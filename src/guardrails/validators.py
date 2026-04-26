import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.settings import get_settings

DISCLAIMER = (
    "\n\n---\n"
    "Disclaimer: This information is for educational purposes only and does not "
    "constitute personalized financial, legal, or tax advice. Please consult a "
    "licensed financial advisor before making investment decisions."
)

PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (re.compile(r"\b\d{16}\b"), "[CARD NUMBER REDACTED]"),
    (re.compile(r"\b\d{9,12}\b"), "[ACCOUNT NUMBER REDACTED]"),
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL REDACTED]"),
]


@dataclass
class GuardrailResult:
    passed: bool
    content: str
    reason: str = ""
    is_off_topic: bool = False


class BaseGuardrail(ABC):
    """Chain of Responsibility — each guardrail processes content and passes it along."""

    def __init__(self, next_guardrail: "BaseGuardrail | None" = None) -> None:
        self._next = next_guardrail

    def handle(self, content: str) -> GuardrailResult:
        result = self.check(content)
        if not result.passed:
            return result
        if self._next:
            return self._next.handle(result.content)
        return result

    @abstractmethod
    def check(self, content: str) -> GuardrailResult:
        pass


class TurnLimitGuardrail(BaseGuardrail):
    """Blocks processing if conversation has exceeded the maximum turn limit."""

    def __init__(self, current_turn: int, next_guardrail: "BaseGuardrail | None" = None) -> None:
        super().__init__(next_guardrail)
        self._current_turn = current_turn
        self._max_turns = get_settings().max_conversation_turns

    def check(self, content: str) -> GuardrailResult:
        if self._current_turn >= self._max_turns:
            return GuardrailResult(
                passed=False,
                content=content,
                reason=f"Turn limit of {self._max_turns} reached. Ending conversation.",
            )
        return GuardrailResult(passed=True, content=content)


class OffTopicGuardrail(BaseGuardrail):
    """
    LLM-based off-topic filter.

    Uses a semantic classifier to determine whether a message is related to financial
    investment topics. This approach understands context, so everyday words used as
    metaphors or analogies (e.g. "treat it like a recipe", "it's a game of patience")
    are never incorrectly flagged as off-topic.
    """

    _SYSTEM_PROMPT = (
        "You are a content classifier for a financial investment advisory system. "
        "Decide if the message is related to financial investment, personal finance, "
        "portfolio management, or wealth management.\n\n"
        "Rules:\n"
        "- Reply with exactly one word: YES if finance-related, NO if not.\n"
        "- Everyday words used as metaphors or analogies in a financial context (e.g. "
        "'treat it like a recipe', 'it's a game of patience', 'the weather in the market') "
        "are finance-related — answer YES.\n"
        "- Only answer NO when the message is genuinely off-topic and has no meaningful "
        "connection to finance (e.g. asking for a cooking recipe, sports scores, "
        "celebrity news, or medical advice)."
    )

    def __init__(
        self,
        next_guardrail: "BaseGuardrail | None" = None,
        soft_mode: bool = False,
    ) -> None:
        """
        Args:
            soft_mode: When True (used for client input), off-topic messages are passed
                       through with an annotation rather than blocked. The advisor is then
                       responsible for redirecting the conversation.
                       When False (default, used for advisor output), off-topic content is
                       blocked and replaced with a rejection reason.
        """
        super().__init__(next_guardrail)
        self._soft_mode = soft_mode
        settings = get_settings()
        self._llm = ChatOpenAI(model=settings.llm_model, temperature=0, max_tokens=10)

    def check(self, content: str) -> GuardrailResult:
        # LLM semantic classifier — understands context so metaphors and analogies
        # (e.g. "treat it like a recipe") are never incorrectly flagged as off-topic.
        response = self._llm.invoke([
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=content),
        ])
        verdict = response.content.strip().lower()
        if "no" in verdict:
            return self._off_topic_result(content, "Off-topic content detected.")
        return GuardrailResult(passed=True, content=content)

    def _off_topic_result(self, content: str, reason: str) -> GuardrailResult:
        if self._soft_mode:
            # Pass the actual client message through but annotate it so the advisor
            # can see what was said and take responsibility for steering back on topic.
            annotated = (
                f"[OFF-TOPIC — {reason} Please acknowledge briefly and redirect to financial matters.]\n"
                f"{content}"
            )
            return GuardrailResult(
                passed=True,
                content=annotated,
                reason=reason,
                is_off_topic=True,
            )
        # Hard mode (advisor output): block and replace with rejection message
        return GuardrailResult(
            passed=False,
            content=content,
            reason=f"{reason} This system only handles financial investment topics.",
            is_off_topic=True,
        )


class PIIScrubbingGuardrail(BaseGuardrail):
    """Redacts PII patterns from content before logging or passing downstream."""

    def check(self, content: str) -> GuardrailResult:
        scrubbed = content
        for pattern, replacement in PII_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return GuardrailResult(passed=True, content=scrubbed)


class DisclaimerGuardrail(BaseGuardrail):
    """Appends a regulatory disclaimer to all advisor-facing outputs."""

    def check(self, content: str) -> GuardrailResult:
        if DISCLAIMER.strip() not in content:
            content = content + DISCLAIMER
        return GuardrailResult(passed=True, content=content)


def build_advisor_output_guardrails(current_turn: int) -> BaseGuardrail:
    """
    Builds the guardrail chain for advisor outputs.
    Order: TurnLimit → OffTopic → PII Scrub → Disclaimer
    """
    disclaimer = DisclaimerGuardrail()
    pii = PIIScrubbingGuardrail(next_guardrail=disclaimer)
    off_topic = OffTopicGuardrail(next_guardrail=pii)
    turn_limit = TurnLimitGuardrail(current_turn=current_turn, next_guardrail=off_topic)
    return turn_limit


def build_input_guardrails() -> BaseGuardrail:
    """
    Builds the guardrail chain for user/client inputs.
    Order: OffTopic (soft) → PII Scrub

    Off-topic client messages are NOT blocked — they are annotated and passed through
    so the advisor can see the actual content and redirect the conversation naturally.
    """
    pii = PIIScrubbingGuardrail()
    off_topic = OffTopicGuardrail(next_guardrail=pii, soft_mode=True)
    return off_topic
