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
    Rejects messages unrelated to finance using LLM semantic classification.
    More robust than keyword matching — handles paraphrasing and avoids false positives.
    """

    _SYSTEM_PROMPT = (
        "You are a content classifier for a financial investment advisory system. "
        "Decide if the message is related to financial investment, personal finance, "
        "portfolio management, or wealth management.\n"
        "Reply with exactly one word: YES if finance-related, NO if not."
    )

    def __init__(self, next_guardrail: "BaseGuardrail | None" = None) -> None:
        super().__init__(next_guardrail)
        settings = get_settings()
        self._llm = ChatOpenAI(model=settings.llm_model, temperature=0, max_tokens=10)

    def check(self, content: str) -> GuardrailResult:
        response = self._llm.invoke([
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=content),
        ])
        verdict = response.content.strip().lower()
        if "no" in verdict:
            return GuardrailResult(
                passed=False,
                content=content,
                reason="Off-topic content detected. This system only handles financial investment topics.",
            )
        return GuardrailResult(passed=True, content=content)


class PIIScrubbingGuardrail(BaseGuardrail):
    """Redacts PII patterns from content before logging or passing downstream."""

    def check(self, content: str) -> GuardrailResult:
        scrubbed = content
        for pattern, replacement in PII_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return GuardrailResult(passed=True, content=scrubbed)


class DisclaimerGuardrail(BaseGuardrail):
    """Appends a regulatory disclaimer to all advisor-facing outputs.
    Skips if any disclaimer is already present (e.g. added by the LLM itself).
    """

    def check(self, content: str) -> GuardrailResult:
        if "disclaimer" not in content.lower():
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
    Order: OffTopic → PII Scrub
    """
    pii = PIIScrubbingGuardrail()
    off_topic = OffTopicGuardrail(next_guardrail=pii)
    return off_topic
