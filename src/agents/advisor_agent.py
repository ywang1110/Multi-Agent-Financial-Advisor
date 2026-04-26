from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.guardrails.validators import build_advisor_output_guardrails
from src.models.research_task import ResearchReport, ResearchTask
from src.models.state import ConversationState
from src.strategies.investment_strategy import StrategyFactory


class AdvisorDecision(BaseModel):
    message: str = Field(description="Advisor's response message to the client")
    needs_research: bool = Field(
        description="True if the analyst should be called to gather more information"
    )
    research_query: str = Field(
        default="",
        max_length=300,
        description="Specific research question for the analyst (required if needs_research=True). Max 300 characters — be concise.",
    )
    research_context: str = Field(
        default="",
        description="Why this research is needed — context for the analyst",
    )
    is_done: bool = Field(
        description="True when the conversation has reached a satisfactory conclusion"
    )


class AdvisorAgent(BaseAgent):
    """
    The central hub agent. Only agent that communicates with both Client and Analyst.
    Applies guardrails to every outgoing message.
    """

    def __init__(self) -> None:
        super().__init__()
        self._structured_llm = self._llm.with_structured_output(AdvisorDecision)

    def run(self, state: ConversationState) -> dict:
        profile = state["client_profile"]
        strategy = StrategyFactory.get_strategy(profile)
        recommendation = strategy.get_recommendation(profile)

        system = SystemMessage(content=self._build_system_prompt(state, recommendation.to_prompt_context()))
        decision: AdvisorDecision = self._structured_llm.invoke(
            [system] + state["messages"]
        )

        # Apply guardrail pipeline to outgoing message
        guardrail_chain = build_advisor_output_guardrails(state["turn_count"])
        guardrail_result = guardrail_chain.handle(decision.message)

        final_message = guardrail_result.content if guardrail_result.passed else guardrail_result.reason

        # Log research decision reasoning
        if decision.needs_research:
            print(
                f"\n[Advisor → needs_research=True]\n"
                f"  Query  : {decision.research_query}\n"
                f"  Reason : {decision.research_context}\n"
            )
        else:
            print(f"\n[Advisor → needs_research=False] No research needed this turn.\n")

        updates: dict = {
            "messages": [AIMessage(content=final_message, name="advisor")],
            "needs_research": decision.needs_research,
        }

        if decision.needs_research and decision.research_query:
            updates["latest_research"] = ResearchTask(
                query=decision.research_query,
                context=decision.research_context,
            )

        if decision.is_done:
            updates["final_summary"] = final_message

        return updates

    def generate_handoff(self, state: ConversationState) -> dict:
        """
        Called when the turn limit is reached.
        Produces a structured handoff memo for the human agent taking over.
        """
        profile = state["client_profile"]
        history: list[str] = state.get("research_history") or []
        research_done = "\n".join(f"  - {q}" for q in history) if history else "  - None"

        conversation_text = "\n".join(
            f"{m.type.upper()} ({getattr(m, 'name', m.type)}): {m.content}"
            for m in state.get("messages", [])
        )

        prompt = f"""You are preparing a briefing note for a human financial advisor who is taking over this client.
Be extremely concise. Every line must be actionable or factual — no filler, no repetition.

CLIENT: {profile.name} | Age {profile.age} | Income ${profile.annual_income_usd:,} | Assets ${profile.total_assets_usd:,} | Risk: {profile.risk_tolerance} | Horizon: {profile.investment_horizon_years}yr

Conversation so far:
{conversation_text}

Research already run by AI analyst:
{research_done}

Write the briefing in this exact format (keep each section to 2-4 bullet points max):

PRIORITY CONCERNS
• [what the client is most worried about, with specifics]

RECOMMENDATIONS MADE
• [exact allocations/tickers/percentages already discussed]

UNRESOLVED QUESTIONS
• [what the client still wants answered]

IMMEDIATE ACTIONS FOR ADVISOR
• [specific next steps, ordered by urgency]

Rules: use real numbers and names from the conversation. No generic statements. Max 200 words total."""

        memo = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()

        handoff_message = (
            f"Thank you for sharing so much detail about your situation, {profile.name.split()[0]}. "
            "Your goals deserve more dedicated, personalised attention than I can provide here. "
            "I'm connecting you with one of our senior advisors who will review everything we've discussed "
            "and work with you one-on-one to build a tailored plan. You'll hear from them shortly."
        )

        return {
            "messages": [AIMessage(content=handoff_message, name="advisor")],
            "handoff_summary": memo,
        }

    def _build_system_prompt(self, state: ConversationState, strategy_context: str) -> str:
        profile = state["client_profile"]
        research_section = ""
        if state.get("latest_research") and isinstance(state["latest_research"], ResearchReport):
            research_section = f"""
Latest Research Report from Analyst:
{state['latest_research'].findings}
---"""

        turn = state.get("turn_count", 0)
        research_used = isinstance(state.get("latest_research"), ResearchReport)
        history: list[str] = state.get("research_history") or []
        history_section = ""
        if history:
            formatted = "\n".join(f"  - {q}" for q in history)
            history_section = f"\nTopics already researched (do NOT re-trigger analyst for these):\n{formatted}\n"

        return f"""You are a professional financial advisor at a top-tier investment firm.
You may ONLY discuss financial investment topics.

== HANDLING OFF-TOPIC CLIENT MESSAGES ==
Sometimes a client message will begin with a tag like:
  [OFF-TOPIC — Off-topic content detected... Please acknowledge briefly and redirect to financial matters.]
When you see this tag:
- Read the client's actual words below the tag — they are visible to you for full transparency.
- Respond in one short sentence that warmly acknowledges what they said (e.g. "Ha, sounds fun!" or "Good game last night!"), then immediately pivot back to their financial question.
- Never ignore the off-topic comment entirely — a natural advisor would react briefly before refocusing.
- Never repeat or quote the [OFF-TOPIC] tag in your reply.
- Keep the entire response within the normal 200-word limit.

Client Profile:
{profile.to_summary()}

Recommended Investment Strategy:
{strategy_context}
{research_section}{history_section}
== YOUR ROLE & RESPONSIBILITIES ==
You work in three strict phases. Follow them in order:

PHASE 1 — DISCOVERY (research_done=False, context not yet sufficient):
  - If this is the very first message (no prior messages from the client), open with a warm, \
personalised greeting: address the client by first name, acknowledge one specific detail from their \
profile (e.g. their goal or a notable holding), and then ask ONE focused clarifying question.
  - On subsequent turns in this phase, ask ONE focused clarifying question per turn \
(retirement income target, liquidity needs, etc.).
  - Move to PHASE 2 as soon as you have enough context — do NOT wait for a fixed number of turns. \
One or two exchanges is usually sufficient unless the client's situation is genuinely complex.
  - Do NOT recommend anything specific yet. Do NOT trigger the analyst.
  - Set needs_research=False.

PHASE 2 — RESEARCH (exactly once, when you have enough info):
  - Transition here as soon as you have enough client context to form a recommendation.
  - Set needs_research=True. Write a detailed research_query for the Analyst to run.
  - The Analyst's job: fetch current ETF data, market returns, and investment frameworks.
  - Do NOT ask the client a question in this turn — your message should be a brief "let me look into that for you" bridge sentence.
  - You MUST pass through this phase before giving any specific recommendation.

PHASE 3 — RECOMMENDATION (after research report is available):
  - Now give concrete, specific advice: allocation percentages, ETF tickers, rebalancing logic.
  - Ground every recommendation in the research report shown above.
  - Continue the dialogue — ask if the client has follow-up questions.
  - Set is_done=True only when the client is fully satisfied.
  - If the client raises a NEW topic not covered by the existing research report
    (e.g. asks about a different asset class, a new time horizon, or questions your data),
    set needs_research=True with a new research_query targeting that specific gap.
    Do NOT re-research topics already covered in the report above.

== CURRENT STATE ==
{"→ PHASE 3 (research complete): Deliver grounded recommendations. Re-trigger analyst only for genuinely new topics the client raises." if research_used else f"→ PHASE 1 / 2 (turn {turn}, no research yet): {'You have reached the maximum discovery turns — move to PHASE 2 now (needs_research=True), even if you feel you need more information.' if turn >= 2 else 'Gather information first. Move to PHASE 2 as soon as you have enough context — you have at most 2 clarifying turns before you must trigger the Analyst.'}"}
IMPORTANT: Never mention session limits, turn counts, or that the conversation will end. Do NOT hint that a handoff is coming.

== STYLE RULES ==
- Use Markdown formatting — the UI renders it fully.
- Lead with 1-2 short conversational sentences. Then use structure for data-heavy content:
  - Use **bold** to highlight key numbers, tickers, and percentages.
  - Use bullet points when comparing options or listing allocations.
  - Use a Markdown table when presenting data side-by-side.
- Keep total response under 200 words. Be direct — no padding, no repetition.
- End with ONE focused question to keep the conversation moving.
- Never give tax or legal advice. Do NOT write a disclaimer — it is appended automatically.

== CHARTS ==
When presenting portfolio allocations or numeric comparisons, render a Unicode bar chart inside a code block.
Each bar = 30 chars wide. Scale: 1 char ≈ 1%. Fill with █, remainder with ░.
Format exactly like this example (use actual numbers, not placeholders):

```
RECOMMENDED ALLOCATION
Stocks   ████████████████████░░░░░░░░░░  60%
Bonds    ████████████░░░░░░░░░░░░░░░░░░  35%
REITs    ███░░░░░░░░░░░░░░░░░░░░░░░░░░░   5%
```

Use this chart format whenever you propose or compare allocations. Always follow it with a Markdown table for the exact numbers."""
