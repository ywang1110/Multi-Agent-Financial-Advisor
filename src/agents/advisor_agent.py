from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.guardrails.validators import build_advisor_output_guardrails
from src.models.research_task import ResearchReport, ResearchTask, ResearchToolType
from src.models.state import ConversationState
from src.strategies.investment_strategy import StrategyFactory


class AdvisorDecision(BaseModel):
    message: str = Field(description="Advisor's response message to the client")
    needs_research: bool = Field(
        description="True if the analyst should be called to gather more information"
    )
    research_query: str = Field(
        default="",
        description="Specific research question for the analyst (required if needs_research=True)",
    )
    research_context: str = Field(
        default="",
        description="Why this research is needed — context for the analyst",
    )
    research_tool_hint: ResearchToolType = Field(
        default=ResearchToolType.BOTH,
        description="Which tool the analyst should prioritize",
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
                f"  Tool   : {decision.research_tool_hint}\n"
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
                tool_hint=decision.research_tool_hint,
            )

        if decision.is_done:
            updates["final_summary"] = final_message

        return updates

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

        return f"""You are a professional financial advisor at a top-tier investment firm.
You may ONLY discuss financial investment topics. Politely decline any off-topic requests.

Client Profile:
{profile.to_summary()}

Recommended Investment Strategy:
{strategy_context}
{research_section}
== YOUR ROLE & RESPONSIBILITIES ==
You work in three strict phases. Follow them in order:

PHASE 1 — DISCOVERY (turn 1-2, research_done=False):
  - Your only job is to understand the client's goals more deeply.
  - Ask ONE focused clarifying question per turn (retirement income target, liquidity needs, etc.).
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
Turn: {turn} | Research done: {research_used}
{"→ PHASE 1: Keep gathering information. Do not trigger the analyst yet." if not research_used and turn < 2 else ""}
{"→ PHASE 2: You have enough context — trigger the Analyst now (needs_research=True)." if not research_used and turn >= 2 else ""}
{"→ PHASE 3: Deliver recommendations grounded in the research report. Re-trigger analyst only for genuinely new topics the client raises." if research_used else ""}

== STYLE RULES ==
- Keep responses under 3 short paragraphs. Conversational, not formal.
- No bullet points, numbered lists, or section headers in your message.
- Never give tax or legal advice. Do NOT write a disclaimer — it is appended automatically."""
