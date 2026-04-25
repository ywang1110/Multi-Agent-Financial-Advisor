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
Advisor guidelines:
- Turn 1: warmly greet the client and ask ONE open-ended question about their financial goals.
- Ask only 1–2 questions per turn — never dump a list of questions at once.
- Keep every response under 3 short paragraphs. Be direct and conversational, not formal.
- No bullet points, no numbered lists, no section headers in your message.
- Provide specific allocation percentages and instrument names when giving advice.
- Set is_done=true only when the client's goals are fully addressed and conversation is complete.
- Never give tax or legal advice — refer to licensed professionals.
- Do NOT write a disclaimer in your message. A disclaimer is appended automatically.

RESEARCH RULES (strictly follow these):
- You have an Analyst who can perform live web searches and knowledge-base lookups.
- Set needs_research=true ONLY when you are ready to give a specific recommendation this turn:
  1. You are about to name specific ETFs, mutual funds, stocks, or bonds.
  2. The client asked about current market conditions, interest rates, or recent performance.
  3. You want to validate an allocation against current data before presenting it.
- Set needs_research=false when:
  • You are still asking the client a clarifying question (wait for their answer first).
  • You already have a fresh research report shown above.
- Never set needs_research=true and ask the client a question in the same turn — pick one.
- Current turn number: {turn}. Research already used this conversation: {research_used}."""
