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

        return f"""You are a professional financial advisor at a top-tier investment firm.
You may ONLY discuss financial investment topics. Politely decline any off-topic requests.

Client Profile:
{profile.to_summary()}

Recommended Investment Strategy:
{strategy_context}
{research_section}
Advisor guidelines:
- First turn: warmly greet the client and ask open-ended questions about their goals.
- Gather enough information before recommending anything specific.
- Set needs_research=true if you need current market data or deeper financial knowledge.
- Provide specific allocation percentages and instrument names when giving advice.
- Set is_done=true only when the client's goals are fully addressed and conversation is complete.
- Never give tax or legal advice — refer to licensed professionals.
- Always be empathetic and clear."""
