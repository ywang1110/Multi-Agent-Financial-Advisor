from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.guardrails.validators import build_input_guardrails
from src.models.client_profile import ClientProfile
from src.models.state import ConversationState


class ClientOutput(BaseModel):
    message: str = Field(description="Client's natural language response to the advisor")
    is_satisfied: bool = Field(
        description="True only when the advisor has fully addressed the client's goals"
    )


class ClientAgent(BaseAgent):
    """
    Simulates a real human client using an LLM persona.
    In production this node would be replaced by a human-in-the-loop UI.
    """

    def __init__(self) -> None:
        super().__init__()
        self._structured_llm = self._llm.with_structured_output(ClientOutput)

    def run(self, state: ConversationState) -> dict:
        profile = state["client_profile"]
        system = SystemMessage(content=self._build_system_prompt(profile))
        response: ClientOutput = self._structured_llm.invoke(
            [system] + state["messages"]
        )

        # Apply input guardrails to the client's outgoing message
        guardrail_chain = build_input_guardrails()
        guardrail_result = guardrail_chain.handle(response.message)
        final_message = guardrail_result.content if guardrail_result.passed else guardrail_result.reason

        return {
            "messages": [HumanMessage(content=final_message, name="client")],
            "is_satisfied": response.is_satisfied,
            "turn_count": state["turn_count"] + 1,
        }

    def _build_system_prompt(self, profile: ClientProfile) -> str:
        return f"""STRICT IDENTITY RULE — READ THIS FIRST:
You ARE {profile.name}. You are not an AI describing {profile.name}. You are not an observer commenting \
on {profile.name}'s profile. You are this person, speaking directly to their financial advisor.

FORBIDDEN (will be rejected):
- "{profile.name} is aggressive" → you are talking ABOUT yourself in third person. Never do this.
- "{profile.name}'s goals are..." → same violation.
- "Based on their profile..." → you have no "profile". You have a life.
- "Since {profile.name.split()[0]} is the one investing..." → you ARE the one investing. Say "I".

REQUIRED: Every sentence must use "I", "my", or "me". Speak exactly as you would in a real meeting \
with your financial advisor — naturally, personally, in first person.

Your background:
{profile.to_summary()}

Behavioral guidelines:
- Ask follow-up questions if anything is unclear or feels too risky for your situation.
- Express genuine concerns about volatility if you are conservative or moderate.
- Do NOT accept vague advice — push for specifics (allocation percentages, instrument names).
- Set is_satisfied=true when the advisor has given you a concrete allocation plan with specific instruments \
and has addressed your main concern(s). You do NOT need every possible question answered — real clients \
accept a solid plan and move on.
- Keep each response concise: 2-4 sentences maximum."""
