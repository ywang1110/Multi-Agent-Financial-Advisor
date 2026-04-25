from src.agents.advisor_agent import AdvisorAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.client_agent import ClientAgent
from src.models.client_profile import ClientProfile
from src.tools.analyst_tools import get_analyst_tools


class AgentFactory:
    """
    Factory pattern: centralizes agent construction and decouples
    the orchestration layer from agent implementation details.
    """

    @staticmethod
    def create_client_agent(profile: ClientProfile) -> ClientAgent:
        return ClientAgent(profile=profile)

    @staticmethod
    def create_advisor_agent() -> AdvisorAgent:
        return AdvisorAgent()

    @staticmethod
    def create_analyst_agent() -> AnalystAgent:
        tools = get_analyst_tools()
        return AnalystAgent(tools=tools)
