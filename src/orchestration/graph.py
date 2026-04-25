from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from src.agents.factory import AgentFactory
from src.config.settings import get_settings
from src.models.client_profile import ClientProfile
from src.models.state import ConversationState


def _make_init_node(client_profile: ClientProfile):
    """
    Returns a node function that injects default values into state.
    Also coerces client_profile from dict → ClientProfile when the state
    arrives via the LangGraph API (Studio serializes it as a plain dict).
    """
    def init_state(state: ConversationState) -> dict:
        updates: dict = {}

        raw_profile = state.get("client_profile")
        if raw_profile is None:
            updates["client_profile"] = client_profile
        elif isinstance(raw_profile, dict):
            updates["client_profile"] = ClientProfile(**raw_profile)

        if state.get("turn_count") is None:
            updates["turn_count"] = 0
        if state.get("is_satisfied") is None:
            updates["is_satisfied"] = False
        if state.get("needs_research") is None:
            updates["needs_research"] = False
        if state.get("latest_research") is None:
            updates["latest_research"] = None
        if not state.get("final_summary"):
            updates["final_summary"] = ""
        return updates
    return init_state


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_advisor(state: ConversationState) -> str:
    """
    Decides next node after the Advisor speaks.
    - Conversation concluded  → END
    - Research needed         → analyst
    - Otherwise               → client
    """
    if state.get("final_summary"):
        return END
    if state.get("needs_research"):
        return "analyst"
    return "client"


def route_after_client(state: ConversationState) -> str:
    """
    Decides next node after the Client responds.
    - Client satisfied        → END
    - Turn limit hit          → END
    - Otherwise               → advisor
    """
    settings = get_settings()
    turn_count = state.get("turn_count", 0)

    if state.get("is_satisfied"):
        return END
    if turn_count >= settings.max_conversation_turns:
        return END
    return "advisor"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(client_profile: ClientProfile) -> CompiledGraph:
    """
    Assembles the LangGraph StateGraph for the multi-agent conversation.

    Node layout:
        START → advisor ──► analyst ──► advisor (loop back with research)
                        ──► client  ──► advisor (loop back with feedback)
                        ──► END
    """
    advisor_agent = AgentFactory.create_advisor_agent()
    analyst_agent = AgentFactory.create_analyst_agent()
    client_agent = AgentFactory.create_client_agent(client_profile)

    builder = StateGraph(ConversationState)

    # Register nodes
    builder.add_node("init", _make_init_node(client_profile))
    builder.add_node("advisor", advisor_agent.run)
    builder.add_node("analyst", analyst_agent.run)
    builder.add_node("client", client_agent.run)

    # Entry point: always run init first to populate defaults
    builder.add_edge(START, "init")
    builder.add_edge("init", "advisor")

    # Advisor → conditional routing
    builder.add_conditional_edges("advisor", route_after_advisor)

    # Analyst always returns to advisor (with research results in state)
    builder.add_edge("analyst", "advisor")

    # Client → conditional routing
    builder.add_conditional_edges("client", route_after_client)

    return builder.compile()


def get_initial_state(client_profile: ClientProfile) -> dict:
    """Returns the initial LangGraph state for a new conversation."""
    return {
        "messages": [],
        "client_profile": client_profile,
        "turn_count": 0,
        "is_satisfied": False,
        "needs_research": False,
        "latest_research": None,
        "final_summary": "",
    }
