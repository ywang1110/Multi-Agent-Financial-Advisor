import json

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from src.agents.factory import AgentFactory
from src.config.settings import get_settings
from src.models.client_profile import ClientProfile
from src.models.state import ConversationState


def _parse_profile_from_message(state: ConversationState) -> ClientProfile | None:
    """
    Fallback: when Studio Chat tab is used, the user's JSON input arrives as a
    HumanMessage rather than a state key. Try to parse client_profile from the
    content of the first message so the graph works in both Chat and State modes.
    """
    messages = state.get("messages") or []
    print(f"[init] _parse_profile_from_message: {len(messages)} message(s) in state")
    for i, msg in enumerate(messages):
        # Handle both LangChain message objects and raw (type, content) tuples
        if isinstance(msg, tuple):
            content = msg[1] if len(msg) > 1 else None
        else:
            content = getattr(msg, "content", None)
        print(f"[init] msg[{i}] type={type(msg).__name__}, content_type={type(content).__name__}, preview={str(content)[:120] if content else None}")
        # Studio Chat sends content as a list of blocks: [{'type': 'text', 'text': '...'}]
        if isinstance(content, list):
            text_parts = [
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            data = json.loads(content.strip())
            # Accept either {"client_profile": {...}} or the profile dict directly
            profile_data = data.get("client_profile", data) if isinstance(data, dict) else None
            if isinstance(profile_data, dict) and "name" in profile_data:
                profile = ClientProfile.model_validate(profile_data)
                print(f"[init] Successfully parsed client_profile for: {profile.name}")
                return profile
        except json.JSONDecodeError:
            print(f"[init] msg[{i}] is not valid JSON, skipping")
        except Exception as e:
            print(f"[init] msg[{i}] JSON parsed but ClientProfile validation failed: {type(e).__name__}: {e}")
    print("[init] No client_profile found in messages")
    return None


def _make_init_node(client_profile: ClientProfile | None = None):
    """
    Returns a node function that injects default values into state.
    Handles three sources for client_profile (in priority order):
      1. Closure argument (hardcoded profile via build_graph)
      2. State key (Studio State-input panel, serialised as plain dict)
      3. First message content (Studio Chat tab, user pastes JSON)
    """
    def init_state(state: ConversationState) -> dict:
        updates: dict = {}

        # messages may be absent when Studio sends only client_profile as a state key
        if not state.get("messages"):
            updates["messages"] = []

        raw_profile = state.get("client_profile")
        if raw_profile is None and client_profile is not None:
            # Source 1: hardcoded profile
            updates["client_profile"] = client_profile
        elif isinstance(raw_profile, dict):
            # Source 2: state key (Studio serialises Pydantic models as plain dicts)
            updates["client_profile"] = ClientProfile(**raw_profile)
        elif raw_profile is None:
            # Source 3: profile embedded in the first chat message (Studio Chat tab)
            parsed = _parse_profile_from_message(state)
            if parsed:
                updates["client_profile"] = parsed
                # Remove the profile message from the conversation so it is not
                # treated as a client question by the advisor.
                updates["messages"] = []

        if state.get("turn_count") is None:
            updates["turn_count"] = 0
        if state.get("is_satisfied") is None:
            updates["is_satisfied"] = False
        if state.get("needs_research") is None:
            updates["needs_research"] = False
        if state.get("latest_research") is None:
            updates["latest_research"] = None
        if state.get("research_history") is None:
            updates["research_history"] = []
        if not state.get("final_summary"):
            updates["final_summary"] = ""
        if not state.get("handoff_summary"):
            updates["handoff_summary"] = ""
        return updates
    return init_state


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_advisor(state: ConversationState) -> str:
    """
    Decides next node after the Advisor speaks.
    - Research needed → analyst
    - Otherwise       → client (Client always gets the final say on satisfaction)
    """
    if state.get("needs_research"):
        return "analyst"
    return "client"


def route_after_client(state: ConversationState) -> str:
    """
    Decides next node after the Client responds.
    - Client satisfied        → END  (takes priority over turn limit)
    - Turn limit hit          → handoff (human agent takeover)
    - Otherwise               → advisor
    """
    settings = get_settings()
    turn_count = state.get("turn_count", 0)

    # Satisfaction takes priority — no handoff needed if client is already happy
    if state.get("is_satisfied"):
        return END
    if turn_count >= settings.max_conversation_turns:
        return "handoff"
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
    client_agent = AgentFactory.create_client_agent()

    builder = StateGraph(ConversationState)

    # Register nodes
    builder.add_node("init", _make_init_node(client_profile))
    builder.add_node("advisor", advisor_agent.run)
    builder.add_node("analyst", analyst_agent.run)
    builder.add_node("client", client_agent.run)
    builder.add_node("handoff", advisor_agent.generate_handoff)

    # Entry point: always run init first to populate defaults
    builder.add_edge(START, "init")
    builder.add_edge("init", "advisor")

    # Advisor → conditional routing
    builder.add_conditional_edges("advisor", route_after_advisor, ["analyst", "client"])

    # Analyst always returns to advisor (with research results in state)
    builder.add_edge("analyst", "advisor")

    # Client → conditional routing
    builder.add_conditional_edges("client", route_after_client, ["advisor", "handoff", END])

    # Handoff → always END after generating the memo
    builder.add_edge("handoff", END)

    return builder.compile()


def build_graph_no_profile() -> CompiledGraph:
    """
    Studio entry point: graph with no hardcoded profile.
    The client_profile must be supplied in the input state at runtime.
    """
    advisor_agent = AgentFactory.create_advisor_agent()
    analyst_agent = AgentFactory.create_analyst_agent()
    client_agent = AgentFactory.create_client_agent()

    builder = StateGraph(ConversationState)

    builder.add_node("init", _make_init_node())
    builder.add_node("advisor", advisor_agent.run)
    builder.add_node("analyst", analyst_agent.run)
    builder.add_node("client", client_agent.run)
    builder.add_node("handoff", advisor_agent.generate_handoff)

    builder.add_edge(START, "init")
    builder.add_edge("init", "advisor")
    builder.add_conditional_edges("advisor", route_after_advisor, ["analyst", "client"])
    builder.add_edge("analyst", "advisor")
    builder.add_conditional_edges("client", route_after_client, ["advisor", "handoff", END])
    builder.add_edge("handoff", END)

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
        "research_history": [],
        "final_summary": "",
        "handoff_summary": "",
    }
