from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.models.client_profile import ClientProfile
from src.models.research_task import ResearchReport


class ConversationState(MessagesState):
    """
    Shared state passed between all nodes in the LangGraph.

    Inherits `messages: Annotated[list[BaseMessage], add_messages]` from MessagesState,
    which automatically appends new messages rather than overwriting.
    """

    # Client context
    client_profile: ClientProfile

    # Conversation control
    turn_count: int
    is_satisfied: bool
    needs_research: bool

    # Analyst research
    latest_research: ResearchReport | None
    research_history: list[str]  # all research queries run so far (for dedup)

    # Final output
    final_summary: str

    # Human handoff memo (populated when turn limit is reached)
    handoff_summary: str
