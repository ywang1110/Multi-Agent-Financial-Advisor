import pytest
from langgraph.graph import END

from src.models.client_profile import DEMO_CLIENT
from src.orchestration.graph import route_after_advisor, route_after_client


def make_state(**kwargs) -> dict:
    """Helper: build a minimal state dict with sensible defaults."""
    return {
        "messages": [],
        "client_profile": DEMO_CLIENT,
        "turn_count": kwargs.get("turn_count", 0),
        "is_satisfied": kwargs.get("is_satisfied", False),
        "needs_research": kwargs.get("needs_research", False),
        "latest_research": kwargs.get("latest_research", None),
        "final_summary": kwargs.get("final_summary", ""),
    }


class TestRouteAfterAdvisor:
    def test_routes_to_analyst_when_research_needed(self):
        state = make_state(needs_research=True)
        assert route_after_advisor(state) == "analyst"

    def test_routes_to_end_when_conversation_done(self):
        state = make_state(final_summary="Here is your final plan.")
        assert route_after_advisor(state) == END

    def test_routes_to_client_by_default(self):
        state = make_state()
        assert route_after_advisor(state) == "client"

    def test_final_summary_takes_priority_over_research(self):
        state = make_state(needs_research=True, final_summary="Done.")
        assert route_after_advisor(state) == END


class TestRouteAfterClient:
    def test_routes_to_end_when_satisfied(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        state = make_state(is_satisfied=True, turn_count=3)
        assert route_after_client(state) == END

    def test_routes_to_end_when_turn_limit_reached(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        state = make_state(is_satisfied=False, turn_count=10)
        assert route_after_client(state) == END

    def test_routes_to_advisor_when_not_satisfied(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        state = make_state(is_satisfied=False, turn_count=3)
        assert route_after_client(state) == "advisor"

    def test_routes_to_end_when_satisfied_at_turn_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        state = make_state(is_satisfied=True, turn_count=10)
        assert route_after_client(state) == END
