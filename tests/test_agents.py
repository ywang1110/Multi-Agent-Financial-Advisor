from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.models.client_profile import ClientProfile, RiskTolerance, InvestmentGoal, Holding
from src.models.research_task import ResearchReport, ResearchTask

_TEST_CLIENT = ClientProfile(
    name="Sarah Chen",
    age=38,
    annual_income_usd=185_000,
    total_assets_usd=620_000,
    total_liabilities_usd=180_000,
    risk_tolerance=RiskTolerance.MODERATE,
    investment_goals=[InvestmentGoal.RETIREMENT, InvestmentGoal.WEALTH_GROWTH],
    investment_horizon_years=25,
)


def make_state(**kwargs) -> dict:
    return {
        "messages": [AIMessage(content="Hello, how can I help?", name="advisor")],
        "client_profile": _TEST_CLIENT,
        "turn_count": kwargs.get("turn_count", 0),
        "is_satisfied": kwargs.get("is_satisfied", False),
        "needs_research": kwargs.get("needs_research", False),
        "latest_research": kwargs.get("latest_research", None),
        "final_summary": kwargs.get("final_summary", ""),
    }


class TestClientAgent:
    def _patch_guardrail_llm(self, verdict: str = "YES"):
        """Patch the input guardrail LLM used inside ClientAgent.run()."""
        mock_guardrail_llm = MagicMock()
        mock_guardrail_llm.invoke.return_value = AIMessage(content=verdict)
        return patch("src.guardrails.validators.ChatOpenAI", return_value=mock_guardrail_llm)

    def test_run_returns_message_and_satisfied_false(self):
        from src.agents.client_agent import ClientAgent, ClientOutput

        mock_output = ClientOutput(
            message="I am concerned about the bond allocation.",
            is_satisfied=False,
        )
        with patch("src.agents.base_agent.ChatOpenAI") as MockLLM, self._patch_guardrail_llm():
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_output
            MockLLM.return_value = mock_llm

            agent = ClientAgent()
            result = agent.run(make_state())

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "I am concerned about the bond allocation."
        assert result["messages"][0].name == "client"
        assert result["is_satisfied"] is False
        assert result["turn_count"] == 1

    def test_run_sets_satisfied_true(self):
        from src.agents.client_agent import ClientAgent, ClientOutput

        mock_output = ClientOutput(
            message="Thank you, I am fully satisfied with the plan.",
            is_satisfied=True,
        )
        with patch("src.agents.base_agent.ChatOpenAI") as MockLLM, self._patch_guardrail_llm():
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_output
            MockLLM.return_value = mock_llm

            agent = ClientAgent()
            result = agent.run(make_state())

        assert result["is_satisfied"] is True


class TestAdvisorAgent:
    def _make_mock_decision(self, **kwargs):
        from src.agents.advisor_agent import AdvisorDecision

        return AdvisorDecision(
            message=kwargs.get("message", "Let me help you with your portfolio."),
            needs_research=kwargs.get("needs_research", False),
            research_query=kwargs.get("research_query", ""),
            research_context=kwargs.get("research_context", ""),
            is_done=kwargs.get("is_done", False),
        )

    def _patch_guardrail_llm(self):
        """Returns a context manager that patches OffTopicGuardrail's LLM with YES verdict."""
        mock_guardrail_llm = MagicMock()
        mock_guardrail_llm.invoke.return_value = AIMessage(content="YES")
        return patch("src.guardrails.validators.ChatOpenAI", return_value=mock_guardrail_llm)

    def test_run_returns_message(self):
        from src.agents.advisor_agent import AdvisorAgent

        mock_decision = self._make_mock_decision()
        with patch("src.agents.base_agent.ChatOpenAI") as MockLLM, self._patch_guardrail_llm():
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_decision
            MockLLM.return_value = mock_llm

            agent = AdvisorAgent()
            result = agent.run(make_state())

        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "advisor"
        assert "Disclaimer" in result["messages"][0].content

    def test_run_sets_needs_research(self):
        from src.agents.advisor_agent import AdvisorAgent

        mock_decision = self._make_mock_decision(
            needs_research=True,
            research_query="Current S&P 500 performance",
            research_context="Client wants to know about market conditions",
        )
        with patch("src.agents.base_agent.ChatOpenAI") as MockLLM, self._patch_guardrail_llm():
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_decision
            MockLLM.return_value = mock_llm

            agent = AdvisorAgent()
            result = agent.run(make_state())

        assert result["needs_research"] is True
        assert isinstance(result["latest_research"], ResearchTask)

    def test_run_sets_final_summary_when_done(self):
        from src.agents.advisor_agent import AdvisorAgent

        mock_decision = self._make_mock_decision(
            message="Here is your final plan.",
            is_done=True,
        )
        with patch("src.agents.base_agent.ChatOpenAI") as MockLLM, self._patch_guardrail_llm():
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_decision
            MockLLM.return_value = mock_llm

            agent = AdvisorAgent()
            result = agent.run(make_state())

        assert result["final_summary"] != ""


class TestAnalystAgent:
    def test_run_returns_research_report(self):
        from src.agents.analyst_agent import AnalystAgent

        task = ResearchTask(
            query="Best ETFs for moderate risk investors",
            context="Client wants ETF recommendations",
        )
        state = make_state(latest_research=task)

        mock_final_message = AIMessage(content="VOO and BND are excellent choices.")
        mock_react_result = {"messages": [mock_final_message]}

        with patch("src.agents.base_agent.ChatOpenAI") as MockLLM:
            mock_llm = MagicMock()
            MockLLM.return_value = mock_llm

            with patch("src.agents.analyst_agent.create_react_agent") as mock_react:
                mock_react.return_value.invoke.return_value = mock_react_result
                agent = AnalystAgent(tools=[])
                result = agent.run(state)

        assert isinstance(result["latest_research"], ResearchReport)
        assert "VOO" in result["latest_research"].findings
