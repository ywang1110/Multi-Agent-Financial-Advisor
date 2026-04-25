from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.guardrails.validators import (
    DisclaimerGuardrail,
    OffTopicGuardrail,
    PIIScrubbingGuardrail,
    TurnLimitGuardrail,
    build_advisor_output_guardrails,
    build_input_guardrails,
)


class TestTurnLimitGuardrail:
    def test_blocks_when_limit_reached(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "5")
        guardrail = TurnLimitGuardrail(current_turn=5)
        result = guardrail.check("some message")
        assert result.passed is False
        assert "5" in result.reason

    def test_passes_below_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "5")
        guardrail = TurnLimitGuardrail(current_turn=3)
        result = guardrail.check("some message")
        assert result.passed is True


class TestOffTopicGuardrail:
    """Tests for LLM-based OffTopicGuardrail — LLM is mocked to avoid real API calls."""

    def _make_guardrail(self, llm_reply: str) -> "OffTopicGuardrail":
        """Return an OffTopicGuardrail whose internal LLM returns llm_reply."""
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage
        guardrail = OffTopicGuardrail.__new__(OffTopicGuardrail)
        guardrail._next = None
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=llm_reply)
        guardrail._llm = mock_llm
        return guardrail

    def test_blocks_off_topic(self):
        guardrail = self._make_guardrail("NO")
        result = guardrail.check("Can you give me a recipe for pasta?")
        assert result.passed is False
        assert "Off-topic" in result.reason

    def test_passes_financial_content(self):
        guardrail = self._make_guardrail("YES")
        result = guardrail.check("I want to invest in index funds for retirement.")
        assert result.passed is True

    def test_passes_financial_content_with_liquidating(self):
        # Previously caused false positive with keyword matching
        guardrail = self._make_guardrail("YES")
        result = guardrail.check("Consider gradually liquidating your Tesla position over 12 months.")
        assert result.passed is True

    def test_blocks_dating_advice(self):
        guardrail = self._make_guardrail("NO")
        result = guardrail.check("Can you help me with dating advice?")
        assert result.passed is False

    def test_llm_called_with_content(self):
        guardrail = self._make_guardrail("YES")
        content = "What ETFs should I buy?"
        guardrail.check(content)
        call_args = guardrail._llm.invoke.call_args[0][0]
        # Last message in the list should contain the user content
        assert any(content in str(msg.content) for msg in call_args)


class TestPIIScrubbingGuardrail:
    def test_redacts_ssn(self):
        guardrail = PIIScrubbingGuardrail()
        result = guardrail.check("My SSN is 123-45-6789")
        assert "123-45-6789" not in result.content
        assert "[SSN REDACTED]" in result.content
        assert result.passed is True

    def test_redacts_email(self):
        guardrail = PIIScrubbingGuardrail()
        result = guardrail.check("Email me at john.doe@example.com please")
        assert "john.doe@example.com" not in result.content
        assert "[EMAIL REDACTED]" in result.content

    def test_redacts_account_number(self):
        guardrail = PIIScrubbingGuardrail()
        result = guardrail.check("Account number: 987654321012")
        assert "987654321012" not in result.content
        assert "[ACCOUNT NUMBER REDACTED]" in result.content

    def test_passes_clean_content(self):
        guardrail = PIIScrubbingGuardrail()
        result = guardrail.check("I would like to discuss my portfolio allocation.")
        assert result.passed is True
        assert result.content == "I would like to discuss my portfolio allocation."


class TestDisclaimerGuardrail:
    def test_appends_disclaimer(self):
        guardrail = DisclaimerGuardrail()
        result = guardrail.check("I recommend index funds.")
        assert "Disclaimer" in result.content
        assert result.passed is True

    def test_does_not_duplicate_disclaimer(self):
        guardrail = DisclaimerGuardrail()
        first = guardrail.check("My advice is X.")
        second = guardrail.check(first.content)
        assert second.content.count("Disclaimer") == 1


class TestGuardrailChains:
    @pytest.fixture(autouse=True)
    def patch_guardrail_llm(self):
        """Patch ChatOpenAI inside OffTopicGuardrail for all chain tests.
        Default verdict is YES (finance-related). Tests that need NO override self._llm_mock.
        """
        with patch("src.guardrails.validators.ChatOpenAI") as MockLLM:
            mock = MagicMock()
            mock.invoke.return_value = AIMessage(content="YES")
            MockLLM.return_value = mock
            self._llm_mock = mock
            yield

    def test_advisor_chain_normal_output(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        chain = build_advisor_output_guardrails(current_turn=2)
        result = chain.handle("I recommend a diversified portfolio.")
        assert result.passed is True
        assert "Disclaimer" in result.content

    def test_advisor_chain_blocks_turn_limit(self, monkeypatch):
        # Turn limit fires before OffTopicGuardrail — LLM is never called
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        chain = build_advisor_output_guardrails(current_turn=10)
        result = chain.handle("Some advice here.")
        assert result.passed is False

    def test_advisor_chain_blocks_off_topic(self, monkeypatch):
        self._llm_mock.invoke.return_value = AIMessage(content="NO")
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        chain = build_advisor_output_guardrails(current_turn=1)
        result = chain.handle("Let me tell you a recipe for success.")
        assert result.passed is False

    def test_input_chain_scrubs_pii(self):
        chain = build_input_guardrails()
        result = chain.handle("My SSN is 111-22-3333 and I want to invest.")
        assert result.passed is True
        assert "111-22-3333" not in result.content

    def test_input_chain_blocks_off_topic(self):
        self._llm_mock.invoke.return_value = AIMessage(content="NO")
        chain = build_input_guardrails()
        result = chain.handle("What is the weather like today?")
        assert result.passed is False
