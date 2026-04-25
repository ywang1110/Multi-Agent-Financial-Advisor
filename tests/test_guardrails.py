import pytest

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
    def test_blocks_recipe_keyword(self):
        guardrail = OffTopicGuardrail()
        result = guardrail.check("Can you give me a recipe for pasta?")
        assert result.passed is False
        assert "recipe" in result.reason

    def test_blocks_sports_keyword(self):
        guardrail = OffTopicGuardrail()
        result = guardrail.check("Who won the sports game last night?")
        assert result.passed is False

    def test_passes_financial_content(self):
        guardrail = OffTopicGuardrail()
        result = guardrail.check("I want to invest in index funds for retirement.")
        assert result.passed is True

    def test_case_insensitive(self):
        guardrail = OffTopicGuardrail()
        result = guardrail.check("Tell me about a MOVIE recommendation")
        assert result.passed is False


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
    def test_advisor_chain_normal_output(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        chain = build_advisor_output_guardrails(current_turn=2)
        result = chain.handle("I recommend a diversified portfolio.")
        assert result.passed is True
        assert "Disclaimer" in result.content

    def test_advisor_chain_blocks_turn_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_CONVERSATION_TURNS", "10")
        chain = build_advisor_output_guardrails(current_turn=10)
        result = chain.handle("Some advice here.")
        assert result.passed is False

    def test_advisor_chain_blocks_off_topic(self, monkeypatch):
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
        chain = build_input_guardrails()
        result = chain.handle("What is the weather like today?")
        assert result.passed is False
