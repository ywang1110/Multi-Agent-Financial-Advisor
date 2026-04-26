import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

load_dotenv()


def _configure_langsmith() -> None:
    """Enable LangSmith tracing if credentials are provided."""
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        api_key = os.getenv("LANGCHAIN_API_KEY", "")
        project = os.getenv("LANGCHAIN_PROJECT", "jpmc-investment-advisor")
        if api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = project
            print(f"LangSmith tracing enabled  →  project: '{project}'")
        else:
            print("LangSmith: LANGCHAIN_API_KEY not set, tracing disabled.")


def _print_transcript(messages: list[BaseMessage]) -> None:
    """Pretty-print the full conversation transcript."""
    role_labels = {"advisor": "Advisor", "client": "Client"}
    separator = "─" * 70

    print(f"\n{'═' * 70}")
    print("  CONVERSATION TRANSCRIPT")
    print(f"{'═' * 70}\n")

    for msg in messages:
        name = getattr(msg, "name", None) or "system"
        label = role_labels.get(name, name.upper())
        print(f"[{label}]")
        print(msg.content)
        print(separator)


def main() -> None:
    from src.models.client_profile import (
        ClientProfile, RiskTolerance, InvestmentGoal, Holding
    )
    from src.orchestration.graph import build_graph, get_initial_state

    _configure_langsmith()

    demo_client = ClientProfile(
        name="Sarah Chen",
        age=38,
        annual_income_usd=185_000,
        total_assets_usd=620_000,
        total_liabilities_usd=180_000,
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goals=[InvestmentGoal.RETIREMENT, InvestmentGoal.WEALTH_GROWTH],
        investment_horizon_years=25,
        current_holdings=[
            Holding(asset_name="S&P 500 Index Fund", asset_type="ETF", value_usd=180_000, allocation_pct=40.0),
            Holding(asset_name="US Treasury Bonds", asset_type="bond", value_usd=90_000, allocation_pct=20.0),
            Holding(asset_name="Apple Inc.", asset_type="stock", value_usd=67_500, allocation_pct=15.0),
            Holding(asset_name="Real Estate (Rental)", asset_type="real_estate", value_usd=112_500, allocation_pct=25.0),
        ],
        additional_notes="Sarah is a senior software engineer. She is concerned about market volatility and prefers a balanced approach. She has a mortgage on her primary residence.",
    )

    print("\nInitializing multi-agent financial advisor system...")
    print(f"\nClient Profile:\n{demo_client.to_summary()}\n")

    graph = build_graph(demo_client)
    initial_state = get_initial_state(demo_client)

    print("Starting conversation...\n")
    final_state = graph.invoke(initial_state)

    _print_transcript(final_state["messages"])

    if final_state.get("final_summary"):
        print("\nFINAL INVESTMENT SUMMARY")
        print("═" * 70)
        print(final_state["final_summary"])

    turns = final_state.get("turn_count", 0)
    satisfied = final_state.get("is_satisfied", False)
    print(f"\nConversation ended — Turns: {turns} | Client satisfied: {satisfied}")


if __name__ == "__main__":
    main()

