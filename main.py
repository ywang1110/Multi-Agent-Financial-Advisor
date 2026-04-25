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
    from src.models.client_profile import DEMO_CLIENT
    from src.orchestration.graph import build_graph, get_initial_state

    _configure_langsmith()

    print("\nInitializing multi-agent financial advisor system...")
    print(f"\nClient Profile:\n{DEMO_CLIENT.to_summary()}\n")

    graph = build_graph(DEMO_CLIENT)
    initial_state = get_initial_state(DEMO_CLIENT)

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

