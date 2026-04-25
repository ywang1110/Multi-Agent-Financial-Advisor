from langchain_core.tools import tool
from tavily import TavilyClient

from src.config.settings import get_settings


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


@tool
def web_search(query: str) -> str:
    """
    Search the internet for real-time financial information.
    Use this for current market data, recent news, interest rates,
    ETF performance, or any time-sensitive financial topics.
    """
    client = _get_tavily_client()
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    results: list[str] = []

    if response.get("answer"):
        results.append(f"Summary: {response['answer']}")

    for i, result in enumerate(response.get("results", []), 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")
        results.append(f"[{i}] {title}\nURL: {url}\n{content}")

    return "\n\n---\n\n".join(results) if results else "No results found."
