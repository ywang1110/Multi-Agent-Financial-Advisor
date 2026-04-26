from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.config.settings import get_settings

_TAVILY_MAX_QUERY_LENGTH = 400


def _get_tavily_client() -> TavilyClient:
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


def _make_concise(query: str) -> str:
    """Compress a verbose query into a concise search phrase using the LLM."""
    settings = get_settings()
    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)
    prompt = (
        f"Rewrite the following into a concise financial search query under 300 characters. "
        f"Preserve all key financial terms and intent. Output only the rewritten query, nothing else.\n\n"
        f"Original: {query}"
    )
    return llm.invoke(prompt).content.strip()


@tool
def web_search(query: str) -> str:
    """
    Search the internet for real-time financial information.
    Use this for current market data, recent news, interest rates,
    ETF performance, or any time-sensitive financial topics.
    """
    if len(query) > _TAVILY_MAX_QUERY_LENGTH:
        query = _make_concise(query)

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
