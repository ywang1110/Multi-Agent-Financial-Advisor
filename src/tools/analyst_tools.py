from functools import lru_cache

from langchain_core.tools import BaseTool, tool

from src.tools.knowledge_store import KnowledgeRepository
from src.tools.web_search import web_search


@lru_cache(maxsize=1)
def _get_repository() -> KnowledgeRepository:
    """Singleton knowledge repository — initialized once, reused across calls."""
    repo = KnowledgeRepository()
    repo.initialize()
    return repo


@tool
def knowledge_base_search(query: str) -> str:
    """
    Search the internal financial knowledge base for investment principles,
    asset allocation strategies, retirement planning guidelines, risk management
    frameworks, and tax optimization techniques.
    Use this for foundational financial knowledge that does not require real-time data.
    """
    repo = _get_repository()
    return repo.query(query, k=4)


def get_analyst_tools() -> list[BaseTool]:
    """
    Returns all tools available to the Analyst Agent.
    Centralizes tool registration — add new tools here as needed.
    """
    return [web_search, knowledge_base_search]
