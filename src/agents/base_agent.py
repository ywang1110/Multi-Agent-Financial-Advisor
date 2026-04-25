from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI

from src.config.settings import get_settings


class BaseAgent(ABC):
    """
    Abstract base for all agents.
    Handles LLM initialization so subclasses stay focused on their role.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )

    @property
    def llm(self) -> ChatOpenAI:
        return self._llm

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Execute agent logic. Receives LangGraph state, returns state updates."""
        pass
