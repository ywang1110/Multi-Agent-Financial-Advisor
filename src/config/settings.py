from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(..., description="OpenAI API key")
    llm_model: str = Field(default="gpt-4.1-mini", description="LLM model name")
    llm_temperature: float = Field(default=0.3, description="LLM temperature")

    # Tavily web search
    tavily_api_key: str = Field(..., description="Tavily search API key")

    # ChromaDB knowledge store
    chroma_persist_dir: str = Field(
        default="data/chroma_db", description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="financial_knowledge", description="ChromaDB collection name"
    )

    # LangSmith tracing
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_project: str = Field(
        default="jpmc-investment-advisor", description="LangSmith project name"
    )

    # Conversation guardrails
    max_conversation_turns: int = Field(
        default=10, description="Maximum conversation turns before forced termination"
    )


def get_settings() -> Settings:
    return Settings()
