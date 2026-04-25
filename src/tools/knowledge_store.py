import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import get_settings

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")


class KnowledgeRepository:
    """
    Repository pattern wrapper around ChromaDB.
    Loads financial documents, chunks them, embeds via OpenAI, and exposes a query interface.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model="text-embedding-3-small",
        )
        self._persist_dir = settings.chroma_persist_dir
        self._collection_name = settings.chroma_collection_name
        self._vectorstore: Chroma | None = None

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        for txt_file in KNOWLEDGE_BASE_DIR.glob("*.txt"):
            content = txt_file.read_text(encoding="utf-8")
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": txt_file.name, "topic": txt_file.stem},
                )
            )
        return docs

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " "],
        )
        return splitter.split_documents(docs)

    def _try_load_existing(self) -> bool:
        """
        Attempt to open an existing ChromaDB at the persist directory.
        Returns True on success, False if the store is missing or corrupted.
        """
        if not os.path.exists(self._persist_dir):
            return False
        try:
            store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
            # Confirm the collection actually contains data
            if store._collection.count() > 0:
                self._vectorstore = store
                return True
        except Exception:
            pass
        return False

    def initialize(self) -> None:
        """Load documents into ChromaDB. Skips if a valid collection already exists."""
        if self._try_load_existing():
            return

        documents = self._load_documents()
        if not documents:
            raise FileNotFoundError(f"No .txt files found in {KNOWLEDGE_BASE_DIR}")

        chunks = self._chunk_documents(documents)
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            collection_name=self._collection_name,
            persist_directory=self._persist_dir,
        )

    def query(self, question: str, k: int = 4) -> str:
        """
        Retrieve the top-k most relevant chunks for a given question.
        Returns a single concatenated string for the LLM to consume.
        """
        if self._vectorstore is None:
            raise RuntimeError("KnowledgeRepository not initialized. Call initialize() first.")

        results = self._vectorstore.similarity_search(question, k=k)
        if not results:
            return "No relevant information found in the knowledge base."

        formatted = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)

    @property
    def is_initialized(self) -> bool:
        return self._vectorstore is not None
