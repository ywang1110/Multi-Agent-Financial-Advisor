from unittest.mock import MagicMock, patch

import pytest


class TestWebSearch:
    def test_returns_formatted_results(self):
        from src.tools.web_search import web_search

        mock_response = {
            "answer": "Index funds are the best for passive investors.",
            "results": [
                {
                    "title": "Best Index Funds 2025",
                    "url": "https://example.com/index-funds",
                    "content": "VOO and VTI are top picks.",
                }
            ],
        }
        with patch("src.tools.web_search._get_tavily_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.search.return_value = mock_response
            mock_client_fn.return_value = mock_client

            result = web_search.invoke("best index funds for passive investing")

        assert "Index funds are the best" in result
        assert "Best Index Funds" in result
        assert "https://example.com/index-funds" in result

    def test_returns_no_results_message(self):
        from src.tools.web_search import web_search

        mock_response = {"answer": None, "results": []}
        with patch("src.tools.web_search._get_tavily_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.search.return_value = mock_response
            mock_client_fn.return_value = mock_client

            result = web_search.invoke("xyzzy nothing here")

        assert result == "No results found."

    def test_calls_tavily_with_advanced_depth(self):
        from src.tools.web_search import web_search

        mock_response = {"answer": "Test answer", "results": []}
        with patch("src.tools.web_search._get_tavily_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.search.return_value = mock_response
            mock_client_fn.return_value = mock_client

            web_search.invoke("S&P 500 performance")

            mock_client.search.assert_called_once_with(
                query="S&P 500 performance",
                search_depth="advanced",
                max_results=5,
                include_answer=True,
            )


class TestKnowledgeBaseSearch:
    def test_returns_query_results(self):
        from src.tools.analyst_tools import knowledge_base_search

        with patch("src.tools.analyst_tools._get_repository") as mock_repo_fn:
            mock_repo = MagicMock()
            mock_repo.query.return_value = "Asset allocation involves stocks and bonds."
            mock_repo_fn.return_value = mock_repo

            result = knowledge_base_search.invoke("asset allocation strategies")

        assert "Asset allocation" in result
        mock_repo.query.assert_called_once_with("asset allocation strategies", k=4)

    def test_raises_when_not_initialized(self):
        from src.tools.knowledge_store import KnowledgeRepository

        repo = KnowledgeRepository.__new__(KnowledgeRepository)
        repo._vectorstore = None
        with pytest.raises(RuntimeError, match="not initialized"):
            repo.query("test query")


class TestGetAnalystTools:
    def test_returns_two_tools(self):
        from src.tools.analyst_tools import get_analyst_tools

        tools = get_analyst_tools()
        assert len(tools) == 2

    def test_tool_names(self):
        from src.tools.analyst_tools import get_analyst_tools

        tools = get_analyst_tools()
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "knowledge_base_search" in names
