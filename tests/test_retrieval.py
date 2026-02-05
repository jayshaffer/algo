"""Tests for trading/retrieval.py - RAG vector search with time decay."""

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch, MagicMock, call

import pytest

from trading.retrieval import (
    retrieve_by_ticker,
    retrieve_by_query,
    retrieve_for_ideation,
    HALF_LIFE_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_row(**kwargs):
    """Create a document dict like what the DB returns."""
    defaults = {
        "id": 1,
        "content": "Sample document content",
        "doc_type": "news",
        "source": "alpaca",
        "source_url": "https://example.com/article",
        "published_at": datetime(2025, 1, 15),
        "recency_score": 0.85,
    }
    defaults.update(kwargs)
    return defaults


def _make_query_doc_row(**kwargs):
    """Create a query result doc row with similarity scores."""
    defaults = {
        "id": 1,
        "content": "Sample document content",
        "ticker": "AAPL",
        "doc_type": "news",
        "source": "alpaca",
        "source_url": "https://example.com/article",
        "published_at": datetime(2025, 1, 15),
        "similarity": 0.92,
        "recency_score": 0.85,
        "final_score": 0.78,
    }
    defaults.update(kwargs)
    return defaults


@pytest.fixture
def mock_retrieval_cursor():
    """Mock cursor for retrieval tests with get_cursor patched."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []

    @contextmanager
    def _get_cursor():
        yield mock_cursor

    patcher = patch("trading.retrieval.get_cursor", _get_cursor)
    patcher.start()
    yield mock_cursor
    patcher.stop()


@pytest.fixture
def mock_retrieval_embed():
    """Mock embed for retrieval tests."""
    with patch("trading.retrieval.embed", return_value=[0.1] * 768) as m:
        yield m


# ---------------------------------------------------------------------------
# retrieve_by_ticker
# ---------------------------------------------------------------------------

class TestRetrieveByTicker:
    """Tests for retrieve_by_ticker()."""

    def test_returns_documents(self, mock_retrieval_cursor):
        expected = [_make_doc_row(), _make_doc_row(id=2)]
        mock_retrieval_cursor.fetchall.return_value = expected

        result = retrieve_by_ticker("AAPL")
        assert result == expected

    def test_sql_contains_ticker_filter(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "ticker = %s" in sql

    def test_sql_contains_recency_score(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "recency_score" in sql
        assert "EXP" in sql

    def test_passes_half_life_days(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert HALF_LIFE_DAYS in params

    def test_passes_ticker_in_params(self, mock_retrieval_cursor):
        retrieve_by_ticker("TSLA")

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert "TSLA" in params

    def test_default_k_is_10(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert 10 in params  # LIMIT param

    def test_custom_k(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL", k=5)

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert 5 in params

    def test_default_days_filter(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert 180 in params

    def test_custom_days_filter(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL", days=30)

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert 30 in params

    def test_without_doc_type_filter(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "doc_type IN" not in sql

    def test_with_doc_type_filter(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL", doc_types=["news", "filing_10k"])

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "doc_type IN" in sql

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert "news" in params
        assert "filing_10k" in params

    def test_single_doc_type_filter(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL", doc_types=["news"])

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "doc_type IN (%s)" in sql

    def test_multiple_doc_type_placeholders(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL", doc_types=["news", "filing_10k", "filing_10q"])

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "doc_type IN (%s, %s, %s)" in sql

    def test_sql_orders_by_recency(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "ORDER BY recency_score DESC" in sql

    def test_sql_has_limit(self, mock_retrieval_cursor):
        retrieve_by_ticker("AAPL")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "LIMIT %s" in sql

    def test_returns_empty_list_when_no_docs(self, mock_retrieval_cursor):
        mock_retrieval_cursor.fetchall.return_value = []
        result = retrieve_by_ticker("AAPL")
        assert result == []


# ---------------------------------------------------------------------------
# retrieve_by_query
# ---------------------------------------------------------------------------

class TestRetrieveByQuery:
    """Tests for retrieve_by_query() - semantic search."""

    def test_calls_embed_with_query(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("AI semiconductor demand")

        mock_retrieval_embed.assert_called_once_with("AI semiconductor demand")

    def test_returns_documents(self, mock_retrieval_cursor, mock_retrieval_embed):
        expected = [_make_query_doc_row()]
        mock_retrieval_cursor.fetchall.return_value = expected

        result = retrieve_by_query("test query")
        assert result == expected

    def test_sql_contains_vector_similarity(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "<=>" in sql  # pgvector cosine distance operator
        assert "similarity" in sql

    def test_sql_contains_time_decay(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "EXP" in sql
        assert "recency_score" in sql

    def test_sql_contains_final_score(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "final_score" in sql
        assert "ORDER BY final_score DESC" in sql

    def test_without_ticker_filter(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "ticker IN" not in sql

    def test_with_ticker_filter(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query", tickers=["AAPL", "TSLA"])

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "ticker IN" in sql

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert "AAPL" in params
        assert "TSLA" in params

    def test_without_doc_type_filter(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "doc_type IN" not in sql

    def test_with_doc_type_filter(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query", doc_types=["news"])

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "doc_type IN" in sql

    def test_default_params(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query")

        params = mock_retrieval_cursor.execute.call_args[0][1]
        # Should include: query_embedding, HALF_LIFE_DAYS, query_embedding, HALF_LIFE_DAYS, days, k
        assert params.count(HALF_LIFE_DAYS) == 2  # Used twice in the query
        assert 180 in params  # default days
        assert 10 in params  # default k

    def test_custom_k_and_days(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query", k=20, days=30)

        params = mock_retrieval_cursor.execute.call_args[0][1]
        assert 20 in params
        assert 30 in params

    def test_embedding_not_null_filter(self, mock_retrieval_cursor, mock_retrieval_embed):
        retrieve_by_query("test query")

        sql = mock_retrieval_cursor.execute.call_args[0][0]
        assert "embedding IS NOT NULL" in sql

    def test_passes_embedding_to_query(self, mock_retrieval_cursor, mock_retrieval_embed):
        mock_retrieval_embed.return_value = [0.5] * 768

        retrieve_by_query("test query")

        params = mock_retrieval_cursor.execute.call_args[0][1]
        # The embedding vector should appear in the params
        assert [0.5] * 768 in params


# ---------------------------------------------------------------------------
# retrieve_for_ideation
# ---------------------------------------------------------------------------

class TestRetrieveForIdeation:
    """Tests for retrieve_for_ideation() - combined retrieval."""

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_returns_combined_structure(self, mock_by_ticker, mock_by_query):
        mock_by_ticker.return_value = [_make_doc_row()]
        mock_by_query.return_value = [_make_query_doc_row()]

        result = retrieve_for_ideation(
            tickers=["AAPL"],
            themes=["AI demand"],
        )

        assert "by_ticker" in result
        assert "by_theme" in result
        assert "AAPL" in result["by_ticker"]
        assert "AI demand" in result["by_theme"]

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_multiple_tickers(self, mock_by_ticker, mock_by_query):
        mock_by_ticker.return_value = [_make_doc_row()]
        mock_by_query.return_value = []

        result = retrieve_for_ideation(
            tickers=["AAPL", "TSLA", "MSFT"],
            themes=[],
        )

        assert mock_by_ticker.call_count == 3
        assert "AAPL" in result["by_ticker"]
        assert "TSLA" in result["by_ticker"]
        assert "MSFT" in result["by_ticker"]

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_multiple_themes(self, mock_by_ticker, mock_by_query):
        mock_by_query.return_value = [_make_query_doc_row()]

        result = retrieve_for_ideation(
            tickers=[],
            themes=["AI demand", "Fed rate cuts"],
        )

        assert mock_by_query.call_count == 2
        assert "AI demand" in result["by_theme"]
        assert "Fed rate cuts" in result["by_theme"]

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_excludes_empty_results(self, mock_by_ticker, mock_by_query):
        mock_by_ticker.side_effect = [[], [_make_doc_row()]]
        mock_by_query.return_value = []

        result = retrieve_for_ideation(
            tickers=["AAPL", "TSLA"],
            themes=["rate cuts"],
        )

        # AAPL returned empty, so should not be in by_ticker
        assert "AAPL" not in result["by_ticker"]
        assert "TSLA" in result["by_ticker"]
        # rate cuts returned empty, so should not be in by_theme
        assert "rate cuts" not in result["by_theme"]

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_passes_k_per_ticker(self, mock_by_ticker, mock_by_query):
        mock_by_ticker.return_value = [_make_doc_row()]
        mock_by_query.return_value = []

        retrieve_for_ideation(
            tickers=["AAPL"],
            themes=[],
            k_per_ticker=15,
        )

        mock_by_ticker.assert_called_once_with("AAPL", k=15)

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_passes_k_per_theme(self, mock_by_ticker, mock_by_query):
        mock_by_query.return_value = [_make_query_doc_row()]

        retrieve_for_ideation(
            tickers=[],
            themes=["AI demand"],
            k_per_theme=20,
        )

        mock_by_query.assert_called_once_with("AI demand", k=20)

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_empty_tickers_and_themes(self, mock_by_ticker, mock_by_query):
        result = retrieve_for_ideation(tickers=[], themes=[])

        assert result == {"by_ticker": {}, "by_theme": {}}
        mock_by_ticker.assert_not_called()
        mock_by_query.assert_not_called()

    @patch("trading.retrieval.retrieve_by_query")
    @patch("trading.retrieval.retrieve_by_ticker")
    def test_default_k_values(self, mock_by_ticker, mock_by_query):
        mock_by_ticker.return_value = [_make_doc_row()]
        mock_by_query.return_value = [_make_query_doc_row()]

        retrieve_for_ideation(
            tickers=["AAPL"],
            themes=["AI demand"],
        )

        mock_by_ticker.assert_called_once_with("AAPL", k=5)
        mock_by_query.assert_called_once_with("AI demand", k=5)
