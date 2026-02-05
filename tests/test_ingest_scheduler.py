"""Tests for trading/ingest_scheduler.py - Scheduled document ingestion."""

from datetime import datetime
from unittest.mock import patch, MagicMock, call

import pytest

from tests.conftest import make_position_row, make_thesis_row


# ---------------------------------------------------------------------------
# Patch targets (all within trading.ingest_scheduler)
# ---------------------------------------------------------------------------

PATCH_GET_POSITIONS = "trading.ingest_scheduler.get_positions"
PATCH_GET_ACTIVE_THESES = "trading.ingest_scheduler.get_active_theses"
PATCH_GET_DEFAULT_UNIVERSE = "trading.ingest_scheduler.get_default_universe"
PATCH_INGEST_NEWS = "trading.ingest_scheduler.ingest_alpaca_news"
PATCH_INGEST_FILINGS = "trading.ingest_scheduler.ingest_sec_filings"
PATCH_CLEANUP = "trading.ingest_scheduler.cleanup_old_documents"
PATCH_DOC_STATS = "trading.ingest_scheduler.get_document_stats"


@pytest.fixture
def mock_ticker_sources():
    """Mock all ticker sources for get_tickers_to_ingest."""
    with patch(PATCH_GET_DEFAULT_UNIVERSE) as m_universe, \
         patch(PATCH_GET_POSITIONS) as m_positions, \
         patch(PATCH_GET_ACTIVE_THESES) as m_theses:
        m_universe.return_value = ["AAPL", "MSFT", "GOOGL"]
        m_positions.return_value = [
            make_position_row(ticker="TSLA"),
            make_position_row(ticker="AAPL", id=2),
        ]
        m_theses.return_value = [
            make_thesis_row(ticker="NVDA"),
            make_thesis_row(ticker="MSFT", id=2),
        ]
        yield {
            "universe": m_universe,
            "positions": m_positions,
            "theses": m_theses,
        }


@pytest.fixture
def mock_ingestion_deps(mock_ticker_sources):
    """Mock all dependencies needed for run_ingestion."""
    with patch(PATCH_INGEST_NEWS) as m_news, \
         patch(PATCH_INGEST_FILINGS) as m_filings, \
         patch(PATCH_CLEANUP) as m_cleanup, \
         patch(PATCH_DOC_STATS) as m_stats:
        m_news.return_value = 25
        m_filings.return_value = 3
        m_cleanup.return_value = 10
        m_stats.return_value = {
            "total_documents": 500,
            "unique_tickers": 20,
            "by_type": {},
        }
        yield {
            **mock_ticker_sources,
            "ingest_news": m_news,
            "ingest_filings": m_filings,
            "cleanup": m_cleanup,
            "doc_stats": m_stats,
        }


# ---------------------------------------------------------------------------
# get_tickers_to_ingest
# ---------------------------------------------------------------------------


class TestGetTickersToIngest:
    """Tests for get_tickers_to_ingest()."""

    def test_combines_all_sources(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        result = get_tickers_to_ingest()

        # Default universe: AAPL, MSFT, GOOGL
        # Positions: TSLA, AAPL (dup)
        # Theses: NVDA, MSFT (dup)
        assert set(result) == {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"}

    def test_deduplicates_tickers(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        result = get_tickers_to_ingest()

        # AAPL appears in universe and positions; MSFT in universe and theses
        assert result.count("AAPL") == 1
        assert result.count("MSFT") == 1

    def test_returns_sorted(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        result = get_tickers_to_ingest()

        assert result == sorted(result)

    def test_empty_positions(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        mock_ticker_sources["positions"].return_value = []

        result = get_tickers_to_ingest()
        # Should still have universe + theses
        assert "AAPL" in result
        assert "NVDA" in result
        assert "TSLA" not in result

    def test_empty_theses(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        mock_ticker_sources["theses"].return_value = []

        result = get_tickers_to_ingest()
        assert "NVDA" not in result
        assert "AAPL" in result
        assert "TSLA" in result

    def test_empty_universe(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        mock_ticker_sources["universe"].return_value = []

        result = get_tickers_to_ingest()
        # Only positions and theses
        assert set(result) == {"AAPL", "TSLA", "NVDA", "MSFT"}

    def test_all_sources_empty(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        mock_ticker_sources["universe"].return_value = []
        mock_ticker_sources["positions"].return_value = []
        mock_ticker_sources["theses"].return_value = []

        result = get_tickers_to_ingest()
        assert result == []

    def test_calls_all_sources(self, mock_ticker_sources):
        from trading.ingest_scheduler import get_tickers_to_ingest

        get_tickers_to_ingest()

        mock_ticker_sources["universe"].assert_called_once()
        mock_ticker_sources["positions"].assert_called_once()
        mock_ticker_sources["theses"].assert_called_once()


# ---------------------------------------------------------------------------
# run_ingestion
# ---------------------------------------------------------------------------


class TestRunIngestion:
    """Tests for run_ingestion()."""

    def test_full_cycle(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        results = run_ingestion(news_days=3, filings=True, cleanup=True)

        assert results["news_ingested"] == 25
        assert results["filings_ingested"] > 0
        assert results["documents_cleaned"] == 10
        assert results["errors"] == []
        assert "timestamp" in results

    def test_skip_filings(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        results = run_ingestion(filings=False)

        mock_ingestion_deps["ingest_filings"].assert_not_called()
        assert results["filings_ingested"] == 0

    def test_skip_cleanup(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        results = run_ingestion(cleanup=False)

        mock_ingestion_deps["cleanup"].assert_not_called()
        assert results["documents_cleaned"] == 0

    def test_skip_both(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        results = run_ingestion(filings=False, cleanup=False)

        mock_ingestion_deps["ingest_filings"].assert_not_called()
        mock_ingestion_deps["cleanup"].assert_not_called()
        assert results["filings_ingested"] == 0
        assert results["documents_cleaned"] == 0
        # News should still run
        mock_ingestion_deps["ingest_news"].assert_called_once()

    def test_news_error_captured(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        mock_ingestion_deps["ingest_news"].side_effect = ConnectionError("API down")

        results = run_ingestion()

        assert results["news_ingested"] == 0
        assert len(results["errors"]) == 1
        assert "News ingestion failed" in results["errors"][0]
        assert "API down" in results["errors"][0]

    def test_filing_error_captured(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        # Make filing ingestion fail for one ticker
        call_count = [0]

        def _side_effect(ticker):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("SEC rate limit")
            return 1

        mock_ingestion_deps["ingest_filings"].side_effect = _side_effect

        results = run_ingestion(filings=True)

        assert len(results["errors"]) == 1
        assert "SEC filing" in results["errors"][0]
        assert "SEC rate limit" in results["errors"][0]
        # Other tickers should still have contributed
        assert results["filings_ingested"] >= 1

    def test_cleanup_error_captured(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        mock_ingestion_deps["cleanup"].side_effect = RuntimeError("Cleanup failed")

        results = run_ingestion(cleanup=True)

        assert results["documents_cleaned"] == 0
        assert any("Cleanup failed" in e for e in results["errors"])

    def test_custom_news_days(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        run_ingestion(news_days=7)

        # The tickers list is sorted and deduped from fixtures
        tickers = sorted({"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"})
        mock_ingestion_deps["ingest_news"].assert_called_once_with(tickers, days=7)

    def test_custom_retention_days(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        run_ingestion(retention_days=30)

        mock_ingestion_deps["cleanup"].assert_called_once_with(30)

    def test_max_filing_tickers_limit(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        # With 5 tickers total, setting max_filing_tickers=2 should only call
        # ingest_sec_filings for the first 2 tickers (alphabetical order).
        results = run_ingestion(max_filing_tickers=2)

        assert mock_ingestion_deps["ingest_filings"].call_count == 2

    def test_max_filing_tickers_larger_than_list(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        # 5 tickers total, max=100 should process all 5
        results = run_ingestion(max_filing_tickers=100)

        assert mock_ingestion_deps["ingest_filings"].call_count == 5

    def test_returns_timestamp(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        results = run_ingestion()

        # Verify timestamp is a valid ISO format string
        ts = results["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)

    def test_doc_stats_called(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        run_ingestion()

        mock_ingestion_deps["doc_stats"].assert_called_once()

    def test_all_filings_error_accumulates(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        mock_ingestion_deps["ingest_filings"].side_effect = RuntimeError("fail")

        results = run_ingestion(filings=True)

        # Should have 5 errors (one per ticker)
        filing_errors = [e for e in results["errors"] if "SEC filing" in e]
        assert len(filing_errors) == 5
        assert results["filings_ingested"] == 0

    def test_multiple_error_types(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        mock_ingestion_deps["ingest_news"].side_effect = ConnectionError("news fail")
        mock_ingestion_deps["ingest_filings"].side_effect = RuntimeError("filing fail")
        mock_ingestion_deps["cleanup"].side_effect = RuntimeError("cleanup fail")

        results = run_ingestion()

        # News error + 5 filing errors + cleanup error = 7 total
        assert len(results["errors"]) == 7
        assert results["news_ingested"] == 0
        assert results["filings_ingested"] == 0
        assert results["documents_cleaned"] == 0

    def test_filings_ingested_sums_across_tickers(self, mock_ingestion_deps):
        from trading.ingest_scheduler import run_ingestion

        # Each ticker returns 2 filings
        mock_ingestion_deps["ingest_filings"].return_value = 2

        results = run_ingestion()

        # 5 tickers * 2 filings each = 10
        assert results["filings_ingested"] == 10
