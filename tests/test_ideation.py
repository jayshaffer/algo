"""Tests for trading/ideation.py - trade idea generation with RAG."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

import pytest

from trading.ideation import (
    MAX_DOC_CONTENT_LENGTH,
    ThesisReview,
    NewThesis,
    IdeationResult,
    format_retrieved_context,
    build_ideation_context,
    run_ideation,
)
from tests.conftest import make_thesis_row, make_position_row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id=1, ticker="AAPL", doc_type="news", content="Some content",
              published_at=None):
    """Create a mock retrieved document dict."""
    return {
        "id": doc_id,
        "ticker": ticker,
        "doc_type": doc_type,
        "content": content,
        "published_at": published_at or datetime.now() - timedelta(days=2),
    }


def _ideation_patches():
    """Return a dict of common patches for run_ideation tests."""
    return {
        "chat_json": patch("trading.ideation.chat_json"),
        "get_positions": patch("trading.ideation.get_positions"),
        "get_active_theses": patch("trading.ideation.get_active_theses"),
        "get_account_info": patch("trading.ideation.get_account_info"),
        "retrieve_for_ideation": patch("trading.ideation.retrieve_for_ideation"),
        "get_market_snapshot": patch("trading.ideation.get_market_snapshot"),
        "format_market_snapshot": patch("trading.ideation.format_market_snapshot"),
        "insert_thesis": patch("trading.ideation.insert_thesis"),
        "update_thesis": patch("trading.ideation.update_thesis"),
        "close_thesis": patch("trading.ideation.close_thesis"),
        "get_portfolio_context": patch("trading.ideation.get_portfolio_context"),
        "get_macro_context": patch("trading.ideation.get_macro_context"),
    }


# ---------------------------------------------------------------------------
# format_retrieved_context
# ---------------------------------------------------------------------------

class TestFormatRetrievedContext:

    def test_empty_docs_shows_no_documents_message(self):
        """When no docs are retrieved, a 'no documents' message should appear."""
        result = format_retrieved_context({"by_ticker": {}, "by_theme": {}})
        assert "No documents retrieved" in result

    def test_completely_empty_dict(self):
        """Empty dict (no by_ticker/by_theme keys) should still work."""
        result = format_retrieved_context({})
        assert "No documents retrieved" in result

    def test_docs_by_ticker(self):
        """Documents organized by ticker should show ticker headers and DOC-IDs."""
        doc = _make_doc(doc_id=42, ticker="NVDA", doc_type="10-K", content="GPU sales up")
        retrieved = {"by_ticker": {"NVDA": [doc]}, "by_theme": {}}

        result = format_retrieved_context(retrieved)
        assert "=== NVDA ===" in result
        assert "[DOC-42]" in result
        assert "10-K" in result
        assert "GPU sales up" in result

    def test_docs_by_theme(self):
        """Documents organized by theme should show theme headers."""
        doc = _make_doc(doc_id=99, ticker="MSFT", doc_type="news",
                        content="AI investment surge")
        retrieved = {"by_ticker": {}, "by_theme": {"AI growth": [doc]}}

        result = format_retrieved_context(retrieved)
        assert "=== Theme: AI growth ===" in result
        assert "[DOC-99]" in result
        assert "[MSFT]" in result

    def test_theme_doc_without_ticker(self):
        """Theme docs without a ticker should not show a ticker tag."""
        doc = _make_doc(doc_id=10)
        doc.pop("ticker", None)
        retrieved = {"by_ticker": {}, "by_theme": {"Macro": [doc]}}

        result = format_retrieved_context(retrieved)
        # Should not have a bracketed ticker after DOC-ID
        assert "[DOC-10]" in result

    def test_content_truncation(self):
        """Content longer than MAX_DOC_CONTENT_LENGTH should be truncated with '...'."""
        long_content = "A" * (MAX_DOC_CONTENT_LENGTH + 500)
        doc = _make_doc(content=long_content)
        retrieved = {"by_ticker": {"AAPL": [doc]}, "by_theme": {}}

        result = format_retrieved_context(retrieved)
        # The truncated content + "..." should appear
        assert "A" * MAX_DOC_CONTENT_LENGTH + "..." in result
        # The full content should NOT appear
        assert long_content not in result

    def test_content_not_truncated_when_short(self):
        """Content shorter than MAX_DOC_CONTENT_LENGTH should not be truncated."""
        short_content = "Short content"
        doc = _make_doc(content=short_content)
        retrieved = {"by_ticker": {"AAPL": [doc]}, "by_theme": {}}

        result = format_retrieved_context(retrieved)
        assert short_content in result
        # Should not end with ... for this doc's content
        lines = result.split("\n")
        content_lines = [l for l in lines if "Short content" in l]
        assert len(content_lines) == 1
        assert not content_lines[0].endswith("...")

    def test_empty_ticker_docs_skipped(self):
        """Tickers with empty doc lists should be skipped."""
        retrieved = {"by_ticker": {"AAPL": []}, "by_theme": {}}
        result = format_retrieved_context(retrieved)
        assert "=== AAPL ===" not in result

    def test_empty_theme_docs_skipped(self):
        """Themes with empty doc lists should be skipped."""
        retrieved = {"by_ticker": {}, "by_theme": {"AI": []}}
        result = format_retrieved_context(retrieved)
        assert "=== Theme: AI ===" not in result

    def test_age_days_shown(self):
        """Document age in days should be shown."""
        doc = _make_doc(published_at=datetime.now() - timedelta(days=3))
        retrieved = {"by_ticker": {"AAPL": [doc]}, "by_theme": {}}

        result = format_retrieved_context(retrieved)
        assert "3d ago" in result

    def test_multiple_tickers_and_themes(self):
        """Multiple tickers and themes should all appear."""
        doc1 = _make_doc(doc_id=1, ticker="AAPL")
        doc2 = _make_doc(doc_id=2, ticker="GOOGL")
        doc3 = _make_doc(doc_id=3, ticker="AMZN")
        retrieved = {
            "by_ticker": {"AAPL": [doc1], "GOOGL": [doc2]},
            "by_theme": {"Cloud computing": [doc3]},
        }

        result = format_retrieved_context(retrieved)
        assert "=== AAPL ===" in result
        assert "=== GOOGL ===" in result
        assert "=== Theme: Cloud computing ===" in result


# ---------------------------------------------------------------------------
# build_ideation_context
# ---------------------------------------------------------------------------

class TestBuildIdeationContext:

    @patch("trading.ideation.format_market_snapshot", return_value="Market Data...")
    @patch("trading.ideation.get_market_snapshot")
    @patch("trading.ideation.get_macro_context", return_value="Macro: Fed steady")
    @patch("trading.ideation.get_active_theses", return_value=[])
    @patch("trading.ideation.get_portfolio_context", return_value="Portfolio: $100k")
    def test_combines_all_sections(self, mock_portfolio, mock_theses,
                                   mock_macro, mock_snap, mock_fmt):
        account_info = {"cash": 100000}
        retrieved = {"by_ticker": {}, "by_theme": {}}

        result = build_ideation_context(account_info, retrieved)
        assert "Portfolio: $100k" in result
        assert "Macro: Fed steady" in result
        assert "Active Theses: None" in result

    @patch("trading.ideation.format_market_snapshot", return_value="Market...")
    @patch("trading.ideation.get_market_snapshot")
    @patch("trading.ideation.get_macro_context", return_value="Macro...")
    @patch("trading.ideation.get_active_theses")
    @patch("trading.ideation.get_portfolio_context", return_value="Portfolio...")
    def test_includes_active_theses_details(self, mock_portfolio, mock_theses,
                                             mock_macro, mock_snap, mock_fmt):
        thesis = make_thesis_row(id=5, ticker="NVDA", direction="long",
                                 confidence="high", thesis="GPU demand strong",
                                 entry_trigger="pullback to $800",
                                 exit_trigger="$1100",
                                 invalidation="revenue drops")
        mock_theses.return_value = [thesis]

        result = build_ideation_context({"cash": 100000}, {"by_ticker": {}, "by_theme": {}})
        assert "ID 5" in result
        assert "NVDA" in result
        assert "GPU demand strong" in result

    @patch("trading.ideation.get_market_snapshot", side_effect=Exception("API fail"))
    @patch("trading.ideation.get_macro_context", return_value="Macro...")
    @patch("trading.ideation.get_active_theses", return_value=[])
    @patch("trading.ideation.get_portfolio_context", return_value="Portfolio...")
    def test_market_snapshot_error_handled(self, mock_portfolio, mock_theses,
                                           mock_macro, mock_snap):
        result = build_ideation_context({"cash": 0}, {"by_ticker": {}, "by_theme": {}})
        assert "Error fetching data" in result


# ---------------------------------------------------------------------------
# run_ideation
# ---------------------------------------------------------------------------

class TestRunIdeation:

    def _run_with_mocks(self, llm_response, positions=None, theses=None,
                        retrieved=None):
        """Helper to run ideation with patched dependencies."""
        with patch("trading.ideation.chat_json", return_value=llm_response) as mock_chat, \
             patch("trading.ideation.get_positions", return_value=positions or []), \
             patch("trading.ideation.get_active_theses", return_value=theses or []), \
             patch("trading.ideation.get_account_info", return_value={"cash": 100000, "buying_power": 50000, "portfolio_value": 150000}), \
             patch("trading.ideation.retrieve_for_ideation", return_value=retrieved or {"by_ticker": {}, "by_theme": {}}), \
             patch("trading.ideation.get_market_snapshot"), \
             patch("trading.ideation.format_market_snapshot", return_value="Market..."), \
             patch("trading.ideation.insert_thesis", return_value=99) as mock_insert, \
             patch("trading.ideation.update_thesis") as mock_update, \
             patch("trading.ideation.close_thesis") as mock_close, \
             patch("trading.ideation.get_portfolio_context", return_value="Portfolio..."), \
             patch("trading.ideation.get_macro_context", return_value="Macro..."):
            result = run_ideation()
            return result, mock_insert, mock_update, mock_close, mock_chat

    def test_returns_ideation_result(self):
        """run_ideation should return an IdeationResult."""
        result, *_ = self._run_with_mocks({
            "reviews": [],
            "new_theses": [],
            "market_observations": "Calm market",
        })
        assert isinstance(result, IdeationResult)
        assert result.market_observations == "Calm market"

    def test_review_keep_action(self):
        """Thesis review with 'keep' action increments theses_kept."""
        theses = [make_thesis_row(id=1)]
        llm_response = {
            "reviews": [{"thesis_id": 1, "action": "keep", "reason": "Still valid"}],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, mock_update, mock_close, _ = self._run_with_mocks(
            llm_response, theses=theses
        )
        assert result.theses_kept == 1
        mock_update.assert_not_called()
        mock_close.assert_not_called()

    def test_review_update_action(self):
        """Thesis review with 'update' action calls update_thesis."""
        theses = [make_thesis_row(id=2)]
        llm_response = {
            "reviews": [{
                "thesis_id": 2,
                "action": "update",
                "reason": "New data",
                "updates": {"thesis": "Updated reasoning", "confidence": "high"},
            }],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, mock_update, _, _ = self._run_with_mocks(
            llm_response, theses=theses
        )
        assert result.theses_updated == 1
        mock_update.assert_called_once_with(
            thesis_id=2,
            thesis="Updated reasoning",
            entry_trigger=None,
            exit_trigger=None,
            invalidation=None,
            confidence="high",
        )

    def test_review_invalidate_action(self):
        """Thesis review with 'invalidate' calls close_thesis with 'invalidated'."""
        theses = [make_thesis_row(id=3)]
        llm_response = {
            "reviews": [{
                "thesis_id": 3,
                "action": "invalidate",
                "reason": "Revenue declined",
            }],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, _, mock_close, _ = self._run_with_mocks(
            llm_response, theses=theses
        )
        assert result.theses_closed == 1
        mock_close.assert_called_once_with(
            thesis_id=3,
            status="invalidated",
            reason="Revenue declined",
        )

    def test_review_expire_action(self):
        """Thesis review with 'expire' calls close_thesis with 'expired'."""
        llm_response = {
            "reviews": [{
                "thesis_id": 4,
                "action": "expire",
                "reason": "Too old",
            }],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, _, mock_close, _ = self._run_with_mocks(llm_response)
        assert result.theses_closed == 1
        mock_close.assert_called_once_with(
            thesis_id=4,
            status="expired",
            reason="Too old",
        )

    def test_new_thesis_creation(self):
        """New theses from LLM should be inserted into the database."""
        llm_response = {
            "reviews": [],
            "new_theses": [{
                "ticker": "NVDA",
                "direction": "long",
                "thesis": "Strong AI demand [DOC-1]",
                "entry_trigger": "Pullback to $800",
                "exit_trigger": "$1100",
                "invalidation": "Revenue drops",
                "confidence": "high",
                "sources": [1],
            }],
            "market_observations": "",
        }
        result, mock_insert, _, _, _ = self._run_with_mocks(llm_response)
        assert result.theses_created == 1
        mock_insert.assert_called_once_with(
            ticker="NVDA",
            direction="long",
            thesis="Strong AI demand [DOC-1]",
            entry_trigger="Pullback to $800",
            exit_trigger="$1100",
            invalidation="Revenue drops",
            confidence="high",
            source="ideation",
        )

    def test_skips_portfolio_tickers(self):
        """New theses for tickers already in portfolio should be skipped."""
        positions = [make_position_row(ticker="AAPL")]
        llm_response = {
            "reviews": [],
            "new_theses": [{
                "ticker": "AAPL",
                "direction": "long",
                "thesis": "Buy more AAPL",
                "sources": [],
            }],
            "market_observations": "",
        }
        result, mock_insert, _, _, _ = self._run_with_mocks(
            llm_response, positions=positions
        )
        assert result.theses_created == 0
        mock_insert.assert_not_called()

    def test_skips_existing_thesis_tickers(self):
        """New theses for tickers with existing active theses should be skipped."""
        theses = [make_thesis_row(id=10, ticker="MSFT")]
        llm_response = {
            "reviews": [],
            "new_theses": [{
                "ticker": "MSFT",
                "direction": "short",
                "thesis": "Overvalued",
                "sources": [],
            }],
            "market_observations": "",
        }
        result, mock_insert, _, _, _ = self._run_with_mocks(
            llm_response, theses=theses
        )
        assert result.theses_created == 0
        mock_insert.assert_not_called()

    def test_skips_duplicate_tickers_in_same_run(self):
        """If LLM generates two theses for the same ticker, only the first is created."""
        llm_response = {
            "reviews": [],
            "new_theses": [
                {
                    "ticker": "AMD",
                    "direction": "long",
                    "thesis": "First thesis",
                    "sources": [],
                },
                {
                    "ticker": "AMD",
                    "direction": "short",
                    "thesis": "Second thesis",
                    "sources": [],
                },
            ],
            "market_observations": "",
        }
        result, mock_insert, _, _, _ = self._run_with_mocks(llm_response)
        assert result.theses_created == 1
        assert mock_insert.call_count == 1

    def test_llm_error_returns_empty_result(self):
        """If chat_json raises, run_ideation should return an empty result."""
        with patch("trading.ideation.chat_json", side_effect=Exception("LLM down")), \
             patch("trading.ideation.get_positions", return_value=[]), \
             patch("trading.ideation.get_active_theses", return_value=[]), \
             patch("trading.ideation.get_account_info", return_value={"cash": 0, "buying_power": 0, "portfolio_value": 0}), \
             patch("trading.ideation.retrieve_for_ideation", return_value={"by_ticker": {}, "by_theme": {}}), \
             patch("trading.ideation.get_market_snapshot"), \
             patch("trading.ideation.format_market_snapshot", return_value=""), \
             patch("trading.ideation.get_portfolio_context", return_value=""), \
             patch("trading.ideation.get_macro_context", return_value=""):
            result = run_ideation()

        assert isinstance(result, IdeationResult)
        assert result.theses_created == 0
        assert result.theses_updated == 0
        assert result.theses_closed == 0
        assert "Error" in result.market_observations

    def test_malformed_review_missing_thesis_id(self):
        """Reviews without thesis_id should be skipped."""
        llm_response = {
            "reviews": [{"action": "keep", "reason": "Valid"}],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, _, _, _ = self._run_with_mocks(llm_response)
        assert len(result.reviews) == 0
        assert result.theses_kept == 0

    def test_malformed_review_missing_action(self):
        """Reviews without action should be skipped."""
        llm_response = {
            "reviews": [{"thesis_id": 1, "reason": "Whatever"}],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, _, _, _ = self._run_with_mocks(llm_response)
        assert len(result.reviews) == 0

    def test_malformed_review_invalid_action(self):
        """Reviews with invalid action should be skipped."""
        llm_response = {
            "reviews": [{"thesis_id": 1, "action": "delete", "reason": "Bad"}],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, _, _, _ = self._run_with_mocks(llm_response)
        assert len(result.reviews) == 0

    def test_malformed_thesis_missing_ticker(self):
        """New theses without ticker should be skipped."""
        llm_response = {
            "reviews": [],
            "new_theses": [{"direction": "long", "thesis": "Buy something"}],
            "market_observations": "",
        }
        result, mock_insert, _, _, _ = self._run_with_mocks(llm_response)
        assert result.theses_created == 0
        mock_insert.assert_not_called()

    def test_retrieval_failure_continues(self):
        """If retrieve_for_ideation fails, ideation should continue with empty docs."""
        with patch("trading.ideation.chat_json", return_value={"reviews": [], "new_theses": [], "market_observations": ""}), \
             patch("trading.ideation.get_positions", return_value=[]), \
             patch("trading.ideation.get_active_theses", return_value=[]), \
             patch("trading.ideation.get_account_info", return_value={"cash": 0, "buying_power": 0, "portfolio_value": 0}), \
             patch("trading.ideation.retrieve_for_ideation", side_effect=Exception("DB down")), \
             patch("trading.ideation.get_market_snapshot"), \
             patch("trading.ideation.format_market_snapshot", return_value=""), \
             patch("trading.ideation.get_portfolio_context", return_value=""), \
             patch("trading.ideation.get_macro_context", return_value=""):
            result = run_ideation()

        assert isinstance(result, IdeationResult)

    def test_account_info_failure_continues(self):
        """If get_account_info fails, ideation should continue with defaults."""
        with patch("trading.ideation.chat_json", return_value={"reviews": [], "new_theses": [], "market_observations": ""}), \
             patch("trading.ideation.get_positions", return_value=[]), \
             patch("trading.ideation.get_active_theses", return_value=[]), \
             patch("trading.ideation.get_account_info", side_effect=Exception("API fail")), \
             patch("trading.ideation.retrieve_for_ideation", return_value={"by_ticker": {}, "by_theme": {}}), \
             patch("trading.ideation.get_market_snapshot"), \
             patch("trading.ideation.format_market_snapshot", return_value=""), \
             patch("trading.ideation.get_portfolio_context", return_value=""), \
             patch("trading.ideation.get_macro_context", return_value=""):
            result = run_ideation()

        assert isinstance(result, IdeationResult)

    def test_update_without_updates_dict_still_counts(self):
        """An update action without updates dict should still increment count."""
        llm_response = {
            "reviews": [{
                "thesis_id": 5,
                "action": "update",
                "reason": "Minor update",
            }],
            "new_theses": [],
            "market_observations": "",
        }
        result, _, mock_update, _, _ = self._run_with_mocks(llm_response)
        assert result.theses_updated == 1
        # update_thesis should not be called when updates is None
        mock_update.assert_not_called()

    def test_new_thesis_defaults(self):
        """Missing optional fields should get defaults."""
        llm_response = {
            "reviews": [],
            "new_theses": [{
                "ticker": "TSLA",
                "thesis": "EV growth",
                "sources": [],
            }],
            "market_observations": "",
        }
        result, mock_insert, _, _, _ = self._run_with_mocks(llm_response)
        assert result.theses_created == 1
        mock_insert.assert_called_once_with(
            ticker="TSLA",
            direction="long",
            thesis="EV growth",
            entry_trigger="",
            exit_trigger="",
            invalidation="",
            confidence="medium",
            source="ideation",
        )
