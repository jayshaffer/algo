"""Tests for trading/ideation.py - trade idea generation."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

import pytest

from trading.ideation import (
    ThesisReview,
    NewThesis,
    IdeationResult,
    build_ideation_context,
    run_ideation,
)
from tests.conftest import make_thesis_row, make_position_row


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

        result = build_ideation_context(account_info)
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

        result = build_ideation_context({"cash": 100000})
        assert "ID 5" in result
        assert "NVDA" in result
        assert "GPU demand strong" in result

    @patch("trading.ideation.get_market_snapshot", side_effect=Exception("API fail"))
    @patch("trading.ideation.get_macro_context", return_value="Macro...")
    @patch("trading.ideation.get_active_theses", return_value=[])
    @patch("trading.ideation.get_portfolio_context", return_value="Portfolio...")
    def test_market_snapshot_error_handled(self, mock_portfolio, mock_theses,
                                           mock_macro, mock_snap):
        result = build_ideation_context({"cash": 0})
        assert "Error fetching data" in result


# ---------------------------------------------------------------------------
# run_ideation
# ---------------------------------------------------------------------------

class TestRunIdeation:

    def _run_with_mocks(self, llm_response, positions=None, theses=None):
        """Helper to run ideation with patched dependencies."""
        with patch("trading.ideation.chat_json", return_value=llm_response) as mock_chat, \
             patch("trading.ideation.get_positions", return_value=positions or []), \
             patch("trading.ideation.get_active_theses", return_value=theses or []), \
             patch("trading.ideation.get_account_info", return_value={"cash": 100000, "buying_power": 50000, "portfolio_value": 150000}), \
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
                "thesis": "Strong AI demand",
                "entry_trigger": "Pullback to $800",
                "exit_trigger": "$1100",
                "invalidation": "Revenue drops",
                "confidence": "high",
            }],
            "market_observations": "",
        }
        result, mock_insert, _, _, _ = self._run_with_mocks(llm_response)
        assert result.theses_created == 1
        mock_insert.assert_called_once_with(
            ticker="NVDA",
            direction="long",
            thesis="Strong AI demand",
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
                },
                {
                    "ticker": "AMD",
                    "direction": "short",
                    "thesis": "Second thesis",
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

    def test_account_info_failure_continues(self):
        """If get_account_info fails, ideation should continue with defaults."""
        with patch("trading.ideation.chat_json", return_value={"reviews": [], "new_theses": [], "market_observations": ""}), \
             patch("trading.ideation.get_positions", return_value=[]), \
             patch("trading.ideation.get_active_theses", return_value=[]), \
             patch("trading.ideation.get_account_info", side_effect=Exception("API fail")), \
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
