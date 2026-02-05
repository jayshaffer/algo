"""Tests for trading/tools.py - tool handler functions for Claude ideation agent."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.tools import (
    reset_session,
    tool_get_market_snapshot,
    tool_get_portfolio_state,
    tool_get_active_theses,
    tool_create_thesis,
    tool_update_thesis,
    tool_close_thesis,
    tool_get_macro_context,
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
)
from tests.conftest import make_thesis_row, make_position_row


# ---------------------------------------------------------------------------
# reset_session
# ---------------------------------------------------------------------------

class TestResetSession:

    def test_reset_session_does_not_raise(self):
        """reset_session should execute without errors."""
        reset_session()


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS and TOOL_HANDLERS
# ---------------------------------------------------------------------------

class TestToolDefinitions:

    def test_tool_definitions_is_list(self):
        assert isinstance(TOOL_DEFINITIONS, list)

    def test_tool_definitions_non_empty(self):
        assert len(TOOL_DEFINITIONS) > 0

    def test_each_definition_has_name(self):
        """Every tool definition must have a 'name' field."""
        for defn in TOOL_DEFINITIONS:
            assert "name" in defn, f"Definition missing 'name': {defn}"

    def test_each_definition_has_description(self):
        """Every tool definition must have a 'description' field."""
        for defn in TOOL_DEFINITIONS:
            assert "description" in defn, f"Definition missing 'description': {defn.get('name')}"

    def test_each_definition_has_input_schema(self):
        """Every tool definition must have an 'input_schema' field."""
        for defn in TOOL_DEFINITIONS:
            assert "input_schema" in defn, f"Definition missing 'input_schema': {defn.get('name')}"

    def test_input_schema_has_type_object(self):
        """Every input_schema should be type=object."""
        for defn in TOOL_DEFINITIONS:
            schema = defn["input_schema"]
            assert schema.get("type") == "object", f"Schema type not 'object' for {defn['name']}"

    def test_all_defined_tools_in_handlers(self):
        """Every tool defined in TOOL_DEFINITIONS should have a handler."""
        defined_names = {d["name"] for d in TOOL_DEFINITIONS}
        handler_names = set(TOOL_HANDLERS.keys())
        assert defined_names == handler_names, (
            f"Mismatch: defined={defined_names - handler_names}, "
            f"handlers={handler_names - defined_names}"
        )

    def test_handlers_are_callable(self):
        """Every handler in TOOL_HANDLERS should be callable."""
        for name, handler in TOOL_HANDLERS.items():
            assert callable(handler), f"Handler for '{name}' is not callable"

    def test_expected_tools_present(self):
        """Check that key tools are present."""
        names = {d["name"] for d in TOOL_DEFINITIONS}
        expected = {
            "get_market_snapshot",
            "get_portfolio_state",
            "get_active_theses",
            "create_thesis",
            "update_thesis",
            "close_thesis",
            "get_macro_context",
        }
        assert expected.issubset(names)


# ---------------------------------------------------------------------------
# tool_get_market_snapshot
# ---------------------------------------------------------------------------

class TestToolGetMarketSnapshot:

    @patch("trading.tools.format_market_snapshot", return_value="Market data text")
    @patch("trading.tools.get_market_snapshot")
    def test_returns_formatted_snapshot(self, mock_snap, mock_fmt):
        """Should call get_market_snapshot and format_market_snapshot."""
        mock_snap.return_value = MagicMock()

        result = tool_get_market_snapshot()
        assert result == "Market data text"
        mock_snap.assert_called_once()
        mock_fmt.assert_called_once()

    @patch("trading.tools.get_market_snapshot", side_effect=Exception("API error"))
    def test_returns_error_on_failure(self, mock_snap):
        """Should return error string if snapshot fails."""
        result = tool_get_market_snapshot()
        assert "Error" in result
        assert "API error" in result


# ---------------------------------------------------------------------------
# tool_get_portfolio_state
# ---------------------------------------------------------------------------

class TestToolGetPortfolioState:

    @patch("trading.tools.get_portfolio_context", return_value="Portfolio: $100k")
    @patch("trading.tools.get_account_info", return_value={"cash": 100000})
    def test_returns_portfolio_context(self, mock_acct, mock_ctx):
        result = tool_get_portfolio_state()
        assert result == "Portfolio: $100k"
        mock_acct.assert_called_once()
        mock_ctx.assert_called_once_with({"cash": 100000})

    @patch("trading.tools.get_account_info", side_effect=Exception("No connection"))
    def test_returns_error_on_failure(self, mock_acct):
        result = tool_get_portfolio_state()
        assert "Error" in result


# ---------------------------------------------------------------------------
# tool_get_active_theses
# ---------------------------------------------------------------------------

class TestToolGetActiveTheses:

    @patch("trading.tools.get_active_theses")
    def test_no_theses(self, mock_theses):
        """Should return 'No active theses.' when none exist."""
        mock_theses.return_value = []

        result = tool_get_active_theses()
        assert result == "No active theses."

    @patch("trading.tools.get_active_theses")
    def test_formats_thesis_details(self, mock_theses):
        """Should format thesis with ID, ticker, direction, etc."""
        thesis = make_thesis_row(
            id=5, ticker="NVDA", direction="long", confidence="high",
            thesis="GPU demand strong",
            entry_trigger="pullback to $800",
            exit_trigger="$1100",
            invalidation="revenue drops",
        )
        mock_theses.return_value = [thesis]

        result = tool_get_active_theses()
        assert "ID 5" in result
        assert "NVDA" in result
        assert "long" in result
        assert "high" in result
        assert "GPU demand strong" in result
        assert "pullback to $800" in result

    @patch("trading.tools.get_active_theses")
    def test_filters_by_ticker(self, mock_theses):
        """Should pass ticker filter to get_active_theses."""
        mock_theses.return_value = []

        tool_get_active_theses(ticker="AAPL")
        mock_theses.assert_called_once_with(ticker="AAPL")

    @patch("trading.tools.get_active_theses")
    def test_none_triggers_show_na(self, mock_theses):
        """None entry/exit/invalidation triggers should show N/A."""
        thesis = make_thesis_row(
            entry_trigger=None, exit_trigger=None, invalidation=None
        )
        mock_theses.return_value = [thesis]

        result = tool_get_active_theses()
        assert "N/A" in result


# ---------------------------------------------------------------------------
# tool_create_thesis
# ---------------------------------------------------------------------------

class TestToolCreateThesis:

    @patch("trading.tools.insert_thesis", return_value=42)
    @patch("trading.tools.get_positions", return_value=[])
    @patch("trading.tools.get_active_theses", return_value=[])
    def test_successful_creation(self, mock_theses, mock_pos, mock_insert):
        """Should create thesis and return success message."""
        result = tool_create_thesis(
            ticker="NVDA",
            direction="long",
            thesis="AI demand",
            entry_trigger="Pullback to $800",
            exit_trigger="$1100",
            invalidation="Revenue miss",
            confidence="high",
        )
        assert "Created thesis ID 42" in result
        assert "NVDA" in result
        mock_insert.assert_called_once_with(
            ticker="NVDA",
            direction="long",
            thesis="AI demand",
            entry_trigger="Pullback to $800",
            exit_trigger="$1100",
            invalidation="Revenue miss",
            confidence="high",
            source="claude_ideation",
        )

    @patch("trading.tools.get_active_theses")
    def test_duplicate_ticker_rejected(self, mock_theses):
        """Should reject if active thesis already exists for ticker."""
        mock_theses.return_value = [make_thesis_row(id=10, ticker="AAPL")]

        result = tool_create_thesis(
            ticker="AAPL",
            direction="long",
            thesis="Buy more",
            entry_trigger="now",
            exit_trigger="later",
            invalidation="never",
            confidence="high",
        )
        assert "Error" in result
        assert "Active thesis already exists" in result
        assert "ID 10" in result

    @patch("trading.tools.get_positions")
    @patch("trading.tools.get_active_theses", return_value=[])
    def test_portfolio_conflict_rejected(self, mock_theses, mock_pos):
        """Should reject if ticker is already in the portfolio."""
        mock_pos.return_value = [make_position_row(ticker="TSLA")]

        result = tool_create_thesis(
            ticker="TSLA",
            direction="long",
            thesis="EV growth",
            entry_trigger="dip",
            exit_trigger="moon",
            invalidation="competition",
            confidence="medium",
        )
        assert "Error" in result
        assert "already in the portfolio" in result


# ---------------------------------------------------------------------------
# tool_update_thesis
# ---------------------------------------------------------------------------

class TestToolUpdateThesis:

    @patch("trading.tools.update_thesis", return_value=True)
    def test_successful_update(self, mock_update):
        """Should return success message when update succeeds."""
        result = tool_update_thesis(
            thesis_id=5,
            thesis="Updated reasoning",
            confidence="high",
        )
        assert "Updated thesis ID 5" in result
        mock_update.assert_called_once_with(
            thesis_id=5,
            thesis="Updated reasoning",
            entry_trigger=None,
            exit_trigger=None,
            invalidation=None,
            confidence="high",
        )

    @patch("trading.tools.update_thesis", return_value=False)
    def test_update_not_found(self, mock_update):
        """Should return error when thesis not found."""
        result = tool_update_thesis(thesis_id=999)
        assert "Error" in result
        assert "999" in result

    @patch("trading.tools.update_thesis", return_value=True)
    def test_partial_update(self, mock_update):
        """Should pass only provided fields, leaving others as None."""
        result = tool_update_thesis(thesis_id=1, entry_trigger="new trigger")
        assert "Updated thesis ID 1" in result
        mock_update.assert_called_once_with(
            thesis_id=1,
            thesis=None,
            entry_trigger="new trigger",
            exit_trigger=None,
            invalidation=None,
            confidence=None,
        )


# ---------------------------------------------------------------------------
# tool_close_thesis
# ---------------------------------------------------------------------------

class TestToolCloseThesis:

    @patch("trading.tools.close_thesis", return_value=True)
    def test_successful_close(self, mock_close):
        """Should return success message when close succeeds."""
        result = tool_close_thesis(thesis_id=7, status="invalidated",
                                   reason="Revenue declined")
        assert "Closed thesis ID 7" in result
        assert "invalidated" in result
        mock_close.assert_called_once_with(
            thesis_id=7, status="invalidated", reason="Revenue declined"
        )

    @patch("trading.tools.close_thesis", return_value=False)
    def test_close_not_found(self, mock_close):
        """Should return error when thesis not found."""
        result = tool_close_thesis(thesis_id=999, status="expired",
                                   reason="Old thesis")
        assert "Error" in result
        assert "999" in result

    @patch("trading.tools.close_thesis", return_value=True)
    def test_close_with_expired_status(self, mock_close):
        """Should handle 'expired' status."""
        result = tool_close_thesis(thesis_id=3, status="expired",
                                   reason="Too old without action")
        assert "Closed thesis ID 3" in result
        assert "expired" in result

    @patch("trading.tools.close_thesis", return_value=True)
    def test_close_with_executed_status(self, mock_close):
        """Should handle 'executed' status."""
        result = tool_close_thesis(thesis_id=8, status="executed",
                                   reason="Thesis was acted upon")
        assert "Closed thesis ID 8" in result
        assert "executed" in result


# ---------------------------------------------------------------------------
# tool_get_macro_context
# ---------------------------------------------------------------------------

class TestToolGetMacroContext:

    @patch("trading.tools.get_macro_context", return_value="Macro: Fed steady (bullish)")
    def test_returns_macro_context(self, mock_macro):
        result = tool_get_macro_context(days=14)
        assert result == "Macro: Fed steady (bullish)"
        mock_macro.assert_called_once_with(days=14)

    @patch("trading.tools.get_macro_context", return_value="Macro Context:\n- No significant macro news")
    def test_default_days(self, mock_macro):
        result = tool_get_macro_context()
        mock_macro.assert_called_once_with(days=7)


# ---------------------------------------------------------------------------
# TOOL_HANDLERS mapping
# ---------------------------------------------------------------------------

class TestToolHandlerMapping:

    def test_get_market_snapshot_handler(self):
        assert TOOL_HANDLERS["get_market_snapshot"] is tool_get_market_snapshot

    def test_get_portfolio_state_handler(self):
        assert TOOL_HANDLERS["get_portfolio_state"] is tool_get_portfolio_state

    def test_get_active_theses_handler(self):
        assert TOOL_HANDLERS["get_active_theses"] is tool_get_active_theses

    def test_create_thesis_handler(self):
        assert TOOL_HANDLERS["create_thesis"] is tool_create_thesis

    def test_update_thesis_handler(self):
        assert TOOL_HANDLERS["update_thesis"] is tool_update_thesis

    def test_close_thesis_handler(self):
        assert TOOL_HANDLERS["close_thesis"] is tool_close_thesis

    def test_get_macro_context_handler(self):
        assert TOOL_HANDLERS["get_macro_context"] is tool_get_macro_context
