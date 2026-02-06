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
    tool_get_news_signals,
    tool_create_thesis,
    tool_update_thesis,
    tool_close_thesis,
    tool_get_macro_context,
    tool_get_signal_attribution,
    tool_get_decision_history,
    tool_write_playbook,
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
)
from tests.conftest import (
    make_thesis_row,
    make_position_row,
    make_news_signal_row,
    make_attribution_row,
    make_decision_row,
)


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

    def _client_tools(self):
        """Return only client-side tool definitions (exclude server-side like web_search)."""
        return [d for d in TOOL_DEFINITIONS if "type" not in d]

    def _server_tools(self):
        """Return only server-side tool definitions (e.g., web_search)."""
        return [d for d in TOOL_DEFINITIONS if "type" in d]

    def test_each_client_definition_has_description(self):
        """Every client tool definition must have a 'description' field."""
        for defn in self._client_tools():
            assert "description" in defn, f"Definition missing 'description': {defn.get('name')}"

    def test_each_client_definition_has_input_schema(self):
        """Every client tool definition must have an 'input_schema' field."""
        for defn in self._client_tools():
            assert "input_schema" in defn, f"Definition missing 'input_schema': {defn.get('name')}"

    def test_input_schema_has_type_object(self):
        """Every client input_schema should be type=object."""
        for defn in self._client_tools():
            schema = defn["input_schema"]
            assert schema.get("type") == "object", f"Schema type not 'object' for {defn['name']}"

    def test_all_client_tools_in_handlers(self):
        """Every client tool defined in TOOL_DEFINITIONS should have a handler."""
        client_names = {d["name"] for d in self._client_tools()}
        handler_names = set(TOOL_HANDLERS.keys())
        assert client_names == handler_names, (
            f"Mismatch: defined={client_names - handler_names}, "
            f"handlers={handler_names - client_names}"
        )

    def test_server_tools_have_type_field(self):
        """Server-side tools should have a 'type' field."""
        for defn in self._server_tools():
            assert "type" in defn
            assert "name" in defn

    def test_web_search_tool_present(self):
        """Web search server-side tool should be in TOOL_DEFINITIONS."""
        server_names = {d["name"] for d in self._server_tools()}
        assert "web_search" in server_names

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
            "get_news_signals",
            "create_thesis",
            "update_thesis",
            "close_thesis",
            "get_macro_context",
            "get_signal_attribution",
            "get_decision_history",
            "write_playbook",
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
# tool_get_news_signals
# ---------------------------------------------------------------------------

class TestToolGetNewsSignals:

    @patch("trading.tools.get_news_signals", return_value=[])
    def test_no_signals_unfiltered(self, mock_signals):
        result = tool_get_news_signals()
        assert "No news signals" in result
        mock_signals.assert_called_once_with(ticker=None, days=7)

    @patch("trading.tools.get_news_signals", return_value=[])
    def test_no_signals_for_ticker(self, mock_signals):
        result = tool_get_news_signals(ticker="AAPL")
        assert "No news signals for AAPL" in result
        mock_signals.assert_called_once_with(ticker="AAPL", days=7)

    @patch("trading.tools.get_news_signals")
    def test_formats_signal_details(self, mock_signals):
        signal = make_news_signal_row(
            ticker="NVDA", headline="NVDA beats Q3 estimates",
            category="earnings", sentiment="bullish", confidence="high",
        )
        mock_signals.return_value = [signal]

        result = tool_get_news_signals()
        assert "NVDA" in result
        assert "earnings" in result
        assert "bullish" in result
        assert "high" in result
        assert "NVDA beats Q3 estimates" in result

    @patch("trading.tools.get_news_signals")
    def test_custom_days(self, mock_signals):
        mock_signals.return_value = []
        tool_get_news_signals(days=3)
        mock_signals.assert_called_once_with(ticker=None, days=3)

    @patch("trading.tools.get_news_signals")
    def test_truncates_long_headlines(self, mock_signals):
        long_headline = "A" * 100
        signal = make_news_signal_row(headline=long_headline)
        mock_signals.return_value = [signal]

        result = tool_get_news_signals()
        assert "..." in result
        assert long_headline not in result

    @patch("trading.tools.get_news_signals")
    def test_multiple_signals(self, mock_signals):
        signals = [
            make_news_signal_row(id=1, ticker="AAPL", sentiment="bullish"),
            make_news_signal_row(id=2, ticker="TSLA", sentiment="bearish"),
        ]
        mock_signals.return_value = signals

        result = tool_get_news_signals()
        assert "AAPL" in result
        assert "TSLA" in result
        assert "bullish" in result
        assert "bearish" in result


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

    def test_get_news_signals_handler(self):
        assert TOOL_HANDLERS["get_news_signals"] is tool_get_news_signals

    def test_get_macro_context_handler(self):
        assert TOOL_HANDLERS["get_macro_context"] is tool_get_macro_context

    def test_get_signal_attribution_handler(self):
        assert TOOL_HANDLERS["get_signal_attribution"] is tool_get_signal_attribution

    def test_get_decision_history_handler(self):
        assert TOOL_HANDLERS["get_decision_history"] is tool_get_decision_history

    def test_write_playbook_handler(self):
        assert TOOL_HANDLERS["write_playbook"] is tool_write_playbook


# ---------------------------------------------------------------------------
# New strategist tools
# ---------------------------------------------------------------------------

class TestNewStrategistTools:

    @patch("trading.tools.get_attribution_summary")
    def test_get_signal_attribution_tool(self, mock_attr):
        """Should return attribution summary."""
        mock_attr.return_value = "Signal Attribution:\n- news:earnings — 20 samples, 7d avg +1.50%, win rate 62%"
        result = tool_get_signal_attribution()
        assert "earnings" in result or "Attribution" in result
        mock_attr.assert_called_once()

    @patch("trading.tools.get_recent_decisions")
    def test_get_decision_history_tool(self, mock_decisions):
        """Should format recent decisions with outcomes."""
        mock_decisions.return_value = [make_decision_row()]
        result = tool_get_decision_history(days=30)
        assert "AAPL" in result
        assert "BUY" in result
        mock_decisions.assert_called_once_with(days=30)

    @patch("trading.tools.get_recent_decisions")
    def test_get_decision_history_empty(self, mock_decisions):
        """Should return message when no decisions."""
        mock_decisions.return_value = []
        result = tool_get_decision_history(days=30)
        assert "No decisions" in result

    @patch("trading.tools.upsert_playbook", return_value=42)
    def test_write_playbook_tool(self, mock_upsert):
        """Should write playbook and return confirmation."""
        result = tool_write_playbook(
            market_outlook="Bullish tech",
            priority_actions=[{"ticker": "NVDA", "action": "buy", "reasoning": "test", "confidence": 0.8}],
            watch_list=["AAPL"],
            risk_notes="Watch Fed",
        )
        assert "Playbook written" in result
        mock_upsert.assert_called_once()

    def test_tool_definitions_include_new_tools(self):
        """TOOL_DEFINITIONS should include the 3 new tools."""
        names = [t.get("name") for t in TOOL_DEFINITIONS if isinstance(t, dict) and "name" in t]
        assert "get_signal_attribution" in names
        assert "get_decision_history" in names
        assert "write_playbook" in names

    def test_tool_handlers_include_new_tools(self):
        """TOOL_HANDLERS should include the 3 new handlers."""
        assert "get_signal_attribution" in TOOL_HANDLERS
        assert "get_decision_history" in TOOL_HANDLERS
        assert "write_playbook" in TOOL_HANDLERS
