"""Tests for v2.tools — tool definitions and handlers."""

import pytest
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, MagicMock

from v2.tools import (
    tool_get_market_snapshot,
    tool_get_portfolio_state,
    tool_get_active_theses,
    tool_create_thesis,
    tool_update_thesis,
    tool_close_thesis,
    tool_get_news_signals,
    tool_get_macro_context,
    tool_get_signal_attribution,
    tool_get_decision_history,
    tool_write_playbook,
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
)


# --- write_playbook tests (V3 rewrite) ---


class TestWritePlaybook:
    def test_stores_playbook_actions(self, mock_db, mock_cursor):
        """write_playbook should store rows in playbook_actions table."""
        mock_cursor.fetchone.return_value = {"id": 1}
        result = tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry hit",
                 "confidence": "high", "max_quantity": 5, "thesis_id": 1},
            ],
            watch_list=["MSFT"],
            risk_notes="Fed meeting",
        )
        assert "Playbook written" in result or "playbook" in result.lower()
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("playbook_actions" in c for c in calls)

    def test_rejects_conflicting_actions(self):
        """Should reject buy + sell same ticker."""
        result = tool_write_playbook(
            market_outlook="Mixed",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "Go long", "confidence": "high"},
                {"ticker": "AAPL", "action": "sell", "reasoning": "Trim", "confidence": "medium"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result or "conflict" in result.lower()

    def test_clears_old_actions_on_upsert(self, mock_db, mock_cursor):
        """Should delete old playbook_actions before inserting new ones."""
        mock_cursor.fetchone.return_value = {"id": 1}
        tool_write_playbook(
            market_outlook="Neutral",
            priority_actions=[{"ticker": "AAPL", "action": "hold", "reasoning": "Wait", "confidence": "low"}],
            watch_list=[],
            risk_notes="",
        )
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("DELETE" in c and "playbook_actions" in c for c in calls)

    def test_returns_playbook_id_and_action_count(self, mock_db, mock_cursor):
        """Should return confirmation with playbook ID and action count."""
        mock_cursor.fetchone.return_value = {"id": 42}
        result = tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry hit", "confidence": "high"},
                {"ticker": "MSFT", "action": "buy", "reasoning": "Breakout", "confidence": "medium"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "42" in result
        assert "2 actions" in result

    def test_empty_priority_actions(self, mock_db, mock_cursor):
        """Should handle empty priority actions list."""
        mock_cursor.fetchone.return_value = {"id": 1}
        result = tool_write_playbook(
            market_outlook="Neutral",
            priority_actions=[],
            watch_list=[],
            risk_notes="",
        )
        assert "0 actions" in result

    def test_multiple_tickers_same_action_allowed(self, mock_db, mock_cursor):
        """Should allow buy + buy for different tickers."""
        mock_cursor.fetchone.return_value = {"id": 1}
        result = tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry hit", "confidence": "high"},
                {"ticker": "MSFT", "action": "sell", "reasoning": "Take profit", "confidence": "medium"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" not in result
        assert "conflict" not in result.lower()

    def test_same_ticker_same_action_allowed(self, mock_db, mock_cursor):
        """Should allow duplicate buy+buy for same ticker (not conflicting)."""
        mock_cursor.fetchone.return_value = {"id": 1}
        result = tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry 1", "confidence": "high"},
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry 2", "confidence": "medium"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" not in result
        assert "conflict" not in result.lower()

    def test_actions_inserted_with_priority(self, mock_db, mock_cursor):
        """Each action should be inserted with its priority (1-indexed)."""
        mock_cursor.fetchone.return_value = {"id": 10}
        tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "First", "confidence": "high", "thesis_id": 1, "max_quantity": 5},
                {"ticker": "MSFT", "action": "buy", "reasoning": "Second", "confidence": "medium", "thesis_id": 2, "max_quantity": 3},
            ],
            watch_list=[],
            risk_notes="",
        )
        # Find the INSERT INTO playbook_actions calls
        insert_calls = [
            c for c in mock_cursor.execute.call_args_list
            if "INSERT" in str(c) and "playbook_actions" in str(c)
        ]
        assert len(insert_calls) == 2

        # Check first action has priority=1
        first_args = insert_calls[0][0][1]  # positional args tuple
        assert first_args[-1] == 1  # priority is last param

        # Check second action has priority=2
        second_args = insert_calls[1][0][1]
        assert second_args[-1] == 2


# --- Market snapshot tests ---


class TestGetMarketSnapshot:
    def test_error_handling(self):
        """Should return error string on failure."""
        with patch("v2.tools.get_market_snapshot", side_effect=Exception("API down")):
            result = tool_get_market_snapshot()
            assert "Error" in result
            assert "API down" in result

    def test_success(self):
        """Should format and return snapshot on success."""
        mock_snapshot = MagicMock()
        with patch("v2.tools.get_market_snapshot", return_value=mock_snapshot), \
             patch("v2.tools.format_market_snapshot", return_value="Market data here"):
            result = tool_get_market_snapshot()
            assert result == "Market data here"


# --- Active theses tests ---


class TestGetActiveTheses:
    def test_no_theses(self, mock_db, mock_cursor):
        """Should return 'No active theses' when none exist."""
        mock_cursor.fetchall.return_value = []
        result = tool_get_active_theses()
        assert "No active theses" in result

    def test_with_theses(self, mock_db, mock_cursor):
        """Should format theses when they exist."""
        mock_cursor.fetchall.return_value = [
            {
                "id": 1, "ticker": "AAPL", "direction": "long",
                "confidence": "high", "thesis": "Strong earnings",
                "entry_trigger": "Price > $150", "exit_trigger": "Price > $180",
                "invalidation": "Earnings miss", "created_at": datetime.now(),
            }
        ]
        result = tool_get_active_theses()
        assert "AAPL" in result
        assert "long" in result
        assert "Strong earnings" in result


# --- Create thesis tests ---


class TestCreateThesis:
    def test_duplicate_rejected(self, mock_db, mock_cursor):
        """Should reject if active thesis exists for ticker."""
        # get_active_theses returns an existing thesis
        mock_cursor.fetchall.return_value = [
            {
                "id": 1, "ticker": "AAPL", "direction": "long",
                "confidence": "high", "thesis": "Existing",
                "entry_trigger": None, "exit_trigger": None,
                "invalidation": None, "created_at": datetime.now(),
            }
        ]
        result = tool_create_thesis(
            ticker="AAPL",
            direction="long",
            thesis="New thesis",
            entry_trigger="Price > $150",
            exit_trigger="Price > $180",
            invalidation="Earnings miss",
            confidence="high",
        )
        assert "Error" in result
        assert "already exists" in result

    def test_success(self, mock_db, mock_cursor):
        """Should create thesis when no duplicate and no position."""
        # get_active_theses returns empty
        mock_cursor.fetchall.return_value = []
        # insert_thesis returns new ID
        mock_cursor.fetchone.return_value = {"id": 5}
        result = tool_create_thesis(
            ticker="GOOG",
            direction="long",
            thesis="AI growth",
            entry_trigger="Price > $100",
            exit_trigger="Price > $150",
            invalidation="AI hype fades",
            confidence="high",
        )
        assert "Created thesis ID 5" in result
        assert "GOOG" in result


# --- Tool completeness tests ---


class TestToolCompleteness:
    def test_tool_handlers_dict_complete(self):
        """TOOL_HANDLERS should have all 14 handler functions."""
        expected_handlers = {
            "get_market_snapshot",
            "get_portfolio_state",
            "get_active_theses",
            "create_thesis",
            "update_thesis",
            "close_thesis",
            "get_news_signals",
            "get_macro_context",
            "get_signal_attribution",
            "get_decision_history",
            "write_playbook",
            "get_strategy_identity",
            "get_strategy_rules",
            "get_strategy_history",
        }
        assert set(TOOL_HANDLERS.keys()) == expected_handlers

    def test_tool_definitions_list_complete(self):
        """TOOL_DEFINITIONS should have 15 entries (14 tools + web_search)."""
        assert len(TOOL_DEFINITIONS) == 15

        # Extract named tools (excluding web_search which has type field)
        tool_names = {
            t["name"] for t in TOOL_DEFINITIONS
            if t.get("name") != "web_search"
        }
        expected_names = {
            "get_market_snapshot",
            "get_portfolio_state",
            "get_active_theses",
            "create_thesis",
            "update_thesis",
            "close_thesis",
            "get_news_signals",
            "get_macro_context",
            "get_signal_attribution",
            "get_decision_history",
            "write_playbook",
            "get_strategy_identity",
            "get_strategy_rules",
            "get_strategy_history",
        }
        assert tool_names == expected_names

    def test_all_handlers_are_callable(self):
        """Every handler in TOOL_HANDLERS should be callable."""
        for name, handler in TOOL_HANDLERS.items():
            assert callable(handler), f"{name} handler is not callable"

    def test_handlers_match_definitions(self):
        """Every non-web-search definition should have a matching handler."""
        for defn in TOOL_DEFINITIONS:
            name = defn.get("name")
            if name == "web_search":
                continue
            assert name in TOOL_HANDLERS, f"No handler for tool definition '{name}'"


# --- Strategy tool tests ---

from tests.v2.conftest import make_strategy_state_row, make_strategy_rule_row, make_strategy_memo_row


class TestGetStrategyIdentity:
    @patch("v2.tools.get_current_strategy_state")
    def test_returns_formatted_identity(self, mock_get):
        from v2.tools import tool_get_strategy_identity
        mock_get.return_value = make_strategy_state_row()
        result = tool_get_strategy_identity()
        assert "Momentum-focused" in result
        assert "moderate" in result

    @patch("v2.tools.get_current_strategy_state")
    def test_returns_null_message_when_empty(self, mock_get):
        from v2.tools import tool_get_strategy_identity
        mock_get.return_value = None
        result = tool_get_strategy_identity()
        assert "No strategy identity" in result


class TestGetStrategyRules:
    @patch("v2.tools.get_active_strategy_rules")
    def test_returns_formatted_rules(self, mock_get):
        from v2.tools import tool_get_strategy_rules
        mock_get.return_value = [make_strategy_rule_row()]
        result = tool_get_strategy_rules()
        assert "Fade legal" in result
        assert "constraint" in result

    @patch("v2.tools.get_active_strategy_rules")
    def test_returns_empty_message(self, mock_get):
        from v2.tools import tool_get_strategy_rules
        mock_get.return_value = []
        result = tool_get_strategy_rules()
        assert "No active" in result


class TestGetStrategyHistory:
    @patch("v2.tools.get_recent_strategy_memos")
    def test_returns_formatted_memos(self, mock_get):
        from v2.tools import tool_get_strategy_history
        mock_get.return_value = [make_strategy_memo_row()]
        result = tool_get_strategy_history(n=5)
        assert "reflection" in result

    @patch("v2.tools.get_recent_strategy_memos")
    def test_returns_empty_message(self, mock_get):
        from v2.tools import tool_get_strategy_history
        mock_get.return_value = []
        result = tool_get_strategy_history(n=5)
        assert "No strategy" in result


class TestStrategyToolDefinitions:
    def test_strategy_tools_in_definitions(self):
        from v2.tools import TOOL_DEFINITIONS
        names = [d.get("name") for d in TOOL_DEFINITIONS]
        assert "get_strategy_identity" in names
        assert "get_strategy_rules" in names
        assert "get_strategy_history" in names

    def test_strategy_tools_in_handlers(self):
        from v2.tools import TOOL_HANDLERS
        assert "get_strategy_identity" in TOOL_HANDLERS
        assert "get_strategy_rules" in TOOL_HANDLERS
        assert "get_strategy_history" in TOOL_HANDLERS
