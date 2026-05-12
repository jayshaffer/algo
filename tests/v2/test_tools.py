"""Tests for v2.tools — tool definitions and handlers."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

from tests.v2.conftest import make_strategy_memo_row, make_strategy_rule_row, make_strategy_state_row
from v2.tools import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    tool_adopt_thesis,
    tool_create_thesis,
    tool_get_active_theses,
    tool_get_market_snapshot,
    tool_write_playbook,
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
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry hit", "confidence": "high", "thesis_id": 1},
                {"ticker": "MSFT", "action": "buy", "reasoning": "Breakout", "confidence": "medium", "thesis_id": 2},
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
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry hit", "confidence": "high", "thesis_id": 1},
                {"ticker": "MSFT", "action": "sell", "reasoning": "Take profit", "confidence": "medium", "thesis_id": 2},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" not in result
        assert "conflict" not in result.lower()

    def test_duplicate_ticker_action_pair_rejected(self, mock_db, mock_cursor):
        """T1.6: two buys for the same ticker were previously accepted, letting
        the executor evaluate both and risk doubling up against the same idea.
        Reject at write-time so the strategist deduplicates in its own loop.
        """
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
        assert "Error" in result
        assert "Duplicate" in result
        assert "AAPL" in result
        # Returns clean string (not raw exception) — strategist tool loop reads it
        # and can retry on the next turn with deduped actions.
        assert "buy" in result

    def test_duplicate_pair_rejection_is_case_insensitive(self, mock_db, mock_cursor):
        """T1.3 + T1.6 interact: 'aapl' and 'AAPL' as buys both normalize to AAPL,
        then dedup catches them.
        """
        mock_cursor.fetchone.return_value = {"id": 1}
        result = tool_write_playbook(
            market_outlook="Mixed",
            priority_actions=[
                {"ticker": "aapl", "action": "buy", "reasoning": "r1", "confidence": "high"},
                {"ticker": "AAPL", "action": "buy", "reasoning": "r2", "confidence": "high"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Duplicate" in result

    def test_normalizes_action_tickers_before_persist(self, mock_db, mock_cursor):
        """T1.3: lowercase / whitespace tickers in playbook actions are
        canonicalized before validation and DB write. Without this, the
        playbook stores 'aapl ' and the executor's downstream
        SECTOR_MAP / position lookups all miss.
        """
        mock_cursor.fetchone.return_value = {"id": 1}
        actions = [
            {"ticker": "aapl ", "action": "buy", "reasoning": "r", "confidence": "high"},
            {"ticker": " msft", "action": "sell", "reasoning": "r", "confidence": "medium"},
        ]
        tool_write_playbook(
            market_outlook="Mixed",
            priority_actions=actions,
            watch_list=[],
            risk_notes="",
        )
        # In-place normalization: the dicts the strategist passed are mutated
        # and that is what reaches replace_playbook_actions_atomic.
        assert actions[0]["ticker"] == "AAPL"
        assert actions[1]["ticker"] == "MSFT"

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

    def test_buy_action_without_thesis_id_rejected(self):
        """Playbook buy/sell actions must cite a thesis. The strategist's
        job is producing thesis-backed actions; allowing NULL thesis_id
        creates orphans in playbook_actions that break attribution joins.
        """
        result = tool_write_playbook(
            market_outlook="neutral",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "intent_type": "open_pct",
                 "intent_magnitude": 0.05, "reasoning": "no thesis here"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result
        assert "thesis_id" in result.lower() or "thesis" in result.lower()

    def test_sell_action_without_thesis_id_rejected(self):
        result = tool_write_playbook(
            market_outlook="neutral",
            priority_actions=[
                {"ticker": "AAPL", "action": "sell", "intent_type": "exit_full",
                 "intent_magnitude": None, "reasoning": "no thesis here"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result
        assert "thesis" in result.lower()

    def test_buy_action_with_thesis_id_accepted(self, mock_db, mock_cursor):
        """Sanity: when thesis_id is provided, the write proceeds."""
        from unittest.mock import patch
        from v2.tools import tool_write_playbook

        with patch("v2.tools.replace_playbook_actions_atomic",
                   return_value=(99, 1)) as mock_write:
            result = tool_write_playbook(
                market_outlook="neutral",
                priority_actions=[
                    {"ticker": "AAPL", "action": "buy", "thesis_id": 42,
                     "intent_type": "open_pct", "intent_magnitude": 0.05,
                     "reasoning": "thesis-backed"},
                ],
                watch_list=[],
                risk_notes="",
            )
        assert "Error" not in result
        mock_write.assert_called_once()


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
        # validate_signal_refs returns the input as-is (one valid ref)
        with patch(
            "v2.tools.validate_signal_refs",
            return_value=[{"type": "news_signal", "id": 1}],
        ), patch("v2.tools.insert_thesis_signals"):
            result = tool_create_thesis(
                ticker="GOOG",
                direction="long",
                thesis="AI growth",
                entry_trigger="Price > $100",
                exit_trigger="Price > $150",
                invalidation="AI hype fades",
                confidence="high",
                signal_refs=[{"type": "news_signal", "id": 1}],
            )
        assert "Created thesis ID 5" in result
        assert "GOOG" in result

    def test_rejects_missing_signal_refs(self, mock_db, mock_cursor):
        """create_thesis without signal_refs is rejected (audit finding 69)."""
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = {"id": 5}
        result = tool_create_thesis(
            ticker="GOOG", direction="long", thesis="AI growth",
            entry_trigger="Price > $100", exit_trigger="Price > $150",
            invalidation="AI hype fades", confidence="high",
        )
        assert "Error" in result and "signal_ref" in result


class TestCreateThesisSignalRefs:
    """Phase 3 of signal-citation wiring: thesis creation captures the
    news/macro signal IDs that justified it. The strategist surfaces these
    via tool_get_news_signals/tool_get_macro_signals and cites them here."""

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.validate_signal_refs")
    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_positions")
    @patch("v2.tools.get_active_theses")
    def test_persists_signal_refs(self, mock_theses, mock_positions,
                                   mock_insert, mock_validate, mock_insert_links):
        mock_theses.return_value = []
        mock_positions.return_value = []
        mock_insert.return_value = 5
        mock_validate.return_value = [
            {"type": "news_signal", "id": 100},
            {"type": "macro_signal", "id": 7},
        ]
        result = tool_create_thesis(
            ticker="GOOG",
            direction="long",
            thesis="AI growth",
            entry_trigger="Price > $100",
            exit_trigger="Price > $150",
            invalidation="AI hype fades",
            confidence="high",
            signal_refs=[
                {"type": "news_signal", "id": 100},
                {"type": "macro_signal", "id": 7},
            ],
        )
        mock_insert_links.assert_called_once_with(
            5,
            [{"type": "news_signal", "id": 100}, {"type": "macro_signal", "id": 7}],
        )
        assert "Created thesis ID 5" in result

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.validate_signal_refs")
    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_positions")
    @patch("v2.tools.get_active_theses")
    def test_warns_on_invalid_refs(self, mock_theses, mock_positions,
                                    mock_insert, mock_validate, mock_insert_links):
        mock_theses.return_value = []
        mock_positions.return_value = []
        mock_insert.return_value = 5
        # 1 of 2 refs survives validation
        mock_validate.return_value = [{"type": "news_signal", "id": 100}]
        result = tool_create_thesis(
            ticker="GOOG", direction="long", thesis="AI growth",
            entry_trigger="...", exit_trigger="...", invalidation="...",
            confidence="high",
            signal_refs=[
                {"type": "news_signal", "id": 100},
                {"type": "news_signal", "id": 999_999},  # not in DB
            ],
        )
        # Strategist must see that one ref was stripped so it can correct
        assert "1" in result and ("strip" in result.lower() or "invalid" in result.lower())

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_positions")
    @patch("v2.tools.get_active_theses")
    def test_rejects_without_signal_refs(self, mock_theses, mock_positions,
                                          mock_insert, mock_insert_links):
        """Audit finding 69 fix: create_thesis with no signal_refs is rejected.
        Use tool_adopt_thesis for orphan-position housekeeping (which still
        accepts empty signal_refs)."""
        mock_theses.return_value = []
        mock_positions.return_value = []
        result = tool_create_thesis(
            ticker="GOOG", direction="long", thesis="AI growth",
            entry_trigger="...", exit_trigger="...", invalidation="...",
            confidence="high",
        )
        assert "Error" in result
        assert "signal_ref" in result
        mock_insert.assert_not_called()
        mock_insert_links.assert_not_called()


class TestAdoptThesisSignalRefs:

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.validate_signal_refs")
    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_active_theses")
    @patch("v2.tools.get_positions")
    def test_persists_signal_refs(self, mock_positions, mock_theses,
                                   mock_insert, mock_validate, mock_insert_links):
        mock_positions.return_value = [{"ticker": "AAPL"}]
        mock_theses.return_value = []
        mock_insert.return_value = 42
        mock_validate.return_value = [{"type": "news_signal", "id": 5}]
        tool_adopt_thesis(
            ticker="AAPL", direction="long",
            thesis="Strong ecosystem", exit_trigger="...",
            invalidation="...", confidence="medium",
            signal_refs=[{"type": "news_signal", "id": 5}],
        )
        mock_insert_links.assert_called_once_with(42, [{"type": "news_signal", "id": 5}])


class TestUpdateThesisSignalRefs:

    @patch("v2.tools.get_thesis_by_id", return_value={"id": 5})
    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.validate_signal_refs")
    @patch("v2.tools.update_thesis")
    def test_adds_signal_refs(self, mock_update, mock_validate, mock_insert_links, _mock_get):
        from v2.tools import tool_update_thesis
        mock_update.return_value = True
        mock_validate.return_value = [{"type": "news_signal", "id": 200}]
        result = tool_update_thesis(
            thesis_id=5,
            add_signal_refs=[{"type": "news_signal", "id": 200}],
        )
        mock_insert_links.assert_called_once_with(5, [{"type": "news_signal", "id": 200}])
        # update_thesis itself only needs to be called if other fields changed;
        # signal-only updates are still a successful no-op result.
        assert "Updated" in result or "5" in result

    @patch("v2.tools.get_thesis_by_id", return_value={"id": 5})
    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.update_thesis")
    def test_no_signal_refs_keeps_existing_behaviour(self, mock_update, mock_insert_links, _mock_get):
        from v2.tools import tool_update_thesis
        mock_update.return_value = True
        result = tool_update_thesis(thesis_id=5, thesis="New text")
        assert "Updated" in result
        mock_insert_links.assert_not_called()


class TestUpdateThesisExistenceGuard:
    """T1.7: invalid thesis_id on the add_signal_refs-only path used to crash
    inside _persist_signal_refs with a raw psycopg2 FK error. Tool callers in
    the strategist loop need a clean string error so they can self-correct.
    """

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.update_thesis")
    @patch("v2.tools.get_thesis_by_id", return_value=None)
    def test_invalid_thesis_id_with_signal_refs_only_returns_clean_error(
        self, mock_get, mock_update, mock_insert,
    ):
        from v2.tools import tool_update_thesis
        result = tool_update_thesis(
            thesis_id=999,
            add_signal_refs=[{"type": "news_signal", "id": 1}],
        )
        assert "Error" in result
        assert "999" in result
        assert "not found" in result.lower()
        mock_update.assert_not_called()
        mock_insert.assert_not_called()

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.update_thesis")
    @patch("v2.tools.get_thesis_by_id", return_value=None)
    def test_invalid_thesis_id_with_field_update_returns_clean_error(
        self, mock_get, mock_update, mock_insert,
    ):
        from v2.tools import tool_update_thesis
        result = tool_update_thesis(thesis_id=999, thesis="New")
        assert "Error" in result
        assert "999" in result
        assert "not found" in result.lower()
        mock_update.assert_not_called()

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.update_thesis")
    @patch("v2.tools.get_thesis_by_id")
    def test_no_updates_at_all_returns_clean_error(
        self, mock_get, mock_update, mock_insert,
    ):
        from v2.tools import tool_update_thesis
        result = tool_update_thesis(thesis_id=5)
        assert "Error" in result
        assert "no updates" in result.lower()
        # Existence check is short-circuited; no DB roundtrip needed
        mock_get.assert_not_called()
        mock_update.assert_not_called()
        mock_insert.assert_not_called()


class TestToolAdoptThesis:

    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_active_theses")
    @patch("v2.tools.get_positions")
    def test_adopt_success(self, mock_positions, mock_theses, mock_insert):
        mock_positions.return_value = [{"ticker": "AAPL"}]
        mock_theses.return_value = []
        mock_insert.return_value = 42

        result = tool_adopt_thesis(
            ticker="AAPL",
            direction="long",
            thesis="Strong ecosystem and services growth",
            exit_trigger="Price drops below $130",
            invalidation="iPhone sales decline 3 consecutive quarters",
            confidence="medium",
        )
        # T2.12: adopt now uses a distinct "Adopted" prefix so count_actions
        # can separate adoption from creation. Older test asserted
        # "Created thesis ID 42"; the new contract is "Adopted thesis ID 42".
        assert "Adopted thesis ID 42" in result
        assert "adopted" in result.lower() or "AAPL" in result
        mock_insert.assert_called_once()
        # Verify source is "adoption" not "claude_ideation"
        call_kwargs = mock_insert.call_args
        assert call_kwargs[1]["source"] == "adoption" or call_kwargs[0][-1] == "adoption"

    @patch("v2.tools.get_active_theses")
    @patch("v2.tools.get_positions")
    def test_reject_not_in_portfolio(self, mock_positions, mock_theses):
        mock_positions.return_value = [{"ticker": "NVDA"}]
        mock_theses.return_value = []

        result = tool_adopt_thesis(
            ticker="TSLA",
            direction="long",
            thesis="EV growth",
            exit_trigger="...",
            invalidation="...",
            confidence="medium",
        )
        assert "Error" in result
        assert "not in the portfolio" in result

    @patch("v2.tools.get_active_theses")
    @patch("v2.tools.get_positions")
    def test_reject_existing_thesis(self, mock_positions, mock_theses):
        mock_positions.return_value = [{"ticker": "AAPL"}]
        mock_theses.return_value = [{"id": 1, "ticker": "AAPL"}]

        result = tool_adopt_thesis(
            ticker="AAPL",
            direction="long",
            thesis="...",
            exit_trigger="...",
            invalidation="...",
            confidence="medium",
        )
        assert "Error" in result
        assert "already exists" in result


# --- Tool completeness tests ---


class TestGetNewsSignals:
    """The strategist must see signal IDs so it can cite them on theses.
    Without IDs the executor downstream would have nothing real to record
    in decision_signals — this was the bug behind the news_signal:unknown
    artifact in attribution."""

    @patch("v2.tools.get_news_signals")
    def test_signal_ids_visible_in_output(self, mock_get):
        """Each signal line must include its DB id so the LLM can cite it."""
        mock_get.return_value = [
            {
                "id": 4242, "ticker": "AAPL", "category": "earnings",
                "sentiment": "bullish", "confidence": "high",
                "headline": "Apple reports record Q4 earnings",
                "published_at": datetime(2026, 4, 30, 14, 22),
            },
        ]
        from v2.tools import tool_get_news_signals
        out = tool_get_news_signals(ticker="AAPL", days=7)
        assert "4242" in out

    @patch("v2.tools.get_news_signals")
    def test_no_signals_message(self, mock_get):
        mock_get.return_value = []
        from v2.tools import tool_get_news_signals
        out = tool_get_news_signals(ticker="AAPL", days=7)
        assert "no" in out.lower() or "No" in out


class TestGetMacroSignals:
    """Macro signals also need IDs surfaced. tool_get_macro_context summarizes
    by category and never shows IDs — so we add a parallel line-per-signal
    tool that the strategist can read when it needs to cite specific macro
    signals on a thesis. Same shape as tool_get_news_signals."""

    @patch("v2.tools.get_macro_signals")
    def test_signal_ids_visible_in_output(self, mock_get):
        mock_get.return_value = [
            {
                "id": 99, "category": "fed",
                "sentiment": "neutral", "confidence": "high",
                "headline": "Fed holds rates steady",
                "published_at": datetime(2026, 4, 30, 14, 0),
                "affected_sectors": ["finance"],
            },
        ]
        from v2.tools import tool_get_macro_signals
        out = tool_get_macro_signals(days=7)
        assert "99" in out

    @patch("v2.tools.get_macro_signals")
    def test_no_signals_message(self, mock_get):
        mock_get.return_value = []
        from v2.tools import tool_get_macro_signals
        out = tool_get_macro_signals(days=7)
        assert "no" in out.lower() or "No" in out


class TestToolCompleteness:
    def test_tool_handlers_dict_complete(self):
        """TOOL_HANDLERS should have all 18 handler functions."""
        expected_handlers = {
            "get_market_snapshot",
            "get_portfolio_state",
            "get_active_theses",
            "create_thesis",
            "adopt_thesis",
            "update_thesis",
            "close_thesis",
            "get_curated_news",
            "get_news_signals",
            "get_macro_context",
            "get_macro_signals",
            "get_signal_attribution",
            "get_decision_history",
            "get_recent_playbooks",
            "write_playbook",
            "get_strategy_identity",
            "get_strategy_rules",
            "get_strategy_history",
        }
        assert set(TOOL_HANDLERS.keys()) == expected_handlers

    def test_tool_definitions_list_complete(self):
        """TOOL_DEFINITIONS should have 19 entries (18 tools + web_search)."""
        assert len(TOOL_DEFINITIONS) == 19

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
            "adopt_thesis",
            "update_thesis",
            "close_thesis",
            "get_curated_news",
            "get_news_signals",
            "get_macro_context",
            "get_macro_signals",
            "get_signal_attribution",
            "get_decision_history",
            "get_recent_playbooks",
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

    def test_thesis_tools_advertise_signal_refs(self):
        """create_thesis/adopt_thesis must let the LLM cite signals;
        update_thesis must let the LLM add signals."""
        by_name = {t.get("name"): t for t in TOOL_DEFINITIONS if t.get("name")}
        for name in ("create_thesis", "adopt_thesis"):
            props = by_name[name]["input_schema"]["properties"]
            assert "signal_refs" in props, f"{name} must advertise signal_refs"
        update_props = by_name["update_thesis"]["input_schema"]["properties"]
        assert "add_signal_refs" in update_props


# --- Strategy tool tests ---


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


class TestGetStrategyHistoryTruncation:
    @patch("v2.tools.get_recent_strategy_memos")
    def test_recent_memos_not_truncated(self, mock_get):
        """Last 2 memos should be returned in full."""
        from v2.tools import tool_get_strategy_history
        long_content = "A" * 2000
        mock_get.return_value = [
            make_strategy_memo_row(content=long_content, session_date=date(2026, 4, 2)),
            make_strategy_memo_row(content=long_content, session_date=date(2026, 4, 1)),
        ]
        result = tool_get_strategy_history(n=5)
        assert long_content in result
        assert result.count("...") == 0

    @patch("v2.tools.get_recent_strategy_memos")
    def test_older_memos_truncated_to_300(self, mock_get):
        """Memos beyond position 2 should be truncated to 300 chars."""
        from v2.tools import tool_get_strategy_history
        long_content = "B" * 2000
        mock_get.return_value = [
            make_strategy_memo_row(content="recent1", session_date=date(2026, 4, 2)),
            make_strategy_memo_row(content="recent2", session_date=date(2026, 4, 1)),
            make_strategy_memo_row(content=long_content, session_date=date(2026, 3, 31)),
        ]
        result = tool_get_strategy_history(n=5)
        lines = result.split("\n")
        assert "B" * 300 in lines[2]
        assert "B" * 301 not in lines[2]
        assert lines[2].endswith("...")

    @patch("v2.tools.get_recent_strategy_memos")
    def test_short_older_memos_not_truncated(self, mock_get):
        """Short older memos (<300 chars) should not get '...' appended."""
        from v2.tools import tool_get_strategy_history
        mock_get.return_value = [
            make_strategy_memo_row(content="recent", session_date=date(2026, 4, 2)),
            make_strategy_memo_row(content="recent", session_date=date(2026, 4, 1)),
            make_strategy_memo_row(content="short old memo", session_date=date(2026, 3, 31)),
        ]
        result = tool_get_strategy_history(n=5)
        assert "short old memo" in result
        assert "..." not in result.split("\n")[2]


class TestToolGetRecentPlaybooks:
    """Strategist tool that surfaces yesterday's planned actions."""

    @patch("v2.tools.get_recent_playbooks_with_actions")
    def test_renders_playbooks_with_actions(self, mock_get):
        from v2.tools import tool_get_recent_playbooks
        mock_get.return_value = [
            {
                "pb_id": 87, "pb_date": date(2026, 5, 4),
                "pb_market_outlook": "fragile",
                "actions": [
                    {"action_id": 401, "ticker": "ANET", "action": "sell",
                     "intent_type": "exit_partial_pct", "intent_magnitude": 50,
                     "reasoning": "trim pre-earnings binary tomorrow"},
                ],
            },
        ]

        result = tool_get_recent_playbooks(n=3)

        assert "2026-05-04" in result
        assert "ANET" in result
        assert "SELL" in result.upper()
        assert "exit_partial_pct=50" in result
        assert "trim pre-earnings" in result

    @patch("v2.tools.get_recent_playbooks_with_actions", return_value=[])
    def test_empty_marker_when_no_playbooks(self, mock_get):
        from v2.tools import tool_get_recent_playbooks
        result = tool_get_recent_playbooks(n=3)
        assert "no recent playbooks" in result.lower() or "none" in result.lower()

    def test_tool_registered_in_definitions_and_handlers(self):
        from v2.tools import TOOL_DEFINITIONS, TOOL_HANDLERS
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "get_recent_playbooks" in names
        assert "get_recent_playbooks" in TOOL_HANDLERS


class TestToolGetCuratedNews:
    """tool_get_curated_news fetches signals from DB, filters via Haiku,
    re-renders in the existing [#id] line format. Falls back to firehose
    on filter failure. Caches per (ticker, days, target_n) within a
    session; reset by reset_session()."""

    def _row(self, id_, ticker="AAPL", summary="full summary"):
        from datetime import datetime, timezone
        return {
            "id": id_,
            "ticker": ticker,
            "headline": f"headline {id_}",
            "category": "momentum",
            "sentiment": "bullish",
            "confidence": "high",
            "published_at": datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
            "summary": summary,
            "alpaca_id": f"alp-{id_}",
        }

    def test_filters_signals_via_haiku(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in [1, 2, 3, 4, 5]]
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "risk-on regime")
        monkeypatch.setattr(
            tools,
            "curate_signals",
            lambda signals, target_n, regime_context: [2, 4],
        )
        tools.reset_session()  # ensure cache is clean

        out = tools.tool_get_curated_news(target_n=2)

        assert "[#2]" in out and "[#4]" in out
        assert "[#1]" not in out
        assert "[#3]" not in out
        assert "[#5]" not in out

    def test_skips_haiku_when_below_target(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in [1, 2, 3]]
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        called = {"count": 0}

        def fake_curate(*args, **kwargs):
            called["count"] += 1
            return [s["id"] for s in args[0]]

        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        out = tools.tool_get_curated_news(target_n=10)

        assert called["count"] == 0, "curate_signals should not be called when candidates <= target_n"
        for i in [1, 2, 3]:
            assert f"[#{i}]" in out

    def test_drops_null_summary_rows(self, monkeypatch):
        from v2 import tools

        rows = [
            self._row(1, summary="full"),
            self._row(2, summary=None),
            self._row(3, summary=""),
            self._row(4, summary="full"),
        ]
        captured = {}

        def fake_curate(signals, target_n, regime_context):
            captured["signal_ids"] = [s["id"] for s in signals]
            return [s["id"] for s in signals]

        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        # Force the filter path by setting target_n below the post-NULL-filter count.
        tools.tool_get_curated_news(target_n=1)

        assert captured["signal_ids"] == [1, 4], (
            f"NULL/empty summaries must be dropped before filter; got {captured['signal_ids']}"
        )

    def test_caches_within_session(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in range(1, 41)]  # 40 rows, above target
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        called = {"count": 0}

        def fake_curate(*args, **kwargs):
            called["count"] += 1
            return [1, 2, 3]

        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        tools.tool_get_curated_news(target_n=3)
        tools.tool_get_curated_news(target_n=3)

        assert called["count"] == 1, "second call with same params must hit cache"

        # Different target_n is a different cache key:
        tools.tool_get_curated_news(target_n=5)
        assert called["count"] == 2, "different target_n must miss cache"

    def test_cache_resets_with_session(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in range(1, 41)]
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        called = {"count": 0}

        def fake_curate(*args, **kwargs):
            called["count"] += 1
            return [1]

        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        tools.tool_get_curated_news(target_n=1)
        tools.reset_session()
        tools.tool_get_curated_news(target_n=1)

        assert called["count"] == 2, "reset_session must clear the curated-news cache"

    def test_ticker_specific_passes_ticker_to_db(self, monkeypatch):
        from v2 import tools

        captured_kwargs = {}

        def fake_get_news_signals(**kw):
            captured_kwargs.update(kw)
            return [self._row(1, ticker="AAPL")]

        monkeypatch.setattr(tools, "get_news_signals", fake_get_news_signals)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        monkeypatch.setattr(tools, "curate_signals", lambda *a, **kw: [1])
        tools.reset_session()

        tools.tool_get_curated_news(ticker="aapl", days=3)

        # Ticker is normalised upstream; assert the DB was queried for AAPL/3.
        assert captured_kwargs.get("ticker") == "AAPL"
        assert captured_kwargs.get("days") == 3
