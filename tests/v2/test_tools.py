"""Tests for v2.tools — tool definitions and handlers."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11
    UTC = timezone.utc  # noqa: UP017

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

    def test_remaps_company_name_in_action(self, mock_db, mock_cursor):
        """Playbook action with ticker='CARLSMED' is remapped to CARL before
        persistence, so executor pricing hits the real symbol."""
        mock_cursor.fetchone.return_value = {"id": 1}
        actions = [
            {"ticker": "CARLSMED", "action": "buy", "reasoning": "r",
             "confidence": "high", "thesis_id": 1},
        ]
        tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=actions,
            watch_list=[],
            risk_notes="",
        )
        assert actions[0]["ticker"] == "CARL"

    def test_rejects_hallucinated_ticker_in_action(self):
        """A playbook action with a drop-listed acronym ticker (e.g. FAANG)
        is rejected at write time so the strategist sees the error in-loop
        and reissues with the real ticker."""
        result = tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=[
                {"ticker": "FAANG", "action": "buy", "reasoning": "r",
                 "confidence": "high", "thesis_id": 1},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result and "FAANG" in result

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


class TestWritePlaybookZeroMagnitude:
    """ALGO-13: reject zero/missing-magnitude intents that need a positive value.
    The strategist was using `exit_partial_pct=0` as a hold proxy; the correct
    shape is to omit the action entirely."""

    def test_exit_partial_pct_zero_rejected(self):
        result = tool_write_playbook(
            market_outlook="neutral",
            priority_actions=[
                {"ticker": "AMZN", "action": "sell", "thesis_id": 7,
                 "intent_type": "exit_partial_pct", "intent_magnitude": 0,
                 "reasoning": "hold via 0%", "confidence": "medium"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result
        assert "exit_partial_pct" in result
        assert "intent_magnitude=0" in result or "magnitude=0" in result
        assert "omit" in result.lower()

    def test_invest_dollar_zero_rejected(self):
        result = tool_write_playbook(
            market_outlook="neutral",
            priority_actions=[
                {"ticker": "MSFT", "action": "buy", "thesis_id": 8,
                 "intent_type": "invest_dollar", "intent_magnitude": 0,
                 "reasoning": "no-op buy", "confidence": "low"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result
        assert "invest_dollar" in result

    def test_invest_portfolio_pct_missing_magnitude_rejected(self):
        result = tool_write_playbook(
            market_outlook="neutral",
            priority_actions=[
                {"ticker": "MSFT", "action": "buy", "thesis_id": 8,
                 "intent_type": "invest_portfolio_pct",
                 "reasoning": "forgot magnitude", "confidence": "low"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result
        assert "invest_portfolio_pct" in result

    def test_exit_full_with_null_magnitude_accepted(self, mock_db, mock_cursor):
        """exit_full is the documented null-magnitude intent; should pass."""
        from unittest.mock import patch

        with patch("v2.tools.replace_playbook_actions_atomic",
                   return_value=(101, 1)) as mock_write, \
             patch("v2.tools.get_positions",
                   return_value=[{"ticker": "AMZN", "shares": 2.5}]):
            result = tool_write_playbook(
                market_outlook="neutral",
                priority_actions=[
                    {"ticker": "AMZN", "action": "sell", "thesis_id": 7,
                     "intent_type": "exit_full", "intent_magnitude": None,
                     "reasoning": "thesis invalidated", "confidence": "high"},
                ],
                watch_list=[],
                risk_notes="",
            )
        assert "Error" not in result
        mock_write.assert_called_once()

    def test_exit_partial_pct_positive_accepted(self, mock_db, mock_cursor):
        from unittest.mock import patch

        with patch("v2.tools.replace_playbook_actions_atomic",
                   return_value=(102, 1)) as mock_write, \
             patch("v2.tools.get_positions",
                   return_value=[{"ticker": "SPY", "shares": 10}]):
            result = tool_write_playbook(
                market_outlook="neutral",
                priority_actions=[
                    {"ticker": "SPY", "action": "sell", "thesis_id": 9,
                     "intent_type": "exit_partial_pct", "intent_magnitude": 50,
                     "reasoning": "trim", "confidence": "medium"},
                ],
                watch_list=[],
                risk_notes="",
            )
        assert "Error" not in result
        mock_write.assert_called_once()


class TestWritePlaybookExitAgainstZeroHeld:
    """ALGO-18: reject exit-style intents whose ticker has 0 live shares.
    Catches the AMD stub-cleanup loop where the strategist re-wrote the same
    exit_full intent for 6 consecutive sessions against a zeroed position."""

    def test_exit_full_against_zero_held_rejected(self, mock_db, mock_cursor):
        from unittest.mock import patch
        with patch("v2.tools.get_positions", return_value=[]):
            result = tool_write_playbook(
                market_outlook="neutral",
                priority_actions=[
                    {"ticker": "AMD", "action": "sell", "thesis_id": 37,
                     "intent_type": "exit_full", "intent_magnitude": None,
                     "reasoning": "stub cleanup", "confidence": "high"},
                ],
                watch_list=[],
                risk_notes="",
            )
        assert "Error" in result
        assert "AMD" in result
        assert "exit_full" in result
        assert "empty" in result.lower() or "held=" in result

    def test_exit_partial_pct_against_zero_held_rejected(self, mock_db, mock_cursor):
        from unittest.mock import patch
        with patch("v2.tools.get_positions",
                   return_value=[{"ticker": "OTHER", "shares": 5}]):
            result = tool_write_playbook(
                market_outlook="neutral",
                priority_actions=[
                    {"ticker": "AMD", "action": "sell", "thesis_id": 37,
                     "intent_type": "exit_partial_pct", "intent_magnitude": 25,
                     "reasoning": "trim", "confidence": "medium"},
                ],
                watch_list=[],
                risk_notes="",
            )
        assert "Error" in result
        assert "AMD" in result

    def test_exit_full_against_held_accepted(self, mock_db, mock_cursor):
        from unittest.mock import patch
        with patch("v2.tools.replace_playbook_actions_atomic",
                   return_value=(201, 1)) as mock_write, \
             patch("v2.tools.get_positions",
                   return_value=[{"ticker": "AMD", "shares": 0.5}]):
            result = tool_write_playbook(
                market_outlook="neutral",
                priority_actions=[
                    {"ticker": "AMD", "action": "sell", "thesis_id": 37,
                     "intent_type": "exit_full", "intent_magnitude": None,
                     "reasoning": "thesis invalidated", "confidence": "high"},
                ],
                watch_list=[],
                risk_notes="",
            )
        assert "Error" not in result
        mock_write.assert_called_once()

    def test_buy_intent_unaffected_when_held_zero(self, mock_db, mock_cursor):
        """ALGO-18 only gates exit intents; open/add intents pass even at held=0."""
        from unittest.mock import patch
        with patch("v2.tools.replace_playbook_actions_atomic",
                   return_value=(202, 1)) as mock_write, \
             patch("v2.tools.get_positions", return_value=[]):
            result = tool_write_playbook(
                market_outlook="neutral",
                priority_actions=[
                    {"ticker": "NVDA", "action": "buy", "thesis_id": 42,
                     "intent_type": "invest_dollar", "intent_magnitude": 500,
                     "reasoning": "fresh entry", "confidence": "high"},
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

    @patch("v2.context._today", return_value=date(2026, 5, 27))
    @patch("v2.tools.get_recent_thesis_decision_dates", return_value={})
    @patch("v2.tools.get_positions", return_value=[
        {"ticker": "AAPL", "shares": 1, "avg_cost": 150},
    ])
    def test_with_theses(self, mock_positions, mock_last_dates, mock_today, mock_db, mock_cursor):
        """Should format theses when they exist."""
        mock_cursor.fetchall.return_value = [
            {
                "id": 1, "ticker": "AAPL", "direction": "long",
                "confidence": "high", "thesis": "Strong earnings",
                "entry_trigger": "Price > $150", "exit_trigger": "Price > $180",
                "invalidation": "Earnings miss", "created_at": datetime(2026, 5, 1),
                "updated_at": datetime(2026, 5, 20),
            }
        ]
        result = tool_get_active_theses()
        assert "Held Theses" in result
        assert "AAPL" in result
        assert "long" in result
        assert "Strong earnings" in result

    @patch("v2.context._today", return_value=date(2026, 5, 27))
    @patch("v2.tools.get_recent_thesis_decision_dates", return_value={})
    @patch("v2.tools.get_positions", return_value=[])
    def test_with_theses_shows_watch_avoid_and_stale(self, mock_positions, mock_last_dates, mock_today, mock_db, mock_cursor):
        """Should expose operational thesis buckets and stale watch/avoid markers."""
        mock_cursor.fetchall.return_value = [
            {
                "id": 1, "ticker": "AVGO", "direction": "long",
                "confidence": "medium", "thesis": "Watch setup",
                "entry_trigger": "Breakout", "exit_trigger": None,
                "invalidation": None, "created_at": datetime(2026, 5, 1),
                "updated_at": datetime(2026, 5, 12),
            },
            {
                "id": 2, "ticker": "TSLA", "direction": "avoid",
                "confidence": "low", "thesis": "Avoid margin pressure",
                "entry_trigger": None, "exit_trigger": None,
                "invalidation": None, "created_at": datetime(2026, 5, 1),
                "updated_at": datetime(2026, 5, 13),
            },
        ]
        result = tool_get_active_theses()
        assert "Watch Theses:\n- AVGO" in result
        assert "Avoid Theses:\n- TSLA" in result
        assert "[STALE: 10d untouched]" in result


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

    def test_remaps_company_name_alias(self, mock_db, mock_cursor):
        """CARLSMED (a company name the LLM emits because Carlsmed IPO'd
        post-knowledge-cutoff) is remapped to CARL before the duplicate /
        position check, so the thesis is created with the canonical ticker."""
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = {"id": 7}
        with patch(
            "v2.tools.validate_signal_refs",
            return_value=[{"type": "news_signal", "id": 1}],
        ), patch("v2.tools.insert_thesis_signals"):
            result = tool_create_thesis(
                ticker="CARLSMED",
                direction="long",
                thesis="Spine implant rollout",
                entry_trigger="t",
                exit_trigger="t",
                invalidation="t",
                confidence="high",
                signal_refs=[{"type": "news_signal", "id": 1}],
            )
        assert "Created thesis ID 7" in result
        assert "CARL" in result and "CARLSMED" not in result

    def test_rejects_hallucinated_acronym(self, mock_db, mock_cursor):
        """create_thesis on FAANG (or any drop-listed acronym) returns an
        error message the strategist sees in its loop, so it can reissue
        with the real underlying ticker."""
        result = tool_create_thesis(
            ticker="FAANG",
            direction="long",
            thesis="t",
            entry_trigger="t",
            exit_trigger="t",
            invalidation="t",
            confidence="high",
            signal_refs=[{"type": "news_signal", "id": 1}],
        )
        assert "Error" in result and "not a recognized equity symbol" in result


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


class TestThesisHandlersSessionId:
    """Task 8: thesis-creation handlers must forward session_id to
    insert_thesis so newly-created theses are tagged with the active
    session row."""

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.validate_signal_refs")
    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_positions")
    @patch("v2.tools.get_active_theses")
    def test_create_thesis_passes_session_id_to_insert(
        self, mock_theses, mock_positions, mock_insert, mock_validate, mock_links
    ):
        mock_theses.return_value = []
        mock_positions.return_value = []
        mock_insert.return_value = 1
        mock_validate.return_value = [{"type": "news_signal", "id": 1}]
        tool_create_thesis(
            ticker="AAPL",
            direction="long",
            thesis="t",
            entry_trigger="e",
            exit_trigger="x",
            invalidation="i",
            confidence="high",
            signal_refs=[{"type": "news_signal", "id": 1}],
            session_id=42,
        )
        assert mock_insert.call_args.kwargs.get("session_id") == 42

    @patch("v2.tools.insert_thesis_signals")
    @patch("v2.tools.validate_signal_refs")
    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_positions")
    @patch("v2.tools.get_active_theses")
    def test_adopt_thesis_passes_session_id_to_insert(
        self, mock_theses, mock_positions, mock_insert, mock_validate, mock_links
    ):
        mock_theses.return_value = []
        mock_positions.return_value = [{"ticker": "AAPL"}]
        mock_insert.return_value = 1
        mock_validate.return_value = []
        tool_adopt_thesis(
            ticker="AAPL",
            direction="long",
            thesis="t",
            exit_trigger="x",
            invalidation="i",
            confidence="high",
            session_id=42,
        )
        assert mock_insert.call_args.kwargs.get("session_id") == 42

    def test_create_thesis_defaults_session_id_to_none(self, mock_db, mock_cursor):
        """Backwards compat: existing callers that don't pass session_id
        get session_id=None forwarded to insert_thesis."""
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = {"id": 5}
        with patch(
            "v2.tools.validate_signal_refs",
            return_value=[{"type": "news_signal", "id": 1}],
        ), patch("v2.tools.insert_thesis_signals"), patch(
            "v2.tools.insert_thesis", return_value=5
        ) as mock_insert:
            tool_create_thesis(
                ticker="GOOG",
                direction="long",
                thesis="t",
                entry_trigger="e",
                exit_trigger="x",
                invalidation="i",
                confidence="high",
                signal_refs=[{"type": "news_signal", "id": 1}],
            )
        assert mock_insert.call_args.kwargs.get("session_id") is None


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
        """TOOL_HANDLERS should have all 29 handler functions."""
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
            "get_retired_rules",
            "get_rule_bind_history",
            "get_theses",
            "get_thesis_lineage",
            "get_recent_decisions",
            "get_decision_detail",
            "get_flip_flop_report",
            "get_executor_behavior_summary",
            "get_session_memos",
            "get_reflection_actions",
            "get_session_summary",
        }
        assert set(TOOL_HANDLERS.keys()) == expected_handlers

    def test_tool_definitions_list_complete(self):
        """TOOL_DEFINITIONS should have 30 entries (29 tools + web_search)."""
        assert len(TOOL_DEFINITIONS) == 30

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
            "get_retired_rules",
            "get_rule_bind_history",
            "get_theses",
            "get_thesis_lineage",
            "get_recent_decisions",
            "get_decision_detail",
            "get_flip_flop_report",
            "get_executor_behavior_summary",
            "get_session_memos",
            "get_reflection_actions",
            "get_session_summary",
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
        assert "lift:" in result

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
                     "status": "deferred", "decision_action": "hold",
                     "outcome_class": "deferred_by_executor",
                     "reasoning": "trim pre-earnings binary tomorrow"},
                ],
            },
        ]

        result = tool_get_recent_playbooks(n=3)

        assert "2026-05-04" in result
        assert "ANET" in result
        assert "SELL" in result.upper()
        assert "exit_partial_pct=50" in result
        assert "deferred_by_executor" in result
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
        from datetime import datetime
        return {
            "id": id_,
            "ticker": ticker,
            "headline": f"headline {id_}",
            "category": "momentum",
            "sentiment": "bullish",
            "confidence": "high",
            "published_at": datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
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


def test_get_retired_rules_returns_recent_retirements(mock_db, mock_cursor):
    from v2.tools import tool_get_retired_rules
    mock_cursor.fetchall.return_value = [
        {
            "id": 7,
            "rule_text": "Avoid earnings-week entries",
            "category": "risk",
            "supporting_evidence": "evidence A",
            "status": "retired",
            "created_at": "2026-04-15T00:00:00+00:00",
            "retired_at": "2026-05-01T00:00:00+00:00",
            "retirement_reason": "low signal",
        },
    ]
    result = tool_get_retired_rules(limit=10)
    assert result == [{
        "rule_id": 7,
        "rule_text": "Avoid earnings-week entries",
        "category": "risk",
        "supporting_evidence": "evidence A",
        "status": "retired",
        "created_at": "2026-04-15T00:00:00+00:00",
        "retired_at": "2026-05-01T00:00:00+00:00",
        "retirement_reason": "low signal",
    }]
    args, _ = mock_cursor.execute.call_args
    assert "status = 'retired'" in args[0]
    assert "LIMIT %s" in args[0]
    assert args[1] == (10,)


def test_get_rule_bind_history_counts_citations(mock_db, mock_cursor):
    from v2.tools import tool_get_rule_bind_history
    mock_cursor.fetchall.return_value = [
        {"id": 101, "date": "2026-05-20", "ticker": "AAPL", "action": "buy", "reasoning": "rule #7 supports caution"},
        {"id": 115, "date": "2026-05-22", "ticker": "MSFT", "action": "sell", "reasoning": "lift_condition for #7 met"},
    ]
    result = tool_get_rule_bind_history(rule_id=7, days=30)
    assert result["rule_id"] == 7
    assert result["bind_count"] == 2
    assert len(result["citations"]) == 2
    assert result["citations"][0]["decision_id"] == 101
    args, _ = mock_cursor.execute.call_args
    assert "decisions" in args[0]
    assert "decision_signals" in args[0]
    assert "signal_type = 'rule_gate'" in args[0]
    assert "%s" in args[0]  # rule_id param


def test_get_theses_filters_by_status(mock_db, mock_cursor):
    from v2.tools import tool_get_theses
    mock_cursor.fetchall.return_value = [
        {"id": 3, "ticker": "AAPL", "thesis": "Long iPhone cycle",
         "entry_trigger": "RSI<40", "exit_trigger": "RSI>70 or earnings",
         "status": "active", "created_at": "2026-05-01T00:00:00+00:00",
         "closed_at": None, "close_reason": None},
    ]
    result = tool_get_theses(status="active", limit=25)
    assert len(result) == 1
    assert result[0]["thesis_id"] == 3
    assert result[0]["status"] == "active"
    assert result[0]["thesis"] == "Long iPhone cycle"
    args, _ = mock_cursor.execute.call_args
    assert "status = %s" in args[0]
    assert args[1] == ("active", 25)


def test_get_theses_all_status_skips_filter(mock_db, mock_cursor):
    from v2.tools import tool_get_theses
    mock_cursor.fetchall.return_value = []
    tool_get_theses(status="all", limit=50)
    args, _ = mock_cursor.execute.call_args
    assert "status = %s" not in args[0]
    assert args[1] == (50,)


def test_get_thesis_lineage_joins_decisions_via_playbook_actions(mock_db, mock_cursor):
    from v2.tools import tool_get_thesis_lineage
    # First fetchone: thesis header
    mock_cursor.fetchone.return_value = {"id": 3, "ticker": "AAPL", "thesis": "Long cycle", "status": "active"}
    # fetchall: decisions joined via playbook_actions
    mock_cursor.fetchall.return_value = [
        {"id": 88, "date": "2026-05-10", "action": "buy", "quantity": 10.0,
         "reasoning": "thesis #3 entry", "outcome_7d": 1.5, "outcome_30d": 3.2},
    ]
    result = tool_get_thesis_lineage(thesis_id=3)
    assert result["thesis_id"] == 3
    assert result["ticker"] == "AAPL"
    assert result["thesis"] == "Long cycle"
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["outcome_7d"] == 1.5
    assert result["decisions"][0]["outcome_30d"] == 3.2
    # Regression guard: the join must cover both paths
    fetchall_sql = mock_cursor.execute.call_args_list[-1].args[0]
    assert "playbook_actions" in fetchall_sql
    assert "pa.thesis_id = %s" in fetchall_sql
    assert "decision_signals" in fetchall_sql
    assert "ds.signal_type = 'thesis'" in fetchall_sql


def test_get_thesis_lineage_returns_error_when_thesis_missing(mock_db, mock_cursor):
    from v2.tools import tool_get_thesis_lineage
    mock_cursor.fetchone.return_value = None
    result = tool_get_thesis_lineage(thesis_id=9999)
    assert result == {"thesis_id": 9999, "error": "not found"}
    # Only the head SELECT should have run; the second query should not execute.
    sqls = [c.args[0] for c in mock_cursor.execute.call_args_list]
    assert len(sqls) == 1


def test_get_recent_decisions_returns_compact_rows(mock_db, mock_cursor):
    from v2.tools import tool_get_recent_decisions
    long_reasoning = "This is a long reasoning that goes on and on and on " * 20
    mock_cursor.fetchall.return_value = [
        {
            "id": 501,
            "date": "2026-05-20",
            "ticker": "AAPL",
            "action": "buy",
            "quantity": 10.0,
            "reasoning": long_reasoning,
            "signals_referenced": "news_signal,thesis",
            "outcome_7d": 1.25,
        },
    ]
    result = tool_get_recent_decisions(days=14)
    assert len(result) == 1
    assert result[0]["decision_id"] == 501
    assert len(result[0]["reasoning_excerpt"]) <= 200
    assert result[0]["signals_referenced"] == "news_signal,thesis"
    args, _ = mock_cursor.execute.call_args
    assert "d.date >= CURRENT_DATE - INTERVAL '1 day' * %s" in args[0]


def test_get_decision_detail_includes_referenced_signals(mock_db, mock_cursor):
    from v2.tools import tool_get_decision_detail
    mock_cursor.fetchone.return_value = {
        "id": 501,
        "date": "2026-05-20",
        "ticker": "AAPL",
        "action": "buy",
        "quantity": 10.0,
        "reasoning": "full reasoning text",
        "outcome_7d": 1.25,
        "outcome_30d": None,
        "price": 178.5,
    }
    mock_cursor.fetchall.return_value = [
        {"signal_id": 12, "signal_type": "news_signal"},
        {"signal_id": 44, "signal_type": "thesis"},
    ]
    result = tool_get_decision_detail(decision_id=501)
    assert result["decision_id"] == 501
    assert result["reasoning"] == "full reasoning text"
    assert len(result["signals"]) == 2
    assert result["signals"][0]["signal_type"] == "news_signal"


def test_get_decision_detail_returns_error_when_missing(mock_db, mock_cursor):
    from v2.tools import tool_get_decision_detail
    mock_cursor.fetchone.return_value = None
    result = tool_get_decision_detail(decision_id=999999)
    assert result == {"decision_id": 999999, "error": "not found"}


def test_get_flip_flop_report_uses_round_trip_analyzer(monkeypatch):
    from v2 import tools
    from v2.patterns import RoundTrip
    monkeypatch.setattr(
        tools,
        "analyze_round_trips",
        lambda **kw: [RoundTrip(ticker="GOOGL", pair_count=11, first_date="2026-05-01", last_date="2026-05-22")],
    )
    result = tools.tool_get_flip_flop_report(days=30, min_reversals=3)
    assert len(result) == 1
    assert result[0]["ticker"] == "GOOGL"
    assert result[0]["reversal_count"] == 11
    assert result[0]["first_date"] == "2026-05-01"
    assert result[0]["last_date"] == "2026-05-22"


def test_get_executor_behavior_summary_returns_aggregates(mock_db, mock_cursor, monkeypatch):
    from v2 import tools
    from v2.patterns import RoundTrip
    # Mock the three DB result sets in the order they're fetched
    mock_cursor.fetchall.side_effect = [
        # Size histogram
        [{"bucket": "small", "n": 12}, {"bucket": "medium", "n": 7}, {"bucket": "large", "n": 1}],
        # Ticker concentration
        [{"ticker": "AAPL", "n": 5}, {"ticker": "GOOGL", "n": 4}, {"ticker": "TSLA", "n": 3}],
    ]
    mock_cursor.fetchone.side_effect = [
        {"n": 20},  # total decisions
        {"n": 4},   # hold count
    ]
    monkeypatch.setattr(
        tools, "analyze_round_trips",
        lambda **kw: [RoundTrip(ticker="GOOGL", pair_count=3, first_date="x", last_date="y")],
    )
    result = tools.tool_get_executor_behavior_summary(days=14)
    assert result["window_days"] == 14
    assert result["decision_count"] == 20
    assert result["hold_rate"] == 0.2
    assert result["size_histogram"] == {"small": 12, "medium": 7, "large": 1}
    assert result["ticker_concentration"] == {"AAPL": 5, "GOOGL": 4, "TSLA": 3}
    assert result["round_trip_count"] == 3


def test_get_session_memos_returns_recent(mock_db, mock_cursor):
    from v2.tools import tool_get_session_memos
    mock_cursor.fetchall.return_value = [
        {"id": 12, "session_date": "2026-05-20", "memo_type": "reflection",
         "content": "Memo body A", "created_at": "2026-05-20T22:00:00+00:00"},
        {"id": 11, "session_date": "2026-05-19", "memo_type": "reflection",
         "content": "Memo body B", "created_at": "2026-05-19T22:00:00+00:00"},
    ]
    result = tool_get_session_memos(limit=5)
    assert len(result) == 2
    assert result[0]["memo_id"] == 12
    args, _ = mock_cursor.execute.call_args
    assert args[1] == (5,)


def test_get_reflection_actions_aggregates_per_session(mock_db, mock_cursor):
    from v2.tools import tool_get_reflection_actions
    mock_cursor.fetchall.return_value = [
        {"session_id": 44, "session_date": "2026-05-20",
         "rules_proposed": 2, "rules_retired": 1,
         "identity_updated": True, "memo_word_count": 312},
        {"session_id": 43, "session_date": "2026-05-19",
         "rules_proposed": 0, "rules_retired": 0,
         "identity_updated": False, "memo_word_count": 95},
    ]
    result = tool_get_reflection_actions(limit=10)
    assert len(result) == 2
    assert result[0]["rules_proposed"] == 2
    assert result[0]["rules_revalidated"] == 0  # always 0 — no DB column
    assert result[0]["identity_updated"] is True
    assert result[1]["memo_word_count"] == 95


def test_get_session_summary_window_aggregates(mock_db, mock_cursor):
    from v2.tools import tool_get_session_summary_window
    mock_cursor.fetchall.return_value = [
        {"session_id": 44, "session_date": "2026-05-20",
         "decisions_count": 14, "pnl_usd": 152.40,
         "stage_failures": 0, "cost_usd": 0.42},
        {"session_id": 43, "session_date": "2026-05-19",
         "decisions_count": 11, "pnl_usd": -33.10,
         "stage_failures": 1, "cost_usd": 0.38},
    ]
    result = tool_get_session_summary_window(days=14)
    assert len(result) == 2
    assert result[0]["decisions_count"] == 14
    assert result[0]["stage_failures"] == 0
    assert result[1]["cost_usd"] == 0.38
    args, _ = mock_cursor.execute.call_args
    assert "session_costs" in args[0]
    assert args[1] == (14,)
