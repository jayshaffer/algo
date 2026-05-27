"""Direct unit tests for dashboard/queries.py SQL functions.

These tests patch `dashboard.queries.get_cursor` to a context manager that
yields a MagicMock cursor — no Flask app, no DB. We assert the SQL string
and bound parameters where it matters, and the shaped return value always.
"""

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    make_agent_event_row,
    make_decision_row,
    make_news_signal_row,
    make_open_order_row,
    make_playbook_action_row,
    make_position_row,
    make_session_row,
    make_strategy_memo_row,
    make_thesis_row,
    make_tweet_row,
)


@pytest.fixture
def cur():
    """Return a MagicMock cursor patched into dashboard.queries.get_cursor."""
    mock_cursor = MagicMock()

    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("dashboard.queries.get_cursor", _get_cursor):
        yield mock_cursor


class TestLookupSessionIdByDate:
    def test_returns_id_when_found(self, cur):
        from dashboard.queries import lookup_session_id_by_date
        cur.fetchone.return_value = {"id": 42}
        result = lookup_session_id_by_date(date(2026, 5, 11))
        assert result == 42

    def test_returns_none_when_no_session(self, cur):
        from dashboard.queries import lookup_session_id_by_date
        cur.fetchone.return_value = None
        result = lookup_session_id_by_date(date(2026, 5, 11))
        assert result is None

    def test_filters_by_session_type(self, cur):
        from dashboard.queries import lookup_session_id_by_date
        cur.fetchone.return_value = {"id": 7}
        lookup_session_id_by_date(date(2026, 5, 11), session_type="premarket")
        # Verify the executed SQL had the type param
        called_sql = cur.execute.call_args[0][0]
        called_params = cur.execute.call_args[0][1]
        assert "session_type" in called_sql
        assert "premarket" in called_params


class TestGetThesis:
    def test_returns_thesis_row(self, cur):
        from dashboard.queries import get_thesis
        cur.fetchone.return_value = make_thesis_row(id=5)
        result = get_thesis(5)
        assert result["id"] == 5

    def test_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_thesis
        cur.fetchone.return_value = None
        assert get_thesis(999) is None

    def test_select_includes_session_id(self, cur):
        from dashboard.queries import get_thesis
        cur.fetchone.return_value = make_thesis_row(id=5)
        get_thesis(5)
        called_sql = cur.execute.call_args[0][0]
        assert "session_id" in called_sql


class TestGetThesisDecisions:
    def test_returns_decisions_joined_through_decision_signals(self, cur):
        from dashboard.queries import get_thesis_decisions
        cur.fetchall.return_value = [make_decision_row(id=1), make_decision_row(id=2)]
        result = get_thesis_decisions(5)
        assert len(result) == 2
        # Verify SQL joins decision_signals filtered by signal_type='thesis'
        called_sql = cur.execute.call_args[0][0]
        assert "decision_signals" in called_sql
        assert "thesis" in called_sql
        assert 5 in cur.execute.call_args[0][1]

    def test_returns_empty_when_no_decisions(self, cur):
        from dashboard.queries import get_thesis_decisions
        cur.fetchall.return_value = []
        assert get_thesis_decisions(5) == []


class TestGetThesisPlaybookActions:
    def test_returns_actions_for_thesis(self, cur):
        from dashboard.queries import get_thesis_playbook_actions
        cur.fetchall.return_value = [make_playbook_action_row(thesis_id=5)]
        result = get_thesis_playbook_actions(5)
        assert len(result) == 1
        assert 5 in cur.execute.call_args[0][1]


class TestGetDecision:
    def test_returns_decision_row(self, cur):
        from dashboard.queries import get_decision
        cur.fetchone.return_value = make_decision_row(id=7)
        assert get_decision(7)["id"] == 7

    def test_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_decision
        cur.fetchone.return_value = None
        assert get_decision(999) is None

    def test_select_includes_session_id(self, cur):
        from dashboard.queries import get_decision
        cur.fetchone.return_value = make_decision_row(id=7)
        get_decision(7)
        called_sql = cur.execute.call_args[0][0]
        assert "session_id" in called_sql


class TestGetStrategyMemos:
    def test_returns_memo_rows(self, cur):
        from dashboard.queries import get_strategy_memos
        cur.fetchall.return_value = [make_strategy_memo_row(id=1), make_strategy_memo_row(id=2)]
        result = get_strategy_memos(days=30)
        assert len(result) == 2

    def test_select_includes_session_id(self, cur):
        from dashboard.queries import get_strategy_memos
        cur.fetchall.return_value = []
        get_strategy_memos(days=30)
        called_sql = cur.execute.call_args[0][0]
        assert "session_id" in called_sql


class TestGetDecisionSignalsFull:
    def test_returns_denormalized_signals(self, cur):
        from dashboard.queries import get_decision_signals_full
        cur.fetchall.return_value = [
            {
                "signal_type": "news_signal",
                "signal_id": 100,
                "news_headline": "AAPL beats Q3",
                "news_summary": "Apple beat estimates",
                "news_category": "earnings",
                "news_sentiment": "bullish",
                "news_confidence": "high",
                "news_published_at": datetime(2026, 5, 11, 10, 0),
                "news_ticker": "AAPL",
                "macro_headline": None,
                "macro_category": None,
                "macro_affected_sectors": None,
                "macro_sentiment": None,
                "macro_published_at": None,
                "thesis_text": None,
                "thesis_ticker": None,
                "thesis_direction": None,
                "thesis_status": None,
            },
        ]
        result = get_decision_signals_full(7)
        assert len(result) == 1
        assert result[0]["signal_type"] == "news_signal"
        assert result[0]["news_headline"] == "AAPL beats Q3"

    def test_returns_empty_when_no_signals(self, cur):
        from dashboard.queries import get_decision_signals_full
        cur.fetchall.return_value = []
        assert get_decision_signals_full(7) == []


class TestGetDecisionTweets:
    def test_returns_tweets_for_decision(self, cur):
        from dashboard.queries import get_decision_tweets
        cur.fetchall.return_value = [make_tweet_row(id=1)]
        result = get_decision_tweets(7)
        assert len(result) == 1
        assert 7 in cur.execute.call_args[0][1]


class TestGetPlaybookAction:
    def test_returns_action_with_thesis_join(self, cur):
        from dashboard.queries import get_playbook_action
        cur.fetchone.return_value = make_playbook_action_row(id=3)
        assert get_playbook_action(3)["id"] == 3

    def test_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_playbook_action
        cur.fetchone.return_value = None
        assert get_playbook_action(999) is None


class TestBehaviorRegressionSummary:
    def test_returns_behavior_summary_sections(self, cur):
        from dashboard.queries import get_behavior_regression_summary

        cur.fetchall.side_effect = [
            [{"status": "deferred", "count": 2}, {"status": "pending", "count": 1}],
            [{
                "id": 7,
                "ticker": "AVGO",
                "status": "pending",
                "decision_id": 70,
                "decision_action": "hold",
            }],
            [{"ticker": "AVGO", "count": 2, "last_deferred_date": date(2026, 5, 26)}],
            [{"id": 9, "session_date": date(2026, 5, 25), "status": "completed"}],
            [
                {
                    "rule_id": 42,
                    "rule_text": "Hold until macro cap lifts",
                    "category": "risk",
                    "direction": "constraint",
                    "lift_condition": None,
                    "decision_date": date(2026, 5, 27),
                    "ticker": "GOOGL",
                    "decision_action": "hold",
                    "playbook_action_status": "deferred",
                },
                {
                    "rule_id": 42,
                    "rule_text": "Hold until macro cap lifts",
                    "category": "risk",
                    "direction": "constraint",
                    "lift_condition": None,
                    "decision_date": date(2026, 5, 26),
                    "ticker": "AVGO",
                    "decision_action": "hold",
                    "playbook_action_status": None,
                },
            ],
            [
                {"role": "watch", "last_touched_date": date(2020, 1, 2)},
                {"role": "watch", "last_touched_date": date(2020, 1, 3)},
                {"role": "held", "last_touched_date": date(2020, 1, 2)},
            ],
        ]

        result = get_behavior_regression_summary(days=30)

        assert result["playbook_actions_by_status"][0]["status"] == "deferred"
        assert result["reviewed_actions_still_pending"][0]["decision_action"] == "hold"
        assert result["deferred_actions_by_ticker"][0]["ticker"] == "AVGO"
        assert result["market_closed_sessions"][0]["session_date"] == date(2026, 5, 25)
        assert result["rule_gate_telemetry"][0]["rule_id"] == 42
        assert result["rule_gate_telemetry"][0]["distinct_tickers"] == ["AVGO", "GOOGL"]
        assert result["active_thesis_role_counts"][1]["role"] == "watch"
        assert result["active_thesis_role_counts"][1]["count"] == 2
        assert result["active_thesis_role_counts"][1]["stale_count"] == 2
        assert cur.execute.call_count == 6

    def test_queries_all_behavior_metrics(self, cur):
        from dashboard.queries import get_behavior_regression_summary

        cur.fetchall.side_effect = [[], [], [], [], [], []]
        get_behavior_regression_summary(days=14)

        status_sql = cur.execute.call_args_list[0][0][0]
        reviewed_sql = cur.execute.call_args_list[1][0][0]
        deferred_sql = cur.execute.call_args_list[2][0][0]
        market_sql = cur.execute.call_args_list[3][0][0]
        rule_sql = cur.execute.call_args_list[4][0][0]
        thesis_sql = cur.execute.call_args_list[5][0][0]
        params = [
            call[0][1] if len(call[0]) > 1 else ()
            for call in cur.execute.call_args_list
        ]

        assert "COALESCE(pa.status, 'pending')" in status_sql
        assert "JOIN decisions d ON d.playbook_action_id = pa.id" in reviewed_sql
        assert "pa.status = 'pending' OR pa.status IS NULL" in reviewed_sql
        assert "pa.status = 'deferred'" in deferred_sql
        assert "GROUP BY pa.ticker" in deferred_sql
        assert "market_closed = TRUE" in market_sql
        assert "ds.signal_type = 'rule_gate'" in rule_sql
        assert "sr.lift_condition" in rule_sql
        assert "LEFT JOIN positions" in thesis_sql
        assert "last_touched_date" in thesis_sql
        assert params == [(14,), (14,), (14,), (14,), (14,), ()]


class TestTickerQueries:
    def test_get_ticker_position_returns_row_when_held(self, cur):
        from dashboard.queries import get_ticker_position
        cur.fetchone.return_value = make_position_row(ticker="AAPL")
        assert get_ticker_position("AAPL")["ticker"] == "AAPL"

    def test_get_ticker_position_returns_none_when_not_held(self, cur):
        from dashboard.queries import get_ticker_position
        cur.fetchone.return_value = None
        assert get_ticker_position("XYZ") is None

    def test_get_ticker_theses_returns_all_statuses(self, cur):
        from dashboard.queries import get_ticker_theses
        cur.fetchall.return_value = [
            make_thesis_row(id=1, ticker="AAPL", status="active"),
            make_thesis_row(id=2, ticker="AAPL", status="expired"),
        ]
        result = get_ticker_theses("AAPL")
        assert len(result) == 2

    def test_get_ticker_decisions_filters_by_days(self, cur):
        from dashboard.queries import get_ticker_decisions
        cur.fetchall.return_value = [make_decision_row(id=1, ticker="AAPL")]
        result = get_ticker_decisions("AAPL", days=90)
        assert len(result) == 1
        assert "AAPL" in cur.execute.call_args[0][1]

    def test_get_ticker_signals_returns_news(self, cur):
        from dashboard.queries import get_ticker_signals
        cur.fetchall.return_value = [make_news_signal_row(ticker="AAPL")]
        assert get_ticker_signals("AAPL")[0]["ticker"] == "AAPL"

    def test_get_ticker_open_orders(self, cur):
        from dashboard.queries import get_ticker_open_orders
        cur.fetchall.return_value = [make_open_order_row(ticker="AAPL")]
        assert len(get_ticker_open_orders("AAPL")) == 1

    def test_get_ticker_attribution_groups_by_category(self, cur):
        from dashboard.queries import get_ticker_attribution
        cur.fetchall.return_value = [
            {"signal_type": "news_signal", "category": "earnings",
             "sample_size": 3, "avg_outcome_7d": Decimal("1.5"),
             "avg_outcome_30d": Decimal("3.0")},
        ]
        result = get_ticker_attribution("AAPL", days=90)
        assert result[0]["category"] == "earnings"


class TestSessionQueries:
    def test_get_session_returns_row(self, cur):
        from dashboard.queries import get_session
        cur.fetchone.return_value = make_session_row(id=1)
        assert get_session(1)["id"] == 1

    def test_get_session_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_session
        cur.fetchone.return_value = None
        assert get_session(999) is None

    def test_get_session_decisions(self, cur):
        from dashboard.queries import get_session_decisions
        cur.fetchall.return_value = [make_decision_row(id=1)]
        result = get_session_decisions(1)
        assert len(result) == 1
        # Should filter by session_id directly (no date-based JOIN)
        called_sql = cur.execute.call_args[0][0]
        assert "d.session_id = %s" in called_sql

    def test_get_session_decisions_is_id_scoped(self, cur):
        """Predicate must be on decisions.session_id, not a date join via sessions."""
        from dashboard.queries import get_session_decisions
        cur.fetchall.return_value = []
        get_session_decisions(42)
        called_sql = cur.execute.call_args[0][0]
        called_params = cur.execute.call_args[0][1]
        assert "d.session_id = %s" in called_sql
        assert "JOIN sessions" not in called_sql
        assert "s.session_date" not in called_sql
        assert called_params == (42,)

    def test_get_session_theses_created(self, cur):
        from dashboard.queries import get_session_theses_created
        cur.fetchall.return_value = [make_thesis_row(id=1)]
        result = get_session_theses_created(1)
        assert len(result) == 1

    def test_get_session_theses_created_is_id_scoped(self, cur):
        from dashboard.queries import get_session_theses_created
        cur.fetchall.return_value = []
        get_session_theses_created(42)
        called_sql = cur.execute.call_args[0][0]
        called_params = cur.execute.call_args[0][1]
        assert "t.session_id = %s" in called_sql
        assert "JOIN sessions" not in called_sql
        assert "s.session_date" not in called_sql
        assert called_params == (42,)

    def test_get_session_memo_returns_row(self, cur):
        from dashboard.queries import get_session_memo
        cur.fetchone.return_value = make_strategy_memo_row(id=1)
        assert get_session_memo(1)["id"] == 1

    def test_get_session_memo_returns_none(self, cur):
        from dashboard.queries import get_session_memo
        cur.fetchone.return_value = None
        assert get_session_memo(999) is None

    def test_get_session_memo_is_id_scoped(self, cur):
        from dashboard.queries import get_session_memo
        cur.fetchone.return_value = None
        get_session_memo(42)
        called_sql = cur.execute.call_args[0][0]
        called_params = cur.execute.call_args[0][1]
        assert "m.session_id = %s" in called_sql
        assert "JOIN sessions" not in called_sql
        assert "s.session_date" not in called_sql
        assert called_params == (42,)

    def test_get_session_tweets(self, cur):
        from dashboard.queries import get_session_tweets
        cur.fetchall.return_value = [make_tweet_row(id=1)]
        assert len(get_session_tweets(1)) == 1

    def test_get_session_tweets_is_id_scoped(self, cur):
        from dashboard.queries import get_session_tweets
        cur.fetchall.return_value = []
        get_session_tweets(42)
        called_sql = cur.execute.call_args[0][0]
        called_params = cur.execute.call_args[0][1]
        assert "tw.session_id = %s" in called_sql
        assert "JOIN sessions" not in called_sql
        assert "s.session_date" not in called_sql
        assert called_params == (42,)

    def test_get_session_events_uses_existing_filter(self, cur):
        from dashboard.queries import get_session_events
        cur.fetchall.return_value = [make_agent_event_row(session_id=1)]
        result = get_session_events(1, limit=50)
        assert len(result) == 1
        assert 1 in cur.execute.call_args[0][1]


class TestCategoryFilters:
    def test_recent_ticker_signals_filters_by_category(self, cur):
        from dashboard.queries import get_recent_ticker_signals
        cur.fetchall.return_value = []
        get_recent_ticker_signals(days=7, limit=50, category="earnings")
        params = cur.execute.call_args[0][1]
        assert "earnings" in params

    def test_recent_ticker_signals_no_category(self, cur):
        from dashboard.queries import get_recent_ticker_signals
        cur.fetchall.return_value = []
        get_recent_ticker_signals(days=7, limit=50)
        called_sql = cur.execute.call_args[0][0]
        # No category WHERE clause when not filtered
        assert "category = " not in called_sql

    def test_recent_macro_signals_filters_by_category(self, cur):
        from dashboard.queries import get_recent_macro_signals
        cur.fetchall.return_value = []
        get_recent_macro_signals(days=7, limit=20, category="fed")
        params = cur.execute.call_args[0][1]
        assert "fed" in params

    def test_signal_attribution_filters_by_category(self, cur):
        from dashboard.queries import get_signal_attribution
        cur.fetchall.return_value = []
        get_signal_attribution(category="news:earnings")
        params = cur.execute.call_args[0][1]
        assert "news:earnings" in params


class TestRecentTweetsExtended:
    def test_returns_decision_id_and_session_id(self, cur):
        from dashboard.queries import get_recent_tweets
        cur.fetchall.return_value = [
            {**make_tweet_row(id=1), "decision_id": 11, "session_id": 42},
        ]
        result = get_recent_tweets()
        assert result[0]["decision_id"] == 11
        assert result[0]["session_id"] == 42

    def test_joins_sessions_by_session_id_not_session_date(self, cur):
        """The session join must use tw.session_id, not tw.session_date.

        Per-run session rows allow multiple sessions on the same date; joining
        by date would fan out tweets across every same-date session row.
        """
        from dashboard.queries import get_recent_tweets
        cur.fetchall.return_value = []
        get_recent_tweets()
        sql = cur.execute.call_args[0][0]
        assert "s.id = tw.session_id" in sql
        assert "s.session_date = tw.session_date" not in sql


class TestGetSessionLlmCalls:
    def test_returns_summary_projection_excluding_jsonb(self, cur):
        from dashboard.queries import get_session_llm_calls
        cur.fetchall.return_value = [
            {
                "id": 1, "stage_name": "trading", "purpose": "executor",
                "sequence": 0, "model": "claude-haiku-4-5-20251001",
                "input_tokens": 120, "output_tokens": 45,
                "cache_read_tokens": 80, "cache_creation_tokens": 10,
                "stop_reason": "end_turn", "duration_ms": 987,
                "created_at": datetime(2026, 5, 13, 16, 30),
            }
        ]
        result = get_session_llm_calls(42)
        assert len(result) == 1
        assert result[0]["model"] == "claude-haiku-4-5-20251001"
        sql = cur.execute.call_args[0][0]
        # The SELECT must list only summary columns; JSONB columns absent.
        assert "messages" not in sql
        assert "tool_definitions" not in sql
        assert "response_content" not in sql
        # And the predicate is on session_id.
        assert "session_id = %s" in sql

    def test_orders_executor_then_strategist_then_reflection(self, cur):
        from dashboard.queries import get_session_llm_calls
        cur.fetchall.return_value = []
        get_session_llm_calls(42)
        sql = cur.execute.call_args[0][0]
        # The CASE expression must put executor=0, strategist_loop=1, reflection_loop=2
        # and then order by sequence.
        assert "executor" in sql
        assert "strategist_loop" in sql
        assert "reflection_loop" in sql
        assert "ORDER BY" in sql
        # Sequence is the tie-breaker.
        assert "sequence" in sql

    def test_passes_session_id_as_param(self, cur):
        from dashboard.queries import get_session_llm_calls
        cur.fetchall.return_value = []
        get_session_llm_calls(99)
        params = cur.execute.call_args[0][1]
        assert params == (99,)


class TestGetLlmCall:
    def test_returns_full_row_including_jsonb(self, cur):
        from dashboard.queries import get_llm_call
        cur.fetchone.return_value = {
            "id": 137, "session_id": 42, "stage_name": "trading",
            "purpose": "executor", "sequence": 0,
            "model": "claude-haiku-4-5-20251001",
            "system_prompt": "you are a trading executor",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_definitions": None,
            "response_content": [{"type": "text", "text": "ok"}],
            "input_tokens": 120, "output_tokens": 45,
            "cache_read_tokens": 80, "cache_creation_tokens": 10,
            "stop_reason": "end_turn", "duration_ms": 987,
            "created_at": datetime(2026, 5, 13, 16, 30),
        }
        result = get_llm_call(137)
        assert result["id"] == 137
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["response_content"] == [{"type": "text", "text": "ok"}]
        sql = cur.execute.call_args[0][0]
        assert "FROM llm_call_contexts" in sql
        assert "id = %s" in sql

    def test_returns_none_when_missing(self, cur):
        from dashboard.queries import get_llm_call
        cur.fetchone.return_value = None
        result = get_llm_call(99999)
        assert result is None

    def test_passes_id_as_param(self, cur):
        from dashboard.queries import get_llm_call
        cur.fetchone.return_value = None
        get_llm_call(137)
        assert cur.execute.call_args[0][1] == (137,)
