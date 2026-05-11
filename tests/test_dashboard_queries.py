"""Direct unit tests for dashboard/queries.py SQL functions.

These tests patch `dashboard.queries.get_cursor` to a context manager that
yields a MagicMock cursor — no Flask app, no DB. We assert the SQL string
and bound parameters where it matters, and the shaped return value always.
"""

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    make_agent_event_row,
    make_decision_row,
    make_macro_signal_row,
    make_news_signal_row,
    make_open_order_row,
    make_playbook_action_row,
    make_position_row,
    make_session_row,
    make_session_stage_cost_row,
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
