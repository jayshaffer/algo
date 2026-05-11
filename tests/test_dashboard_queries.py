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
