"""Tests for new database functions: playbooks, decision_signals, signal_attribution."""

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest

from trading.db import (
    upsert_playbook,
    get_playbook,
    insert_decision_signal,
    insert_decision_signals_batch,
    get_decision_signals,
    upsert_signal_attribution,
    get_signal_attribution,
)


@pytest.fixture
def mock_cursor():
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 1}
    cursor.fetchall.return_value = []
    cursor.rowcount = 1
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("trading.db.get_cursor", _get_cursor), \
         patch("trading.db.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor


class TestPlaybooks:
    def test_upsert_playbook(self, mock_db):
        result = upsert_playbook(
            playbook_date=date.today(),
            market_outlook="Bullish tech",
            priority_actions=[{"ticker": "NVDA", "action": "buy"}],
            watch_list=["AAPL", "MSFT"],
            risk_notes="Watch Fed meeting",
        )
        assert result == 1
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO playbooks" in sql
        assert "ON CONFLICT (date)" in sql

    def test_get_playbook_returns_row(self, mock_db):
        mock_db.fetchone.return_value = {
            "id": 1,
            "date": date.today(),
            "market_outlook": "Bullish",
            "priority_actions": [],
            "watch_list": [],
            "risk_notes": "",
        }
        result = get_playbook(date.today())
        assert result["market_outlook"] == "Bullish"

    def test_get_playbook_returns_none(self, mock_db):
        mock_db.fetchone.return_value = None
        result = get_playbook(date.today())
        assert result is None


class TestDecisionSignals:
    def test_insert_decision_signal(self, mock_db):
        insert_decision_signal(decision_id=1, signal_type="news_signal", signal_id=42)
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO decision_signals" in sql

    @patch("trading.db.execute_values")
    def test_insert_batch(self, mock_exec_values, mock_db):
        signals = [
            (1, "news_signal", 42),
            (1, "thesis", 5),
        ]
        result = insert_decision_signals_batch(signals)
        assert result == 2
        mock_exec_values.assert_called_once()

    def test_insert_batch_empty(self, mock_db):
        result = insert_decision_signals_batch([])
        assert result == 0

    def test_get_decision_signals(self, mock_db):
        mock_db.fetchall.return_value = [
            {"decision_id": 1, "signal_type": "news_signal", "signal_id": 42},
        ]
        result = get_decision_signals(decision_id=1)
        assert len(result) == 1


class TestSignalAttribution:
    def test_upsert_signal_attribution(self, mock_db):
        upsert_signal_attribution(
            category="news:earnings",
            sample_size=20,
            avg_outcome_7d=Decimal("1.5"),
            avg_outcome_30d=Decimal("3.2"),
            win_rate_7d=Decimal("0.62"),
            win_rate_30d=Decimal("0.55"),
        )
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO signal_attribution" in sql
        assert "ON CONFLICT (category)" in sql

    def test_get_signal_attribution(self, mock_db):
        mock_db.fetchall.return_value = [
            {"category": "news:earnings", "sample_size": 20, "win_rate_7d": Decimal("0.62")},
        ]
        result = get_signal_attribution()
        assert len(result) == 1
