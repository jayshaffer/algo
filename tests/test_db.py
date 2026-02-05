"""Tests for trading/db.py - Database CRUD operations."""

from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, MagicMock, call

import pytest

from tests.conftest import (
    make_news_signal_row,
    make_macro_signal_row,
    make_snapshot_row,
    make_decision_row,
    make_position_row,
    make_thesis_row,
)


# ---------------------------------------------------------------------------
# get_connection
# ---------------------------------------------------------------------------

class TestGetConnection:
    """Tests for get_connection()."""

    def test_raises_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from trading.db import get_connection
        with pytest.raises(ValueError, match="DATABASE_URL must be set"):
            get_connection()

    @patch("trading.db.psycopg2.connect")
    def test_connects_with_database_url(self, mock_connect, db_env):
        from trading.db import get_connection
        get_connection()
        mock_connect.assert_called_once_with("postgresql://test:test@localhost:5432/test")


# ---------------------------------------------------------------------------
# News Signals
# ---------------------------------------------------------------------------

class TestInsertNewsSignal:
    """Tests for insert_news_signal()."""

    def test_inserts_and_returns_id(self, mock_db):
        from trading.db import insert_news_signal
        mock_db.fetchone.return_value = {"id": 42}

        result = insert_news_signal(
            ticker="AAPL",
            headline="AAPL earnings beat",
            category="earnings",
            sentiment="bullish",
            confidence="high",
            published_at=datetime(2025, 1, 15, 10, 0, 0),
        )

        assert result == 42
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO news_signals" in sql
        assert "RETURNING id" in sql

    def test_passes_correct_params(self, mock_db):
        from trading.db import insert_news_signal
        ts = datetime(2025, 1, 15, 10, 0, 0)
        insert_news_signal("TSLA", "Tesla news", "product", "bearish", "low", ts)

        params = mock_db.execute.call_args[0][1]
        assert params == ("TSLA", "Tesla news", "product", "bearish", "low", ts)


class TestInsertNewsSignalsBatch:
    """Tests for insert_news_signals_batch()."""

    def test_empty_list_returns_zero(self, mock_db):
        from trading.db import insert_news_signals_batch
        result = insert_news_signals_batch([])
        assert result == 0
        mock_db.execute.assert_not_called()

    @patch("trading.db.execute_values")
    def test_batch_insert_returns_count(self, mock_exec_values, mock_db):
        from trading.db import insert_news_signals_batch
        signals = [
            ("AAPL", "headline1", "earnings", "bullish", "high", datetime.now()),
            ("TSLA", "headline2", "product", "bearish", "medium", datetime.now()),
        ]
        result = insert_news_signals_batch(signals)
        assert result == 2
        mock_exec_values.assert_called_once()


class TestGetNewsSignals:
    """Tests for get_news_signals()."""

    def test_without_ticker_filter(self, mock_db):
        from trading.db import get_news_signals
        expected = [make_news_signal_row(), make_news_signal_row(id=2)]
        mock_db.fetchall.return_value = expected

        result = get_news_signals()
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "ticker = %s" not in sql

    def test_with_ticker_filter(self, mock_db):
        from trading.db import get_news_signals
        expected = [make_news_signal_row(ticker="AAPL")]
        mock_db.fetchall.return_value = expected

        result = get_news_signals(ticker="AAPL")
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "ticker = %s" in sql

    def test_custom_days_param(self, mock_db):
        from trading.db import get_news_signals
        get_news_signals(days=30)
        params = mock_db.execute.call_args[0][1]
        assert 30 in params


# ---------------------------------------------------------------------------
# Macro Signals
# ---------------------------------------------------------------------------

class TestInsertMacroSignal:
    """Tests for insert_macro_signal()."""

    def test_inserts_and_returns_id(self, mock_db):
        from trading.db import insert_macro_signal
        mock_db.fetchone.return_value = {"id": 55}

        result = insert_macro_signal(
            headline="Fed holds rates",
            category="fed",
            affected_sectors=["finance", "tech"],
            sentiment="bullish",
            published_at=datetime(2025, 1, 15, 10, 0, 0),
        )

        assert result == 55
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO macro_signals" in sql

    def test_passes_correct_params(self, mock_db):
        from trading.db import insert_macro_signal
        ts = datetime(2025, 6, 1)
        insert_macro_signal("Tariff news", "trade", ["manufacturing"], "bearish", ts)

        params = mock_db.execute.call_args[0][1]
        assert params == ("Tariff news", "trade", ["manufacturing"], "bearish", ts)


class TestInsertMacroSignalsBatch:
    """Tests for insert_macro_signals_batch()."""

    def test_empty_list_returns_zero(self, mock_db):
        from trading.db import insert_macro_signals_batch
        result = insert_macro_signals_batch([])
        assert result == 0

    @patch("trading.db.execute_values")
    def test_batch_insert_returns_count(self, mock_exec_values, mock_db):
        from trading.db import insert_macro_signals_batch
        signals = [
            ("headline1", "fed", ["finance"], "bullish", datetime.now()),
            ("headline2", "trade", ["manufacturing"], "bearish", datetime.now()),
            ("headline3", "geopolitical", ["energy"], "neutral", datetime.now()),
        ]
        result = insert_macro_signals_batch(signals)
        assert result == 3


class TestGetMacroSignals:
    """Tests for get_macro_signals()."""

    def test_without_category_filter(self, mock_db):
        from trading.db import get_macro_signals
        expected = [make_macro_signal_row()]
        mock_db.fetchall.return_value = expected

        result = get_macro_signals()
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "category = %s" not in sql

    def test_with_category_filter(self, mock_db):
        from trading.db import get_macro_signals
        expected = [make_macro_signal_row(category="fed")]
        mock_db.fetchall.return_value = expected

        result = get_macro_signals(category="fed")
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "category = %s" in sql

    def test_custom_days_param(self, mock_db):
        from trading.db import get_macro_signals
        get_macro_signals(days=14)
        params = mock_db.execute.call_args[0][1]
        assert 14 in params


# ---------------------------------------------------------------------------
# Account Snapshots
# ---------------------------------------------------------------------------

class TestInsertAccountSnapshot:
    """Tests for insert_account_snapshot()."""

    def test_inserts_and_returns_id(self, mock_db):
        from trading.db import insert_account_snapshot
        mock_db.fetchone.return_value = {"id": 10}

        result = insert_account_snapshot(
            snapshot_date=date(2025, 1, 15),
            cash=Decimal("100000"),
            portfolio_value=Decimal("150000"),
            buying_power=Decimal("200000"),
        )

        assert result == 10
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO account_snapshots" in sql
        assert "ON CONFLICT (date) DO UPDATE" in sql

    def test_passes_optional_market_values(self, mock_db):
        from trading.db import insert_account_snapshot
        insert_account_snapshot(
            snapshot_date=date(2025, 1, 15),
            cash=Decimal("100000"),
            portfolio_value=Decimal("150000"),
            buying_power=Decimal("200000"),
            long_market_value=Decimal("50000"),
            short_market_value=Decimal("10000"),
        )

        params = mock_db.execute.call_args[0][1]
        assert Decimal("50000") in params
        assert Decimal("10000") in params

    def test_optional_market_values_default_none(self, mock_db):
        from trading.db import insert_account_snapshot
        insert_account_snapshot(
            snapshot_date=date(2025, 1, 15),
            cash=Decimal("100000"),
            portfolio_value=Decimal("150000"),
            buying_power=Decimal("200000"),
        )

        params = mock_db.execute.call_args[0][1]
        # long_market_value and short_market_value should be None (positions 4, 5)
        assert params[4] is None
        assert params[5] is None


class TestGetAccountSnapshots:
    """Tests for get_account_snapshots()."""

    def test_returns_list(self, mock_db):
        from trading.db import get_account_snapshots
        expected = [make_snapshot_row(), make_snapshot_row(id=2)]
        mock_db.fetchall.return_value = expected

        result = get_account_snapshots()
        assert result == expected

    def test_default_30_days(self, mock_db):
        from trading.db import get_account_snapshots
        get_account_snapshots()
        params = mock_db.execute.call_args[0][1]
        assert 30 in params

    def test_custom_days(self, mock_db):
        from trading.db import get_account_snapshots
        get_account_snapshots(days=90)
        params = mock_db.execute.call_args[0][1]
        assert 90 in params


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

class TestInsertDecision:
    """Tests for insert_decision()."""

    def test_inserts_and_returns_id(self, mock_db):
        from trading.db import insert_decision
        mock_db.fetchone.return_value = {"id": 77}

        result = insert_decision(
            decision_date=date(2025, 1, 15),
            ticker="AAPL",
            action="buy",
            quantity=Decimal("10"),
            price=Decimal("150.00"),
            reasoning="Strong earnings",
            signals_used={"summary": "bullish"},
            account_equity=Decimal("100000"),
            buying_power=Decimal("50000"),
        )

        assert result == 77
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO decisions" in sql

    def test_wraps_signals_with_json(self, mock_db):
        from trading.db import insert_decision
        signals = {"market_summary": "bullish", "news": ["AAPL earnings"]}

        insert_decision(
            decision_date=date(2025, 1, 15),
            ticker="AAPL",
            action="buy",
            quantity=Decimal("10"),
            price=Decimal("150.00"),
            reasoning="Test",
            signals_used=signals,
            account_equity=Decimal("100000"),
            buying_power=Decimal("50000"),
        )

        # The 7th param (index 6) should be a Json wrapper
        params = mock_db.execute.call_args[0][1]
        from psycopg2.extras import Json
        assert isinstance(params[6], Json)


class TestGetRecentDecisions:
    """Tests for get_recent_decisions()."""

    def test_returns_list(self, mock_db):
        from trading.db import get_recent_decisions
        expected = [make_decision_row()]
        mock_db.fetchall.return_value = expected

        result = get_recent_decisions()
        assert result == expected

    def test_default_30_days(self, mock_db):
        from trading.db import get_recent_decisions
        get_recent_decisions()
        params = mock_db.execute.call_args[0][1]
        assert 30 in params

    def test_custom_days(self, mock_db):
        from trading.db import get_recent_decisions
        get_recent_decisions(days=60)
        params = mock_db.execute.call_args[0][1]
        assert 60 in params


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

class TestUpsertPosition:
    """Tests for upsert_position()."""

    def test_upserts_and_returns_id(self, mock_db):
        from trading.db import upsert_position
        mock_db.fetchone.return_value = {"id": 5}

        result = upsert_position("AAPL", Decimal("10"), Decimal("150.00"))

        assert result == 5
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO positions" in sql
        assert "ON CONFLICT (ticker) DO UPDATE" in sql

    def test_passes_correct_params(self, mock_db):
        from trading.db import upsert_position
        upsert_position("TSLA", Decimal("5"), Decimal("250.00"))

        params = mock_db.execute.call_args[0][1]
        assert params == ("TSLA", Decimal("5"), Decimal("250.00"))


class TestGetPositions:
    """Tests for get_positions()."""

    def test_returns_ordered_list(self, mock_db):
        from trading.db import get_positions
        expected = [
            make_position_row(ticker="AAPL"),
            make_position_row(ticker="TSLA", id=2),
        ]
        mock_db.fetchall.return_value = expected

        result = get_positions()
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "ORDER BY ticker" in sql

    def test_returns_empty_list(self, mock_db):
        from trading.db import get_positions
        mock_db.fetchall.return_value = []
        result = get_positions()
        assert result == []


class TestDeletePosition:
    """Tests for delete_position()."""

    def test_returns_true_when_deleted(self, mock_db):
        from trading.db import delete_position
        mock_db.rowcount = 1

        result = delete_position("AAPL")
        assert result is True

        sql = mock_db.execute.call_args[0][0]
        assert "DELETE FROM positions" in sql

    def test_returns_false_when_not_found(self, mock_db):
        from trading.db import delete_position
        mock_db.rowcount = 0

        result = delete_position("NONEXISTENT")
        assert result is False


# ---------------------------------------------------------------------------
# Theses
# ---------------------------------------------------------------------------

class TestInsertThesis:
    """Tests for insert_thesis()."""

    def test_inserts_and_returns_id(self, mock_db):
        from trading.db import insert_thesis
        mock_db.fetchone.return_value = {"id": 99}

        result = insert_thesis(
            ticker="AAPL",
            direction="long",
            thesis="Strong fundamentals",
        )

        assert result == 99
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO theses" in sql

    def test_uses_defaults_for_optional_params(self, mock_db):
        from trading.db import insert_thesis
        insert_thesis(ticker="AAPL", direction="long", thesis="Test")

        params = mock_db.execute.call_args[0][1]
        # (ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source)
        assert params[0] == "AAPL"
        assert params[1] == "long"
        assert params[2] == "Test"
        assert params[3] is None  # entry_trigger
        assert params[4] is None  # exit_trigger
        assert params[5] is None  # invalidation
        assert params[6] == "medium"  # confidence default
        assert params[7] == "ideation"  # source default

    def test_all_params_passed(self, mock_db):
        from trading.db import insert_thesis
        insert_thesis(
            ticker="TSLA",
            direction="short",
            thesis="Overvalued",
            entry_trigger="Price above $300",
            exit_trigger="Price below $200",
            invalidation="Revenue accelerates",
            confidence="high",
            source="manual",
        )

        params = mock_db.execute.call_args[0][1]
        assert params == (
            "TSLA", "short", "Overvalued", "Price above $300",
            "Price below $200", "Revenue accelerates", "high", "manual",
        )


class TestGetActiveTheses:
    """Tests for get_active_theses()."""

    def test_without_ticker_filter(self, mock_db):
        from trading.db import get_active_theses
        expected = [make_thesis_row()]
        mock_db.fetchall.return_value = expected

        result = get_active_theses()
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "status = 'active'" in sql
        assert "ticker = %s" not in sql

    def test_with_ticker_filter(self, mock_db):
        from trading.db import get_active_theses
        expected = [make_thesis_row(ticker="AAPL")]
        mock_db.fetchall.return_value = expected

        result = get_active_theses(ticker="AAPL")
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "status = 'active'" in sql
        assert "ticker = %s" in sql


class TestUpdateThesis:
    """Tests for update_thesis()."""

    def test_returns_false_when_no_updates(self, mock_db):
        from trading.db import update_thesis
        result = update_thesis(thesis_id=1)
        assert result is False
        mock_db.execute.assert_not_called()

    def test_updates_single_field(self, mock_db):
        from trading.db import update_thesis
        mock_db.rowcount = 1

        result = update_thesis(thesis_id=1, thesis="Updated thesis")
        assert result is True

        sql = mock_db.execute.call_args[0][0]
        assert "thesis = %s" in sql
        assert "updated_at = NOW()" in sql

    def test_updates_multiple_fields(self, mock_db):
        from trading.db import update_thesis
        mock_db.rowcount = 1

        result = update_thesis(
            thesis_id=1,
            thesis="New thesis",
            confidence="low",
            entry_trigger="New entry",
        )
        assert result is True

        sql = mock_db.execute.call_args[0][0]
        assert "thesis = %s" in sql
        assert "confidence = %s" in sql
        assert "entry_trigger = %s" in sql

    def test_returns_false_when_row_not_found(self, mock_db):
        from trading.db import update_thesis
        mock_db.rowcount = 0

        result = update_thesis(thesis_id=999, thesis="Updated")
        assert result is False

    def test_dynamic_set_clause_includes_all_fields(self, mock_db):
        from trading.db import update_thesis
        mock_db.rowcount = 1

        update_thesis(
            thesis_id=1,
            thesis="t",
            entry_trigger="e",
            exit_trigger="x",
            invalidation="i",
            confidence="c",
        )

        sql = mock_db.execute.call_args[0][0]
        for field in ["thesis", "entry_trigger", "exit_trigger", "invalidation", "confidence"]:
            assert f"{field} = %s" in sql


class TestCloseThesis:
    """Tests for close_thesis()."""

    def test_closes_and_returns_true(self, mock_db):
        from trading.db import close_thesis
        mock_db.rowcount = 1

        result = close_thesis(thesis_id=1, status="invalidated", reason="Revenue declined")
        assert result is True

        sql = mock_db.execute.call_args[0][0]
        assert "status = %s" in sql
        assert "close_reason = %s" in sql
        assert "closed_at = NOW()" in sql

    def test_returns_false_when_not_found(self, mock_db):
        from trading.db import close_thesis
        mock_db.rowcount = 0

        result = close_thesis(thesis_id=999, status="closed", reason="Done")
        assert result is False

    def test_passes_correct_params(self, mock_db):
        from trading.db import close_thesis
        close_thesis(thesis_id=42, status="realized", reason="Target hit")

        params = mock_db.execute.call_args[0][1]
        assert params == ("realized", "Target hit", 42)


# ---------------------------------------------------------------------------
# Open Orders
# ---------------------------------------------------------------------------

class TestUpsertOpenOrder:
    """Tests for upsert_open_order()."""

    def test_upserts_and_returns_id(self, mock_db):
        from trading.db import upsert_open_order
        mock_db.fetchone.return_value = {"id": 33}

        result = upsert_open_order(
            order_id="ord-123",
            ticker="AAPL",
            side="buy",
            order_type="limit",
            qty=Decimal("10"),
            filled_qty=Decimal("0"),
            limit_price=Decimal("150.00"),
            stop_price=None,
            status="new",
            submitted_at=datetime(2025, 1, 15, 10, 0, 0),
        )

        assert result == 33
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO open_orders" in sql
        assert "ON CONFLICT (order_id) DO UPDATE" in sql

    def test_passes_all_params(self, mock_db):
        from trading.db import upsert_open_order
        ts = datetime(2025, 6, 1, 14, 30)
        upsert_open_order(
            order_id="ord-456",
            ticker="TSLA",
            side="sell",
            order_type="market",
            qty=Decimal("5"),
            filled_qty=Decimal("3"),
            limit_price=None,
            stop_price=Decimal("200.00"),
            status="partially_filled",
            submitted_at=ts,
        )

        params = mock_db.execute.call_args[0][1]
        assert params == (
            "ord-456", "TSLA", "sell", "market",
            Decimal("5"), Decimal("3"), None, Decimal("200.00"),
            "partially_filled", ts,
        )


class TestGetOpenOrders:
    """Tests for get_open_orders()."""

    def test_returns_list(self, mock_db):
        from trading.db import get_open_orders
        expected = [{"id": 1, "order_id": "ord-1", "ticker": "AAPL"}]
        mock_db.fetchall.return_value = expected

        result = get_open_orders()
        assert result == expected

        sql = mock_db.execute.call_args[0][0]
        assert "ORDER BY submitted_at DESC" in sql

    def test_returns_empty_list(self, mock_db):
        from trading.db import get_open_orders
        mock_db.fetchall.return_value = []
        result = get_open_orders()
        assert result == []


class TestDeleteOpenOrder:
    """Tests for delete_open_order()."""

    def test_returns_true_when_deleted(self, mock_db):
        from trading.db import delete_open_order
        mock_db.rowcount = 1

        result = delete_open_order("ord-123")
        assert result is True

    def test_returns_false_when_not_found(self, mock_db):
        from trading.db import delete_open_order
        mock_db.rowcount = 0

        result = delete_open_order("nonexistent")
        assert result is False

    def test_deletes_by_order_id(self, mock_db):
        from trading.db import delete_open_order
        delete_open_order("ord-789")

        sql = mock_db.execute.call_args[0][0]
        assert "DELETE FROM open_orders WHERE order_id = %s" in sql
        params = mock_db.execute.call_args[0][1]
        assert params == ("ord-789",)


class TestClearClosedOrders:
    """Tests for clear_closed_orders()."""

    def test_returns_count_of_deleted(self, mock_db):
        from trading.db import clear_closed_orders
        mock_db.rowcount = 5

        result = clear_closed_orders()
        assert result == 5

    def test_returns_zero_when_none_cleared(self, mock_db):
        from trading.db import clear_closed_orders
        mock_db.rowcount = 0

        result = clear_closed_orders()
        assert result == 0

    def test_deletes_non_open_statuses(self, mock_db):
        from trading.db import clear_closed_orders
        clear_closed_orders()

        sql = mock_db.execute.call_args[0][0]
        assert "DELETE FROM open_orders" in sql
        assert "status NOT IN" in sql
        for status in ["new", "accepted", "pending_new", "partially_filled"]:
            assert status in sql
