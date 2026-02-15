"""Tests for v2 database layer."""
import pytest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal


class TestGetConnection:
    def test_reads_database_url(self):
        with patch("v2.database.connection.psycopg2") as mock_pg:
            with patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}):
                from v2.database.connection import get_connection
                get_connection()
            mock_pg.connect.assert_called_once_with("postgresql://test")

    def test_raises_without_database_url(self):
        with patch.dict("os.environ", {}, clear=True):
            from v2.database.connection import get_connection
            with pytest.raises(ValueError, match="DATABASE_URL"):
                get_connection()


class TestGetCursor:
    def test_yields_realdict_cursor_and_commits(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("v2.database.connection.get_connection", return_value=mock_conn):
            from v2.database.connection import get_cursor
            with get_cursor() as cur:
                cur.execute("SELECT 1")
            mock_conn.commit.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_rollback_on_exception(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("v2.database.connection.get_connection", return_value=mock_conn):
            from v2.database.connection import get_cursor
            with pytest.raises(RuntimeError):
                with get_cursor() as cur:
                    raise RuntimeError("test error")
            mock_conn.rollback.assert_called_once()
            mock_conn.close.assert_called_once()


class TestNewsSignals:
    def test_insert_news_signal(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_news_signal
        result = insert_news_signal("AAPL", "Test", "earnings", "bullish", "high", datetime.now())
        assert result == 1
        assert "INSERT INTO news_signals" in mock_cursor.execute.call_args[0][0]

    def test_insert_news_signals_batch_empty(self, mock_db, mock_cursor):
        from v2.database.trading_db import insert_news_signals_batch
        assert insert_news_signals_batch([]) == 0

    @patch("v2.database.trading_db.execute_values")
    def test_insert_news_signals_batch(self, mock_exec_values, mock_db, mock_cursor):
        from v2.database.trading_db import insert_news_signals_batch
        signals = [("AAPL", "Test", "earnings", "bullish", "high", datetime.now())]
        result = insert_news_signals_batch(signals)
        assert result == 1
        mock_exec_values.assert_called_once()

    def test_get_news_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "ticker": "AAPL"}]
        from v2.database.trading_db import get_news_signals
        result = get_news_signals(days=7)
        assert len(result) == 1

    def test_get_news_signals_by_ticker(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.trading_db import get_news_signals
        get_news_signals(ticker="AAPL", days=7)
        sql = mock_cursor.execute.call_args[0][0]
        assert "ticker = %s" in sql


class TestMacroSignals:
    def test_insert_macro_signal(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_macro_signal
        result = insert_macro_signal("Fed holds", "fed", ["finance"], "neutral", datetime.now())
        assert result == 1

    def test_insert_macro_signals_batch_empty(self, mock_db, mock_cursor):
        from v2.database.trading_db import insert_macro_signals_batch
        assert insert_macro_signals_batch([]) == 0

    @patch("v2.database.trading_db.execute_values")
    def test_insert_macro_signals_batch(self, mock_exec_values, mock_db, mock_cursor):
        from v2.database.trading_db import insert_macro_signals_batch
        signals = [("Fed holds", "fed", ["finance"], "neutral", datetime.now())]
        result = insert_macro_signals_batch(signals)
        assert result == 1
        mock_exec_values.assert_called_once()

    def test_get_macro_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "category": "fed"}]
        from v2.database.trading_db import get_macro_signals
        result = get_macro_signals(days=7)
        assert len(result) == 1

    def test_get_macro_signals_by_category(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.trading_db import get_macro_signals
        get_macro_signals(category="fed", days=7)
        sql = mock_cursor.execute.call_args[0][0]
        assert "category = %s" in sql


class TestAccountSnapshots:
    def test_insert_account_snapshot(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_account_snapshot
        result = insert_account_snapshot(
            date.today(),
            Decimal("50000"),
            Decimal("100000"),
            Decimal("50000"),
            Decimal("50000"),
            Decimal("0")
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO account_snapshots" in sql
        assert "ON CONFLICT (date) DO UPDATE" in sql

    def test_get_account_snapshots(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"date": date.today(), "portfolio_value": Decimal("100000")}
        ]
        from v2.database.trading_db import get_account_snapshots
        result = get_account_snapshots(days=30)
        assert len(result) == 1


class TestPositions:
    def test_upsert_position(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import upsert_position
        result = upsert_position("AAPL", Decimal("10"), Decimal("150"))
        assert result == 1

    def test_get_positions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"ticker": "AAPL", "shares": Decimal("10")}]
        from v2.database.trading_db import get_positions
        result = get_positions()
        assert len(result) == 1

    def test_delete_position(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 1
        from v2.database.trading_db import delete_position
        result = delete_position("AAPL")
        assert result is True

    def test_delete_all_positions(self, mock_db, mock_cursor):
        from v2.database.trading_db import delete_all_positions
        delete_all_positions()
        assert "DELETE FROM positions" in mock_cursor.execute.call_args[0][0]


class TestDecisions:
    def test_insert_decision_with_v3_fields(self, mock_db, mock_cursor):
        """V3: insert_decision should accept playbook_action_id and is_off_playbook."""
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_decision
        result = insert_decision(
            date.today(), "AAPL", "buy", Decimal("5"), Decimal("150"),
            "Test", {}, Decimal("100000"), Decimal("50000"),
            playbook_action_id=1, is_off_playbook=False
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "playbook_action_id" in sql
        assert "is_off_playbook" in sql

    def test_insert_decision_without_v3_fields(self, mock_db, mock_cursor):
        """Test backward compatibility when V3 fields are not provided."""
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_decision
        result = insert_decision(
            date.today(), "AAPL", "buy", Decimal("5"), Decimal("150"),
            "Test", {}, Decimal("100000"), Decimal("50000")
        )
        assert result == 1

    def test_get_recent_decisions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.trading_db import get_recent_decisions
        result = get_recent_decisions(days=30)
        assert result == []

    def test_update_decision_outcome_7d(self, mock_db, mock_cursor):
        from v2.database.trading_db import update_decision_outcome
        update_decision_outcome(1, outcome_7d=Decimal("2.5"))
        sql = mock_cursor.execute.call_args[0][0]
        assert "outcome_7d" in sql

    def test_update_decision_outcome_30d(self, mock_db, mock_cursor):
        from v2.database.trading_db import update_decision_outcome
        update_decision_outcome(1, outcome_30d=Decimal("5.0"))
        sql = mock_cursor.execute.call_args[0][0]
        assert "outcome_30d" in sql

    def test_update_decision_outcome_both(self, mock_db, mock_cursor):
        from v2.database.trading_db import update_decision_outcome
        update_decision_outcome(1, outcome_7d=Decimal("2.5"), outcome_30d=Decimal("5.0"))
        sql = mock_cursor.execute.call_args[0][0]
        assert "outcome_7d" in sql
        assert "outcome_30d" in sql

    def test_update_decision_outcome_neither(self, mock_db, mock_cursor):
        from v2.database.trading_db import update_decision_outcome
        update_decision_outcome(1)
        # Should not execute any query
        mock_cursor.execute.assert_not_called()

    def test_get_decisions_needing_backfill_7d(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.trading_db import get_decisions_needing_backfill_7d
        result = get_decisions_needing_backfill_7d()
        sql = mock_cursor.execute.call_args[0][0]
        assert "outcome_7d IS NULL" in sql
        assert "7 days" in sql

    def test_get_decisions_needing_backfill_30d(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.trading_db import get_decisions_needing_backfill_30d
        result = get_decisions_needing_backfill_30d()
        sql = mock_cursor.execute.call_args[0][0]
        assert "outcome_30d IS NULL" in sql
        assert "30 days" in sql


class TestTheses:
    def test_insert_thesis(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_thesis
        result = insert_thesis("AAPL", "long", "Strong earnings")
        assert result == 1

    def test_insert_thesis_with_all_fields(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_thesis
        result = insert_thesis(
            "AAPL", "long", "Strong earnings",
            entry_trigger="Price > $150",
            exit_trigger="Price > $180",
            invalidation="Earnings miss",
            confidence="high",
            source="ideation"
        )
        assert result == 1

    def test_get_active_theses(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "ticker": "AAPL", "status": "active"}]
        from v2.database.trading_db import get_active_theses
        result = get_active_theses()
        assert len(result) == 1

    def test_get_active_theses_by_ticker(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.trading_db import get_active_theses
        get_active_theses(ticker="AAPL")
        sql = mock_cursor.execute.call_args[0][0]
        assert "ticker = %s" in sql

    def test_get_thesis_by_id(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1, "ticker": "AAPL"}
        from v2.database.trading_db import get_thesis_by_id
        result = get_thesis_by_id(1)
        assert result["ticker"] == "AAPL"

    def test_get_thesis_by_id_not_found(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.trading_db import get_thesis_by_id
        result = get_thesis_by_id(999)
        assert result is None

    def test_update_thesis(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 1
        from v2.database.trading_db import update_thesis
        result = update_thesis(1, thesis="Updated thesis")
        assert result is True

    def test_update_thesis_multiple_fields(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 1
        from v2.database.trading_db import update_thesis
        result = update_thesis(
            1,
            thesis="Updated thesis",
            entry_trigger="New trigger",
            confidence="high"
        )
        assert result is True

    def test_update_thesis_no_fields(self, mock_db, mock_cursor):
        from v2.database.trading_db import update_thesis
        result = update_thesis(1)
        assert result is False

    def test_close_thesis(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 1
        from v2.database.trading_db import close_thesis
        result = close_thesis(1, "executed", "Trade executed")
        assert result is True
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = %s" in sql
        assert "close_reason = %s" in sql


class TestOpenOrders:
    def test_upsert_open_order(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import upsert_open_order
        result = upsert_open_order(
            "order123", "AAPL", "buy", "market",
            Decimal("10"), Decimal("0"), None, None,
            "accepted", datetime.now()
        )
        assert result == 1

    def test_get_open_orders(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"order_id": "order123", "ticker": "AAPL"}]
        from v2.database.trading_db import get_open_orders
        result = get_open_orders()
        assert len(result) == 1

    def test_delete_open_order(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 1
        from v2.database.trading_db import delete_open_order
        result = delete_open_order("order123")
        assert result is True

    def test_delete_all_open_orders(self, mock_db, mock_cursor):
        from v2.database.trading_db import delete_all_open_orders
        delete_all_open_orders()
        assert "DELETE FROM open_orders" in mock_cursor.execute.call_args[0][0]

    def test_clear_closed_orders(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 2
        from v2.database.trading_db import clear_closed_orders
        result = clear_closed_orders()
        assert result == 2


class TestPlaybooks:
    def test_upsert_playbook(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import upsert_playbook
        result = upsert_playbook(date.today(), "Bullish", [], ["AAPL"], "None")
        assert result == 1

    def test_get_playbook(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1, "date": date.today()}
        from v2.database.trading_db import get_playbook
        result = get_playbook(date.today())
        assert result["id"] == 1

    def test_get_playbook_not_found(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.trading_db import get_playbook
        result = get_playbook(date.today())
        assert result is None


class TestPlaybookActions:
    def test_insert_playbook_action(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_playbook_action
        result = insert_playbook_action(1, "AAPL", "buy", 1, "Entry hit", "high", Decimal("5"), 1)
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "playbook_actions" in sql

    def test_get_playbook_actions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "ticker": "AAPL"}]
        from v2.database.trading_db import get_playbook_actions
        result = get_playbook_actions(1)
        assert len(result) == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "ORDER BY priority" in sql

    def test_delete_playbook_actions(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 2
        from v2.database.trading_db import delete_playbook_actions
        result = delete_playbook_actions(1)
        assert result == 2


class TestSignalAttribution:
    def test_upsert_signal_attribution(self, mock_db, mock_cursor):
        from v2.database.trading_db import upsert_signal_attribution
        upsert_signal_attribution("news_signal:earnings", 10, Decimal("1.5"), Decimal("2.0"), Decimal("0.6"), Decimal("0.55"))
        sql = mock_cursor.execute.call_args[0][0]
        assert "signal_attribution" in sql
        assert "ON CONFLICT (category) DO UPDATE" in sql

    def test_get_signal_attribution(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.trading_db import get_signal_attribution
        result = get_signal_attribution()
        assert result == []


class TestDecisionSignals:
    def test_insert_decision_signal(self, mock_db, mock_cursor):
        from v2.database.trading_db import insert_decision_signal
        insert_decision_signal(1, "news_signal", 1)
        sql = mock_cursor.execute.call_args[0][0]
        assert "decision_signals" in sql

    def test_insert_decision_signals_batch_empty(self, mock_db, mock_cursor):
        from v2.database.trading_db import insert_decision_signals_batch
        assert insert_decision_signals_batch([]) == 0

    @patch("v2.database.trading_db.execute_values")
    def test_insert_decision_signals_batch(self, mock_exec_values, mock_db, mock_cursor):
        from v2.database.trading_db import insert_decision_signals_batch
        signals = [(1, "news_signal", 1), (1, "macro_signal", 2)]
        result = insert_decision_signals_batch(signals)
        assert result == 2
        mock_exec_values.assert_called_once()

    def test_get_decision_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"signal_type": "news_signal", "signal_id": 1}]
        from v2.database.trading_db import get_decision_signals
        result = get_decision_signals(1)
        assert len(result) == 1


class TestDashboardQueries:
    def test_get_positions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.dashboard_db import get_positions
        result = get_positions()
        assert result == []

    def test_get_latest_snapshot(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"portfolio_value": Decimal("100000")}
        from v2.database.dashboard_db import get_latest_snapshot
        result = get_latest_snapshot()
        assert result["portfolio_value"] == Decimal("100000")

    def test_get_latest_snapshot_none(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.dashboard_db import get_latest_snapshot
        result = get_latest_snapshot()
        assert result is None

    def test_get_recent_ticker_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"ticker": "AAPL", "category": "earnings"}]
        from v2.database.dashboard_db import get_recent_ticker_signals
        result = get_recent_ticker_signals(days=7, limit=50)
        assert len(result) == 1

    def test_get_recent_macro_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"category": "fed", "sentiment": "neutral"}]
        from v2.database.dashboard_db import get_recent_macro_signals
        result = get_recent_macro_signals(days=7, limit=20)
        assert len(result) == 1

    def test_get_signal_summary(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"ticker": "AAPL", "total": 5, "bullish": 3, "bearish": 1, "neutral": 1}
        ]
        from v2.database.dashboard_db import get_signal_summary
        result = get_signal_summary(days=7)
        assert len(result) == 1

    def test_get_recent_decisions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"ticker": "AAPL", "action": "buy"}]
        from v2.database.dashboard_db import get_recent_decisions
        result = get_recent_decisions(days=30, limit=50)
        assert len(result) == 1

    def test_get_decision_stats(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {
            "total_decisions": 10, "buys": 5, "sells": 3, "holds": 2,
            "avg_outcome_7d": Decimal("1.5"), "avg_outcome_30d": Decimal("3.0")
        }
        from v2.database.dashboard_db import get_decision_stats
        result = get_decision_stats(days=30)
        assert result["total_decisions"] == 10

    def test_get_equity_curve(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.dashboard_db import get_equity_curve
        result = get_equity_curve(days=90)
        assert result == []

    def test_get_performance_metrics_with_data(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {
            "start_value": Decimal("100000"),
            "end_value": Decimal("105000"),
            "start_date": date.today(),
            "end_date": date.today()
        }
        from v2.database.dashboard_db import get_performance_metrics
        result = get_performance_metrics(days=30)
        assert result is not None
        assert result["pnl"] == 5000.0
        assert abs(result["pnl_pct"] - 5.0) < 0.001

    def test_get_performance_metrics_no_data(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.dashboard_db import get_performance_metrics
        result = get_performance_metrics(days=30)
        assert result is None

    def test_get_performance_metrics_missing_values(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"start_value": None, "end_value": None}
        from v2.database.dashboard_db import get_performance_metrics
        result = get_performance_metrics(days=30)
        assert result is None

    def test_get_today_playbook(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.dashboard_db import get_today_playbook
        result = get_today_playbook()
        assert result is None

    def test_get_today_playbook_with_data(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1, "market_outlook": "Bullish"}
        from v2.database.dashboard_db import get_today_playbook
        result = get_today_playbook()
        assert result["id"] == 1

    def test_get_signal_attribution(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"category": "earnings", "sample_size": 10, "avg_outcome_7d": Decimal("2.0")}
        ]
        from v2.database.dashboard_db import get_signal_attribution
        result = get_signal_attribution()
        assert len(result) == 1

    def test_get_decision_signal_refs(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"signal_type": "news_signal", "signal_id": 1}
        ]
        from v2.database.dashboard_db import get_decision_signal_refs
        result = get_decision_signal_refs(1)
        assert len(result) == 1

    def test_get_thesis_stats(self, mock_db, mock_cursor):
        # First call for counts
        mock_cursor.fetchone.return_value = {
            "active": 5, "executed": 3, "invalidated": 1, "expired": 2
        }
        # Second call for confidence distribution
        mock_cursor.fetchall.return_value = [
            {"confidence": "high", "count": 3},
            {"confidence": "medium", "count": 2}
        ]
        from v2.database.dashboard_db import get_thesis_stats
        result = get_thesis_stats()
        assert result["active"] == 5
        assert result["executed"] == 3
        assert result["confidence_dist"]["high"] == 3

    def test_close_thesis(self, mock_db, mock_cursor):
        mock_conn = MagicMock()
        mock_cursor_obj = MagicMock()
        mock_cursor_obj.rowcount = 1
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor_obj)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("v2.database.dashboard_db.get_connection", return_value=mock_conn):
            from v2.database.dashboard_db import close_thesis
            result = close_thesis(1, "invalidated", "Market changed")
            assert result is True

    def test_close_thesis_invalid_status(self, mock_db, mock_cursor):
        from v2.database.dashboard_db import close_thesis
        with pytest.raises(ValueError, match="Invalid close status"):
            close_thesis(1, "executed", "Should fail")

    def test_get_theses_active(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "status": "active"}]
        from v2.database.dashboard_db import get_theses
        result = get_theses(status_filter="active", sort_by="newest")
        assert len(result) == 1

    def test_get_theses_all(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.dashboard_db import get_theses
        result = get_theses(status_filter="all", sort_by="ticker")
        assert result == []

    def test_get_theses_sort_by_confidence(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.database.dashboard_db import get_theses
        get_theses(status_filter="active", sort_by="confidence")
        sql = mock_cursor.execute.call_args[0][0]
        assert "CASE confidence" in sql


class TestStrategyState:
    def test_insert_strategy_state(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_strategy_state
        result = insert_strategy_state(
            identity_text="Momentum-focused trader",
            risk_posture="moderate",
            sector_biases={"tech": "overweight"},
            preferred_signals=["earnings"],
            avoided_signals=["legal"],
            version=1,
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO strategy_state" in sql

    def test_get_current_strategy_state(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1, "identity_text": "test", "is_current": True}
        from v2.database.trading_db import get_current_strategy_state
        result = get_current_strategy_state()
        assert result is not None
        sql = mock_cursor.execute.call_args[0][0]
        assert "is_current = TRUE" in sql

    def test_get_current_strategy_state_none(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.trading_db import get_current_strategy_state
        result = get_current_strategy_state()
        assert result is None

    def test_clear_current_strategy_state(self, mock_db, mock_cursor):
        from v2.database.trading_db import clear_current_strategy_state
        clear_current_strategy_state()
        sql = mock_cursor.execute.call_args[0][0]
        assert "UPDATE strategy_state" in sql
        assert "is_current = FALSE" in sql


class TestStrategyRules:
    def test_insert_strategy_rule(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_strategy_rule
        result = insert_strategy_rule(
            rule_text="Fade legal news signals",
            category="news_signal:legal",
            direction="constraint",
            confidence=0.8,
            supporting_evidence="38% win rate over 12 trades",
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO strategy_rules" in sql

    def test_get_active_strategy_rules(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "rule_text": "test", "status": "active"}]
        from v2.database.trading_db import get_active_strategy_rules
        result = get_active_strategy_rules()
        assert len(result) == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'active'" in sql

    def test_retire_strategy_rule(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 1
        from v2.database.trading_db import retire_strategy_rule
        result = retire_strategy_rule(rule_id=1)
        assert result is True
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'retired'" in sql
        assert "retired_at" in sql

    def test_retire_strategy_rule_not_found(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 0
        from v2.database.trading_db import retire_strategy_rule
        result = retire_strategy_rule(rule_id=999)
        assert result is False


class TestStrategyMemos:
    def test_insert_strategy_memo(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_strategy_memo
        result = insert_strategy_memo(
            session_date="2026-02-15",
            memo_type="reflection",
            content="Today we learned...",
            strategy_state_id=1,
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO strategy_memos" in sql

    def test_get_recent_strategy_memos(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "content": "test"}]
        from v2.database.trading_db import get_recent_strategy_memos
        result = get_recent_strategy_memos(n=5)
        assert len(result) == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "LIMIT" in sql
        assert "ORDER BY" in sql
