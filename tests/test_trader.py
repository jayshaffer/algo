"""Tests for trading/trader.py - trading session orchestrator."""

from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, MagicMock, call

import pytest

from trading.trader import (
    TradeResult,
    TradingSessionResult,
    run_trading_session,
)
from trading.agent import (
    TradingDecision,
    ThesisInvalidation,
    AgentResponse,
)
from tests.conftest import make_trading_decision, make_agent_response


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestTradeResult:
    def test_success(self):
        d = make_trading_decision()
        r = TradeResult(
            decision=d,
            executed=True,
            order_id="abc",
            filled_price=Decimal("150"),
            error=None,
        )
        assert r.executed is True
        assert r.error is None

    def test_failure(self):
        d = make_trading_decision()
        r = TradeResult(
            decision=d,
            executed=False,
            order_id=None,
            filled_price=None,
            error="Rejected",
        )
        assert r.executed is False


class TestTradingSessionResult:
    def test_creates_correctly(self):
        r = TradingSessionResult(
            timestamp=datetime.now(),
            account_snapshot_id=1,
            positions_synced=2,
            orders_synced=3,
            decisions_made=4,
            trades_executed=2,
            trades_failed=1,
            total_buy_value=Decimal("1000"),
            total_sell_value=Decimal("500"),
            errors=["some error"],
        )
        assert r.positions_synced == 2
        assert r.trades_executed == 2
        assert len(r.errors) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common account_info dict
ACCOUNT_INFO = {
    "account_id": "test",
    "status": "ACTIVE",
    "cash": Decimal("100000"),
    "portfolio_value": Decimal("150000"),
    "buying_power": Decimal("200000"),
    "long_market_value": Decimal("50000"),
    "short_market_value": Decimal("0"),
    "equity": Decimal("150000"),
    "daytrade_count": 0,
    "pattern_day_trader": False,
}


def _make_order_result(success=True, order_id="DRY_RUN", error=None):
    """Build a mock OrderResult."""
    from trading.executor import OrderResult
    return OrderResult(
        success=success,
        order_id=order_id,
        filled_qty=Decimal("10") if success else None,
        filled_avg_price=Decimal("150") if success else None,
        error=error,
    )


def _patch_all():
    """Return a dict of patch contexts for all external dependencies."""
    return {
        "sync_positions": patch("trading.trader.sync_positions_from_alpaca", return_value=2),
        "sync_orders": patch("trading.trader.sync_orders_from_alpaca", return_value=1),
        "get_account": patch("trading.trader.get_account_info", return_value=ACCOUNT_INFO),
        "snapshot": patch("trading.trader.take_account_snapshot", return_value=42),
        "build_ctx": patch("trading.trader.build_trading_context", return_value="mock context"),
        "get_decisions": patch("trading.trader.get_trading_decisions"),
        "validate": patch("trading.trader.validate_decision", return_value=(True, "OK")),
        "execute": patch("trading.trader.execute_market_order", return_value=_make_order_result()),
        "latest_price": patch("trading.trader.get_latest_price", return_value=Decimal("150")),
        "get_positions": patch("trading.trader.get_positions", return_value=[]),
        "insert_decision": patch("trading.trader.insert_decision", return_value=1),
        "insert_signals_batch": patch("trading.trader.insert_decision_signals_batch", return_value=0),
        "close_thesis": patch("trading.trader.close_thesis", return_value=True),
        "format_log": patch("trading.trader.format_decisions_for_logging", return_value={"market_summary": "ok"}),
        "calc_size": patch("trading.trader.calculate_position_size", return_value=10),
    }


# ---------------------------------------------------------------------------
# run_trading_session
# ---------------------------------------------------------------------------


class TestRunTradingSession:
    def test_successful_session_with_buy(self):
        """Full session with a buy decision that executes."""
        patches = _patch_all()
        with patches["sync_positions"] as m_sync_pos, \
             patches["sync_orders"] as m_sync_ord, \
             patches["get_account"] as m_acct, \
             patches["snapshot"] as m_snap, \
             patches["build_ctx"] as m_ctx, \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"] as m_exec, \
             patches["latest_price"] as m_price, \
             patches["get_positions"] as m_pos, \
             patches["insert_decision"] as m_insert, \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy_decision = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy_decision])

            result = run_trading_session(dry_run=True)

            assert result.trades_executed == 1
            assert result.trades_failed == 0
            assert result.decisions_made == 1
            assert result.total_buy_value > 0
            assert result.account_snapshot_id == 42
            assert result.positions_synced == 2
            assert result.orders_synced == 1

    def test_successful_session_with_sell(self):
        """Full session with a sell decision."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            sell_decision = make_trading_decision(action="sell", ticker="AAPL", quantity=5)
            m_dec.return_value = make_agent_response(decisions=[sell_decision])

            result = run_trading_session(dry_run=True)

            assert result.trades_executed == 1
            assert result.total_sell_value > 0
            assert result.total_buy_value == 0

    def test_hold_decision_not_executed(self):
        """Hold decisions should not trigger execution."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"] as m_exec, \
             patches["latest_price"] as m_price, \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            hold_decision = make_trading_decision(action="hold", ticker="AAPL", quantity=None)
            m_dec.return_value = make_agent_response(decisions=[hold_decision])

            result = run_trading_session(dry_run=True)

            # Hold should skip execution entirely
            assert result.trades_executed == 0
            assert result.trades_failed == 0
            m_exec.assert_not_called()

    def test_account_snapshot_failure_returns_early(self):
        """When account snapshot fails, session should return early."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"] as m_acct, \
             patches["snapshot"], \
             patches["build_ctx"] as m_ctx, \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            m_acct.side_effect = Exception("Cannot connect to Alpaca")

            result = run_trading_session(dry_run=True)

            assert result.decisions_made == 0
            assert result.trades_executed == 0
            assert result.account_snapshot_id == 0
            assert any("Account snapshot failed" in e for e in result.errors)
            # Should not have called LLM
            m_dec.assert_not_called()

    def test_llm_failure_returns_early(self):
        """When LLM call fails, session should return early."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            m_dec.side_effect = ValueError("Bad JSON from LLM")

            result = run_trading_session(dry_run=True)

            assert result.decisions_made == 0
            assert result.trades_executed == 0
            assert any("LLM decision failed" in e for e in result.errors)
            m_exec.assert_not_called()

    def test_validation_failure_increments_trades_failed(self):
        """Invalid decision should increment trades_failed."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy])
            m_val.return_value = (False, "Insufficient buying power")

            result = run_trading_session(dry_run=True)

            assert result.trades_failed == 1
            assert result.trades_executed == 0
            m_exec.assert_not_called()

    def test_execution_error_increments_trades_failed(self):
        """Order execution failure should increment trades_failed."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy])
            m_exec.return_value = _make_order_result(success=False, error="API error")

            result = run_trading_session(dry_run=True)

            assert result.trades_failed == 1
            assert result.trades_executed == 0

    def test_price_unavailable_increments_trades_failed(self):
        """When get_latest_price returns None, trade should fail."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"] as m_exec, \
             patches["latest_price"] as m_price, \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="BADTICKER", quantity=5)
            m_dec.return_value = make_agent_response(decisions=[buy])
            m_price.return_value = None

            result = run_trading_session(dry_run=True)

            assert result.trades_failed == 1
            assert result.trades_executed == 0
            m_val.assert_not_called()
            m_exec.assert_not_called()

    def test_thesis_invalidation_calls_close_thesis(self):
        """Thesis invalidations should call close_thesis."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"] as m_close, \
             patches["format_log"], \
             patches["calc_size"]:

            hold = make_trading_decision(action="hold", quantity=None)
            inv = ThesisInvalidation(thesis_id=99, reason="Market shifted")
            m_dec.return_value = make_agent_response(
                decisions=[hold],
                thesis_invalidations=[inv],
            )

            result = run_trading_session(dry_run=False)

            m_close.assert_any_call(
                thesis_id=99,
                status="invalidated",
                reason="Market shifted",
            )

    def test_thesis_invalidation_error_captured(self):
        """Error closing a thesis should be captured, not crash."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"] as m_close, \
             patches["format_log"], \
             patches["calc_size"]:

            hold = make_trading_decision(action="hold", quantity=None)
            inv = ThesisInvalidation(thesis_id=99, reason="test")
            m_dec.return_value = make_agent_response(
                decisions=[hold],
                thesis_invalidations=[inv],
            )
            m_close.side_effect = Exception("DB error")

            result = run_trading_session(dry_run=False)

            assert any("Failed to invalidate thesis" in e for e in result.errors)

    def test_sync_positions_error_does_not_abort(self):
        """Position sync failure should be logged but not abort session."""
        patches = _patch_all()
        with patches["sync_positions"] as m_sync, \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            m_sync.side_effect = Exception("Alpaca down")
            hold = make_trading_decision(action="hold", quantity=None)
            m_dec.return_value = make_agent_response(decisions=[hold])

            result = run_trading_session(dry_run=True)

            # Session should still complete
            assert result.positions_synced == 0
            assert result.decisions_made == 1
            assert any("Position sync failed" in e for e in result.errors)

    def test_sync_orders_error_does_not_abort(self):
        """Order sync failure should be logged but not abort session."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"] as m_sync, \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            m_sync.side_effect = Exception("Alpaca down")
            hold = make_trading_decision(action="hold", quantity=None)
            m_dec.return_value = make_agent_response(decisions=[hold])

            result = run_trading_session(dry_run=True)

            assert result.orders_synced == 0
            assert any("Order sync failed" in e for e in result.errors)

    def test_multiple_decisions_mixed_results(self):
        """Session with buy (success), sell (fail validation), hold (skip)."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            sell = make_trading_decision(action="sell", ticker="MSFT", quantity=20)
            hold = make_trading_decision(action="hold", ticker="GOOG", quantity=None)
            m_dec.return_value = make_agent_response(decisions=[buy, sell, hold])

            # buy passes, sell fails validation
            m_val.side_effect = [
                (True, "OK"),
                (False, "Insufficient shares"),
            ]

            result = run_trading_session(dry_run=True)

            assert result.decisions_made == 3
            assert result.trades_executed == 1
            assert result.trades_failed == 1  # sell failed validation

    def test_dry_run_flag_passed_to_executor(self):
        """dry_run should be passed to execute_market_order."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy])

            run_trading_session(dry_run=True)

            call_kwargs = m_exec.call_args.kwargs
            assert call_kwargs["dry_run"] is True

    def test_live_mode_passes_dry_run_false(self):
        """Live mode should pass dry_run=False."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy])

            run_trading_session(dry_run=False)

            call_kwargs = m_exec.call_args.kwargs
            assert call_kwargs["dry_run"] is False

    def test_model_passed_to_get_decisions(self):
        """Custom model should be forwarded to get_trading_decisions."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            hold = make_trading_decision(action="hold", quantity=None)
            m_dec.return_value = make_agent_response(decisions=[hold])

            run_trading_session(dry_run=True, model="llama3:8b")

            m_dec.assert_called_once()
            call_kwargs = m_dec.call_args.kwargs
            assert call_kwargs.get("model") == "llama3:8b"

    def test_decisions_logged_to_database(self):
        """All decisions should be inserted to the database."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"] as m_insert, \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            d1 = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            d2 = make_trading_decision(action="hold", ticker="MSFT", quantity=None)
            m_dec.return_value = make_agent_response(decisions=[d1, d2])

            run_trading_session(dry_run=True)

            assert m_insert.call_count == 2

    def test_insert_decision_error_captured(self):
        """Error logging a decision should be captured, not crash."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"] as m_insert, \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy])
            m_insert.side_effect = Exception("DB write error")

            result = run_trading_session(dry_run=True)

            assert any("Failed to log decision" in e for e in result.errors)

    def test_buying_power_decremented_after_buy(self):
        """After a buy, remaining buying power should decrease for subsequent trades."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy1 = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            buy2 = make_trading_decision(action="buy", ticker="MSFT", quantity=5)
            m_dec.return_value = make_agent_response(decisions=[buy1, buy2])

            run_trading_session(dry_run=True)

            # Both should have been validated (validate_decision called twice)
            assert m_val.call_count == 2

    def test_empty_decisions_list(self):
        """Session with no decisions from LLM."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            m_dec.return_value = make_agent_response(decisions=[])

            result = run_trading_session(dry_run=True)

            assert result.decisions_made == 0
            assert result.trades_executed == 0
            assert result.trades_failed == 0
            m_val.assert_not_called()
            m_exec.assert_not_called()

    def test_thesis_executed_on_live_buy(self):
        """Buy with thesis_id in live mode should close the thesis as executed."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"] as m_close, \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(
                action="buy", ticker="AAPL", quantity=10, thesis_id=55
            )
            m_dec.return_value = make_agent_response(decisions=[buy])

            run_trading_session(dry_run=False)

            m_close.assert_any_call(
                thesis_id=55,
                status="executed",
                reason="Trade executed: buy 10 shares",
            )

    def test_thesis_not_closed_on_dry_run(self):
        """In dry_run mode, thesis should NOT be marked as executed."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"] as m_close, \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(
                action="buy", ticker="AAPL", quantity=10, thesis_id=55
            )
            m_dec.return_value = make_agent_response(
                decisions=[buy],
                thesis_invalidations=[],
            )

            run_trading_session(dry_run=True)

            # close_thesis should NOT have been called with status="executed"
            for c in m_close.call_args_list:
                if c.kwargs.get("status") == "executed":
                    pytest.fail("Thesis should not be marked executed in dry_run mode")

    def test_result_timestamp_is_set(self):
        """Result should have a timestamp."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            hold = make_trading_decision(action="hold", quantity=None)
            m_dec.return_value = make_agent_response(decisions=[hold])

            result = run_trading_session(dry_run=True)

            assert isinstance(result.timestamp, datetime)

    def test_logs_decision_signal_links(self):
        """After logging a decision, signal_refs should be inserted into decision_signals."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"] as m_insert, \
             patches["insert_signals_batch"] as m_signals, \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            refs = [
                {"type": "news_signal", "id": 15},
                {"type": "thesis", "id": 42},
            ]
            buy = make_trading_decision(
                action="buy", ticker="AAPL", quantity=10, signal_refs=refs
            )
            m_dec.return_value = make_agent_response(decisions=[buy])
            m_insert.return_value = 77  # decision_id

            run_trading_session(dry_run=True)

            m_signals.assert_called_once_with([
                (77, "news_signal", 15),
                (77, "thesis", 42),
            ])

    def test_signal_links_not_called_when_no_refs(self):
        """When signal_refs is empty, insert_decision_signals_batch should not be called."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"] as m_signals, \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy])

            run_trading_session(dry_run=True)

            m_signals.assert_not_called()

    def test_signal_links_error_captured(self):
        """Error inserting signal links should be captured, not crash."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"] as m_signals, \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            refs = [{"type": "news_signal", "id": 15}]
            buy = make_trading_decision(
                action="buy", ticker="AAPL", quantity=10, signal_refs=refs
            )
            m_dec.return_value = make_agent_response(decisions=[buy])
            m_signals.side_effect = Exception("DB error on signals")

            result = run_trading_session(dry_run=True)

            assert any("Failed to log signal links" in e for e in result.errors)

    def test_signal_links_skipped_when_insert_decision_fails(self):
        """When insert_decision fails, signal links should not be attempted."""
        patches = _patch_all()
        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"] as m_insert, \
             patches["insert_signals_batch"] as m_signals, \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"]:

            refs = [{"type": "news_signal", "id": 15}]
            buy = make_trading_decision(
                action="buy", ticker="AAPL", quantity=10, signal_refs=refs
            )
            m_dec.return_value = make_agent_response(decisions=[buy])
            m_insert.side_effect = Exception("DB write error")

            result = run_trading_session(dry_run=True)

            # insert_decision failed, so signal links should NOT be attempted
            m_signals.assert_not_called()
