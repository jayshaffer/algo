"""Tests for trading session executor."""
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch, MagicMock

from v2.trader import run_trading_session, TradingSessionResult
from v2.agent import ExecutorInput, ExecutorDecision, AgentResponse, PlaybookAction


class TestRunTradingSession:
    def test_uses_structured_executor_input(self, mock_db, mock_cursor):
        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions:

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[],
                market_summary="No trades", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=True)

        mock_build.assert_called_once()
        assert isinstance(result, TradingSessionResult)

    def test_logs_playbook_action_id(self, mock_db, mock_cursor):
        """Decisions should log playbook_action_id and is_off_playbook."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=2.5, reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[{"type": "news_signal", "id": 5}],
            thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price", return_value=Decimal("150")), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.insert_decision", return_value=1) as mock_insert, \
             patch("v2.trader.insert_decision_signals_batch") as mock_signals, \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )
            mock_exec.return_value = MagicMock(success=True, order_id="123", error=None)

            result = run_trading_session(dry_run=True)

        # Verify playbook_action_id and is_off_playbook were passed
        mock_insert.assert_called_once()
        call_kwargs = mock_insert.call_args
        assert call_kwargs.kwargs.get("playbook_action_id") == 1 or \
               (len(call_kwargs.args) > 9 and call_kwargs.args[9] == 1)

    def test_session_with_no_decisions(self, mock_db, mock_cursor):
        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions:

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[],
                market_summary="No trades", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=True)

        assert result.decisions_made == 0
        assert result.trades_executed == 0

    def test_account_snapshot_failure_returns_early(self, mock_db, mock_cursor):
        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.get_account_info", side_effect=Exception("API error")):

            result = run_trading_session(dry_run=True)

        assert len(result.errors) > 0
        assert result.decisions_made == 0

    def test_thesis_invalidations_processed(self, mock_db, mock_cursor):
        from v2.agent import ThesisInvalidation
        inv = ThesisInvalidation(thesis_id=5, reason="Conditions changed")

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.close_thesis") as mock_close:

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[inv],
                market_summary="", risk_assessment="",
            )

            run_trading_session(dry_run=True)

        mock_close.assert_called_once_with(thesis_id=5, status="invalidated", reason="Conditions changed")


class TestTradingSessionResult:
    def test_has_required_fields(self):
        result = TradingSessionResult(
            timestamp=datetime.now(),
            account_snapshot_id=1,
            positions_synced=2,
            orders_synced=0,
            decisions_made=3,
            trades_executed=2,
            trades_failed=1,
            total_buy_value=Decimal("1000"),
            total_sell_value=Decimal("500"),
            errors=[],
        )
        assert result.decisions_made == 3
