"""Tests for trading session executor."""
from contextlib import ExitStack
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.agent import AgentResponse, ExecutorDecision, ExecutorInput
from v2.trader import TradingSessionResult, run_trading_session

# ---------------------------------------------------------------------------
# Helpers for branch-coverage tests
# ---------------------------------------------------------------------------

_DEFAULT_ACCOUNT = {
    "portfolio_value": Decimal("100000"),
    "cash": Decimal("50000"),
    "buying_power": Decimal("50000"),
}


def _make_decision(
    ticker="AAPL",
    action="buy",
    intent_type="invest_dollar",
    intent_magnitude=Decimal("500"),
    playbook_action_id=1,
    signal_refs=None,
    thesis_id=None,
    reasoning="test",
    is_off_playbook=False,
):
    return ExecutorDecision(
        playbook_action_id=playbook_action_id,
        ticker=ticker,
        action=action,
        intent_type=intent_type,
        intent_magnitude=intent_magnitude,
        reasoning=reasoning,
        confidence="high",
        is_off_playbook=is_off_playbook,
        signal_refs=signal_refs if signal_refs is not None else [],
        thesis_id=thesis_id,
    )


def _playbook_actions_for(decisions, invalidations=None):
    """Build the PlaybookActions the executor would have been shown.

    One per decision citing a playbook_action_id, matching on ticker/action/
    thesis_id so the A.6 id validator recognizes them as real. Decisions whose
    thesis_id is meant to come from an active thesis rather than the playbook
    still validate via the get_active_theses fallback.

    Invalidated thesis ids get a carrier action too: the executor can only
    invalidate a thesis it was shown, so a test invalidating thesis N implies
    N was in today's playbook.
    """
    from v2.agent import PlaybookAction

    actions = []
    for d in decisions or []:
        if d.playbook_action_id is None:
            continue
        actions.append(PlaybookAction(
            id=d.playbook_action_id,
            ticker=d.ticker,
            action=d.action,
            thesis_id=d.thesis_id,
            reasoning="from playbook",
            confidence="high",
            intent_type=d.intent_type,
            intent_magnitude=d.intent_magnitude,
            priority=len(actions) + 1,
        ))

    known = {a.thesis_id for a in actions}
    for inv in invalidations or []:
        if inv.thesis_id in known:
            continue
        known.add(inv.thesis_id)
        actions.append(PlaybookAction(
            id=9000 + len(actions),
            ticker="ZZZ",
            action="hold",
            thesis_id=inv.thesis_id,
            reasoning="carries the invalidated thesis into the executor's view",
            confidence="medium",
            intent_type=None,
            intent_magnitude=None,
            priority=len(actions) + 1,
        ))
    return actions


def _happy_path(stack, *, decisions=None, invalidations=None, overrides=None):
    """Enter all default trader.py happy-path patches on `stack`.

    Returns a dict of the mocks keyed by the dependency name so tests can
    make assertions on them. `overrides` is a dict mapping dependency name
    (unqualified — patched at v2.trader.<name>) to MagicMock.

    The default ExecutorInput carries a matching PlaybookAction for every
    decision that cites a playbook_action_id, mirroring production: the
    executor's decisions reference actions build_executor_input actually
    showed it. Without this the A.6 validator would (correctly) treat every
    decision in the suite as citing a hallucinated id and null it out. Tests
    that want to simulate a hallucination override build_executor_input.
    """
    overrides = overrides or {}
    defaults = {
        "sync_positions_from_alpaca": MagicMock(return_value=0),
        "sync_orders_from_alpaca": MagicMock(return_value=0),
        "is_market_open": MagicMock(return_value=True),
        "get_account_info": MagicMock(return_value=_DEFAULT_ACCOUNT),
        "take_account_snapshot": MagicMock(return_value=1),
        "build_executor_input": MagicMock(return_value=ExecutorInput(
            playbook_actions=_playbook_actions_for(decisions, invalidations),
            positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )),
        "get_trading_decisions": MagicMock(return_value=AgentResponse(
            decisions=decisions or [],
            thesis_invalidations=invalidations or [],
            market_summary="", risk_assessment="",
        )),
        "get_latest_price_with_reason": MagicMock(return_value=(Decimal("150"), None)),
        "get_latest_trade_price": MagicMock(return_value=Decimal("150")),
        "get_live_available_qty": MagicMock(return_value=Decimal("1000")),
        "execute_market_order": MagicMock(return_value=MagicMock(
            success=True, order_id="ord-1", error=None,
            filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
        )),
        "wait_for_fill": MagicMock(return_value=MagicMock(
            success=True, order_id="ord-1", error=None,
            filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
        )),
        "get_positions": MagicMock(return_value=[]),
        "check_decision_exists": MagicMock(return_value=None),
        "insert_decision": MagicMock(return_value=1),
        "insert_decision_signals_batch": MagicMock(),
        "validate_signal_refs": MagicMock(side_effect=lambda refs: refs),
        "close_thesis": MagicMock(),
        "get_active_theses": MagicMock(return_value=[]),
        "format_decisions_for_logging": MagicMock(return_value={}),
        "check_sector_concentration": MagicMock(return_value=[]),
        "update_playbook_action_status": MagicMock(),
        "get_pending_playbook_action_for_ticker": MagicMock(return_value=None),
        "StockHistoricalDataClient": MagicMock(return_value=MagicMock()),
    }
    defaults.update(overrides)
    mocks = {}
    for name, mock in defaults.items():
        mocks[name] = stack.enter_context(patch(f"v2.trader.{name}", mock))
    return mocks


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


class TestDecisionRejectionTelemetry:
    def test_price_rejection_emits_decision_rejected_event(self):
        from v2.trader import _ExecutionTotals, _prepare_decision

        decision = _make_decision(ticker="AAPL")
        totals = _ExecutionTotals()

        with patch("v2.trader.get_latest_price_with_reason",
                   return_value=(None, "wide quote spread")), \
             patch("v2.trader.record_event") as mock_record:
            price = _prepare_decision(
                decision=decision,
                positions={},
                data_client=None,
                dry_run=False,
                portfolio_value=Decimal("100000"),
                buying_power=Decimal("50000"),
                totals=totals,
                errors=[],
                session_date=date(2026, 5, 15),
                session_id=42,
            )

        assert price is None
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["event_type"] == "decision_rejected"
        payload = mock_record.call_args.kwargs["payload"]
        assert payload["reason_code"] == "pricing"
        assert payload["ticker"] == "AAPL"

    def test_duplicate_decision_emits_decision_rejected_event(self):
        from v2.trader import _ExecutionTotals, _prepare_decision

        decision = _make_decision(ticker="AAPL")
        totals = _ExecutionTotals()

        with patch("v2.trader.get_latest_price_with_reason",
                   return_value=(Decimal("100"), None)), \
             patch("v2.trader.check_churn_gate", return_value=None), \
             patch("v2.trader.check_sector_cap_for_buy", return_value=None), \
             patch("v2.trader.check_decision_exists", return_value=123), \
             patch("v2.trader.record_event") as mock_record:
            price = _prepare_decision(
                decision=decision,
                positions={},
                data_client=None,
                dry_run=False,
                portfolio_value=Decimal("100000"),
                buying_power=Decimal("50000"),
                totals=totals,
                errors=[],
                session_date=date(2026, 5, 15),
                position_values={},
                session_id=42,
            )

        assert price is None
        payload = mock_record.call_args.kwargs["payload"]
        assert payload["reason_code"] == "duplicate_decision"
        assert payload["existing_decision_id"] == 123

    def test_logs_playbook_action_id(self, mock_db, mock_cursor):
        """Decisions should log playbook_action_id and is_off_playbook."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("375"),
            reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[{"type": "news_signal", "id": 5}],
            thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("150"), None)), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.insert_decision", return_value=1) as mock_insert, \
             patch("v2.trader.insert_decision_signals_batch") as mock_signals, \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            # The playbook must contain the action the decision cites, or the
            # A.6 validator correctly treats the id as hallucinated and nulls it.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([decision]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )
            mock_exec.return_value = MagicMock(
                success=True, order_id="123", error=None,
                filled_qty=Decimal("2.5"), filled_avg_price=Decimal("150"),
            )

            result = run_trading_session(dry_run=True)

        # Verify playbook_action_id and is_off_playbook were passed
        mock_insert.assert_called_once()
        call_kwargs = mock_insert.call_args
        assert call_kwargs.kwargs.get("playbook_action_id") == 1 or \
               (len(call_kwargs.args) > 9 and call_kwargs.args[9] == 1)
        # Verify filled_qty is used (not requested quantity)
        assert call_kwargs.kwargs.get("quantity") == Decimal("2.5")

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

    def test_missing_price_decision_logged_as_invalid(self, mock_db, mock_cursor):
        """When get_latest_price returns None for a decision ticker, the decision
        should still be logged as action='invalid' (with NULL price) so the
        model's intent is recorded — not silently dropped."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="ZZZZ", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("500"),
            reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(None, "quote stale: 90s (max 60s)")), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.insert_decision", return_value=1) as mock_insert, \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("10000"), "cash": Decimal("5000"), "buying_power": Decimal("5000")}
            # The playbook must contain the action the decision cites, or the
            # A.6 validator correctly treats the id as hallucinated and nulls it.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([decision]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=True)

        mock_exec.assert_not_called()
        mock_insert.assert_called_once()
        kwargs = mock_insert.call_args.kwargs
        assert kwargs.get("action") == "invalid"
        assert kwargs.get("price") is None
        assert kwargs.get("order_id") is None
        reasoning = kwargs.get("reasoning") or ""
        # ALGO-14: rejection string carries the structured reason from
        # get_latest_price_with_reason (not the legacy "no price available").
        assert "[REJECTED:" in reasoning
        assert "quote stale" in reasoning  # matches the mock return value above
        assert result.trades_failed == 1

    def test_zero_available_sell_logged_as_invalid(self, mock_db, mock_cursor):
        """Pre-submit Alpaca live-availability check can reject a sell when DB
        says shares exist but Alpaca reports 0 available. That skip must be
        recorded as action='invalid' too, not left as a phantom 'sell'."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="sell",
            intent_type="exit_full", intent_magnitude=None,
            reasoning="Take profit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=1), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("100"), None)), \
             patch("v2.trader.get_live_available_qty", return_value=Decimal("0")), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.insert_decision", return_value=1) as mock_insert, \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]):

            mock_acct.return_value = {"portfolio_value": Decimal("10000"), "cash": Decimal("5000"), "buying_power": Decimal("5000")}
            # The playbook must contain the action the decision cites, or the
            # A.6 validator correctly treats the id as hallucinated and nulls it.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([decision]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        mock_exec.assert_not_called()
        mock_insert.assert_called_once()
        kwargs = mock_insert.call_args.kwargs
        assert kwargs.get("action") == "invalid"
        assert kwargs.get("order_id") is None
        reasoning = kwargs.get("reasoning") or ""
        assert "0 available" in reasoning or "available" in reasoning.lower()
        assert result.trades_failed == 1

    def test_stale_exit_full_sells_actual_holding_not_playbook_qty(self, mock_db, mock_cursor):
        """The canonical AMZN incident: playbook wants to exit AMZN but the
        LLM-authored max_quantity drifted from the held shares. With intent-
        based sizing, the trader resolves `exit_full` against live positions
        and sells exactly 1.0 — regardless of what arithmetic the strategist
        baked into the playbook.

        Regression test for 2026-04-17 oversell incident (playbook action #14).
        """
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AMZN", action="sell",
            intent_type="exit_full", intent_magnitude=None,
            reasoning="Target hit — exit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=1), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("248.66"), None)), \
             patch("v2.trader.get_live_available_qty", return_value=Decimal("1.0")), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.wait_for_fill") as mock_wait, \
             patch("v2.trader.insert_decision", return_value=1) as mock_insert, \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[
                 {"ticker": "AMZN", "shares": Decimal("1.0")}
             ]):

            mock_acct.return_value = {
                "portfolio_value": Decimal("100000"),
                "cash": Decimal("50000"),
                "buying_power": Decimal("50000"),
            }
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="", risk_assessment="",
            )
            mock_exec.return_value = MagicMock(
                success=True, order_id="test-order",
                filled_qty=Decimal("1.0"), filled_avg_price=Decimal("248.66"),
                error=None,
            )
            mock_wait.return_value = mock_exec.return_value

            result = run_trading_session(dry_run=False)

        # Resolver turned exit_full into exactly 1.0 (the held shares)
        mock_exec.assert_called_once()
        call_kwargs = mock_exec.call_args.kwargs
        assert call_kwargs["qty"] == Decimal("1.0")
        assert call_kwargs["side"] == "sell"
        assert result.trades_executed == 1
        assert result.trades_failed == 0
        # Logged decision records the resolved quantity
        assert mock_insert.call_args.kwargs["quantity"] == Decimal("1.0")

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
            # A.6: the executor may only invalidate a thesis it was shown, so
            # thesis 5 must be present in today's playbook for this to apply.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([], [inv]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[inv],
                market_summary="", risk_assessment="",
            )

            run_trading_session(dry_run=True)

        mock_close.assert_called_once_with(thesis_id=5, status="invalidated", reason="Conditions changed")

    def test_positions_dict_refreshed_after_each_fill(self, mock_db, mock_cursor):
        """A later decision for a ticker must see the post-fill share count,
        not the pre-loop snapshot — the thesis-lifecycle close depends on
        `held_before - filled_qty == 0`.

        Exercised with buy-then-sell: A.2 now rejects two same-action
        decisions for one ticker as an intra-batch duplicate (the DB's
        idx_decisions_dedup could never record both anyway), so buy→sell is
        the remaining way two fills touch the same ticker in one batch.
        """
        from datetime import date
        from decimal import Decimal
        from unittest.mock import MagicMock, patch

        from v2.agent import AgentResponse
        from v2.trader import ExecutorDecision, _execute_decisions

        decisions = [
            ExecutorDecision(
                ticker="AAPL", action="buy",
                reasoning="add 10 shares", thesis_id=42,
                intent_type="invest_dollar", intent_magnitude=Decimal("1500"),
                playbook_action_id=None, is_off_playbook=False,
                confidence="high",
            ),
            ExecutorDecision(
                ticker="AAPL", action="sell",
                reasoning="exit everything", thesis_id=42,
                intent_type="exit_full", intent_magnitude=None,
                playbook_action_id=None, is_off_playbook=False,
                confidence="high",
            ),
        ]
        response = AgentResponse(
            decisions=decisions, market_summary="", risk_assessment="",
            thesis_invalidations=[],
        )
        # 10 shares @ $150 = $1,500 of a $50,000 book — well under the
        # max-position cap, so the buy resolves rather than clamping to 0.
        positions = {"AAPL": Decimal("10")}
        account_info = {"buying_power": Decimal("10000"), "portfolio_value": Decimal("50000")}
        errors: list[str] = []

        closed_thesis_ids: list[int] = []

        def fake_close_thesis(*, thesis_id, status, reason):
            closed_thesis_ids.append(thesis_id)

        def fake_order(*, ticker, side, qty, dry_run, simulated_price, client_order_id):
            return MagicMock(success=True, order_id="O1", error=None,
                             filled_qty=qty, filled_avg_price=Decimal("150"))

        with patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("150"), None)), \
             patch("v2.trader._precheck_sell_against_alpaca", return_value=True), \
             patch("v2.trader.execute_market_order", side_effect=fake_order) as mock_order, \
             patch("v2.trader.wait_for_fill", side_effect=lambda oid: MagicMock(
                 success=True, order_id=oid, error=None,
                 filled_qty=mock_order.call_args.kwargs["qty"],
                 filled_avg_price=Decimal("150"))), \
             patch("v2.trader._refresh_buying_power",
                   return_value=(Decimal("10000"), Decimal("50000"), None)), \
             patch("v2.trader.close_thesis", side_effect=fake_close_thesis), \
             patch("v2.trader.check_sector_concentration", return_value=[]), \
             patch("v2.trader.check_decision_exists", return_value=None):

            _execute_decisions(
                response, positions, account_info, MagicMock(),
                False, errors, date(2026, 5, 3),
            )

        # $1,500 / $150 = 10 shares bought → 20 held. The exit_full must size
        # against 20, not the stale pre-loop 10.
        sell_qty = mock_order.call_args_list[1].kwargs["qty"]
        assert sell_qty == Decimal("20"), (
            f"exit_full should size against the post-buy 20 shares, got {sell_qty}. "
            "positions dict was not refreshed between decisions."
        )
        assert 42 in closed_thesis_ids, (
            "Thesis 42 should close once the full position is sold."
        )
        assert positions["AAPL"] == Decimal("0"), (
            f"positions['AAPL'] should be 0 after the exit, got {positions['AAPL']}"
        )


class TestDailyLossCircuitBreaker:
    """Kill switch: a session that opens (or goes) more than the daily loss
    limit under the previous close must stop trading instead of letting the
    executor keep bleeding capital."""

    def test_session_halts_before_decisions_when_limit_breached(self, mock_db, mock_cursor):
        account = {
            **_DEFAULT_ACCOUNT,
            "equity": Decimal("95000"),
            "last_equity": Decimal("100000"),  # -5% vs 3% default limit
        }
        with ExitStack() as stack:
            mocks = _happy_path(stack, overrides={
                "get_account_info": MagicMock(return_value=account),
            })
            result = run_trading_session(dry_run=True)

        assert any("daily loss" in e.lower() for e in result.errors)
        mocks["get_trading_decisions"].assert_not_called()
        assert result.decisions_made == 0

    def test_session_proceeds_when_within_limit(self, mock_db, mock_cursor):
        account = {
            **_DEFAULT_ACCOUNT,
            "equity": Decimal("99500"),
            "last_equity": Decimal("100000"),  # -0.5%
        }
        with ExitStack() as stack:
            mocks = _happy_path(stack, overrides={
                "get_account_info": MagicMock(return_value=account),
            })
            result = run_trading_session(dry_run=True)

        mocks["get_trading_decisions"].assert_called_once()
        assert not any("daily loss" in e.lower() for e in result.errors)

    def test_mid_loop_breach_stops_remaining_trades(self):
        """After a fill, the refreshed account state is re-checked; a breach
        mid-loop halts the remaining decisions."""
        from v2.trader import _execute_decisions

        decisions = [
            _make_decision(ticker="AAPL", playbook_action_id=None),
            _make_decision(ticker="MSFT", playbook_action_id=None),
        ]
        response = AgentResponse(
            decisions=decisions, thesis_invalidations=[],
            market_summary="", risk_assessment="",
        )
        account_info = {
            "buying_power": Decimal("50000"),
            "portfolio_value": Decimal("100000"),
        }
        breached = {
            "buying_power": Decimal("45000"),
            "portfolio_value": Decimal("94000"),
            "equity": Decimal("94000"),
            "last_equity": Decimal("100000"),  # -6% after the first fill
        }
        errors: list[str] = []

        order_result = MagicMock(
            success=True, order_id="O1", error=None,
            filled_qty=Decimal("3"), filled_avg_price=Decimal("150"),
        )
        with patch("v2.trader.get_latest_price_with_reason",
                   return_value=(Decimal("150"), None)), \
             patch("v2.trader.check_churn_gate", return_value=None), \
             patch("v2.trader.check_decision_exists", return_value=None), \
             patch("v2.trader.execute_market_order", return_value=order_result) as mock_exec, \
             patch("v2.trader.wait_for_fill", return_value=order_result), \
             patch("v2.trader.get_account_info", return_value=breached):
            _execute_decisions(
                response, {}, account_info, MagicMock(),
                False, errors, date(2026, 6, 10),
            )

        assert mock_exec.call_count == 1, (
            "Second decision must not execute after the daily loss limit "
            "was breached mid-session"
        )
        assert any("daily loss" in e.lower() for e in errors)


class TestThesisClosureUsesFilledQty:
    """Thesis lifecycle must key off what actually filled, not what was
    submitted. A partially-filled exit leaves real shares held; closing the
    thesis would orphan that position."""

    def _run_single_sell(self, filled_qty):
        from v2.trader import _execute_decisions

        decision = ExecutorDecision(
            ticker="AAPL", action="sell",
            reasoning="exit", thesis_id=42,
            intent_type="exit_full", intent_magnitude=None,
            playbook_action_id=None, is_off_playbook=False,
            confidence="high",
        )
        response = AgentResponse(
            decisions=[decision], thesis_invalidations=[],
            market_summary="", risk_assessment="",
        )
        positions = {"AAPL": Decimal("100")}
        account_info = {
            "buying_power": Decimal("10000"),
            "portfolio_value": Decimal("50000"),
        }
        errors: list[str] = []
        closed: list[int] = []

        order_result = MagicMock(
            success=True, order_id="O1", error=None,
            filled_qty=filled_qty, filled_avg_price=Decimal("150"),
        )
        with patch("v2.trader.get_latest_price_with_reason",
                   return_value=(Decimal("150"), None)), \
             patch("v2.trader.check_churn_gate", return_value=None), \
             patch("v2.trader.check_decision_exists", return_value=None), \
             patch("v2.trader._precheck_sell_against_alpaca", return_value=True), \
             patch("v2.trader.execute_market_order", return_value=order_result), \
             patch("v2.trader.wait_for_fill", return_value=order_result), \
             patch("v2.trader._refresh_buying_power",
                   return_value=(Decimal("10000"), Decimal("50000"), None)), \
             patch("v2.trader.close_thesis",
                   side_effect=lambda *, thesis_id, status, reason: closed.append(thesis_id)):
            _execute_decisions(
                response, positions, account_info, MagicMock(),
                False, errors, date(2026, 6, 10),
            )
        return closed

    def test_partial_fill_keeps_thesis_open(self):
        # exit_full submits 100 shares but only 40 fill before timeout/cancel.
        closed = self._run_single_sell(filled_qty=Decimal("40"))
        assert closed == [], (
            "Thesis must stay open when only 40 of 100 shares actually "
            "filled — 60 shares are still held"
        )

    def test_full_fill_closes_thesis(self):
        closed = self._run_single_sell(filled_qty=Decimal("100"))
        assert closed == [42]


class TestSessionDateConsistency:
    """T2.10: session_date must be captured once at the top of
    `run_trading_session` and threaded through every callsite that
    needed today's date — client_order_id signing, dedup checks, and
    decision-row inserts. A session running across midnight ET would
    otherwise produce decision rows tagged with multiple dates.
    """

    def test_session_date_threaded_to_log_decisions(self, mock_db, mock_cursor):
        from datetime import date as date_cls
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=1500.0,
            quantity=Decimal("10"),
            reasoning="t", confidence="high", is_off_playbook=False,
            signal_refs=[],
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            # Anchor the session-date wall clock to a fixed ET datetime.
            stack.enter_context(patch(
                "v2.trader.datetime",
                MagicMock(
                    now=MagicMock(return_value=__import__(
                        "datetime"
                    ).datetime(2026, 5, 3, 23, 30)),
                ),
            ))
            run_trading_session(dry_run=True)
        # Verify insert_decision was called with the captured session_date.
        # In dry_run, execute_market_order returns DRY_RUN order id; the
        # dedup check / log_decisions still receive the date arg.
        called_dates = {
            (c.kwargs.get("decision_date") or c.args[0])
            for c in mocks["insert_decision"].call_args_list
        }
        # At minimum: the session_date is a date instance and there is
        # exactly one distinct value across all logged rows.
        assert all(isinstance(d, date_cls) for d in called_dates)
        assert len(called_dates) == 1


class TestSessionIdThreading:
    """Task 7: run_trading_session must thread its session_id parameter
    through _log_decisions and _insert_decision_with_retry down to the
    insert_decision DB call, so every decision row is owned by the
    current session.
    """

    def test_session_id_threaded_to_insert_decision(self, mock_db, mock_cursor):
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=1500.0,
            quantity=Decimal("10"),
            reasoning="t", confidence="high", is_off_playbook=False,
            signal_refs=[],
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            run_trading_session(dry_run=True, session_id=42)
        # Every insert_decision call must carry the session_id kwarg.
        called_session_ids = [
            c.kwargs.get("session_id")
            for c in mocks["insert_decision"].call_args_list
        ]
        assert called_session_ids, "insert_decision was not called"
        assert all(sid == 42 for sid in called_session_ids)

    def test_session_id_defaults_to_none(self, mock_db, mock_cursor):
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=1500.0,
            quantity=Decimal("10"),
            reasoning="t", confidence="high", is_off_playbook=False,
            signal_refs=[],
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            run_trading_session(dry_run=True)
        called_session_ids = [
            c.kwargs.get("session_id")
            for c in mocks["insert_decision"].call_args_list
        ]
        assert called_session_ids, "insert_decision was not called"
        assert all(sid is None for sid in called_session_ids)

    def test_session_id_included_in_orphan_payload(self, mock_db, mock_cursor):
        """JSONL orphan fallback must capture session_id from the payload.

        Regression for: session_id used to ride as a separate kwarg through
        _insert_decision_with_retry, so when the retry exhausted and the
        orphan-JSONL path serialized `payload` alone, the owning session was
        lost from the operator-recovery record. Folding session_id into
        `payload` itself fixes this end-to-end.
        """
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=1500.0,
            quantity=Decimal("10"),
            reasoning="t", confidence="high", is_off_playbook=False,
            signal_refs=[],
        )
        captured_payloads: list[dict] = []

        def capture_orphan(**kwargs):
            captured_payloads.append(kwargs["payload"])

        with ExitStack() as stack:
            _happy_path(
                stack,
                decisions=[decision],
                overrides={
                    # Force retries to exhaust so the orphan path runs.
                    "insert_decision": MagicMock(side_effect=RuntimeError("db down")),
                },
            )
            stack.enter_context(patch("v2.trader.time.sleep", lambda s: None))
            stack.enter_context(
                patch("v2.trader._persist_orphan_decision", side_effect=capture_orphan)
            )
            run_trading_session(dry_run=True, session_id=42)

        assert captured_payloads, (
            "_persist_orphan_decision was not called — orphan path didn't trigger"
        )
        # Every orphan payload must carry the owning session_id so an operator
        # reconciling the JSONL record can correlate it back to `sessions`.
        for payload in captured_payloads:
            assert payload.get("session_id") == 42, (
                f"orphan payload missing session_id: {payload!r}"
            )


class TestMarketHoursGate:
    def test_skips_trading_when_market_closed(self, mock_db, mock_cursor):
        """When market is closed and not dry_run, return a neutral no-op result."""
        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot") as mock_snap:

            result = run_trading_session(dry_run=False)

        mock_acct.assert_not_called()
        assert result.trades_executed == 0
        assert result.errors == []
        assert result.market_closed is True

    def test_allows_trading_when_market_open(self, mock_db, mock_cursor):
        """When market is open, should proceed normally."""
        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
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

            result = run_trading_session(dry_run=False)

        mock_acct.assert_called_once()

    def test_dry_run_bypasses_market_hours_check(self, mock_db, mock_cursor):
        """Dry run should work even when market is closed."""
        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False), \
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

        mock_acct.assert_called_once()


class TestFillConfirmation:
    def test_uses_fill_price_for_logging(self, mock_db, mock_cursor):
        """After fill confirmation, logged price should be fill price, not quote."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("375"),
            reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        submit_result = MagicMock(success=True, order_id="order-123", filled_avg_price=None, error=None)
        fill_result = MagicMock(success=True, order_id="order-123", filled_qty=Decimal("2.5"), filled_avg_price=Decimal("151.50"), error=None)

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("150.00"), None)), \
             patch("v2.trader.execute_market_order", return_value=submit_result), \
             patch("v2.trader.wait_for_fill", return_value=fill_result), \
             patch("v2.trader.insert_decision", return_value=1) as mock_insert, \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            # The playbook must contain the action the decision cites, or the
            # A.6 validator correctly treats the id as hallucinated and nulls it.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([decision]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        # Logged price should be fill price (151.50), not quote price (150.00)
        mock_insert.assert_called_once()
        call_kwargs = mock_insert.call_args
        logged_price = call_kwargs.kwargs.get("price") or call_kwargs[1].get("price")
        assert logged_price == Decimal("151.50")

    def test_dry_run_skips_wait_for_fill(self, mock_db, mock_cursor):
        """Dry run should NOT call wait_for_fill."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("375"),
            reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("150.00"), None)), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.wait_for_fill") as mock_wait, \
             patch("v2.trader.insert_decision", return_value=1), \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            # The playbook must contain the action the decision cites, or the
            # A.6 validator correctly treats the id as hallucinated and nulls it.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([decision]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )
            mock_exec.return_value = MagicMock(success=True, order_id="DRY_RUN", filled_qty=Decimal("2.5"), filled_avg_price=None, error=None)

            result = run_trading_session(dry_run=True)

        mock_wait.assert_not_called()

    def test_fill_timeout_marks_trade_failed(self, mock_db, mock_cursor):
        """If fill confirmation times out, trade should be marked failed."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("375"),
            reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        submit_result = MagicMock(success=True, order_id="order-123", filled_avg_price=None, error=None)
        fill_result = MagicMock(success=False, order_id="order-123", filled_qty=None, filled_avg_price=None, error="Timeout")

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("150.00"), None)), \
             patch("v2.trader.execute_market_order", return_value=submit_result), \
             patch("v2.trader.wait_for_fill", return_value=fill_result), \
             patch("v2.trader.insert_decision", return_value=1), \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            # The playbook must contain the action the decision cites, or the
            # A.6 validator correctly treats the id as hallucinated and nulls it.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([decision]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        assert result.trades_failed == 1


class TestBuyingPowerRefresh:
    def test_refreshes_buying_power_after_fill(self, mock_db, mock_cursor):
        """After a fill, buying power should be re-fetched from Alpaca."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("375"),
            reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        submit_result = MagicMock(success=True, order_id="order-123", filled_avg_price=None, error=None)
        fill_result = MagicMock(success=True, order_id="order-123", filled_qty=Decimal("2.5"), filled_avg_price=Decimal("150.00"), error=None)

        # First call returns initial account info, second returns refreshed
        account_calls = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")},
            {"portfolio_value": Decimal("99625"), "cash": Decimal("49625"), "buying_power": Decimal("49625")},
        ]
        call_count = [0]
        def mock_get_account():
            idx = min(call_count[0], len(account_calls) - 1)
            call_count[0] += 1
            return account_calls[idx]

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info", side_effect=mock_get_account), \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price_with_reason", return_value=(Decimal("150.00"), None)), \
             patch("v2.trader.execute_market_order", return_value=submit_result), \
             patch("v2.trader.wait_for_fill", return_value=fill_result), \
             patch("v2.trader.validate_signal_refs", return_value=[]), \
             patch("v2.trader.insert_decision", return_value=1), \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            # The playbook must contain the action the decision cites, or the
            # A.6 validator correctly treats the id as hallucinated and nulls it.
            mock_build.return_value = ExecutorInput(
                playbook_actions=_playbook_actions_for([decision]),
                positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        # get_account_info should be called twice: once at snapshot, once after fill
        assert call_count[0] >= 2

    def test_each_decision_logged_with_account_state_at_trade_time(self, mock_db, mock_cursor):
        """Each decision row's account_equity/buying_power must reflect the live
        account state at the moment that trade executed, not the pre-session
        snapshot. Without this, every row in a multi-trade session shares the
        same pre-session figures and corrupts the historical signal that
        strategy reflection learns from.
        """
        decision_a = _make_decision(
            ticker="AAPL", playbook_action_id=1,
            intent_type="invest_dollar", intent_magnitude=Decimal("150"),
        )
        decision_b = _make_decision(
            ticker="MSFT", playbook_action_id=2,
            intent_type="invest_dollar", intent_magnitude=Decimal("150"),
        )

        pre_session = {
            "portfolio_value": Decimal("100000"), "cash": Decimal("50000"),
            "buying_power": Decimal("50000"),
        }
        after_first_fill = {
            "portfolio_value": Decimal("99850"), "cash": Decimal("49850"),
            "buying_power": Decimal("49850"),
        }
        after_second_fill = {
            "portfolio_value": Decimal("99700"), "cash": Decimal("49700"),
            "buying_power": Decimal("49700"),
        }
        account_returns = iter([
            pre_session, after_first_fill, after_second_fill, after_second_fill,
        ])

        with ExitStack() as stack:
            mocks = _happy_path(
                stack,
                decisions=[decision_a, decision_b],
                overrides={
                    "get_account_info": MagicMock(
                        side_effect=lambda: next(account_returns),
                    ),
                },
            )
            run_trading_session(dry_run=False)

        insert_calls = mocks["insert_decision"].call_args_list
        assert len(insert_calls) == 2, "expected one insert_decision per decision"

        # First decision: state at trade time == pre-session snapshot
        assert insert_calls[0].kwargs["buying_power"] == pre_session["buying_power"]
        assert insert_calls[0].kwargs["account_equity"] == pre_session["portfolio_value"]

        # Second decision: state at trade time == AFTER first fill, NOT pre-session
        assert insert_calls[1].kwargs["buying_power"] == after_first_fill["buying_power"]
        assert insert_calls[1].kwargs["account_equity"] == after_first_fill["portfolio_value"]


class TestRefreshBuyingPowerLocalEstimate:
    """T1.1: when get_account_info raises, the local-estimate fallback must
    debit buys AND credit sells. Pre-fix the sell branch was a no-op, leaving
    later size-checks reading a stale buying_power that ignored cash freed by
    earlier sells in the same loop."""

    def _decision(self, action: str) -> ExecutorDecision:
        return ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action=action,
            intent_type="invest_dollar", intent_magnitude=Decimal("500"),
            reasoning="x", confidence="medium",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

    def test_sell_credits_buying_power_real_branch(self):
        from v2.trader import _refresh_buying_power
        bp_before = Decimal("1000")
        pv_before = Decimal("10000")
        trade_value = Decimal("250")

        with patch("v2.trader.get_account_info", side_effect=Exception("api down")):
            bp_after, pv_after, refreshed = _refresh_buying_power(
                self._decision("sell"), bp_before, pv_before, trade_value, dry_run=False,
            )
        assert bp_after == bp_before + trade_value
        assert pv_after == pv_before
        # No live account state available on the fallback path.
        assert refreshed is None

    def test_sell_credits_buying_power_dry_run(self):
        from v2.trader import _refresh_buying_power
        bp_before = Decimal("1000")
        pv_before = Decimal("10000")
        trade_value = Decimal("250")

        bp_after, pv_after, refreshed = _refresh_buying_power(
            self._decision("sell"), bp_before, pv_before, trade_value, dry_run=True,
        )
        assert bp_after == bp_before + trade_value
        assert pv_after == pv_before
        assert refreshed is None

    def test_buy_still_debits_buying_power(self):
        from v2.trader import _refresh_buying_power
        bp_before = Decimal("1000")
        trade_value = Decimal("250")

        with patch("v2.trader.get_account_info", side_effect=Exception("api down")):
            bp_after, _, _ = _refresh_buying_power(
                self._decision("buy"), bp_before, Decimal("10000"), trade_value, dry_run=False,
            )
        assert bp_after == bp_before - trade_value


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


# ---------------------------------------------------------------------------
# Branch-coverage completion tests
# ---------------------------------------------------------------------------


class TestSyncFailures:
    def test_position_sync_failure_records_error(self, mock_db, mock_cursor):
        with ExitStack() as stack:
            _happy_path(stack, overrides={
                "sync_positions_from_alpaca": MagicMock(side_effect=RuntimeError("pos sync down")),
            })
            result = run_trading_session(dry_run=True)
        assert any("Position sync failed" in e for e in result.errors)
        assert result.positions_synced == 0

    def test_order_sync_failure_records_error(self, mock_db, mock_cursor):
        with ExitStack() as stack:
            _happy_path(stack, overrides={
                "sync_orders_from_alpaca": MagicMock(side_effect=RuntimeError("ord sync down")),
            })
            result = run_trading_session(dry_run=True)
        assert any("Order sync failed" in e for e in result.errors)
        assert result.orders_synced == 0


class TestContextBuild:
    def test_build_executor_input_fallback_on_exception(self, mock_db, mock_cursor):
        """When build_executor_input raises, a fallback ExecutorInput is used."""
        with ExitStack() as stack:
            _happy_path(stack, overrides={
                "build_executor_input": MagicMock(side_effect=RuntimeError("context dead")),
            })
            result = run_trading_session(dry_run=True)
        assert any("Context build failed" in e for e in result.errors)
        # Session proceeds despite context-build failure
        assert isinstance(result, TradingSessionResult)

    def test_sector_warnings_append_to_existing_risk_notes(self, mock_db, mock_cursor):
        existing_input = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="existing note",
        )
        with ExitStack() as stack:
            _happy_path(stack, overrides={
                "build_executor_input": MagicMock(return_value=existing_input),
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("10")}]),
                "check_sector_concentration": MagicMock(return_value=["tech heavy"]),
            })
            run_trading_session(dry_run=True)
        assert "existing note" in existing_input.risk_notes
        assert "tech heavy" in existing_input.risk_notes

    def test_sector_warnings_set_empty_risk_notes(self, mock_db, mock_cursor):
        empty_input = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )
        with ExitStack() as stack:
            _happy_path(stack, overrides={
                "build_executor_input": MagicMock(return_value=empty_input),
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("10")}]),
                "check_sector_concentration": MagicMock(return_value=["tech heavy"]),
            })
            run_trading_session(dry_run=True)
        assert "tech heavy" in empty_input.risk_notes

    def test_position_loop_skips_ticker_when_price_is_none(self, mock_db, mock_cursor):
        """The loop that builds position_values for sector check skips tickers
        whose price lookup returns None (covers the 'if price:' False branch).
        Position valuation uses the trade-price helper (not the quote-based
        one) so wide IEX spreads near the close don't silently drop positions
        from the sector concentration check.
        """
        with ExitStack() as stack:
            mocks = _happy_path(stack, overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("10")}]),
                "get_latest_trade_price": MagicMock(return_value=None),
                "check_sector_concentration": MagicMock(return_value=[]),
            })
            run_trading_session(dry_run=True)
        # check_sector_concentration called with empty position_values dict
        assert mocks["check_sector_concentration"].called
        args, _ = mocks["check_sector_concentration"].call_args
        assert args[0] == {}


class TestLLMDecisionFailure:
    def test_llm_failure_returns_early(self, mock_db, mock_cursor):
        with ExitStack() as stack:
            _happy_path(stack, overrides={
                "get_trading_decisions": MagicMock(side_effect=RuntimeError("claude down")),
            })
            result = run_trading_session(dry_run=True)
        assert any("LLM decision failed" in e for e in result.errors)
        assert result.decisions_made == 0


class TestDecisionLoopBranches:
    def test_hold_decision_is_skipped(self, mock_db, mock_cursor):
        decision = _make_decision(action="hold", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            result = run_trading_session(dry_run=True)
        mocks["execute_market_order"].assert_not_called()
        assert result.trades_executed == 0

    def test_trade_limit_halts_loop(self, mock_db, mock_cursor):
        """After 10 successful trades, subsequent decisions are skipped (break)."""
        decisions = [
            _make_decision(ticker=f"T{i}", action="buy")
            for i in range(12)
        ]
        with ExitStack() as stack:
            _happy_path(stack, decisions=decisions)
            result = run_trading_session(dry_run=True)
        # Exactly 10 trades executed; two decisions skipped by the limit.
        assert result.trades_executed == 10

class TestIntentResolution:
    def test_buy_without_magnitude_raises_intent_error(self, mock_db, mock_cursor):
        decision = _make_decision(action="buy", intent_magnitude=None)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision])
            result = run_trading_session(dry_run=True)
        assert result.trades_failed == 1
        assert decision.action == "invalid"
        assert "intent error" in decision.reasoning.lower()

    def test_unsupported_action_raises_intent_error(self, mock_db, mock_cursor):
        # "short" is not in (hold, buy, sell) → IntentError("unsupported action")
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="short",
            intent_type="invest_dollar", intent_magnitude=Decimal("500"),
            reasoning="test", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision])
            result = run_trading_session(dry_run=True)
        assert result.trades_failed == 1
        assert decision.action == "invalid"
        assert "unsupported action" in decision.reasoning.lower()


class TestSectorCapHardGate:
    """P3.30: hard pre-submit sector-concentration gate. The advisory text
    in risk_notes was the only line of defense before — nothing prevented
    the executor LLM from emitting a buy that ran the book over the cap.
    The gate enforces the same threshold structurally on the buy path."""

    def test_buy_rejected_when_breaches_sector_cap(self, mock_db, mock_cursor):
        """$39k of AAPL (tech, 39%) + $2k MSFT buy (tech) → 41% > 40% cap.
        Decision should be marked invalid before submit."""
        decision = _make_decision(
            ticker="MSFT", action="buy",
            intent_type="invest_dollar",
            intent_magnitude=Decimal("2000"),
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[
                    {"ticker": "AAPL", "shares": Decimal("260")},
                ]),
            })
            run_trading_session(dry_run=False)
        mocks["execute_market_order"].assert_not_called()
        assert decision.action == "invalid"
        assert "sector" in decision.reasoning.lower()
        assert "tech" in decision.reasoning.lower()

    def test_buy_allowed_under_sector_cap(self, mock_db, mock_cursor):
        """$30k of AAPL (tech, 30%) + $2k MSFT (tech) → 32% < 40% cap.
        Buy should proceed normally."""
        decision = _make_decision(
            ticker="MSFT", action="buy",
            intent_type="invest_dollar",
            intent_magnitude=Decimal("2000"),
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[
                    {"ticker": "AAPL", "shares": Decimal("200")},
                ]),
            })
            run_trading_session(dry_run=False)
        mocks["execute_market_order"].assert_called()
        assert decision.action == "buy"

    def test_sell_not_blocked_by_sector_cap(self, mock_db, mock_cursor):
        """Sells reduce sector exposure — gate only fires on buys."""
        decision = _make_decision(
            ticker="AAPL", action="sell",
            intent_type="exit_full",
            intent_magnitude=None,
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                # Massively concentrated tech book — irrelevant for sells.
                "get_positions": MagicMock(return_value=[
                    {"ticker": "AAPL", "shares": Decimal("1000")},
                ]),
            })
            run_trading_session(dry_run=False)
        # Sell goes through (no sector-cap rejection); other gates may
        # still affect it but the sector check must not.
        assert "sector" not in decision.reasoning.lower()

    def test_sector_cap_refreshed_mid_loop(self, mock_db, mock_cursor):
        """T1.2: position_values must update after each fill so cumulative
        same-sector buys can't sneak past the gate.

        Setup: $100k portfolio, $20k existing tech (NVDA), MAX_SECTOR_PCT=40%.
        Three new tech buys at $10k each (capped by MAX_POSITION_PCT=10%):
          - Buy 1 (AAPL): projected sector = $30k → pass
          - Buy 2 (MSFT): projected sector = $40k → at cap (not > 40%) → pass
          - Buy 3 (GOOGL): projected sector = $50k → 50% > 40% → REJECT
        Pre-fix: all three would see the stale $20k pre-loop snapshot and pass.
        """
        d1 = _make_decision(ticker="AAPL", action="buy", intent_magnitude=Decimal("10000"), playbook_action_id=1)
        d2 = _make_decision(ticker="MSFT", action="buy", intent_magnitude=Decimal("10000"), playbook_action_id=2)
        d3 = _make_decision(ticker="GOOGL", action="buy", intent_magnitude=Decimal("10000"), playbook_action_id=3)

        # Existing $20k tech position: 100 * $200 NVDA. Need a per-ticker price
        # source: get_latest_trade_price values the existing book; get_latest_price
        # quotes new buys. Set both to $200 so 100 shares = $20k for NVDA and
        # $10k buys resolve to 50 shares (50 * $200 = $10k trade_value).
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[d1, d2, d3], overrides={
                "get_positions": MagicMock(return_value=[
                    {"ticker": "NVDA", "shares": Decimal("100")},
                ]),
                "get_latest_price_with_reason": MagicMock(return_value=(Decimal("200"), None)),
                "get_latest_trade_price": MagicMock(return_value=Decimal("200")),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="ord-1", error=None,
                    filled_qty=Decimal("50"), filled_avg_price=Decimal("200"),
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=True, order_id="ord-1", error=None,
                    filled_qty=Decimal("50"), filled_avg_price=Decimal("200"),
                )),
            })
            run_trading_session(dry_run=False)

        assert d1.action == "buy", f"buy 1 should pass, got: {d1.reasoning}"
        assert d2.action == "buy", f"buy 2 should pass, got: {d2.reasoning}"
        assert d3.action == "invalid", f"buy 3 should be rejected by sector cap, got: {d3.reasoning}"
        assert "sector" in d3.reasoning.lower()


class TestRiskBlockTelemetry:
    """When the sector-cap gate rejects a buy, the trader must emit a
    `risk_block` event so the auditor can detect ticker hotspots and bursts
    (RISK_BLOCK_HOTSPOT, RISK_BLOCK_BURST). No event when the gate doesn't trip."""

    def test_emits_risk_block_event_on_sector_cap_breach(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="MSFT", action="buy",
            intent_type="invest_dollar",
            intent_magnitude=Decimal("2000"),
        )
        with ExitStack() as stack:
            mock_rec = stack.enter_context(patch("v2.trader.record_event"))
            _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[
                    {"ticker": "AAPL", "shares": Decimal("260")},
                ]),
            })
            run_trading_session(dry_run=False)

        risk_events = [
            c for c in mock_rec.call_args_list
            if c.kwargs.get("event_type") == "risk_block"
        ]
        assert len(risk_events) == 1
        ev = risk_events[0].kwargs
        assert ev["stage_name"] == "trading"
        payload = ev["payload"]
        assert payload["ticker"] == "MSFT"
        assert payload["sector"] == "tech"
        assert payload["proposed_qty"] is not None
        assert payload["price"] is not None
        assert payload["sector_pct_after"] is not None
        assert payload["cap"] is not None
        assert "sector" in payload["reason_text"].lower()

    def test_no_risk_block_event_when_under_cap(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="MSFT", action="buy",
            intent_type="invest_dollar",
            intent_magnitude=Decimal("2000"),
        )
        with ExitStack() as stack:
            mock_rec = stack.enter_context(patch("v2.trader.record_event"))
            _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[
                    {"ticker": "AAPL", "shares": Decimal("200")},
                ]),
            })
            run_trading_session(dry_run=False)

        risk_events = [
            c for c in mock_rec.call_args_list
            if c.kwargs.get("event_type") == "risk_block"
        ]
        assert risk_events == []


class TestZeroResolvedQty:
    def test_sell_resolves_to_zero_when_no_holdings(self, mock_db, mock_cursor):
        """exit_full on a ticker with 0 shares → resolved_qty=0 → skip path."""
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[]),  # no holdings
            })
            run_trading_session(dry_run=True)
        mocks["execute_market_order"].assert_not_called()
        assert decision.action == "invalid"
        assert "resolved to 0 shares" in decision.reasoning


class TestAlpacaPrecheck:
    def test_live_availability_check_exception_skips_sell(self, mock_db, mock_cursor):
        """P1.7: when get_live_available_qty raises a transient error, the sell
        is skipped (fail-closed). Submitting without a precheck during Alpaca
        degradation is exactly when stale state is most likely — would rather
        skip a legit sell (recoverable) than submit a bad one.
        """
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_live_available_qty": MagicMock(side_effect=RuntimeError("alpaca 500")),
            })
            result = run_trading_session(dry_run=False)
        mocks["execute_market_order"].assert_not_called()
        assert decision.action == "invalid"
        assert "live availability check failed" in decision.reasoning
        assert result.trades_executed == 0
        assert result.trades_failed == 1

    def test_live_availability_check_exception_updates_playbook_action(self, mock_db, mock_cursor):
        """When precheck fails closed, the playbook action is marked skipped
        so resume logic doesn't re-attempt it (mirrors zero-available branch).
        """
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None,
                                  playbook_action_id=42)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_live_available_qty": MagicMock(side_effect=RuntimeError("alpaca 503")),
            })
            run_trading_session(dry_run=False)
        mocks["update_playbook_action_status"].assert_called_with(42, "skipped")

    def test_no_position_at_alpaca_rejects_sell(self, mock_db, mock_cursor):
        """A.1: get_live_available_qty returns None specifically when Alpaca
        says the position does not exist — strictly worse than '0 available',
        which we already reject. Position sync failures are non-fatal, so a
        stale DB row can outlive a closed position; on a live margin account
        a market sell of a non-held symbol opens an unintended short.
        """
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_live_available_qty": MagicMock(return_value=None),
            })
            result = run_trading_session(dry_run=False)
        mocks["execute_market_order"].assert_not_called()
        assert decision.action == "invalid"
        assert "no position" in decision.reasoning
        assert result.trades_executed == 0
        assert result.trades_failed == 1

    def test_no_position_marks_playbook_action_skipped(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None,
                                  playbook_action_id=42)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_live_available_qty": MagicMock(return_value=None),
            })
            run_trading_session(dry_run=False)
        mocks["update_playbook_action_status"].assert_called_with(42, "skipped")

    def test_sell_trimmed_to_alpaca_available(self, mock_db, mock_cursor):
        """DB says 10 shares but Alpaca reports 4 available → trim to 4."""
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("10")}]),
                "get_live_available_qty": MagicMock(return_value=Decimal("4")),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("4"), filled_avg_price=Decimal("150"),
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("4"), filled_avg_price=Decimal("150"),
                )),
            })
            run_trading_session(dry_run=False)
        call_kwargs = mocks["execute_market_order"].call_args.kwargs
        assert call_kwargs["qty"] == Decimal("4")

    def test_zero_available_without_playbook_action(self, mock_db, mock_cursor):
        """Zero-available rejection branch when playbook_action_id is None
        (covers the 'if decision.playbook_action_id:' False branch, L344->350).
        """
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None,
                                  playbook_action_id=None)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_live_available_qty": MagicMock(return_value=Decimal("0")),
            })
            result = run_trading_session(dry_run=False)
        assert decision.action == "invalid"
        assert result.trades_failed == 1

    def test_zero_available_with_playbook_action_status_update_failure(self, mock_db, mock_cursor):
        """playbook_action_id is set and update_playbook_action_status raises
        inside the zero-available handler — exception is swallowed.
        """
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None,
                                  playbook_action_id=42)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_live_available_qty": MagicMock(return_value=Decimal("0")),
                "update_playbook_action_status": MagicMock(
                    side_effect=RuntimeError("db down"),
                ),
            })
            result = run_trading_session(dry_run=False)
        assert decision.action == "invalid"
        assert result.trades_failed == 1


class TestExecutionSuccessBranches:
    def test_update_playbook_action_status_executed_failure_is_logged(self, mock_db, mock_cursor):
        """After successful execution, update_playbook_action_status raises —
        logged as warning but does not break the flow.
        """
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "update_playbook_action_status": MagicMock(
                    side_effect=RuntimeError("db down"),
                ),
            })
            result = run_trading_session(dry_run=False)
        assert result.trades_executed == 1

    def test_successful_execution_without_playbook_action_id(self, mock_db, mock_cursor):
        """Covers L386->395: 'if decision.playbook_action_id:' False branch
        when execution succeeds but no playbook_action_id is set.
        """
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision])
            result = run_trading_session(dry_run=False)
        assert result.trades_executed == 1

    def test_buying_power_refresh_failure_falls_back_to_local(self, mock_db, mock_cursor):
        """Second call to get_account_info (refresh) raises → use local estimate."""
        decision = _make_decision(ticker="AAPL", action="buy")
        calls = [0]

        def acct_side_effect():
            calls[0] += 1
            if calls[0] == 1:
                return _DEFAULT_ACCOUNT
            raise RuntimeError("refresh flaky")

        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "get_account_info": MagicMock(side_effect=acct_side_effect),
            })
            result = run_trading_session(dry_run=False)
        assert result.trades_executed == 1

    def test_dry_run_buy_adjusts_buying_power_locally(self, mock_db, mock_cursor):
        """dry_run buy path → local buying_power adjustment (L416-417)."""
        decision = _make_decision(ticker="AAPL", action="buy")
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="DRY_RUN", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
            })
            result = run_trading_session(dry_run=True)
        assert result.trades_executed == 1

    def test_dry_run_sell_takes_no_buying_power_adjustment(self, mock_db, mock_cursor):
        """dry_run sell path hits the 'if action == buy' False branch (L416->419)."""
        decision = _make_decision(
            ticker="AAPL", action="sell",
            intent_type="exit_full", intent_magnitude=None,
        )
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="DRY_RUN", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
            })
            result = run_trading_session(dry_run=True)
        assert result.trades_executed == 1

    def test_buying_power_refresh_failure_on_sell(self, mock_db, mock_cursor):
        """Real-trade refresh fails for a SELL — local fallback does NOT adjust
        buying_power (covers L412->419 branch where action != buy).
        """
        decision = _make_decision(
            ticker="AAPL", action="sell",
            intent_type="exit_full", intent_magnitude=None,
        )
        calls = [0]

        def acct_side_effect():
            calls[0] += 1
            if calls[0] == 1:
                return _DEFAULT_ACCOUNT
            raise RuntimeError("refresh flaky")

        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "get_account_info": MagicMock(side_effect=acct_side_effect),
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="ord-1", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=True, order_id="ord-1", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
            })
            result = run_trading_session(dry_run=False)
        assert result.trades_executed == 1

    def test_fill_failure_marks_trade_failed_and_continues(self, mock_db, mock_cursor):
        """wait_for_fill returns success=False → trades_failed++, continue."""
        decision = _make_decision(ticker="AAPL", action="buy")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="real-order", error=None,
                    filled_qty=None, filled_avg_price=None,
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=False, error="timed out",
                    filled_qty=None, filled_avg_price=None,
                )),
            })
            result = run_trading_session(dry_run=False)
        assert result.trades_executed == 0
        assert result.trades_failed == 1
        # The decision is still logged in Step 6 (the execution `continue` skips
        # the order-tracking path but not the bulk logging loop).
        mocks["insert_decision"].assert_called_once()


class TestHoldPlaybookLifecycle:
    def test_playbook_backed_hold_marks_action_deferred(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="AVGO",
            action="hold",
            playbook_action_id=42,
            is_off_playbook=False,
        )

        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            result = run_trading_session(dry_run=True, session_date=date(2026, 5, 26))

        assert result.decisions_made == 1
        mocks["update_playbook_action_status"].assert_called_once_with(42, "deferred")
        mocks["get_pending_playbook_action_for_ticker"].assert_not_called()

    def test_hold_without_playbook_action_id_does_not_mark_deferred(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="AVGO",
            action="hold",
            playbook_action_id=None,
            is_off_playbook=False,
        )

        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            result = run_trading_session(dry_run=True, session_date=date(2026, 5, 26))

        assert result.decisions_made == 1
        mocks["update_playbook_action_status"].assert_not_called()
        mocks["get_pending_playbook_action_for_ticker"].assert_not_called()

    def test_deferred_status_update_failure_does_not_block_logging(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="AVGO",
            action="hold",
            playbook_action_id=42,
            is_off_playbook=False,
        )

        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "update_playbook_action_status": MagicMock(
                    side_effect=RuntimeError("status write failed"),
                ),
            })
            result = run_trading_session(dry_run=True, session_date=date(2026, 5, 26))

        assert result.errors == []
        assert result.decisions_made == 1
        mocks["insert_decision"].assert_called_once()

    def test_off_playbook_hold_with_pending_action_warns(self, mock_db, mock_cursor, caplog):
        decision = _make_decision(
            ticker="AVGO",
            action="hold",
            playbook_action_id=None,
            is_off_playbook=True,
        )

        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_pending_playbook_action_for_ticker": MagicMock(
                    return_value={"id": 99, "ticker": "AVGO"},
                ),
            })
            with caplog.at_level("WARNING", logger="trader"):
                result = run_trading_session(
                    dry_run=True,
                    session_date=date(2026, 5, 26),
                )

        assert result.decisions_made == 1
        mocks["get_pending_playbook_action_for_ticker"].assert_called_once_with(
            date(2026, 5, 26), "AVGO",
        )
        assert "off-playbook HOLD has pending playbook action 99" in caplog.text


class TestThesisLifecycle:
    def test_full_sell_closes_thesis(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="AAPL", action="sell",
            intent_type="exit_full", intent_magnitude=None,
            thesis_id=9,
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
            })
            run_trading_session(dry_run=False)
        # Expect close_thesis called (once for this fill; invalidations loop is empty).
        close_calls = mocks["close_thesis"].call_args_list
        assert any(c.kwargs.get("status") == "closed" for c in close_calls)

    def test_partial_sell_keeps_thesis_active(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="AAPL", action="sell",
            intent_type="exit_partial_pct", intent_magnitude=Decimal("50"),
            thesis_id=9,
        )
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("10")}]),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("5"), filled_avg_price=Decimal("150"),
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("5"), filled_avg_price=Decimal("150"),
                )),
            })
            run_trading_session(dry_run=False)
        # No "closed" status — thesis stays active on partial.
        close_calls = mocks["close_thesis"].call_args_list
        assert not any(c.kwargs.get("status") == "closed" for c in close_calls)

    def test_close_thesis_exception_on_fill_is_recorded(self, mock_db, mock_cursor):
        decision = _make_decision(
            ticker="AAPL", action="sell",
            intent_type="exit_full", intent_magnitude=None,
            thesis_id=9,
        )
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=Decimal("1"), filled_avg_price=Decimal("150"),
                )),
                "close_thesis": MagicMock(side_effect=RuntimeError("thesis table down")),
            })
            result = run_trading_session(dry_run=False)
        assert any("thesis" in e.lower() for e in result.errors)


class TestOrderFailure:
    def test_order_failure_counts_and_updates_playbook(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, order_id=None, error="rejected",
                    filled_qty=None, filled_avg_price=None,
                    duplicate_client_order_id=False,
                )),
            })
            result = run_trading_session(dry_run=False)
        assert result.trades_failed == 1
        assert any("execution failed" in e for e in result.errors)

    def test_order_failure_update_playbook_exception_is_swallowed(self, mock_db, mock_cursor):
        """Order failed, playbook_action_id set, update_playbook_action_status raises."""
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, order_id=None, error="rejected",
                    filled_qty=None, filled_avg_price=None,
                    duplicate_client_order_id=False,
                )),
                "update_playbook_action_status": MagicMock(
                    side_effect=RuntimeError("db down"),
                ),
            })
            result = run_trading_session(dry_run=False)
        assert result.trades_failed == 1

    def test_order_failure_without_playbook_action_id(self, mock_db, mock_cursor):
        """Order fails and decision.playbook_action_id is None — the
        update_playbook_action_status block is skipped (covers L450->249).
        """
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, order_id=None, error="rejected",
                    filled_qty=None, filled_avg_price=None,
                    duplicate_client_order_id=False,
                )),
            })
            result = run_trading_session(dry_run=False)
        assert result.trades_failed == 1


class TestThesisInvalidationExceptions:
    def test_invalidation_close_thesis_exception_is_recorded(self, mock_db, mock_cursor):
        from v2.agent import ThesisInvalidation
        inv = ThesisInvalidation(thesis_id=5, reason="changed")
        with ExitStack() as stack:
            _happy_path(stack, invalidations=[inv], overrides={
                "close_thesis": MagicMock(side_effect=RuntimeError("thesis down")),
            })
            result = run_trading_session(dry_run=True)
        assert any("invalidate thesis 5" in e for e in result.errors)


class TestDecisionLoggingBranches:
    def test_duplicate_decision_is_skipped(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "check_decision_exists": MagicMock(return_value=42),
            })
            run_trading_session(dry_run=True)
        mocks["insert_decision"].assert_not_called()

    def test_insert_decision_exception_recorded_and_loop_continues(self, mock_db, mock_cursor):
        """T1.5: a decision that fails ALL retries records an error and the
        loop continues to the next decision. With retry=3, the first three
        calls fail then the 4th (next decision) succeeds.
        """
        d1 = _make_decision(ticker="AAA", action="buy")
        d2 = _make_decision(ticker="BBB", action="buy")
        with ExitStack() as stack:
            insert_calls = [0]

            def fail_3_then_pass(*a, **kw):
                insert_calls[0] += 1
                if insert_calls[0] <= 3:
                    raise RuntimeError("insert broke")
                return 2

            mocks = _happy_path(stack, decisions=[d1, d2], overrides={
                "insert_decision": MagicMock(side_effect=fail_3_then_pass),
            })
            stack.enter_context(patch("v2.trader.time.sleep"))
            result = run_trading_session(dry_run=True)
        assert any("AAA" in e and "after retries" in e for e in result.errors)
        # 3 retries on AAA + 1 success on BBB
        assert mocks["insert_decision"].call_count == 4

    def test_logged_qty_falls_back_to_decision_quantity(self, mock_db, mock_cursor):
        """Dry run: result has no filled_qty → logged_qty uses decision.quantity."""
        decision = _make_decision(ticker="AAPL", action="buy")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="DRY_RUN", error=None,
                    filled_qty=None, filled_avg_price=None,
                )),
            })
            run_trading_session(dry_run=True)
        inserted_qty = mocks["insert_decision"].call_args.kwargs["quantity"]
        assert inserted_qty is not None
        assert inserted_qty > 0

    def test_logged_qty_is_none_for_hold(self, mock_db, mock_cursor):
        """Hold decision: no execution, no decision.quantity → logged_qty=None."""
        # But HOLDs also never reach the log block via the execution path — yet they
        # still get inserted via the logging loop. Confirm quantity is None.
        decision = _make_decision(action="hold", intent_magnitude=None)
        decision.quantity = None
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            run_trading_session(dry_run=True)
        mocks["insert_decision"].assert_called_once()
        assert mocks["insert_decision"].call_args.kwargs["quantity"] is None

    def test_price_is_none_for_buy_skips_log(self, mock_db, mock_cursor):
        """Price lookup returns None for a buy/sell in the logging loop → skip,
        append error (L488-490).

        The logging loop uses get_latest_trade_price (trade-price reference)
        while _prepare_decision still uses get_latest_price (quote+spread check
        before an order). Simulate a successful buy whose order returned no
        filled_avg_price, then a None trade-price lookup during logging.
        """
        decision = _make_decision(ticker="AAPL", action="buy")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_latest_trade_price": MagicMock(return_value=None),
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=None, filled_avg_price=None,
                )),
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=True, order_id="o", error=None,
                    filled_qty=None, filled_avg_price=None,
                )),
            })
            result = run_trading_session(dry_run=False)
        assert any("No price available" in e for e in result.errors)
        mocks["insert_decision"].assert_not_called()

    def test_log_decisions_skips_dedup_for_invalid_actions(self, mock_db, mock_cursor):
        """Rejected decisions (action='invalid') should not query
        check_decision_exists. The dedup gate is meaningful only for
        buy/sell/hold; running it on 'invalid' suppresses distinct
        rejection audit rows on the same ticker.
        """
        from v2.agent import AgentResponse, ExecutorDecision
        from v2.trader import _log_decisions

        decisions = [
            ExecutorDecision(
                playbook_action_id=None,
                ticker="AAPL",
                action="invalid",
                intent_type=None,
                intent_magnitude=None,
                reasoning="[REJECTED: no price] -",
                confidence="low",
                is_off_playbook=False,
                thesis_id=None,
            ),
            ExecutorDecision(
                playbook_action_id=None,
                ticker="AAPL",
                action="invalid",
                intent_type=None,
                intent_magnitude=None,
                reasoning="[REJECTED: intent error] -",
                confidence="low",
                is_off_playbook=False,
                thesis_id=None,
            ),
        ]
        response = AgentResponse(
            decisions=decisions,
            thesis_invalidations=[],
            market_summary="",
            risk_assessment="",
        )
        account_info = {
            "buying_power": Decimal("10000"),
            "portfolio_value": Decimal("50000"),
        }
        errors: list = []

        with patch("v2.trader.check_decision_exists") as mock_check, \
             patch("v2.trader.get_latest_trade_price", return_value=Decimal("150")), \
             patch("v2.trader._insert_decision_with_retry", return_value=1), \
             patch("v2.trader.format_decisions_for_logging", return_value={}), \
             patch("v2.trader.insert_decision_signals_batch"):
            _log_decisions(
                response, {}, {}, MagicMock(), account_info, errors,
                date(2026, 5, 3),
            )

        assert mock_check.call_count == 0, (
            f"check_decision_exists should not be called for action='invalid'; "
            f"got {mock_check.call_count} calls"
        )


class TestInsertDecisionRetryAndOrphanFallback:
    """T1.5: filled order + failed insert_decision must not produce a silent
    orphan position. Three bounded retries first; final failure with a real
    fill triggers logs/orphan_decisions.jsonl. No fill → no orphan.
    """

    def test_retry_succeeds_on_second_attempt(self, mock_db, mock_cursor, tmp_path, monkeypatch):
        from v2.trader import _insert_decision_with_retry
        monkeypatch.setattr("v2.trader._ORPHAN_DECISIONS_LOG", tmp_path / "orphans.jsonl")
        monkeypatch.setattr("v2.trader.time.sleep", lambda s: None)

        decision = _make_decision(ticker="AAPL", action="buy")
        order_result = MagicMock(success=True, filled_qty=Decimal("5"), filled_avg_price=Decimal("150"))
        attempts = [0]

        def insert(**kwargs):
            attempts[0] += 1
            if attempts[0] < 2:
                raise RuntimeError("transient")
            return 42

        with patch("v2.trader.insert_decision", side_effect=insert):
            result = _insert_decision_with_retry(
                decision=decision, order_id="ord-1", order_result=order_result,
                payload={"ticker": "AAPL", "action": "buy", "quantity": Decimal("5"),
                         "decision_date": date(2026, 5, 3), "price": Decimal("150"),
                         "reasoning": "r", "signals_used": {},
                         "account_equity": Decimal("100000"), "buying_power": Decimal("50000"),
                         "playbook_action_id": 1, "is_off_playbook": False, "order_id": "ord-1"},
            )
        assert result == 42
        assert attempts[0] == 2
        # No orphan file written on success
        assert not (tmp_path / "orphans.jsonl").exists()

    def test_filled_order_then_persistent_failure_writes_orphan_jsonl(
        self, mock_db, mock_cursor, tmp_path, monkeypatch,
    ):
        from v2.trader import _insert_decision_with_retry
        orphan_log = tmp_path / "orphans.jsonl"
        monkeypatch.setattr("v2.trader._ORPHAN_DECISIONS_LOG", orphan_log)
        monkeypatch.setattr("v2.trader.time.sleep", lambda s: None)

        decision = _make_decision(
            ticker="AAPL", action="buy",
            signal_refs=[{"type": "news_signal", "id": 99}],
            playbook_action_id=7, thesis_id=12,
        )
        order_result = MagicMock(
            success=True, filled_qty=Decimal("5"),
            filled_avg_price=Decimal("150.25"),
        )

        with patch("v2.trader.insert_decision", side_effect=RuntimeError("db down")):
            result = _insert_decision_with_retry(
                decision=decision, order_id="ord-abc", order_result=order_result,
                payload={"ticker": "AAPL", "action": "buy", "quantity": Decimal("5"),
                         "decision_date": date(2026, 5, 3), "price": Decimal("150.25"),
                         "reasoning": "buy thesis hit", "signals_used": {},
                         "account_equity": Decimal("100000"), "buying_power": Decimal("50000"),
                         "playbook_action_id": 7, "is_off_playbook": False, "order_id": "ord-abc"},
            )

        assert result is None
        assert orphan_log.exists(), "orphan JSONL must be written when fill happened"
        import json
        line = orphan_log.read_text().strip()
        record = json.loads(line)
        assert record["ticker"] == "AAPL"
        assert record["action"] == "buy"
        assert record["order_id"] == "ord-abc"
        assert record["filled_qty"] == "5"
        assert record["filled_avg_price"] == "150.25"
        assert record["playbook_action_id"] == 7
        assert record["thesis_id"] == 12
        assert record["signal_refs"] == [{"type": "news_signal", "id": 99}]
        assert "db down" in record["last_error"]

    def test_unfilled_order_failure_does_not_write_orphan(
        self, mock_db, mock_cursor, tmp_path, monkeypatch,
    ):
        """A decision that wasn't actually executed (no real position) must
        NOT pollute the orphan log. Operator reconciliation only matters when
        Alpaca shows shares moved."""
        from v2.trader import _insert_decision_with_retry
        orphan_log = tmp_path / "orphans.jsonl"
        monkeypatch.setattr("v2.trader._ORPHAN_DECISIONS_LOG", orphan_log)
        monkeypatch.setattr("v2.trader.time.sleep", lambda s: None)

        decision = _make_decision(ticker="AAPL", action="hold")
        # No order_result: decision never went to Alpaca
        with patch("v2.trader.insert_decision", side_effect=RuntimeError("db down")):
            result = _insert_decision_with_retry(
                decision=decision, order_id=None, order_result=None,
                payload={"ticker": "AAPL", "action": "hold", "quantity": None,
                         "decision_date": date(2026, 5, 3), "price": Decimal("150"),
                         "reasoning": "r", "signals_used": {},
                         "account_equity": Decimal("100000"), "buying_power": Decimal("50000"),
                         "playbook_action_id": 1, "is_off_playbook": False, "order_id": None},
            )
        assert result is None
        assert not orphan_log.exists()

    def test_zero_fill_does_not_produce_orphan(
        self, mock_db, mock_cursor, tmp_path, monkeypatch,
    ):
        """Order submitted but filled_qty=0 (e.g., canceled before fill) is not
        an orphan — nothing to reconcile.
        """
        from v2.trader import _insert_decision_with_retry
        orphan_log = tmp_path / "orphans.jsonl"
        monkeypatch.setattr("v2.trader._ORPHAN_DECISIONS_LOG", orphan_log)
        monkeypatch.setattr("v2.trader.time.sleep", lambda s: None)

        decision = _make_decision(ticker="AAPL", action="buy")
        # T2.8: explicitly set unknown_partial_fill=False so this test
        # exercises the genuine zero-fill path. Without the explicit
        # assignment a MagicMock attribute is truthy and would mis-route
        # this case through the unknown-fill orphan branch.
        order_result = MagicMock(
            success=True, filled_qty=Decimal("0"), filled_avg_price=None,
            unknown_partial_fill=False,
        )
        with patch("v2.trader.insert_decision", side_effect=RuntimeError("db down")):
            _insert_decision_with_retry(
                decision=decision, order_id="ord-1", order_result=order_result,
                payload={"ticker": "AAPL", "action": "buy", "quantity": Decimal("0"),
                         "decision_date": date(2026, 5, 3), "price": Decimal("150"),
                         "reasoning": "r", "signals_used": {},
                         "account_equity": Decimal("100000"), "buying_power": Decimal("50000"),
                         "playbook_action_id": 1, "is_off_playbook": False, "order_id": "ord-1"},
            )
        assert not orphan_log.exists()

    def test_unknown_partial_fill_produces_orphan(
        self, mock_db, mock_cursor, tmp_path, monkeypatch,
    ):
        """T2.8: when post-cancel re-fetch failed, the order_result carries
        `unknown_partial_fill=True`. The trader can't tell whether anything
        filled, so an orphan reconciliation log is the safe choice — better
        a redundant entry than a silently missed fill.
        """
        from v2.trader import _insert_decision_with_retry
        orphan_log = tmp_path / "orphans.jsonl"
        monkeypatch.setattr("v2.trader._ORPHAN_DECISIONS_LOG", orphan_log)
        monkeypatch.setattr("v2.trader.time.sleep", lambda s: None)

        decision = _make_decision(ticker="AAPL", action="buy")
        order_result = MagicMock(
            success=False, filled_qty=None, filled_avg_price=None,
            unknown_partial_fill=True,
        )
        with patch("v2.trader.insert_decision", side_effect=RuntimeError("db down")):
            _insert_decision_with_retry(
                decision=decision, order_id="ord-7", order_result=order_result,
                payload={"ticker": "AAPL", "action": "buy", "quantity": Decimal("1"),
                         "decision_date": date(2026, 5, 3), "price": Decimal("150"),
                         "reasoning": "r", "signals_used": {},
                         "account_equity": Decimal("100000"), "buying_power": Decimal("50000"),
                         "playbook_action_id": 1, "is_off_playbook": False, "order_id": "ord-7"},
            )
        assert orphan_log.exists()
        contents = orphan_log.read_text()
        assert "ord-7" in contents


class TestSignalRefValidation:
    def test_stripped_invalid_signal_refs_are_logged(self, mock_db, mock_cursor):
        """validated_refs is shorter than original → warning log, insert batch w/ valid refs."""
        decision = _make_decision(ticker="AAPL", action="buy", signal_refs=[
            {"type": "news_signal", "id": 1},
            {"type": "news_signal", "id": 2},
        ])
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "validate_signal_refs": MagicMock(return_value=[{"type": "news_signal", "id": 1}]),
            })
            run_trading_session(dry_run=True)
        mocks["insert_decision_signals_batch"].assert_called_once()

    def test_validate_signal_refs_all_valid(self, mock_db, mock_cursor):
        """validated_refs == original length → no warning, batch inserted."""
        refs = [{"type": "news_signal", "id": 1}]
        decision = _make_decision(ticker="AAPL", action="buy", signal_refs=refs)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            run_trading_session(dry_run=True)
        mocks["insert_decision_signals_batch"].assert_called_once()

    def test_validated_refs_empty_emits_signal_gap_marker(self, mock_db, mock_cursor):
        """validated_refs is empty on buy/sell → signal_gap marker row inserted.

        Closes audit Rule 32 enforcement: incompletely attributed decisions
        get a 'signal_gap' decision_signals row flagging the attribution gap.
        """
        decision = _make_decision(ticker="AAPL", action="buy", signal_refs=[
            {"type": "junk", "id": 1},
        ])
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "validate_signal_refs": MagicMock(return_value=[]),
            })
            run_trading_session(dry_run=True)
        # Exactly one call with the signal_gap marker row.
        calls = mocks["insert_decision_signals_batch"].call_args_list
        assert len(calls) == 1
        rows = calls[0].args[0]
        assert any(r[1] == "signal_gap" for r in rows), rows

    def test_signal_link_exception_is_recorded(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", signal_refs=[
            {"type": "news_signal", "id": 1},
        ])
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "insert_decision_signals_batch": MagicMock(side_effect=RuntimeError("link table down")),
            })
            result = run_trading_session(dry_run=True)
        assert any("signal links" in e for e in result.errors)

    def test_buy_without_signal_refs_logs_warning(self, mock_db, mock_cursor, caplog):
        """action in (buy, sell) with empty signal_refs → warning log (L543-545)."""
        decision = _make_decision(ticker="AAPL", action="buy", signal_refs=[])
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision])
            with caplog.at_level("WARNING", logger="trader"):
                run_trading_session(dry_run=True)
        assert any("no concrete signal_refs cited" in r.getMessage() for r in caplog.records)


class TestTraderCliMain:
    def test_main_forwards_args(self, mock_db, mock_cursor):
        import sys as _sys

        from v2.trader import TradingSessionResult as TSR
        from v2.trader import main

        argv = ["v2.trader", "--dry-run"]
        ok = TSR(
            timestamp=datetime.now(), account_snapshot_id=1,
            positions_synced=0, orders_synced=0,
            decisions_made=0, trades_executed=0, trades_failed=0,
            total_buy_value=Decimal(0), total_sell_value=Decimal(0),
            errors=[],
        )
        with patch.object(_sys, "argv", argv), \
             patch("v2.log_config.setup_logging"), \
             patch("v2.trader.run_trading_session", return_value=ok) as mock_run:
            main()
        mock_run.assert_called_once()

class TestPreSubmitDedup:
    """P1.6: dedup must run BEFORE order submission. The previous post-submit
    check left a window where insert_decision failure + operator rerun would
    re-submit the order with no DB or broker-side guard.
    """

    def test_pre_submit_dedup_blocks_buy_when_decision_already_exists(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy",
                                  intent_type="invest_dollar",
                                  intent_magnitude=Decimal("500"))
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "check_decision_exists": MagicMock(return_value=99),
            })
            result = run_trading_session(dry_run=False)
        mocks["execute_market_order"].assert_not_called()
        assert decision.action == "invalid"
        assert "duplicate decision today" in decision.reasoning
        assert result.trades_executed == 0
        assert result.trades_failed == 1

    def test_pre_submit_dedup_blocks_sell_when_decision_already_exists(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "check_decision_exists": MagicMock(return_value=42),
            })
            result = run_trading_session(dry_run=False)
        mocks["execute_market_order"].assert_not_called()
        assert decision.action == "invalid"
        assert result.trades_failed == 1

    def test_dry_run_does_not_block_preview_on_existing_row(self, mock_db, mock_cursor):
        """Dry-run never submits a real order, so the pre-submit dedup gate is
        deliberately skipped — preview output should still reflect what *would*
        happen even if a row was already logged today. (The post-log dedup at
        step 6 still fires; that's expected, it just gates the duplicate insert.)
        """
        decision = _make_decision(ticker="AAPL", action="buy",
                                  intent_type="invest_dollar",
                                  intent_magnitude=Decimal("500"))
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "check_decision_exists": MagicMock(return_value=99),
            })
            run_trading_session(dry_run=True)
        # Preview path runs in spite of the existing row (action stays "buy",
        # the dry-run executor returns DRY_RUN). The pre-submit dedup is
        # bypassed; the order_action is unchanged.
        assert decision.action == "buy"
        mocks["execute_market_order"].assert_called_once()


class TestClientOrderIdPlumbing:
    """P1.6: deterministic client_order_id sent to Alpaca for broker-side
    idempotency. Same decision → same key → broker rejects duplicate submit.
    """

    def test_buy_passes_deterministic_client_order_id(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy",
                                  intent_type="invest_dollar",
                                  intent_magnitude=Decimal("500"),
                                  playbook_action_id=42)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            run_trading_session(dry_run=False)
        kwargs = mocks["execute_market_order"].call_args.kwargs
        # Format: algo-YYYYMMDD-{action[0]}-{TICKER}-{playbook_action_id}
        coid = kwargs["client_order_id"]
        assert coid.startswith("algo-")
        assert coid.endswith("-b-AAPL-42")

    def test_off_playbook_decision_uses_op_marker(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="NVDA", action="buy",
                                  intent_type="invest_dollar",
                                  intent_magnitude=Decimal("500"),
                                  playbook_action_id=None,
                                  is_off_playbook=True)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision])
            run_trading_session(dry_run=False)
        coid = mocks["execute_market_order"].call_args.kwargs["client_order_id"]
        assert coid.endswith("-b-NVDA-op")


    def test_main_exits_nonzero_on_errors(self, mock_db, mock_cursor):
        import sys as _sys

        from v2.trader import TradingSessionResult as TSR
        from v2.trader import main

        failed = TSR(
            timestamp=datetime.now(), account_snapshot_id=1,
            positions_synced=0, orders_synced=0,
            decisions_made=0, trades_executed=0, trades_failed=0,
            total_buy_value=Decimal(0), total_sell_value=Decimal(0),
            errors=["boom"],
        )
        with patch.object(_sys, "argv", ["v2.trader"]), \
             patch("v2.log_config.setup_logging"), \
             patch("v2.trader.run_trading_session", return_value=failed):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestIntraBatchDedup:
    """A.2: all three dedup layers are blind to duplicates *within one*
    executor response — decision rows are only written after the execution
    loop, and a playbook buy vs an off-playbook buy of the same ticker sign
    different client_order_ids, so Alpaca accepts both. Both filled; only one
    got a decision row (the second hit the dedup skip at logging time).
    """

    def test_duplicate_buys_execute_once_playbook_wins(self, mock_db, mock_cursor):
        off = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                             is_off_playbook=True)
        pb = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[off, pb])
            result = run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 1
        assert off.action == "invalid"
        assert "duplicate" in off.reasoning
        assert pb.action == "buy"
        assert result.trades_executed == 1
        assert result.trades_failed == 1

    def test_duplicate_buys_playbook_first_still_keeps_playbook(self, mock_db, mock_cursor):
        """Order within the batch must not decide the winner."""
        pb = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        off = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                             is_off_playbook=True)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[pb, off])
            run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 1
        assert off.action == "invalid"
        assert pb.action == "buy"

    def test_two_off_playbook_duplicates_keep_first(self, mock_db, mock_cursor):
        a = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                           is_off_playbook=True, reasoning="first")
        b = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                           is_off_playbook=True, reasoning="second")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[a, b])
            run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 1
        assert a.action == "buy"
        assert b.action == "invalid"

    def test_buy_and_sell_same_ticker_not_deduped(self, mock_db, mock_cursor):
        """Different actions are not duplicates — only (ticker, action) pairs."""
        buy = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        sell = _make_decision(ticker="MSFT", action="sell", playbook_action_id=8,
                              intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[buy, sell], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "MSFT", "shares": Decimal("5")}]),
            })
            run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 2

    def test_duplicate_holds_not_rejected(self, mock_db, mock_cursor):
        """Holds submit no orders; dedup'ing them would lose override reasoning."""
        h1 = _make_decision(ticker="AAPL", action="hold", playbook_action_id=None)
        h2 = _make_decision(ticker="AAPL", action="hold", playbook_action_id=None)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[h1, h2])
            run_trading_session(dry_run=False)
        assert h1.action == "hold"
        assert h2.action == "hold"

    def test_staged_same_ticker_sells_rejected(self, mock_db, mock_cursor):
        """A 'staged exit' (exit_partial_pct then exit_full for one ticker in
        one batch) is a duplicate, not a feature: tool_write_playbook already
        rejects duplicate (ticker, action) pairs, and idx_decisions_dedup —
        UNIQUE (date, ticker, action) for buy/sell — means the second could
        never be recorded. Executing it produced an untracked fill.
        """
        first = _make_decision(ticker="AAPL", action="sell", playbook_action_id=None,
                               is_off_playbook=True, thesis_id=42,
                               intent_type="exit_partial_pct",
                               intent_magnitude=Decimal("50"))
        second = _make_decision(ticker="AAPL", action="sell", playbook_action_id=None,
                                is_off_playbook=True, thesis_id=42,
                                intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[first, second], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("100")}]),
            })
            run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 1
        assert first.action == "sell"
        assert second.action == "invalid"

    def test_rejected_duplicate_not_double_counted(self, mock_db, mock_cursor):
        """The stamped-invalid loser must be skipped by the execution loop —
        resolving its intent would raise IntentError ('unsupported action:
        invalid') and count the same rejection twice.
        """
        off = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                             is_off_playbook=True)
        pb = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[off, pb])
            result = run_trading_session(dry_run=False)
        assert result.trades_failed == 1
        assert not [e for e in result.errors if "intent error" in e.lower()]

    def test_dedup_emits_telemetry(self, mock_db, mock_cursor):
        off = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                             is_off_playbook=True)
        pb = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[off, pb])
            mock_rec = stack.enter_context(patch("v2.trader.record_event"))
            run_trading_session(dry_run=False, session_id=11)
        codes = [
            c.kwargs.get("payload", {}).get("reason_code")
            for c in mock_rec.call_args_list
        ]
        assert "intra_batch_duplicate" in codes


class TestFailedOrderLogging:
    """A.3: submit/fill failures left decision.action untouched, unlike every
    pre-submit rejection path (which stamps [REJECTED:] + action='invalid').
    _log_decisions then inserted a real buy/sell row with a quantity and a
    fresh trade price — a trade that never happened, eligible for backfill
    (action IN ('buy','sell') AND price IS NOT NULL) and attribution. The
    phantom row also tripped pre-submit dedup, so even --force couldn't place
    the intended trade that day.
    """

    def test_failed_submit_stamped_invalid(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, error="insufficient buying power",
                    duplicate_client_order_id=False)),
            })
            result = run_trading_session(dry_run=False)
        assert decision.action == "invalid"
        assert "[FAILED:" in decision.reasoning
        assert result.trades_executed == 0
        assert result.trades_failed == 1
        assert mocks["insert_decision"].call_args.kwargs["action"] == "invalid"

    def test_failed_submit_marks_playbook_action_failed(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, error="rejected", duplicate_client_order_id=False)),
            })
            run_trading_session(dry_run=False)
        mocks["update_playbook_action_status"].assert_called_with(7, "failed")

    def test_failed_fill_stamped_invalid(self, mock_db, mock_cursor):
        """The 4 PM queue/timeout case: submitted, never filled, cancelled."""
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=False, error="timeout after 30s", order_id="ord-1",
                    filled_qty=None, filled_avg_price=None)),
            })
            result = run_trading_session(dry_run=False)
        assert decision.action == "invalid"
        assert "[FAILED:" in decision.reasoning
        assert result.trades_executed == 0
        assert mocks["insert_decision"].call_args.kwargs["action"] == "invalid"

    def test_failed_fill_marks_playbook_action_failed(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=False, error="timeout", order_id="ord-1",
                    filled_qty=None, filled_avg_price=None)),
            })
            run_trading_session(dry_run=False)
        mocks["update_playbook_action_status"].assert_called_with(7, "failed")

    def test_duplicate_client_order_id_not_stamped(self, mock_db, mock_cursor):
        """P1.6 race-loser stays a benign skip — the winner's row is the real
        record, and stamping the loser would write a bogus rejection row.
        """
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, error="duplicate client_order_id",
                    duplicate_client_order_id=True)),
            })
            run_trading_session(dry_run=False)
        assert decision.action == "buy"
        assert "[FAILED:" not in decision.reasoning
        for call in mocks["update_playbook_action_status"].call_args_list:
            assert call.args[1] != "failed"

    def test_failed_order_not_eligible_for_backfill(self, mock_db, mock_cursor):
        """The point of the fix: backfill selects action IN ('buy','sell'),
        so an invalid row can never become a phantom outcome.
        """
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, error="boom", duplicate_client_order_id=False)),
            })
            run_trading_session(dry_run=False)
        logged = mocks["insert_decision"].call_args.kwargs
        assert logged["action"] not in ("buy", "sell")
        assert logged["order_id"] is None


class TestLlmIdValidation:
    """A.6/D.3: thesis_id and playbook_action_id are LLM-authored and were the
    only such pointers written to the DB unvalidated — signal_refs get DB
    validation, tickers get normalization. A hallucinated or transposed id
    could mark an arbitrary historical playbook action executed, or close /
    invalidate an unrelated active thesis. Theses are the system's run-to-run
    memory, and the executor (smallest model, least context) never sees their
    text — only integers.
    """

    def _action(self, id=7, ticker="AAPL", thesis_id=3):
        from v2.agent import PlaybookAction
        return PlaybookAction(
            id=id, ticker=ticker, action="buy", thesis_id=thesis_id,
            reasoning="r", confidence="high", intent_type="invest_dollar",
            intent_magnitude=Decimal("500"), priority=1,
        )

    def _input(self, **kw):
        return ExecutorInput(
            playbook_actions=[self._action(**kw)], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )

    def test_hallucinated_playbook_action_id_nulled(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=999)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "build_executor_input": MagicMock(return_value=self._input()),
            })
            run_trading_session(dry_run=False)
        assert decision.playbook_action_id is None
        assert decision.is_off_playbook is True
        for call in mocks["update_playbook_action_status"].call_args_list:
            assert call.args[0] != 999

    def test_ticker_mismatch_nulls_playbook_action_id(self, mock_db, mock_cursor):
        """Action 7 is AAPL's; a MSFT decision must not claim it."""
        decision = _make_decision(ticker="MSFT", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "build_executor_input": MagicMock(return_value=self._input(ticker="AAPL")),
            })
            run_trading_session(dry_run=False)
        assert decision.playbook_action_id is None
        assert decision.is_off_playbook is True

    def test_valid_playbook_action_id_kept(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "build_executor_input": MagicMock(return_value=self._input()),
            })
            run_trading_session(dry_run=False)
        assert decision.playbook_action_id == 7
        assert decision.is_off_playbook is False
        mocks["update_playbook_action_status"].assert_any_call(7, "executed")

    def test_hallucinated_thesis_id_not_closed_blindly(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="sell", playbook_action_id=None,
                                  is_off_playbook=True, intent_type="exit_full",
                                  intent_magnitude=None, thesis_id=555)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "get_active_theses": MagicMock(return_value=[]),
            })
            run_trading_session(dry_run=False)
        assert decision.thesis_id is None
        mocks["close_thesis"].assert_not_called()

    def test_thesis_id_matching_active_thesis_kept(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="sell", playbook_action_id=None,
                                  is_off_playbook=True, intent_type="exit_full",
                                  intent_magnitude=None, thesis_id=42)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "get_active_theses": MagicMock(return_value=[{"id": 42}]),
            })
            run_trading_session(dry_run=False)
        assert decision.thesis_id == 42
        mocks["close_thesis"].assert_called_once()

    def test_thesis_id_from_playbook_action_kept_without_db_lookup(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="sell", playbook_action_id=7,
                                  intent_type="exit_full", intent_magnitude=None,
                                  thesis_id=3)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "build_executor_input": MagicMock(
                    return_value=ExecutorInput(
                        playbook_actions=[self._action(id=7, ticker="AAPL", thesis_id=3)],
                        positions=[], account={}, attribution_summary={},
                        recent_outcomes=[], market_outlook="", risk_notes="")),
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "get_active_theses": MagicMock(return_value=[]),
            })
            run_trading_session(dry_run=False)
        assert decision.thesis_id == 3
        mocks["get_active_theses"].assert_not_called()

    def test_thesis_lookup_failure_drops_id(self, mock_db, mock_cursor):
        """Fail closed: an unverifiable id must not close a thesis."""
        decision = _make_decision(ticker="AAPL", action="sell", playbook_action_id=None,
                                  is_off_playbook=True, intent_type="exit_full",
                                  intent_magnitude=None, thesis_id=42)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("1")}]),
                "get_active_theses": MagicMock(side_effect=RuntimeError("db down")),
            })
            run_trading_session(dry_run=False)
        assert decision.thesis_id is None
        mocks["close_thesis"].assert_not_called()

    def test_unknown_thesis_invalidation_dropped(self, mock_db, mock_cursor):
        from v2.agent import ThesisInvalidation
        inv = ThesisInvalidation(thesis_id=888, reason="gone")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[], invalidations=[inv], overrides={
                "build_executor_input": MagicMock(return_value=self._input(thesis_id=3)),
            })
            run_trading_session(dry_run=False)
        mocks["close_thesis"].assert_not_called()

    def test_known_thesis_invalidation_processed(self, mock_db, mock_cursor):
        """A thesis the executor actually saw may still be invalidated."""
        from v2.agent import ThesisInvalidation
        inv = ThesisInvalidation(thesis_id=3, reason="broken")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[], invalidations=[inv], overrides={
                "build_executor_input": MagicMock(return_value=self._input(thesis_id=3)),
            })
            run_trading_session(dry_run=False)
        mocks["close_thesis"].assert_called_once_with(
            thesis_id=3, status="invalidated", reason="broken")
