"""Trading agent orchestrator -- daily automation entry point."""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient

from .agent import (
    DEFAULT_EXECUTOR_MODEL,
    ExecutorDecision,
    ExecutorInput,
    format_decisions_for_logging,
    get_trading_decisions,
    validate_signal_refs,
)
from .context import build_executor_input
from .database.trading_db import (
    check_decision_exists,
    close_thesis,
    get_positions,
    insert_decision,
    insert_decision_signals_batch,
    update_playbook_action_status,
)
from .executor import (
    execute_market_order,
    get_account_info,
    get_latest_price,
    get_latest_trade_price,
    get_live_available_qty,
    is_market_open,
    sync_orders_from_alpaca,
    sync_positions_from_alpaca,
    take_account_snapshot,
    wait_for_fill,
)
from .intents import (
    BuyIntent,
    IntentError,
    SellIntent,
    resolve_buy_intent,
    resolve_sell_intent,
)
from .risk import check_sector_cap_for_buy, check_sector_concentration

logger = logging.getLogger("trader")


@dataclass
class TradeResult:
    """Result of executing a trading decision."""
    decision: ExecutorDecision
    executed: bool
    order_id: str | None
    filled_price: Decimal | None
    error: str | None


@dataclass
class TradingSessionResult:
    """Result of a complete trading session."""
    timestamp: datetime
    account_snapshot_id: int
    positions_synced: int
    orders_synced: int
    decisions_made: int
    trades_executed: int
    trades_failed: int
    total_buy_value: Decimal
    total_sell_value: Decimal
    errors: list[str]
    market_summary: str = ""
    risk_assessment: str = ""


def _empty_result(timestamp, positions_synced, orders_synced, snapshot_id, errors):
    return TradingSessionResult(
        timestamp=timestamp,
        account_snapshot_id=snapshot_id,
        positions_synced=positions_synced,
        orders_synced=orders_synced,
        decisions_made=0,
        trades_executed=0,
        trades_failed=0,
        total_buy_value=Decimal(0),
        total_sell_value=Decimal(0),
        errors=errors,
    )


def _sync_from_alpaca(errors: list[str]) -> tuple[int, int]:
    """Sync positions and open orders; return counts, append failures to errors."""
    try:
        positions_synced = sync_positions_from_alpaca()
        logger.info("Synced %d positions", positions_synced)
    except Exception as e:
        errors.append(f"Position sync failed: {e}")
        logger.error("Position sync failed: %s", e)
        positions_synced = 0

    try:
        orders_synced = sync_orders_from_alpaca()
        logger.info("Synced %d open orders", orders_synced)
    except Exception as e:
        errors.append(f"Order sync failed: {e}")
        logger.error("Order sync failed: %s", e)
        orders_synced = 0

    return positions_synced, orders_synced


def _snapshot_account(errors: list[str]) -> tuple[dict | None, int]:
    """Take account snapshot; return (account_info, snapshot_id) or (None, 0) on failure."""
    try:
        account_info = get_account_info()
        snapshot_id = take_account_snapshot()
        logger.info("Snapshot ID: %d", snapshot_id)
        logger.info(
            "Portfolio value: $%s  Buying power: $%s",
            f"{float(account_info['portfolio_value']):,.2f}",
            f"{float(account_info['buying_power']):,.2f}",
        )
        return account_info, snapshot_id
    except Exception as e:
        errors.append(f"Account snapshot failed: {e}")
        logger.error("Account snapshot failed: %s", e, exc_info=True)
        return None, 0


def _build_executor_context(account_info: dict, data_client, errors: list[str]) -> ExecutorInput:
    """Build executor input and augment risk_notes with sector concentration warnings."""
    try:
        executor_input = build_executor_input(account_info)
        logger.info("Executor input built")
    except Exception as e:
        errors.append(f"Context build failed: {e}")
        logger.error("Context build failed: %s", e, exc_info=True)
        executor_input = ExecutorInput(
            playbook_actions=[],
            positions=[],
            account=account_info,
            attribution_summary={},
            recent_outcomes=[],
            market_outlook=f"Error building context: {e}",
            risk_notes="",
        )

    position_values = {}
    for p in get_positions():
        price = get_latest_trade_price(p["ticker"], client=data_client)
        if price:
            position_values[p["ticker"]] = p["shares"] * price
    sector_warnings = check_sector_concentration(position_values, account_info["portfolio_value"])
    if sector_warnings:
        logger.warning("Sector concentration warnings: %s", sector_warnings)
        if executor_input.risk_notes:
            executor_input.risk_notes += "\n" + "\n".join(sector_warnings)
        else:
            executor_input.risk_notes = "\n".join(sector_warnings)

    return executor_input


def _resolve_decision_qty(
    decision: ExecutorDecision,
    held: Decimal,
    price: Decimal,
    portfolio_value: Decimal,
    buying_power: Decimal,
) -> Decimal:
    """Resolve an intent to a concrete share count. Raises IntentError on failure."""
    if decision.action == "sell":
        intent = SellIntent(
            type=decision.intent_type,
            magnitude=(
                Decimal(str(decision.intent_magnitude))
                if decision.intent_magnitude is not None else None
            ),
        )
        return resolve_sell_intent(
            intent, held=held, price=price, portfolio_value=portfolio_value,
        )
    if decision.action == "buy":
        if decision.intent_magnitude is None:
            raise IntentError("buy intents require a magnitude")
        intent = BuyIntent(
            type=decision.intent_type,
            magnitude=Decimal(str(decision.intent_magnitude)),
        )
        return resolve_buy_intent(
            intent, held=held, price=price,
            portfolio_value=portfolio_value, buying_power=buying_power,
        )
    raise IntentError(f"unsupported action: {decision.action}")


def _client_order_id(decision: ExecutorDecision, session_date: date) -> str:
    """P1.6: deterministic broker-side idempotency key.

    Format: ``algo-YYYYMMDD-{b/s}-{TICKER}-{playbook_action_id|"op"}``. Same
    decision → same key, so an Alpaca submission retry after our DB write
    failed will be rejected by the broker as a duplicate (no double order).
    Alpaca's max length is 48 chars; the format above stays well under that
    for any plausible ticker (max ~6) + playbook id.
    """
    pba = decision.playbook_action_id if decision.playbook_action_id is not None else "op"
    return f"algo-{session_date.strftime('%Y%m%d')}-{decision.action[:1]}-{decision.ticker}-{pba}"


def _precheck_sell_against_alpaca(
    decision: ExecutorDecision, held: Decimal, errors: list[str],
) -> bool:
    """Return True if the sell should proceed; False if it was fully rejected.

    Returning False always represents a rejection worth counting toward
    trades_failed, so the caller should increment that counter on False.
    Trim-to-available is not a rejection and returns True after mutating
    decision.quantity.
    """
    def _reject(reason: str) -> bool:
        errors.append(f"{decision.ticker} pre-submit check failed: {reason}")
        logger.warning("%s: SKIP - %s", decision.ticker, reason)
        if decision.playbook_action_id:
            try:
                update_playbook_action_status(decision.playbook_action_id, "skipped")
            except Exception:
                pass
        decision.reasoning = f"[REJECTED: {reason}] {decision.reasoning}"
        decision.action = "invalid"
        return False

    try:
        available = get_live_available_qty(decision.ticker)
    except Exception as e:
        # Fail closed: skipping a legit sell is recoverable next session;
        # submitting on stale state during Alpaca degradation is not.
        return _reject(f"live availability check failed ({e})")

    if available is None or available >= decision.quantity:
        return True

    if available <= Decimal("0.0001"):
        return _reject(f"Alpaca reports 0 available shares (DB said {held})")

    logger.info(
        "%s: trimming sell from %s to %s (Alpaca available)",
        decision.ticker, decision.quantity, available,
    )
    decision.quantity = available
    return True


@dataclass
class _DecisionOutcome:
    executed: bool
    fill_price: Decimal | None
    order_id: str | None
    order_result: object | None
    trade_value: Decimal


def _execute_decision_order(
    decision: ExecutorDecision,
    price: Decimal,
    positions: dict,
    dry_run: bool,
    errors: list[str],
    session_date: date,
) -> _DecisionOutcome:
    """Submit the order, wait for fill, run post-fill side effects.

    Updates playbook_action_status on success/failure, handles thesis lifecycle
    on sell fills, and returns the outcome so the caller can update aggregate
    counters.

    T2.10: `session_date` is captured once at the top of the trading
    session and threaded through every callsite that needs "today's
    date". Without this, a session that crosses midnight ET could log
    decisions, dedup-check, and sign client_order_ids against three
    different dates within a single run.
    """
    logger.info("%s: %s %s @ ~$%.2f", decision.ticker, decision.action.upper(),
                decision.quantity, price)

    result = execute_market_order(
        ticker=decision.ticker,
        side=decision.action,
        qty=decision.quantity,
        dry_run=dry_run,
        simulated_price=price,
        client_order_id=_client_order_id(decision, session_date),
    )

    if not result.success:
        # P1.6 race-loser: a concurrent run already submitted this order with
        # the same deterministic client_order_id, so Alpaca rejected ours.
        # The DB-side dedup will keep the loser's decision row from being
        # written too. Log it as a benign skip rather than an error and don't
        # mark the playbook action as failed — the winner will mark it
        # executed.
        if getattr(result, "duplicate_client_order_id", False):
            logger.warning(
                "  %s: skipped — duplicate client_order_id rejected by broker "
                "(concurrent run already submitted): %s",
                decision.ticker, result.error,
            )
            return _DecisionOutcome(False, None, None, None, Decimal(0))
        errors.append(f"{decision.ticker} execution failed: {result.error}")
        logger.error("  %s: execution failed: %s", decision.ticker, result.error)
        if decision.playbook_action_id:
            try:
                update_playbook_action_status(decision.playbook_action_id, "failed")
            except Exception:
                pass
        return _DecisionOutcome(False, None, None, None, Decimal(0))

    # Wait for fill confirmation (skip for dry run — fills are instant)
    if not dry_run and result.order_id != "DRY_RUN":
        fill = wait_for_fill(result.order_id)
        if not fill.success:
            errors.append(f"{decision.ticker} fill failed: {fill.error}")
            logger.error("  %s: fill failed: %s", decision.ticker, fill.error)
            return _DecisionOutcome(False, None, None, None, Decimal(0))
        result = fill  # Use fill data

    if decision.playbook_action_id:
        try:
            update_playbook_action_status(decision.playbook_action_id, "executed")
        except Exception as e:
            logger.warning("Could not mark playbook action %d as executed: %s",
                           decision.playbook_action_id, e)

    fill_price = result.filled_avg_price if result.filled_avg_price else price
    trade_value = fill_price * decision.quantity

    status = "[DRY RUN]" if dry_run else f"Order {result.order_id} filled @ ${fill_price}"
    logger.info("  %s - Success", status)

    # Thesis lifecycle on fill: buys leave active, full sells close,
    # partial sells keep active.
    if decision.thesis_id and not dry_run and decision.action == "sell":
        try:
            held_before = positions.get(decision.ticker, Decimal(0))
            remaining = held_before - decision.quantity
            if remaining <= Decimal("0.0001"):
                close_thesis(
                    thesis_id=decision.thesis_id,
                    status="closed",
                    reason=f"Position exited: sold {decision.quantity} shares @ ${fill_price}",
                )
                logger.info("  Thesis %d closed (position exited)", decision.thesis_id)
            else:
                logger.info(
                    "  Thesis %d kept active (partial exit, %s shares remaining)",
                    decision.thesis_id, remaining,
                )
        except Exception as e:
            errors.append(f"Failed to update thesis {decision.thesis_id}: {e}")

    return _DecisionOutcome(True, fill_price, result.order_id, result, trade_value)


def _refresh_buying_power(
    decision: ExecutorDecision,
    buying_power: Decimal,
    portfolio_value: Decimal,
    trade_value: Decimal,
    dry_run: bool,
) -> tuple[Decimal, Decimal]:
    """After a fill, refresh buying power / portfolio value from Alpaca (real
    trades) or adjust locally (dry run).
    """
    if not dry_run:
        try:
            refreshed = get_account_info()
            return refreshed["buying_power"], refreshed["portfolio_value"]
        except Exception as e:
            logger.warning("Could not refresh buying power: %s — using local estimate", e)
            if decision.action == "buy":
                buying_power -= trade_value
            elif decision.action == "sell":
                buying_power += trade_value
            return buying_power, portfolio_value
    # Dry run: use local estimate
    if decision.action == "buy":
        buying_power -= trade_value
    elif decision.action == "sell":
        buying_power += trade_value
    return buying_power, portfolio_value


def _handle_thesis_invalidations(invalidations, errors: list[str]) -> None:
    if not invalidations:
        return
    logger.info("[Step 5b] Processing thesis invalidations")
    for inv in invalidations:
        try:
            close_thesis(thesis_id=inv.thesis_id, status="invalidated", reason=inv.reason)
            logger.info("Thesis %d: INVALIDATED - %s...", inv.thesis_id, inv.reason[:50])
        except Exception as e:
            errors.append(f"Failed to invalidate thesis {inv.thesis_id}: {e}")
            logger.error("Error invalidating thesis %d: %s", inv.thesis_id, e)


def _resolve_logged_qty(result, decision) -> Decimal | None:
    if result and result.filled_qty is not None:
        return Decimal(str(result.filled_qty))
    if decision.quantity:
        return decision.quantity
    return None


def _log_signal_links(decision_id: int, decision, errors: list[str]) -> None:
    if decision.signal_refs:
        try:
            validated_refs = validate_signal_refs(decision.signal_refs)
            if len(validated_refs) < len(decision.signal_refs):
                logger.warning(
                    "%s: stripped %d invalid signal refs",
                    decision.ticker,
                    len(decision.signal_refs) - len(validated_refs),
                )
            if validated_refs:
                signal_links = [
                    (decision_id, ref["type"], ref["id"])
                    for ref in validated_refs
                ]
                insert_decision_signals_batch(signal_links)
        except Exception as e:
            errors.append(f"Failed to log signal links for {decision.ticker}: {e}")
    elif decision.action in ("buy", "sell"):
        logger.warning(
            "%s: no signal_refs cited — decision will be excluded from attribution",
            decision.ticker,
        )


@dataclass
class _ExecutionTotals:
    trades_executed: int = 0
    trades_failed: int = 0
    total_buy_value: Decimal = Decimal(0)
    total_sell_value: Decimal = Decimal(0)


def _prepare_decision(
    decision: ExecutorDecision,
    positions: dict,
    data_client,
    dry_run: bool,
    portfolio_value: Decimal,
    buying_power: Decimal,
    totals: "_ExecutionTotals",
    errors: list[str],
    session_date: date,
    position_values: dict | None = None,
) -> Decimal | None:
    """Price-lookup, resolve intent to share count, and run sell precheck.

    Returns the price to use for execution, or None if the decision was
    rejected / skipped. Mutates the decision (reasoning, action, quantity)
    and totals.trades_failed in the rejection/skip paths.
    """
    price = get_latest_price(decision.ticker, client=data_client)
    if price is None:
        errors.append(f"Could not get price for {decision.ticker}")
        logger.error("%s: Could not get price", decision.ticker)
        totals.trades_failed += 1
        decision.reasoning = f"[REJECTED: no price available] {decision.reasoning}"
        decision.action = "invalid"
        return None

    # Resolve intent → concrete share count using live portfolio state.
    # The LLM does NOT author share counts; it authors intents (e.g.
    # exit_full, invest_dollar) and the system computes the exact Decimal
    # from positions/account. This structurally prevents oversells and
    # overspends regardless of LLM drift.
    held = positions.get(decision.ticker, Decimal(0))
    try:
        resolved_qty = _resolve_decision_qty(
            decision, held=held, price=price,
            portfolio_value=portfolio_value, buying_power=buying_power,
        )
    except IntentError as e:
        errors.append(f"{decision.ticker} intent error: {e}")
        logger.warning("%s: INVALID - intent error: %s", decision.ticker, e)
        totals.trades_failed += 1
        decision.reasoning = f"[REJECTED: intent error: {e}] {decision.reasoning}"
        decision.action = "invalid"
        return None

    if resolved_qty <= Decimal("0.0001"):
        logger.info(
            "%s: %s resolves to zero shares (held=%s, intent=%s, magnitude=%s) — skipping",
            decision.ticker, decision.action.upper(),
            held, decision.intent_type, decision.intent_magnitude,
        )
        decision.reasoning = (
            f"[SKIPPED: intent {decision.intent_type} resolved to 0 shares "
            f"against held={held}] {decision.reasoning}"
        )
        decision.action = "invalid"
        return None

    decision.quantity = resolved_qty

    # P3.30: hard sector-concentration gate for buys. Sells naturally reduce
    # exposure; this check only fires on buys. The same MAX_SECTOR_PCT is
    # injected as advisory text to the LLM via risk_notes — this gate is the
    # belt-and-suspenders backstop when the LLM ignores the warning.
    if decision.action == "buy" and position_values is not None:
        breach = check_sector_cap_for_buy(
            ticker=decision.ticker,
            new_qty=resolved_qty,
            price=price,
            position_values=position_values,
            portfolio_value=portfolio_value,
        )
        if breach:
            errors.append(f"{decision.ticker} sector cap: {breach}")
            logger.warning("%s: REJECTED (sector cap) - %s", decision.ticker, breach)
            decision.reasoning = f"[REJECTED: {breach}] {decision.reasoning}"
            decision.action = "invalid"
            totals.trades_failed += 1
            return None

    # P1.6: pre-submit dedup. The previous post-submit check left a window
    # where a network blip mid-`insert_decision` after a successful submit,
    # plus an operator rerun, would re-submit the order. The check here
    # short-circuits at the DB row that the prior run did manage to write
    # (or was about to). The broker-side `client_order_id` covers the
    # narrower case where the prior run's DB write itself failed.
    if not dry_run:
        existing_id = check_decision_exists(session_date, decision.ticker, decision.action)
        if existing_id:
            logger.warning(
                "%s: duplicate %s decision today (existing id=%d) — skipping submit",
                decision.ticker, decision.action, existing_id,
            )
            decision.reasoning = (
                f"[REJECTED: duplicate decision today (existing id={existing_id})] {decision.reasoning}"
            )
            decision.action = "invalid"
            totals.trades_failed += 1
            return None

    # Defense-in-depth: pre-submit live Alpaca check for sells. Catches the
    # edge case where Alpaca disagrees with the DB (untracked orders, partial
    # fills from a prior session, externally closed positions).
    if (
        decision.action == "sell"
        and not dry_run
        and not _precheck_sell_against_alpaca(decision, held, errors)
    ):
        totals.trades_failed += 1
        return None

    return price


def _execute_decisions(
    response,
    positions: dict,
    account_info: dict,
    data_client,
    dry_run: bool,
    errors: list[str],
    session_date: date,
) -> tuple[_ExecutionTotals, dict, dict, dict]:
    """Run the per-decision execution loop.

    Returns (totals, order_ids, order_results, decision_account_states).
    decision_account_states maps decision index → {portfolio_value, buying_power}
    captured at the moment the decision was evaluated, so each logged row
    reflects the live account state at trade time rather than the pre-session
    snapshot.
    """
    buying_power = account_info["buying_power"]
    portfolio_value = account_info["portfolio_value"]

    totals = _ExecutionTotals()
    max_trades_per_session = 10
    order_ids: dict = {}
    order_results: dict = {}
    decision_account_states: dict[int, dict] = {}

    # Pre-compute current position values for the sector-concentration gate.
    # One per-ticker price lookup; reused across every buy in the loop.
    position_values: dict = {}
    for ticker, shares in positions.items():
        try:
            price = get_latest_trade_price(ticker, client=data_client)
        except Exception:
            price = None
        if price:
            position_values[ticker] = shares * price

    for i, decision in enumerate(response.decisions):
        decision_account_states[i] = {
            "portfolio_value": portfolio_value,
            "buying_power": buying_power,
        }

        if decision.action == "hold":
            logger.info("%s: HOLD - %s...", decision.ticker, decision.reasoning[:50])
            continue

        if totals.trades_executed >= max_trades_per_session:
            logger.warning("Trade limit reached (%d trades). Skipping remaining decisions.",
                           max_trades_per_session)
            break

        price = _prepare_decision(
            decision, positions, data_client, dry_run,
            portfolio_value, buying_power, totals, errors,
            session_date,
            position_values=position_values,
        )
        if price is None:
            continue

        outcome = _execute_decision_order(
            decision, price, positions, dry_run, errors, session_date,
        )
        if not outcome.executed:
            totals.trades_failed += 1
            continue

        totals.trades_executed += 1
        order_ids[i] = outcome.order_id
        order_results[i] = outcome.order_result
        if decision.action == "buy":
            totals.total_buy_value += outcome.trade_value
        else:
            totals.total_sell_value += outcome.trade_value

        # T1.2: refresh sector-cap inputs mid-loop. Without this, multiple buys
        # in the same sector each compare against the pre-loop snapshot, so a
        # sequence that individually fits but cumulatively breaches MAX_SECTOR_PCT
        # all clears. Sells reduce exposure → subtract; clamp at 0 to defend
        # against precision drift.
        if decision.action == "buy":
            position_values[decision.ticker] = (
                position_values.get(decision.ticker, Decimal(0)) + outcome.trade_value
            )
        elif decision.action == "sell":
            new_val = position_values.get(decision.ticker, Decimal(0)) - outcome.trade_value
            position_values[decision.ticker] = new_val if new_val > 0 else Decimal(0)

        # Mirror the position_values refresh for the share-count dict.
        # Without this, a second sell of the same ticker reads the
        # pre-loop holding and the thesis-lifecycle close check at
        # _execute_decision_order computes the wrong `remaining`.
        held = positions.get(decision.ticker, Decimal(0))
        if decision.action == "buy":
            positions[decision.ticker] = held + decision.quantity
        elif decision.action == "sell":
            new_held = held - decision.quantity
            positions[decision.ticker] = new_held if new_held > Decimal(0) else Decimal(0)

        buying_power, portfolio_value = _refresh_buying_power(
            decision, buying_power, portfolio_value, outcome.trade_value, dry_run,
        )

    return totals, order_ids, order_results, decision_account_states


def _build_final_result(
    timestamp, snapshot_id, positions_synced, orders_synced,
    response, totals: "_ExecutionTotals", errors: list[str],
) -> TradingSessionResult:
    return TradingSessionResult(
        timestamp=timestamp,
        account_snapshot_id=snapshot_id,
        positions_synced=positions_synced,
        orders_synced=orders_synced,
        decisions_made=len(response.decisions),
        trades_executed=totals.trades_executed,
        trades_failed=totals.trades_failed,
        total_buy_value=totals.total_buy_value,
        total_sell_value=totals.total_sell_value,
        errors=errors,
        market_summary=response.market_summary,
        risk_assessment=response.risk_assessment,
    )


def _get_decisions(executor_input, model: str, errors: list[str]):
    """Call the executor LLM; return response or None on failure."""
    try:
        response = get_trading_decisions(executor_input, model=model)
        logger.info("Received %d decisions", len(response.decisions))
        logger.info("Market summary: %s...", response.market_summary[:100])
        return response
    except Exception as e:
        errors.append(f"LLM decision failed: {e}")
        logger.error("LLM decision failed: %s", e, exc_info=True)
        return None


def _log_session_summary(response, totals: _ExecutionTotals, errors: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("Trading Session Complete")
    logger.info("Decisions: %d | Executed: %d | Failed: %d",
                len(response.decisions), totals.trades_executed, totals.trades_failed)
    logger.info("Buy value: $%s | Sell value: $%s",
                f"{float(totals.total_buy_value):,.2f}",
                f"{float(totals.total_sell_value):,.2f}")
    if errors:
        logger.info("Errors: %d", len(errors))


_ORPHAN_DECISIONS_LOG = Path("logs/orphan_decisions.jsonl")
_INSERT_RETRY_ATTEMPTS = 3
_INSERT_RETRY_BASE_DELAY = 0.5  # seconds; doubles each attempt


def _persist_orphan_decision(
    *,
    decision: ExecutorDecision,
    order_id: str | None,
    order_result,
    payload: dict,
    last_error: Exception,
) -> None:
    """T1.5: write a JSONL row so an operator can manually reconcile.

    Reachable when an Alpaca order has filled but every retry of insert_decision
    failed. Without this, the system has a real position with no DB record —
    backfill / attribution / strategy reflection all run against an incomplete
    history. Decimals serialize as strings; payload may already contain Decimals.
    """
    def _enc(v):
        if isinstance(v, Decimal):
            return str(v)
        if isinstance(v, (date, datetime)):
            return v.isoformat()
        return str(v)

    record = {
        "logged_at": datetime.now().isoformat(),
        "ticker": decision.ticker,
        "action": decision.action,
        "order_id": order_id,
        "filled_qty": str(getattr(order_result, "filled_qty", None)),
        "filled_avg_price": str(getattr(order_result, "filled_avg_price", None)),
        "reasoning": decision.reasoning,
        "signal_refs": decision.signal_refs,
        "playbook_action_id": decision.playbook_action_id,
        "thesis_id": decision.thesis_id,
        "insert_payload": {k: _enc(v) for k, v in payload.items()},
        "last_error": f"{type(last_error).__name__}: {last_error}",
    }
    try:
        _ORPHAN_DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _ORPHAN_DECISIONS_LOG.open("a") as f:
            f.write(json.dumps(record, default=_enc) + "\n")
        logger.error(
            "Orphan decision persisted to %s for manual reconciliation: %s",
            _ORPHAN_DECISIONS_LOG, record,
        )
    except Exception as fs_err:
        # File-system write failed — last-ditch full-payload log so an operator
        # can recover from journald/stderr. Do not raise; the trading loop
        # should continue rather than crash on an unrelated I/O issue.
        logger.error(
            "Orphan-log write failed (%s); inline payload: %s", fs_err, record,
        )


def _insert_decision_with_retry(
    *,
    decision: ExecutorDecision,
    order_id: str | None,
    order_result,
    payload: dict,
) -> int | None:
    """T1.5: bounded retry around insert_decision with JSONL fallback.

    A filled order with a failed DB write produces an orphan position. Three
    attempts (0.5s, 1s) cover transient pool/network blips; final failure
    appends to logs/orphan_decisions.jsonl ONLY when the order actually filled
    (Alpaca confirmed shares moved). No-op decisions and pre-fill failures
    don't produce orphans.
    """
    last_exc: Exception | None = None
    for attempt in range(_INSERT_RETRY_ATTEMPTS):
        try:
            return insert_decision(**payload)
        except Exception as e:
            last_exc = e
            if attempt < _INSERT_RETRY_ATTEMPTS - 1:
                delay = _INSERT_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "insert_decision attempt %d/%d failed for %s: %s; retrying in %.1fs",
                    attempt + 1, _INSERT_RETRY_ATTEMPTS, decision.ticker, e, delay,
                )
                time.sleep(delay)

    assert last_exc is not None
    filled_qty = getattr(order_result, "filled_qty", None)
    unknown_partial_fill = bool(getattr(order_result, "unknown_partial_fill", False))
    is_orphan = bool(
        order_result
        and getattr(order_result, "success", False)
        and filled_qty is not None
        and Decimal(str(filled_qty)) > 0
    )
    # T2.8: when post-cancel re-fetch failed we don't know whether a
    # partial fill landed. Persist the orphan record so an operator can
    # reconcile against the broker — better a redundant log line than a
    # silent missed fill.
    if not is_orphan and unknown_partial_fill:
        is_orphan = True
    if is_orphan:
        _persist_orphan_decision(
            decision=decision, order_id=order_id, order_result=order_result,
            payload=payload, last_error=last_exc,
        )
    else:
        logger.error(
            "insert_decision failed after %d attempts for %s (%s) — no fill, no orphan: %s",
            _INSERT_RETRY_ATTEMPTS, decision.ticker, decision.action, last_exc,
        )
    return None


def _log_decisions(
    response,
    order_ids: dict,
    order_results: dict,
    data_client,
    account_info: dict,
    errors: list[str],
    session_date: date,
    decision_account_states: dict | None = None,
) -> int:
    """Insert decision rows and signal-links. Returns count of successfully logged decisions.

    Holds are logged so override reasoning (e.g., playbook said buy but executor
    held) is auditable. Backfill, attribution, and patterns filter action IN
    ('buy','sell') so unfillable holds don't pollute outcome metrics.
    """
    signals_used = format_decisions_for_logging(response)
    logged_count = 0
    for i, decision in enumerate(response.decisions):
        try:
            # Prefer filled price from order, fall back to latest quote.
            # Invalid (rejected) decisions may have no price available — that's
            # fine, the CHECK constraint only requires price for buy/sell.
            result = order_results.get(i)
            price = (
                result.filled_avg_price if result and result.filled_avg_price
                else get_latest_trade_price(decision.ticker, client=data_client)
            )
            if price is None and decision.action in ("buy", "sell"):
                errors.append(f"No price available for {decision.ticker} — skipping decision log")
                logger.error("Cannot log decision for %s: no price available", decision.ticker)
                continue

            # Skip duplicate decisions (same ticker+action already logged today).
            # Only meaningful for buy/sell/hold; rejected decisions carry
            # action='invalid' and dedup'ing them would collapse distinct
            # rejection audit rows for the same ticker.
            if decision.action in ("buy", "sell", "hold"):
                existing_id = check_decision_exists(session_date, decision.ticker, decision.action)
                if existing_id:
                    logger.warning(
                        "%s: duplicate %s decision — already logged as ID %d",
                        decision.ticker, decision.action, existing_id,
                    )
                    continue

            logged_qty = _resolve_logged_qty(result, decision)

            state = (decision_account_states or {}).get(i, account_info)
            payload = dict(
                decision_date=session_date,
                ticker=decision.ticker,
                action=decision.action,
                quantity=logged_qty,
                price=price,
                reasoning=decision.reasoning,
                signals_used=signals_used,
                account_equity=state["portfolio_value"],
                buying_power=state["buying_power"],
                playbook_action_id=decision.playbook_action_id,
                is_off_playbook=decision.is_off_playbook,
                order_id=order_ids.get(i),
            )
            decision_id = _insert_decision_with_retry(
                decision=decision,
                order_id=order_ids.get(i),
                order_result=result,
                payload=payload,
            )
            if decision_id is None:
                errors.append(
                    f"Failed to log decision for {decision.ticker} after retries"
                )
                continue
            logged_count += 1
        except Exception as e:
            errors.append(f"Failed to log decision for {decision.ticker}: {e}")
            logger.error("Error logging %s: %s", decision.ticker, e)
            continue

        _log_signal_links(decision_id, decision, errors)
    return logged_count


def run_trading_session(
    dry_run: bool = False,
    model: str = DEFAULT_EXECUTOR_MODEL,
) -> TradingSessionResult:
    """
    Run a complete trading session.

    1. Sync positions from Alpaca
    2. Take account snapshot
    3. Build executor input (structured)
    4. Get decisions from Claude Haiku
    5. Validate and execute trades
    6. Log decisions to database

    Args:
        dry_run: If True, don't execute real trades
        model: Claude model to use for decisions

    Returns:
        TradingSessionResult with session details
    """
    errors: list[str] = []
    timestamp = datetime.now()
    # T2.10: capture session date once, in market-local time. Every later
    # callsite (client_order_id signing, dedup checks, decision row
    # insertion) reuses this value so a session that runs across midnight
    # ET still shows up as a single trading day.
    session_date = datetime.now(ZoneInfo("America/New_York")).date()

    logger.info(
        "Starting trading session (dry_run=%s, model=%s, session_date=%s)",
        dry_run, model, session_date,
    )

    # Step 1: Sync positions and orders
    logger.info("[Step 1] Syncing positions and orders from Alpaca")
    positions_synced, orders_synced = _sync_from_alpaca(errors)

    # Market hours gate — skip trading if market is closed (dry_run bypasses)
    if not dry_run and not is_market_open():
        logger.warning("Market is closed. Skipping trading session (use --dry-run to bypass)")
        errors.append("Market is closed — skipped trading")
        return _empty_result(timestamp, positions_synced, orders_synced, 0, errors)

    data_client = StockHistoricalDataClient(
        os.environ.get("ALPACA_API_KEY"),
        os.environ.get("ALPACA_SECRET_KEY"),
    )

    # Step 2: Take account snapshot
    logger.info("[Step 2] Taking account snapshot")
    account_info, snapshot_id = _snapshot_account(errors)
    if account_info is None:
        return _empty_result(timestamp, positions_synced, orders_synced, 0, errors)

    # Step 3: Build executor input (structured, not string)
    logger.info("[Step 3] Building executor input")
    executor_input = _build_executor_context(account_info, data_client, errors)

    # Step 4: Get LLM decisions
    logger.info("[Step 4] Getting trading decisions from executor")
    response = _get_decisions(executor_input, model, errors)
    if response is None:
        return _empty_result(timestamp, positions_synced, orders_synced, snapshot_id, errors)

    # Step 5: Validate and execute trades
    logger.info("[Step 5] Executing trades")
    positions = {p["ticker"]: p["shares"] for p in get_positions()}
    totals, order_ids, order_results, decision_account_states = _execute_decisions(
        response, positions, account_info, data_client, dry_run, errors,
        session_date,
    )

    # Step 5b: Process thesis invalidations
    _handle_thesis_invalidations(response.thesis_invalidations, errors)

    # Step 6: Log decisions
    logger.info("[Step 6] Logging decisions")
    logged_count = _log_decisions(
        response, order_ids, order_results, data_client, account_info, errors,
        session_date,
        decision_account_states=decision_account_states,
    )
    logger.info("Logged %d decisions (%d emitted by executor)", logged_count, len(response.decisions))

    _log_session_summary(response, totals, errors)
    return _build_final_result(
        timestamp, snapshot_id, positions_synced, orders_synced,
        response, totals, errors,
    )


def main():
    """CLI entry point for trading agent."""
    import argparse

    from .log_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run trading agent session")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute real trades")
    parser.add_argument("--model", default=DEFAULT_EXECUTOR_MODEL, help="Claude model to use")

    args = parser.parse_args()

    result = run_trading_session(
        dry_run=args.dry_run,
        model=args.model,
    )

    if result.errors:
        logger.error("Errors encountered:")
        for error in result.errors:
            logger.error("  - %s", error)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
