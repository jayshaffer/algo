"""Trading agent orchestrator -- daily automation entry point."""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal

from .context import build_executor_input
from .executor import (
    get_account_info,
    take_account_snapshot,
    sync_positions_from_alpaca,
    sync_orders_from_alpaca,
    execute_market_order,
    get_latest_price,
    calculate_position_size,
    is_market_open,
    wait_for_fill,
)
from .agent import (
    get_trading_decisions,
    validate_decision,
    validate_signal_refs,
    format_decisions_for_logging,
    AgentResponse,
    ExecutorDecision,
    DEFAULT_EXECUTOR_MODEL,
)
from .database.trading_db import insert_decision, check_decision_exists, get_positions, close_thesis, insert_decision_signals_batch, get_open_orders

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
    errors = []
    timestamp = datetime.now()

    logger.info("Starting trading session (dry_run=%s, model=%s)", dry_run, model)

    # Step 1: Sync positions and orders
    logger.info("[Step 1] Syncing positions and orders from Alpaca")
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

    # Market hours gate — skip trading if market is closed (dry_run bypasses)
    if not dry_run and not is_market_open():
        logger.warning("Market is closed. Skipping trading session (use --dry-run to bypass)")
        errors.append("Market is closed — skipped trading")
        return TradingSessionResult(
            timestamp=timestamp,
            account_snapshot_id=0,
            positions_synced=positions_synced,
            orders_synced=orders_synced,
            decisions_made=0,
            trades_executed=0,
            trades_failed=0,
            total_buy_value=Decimal(0),
            total_sell_value=Decimal(0),
            errors=errors,
        )

    # Create a shared data client for price lookups
    from alpaca.data.historical import StockHistoricalDataClient
    import os
    data_client = StockHistoricalDataClient(
        os.environ.get("ALPACA_API_KEY"),
        os.environ.get("ALPACA_SECRET_KEY"),
    )

    # Step 2: Take account snapshot
    logger.info("[Step 2] Taking account snapshot")
    try:
        account_info = get_account_info()
        snapshot_id = take_account_snapshot()
        logger.info("Snapshot ID: %d", snapshot_id)
        logger.info("Portfolio value: $%s  Buying power: $%s",
                     f"{float(account_info['portfolio_value']):,.2f}",
                     f"{float(account_info['buying_power']):,.2f}")
    except Exception as e:
        errors.append(f"Account snapshot failed: {e}")
        logger.error("Account snapshot failed: %s", e, exc_info=True)
        return TradingSessionResult(
            timestamp=timestamp,
            account_snapshot_id=0,
            positions_synced=positions_synced,
            orders_synced=orders_synced,
            decisions_made=0,
            trades_executed=0,
            trades_failed=0,
            total_buy_value=Decimal(0),
            total_sell_value=Decimal(0),
            errors=errors,
        )

    # Step 3: Build executor input (structured, not string)
    logger.info("[Step 3] Building executor input")
    try:
        executor_input = build_executor_input(account_info)
        logger.info("Executor input built")
    except Exception as e:
        errors.append(f"Context build failed: {e}")
        logger.error("Context build failed: %s", e, exc_info=True)
        # Build a minimal fallback input
        from .agent import ExecutorInput
        executor_input = ExecutorInput(
            playbook_actions=[],
            positions=[],
            account=account_info,
            attribution_summary={},
            recent_outcomes=[],
            market_outlook=f"Error building context: {e}",
            risk_notes="",
        )

    # Check sector concentration and inject warnings
    from .risk import check_sector_concentration
    position_values = {}
    for p in get_positions():
        price = get_latest_price(p["ticker"], client=data_client)
        if price:
            position_values[p["ticker"]] = p["shares"] * price
    sector_warnings = check_sector_concentration(position_values, account_info["portfolio_value"])
    if sector_warnings:
        logger.warning("Sector concentration warnings: %s", sector_warnings)
        if executor_input.risk_notes:
            executor_input.risk_notes += "\n" + "\n".join(sector_warnings)
        else:
            executor_input.risk_notes = "\n".join(sector_warnings)

    # Step 4: Get LLM decisions
    logger.info("[Step 4] Getting trading decisions from executor")
    try:
        response = get_trading_decisions(executor_input, model=model)
        logger.info("Received %d decisions", len(response.decisions))
        logger.info("Market summary: %s...", response.market_summary[:100])
    except Exception as e:
        errors.append(f"LLM decision failed: {e}")
        logger.error("LLM decision failed: %s", e, exc_info=True)
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

    # Step 5: Validate and execute trades
    logger.info("[Step 5] Executing trades")
    positions = {p["ticker"]: p["shares"] for p in get_positions()}
    buying_power = account_info["buying_power"]
    portfolio_value = account_info["portfolio_value"]

    # Build pending sell orders map for validation
    open_orders_list = get_open_orders()
    open_sell_orders = {}
    for order in open_orders_list:
        if order["side"] == "sell" and order["status"] in ("new", "accepted", "partially_filled"):
            ticker = order["ticker"]
            remaining = order["qty"] - (order.get("filled_qty") or Decimal(0))
            open_sell_orders[ticker] = open_sell_orders.get(ticker, Decimal(0)) + remaining

    trades_executed = 0
    trades_failed = 0
    total_buy_value = Decimal(0)
    total_sell_value = Decimal(0)
    max_trades_per_session = 10

    order_ids = {}
    order_results = {}
    for i, decision in enumerate(response.decisions):
        if decision.action == "hold":
            logger.info("%s: HOLD - %s...", decision.ticker, decision.reasoning[:50])
            continue

        if trades_executed >= max_trades_per_session:
            logger.warning("Trade limit reached (%d trades). Skipping remaining decisions.", max_trades_per_session)
            break

        # Get current price for validation
        price = get_latest_price(decision.ticker, client=data_client)
        if price is None:
            errors.append(f"Could not get price for {decision.ticker}")
            logger.error("%s: Could not get price", decision.ticker)
            trades_failed += 1
            continue

        # Validate decision
        is_valid, reason = validate_decision(
            decision, buying_power, price, positions, portfolio_value,
            open_sell_orders=open_sell_orders,
        )

        if not is_valid:
            errors.append(f"{decision.ticker} validation failed: {reason}")
            logger.warning("%s: INVALID - %s", decision.ticker, reason)
            trades_failed += 1
            continue

        # Execute trade
        logger.info("%s: %s %.4g @ ~$%.2f", decision.ticker, decision.action.upper(), decision.quantity, price)

        result = execute_market_order(
            ticker=decision.ticker,
            side=decision.action,
            qty=Decimal(decision.quantity),
            dry_run=dry_run,
            simulated_price=price,
        )

        if result.success:
            # Wait for fill confirmation (skip for dry run — fills are instant)
            if not dry_run and result.order_id != "DRY_RUN":
                fill = wait_for_fill(result.order_id)
                if not fill.success:
                    trades_failed += 1
                    errors.append(f"{decision.ticker} fill failed: {fill.error}")
                    logger.error("  %s: fill failed: %s", decision.ticker, fill.error)
                    continue
                # Update result with fill data
                result = fill

            trades_executed += 1
            order_ids[i] = result.order_id
            order_results[i] = result

            if decision.playbook_action_id:
                try:
                    from .database.trading_db import update_playbook_action_status
                    update_playbook_action_status(decision.playbook_action_id, "executed")
                except Exception as e:
                    logger.warning("Could not mark playbook action %d as executed: %s",
                                   decision.playbook_action_id, e)

            # Use fill price if available, fall back to quote price
            fill_price = result.filled_avg_price if result.filled_avg_price else price
            trade_value = fill_price * Decimal(str(decision.quantity))

            if decision.action == "buy":
                total_buy_value += trade_value
            else:
                total_sell_value += trade_value

            # Refresh buying power from Alpaca after real trades
            if not dry_run:
                try:
                    refreshed = get_account_info()
                    buying_power = refreshed["buying_power"]
                    portfolio_value = refreshed["portfolio_value"]
                except Exception as e:
                    # Fall back to local estimate if refresh fails
                    logger.warning("Could not refresh buying power: %s — using local estimate", e)
                    if decision.action == "buy":
                        buying_power -= trade_value
            else:
                # Dry run: use local estimate
                if decision.action == "buy":
                    buying_power -= trade_value

            status = "[DRY RUN]" if dry_run else f"Order {result.order_id} filled @ ${fill_price}"
            logger.info("  %s - Success", status)

            # Mark thesis as executed if trade was based on one
            if decision.thesis_id and not dry_run:
                try:
                    close_thesis(
                        thesis_id=decision.thesis_id,
                        status="executed",
                        reason=f"Trade executed: {decision.action} {decision.quantity} shares @ ${fill_price}"
                    )
                    logger.info("  Thesis %d marked as executed", decision.thesis_id)
                except Exception as e:
                    errors.append(f"Failed to update thesis {decision.thesis_id}: {e}")
        else:
            trades_failed += 1
            errors.append(f"{decision.ticker} execution failed: {result.error}")
            logger.error("  %s: execution failed: %s", decision.ticker, result.error)
            if hasattr(decision, 'playbook_action_id') and decision.playbook_action_id:
                try:
                    from .database.trading_db import update_playbook_action_status
                    update_playbook_action_status(decision.playbook_action_id, "failed")
                except Exception:
                    pass

    # Step 5b: Process thesis invalidations
    if response.thesis_invalidations:
        logger.info("[Step 5b] Processing thesis invalidations")
        for inv in response.thesis_invalidations:
            try:
                close_thesis(
                    thesis_id=inv.thesis_id,
                    status="invalidated",
                    reason=inv.reason
                )
                logger.info("Thesis %d: INVALIDATED - %s...", inv.thesis_id, inv.reason[:50])
            except Exception as e:
                errors.append(f"Failed to invalidate thesis {inv.thesis_id}: {e}")
                logger.error("Error invalidating thesis %d: %s", inv.thesis_id, e)

    # Step 6: Log decisions
    logger.info("[Step 6] Logging decisions")
    signals_used = format_decisions_for_logging(response)

    for i, decision in enumerate(response.decisions):
        # Skip logging hold decisions — they pollute the database with unfillable outcomes
        if decision.action == "hold":
            continue

        try:
            # Prefer filled price from order, fall back to latest quote
            result = order_results.get(i)
            price = result.filled_avg_price if result and result.filled_avg_price else get_latest_price(decision.ticker, client=data_client)
            if price is None:
                errors.append(f"No price available for {decision.ticker} — skipping decision log")
                logger.error("Cannot log decision for %s: no price available", decision.ticker)
                continue

            # Skip duplicate decisions (same ticker+action already logged today)
            existing_id = check_decision_exists(date.today(), decision.ticker, decision.action)
            if existing_id:
                logger.warning("%s: duplicate %s decision — already logged as ID %d",
                               decision.ticker, decision.action, existing_id)
                continue

            # Use filled quantity when available (handles partial fills)
            if result and result.filled_qty is not None:
                logged_qty = Decimal(str(result.filled_qty))
            elif decision.quantity:
                logged_qty = Decimal(str(decision.quantity))
            else:
                logged_qty = None

            decision_id = insert_decision(
                decision_date=date.today(),
                ticker=decision.ticker,
                action=decision.action,
                quantity=logged_qty,
                price=price,
                reasoning=decision.reasoning,
                signals_used=signals_used,
                account_equity=account_info["portfolio_value"],
                buying_power=account_info["buying_power"],
                playbook_action_id=decision.playbook_action_id,
                is_off_playbook=decision.is_off_playbook,
                order_id=order_ids.get(i),
            )
        except Exception as e:
            errors.append(f"Failed to log decision for {decision.ticker}: {e}")
            logger.error("Error logging %s: %s", decision.ticker, e)
            continue

        # Log signal-decision links for attribution (validate first)
        if decision.signal_refs:
            try:
                validated_refs = validate_signal_refs(decision.signal_refs)
                if len(validated_refs) < len(decision.signal_refs):
                    logger.warning("%s: stripped %d invalid signal refs",
                                   decision.ticker,
                                   len(decision.signal_refs) - len(validated_refs))
                if validated_refs:
                    signal_links = [
                        (decision_id, ref["type"], ref["id"])
                        for ref in validated_refs
                    ]
                    insert_decision_signals_batch(signal_links)
            except Exception as e:
                errors.append(f"Failed to log signal links for {decision.ticker}: {e}")
        else:
            logger.warning("%s: no signal_refs cited — decision will be excluded from attribution",
                           decision.ticker)

    logger.info("Logged %d decisions", len(response.decisions))

    # Summary
    logger.info("=" * 60)
    logger.info("Trading Session Complete")
    logger.info("Decisions: %d | Executed: %d | Failed: %d",
                len(response.decisions), trades_executed, trades_failed)
    logger.info("Buy value: $%s | Sell value: $%s",
                f"{float(total_buy_value):,.2f}", f"{float(total_sell_value):,.2f}")
    if errors:
        logger.info("Errors: %d", len(errors))

    return TradingSessionResult(
        timestamp=timestamp,
        account_snapshot_id=snapshot_id,
        positions_synced=positions_synced,
        orders_synced=orders_synced,
        decisions_made=len(response.decisions),
        trades_executed=trades_executed,
        trades_failed=trades_failed,
        total_buy_value=total_buy_value,
        total_sell_value=total_sell_value,
        errors=errors,
        market_summary=response.market_summary,
        risk_assessment=response.risk_assessment,
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


if __name__ == "__main__":
    main()
