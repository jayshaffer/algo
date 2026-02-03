"""Trading agent orchestrator - daily automation entry point."""

import sys
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal

from .context import build_trading_context
from .executor import (
    get_account_info,
    take_account_snapshot,
    sync_positions_from_alpaca,
    execute_market_order,
    get_latest_price,
    calculate_position_size,
)
from .agent import (
    get_trading_decisions,
    validate_decision,
    format_decisions_for_logging,
    AgentResponse,
    TradingDecision,
)
from .db import insert_decision, get_positions


@dataclass
class TradeResult:
    """Result of executing a trading decision."""
    decision: TradingDecision
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
    decisions_made: int
    trades_executed: int
    trades_failed: int
    total_buy_value: Decimal
    total_sell_value: Decimal
    errors: list[str]


def run_trading_session(
    dry_run: bool = False,
    model: str = "qwen2.5:14b"
) -> TradingSessionResult:
    """
    Run a complete trading session.

    1. Sync positions from Alpaca
    2. Take account snapshot
    3. Build trading context
    4. Get decisions from local LLM via Ollama
    5. Validate and execute trades
    6. Log decisions to database

    Args:
        dry_run: If True, don't execute real trades
        model: Ollama model to use for decisions

    Returns:
        TradingSessionResult with session details
    """
    errors = []
    timestamp = datetime.now()

    print(f"[{timestamp.isoformat()}] Starting trading session")
    print(f"  Dry run: {dry_run}")
    print(f"  Model: {model}")

    # Step 1: Sync positions
    print("\n[Step 1] Syncing positions from Alpaca...")
    try:
        positions_synced = sync_positions_from_alpaca()
        print(f"  Synced {positions_synced} positions")
    except Exception as e:
        errors.append(f"Position sync failed: {e}")
        print(f"  Error: {e}")
        positions_synced = 0

    # Step 2: Take account snapshot
    print("\n[Step 2] Taking account snapshot...")
    try:
        account_info = get_account_info()
        snapshot_id = take_account_snapshot()
        print(f"  Snapshot ID: {snapshot_id}")
        print(f"  Portfolio value: ${float(account_info['portfolio_value']):,.2f}")
        print(f"  Buying power: ${float(account_info['buying_power']):,.2f}")
    except Exception as e:
        errors.append(f"Account snapshot failed: {e}")
        print(f"  Error: {e}")
        return TradingSessionResult(
            timestamp=timestamp,
            account_snapshot_id=0,
            positions_synced=positions_synced,
            decisions_made=0,
            trades_executed=0,
            trades_failed=0,
            total_buy_value=Decimal(0),
            total_sell_value=Decimal(0),
            errors=errors,
        )

    # Step 3: Build trading context
    print("\n[Step 3] Building trading context...")
    try:
        context = build_trading_context(account_info)
        print(f"  Context built ({len(context)} chars)")
    except Exception as e:
        errors.append(f"Context build failed: {e}")
        print(f"  Error: {e}")
        context = f"Error building context: {e}"

    # Step 4: Get LLM decisions
    print("\n[Step 4] Getting trading decisions from Ollama...")
    try:
        response = get_trading_decisions(context, model=model)
        print(f"  Received {len(response.decisions)} decisions")
        print(f"  Market summary: {response.market_summary[:100]}...")
    except Exception as e:
        errors.append(f"LLM decision failed: {e}")
        print(f"  Error: {e}")
        return TradingSessionResult(
            timestamp=timestamp,
            account_snapshot_id=snapshot_id,
            positions_synced=positions_synced,
            decisions_made=0,
            trades_executed=0,
            trades_failed=0,
            total_buy_value=Decimal(0),
            total_sell_value=Decimal(0),
            errors=errors,
        )

    # Step 5: Validate and execute trades
    print("\n[Step 5] Executing trades...")
    positions = {p["ticker"]: p["shares"] for p in get_positions()}
    buying_power = account_info["buying_power"]

    trades_executed = 0
    trades_failed = 0
    total_buy_value = Decimal(0)
    total_sell_value = Decimal(0)

    for decision in response.decisions:
        if decision.action == "hold":
            print(f"  {decision.ticker}: HOLD - {decision.reasoning[:50]}...")
            continue

        # Get current price for validation
        price = get_latest_price(decision.ticker)
        if price is None:
            errors.append(f"Could not get price for {decision.ticker}")
            print(f"  {decision.ticker}: ERROR - Could not get price")
            trades_failed += 1
            continue

        # Validate decision
        is_valid, reason = validate_decision(
            decision, buying_power, price, positions
        )

        if not is_valid:
            errors.append(f"{decision.ticker} validation failed: {reason}")
            print(f"  {decision.ticker}: INVALID - {reason}")
            trades_failed += 1
            continue

        # Execute trade
        print(f"  {decision.ticker}: {decision.action.upper()} {decision.quantity} @ ~${price:.2f}")

        result = execute_market_order(
            ticker=decision.ticker,
            side=decision.action,
            qty=Decimal(decision.quantity),
            dry_run=dry_run,
        )

        if result.success:
            trades_executed += 1
            trade_value = price * decision.quantity

            if decision.action == "buy":
                total_buy_value += trade_value
                buying_power -= trade_value
            else:
                total_sell_value += trade_value

            status = "[DRY RUN]" if dry_run else f"Order {result.order_id}"
            print(f"    {status} - Success")
        else:
            trades_failed += 1
            errors.append(f"{decision.ticker} execution failed: {result.error}")
            print(f"    ERROR: {result.error}")

    # Step 6: Log decisions
    print("\n[Step 6] Logging decisions...")
    signals_used = format_decisions_for_logging(response)

    for decision in response.decisions:
        try:
            price = get_latest_price(decision.ticker)
            insert_decision(
                decision_date=date.today(),
                ticker=decision.ticker,
                action=decision.action,
                quantity=Decimal(decision.quantity) if decision.quantity else None,
                price=price,
                reasoning=decision.reasoning,
                signals_used=signals_used,
                account_equity=account_info["portfolio_value"],
                buying_power=account_info["buying_power"],
            )
        except Exception as e:
            errors.append(f"Failed to log decision for {decision.ticker}: {e}")
            print(f"  Error logging {decision.ticker}: {e}")

    print(f"  Logged {len(response.decisions)} decisions")

    # Summary
    print("\n" + "=" * 60)
    print("Trading Session Complete")
    print("=" * 60)
    print(f"  Decisions: {len(response.decisions)}")
    print(f"  Executed: {trades_executed}")
    print(f"  Failed: {trades_failed}")
    print(f"  Buy value: ${float(total_buy_value):,.2f}")
    print(f"  Sell value: ${float(total_sell_value):,.2f}")
    print(f"  Errors: {len(errors)}")

    return TradingSessionResult(
        timestamp=timestamp,
        account_snapshot_id=snapshot_id,
        positions_synced=positions_synced,
        decisions_made=len(response.decisions),
        trades_executed=trades_executed,
        trades_failed=trades_failed,
        total_buy_value=total_buy_value,
        total_sell_value=total_sell_value,
        errors=errors,
    )


def main():
    """CLI entry point for trading agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Run trading agent session")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute real trades")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model to use")

    args = parser.parse_args()

    result = run_trading_session(
        dry_run=args.dry_run,
        model=args.model,
    )

    if result.errors:
        print("\nErrors encountered:")
        for error in result.errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
