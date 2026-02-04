"""Outcome backfill job - fills in 7d and 30d P&L for past decisions."""

import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .db import get_cursor


def get_data_client() -> StockHistoricalDataClient:
    """Create Alpaca data client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    return StockHistoricalDataClient(api_key, secret_key)


def get_price_on_date(client: StockHistoricalDataClient, ticker: str, target_date: date) -> Decimal | None:
    """
    Get closing price for a ticker on a specific date.

    Returns None if no data available (weekend, holiday, etc.)
    """
    # Try to get bar for the target date, with a few days buffer for weekends
    start = datetime.combine(target_date, datetime.min.time())
    end = datetime.combine(target_date + timedelta(days=5), datetime.min.time())

    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=5,
        )
        bars = client.get_stock_bars(request)

        if ticker in bars and bars[ticker]:
            # Find the first bar on or after target date
            for bar in bars[ticker]:
                bar_date = bar.timestamp.date()
                if bar_date >= target_date:
                    return Decimal(str(bar.close))

        return None
    except Exception as e:
        print(f"  Error fetching price for {ticker} on {target_date}: {e}")
        return None


def get_decisions_needing_backfill(days_threshold: int) -> list:
    """
    Get decisions that need outcome backfill.

    Args:
        days_threshold: 7 or 30

    Returns:
        List of decisions needing backfill for that timeframe
    """
    outcome_col = f"outcome_{days_threshold}d"
    cutoff_date = date.today() - timedelta(days=days_threshold)

    with get_cursor() as cur:
        cur.execute(f"""
            SELECT id, date, ticker, action, price
            FROM decisions
            WHERE {outcome_col} IS NULL
              AND date <= %s
              AND action IN ('buy', 'sell')
              AND price IS NOT NULL
            ORDER BY date ASC
        """, (cutoff_date,))
        return cur.fetchall()


def calculate_outcome(
    action: str,
    entry_price: Decimal,
    exit_price: Decimal
) -> Decimal:
    """
    Calculate P&L percentage for a decision.

    For BUY: positive if price went up
    For SELL: positive if price went down (avoided loss)
    """
    if entry_price == 0:
        return Decimal(0)

    price_change_pct = ((exit_price - entry_price) / entry_price) * 100

    if action == "buy":
        # BUY benefits from price increase
        return price_change_pct
    else:
        # SELL benefits from price decrease (we avoided holding)
        return -price_change_pct


def update_outcome(decision_id: int, days: int, outcome: Decimal):
    """Update the outcome column for a decision."""
    outcome_col = f"outcome_{days}d"

    with get_cursor() as cur:
        cur.execute(f"""
            UPDATE decisions
            SET {outcome_col} = %s
            WHERE id = %s
        """, (outcome, decision_id))


def backfill_outcomes(days: int = 7, dry_run: bool = False) -> dict:
    """
    Backfill outcomes for decisions that have reached the threshold.

    Args:
        days: 7 or 30
        dry_run: If True, don't actually update database

    Returns:
        Stats dict with counts
    """
    stats = {
        "decisions_found": 0,
        "outcomes_filled": 0,
        "skipped_no_price": 0,
        "errors": 0,
    }

    print(f"Backfilling {days}-day outcomes...")

    decisions = get_decisions_needing_backfill(days)
    stats["decisions_found"] = len(decisions)
    print(f"  Found {len(decisions)} decisions needing {days}d backfill")

    if not decisions:
        return stats

    client = get_data_client()

    for decision in decisions:
        decision_id = decision["id"]
        ticker = decision["ticker"]
        action = decision["action"]
        entry_price = Decimal(str(decision["price"]))
        decision_date = decision["date"]
        exit_date = decision_date + timedelta(days=days)

        # Get price on exit date
        exit_price = get_price_on_date(client, ticker, exit_date)

        if exit_price is None:
            print(f"  [{decision_id}] {ticker}: No price data for {exit_date}")
            stats["skipped_no_price"] += 1
            continue

        # Calculate outcome
        outcome = calculate_outcome(action, entry_price, exit_price)

        if dry_run:
            print(f"  [{decision_id}] {ticker} {action}: ${entry_price} -> ${exit_price} = {outcome:+.2f}% [DRY RUN]")
        else:
            try:
                update_outcome(decision_id, days, outcome)
                print(f"  [{decision_id}] {ticker} {action}: ${entry_price} -> ${exit_price} = {outcome:+.2f}%")
                stats["outcomes_filled"] += 1
            except Exception as e:
                print(f"  [{decision_id}] Error updating: {e}")
                stats["errors"] += 1

    return stats


def run_backfill(dry_run: bool = False) -> dict:
    """
    Run full backfill for both 7d and 30d outcomes.

    Args:
        dry_run: If True, don't actually update database

    Returns:
        Combined stats dict
    """
    print(f"[{datetime.now().isoformat()}] Starting outcome backfill")
    print(f"  Dry run: {dry_run}")

    stats_7d = backfill_outcomes(days=7, dry_run=dry_run)
    print()
    stats_30d = backfill_outcomes(days=30, dry_run=dry_run)

    combined = {
        "7d": stats_7d,
        "30d": stats_30d,
        "total_filled": stats_7d["outcomes_filled"] + stats_30d["outcomes_filled"],
    }

    print("\n" + "=" * 50)
    print("Backfill Complete")
    print("=" * 50)
    print(f"  7d outcomes filled: {stats_7d['outcomes_filled']}")
    print(f"  30d outcomes filled: {stats_30d['outcomes_filled']}")
    print(f"  Total: {combined['total_filled']}")

    return combined


def main():
    """CLI entry point for backfill job."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill decision outcomes")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--days", type=int, choices=[7, 30], help="Only backfill specific timeframe")

    args = parser.parse_args()

    if args.days:
        backfill_outcomes(days=args.days, dry_run=args.dry_run)
    else:
        run_backfill(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
