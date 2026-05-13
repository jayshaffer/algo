"""Outcome backfill job - fills in 7d and 30d P&L for past decisions."""

import os
from datetime import date, datetime, timedelta
from decimal import Decimal

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .database.connection import get_cursor
from .market_calendar import NYSE_HOLIDAYS  # noqa: F401  alias preserves internal call sites
from .market_calendar import is_trading_day as _is_trading_day


def trading_day_offset(start: date, trading_days: int) -> date:
    """Advance start date by N trading days (skipping weekends + NYSE holidays).

    Returns the date that is `trading_days` market sessions after `start`.
    If `trading_days=0`, returns the next trading day on or after `start`
    (so a Saturday/holiday input advances to the next session).
    """
    current = start
    days_counted = 0
    while days_counted < trading_days:
        current += timedelta(days=1)
        if _is_trading_day(current):
            days_counted += 1
    while not _is_trading_day(current):
        current += timedelta(days=1)
    return current


def trading_day_cutoff(today: date, trading_days: int) -> date:
    """Return the date `trading_days` trading days before `today`.

    The eligibility filter for backfill compares a decision's date against
    this cutoff: `date <= cutoff` means the N-trading-day outcome window
    has closed. Using calendar days (`today - timedelta(days=N)`) lets a
    Friday decision become "eligible" on the next Friday — only 5 trading
    days have passed for a 7d window, so the price hasn't actually moved
    7 sessions yet. NYSE holidays are also skipped so a window crossing
    a holiday doesn't open eligibility a day early.
    """
    current = today
    days_counted = 0
    while days_counted < trading_days:
        current -= timedelta(days=1)
        if _is_trading_day(current):
            days_counted += 1
    return current


def get_data_client() -> StockHistoricalDataClient:
    """Create Alpaca data client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    return StockHistoricalDataClient(api_key, secret_key)


def get_price_on_date(client: StockHistoricalDataClient, ticker: str, target_date: date) -> Decimal | None:
    """Get closing price for a ticker on a specific date."""
    start = datetime.combine(target_date, datetime.min.time())
    end = datetime.combine(target_date + timedelta(days=5), datetime.min.time())

    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=5,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request)

        if ticker in bars.data and bars[ticker]:
            for bar in bars[ticker]:
                bar_date = bar.timestamp.date()
                if bar_date >= target_date:
                    return Decimal(str(bar.close))

        return None
    except Exception as e:
        print(f"  Error fetching price for {ticker} on {target_date}: {e}")
        return None


def get_decisions_needing_backfill(days_threshold: int) -> list:
    """Get decisions that need outcome backfill."""
    outcome_col = f"outcome_{days_threshold}d"
    cutoff_date = trading_day_cutoff(date.today(), days_threshold)

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
    """Calculate P&L percentage for a decision."""
    if entry_price <= 0 or exit_price <= 0:
        return Decimal(0)

    price_change_pct = ((exit_price - entry_price) / entry_price) * 100

    if action == "buy":
        return price_change_pct
    else:
        return -price_change_pct


BENCHMARK_TICKER = "SPY"


def update_outcome(decision_id: int, days: int, outcome: Decimal, benchmark: Decimal = None):
    """Update the outcome and benchmark columns for a decision."""
    outcome_col = f"outcome_{days}d"
    benchmark_col = f"benchmark_{days}d"

    with get_cursor() as cur:
        cur.execute(f"""
            UPDATE decisions
            SET {outcome_col} = %s, {benchmark_col} = %s
            WHERE id = %s
        """, (outcome, benchmark, decision_id))


def backfill_outcomes(days: int = 7, dry_run: bool = False) -> dict:
    """Backfill outcomes for decisions that have reached the threshold."""
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

    # Pre-fetch SPY prices for benchmark computation. Two caches keyed by
    # date — entries (decision_date) and exits (exit_date) often overlap
    # across decisions, so each call should hit the cache for the second
    # ticker that shares the same date. T1.10: only cache successful
    # fetches; caching None on a transient failure poisons every later
    # decision sharing that date.
    spy_entry_prices: dict[date, Decimal] = {}
    spy_exit_prices: dict[date, Decimal] = {}

    def _spy_price(target_date: date, cache: dict[date, Decimal]) -> Decimal | None:
        if target_date in cache:
            return cache[target_date]
        price = get_price_on_date(client, BENCHMARK_TICKER, target_date)
        if price is not None:
            cache[target_date] = price
        return price

    for decision in decisions:
        decision_id = decision["id"]
        ticker = decision["ticker"]
        action = decision["action"]
        entry_price = Decimal(str(decision["price"]))
        decision_date = decision["date"]
        exit_date = trading_day_offset(decision_date, days)

        exit_price = get_price_on_date(client, ticker, exit_date)

        if exit_price is None:
            print(f"  [{decision_id}] {ticker}: No price data for {exit_date} — skipping")
            stats["skipped_no_price"] += 1
            continue

        outcome = calculate_outcome(action, entry_price, exit_price)

        # Compute SPY benchmark for the same window. For sells, negate the
        # benchmark too — outcome is sign-flipped for sells (rising stock after
        # a sell = negative outcome for the signal), so benchmark must flip
        # the same way for `alpha = outcome - benchmark` to measure whether
        # the sell beat the market. Without this, every sell during a bull
        # market gets a wrongly-negative alpha.
        benchmark = None
        spy_entry = _spy_price(decision_date, spy_entry_prices)
        spy_exit = _spy_price(exit_date, spy_exit_prices)
        if spy_entry and spy_exit and spy_entry > 0 and spy_exit > 0:
            benchmark = ((spy_exit - spy_entry) / spy_entry) * 100
            if action == "sell":
                benchmark = -benchmark

        alpha_str = f" alpha={outcome - benchmark:+.2f}%" if benchmark is not None else ""

        if dry_run:
            print(f"  [{decision_id}] {ticker} {action}: {outcome:+.2f}% (SPY {benchmark:+.2f}%{alpha_str}) [DRY RUN]" if benchmark is not None
                  else f"  [{decision_id}] {ticker} {action}: {outcome:+.2f}% (no benchmark) [DRY RUN]")
        else:
            try:
                update_outcome(decision_id, days, outcome, benchmark)
                print(f"  [{decision_id}] {ticker} {action}: {outcome:+.2f}%{alpha_str}")
                stats["outcomes_filled"] += 1
            except Exception as e:
                print(f"  [{decision_id}] Error updating: {e}")
                stats["errors"] += 1

    return stats


def backfill_decision_outcomes(decision_id: int) -> dict:
    """Re-run 7d and 30d outcome+benchmark backfill for a single decision.

    Used by the audit auto-fix path when a decision past the window still
    has NULL outcome/benchmark. Always queries Alpaca; skips writes only
    when no exit price is available.
    """
    stats = {"decision_id": decision_id, "windows_filled": []}

    with get_cursor() as cur:
        cur.execute(
            "SELECT id, date, ticker, action, price FROM decisions WHERE id=%s",
            (decision_id,),
        )
        row = cur.fetchone()
    if not row or row["action"] not in ("buy", "sell") or row["price"] is None:
        return stats

    ticker = row["ticker"]
    action = row["action"]
    entry_price = Decimal(str(row["price"]))
    decision_date = row["date"]

    client = get_data_client()
    spy_entry = get_price_on_date(client, BENCHMARK_TICKER, decision_date)

    for days in (7, 30):
        exit_date = trading_day_offset(decision_date, days)
        exit_price = get_price_on_date(client, ticker, exit_date)
        if exit_price is None:
            continue

        outcome = calculate_outcome(action, entry_price, exit_price)

        benchmark = None
        spy_exit = get_price_on_date(client, BENCHMARK_TICKER, exit_date)
        if spy_entry and spy_exit and spy_entry > 0 and spy_exit > 0:
            benchmark = ((spy_exit - spy_entry) / spy_entry) * 100
            if action == "sell":
                benchmark = -benchmark

        update_outcome(decision_id, days, outcome, benchmark)
        stats["windows_filled"].append(days)

    return stats


def run_backfill(dry_run: bool = False) -> dict:
    """Run full backfill for both 7d and 30d outcomes."""
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
    parser.add_argument(
        "--decision-id",
        type=int,
        help="Re-run 7d+30d backfill for a single decision (audit auto-fix path).",
    )

    args = parser.parse_args()

    if args.decision_id is not None:
        if args.dry_run:
            parser.error("--decision-id does not support --dry-run")
        if args.days is not None:
            parser.error("--decision-id always backfills both 7d and 30d; drop --days")
        stats = backfill_decision_outcomes(args.decision_id)
        filled = stats.get("windows_filled") or []
        if not filled:
            print(f"[{args.decision_id}] No windows filled (decision missing, not buy/sell, or no exit price).")
        else:
            print(f"[{args.decision_id}] Filled windows: {filled}")
        return

    if args.days:
        backfill_outcomes(days=args.days, dry_run=args.dry_run)
    else:
        run_backfill(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
