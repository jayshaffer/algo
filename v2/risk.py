"""Portfolio-level risk checks."""
import os
from datetime import date, timedelta
from decimal import Decimal

SECTOR_MAP = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "GOOG": "tech",
    "AMZN": "tech", "META": "tech", "NVDA": "tech", "TSM": "tech",
    "AVGO": "tech", "ORCL": "tech", "CRM": "tech", "AMD": "tech",
    "INTC": "tech", "ADBE": "tech", "CSCO": "tech", "QCOM": "tech",
    "JPM": "finance", "BAC": "finance", "WFC": "finance", "GS": "finance",
    "MS": "finance", "C": "finance", "BLK": "finance", "SCHW": "finance",
    "V": "finance", "MA": "finance", "AXP": "finance",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "EOG": "energy", "OXY": "energy",
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "LLY": "healthcare", "TMO": "healthcare",
    "WMT": "consumer", "PG": "consumer", "KO": "consumer", "PEP": "consumer",
    "COST": "consumer", "NKE": "consumer", "SBUX": "consumer", "MCD": "consumer",
    "CAT": "industrial", "DE": "industrial", "HON": "industrial",
    "UPS": "industrial", "BA": "industrial", "GE": "industrial",
    "LMT": "defense", "RTX": "defense", "NOC": "defense", "GD": "defense",
}

MAX_SECTOR_PCT = Decimal("0.40")
MAX_POSITION_PCT = Decimal("0.10")  # max single-ticker exposure as a fraction of portfolio

# Rule 43: churn-review gate. Six or more round-trip pairs (opposing
# buy/sell actions within 7 days of each other) on a single symbol in a
# 30-day window triggers a mandatory review before further trading.
CHURN_PAIR_THRESHOLD = 6
CHURN_WINDOW_DAYS = 30
CHURN_PAIR_WINDOW_DAYS = 7

# Daily-loss circuit breaker: halt all trading when account equity has
# dropped more than this percentage versus the previous close (Alpaca
# last_equity). Read at module import like the other ALGO_* knobs —
# container restart required after changing. Set <= 0 to disable.
DAILY_LOSS_LIMIT_PCT = Decimal(os.environ.get("ALGO_DAILY_LOSS_LIMIT_PCT", "3.0"))


def check_sector_concentration(
    position_values: dict[str, Decimal],
    portfolio_value: Decimal,
) -> list[str]:
    """Check for sector concentration risk.

    Args:
        position_values: Dict of ticker -> market value (shares * price)
        portfolio_value: Total portfolio value

    Returns:
        List of warning strings (empty if no issues).
    """
    if portfolio_value <= 0:
        return []

    sector_totals: dict[str, Decimal] = {}
    for ticker, value in position_values.items():
        sector = SECTOR_MAP.get(ticker, "other")
        sector_totals[sector] = sector_totals.get(sector, Decimal(0)) + value

    warnings = []
    for sector, total in sector_totals.items():
        pct = total / portfolio_value
        if pct > MAX_SECTOR_PCT:
            warnings.append(
                f"Sector '{sector}' concentration {pct:.0%} exceeds {MAX_SECTOR_PCT:.0%} limit "
                f"(${total:,.0f} of ${portfolio_value:,.0f})"
            )

    return warnings


def check_sector_cap_for_buy(
    ticker: str,
    new_qty: Decimal,
    price: Decimal,
    position_values: dict[str, Decimal],
    portfolio_value: Decimal,
    cap: Decimal = MAX_SECTOR_PCT,
) -> str | None:
    """Hard pre-submit gate for sector concentration on buy orders.

    Returns a breach message if buying `new_qty` of `ticker` at `price`
    would push the ticker's sector over `cap`; otherwise None.

    Companion to `check_sector_concentration`, which is advisory-only and
    fed to the LLM as a string warning. The advisory path is brittle —
    nothing prevents the executor LLM from ignoring it under context
    pressure. This gate enforces the same threshold structurally so a
    single LLM lapse can't run the book over the cap. Sells naturally
    reduce sector exposure, so the gate only fires on buys.
    """
    if portfolio_value <= 0:
        return None
    sector = SECTOR_MAP.get(ticker, "other")
    new_value = new_qty * price
    sector_total = sum(
        (v for t, v in position_values.items()
         if SECTOR_MAP.get(t, "other") == sector),
        Decimal(0),
    )
    projected = sector_total + new_value
    pct = projected / portfolio_value
    if pct > cap:
        return (
            f"Sector '{sector}' projected exposure {pct:.0%} would exceed "
            f"{cap:.0%} cap (${projected:,.0f} of ${portfolio_value:,.0f})"
        )
    return None


def check_daily_loss_limit(
    equity: Decimal | None,
    last_equity: Decimal | None,
    limit_pct: Decimal | None = None,
) -> str | None:
    """Kill switch: returns a breach message when equity has dropped more
    than `limit_pct`% versus the previous close, else None.

    Inputs come from Alpaca account info (`equity`, `last_equity`). Missing
    or non-positive last_equity skips the check — fail open here because the
    breaker is a backstop, not the primary control, and a transient data gap
    must not permanently halt the system. The caller halts the trading stage
    (and re-checks after each fill) when a message is returned.
    """
    if limit_pct is None:
        limit_pct = DAILY_LOSS_LIMIT_PCT
    if limit_pct <= 0:
        return None
    if equity is None or last_equity is None or last_equity <= 0:
        return None
    loss_pct = (equity - last_equity) / last_equity * Decimal(100)
    if loss_pct <= -limit_pct:
        return (
            f"Daily loss limit: equity ${equity:,.0f} is {loss_pct:.2f}% vs "
            f"previous close ${last_equity:,.0f} (limit -{limit_pct}%). "
            "Trading halted for the session."
        )
    return None


def count_round_trip_pairs(
    ticker: str,
    reference_date: date | None = None,
    window_days: int = CHURN_WINDOW_DAYS,
    pair_window_days: int = CHURN_PAIR_WINDOW_DAYS,
) -> int:
    """Count opposing buy/sell pairs on `ticker` within `pair_window_days`
    of each other over the last `window_days`.

    Algorithm: greedy match — sort the symbol's decisions ascending by
    date; for each entry, if an opposite-action decision sits within
    `pair_window_days`, mark both consumed and count one pair. A buy
    consumed against an earlier sell counts the same as the reverse.
    """
    from .database.connection import get_cursor

    if reference_date is None:
        reference_date = date.today()
    cutoff = reference_date - timedelta(days=window_days)

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, date, action
            FROM decisions
            WHERE ticker = %s
              AND action IN ('buy', 'sell')
              AND date > %s
              AND date <= %s
            ORDER BY date ASC, id ASC
            """,
            (ticker, cutoff, reference_date),
        )
        rows = list(cur.fetchall())

    consumed = set()
    pairs = 0
    for i, row in enumerate(rows):
        if row["id"] in consumed:
            continue
        for j in range(i + 1, len(rows)):
            other = rows[j]
            if other["id"] in consumed:
                continue
            if (other["date"] - row["date"]).days > pair_window_days:
                break
            if other["action"] != row["action"]:
                consumed.add(row["id"])
                consumed.add(other["id"])
                pairs += 1
                break
    return pairs


def check_churn_gate(
    ticker: str,
    reference_date: date | None = None,
) -> str | None:
    """Rule 43 gate: block trading if `ticker` has accumulated
    CHURN_PAIR_THRESHOLD or more round-trip pairs in the last
    CHURN_WINDOW_DAYS days.

    Returns a breach message if the threshold is met, else None. The
    caller is responsible for converting the breach into a HOLD decision
    and emitting a `rule_gate:43` decision_signals row.
    """
    pairs = count_round_trip_pairs(ticker, reference_date=reference_date)
    if pairs >= CHURN_PAIR_THRESHOLD:
        return (
            f"Churn gate (Rule 43): {ticker} has {pairs} round-trip pairs "
            f"in the last {CHURN_WINDOW_DAYS}d "
            f"(threshold {CHURN_PAIR_THRESHOLD}). Churn review required "
            f"before further trading."
        )
    return None
