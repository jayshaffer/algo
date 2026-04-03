"""Cooldown enforcement — structural prevention of rapid action reversals.

Enforces rules that the LLM has repeatedly violated:
  Rule #20: HOLD → SELL: 3 business days
  Rule #3:  BUY  → SELL: 3 business days
  Rule #6:  SELL → BUY:  5 business days
  Rule #10: SELL (after 3+ BUYs) → BUY: 10 business days
"""

import logging
from datetime import date, timedelta
from dataclasses import dataclass

from .database.trading_db import get_recent_decisions

logger = logging.getLogger(__name__)


@dataclass
class CooldownViolation:
    ticker: str
    proposed_action: str
    blocking_action: str
    blocking_date: date
    cooldown_expires: date
    rule: str


def _add_business_days(start: date, days: int) -> date:
    """Add business days (Mon-Fri) to a date."""
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def get_playbook_dates(lookback_days: int = 15) -> set[date]:
    """Get dates that have a playbook (used to identify fallback decisions)."""
    from .database.connection import get_cursor
    cutoff = date.today() - timedelta(days=lookback_days)
    with get_cursor() as cur:
        cur.execute("SELECT DISTINCT date FROM playbooks WHERE date >= %s", (cutoff,))
        return {row["date"] for row in cur.fetchall()}


def get_ticker_cooldowns(lookback_days: int = 15) -> dict[str, CooldownViolation]:
    """Compute active cooldowns for all tickers based on recent decisions.

    Returns dict of ticker -> CooldownViolation for tickers currently in cooldown.
    Fallback HOLDs (from sessions without a playbook) are excluded.
    """
    decisions = get_recent_decisions(days=lookback_days)
    playbook_dates = get_playbook_dates(lookback_days)
    today = date.today()
    cooldowns: dict[str, CooldownViolation] = {}

    # Group decisions by ticker, most recent first (query returns DESC)
    by_ticker: dict[str, list] = {}
    for d in decisions:
        ticker = d["ticker"]
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(d)

    for ticker, ticker_decisions in by_ticker.items():
        if not ticker_decisions:
            continue

        latest = ticker_decisions[0]
        latest_action = latest["action"]
        latest_date = latest["date"]

        # Skip fallback HOLDs (no playbook existed for that date)
        if latest_action == "hold" and latest_date not in playbook_dates:
            continue

        # HOLD → SELL lockout: 3 business days (Rule #20)
        if latest_action == "hold":
            expires = _add_business_days(latest_date, 3)
            if today < expires:
                cooldowns[ticker] = CooldownViolation(
                    ticker=ticker, proposed_action="sell",
                    blocking_action="hold", blocking_date=latest_date,
                    cooldown_expires=expires,
                    rule="HOLD\u2192SELL 3-day lockout (Rule #20)",
                )

        # BUY → SELL lockout: 3 business days (Rule #3)
        elif latest_action == "buy":
            expires = _add_business_days(latest_date, 3)
            if today < expires:
                cooldowns[ticker] = CooldownViolation(
                    ticker=ticker, proposed_action="sell",
                    blocking_action="buy", blocking_date=latest_date,
                    cooldown_expires=expires,
                    rule="BUY\u2192SELL 3-day lockout (Rule #3)",
                )

        # SELL → BUY lockout: 5 or 10 business days (Rule #6 / #10)
        elif latest_action == "sell":
            # Check for accumulation streak (3+ consecutive BUYs before the SELL)
            buy_streak = 0
            for d in ticker_decisions[1:]:
                if d["action"] == "buy":
                    buy_streak += 1
                else:
                    break

            if buy_streak >= 3:
                expires = _add_business_days(latest_date, 10)
                rule = f"Post-accumulation SELL\u2192BUY 10-day lockout (Rule #10, {buy_streak} prior BUYs)"
            else:
                expires = _add_business_days(latest_date, 5)
                rule = "SELL\u2192BUY/HOLD 5-day lockout (Rule #6)"

            if today < expires:
                cooldowns[ticker] = CooldownViolation(
                    ticker=ticker, proposed_action="buy",
                    blocking_action="sell", blocking_date=latest_date,
                    cooldown_expires=expires, rule=rule,
                )

    return cooldowns


def check_cooldown(
    ticker: str, proposed_action: str, cooldowns: dict[str, CooldownViolation],
) -> tuple[bool, str]:
    """Check if a proposed action violates cooldown. Returns (is_blocked, reason)."""
    if ticker not in cooldowns:
        return False, ""

    violation = cooldowns[ticker]
    if proposed_action == violation.proposed_action:
        return True, (
            f"Cooldown: {violation.rule} \u2014 "
            f"{violation.blocking_action.upper()} on {violation.blocking_date}, "
            f"expires {violation.cooldown_expires}"
        )

    return False, ""


def format_cooldown_map(cooldowns: dict[str, CooldownViolation]) -> dict[str, str]:
    """Format cooldowns as human-readable strings for the executor."""
    return {
        ticker: f"{v.rule} (expires {v.cooldown_expires})"
        for ticker, v in cooldowns.items()
    }
