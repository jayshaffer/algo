"""Intent-based sizing resolver.

LLMs pick intents + magnitudes (dollars or percentages); the system resolves
to exact Decimal share counts against live portfolio state. This keeps share
arithmetic out of the LLM's hands — the one thing it reliably botches under
context pressure.

Pure functions: no DB, no API. Inputs are pre-fetched values; output is a
clamped, feasible share count.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

MAX_POSITION_PCT = Decimal("0.10")  # mirror v2/agent.py

SELL_INTENT_TYPES = frozenset({
    "exit_full",
    "exit_partial_pct",
    "trim_to_portfolio_pct",
    "exit_dollar",
})

BUY_INTENT_TYPES = frozenset({
    "invest_dollar",
    "invest_portfolio_pct",
    "invest_buying_power_pct",
    "add_to_target_pct",
})


class IntentError(ValueError):
    """Raised when an intent is malformed or its magnitude is out of range."""


@dataclass
class SellIntent:
    type: str
    magnitude: Decimal | None  # None for exit_full; pct or dollars otherwise


@dataclass
class BuyIntent:
    type: str
    magnitude: Decimal  # dollars or pct — required


def _require_pct(magnitude: Decimal | None, label: str) -> Decimal:
    if magnitude is None:
        raise IntentError(f"{label} requires magnitude (percentage)")
    if magnitude < 0 or magnitude > 100:
        raise IntentError(f"{label} magnitude must be between 0 and 100, got {magnitude}")
    return magnitude


def _require_dollar(magnitude: Decimal | None, label: str) -> Decimal:
    if magnitude is None or magnitude <= 0:
        raise IntentError(f"{label} requires positive dollar magnitude")
    return magnitude


def resolve_sell_intent(
    intent: SellIntent,
    held: Decimal,
    price: Decimal,
    portfolio_value: Decimal,
) -> Decimal:
    """Resolve a sell intent to shares. Always clamped to held shares."""
    if held <= 0:
        return Decimal("0")
    if price <= 0:
        raise IntentError("price must be positive")

    t = intent.type
    if t == "exit_full":
        return held

    if t == "exit_partial_pct":
        pct = _require_pct(intent.magnitude, "exit_partial_pct")
        shares = held * (pct / Decimal("100"))
        return min(shares, held)

    if t == "exit_dollar":
        dollars = _require_dollar(intent.magnitude, "exit_dollar")
        shares = dollars / price
        return min(shares, held)

    if t == "trim_to_portfolio_pct":
        pct = _require_pct(intent.magnitude, "trim_to_portfolio_pct")
        target_value = portfolio_value * (pct / Decimal("100"))
        current_value = held * price
        if current_value <= target_value:
            return Decimal("0")
        delta_value = current_value - target_value
        shares = delta_value / price
        return min(shares, held)

    raise IntentError(f"unknown sell intent: {t}")


def resolve_buy_intent(
    intent: BuyIntent,
    held: Decimal,
    price: Decimal,
    portfolio_value: Decimal,
    buying_power: Decimal,
) -> Decimal:
    """Resolve a buy intent to shares. Clamped to buying_power and MAX_POSITION_PCT."""
    if price <= 0:
        raise IntentError("price must be positive")
    if buying_power <= 0:
        return Decimal("0")

    t = intent.type
    if t == "invest_dollar":
        dollars = _require_dollar(intent.magnitude, "invest_dollar")
        desired = dollars
    elif t == "invest_portfolio_pct":
        pct = _require_pct(intent.magnitude, "invest_portfolio_pct")
        desired = portfolio_value * (pct / Decimal("100"))
    elif t == "invest_buying_power_pct":
        pct = _require_pct(intent.magnitude, "invest_buying_power_pct")
        desired = buying_power * (pct / Decimal("100"))
    elif t == "add_to_target_pct":
        pct = _require_pct(intent.magnitude, "add_to_target_pct")
        target_value = portfolio_value * (pct / Decimal("100"))
        current_value = held * price
        if current_value >= target_value:
            return Decimal("0")
        desired = target_value - current_value
    else:
        raise IntentError(f"unknown buy intent: {t}")

    # Clamp to buying_power
    desired = min(desired, buying_power)

    # Clamp to MAX_POSITION_PCT (account for existing holding)
    if portfolio_value > 0:
        max_total_value = portfolio_value * MAX_POSITION_PCT
        current_value = held * price
        headroom = max_total_value - current_value
        if headroom <= 0:
            return Decimal("0")
        desired = min(desired, headroom)

    if desired <= 0:
        return Decimal("0")
    return desired / price
