"""Portfolio-level risk checks."""
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
MAX_DAILY_LOSS_PCT = Decimal("0.03")  # 3% daily loss limit


def check_daily_loss_limit(
    current_value: Decimal,
    previous_value: Decimal | None,
    max_loss_pct: Decimal = MAX_DAILY_LOSS_PCT,
) -> str | None:
    """Check if daily loss exceeds threshold.

    Returns warning string if limit breached, None if OK.
    """
    if previous_value is None or previous_value <= 0:
        return None

    change_pct = (current_value - previous_value) / previous_value
    if change_pct < -max_loss_pct:
        return (
            f"CIRCUIT BREAKER: Daily loss {change_pct:.2%} exceeds "
            f"{-max_loss_pct:.1%} limit (${current_value:,.2f} vs "
            f"prior ${previous_value:,.2f}). Trading halted."
        )
    return None


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
