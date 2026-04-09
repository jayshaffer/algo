"""Formation mode detection and context generation.

Detects cold-start conditions (no completed trade cycles, orphan positions)
and generates context injections that encourage exploratory behavior until
the system has enough data to self-sustain.
"""

import logging
from datetime import date

from .database.trading_db import (
    get_recent_decisions,
    get_positions,
    get_active_theses,
)

logger = logging.getLogger(__name__)

FORMATION_TRADE_THRESHOLD = 5


def is_formation_mode() -> bool:
    """Check if the system is in formation mode.

    Formation mode is active when fewer than FORMATION_TRADE_THRESHOLD
    buy/sell decisions have completed 7-day outcome cycles.
    """
    decisions = get_recent_decisions(days=90)
    completed_cycles = sum(
        1 for d in decisions
        if d.get("outcome_7d") is not None and d["action"] in ("buy", "sell")
    )
    return completed_cycles < FORMATION_TRADE_THRESHOLD


def get_orphan_positions() -> list[dict]:
    """Find positions with no active thesis.

    Returns list of position dicts for tickers that have no
    corresponding active thesis.
    """
    positions = get_positions()
    if not positions:
        return []

    theses = get_active_theses()
    covered_tickers = {t["ticker"] for t in theses}

    return [p for p in positions if p["ticker"] not in covered_tickers]
