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


def build_formation_context() -> str:
    """Build context injection for strategist/reflection prompts.

    Returns a string to append to the system prompt. Empty string
    if neither formation mode nor orphan positions apply.
    """
    in_formation = is_formation_mode()
    orphans = get_orphan_positions()

    if not in_formation and not orphans:
        return ""

    parts = []

    if in_formation:
        parts.append(f"""## FORMATION MODE ACTIVE

This system has fewer than {FORMATION_TRADE_THRESHOLD} completed trade cycles. \
It cannot learn without trades. Your priority is to break the cold-start loop:

1. **Generate actionable playbook actions.** Every session MUST produce at least 2-3 \
concrete buy/sell actions in the playbook. Conservative inaction is the wrong move \
during formation — the system needs live trades to build its evidence base.
2. **Size positions small.** Use small position sizes (1-5 shares or $200-$500) to \
limit downside while generating learning data.
3. **Prefer high-conviction, well-researched plays.** Don't trade randomly — but don't \
let the absence of historical data paralyze you either. Use your market research to \
form theses and act on them.
4. **Formation mode exits automatically** once {FORMATION_TRADE_THRESHOLD} trades have \
completed their 7-day outcome measurement.""")

    if orphans:
        orphan_lines = []
        for p in orphans:
            orphan_lines.append(
                f"- {p['ticker']}: {p['shares']} shares @ ${float(p['avg_cost']):.2f} avg"
            )
        orphan_list = "\n".join(orphan_lines)

        if in_formation:
            parts.append(f"""
## ORPHAN POSITIONS (no thesis)

These positions exist in the portfolio but have no active thesis. The system \
cannot reason about them, set exit triggers, or learn from them.

{orphan_list}

Use the `adopt_thesis` tool to create theses for these positions. For each one, \
decide: is this a hold worth keeping (create a long thesis with exit/invalidation \
triggers), or should it be exited (create a playbook sell action)?""")
        else:
            parts.append(f"""
## ORPHAN POSITIONS (no thesis)

These positions have no active thesis. Use `adopt_thesis` to create theses so the \
system can reason about exit triggers and learn from outcomes.

{orphan_list}""")

    return "\n\n".join(parts)
