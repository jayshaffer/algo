"""Twitter integration -- Bikini Bottom Capital (v2 pipeline).

Generates and posts tweets about trading activity using Ollama
in the voice of Mr. Krabs.
"""

import logging
import os
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

from .database.connection import get_cursor
from .database.trading_db import insert_tweet

logger = logging.getLogger("twitter")


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

def gather_tweet_context(session_date: Optional[date] = None) -> str:
    """Build a plain-text summary of today's trading session for tweet generation.

    Queries decisions, positions, active theses, latest snapshot, and latest
    strategy memo for the given date (defaults to today).
    """
    if session_date is None:
        session_date = date.today()

    sections = []

    with get_cursor() as cur:
        # Decisions for today
        cur.execute(
            "SELECT ticker, action, quantity, price, reasoning FROM decisions WHERE date = %s ORDER BY id",
            (session_date,),
        )
        decisions = cur.fetchall()
        if decisions:
            lines = ["TODAY'S DECISIONS:"]
            for d in decisions:
                qty = d['quantity']
                if d.get('price'):
                    lines.append(f"  {d['action'].upper()} {qty} {d['ticker']} @ ${d['price']}: {d['reasoning']}")
                else:
                    lines.append(f"  {d['action'].upper()} {qty} {d['ticker']}: {d['reasoning']}")
            sections.append("\n".join(lines))

        # Current positions
        cur.execute("SELECT ticker, shares, avg_cost FROM positions ORDER BY ticker")
        positions = cur.fetchall()
        if positions:
            lines = ["CURRENT POSITIONS:"]
            for p in positions:
                lines.append(f"  {p['ticker']}: {p['shares']} shares @ ${p['avg_cost']}")
            sections.append("\n".join(lines))

        # Active theses
        cur.execute(
            "SELECT ticker, direction, thesis, confidence FROM theses WHERE status = 'active' ORDER BY created_at DESC LIMIT 10"
        )
        theses = cur.fetchall()
        if theses:
            lines = ["ACTIVE THESES:"]
            for t in theses:
                lines.append(f"  {t['ticker']} ({t['direction']}, {t['confidence']}): {t['thesis']}")
            sections.append("\n".join(lines))

        # Latest snapshot
        cur.execute(
            "SELECT portfolio_value, cash, buying_power FROM account_snapshots ORDER BY date DESC LIMIT 1"
        )
        snapshot = cur.fetchone()
        if snapshot:
            sections.append(
                f"ACCOUNT: portfolio=${snapshot['portfolio_value']}, "
                f"cash=${snapshot['cash']}, buying_power=${snapshot['buying_power']}"
            )

        # Latest strategy memo
        cur.execute(
            "SELECT content FROM strategy_memos ORDER BY created_at DESC LIMIT 1"
        )
        memo = cur.fetchone()
        if memo:
            sections.append(f"STRATEGY MEMO:\n  {memo['content']}")

    if not sections:
        return "No trading activity today."

    return "\n\n".join(sections)
