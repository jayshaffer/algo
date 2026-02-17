"""Dashboard data gathering for public GitHub Pages dashboard.

Queries the DB and structures data for JSON export.
"""

import json
import logging
from datetime import date, datetime
from decimal import Decimal

from .database.connection import get_cursor

logger = logging.getLogger("dashboard_publish")


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal, date, and datetime types."""

    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def gather_dashboard_data(session_date: date) -> dict:
    """Gather all dashboard data in a single DB connection.

    Returns dict with keys: summary, snapshots, positions, decisions, theses.
    Handles empty DB gracefully.
    """
    with get_cursor() as cur:
        # Snapshots: last 90 days, ordered ASC
        cur.execute(
            """
            SELECT date, portfolio_value, cash, buying_power
            FROM account_snapshots
            WHERE date > %s - INTERVAL '90 days'
            ORDER BY date ASC
            """,
            (session_date,),
        )
        snapshots = cur.fetchall()

        # Positions: all, ordered by ticker
        cur.execute(
            "SELECT ticker, shares, avg_cost, updated_at FROM positions ORDER BY ticker"
        )
        positions = cur.fetchall()

        # Decisions: last 30 days, ordered DESC
        cur.execute(
            """
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   outcome_7d, outcome_30d, order_id
            FROM decisions
            WHERE date > %s - INTERVAL '30 days'
            ORDER BY date DESC, id DESC
            """,
            (session_date,),
        )
        decisions = cur.fetchall()

        # Theses: active only, ordered DESC
        cur.execute(
            """
            SELECT id, ticker, direction, confidence, thesis,
                   entry_trigger, exit_trigger, created_at
            FROM theses
            WHERE status = 'active'
            ORDER BY created_at DESC
            """
        )
        theses = cur.fetchall()

        # Latest snapshot for summary
        cur.execute(
            """
            SELECT portfolio_value, cash, long_market_value
            FROM account_snapshots
            ORDER BY date DESC LIMIT 1
            """
        )
        latest = cur.fetchone()

        # First snapshot ever (for total return)
        cur.execute(
            """
            SELECT portfolio_value, date
            FROM account_snapshots
            ORDER BY date ASC LIMIT 1
            """
        )
        first = cur.fetchone()

        # Previous snapshot (for daily P&L)
        cur.execute(
            """
            SELECT portfolio_value
            FROM account_snapshots
            ORDER BY date DESC LIMIT 1 OFFSET 1
            """
        )
        previous = cur.fetchone()

    # Build summary
    summary = _build_summary(latest, first, previous, len(positions), session_date)

    return {
        "summary": summary,
        "snapshots": [dict(r) for r in snapshots],
        "positions": [dict(r) for r in positions],
        "decisions": [dict(r) for r in decisions],
        "theses": [dict(r) for r in theses],
    }


def _build_summary(latest, first, previous, positions_count, session_date):
    """Build summary dict from query results."""
    if not latest:
        return {
            "portfolio_value": 0,
            "cash": 0,
            "invested": 0,
            "positions_count": 0,
            "last_updated": session_date.isoformat(),
            "daily_pnl": 0,
            "daily_pnl_pct": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "inception_date": None,
        }

    portfolio_value = latest["portfolio_value"]
    cash = latest["cash"]
    long_market_value = latest.get("long_market_value") or (portfolio_value - cash)

    # Daily P&L
    daily_pnl = Decimal("0")
    daily_pnl_pct = Decimal("0")
    if previous and previous["portfolio_value"]:
        prev_value = previous["portfolio_value"]
        daily_pnl = portfolio_value - prev_value
        if prev_value != 0:
            daily_pnl_pct = (daily_pnl / prev_value) * 100

    # Total P&L
    total_pnl = Decimal("0")
    total_pnl_pct = Decimal("0")
    inception_date = None
    if first and first["portfolio_value"]:
        first_value = first["portfolio_value"]
        inception_date = first["date"]
        total_pnl = portfolio_value - first_value
        if first_value != 0:
            total_pnl_pct = (total_pnl / first_value) * 100

    return {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "invested": long_market_value,
        "positions_count": positions_count,
        "last_updated": session_date.isoformat(),
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "inception_date": inception_date,
    }
