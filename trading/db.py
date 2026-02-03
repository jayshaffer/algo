"""Database client for signal storage and retrieval."""

import os
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor, Json


def get_connection():
    """Create database connection from environment."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL must be set")
    return psycopg2.connect(database_url)


@contextmanager
def get_cursor():
    """Context manager for database cursor."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --- News Signals ---

def insert_news_signal(
    ticker: str,
    headline: str,
    category: str,
    sentiment: str,
    confidence: str,
    published_at: datetime
) -> int:
    """Insert a ticker-specific news signal."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO news_signals (ticker, headline, category, sentiment, confidence, published_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (ticker, headline, category, sentiment, confidence, published_at))
        return cur.fetchone()["id"]


def get_news_signals(ticker: Optional[str] = None, days: int = 7) -> list:
    """Get recent news signals, optionally filtered by ticker."""
    with get_cursor() as cur:
        if ticker:
            cur.execute("""
                SELECT * FROM news_signals
                WHERE ticker = %s AND published_at > NOW() - INTERVAL '%s days'
                ORDER BY published_at DESC
            """, (ticker, days))
        else:
            cur.execute("""
                SELECT * FROM news_signals
                WHERE published_at > NOW() - INTERVAL '%s days'
                ORDER BY published_at DESC
            """, (days,))
        return cur.fetchall()


# --- Macro Signals ---

def insert_macro_signal(
    headline: str,
    category: str,
    affected_sectors: list[str],
    sentiment: str,
    published_at: datetime
) -> int:
    """Insert a macro/political news signal."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO macro_signals (headline, category, affected_sectors, sentiment, published_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (headline, category, affected_sectors, sentiment, published_at))
        return cur.fetchone()["id"]


def get_macro_signals(category: Optional[str] = None, days: int = 7) -> list:
    """Get recent macro signals, optionally filtered by category."""
    with get_cursor() as cur:
        if category:
            cur.execute("""
                SELECT * FROM macro_signals
                WHERE category = %s AND published_at > NOW() - INTERVAL '%s days'
                ORDER BY published_at DESC
            """, (category, days))
        else:
            cur.execute("""
                SELECT * FROM macro_signals
                WHERE published_at > NOW() - INTERVAL '%s days'
                ORDER BY published_at DESC
            """, (days,))
        return cur.fetchall()


# --- Account Snapshots ---

def insert_account_snapshot(
    snapshot_date: date,
    cash: Decimal,
    portfolio_value: Decimal,
    buying_power: Decimal,
    long_market_value: Optional[Decimal] = None,
    short_market_value: Optional[Decimal] = None
) -> int:
    """Insert or update daily account snapshot."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO account_snapshots (date, cash, portfolio_value, buying_power, long_market_value, short_market_value)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                cash = EXCLUDED.cash,
                portfolio_value = EXCLUDED.portfolio_value,
                buying_power = EXCLUDED.buying_power,
                long_market_value = EXCLUDED.long_market_value,
                short_market_value = EXCLUDED.short_market_value,
                created_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (snapshot_date, cash, portfolio_value, buying_power, long_market_value, short_market_value))
        return cur.fetchone()["id"]


def get_account_snapshots(days: int = 30) -> list:
    """Get recent account snapshots for equity curve."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM account_snapshots
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
        """, (days,))
        return cur.fetchall()


# --- Decisions ---

def insert_decision(
    decision_date: date,
    ticker: str,
    action: str,
    quantity: Optional[Decimal],
    price: Optional[Decimal],
    reasoning: str,
    signals_used: dict,
    account_equity: Decimal,
    buying_power: Decimal
) -> int:
    """Insert a trading decision."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO decisions (date, ticker, action, quantity, price, reasoning, signals_used, account_equity, buying_power)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (decision_date, ticker, action, quantity, price, reasoning, Json(signals_used), account_equity, buying_power))
        return cur.fetchone()["id"]


def get_recent_decisions(days: int = 30) -> list:
    """Get recent decisions with outcomes."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
        """, (days,))
        return cur.fetchall()


# --- Positions ---

def upsert_position(ticker: str, shares: Decimal, avg_cost: Decimal) -> int:
    """Insert or update a position."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO positions (ticker, shares, avg_cost, updated_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (ticker) DO UPDATE SET
                shares = EXCLUDED.shares,
                avg_cost = EXCLUDED.avg_cost,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (ticker, shares, avg_cost))
        return cur.fetchone()["id"]


def get_positions() -> list:
    """Get all current positions."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM positions ORDER BY ticker")
        return cur.fetchall()


def delete_position(ticker: str) -> bool:
    """Delete a position (when fully sold)."""
    with get_cursor() as cur:
        cur.execute("DELETE FROM positions WHERE ticker = %s", (ticker,))
        return cur.rowcount > 0


# --- Theses ---

def insert_thesis(
    ticker: str,
    direction: str,
    thesis: str,
    entry_trigger: Optional[str] = None,
    exit_trigger: Optional[str] = None,
    invalidation: Optional[str] = None,
    confidence: str = "medium",
    source: str = "ideation"
) -> int:
    """Insert a new trade thesis."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO theses (ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source))
        return cur.fetchone()["id"]


def get_active_theses(ticker: Optional[str] = None) -> list:
    """Get all active theses, optionally filtered by ticker."""
    with get_cursor() as cur:
        if ticker:
            cur.execute("""
                SELECT * FROM theses
                WHERE status = 'active' AND ticker = %s
                ORDER BY created_at DESC
            """, (ticker,))
        else:
            cur.execute("""
                SELECT * FROM theses
                WHERE status = 'active'
                ORDER BY created_at DESC
            """)
        return cur.fetchall()


def update_thesis(
    thesis_id: int,
    thesis: Optional[str] = None,
    entry_trigger: Optional[str] = None,
    exit_trigger: Optional[str] = None,
    invalidation: Optional[str] = None,
    confidence: Optional[str] = None
) -> bool:
    """Update an existing thesis."""
    updates = []
    params = []

    if thesis is not None:
        updates.append("thesis = %s")
        params.append(thesis)
    if entry_trigger is not None:
        updates.append("entry_trigger = %s")
        params.append(entry_trigger)
    if exit_trigger is not None:
        updates.append("exit_trigger = %s")
        params.append(exit_trigger)
    if invalidation is not None:
        updates.append("invalidation = %s")
        params.append(invalidation)
    if confidence is not None:
        updates.append("confidence = %s")
        params.append(confidence)

    if not updates:
        return False

    updates.append("updated_at = NOW()")
    params.append(thesis_id)

    with get_cursor() as cur:
        cur.execute(f"""
            UPDATE theses
            SET {', '.join(updates)}
            WHERE id = %s
        """, params)
        return cur.rowcount > 0


def close_thesis(thesis_id: int, status: str, reason: str) -> bool:
    """Close a thesis with a status and reason."""
    with get_cursor() as cur:
        cur.execute("""
            UPDATE theses
            SET status = %s, close_reason = %s, closed_at = NOW(), updated_at = NOW()
            WHERE id = %s
        """, (status, reason, thesis_id))
        return cur.rowcount > 0
