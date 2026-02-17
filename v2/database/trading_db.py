"""
Trading database CRUD operations.

This module provides all database operations for the trading system,
including positions, decisions, theses, signals, and playbooks.
"""

from psycopg2.extras import execute_values, Json

from .connection import get_cursor


# --- News Signals ---

def insert_news_signal(ticker, headline, category, sentiment, confidence, published_at) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO news_signals (ticker, headline, category, sentiment, confidence, published_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (ticker, headline, category, sentiment, confidence, published_at))
        return cur.fetchone()["id"]


def insert_news_signals_batch(signals: list[tuple]) -> int:
    if not signals:
        return 0
    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO news_signals (ticker, headline, category, sentiment, confidence, published_at)
            VALUES %s
        """, signals)
        return len(signals)


def get_news_signals(ticker=None, days=7) -> list:
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

def insert_macro_signal(headline, category, affected_sectors, sentiment, published_at) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO macro_signals (headline, category, affected_sectors, sentiment, published_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (headline, category, affected_sectors, sentiment, published_at))
        return cur.fetchone()["id"]


def insert_macro_signals_batch(signals: list[tuple]) -> int:
    if not signals:
        return 0
    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO macro_signals (headline, category, affected_sectors, sentiment, published_at)
            VALUES %s
        """, signals)
        return len(signals)


def get_macro_signals(category=None, days=7) -> list:
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

def insert_account_snapshot(snapshot_date, cash, portfolio_value, buying_power, long_market_value=None, short_market_value=None) -> int:
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


def get_account_snapshots(days=30) -> list:
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM account_snapshots
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
        """, (days,))
        return cur.fetchall()


# --- Decisions ---

def insert_decision(decision_date, ticker, action, quantity, price, reasoning, signals_used, account_equity, buying_power, playbook_action_id=None, is_off_playbook=False, order_id=None) -> int:
    """
    Insert a trading decision.

    V3 additions:
    - playbook_action_id: Links decision to a specific playbook action
    - is_off_playbook: Marks decisions made outside the playbook
    - order_id: Alpaca order ID for trade verification
    """
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO decisions (date, ticker, action, quantity, price, reasoning, signals_used, account_equity, buying_power, playbook_action_id, is_off_playbook, order_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (decision_date, ticker, action, quantity, price, reasoning, Json(signals_used), account_equity, buying_power, playbook_action_id, is_off_playbook, order_id))
        return cur.fetchone()["id"]


def get_recent_decisions(days=30) -> list:
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
        """, (days,))
        return cur.fetchall()


def get_decisions_needing_backfill_7d() -> list:
    """Get decisions needing 7-day outcome backfill."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM decisions
            WHERE outcome_7d IS NULL AND date <= CURRENT_DATE - INTERVAL '7 days'
              AND action IN ('buy', 'sell')
            ORDER BY date
        """)
        return cur.fetchall()


def get_decisions_needing_backfill_30d() -> list:
    """Get decisions needing 30-day outcome backfill."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM decisions
            WHERE outcome_30d IS NULL AND date <= CURRENT_DATE - INTERVAL '30 days'
              AND action IN ('buy', 'sell')
            ORDER BY date
        """)
        return cur.fetchall()


def update_decision_outcome(decision_id, outcome_7d=None, outcome_30d=None):
    """Update decision outcomes after 7 or 30 days."""
    updates = []
    params = []
    if outcome_7d is not None:
        updates.append("outcome_7d = %s")
        params.append(outcome_7d)
    if outcome_30d is not None:
        updates.append("outcome_30d = %s")
        params.append(outcome_30d)
    if not updates:
        return
    params.append(decision_id)
    with get_cursor() as cur:
        cur.execute(f"UPDATE decisions SET {', '.join(updates)} WHERE id = %s", params)


# --- Positions ---

def upsert_position(ticker, shares, avg_cost) -> int:
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
    with get_cursor() as cur:
        cur.execute("SELECT * FROM positions ORDER BY ticker")
        return cur.fetchall()


def delete_position(ticker) -> bool:
    with get_cursor() as cur:
        cur.execute("DELETE FROM positions WHERE ticker = %s", (ticker,))
        return cur.rowcount > 0


def delete_all_positions():
    """Delete all positions (used during sync)."""
    with get_cursor() as cur:
        cur.execute("DELETE FROM positions")


# --- Theses ---

def insert_thesis(ticker, direction, thesis, entry_trigger=None, exit_trigger=None, invalidation=None, confidence="medium", source="ideation") -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO theses (ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source))
        return cur.fetchone()["id"]


def get_active_theses(ticker=None) -> list:
    with get_cursor() as cur:
        if ticker:
            cur.execute("SELECT * FROM theses WHERE status = 'active' AND ticker = %s ORDER BY created_at DESC", (ticker,))
        else:
            cur.execute("SELECT * FROM theses WHERE status = 'active' ORDER BY created_at DESC")
        return cur.fetchall()


def get_thesis_by_id(thesis_id) -> dict | None:
    """Get a specific thesis by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM theses WHERE id = %s", (thesis_id,))
        return cur.fetchone()


def update_thesis(thesis_id, thesis=None, entry_trigger=None, exit_trigger=None, invalidation=None, confidence=None) -> bool:
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
        cur.execute(f"UPDATE theses SET {', '.join(updates)} WHERE id = %s", params)
        return cur.rowcount > 0


def close_thesis(thesis_id, status, reason) -> bool:
    with get_cursor() as cur:
        cur.execute("""
            UPDATE theses SET status = %s, close_reason = %s, closed_at = NOW(), updated_at = NOW()
            WHERE id = %s
        """, (status, reason, thesis_id))
        return cur.rowcount > 0


# --- Open Orders ---

def upsert_open_order(order_id, ticker, side, order_type, qty, filled_qty, limit_price, stop_price, status, submitted_at) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO open_orders (order_id, ticker, side, order_type, qty, filled_qty, limit_price, stop_price, status, submitted_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (order_id) DO UPDATE SET
                filled_qty = EXCLUDED.filled_qty,
                status = EXCLUDED.status,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (order_id, ticker, side, order_type, qty, filled_qty, limit_price, stop_price, status, submitted_at))
        return cur.fetchone()["id"]


def get_open_orders() -> list:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM open_orders ORDER BY submitted_at DESC")
        return cur.fetchall()


def delete_open_order(order_id) -> bool:
    with get_cursor() as cur:
        cur.execute("DELETE FROM open_orders WHERE order_id = %s", (order_id,))
        return cur.rowcount > 0


def delete_all_open_orders():
    """Delete all open orders (used during sync)."""
    with get_cursor() as cur:
        cur.execute("DELETE FROM open_orders")


def clear_closed_orders() -> int:
    with get_cursor() as cur:
        cur.execute("DELETE FROM open_orders WHERE status NOT IN ('new', 'accepted', 'pending_new', 'partially_filled')")
        return cur.rowcount


# --- Playbooks ---

def upsert_playbook(playbook_date, market_outlook, priority_actions, watch_list, risk_notes) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO playbooks (date, market_outlook, priority_actions, watch_list, risk_notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                market_outlook = EXCLUDED.market_outlook,
                priority_actions = EXCLUDED.priority_actions,
                watch_list = EXCLUDED.watch_list,
                risk_notes = EXCLUDED.risk_notes,
                created_at = NOW()
            RETURNING id
        """, (playbook_date, market_outlook, Json(priority_actions), watch_list, risk_notes))
        return cur.fetchone()["id"]


def get_playbook(playbook_date) -> dict | None:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM playbooks WHERE date = %s", (playbook_date,))
        return cur.fetchone()


# --- Playbook Actions (V3) ---

def insert_playbook_action(playbook_id, ticker, action, thesis_id, reasoning, confidence, max_quantity, priority) -> int:
    """Insert a playbook action for V3 architecture."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO playbook_actions (playbook_id, ticker, action, thesis_id, reasoning, confidence, max_quantity, priority)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (playbook_id, ticker, action, thesis_id, reasoning, confidence, max_quantity, priority))
        return cur.fetchone()["id"]


def get_playbook_actions(playbook_id) -> list:
    """Get all actions for a playbook, ordered by priority."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM playbook_actions WHERE playbook_id = %s ORDER BY priority", (playbook_id,))
        return cur.fetchall()


def delete_playbook_actions(playbook_id) -> int:
    """Delete all actions for a playbook, clearing decision references first."""
    with get_cursor() as cur:
        cur.execute(
            "UPDATE decisions SET playbook_action_id = NULL "
            "WHERE playbook_action_id IN (SELECT id FROM playbook_actions WHERE playbook_id = %s)",
            (playbook_id,),
        )
        cur.execute("DELETE FROM playbook_actions WHERE playbook_id = %s", (playbook_id,))
        return cur.rowcount


# --- Decision Signals ---

def insert_decision_signal(decision_id, signal_type, signal_id):
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO decision_signals (decision_id, signal_type, signal_id)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
        """, (decision_id, signal_type, signal_id))


def insert_decision_signals_batch(signals: list[tuple]) -> int:
    if not signals:
        return 0
    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO decision_signals (decision_id, signal_type, signal_id)
            VALUES %s ON CONFLICT DO NOTHING
        """, signals)
        return len(signals)


def get_decision_signals(decision_id) -> list:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM decision_signals WHERE decision_id = %s", (decision_id,))
        return cur.fetchall()


# --- Signal Attribution ---

def upsert_signal_attribution(category, sample_size, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d):
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO signal_attribution (category, sample_size, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (category) DO UPDATE SET
                sample_size = EXCLUDED.sample_size,
                avg_outcome_7d = EXCLUDED.avg_outcome_7d,
                avg_outcome_30d = EXCLUDED.avg_outcome_30d,
                win_rate_7d = EXCLUDED.win_rate_7d,
                win_rate_30d = EXCLUDED.win_rate_30d,
                updated_at = NOW()
        """, (category, sample_size, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d))


def get_signal_attribution() -> list:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM signal_attribution ORDER BY sample_size DESC")
        return cur.fetchall()


# --- Strategy State ---

def insert_strategy_state(identity_text, risk_posture, sector_biases, preferred_signals, avoided_signals, version) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy_state (identity_text, risk_posture, sector_biases, preferred_signals, avoided_signals, version)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (identity_text, risk_posture, Json(sector_biases), Json(preferred_signals), Json(avoided_signals), version))
        return cur.fetchone()["id"]


def get_current_strategy_state() -> dict | None:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM strategy_state WHERE is_current = TRUE LIMIT 1")
        return cur.fetchone()


def clear_current_strategy_state():
    with get_cursor() as cur:
        cur.execute("UPDATE strategy_state SET is_current = FALSE WHERE is_current = TRUE")


# --- Strategy Rules ---

def insert_strategy_rule(rule_text, category, direction, confidence, supporting_evidence=None) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy_rules (rule_text, category, direction, confidence, supporting_evidence)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (rule_text, category, direction, confidence, supporting_evidence))
        return cur.fetchone()["id"]


def get_active_strategy_rules() -> list:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM strategy_rules WHERE status = 'active' ORDER BY created_at DESC")
        return cur.fetchall()


def retire_strategy_rule(rule_id) -> bool:
    with get_cursor() as cur:
        cur.execute("""
            UPDATE strategy_rules SET status = 'retired', retired_at = NOW()
            WHERE id = %s AND status = 'active'
        """, (rule_id,))
        return cur.rowcount > 0


# --- Strategy Memos ---

def insert_strategy_memo(session_date, memo_type, content, strategy_state_id=None) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy_memos (session_date, memo_type, content, strategy_state_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (session_date, memo_type, content, strategy_state_id))
        return cur.fetchone()["id"]


def get_recent_strategy_memos(n=5) -> list:
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM strategy_memos
            ORDER BY created_at DESC
            LIMIT %s
        """, (n,))
        return cur.fetchall()


# --- Tweets ---

def insert_tweet(session_date, tweet_type, tweet_text, tweet_id=None, posted=False, error=None, platform="twitter") -> int:
    """Insert a tweet record and return its id."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (session_date, tweet_type, tweet_text, tweet_id, posted, error, platform)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id, posted, error, platform))
        return cur.fetchone()["id"]


def get_tweets_for_date(session_date) -> list:
    """Get all tweets for a given session date."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM tweets WHERE session_date = %s ORDER BY created_at",
            (session_date,),
        )
        return cur.fetchall()
