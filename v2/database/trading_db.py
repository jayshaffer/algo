"""
Trading database CRUD operations.

This module provides all database operations for the trading system,
including positions, decisions, theses, signals, and playbooks.
"""

from psycopg2.extras import Json, execute_values

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
    """Batch-insert news_signals rows.

    Tuple shape: (ticker, headline, category, sentiment, confidence,
    published_at, alpaca_id?, summary?). Legacy 6/7-tuples are padded
    with None for missing trailing fields so callers updated in
    different commits don't break each other.
    """
    if not signals:
        return 0

    def _normalize(s):
        if len(s) == 6:
            return (*s, None, None)
        if len(s) == 7:
            return (*s, None)
        return s

    normalized = [_normalize(s) for s in signals]

    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO news_signals (ticker, headline, category, sentiment, confidence, published_at, alpaca_id, summary)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, normalized)
        # P2.21: report what was actually inserted, not the input count.
        return cur.rowcount if cur.rowcount is not None else 0


def get_news_signals(ticker=None, days=7) -> list:
    with get_cursor() as cur:
        if ticker:
            cur.execute("""
                SELECT * FROM news_signals
                WHERE ticker = %s AND published_at > NOW() - INTERVAL '1 day' * %s
                ORDER BY published_at DESC
            """, (ticker, days))
        else:
            cur.execute("""
                SELECT * FROM news_signals
                WHERE published_at > NOW() - INTERVAL '1 day' * %s
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
    """Batch-insert macro_signals rows.

    Tuple shape: (headline, category, affected_sectors, sentiment, published_at, alpaca_id?).
    """
    if not signals:
        return 0
    # P2.16: pad legacy 5-tuples with None for alpaca_id.
    normalized = [
        (*s, None) if len(s) == 5 else s
        for s in signals
    ]
    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO macro_signals (headline, category, affected_sectors, sentiment, published_at, alpaca_id)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, normalized)
        # P2.21: report what was actually inserted, not the input count.
        return cur.rowcount if cur.rowcount is not None else 0


def get_macro_signals(category=None, days=7) -> list:
    with get_cursor() as cur:
        if category:
            cur.execute("""
                SELECT * FROM macro_signals
                WHERE category = %s AND published_at > NOW() - INTERVAL '1 day' * %s
                ORDER BY published_at DESC
            """, (category, days))
        else:
            cur.execute("""
                SELECT * FROM macro_signals
                WHERE published_at > NOW() - INTERVAL '1 day' * %s
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
    """Return snapshots within the last `days` calendar days, inclusive
    of the boundary day. T2.13: `>` excluded the boundary day, which
    silently shrank every lookback window by one day. The semantic now
    matches `get_recent_decisions` and the rest of the codebase: a
    `days=30` window covers exactly 30 calendar days back through today.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM account_snapshots
            WHERE date >= CURRENT_DATE - INTERVAL '1 day' * %s
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


def check_decision_exists(decision_date, ticker: str, action: str) -> int | None:
    """Check if a decision already exists for this (date, ticker, action).

    Applies to any action including 'hold' so same-day duplicate rows are
    suppressed even when the trader runs multiple times per day. The DB-level
    unique index only covers buy/sell; this function covers the rest.

    Returns the existing decision ID if found, None otherwise.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id FROM decisions
            WHERE date = %s AND ticker = %s AND action = %s
            LIMIT 1
        """, (decision_date, ticker, action))
        row = cur.fetchone()
        return row["id"] if row else None


def get_recent_decisions(days=30) -> list:
    """Return decisions within the last `days` calendar days, inclusive
    of the boundary day. T2.13: `>` excluded a decision logged exactly
    `days` days ago — formation reads this with `days=7` and was
    silently missing decisions from the same day a week earlier.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM decisions
            WHERE date >= CURRENT_DATE - INTERVAL '1 day' * %s
            ORDER BY date DESC
        """, (days,))
        return cur.fetchall()


def select_postable_decisions_for_date(
    session_date,
    min_notional: float,
    limit: int,
) -> list[dict]:
    """Return today's non-hold decisions worth posting about.

    Joined with their underlying thesis via playbook_actions; off-playbook
    decisions return rows with NULL thesis fields. Ordered by absolute
    notional value descending so the top `limit` are the highest-impact
    trades for the day. Filtered to ABS(quantity*price) >= min_notional
    so micro-trades don't spam the social feed.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                d.id,
                d.date,
                d.ticker,
                d.action,
                d.quantity,
                d.price,
                d.reasoning,
                d.is_off_playbook,
                pa.thesis_id AS thesis_id,
                t.thesis     AS thesis_text,
                t.direction  AS thesis_direction
            FROM decisions d
            LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
            LEFT JOIN theses t            ON t.id  = pa.thesis_id
            WHERE d.date = %s
              AND d.action != 'hold'
              AND d.price IS NOT NULL
              AND d.quantity IS NOT NULL
              AND ABS(d.quantity * d.price) >= %s
            ORDER BY ABS(d.quantity * d.price) DESC
            LIMIT %s
        """, (session_date, min_notional, limit))
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
        cur.execute("""
            SELECT * FROM open_orders
            WHERE status IN ('new', 'accepted', 'pending_new', 'partially_filled')
            ORDER BY submitted_at DESC
        """)
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

def insert_playbook_action(playbook_id, ticker, action, thesis_id, reasoning,
                           confidence, intent_type, intent_magnitude, priority) -> int:
    """Insert a playbook action with intent-based sizing."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO playbook_actions
                (playbook_id, ticker, action, thesis_id, reasoning,
                 confidence, intent_type, intent_magnitude, priority)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (playbook_id, ticker, action, thesis_id, reasoning,
              confidence, intent_type, intent_magnitude, priority))
        return cur.fetchone()["id"]


def get_playbook_actions(playbook_id) -> list:
    """Get all actions for a playbook, ordered by priority."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM playbook_actions WHERE playbook_id = %s ORDER BY priority", (playbook_id,))
        return cur.fetchall()


def get_recent_playbooks_with_actions(n: int = 3) -> list[dict]:
    """Return the N most recent playbooks with their actions nested.

    Used by the strategist's reversal-justification flow: shows what was
    *planned* in recent sessions (not just what got executed), so the
    strategist can detect when today's plan reverses a recent one.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                p.id              AS pb_id,
                p.date            AS pb_date,
                p.market_outlook  AS pb_market_outlook,
                pa.id             AS action_id,
                pa.ticker         AS ticker,
                pa.action         AS action,
                pa.intent_type    AS intent_type,
                pa.intent_magnitude AS intent_magnitude,
                pa.reasoning      AS reasoning
            FROM playbooks p
            LEFT JOIN playbook_actions pa ON pa.playbook_id = p.id
            WHERE p.id IN (
                SELECT id FROM playbooks ORDER BY date DESC LIMIT %s
            )
            ORDER BY p.date DESC, pa.priority ASC NULLS LAST, pa.id ASC
        """, (n,))
        rows = cur.fetchall()

    by_pb: dict[int, dict] = {}
    order: list[int] = []
    for row in rows:
        pb_id = row["pb_id"]
        if pb_id not in by_pb:
            by_pb[pb_id] = {
                "pb_id": pb_id,
                "pb_date": row["pb_date"],
                "pb_market_outlook": row.get("pb_market_outlook"),
                "actions": [],
            }
            order.append(pb_id)
        if row.get("action_id") is not None:
            by_pb[pb_id]["actions"].append({
                "action_id": row["action_id"],
                "ticker": row["ticker"],
                "action": row["action"],
                "intent_type": row.get("intent_type"),
                "intent_magnitude": row.get("intent_magnitude"),
                "reasoning": row.get("reasoning") or "",
            })
    return [by_pb[pb_id] for pb_id in order]


def update_playbook_action_status(action_id: int, status: str):
    """Update the status of a playbook action (e.g. pending -> executed)."""
    with get_cursor() as cur:
        cur.execute("UPDATE playbook_actions SET status = %s WHERE id = %s", (status, action_id))


def get_pending_playbook_actions(playbook_id: int) -> list[dict]:
    """Get pending (not yet executed) actions for a playbook."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM playbook_actions
            WHERE playbook_id = %s AND (status = 'pending' OR status IS NULL)
            ORDER BY priority ASC
        """, (playbook_id,))
        return [dict(row) for row in cur.fetchall()]


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


def replace_playbook_actions_atomic(
    playbook_date, market_outlook, priority_actions, watch_list, risk_notes,
    actions: list[dict],
) -> tuple[int, int]:
    """P2.22: atomic upsert-playbook + clear-old-actions + insert-new-actions.

    The previous tool_write_playbook flow used three separate `get_cursor()`
    contexts (own connection, own transaction). Mid-loop failure left the
    playbook row + half the actions in DB with no rollback, so the executor
    traded on incomplete intent. Wrapping it in one transaction makes the
    operation truly all-or-nothing.

    Returns (playbook_id, action_count).
    """
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
        playbook_id = cur.fetchone()["id"]

        cur.execute(
            "UPDATE decisions SET playbook_action_id = NULL "
            "WHERE playbook_action_id IN (SELECT id FROM playbook_actions WHERE playbook_id = %s)",
            (playbook_id,),
        )
        cur.execute("DELETE FROM playbook_actions WHERE playbook_id = %s", (playbook_id,))

        from decimal import Decimal as _Decimal
        for i, action in enumerate(actions):
            intent_magnitude = action.get("intent_magnitude")
            cur.execute("""
                INSERT INTO playbook_actions
                    (playbook_id, ticker, action, thesis_id, reasoning,
                     confidence, intent_type, intent_magnitude, priority)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                playbook_id,
                action.get("ticker"),
                action.get("action"),
                action.get("thesis_id"),
                action.get("reasoning", ""),
                action.get("confidence", "medium"),
                action.get("intent_type"),
                _Decimal(str(intent_magnitude)) if intent_magnitude is not None else None,
                i + 1,
            ))
    return playbook_id, len(actions)


# --- Decision Signals ---

def insert_thesis_signals(thesis_id: int, signal_refs: list[dict]) -> int:
    """Persist the news/macro/thesis signals that back a thesis.

    signal_refs: [{"type": "news_signal"|"macro_signal"|"thesis", "id": int}, ...]
    Returns the number of refs submitted (writes are idempotent on PK).
    """
    if not signal_refs:
        return 0
    rows = [(thesis_id, ref["type"], ref["id"]) for ref in signal_refs]
    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO thesis_signals (thesis_id, signal_type, signal_id)
            VALUES %s ON CONFLICT DO NOTHING
        """, rows)
        # P2.26: return what was actually inserted, not the input count.
        # ON CONFLICT DO NOTHING means refs already cited on this thesis
        # are skipped — reporting len() lies to the caller (and to the
        # strategist via _persist_signal_refs's "Cited N signals" note).
        return cur.rowcount if cur.rowcount is not None else 0


def get_thesis_signals(thesis_id: int) -> list:
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM thesis_signals WHERE thesis_id = %s ORDER BY signal_type, signal_id",
            (thesis_id,),
        )
        return cur.fetchall()


def delete_thesis_signals(thesis_id: int) -> int:
    with get_cursor() as cur:
        cur.execute("DELETE FROM thesis_signals WHERE thesis_id = %s", (thesis_id,))
        return cur.rowcount


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
        # P2.26: same shape as P2.21 (news/macro batches). Return what was
        # actually persisted so a rerun of an already-logged decision
        # doesn't falsely claim N new links.
        return cur.rowcount if cur.rowcount is not None else 0


def get_decision_signals(decision_id) -> list:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM decision_signals WHERE decision_id = %s", (decision_id,))
        return cur.fetchall()


# --- Signal Attribution ---

def upsert_signal_attribution(category, sample_size, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d, sample_size_30d=None):
    """P2.17: sample_size_30d is the count of decisions with both 7d AND 30d
    outcomes. The 7d-only `sample_size` is preserved for backward compatibility.
    """
    if sample_size_30d is None:
        sample_size_30d = sample_size  # legacy fallback
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO signal_attribution (category, sample_size, sample_size_30d, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (category) DO UPDATE SET
                sample_size = EXCLUDED.sample_size,
                sample_size_30d = EXCLUDED.sample_size_30d,
                avg_outcome_7d = EXCLUDED.avg_outcome_7d,
                avg_outcome_30d = EXCLUDED.avg_outcome_30d,
                win_rate_7d = EXCLUDED.win_rate_7d,
                win_rate_30d = EXCLUDED.win_rate_30d,
                updated_at = NOW()
        """, (category, sample_size, sample_size_30d, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d))


def get_signal_attribution() -> list:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM signal_attribution ORDER BY sample_size DESC")
        return cur.fetchall()


# --- Session Tracking ---

def insert_session_record(session_date, session_type="daily") -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO sessions (session_date, session_type, status)
            VALUES (%s, %s, 'running')
            RETURNING id
        """, (session_date, session_type))
        return cur.fetchone()["id"]


def get_session_for_date(session_date, session_type="daily"):
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM sessions
            WHERE session_date = %s AND session_type = %s
            ORDER BY id DESC LIMIT 1
        """, (session_date, session_type))
        return cur.fetchone()


def complete_session(session_id):
    with get_cursor() as cur:
        cur.execute("""
            UPDATE sessions SET status = 'completed', completed_at = NOW()
            WHERE id = %s
        """, (session_id,))


def fail_session(session_id, error_text):
    with get_cursor() as cur:
        cur.execute("""
            UPDATE sessions SET status = 'failed', completed_at = NOW(), error = %s
            WHERE id = %s
        """, (error_text, session_id))


# --- Strategy State ---

def insert_strategy_state(identity_text, risk_posture, sector_biases, preferred_signals, avoided_signals, version) -> int:
    """Atomically clear the prior current state and insert the new one.

    The clear+insert pair shares one transaction so a mid-write failure
    leaves the prior is_current=TRUE row intact rather than wiping all
    state. Caller no longer needs to call clear_current_strategy_state()
    separately.
    """
    with get_cursor() as cur:
        cur.execute("UPDATE strategy_state SET is_current = FALSE WHERE is_current = TRUE")
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


def get_strategy_rule(rule_id: int) -> dict | None:
    """Fetch a single strategy rule by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM strategy_rules WHERE id = %s", (rule_id,))
        return cur.fetchone()


def retire_strategy_rule(rule_id, reason=None) -> bool:
    with get_cursor() as cur:
        cur.execute("""
            UPDATE strategy_rules
            SET status = 'retired', retired_at = NOW(), retirement_reason = %s
            WHERE id = %s AND status = 'active'
        """, (reason, rule_id))
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

def insert_tweet(
    session_date,
    tweet_type: str,
    tweet_text: str,
    tweet_id: str | None = None,
    posted: bool = False,
    error: str | None = None,
    platform: str = "twitter",
    decision_id: int | None = None,
) -> int:
    """Log a tweet/post to the audit table.

    decision_id (new) ties a per-trade post back to its source decision,
    enabling the (decision_id, platform) rerun guard used by the live-
    trade pipeline. Recap and entertainment posts leave it NULL.
    """
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (
                session_date, tweet_type, tweet_text, tweet_id,
                posted, error, platform, decision_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id,
              posted, error, platform, decision_id))
        return cur.fetchone()["id"]


def posted_tweet_for_decision_exists(decision_id: int, platform: str) -> bool:
    """True if a successful (posted=TRUE) tweet already exists for this
    decision on this platform. Used by run_trade_posts_stage to skip
    decisions that were posted on a prior session run."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT 1
            FROM tweets
            WHERE decision_id = %s
              AND platform = %s
              AND posted = TRUE
            LIMIT 1
        """, (decision_id, platform))
        return cur.fetchone() is not None


def posted_tweet_exists(session_date, tweet_type: str, platform: str) -> bool:
    """Return True if a posted=TRUE tweet already exists for this slot.

    Used by Twitter/Bluesky stages to short-circuit on rerun and avoid
    re-generating + re-posting an already-published tweet (P1.9).
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM tweets
            WHERE session_date = %s
              AND tweet_type = %s
              AND platform = %s
              AND posted = TRUE
            LIMIT 1
            """,
            (session_date, tweet_type, platform),
        )
        return cur.fetchone() is not None


def get_tweets_for_date(session_date) -> list:
    """Get all tweets for a given session date."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM tweets WHERE session_date = %s ORDER BY created_at",
            (session_date,),
        )
        return cur.fetchall()


# --- Session Stages ---

def insert_session_stage(session_id: int, stage_name: str) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO session_stages (session_id, stage_name, status)
            VALUES (%s, %s, 'running')
            ON CONFLICT (session_id, stage_name) DO UPDATE SET
                status = 'running', started_at = NOW(), completed_at = NULL, error = NULL
            RETURNING id
        """, (session_id, stage_name))
        return cur.fetchone()["id"]


def complete_session_stage(session_id: int, stage_name: str, usage=None):
    """Mark a session_stages row complete, optionally recording token usage.

    `usage` is a v2.claude_client.UsageAccumulator. When supplied with a
    non-None .model, the five token columns are written. When None or
    .model is None (stage made no Claude calls), the columns stay NULL.
    """
    with get_cursor() as cur:
        if usage is not None and usage.model is not None:
            cur.execute("""
                UPDATE session_stages
                SET status = 'completed',
                    completed_at = NOW(),
                    model = %s,
                    input_tokens = %s,
                    output_tokens = %s,
                    cache_creation_tokens = %s,
                    cache_read_tokens = %s
                WHERE session_id = %s AND stage_name = %s
            """, (
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_tokens,
                usage.cache_read_tokens,
                session_id, stage_name,
            ))
        else:
            cur.execute("""
                UPDATE session_stages SET status = 'completed', completed_at = NOW()
                WHERE session_id = %s AND stage_name = %s
            """, (session_id, stage_name))


def fail_session_stage(session_id: int, stage_name: str, error: str, usage=None):
    """Mark a session_stages row failed; record any partial token usage."""
    with get_cursor() as cur:
        if usage is not None and usage.model is not None:
            cur.execute("""
                UPDATE session_stages
                SET status = 'failed',
                    completed_at = NOW(),
                    error = %s,
                    model = %s,
                    input_tokens = %s,
                    output_tokens = %s,
                    cache_creation_tokens = %s,
                    cache_read_tokens = %s
                WHERE session_id = %s AND stage_name = %s
            """, (
                error,
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_tokens,
                usage.cache_read_tokens,
                session_id, stage_name,
            ))
        else:
            cur.execute("""
                UPDATE session_stages SET status = 'failed', completed_at = NOW(), error = %s
                WHERE session_id = %s AND stage_name = %s
            """, (error, session_id, stage_name))


def close_orphan_running_stages(session_id: int, reason: str = "session ended with stage still running") -> list[str]:
    """Flip any session_stages still 'running' for this session_id to 'failed'.

    Belt-and-suspenders for cases where a stage entered try/except but the
    process was killed before fail_session_stage could write, leaving the row
    pinned at 'running' forever (the STAGE_RUNNING_STALE audit check).
    Returns the names of stages that were swept.
    """
    with get_cursor() as cur:
        cur.execute("""
            UPDATE session_stages
            SET status = 'failed', completed_at = NOW(), error = %s
            WHERE session_id = %s AND status = 'running'
            RETURNING stage_name
        """, (reason, session_id))
        return [row["stage_name"] for row in cur.fetchall()]


def get_completed_stages(session_id: int) -> set[str]:
    with get_cursor() as cur:
        cur.execute("""
            SELECT stage_name FROM session_stages
            WHERE session_id = %s AND status = 'completed'
        """, (session_id,))
        return {row["stage_name"] for row in cur.fetchall()}


def get_closed_losers(reference_date, limit: int = 15) -> list[dict]:
    """Decisions in the last 30 days with resolved 30-day outcomes < 0.

    Ordered worst first. Used by the dashboard /mistakes/ page and the
    weekly mistakes social post. `reference_date` is treated as 'today';
    the window is `reference_date - 30 days .. reference_date`.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   outcome_7d, outcome_30d
            FROM decisions
            WHERE date > %s::date - INTERVAL '30 days'
              AND outcome_30d IS NOT NULL
              AND outcome_30d < 0
            ORDER BY outcome_30d ASC
            LIMIT %s
        """, (reference_date, limit))
        return cur.fetchall()


def get_retired_rules(reference_date, limit: int = 10) -> list[dict]:
    """Rules retired in the last 90 days, most recent first.

    `reference_date` is treated as 'today'; the window is
    `reference_date - 90 days .. reference_date`.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, rule_text, category, direction, confidence,
                   retired_at, retirement_reason
            FROM strategy_rules
            WHERE status = 'retired'
              AND retired_at IS NOT NULL
              AND retired_at > %s::date - INTERVAL '90 days'
            ORDER BY retired_at DESC
            LIMIT %s
        """, (reference_date, limit))
        return cur.fetchall()


# --- Audit (self-healing) ---

# Stable advisory-lock key for the audit runner.
_AUDIT_ADVISORY_LOCK_KEY = 7341984512


def insert_audit_run(mode: str) -> int:
    """Create a new audit_runs row; returns the id."""
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO audit_runs (mode) VALUES (%s) RETURNING id",
            (mode,),
        )
        return cur.fetchone()["id"]


def insert_audit_finding(
    *, audit_run_id: int, check_code: str, tier: int, severity: str,
    title: str, body: str, affected_count: int, evidence: dict,
    fingerprint: str, status: str = "open", resolved_at=None,
    resolved_note=None,
) -> int | None:
    """Insert a finding row. Returns id, or None if an open finding with the
    same fingerprint already exists (idempotent re-emit)."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit_findings
                (audit_run_id, check_code, tier, severity, title, body,
                 affected_count, evidence, status, fingerprint,
                 resolved_at, resolved_note)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (fingerprint) WHERE status='open' DO NOTHING
            RETURNING id
            """,
            (audit_run_id, check_code, tier, severity, title, body,
             affected_count, Json(evidence), status, fingerprint,
             resolved_at, resolved_note),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def supersede_stale_open_findings(*, run_id: int, current_fingerprints: set[str]) -> int:
    """Mark all open findings whose fingerprint is not in current_fingerprints
    as superseded with a reference to this run id. Returns count updated."""
    with get_cursor() as cur:
        if current_fingerprints:
            cur.execute(
                """
                UPDATE audit_findings
                SET status='superseded',
                    resolved_at=now(),
                    resolved_note=%s
                WHERE status='open'
                  AND fingerprint NOT IN %s
                """,
                (f"not detected by run #{run_id}", tuple(current_fingerprints)),
            )
        else:
            cur.execute(
                """
                UPDATE audit_findings
                SET status='superseded',
                    resolved_at=now(),
                    resolved_note=%s
                WHERE status='open'
                """,
                (f"not detected by run #{run_id}",),
            )
        return cur.rowcount


def finalize_audit_run(*, run_id: int, total_findings: int, auto_fixed: int,
                       failed_checks: int, model: str | None = None,
                       input_tokens: int | None = None,
                       output_tokens: int | None = None,
                       cache_creation_tokens: int | None = None,
                       cache_read_tokens: int | None = None) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE audit_runs
            SET completed_at=now(), total_findings=%s, auto_fixed=%s,
                failed_checks=%s, model=%s, input_tokens=%s, output_tokens=%s,
                cache_creation_tokens=%s, cache_read_tokens=%s
            WHERE id=%s
            """,
            (total_findings, auto_fixed, failed_checks, model,
             input_tokens, output_tokens, cache_creation_tokens,
             cache_read_tokens, run_id),
        )


def insert_audit_llm_call(
    *,
    audit_run_id: int,
    purpose: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    latency_ms: int | None = None,
) -> int:
    """Record one LLM call's token usage for an audit run.

    Returns the new row id. Inserts into audit_llm_calls (migration 029).
    """
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit_llm_calls
                (audit_run_id, purpose, model, input_tokens, output_tokens,
                 cache_creation_tokens, cache_read_tokens, latency_ms)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (audit_run_id, purpose, model, input_tokens, output_tokens,
             cache_creation_tokens, cache_read_tokens, latency_ms),
        )
        return cur.fetchone()["id"]


def try_advisory_audit_lock() -> bool:
    """Returns True if the lock was acquired; False if already held."""
    with get_cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_AUDIT_ADVISORY_LOCK_KEY,))
        return bool(cur.fetchone()["pg_try_advisory_lock"])


def release_advisory_audit_lock() -> None:
    with get_cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s)", (_AUDIT_ADVISORY_LOCK_KEY,))


def delete_orphan_decision_signals(signal_type: str, ids: list[int]) -> int:
    """Auto-fix helper: delete orphan decision_signals rows for a given
    signal_type and list of orphan signal_ids. Returns count deleted."""
    if not ids:
        return 0
    with get_cursor() as cur:
        cur.execute(
            "DELETE FROM decision_signals WHERE signal_type=%s AND signal_id = ANY(%s)",
            (signal_type, ids),
        )
        return cur.rowcount


def get_open_audit_findings():
    """Read open findings for the dashboard, ordered by severity then created_at."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, audit_run_id, check_code, tier, severity, title,
                   affected_count, created_at, evidence
            FROM audit_findings
            WHERE status='open'
            ORDER BY
              CASE severity WHEN 'critical' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END,
              tier, created_at DESC
        """)
        return cur.fetchall()


def get_audit_finding(finding_id: int):
    with get_cursor() as cur:
        cur.execute("SELECT * FROM audit_findings WHERE id=%s", (finding_id,))
        return cur.fetchone()


def get_recent_audit_runs(limit: int = 14):
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, started_at, completed_at, mode, total_findings,
                   auto_fixed, failed_checks, model,
                   input_tokens, output_tokens,
                   cache_creation_tokens, cache_read_tokens
            FROM audit_runs ORDER BY started_at DESC LIMIT %s
        """, (limit,))
        return cur.fetchall()


def update_audit_finding_status(finding_id: int, status: str, note: str | None) -> None:
    if status not in ("acknowledged", "resolved"):
        raise ValueError(f"manual status must be acknowledged or resolved, got {status!r}")
    with get_cursor() as cur:
        cur.execute(
            "UPDATE audit_findings SET status=%s, resolved_at=now(), resolved_note=%s WHERE id=%s",
            (status, note, finding_id),
        )
