"""Database queries for dashboard views."""

import os
from contextlib import contextmanager
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

from v2.market_calendar import is_trading_day

RULE_REVALIDATION_STREAK = 5
THESIS_STALE_TRADING_DAYS = 10


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
    finally:
        conn.close()


# --- Portfolio ---

def get_positions():
    """Get all current positions."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT ticker, shares, avg_cost, updated_at
            FROM positions
            ORDER BY ticker
        """)
        return cur.fetchall()


def get_latest_snapshot():
    """Get most recent account snapshot."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT date, cash, portfolio_value, buying_power,
                   long_market_value, short_market_value
            FROM account_snapshots
            ORDER BY date DESC
            LIMIT 1
        """)
        return cur.fetchone()


def get_open_orders():
    """Get active open orders."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT order_id, ticker, side, order_type, qty, filled_qty,
                   limit_price, stop_price, status, submitted_at
            FROM open_orders
            WHERE status IN ('new', 'partially_filled', 'accepted', 'pending_new')
            ORDER BY submitted_at DESC
        """)
        return cur.fetchall()


# --- Signals ---

def get_recent_ticker_signals(days: int = 7, limit: int = 50, category: str | None = None):
    """Fetch recent ticker-specific news signals, optionally filtered by category."""
    where = ["published_at > NOW() - INTERVAL '%s days'"]
    params: list = [days]
    if category:
        where.append("category = %s")
        params.append(category)
    params.append(limit)
    with get_cursor() as cur:
        cur.execute(f"""
            SELECT id, ticker, headline, summary, category, sentiment, confidence,
                   published_at, processed_at
            FROM news_signals
            WHERE {" AND ".join(where)}
            ORDER BY published_at DESC
            LIMIT %s
        """, params)
        return cur.fetchall()


def get_recent_macro_signals(days: int = 7, limit: int = 20, category: str | None = None):
    """Fetch recent macro signals, optionally filtered by category."""
    where = ["published_at > NOW() - INTERVAL '%s days'"]
    params: list = [days]
    if category:
        where.append("category = %s")
        params.append(category)
    params.append(limit)
    with get_cursor() as cur:
        cur.execute(f"""
            SELECT id, headline, category, affected_sectors, sentiment, published_at
            FROM macro_signals
            WHERE {" AND ".join(where)}
            ORDER BY published_at DESC
            LIMIT %s
        """, params)
        return cur.fetchall()


def get_signal_summary(days: int = 7):
    """Get aggregated signal counts by ticker and sentiment."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT ticker,
                   COUNT(*) as total,
                   SUM(CASE WHEN sentiment = 'bullish' THEN 1 ELSE 0 END) as bullish,
                   SUM(CASE WHEN sentiment = 'bearish' THEN 1 ELSE 0 END) as bearish,
                   SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral
            FROM news_signals
            WHERE published_at > NOW() - INTERVAL '%s days'
            GROUP BY ticker
            ORDER BY total DESC
        """, (days,))
        return cur.fetchall()


# --- Decisions ---

def get_recent_decisions(days: int = 30, limit: int = 50):
    """Get recent trading decisions with outcomes."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   account_equity, buying_power, outcome_7d, outcome_30d,
                   is_off_playbook, playbook_action_id
            FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC, id DESC
            LIMIT %s
        """, (days, limit))
        return cur.fetchall()


def get_decision_stats(days: int = 30):
    """Get decision statistics."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total_decisions,
                SUM(CASE WHEN action = 'buy' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN action = 'sell' THEN 1 ELSE 0 END) as sells,
                SUM(CASE WHEN action = 'hold' THEN 1 ELSE 0 END) as holds,
                AVG(outcome_7d) as avg_outcome_7d,
                AVG(outcome_30d) as avg_outcome_30d
            FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
        """, (days,))
        return cur.fetchone()


# --- Performance ---

def get_equity_curve(days: int = 90):
    """Get account snapshots for equity curve chart."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT date, cash, portfolio_value, buying_power
            FROM account_snapshots
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date ASC
        """, (days,))
        return cur.fetchall()


def get_performance_metrics(days: int = 30):
    """Calculate performance metrics."""
    with get_cursor() as cur:
        # Get first and last snapshots in period
        cur.execute("""
            WITH period_data AS (
                SELECT * FROM account_snapshots
                WHERE date > CURRENT_DATE - INTERVAL '%s days'
            )
            SELECT
                (SELECT portfolio_value FROM period_data ORDER BY date ASC LIMIT 1) as start_value,
                (SELECT portfolio_value FROM period_data ORDER BY date DESC LIMIT 1) as end_value,
                (SELECT date FROM period_data ORDER BY date ASC LIMIT 1) as start_date,
                (SELECT date FROM period_data ORDER BY date DESC LIMIT 1) as end_date
        """, (days,))
        result = cur.fetchone()

        if result and result['start_value'] and result['end_value']:
            start_val = float(result['start_value'])
            end_val = float(result['end_value'])
            pnl = end_val - start_val
            pnl_pct = ((end_val / start_val) - 1) * 100 if start_val > 0 else 0

            return {
                'start_value': start_val,
                'end_value': end_val,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'start_date': result['start_date'],
                'end_date': result['end_date'],
            }

        return None


# --- Playbook ---

def get_today_playbook():
    """Get today's playbook."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM playbooks
            WHERE date = CURRENT_DATE
            ORDER BY created_at DESC
            LIMIT 1
        """)
        return cur.fetchone()


def get_playbook_actions(playbook_id):
    """Get structured playbook actions with thesis details."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT pa.id, pa.playbook_id, pa.ticker, pa.action, pa.thesis_id,
                   pa.reasoning, pa.confidence, pa.intent_type, pa.intent_magnitude,
                   pa.priority, pa.created_at,
                   t.thesis AS thesis_text, t.direction AS thesis_direction
            FROM playbook_actions pa
            LEFT JOIN theses t ON pa.thesis_id = t.id
            WHERE pa.playbook_id = %s
            ORDER BY pa.priority ASC NULLS LAST
        """, (playbook_id,))
        return cur.fetchall()


def get_behavior_regression_summary(days: int = 30):
    """Return operator-facing behavior checks for strategist/executor drift."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT COALESCE(pa.status, 'pending') AS status,
                   COUNT(*) AS count
            FROM playbook_actions pa
            JOIN playbooks p ON p.id = pa.playbook_id
            WHERE p.date > CURRENT_DATE - INTERVAL '%s days'
            GROUP BY COALESCE(pa.status, 'pending')
            ORDER BY status
        """, (days,))
        playbook_actions_by_status = cur.fetchall()

        cur.execute("""
            SELECT pa.id, pa.playbook_id, pa.ticker, pa.action,
                   COALESCE(pa.status, 'pending') AS status,
                   p.date AS playbook_date,
                   d.id AS decision_id,
                   d.action AS decision_action,
                   d.reasoning AS decision_reasoning
            FROM playbook_actions pa
            JOIN playbooks p ON p.id = pa.playbook_id
            JOIN decisions d ON d.playbook_action_id = pa.id
            WHERE p.date > CURRENT_DATE - INTERVAL '%s days'
              AND (pa.status = 'pending' OR pa.status IS NULL)
            ORDER BY p.date DESC, pa.ticker, pa.id
        """, (days,))
        reviewed_actions_still_pending = cur.fetchall()

        cur.execute("""
            SELECT pa.ticker,
                   COUNT(*) AS count,
                   MAX(p.date) AS last_deferred_date
            FROM playbook_actions pa
            JOIN playbooks p ON p.id = pa.playbook_id
            WHERE p.date > CURRENT_DATE - INTERVAL '%s days'
              AND pa.status = 'deferred'
            GROUP BY pa.ticker
            ORDER BY count DESC, pa.ticker
        """, (days,))
        deferred_actions_by_ticker = cur.fetchall()

        cur.execute("""
            SELECT id, session_date, status, completed_at
            FROM sessions
            WHERE session_date > CURRENT_DATE - INTERVAL '%s days'
              AND market_closed = TRUE
            ORDER BY session_date DESC, id DESC
        """, (days,))
        market_closed_sessions = cur.fetchall()

        cur.execute("""
            SELECT
                sr.id AS rule_id,
                sr.rule_text,
                sr.category,
                sr.direction,
                sr.lift_condition,
                d.date AS decision_date,
                d.ticker,
                d.action AS decision_action,
                pa.status AS playbook_action_status
            FROM strategy_rules sr
            JOIN decision_signals ds
              ON ds.signal_type = 'rule_gate'
             AND ds.signal_id = sr.id
            JOIN decisions d ON d.id = ds.decision_id
            LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
            WHERE sr.status = 'active'
              AND d.date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY sr.id, d.date DESC, d.id DESC
        """, (days,))
        rule_gate_rows = cur.fetchall()

        cur.execute("""
            WITH thesis_decisions AS (
                SELECT pa.thesis_id AS thesis_id, MAX(d.date) AS last_decision_date
                FROM decisions d
                JOIN playbook_actions pa ON pa.id = d.playbook_action_id
                WHERE pa.thesis_id IS NOT NULL
                GROUP BY pa.thesis_id

                UNION ALL

                SELECT ds.signal_id AS thesis_id, MAX(d.date) AS last_decision_date
                FROM decisions d
                JOIN decision_signals ds ON ds.decision_id = d.id
                WHERE ds.signal_type = 'thesis'
                GROUP BY ds.signal_id
            ),
            last_touched AS (
                SELECT thesis_id, MAX(last_decision_date) AS last_decision_date
                FROM thesis_decisions
                GROUP BY thesis_id
            ),
            classified AS (
                SELECT
                    CASE
                        WHEN p.ticker IS NOT NULL THEN 'held'
                        WHEN t.direction = 'avoid' THEN 'avoid'
                        ELSE 'watch'
                    END AS role,
                    GREATEST(
                        t.updated_at::date,
                        COALESCE(lt.last_decision_date, t.created_at::date)
                    ) AS last_touched_date
                FROM theses t
                LEFT JOIN positions p ON p.ticker = t.ticker
                LEFT JOIN last_touched lt ON lt.thesis_id = t.id
                WHERE t.status = 'active'
            )
            SELECT role, last_touched_date
            FROM classified
            ORDER BY role, last_touched_date
        """)
        active_thesis_role_rows = cur.fetchall()

    return {
        "playbook_actions_by_status": playbook_actions_by_status,
        "reviewed_actions_still_pending": reviewed_actions_still_pending,
        "deferred_actions_by_ticker": deferred_actions_by_ticker,
        "market_closed_sessions": market_closed_sessions,
        "rule_gate_telemetry": _shape_rule_gate_telemetry(rule_gate_rows),
        "active_thesis_role_counts": _shape_active_thesis_role_counts(active_thesis_role_rows),
    }


def _previous_trading_day(day):
    current = day - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current


def _consecutive_rule_cite_streak(dates) -> int:
    ordered = sorted(set(dates), reverse=True)
    if not ordered:
        return 0
    streak = 1
    expected = _previous_trading_day(ordered[0])
    for cited_date in ordered[1:]:
        if cited_date != expected:
            break
        streak += 1
        expected = _previous_trading_day(cited_date)
    return streak


def _shape_rule_gate_telemetry(rows):
    by_rule = {}
    for row in rows:
        rule = by_rule.setdefault(row["rule_id"], {
            "rule_id": row["rule_id"],
            "rule_text": row["rule_text"],
            "category": row["category"],
            "direction": row["direction"],
            "lift_condition": row.get("lift_condition"),
            "has_lift_condition": bool(row.get("lift_condition")),
            "cite_count": 0,
            "distinct_tickers": set(),
            "last_cited_date": None,
            "hold_deferred_count": 0,
            "_dates": [],
        })
        rule["cite_count"] += 1
        rule["distinct_tickers"].add(row["ticker"])
        decision_date = row["decision_date"]
        rule["_dates"].append(decision_date)
        if rule["last_cited_date"] is None or decision_date > rule["last_cited_date"]:
            rule["last_cited_date"] = decision_date
        if row["decision_action"] == "hold" or row.get("playbook_action_status") == "deferred":
            rule["hold_deferred_count"] += 1

    shaped = []
    for rule in by_rule.values():
        rule["consecutive_session_streak"] = _consecutive_rule_cite_streak(rule.pop("_dates"))
        rule["distinct_tickers"] = sorted(rule["distinct_tickers"])
        rule["is_gated"] = (
            not rule["has_lift_condition"]
            and rule["consecutive_session_streak"] >= RULE_REVALIDATION_STREAK
        )
        shaped.append(rule)
    return sorted(
        shaped,
        key=lambda r: (r["consecutive_session_streak"], r["cite_count"], r["rule_id"]),
        reverse=True,
    )


def _trading_days_since(start, end) -> int:
    if start >= end:
        return 0
    current = start + timedelta(days=1)
    days = 0
    while current <= end:
        if is_trading_day(current):
            days += 1
        current += timedelta(days=1)
    return days


def _shape_active_thesis_role_counts(rows):
    today = date.today()
    by_role = {}
    for row in rows:
        role = row["role"]
        bucket = by_role.setdefault(role, {"role": role, "count": 0, "stale_count": 0})
        bucket["count"] += 1
        if (
            role in ("watch", "avoid")
            and _trading_days_since(row["last_touched_date"], today) >= THESIS_STALE_TRADING_DAYS
        ):
            bucket["stale_count"] += 1
    return [by_role[role] for role in sorted(by_role)]


# --- Signal Attribution ---

def get_signal_attribution(category: str | None = None):
    """Get latest attribution scores, optionally filtered to one category."""
    where = ""
    params: list = []
    if category:
        where = "WHERE category = %s"
        params.append(category)
    with get_cursor() as cur:
        cur.execute(f"""
            SELECT category, sample_size, avg_outcome_7d, avg_outcome_30d,
                   win_rate_7d, win_rate_30d, updated_at
            FROM signal_attribution
            {where}
            ORDER BY sample_size DESC
        """, params)
        return cur.fetchall()


# --- Decision Signals ---

def get_decision_signal_refs(decision_id):
    """Get signal refs for a decision."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT signal_type, signal_id
            FROM decision_signals
            WHERE decision_id = %s
        """, (decision_id,))
        return cur.fetchall()


def get_decision_signal_refs_batch(decision_ids):
    """Get signal refs for multiple decisions in one query.

    Returns dict[int, list] grouped by decision_id.
    """
    if not decision_ids:
        return {}
    with get_cursor() as cur:
        cur.execute("""
            SELECT ds.decision_id, ds.signal_type, ds.signal_id,
                   ns.headline AS news_headline,
                   ms.headline AS macro_headline,
                   t.thesis AS thesis_text
            FROM decision_signals ds
            LEFT JOIN news_signals ns
                ON ds.signal_type = 'news_signal' AND ds.signal_id = ns.id
            LEFT JOIN macro_signals ms
                ON ds.signal_type = 'macro_signal' AND ds.signal_id = ms.id
            LEFT JOIN theses t
                ON ds.signal_type = 'thesis' AND ds.signal_id = t.id
            WHERE ds.decision_id = ANY(%s)
            ORDER BY ds.decision_id, ds.signal_type
        """, (decision_ids,))
        rows = cur.fetchall()

    result = {}
    for row in rows:
        did = row['decision_id']
        label = (row['news_headline']
                 or row['macro_headline']
                 or row['thesis_text']
                 or f"{row['signal_type']}#{row['signal_id']}")
        result.setdefault(did, []).append({
            'signal_type': row['signal_type'],
            'signal_id': row['signal_id'],
            'label': label,
        })
    return result


# --- Theses ---

def get_thesis_stats():
    """Return dict with counts by status and success rate."""
    with get_cursor() as cur:
        # Get counts by status
        cur.execute("""
            SELECT
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'executed' THEN 1 ELSE 0 END) as executed,
                SUM(CASE WHEN status = 'invalidated' THEN 1 ELSE 0 END) as invalidated,
                SUM(CASE WHEN status = 'expired' THEN 1 ELSE 0 END) as expired
            FROM theses
        """)
        counts = cur.fetchone()

        # Get confidence distribution for active theses
        cur.execute("""
            SELECT confidence, COUNT(*) as count
            FROM theses
            WHERE status = 'active'
            GROUP BY confidence
        """)
        confidence_rows = cur.fetchall()
        confidence_dist = {row['confidence']: row['count'] for row in confidence_rows}

        executed = counts['executed'] or 0
        invalidated = counts['invalidated'] or 0
        expired = counts['expired'] or 0
        # Execution conversion rate among closed theses (executed vs.
        # dropped). A more rigorous "success" would join through
        # decision_signals to alpha; this cheap count avoids the N/A
        # placeholder until that lands.
        closed = executed + invalidated + expired
        success_rate = (executed / closed * 100) if closed else None
        return {
            'active': counts['active'] or 0,
            'executed': executed,
            'invalidated': invalidated,
            'expired': expired,
            'success_rate': success_rate,
            'confidence_dist': confidence_dist,
        }


def close_thesis(thesis_id: int, status: str, reason: str = None) -> bool:
    """Close a thesis with the given status and optional reason."""
    if status not in ('invalidated', 'expired'):
        raise ValueError(f"Invalid close status: {status}")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE theses
                SET status = %s,
                    close_reason = %s,
                    closed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s AND status = 'active'
            """, (status, reason, thesis_id))
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()



# --- Strategy ---

def get_current_strategy():
    """Fetch the current strategy state."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, identity_text, risk_posture, sector_biases,
                   preferred_signals, avoided_signals, version, created_at
            FROM strategy_state
            WHERE is_current = TRUE
            LIMIT 1
        """)
        return cur.fetchone()


def get_strategy_rules(status='active'):
    """Fetch strategy rules filtered by status."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, rule_text, category, direction, confidence,
                   supporting_evidence, status, created_at, retired_at
            FROM strategy_rules
            WHERE status = %s
            ORDER BY confidence DESC
        """, (status,))
        return cur.fetchall()


def get_strategy_memos(days=30):
    """Fetch recent strategy memos."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, session_date, memo_type, content, created_at, session_id
            FROM strategy_memos
            WHERE session_date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY session_date DESC, created_at DESC
        """, (days,))
        return cur.fetchall()


# --- Tweets ---

def get_recent_tweets(days=30, limit=50):
    """Fetch recent tweets joined with their session row."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT tw.id, tw.session_date, tw.tweet_type, tw.tweet_text,
                   tw.platform, tw.posted, tw.error, tw.created_at,
                   tw.decision_id,
                   s.id AS session_id
            FROM tweets tw
            LEFT JOIN sessions s ON s.id = tw.session_id
            WHERE tw.session_date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY tw.session_date DESC, tw.created_at DESC
            LIMIT %s
        """, (days, limit))
        return cur.fetchall()


def get_theses(status_filter: str = 'active', sort_by: str = 'newest'):
    """Return filtered/sorted thesis list."""
    with get_cursor() as cur:
        # Build WHERE clause
        where_clause = ""
        params = []
        if status_filter and status_filter != 'all':
            where_clause = "WHERE status = %s"
            params.append(status_filter)

        # Build ORDER BY clause
        order_map = {
            'newest': 'created_at DESC',
            'oldest': 'created_at ASC',
            'confidence': "CASE confidence WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END",
            'ticker': 'ticker ASC',
        }
        order_clause = order_map.get(sort_by, 'created_at DESC')

        cur.execute(f"""
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, source, status, created_at, updated_at
            FROM theses
            {where_clause}
            ORDER BY {order_clause}
        """, params)
        return cur.fetchall()


# --- Session cost / token usage ---

def get_recent_session_costs(limit: int = 30):
    """Return recent sessions with aggregated token usage and USD cost.

    Joins `session_costs` view back to `sessions` for ordering / display.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT sc.session_id, sc.session_date, sc.session_type, sc.status,
                   sc.total_cost_usd, sc.total_input_tokens, sc.total_output_tokens,
                   sc.total_cache_creation_tokens, sc.total_cache_read_tokens,
                   s.started_at, s.completed_at
            FROM session_costs sc
            JOIN sessions s ON s.id = sc.session_id
            ORDER BY sc.session_date DESC, sc.session_id DESC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


def get_session_stage_costs(session_id: int):
    """Return per-stage token usage + USD cost for a session."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, stage_name, status, started_at, completed_at,
                   model, input_tokens, output_tokens,
                   cache_creation_tokens, cache_read_tokens, cost_usd
            FROM session_stage_costs
            WHERE session_id = %s
            ORDER BY started_at ASC, id ASC
        """, (session_id,))
        return cur.fetchall()


# --- Agent events (telemetry substrate) ---

def get_recent_agent_events(limit: int = 100, event_type: str = None,
                            session_id: int = None):
    """Return recent agent_events with optional event_type / session_id filter."""
    where = []
    params = []
    if event_type:
        where.append("event_type = %s")
        params.append(event_type)
    if session_id:
        where.append("session_id = %s")
        params.append(session_id)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    with get_cursor() as cur:
        cur.execute(f"""
            SELECT id, session_id, stage_name, event_type, payload, occurred_at
            FROM agent_events
            {where_sql}
            ORDER BY occurred_at DESC, id DESC
            LIMIT %s
        """, params)
        return cur.fetchall()


def get_agent_event_types(days: int = 14):
    """Return distinct event_type values with counts over the last N days."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT event_type, COUNT(*) AS n
            FROM agent_events
            WHERE occurred_at > NOW() - INTERVAL '%s days'
            GROUP BY event_type
            ORDER BY n DESC
        """, (days,))
        return cur.fetchall()


def get_session(session_id: int):
    """Return one sessions row, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, session_date, session_type, status, started_at,
                   completed_at, error
            FROM sessions
            WHERE id = %s
        """, (session_id,))
        return cur.fetchone()


def get_session_decisions(session_id: int):
    """Return decisions made during this session (by session_id)."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT d.id, d.date, d.ticker, d.action, d.quantity, d.price,
                   d.reasoning, d.account_equity, d.outcome_7d, d.outcome_30d,
                   d.is_off_playbook, d.playbook_action_id
            FROM decisions d
            WHERE d.session_id = %s
            ORDER BY d.id ASC
        """, (session_id,))
        return cur.fetchall()


def get_session_theses_created(session_id: int):
    """Return theses created during this session (by session_id)."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT t.id, t.ticker, t.direction, t.thesis, t.entry_trigger,
                   t.exit_trigger, t.invalidation, t.confidence, t.source,
                   t.status, t.created_at, t.updated_at
            FROM theses t
            WHERE t.session_id = %s
            ORDER BY t.created_at ASC
        """, (session_id,))
        return cur.fetchall()


def get_session_memo(session_id: int):
    """Return the strategy_memos row for this session, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT m.id, m.session_date, m.memo_type, m.content, m.created_at
            FROM strategy_memos m
            WHERE m.session_id = %s
            ORDER BY m.created_at DESC
            LIMIT 1
        """, (session_id,))
        return cur.fetchone()


def get_session_tweets(session_id: int):
    """Return tweets posted during this session (by session_id)."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT tw.id, tw.session_date, tw.tweet_type, tw.tweet_text,
                   tw.platform, tw.posted, tw.error, tw.created_at,
                   tw.decision_id
            FROM tweets tw
            WHERE tw.session_id = %s
            ORDER BY tw.created_at DESC
        """, (session_id,))
        return cur.fetchall()


def get_session_events(session_id: int, limit: int = 200):
    """Return agent_events filtered to this session (thin wrapper)."""
    return get_recent_agent_events(limit=limit, session_id=session_id)


def get_session_llm_calls(session_id: int):
    """Return summary rows from llm_call_contexts for one session.

    Excludes the JSONB columns (messages, tool_definitions,
    response_content) because they are large and the list view does not
    need them. Ordering puts the executor row first, then strategist
    loop turns in sequence, then reflection loop turns in sequence.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, stage_name, purpose, sequence, model,
                   input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens,
                   stop_reason, duration_ms, created_at
            FROM llm_call_contexts
            WHERE session_id = %s
            ORDER BY
              CASE purpose
                WHEN 'executor' THEN 0
                WHEN 'strategist_loop' THEN 1
                WHEN 'reflection_loop' THEN 2
                ELSE 3
              END,
              sequence
        """, (session_id,))
        return cur.fetchall()


def get_llm_call(call_id: int):
    """Return a single llm_call_contexts row by id, or None.

    Returns all columns including the JSONB payloads (messages,
    tool_definitions, response_content) — the detail view needs them.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, session_id, stage_name, purpose, sequence, model,
                   system_prompt, messages, tool_definitions,
                   response_content,
                   input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens,
                   stop_reason, duration_ms, created_at
            FROM llm_call_contexts
            WHERE id = %s
        """, (call_id,))
        return cur.fetchone()


def lookup_session_id_by_date(d, session_type: str = 'daily'):
    """Return the most recent sessions.id for a given date + type, or None.

    Legacy/fallback lookup. Per-run session IDs are now the norm: decisions,
    theses, strategy_memos, and tweets all carry a `session_id` FK directly
    (see migration 029), and the dashboard prefers that field. This helper
    is retained as a fallback for historical rows where `session_id` is
    NULL (pre-migration data, or rows written before per-run threading
    landed in every code path). The UNIQUE (session_date, session_type)
    constraint was dropped in migration 029, so multiple sessions per
    (date, type) are now possible — we ORDER BY started_at DESC and return
    the most recent.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id
            FROM sessions
            WHERE session_date = %s AND session_type = %s
            ORDER BY started_at DESC
            LIMIT 1
        """, (d, session_type))
        row = cur.fetchone()
        return row["id"] if row else None


def get_thesis(thesis_id: int):
    """Return one thesis row, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, source, status,
                   created_at, updated_at, closed_at, close_reason, session_id
            FROM theses
            WHERE id = %s
        """, (thesis_id,))
        return cur.fetchone()


def get_thesis_decisions(thesis_id: int):
    """Return decisions that cited this thesis (via decision_signals)."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT d.id, d.date, d.ticker, d.action, d.quantity, d.price,
                   d.reasoning, d.account_equity, d.outcome_7d, d.outcome_30d,
                   d.is_off_playbook, d.playbook_action_id
            FROM decisions d
            JOIN decision_signals ds ON ds.decision_id = d.id
            WHERE ds.signal_type = 'thesis' AND ds.signal_id = %s
            ORDER BY d.date DESC, d.id DESC
        """, (thesis_id,))
        return cur.fetchall()


def get_thesis_playbook_actions(thesis_id: int):
    """Return playbook_actions that reference this thesis, with playbook date."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT pa.id, pa.playbook_id, pa.ticker, pa.action, pa.thesis_id,
                   pa.reasoning, pa.confidence, pa.intent_type,
                   pa.intent_magnitude, pa.priority, pa.created_at,
                   p.date AS playbook_date
            FROM playbook_actions pa
            JOIN playbooks p ON p.id = pa.playbook_id
            WHERE pa.thesis_id = %s
            ORDER BY p.date DESC, pa.priority ASC NULLS LAST
        """, (thesis_id,))
        return cur.fetchall()


def get_decision(decision_id: int):
    """Return one decision row, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   signals_used, account_equity, buying_power,
                   outcome_7d, outcome_30d, is_off_playbook, playbook_action_id,
                   session_id
            FROM decisions
            WHERE id = %s
        """, (decision_id,))
        return cur.fetchone()


def get_decision_signals_full(decision_id: int):
    """Return decision_signals rows denormalized with the full signal record.

    Result rows always contain all three signal blocks; only the matching
    one is populated for any given row. Template renders the populated one.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT ds.signal_type, ds.signal_id,
                   ns.headline   AS news_headline,
                   ns.summary    AS news_summary,
                   ns.category   AS news_category,
                   ns.sentiment  AS news_sentiment,
                   ns.confidence AS news_confidence,
                   ns.published_at AS news_published_at,
                   ns.ticker     AS news_ticker,
                   ms.headline   AS macro_headline,
                   ms.category   AS macro_category,
                   ms.affected_sectors AS macro_affected_sectors,
                   ms.sentiment  AS macro_sentiment,
                   ms.published_at AS macro_published_at,
                   t.thesis      AS thesis_text,
                   t.ticker      AS thesis_ticker,
                   t.direction   AS thesis_direction,
                   t.status      AS thesis_status
            FROM decision_signals ds
            LEFT JOIN news_signals  ns ON ds.signal_type = 'news_signal'  AND ds.signal_id = ns.id
            LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ds.signal_id = ms.id
            LEFT JOIN theses        t  ON ds.signal_type = 'thesis'       AND ds.signal_id = t.id
            WHERE ds.decision_id = %s
            ORDER BY ds.signal_type, ds.signal_id
        """, (decision_id,))
        return cur.fetchall()


def get_decision_tweets(decision_id: int):
    """Return tweets posted for a given decision_id."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, session_date, tweet_type, tweet_text, platform,
                   posted, error, created_at
            FROM tweets
            WHERE decision_id = %s
            ORDER BY created_at DESC
        """, (decision_id,))
        return cur.fetchall()


def get_playbook_action(action_id: int):
    """Return one playbook_action joined with its thesis info, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT pa.id, pa.playbook_id, pa.ticker, pa.action, pa.thesis_id,
                   pa.reasoning, pa.confidence, pa.intent_type,
                   pa.intent_magnitude, pa.priority, pa.created_at,
                   t.thesis    AS thesis_text,
                   t.direction AS thesis_direction,
                   t.status    AS thesis_status,
                   p.date      AS playbook_date
            FROM playbook_actions pa
            LEFT JOIN theses    t ON t.id = pa.thesis_id
            LEFT JOIN playbooks p ON p.id = pa.playbook_id
            WHERE pa.id = %s
        """, (action_id,))
        return cur.fetchone()


# --- Ticker overview ---

def get_ticker_position(sym: str):
    """Return one position row for this ticker, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT ticker, shares, avg_cost, updated_at
            FROM positions
            WHERE ticker = %s
        """, (sym,))
        return cur.fetchone()


def get_ticker_theses(sym: str):
    """Return all theses (any status) for this ticker, newest first."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, source, status,
                   created_at, updated_at, closed_at, close_reason
            FROM theses
            WHERE ticker = %s
            ORDER BY created_at DESC
        """, (sym,))
        return cur.fetchall()


def get_ticker_decisions(sym: str, days: int = 90, limit: int = 50):
    """Return recent decisions for this ticker."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   account_equity, outcome_7d, outcome_30d,
                   is_off_playbook, playbook_action_id
            FROM decisions
            WHERE ticker = %s
              AND date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC, id DESC
            LIMIT %s
        """, (sym, days, limit))
        return cur.fetchall()


def get_ticker_signals(sym: str, days: int = 30, limit: int = 50):
    """Return recent news signals for this ticker."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, ticker, headline, summary, category, sentiment,
                   confidence, published_at
            FROM news_signals
            WHERE ticker = %s
              AND published_at > NOW() - INTERVAL '%s days'
            ORDER BY published_at DESC
            LIMIT %s
        """, (sym, days, limit))
        return cur.fetchall()


def get_ticker_open_orders(sym: str):
    """Return open orders for this ticker."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT order_id, ticker, side, order_type, qty, filled_qty,
                   limit_price, stop_price, status, submitted_at, updated_at
            FROM open_orders
            WHERE ticker = %s
            ORDER BY submitted_at DESC
        """, (sym,))
        return cur.fetchall()


def get_ticker_attribution(sym: str, days: int = 90):
    """Per-category attribution for signals that fed decisions on this ticker.

    Joins decision_signals through to news/macro categories and aggregates
    decision outcomes. Theses are excluded (not a 'category').
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                CASE ds.signal_type
                    WHEN 'news_signal'  THEN 'news:'  || ns.category
                    WHEN 'macro_signal' THEN 'macro:' || ms.category
                END AS category,
                ds.signal_type,
                COUNT(*) AS sample_size,
                AVG(d.outcome_7d)::numeric(8,4)  AS avg_outcome_7d,
                AVG(d.outcome_30d)::numeric(8,4) AS avg_outcome_30d
            FROM decision_signals ds
            JOIN decisions d ON d.id = ds.decision_id
            LEFT JOIN news_signals  ns ON ds.signal_type = 'news_signal'  AND ds.signal_id = ns.id
            LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ds.signal_id = ms.id
            WHERE d.ticker = %s
              AND d.date > CURRENT_DATE - INTERVAL '%s days'
              AND ds.signal_type IN ('news_signal', 'macro_signal')
              AND ds.signal_id IS NOT NULL
            GROUP BY 1, ds.signal_type
            ORDER BY sample_size DESC
        """, (sym, days))
        return cur.fetchall()


# --- Supervisor memos ---

def get_recent_supervisor_memos(limit: int = 10) -> list[dict]:
    """Compact recent supervisor memos for the dashboard sidebar / index page."""
    sql = """
        SELECT id, created_at, model, prompt_version, status, turns_used, cost_usd
        FROM supervisor_memos
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
            "model": r["model"],
            "prompt_version": r["prompt_version"],
            "status": r["status"],
            "turns_used": r["turns_used"],
            "cost_usd": float(r["cost_usd"]) if r["cost_usd"] is not None else None,
        }
        for r in rows
    ]


def get_supervisor_memo(memo_id: int) -> dict | None:
    """Full supervisor memo for the detail page. Returns None if not found."""
    sql = """
        SELECT id, created_at, model, prompt_version, content, status, turns_used,
               tool_calls, input_tokens, output_tokens, cost_usd, error_message
        FROM supervisor_memos
        WHERE id = %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (memo_id,))
        r = cur.fetchone()
    if r is None:
        return None
    return {
        "id": r["id"],
        "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
        "model": r["model"],
        "prompt_version": r["prompt_version"],
        "content": r["content"],
        "status": r["status"],
        "turns_used": r["turns_used"],
        "tool_calls": r["tool_calls"],
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
        "cost_usd": float(r["cost_usd"]) if r["cost_usd"] is not None else None,
        "error_message": r["error_message"],
    }
