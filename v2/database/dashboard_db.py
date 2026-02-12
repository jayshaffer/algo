"""Database queries for dashboard views."""
from .connection import get_cursor, get_connection


def get_positions():
    with get_cursor() as cur:
        cur.execute("SELECT ticker, shares, avg_cost, updated_at FROM positions ORDER BY ticker")
        return cur.fetchall()


def get_latest_snapshot():
    with get_cursor() as cur:
        cur.execute("""
            SELECT date, cash, portfolio_value, buying_power, long_market_value, short_market_value
            FROM account_snapshots ORDER BY date DESC LIMIT 1
        """)
        return cur.fetchone()


def get_recent_ticker_signals(days=7, limit=50):
    with get_cursor() as cur:
        cur.execute("""
            SELECT ticker, headline, category, sentiment, confidence, published_at
            FROM news_signals WHERE published_at > NOW() - INTERVAL '%s days'
            ORDER BY published_at DESC LIMIT %s
        """, (days, limit))
        return cur.fetchall()


def get_recent_macro_signals(days=7, limit=20):
    with get_cursor() as cur:
        cur.execute("""
            SELECT headline, category, affected_sectors, sentiment, published_at
            FROM macro_signals WHERE published_at > NOW() - INTERVAL '%s days'
            ORDER BY published_at DESC LIMIT %s
        """, (days, limit))
        return cur.fetchall()


def get_signal_summary(days=7):
    with get_cursor() as cur:
        cur.execute("""
            SELECT ticker,
                   COUNT(*) as total,
                   SUM(CASE WHEN sentiment = 'bullish' THEN 1 ELSE 0 END) as bullish,
                   SUM(CASE WHEN sentiment = 'bearish' THEN 1 ELSE 0 END) as bearish,
                   SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral
            FROM news_signals
            WHERE published_at > NOW() - INTERVAL '%s days'
            GROUP BY ticker ORDER BY total DESC
        """, (days,))
        return cur.fetchall()


def get_recent_decisions(days=30, limit=50):
    with get_cursor() as cur:
        cur.execute("""
            SELECT date, ticker, action, quantity, price, reasoning,
                   account_equity, buying_power, outcome_7d, outcome_30d
            FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC, id DESC LIMIT %s
        """, (days, limit))
        return cur.fetchall()


def get_decision_stats(days=30):
    with get_cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) as total_decisions,
                   SUM(CASE WHEN action = 'buy' THEN 1 ELSE 0 END) as buys,
                   SUM(CASE WHEN action = 'sell' THEN 1 ELSE 0 END) as sells,
                   SUM(CASE WHEN action = 'hold' THEN 1 ELSE 0 END) as holds,
                   AVG(outcome_7d) as avg_outcome_7d, AVG(outcome_30d) as avg_outcome_30d
            FROM decisions WHERE date > CURRENT_DATE - INTERVAL '%s days'
        """, (days,))
        return cur.fetchone()


def get_equity_curve(days=90):
    with get_cursor() as cur:
        cur.execute("""
            SELECT date, cash, portfolio_value, buying_power
            FROM account_snapshots
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date ASC
        """, (days,))
        return cur.fetchall()


def get_performance_metrics(days=30):
    with get_cursor() as cur:
        cur.execute("""
            WITH period_data AS (
                SELECT * FROM account_snapshots WHERE date > CURRENT_DATE - INTERVAL '%s days'
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
            return {'start_value': start_val, 'end_value': end_val, 'pnl': pnl, 'pnl_pct': pnl_pct,
                    'start_date': result['start_date'], 'end_date': result['end_date']}
        return None


def get_today_playbook():
    with get_cursor() as cur:
        cur.execute("SELECT * FROM playbooks WHERE date = CURRENT_DATE ORDER BY created_at DESC LIMIT 1")
        return cur.fetchone()


def get_signal_attribution():
    with get_cursor() as cur:
        cur.execute("""
            SELECT category, sample_size, avg_outcome_7d, avg_outcome_30d,
                   win_rate_7d, win_rate_30d, updated_at
            FROM signal_attribution ORDER BY sample_size DESC
        """)
        return cur.fetchall()


def get_decision_signal_refs(decision_id):
    with get_cursor() as cur:
        cur.execute("SELECT signal_type, signal_id FROM decision_signals WHERE decision_id = %s", (decision_id,))
        return cur.fetchall()


def get_thesis_stats():
    with get_cursor() as cur:
        cur.execute("""
            SELECT SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                   SUM(CASE WHEN status = 'executed' THEN 1 ELSE 0 END) as executed,
                   SUM(CASE WHEN status = 'invalidated' THEN 1 ELSE 0 END) as invalidated,
                   SUM(CASE WHEN status = 'expired' THEN 1 ELSE 0 END) as expired
            FROM theses
        """)
        counts = cur.fetchone()
        cur.execute("SELECT confidence, COUNT(*) as count FROM theses WHERE status = 'active' GROUP BY confidence")
        confidence_rows = cur.fetchall()
        confidence_dist = {row['confidence']: row['count'] for row in confidence_rows}
        return {'active': counts['active'] or 0, 'executed': counts['executed'] or 0,
                'invalidated': counts['invalidated'] or 0, 'expired': counts['expired'] or 0,
                'success_rate': None, 'confidence_dist': confidence_dist}


def close_thesis(thesis_id, status, reason=None):
    if status not in ('invalidated', 'expired'):
        raise ValueError(f"Invalid close status: {status}")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE theses SET status = %s, close_reason = %s, closed_at = NOW(), updated_at = NOW()
                WHERE id = %s AND status = 'active'
            """, (status, reason, thesis_id))
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()


def get_theses(status_filter='active', sort_by='newest'):
    with get_cursor() as cur:
        where_clause = ""
        params = []
        if status_filter and status_filter != 'all':
            where_clause = "WHERE status = %s"
            params.append(status_filter)
        order_map = {'newest': 'created_at DESC', 'oldest': 'created_at ASC',
                     'confidence': "CASE confidence WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END",
                     'ticker': 'ticker ASC'}
        order_clause = order_map.get(sort_by, 'created_at DESC')
        cur.execute(f"""
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, source, status, created_at, updated_at
            FROM theses {where_clause} ORDER BY {order_clause}
        """, params)
        return cur.fetchall()
