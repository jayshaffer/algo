"""Pattern analysis for learning from past decisions.

All signal-level queries go through decision_signals FK (not time-window JOINs).
This is the single source of truth for signal metrics.
"""

from dataclasses import dataclass
from .database.connection import get_cursor


@dataclass
class SignalPerformance:
    category: str
    total_signals: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    win_rate_7d: float | None
    win_rate_30d: float | None


@dataclass
class SentimentPerformance:
    sentiment: str
    total_decisions: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    win_rate_7d: float | None


@dataclass
class TickerPerformance:
    ticker: str
    total_decisions: int
    buys: int
    sells: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    total_pnl_7d: float | None


@dataclass
class ConfidenceCorrelation:
    confidence: str
    total_decisions: int
    avg_outcome_7d: float | None
    win_rate_7d: float | None


def analyze_signal_categories(days: int = 90) -> list[SignalPerformance]:
    """Analyze which signal categories lead to profitable trades.
    Uses decision_signals FK as single source of truth."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                CASE
                    WHEN ds.signal_type = 'news_signal' THEN
                        'news_signal:' || COALESCE(ns.category, 'unknown')
                    WHEN ds.signal_type = 'macro_signal' THEN
                        'macro_signal:' || COALESCE(ms.category, 'unknown')
                    ELSE ds.signal_type
                END AS category,
                COUNT(DISTINCT ds.decision_id) as total_signals,
                AVG(d.outcome_7d) as avg_outcome_7d,
                AVG(d.outcome_30d) as avg_outcome_30d,
                AVG(CASE WHEN d.outcome_7d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_7d,
                AVG(CASE WHEN d.outcome_30d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_30d
            FROM decision_signals ds
            JOIN decisions d ON d.id = ds.decision_id
            LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
            LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
            WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
              AND d.action IN ('buy', 'sell')
              AND d.outcome_7d IS NOT NULL
            GROUP BY category
            ORDER BY avg_outcome_7d DESC NULLS LAST
        """, (days,))

        results = []
        for row in cur.fetchall():
            results.append(SignalPerformance(
                category=row["category"],
                total_signals=row["total_signals"],
                avg_outcome_7d=float(row["avg_outcome_7d"]) if row["avg_outcome_7d"] else None,
                avg_outcome_30d=float(row["avg_outcome_30d"]) if row["avg_outcome_30d"] else None,
                win_rate_7d=float(row["win_rate_7d"]) if row["win_rate_7d"] else None,
                win_rate_30d=float(row["win_rate_30d"]) if row["win_rate_30d"] else None,
            ))
        return results


def analyze_sentiment_performance(days: int = 90) -> list[SentimentPerformance]:
    """Analyze performance by signal sentiment. JOINs through decision_signals FK."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                ns.sentiment,
                COUNT(DISTINCT ds.decision_id) as total_decisions,
                AVG(d.outcome_7d) as avg_outcome_7d,
                AVG(d.outcome_30d) as avg_outcome_30d,
                AVG(CASE WHEN d.outcome_7d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_7d
            FROM decision_signals ds
            JOIN decisions d ON d.id = ds.decision_id
            LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
            WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
              AND d.action IN ('buy', 'sell')
              AND ns.sentiment IS NOT NULL
            GROUP BY ns.sentiment
            ORDER BY avg_outcome_7d DESC NULLS LAST
        """, (days,))

        results = []
        for row in cur.fetchall():
            results.append(SentimentPerformance(
                sentiment=row["sentiment"],
                total_decisions=row["total_decisions"],
                avg_outcome_7d=float(row["avg_outcome_7d"]) if row["avg_outcome_7d"] else None,
                avg_outcome_30d=float(row["avg_outcome_30d"]) if row["avg_outcome_30d"] else None,
                win_rate_7d=float(row["win_rate_7d"]) if row["win_rate_7d"] else None,
            ))
        return results


def analyze_ticker_performance(days: int = 90) -> list[TickerPerformance]:
    """Performance by ticker. No signal JOIN needed — groups decisions directly."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                ticker,
                COUNT(*) as total_decisions,
                SUM(CASE WHEN action = 'buy' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN action = 'sell' THEN 1 ELSE 0 END) as sells,
                AVG(outcome_7d) as avg_outcome_7d,
                AVG(outcome_30d) as avg_outcome_30d,
                SUM(outcome_7d) as total_pnl_7d
            FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
              AND action IN ('buy', 'sell')
            GROUP BY ticker
            ORDER BY total_pnl_7d DESC NULLS LAST
        """, (days,))

        results = []
        for row in cur.fetchall():
            results.append(TickerPerformance(
                ticker=row["ticker"],
                total_decisions=row["total_decisions"],
                buys=row["buys"],
                sells=row["sells"],
                avg_outcome_7d=float(row["avg_outcome_7d"]) if row["avg_outcome_7d"] else None,
                avg_outcome_30d=float(row["avg_outcome_30d"]) if row["avg_outcome_30d"] else None,
                total_pnl_7d=float(row["total_pnl_7d"]) if row["total_pnl_7d"] else None,
            ))
        return results


def analyze_confidence_correlation(days: int = 90) -> list[ConfidenceCorrelation]:
    """Correlation between stated confidence and actual outcomes.
    JOINs through decision_signals FK."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                ns.confidence,
                COUNT(DISTINCT ds.decision_id) as total_decisions,
                AVG(d.outcome_7d) as avg_outcome_7d,
                AVG(CASE WHEN d.outcome_7d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_7d
            FROM decision_signals ds
            JOIN decisions d ON d.id = ds.decision_id
            LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
            WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
              AND d.action IN ('buy', 'sell')
              AND ns.confidence IS NOT NULL
            GROUP BY ns.confidence
            ORDER BY
                CASE ns.confidence
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                END
        """, (days,))

        results = []
        for row in cur.fetchall():
            results.append(ConfidenceCorrelation(
                confidence=row["confidence"],
                total_decisions=row["total_decisions"],
                avg_outcome_7d=float(row["avg_outcome_7d"]) if row["avg_outcome_7d"] else None,
                win_rate_7d=float(row["win_rate_7d"]) if row["win_rate_7d"] else None,
            ))
        return results


def get_best_performing_signals(days: int = 90, min_occurrences: int = 3) -> list[dict]:
    """Get signal categories with best outcomes. Reads from signal_attribution table."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT category, avg_outcome_7d as avg_outcome, sample_size as occurrences,
                   win_rate_7d
            FROM signal_attribution
            WHERE sample_size >= %s AND avg_outcome_7d IS NOT NULL
            ORDER BY avg_outcome_7d DESC
            LIMIT 10
        """, (min_occurrences,))
        return [dict(row) for row in cur.fetchall()]


def get_worst_performing_signals(days: int = 90, min_occurrences: int = 3) -> list[dict]:
    """Get signal categories with worst outcomes. Reads from signal_attribution table."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT category, avg_outcome_7d as avg_outcome, sample_size as occurrences,
                   win_rate_7d
            FROM signal_attribution
            WHERE sample_size >= %s AND avg_outcome_7d IS NOT NULL
            ORDER BY avg_outcome_7d ASC
            LIMIT 10
        """, (min_occurrences,))
        return [dict(row) for row in cur.fetchall()]


def generate_pattern_report(days: int = 90) -> str:
    """Orchestrate all analysis functions into a text report."""
    lines = [
        f"Pattern Analysis Report ({days} days)",
        "=" * 50,
        "",
    ]

    # Signal categories
    signal_perf = analyze_signal_categories(days)
    if signal_perf:
        lines.append("Signal Category Performance:")
        for sp in signal_perf:
            outcome = f"{sp.avg_outcome_7d:+.2f}%" if sp.avg_outcome_7d else "N/A"
            win_rate = f"{sp.win_rate_7d:.0f}%" if sp.win_rate_7d else "N/A"
            lines.append(f"  {sp.category}: {outcome} avg (win rate: {win_rate}, n={sp.total_signals})")
        lines.append("")

    # Sentiment performance
    sentiment_perf = analyze_sentiment_performance(days)
    if sentiment_perf:
        lines.append("Sentiment Performance:")
        for sp in sentiment_perf:
            outcome = f"{sp.avg_outcome_7d:+.2f}%" if sp.avg_outcome_7d else "N/A"
            win_rate = f"{sp.win_rate_7d:.0f}%" if sp.win_rate_7d else "N/A"
            lines.append(f"  {sp.sentiment}: {outcome} avg (win rate: {win_rate}, n={sp.total_decisions})")
        lines.append("")

    # Top tickers
    ticker_perf = analyze_ticker_performance(days)
    if ticker_perf:
        lines.append("Ticker Performance (by total P&L):")
        for tp in ticker_perf[:5]:
            total = f"{tp.total_pnl_7d:+.2f}%" if tp.total_pnl_7d else "N/A"
            lines.append(f"  {tp.ticker}: {total} total ({tp.buys} buys, {tp.sells} sells)")
        lines.append("")

    # Confidence correlation
    conf_perf = analyze_confidence_correlation(days)
    if conf_perf:
        lines.append("Confidence vs Outcomes:")
        for cp in conf_perf:
            outcome = f"{cp.avg_outcome_7d:+.2f}%" if cp.avg_outcome_7d else "N/A"
            win_rate = f"{cp.win_rate_7d:.0f}%" if cp.win_rate_7d else "N/A"
            lines.append(f"  {cp.confidence}: {outcome} avg (win rate: {win_rate})")
        lines.append("")

    # Best/worst signals
    best = get_best_performing_signals(days)
    if best:
        lines.append("Best Performing Signal Categories:")
        for b in best[:3]:
            lines.append(f"  {b['category']}: {float(b['avg_outcome']):+.2f}% avg (n={b['occurrences']})")
        lines.append("")

    worst = get_worst_performing_signals(days)
    if worst:
        lines.append("Worst Performing Signal Categories:")
        for w in worst[:3]:
            lines.append(f"  {w['category']}: {float(w['avg_outcome']):+.2f}% avg (n={w['occurrences']})")

    return "\n".join(lines)
