"""Pattern analysis for learning from past decisions."""

from dataclasses import dataclass
from decimal import Decimal

from .db import get_cursor


@dataclass
class SignalPerformance:
    """Performance metrics for a signal category."""
    category: str
    total_signals: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    win_rate_7d: float | None  # % of positive outcomes
    win_rate_30d: float | None


@dataclass
class SentimentPerformance:
    """Performance by sentiment."""
    sentiment: str
    total_decisions: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    win_rate_7d: float | None


@dataclass
class TickerPerformance:
    """Performance by ticker."""
    ticker: str
    total_decisions: int
    buys: int
    sells: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    total_pnl_7d: float | None


@dataclass
class ConfidenceCorrelation:
    """Correlation between stated confidence and actual outcomes."""
    confidence: str
    total_decisions: int
    avg_outcome_7d: float | None
    win_rate_7d: float | None


def analyze_signal_categories(days: int = 90) -> list[SignalPerformance]:
    """
    Analyze which signal categories lead to profitable trades.

    Joins decisions with the signals that informed them.
    """
    with get_cursor() as cur:
        cur.execute("""
            WITH decision_signals AS (
                SELECT
                    d.id as decision_id,
                    d.outcome_7d,
                    d.outcome_30d,
                    ns.category
                FROM decisions d
                JOIN news_signals ns ON ns.ticker = d.ticker
                    AND ns.published_at::date <= d.date
                    AND ns.published_at::date >= d.date - INTERVAL '7 days'
                WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
                  AND d.action IN ('buy', 'sell')
            )
            SELECT
                category,
                COUNT(DISTINCT decision_id) as total_signals,
                AVG(outcome_7d) as avg_outcome_7d,
                AVG(outcome_30d) as avg_outcome_30d,
                AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_7d,
                AVG(CASE WHEN outcome_30d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_30d
            FROM decision_signals
            WHERE category IS NOT NULL
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
    """
    Analyze how decisions based on different sentiments perform.
    """
    with get_cursor() as cur:
        cur.execute("""
            WITH decision_sentiment AS (
                SELECT
                    d.id as decision_id,
                    d.outcome_7d,
                    d.outcome_30d,
                    ns.sentiment
                FROM decisions d
                JOIN news_signals ns ON ns.ticker = d.ticker
                    AND ns.published_at::date <= d.date
                    AND ns.published_at::date >= d.date - INTERVAL '3 days'
                WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
                  AND d.action IN ('buy', 'sell')
            )
            SELECT
                sentiment,
                COUNT(DISTINCT decision_id) as total_decisions,
                AVG(outcome_7d) as avg_outcome_7d,
                AVG(outcome_30d) as avg_outcome_30d,
                AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_7d
            FROM decision_sentiment
            WHERE sentiment IS NOT NULL
            GROUP BY sentiment
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
    """
    Analyze performance by ticker.
    """
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
    """
    Analyze correlation between stated confidence and actual outcomes.
    """
    with get_cursor() as cur:
        # Extract confidence from signals_used JSONB or join with news_signals
        cur.execute("""
            WITH decision_confidence AS (
                SELECT
                    d.id,
                    d.outcome_7d,
                    ns.confidence
                FROM decisions d
                JOIN news_signals ns ON ns.ticker = d.ticker
                    AND ns.published_at::date <= d.date
                    AND ns.published_at::date >= d.date - INTERVAL '3 days'
                WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
                  AND d.action IN ('buy', 'sell')
            )
            SELECT
                confidence,
                COUNT(DISTINCT id) as total_decisions,
                AVG(outcome_7d) as avg_outcome_7d,
                AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_7d
            FROM decision_confidence
            WHERE confidence IS NOT NULL
            GROUP BY confidence
            ORDER BY
                CASE confidence
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
    """
    Get the signal types that have led to the best outcomes.

    Returns list of dicts with category, sentiment, avg_outcome, count.
    """
    with get_cursor() as cur:
        cur.execute("""
            WITH signal_outcomes AS (
                SELECT
                    ns.category,
                    ns.sentiment,
                    d.outcome_7d
                FROM decisions d
                JOIN news_signals ns ON ns.ticker = d.ticker
                    AND ns.published_at::date <= d.date
                    AND ns.published_at::date >= d.date - INTERVAL '7 days'
                WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
                  AND d.action = 'buy'
                  AND d.outcome_7d IS NOT NULL
            )
            SELECT
                category,
                sentiment,
                AVG(outcome_7d) as avg_outcome,
                COUNT(*) as occurrences
            FROM signal_outcomes
            GROUP BY category, sentiment
            HAVING COUNT(*) >= %s
            ORDER BY avg_outcome DESC
            LIMIT 10
        """, (days, min_occurrences))

        return [dict(row) for row in cur.fetchall()]


def get_worst_performing_signals(days: int = 90, min_occurrences: int = 3) -> list[dict]:
    """
    Get the signal types that have led to the worst outcomes.
    """
    with get_cursor() as cur:
        cur.execute("""
            WITH signal_outcomes AS (
                SELECT
                    ns.category,
                    ns.sentiment,
                    d.outcome_7d
                FROM decisions d
                JOIN news_signals ns ON ns.ticker = d.ticker
                    AND ns.published_at::date <= d.date
                    AND ns.published_at::date >= d.date - INTERVAL '7 days'
                WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
                  AND d.action = 'buy'
                  AND d.outcome_7d IS NOT NULL
            )
            SELECT
                category,
                sentiment,
                AVG(outcome_7d) as avg_outcome,
                COUNT(*) as occurrences
            FROM signal_outcomes
            GROUP BY category, sentiment
            HAVING COUNT(*) >= %s
            ORDER BY avg_outcome ASC
            LIMIT 10
        """, (days, min_occurrences))

        return [dict(row) for row in cur.fetchall()]


def generate_pattern_report(days: int = 90) -> str:
    """
    Generate a human-readable pattern analysis report.
    """
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
        lines.append("Best Performing Signal Combinations:")
        for b in best[:3]:
            lines.append(f"  {b['category']} + {b['sentiment']}: {b['avg_outcome']:+.2f}% (n={b['occurrences']})")
        lines.append("")

    worst = get_worst_performing_signals(days)
    if worst:
        lines.append("Worst Performing Signal Combinations:")
        for w in worst[:3]:
            lines.append(f"  {w['category']} + {w['sentiment']}: {w['avg_outcome']:+.2f}% (n={w['occurrences']})")

    return "\n".join(lines)
