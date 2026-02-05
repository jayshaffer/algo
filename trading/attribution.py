"""Signal attribution engine — computes which signal types are predictive."""

from decimal import Decimal

from .db import get_cursor, upsert_signal_attribution, get_signal_attribution


def compute_signal_attribution() -> list[dict]:
    """
    Compute signal attribution scores from decision_signals joined with decisions.

    Groups by signal category (signal_type + news category for news signals)
    and computes avg outcomes and win rates.

    Returns:
        List of attribution dicts, also upserts into signal_attribution table.
    """
    with get_cursor() as cur:
        cur.execute("""
            WITH categorized AS (
                SELECT
                    ds.decision_id,
                    CASE
                        WHEN ds.signal_type = 'news_signal' THEN
                            'news_signal:' || COALESCE(ns.category, 'unknown')
                        WHEN ds.signal_type = 'macro_signal' THEN
                            'macro_signal:' || COALESCE(ms.category, 'unknown')
                        ELSE ds.signal_type
                    END AS category,
                    d.outcome_7d,
                    d.outcome_30d
                FROM decision_signals ds
                JOIN decisions d ON d.id = ds.decision_id
                LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
                LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
                WHERE d.action IN ('buy', 'sell')
            )
            SELECT
                category,
                COUNT(DISTINCT decision_id) AS sample_size,
                AVG(outcome_7d) AS avg_outcome_7d,
                AVG(outcome_30d) AS avg_outcome_30d,
                AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_7d,
                AVG(CASE WHEN outcome_30d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_30d
            FROM categorized
            WHERE outcome_7d IS NOT NULL
            GROUP BY category
            ORDER BY sample_size DESC
        """)
        results = [dict(row) for row in cur.fetchall()]

    # Upsert each result into signal_attribution table
    for row in results:
        upsert_signal_attribution(
            category=row["category"],
            sample_size=row["sample_size"],
            avg_outcome_7d=row["avg_outcome_7d"] or Decimal(0),
            avg_outcome_30d=row["avg_outcome_30d"] or Decimal(0),
            win_rate_7d=row["win_rate_7d"] or Decimal(0),
            win_rate_30d=row["win_rate_30d"] or Decimal(0),
        )

    return results


def get_attribution_summary() -> str:
    """
    Format signal attribution scores as a text summary for LLM context.

    Returns:
        Formatted string with predictive and non-predictive signal categories.
    """
    rows = get_signal_attribution()

    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    lines = ["Signal Attribution:"]

    # Sort into predictive vs non-predictive
    predictive = [r for r in rows if r.get("win_rate_7d") and r["win_rate_7d"] > Decimal("0.5")]
    weak = [r for r in rows if r.get("win_rate_7d") and r["win_rate_7d"] <= Decimal("0.5")]

    if predictive:
        lines.append("Predictive signal types:")
        for r in predictive:
            wr = float(r["win_rate_7d"]) * 100
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: {wr:.0f}% win rate, {avg:+.2f}% avg 7d return (n={r['sample_size']})")

    if weak:
        lines.append("Weak/non-predictive signal types:")
        for r in weak:
            wr = float(r["win_rate_7d"]) * 100
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: {wr:.0f}% win rate, {avg:+.2f}% avg 7d return (n={r['sample_size']})")

    return "\n".join(lines)
