"""Signal attribution engine — computes which signal types are predictive."""

from decimal import Decimal

from .database.trading_db import upsert_signal_attribution, get_signal_attribution
from .database.connection import get_cursor


def compute_signal_attribution(days: int = 90) -> list[dict]:
    """
    Compute signal attribution scores from decision_signals joined with decisions.

    Groups by composite category (signal_type + news category for news signals).
    JOINs through decision_signals FK — not time-window JOINs.
    Upserts results into signal_attribution table.
    """
    from datetime import date, timedelta
    cutoff_date = date.today() - timedelta(days=days)

    with get_cursor() as cur:
        cur.execute("""
            WITH categorized AS (
                SELECT
                    ds.decision_id,
                    CASE
                        WHEN ds.signal_type = 'news_signal' THEN
                            'news_signal:' || COALESCE(ns.category, 'unknown') || ':' || d.action
                        WHEN ds.signal_type = 'macro_signal' THEN
                            'macro_signal:' || COALESCE(ms.category, 'unknown') || ':' || d.action
                        ELSE ds.signal_type || ':' || d.action
                    END AS category,
                    d.outcome_7d,
                    d.outcome_30d
                FROM decision_signals ds
                JOIN decisions d ON d.id = ds.decision_id
                LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
                LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
                WHERE d.action IN ('buy', 'sell')
                  AND d.date >= %s
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
        """, (cutoff_date,))
        results = [dict(row) for row in cur.fetchall()]

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
    """Format attribution scores as advisory text for LLM context."""
    rows = get_signal_attribution()
    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    lines = ["Signal Attribution:"]
    profitable = [r for r in rows if r.get("avg_outcome_7d") and float(r["avg_outcome_7d"]) > 0]
    unprofitable = [r for r in rows if r.get("avg_outcome_7d") and float(r["avg_outcome_7d"]) <= 0]

    if profitable:
        lines.append("Profitable signal types (positive avg 7d return):")
        for r in profitable:
            avg = float(r.get("avg_outcome_7d") or 0)
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0
            lines.append(f"  - {r['category']}: {avg:+.2f}% avg 7d return, {wr:.0f}% win rate (n={r['sample_size']})")

    if unprofitable:
        lines.append("Unprofitable signal types (negative avg 7d return):")
        for r in unprofitable:
            avg = float(r.get("avg_outcome_7d") or 0)
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0
            lines.append(f"  - {r['category']}: {avg:+.2f}% avg 7d return, {wr:.0f}% win rate (n={r['sample_size']})")

    return "\n".join(lines)


def build_attribution_constraints(min_samples: int = 5) -> str:
    """Format signal attribution into constraint block for strategist system prompt.

    Uses expected value (avg_outcome_7d) as the primary metric instead of win rate.

    Categories:
      STRONG: avg 7d return > +0.5%, >= min_samples
      WEAK: avg 7d return < -0.5%, >= min_samples
      INSUFFICIENT DATA: < min_samples
    """
    rows = get_signal_attribution()
    if not rows:
        return ""

    strong, weak, insufficient = [], [], []

    for r in rows:
        cat = r["category"]
        n = r["sample_size"]
        avg = float(r["avg_outcome_7d"]) if r.get("avg_outcome_7d") else 0
        wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0

        if n < min_samples:
            insufficient.append(f"{cat} (n={n})")
        elif avg > 0.5:
            strong.append(f"{cat} ({avg:+.2f}% avg 7d return, {wr:.0f}% win rate, n={n})")
        elif avg < -0.5:
            weak.append(f"{cat} ({avg:+.2f}% avg 7d return, {wr:.0f}% win rate, n={n})")

    lines = ["SIGNAL PERFORMANCE (rolling window):"]
    if strong:
        lines.append(f"  STRONG (positive avg return): {', '.join(strong)}")
    if weak:
        lines.append(f"  WEAK (negative avg return): {', '.join(weak)}")
    if insufficient:
        lines.append(f"  INSUFFICIENT DATA (<{min_samples} samples): {', '.join(insufficient)}")

    lines.append("")
    lines.append("CONSTRAINT: Do not create theses primarily based on WEAK signal categories")
    lines.append("(negative avg 7d return) unless you have a specific reason to override (explain in thesis text).")

    return "\n".join(lines)


def main():
    """CLI entry point for attribution."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute signal attribution scores")
    parser.add_argument("--constraints", action="store_true", help="Also print attribution constraints")
    args = parser.parse_args()

    print("Computing signal attribution...")
    results = compute_signal_attribution()
    print(f"  Updated {len(results)} categories")

    print()
    print(get_attribution_summary())

    if args.constraints:
        print()
        constraints = build_attribution_constraints()
        print(constraints if constraints else "No constraints generated (insufficient data)")


if __name__ == "__main__":
    main()
