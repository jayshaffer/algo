"""Signal attribution engine — computes which signal types are predictive."""

from decimal import Decimal

from .database.connection import get_cursor
from .database.trading_db import get_signal_attribution, upsert_signal_attribution

# Per-session memo for `get_attribution_summary`. The attribution table is
# refreshed once at the top of each session (`compute_signal_attribution` in
# Stage 0) and read multiple times downstream — once by the strategist tool
# loop and once by `build_trading_context` / `build_executor_input`. Caching
# the formatted summary skips redundant DB roundtrips and Decimal formatting
# work. Explicit clear so callers can invalidate after a recompute.
_attribution_summary_cache: str | None = None


def clear_attribution_summary_cache() -> None:
    """Invalidate the memoized `get_attribution_summary` result."""
    global _attribution_summary_cache
    _attribution_summary_cache = None


def compute_signal_attribution(days: int = 90) -> list[dict]:
    """
    Compute signal attribution scores from decision_signals joined with decisions.

    Uses alpha (outcome - benchmark) instead of raw returns so attribution
    reflects signal quality independent of market conditions.

    Groups by 2-part category key (signal_type:subcategory), e.g.:
      - news_signal:earnings
      - macro_signal:geopolitical
      - thesis
    Sentiment and action are excluded from the key to prevent low-N fragmentation.
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
                            'news_signal:' || ns.category
                        WHEN ds.signal_type = 'macro_signal' THEN
                            'macro_signal:' || ms.category
                        ELSE ds.signal_type
                    END AS category,
                    d.outcome_7d,
                    d.outcome_30d,
                    d.benchmark_7d,
                    d.benchmark_30d,
                    CASE WHEN d.benchmark_7d IS NOT NULL
                         THEN d.outcome_7d - d.benchmark_7d
                         ELSE NULL END AS alpha_7d,
                    CASE WHEN d.benchmark_30d IS NOT NULL
                         THEN d.outcome_30d - d.benchmark_30d
                         ELSE NULL END AS alpha_30d
                FROM decision_signals ds
                JOIN decisions d ON d.id = ds.decision_id
                LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
                LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
                LEFT JOIN theses t ON ds.signal_type = 'thesis' AND t.id = ds.signal_id
                WHERE d.action IN ('buy', 'sell')
                  AND d.date >= %s
                  AND (ds.signal_type != 'news_signal' OR ns.id IS NOT NULL)
                  AND (ds.signal_type != 'macro_signal' OR ms.id IS NOT NULL)
                  AND (ds.signal_type != 'thesis' OR t.id IS NOT NULL)
            )
            SELECT
                category,
                COUNT(DISTINCT decision_id) AS sample_size,
                -- P2.17: 30d cohort is a subset; track its size separately so
                -- consumers don't read sample_size as supporting both metrics.
                COUNT(DISTINCT CASE WHEN alpha_30d IS NOT NULL THEN decision_id END)
                    AS sample_size_30d,
                AVG(alpha_7d) AS avg_outcome_7d,
                AVG(alpha_30d) AS avg_outcome_30d,
                AVG(CASE WHEN alpha_7d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_7d,
                -- Bug fix: NULL alpha_30d (decision too young for 30d outcome)
                -- must propagate through AVG, not collapse to ELSE 0.0.
                -- Without the explicit IS NULL guard, NULL > 0 evaluates to
                -- NULL → falls into ELSE → counted as a loss, biasing win
                -- rate toward 0 for any cohort whose 30d window hasn't closed.
                AVG(CASE WHEN alpha_30d IS NULL THEN NULL
                         WHEN alpha_30d > 0 THEN 1.0
                         ELSE 0.0 END) AS win_rate_30d
            FROM categorized
            WHERE outcome_7d IS NOT NULL
              AND alpha_7d IS NOT NULL
            GROUP BY category
            ORDER BY sample_size DESC
        """, (cutoff_date,))
        results = [dict(row) for row in cur.fetchall()]

    for row in results:
        # Pass NULL through for 30d metrics so "no eligible decisions" stays
        # distinguishable from "0% win rate". Coercing NULL→0 here would
        # silently re-introduce the bias the SQL fix above just removed.
        upsert_signal_attribution(
            category=row["category"],
            sample_size=row["sample_size"],
            sample_size_30d=row.get("sample_size_30d") or 0,
            avg_outcome_7d=row["avg_outcome_7d"] if row["avg_outcome_7d"] is not None else Decimal(0),
            avg_outcome_30d=row["avg_outcome_30d"],
            win_rate_7d=row["win_rate_7d"] if row["win_rate_7d"] is not None else Decimal(0),
            win_rate_30d=row["win_rate_30d"],
        )

    # New attribution numbers — invalidate the formatted-summary memo so the
    # next reader sees the fresh data.
    clear_attribution_summary_cache()

    return results


def get_attribution_summary() -> str:
    """Format attribution scores as advisory text for LLM context.

    All metrics are benchmark-relative (alpha vs SPY). A positive alpha means
    the signal outperformed the market; negative means it underperformed.

    Memoized per process: the strategist tool loop and `build_trading_context`
    both call this 1-2 times per session. Cache is invalidated by
    `compute_signal_attribution` whenever the table is recomputed.
    """
    global _attribution_summary_cache
    if _attribution_summary_cache is not None:
        return _attribution_summary_cache
    _attribution_summary_cache = _format_attribution_summary()
    return _attribution_summary_cache


def _format_attribution_summary() -> str:
    rows = get_signal_attribution()
    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    min_samples = 3
    sufficient = [r for r in rows if r["sample_size"] >= min_samples]
    insufficient = [r for r in rows if r["sample_size"] < min_samples]

    lines = ["Signal Attribution (alpha vs SPY benchmark):"]
    # P3.41: explicit `is not None` so a row with exactly 0 alpha lands in
    # underperforming (where `<= 0` covers it). The previous truthy check
    # short-circuited on Decimal(0) and the row was silently dropped from
    # both buckets — invisible to the strategist.
    outperforming = [r for r in sufficient if r.get("avg_outcome_7d") is not None and float(r["avg_outcome_7d"]) > 0]
    underperforming = [r for r in sufficient if r.get("avg_outcome_7d") is not None and float(r["avg_outcome_7d"]) <= 0]

    if outperforming:
        lines.append("Outperforming signals (positive alpha vs SPY):")
        for r in outperforming:
            avg = float(r["avg_outcome_7d"]) if r.get("avg_outcome_7d") is not None else 0.0
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") is not None else 0.0
            lines.append(f"  - {r['category']}: {avg:+.2f}% avg 7d alpha, {wr:.0f}% beat-market rate (n={r['sample_size']})")

    if underperforming:
        lines.append("Underperforming signals (negative alpha vs SPY):")
        for r in underperforming:
            avg = float(r["avg_outcome_7d"]) if r.get("avg_outcome_7d") is not None else 0.0
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") is not None else 0.0
            lines.append(f"  - {r['category']}: {avg:+.2f}% avg 7d alpha, {wr:.0f}% beat-market rate (n={r['sample_size']})")

    if insufficient:
        names = [f"{r['category']} (n={r['sample_size']})" for r in insufficient]
        lines.append(f"Insufficient data (<{min_samples} samples): {', '.join(names)}")

    return "\n".join(lines)


def build_attribution_constraints(min_samples: int = 5) -> str:
    """Format signal attribution into constraint block for strategist system prompt.

    All metrics are benchmark-relative (alpha vs SPY).

    Categories:
      STRONG: avg 7d alpha > +0.5%, >= min_samples
      WEAK: avg 7d alpha < -0.5%, >= min_samples
      INSUFFICIENT DATA: < min_samples
    """
    rows = get_signal_attribution()
    if not rows:
        return ""

    strong, weak, insufficient = [], [], []

    for r in rows:
        cat = r["category"]
        n = r["sample_size"]
        avg = float(r["avg_outcome_7d"]) if r.get("avg_outcome_7d") is not None else 0.0
        wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") is not None else 0.0

        if n < min_samples:
            insufficient.append(f"{cat} (n={n})")
        elif avg > 0.5:
            strong.append(f"{cat} ({avg:+.2f}% avg alpha, {wr:.0f}% beat-market rate, n={n})")
        elif avg < -0.5:
            weak.append(f"{cat} ({avg:+.2f}% avg alpha, {wr:.0f}% beat-market rate, n={n})")

    lines = ["SIGNAL PERFORMANCE (alpha vs SPY, rolling window):"]
    if strong:
        lines.append(f"  STRONG (outperforms market): {', '.join(strong)}")
    if weak:
        lines.append(f"  WEAK (underperforms market): {', '.join(weak)}")
    if insufficient:
        lines.append(f"  INSUFFICIENT DATA (<{min_samples} samples): {', '.join(insufficient)}")

    lines.append("")
    lines.append("CONSTRAINT: Do not create theses primarily based on WEAK signal categories")
    lines.append("(negative alpha vs SPY) unless you have a specific reason to override (explain in thesis text).")

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
