"""Strategy evolution based on learning from past performance."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .db import get_cursor
from .patterns import (
    analyze_ticker_performance,
    analyze_signal_categories,
    analyze_sentiment_performance,
    get_best_performing_signals,
    get_worst_performing_signals,
)


@dataclass
class StrategyRecommendation:
    """Recommended strategy adjustments."""
    watchlist: list[str]
    avoid_list: list[str]
    risk_tolerance: str
    focus_sectors: list[str]
    description: str
    reasoning: list[str]


def get_current_strategy() -> dict | None:
    """Get the current strategy from database."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT date, description, watchlist, risk_tolerance, focus_sectors
            FROM strategy
            ORDER BY date DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        return dict(row) if row else None


def save_strategy(
    description: str,
    watchlist: list[str],
    risk_tolerance: str,
    focus_sectors: list[str]
) -> int:
    """Save a new strategy to database."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy (date, description, watchlist, risk_tolerance, focus_sectors)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (date.today(), description, watchlist, risk_tolerance, focus_sectors))
        return cur.fetchone()["id"]


def calculate_overall_performance(days: int = 30) -> dict:
    """Calculate overall trading performance metrics."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total_trades,
                AVG(outcome_7d) as avg_return,
                AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                SUM(outcome_7d) as total_return
            FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '%s days'
              AND action IN ('buy', 'sell')
              AND outcome_7d IS NOT NULL
        """, (days,))
        row = cur.fetchone()

        return {
            "total_trades": row["total_trades"] or 0,
            "avg_return": float(row["avg_return"]) if row["avg_return"] else 0,
            "win_rate": float(row["win_rate"]) if row["win_rate"] else 0,
            "total_return": float(row["total_return"]) if row["total_return"] else 0,
        }


def determine_risk_tolerance(performance: dict) -> str:
    """
    Determine appropriate risk tolerance based on recent performance.

    - If win rate > 60% and avg return > 2%: aggressive
    - If win rate > 50% and avg return > 0%: moderate
    - Otherwise: conservative
    """
    win_rate = performance["win_rate"]
    avg_return = performance["avg_return"]

    if win_rate > 0.6 and avg_return > 2:
        return "aggressive"
    elif win_rate > 0.5 and avg_return > 0:
        return "moderate"
    else:
        return "conservative"


def build_watchlist(days: int = 60, max_tickers: int = 10) -> list[str]:
    """
    Build watchlist based on best performing tickers.
    """
    ticker_perf = analyze_ticker_performance(days)

    # Filter to tickers with positive outcomes and at least 2 decisions
    good_tickers = [
        tp.ticker for tp in ticker_perf
        if tp.avg_outcome_7d and tp.avg_outcome_7d > 0 and tp.total_decisions >= 2
    ]

    return good_tickers[:max_tickers]


def build_avoid_list(days: int = 60, max_tickers: int = 5) -> list[str]:
    """
    Build list of tickers to avoid based on poor performance.
    """
    ticker_perf = analyze_ticker_performance(days)

    # Filter to tickers with negative outcomes
    bad_tickers = [
        tp.ticker for tp in ticker_perf
        if tp.avg_outcome_7d and tp.avg_outcome_7d < -2 and tp.total_decisions >= 2
    ]

    return bad_tickers[:max_tickers]


def identify_focus_sectors(days: int = 60) -> list[str]:
    """
    Identify sectors to focus on based on macro signal performance.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT DISTINCT unnest(affected_sectors) as sector
            FROM macro_signals
            WHERE sentiment = 'bullish'
              AND published_at > NOW() - INTERVAL '%s days'
        """, (days,))
        bullish_sectors = [row["sector"] for row in cur.fetchall()]

    # Common sector mapping
    sector_priority = ["tech", "finance", "healthcare", "energy", "consumer", "defense"]

    # Return sectors that appear in bullish signals, maintaining priority order
    return [s for s in sector_priority if s in bullish_sectors][:3] or ["tech"]


def generate_strategy_recommendation(days: int = 60) -> StrategyRecommendation:
    """
    Generate strategy recommendation based on pattern analysis.
    """
    reasoning = []

    # Analyze overall performance
    performance = calculate_overall_performance(days)
    reasoning.append(f"Overall: {performance['total_trades']} trades, "
                    f"{performance['win_rate']*100:.0f}% win rate, "
                    f"{performance['avg_return']:+.2f}% avg return")

    # Determine risk tolerance
    risk_tolerance = determine_risk_tolerance(performance)
    reasoning.append(f"Risk tolerance set to {risk_tolerance} based on performance")

    # Build watchlist
    watchlist = build_watchlist(days)
    if watchlist:
        reasoning.append(f"Watchlist: {', '.join(watchlist)} (positive historical performance)")

    # Build avoid list
    avoid_list = build_avoid_list(days)
    if avoid_list:
        reasoning.append(f"Avoiding: {', '.join(avoid_list)} (negative historical performance)")

    # Identify focus sectors
    focus_sectors = identify_focus_sectors(days)
    reasoning.append(f"Focus sectors: {', '.join(focus_sectors)}")

    # Analyze best/worst signal patterns
    best_signals = get_best_performing_signals(days, min_occurrences=2)
    if best_signals:
        best = best_signals[0]
        reasoning.append(f"Best signal pattern: {best['category']}+{best['sentiment']} "
                        f"({best['avg_outcome']:+.2f}%)")

    worst_signals = get_worst_performing_signals(days, min_occurrences=2)
    if worst_signals:
        worst = worst_signals[0]
        reasoning.append(f"Avoid signal pattern: {worst['category']}+{worst['sentiment']} "
                        f"({worst['avg_outcome']:+.2f}%)")

    # Build description
    if performance["win_rate"] > 0.5:
        description = f"Momentum strategy with {risk_tolerance} risk. "
    else:
        description = f"Conservative value strategy. "

    if focus_sectors:
        description += f"Focus on {', '.join(focus_sectors)} sectors. "

    if watchlist:
        description += f"Prioritize {', '.join(watchlist[:3])}."

    return StrategyRecommendation(
        watchlist=watchlist,
        avoid_list=avoid_list,
        risk_tolerance=risk_tolerance,
        focus_sectors=focus_sectors,
        description=description,
        reasoning=reasoning,
    )


def evolve_strategy(days: int = 60, dry_run: bool = False) -> StrategyRecommendation:
    """
    Analyze patterns and evolve the trading strategy.

    Args:
        days: Days of history to analyze
        dry_run: If True, don't save to database

    Returns:
        The new strategy recommendation
    """
    print(f"Analyzing {days} days of trading history...")

    recommendation = generate_strategy_recommendation(days)

    print("\nStrategy Recommendation:")
    print("-" * 40)
    print(f"Description: {recommendation.description}")
    print(f"Risk Tolerance: {recommendation.risk_tolerance}")
    print(f"Watchlist: {', '.join(recommendation.watchlist) or 'None'}")
    print(f"Avoid: {', '.join(recommendation.avoid_list) or 'None'}")
    print(f"Focus Sectors: {', '.join(recommendation.focus_sectors)}")
    print("\nReasoning:")
    for reason in recommendation.reasoning:
        print(f"  - {reason}")

    if not dry_run:
        strategy_id = save_strategy(
            description=recommendation.description,
            watchlist=recommendation.watchlist,
            risk_tolerance=recommendation.risk_tolerance,
            focus_sectors=recommendation.focus_sectors,
        )
        print(f"\nStrategy saved (ID: {strategy_id})")
    else:
        print("\n[DRY RUN] Strategy not saved")

    return recommendation
