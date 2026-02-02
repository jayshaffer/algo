"""Learning loop orchestrator - backfill, analyze, evolve."""

import sys
from dataclasses import dataclass
from datetime import datetime

from .backfill import run_backfill
from .patterns import generate_pattern_report
from .strategy import evolve_strategy, StrategyRecommendation


@dataclass
class LearningResult:
    """Result of a learning loop run."""
    timestamp: datetime
    outcomes_backfilled: int
    pattern_report: str
    strategy: StrategyRecommendation | None
    errors: list[str]


def run_learning_loop(
    analysis_days: int = 60,
    dry_run: bool = False,
    skip_backfill: bool = False,
    skip_strategy: bool = False,
) -> LearningResult:
    """
    Run the complete learning loop.

    1. Backfill decision outcomes (7d, 30d P&L)
    2. Analyze patterns in historical data
    3. Evolve strategy based on what's working

    Args:
        analysis_days: Days of history to analyze
        dry_run: If True, don't update database
        skip_backfill: Skip the backfill step
        skip_strategy: Skip strategy evolution

    Returns:
        LearningResult with details
    """
    timestamp = datetime.now()
    errors = []
    outcomes_backfilled = 0
    pattern_report = ""
    strategy = None

    print("=" * 60)
    print(f"Learning Loop - {timestamp.isoformat()}")
    print("=" * 60)
    print(f"  Analysis period: {analysis_days} days")
    print(f"  Dry run: {dry_run}")
    print()

    # Step 1: Backfill outcomes
    if not skip_backfill:
        print("[Step 1] Backfilling decision outcomes...")
        print("-" * 40)
        try:
            backfill_result = run_backfill(dry_run=dry_run)
            outcomes_backfilled = backfill_result["total_filled"]
        except Exception as e:
            errors.append(f"Backfill failed: {e}")
            print(f"Error: {e}")
        print()
    else:
        print("[Step 1] Skipping backfill")
        print()

    # Step 2: Analyze patterns
    print("[Step 2] Analyzing patterns...")
    print("-" * 40)
    try:
        pattern_report = generate_pattern_report(days=analysis_days)
        print(pattern_report)
    except Exception as e:
        errors.append(f"Pattern analysis failed: {e}")
        print(f"Error: {e}")
    print()

    # Step 3: Evolve strategy
    if not skip_strategy:
        print("[Step 3] Evolving strategy...")
        print("-" * 40)
        try:
            strategy = evolve_strategy(days=analysis_days, dry_run=dry_run)
        except Exception as e:
            errors.append(f"Strategy evolution failed: {e}")
            print(f"Error: {e}")
        print()
    else:
        print("[Step 3] Skipping strategy evolution")
        print()

    # Summary
    print("=" * 60)
    print("Learning Loop Complete")
    print("=" * 60)
    print(f"  Outcomes backfilled: {outcomes_backfilled}")
    print(f"  Strategy updated: {'Yes' if strategy and not dry_run else 'No'}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")

    return LearningResult(
        timestamp=timestamp,
        outcomes_backfilled=outcomes_backfilled,
        pattern_report=pattern_report,
        strategy=strategy,
        errors=errors,
    )


def main():
    """CLI entry point for learning loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Run learning loop")
    parser.add_argument("--days", type=int, default=60, help="Days of history to analyze")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--skip-backfill", action="store_true", help="Skip outcome backfill")
    parser.add_argument("--skip-strategy", action="store_true", help="Skip strategy evolution")
    parser.add_argument("--patterns-only", action="store_true", help="Only run pattern analysis")

    args = parser.parse_args()

    if args.patterns_only:
        # Just print pattern report
        report = generate_pattern_report(days=args.days)
        print(report)
        return

    result = run_learning_loop(
        analysis_days=args.days,
        dry_run=args.dry_run,
        skip_backfill=args.skip_backfill,
        skip_strategy=args.skip_strategy,
    )

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
