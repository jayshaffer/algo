"""Learning loop orchestrator - backfill and compute attribution."""

import sys
from dataclasses import dataclass
from datetime import datetime

from .backfill import run_backfill
from .attribution import compute_signal_attribution
from .patterns import generate_pattern_report


@dataclass
class LearningResult:
    """Result of a learning loop run."""
    timestamp: datetime
    outcomes_backfilled: int
    attribution_computed: int
    pattern_report: str
    errors: list[str]


def run_learning_loop(
    analysis_days: int = 60,
    dry_run: bool = False,
    skip_backfill: bool = False,
    skip_attribution: bool = False,
) -> LearningResult:
    """Run the complete learning loop."""
    timestamp = datetime.now()
    errors = []
    outcomes_backfilled = 0
    pattern_report = ""
    attribution_computed = 0

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

    # Step 3: Compute signal attribution
    if not skip_attribution:
        print("[Step 3] Computing signal attribution...")
        print("-" * 40)
        try:
            attribution_results = compute_signal_attribution()
            attribution_computed = len(attribution_results)
            print(f"  Computed attribution for {attribution_computed} signal categories")
        except Exception as e:
            errors.append(f"Attribution computation failed: {e}")
            print(f"Error: {e}")
        print()
    else:
        print("[Step 3] Skipping attribution computation")
        print()

    # Summary
    print("=" * 60)
    print("Learning Loop Complete")
    print("=" * 60)
    print(f"  Outcomes backfilled: {outcomes_backfilled}")
    print(f"  Attribution categories computed: {attribution_computed}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")

    return LearningResult(
        timestamp=timestamp,
        outcomes_backfilled=outcomes_backfilled,
        attribution_computed=attribution_computed,
        pattern_report=pattern_report,
        errors=errors,
    )


def main():
    """CLI entry point for learning loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Run learning loop")
    parser.add_argument("--days", type=int, default=60, help="Days of history to analyze")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--skip-backfill", action="store_true", help="Skip outcome backfill")
    parser.add_argument("--skip-attribution", action="store_true", help="Skip attribution computation")
    parser.add_argument("--patterns-only", action="store_true", help="Only run pattern analysis")

    args = parser.parse_args()

    if args.patterns_only:
        report = generate_pattern_report(days=args.days)
        print(report)
        return

    result = run_learning_loop(
        analysis_days=args.days,
        dry_run=args.dry_run,
        skip_backfill=args.skip_backfill,
        skip_attribution=args.skip_attribution,
    )

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
