"""News pipeline orchestrator - fetch, filter, classify, store."""

import sys
from datetime import datetime
from dataclasses import dataclass

from .news import fetch_broad_news, NewsItem
from .filter import filter_by_relevance, FilteredNewsItem, DEFAULT_STRATEGY_CONTEXT
from .classifier import classify_filtered_news, ClassificationResult
from .db import insert_news_signal, insert_macro_signal
from .ollama import check_ollama_health, list_models


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    news_fetched: int
    news_filtered: int
    ticker_signals_stored: int
    macro_signals_stored: int
    noise_dropped: int
    errors: int


def run_pipeline(
    hours: int = 24,
    limit: int = 300,
    relevance_threshold: float = 0.3,
    strategy_context: str = DEFAULT_STRATEGY_CONTEXT,
    dry_run: bool = False
) -> PipelineStats:
    """
    Run the full news processing pipeline.

    1. Fetch news from Alpaca (broad coverage, no symbol filter)
    2. Filter by embedding relevance
    3. Classify with Phi-3
    4. Store signals in database

    Args:
        hours: How far back to fetch news
        limit: Max news items to fetch
        relevance_threshold: Min similarity score to keep
        strategy_context: Text defining relevant topics
        dry_run: If True, don't store to database

    Returns:
        PipelineStats with counts
    """
    stats = PipelineStats(
        news_fetched=0,
        news_filtered=0,
        ticker_signals_stored=0,
        macro_signals_stored=0,
        noise_dropped=0,
        errors=0
    )

    print(f"[{datetime.now().isoformat()}] Starting news pipeline")
    print(f"  Hours: {hours}, Limit: {limit}, Threshold: {relevance_threshold}")

    # Step 1: Fetch news
    print("\n[Step 1] Fetching news from Alpaca...")
    try:
        news_items = fetch_broad_news(hours=hours, limit=limit)
        stats.news_fetched = len(news_items)
        print(f"  Fetched {stats.news_fetched} news items")
    except Exception as e:
        print(f"  Error fetching news: {e}")
        stats.errors += 1
        return stats

    if not news_items:
        print("  No news items to process")
        return stats

    # Step 2: Filter by relevance
    print("\n[Step 2] Filtering by relevance...")
    try:
        filtered_items = filter_by_relevance(
            news_items,
            strategy_context=strategy_context,
            threshold=relevance_threshold
        )
        stats.news_filtered = len(filtered_items)
        dropped = stats.news_fetched - stats.news_filtered
        print(f"  Kept {stats.news_filtered} items, dropped {dropped} below threshold")
    except Exception as e:
        print(f"  Error filtering news: {e}")
        stats.errors += 1
        return stats

    if not filtered_items:
        print("  No items passed relevance filter")
        return stats

    # Step 3: Classify with Phi-3
    print("\n[Step 3] Classifying with Phi-3...")
    try:
        classifications = classify_filtered_news(filtered_items)
        print(f"  Classified {len(classifications)} items")
    except Exception as e:
        print(f"  Error classifying news: {e}")
        stats.errors += 1
        return stats

    # Step 4: Store signals
    print("\n[Step 4] Storing signals...")
    for result in classifications:
        if result.news_type == "noise":
            stats.noise_dropped += 1
            continue

        # Store ticker signals
        for signal in result.ticker_signals:
            if dry_run:
                print(f"  [DRY RUN] Ticker signal: {signal.ticker} - {signal.sentiment} ({signal.category})")
            else:
                try:
                    insert_news_signal(
                        ticker=signal.ticker,
                        headline=signal.headline,
                        category=signal.category,
                        sentiment=signal.sentiment,
                        confidence=signal.confidence,
                        published_at=signal.published_at
                    )
                    stats.ticker_signals_stored += 1
                except Exception as e:
                    print(f"  Error storing ticker signal: {e}")
                    stats.errors += 1

        # Store macro signal
        if result.macro_signal:
            signal = result.macro_signal
            if dry_run:
                print(f"  [DRY RUN] Macro signal: {signal.category} - {signal.sentiment}")
            else:
                try:
                    insert_macro_signal(
                        headline=signal.headline,
                        category=signal.category,
                        affected_sectors=signal.affected_sectors,
                        sentiment=signal.sentiment,
                        published_at=signal.published_at
                    )
                    stats.macro_signals_stored += 1
                except Exception as e:
                    print(f"  Error storing macro signal: {e}")
                    stats.errors += 1

    print(f"\n[Complete]")
    print(f"  Ticker signals: {stats.ticker_signals_stored}")
    print(f"  Macro signals: {stats.macro_signals_stored}")
    print(f"  Noise dropped: {stats.noise_dropped}")
    print(f"  Errors: {stats.errors}")

    return stats


def check_dependencies() -> bool:
    """Check that required services are available."""
    print("Checking dependencies...")

    # Check Ollama
    print("  Ollama: ", end="")
    if check_ollama_health():
        models = list_models()
        print(f"OK ({len(models)} models)")

        # Check for required models
        required = ["phi3:mini", "nomic-embed-text"]
        for model in required:
            if not any(model in m for m in models):
                print(f"  WARNING: Model '{model}' not found. Run setup-ollama.sh")
    else:
        print("NOT AVAILABLE")
        return False

    return True


def main():
    """CLI entry point for news pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run news processing pipeline")
    parser.add_argument("--hours", type=int, default=24, help="Hours of news to fetch")
    parser.add_argument("--limit", type=int, default=50, help="Max news items")
    parser.add_argument("--threshold", type=float, default=0.3, help="Relevance threshold")
    parser.add_argument("--dry-run", action="store_true", help="Don't store to database")
    parser.add_argument("--check", action="store_true", help="Only check dependencies")

    args = parser.parse_args()

    if args.check:
        if check_dependencies():
            print("\nAll dependencies OK")
            sys.exit(0)
        else:
            print("\nDependency check failed")
            sys.exit(1)

    # Run pipeline
    stats = run_pipeline(
        hours=args.hours,
        limit=args.limit,
        relevance_threshold=args.threshold,
        dry_run=args.dry_run
    )

    if stats.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
