"""News pipeline orchestrator - fetch, filter, classify, store."""

import logging
import sys
from datetime import datetime
from dataclasses import dataclass

from .news import fetch_broad_news, NewsItem
from .filter import filter_by_relevance, FilteredNewsItem, DEFAULT_STRATEGY_CONTEXT
from .classifier import classify_news_batch
from .db import insert_news_signals_batch, insert_macro_signals_batch
from .ollama import check_ollama_health, list_models

logger = logging.getLogger("pipeline")


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

    logger.info("Starting news pipeline (hours=%d, limit=%d, threshold=%.2f)",
                hours, limit, relevance_threshold)

    # Step 1: Fetch news
    logger.info("[Step 1] Fetching news from Alpaca")
    try:
        news_items = fetch_broad_news(hours=hours, limit=limit)
        stats.news_fetched = len(news_items)
        logger.info("Fetched %d news items", stats.news_fetched)
    except Exception as e:
        logger.error("Error fetching news: %s", e, exc_info=True)
        stats.errors += 1
        return stats

    if not news_items:
        logger.info("No news items to process")
        return stats

    # Step 2: Filter by relevance
    logger.info("[Step 2] Filtering by relevance")
    try:
        filtered_items = filter_by_relevance(
            news_items,
            strategy_context=strategy_context,
            threshold=relevance_threshold
        )
        stats.news_filtered = len(filtered_items)
        dropped = stats.news_fetched - stats.news_filtered
        logger.info("Kept %d items, dropped %d below threshold",
                     stats.news_filtered, dropped)
    except Exception as e:
        logger.error("Error filtering news: %s", e, exc_info=True)
        stats.errors += 1
        return stats

    if not filtered_items:
        logger.info("No items passed relevance filter")
        return stats

    # Step 3: Classify with qwen2.5:14b (batched)
    logger.info("[Step 3] Classifying with qwen2.5:14b (batched)")
    total = len(filtered_items)
    ticker_signals_batch = []
    macro_signals_batch = []

    headlines = [f.item.headline for f in filtered_items]
    published_ats = [f.item.published_at for f in filtered_items]

    results = classify_news_batch(headlines, published_ats)

    for result in results:
        if result.news_type == "noise":
            stats.noise_dropped += 1
            continue

        for signal in result.ticker_signals:
            if dry_run:
                logger.info("[DRY RUN] Ticker signal: %s - %s (%s)",
                            signal.ticker, signal.sentiment, signal.category)
            else:
                ticker_signals_batch.append((
                    signal.ticker, signal.headline, signal.category,
                    signal.sentiment, signal.confidence, signal.published_at
                ))

        if result.macro_signal:
            signal = result.macro_signal
            if dry_run:
                logger.info("[DRY RUN] Macro signal: %s - %s",
                            signal.category, signal.sentiment)
            else:
                macro_signals_batch.append((
                    signal.headline, signal.category, signal.affected_sectors,
                    signal.sentiment, signal.published_at
                ))

    logger.info("Classified %d items", total)

    # Step 4: Batch store signals (single transaction per type)
    if not dry_run:
        logger.info("[Step 4] Storing signals")
        try:
            stats.ticker_signals_stored = insert_news_signals_batch(ticker_signals_batch)
        except Exception as e:
            logger.error("Error batch-inserting ticker signals: %s", e, exc_info=True)
            stats.errors += 1

        try:
            stats.macro_signals_stored = insert_macro_signals_batch(macro_signals_batch)
        except Exception as e:
            logger.error("Error batch-inserting macro signals: %s", e, exc_info=True)
            stats.errors += 1

    logger.info("Pipeline complete — ticker: %d, macro: %d, noise: %d, errors: %d",
                stats.ticker_signals_stored, stats.macro_signals_stored,
                stats.noise_dropped, stats.errors)

    return stats


def check_dependencies() -> bool:
    """Check that required services are available."""
    logger.info("Checking dependencies")

    # Check Ollama
    if check_ollama_health():
        models = list_models()
        logger.info("Ollama: OK (%d models)", len(models))

        # Check for required models
        required = ["qwen2.5:14b", "nomic-embed-text"]
        for model in required:
            if not any(model in m for m in models):
                logger.warning("Model '%s' not found. Run setup-ollama.sh", model)
    else:
        logger.warning("Ollama: NOT AVAILABLE")
        return False

    return True


def main():
    """CLI entry point for news pipeline."""
    import argparse
    from .log_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run news processing pipeline")
    parser.add_argument("--hours", type=int, default=24, help="Hours of news to fetch")
    parser.add_argument("--limit", type=int, default=50, help="Max news items")
    parser.add_argument("--threshold", type=float, default=0.3, help="Relevance threshold")
    parser.add_argument("--dry-run", action="store_true", help="Don't store to database")
    parser.add_argument("--check", action="store_true", help="Only check dependencies")

    args = parser.parse_args()

    if args.check:
        if check_dependencies():
            logger.info("All dependencies OK")
            sys.exit(0)
        else:
            logger.error("Dependency check failed")
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
