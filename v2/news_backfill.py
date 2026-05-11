"""One-shot backfill for the news_signals.summary column.

Re-fetches the past N hours of news from Alpaca and runs
UPDATE news_signals SET summary = ... WHERE alpaca_id = ... AND summary IS NULL
for each item whose alpaca_id matches an existing row.

Idempotent — the WHERE summary IS NULL guard means re-running is safe.
Run manually after schema migration + before relying on tool_get_curated_news.
"""
import logging

from .database.connection import get_cursor
from .news import fetch_news

logger = logging.getLogger(__name__)


def run(hours: int = 168) -> dict:
    """Backfill summary on existing news_signals rows.

    Args:
        hours: How far back to fetch from Alpaca. Default 7 days.

    Returns:
        {"fetched": int, "updated": int, "skipped_or_no_match": int}
    """
    logger.info("Starting news backfill (hours=%d)", hours)
    items = fetch_news(hours=hours, symbols=None, limit=10000)
    stats = {"fetched": len(items), "updated": 0, "skipped_or_no_match": 0}

    if not items:
        logger.info("No news items fetched; nothing to backfill")
        return stats

    for item in items:
        if not item.summary:
            stats["skipped_or_no_match"] += 1
            continue

        with get_cursor() as cur:
            cur.execute(
                """
                UPDATE news_signals
                   SET summary = %s
                 WHERE alpaca_id = %s
                   AND (summary IS NULL OR summary = '')
                """,
                (item.summary, item.id),
            )
            if cur.rowcount and cur.rowcount > 0:
                stats["updated"] += cur.rowcount
            else:
                stats["skipped_or_no_match"] += 1

    logger.info(
        "Backfill complete: fetched=%d updated=%d skipped/no-match=%d",
        stats["fetched"], stats["updated"], stats["skipped_or_no_match"],
    )
    return stats


def main():
    """CLI entry point."""
    import argparse

    from .log_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Backfill news_signals.summary from Alpaca")
    parser.add_argument("--hours", type=int, default=168, help="Hours to backfill (default 168 = 7 days)")
    args = parser.parse_args()

    stats = run(hours=args.hours)
    print(f"fetched={stats['fetched']} updated={stats['updated']} skipped={stats['skipped_or_no_match']}")


if __name__ == "__main__":
    main()
