"""Scheduled document ingestion job."""

import argparse
from datetime import datetime

from .db import get_positions, get_active_theses
from .ingest import (
    ingest_alpaca_news,
    ingest_sec_filings,
    cleanup_old_documents,
    get_document_stats,
)
from .market_data import get_default_universe


def get_tickers_to_ingest() -> list[str]:
    """
    Get tickers that need document ingestion.

    Includes: positions, active theses, default universe.
    """
    tickers = set(get_default_universe())

    # Add position tickers
    for pos in get_positions():
        tickers.add(pos["ticker"])

    # Add thesis tickers
    for thesis in get_active_theses():
        tickers.add(thesis["ticker"])

    return sorted(tickers)


def run_ingestion(
    news_days: int = 3,
    filings: bool = True,
    cleanup: bool = True,
    retention_days: int = 180,
    max_filing_tickers: int = 20,
) -> dict:
    """
    Run full ingestion cycle.

    Args:
        news_days: Days of news to fetch
        filings: Whether to check for new SEC filings
        cleanup: Whether to remove old documents
        retention_days: Document retention period
        max_filing_tickers: Max tickers to fetch filings for (rate limit)

    Returns:
        Summary of ingestion results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "news_ingested": 0,
        "filings_ingested": 0,
        "documents_cleaned": 0,
        "errors": [],
    }

    tickers = get_tickers_to_ingest()
    print(f"[Ingestion] Processing {len(tickers)} tickers")
    print(f"  News days: {news_days}")
    print(f"  Filings: {filings}")
    print(f"  Cleanup: {cleanup} (retention: {retention_days} days)")

    # Ingest news
    print(f"\n[Step 1] Ingesting Alpaca news...")
    try:
        results["news_ingested"] = ingest_alpaca_news(tickers, days=news_days)
        print(f"  Ingested {results['news_ingested']} news documents")
    except Exception as e:
        results["errors"].append(f"News ingestion failed: {e}")
        print(f"  Error: {e}")

    # Ingest SEC filings (limited to avoid rate limits)
    if filings:
        print(f"\n[Step 2] Ingesting SEC filings (max {max_filing_tickers} tickers)...")
        for ticker in tickers[:max_filing_tickers]:
            try:
                count = ingest_sec_filings(ticker)
                results["filings_ingested"] += count
            except Exception as e:
                results["errors"].append(f"SEC filing {ticker}: {e}")
        print(f"  Ingested {results['filings_ingested']} filing documents")

    # Cleanup old documents
    if cleanup:
        print(f"\n[Step 3] Cleaning documents older than {retention_days} days...")
        try:
            results["documents_cleaned"] = cleanup_old_documents(retention_days)
            print(f"  Removed {results['documents_cleaned']} old documents")
        except Exception as e:
            results["errors"].append(f"Cleanup failed: {e}")
            print(f"  Error: {e}")

    # Summary
    stats = get_document_stats()
    print("\n" + "=" * 50)
    print("Ingestion Complete")
    print("=" * 50)
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Unique tickers: {stats['unique_tickers']}")
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}")
        for err in results["errors"]:
            print(f"    - {err}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run document ingestion")
    parser.add_argument("--news-days", type=int, default=3, help="Days of news to fetch")
    parser.add_argument("--skip-filings", action="store_true", help="Skip SEC filings")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip old doc cleanup")
    parser.add_argument("--retention", type=int, default=180, help="Retention days")
    parser.add_argument("--stats", action="store_true", help="Show stats only")

    args = parser.parse_args()

    if args.stats:
        stats = get_document_stats()
        print(f"Total documents: {stats['total_documents']}")
        print(f"Unique tickers: {stats['unique_tickers']}")
        for doc_type, info in stats.get("by_type", {}).items():
            print(f"  {doc_type}: {info['count']} docs")
        return

    run_ingestion(
        news_days=args.news_days,
        filings=not args.skip_filings,
        cleanup=not args.skip_cleanup,
        retention_days=args.retention,
    )


if __name__ == "__main__":
    main()
