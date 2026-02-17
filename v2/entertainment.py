"""Entertainment tweet pipeline -- Bikini Bottom Capital.

Generates and posts entertaining Mr. Krabs tweets based on live market
news and trends, independent of the daily trading session.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from .claude_client import get_claude_client, _call_with_retry
from .news import fetch_broad_news
from .market_data import get_market_snapshot, format_market_snapshot
from .twitter import get_twitter_client, post_tweets
from .database.connection import get_cursor
from .database.trading_db import insert_tweet

logger = logging.getLogger("entertainment")


def gather_market_context(news_hours: int = 24, news_limit: int = 20) -> str:
    """Fetch live news headlines and market snapshot for tweet context."""
    sections = []

    # News headlines
    try:
        news_items = fetch_broad_news(hours=news_hours, limit=news_limit)
        if news_items:
            lines = ["NEWS HEADLINES:"]
            for item in news_items:
                tickers = ", ".join(item.symbols) if item.symbols else ""
                prefix = f"  [{tickers}] " if tickers else "  "
                lines.append(f"{prefix}{item.headline}")
            sections.append("\n".join(lines))
    except Exception as e:
        logger.warning("Failed to fetch news: %s", e)

    # Market data
    try:
        snapshot = get_market_snapshot()
        formatted = format_market_snapshot(snapshot)
        sections.append(f"MARKET DATA:\n{formatted}")
    except Exception as e:
        logger.warning("Failed to fetch market data: %s", e)

    if not sections:
        return "No market data available."

    return "\n\n".join(sections)
