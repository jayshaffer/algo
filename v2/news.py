"""Alpaca news fetching."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest


@dataclass
class NewsItem:
    """Structured news item from Alpaca."""
    id: str
    headline: str
    summary: str
    author: str
    source: str
    symbols: list[str]
    published_at: datetime
    url: str


def get_news_client() -> NewsClient:
    """Create Alpaca news client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    return NewsClient(api_key, secret_key)


def fetch_news(
    hours: int = 24,
    symbols: Optional[list[str]] = None,
    limit: int = 50
) -> list[NewsItem]:
    """Fetch news from Alpaca News API."""
    client = get_news_client()

    start_time = datetime.now() - timedelta(hours=hours)

    request_params = {
        "start": start_time,
        "limit": limit,
        "sort": "desc"
    }

    if symbols:
        request_params["symbols"] = symbols

    request = NewsRequest(**request_params)
    response = client.get_news(request)

    items = []
    for news in response.data.get("news", []):
        items.append(NewsItem(
            id=str(news.id),
            headline=news.headline,
            summary=news.summary or "",
            author=news.author or "",
            source=news.source or "",
            symbols=news.symbols or [],
            published_at=news.created_at,
            url=news.url or ""
        ))

    return items


def fetch_broad_news(hours: int = 24, limit: int = 50) -> list[NewsItem]:
    """Fetch broad market news without symbol filter."""
    return fetch_news(hours=hours, symbols=None, limit=limit)


def fetch_ticker_news(ticker: str, hours: int = 24, limit: int = 20) -> list[NewsItem]:
    """Fetch news for a specific ticker."""
    return fetch_news(hours=hours, symbols=[ticker], limit=limit)
