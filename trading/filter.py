"""Embedding-based relevance filter for news items."""

from dataclasses import dataclass

from .news import NewsItem
from .ollama import embed, cosine_similarity


# Default strategy context for relevance filtering
DEFAULT_STRATEGY_CONTEXT = """
Stock market trading signals. Earnings reports, analyst ratings, guidance updates.
Federal Reserve policy, interest rates, inflation. Trade policy, tariffs, sanctions.
Sector news for technology, healthcare, finance, energy, defense.
Company acquisitions, mergers, IPOs, stock splits, dividends.
Economic indicators, employment data, GDP, consumer sentiment.
"""


@dataclass
class FilteredNewsItem:
    """News item with relevance score."""
    item: NewsItem
    relevance_score: float


def filter_by_relevance(
    news_items: list[NewsItem],
    strategy_context: str = DEFAULT_STRATEGY_CONTEXT,
    threshold: float = 0.3
) -> list[FilteredNewsItem]:
    """
    Filter news items by relevance to trading strategy.

    Uses embedding similarity to drop irrelevant news items.

    Args:
        news_items: List of news items to filter
        strategy_context: Text describing what's relevant to the strategy
        threshold: Minimum cosine similarity to keep (default 0.3)

    Returns:
        List of FilteredNewsItem with relevance scores, sorted by relevance
    """
    if not news_items:
        return []

    # Embed strategy context once
    context_embedding = embed(strategy_context)

    filtered = []
    for item in news_items:
        # Embed headline (more efficient than full summary)
        headline_embedding = embed(item.headline)

        # Calculate relevance
        score = cosine_similarity(context_embedding, headline_embedding)

        if score >= threshold:
            filtered.append(FilteredNewsItem(item=item, relevance_score=score))

    # Sort by relevance (highest first)
    filtered.sort(key=lambda x: x.relevance_score, reverse=True)

    return filtered


def filter_news_batch(
    news_items: list[NewsItem],
    strategy_context: str = DEFAULT_STRATEGY_CONTEXT,
    threshold: float = 0.3,
    batch_size: int = 10
) -> list[FilteredNewsItem]:
    """
    Filter news with progress tracking for larger batches.

    Args:
        news_items: List of news items to filter
        strategy_context: Text describing what's relevant
        threshold: Minimum similarity threshold
        batch_size: Items to process before yielding progress

    Returns:
        List of FilteredNewsItem passing the threshold
    """
    if not news_items:
        return []

    # Embed strategy context once
    context_embedding = embed(strategy_context)

    filtered = []
    total = len(news_items)

    for i, item in enumerate(news_items):
        headline_embedding = embed(item.headline)
        score = cosine_similarity(context_embedding, headline_embedding)

        if score >= threshold:
            filtered.append(FilteredNewsItem(item=item, relevance_score=score))

        # Progress logging
        if (i + 1) % batch_size == 0:
            print(f"  Filtered {i + 1}/{total} items, {len(filtered)} passed threshold")

    # Sort by relevance
    filtered.sort(key=lambda x: x.relevance_score, reverse=True)

    return filtered
