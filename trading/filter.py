"""Embedding-based relevance filter for news items."""

from dataclasses import dataclass

from .news import NewsItem
from .ollama import embed, embed_batch, cosine_similarity_batch


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

    Uses batch embedding + vectorized cosine similarity to score all
    headlines in one pass instead of individual API calls per item.

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

    # Batch embed all headlines in a single API call
    headlines = [item.headline for item in news_items]
    headline_embeddings = embed_batch(headlines)

    # Vectorized cosine similarity against all embeddings at once
    scores = cosine_similarity_batch(context_embedding, headline_embeddings)

    filtered = [
        FilteredNewsItem(item=item, relevance_score=score)
        for item, score in zip(news_items, scores)
        if score >= threshold
    ]

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

    Delegates to filter_by_relevance which already uses batch embedding.
    The batch_size parameter is kept for API compatibility but embedding
    is done in a single call regardless.

    Args:
        news_items: List of news items to filter
        strategy_context: Text describing what's relevant
        threshold: Minimum similarity threshold
        batch_size: Unused (kept for API compatibility)

    Returns:
        List of FilteredNewsItem passing the threshold
    """
    total = len(news_items)
    print(f"  Batch filtering {total} items...")

    filtered = filter_by_relevance(news_items, strategy_context, threshold)

    print(f"  Filtered {total} items, {len(filtered)} passed threshold")
    return filtered
