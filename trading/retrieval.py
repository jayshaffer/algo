"""Retrieval layer for RAG - vector search with time decay."""

from datetime import datetime
from typing import Optional

from .db import get_cursor
from .ollama import embed

# Time decay: documents lose 50% relevance after this many days
HALF_LIFE_DAYS = 30


def retrieve_by_ticker(
    ticker: str,
    k: int = 10,
    doc_types: Optional[list[str]] = None,
    days: int = 180,
) -> list[dict]:
    """
    Retrieve recent documents for a specific ticker.

    Returns documents sorted by recency (time-decayed).
    """
    with get_cursor() as cur:
        type_filter = ""
        params = [HALF_LIFE_DAYS, ticker, days]

        if doc_types:
            placeholders = ", ".join(["%s"] * len(doc_types))
            type_filter = f"AND doc_type IN ({placeholders})"
            params.extend(doc_types)

        params.append(k)

        cur.execute(f"""
            SELECT
                id, content, doc_type, source, source_url, published_at,
                EXP(-EXTRACT(EPOCH FROM (NOW() - published_at)) / (86400 * %s)) as recency_score
            FROM documents
            WHERE ticker = %s
              AND published_at > NOW() - INTERVAL '%s days'
              {type_filter}
            ORDER BY recency_score DESC
            LIMIT %s
        """, params)

        return cur.fetchall()


def retrieve_by_query(
    query: str,
    k: int = 10,
    tickers: Optional[list[str]] = None,
    doc_types: Optional[list[str]] = None,
    days: int = 180,
) -> list[dict]:
    """
    Semantic search across all documents.

    Combines vector similarity with time decay:
        final_score = cosine_similarity * recency_score
    """
    query_embedding = embed(query)

    with get_cursor() as cur:
        filters = ["published_at > NOW() - INTERVAL '%s days'"]
        params = [days]

        if tickers:
            placeholders = ", ".join(["%s"] * len(tickers))
            filters.append(f"ticker IN ({placeholders})")
            params.extend(tickers)

        if doc_types:
            placeholders = ", ".join(["%s"] * len(doc_types))
            filters.append(f"doc_type IN ({placeholders})")
            params.extend(doc_types)

        where_clause = " AND ".join(filters)

        # pgvector uses <=> for cosine distance (0 = identical, 2 = opposite)
        # Convert to similarity: 1 - distance
        cur.execute(f"""
            SELECT
                id, content, ticker, doc_type, source, source_url, published_at,
                1 - (embedding <=> %s::vector) as similarity,
                EXP(-EXTRACT(EPOCH FROM (NOW() - published_at)) / (86400 * %s)) as recency_score,
                (1 - (embedding <=> %s::vector)) *
                    EXP(-EXTRACT(EPOCH FROM (NOW() - published_at)) / (86400 * %s)) as final_score
            FROM documents
            WHERE {where_clause}
              AND embedding IS NOT NULL
            ORDER BY final_score DESC
            LIMIT %s
        """, [query_embedding, HALF_LIFE_DAYS, query_embedding, HALF_LIFE_DAYS] + params + [k])

        return cur.fetchall()


def retrieve_for_ideation(
    tickers: list[str],
    themes: list[str],
    k_per_ticker: int = 5,
    k_per_theme: int = 5,
) -> dict:
    """
    Combined retrieval for ideation session.

    Args:
        tickers: Tickers to get specific docs for (positions, thesis tickers)
        themes: Semantic queries ("AI semiconductor demand", "Fed rate cuts")

    Returns:
        {
            "by_ticker": {ticker: [docs]},
            "by_theme": {theme: [docs]},
        }
    """
    result = {"by_ticker": {}, "by_theme": {}}

    for ticker in tickers:
        docs = retrieve_by_ticker(ticker, k=k_per_ticker)
        if docs:
            result["by_ticker"][ticker] = docs

    for theme in themes:
        docs = retrieve_by_query(theme, k=k_per_theme)
        if docs:
            result["by_theme"][theme] = docs

    return result
