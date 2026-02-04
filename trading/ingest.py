"""Document ingestion for RAG - chunking, embedding, and storage."""

import os
from datetime import datetime, timedelta
from typing import Optional

import httpx

from .db import get_cursor
from .ollama import embed


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.

    Tries to break at sentence boundaries for cleaner chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings in the last quarter of the chunk
            search_start = start + (max_chars * 3 // 4)
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                idx = text.rfind(sep, search_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def document_exists(source_url: str) -> bool:
    """Check if a document with this source URL already exists."""
    if not source_url:
        return False

    with get_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM documents WHERE source_url = %s LIMIT 1",
            (source_url,)
        )
        return cur.fetchone() is not None


def ingest_document(
    content: str,
    ticker: Optional[str],
    doc_type: str,
    source: str,
    source_url: Optional[str],
    published_at: datetime,
) -> list[int]:
    """
    Ingest a document: chunk, embed, and store.

    Returns list of created document IDs.
    """
    # Skip if already ingested
    if source_url and document_exists(source_url):
        return []

    chunks = chunk_text(content)
    doc_ids = []
    parent_id = None

    with get_cursor() as cur:
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = embed(chunk)

            cur.execute("""
                INSERT INTO documents
                    (content, embedding, ticker, doc_type, source, source_url,
                     published_at, parent_id, chunk_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (chunk, embedding, ticker, doc_type, source, source_url,
                  published_at, parent_id, i))

            doc_id = cur.fetchone()["id"]
            doc_ids.append(doc_id)

            # First chunk becomes parent for subsequent chunks
            if parent_id is None:
                parent_id = doc_id

    return doc_ids


def get_document_stats() -> dict:
    """Get statistics about the document store."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                doc_type,
                COUNT(*) as count,
                COUNT(DISTINCT ticker) as tickers,
                MIN(published_at) as oldest,
                MAX(published_at) as newest
            FROM documents
            GROUP BY doc_type
        """)
        by_type = {row["doc_type"]: dict(row) for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) as total FROM documents")
        total = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(DISTINCT ticker) as tickers FROM documents WHERE ticker IS NOT NULL")
        unique_tickers = cur.fetchone()["tickers"]

    return {
        "total_documents": total,
        "unique_tickers": unique_tickers,
        "by_type": by_type,
    }


def cleanup_old_documents(days: int = 180) -> int:
    """Remove documents older than retention period."""
    with get_cursor() as cur:
        cur.execute("""
            DELETE FROM documents
            WHERE published_at < NOW() - INTERVAL '%s days'
        """, (days,))
        return cur.rowcount


def fetch_alpaca_news(
    tickers: list[str],
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Fetch news from Alpaca News API.

    Returns list of news articles.
    """
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not set")

    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    all_articles = []
    page_token = None

    while True:
        params = {
            "symbols": ",".join(tickers),
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 50,
            "sort": "desc",
        }
        if page_token:
            params["page_token"] = page_token

        response = httpx.get(url, headers=headers, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        all_articles.extend(data.get("news", []))

        page_token = data.get("next_page_token")
        if not page_token:
            break

    return all_articles


def ingest_alpaca_news(tickers: list[str], days: int = 3) -> int:
    """
    Ingest recent news for given tickers from Alpaca.

    Returns count of documents ingested.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    print(f"  Fetching Alpaca news for {len(tickers)} tickers...")
    articles = fetch_alpaca_news(tickers, start, end)
    print(f"  Found {len(articles)} articles")

    count = 0
    for article in articles:
        # Combine headline and summary
        headline = article.get("headline", "")
        summary = article.get("summary", "")
        content = f"{headline}\n\n{summary}" if summary else headline

        if not content.strip():
            continue

        # Parse timestamp
        created_at = article.get("created_at", "")
        try:
            published_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            published_at = datetime.utcnow()

        # Ingest for each mentioned ticker we care about
        article_tickers = article.get("symbols", [])
        for ticker in article_tickers:
            if ticker in tickers:
                doc_ids = ingest_document(
                    content=content,
                    ticker=ticker,
                    doc_type="news",
                    source="alpaca",
                    source_url=article.get("url"),
                    published_at=published_at,
                )
                if doc_ids:
                    count += 1

    return count
