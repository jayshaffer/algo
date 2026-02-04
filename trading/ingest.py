"""Document ingestion for RAG - chunking, embedding, and storage."""

from datetime import datetime
from typing import Optional

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
