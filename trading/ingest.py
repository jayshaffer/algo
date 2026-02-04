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


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """
    Get SEC CIK number for a ticker symbol.

    Uses SEC's company tickers JSON.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "AlgoTrading contact@example.com"}

    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                # CIK needs to be zero-padded to 10 digits
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass

    return None


def fetch_sec_filings(
    ticker: str,
    filing_types: list[str] = ["10-K", "10-Q", "8-K"],
    count: int = 5,
) -> list[dict]:
    """
    Fetch recent SEC filings for a ticker.

    Returns list of filing metadata with content URLs.
    """
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return []

    # Get company submissions
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "AlgoTrading contact@example.com"}

    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    filings = []
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form in filing_types and len(filings) < count:
            accession = accessions[i].replace("-", "")
            doc = primary_docs[i]

            filings.append({
                "type": form,
                "filed_at": datetime.strptime(dates[i], "%Y-%m-%d"),
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}",
                "accession": accessions[i],
            })

    return filings


def fetch_filing_content(url: str, max_chars: int = 50000) -> str:
    """
    Fetch and extract text from an SEC filing.

    Truncates to max_chars to avoid huge filings.
    """
    headers = {"User-Agent": "AlgoTrading contact@example.com"}

    try:
        response = httpx.get(url, headers=headers, timeout=60.0)
        response.raise_for_status()
        content = response.text

        # Basic HTML stripping (filings are often HTML)
        import re
        # Remove script/style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        return content[:max_chars]
    except Exception:
        return ""


def ingest_sec_filings(ticker: str, filing_types: list[str] = ["10-K", "10-Q", "8-K"]) -> int:
    """
    Ingest recent SEC filings for a ticker.

    Returns count of documents ingested.
    """
    print(f"  Fetching SEC filings for {ticker}...")
    filings = fetch_sec_filings(ticker, filing_types)
    print(f"  Found {len(filings)} filings")

    count = 0
    for filing in filings:
        # Check if already ingested
        if document_exists(filing["url"]):
            continue

        content = fetch_filing_content(filing["url"])
        if not content or len(content) < 100:
            continue

        doc_type = f"filing_{filing['type'].lower().replace('-', '')}"

        doc_ids = ingest_document(
            content=content,
            ticker=ticker,
            doc_type=doc_type,
            source="sec_edgar",
            source_url=filing["url"],
            published_at=filing["filed_at"],
        )
        if doc_ids:
            count += 1

    return count
