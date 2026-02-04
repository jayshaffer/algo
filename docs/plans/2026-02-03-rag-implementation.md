# RAG Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add retrieval-augmented generation to ground trade theses in fresh documents rather than stale model knowledge.

**Architecture:** pgvector stores embedded document chunks; retrieval layer fetches by ticker or semantic query with time-decay scoring; ingestion pipeline pulls from Alpaca News and SEC EDGAR; ideation engine injects retrieved docs and enforces citations.

**Tech Stack:** PostgreSQL + pgvector, nomic-embed-text (Ollama), Alpaca News API, SEC EDGAR API

---

## Task 1: Add pgvector Schema

**Files:**
- Create: `db/init/004_documents.sql`

**Step 1: Write the schema file**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Core documents table for RAG
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,

    -- Content
    content TEXT NOT NULL,
    embedding vector(768),              -- nomic-embed-text produces 768-dim vectors

    -- Metadata for filtering
    ticker VARCHAR(10),                 -- NULL for macro/general docs
    doc_type VARCHAR(20) NOT NULL,      -- 'news', 'filing_10k', 'filing_10q', 'filing_8k'
    source VARCHAR(50) NOT NULL,        -- 'alpaca', 'sec_edgar', 'manual'
    source_url TEXT,                    -- Original URL for citation/dedup

    -- Timestamps
    published_at TIMESTAMP NOT NULL,    -- When the source was published
    created_at TIMESTAMP DEFAULT NOW(), -- When we ingested it

    -- Chunking metadata
    parent_id INTEGER REFERENCES documents(id),
    chunk_index INTEGER DEFAULT 0
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX idx_documents_embedding ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Filtering indexes
CREATE INDEX idx_documents_ticker ON documents(ticker);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
CREATE INDEX idx_documents_published_at ON documents(published_at DESC);
CREATE INDEX idx_documents_ticker_published ON documents(ticker, published_at DESC);
CREATE INDEX idx_documents_source_url ON documents(source_url);
```

**Step 2: Add pgvector to requirements**

Add to `trading/requirements.txt`:
```
pgvector>=0.2.0
```

**Step 3: Commit**

```bash
git add db/init/004_documents.sql trading/requirements.txt
git commit -m "feat: add pgvector schema for RAG documents"
```

---

## Task 2: Implement Document Retrieval Layer

**Files:**
- Create: `trading/retrieval.py`

**Step 1: Write the retrieval module**

```python
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
```

**Step 2: Commit**

```bash
git add trading/retrieval.py
git commit -m "feat: add retrieval layer with time-decay scoring"
```

---

## Task 3: Implement Document Ingestion Core

**Files:**
- Create: `trading/ingest.py`

**Step 1: Write chunking and ingestion functions**

```python
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
```

**Step 2: Commit**

```bash
git add trading/ingest.py
git commit -m "feat: add document ingestion with chunking and embedding"
```

---

## Task 4: Implement Alpaca News Ingestion

**Files:**
- Modify: `trading/ingest.py` (append to existing)

**Step 1: Add Alpaca News fetching**

Append to `trading/ingest.py`:

```python
import os
from datetime import timedelta

import httpx


def fetch_alpaca_news(
    tickers: list[str],
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Fetch news from Alpaca News API.

    Returns list of news articles.
    """
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY_ID")
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
```

**Step 2: Commit**

```bash
git add trading/ingest.py
git commit -m "feat: add Alpaca News ingestion"
```

---

## Task 5: Implement SEC EDGAR Ingestion

**Files:**
- Modify: `trading/ingest.py` (append to existing)

**Step 1: Add SEC EDGAR fetching**

Append to `trading/ingest.py`:

```python
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
```

**Step 2: Commit**

```bash
git add trading/ingest.py
git commit -m "feat: add SEC EDGAR filing ingestion"
```

---

## Task 6: Implement Ingestion Scheduler

**Files:**
- Create: `trading/ingest_scheduler.py`

**Step 1: Write the scheduler module**

```python
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
```

**Step 2: Commit**

```bash
git add trading/ingest_scheduler.py
git commit -m "feat: add ingestion scheduler for daily document updates"
```

---

## Task 7: Integrate RAG into Ideation Engine

**Files:**
- Modify: `trading/ideation.py`

**Step 1: Update the system prompt**

Replace `IDEATION_SYSTEM_PROMPT` in `trading/ideation.py`:

```python
IDEATION_SYSTEM_PROMPT = """You are an investment ideation agent. Your job is to generate trade theses based ONLY on the provided documents.

CRITICAL RULES:
1. You must ONLY use information from the retrieved documents below
2. Do NOT use your training knowledge about companies - it may be outdated
3. Every thesis MUST cite at least one source document by [DOC-ID]
4. If you don't have recent information about a company, say so - don't guess

You will receive:
1. Current portfolio positions
2. Active theses you previously generated
3. Retrieved documents organized by ticker and theme (with DOC-IDs)
4. Market snapshot (sector performance, movers)

Your tasks:
1. REVIEW existing active theses - check if retrieved documents support or contradict them
2. GENERATE new trade ideas (3-5) based on retrieved documents

For each new thesis, provide:
- ticker: Stock symbol
- direction: "long", "short", or "avoid"
- thesis: Core reasoning citing [DOC-ID] sources
- entry_trigger: Specific conditions (price levels, events)
- exit_trigger: Target or stop conditions
- invalidation: What would prove the thesis wrong
- confidence: "high", "medium", or "low"
- sources: List of DOC-IDs used

Respond with valid JSON only:
{
    "reviews": [
        {
            "thesis_id": 123,
            "action": "keep" | "update" | "invalidate" | "expire",
            "reason": "Brief explanation citing [DOC-ID] if relevant",
            "updates": {...}
        }
    ],
    "new_theses": [
        {
            "ticker": "NVDA",
            "direction": "long",
            "thesis": "Strong datacenter demand per Q3 earnings [DOC-142]. New AI chip approach [DOC-156].",
            "entry_trigger": "Pullback to $800 support",
            "exit_trigger": "Target $1100 or break below $750",
            "invalidation": "Datacenter revenue growth decelerates below 50% YoY",
            "confidence": "high",
            "sources": [142, 156]
        }
    ],
    "market_observations": "..."
}"""
```

**Step 2: Add import for retrieval**

Add to imports at top of `trading/ideation.py`:

```python
from .retrieval import retrieve_for_ideation
```

**Step 3: Add context formatting function**

Add after the dataclass definitions in `trading/ideation.py`:

```python
def format_retrieved_context(retrieved: dict) -> str:
    """Format retrieved documents for LLM context."""
    lines = ["Retrieved Documents:", ""]

    # By ticker
    for ticker, docs in retrieved.get("by_ticker", {}).items():
        if not docs:
            continue
        lines.append(f"=== {ticker} ===")
        for doc in docs:
            age_days = (datetime.now() - doc["published_at"]).days
            lines.append(f"[DOC-{doc['id']}] ({doc['doc_type']}, {age_days}d ago)")
            # Truncate long content
            content = doc["content"][:1500]
            if len(doc["content"]) > 1500:
                content += "..."
            lines.append(content)
            lines.append("")

    # By theme
    for theme, docs in retrieved.get("by_theme", {}).items():
        if not docs:
            continue
        lines.append(f"=== Theme: {theme} ===")
        for doc in docs:
            age_days = (datetime.now() - doc["published_at"]).days
            ticker_note = f" [{doc['ticker']}]" if doc.get("ticker") else ""
            lines.append(f"[DOC-{doc['id']}]{ticker_note} ({doc['doc_type']}, {age_days}d ago)")
            content = doc["content"][:1500]
            if len(doc["content"]) > 1500:
                content += "..."
            lines.append(content)
            lines.append("")

    if len(lines) == 2:  # Only header
        lines.append("No documents retrieved. Unable to generate grounded theses.")

    return "\n".join(lines)
```

**Step 4: Update build_ideation_context function**

Replace the `build_ideation_context` function:

```python
def build_ideation_context(account_info: dict, retrieved: dict) -> str:
    """Build context string for ideation LLM call with RAG."""
    sections = []

    # Portfolio context
    sections.append(get_portfolio_context(account_info))
    sections.append("")

    # Retrieved documents (RAG)
    sections.append(format_retrieved_context(retrieved))
    sections.append("")

    # Active theses
    theses = get_active_theses()
    if theses:
        sections.append("Active Theses:")
        for t in theses:
            age_days = (datetime.now() - t["created_at"]).days
            sections.append(f"  ID {t['id']}: {t['ticker']} ({t['direction']}) - {t['confidence']} confidence")
            sections.append(f"    Thesis: {t['thesis']}")
            sections.append(f"    Entry trigger: {t['entry_trigger'] or 'Not specified'}")
            sections.append(f"    Exit trigger: {t['exit_trigger'] or 'Not specified'}")
            sections.append(f"    Invalidation criteria: {t['invalidation'] or 'Not specified'}")
            sections.append(f"    Age: {age_days} days")
    else:
        sections.append("Active Theses: None")
    sections.append("")

    # Macro context
    sections.append(get_macro_context(days=7))
    sections.append("")

    # Market snapshot
    try:
        snapshot = get_market_snapshot()
        sections.append(format_market_snapshot(snapshot))
    except Exception as e:
        sections.append(f"Market Snapshot: Error fetching data - {e}")

    return "\n".join(sections)
```

**Step 5: Update run_ideation function**

Replace the `run_ideation` function:

```python
def run_ideation(model: str = "qwen2.5:14b") -> IdeationResult:
    """
    Run an ideation session with RAG retrieval.

    1. Determine tickers and themes to research
    2. Retrieve relevant documents
    3. Build context with retrieved docs
    4. Call LLM for thesis review and generation
    5. Validate citations and apply changes

    Args:
        model: Ollama model to use

    Returns:
        IdeationResult with session details
    """
    timestamp = datetime.now()
    print(f"[{timestamp.isoformat()}] Starting ideation session (RAG-enabled)")
    print(f"  Model: {model}")

    # Step 1: Determine what to retrieve
    print("\n[Step 1] Determining retrieval targets...")
    positions = get_positions()
    position_tickers = [p["ticker"] for p in positions]

    active_theses = get_active_theses()
    thesis_tickers = [t["ticker"] for t in active_theses]

    tickers_to_research = list(set(position_tickers + thesis_tickers))
    print(f"  Tickers: {', '.join(tickers_to_research) or 'None'}")

    # Default themes for market research
    themes = [
        "Federal Reserve interest rate policy",
        "AI and semiconductor demand",
        "earnings surprises guidance raised",
        "sector rotation defensive stocks",
    ]
    print(f"  Themes: {len(themes)}")

    # Step 2: Retrieve documents
    print("\n[Step 2] Retrieving documents...")
    retrieved = retrieve_for_ideation(
        tickers=tickers_to_research,
        themes=themes,
        k_per_ticker=5,
        k_per_theme=5,
    )

    doc_count = sum(len(docs) for docs in retrieved.get("by_ticker", {}).values())
    doc_count += sum(len(docs) for docs in retrieved.get("by_theme", {}).values())
    print(f"  Retrieved {doc_count} documents")

    # Step 3: Build context
    print("\n[Step 3] Building ideation context...")
    try:
        account_info = get_account_info()
    except Exception as e:
        print(f"  Warning: Could not get account info: {e}")
        account_info = {"cash": 0, "portfolio_value": 0, "buying_power": 0}

    context = build_ideation_context(account_info, retrieved)
    print(f"  Context built ({len(context)} chars)")

    # Get current state for exclusion
    position_set = {p["ticker"] for p in positions}
    active_thesis_tickers = {t["ticker"] for t in active_theses}

    # Step 4: Call LLM
    print("\n[Step 4] Calling LLM for ideation...")
    prompt = f"""Here is the current market context with retrieved documents. Review existing theses and generate new trade ideas.

IMPORTANT: Base your analysis ONLY on the retrieved documents. Cite [DOC-ID] for every claim.

Exclude these tickers (already in portfolio): {', '.join(position_set) or 'None'}
Exclude these tickers (already have active thesis): {', '.join(active_thesis_tickers) or 'None'}

{context}"""

    try:
        response = chat_json(prompt, model=model, system=IDEATION_SYSTEM_PROMPT)
    except Exception as e:
        print(f"  Error: {e}")
        return IdeationResult(
            timestamp=timestamp,
            reviews=[],
            new_theses=[],
            market_observations=f"Error: {e}",
            theses_kept=0,
            theses_updated=0,
            theses_closed=0,
            theses_created=0,
        )

    print(f"  Received response")
    print(f"  Market observations: {response.get('market_observations', '')[:100]}...")

    # Step 5: Validate citations
    print("\n[Step 5] Validating citations...")
    for thesis_data in response.get("new_theses", []):
        sources = thesis_data.get("sources", [])
        if not sources:
            print(f"  Warning: {thesis_data.get('ticker', '?')} thesis has no citations")

    # Process reviews
    print("\n[Step 6] Processing thesis reviews...")
    reviews = []
    theses_kept = 0
    theses_updated = 0
    theses_closed = 0

    for review_data in response.get("reviews", []):
        review = ThesisReview(
            thesis_id=review_data["thesis_id"],
            action=review_data["action"],
            reason=review_data.get("reason", ""),
            updates=review_data.get("updates"),
        )
        reviews.append(review)

        if review.action == "keep":
            theses_kept += 1
            print(f"  Thesis {review.thesis_id}: KEEP - {review.reason}")

        elif review.action == "update":
            if review.updates:
                update_thesis(
                    thesis_id=review.thesis_id,
                    thesis=review.updates.get("thesis"),
                    entry_trigger=review.updates.get("entry_trigger"),
                    exit_trigger=review.updates.get("exit_trigger"),
                    invalidation=review.updates.get("invalidation"),
                    confidence=review.updates.get("confidence"),
                )
            theses_updated += 1
            print(f"  Thesis {review.thesis_id}: UPDATE - {review.reason}")

        elif review.action in ("invalidate", "expire"):
            close_thesis(
                thesis_id=review.thesis_id,
                status=review.action + "d",
                reason=review.reason,
            )
            theses_closed += 1
            print(f"  Thesis {review.thesis_id}: {review.action.upper()} - {review.reason}")

    # Process new theses
    print("\n[Step 7] Creating new theses...")
    new_theses = []
    theses_created = 0

    for thesis_data in response.get("new_theses", []):
        ticker = thesis_data["ticker"]
        if ticker in position_set:
            print(f"  Skipping {ticker}: already in portfolio")
            continue
        if ticker in active_thesis_tickers:
            print(f"  Skipping {ticker}: already has active thesis")
            continue

        thesis = NewThesis(
            ticker=ticker,
            direction=thesis_data["direction"],
            thesis=thesis_data["thesis"],
            entry_trigger=thesis_data.get("entry_trigger", ""),
            exit_trigger=thesis_data.get("exit_trigger", ""),
            invalidation=thesis_data.get("invalidation", ""),
            confidence=thesis_data.get("confidence", "medium"),
        )
        new_theses.append(thesis)

        thesis_id = insert_thesis(
            ticker=thesis.ticker,
            direction=thesis.direction,
            thesis=thesis.thesis,
            entry_trigger=thesis.entry_trigger,
            exit_trigger=thesis.exit_trigger,
            invalidation=thesis.invalidation,
            confidence=thesis.confidence,
            source="ideation",
        )
        theses_created += 1
        active_thesis_tickers.add(ticker)
        print(f"  Created thesis {thesis_id}: {ticker} ({thesis.direction}) - {thesis.confidence}")

    # Summary
    print("\n" + "=" * 60)
    print("Ideation Session Complete (RAG-enabled)")
    print("=" * 60)
    print(f"  Documents retrieved: {doc_count}")
    print(f"  Theses kept: {theses_kept}")
    print(f"  Theses updated: {theses_updated}")
    print(f"  Theses closed: {theses_closed}")
    print(f"  Theses created: {theses_created}")

    return IdeationResult(
        timestamp=timestamp,
        reviews=reviews,
        new_theses=new_theses,
        market_observations=response.get("market_observations", ""),
        theses_kept=theses_kept,
        theses_updated=theses_updated,
        theses_closed=theses_closed,
        theses_created=theses_created,
    )
```

**Step 6: Commit**

```bash
git add trading/ideation.py
git commit -m "feat: integrate RAG retrieval into ideation engine"
```

---

## Task 8: Update CLAUDE.md with RAG Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add RAG section to CLAUDE.md**

Add after the "Key Concepts" section:

```markdown
### RAG System (`retrieval.py`, `ingest.py`)

The ideation engine uses Retrieval-Augmented Generation to ground theses in fresh documents:

**Document Sources:**
- Alpaca News API (free with trading account)
- SEC EDGAR (10-K, 10-Q, 8-K filings)

**Retrieval modes:**
- `retrieve_by_ticker(ticker)` - Get recent docs for a specific company
- `retrieve_by_query(query)` - Semantic search across all documents
- `retrieve_for_ideation(tickers, themes)` - Combined retrieval for ideation

**Time decay scoring:**
Documents lose 50% relevance after 30 days: `score = similarity * exp(-age/30)`

**Citation requirement:**
The LLM must cite `[DOC-ID]` for every claim. Theses without citations are flagged.

**Ingestion schedule:**
```bash
# Run daily at 6am ET
docker compose exec trading python -m trading.ingest_scheduler
```
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add RAG system documentation to CLAUDE.md"
```

---

## Task 9: End-to-End Testing

**Step 1: Rebuild containers with pgvector**

```bash
docker compose build trading
docker compose up -d db
# Wait for db to be healthy
sleep 10
```

**Step 2: Verify pgvector extension**

```bash
docker compose exec db psql -U algo -d trading -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker compose exec db psql -U algo -d trading -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

Expected: Shows pgvector version (e.g., "0.5.1")

**Step 3: Apply new schema**

```bash
docker compose exec db psql -U algo -d trading -f /docker-entrypoint-initdb.d/004_documents.sql
```

**Step 4: Pull embedding model**

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

**Step 5: Test ingestion**

```bash
docker compose exec trading python -c "
from trading.ingest import ingest_alpaca_news, get_document_stats
count = ingest_alpaca_news(['AAPL', 'NVDA', 'MSFT'], days=1)
print(f'Ingested {count} documents')
print(get_document_stats())
"
```

**Step 6: Test retrieval**

```bash
docker compose exec trading python -c "
from trading.retrieval import retrieve_by_ticker, retrieve_by_query
docs = retrieve_by_ticker('AAPL', k=3)
print(f'By ticker: {len(docs)} docs')
for d in docs:
    print(f'  [{d[\"id\"]}] {d[\"content\"][:80]}...')

docs = retrieve_by_query('AI semiconductor demand', k=3)
print(f'By query: {len(docs)} docs')
for d in docs:
    print(f'  [{d[\"id\"]}] {d.get(\"ticker\", \"?\")} - {d[\"content\"][:60]}...')
"
```

**Step 7: Test ideation with RAG**

```bash
docker compose exec trading python -m trading.ideation --model qwen2.5:14b
```

Expected: Ideation output shows "RAG-enabled", document retrieval count, and theses with `[DOC-ID]` citations.

**Step 8: Final commit**

```bash
git add -A
git commit -m "test: verify RAG integration end-to-end"
```

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `db/init/004_documents.sql`, `trading/requirements.txt` | pgvector schema |
| 2 | `trading/retrieval.py` | Retrieval layer with time decay |
| 3 | `trading/ingest.py` | Core ingestion (chunking, embedding) |
| 4 | `trading/ingest.py` | Alpaca News ingestion |
| 5 | `trading/ingest.py` | SEC EDGAR ingestion |
| 6 | `trading/ingest_scheduler.py` | Scheduled ingestion job |
| 7 | `trading/ideation.py` | RAG integration |
| 8 | `CLAUDE.md` | Documentation |
| 9 | - | End-to-end testing |
