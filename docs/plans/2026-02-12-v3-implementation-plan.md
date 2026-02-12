# V3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a trading platform with a closed learning loop, structured data flow, and simplified infrastructure (Claude-only, no Ollama).

**Architecture:** 4-stage session (backfill → pipeline → strategist → executor). Haiku classifies news. Opus strategizes with attribution constraints in system prompt. Executor receives structured JSON, not prose. All signal metrics via `decision_signals` FK.

**Tech Stack:** Python 3.12, psycopg2, anthropic SDK, alpaca-py, Flask, PostgreSQL 16, Docker Compose

**V2 Reference:** The existing codebase serves as reference material for API integration patterns, prompt templates, test fixtures, and SQL queries. `V3_ARCHITECTURE.md` at repo root has exhaustive documentation. `docs/plans/2026-02-12-v3-design.md` has the approved design.

---

## Phase 1: Foundation (Schema + Infrastructure)

### Task 1: Schema Migration

Build the additive schema migration that adds the `playbook_actions` table, new columns on `decisions`, and a missing index on `decision_signals`.

**Files:**
- Build: `db/init/006_v3.sql`

**Step 1: Write the migration**

```sql
-- V3: Structured playbook actions + decision linkage
CREATE TABLE IF NOT EXISTS playbook_actions (
    id SERIAL PRIMARY KEY,
    playbook_id INT REFERENCES playbooks(id),
    ticker VARCHAR(128) NOT NULL,
    action VARCHAR(10) NOT NULL,
    thesis_id INT REFERENCES theses(id),
    reasoning TEXT,
    confidence VARCHAR(10),
    max_quantity DECIMAL,
    priority INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_playbook_actions_playbook_id ON playbook_actions(playbook_id);

ALTER TABLE decisions ADD COLUMN IF NOT EXISTS playbook_action_id INT REFERENCES playbook_actions(id);
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS is_off_playbook BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_decision_signals_decision_id ON decision_signals(decision_id);
```

**Step 2: Verify migration is valid SQL**

Run: `docker compose exec db psql -U algo -d trading -f /docker-entrypoint-initdb.d/006_v3.sql`

If DB is fresh, the init scripts run in order. For existing DBs, run the migration manually.

**Step 3: Commit**

```bash
git add db/init/006_v3.sql
git commit -m "feat: add V3 schema migration — playbook_actions table + decision columns"
```

---

### Task 2: Docker Compose

Build the Docker Compose stack for V3. Three services: PostgreSQL, trading agent, dashboard. No Ollama, no GPU.

**Files:**
- Build: `docker-compose.yml`

**Reference:** V2's `docker-compose.yml` has the Postgres healthcheck and volume patterns. V3 drops the `ollama` service and `ollama_data` volume.

**Step 1: Write `docker-compose.yml`**

```yaml
services:
  db:
    image: postgres:16
    env_file:
      - .env
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U algo -d trading"]
      interval: 5s
      timeout: 5s
      retries: 5

  trading:
    build: ./trading
    depends_on:
      db:
        condition: service_healthy
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    command: ["sleep", "infinity"]

  dashboard:
    build: ./dashboard
    ports:
      - "3000:3000"
    depends_on:
      db:
        condition: service_healthy
    env_file:
      - .env

volumes:
  postgres_data:
```

**Step 2: Update `trading/requirements.txt`**

The dependencies list should include `anthropic`, `alpaca-py`, `psycopg2-binary`, `flask`, and standard library dependencies. It should NOT include `ollama` or `pgvector`.

**Reference:** V2's `trading/requirements.txt` has the version pins. Keep the same pins for shared dependencies.

**Step 3: Verify Docker builds**

Run: `docker compose build`

**Step 4: Commit**

```bash
git add docker-compose.yml trading/requirements.txt
git commit -m "feat: V3 Docker stack — three services, no Ollama"
```

---

### Task 3: Database Layer

Build the consolidated database access layer as a Python package: shared connection factory, trading CRUD operations, and dashboard read queries.

**Files:**
- Build: `trading/database/__init__.py`
- Build: `trading/database/connection.py`
- Build: `trading/database/trading_db.py`
- Build: `trading/database/dashboard_db.py`
- Test: `tests/test_db.py`

**Reference:** V2's `trading/db.py` has all trading CRUD functions. V2's `dashboard/queries.py` has all dashboard read queries. Both have their own `get_connection()` / `get_cursor()` — V3 unifies these into `connection.py`.

**Step 1: Write failing tests for the connection module**

```python
"""Tests for database connection layer."""
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from trading.database.connection import get_connection, get_cursor


class TestGetConnection:
    def test_reads_database_url(self):
        with patch("trading.database.connection.psycopg2") as mock_pg:
            with patch.dict("os.environ", {"DATABASE_URL": "postgresql://test"}):
                get_connection()
            mock_pg.connect.assert_called_once_with("postgresql://test")

    def test_raises_without_database_url(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL"):
                get_connection()


class TestGetCursor:
    def test_yields_realdict_cursor_and_commits(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("trading.database.connection.get_connection", return_value=mock_conn):
            with get_cursor() as cur:
                cur.execute("SELECT 1")
            mock_conn.commit.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_rollback_on_exception(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("trading.database.connection.get_connection", return_value=mock_conn):
            with pytest.raises(RuntimeError):
                with get_cursor() as cur:
                    raise RuntimeError("test error")
            mock_conn.rollback.assert_called_once()
            mock_conn.close.assert_called_once()
```

Run: `python3 -m pytest tests/test_db.py::TestGetConnection -x -v`

Expected: FAIL (module doesn't exist yet).

**Step 2: Build `trading/database/connection.py`**

```python
"""Shared database connection factory."""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection():
    """Create database connection from DATABASE_URL."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL must be set")
    return psycopg2.connect(database_url)


@contextmanager
def get_cursor():
    """Context manager for database cursor with auto-commit/rollback."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

**Step 3: Build `trading/database/__init__.py`**

```python
"""Database access layer — shared connection, separate query namespaces."""

from .connection import get_connection, get_cursor

__all__ = ["get_connection", "get_cursor"]
```

**Step 4: Build `trading/database/trading_db.py`**

All trading CRUD operations. Functions to build:

**Reference:** V2's `trading/db.py` has the SQL for every function listed below. Use the same query patterns but import `get_cursor` from `.connection`.

```python
"""Trading CRUD operations."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from psycopg2.extras import Json, execute_values
from .connection import get_cursor, get_connection


# --- News Signals ---
def insert_news_signal(ticker, headline, category, sentiment, confidence, published_at) -> int
def insert_news_signals_batch(signals: list[tuple]) -> int

# --- Macro Signals ---
def insert_macro_signal(headline, category, affected_sectors, sentiment, published_at) -> int
def insert_macro_signals_batch(signals: list[tuple]) -> int

# --- Positions ---
def upsert_position(ticker, shares, avg_cost) -> int
def get_positions() -> list
def delete_all_positions()

# --- Open Orders ---
def upsert_open_order(order_id, ticker, side, qty, filled_qty, order_type, limit_price, status) -> int
def get_open_orders() -> list
def delete_all_open_orders()

# --- Decisions ---
def insert_decision(decision_date, ticker, action, quantity, price, reasoning,
                    signals_used, account_equity, buying_power,
                    playbook_action_id=None, is_off_playbook=False) -> int
def get_recent_decisions(days=30) -> list
def update_decision_outcome(decision_id, outcome_7d=None, outcome_30d=None)
def get_decisions_needing_backfill_7d() -> list
def get_decisions_needing_backfill_30d() -> list

# --- Decision Signals ---
def insert_decision_signal(decision_id, signal_type, signal_id) -> int
def insert_decision_signals_batch(signal_links: list[tuple]) -> int

# --- Theses ---
def insert_thesis(ticker, direction, confidence, thesis, entry_trigger, exit_trigger, invalidation, source_signals) -> int
def update_thesis(thesis_id, **fields)
def close_thesis(thesis_id, status, reason)
def get_active_theses(ticker=None) -> list
def get_thesis_by_id(thesis_id) -> dict | None

# --- Playbooks ---
def upsert_playbook(playbook_date, market_outlook, priority_actions, watch_list, risk_notes) -> int
def get_playbook(playbook_date) -> dict | None

# --- Playbook Actions (V3) ---
def insert_playbook_action(playbook_id, ticker, action, thesis_id, reasoning, confidence, max_quantity, priority) -> int
def get_playbook_actions(playbook_id) -> list
def delete_playbook_actions(playbook_id) -> int

# --- Account Snapshots ---
def insert_account_snapshot(portfolio_value, cash, buying_power) -> int
def get_account_snapshots(limit=30) -> list

# --- Signal Attribution ---
def upsert_signal_attribution(category, sample_size, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d) -> int
def get_signal_attribution() -> list

# --- Read helpers ---
def get_news_signals(days=7, ticker=None) -> list
def get_macro_signals(days=7) -> list
```

Each function follows the same pattern: `with get_cursor() as cur:` → execute SQL → return result.

For the exact SQL queries, reference V2's `trading/db.py`. The queries are identical — the only additions are:
- `insert_playbook_action()`, `get_playbook_actions()`, `delete_playbook_actions()` — new V3 CRUD
- `insert_decision()` now accepts `playbook_action_id` and `is_off_playbook` parameters

**Step 5: Build `trading/database/dashboard_db.py`**

All dashboard read queries.

**Reference:** V2's `dashboard/queries.py` has all the queries. Use the same SQL but import from `.connection`.

```python
"""Dashboard read-only queries."""

from .connection import get_cursor


def get_positions() -> list
def get_latest_snapshot() -> dict | None
def get_recent_ticker_signals(limit=50) -> list
def get_recent_macro_signals(limit=50) -> list
def get_recent_decisions(limit=50) -> list
def get_equity_curve() -> list
def get_decision_outcomes() -> list
def get_active_theses() -> list
def get_latest_playbook() -> dict | None
def get_playbook_history(limit=10) -> list
def get_signal_attribution() -> list
def get_performance_stats() -> dict
def close_thesis(thesis_id, status, reason)
```

**Step 6: Write comprehensive tests**

Test key CRUD functions for `trading_db.py` and `dashboard_db.py`. Mock `get_cursor` at `trading.database.connection.get_cursor`.

**Reference:** V2's `tests/test_db.py` has tests for most functions. V2's `conftest.py` has the `mock_db` and `mock_cursor` fixtures.

Run: `python3 -m pytest tests/test_db.py -x -v`

Expected: PASS

**Step 7: Commit**

```bash
git add trading/database/ tests/test_db.py
git commit -m "feat: build consolidated database layer — shared connection, trading + dashboard queries"
```

---

## Phase 2: Pipeline (Haiku Classification)

### Task 4: News Classifier

Build the news classifier that uses Claude Haiku to categorize headlines into ticker-specific, macro-political, sector, or noise signals.

**Files:**
- Build: `trading/classifier.py`
- Test: `tests/test_classifier.py`

**Reference:** V2's `trading/classifier.py` has the dataclasses (`TickerSignal`, `MacroSignal`, `ClassificationResult`), the prompt templates (`CLASSIFICATION_PROMPT`, `BATCH_CLASSIFICATION_PROMPT`), and the `_build_classification_result()` helper. V3 keeps the same dataclasses and prompts but calls Claude Haiku instead of Ollama's `chat_json`.

**Step 1: Write failing tests**

```python
"""Tests for news classification via Claude Haiku."""
from unittest.mock import patch, MagicMock
from datetime import datetime

from trading.classifier import (
    classify_news_batch,
    classify_news,
    _build_classification_result,
    ClassificationResult,
    TickerSignal,
    MacroSignal,
    CLASSIFICATION_MODEL,
)


class TestClassifyNewsBatch:
    def test_batch_calls_haiku(self):
        """Should call Claude Haiku for classification."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"type":"noise","tickers":[],"category":"noise","sentiment":"neutral","confidence":"low","affected_sectors":[]}]')]
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("trading.classifier.get_claude_client", return_value=mock_client):
            results = classify_news_batch(["Test headline"], [datetime.now()])

        assert len(results) == 1
        assert results[0].news_type == "noise"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == CLASSIFICATION_MODEL

    def test_batch_size_50(self):
        """Should batch 50 headlines per API call."""
        headlines = [f"Headline {i}" for i in range(75)]
        dates = [datetime.now()] * 75

        noise_entry = '{"type":"noise","tickers":[],"category":"noise","sentiment":"neutral","confidence":"low","affected_sectors":[]}'
        mock_response_50 = MagicMock()
        mock_response_50.content = [MagicMock(text='[' + ','.join([noise_entry] * 50) + ']')]
        mock_response_25 = MagicMock()
        mock_response_25.content = [MagicMock(text='[' + ','.join([noise_entry] * 25) + ']')]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [mock_response_50, mock_response_25]

        with patch("trading.classifier.get_claude_client", return_value=mock_client):
            results = classify_news_batch(headlines, dates, batch_size=50)

        assert len(results) == 75
        assert mock_client.messages.create.call_count == 2

    def test_fallback_on_batch_failure(self):
        """Should fall back to individual classification when batch fails."""
        single_response = MagicMock()
        single_response.content = [MagicMock(text='{"type":"noise","tickers":[],"category":"noise","sentiment":"neutral","confidence":"low","affected_sectors":[]}')]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            Exception("batch failed"),  # batch call fails
            single_response,            # individual fallback
        ]

        with patch("trading.classifier.get_claude_client", return_value=mock_client):
            results = classify_news_batch(["Headline"], [datetime.now()], batch_size=50)

        assert len(results) == 1


class TestBuildClassificationResult:
    def test_ticker_specific(self):
        entry = {"type": "ticker_specific", "tickers": ["AAPL"], "category": "earnings", "sentiment": "bullish", "confidence": "high"}
        result = _build_classification_result(entry, "Apple beats earnings", datetime.now())
        assert result.news_type == "ticker_specific"
        assert len(result.ticker_signals) == 1
        assert result.ticker_signals[0].ticker == "AAPL"
        assert result.ticker_signals[0].sentiment == "bullish"

    def test_macro_political(self):
        entry = {"type": "macro_political", "category": "fed", "sentiment": "bearish", "affected_sectors": ["finance"]}
        result = _build_classification_result(entry, "Fed raises rates", datetime.now())
        assert result.news_type == "macro_political"
        assert result.macro_signal is not None
        assert result.macro_signal.category == "fed"

    def test_noise(self):
        entry = {"type": "noise"}
        result = _build_classification_result(entry, "Celebrity gossip", datetime.now())
        assert result.news_type == "noise"
        assert len(result.ticker_signals) == 0
        assert result.macro_signal is None
```

Run: `python3 -m pytest tests/test_classifier.py -x -v`

Expected: FAIL (module doesn't exist yet).

**Step 2: Build `trading/classifier.py`**

```python
"""News classification using Claude Haiku with batch support."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .claude_client import get_claude_client

logger = logging.getLogger(__name__)

CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class TickerSignal:
    """Classified ticker-specific news signal."""
    ticker: str
    headline: str
    category: str       # earnings, guidance, analyst, product, legal, noise
    sentiment: str      # bullish, bearish, neutral
    confidence: str     # high, medium, low
    published_at: datetime


@dataclass
class MacroSignal:
    """Classified macro/political news signal."""
    headline: str
    category: str            # fed, trade, regulation, geopolitical, fiscal, election
    affected_sectors: list[str]
    sentiment: str           # bullish, bearish, neutral
    published_at: datetime


@dataclass
class ClassificationResult:
    """Result of news classification."""
    news_type: str  # ticker_specific, macro_political, sector, noise
    ticker_signals: list[TickerSignal]
    macro_signal: Optional[MacroSignal]
```

Include the three prompt templates (`CLASSIFICATION_PROMPT`, `BATCH_CLASSIFICATION_PROMPT`, `TICKER_CLASSIFICATION_PROMPT`) — same text as V2.

**Reference:** V2's `trading/classifier.py` lines 41-109 have the exact prompt text.

Then implement:

```python
def _build_classification_result(entry: dict, headline: str, published_at: datetime) -> ClassificationResult:
    """Build a ClassificationResult from a parsed JSON entry."""
    # Same logic as V2 — handles ticker_specific, macro_political, sector, noise
    ...


def classify_news(headline: str, published_at: datetime) -> ClassificationResult:
    """Classify a single headline using Claude Haiku."""
    client = get_claude_client()
    response = client.messages.create(
        model=CLASSIFICATION_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    ...
    result = json.loads(text)
    return _build_classification_result(result, headline, published_at)


def classify_news_batch(
    headlines: list[str],
    published_ats: list[datetime],
    batch_size: int = 50,
) -> list[ClassificationResult]:
    """Classify headlines in batched Haiku calls. Falls back to individual on failure."""
    results = []
    for start in range(0, len(headlines), batch_size):
        batch = headlines[start:start + batch_size]
        batch_dates = published_ats[start:start + batch_size]
        try:
            results.extend(_classify_batch(batch, batch_dates))
        except Exception as e:
            logger.warning("Batch failed (%s), falling back to individual", e)
            for headline, pub_date in zip(batch, batch_dates):
                try:
                    results.append(classify_news(headline, pub_date))
                except Exception:
                    results.append(ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None))
    return results


def _classify_batch(headlines: list[str], published_ats: list[datetime]) -> list[ClassificationResult]:
    """Classify a single batch in one Haiku call."""
    headlines_block = "\n".join(f'{i+1}. "{h}"' for i, h in enumerate(headlines))
    prompt = BATCH_CLASSIFICATION_PROMPT.format(headlines_block=headlines_block, count=len(headlines))

    client = get_claude_client()
    response = client.messages.create(
        model=CLASSIFICATION_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown fences, parse JSON array, build results
    # Pad with noise if array shorter than input
    ...
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_classifier.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/classifier.py tests/test_classifier.py
git commit -m "feat: build Haiku news classifier — batch size 50, fallback to individual"
```

---

### Task 5: News Pipeline

Build the pipeline orchestrator that fetches news from Alpaca, classifies with Haiku, and stores signals in the database. Two steps: fetch → classify+store (no filter step).

**Files:**
- Build: `trading/pipeline.py`
- Test: `tests/test_pipeline.py`

**Reference:** V2's `trading/pipeline.py` has the `PipelineStats` dataclass, `run_pipeline()` function, and signal storage logic. V3 removes the filter step and the `check_dependencies()` call (was checking Ollama).

**Step 1: Write failing tests**

```python
"""Tests for news pipeline."""
from unittest.mock import patch, MagicMock
from trading.pipeline import run_pipeline, PipelineStats
from trading.classifier import ClassificationResult


class TestRunPipeline:
    def test_pipeline_fetch_classify_store(self, mock_db):
        """Pipeline runs: fetch → classify → store."""
        with patch("trading.pipeline.fetch_broad_news") as mock_fetch, \
             patch("trading.pipeline.classify_news_batch") as mock_classify:
            mock_fetch.return_value = [MagicMock(headline="Test", published_at=datetime.now())]
            mock_classify.return_value = [ClassificationResult(
                news_type="noise", ticker_signals=[], macro_signal=None
            )]
            stats = run_pipeline(hours=24, limit=50)

        assert stats.news_fetched == 1
        assert stats.noise_dropped == 1
        mock_classify.assert_called_once()

    def test_no_filter_step(self):
        """PipelineStats should not have a news_filtered attribute."""
        stats = PipelineStats(news_fetched=0, ticker_signals_stored=0,
                              macro_signals_stored=0, noise_dropped=0, errors=0)
        assert not hasattr(stats, 'news_filtered')
```

Run: `python3 -m pytest tests/test_pipeline.py -x -v`

Expected: FAIL

**Step 2: Build `trading/pipeline.py`**

```python
"""News pipeline orchestrator — fetch, classify, store."""

import logging
from dataclasses import dataclass

from .news import fetch_broad_news
from .classifier import classify_news_batch
from .database.trading_db import insert_news_signals_batch, insert_macro_signals_batch

logger = logging.getLogger("pipeline")


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    news_fetched: int
    ticker_signals_stored: int
    macro_signals_stored: int
    noise_dropped: int
    errors: int


def run_pipeline(hours: int = 24, limit: int = 300, dry_run: bool = False) -> PipelineStats:
    """Run the news pipeline: fetch → classify → store."""
    stats = PipelineStats(news_fetched=0, ticker_signals_stored=0,
                          macro_signals_stored=0, noise_dropped=0, errors=0)

    # Step 1: Fetch news from Alpaca
    news_items = fetch_broad_news(hours=hours, limit=limit)
    stats.news_fetched = len(news_items)

    if not news_items:
        return stats

    # Step 2: Classify with Haiku (batched)
    headlines = [item.headline for item in news_items]
    published_ats = [item.published_at for item in news_items]
    results = classify_news_batch(headlines, published_ats)

    # Step 3: Store signals
    ticker_signals_batch = []
    macro_signals_batch = []

    for result in results:
        if result.news_type == "noise":
            stats.noise_dropped += 1
            continue

        for signal in result.ticker_signals:
            ticker_signals_batch.append((
                signal.ticker, signal.headline, signal.category,
                signal.sentiment, signal.confidence, signal.published_at
            ))

        if result.macro_signal:
            s = result.macro_signal
            macro_signals_batch.append((
                s.headline, s.category, s.affected_sectors,
                s.sentiment, s.published_at
            ))

    if not dry_run:
        try:
            stats.ticker_signals_stored = insert_news_signals_batch(ticker_signals_batch)
        except Exception as e:
            logger.error("Error batch-inserting ticker signals: %s", e)
            stats.errors += 1
        try:
            stats.macro_signals_stored = insert_macro_signals_batch(macro_signals_batch)
        except Exception as e:
            logger.error("Error batch-inserting macro signals: %s", e)
            stats.errors += 1

    return stats
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_pipeline.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/pipeline.py tests/test_pipeline.py
git commit -m "feat: build news pipeline — fetch, classify with Haiku, store"
```

---

## Phase 3: Learning Loop

### Task 6: Attribution Engine with Constraints Builder

Build the attribution engine that computes signal performance scores from `decision_signals` and formats them as constraints for the strategist system prompt.

**Files:**
- Build: `trading/attribution.py`
- Test: `tests/test_attribution.py`

**Reference:** V2's `trading/attribution.py` has `compute_signal_attribution()` and `get_attribution_summary()`. V3 keeps both and adds `build_attribution_constraints()` — the function that closes the learning loop.

**Step 1: Write failing tests**

```python
"""Tests for signal attribution engine."""
from decimal import Decimal
from unittest.mock import patch

from trading.attribution import (
    compute_signal_attribution,
    get_attribution_summary,
    build_attribution_constraints,
)


class TestBuildAttributionConstraints:
    def test_formats_strong_and_weak_categories(self, mock_db, mock_cursor):
        """Should categorize signals into STRONG, WEAK, INSUFFICIENT DATA."""
        mock_cursor.fetchall.return_value = [
            {"category": "news_signal:earnings", "sample_size": 20,
             "win_rate_7d": Decimal("0.62"), "avg_outcome_7d": Decimal("1.5"),
             "avg_outcome_30d": Decimal("2.0"), "win_rate_30d": Decimal("0.55")},
            {"category": "news_signal:legal", "sample_size": 10,
             "win_rate_7d": Decimal("0.38"), "avg_outcome_7d": Decimal("-0.8"),
             "avg_outcome_30d": Decimal("-1.0"), "win_rate_30d": Decimal("0.40")},
            {"category": "news_signal:product", "sample_size": 3,
             "win_rate_7d": Decimal("0.50"), "avg_outcome_7d": Decimal("0.2"),
             "avg_outcome_30d": Decimal("0.3"), "win_rate_30d": Decimal("0.50")},
        ]

        result = build_attribution_constraints()

        assert "STRONG" in result
        assert "news_signal:earnings" in result
        assert "WEAK" in result
        assert "news_signal:legal" in result
        assert "INSUFFICIENT DATA" in result
        assert "news_signal:product" in result
        assert "CONSTRAINT" in result

    def test_empty_when_no_data(self, mock_db, mock_cursor):
        """Should return empty string when no attribution data."""
        mock_cursor.fetchall.return_value = []
        result = build_attribution_constraints()
        assert result == ""


class TestComputeSignalAttribution:
    def test_joins_through_decision_signals_fk(self, mock_db, mock_cursor):
        """Should use decision_signals FK, not time-window JOINs."""
        mock_cursor.fetchall.return_value = [
            {"category": "news_signal:earnings", "sample_size": 15,
             "avg_outcome_7d": Decimal("1.2"), "avg_outcome_30d": Decimal("2.0"),
             "win_rate_7d": Decimal("0.6"), "win_rate_30d": Decimal("0.55")}
        ]

        result = compute_signal_attribution()

        # Verify the SQL uses decision_signals table
        sql = mock_cursor.execute.call_args[0][0]
        assert "decision_signals" in sql
        assert "ns.ticker = d.ticker" not in sql  # No time-window JOIN
```

Run: `python3 -m pytest tests/test_attribution.py -x -v`

Expected: FAIL

**Step 2: Build `trading/attribution.py`**

```python
"""Signal attribution engine — computes which signal types are predictive."""

from decimal import Decimal

from .database.trading_db import upsert_signal_attribution, get_signal_attribution
from .database.connection import get_cursor


def compute_signal_attribution() -> list[dict]:
    """
    Compute signal attribution scores from decision_signals joined with decisions.

    Groups by composite category (signal_type + news category for news signals).
    JOINs through decision_signals FK — not time-window JOINs.
    Upserts results into signal_attribution table.
    """
    with get_cursor() as cur:
        cur.execute("""
            WITH categorized AS (
                SELECT
                    ds.decision_id,
                    CASE
                        WHEN ds.signal_type = 'news_signal' THEN
                            'news_signal:' || COALESCE(ns.category, 'unknown')
                        WHEN ds.signal_type = 'macro_signal' THEN
                            'macro_signal:' || COALESCE(ms.category, 'unknown')
                        ELSE ds.signal_type
                    END AS category,
                    d.outcome_7d,
                    d.outcome_30d
                FROM decision_signals ds
                JOIN decisions d ON d.id = ds.decision_id
                LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
                LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
                WHERE d.action IN ('buy', 'sell')
            )
            SELECT
                category,
                COUNT(DISTINCT decision_id) AS sample_size,
                AVG(outcome_7d) AS avg_outcome_7d,
                AVG(outcome_30d) AS avg_outcome_30d,
                AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_7d,
                AVG(CASE WHEN outcome_30d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_30d
            FROM categorized
            WHERE outcome_7d IS NOT NULL
            GROUP BY category
            ORDER BY sample_size DESC
        """)
        results = [dict(row) for row in cur.fetchall()]

    for row in results:
        upsert_signal_attribution(
            category=row["category"],
            sample_size=row["sample_size"],
            avg_outcome_7d=row["avg_outcome_7d"] or Decimal(0),
            avg_outcome_30d=row["avg_outcome_30d"] or Decimal(0),
            win_rate_7d=row["win_rate_7d"] or Decimal(0),
            win_rate_30d=row["win_rate_30d"] or Decimal(0),
        )

    return results


def get_attribution_summary() -> str:
    """Format attribution scores as advisory text for LLM context."""
    rows = get_signal_attribution()
    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    lines = ["Signal Attribution:"]
    predictive = [r for r in rows if r.get("win_rate_7d") and r["win_rate_7d"] > Decimal("0.5")]
    weak = [r for r in rows if r.get("win_rate_7d") and r["win_rate_7d"] <= Decimal("0.5")]

    if predictive:
        lines.append("Predictive signal types:")
        for r in predictive:
            wr = float(r["win_rate_7d"]) * 100
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: {wr:.0f}% win rate, {avg:+.2f}% avg 7d return (n={r['sample_size']})")

    if weak:
        lines.append("Weak/non-predictive signal types:")
        for r in weak:
            wr = float(r["win_rate_7d"]) * 100
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: {wr:.0f}% win rate, {avg:+.2f}% avg 7d return (n={r['sample_size']})")

    return "\n".join(lines)


def build_attribution_constraints(min_samples: int = 5) -> str:
    """Format signal attribution into constraint block for strategist system prompt.

    This is the function that closes the learning loop. The output is injected
    into the strategist's system prompt, making attribution scores enforceable.

    Categories:
      STRONG: >55% win rate, >= min_samples
      WEAK: <45% win rate, >= min_samples
      INSUFFICIENT DATA: < min_samples
    """
    rows = get_signal_attribution()
    if not rows:
        return ""

    strong, weak, insufficient = [], [], []

    for r in rows:
        cat = r["category"]
        n = r["sample_size"]
        wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0

        if n < min_samples:
            insufficient.append(f"{cat} (n={n})")
        elif wr > 55:
            strong.append(f"{cat} ({wr:.0f}%, n={n})")
        elif wr < 45:
            weak.append(f"{cat} ({wr:.0f}%, n={n})")

    lines = ["SIGNAL PERFORMANCE (last 60 days):"]
    if strong:
        lines.append(f"  STRONG (>55% win rate): {', '.join(strong)}")
    if weak:
        lines.append(f"  WEAK (<45% win rate): {', '.join(weak)}")
    if insufficient:
        lines.append(f"  INSUFFICIENT DATA (<{min_samples} samples): {', '.join(insufficient)}")

    lines.append("")
    lines.append("CONSTRAINT: Do not create theses primarily based on WEAK signal categories")
    lines.append("unless you have a specific reason to override (explain in thesis text).")

    return "\n".join(lines)
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_attribution.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/attribution.py tests/test_attribution.py
git commit -m "feat: build attribution engine with constraints builder for learning loop"
```

---

### Task 7: Pattern Analysis (FK-based queries)

Build the pattern analysis module that computes signal performance, sentiment, ticker, and confidence metrics. All signal-level queries go through the `decision_signals` FK — no time-window JOINs.

**Files:**
- Build: `trading/patterns.py`
- Test: `tests/test_patterns.py`

**Reference:** V2's `trading/patterns.py` has the dataclasses (`SignalPerformance`, `SentimentPerformance`, `TickerPerformance`, `ConfidenceCorrelation`) and the function signatures. V3 rebuilds the SQL queries to use `decision_signals` FK instead of V2's time-window JOINs (`ns.ticker = d.ticker AND ns.published_at::date <= d.date`).

**Step 1: Write failing tests**

```python
"""Tests for pattern analysis."""
from trading.patterns import (
    analyze_signal_categories,
    analyze_sentiment_performance,
    analyze_ticker_performance,
    analyze_confidence_correlation,
    get_best_performing_signals,
    get_worst_performing_signals,
    SignalPerformance,
)


class TestAnalyzeSignalCategories:
    def test_uses_decision_signals_fk(self, mock_db, mock_cursor):
        """SQL must JOIN through decision_signals, not time-window."""
        mock_cursor.fetchall.return_value = []
        analyze_signal_categories(days=90)
        sql = mock_cursor.execute.call_args[0][0]
        assert "decision_signals" in sql
        assert "ns.ticker = d.ticker" not in sql

    def test_returns_signal_performance_objects(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"category": "news_signal:earnings", "total_signals": 10,
             "avg_outcome_7d": 1.5, "avg_outcome_30d": 2.0,
             "win_rate_7d": 60.0, "win_rate_30d": 55.0}
        ]
        results = analyze_signal_categories()
        assert len(results) == 1
        assert isinstance(results[0], SignalPerformance)
        assert results[0].category == "news_signal:earnings"


class TestBestWorstPerforming:
    def test_best_reads_from_signal_attribution(self, mock_db, mock_cursor):
        """Should read directly from signal_attribution table."""
        mock_cursor.fetchall.return_value = []
        get_best_performing_signals()
        sql = mock_cursor.execute.call_args[0][0]
        assert "signal_attribution" in sql
```

Run: `python3 -m pytest tests/test_patterns.py -x -v`

Expected: FAIL

**Step 2: Build `trading/patterns.py`**

```python
"""Pattern analysis for learning from past decisions.

All signal-level queries go through decision_signals FK (not time-window JOINs).
This is the single source of truth for signal metrics.
"""

from dataclasses import dataclass
from .database.connection import get_cursor


@dataclass
class SignalPerformance:
    category: str
    total_signals: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    win_rate_7d: float | None
    win_rate_30d: float | None


@dataclass
class SentimentPerformance:
    sentiment: str
    total_decisions: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    win_rate_7d: float | None


@dataclass
class TickerPerformance:
    ticker: str
    total_decisions: int
    buys: int
    sells: int
    avg_outcome_7d: float | None
    avg_outcome_30d: float | None
    total_pnl_7d: float | None


@dataclass
class ConfidenceCorrelation:
    confidence: str
    total_decisions: int
    avg_outcome_7d: float | None
    win_rate_7d: float | None


def analyze_signal_categories(days: int = 90) -> list[SignalPerformance]:
    """Analyze which signal categories lead to profitable trades.

    Uses decision_signals FK as single source of truth.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                CASE
                    WHEN ds.signal_type = 'news_signal' THEN
                        'news_signal:' || COALESCE(ns.category, 'unknown')
                    WHEN ds.signal_type = 'macro_signal' THEN
                        'macro_signal:' || COALESCE(ms.category, 'unknown')
                    ELSE ds.signal_type
                END AS category,
                COUNT(DISTINCT ds.decision_id) as total_signals,
                AVG(d.outcome_7d) as avg_outcome_7d,
                AVG(d.outcome_30d) as avg_outcome_30d,
                AVG(CASE WHEN d.outcome_7d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_7d,
                AVG(CASE WHEN d.outcome_30d > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_30d
            FROM decision_signals ds
            JOIN decisions d ON d.id = ds.decision_id
            LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
            LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
            WHERE d.date > CURRENT_DATE - INTERVAL '%s days'
              AND d.action IN ('buy', 'sell')
              AND d.outcome_7d IS NOT NULL
            GROUP BY category
            ORDER BY avg_outcome_7d DESC NULLS LAST
        """, (days,))
        return [SignalPerformance(**row) for row in cur.fetchall()]


def analyze_sentiment_performance(days: int = 90) -> list[SentimentPerformance]:
    """Analyze performance by signal sentiment. JOINs through decision_signals FK."""
    ...


def analyze_ticker_performance(days: int = 90) -> list[TickerPerformance]:
    """Performance by ticker. No signal JOIN needed — groups decisions directly."""
    ...


def analyze_confidence_correlation(days: int = 90) -> list[ConfidenceCorrelation]:
    """Correlation between stated confidence and actual outcomes."""
    ...


def get_best_performing_signals(days: int = 90, min_occurrences: int = 3) -> list[dict]:
    """Get signal categories with best outcomes. Reads from signal_attribution table."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT category, avg_outcome_7d as avg_outcome, sample_size as occurrences,
                   win_rate_7d
            FROM signal_attribution
            WHERE sample_size >= %s AND avg_outcome_7d IS NOT NULL
            ORDER BY avg_outcome_7d DESC
            LIMIT 10
        """, (min_occurrences,))
        return [dict(row) for row in cur.fetchall()]


def get_worst_performing_signals(days: int = 90, min_occurrences: int = 3) -> list[dict]:
    """Get signal categories with worst outcomes. Reads from signal_attribution table."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT category, avg_outcome_7d as avg_outcome, sample_size as occurrences,
                   win_rate_7d
            FROM signal_attribution
            WHERE sample_size >= %s AND avg_outcome_7d IS NOT NULL
            ORDER BY avg_outcome_7d ASC
            LIMIT 10
        """, (min_occurrences,))
        return [dict(row) for row in cur.fetchall()]


def generate_pattern_report(days: int = 90) -> str:
    """Orchestrate all analysis functions into a text report."""
    ...
```

**Reference:** V2's `trading/patterns.py` has the `analyze_sentiment_performance()`, `analyze_ticker_performance()`, `analyze_confidence_correlation()`, and `generate_pattern_report()` implementations. Port the logic but replace time-window JOINs with `decision_signals` FK JOINs for signal-level queries.

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_patterns.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/patterns.py tests/test_patterns.py
git commit -m "feat: build pattern analysis with FK-based queries — single source of truth"
```

---

## Phase 4: Data Flow

### Task 8: Write Playbook Tool

Build the `write_playbook` tool handler that stores structured actions in both the `playbooks` JSONB column (backward compat) and the new `playbook_actions` table (V3 executor input).

**Files:**
- Build: `trading/tools.py` (the `tool_write_playbook` function and supporting tool handlers)
- Test: `tests/test_tools.py`

**Reference:** V2's `trading/tools.py` has all 11 tool definitions, their handlers, and the `TOOL_DEFINITIONS` / `TOOL_HANDLERS` dicts. V3 keeps all tools and rewrites `tool_write_playbook()` to also write `playbook_actions` rows.

**Step 1: Write failing tests for playbook actions**

```python
"""Tests for write_playbook tool handler."""

class TestWritePlaybook:
    def test_stores_playbook_actions(self, mock_db, mock_cursor):
        """write_playbook should store rows in playbook_actions table."""
        mock_cursor.fetchone.return_value = {"id": 1}
        result = tool_write_playbook(
            market_outlook="Bullish",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "Entry hit",
                 "confidence": "high", "max_quantity": 5, "thesis_id": 1},
            ],
            watch_list=["MSFT"],
            risk_notes="Fed meeting",
        )
        assert "Playbook written" in result
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("playbook_actions" in c for c in calls)

    def test_rejects_conflicting_actions(self):
        """Should reject buy + sell same ticker."""
        result = tool_write_playbook(
            market_outlook="Mixed",
            priority_actions=[
                {"ticker": "AAPL", "action": "buy", "reasoning": "Go long", "confidence": "high"},
                {"ticker": "AAPL", "action": "sell", "reasoning": "Trim", "confidence": "medium"},
            ],
            watch_list=[],
            risk_notes="",
        )
        assert "Error" in result or "conflict" in result.lower()

    def test_clears_old_actions_on_upsert(self, mock_db, mock_cursor):
        """Should delete old playbook_actions before inserting new ones."""
        mock_cursor.fetchone.return_value = {"id": 1}
        tool_write_playbook(
            market_outlook="Neutral",
            priority_actions=[{"ticker": "AAPL", "action": "hold", "reasoning": "Wait", "confidence": "low"}],
            watch_list=[],
            risk_notes="",
        )
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("DELETE" in c and "playbook_actions" in c for c in calls)
```

Run: `python3 -m pytest tests/test_tools.py::TestWritePlaybook -x -v`

Expected: FAIL

**Step 2: Build `trading/tools.py`**

All 11 tool definitions + handlers + the `write_playbook` rewrite.

**Reference:** V2's `trading/tools.py` has all tool handler implementations. The only handler that changes is `tool_write_playbook()`. All others are ported as-is (they call DB functions that now live in `trading.database.trading_db`).

Key `tool_write_playbook()` changes:
- Validates no conflicting actions (buy + sell same ticker)
- Writes `playbooks` row via `upsert_playbook()` (same as V2 — backward compat)
- Deletes old `playbook_actions` for this playbook via `delete_playbook_actions()`
- Inserts new `playbook_actions` rows via `insert_playbook_action()` for each priority action
- Returns confirmation with playbook ID and action count

Also build the `TOOL_DEFINITIONS` list and `TOOL_HANDLERS` dict, and the `web_search` server tool.

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_tools.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/tools.py tests/test_tools.py
git commit -m "feat: build strategist tools — write_playbook stores structured actions"
```

---

### Task 9: Executor Input/Output Contracts

Build the executor's typed contracts (`ExecutorInput`, `ExecutorDecision`, `PlaybookAction`) and the LLM integration that accepts structured JSON instead of prose context.

**Files:**
- Build: `trading/agent.py`
- Test: `tests/test_agent.py`

**Reference:** V2's `trading/agent.py` has `TradingDecision`, `AgentResponse`, `ThesisInvalidation`, `get_trading_decisions()`, `validate_decision()`, `TRADING_SYSTEM_PROMPT`. V3 rebuilds these around structured contracts.

**Step 1: Write failing tests**

```python
"""Tests for executor LLM integration."""
from decimal import Decimal
from unittest.mock import patch, MagicMock

from trading.agent import (
    ExecutorInput,
    ExecutorDecision,
    PlaybookAction,
    AgentResponse,
    ThesisInvalidation,
    get_trading_decisions,
    validate_decision,
    DEFAULT_EXECUTOR_MODEL,
)


class TestExecutorContracts:
    def test_executor_input_serializable(self):
        """ExecutorInput should be JSON-serializable."""
        inp = ExecutorInput(
            playbook_actions=[PlaybookAction(
                id=1, ticker="AAPL", action="buy", thesis_id=1,
                reasoning="Entry hit", confidence="high",
                max_quantity=Decimal("5"), priority=1,
            )],
            positions=[{"ticker": "MSFT", "shares": "10"}],
            account={"cash": "50000", "buying_power": "50000"},
            attribution_summary={"news_signal:earnings": {"win_rate_7d": 0.6, "sample_size": 20}},
            recent_outcomes=[],
            market_outlook="Bullish",
            risk_notes="Fed meeting tomorrow",
        )
        assert inp.playbook_actions[0].ticker == "AAPL"

    def test_executor_decision_has_playbook_action_id(self):
        """ExecutorDecision should have playbook_action_id field."""
        d = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=2.5, reasoning="Entry hit", confidence="high",
            is_off_playbook=False,
        )
        assert d.playbook_action_id == 1
        assert d.is_off_playbook is False

    def test_off_playbook_decision(self):
        """Off-playbook decisions should have playbook_action_id=None."""
        d = ExecutorDecision(
            playbook_action_id=None, ticker="NVDA", action="buy",
            quantity=1.0, reasoning="Urgent opportunity", confidence="medium",
            is_off_playbook=True,
        )
        assert d.playbook_action_id is None
        assert d.is_off_playbook is True


class TestGetTradingDecisions:
    def test_calls_haiku_with_structured_input(self):
        """Should serialize ExecutorInput as JSON for the user message."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"decisions":[],"thesis_invalidations":[],"market_summary":"Quiet day","risk_assessment":"Low"}')]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = MagicMock()

        executor_input = ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "50000"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="Neutral", risk_notes="",
        )

        with patch("trading.agent.get_claude_client", return_value=mock_client), \
             patch("trading.agent._call_with_retry", return_value=mock_response):
            response = get_trading_decisions(executor_input)

        assert isinstance(response, AgentResponse)
        assert response.market_summary == "Quiet day"


class TestValidateDecision:
    def test_buy_valid(self):
        d = ExecutorDecision(playbook_action_id=1, ticker="AAPL", action="buy",
                             quantity=2.0, reasoning="Buy", confidence="high",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {})
        assert valid is True

    def test_buy_exceeds_buying_power(self):
        d = ExecutorDecision(playbook_action_id=1, ticker="AAPL", action="buy",
                             quantity=1000, reasoning="Buy", confidence="high",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("100"), Decimal("150"), {})
        assert valid is False
        assert "buying power" in reason.lower()

    def test_sell_exceeds_held_shares(self):
        d = ExecutorDecision(playbook_action_id=1, ticker="AAPL", action="sell",
                             quantity=100, reasoning="Sell", confidence="high",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {"AAPL": Decimal("10")})
        assert valid is False
        assert "shares" in reason.lower()

    def test_hold_always_valid(self):
        d = ExecutorDecision(playbook_action_id=None, ticker="AAPL", action="hold",
                             quantity=None, reasoning="Wait", confidence="low",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {})
        assert valid is True
```

Run: `python3 -m pytest tests/test_agent.py -x -v`

Expected: FAIL

**Step 2: Build `trading/agent.py`**

```python
"""Executor LLM integration — gets structured trading decisions from Claude Haiku."""

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .claude_client import get_claude_client, _call_with_retry

logger = logging.getLogger(__name__)

DEFAULT_EXECUTOR_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class PlaybookAction:
    """A structured action from the playbook."""
    id: int
    ticker: str
    action: str
    thesis_id: int | None
    reasoning: str
    confidence: str
    max_quantity: Decimal | None
    priority: int


@dataclass
class ExecutorInput:
    """Structured input for the executor."""
    playbook_actions: list[PlaybookAction]
    positions: list[dict]
    account: dict
    attribution_summary: dict
    recent_outcomes: list[dict]
    market_outlook: str
    risk_notes: str


@dataclass
class ExecutorDecision:
    """A trading decision from the executor."""
    playbook_action_id: int | None
    ticker: str
    action: str
    quantity: float | None
    reasoning: str
    confidence: str
    is_off_playbook: bool
    signal_refs: list = None
    thesis_id: int | None = None

    def __post_init__(self):
        if self.signal_refs is None:
            self.signal_refs = []


@dataclass
class ThesisInvalidation:
    """A thesis flagged for invalidation."""
    thesis_id: int
    reason: str


@dataclass
class AgentResponse:
    """Full response from trading executor."""
    decisions: list[ExecutorDecision]
    thesis_invalidations: list[ThesisInvalidation]
    market_summary: str
    risk_assessment: str


TRADING_SYSTEM_PROMPT = """You are a trading executor. You receive structured JSON with playbook actions, positions, account state, and attribution data. You output trading decisions as JSON.

OUTPUT FORMAT: Respond with a single JSON object. No commentary, no markdown, no explanation outside the JSON.

INPUT FIELDS:
- playbook_actions: Today's priority actions from the strategist, each with id, ticker, action, reasoning, confidence, max_quantity
- positions: Current portfolio holdings
- account: Cash, buying power, equity
- attribution_summary: Signal category performance (win rates, sample sizes)
- recent_outcomes: Recent decision P&L results
- market_outlook: Strategist's market view
- risk_notes: Warnings, upcoming events

RULES:
- For each playbook action: execute as-is, adjust quantity, or skip with justification
- Include playbook_action_id in each decision to trace back to the playbook
- Set is_off_playbook: true for trades NOT in the playbook (urgent opportunities only)
- Conservative sizing: 1-5% of buying power per trade
- Never exceed available buying power
- Fractional shares supported — size by dollar amount (e.g., $500 in a $200 stock = 2.5 shares)
- If no playbook: hold everything, no new positions
- If uncertain: HOLD
- Every decision MUST cite signal_refs for the learning loop

JSON SCHEMA:
{"decisions": [{"playbook_action_id": 1, "ticker": "SYMBOL", "action": "buy|sell|hold", "quantity": 2.5, "reasoning": "...", "confidence": "high|medium|low", "is_off_playbook": false, "signal_refs": [{"type": "news_signal|thesis", "id": 0}]}], "thesis_invalidations": [{"thesis_id": 0, "reason": "..."}], "market_summary": "...", "risk_assessment": "..."}

If no trades: empty decisions array, explain in market_summary.
If no invalidations: empty thesis_invalidations array."""


def get_trading_decisions(executor_input: ExecutorInput, model: str = DEFAULT_EXECUTOR_MODEL) -> AgentResponse:
    """Get trading decisions from Claude Haiku given structured input."""
    client = get_claude_client()

    input_json = json.dumps({
        "playbook_actions": [
            {"id": a.id, "ticker": a.ticker, "action": a.action,
             "thesis_id": a.thesis_id, "reasoning": a.reasoning,
             "confidence": a.confidence,
             "max_quantity": float(a.max_quantity) if a.max_quantity else None,
             "priority": a.priority}
            for a in executor_input.playbook_actions
        ],
        "positions": executor_input.positions,
        "account": executor_input.account,
        "attribution_summary": executor_input.attribution_summary,
        "recent_outcomes": executor_input.recent_outcomes,
        "market_outlook": executor_input.market_outlook,
        "risk_notes": executor_input.risk_notes,
    }, default=str)

    response = _call_with_retry(
        client, model=model, max_tokens=4096,
        system=TRADING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": input_json}],
    )

    # Extract text, strip markdown fences, parse JSON
    response_text = "".join(b.text for b in response.content if hasattr(b, "text"))
    logger.info("Haiku tokens — input: %d, output: %d, stop_reason: %s",
                response.usage.input_tokens, response.usage.output_tokens, response.stop_reason)

    if response.stop_reason == "max_tokens":
        raise ValueError(f"Executor response truncated at {response.usage.output_tokens} tokens.")

    text = response_text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    data = json.loads(text.strip())

    decisions = [ExecutorDecision(
        playbook_action_id=d.get("playbook_action_id"),
        ticker=d.get("ticker", ""),
        action=d.get("action", "hold"),
        quantity=d.get("quantity"),
        reasoning=d.get("reasoning", ""),
        confidence=d.get("confidence", "low"),
        is_off_playbook=d.get("is_off_playbook", False),
        signal_refs=d.get("signal_refs", []),
        thesis_id=d.get("thesis_id"),
    ) for d in data.get("decisions", [])]

    invalidations = [ThesisInvalidation(
        thesis_id=inv["thesis_id"], reason=inv.get("reason", "")
    ) for inv in data.get("thesis_invalidations", [])]

    return AgentResponse(
        decisions=decisions,
        thesis_invalidations=invalidations,
        market_summary=data.get("market_summary", ""),
        risk_assessment=data.get("risk_assessment", ""),
    )


def validate_decision(
    decision: ExecutorDecision,
    buying_power: Decimal,
    current_price: Decimal,
    positions: dict[str, Decimal],
) -> tuple[bool, str]:
    """Validate a trading decision before execution."""
    if decision.action == "hold":
        return True, "Hold requires no validation"

    if decision.action == "buy":
        if decision.quantity is None or decision.quantity <= 0:
            return False, "Buy requires positive quantity"
        cost = current_price * Decimal(str(decision.quantity))
        if cost > buying_power:
            return False, f"Insufficient buying power: need ${cost:.2f}, have ${buying_power:.2f}"
        return True, "Buy order validated"

    if decision.action == "sell":
        if decision.quantity is None or decision.quantity <= 0:
            return False, "Sell requires positive quantity"
        held = positions.get(decision.ticker, Decimal(0))
        if Decimal(str(decision.quantity)) > held:
            return False, f"Insufficient shares: want to sell {decision.quantity}, hold {held}"
        return True, "Sell order validated"

    return False, f"Unknown action: {decision.action}"


def format_decisions_for_logging(response: AgentResponse) -> dict:
    """Format agent response for database logging."""
    return {
        "market_summary": response.market_summary,
        "risk_assessment": response.risk_assessment,
        "decision_count": len(response.decisions),
        "thesis_invalidation_count": len(response.thesis_invalidations),
    }
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_agent.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/agent.py tests/test_agent.py
git commit -m "feat: build executor contracts — structured JSON input, playbook traceability"
```

---

### Task 10: Executor Input Builder

Build the context builder that assembles structured `ExecutorInput` from database queries. Replaces V2's prose `build_trading_context()` with typed data, fetched one query per data source (no N+1 pattern).

**Files:**
- Build: `trading/context.py`
- Test: `tests/test_context.py`

**Reference:** V2's `trading/context.py` has the prose context functions (`get_portfolio_context()`, `get_macro_context()`, etc.) used by strategist tools, plus `build_trading_context()` for the executor. V3 keeps the prose functions (strategist tools still use them) and adds `build_executor_input()`.

**Step 1: Write failing tests**

```python
"""Tests for executor input builder."""
from datetime import date
from decimal import Decimal
from unittest.mock import patch

from trading.context import build_executor_input
from trading.agent import ExecutorInput, PlaybookAction


class TestBuildExecutorInput:
    def test_returns_executor_input(self, mock_db, mock_cursor):
        """Should return an ExecutorInput dataclass."""
        # Mock playbook lookup
        mock_cursor.fetchone.side_effect = [
            {"id": 1, "market_outlook": "Bullish", "risk_notes": "Fed", "priority_actions": [], "watch_list": []},
        ]
        mock_cursor.fetchall.side_effect = [
            [{"id": 1, "ticker": "AAPL", "action": "buy", "thesis_id": 1,
              "reasoning": "Entry", "confidence": "high", "max_quantity": Decimal("5"), "priority": 1}],
            [{"ticker": "MSFT", "shares": Decimal("10"), "avg_cost": Decimal("300")}],
            [],  # recent decisions
            [],  # attribution
        ]

        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        assert isinstance(result, ExecutorInput)
        assert len(result.playbook_actions) == 1
        assert result.playbook_actions[0].ticker == "AAPL"
        assert result.market_outlook == "Bullish"

    def test_no_playbook_returns_empty_actions(self, mock_db, mock_cursor):
        """Should return empty playbook_actions when no playbook exists."""
        mock_cursor.fetchone.return_value = None  # no playbook
        mock_cursor.fetchall.side_effect = [[], [], []]

        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        assert result.playbook_actions == []
        assert "No playbook" in result.market_outlook
```

Run: `python3 -m pytest tests/test_context.py::TestBuildExecutorInput -x -v`

Expected: FAIL

**Step 2: Build `trading/context.py`**

Keep all the prose context functions from V2 (strategist tools use them). Add the new function:

```python
from .agent import ExecutorInput, PlaybookAction
from .database.trading_db import (
    get_positions, get_playbook, get_playbook_actions,
    get_recent_decisions, get_open_orders,
)
from .database.connection import get_cursor
from .attribution import get_attribution_summary


def build_executor_input(account_info: dict, playbook_date: date = None) -> ExecutorInput:
    """Build structured executor input. Each data element fetched once."""
    if playbook_date is None:
        playbook_date = date.today()

    playbook = get_playbook(playbook_date)
    playbook_actions_rows = get_playbook_actions(playbook["id"]) if playbook else []

    actions = [
        PlaybookAction(
            id=row["id"], ticker=row["ticker"], action=row["action"],
            thesis_id=row.get("thesis_id"), reasoning=row.get("reasoning", ""),
            confidence=row.get("confidence", "medium"),
            max_quantity=row.get("max_quantity"), priority=row.get("priority", 99),
        )
        for row in playbook_actions_rows
    ]

    positions = get_positions()
    recent = get_recent_decisions(days=30)
    attribution_rows = get_signal_attribution()
    attribution_summary = {
        r["category"]: {"win_rate_7d": float(r["win_rate_7d"] or 0), "sample_size": r["sample_size"]}
        for r in attribution_rows
    }

    recent_outcomes = [
        {"date": str(d["date"]), "ticker": d["ticker"], "action": d["action"],
         "outcome_7d": float(d["outcome_7d"]) if d.get("outcome_7d") is not None else None}
        for d in recent if d.get("outcome_7d") is not None
    ][:10]

    return ExecutorInput(
        playbook_actions=actions,
        positions=[dict(p) for p in positions],
        account=account_info,
        attribution_summary=attribution_summary,
        recent_outcomes=recent_outcomes,
        market_outlook=playbook.get("market_outlook", "") if playbook else "No playbook available",
        risk_notes=playbook.get("risk_notes", "") if playbook else "",
    )
```

**Reference:** V2's `trading/context.py` has all the prose helper functions that need to be included. Use V2's `build_trading_context()` as a guide for which data sources to assemble — but return a typed dataclass, not a string.

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_context.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/context.py tests/test_context.py
git commit -m "feat: build executor input builder — structured data, no N+1 queries"
```

---

### Task 11: Trading Session Orchestrator

Build the trader module that runs the executor: syncs positions, takes snapshots, builds executor input, gets decisions from Haiku, validates, executes trades, and logs decisions with `playbook_action_id`.

**Files:**
- Build: `trading/trader.py`
- Test: `tests/test_trader.py`

**Reference:** V2's `trading/trader.py` has the `TradeResult`, `TradingSessionResult` dataclasses, and `run_trading_session()`. V3 replaces `build_trading_context()` with `build_executor_input()` and logs `playbook_action_id` + `is_off_playbook` on each decision.

**Step 1: Write failing tests**

```python
"""Tests for trading session executor."""
from decimal import Decimal
from unittest.mock import patch, MagicMock

from trading.trader import run_trading_session, TradingSessionResult
from trading.agent import ExecutorInput, ExecutorDecision, AgentResponse


class TestRunTradingSession:
    def test_uses_structured_executor_input(self, mock_db, mock_cursor):
        """Should call build_executor_input, not build_trading_context."""
        mock_cursor.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")},
            {"id": 1},  # snapshot
        ]

        with patch("trading.trader.sync_positions_from_alpaca", return_value=2), \
             patch("trading.trader.sync_orders_from_alpaca", return_value=0), \
             patch("trading.trader.get_account_info") as mock_acct, \
             patch("trading.trader.take_account_snapshot", return_value=1), \
             patch("trading.trader.build_executor_input") as mock_build, \
             patch("trading.trader.get_trading_decisions") as mock_decisions:

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[],
                market_summary="No trades", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=True)

        mock_build.assert_called_once()
        assert isinstance(result, TradingSessionResult)

    def test_logs_playbook_action_id(self, mock_db, mock_cursor):
        """Decisions should log playbook_action_id and is_off_playbook."""
        # Test that insert_decision is called with the new fields
        ...
```

Run: `python3 -m pytest tests/test_trader.py -x -v`

Expected: FAIL

**Step 2: Build `trading/trader.py`**

```python
"""Trading agent orchestrator — daily automation entry point."""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal

from .context import build_executor_input
from .executor import (
    get_account_info, take_account_snapshot,
    sync_positions_from_alpaca, sync_orders_from_alpaca,
    execute_market_order, get_latest_price,
)
from .agent import (
    get_trading_decisions, validate_decision, format_decisions_for_logging,
    AgentResponse, ExecutorDecision, DEFAULT_EXECUTOR_MODEL,
)
from .database.trading_db import (
    insert_decision, get_positions, close_thesis, insert_decision_signals_batch,
)

logger = logging.getLogger("trader")


@dataclass
class TradeResult:
    decision: ExecutorDecision
    executed: bool
    order_id: str | None
    filled_price: Decimal | None
    error: str | None


@dataclass
class TradingSessionResult:
    timestamp: datetime
    account_snapshot_id: int
    positions_synced: int
    orders_synced: int
    decisions_made: int
    trades_executed: int
    trades_failed: int
    total_buy_value: Decimal
    total_sell_value: Decimal
    errors: list[str]


def run_trading_session(dry_run: bool = False, model: str = DEFAULT_EXECUTOR_MODEL) -> TradingSessionResult:
    """
    Run a complete trading session.

    1. Sync positions/orders from Alpaca
    2. Take account snapshot
    3. Build structured ExecutorInput
    4. Get decisions from Haiku
    5. Validate and execute trades
    6. Log decisions with playbook_action_id
    """
    errors = []
    timestamp = datetime.now()

    # Step 1: Sync positions and orders
    # ... same pattern as V2 ...

    # Step 2: Account snapshot
    # ... same pattern as V2 ...

    # Step 3: Build structured executor input (replaces prose build_trading_context)
    executor_input = build_executor_input(account_info)

    # Step 4: Get LLM decisions
    response = get_trading_decisions(executor_input, model=model)

    # Step 5: Validate and execute trades
    positions = {p["ticker"]: p["shares"] for p in get_positions()}
    buying_power = account_info["buying_power"]

    for decision in response.decisions:
        if decision.action == "hold":
            continue

        price = get_latest_price(decision.ticker)
        is_valid, reason = validate_decision(decision, buying_power, price, positions)
        if not is_valid:
            continue

        result = execute_market_order(
            ticker=decision.ticker, side=decision.action,
            qty=Decimal(decision.quantity), dry_run=dry_run,
        )

        # ... handle result, update buying_power ...

        # Mark thesis as executed if applicable
        if decision.thesis_id and not dry_run and result.success:
            close_thesis(thesis_id=decision.thesis_id, status="executed",
                         reason=f"Trade executed: {decision.action} {decision.quantity} shares")

    # Step 5b: Process thesis invalidations
    # ... same as V2 ...

    # Step 6: Log decisions (with new playbook_action_id + is_off_playbook)
    signals_used = format_decisions_for_logging(response)

    for decision in response.decisions:
        price = get_latest_price(decision.ticker)
        decision_id = insert_decision(
            decision_date=date.today(),
            ticker=decision.ticker,
            action=decision.action,
            quantity=Decimal(decision.quantity) if decision.quantity else None,
            price=price,
            reasoning=decision.reasoning,
            signals_used=signals_used,
            account_equity=account_info["portfolio_value"],
            buying_power=account_info["buying_power"],
            playbook_action_id=decision.playbook_action_id,  # V3 addition
            is_off_playbook=decision.is_off_playbook,        # V3 addition
        )

        if decision.signal_refs:
            signal_links = [(decision_id, ref["type"], ref["id"]) for ref in decision.signal_refs]
            insert_decision_signals_batch(signal_links)

    return TradingSessionResult(...)
```

**Reference:** V2's `trading/trader.py` has the complete error handling, summary logging, and return structure. The structure is the same in V3 — the key differences are:
1. `build_executor_input()` instead of `build_trading_context()`
2. `ExecutorDecision` instead of `TradingDecision`
3. `insert_decision()` receives `playbook_action_id` and `is_off_playbook`

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_trader.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/trader.py tests/test_trader.py
git commit -m "feat: build trading session — structured executor input, playbook traceability"
```

---

## Phase 5: Integration

### Task 12: Strategist with Attribution Constraints

Build the strategist orchestration module that runs the Claude agentic loop with attribution constraints injected into the system prompt.

**Files:**
- Build: `trading/ideation_claude.py`
- Test: `tests/test_ideation_claude.py`

**Reference:** V2's `trading/ideation_claude.py` has the system prompts (`CLAUDE_IDEATION_SYSTEM`, `_STRATEGIST_TEMPLATE`), `run_strategist_loop()`, and `run_strategist_session()`. V3 keeps the prompts and loop structure but changes two things:
1. `run_strategist_loop()` accepts `attribution_constraints: str` and appends it to the system prompt
2. `run_strategist_session()` no longer runs backfill/attribution internally — that's Stage 0's job

**Step 1: Write failing tests**

```python
"""Tests for Claude strategist."""
from unittest.mock import patch, MagicMock

from trading.ideation_claude import run_strategist_loop, ClaudeIdeationResult


class TestStrategistAttributionConstraints:
    def test_attribution_injected_into_system_prompt(self):
        """Attribution constraints should be appended to the system prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Analysis complete.")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100,
                                        cache_creation_input_tokens=0, cache_read_input_tokens=0)

        constraints = "SIGNAL PERFORMANCE:\n  WEAK: news_signal:legal (38%, n=12)\nCONSTRAINT: Do not use WEAK signals"

        with patch("trading.ideation_claude.get_claude_client", return_value=mock_client), \
             patch("trading.ideation_claude.run_agentic_loop") as mock_loop:
            mock_loop.return_value = ([], mock_response.usage)

            result = run_strategist_loop(
                model="claude-opus-4-6",
                max_turns=1,
                attribution_constraints=constraints,
            )

        # Verify the system prompt includes constraints
        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert "WEAK" in system_prompt
        assert "CONSTRAINT" in system_prompt

    def test_runs_without_constraints(self):
        """Should work fine with empty attribution_constraints."""
        ...
```

Run: `python3 -m pytest tests/test_ideation_claude.py -x -v`

Expected: FAIL

**Step 2: Build `trading/ideation_claude.py`**

```python
"""Claude-based ideation module for autonomous trade idea generation."""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

from .claude_client import get_claude_client, run_agentic_loop, extract_final_text
from .tools import TOOL_DEFINITIONS, TOOL_HANDLERS, reset_session

logger = logging.getLogger(__name__)

# System prompts — same text as V2
CLAUDE_IDEATION_SYSTEM = """..."""      # Reference: V2 lines 16-54
_STRATEGIST_TEMPLATE = """..."""         # Reference: V2 lines 57-80+

CLAUDE_SESSION_STRATEGIST_SYSTEM = _STRATEGIST_TEMPLATE.format(
    timing="after market close",
    review_scope="today's signals, decision outcomes, and thesis status",
    date_ref="tomorrow's",
    executor_note=" that will run before market open",
)


@dataclass
class ClaudeIdeationResult:
    """Result from a strategist session."""
    theses_created: int
    theses_updated: int
    theses_closed: int
    playbook_written: bool
    turns_used: int
    total_input_tokens: int
    total_output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    final_summary: str


def run_strategist_loop(
    model: str = "claude-opus-4-6",
    max_turns: int = 25,
    system_prompt: str = None,
    attribution_constraints: str = "",
) -> ClaudeIdeationResult:
    """Run the strategist agentic loop.

    attribution_constraints is appended to the system prompt, making
    signal performance data enforceable rather than advisory.
    """
    base_prompt = system_prompt or CLAUDE_SESSION_STRATEGIST_SYSTEM
    if attribution_constraints:
        base_prompt = base_prompt + "\n\n" + attribution_constraints

    reset_session()
    client = get_claude_client()

    # Run agentic loop with tools
    tool_results, usage = run_agentic_loop(
        client=client,
        model=model,
        system=base_prompt,
        tools=TOOL_DEFINITIONS,
        tool_handlers=TOOL_HANDLERS,
        max_turns=max_turns,
    )

    # Count actions from tool results
    # ... same logic as V2 ...

    return ClaudeIdeationResult(...)


def run_strategist_session(
    model: str = "claude-opus-4-6",
    max_turns: int = 25,
    attribution_constraints: str = "",
) -> ClaudeIdeationResult:
    """Entry point for strategist. No longer runs backfill/attribution internally."""
    return run_strategist_loop(
        model=model,
        max_turns=max_turns,
        attribution_constraints=attribution_constraints,
    )
```

**Reference:** V2's `trading/ideation_claude.py` has the full `run_strategist_loop()` and `run_strategist_session()` implementations including token counting, tool result counting, and the `ClaudeIdeationResult` construction. Port all of that, changing only: (1) `attribution_constraints` parameter on `run_strategist_loop()`, (2) remove backfill/attribution calls from `run_strategist_session()`.

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_ideation_claude.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/ideation_claude.py tests/test_ideation_claude.py
git commit -m "feat: build strategist — attribution constraints in system prompt"
```

---

### Task 13: Session Orchestrator (4 Stages)

Build the session orchestrator that runs all four stages in sequence: learning refresh → news pipeline → strategist → executor. Stage 0 produces attribution constraints that Stage 2 consumes.

**Files:**
- Build: `trading/session.py`
- Test: `tests/test_session.py`

**Reference:** V2's `trading/session.py` has the `SessionResult` dataclass and `run_session()` with 3 stages. V3 adds Stage 0 (learning refresh) and passes `attribution_constraints` to Stage 2.

**Step 1: Write failing tests**

```python
"""Tests for 4-stage session orchestrator."""
from unittest.mock import patch, MagicMock

from trading.session import run_session, SessionResult


class TestRunSession:
    def test_stage_0_runs_before_pipeline(self):
        """Learning refresh (backfill + attribution) runs before news pipeline."""
        call_order = []

        with patch("trading.session.run_backfill") as mock_backfill, \
             patch("trading.session.compute_signal_attribution") as mock_attr, \
             patch("trading.session.build_attribution_constraints", return_value="CONSTRAINTS") as mock_constraints, \
             patch("trading.session.run_pipeline") as mock_pipeline, \
             patch("trading.session.run_strategist_loop") as mock_strat, \
             patch("trading.session.run_trading_session") as mock_trade:

            mock_backfill.side_effect = lambda: call_order.append("backfill")
            mock_attr.side_effect = lambda: call_order.append("attribution") or []
            mock_pipeline.side_effect = lambda **kw: call_order.append("pipeline")
            mock_strat.side_effect = lambda **kw: call_order.append("strategist")
            mock_trade.side_effect = lambda **kw: call_order.append("trader")

            run_session(dry_run=True)

        assert call_order.index("backfill") < call_order.index("pipeline")
        assert call_order.index("attribution") < call_order.index("strategist")

    def test_attribution_constraints_passed_to_strategist(self):
        """Stage 0's constraints should be passed to Stage 2."""
        with patch("trading.session.run_backfill"), \
             patch("trading.session.compute_signal_attribution", return_value=[]), \
             patch("trading.session.build_attribution_constraints", return_value="STRONG: earnings"), \
             patch("trading.session.run_pipeline"), \
             patch("trading.session.run_strategist_loop") as mock_strat, \
             patch("trading.session.run_trading_session"):

            run_session(dry_run=True)

        mock_strat.assert_called_once()
        assert "STRONG: earnings" in str(mock_strat.call_args)

    def test_stage_0_failure_does_not_block(self):
        """If learning refresh fails, session continues with stale data."""
        with patch("trading.session.run_backfill", side_effect=Exception("DB error")), \
             patch("trading.session.run_pipeline"), \
             patch("trading.session.run_strategist_loop"), \
             patch("trading.session.run_trading_session"):

            result = run_session(dry_run=True)

        assert result.learning_error is not None
        assert "DB error" in result.learning_error


class TestSessionResult:
    def test_has_learning_error_field(self):
        result = SessionResult()
        assert hasattr(result, 'learning_error')
```

Run: `python3 -m pytest tests/test_session.py -x -v`

Expected: FAIL

**Step 2: Build `trading/session.py`**

```python
"""Consolidated daily session orchestrator.

Runs the full daily pipeline in a single invocation:
  Stage 0: Learning refresh (backfill + attribution)
  Stage 1: News pipeline (fetch, classify, store)
  Stage 2: Claude strategist (thesis management + playbook generation)
  Stage 3: Trading executor (decisions + order execution)

Each stage is independent — failures are captured and do not prevent
subsequent stages from running.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from .log_config import setup_logging
from .backfill import run_backfill
from .attribution import compute_signal_attribution, build_attribution_constraints
from .pipeline import PipelineStats, run_pipeline
from .ideation_claude import ClaudeIdeationResult, run_strategist_loop
from .agent import DEFAULT_EXECUTOR_MODEL
from .trader import TradingSessionResult, run_trading_session

logger = logging.getLogger("session")


@dataclass
class SessionResult:
    pipeline_result: Optional[PipelineStats] = None
    strategist_result: Optional[ClaudeIdeationResult] = None
    trading_result: Optional[TradingSessionResult] = None

    learning_error: Optional[str] = None     # V3: Stage 0
    pipeline_error: Optional[str] = None
    strategist_error: Optional[str] = None
    trading_error: Optional[str] = None

    skipped_pipeline: bool = False
    skipped_ideation: bool = False
    duration_seconds: float = 0.0

    @property
    def has_errors(self) -> bool:
        return any([self.learning_error, self.pipeline_error,
                    self.strategist_error, self.trading_error])


def run_session(
    dry_run: bool = False,
    model: str = "claude-opus-4-6",
    executor_model: str = DEFAULT_EXECUTOR_MODEL,
    max_turns: int = 25,
    skip_pipeline: bool = False,
    skip_ideation: bool = False,
    pipeline_hours: int = 24,
    pipeline_limit: int = 300,
) -> SessionResult:
    start = time.monotonic()
    result = SessionResult(skipped_pipeline=skip_pipeline, skipped_ideation=skip_ideation)

    # Stage 0: Refresh learning data
    attribution_constraints = ""
    logger.info("[Stage 0] Refreshing learning data")
    try:
        run_backfill()
        compute_signal_attribution()
        attribution_constraints = build_attribution_constraints()
        logger.info("Learning refresh complete")
    except Exception as e:
        result.learning_error = str(e)
        logger.warning("Learning refresh failed: %s — continuing with stale data", e)

    # Stage 1: News pipeline
    if skip_pipeline:
        logger.info("[Stage 1] News pipeline — SKIPPED")
    else:
        logger.info("[Stage 1] Running news pipeline")
        try:
            result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit)
        except Exception as e:
            result.pipeline_error = str(e)
            logger.error("Pipeline failed: %s — continuing with existing signals", e)

    # Stage 2: Claude strategist (receives attribution constraints)
    if skip_ideation:
        logger.info("[Stage 2] Strategist — SKIPPED")
    else:
        logger.info("[Stage 2] Running Claude strategist")
        try:
            result.strategist_result = run_strategist_loop(
                model=model,
                max_turns=max_turns,
                attribution_constraints=attribution_constraints,
            )
        except Exception as e:
            result.strategist_error = str(e)
            logger.error("Strategist failed: %s — continuing with existing playbook", e)

    # Stage 3: Trading session
    logger.info("[Stage 3] Running trading session")
    try:
        result.trading_result = run_trading_session(dry_run=dry_run, model=executor_model)
    except Exception as e:
        result.trading_error = str(e)
        logger.error("Trading session failed: %s", e)

    result.duration_seconds = time.monotonic() - start

    # Summary
    logger.info("=" * 60)
    logger.info("Session complete in %.1fs", result.duration_seconds)
    if result.has_errors:
        for field_name in ["learning_error", "pipeline_error", "strategist_error", "trading_error"]:
            err = getattr(result, field_name)
            if err:
                logger.error("  %s: %s", field_name, err)
    else:
        logger.info("  All stages completed successfully")
    logger.info("=" * 60)

    return result


def main():
    """CLI entry point for consolidated daily session."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run consolidated daily trading session")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--executor-model", default=DEFAULT_EXECUTOR_MODEL)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-ideation", action="store_true")
    parser.add_argument("--pipeline-hours", type=int, default=24)

    args = parser.parse_args()
    result = run_session(
        dry_run=args.dry_run, model=args.model, executor_model=args.executor_model,
        max_turns=args.max_turns, skip_pipeline=args.skip_pipeline,
        skip_ideation=args.skip_ideation, pipeline_hours=args.pipeline_hours,
    )
    if result.has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_session.py -x -v`

Expected: PASS

**Step 4: Commit**

```bash
git add trading/session.py tests/test_session.py
git commit -m "feat: build 4-stage session — learning refresh before pipeline/strategist/executor"
```

---

### Task 14: Test Fixtures and Conftest

Build the test configuration: shared fixtures, factory functions, and mock setup for the V3 module structure.

**Files:**
- Build: `tests/conftest.py`

**Reference:** V2's `tests/conftest.py` has `mock_cursor`, `mock_db`, and factory functions (`make_news_item()`, `make_trading_decision()`, `make_agent_response()`, etc.). V3 keeps the same pattern but:
- Mocks `trading.database.connection.get_cursor` instead of `trading.db.get_cursor`
- Removes Ollama fixtures (`mock_embed`, `mock_chat_json`, `ollama_env`)
- Removes `FilteredNewsItem` import
- Adds `make_playbook_action_row()` factory
- Adds `make_executor_decision()` factory (for `ExecutorDecision`)

**Step 1: Build `tests/conftest.py`**

```python
"""Shared test fixtures and factory functions."""

import os
import sys
import pytest
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import MagicMock, patch


# --- Core DB Fixtures ---

@pytest.fixture
def mock_cursor():
    """Create a mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    """Patch get_cursor to yield a mock cursor."""
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("trading.database.connection.get_cursor", _get_cursor), \
         patch("trading.database.connection.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor


# --- Claude Fixtures ---

@pytest.fixture
def mock_claude_client():
    """Mock Claude API client."""
    client = MagicMock()
    with patch("trading.claude_client.get_claude_client", return_value=client):
        yield client


# --- Factory Functions ---

def make_news_item(**kwargs):
    """Create a news item dict like what Alpaca returns."""
    defaults = {
        "headline": "Test headline",
        "published_at": datetime.now(),
        "source": "test",
        "url": "https://example.com",
        "symbols": ["AAPL"],
    }
    defaults.update(kwargs)

    class NewsItem:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    return NewsItem(**defaults)


def make_trading_decision(**kwargs):
    """Create an ExecutorDecision for testing."""
    from trading.agent import ExecutorDecision
    defaults = {
        "playbook_action_id": 1,
        "ticker": "AAPL",
        "action": "buy",
        "quantity": 2.5,
        "reasoning": "Entry trigger hit",
        "confidence": "high",
        "is_off_playbook": False,
        "signal_refs": [{"type": "news_signal", "id": 1}],
        "thesis_id": 1,
    }
    defaults.update(kwargs)
    return ExecutorDecision(**defaults)


def make_agent_response(**kwargs):
    """Create an AgentResponse for testing."""
    from trading.agent import AgentResponse
    defaults = {
        "decisions": [],
        "thesis_invalidations": [],
        "market_summary": "Test summary",
        "risk_assessment": "Low risk",
    }
    defaults.update(kwargs)
    return AgentResponse(**defaults)


def make_ticker_signal(**kwargs):
    """Create a TickerSignal for testing."""
    from trading.classifier import TickerSignal
    defaults = {
        "ticker": "AAPL",
        "headline": "Apple reports earnings",
        "category": "earnings",
        "sentiment": "bullish",
        "confidence": "high",
        "published_at": datetime.now(),
    }
    defaults.update(kwargs)
    return TickerSignal(**defaults)


def make_position_row(**kwargs):
    defaults = {"ticker": "AAPL", "shares": Decimal("10"), "avg_cost": Decimal("150.00")}
    defaults.update(kwargs)
    return defaults


def make_thesis_row(**kwargs):
    defaults = {
        "id": 1, "ticker": "AAPL", "direction": "long", "confidence": "high",
        "thesis": "Strong earnings growth", "entry_trigger": "Price > $150",
        "exit_trigger": "Price > $180", "invalidation": "Earnings miss",
        "status": "active", "source_signals": None,
        "created_at": datetime.now(), "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_decision_row(**kwargs):
    defaults = {
        "id": 1, "date": date.today(), "ticker": "AAPL", "action": "buy",
        "quantity": Decimal("5"), "price": Decimal("150"), "reasoning": "Test",
        "signals_used": {}, "account_equity": Decimal("100000"),
        "buying_power": Decimal("50000"), "outcome_7d": Decimal("2.5"),
        "outcome_30d": None, "playbook_action_id": None, "is_off_playbook": False,
    }
    defaults.update(kwargs)
    return defaults


def make_snapshot_row(**kwargs):
    defaults = {
        "id": 1, "portfolio_value": Decimal("100000"),
        "cash": Decimal("50000"), "buying_power": Decimal("50000"),
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_news_signal_row(**kwargs):
    defaults = {
        "id": 1, "ticker": "AAPL", "headline": "Test", "category": "earnings",
        "sentiment": "bullish", "confidence": "high", "published_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_macro_signal_row(**kwargs):
    defaults = {
        "id": 1, "headline": "Fed holds rates", "category": "fed",
        "affected_sectors": ["finance"], "sentiment": "neutral",
        "published_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_playbook_action_row(**kwargs):
    """Create a playbook action dict like what DB returns."""
    defaults = {
        "id": 1, "playbook_id": 1, "ticker": "AAPL", "action": "buy",
        "thesis_id": 1, "reasoning": "Entry trigger hit", "confidence": "high",
        "max_quantity": Decimal("5"), "priority": 1, "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults
```

**Reference:** V2's `tests/conftest.py` has the complete fixtures. Port all factory functions, updating imports and field names for V3 contracts.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -v`

Expected: All tests pass.

**Step 3: Run coverage**

Run: `python3 -m pytest tests/ --cov=trading --cov=dashboard`

**Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: build test fixtures for V3 — shared DB mock, factory functions, no Ollama"
```

---

### Task 15: Remaining Modules (Carry-Forward)

Build the modules that carry forward from V2 with minimal changes. These modules have stable interfaces that V3 reuses directly.

**Files:**
- Build: `trading/news.py` — Alpaca News API client
- Build: `trading/executor.py` — Alpaca trade execution
- Build: `trading/backfill.py` — Outcome backfill (7d/30d P&L)
- Build: `trading/claude_client.py` — Claude API wrapper + agentic loop
- Build: `trading/market_data.py` — Sector/index/mover data
- Build: `trading/learn.py` — Standalone learning loop
- Build: `trading/log_config.py` — Logging setup
- Build: `trading/__init__.py` — Package init
- Build: `dashboard/app.py` — Flask routes

**Reference:** V2 has working implementations of all these modules. The only change required is import paths — they should import from `trading.database.trading_db` or `trading.database.dashboard_db` instead of `trading.db` or `queries`.

For `dashboard/app.py`: V2 does `from queries import ...`. V3 should do `from trading.database.dashboard_db import ...`.

**Step 1: Build each module**

For each module, use the V2 implementation as the source. Update imports:
- `from .db import X` → `from .database.trading_db import X`
- `from .db import get_cursor` → `from .database.connection import get_cursor`
- Remove any `from .ollama import ...` or `from .filter import ...`

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -v`

**Step 3: Commit**

```bash
git add trading/ dashboard/
git commit -m "feat: build carry-forward modules — news, executor, backfill, claude_client, dashboard"
```

---

### Task 16: Final Integration Test

Verify the complete V3 build works end-to-end.

**Step 1: Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: All tests pass.

**Step 2: Run coverage check**

```bash
python3 -m pytest tests/ --cov=trading --cov=dashboard --cov-report=term-missing
```

**Step 3: Verify Docker builds**

```bash
docker compose build
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: V3 complete — closed learning loop, structured data flow, Claude-only"
```

---

## Task Dependency Graph

```
Phase 1 (Foundation):
  Task 1 (Schema) ─────────────────┐
  Task 2 (Docker) ─────────────────┤
  Task 3 (DB Layer) ───────────────┤
                                    │
Phase 2 (Pipeline):                 │
  Task 4 (Classifier) ─── depends on Task 3
  Task 5 (Pipeline) ───── depends on Task 4
                                    │
Phase 3 (Learning):                 │
  Task 6 (Attribution) ── depends on Task 3
  Task 7 (Patterns) ───── depends on Task 3
                                    │
Phase 4 (Data Flow):                │
  Task 8 (Playbook Tool) ─ depends on Tasks 1, 3
  Task 9 (Agent Contracts)─ depends on Task 1
  Task 10 (Context) ───── depends on Tasks 1, 3, 9
  Task 11 (Trader) ────── depends on Tasks 9, 10
                                    │
Phase 5 (Integration):              │
  Task 12 (Strategist) ── depends on Task 6
  Task 13 (Session) ───── depends on Tasks 5, 6, 11, 12
  Task 14 (Tests) ─────── depends on all above
  Task 15 (Carry-Forward)─ depends on Task 3
  Task 16 (Final) ─────── depends on all above
```

## Parallelizable Tasks

Within each phase, some tasks can run in parallel:
- **Phase 1:** Tasks 1, 2 in parallel. Task 3 after both.
- **Phase 2:** Sequential (4 then 5).
- **Phase 3:** Tasks 6, 7 in parallel.
- **Phase 4:** Task 8 in parallel with 9. Task 10 after 9. Task 11 after 10.
- **Phase 5:** Task 15 can run in parallel with Tasks 12-13. Task 14 after all. Task 16 last.
