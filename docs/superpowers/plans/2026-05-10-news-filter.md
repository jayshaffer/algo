# Haiku News Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a Haiku 4.5 relevance filter between the local `news_signals` cache and the strategist, so Opus receives a curated subset of news rather than the 936-row firehose, while keeping the raw firehose accessible via the unchanged `get_news_signals` tool.

**Architecture:** Five tightly-coupled components: (1) a schema migration adding a nullable `summary TEXT` column to `news_signals`, (2) a pipeline change that persists `NewsItem.summary` (already plumbed from Alpaca), (3) a new `v2/news_filter.py` exposing `curate_signals(signals, target_n, regime_context) -> list[int]` that calls Haiku and degrades to firehose on any error, (4) a new `tool_get_curated_news` in `v2/tools.py` plus tool registration, and (5) a one-shot `v2/news_backfill.py` that re-fetches the last 7 days from Alpaca and fills `summary` on existing rows. A single sentence is added to the strategist system prompt; the existing `get_news_signals` tool is unchanged.

**Tech Stack:** Python 3, anthropic SDK (Haiku 4.5), psycopg2, PostgreSQL, pytest, Alpaca News API.

**Spec:** `docs/superpowers/specs/2026-05-10-news-filter-design.md`

---

## File Structure

**Create:**
- `db/init/027_news_signals_summary.sql` — schema migration
- `v2/news_filter.py` — Haiku filter (one public function: `curate_signals`)
- `v2/news_backfill.py` — one-shot CLI to backfill summaries
- `tests/v2/test_news_filter.py`
- `tests/v2/test_news_backfill.py`

**Modify:**
- `v2/database/trading_db.py` — `insert_news_signals_batch` carries `summary`; `get_news_signals` returns it
- `v2/pipeline.py` — pass `signal.summary` into the batch insert tuple (signal struct change in `classifier.py` if needed; see Task 2)
- `v2/classifier.py` — propagate `summary` from `NewsItem` through to the per-signal record
- `v2/tools.py` — add `tool_get_curated_news`, register in `TOOL_DEFINITIONS` + `TOOL_HANDLERS`, update existing `get_news_signals` tool description
- `v2/ideation_claude.py` — one-sentence prompt nudge in `_STRATEGIST_TEMPLATE`
- `tests/v2/test_tools.py` — new test class for curated tool
- `tests/v2/test_pipeline.py` (or create if missing) — summary persistence tests

---

## Task 1: Schema migration

**Files:**
- Create: `db/init/027_news_signals_summary.sql`

- [ ] **Step 1: Write the migration SQL**

Create `db/init/027_news_signals_summary.sql`:

```sql
-- db/init/027_news_signals_summary.sql
-- Adds a nullable `summary` column to news_signals so the Haiku
-- relevance filter has more than 60 chars of headline to rank on.
-- Nullable so pre-backfill rows aren't broken and existing inserts
-- without the column keep working until pipeline is updated.
-- See docs/superpowers/specs/2026-05-10-news-filter-design.md

ALTER TABLE news_signals
    ADD COLUMN IF NOT EXISTS summary TEXT;
```

- [ ] **Step 2: Apply the migration to both paper and prod DBs**

The DB init scripts run only on fresh container creation. For an existing DB, apply the ALTER manually:

```
docker exec algo-db-paper-1 psql -U algo -d trading -f /docker-entrypoint-initdb.d/027_news_signals_summary.sql
docker exec algo-db-1       psql -U algo -d trading -f /docker-entrypoint-initdb.d/027_news_signals_summary.sql
```

Verify:
```
docker exec algo-db-paper-1 psql -U algo -d trading -c "\d news_signals" | grep summary
docker exec algo-db-1       psql -U algo -d trading -c "\d news_signals" | grep summary
```

Expected: both show ` summary | text ` in the column list.

- [ ] **Step 3: Commit**

```
git add db/init/027_news_signals_summary.sql
git commit -m "$(cat <<'EOF'
feat(schema): add summary column to news_signals

Nullable TEXT column for storing Alpaca-provided article summaries.
The Haiku news filter (next commits) needs more than 60 chars of
headline to rank signals by relevance. Backfill of existing rows
ships in a separate one-shot script.

Spec: docs/superpowers/specs/2026-05-10-news-filter-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Pipeline persists summary

The strategist's news pipeline already fetches `summary` from Alpaca via `v2/news.py` (the `NewsItem` dataclass has `summary: str`), but `v2/classifier.py` and `v2/pipeline.py` drop it before the DB insert. This task plumbs it through.

**Files:**
- Modify: `v2/database/trading_db.py:24-44` (insert_news_signals_batch)
- Modify: `v2/database/trading_db.py:47-61` (get_news_signals SELECT *)
- Modify: `v2/classifier.py` (TickerSignal dataclass — add summary field; classifier propagates it)
- Modify: `v2/pipeline.py:50-55` (include summary in tuple)
- Test: `tests/v2/test_pipeline.py` (create if missing)

- [ ] **Step 1: Check the current TickerSignal dataclass shape**

Run:
```
grep -n "class TickerSignal\|class ClassificationResult\|class MacroSignal" v2/classifier.py
```

Note the dataclass definitions you find. The next steps assume `TickerSignal` already has fields `(ticker, headline, category, sentiment, confidence, published_at, alpaca_id)`. You'll add `summary: str = ""` to it.

- [ ] **Step 2: Write the failing test for pipeline summary persistence**

If `tests/v2/test_pipeline.py` does not exist yet, create it with the imports below. Otherwise append the test class.

Create or append to `tests/v2/test_pipeline.py`:

```python
"""Tests for v2/pipeline.py - news ingestion summary persistence."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from v2.pipeline import run_pipeline


class TestPipelineSummaryPersistence:
    """News pipeline must persist NewsItem.summary so the Haiku filter
    has more than the 60-char headline excerpt to rank on."""

    @patch("v2.pipeline.insert_news_signals_batch")
    @patch("v2.pipeline.insert_macro_signals_batch")
    @patch("v2.pipeline.classify_news_batch")
    @patch("v2.pipeline.fetch_broad_news")
    def test_pipeline_passes_summary_to_db_layer(
        self, mock_fetch, mock_classify, mock_macro_insert, mock_ticker_insert
    ):
        from v2.news import NewsItem
        from v2.classifier import ClassificationResult, TickerSignal

        # Alpaca returned one news item with a real summary.
        mock_fetch.return_value = [
            NewsItem(
                id="alp-123",
                headline="AAPL hits ATH",
                summary="Apple closed at $300 after the Foxconn deal.",
                author="x",
                source="Reuters",
                symbols=["AAPL"],
                published_at=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
                url="https://example.com",
            ),
        ]

        # Classifier produces one ticker signal carrying the summary through.
        mock_classify.return_value = [
            ClassificationResult(
                news_type="ticker",
                ticker_signals=[
                    TickerSignal(
                        ticker="AAPL",
                        headline="AAPL hits ATH",
                        summary="Apple closed at $300 after the Foxconn deal.",
                        category="momentum",
                        sentiment="bullish",
                        confidence="high",
                        published_at=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
                        alpaca_id="alp-123",
                    )
                ],
                macro_signal=None,
            ),
        ]
        mock_ticker_insert.return_value = 1
        mock_macro_insert.return_value = 0

        run_pipeline(hours=1, limit=10, dry_run=False)

        # The pipeline must include summary in the tuple passed to the DB.
        assert mock_ticker_insert.call_count == 1
        ticker_tuples = mock_ticker_insert.call_args[0][0]
        assert len(ticker_tuples) == 1, f"expected 1 tuple, got {ticker_tuples}"
        # The tuple shape after this change is:
        # (ticker, headline, category, sentiment, confidence, published_at, alpaca_id, summary)
        assert ticker_tuples[0][-1] == "Apple closed at $300 after the Foxconn deal.", (
            f"summary not at last position; tuple was {ticker_tuples[0]}"
        )
```

- [ ] **Step 3: Run the test and verify it fails**

From `/home/jay/dev/algo`:
```
python3 -m pytest tests/v2/test_pipeline.py::TestPipelineSummaryPersistence -v
```

Expected: FAIL because `TickerSignal` doesn't accept a `summary` argument yet (or because the pipeline doesn't pass it through).

- [ ] **Step 4: Add `summary` field to `TickerSignal`**

In `v2/classifier.py`, find the `TickerSignal` dataclass. Add `summary: str = ""` as a field (default empty string so existing test fixtures don't break). Keep all other fields unchanged.

After the edit, confirm with grep:
```
grep -A8 "class TickerSignal" v2/classifier.py
```

Expected: the dataclass body now lists `summary: str = ""` among its fields.

- [ ] **Step 5: Make the classifier carry summary from NewsItem to TickerSignal**

Find where the classifier constructs `TickerSignal` objects from incoming news items. The classifier likely processes a batch and emits a result per item. At each construction site, pass `summary=item.summary` (or equivalent — find the `NewsItem` reference in scope at the construction site).

Grep to find construction sites:
```
grep -n "TickerSignal(" v2/classifier.py
```

For each `TickerSignal(...)` call, add `summary=<news_item>.summary` where `<news_item>` is the variable referencing the input `NewsItem` at that site. If there's no `NewsItem` in scope at the construction site (e.g., the classifier internally only sees the headline), the easiest plumbing is to pass `summary` alongside `headline` through whichever helper assembles the inputs. Inspect the function signature you find and adapt.

If the classifier currently receives only `(headlines, published_ats, alpaca_ids)`, also accept a `summaries` parameter and thread it through. Update its call site in `v2/pipeline.py:36-39` to pass `summaries = [item.summary for item in news_items]`.

- [ ] **Step 6: Update `v2/pipeline.py` to include summary in the batch tuple**

In `v2/pipeline.py`, find lines 50-55 (the `for signal in result.ticker_signals` loop). The current tuple is:

```python
ticker_signals_batch.append((
    signal.ticker, signal.headline, signal.category,
    signal.sentiment, signal.confidence, signal.published_at,
    signal.alpaca_id,
))
```

Change to:

```python
ticker_signals_batch.append((
    signal.ticker, signal.headline, signal.category,
    signal.sentiment, signal.confidence, signal.published_at,
    signal.alpaca_id, signal.summary,
))
```

- [ ] **Step 7: Update `insert_news_signals_batch` to accept the 8th element**

In `v2/database/trading_db.py`, find `insert_news_signals_batch` (lines 24-44). The current implementation tolerates 6-tuples by padding with `None` for `alpaca_id`. Extend the same tolerance for the new 8th element.

Replace lines 24-44 with:

```python
def insert_news_signals_batch(signals: list[tuple]) -> int:
    """Batch-insert news_signals rows.

    Tuple shape: (ticker, headline, category, sentiment, confidence,
    published_at, alpaca_id?, summary?). Legacy 6/7-tuples are padded
    with None for missing trailing fields so callers updated in
    different commits don't break each other.
    """
    if not signals:
        return 0

    def _normalize(s):
        if len(s) == 6:
            return (*s, None, None)
        if len(s) == 7:
            return (*s, None)
        return s

    normalized = [_normalize(s) for s in signals]

    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO news_signals (ticker, headline, category, sentiment, confidence, published_at, alpaca_id, summary)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, normalized)
        return cur.rowcount if cur.rowcount is not None else 0
```

The `execute_values` import is already at the top of the file (verify with `grep "execute_values" v2/database/trading_db.py`); if missing, add `from psycopg2.extras import execute_values` to the imports.

- [ ] **Step 8: Run the pipeline test to verify it passes**

```
python3 -m pytest tests/v2/test_pipeline.py::TestPipelineSummaryPersistence -v
```

Expected: PASS.

- [ ] **Step 9: Run the broader test suite to catch regressions in callers of `insert_news_signals_batch`**

```
python3 -m pytest tests/ -q -k "pipeline or classifier or news_signals" 2>&1 | tail -30
```

Expected: tests pass (or only fail for unrelated reasons not introduced by this change — flag anything ambiguous). The 6/7-tuple tolerance preserves backward compatibility for any existing test that uses the legacy shape.

- [ ] **Step 10: Commit**

```
git add v2/classifier.py v2/pipeline.py v2/database/trading_db.py tests/v2/test_pipeline.py
git commit -m "$(cat <<'EOF'
feat(pipeline): persist news summary alongside headline

Alpaca's NewsItem.summary was being discarded at the classifier
boundary. This commit plumbs it through TickerSignal and into the
news_signals.summary column. The batch insert tolerates legacy
6/7-tuple shapes for callers not yet updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: get_news_signals returns summary

The DB function `get_news_signals` uses `SELECT *` so the new column comes through automatically; psycopg2's `DictCursor` (per existing pattern) returns it as a key in each row dict. No code change is required, BUT we need a regression test to lock this in.

**Files:**
- Test: `tests/v2/test_news_signals_query.py` (create)

- [ ] **Step 1: Write the test**

Create `tests/v2/test_news_signals_query.py`:

```python
"""Regression test: get_news_signals must return the summary column."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


class TestGetNewsSignalsReturnsSummary:
    @patch("v2.database.trading_db.get_cursor")
    def test_get_news_signals_includes_summary_in_each_row(self, mock_get_cursor):
        from v2.database.trading_db import get_news_signals

        # Mock the cursor context manager + fetchall to return rows that
        # include the new summary key.
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "id": 1,
                "ticker": "AAPL",
                "headline": "AAPL hits ATH",
                "category": "momentum",
                "sentiment": "bullish",
                "confidence": "high",
                "published_at": datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
                "alpaca_id": "alp-123",
                "summary": "Apple closed at $300 after the Foxconn deal.",
                "processed_at": datetime(2026, 5, 9, 12, 5),
            },
        ]
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        rows = get_news_signals(days=7)

        assert len(rows) == 1
        assert "summary" in rows[0], f"summary missing from row keys: {list(rows[0].keys())}"
        assert rows[0]["summary"] == "Apple closed at $300 after the Foxconn deal."
```

- [ ] **Step 2: Run the test**

```
python3 -m pytest tests/v2/test_news_signals_query.py -v
```

Expected: PASS (the test only verifies that when the DB layer returns a `summary` key, the function passes it through — it should pass without any source change since `SELECT *` is the existing query).

- [ ] **Step 3: Commit**

```
git add tests/v2/test_news_signals_query.py
git commit -m "$(cat <<'EOF'
test(db): regression test that get_news_signals returns summary

Locks in the SELECT * contract for the new news_signals.summary
column so a future refactor to explicit column lists doesn't
silently drop it from the strategist's tool output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: News filter module (`v2/news_filter.py`)

**Files:**
- Create: `v2/news_filter.py`
- Test: `tests/v2/test_news_filter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/v2/test_news_filter.py`:

```python
"""Tests for v2/news_filter.py — Haiku relevance filter."""

import json
from unittest.mock import MagicMock, patch

import pytest


def _signal(id_: int, ticker: str = "AAPL", summary: str = "summary text") -> dict:
    """Build a minimal signal dict matching the news_signals row shape."""
    return {
        "id": id_,
        "ticker": ticker,
        "headline": f"headline {id_}",
        "category": "momentum",
        "sentiment": "bullish",
        "summary": summary,
    }


def _mock_haiku_response(json_payload: dict) -> MagicMock:
    """Build a fake anthropic Message with one text block."""
    block = MagicMock()
    block.text = json.dumps(json_payload)
    msg = MagicMock()
    msg.content = [block]
    return msg


class TestCurateSignals:
    @patch("v2.news_filter._call_haiku")
    def test_returns_top_n_ids(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [3, 7, 12]})
        signals = [_signal(i) for i in [3, 5, 7, 9, 12]]

        result = curate_signals(signals, target_n=3, regime_context="risk-on")
        assert result == [3, 7, 12]

    @patch("v2.news_filter._call_haiku")
    def test_drops_hallucinated_ids(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [3, 999]})
        signals = [_signal(i) for i in [3, 5]]

        result = curate_signals(signals, target_n=5, regime_context="x")
        assert result == [3]

    @patch("v2.news_filter._call_haiku")
    def test_falls_back_on_api_error(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.side_effect = RuntimeError("Haiku 500")
        signals = [_signal(i) for i in [1, 2, 3]]

        result = curate_signals(signals, target_n=2, regime_context="x")
        assert result == [1, 2, 3], "API error must degrade to firehose (all input IDs)"

    @patch("v2.news_filter._call_haiku")
    def test_falls_back_on_malformed_json(self, mock_call):
        from v2.news_filter import curate_signals

        block = MagicMock()
        block.text = "not json at all"
        msg = MagicMock()
        msg.content = [block]
        mock_call.return_value = msg

        signals = [_signal(i) for i in [1, 2]]
        result = curate_signals(signals, target_n=1, regime_context="x")
        assert result == [1, 2]

    @patch("v2.news_filter._call_haiku")
    def test_falls_back_on_empty_intersection(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [999, 1000]})
        signals = [_signal(i) for i in [1, 2]]

        result = curate_signals(signals, target_n=1, regime_context="x")
        assert result == [1, 2]

    @patch("v2.news_filter._call_haiku")
    def test_passes_regime_context_to_haiku(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [1]})
        signals = [_signal(1)]

        curate_signals(signals, target_n=1, regime_context="VIX 12, risk-on tape")

        sent_kwargs = mock_call.call_args.kwargs
        sent_messages = sent_kwargs.get("messages") or mock_call.call_args.args[1]
        prompt_text = json.dumps(sent_messages)
        assert "VIX 12, risk-on tape" in prompt_text

    @patch("v2.news_filter._call_haiku")
    def test_input_includes_summary_text(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [1]})
        signals = [_signal(1, summary="Foxconn deal pushes AAPL +5%")]

        curate_signals(signals, target_n=1, regime_context="x")

        sent_kwargs = mock_call.call_args.kwargs
        sent_messages = sent_kwargs.get("messages") or mock_call.call_args.args[1]
        prompt_text = json.dumps(sent_messages)
        assert "Foxconn deal pushes AAPL +5%" in prompt_text

    @patch("v2.news_filter._call_haiku")
    def test_empty_input_returns_empty(self, mock_call):
        from v2.news_filter import curate_signals

        result = curate_signals([], target_n=30, regime_context="x")
        assert result == []
        mock_call.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python3 -m pytest tests/v2/test_news_filter.py -v
```

Expected: every test FAILS with `ImportError: No module named 'v2.news_filter'`.

- [ ] **Step 3: Create `v2/news_filter.py`**

```python
"""Haiku-based relevance filter for the news firehose.

Sits between the local news_signals cache and the strategist (Opus).
Called by tool_get_curated_news. Pure function: takes signals + a
regime context blob, returns the IDs of the top-N most relevant
signals for today.

Degrades gracefully — any error (API failure, malformed JSON,
all-hallucinated IDs) returns all input IDs so the caller serves the
firehose unfiltered. Telemetry is the caller's job.
"""
import json
import logging

import anthropic

from .claude_client import _call_with_retry, get_claude_client

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """You rank financial news signals by relevance for a trading strategist.

You receive:
  - A short regime_context describing today's market backdrop.
  - A list of news signals, each with: id, ticker, category, sentiment, headline, summary.

You return a JSON object: {"top_ids": [<int>, <int>, ...]} listing the IDs of the
top-N most relevant signals for the strategist to consider today, in order of
relevance. Optimize for: news-worthiness (real catalysts vs. noise), regime-fit
(consistent with today's backdrop), and de-duplication (one slot per distinct
story, not multiple variants of the same event).

Return ONLY the JSON object. No prose. No code fences. No commentary."""


def _call_haiku(client, *, messages: list[dict]):
    """Indirection point so tests can patch this without monkeypatching the SDK.

    Returns the anthropic Message object.
    """
    return _call_with_retry(
        client,
        model=HAIKU_MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )


def _build_user_message(signals: list[dict], target_n: int, regime_context: str) -> str:
    lines = [
        f"regime_context: {regime_context}",
        f"target_n: {target_n}",
        "",
        "signals:",
    ]
    for s in signals:
        lines.append(
            f"[#{s['id']}] {s.get('ticker','?')} {s.get('category','?')}/{s.get('sentiment','?')}: "
            f"{s.get('headline','')}\n  summary: {s.get('summary','')}"
        )
    return "\n".join(lines)


def curate_signals(
    signals: list[dict],
    target_n: int,
    regime_context: str,
) -> list[int]:
    """Return IDs of the top-N most relevant signals.

    Falls back to all input IDs on any error. Empty input returns
    empty list without calling the API.
    """
    if not signals:
        return []

    input_ids = {s["id"] for s in signals}
    fallback = [s["id"] for s in signals]

    user_message = _build_user_message(signals, target_n, regime_context)

    try:
        client = get_claude_client()
        response = _call_haiku(
            client,
            messages=[{"role": "user", "content": user_message}],
        )
    except (anthropic.APIError, anthropic.APIConnectionError, RuntimeError) as e:
        logger.warning("Haiku news filter call failed (%s); falling back to firehose", e)
        return fallback

    try:
        text = response.content[0].text.strip()
        # Strip optional ```json fences just in case Haiku ignores instructions.
        if text.startswith("```"):
            parts = text.split("\n", 1)
            text = parts[1] if len(parts) == 2 else parts[0].lstrip("`")
            text = text.rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        raw_ids = parsed.get("top_ids", [])
        if not isinstance(raw_ids, list):
            raise ValueError(f"top_ids is not a list: {raw_ids!r}")
    except (ValueError, KeyError, AttributeError, IndexError) as e:
        logger.warning("Haiku response parse failed (%s); falling back to firehose", e)
        return fallback

    valid_ids = [int(i) for i in raw_ids if isinstance(i, int) and i in input_ids]
    if not valid_ids:
        logger.warning("Haiku returned no valid IDs (input=%d, returned=%d); falling back",
                       len(input_ids), len(raw_ids))
        return fallback

    return valid_ids
```

- [ ] **Step 4: Run the filter tests to verify they pass**

```
python3 -m pytest tests/v2/test_news_filter.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```
git add v2/news_filter.py tests/v2/test_news_filter.py
git commit -m "$(cat <<'EOF'
feat(news_filter): add Haiku-based relevance filter

Pure function curate_signals(signals, target_n, regime_context)
returns the top-N most relevant signal IDs as ranked by Haiku 4.5.
Degrades to firehose (returns all input IDs) on API error,
malformed JSON, or all-hallucinated IDs so the caller is never
left empty-handed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `tool_get_curated_news` + tool registration

**Files:**
- Modify: `v2/tools.py` (add the new tool function near `tool_get_news_signals` at line 285; register in `TOOL_DEFINITIONS` near line 694; register in `TOOL_HANDLERS` near line 832; update `tool_get_news_signals` description in `TOOL_DEFINITIONS`)
- Test: `tests/v2/test_tools.py` (append new test class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_tools.py` (create the file if it doesn't exist with `from unittest.mock import MagicMock, patch` plus `import pytest` at the top):

```python
class TestToolGetCuratedNews:
    """tool_get_curated_news fetches signals from DB, filters via Haiku,
    re-renders in the existing [#id] line format. Falls back to firehose
    on filter failure. Caches per (ticker, days, target_n) within a
    session; reset by reset_session()."""

    def _row(self, id_, ticker="AAPL", summary="full summary"):
        from datetime import datetime, timezone
        return {
            "id": id_,
            "ticker": ticker,
            "headline": f"headline {id_}",
            "category": "momentum",
            "sentiment": "bullish",
            "confidence": "high",
            "published_at": datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
            "summary": summary,
            "alpaca_id": f"alp-{id_}",
        }

    def test_filters_signals_via_haiku(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in [1, 2, 3, 4, 5]]
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "risk-on regime")
        monkeypatch.setattr(
            tools,
            "curate_signals",
            lambda signals, target_n, regime_context: [2, 4],
        )
        tools.reset_session()  # ensure cache is clean

        out = tools.tool_get_curated_news(target_n=2)

        assert "[#2]" in out and "[#4]" in out
        assert "[#1]" not in out
        assert "[#3]" not in out
        assert "[#5]" not in out

    def test_skips_haiku_when_below_target(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in [1, 2, 3]]
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        called = {"count": 0}

        def fake_curate(*args, **kwargs):
            called["count"] += 1
            return [s["id"] for s in args[0]]

        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        out = tools.tool_get_curated_news(target_n=10)

        assert called["count"] == 0, "curate_signals should not be called when candidates <= target_n"
        for i in [1, 2, 3]:
            assert f"[#{i}]" in out

    def test_drops_null_summary_rows(self, monkeypatch):
        from v2 import tools

        rows = [
            self._row(1, summary="full"),
            self._row(2, summary=None),
            self._row(3, summary=""),
            self._row(4, summary="full"),
        ]
        captured = {}

        def fake_curate(signals, target_n, regime_context):
            captured["signal_ids"] = [s["id"] for s in signals]
            return [s["id"] for s in signals]

        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        # Force the filter path by setting target_n below the post-NULL-filter count.
        tools.tool_get_curated_news(target_n=1)

        assert captured["signal_ids"] == [1, 4], (
            f"NULL/empty summaries must be dropped before filter; got {captured['signal_ids']}"
        )

    def test_caches_within_session(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in range(1, 41)]  # 40 rows, above target
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        called = {"count": 0}

        def fake_curate(*args, **kwargs):
            called["count"] += 1
            return [1, 2, 3]

        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        tools.tool_get_curated_news(target_n=3)
        tools.tool_get_curated_news(target_n=3)

        assert called["count"] == 1, "second call with same params must hit cache"

        # Different target_n is a different cache key:
        tools.tool_get_curated_news(target_n=5)
        assert called["count"] == 2, "different target_n must miss cache"

    def test_cache_resets_with_session(self, monkeypatch):
        from v2 import tools

        rows = [self._row(i) for i in range(1, 41)]
        monkeypatch.setattr(tools, "get_news_signals", lambda **kw: rows)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        called = {"count": 0}

        def fake_curate(*args, **kwargs):
            called["count"] += 1
            return [1]

        monkeypatch.setattr(tools, "curate_signals", fake_curate)
        tools.reset_session()

        tools.tool_get_curated_news(target_n=1)
        tools.reset_session()
        tools.tool_get_curated_news(target_n=1)

        assert called["count"] == 2, "reset_session must clear the curated-news cache"

    def test_ticker_specific_passes_ticker_to_db(self, monkeypatch):
        from v2 import tools

        captured_kwargs = {}

        def fake_get_news_signals(**kw):
            captured_kwargs.update(kw)
            return [self._row(1, ticker="AAPL")]

        monkeypatch.setattr(tools, "get_news_signals", fake_get_news_signals)
        monkeypatch.setattr(tools, "get_macro_context", lambda days: "x")
        monkeypatch.setattr(tools, "curate_signals", lambda *a, **kw: [1])
        tools.reset_session()

        tools.tool_get_curated_news(ticker="aapl", days=3)

        # Ticker is normalised upstream; assert the DB was queried for AAPL/3.
        assert captured_kwargs.get("ticker") == "AAPL"
        assert captured_kwargs.get("days") == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python3 -m pytest tests/v2/test_tools.py::TestToolGetCuratedNews -v
```

Expected: all 6 tests FAIL with `AttributeError: module 'v2.tools' has no attribute 'tool_get_curated_news'` (or similar — the function doesn't exist yet).

- [ ] **Step 3: Import `curate_signals` and `get_macro_context` if needed**

`get_macro_context` is already imported at the top of `v2/tools.py:8`. Add the news-filter import below it. The imports section near the top should include:

```python
from .news_filter import curate_signals
```

Add this import line in alphabetical position among the existing `from .` imports (after `from .market_data import format_market_snapshot, get_market_snapshot`).

- [ ] **Step 4: Update `reset_session` to clear the curated-news cache**

In `v2/tools.py`, the existing `reset_session` at line 33 currently only logs. Replace it with:

```python
_curated_news_cache: dict[tuple, list[int]] = {}


def reset_session():
    """Reset session state. Call at start of each ideation run."""
    _curated_news_cache.clear()
    logger.info("Session state reset")
```

Place the `_curated_news_cache` dict immediately above the `reset_session` function so it's clearly the state the function manages.

- [ ] **Step 5: Implement `tool_get_curated_news`**

In `v2/tools.py`, immediately after `tool_get_news_signals` (which ends around line 311), add:

```python
def tool_get_curated_news(
    ticker: str = None,
    days: int = 7,
    target_n: int = 30,
) -> str:
    """Curated news signals — Haiku-filtered to the ~target_n most
    relevant for today's market regime. Use this by default for thesis
    research; use get_news_signals if you need the raw firehose.

    Each line is prefixed with [#<id>] (same format as get_news_signals)
    so signal_refs citation works unchanged.
    """
    ticker = _norm_ticker(ticker)
    cache_key = (ticker, days, target_n)
    cached_ids = _curated_news_cache.get(cache_key)

    logger.info("Getting curated news (ticker=%s, days=%d, target_n=%d)", ticker, days, target_n)
    rows = get_news_signals(ticker=ticker, days=days)

    # Drop rows without usable summaries — pre-backfill safety + empty-summary cleanup.
    candidates = [r for r in rows if r.get("summary")]

    if not candidates:
        if ticker:
            return f"No news signals for {ticker} in the last {days} days."
        return f"No news signals in the last {days} days."

    if cached_ids is None:
        if len(candidates) <= target_n:
            # No filtering needed — return everything.
            selected_ids = [r["id"] for r in candidates]
        else:
            regime_context = get_macro_context(days=2)
            selected_ids = curate_signals(
                candidates,
                target_n=target_n,
                regime_context=regime_context,
            )
        _curated_news_cache[cache_key] = selected_ids
    else:
        selected_ids = cached_ids

    selected_set = set(selected_ids)
    selected_rows = [r for r in candidates if r["id"] in selected_set]
    # Preserve curate_signals' ordering (rank order), not DB order:
    order = {sid: i for i, sid in enumerate(selected_ids)}
    selected_rows.sort(key=lambda r: order.get(r["id"], 1_000_000))

    lines = []
    for s in selected_rows:
        date_str = s["published_at"].strftime("%m-%d %H:%M")
        headline = s["headline"][:60]
        lines.append(
            f"[#{s['id']}] {date_str} {s['ticker']} "
            f"{s['category']}/{s['sentiment']}/{s['confidence']}: {headline}"
        )

    return "\n".join(lines)
```

- [ ] **Step 6: Register the tool in `TOOL_DEFINITIONS` and `TOOL_HANDLERS`**

In `v2/tools.py`, find `TOOL_DEFINITIONS` around line 536. Locate the existing entry for `get_news_signals` (around line 694). Update that entry's description to note its raw-firehose role, AND add a new entry for `get_curated_news` immediately above it:

Replace the existing entry:

```python
{
    "name": "get_news_signals",
    "description": "Recent ticker news signals: headlines, sentiment, category, confidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Filter by ticker"},
            "days": {"type": "integer", "description": "Lookback days (default: 7)"},
        },
        "required": [],
    },
},
```

With:

```python
{
    "name": "get_curated_news",
    "description": (
        "PREFERRED: Curated news signals — Haiku-filtered to the most relevant "
        "for today's market regime. Use this by default. Each line is [#<id>] "
        "(use IDs for signal_refs citation, same as get_news_signals)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Filter by ticker"},
            "days": {"type": "integer", "description": "Lookback days (default: 7)"},
            "target_n": {"type": "integer", "description": "Approx. number of signals to return (default: 30)"},
        },
        "required": [],
    },
},
{
    "name": "get_news_signals",
    "description": (
        "RAW FIREHOSE: Unfiltered recent ticker news signals (could be 900+ items "
        "across 7 days). Use get_curated_news by default; reach for this only when "
        "you need to look past the filter for a specific ticker or theme."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Filter by ticker"},
            "days": {"type": "integer", "description": "Lookback days (default: 7)"},
        },
        "required": [],
    },
},
```

Then find `TOOL_HANDLERS` around line 824. Add `"get_curated_news": tool_get_curated_news,` immediately above the existing `"get_news_signals": tool_get_news_signals,` line.

- [ ] **Step 7: Run the tool tests**

```
python3 -m pytest tests/v2/test_tools.py::TestToolGetCuratedNews -v
```

Expected: all 6 tests PASS.

- [ ] **Step 8: Run the full tools test file to ensure no regressions**

```
python3 -m pytest tests/v2/test_tools.py -v 2>&1 | tail -20
```

Expected: all tests pass. Existing `tool_get_news_signals` tests continue to pass (its behavior is unchanged).

- [ ] **Step 9: Commit**

```
git add v2/tools.py tests/v2/test_tools.py
git commit -m "$(cat <<'EOF'
feat(tools): add tool_get_curated_news using Haiku filter

New strategist tool that fetches news signals, drops NULL-summary
rows, calls v2/news_filter.curate_signals for relevance ranking, and
re-renders the selected signals in the existing [#id] line format.

Caches per (ticker, days, target_n) within a session; reset by
reset_session() at the start of each strategist loop.

Tool descriptions updated so get_curated_news is the explicit
default; get_news_signals stays available as the raw firehose
escape hatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Strategist prompt nudge

**Files:**
- Modify: `v2/ideation_claude.py:60-65` (Tool Usage bullets in `_STRATEGIST_TEMPLATE`)

- [ ] **Step 1: Update the prompt template**

In `v2/ideation_claude.py`, find lines 60-65 — the `## Tool Usage` bullets within `_STRATEGIST_TEMPLATE`. Locate the current `get_news_signals` line:

```
- Use `get_news_signals` to check recent news for specific tickers you're researching (each line is prefixed with `[#id]` — those IDs are what you cite via `signal_refs` on `create_thesis` / `update_thesis`)
```

Replace with:

```
- Use `get_curated_news` by default for thesis research — it returns a Haiku-curated subset of news ranked by relevance to today's market. Use `get_news_signals` (raw firehose) if you need to look past the filter (e.g., specific ticker deep-dive, or you suspect the filter is missing something). Both return `[#id]`-prefixed lines for signal_refs citation.
- News is one input among many — you can also generate theses from market structure (`get_market_snapshot`), attribution patterns (`get_signal_attribution`), macro themes (`get_macro_context`, `get_macro_signals`), or a fresh read of existing positions. Don't anchor thesis generation exclusively on what the news filter surfaces.
```

- [ ] **Step 2: Verify both system prompt variants compile**

The template is used to format two constants (`CLAUDE_STRATEGIST_SYSTEM` and `CLAUDE_SESSION_STRATEGIST_SYSTEM`). Run a quick import check:

```
python3 -c "from v2.ideation_claude import CLAUDE_SESSION_STRATEGIST_SYSTEM; print(len(CLAUDE_SESSION_STRATEGIST_SYSTEM), 'chars')"
```

Expected: prints a number ≥ 5000. If it errors, your edit broke an f-string brace or template placeholder — re-check the replaced lines.

- [ ] **Step 3: Run existing ideation_claude tests to catch breakage**

```
python3 -m pytest tests/v2/test_ideation_claude.py -v 2>&1 | tail -15
```

Expected: tests pass (or fail only for unrelated reasons — flag anything ambiguous).

- [ ] **Step 4: Commit**

```
git add v2/ideation_claude.py
git commit -m "$(cat <<'EOF'
feat(prompt): direct strategist to get_curated_news + nudge non-news ideation

Tool Usage section now points at get_curated_news as the default
for news research, with get_news_signals positioned as the raw
firehose escape hatch. Adds one sentence reminding the strategist
that news is one input among many — market structure, attribution,
macro, and position re-reads are all viable thesis sources, so the
filter doesn't become a creativity bottleneck.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Backfill module (`v2/news_backfill.py`)

**Files:**
- Create: `v2/news_backfill.py`
- Test: `tests/v2/test_news_backfill.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/v2/test_news_backfill.py`:

```python
"""Tests for v2/news_backfill.py — one-shot summary backfill from Alpaca."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


class TestNewsBackfill:
    def _news_item(self, id_: str, summary: str):
        from v2.news import NewsItem
        return NewsItem(
            id=id_,
            headline=f"headline {id_}",
            summary=summary,
            author="x",
            source="Reuters",
            symbols=["AAPL"],
            published_at=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
            url="https://example.com",
        )

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_updates_summary_for_matching_alpaca_id(self, mock_fetch, mock_get_cursor):
        from v2.news_backfill import run

        mock_fetch.return_value = [self._news_item("alp-1", "Foxconn deal pushes AAPL")]
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert mock_cur.execute.called
        executed_sql, executed_params = mock_cur.execute.call_args[0]
        assert "UPDATE news_signals" in executed_sql
        assert "summary IS NULL" in executed_sql or "summary IS NULL OR summary = ''" in executed_sql
        assert "Foxconn deal pushes AAPL" in executed_params
        assert "alp-1" in executed_params
        assert stats["updated"] == 1

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_idempotent_skips_already_populated_rows(self, mock_fetch, mock_get_cursor):
        from v2.news_backfill import run

        mock_fetch.return_value = [self._news_item("alp-1", "new summary")]
        # Simulate the WHERE summary IS NULL guard: no row updated.
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert stats["updated"] == 0
        assert stats["skipped_or_no_match"] == 1

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_skips_items_with_empty_summary(self, mock_fetch, mock_get_cursor):
        """Don't push empty strings into the column."""
        from v2.news_backfill import run

        mock_fetch.return_value = [self._news_item("alp-1", "")]
        mock_cur = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert not mock_cur.execute.called, "must not UPDATE for items with empty summary"
        assert stats["updated"] == 0
        assert stats["skipped_or_no_match"] == 1

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_no_news_fetched_is_clean_exit(self, mock_fetch, mock_get_cursor):
        from v2.news_backfill import run

        mock_fetch.return_value = []
        mock_cur = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert stats == {"fetched": 0, "updated": 0, "skipped_or_no_match": 0}
        mock_cur.execute.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

```
python3 -m pytest tests/v2/test_news_backfill.py -v
```

Expected: all 4 tests FAIL with `ImportError`.

- [ ] **Step 3: Create `v2/news_backfill.py`**

```python
"""One-shot backfill for the news_signals.summary column.

Re-fetches the past N hours of news from Alpaca and runs
UPDATE news_signals SET summary = ... WHERE alpaca_id = ... AND summary IS NULL
for each item whose alpaca_id matches an existing row.

Idempotent — the WHERE summary IS NULL guard means re-running is safe.
Run manually after schema migration + before relying on tool_get_curated_news.
"""
import logging

from .database.connection import get_cursor
from .news import fetch_news

logger = logging.getLogger(__name__)


def run(hours: int = 168) -> dict:
    """Backfill summary on existing news_signals rows.

    Args:
        hours: How far back to fetch from Alpaca. Default 7 days.

    Returns:
        {"fetched": int, "updated": int, "skipped_or_no_match": int}
    """
    logger.info("Starting news backfill (hours=%d)", hours)
    items = fetch_news(hours=hours, symbols=None, limit=10000)
    stats = {"fetched": len(items), "updated": 0, "skipped_or_no_match": 0}

    if not items:
        logger.info("No news items fetched; nothing to backfill")
        return stats

    for item in items:
        if not item.summary:
            stats["skipped_or_no_match"] += 1
            continue

        with get_cursor() as cur:
            cur.execute(
                """
                UPDATE news_signals
                   SET summary = %s
                 WHERE alpaca_id = %s
                   AND (summary IS NULL OR summary = '')
                """,
                (item.summary, item.id),
            )
            if cur.rowcount and cur.rowcount > 0:
                stats["updated"] += cur.rowcount
            else:
                stats["skipped_or_no_match"] += 1

    logger.info(
        "Backfill complete: fetched=%d updated=%d skipped/no-match=%d",
        stats["fetched"], stats["updated"], stats["skipped_or_no_match"],
    )
    return stats


def main():
    """CLI entry point."""
    import argparse

    from .log_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Backfill news_signals.summary from Alpaca")
    parser.add_argument("--hours", type=int, default=168, help="Hours to backfill (default 168 = 7 days)")
    args = parser.parse_args()

    stats = run(hours=args.hours)
    print(f"fetched={stats['fetched']} updated={stats['updated']} skipped={stats['skipped_or_no_match']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the backfill tests**

```
python3 -m pytest tests/v2/test_news_backfill.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```
git add v2/news_backfill.py tests/v2/test_news_backfill.py
git commit -m "$(cat <<'EOF'
feat(backfill): one-shot news_signals.summary backfill

Re-fetches past 7 days from Alpaca and UPDATEs the new summary
column on rows where it's currently NULL/empty, matched by
alpaca_id. Idempotent (WHERE summary IS NULL guard). Standalone
CLI: python -m v2.news_backfill.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Manual backfill execution + smoke test (user action)

This is a one-shot operational step, not code. Run after Tasks 1–7 have all landed.

**Files:**
- No code changes. Adds a "Validation results" section to the spec on completion.

- [ ] **Step 1: Verify both DBs have the summary column**

```
docker exec algo-db-paper-1 psql -U algo -d trading -c "\d news_signals" | grep summary
docker exec algo-db-1       psql -U algo -d trading -c "\d news_signals" | grep summary
```

Expected: both show `summary | text`. If missing, re-run Task 1 Step 2.

- [ ] **Step 2: Run backfill on paper**

```
docker exec algo-trading-paper-1 python -m v2.news_backfill --hours 168
```

Expected: prints `fetched=N updated=M skipped=K`. M should be in the hundreds (close to but possibly less than the 936 rows currently in `news_signals`, since Alpaca's 7-day window may differ slightly from the local table's content).

- [ ] **Step 3: Verify backfill populated rows**

```
docker exec algo-db-paper-1 psql -U algo -d trading -c "SELECT COUNT(*) AS total, COUNT(summary) AS with_summary, COUNT(*) FILTER (WHERE summary IS NULL) AS missing FROM news_signals WHERE published_at > NOW() - INTERVAL '7 days';"
```

Expected: `with_summary` count > 0 and substantially less than `total` is fine (some rows may have no summary on the Alpaca side). If `with_summary` is 0, debug — likely an alpaca_id mismatch.

- [ ] **Step 4: Run backfill on prod (optional but recommended)**

```
docker exec algo-trading-1 python -m v2.news_backfill --hours 168
```

Same verification as Step 3 against `algo-db-1`.

- [ ] **Step 5: Smoke test — run a paper strategist session**

Clear any stale session for today's date first:

```
docker exec algo-db-paper-1 psql -U algo -d trading -c "DELETE FROM session_stages WHERE session_id IN (SELECT id FROM sessions WHERE session_date=CURRENT_DATE); DELETE FROM sessions WHERE session_date=CURRENT_DATE;"
```

Then:

```
task paper:session
```

Expected: session completes. In the strategist log (visible during the run), confirm:
- At least one `Executing tool: get_curated_news` line
- The strategist still uses other tools (`get_market_snapshot`, etc.) — not just news

- [ ] **Step 6: Compare outputs**

Note the new session's id (top of `SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1` on `algo-db-paper-1`). Then:

```
docker exec algo-db-paper-1 psql -U algo -d trading -c "SELECT payload->>'tool_name' AS tool, COUNT(*) AS calls, AVG((payload->>'output_chars')::int)::int AS avg_chars FROM agent_events WHERE event_type='tool_invocation' AND session_id=<NEW_SESSION_ID> AND stage_name='ideation' GROUP BY 1 ORDER BY 1;"
```

Compare against paper session 241 (pre-filter baseline, recorded in the abandoned cache-fix spec history if you want exact numbers, otherwise compare ad hoc):

- `get_curated_news` should appear with `avg_chars` ≪ 100K (expect ~5–8K).
- `get_news_signals` should appear 0 or 1 times (vs 2 on session 241). If it's >2, the strategist is reaching past the filter often — may indicate the curated output is too thin (raise `target_n` next iteration).

- [ ] **Step 7: Eyeball the post-fix playbook**

```
docker exec algo-db-paper-1 psql -U algo -d trading -c "SELECT id, date, market_outlook FROM playbooks WHERE date = CURRENT_DATE ORDER BY id DESC LIMIT 1;"
docker exec algo-db-paper-1 psql -U algo -d trading -c "SELECT id, ticker, action, reasoning FROM playbook_actions WHERE playbook_id = (SELECT id FROM playbooks WHERE date = CURRENT_DATE ORDER BY id DESC LIMIT 1) ORDER BY id;"
```

Compare against the pre-filter sample (playbook 12 from session 241 — 5 actions, avg reasoning 361 chars, 2 new theses created):

- Action count: ≥ 4 (some variability OK; <3 sustained over multiple sessions = regression)
- Reasoning length: ≥ 250 chars avg (similar caveat)
- Spot-check: does the rationale read coherently? Do action choices reference specific news/signal IDs?

- [ ] **Step 8: Record results in the spec**

Append a "Validation results" section at the bottom of `docs/superpowers/specs/2026-05-10-news-filter-design.md`:

```markdown
## Validation results

- Post-merge paper session: <SESSION_ID> on <DATE>
- get_curated_news avg output: <CHARS> chars (vs pre-filter firehose: ~100K)
- get_news_signals calls in session: <COUNT> (vs pre-filter baseline: 2)
- Playbook actions: <N> (vs baseline: 5)
- Avg reasoning chars per action: <CHARS> (vs baseline: 361)
- Theses created: <N> (vs baseline: 2)
- Subjective playbook quality: <pass | concerns: ...>
- Strategist cost: $<COST> (vs pre-filter average $5.18)
```

Fill in actual numbers from Steps 5–7.

- [ ] **Step 9: Commit the validation results**

```
git add docs/superpowers/specs/2026-05-10-news-filter-design.md
git commit -m "$(cat <<'EOF'
docs(specs): record post-merge validation for news filter

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review checklist (post-implementation)

After Task 8 completes, run through this list:

- All tasks above show green checkboxes.
- `python3 -m pytest tests/v2/ -q` passes (or only fails on unrelated pre-existing issues — flag, don't fix in this branch).
- `grep -rn "tool_get_curated_news\|curate_signals" v2/ tests/` shows expected wiring (tool function, registration in TOOL_HANDLERS and TOOL_DEFINITIONS, tests, filter module).
- `docker exec algo-db-paper-1 psql -U algo -d trading -c "SELECT COUNT(*) FROM news_signals WHERE summary IS NOT NULL"` returns hundreds.
- Post-merge paper session shows `get_curated_news` in use and playbook quality is at or above baseline.
- Validation results recorded in the spec.

If any item fails, fix before merging the branch.

---

## Out of scope (do NOT do as part of this plan)

- Filtering `get_macro_signals` or any other tool. Only news is firehose-shaped.
- A configuration UI for `target_n`. Default 30 is hardcoded.
- Trimming the strategist's pre-seeded context (a separate optimization).
- Sonnet swap or A/B harness.
- Storing full article bodies beyond Alpaca's `summary` field.
- Re-classifying sentiment/category via Haiku. Existing pipeline classification stays.
- Caching Haiku output across sessions. In-session cache is sufficient and the daily news set changes.
