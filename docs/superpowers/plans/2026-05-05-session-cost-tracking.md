# Session Cost Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist Claude API token usage per session stage so daily session cost is queryable after the fact, and replace the buggy `_print_cost_summary` with a correct, reusable helper.

**Architecture:** Add five token columns to `session_stages`. Store raw API counters; compute USD cost on read via SQL view backed by a `model_pricing` lookup table. Capture usage at the `_call_with_retry` boundary using a `contextvars`-bound accumulator opened by `session.py` per stage — stage code is unchanged.

**Tech Stack:** PostgreSQL 16, Python 3, `anthropic` SDK, `psycopg2`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-05-session-cost-tracking-design.md`

---

## File map

**Create:**
- `db/init/024_session_stage_token_usage.sql` — migration: alter `session_stages`, create `model_pricing`, create views
- `v2/pricing.py` — DB-backed cost lookup helper
- `tests/test_pricing.py` — unit tests for pricing helper
- `tests/test_usage_accumulator.py` — unit tests for accumulator
- `tests/test_capture_usage.py` — unit tests for context manager
- `tests/test_claude_client_usage_capture.py` — verify `_call_with_retry` records usage
- `tests/test_pricing_coverage.py` — meta-test: every hardcoded `claude-*` model literal in `v2/` has a `model_pricing` row

**Modify:**
- `v2/claude_client.py` — add `UsageAccumulator`, `capture_usage`, `_record_usage`; call `_record_usage` from `_call_with_retry`
- `v2/database/trading_db.py` — extend `complete_session_stage` and `fail_session_stage` to accept usage and write the five columns
- `v2/session.py` — wrap each `_run_*_stage` body in `capture_usage()`; pass usage to complete/fail; add end-of-session cost log line
- `v2/ideation_claude.py` — delete `_print_cost_summary`, replace with a call into `v2/pricing.py`
- `v2/BUGS.md` — mark line 354 fixed

---

## Task 1: Migration — schema + price seed + views

**Files:**
- Create: `db/init/024_session_stage_token_usage.sql`

- [ ] **Step 1: Write the migration**

Create `db/init/024_session_stage_token_usage.sql`:

```sql
-- Per-stage Claude API token usage + USD cost views.
-- See docs/superpowers/specs/2026-05-05-session-cost-tracking-design.md

ALTER TABLE session_stages
    ADD COLUMN model VARCHAR(64),
    ADD COLUMN input_tokens INT,
    ADD COLUMN output_tokens INT,
    ADD COLUMN cache_creation_tokens INT,
    ADD COLUMN cache_read_tokens INT;

CREATE TABLE model_pricing (
    model VARCHAR(64) PRIMARY KEY,
    input_per_mtok NUMERIC(10,4) NOT NULL,
    output_per_mtok NUMERIC(10,4) NOT NULL,
    cache_creation_per_mtok NUMERIC(10,4) NOT NULL,
    cache_read_per_mtok NUMERIC(10,4) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed both bare aliases and dated pins. Callsites pass the dated pin
-- (e.g. classifier uses "claude-haiku-4-5-20251001"); ideation/session
-- pass the bare alias. Both must resolve.
INSERT INTO model_pricing (model, input_per_mtok, output_per_mtok, cache_creation_per_mtok, cache_read_per_mtok) VALUES
    ('claude-opus-4-7',              15.00, 75.00, 18.75, 1.50),
    ('claude-opus-4-6',              15.00, 75.00, 18.75, 1.50),
    ('claude-sonnet-4-6',             3.00, 15.00,  3.75, 0.30),
    ('claude-haiku-4-5',              1.00,  5.00,  1.25, 0.10),
    ('claude-haiku-4-5-20251001',     1.00,  5.00,  1.25, 0.10);

CREATE VIEW session_stage_costs AS
SELECT
    ss.*,
    CASE
        WHEN ss.model IS NULL OR mp.model IS NULL THEN NULL
        ELSE (
            COALESCE(ss.input_tokens, 0)          * mp.input_per_mtok +
            COALESCE(ss.output_tokens, 0)         * mp.output_per_mtok +
            COALESCE(ss.cache_creation_tokens, 0) * mp.cache_creation_per_mtok +
            COALESCE(ss.cache_read_tokens, 0)     * mp.cache_read_per_mtok
        ) / 1000000.0
    END AS cost_usd
FROM session_stages ss
LEFT JOIN model_pricing mp ON mp.model = ss.model;

CREATE VIEW session_costs AS
SELECT
    s.id AS session_id, s.session_date, s.session_type, s.status,
    SUM(sc.cost_usd)              AS total_cost_usd,
    SUM(sc.input_tokens)          AS total_input_tokens,
    SUM(sc.output_tokens)         AS total_output_tokens,
    SUM(sc.cache_creation_tokens) AS total_cache_creation_tokens,
    SUM(sc.cache_read_tokens)     AS total_cache_read_tokens
FROM sessions s
LEFT JOIN session_stage_costs sc ON sc.session_id = s.id
GROUP BY s.id, s.session_date, s.session_type, s.status;
```

- [ ] **Step 2: Apply to the running paper DB and verify**

The paper stack is the only currently-running container set. Apply there to verify SQL parses and runs:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    cp db/init/024_session_stage_token_usage.sql \
    db-paper:/tmp/024.sql

docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/024.sql
```

Expected: no errors. Then verify:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -c "\d session_stages" \
    -c "SELECT * FROM model_pricing;" \
    -c "SELECT * FROM session_costs LIMIT 1;"
```

Expected: five new columns shown on `session_stages`; five seeded rows in `model_pricing`; `session_costs` view returns rows (with NULL costs since no token data yet).

- [ ] **Step 3: Commit**

```bash
git add db/init/024_session_stage_token_usage.sql
git commit -m "feat(db): migration 024 — session_stages token usage + cost views"
```

---

## Task 2: `v2/pricing.py` helper

**Files:**
- Create: `v2/pricing.py`
- Test: `tests/test_pricing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pricing.py`:

```python
"""Unit tests for v2/pricing.py — DB-backed cost lookup helper."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from v2.pricing import UnknownModelError, stage_cost_usd


@contextmanager
def _mock_cursor(rows_by_query):
    """Patch v2.database.trading_db.get_cursor; rows_by_query maps
    fetchone() return values keyed by call order."""
    cursor = MagicMock()
    cursor.fetchone.side_effect = rows_by_query
    @contextmanager
    def _gc():
        yield cursor
    with patch("v2.pricing.get_cursor", _gc):
        yield cursor


def test_haiku_cost_matches_hand_computation():
    rates = {
        "input_per_mtok": 1.00,
        "output_per_mtok": 5.00,
        "cache_creation_per_mtok": 1.25,
        "cache_read_per_mtok": 0.10,
    }
    with _mock_cursor([rates]):
        cost = stage_cost_usd(
            model="claude-haiku-4-5-20251001",
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_creation_tokens=200_000,
            cache_read_tokens=800_000,
        )
    # 1.00 + 2.50 + 0.25 + 0.08 = 3.83
    assert cost == pytest.approx(3.83)


def test_opus_cost_matches_hand_computation():
    rates = {
        "input_per_mtok": 15.00,
        "output_per_mtok": 75.00,
        "cache_creation_per_mtok": 18.75,
        "cache_read_per_mtok": 1.50,
    }
    with _mock_cursor([rates]):
        cost = stage_cost_usd(
            model="claude-opus-4-7",
            input_tokens=10_000,
            output_tokens=5_000,
            cache_creation_tokens=100_000,
            cache_read_tokens=400_000,
        )
    # 0.00015 + 0.000375 + 0.001875 + 0.00060 = 0.0030
    assert cost == pytest.approx(0.003)


def test_zero_tokens_returns_zero():
    rates = {
        "input_per_mtok": 1.00,
        "output_per_mtok": 5.00,
        "cache_creation_per_mtok": 1.25,
        "cache_read_per_mtok": 0.10,
    }
    with _mock_cursor([rates]):
        cost = stage_cost_usd("claude-haiku-4-5", 0, 0, 0, 0)
    assert cost == 0.0


def test_unknown_model_raises():
    with _mock_cursor([None]):
        with pytest.raises(UnknownModelError):
            stage_cost_usd("claude-future-99", 1000, 1000, 0, 0)
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_pricing.py -v
```

Expected: ImportError or ModuleNotFoundError for `v2.pricing` — module does not exist yet.

- [ ] **Step 3: Implement `v2/pricing.py`**

Create `v2/pricing.py`:

```python
"""USD cost lookup for Claude API token usage.

Reads rates from the model_pricing table. Source of truth for prices
lives in the DB (db/init/024_session_stage_token_usage.sql), not in
code, so a price update is a one-row SQL change rather than a redeploy.
"""

from v2.database.trading_db import get_cursor


class UnknownModelError(KeyError):
    """Raised when a model is not present in model_pricing."""


def stage_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> float:
    """Return USD cost for a given (model, token counts) tuple.

    Mirrors the cost formula in the session_stage_costs SQL view so
    Python-side numbers match what the DB reports.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT input_per_mtok, output_per_mtok,
                   cache_creation_per_mtok, cache_read_per_mtok
            FROM model_pricing WHERE model = %s
            """,
            (model,),
        )
        row = cur.fetchone()
    if row is None:
        raise UnknownModelError(model)
    return (
        input_tokens          * float(row["input_per_mtok"])
        + output_tokens         * float(row["output_per_mtok"])
        + cache_creation_tokens * float(row["cache_creation_per_mtok"])
        + cache_read_tokens     * float(row["cache_read_per_mtok"])
    ) / 1_000_000.0
```

- [ ] **Step 4: Run the test — verify it passes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_pricing.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/pricing.py tests/test_pricing.py
git commit -m "feat(v2): pricing.stage_cost_usd helper backed by model_pricing table"
```

---

## Task 3: `UsageAccumulator` dataclass

**Files:**
- Modify: `v2/claude_client.py` (add to existing module, near top with other dataclasses around line 37)
- Test: `tests/test_usage_accumulator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_usage_accumulator.py`:

```python
"""Unit tests for UsageAccumulator — sums token counts across calls."""

from types import SimpleNamespace

from v2.claude_client import UsageAccumulator


def _usage(input_t=0, output_t=0, cache_create=0, cache_read=0):
    return SimpleNamespace(
        input_tokens=input_t,
        output_tokens=output_t,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    )


def test_initial_state_is_zero():
    acc = UsageAccumulator()
    assert acc.model is None
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.cache_creation_tokens == 0
    assert acc.cache_read_tokens == 0
    assert acc.mixed_models is False


def test_add_records_first_model():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=100, output_t=50))
    assert acc.model == "claude-haiku-4-5"
    assert acc.input_tokens == 100
    assert acc.output_tokens == 50
    assert acc.mixed_models is False


def test_add_sums_across_calls_same_model():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=100, output_t=50, cache_create=10, cache_read=20))
    acc.add("claude-haiku-4-5", _usage(input_t=200, output_t=80, cache_create=5,  cache_read=15))
    assert acc.input_tokens == 300
    assert acc.output_tokens == 130
    assert acc.cache_creation_tokens == 15
    assert acc.cache_read_tokens == 35
    assert acc.mixed_models is False


def test_add_flips_mixed_models_on_second_model():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=100))
    acc.add("claude-opus-4-7",  _usage(input_t=200))
    assert acc.mixed_models is True
    # First model is preserved as the recorded model
    assert acc.model == "claude-haiku-4-5"
    assert acc.input_tokens == 300


def test_add_handles_missing_cache_attrs():
    """Older API responses may omit cache_creation_input_tokens / cache_read_input_tokens."""
    acc = UsageAccumulator()
    bare = SimpleNamespace(input_tokens=100, output_tokens=50)
    acc.add("claude-haiku-4-5", bare)
    assert acc.cache_creation_tokens == 0
    assert acc.cache_read_tokens == 0


def test_add_handles_none_token_values():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=None, output_t=None))
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_usage_accumulator.py -v
```

Expected: ImportError — `UsageAccumulator` not defined.

- [ ] **Step 3: Implement `UsageAccumulator`**

Add to `v2/claude_client.py`, just below the existing `AgenticLoopResult` dataclass (around line 47):

```python
@dataclass
class UsageAccumulator:
    """Sums Claude API token usage across calls within a stage.

    Populated by `_record_usage` (called from `_call_with_retry`) when
    a `capture_usage()` block is active; consumed by session.py to
    write per-stage token counts to the database.
    """

    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    mixed_models: bool = False

    def add(self, model: str, usage) -> None:
        if self.model is None:
            self.model = model
        elif self.model != model:
            self.mixed_models = True
        self.input_tokens          += (usage.input_tokens or 0)
        self.output_tokens         += (usage.output_tokens or 0)
        self.cache_creation_tokens += (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.cache_read_tokens     += (getattr(usage, "cache_read_input_tokens", 0) or 0)
```

- [ ] **Step 4: Run the test — verify it passes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_usage_accumulator.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/claude_client.py tests/test_usage_accumulator.py
git commit -m "feat(v2): UsageAccumulator dataclass for per-stage token tracking"
```

---

## Task 4: `capture_usage` context manager + `_record_usage`

**Files:**
- Modify: `v2/claude_client.py`
- Test: `tests/test_capture_usage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_capture_usage.py`:

```python
"""Tests for capture_usage context manager + _record_usage helper."""

from types import SimpleNamespace

from v2.claude_client import _record_usage, capture_usage


def _usage(input_t=0, output_t=0):
    return SimpleNamespace(
        input_tokens=input_t,
        output_tokens=output_t,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def test_record_usage_outside_block_is_noop():
    """Calls without an active capture_usage() block do nothing — no error,
    no global state mutation."""
    _record_usage("claude-haiku-4-5", _usage(input_t=100))
    # Re-entering a capture block must start fresh
    with capture_usage() as acc:
        assert acc.input_tokens == 0


def test_capture_usage_collects_within_block():
    with capture_usage() as acc:
        _record_usage("claude-haiku-4-5", _usage(input_t=100, output_t=50))
        _record_usage("claude-haiku-4-5", _usage(input_t=200, output_t=80))
    assert acc.input_tokens == 300
    assert acc.output_tokens == 130
    assert acc.model == "claude-haiku-4-5"


def test_sequential_blocks_are_isolated():
    with capture_usage() as a:
        _record_usage("claude-haiku-4-5", _usage(input_t=100))
    with capture_usage() as b:
        _record_usage("claude-opus-4-7", _usage(input_t=200))
    assert a.input_tokens == 100
    assert a.model == "claude-haiku-4-5"
    assert b.input_tokens == 200
    assert b.model == "claude-opus-4-7"


def test_nested_blocks_inner_collects_only_inner_calls():
    """The contextvars-based scoping means inner block sees only its own
    calls; outer block's own calls (before and after) accumulate to outer."""
    with capture_usage() as outer:
        _record_usage("claude-opus-4-7", _usage(input_t=10))
        with capture_usage() as inner:
            _record_usage("claude-haiku-4-5", _usage(input_t=100))
        _record_usage("claude-opus-4-7", _usage(input_t=20))
    assert inner.input_tokens == 100
    assert inner.model == "claude-haiku-4-5"
    assert outer.input_tokens == 30
    assert outer.model == "claude-opus-4-7"


def test_block_resets_after_exception():
    try:
        with capture_usage() as _:
            _record_usage("claude-haiku-4-5", _usage(input_t=100))
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # New block sees fresh state
    with capture_usage() as fresh:
        assert fresh.input_tokens == 0
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_capture_usage.py -v
```

Expected: ImportError — `capture_usage` / `_record_usage` not defined.

- [ ] **Step 3: Implement `capture_usage` and `_record_usage`**

Add to `v2/claude_client.py` (top of file, alongside other imports — `contextlib` and `contextvars`):

```python
import contextlib
import contextvars
```

Then add below `UsageAccumulator` (immediately after the dataclass from Task 3):

```python
_current_usage: contextvars.ContextVar[UsageAccumulator | None] = contextvars.ContextVar(
    "_current_usage", default=None
)


@contextlib.contextmanager
def capture_usage():
    """Open an accumulator that records all Claude calls until the block exits.

    Stage code is unaffected — instrumentation lives at `_call_with_retry`.
    Sessions wrap each stage in this block to collect per-stage usage.
    """
    acc = UsageAccumulator()
    token = _current_usage.set(acc)
    try:
        yield acc
    finally:
        _current_usage.reset(token)


def _record_usage(model: str, usage) -> None:
    """Record one API call's usage into the active accumulator (if any).

    No-op when called outside a capture_usage() block — the function is
    safe to call from any code path; production callers don't need to
    know whether tracking is on.
    """
    acc = _current_usage.get()
    if acc is not None:
        acc.add(model, usage)
```

- [ ] **Step 4: Run the test — verify it passes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_capture_usage.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/claude_client.py tests/test_capture_usage.py
git commit -m "feat(v2): capture_usage context manager + _record_usage helper"
```

---

## Task 5: Wire `_record_usage` into `_call_with_retry`

**Files:**
- Modify: `v2/claude_client.py:63-97` (the `_call_with_retry` function)
- Test: `tests/test_claude_client_usage_capture.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_claude_client_usage_capture.py`:

```python
"""Verify that every Claude call routed through _call_with_retry feeds
its usage into the active capture_usage() block."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from v2.claude_client import _call_with_retry, capture_usage


def _make_mock_client(usage):
    """Build a stand-in for the anthropic.Anthropic client whose
    messages.stream() returns a context manager whose final message
    carries the given usage."""
    final_message = SimpleNamespace(
        usage=usage,
        content=[],
        stop_reason="end_turn",
    )
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=SimpleNamespace(
        get_final_message=MagicMock(return_value=final_message)
    ))
    stream_cm.__exit__ = MagicMock(return_value=False)
    client = MagicMock()
    client.messages.stream.return_value = stream_cm
    return client


def test_call_with_retry_records_usage_in_active_block():
    client = _make_mock_client(SimpleNamespace(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=10, cache_read_input_tokens=20,
    ))
    with capture_usage() as acc:
        _call_with_retry(client, model="claude-haiku-4-5-20251001",
                         max_tokens=100, messages=[])
    assert acc.model == "claude-haiku-4-5-20251001"
    assert acc.input_tokens == 100
    assert acc.output_tokens == 50
    assert acc.cache_creation_tokens == 10
    assert acc.cache_read_tokens == 20


def test_call_with_retry_does_nothing_outside_capture_block():
    """Production callers that don't open a capture block must not break."""
    client = _make_mock_client(SimpleNamespace(
        input_tokens=1, output_tokens=1,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    ))
    # Should not raise
    result = _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
    assert result.usage.input_tokens == 1


def test_multiple_calls_in_one_block_sum():
    client = _make_mock_client(SimpleNamespace(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    ))
    with capture_usage() as acc:
        _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
        _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
        _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
    assert acc.input_tokens == 300
    assert acc.output_tokens == 150


def test_agentic_loop_does_not_double_count_simulated():
    """If run_agentic_loop is ever modified to also call _record_usage with its
    AgenticLoopResult totals, every token would be counted twice. This
    test asserts that the only path producing recorded usage is
    _call_with_retry."""
    # Simulate three "turns" of an agentic loop — three _call_with_retry
    # invocations. Expectation: exactly summed tokens, never doubled.
    client = _make_mock_client(SimpleNamespace(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    ))
    with capture_usage() as acc:
        for _ in range(3):
            _call_with_retry(client, model="claude-opus-4-7", max_tokens=10, messages=[])
    # Three calls × 100 input each = 300 total. If double-counting, would be 600.
    assert acc.input_tokens == 300
    assert acc.output_tokens == 150
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_claude_client_usage_capture.py -v
```

Expected: assertions fail — `acc.input_tokens == 0` because `_call_with_retry` does not yet call `_record_usage`.

- [ ] **Step 3: Modify `_call_with_retry`**

In `v2/claude_client.py`, edit the `_call_with_retry` function (currently at lines 63-97). Replace:

```python
    for attempt in range(max_retries + 1):
        try:
            with client.messages.stream(**create_kwargs) as stream:
                return stream.get_final_message()
```

with:

```python
    for attempt in range(max_retries + 1):
        try:
            with client.messages.stream(**create_kwargs) as stream:
                message = stream.get_final_message()
            _record_usage(create_kwargs["model"], message.usage)
            return message
```

Note the dedent: `_record_usage` is called *after* the `with` block exits to keep the stream cleanup tight, but still inside the `try`. Returning `message` afterwards preserves caller behavior.

- [ ] **Step 4: Run the test — verify it passes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_claude_client_usage_capture.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full claude_client test file to make sure nothing regressed**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_claude_client.py -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add v2/claude_client.py tests/test_claude_client_usage_capture.py
git commit -m "feat(v2): record API usage on every _call_with_retry"
```

---

## Task 6: Extend `complete_session_stage` / `fail_session_stage` to write usage

**Files:**
- Modify: `v2/database/trading_db.py:866-895` (the three session_stages helpers)
- Modify: `tests/test_db.py` (add usage-write tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_db.py` (or create a new section):

```python
def test_complete_session_stage_writes_usage_columns(mock_db):
    """complete_session_stage should write the five token columns when
    a usage object is supplied."""
    from v2.claude_client import UsageAccumulator
    from v2.database.trading_db import complete_session_stage

    usage = UsageAccumulator(
        model="claude-haiku-4-5-20251001",
        input_tokens=1234,
        output_tokens=567,
        cache_creation_tokens=89,
        cache_read_tokens=10,
    )
    complete_session_stage(session_id=42, stage_name="executor", usage=usage)

    sql_calls = [c.args[0] for c in mock_db.execute.call_args_list]
    last = sql_calls[-1]
    assert "model" in last
    assert "input_tokens" in last
    assert "output_tokens" in last
    assert "cache_creation_tokens" in last
    assert "cache_read_tokens" in last

    params = mock_db.execute.call_args_list[-1].args[1]
    assert "claude-haiku-4-5-20251001" in params
    assert 1234 in params
    assert 567 in params


def test_complete_session_stage_without_usage_omits_columns(mock_db):
    """When usage is None or has no model (no Claude calls fired), the
    five new columns must not appear in the UPDATE — preserves existing
    behavior for stages like learning/dashboard."""
    from v2.database.trading_db import complete_session_stage

    complete_session_stage(session_id=42, stage_name="dashboard", usage=None)
    last = mock_db.execute.call_args_list[-1].args[0]
    assert "model" not in last
    assert "input_tokens" not in last


def test_fail_session_stage_writes_partial_usage(mock_db):
    """A stage that fails after firing some calls should still record the
    partial token usage."""
    from v2.claude_client import UsageAccumulator
    from v2.database.trading_db import fail_session_stage

    partial = UsageAccumulator(
        model="claude-opus-4-7",
        input_tokens=500,
        output_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )
    fail_session_stage(session_id=42, stage_name="strategist", error_text="boom", usage=partial)

    last = mock_db.execute.call_args_list[-1].args[0]
    assert "model" in last
    assert "input_tokens" in last
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_db.py -v -k "session_stage_writes_usage or session_stage_without_usage or partial_usage"
```

Expected: TypeError — `complete_session_stage`/`fail_session_stage` do not accept `usage` keyword.

- [ ] **Step 3: Modify `v2/database/trading_db.py`**

Locate `complete_session_stage` (around line 877) and `fail_session_stage` (around line 884). Replace both with:

```python
def insert_session_stage(session_id, stage_name) -> None:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO session_stages (session_id, stage_name, status)
            VALUES (%s, %s, 'running')
            ON CONFLICT (session_id, stage_name) DO NOTHING
        """, (session_id, stage_name))


def complete_session_stage(session_id, stage_name, usage=None) -> None:
    """Mark a session_stages row complete, optionally recording token usage.

    `usage` is a v2.claude_client.UsageAccumulator. When supplied with a
    non-None .model, the five token columns are written. When None or
    .model is None (stage made no Claude calls), the columns stay NULL.
    """
    with get_cursor() as cur:
        if usage is not None and usage.model is not None:
            cur.execute("""
                UPDATE session_stages
                SET status = 'completed',
                    completed_at = NOW(),
                    model = %s,
                    input_tokens = %s,
                    output_tokens = %s,
                    cache_creation_tokens = %s,
                    cache_read_tokens = %s
                WHERE session_id = %s AND stage_name = %s
            """, (
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_tokens,
                usage.cache_read_tokens,
                session_id, stage_name,
            ))
        else:
            cur.execute("""
                UPDATE session_stages SET status = 'completed', completed_at = NOW()
                WHERE session_id = %s AND stage_name = %s
            """, (session_id, stage_name))


def fail_session_stage(session_id, stage_name, error_text, usage=None) -> None:
    """Mark a session_stages row failed; record any partial token usage."""
    with get_cursor() as cur:
        if usage is not None and usage.model is not None:
            cur.execute("""
                UPDATE session_stages
                SET status = 'failed',
                    completed_at = NOW(),
                    error = %s,
                    model = %s,
                    input_tokens = %s,
                    output_tokens = %s,
                    cache_creation_tokens = %s,
                    cache_read_tokens = %s
                WHERE session_id = %s AND stage_name = %s
            """, (
                error_text,
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_tokens,
                usage.cache_read_tokens,
                session_id, stage_name,
            ))
        else:
            cur.execute("""
                UPDATE session_stages
                SET status = 'failed', completed_at = NOW(), error = %s
                WHERE session_id = %s AND stage_name = %s
            """, (error_text, session_id, stage_name))
```

(Keep `insert_session_stage` and `get_completed_stages` untouched; the snippet above includes `insert_session_stage` for context — if it already differs in the file, leave it as-is.)

- [ ] **Step 4: Run the test — verify it passes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_db.py -v -k "session_stage"
```

Expected: all session_stage tests pass.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/test_db.py
git commit -m "feat(v2): persist token usage in complete/fail_session_stage"
```

---

## Task 7: Wrap each stage in `session.py` with `capture_usage()`

**Files:**
- Modify: `v2/session.py` (the `_complete_stage`, `_fail_stage` helpers and every `_run_*_stage` function)
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session.py`:

```python
def test_run_pipeline_stage_passes_usage_to_complete(mock_db, monkeypatch):
    """When the pipeline stage runs, the captured token usage should be
    forwarded to complete_session_stage."""
    from v2 import session
    from v2.claude_client import _record_usage

    captured = {}

    def fake_complete(session_id, stage_name, usage=None):
        captured["stage"] = stage_name
        captured["model"] = usage.model if usage else None
        captured["input_tokens"] = usage.input_tokens if usage else 0

    monkeypatch.setattr(session, "complete_session_stage", fake_complete)
    monkeypatch.setattr(session, "insert_session_stage", lambda *a, **kw: None)

    def fake_pipeline(*args, **kwargs):
        from types import SimpleNamespace
        _record_usage("claude-haiku-4-5-20251001", SimpleNamespace(
            input_tokens=500, output_tokens=200,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        ))
        return MagicMock()

    monkeypatch.setattr(session, "run_pipeline", fake_pipeline)

    from v2.session import SessionResult, _run_pipeline_stage
    result = SessionResult()
    _run_pipeline_stage(result, session_id=1, completed_stages=set(),
                        skip=False, pipeline_hours=24, pipeline_limit=300)

    assert captured["stage"] == "pipeline"
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["input_tokens"] == 500
```

(Adjust `MagicMock` import at the top of the test file if not already present.)

- [ ] **Step 2: Run the test — verify it fails**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_session.py -v -k "passes_usage_to_complete"
```

Expected: assertion fails — `captured["model"]` is None because `_run_pipeline_stage` does not yet open `capture_usage()`.

- [ ] **Step 3: Modify `_complete_stage` / `_fail_stage` helpers in `v2/session.py`**

Replace the helpers at lines 110-125 with:

```python
def _complete_stage(session_id: int | None, stage: str, usage=None) -> None:
    if session_id is None:
        return
    try:
        complete_session_stage(session_id, stage, usage=usage)
    except Exception:
        pass


def _fail_stage(session_id: int | None, stage: str, error: str, usage=None) -> None:
    if session_id is None:
        return
    try:
        fail_session_stage(session_id, stage, error, usage=usage)
    except Exception:
        pass
```

- [ ] **Step 4: Wrap each `_run_*_stage` function with `capture_usage()`**

Add the import at the top of `v2/session.py`:

```python
from v2.claude_client import capture_usage
```

For each of the following functions in `v2/session.py`, replace the body inside `try:` so the stage code runs inside a `with capture_usage() as usage:` block, and pass `usage` to `_complete_stage` / `_fail_stage`:

- `_run_learning_refresh` (Stage 0 — no Claude calls today, but wrap defensively)
- `_run_pipeline_stage` (Stage 1)
- `_run_strategist_stage` (Stage 2)
- `_run_executor_stage` (Stage 3)
- `_run_strategy_stage` (Stage 4)
- `_run_twitter_stage` / `_run_bluesky_stage` / `_run_trade_posts_stage` (Stage 5 variants)
- (Skip Stage 6 dashboard publish — no Claude calls; leave untouched.)

The transformation pattern, illustrated for `_run_pipeline_stage`:

**Before:**

```python
    logger.info("[Stage 1] Running news pipeline")
    _start_stage(session_id, "pipeline")
    try:
        result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit)
        _complete_stage(session_id, "pipeline")
    except Exception as e:
        result.pipeline_error = str(e)
        _fail_stage(session_id, "pipeline", str(e))
        logger.error("Pipeline failed: %s — continuing with existing signals", e)
```

**After:**

```python
    logger.info("[Stage 1] Running news pipeline")
    _start_stage(session_id, "pipeline")
    with capture_usage() as usage:
        try:
            result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit)
            _complete_stage(session_id, "pipeline", usage=usage)
        except Exception as e:
            result.pipeline_error = str(e)
            _fail_stage(session_id, "pipeline", str(e), usage=usage)
            logger.error("Pipeline failed: %s — continuing with existing signals", e)
```

Apply the same transform to every listed function. Be sure to keep all other behavior identical (early returns, skip logic, post-stage memo persistence, etc.).

- [ ] **Step 5: Run the test — verify it passes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_session.py -v -k "passes_usage_to_complete"
```

Expected: pass.

- [ ] **Step 6: Run the full session test file**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_session.py -v
```

Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add v2/session.py tests/test_session.py
git commit -m "feat(v2): per-stage capture_usage in session.py"
```

---

## Task 8: End-of-session cost log line

**Files:**
- Modify: `v2/session.py` (add a helper + call it from the success path)
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_session.py`:

```python
def test_log_session_costs_emits_per_stage_breakdown(mock_db, caplog):
    """The end-of-session cost summary should log one line per stage with a
    Claude call, and a Total: line."""
    from v2.session import _log_session_costs

    # Mock the cursor.fetchall() to return three stages with cost data
    mock_db.fetchall.return_value = [
        {"stage_name": "pipeline",   "model": "claude-haiku-4-5-20251001",
         "cost_usd": 0.04, "input_tokens": 100, "output_tokens": 50,
         "cache_creation_tokens": 0, "cache_read_tokens": 0},
        {"stage_name": "strategist", "model": "claude-opus-4-7",
         "cost_usd": 1.28, "input_tokens": 5000, "output_tokens": 2000,
         "cache_creation_tokens": 100000, "cache_read_tokens": 500000},
        {"stage_name": "dashboard",  "model": None,
         "cost_usd": None, "input_tokens": None, "output_tokens": None,
         "cache_creation_tokens": None, "cache_read_tokens": None},
    ]

    with caplog.at_level("INFO", logger="session"):
        _log_session_costs(session_id=42)

    log_text = "\n".join(caplog.messages)
    assert "pipeline" in log_text
    assert "strategist" in log_text
    assert "$0.04" in log_text
    assert "$1.28" in log_text
    assert "$1.32" in log_text  # Total
    # Stages with NULL cost are omitted from the breakdown
    assert "dashboard" not in log_text
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_session.py -v -k "log_session_costs"
```

Expected: ImportError or AttributeError — `_log_session_costs` not defined.

- [ ] **Step 3: Implement `_log_session_costs` and wire into the success path**

Add to `v2/session.py`, near the top with the other private helpers (e.g. just below `_fail_stage`):

```python
def _log_session_costs(session_id: int | None) -> None:
    """Emit a per-stage cost breakdown to the session logger.

    Reads from the session_stage_costs view (defined in
    db/init/024_session_stage_token_usage.sql). Stages with NULL cost
    (no Claude calls or unseeded model) are omitted from the breakdown
    but contribute NULL to the SUM (i.e. nothing).
    """
    if session_id is None:
        return
    try:
        from v2.database.trading_db import get_cursor
        with get_cursor() as cur:
            cur.execute("""
                SELECT stage_name, model, cost_usd
                FROM session_stage_costs
                WHERE session_id = %s
                ORDER BY id
            """, (session_id,))
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("Could not load session costs: %s", e)
        return

    priced = [r for r in rows if r["cost_usd"] is not None]
    if not priced:
        return

    logger.info("Stage costs (USD):")
    for r in priced:
        logger.info("  %-12s $%.4f  (%s)", r["stage_name"] + ":", float(r["cost_usd"]), r["model"])
    total = sum(float(r["cost_usd"]) for r in priced)
    logger.info("  Total: $%.4f", total)
```

Then, in the session orchestrator's success-path log block (search for `"Session complete in %.1fs"` and `"All stages completed successfully"`), insert a call:

```python
_log_session_costs(session_id)
```

immediately before the `"Session complete in %.1fs"` line so the cost summary lands inside the trailing `=== ... ===` block.

- [ ] **Step 4: Run the test — verify it passes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_session.py -v -k "log_session_costs"
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add v2/session.py tests/test_session.py
git commit -m "feat(v2): log per-stage cost breakdown at session end"
```

---

## Task 9: Fix `_print_cost_summary` to use `pricing.stage_cost_usd`

**Files:**
- Modify: `v2/ideation_claude.py` (the `_print_cost_summary` function at line 245; the call site at line 229 is left untouched)

The function has exactly one caller (line 229). Keep the function, but rewrite its body to use `pricing.stage_cost_usd` and accept the `model` so it can resolve the right price. Update the single call site to pass `model`.

- [ ] **Step 1: Add the import**

At the top of `v2/ideation_claude.py`:

```python
from v2.pricing import UnknownModelError, stage_cost_usd
```

- [ ] **Step 2: Rewrite `_print_cost_summary`**

Replace the entire function body (currently lines 245-271):

```python
def _print_cost_summary(label, result, model, created, updated, closed, summary, adopted=0):
    """Print token usage and estimated USD cost for an agentic loop result.

    Uses v2.pricing.stage_cost_usd, which reads rates from the
    model_pricing DB table — single source of truth, matches the SQL
    cost view exactly. Prior version hardcoded Opus pricing inline and
    miscomputed uncached input by double-subtracting cache tokens.
    """
    try:
        cost = stage_cost_usd(
            model=model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_creation_tokens=result.cache_creation_input_tokens,
            cache_read_tokens=result.cache_read_input_tokens,
        )
        cost_str = f"${cost:.4f}"
    except UnknownModelError:
        cost_str = f"(no price seeded for model {model})"

    print("\n" + "=" * 60)
    print(f"{label} Complete")
    print("=" * 60)
    print(f"  Turns used: {result.turns_used}")
    print(f"  Stop reason: {result.stop_reason}")
    print(f"  Theses created: {created}")
    print(f"  Theses updated: {updated}")
    print(f"  Theses closed: {closed}")
    if adopted:
        print(f"  Theses adopted: {adopted}")
    print("\nToken usage:")
    print(f"  Input tokens: {result.input_tokens:,}")
    if result.cache_read_input_tokens:
        print(f"  Cache read tokens: {result.cache_read_input_tokens:,}")
        print(f"  Cache write tokens: {result.cache_creation_input_tokens:,}")
    print(f"  Output tokens: {result.output_tokens:,}")
    print(f"  Estimated cost: {cost_str}")
    print(f"\nSummary:\n{summary[:1000]}{'...' if len(summary) > 1000 else ''}")
```

- [ ] **Step 3: Update the single call site at line 229**

Find the call:

```python
    _print_cost_summary(label, result, created, updated, closed, summary, adopted=adopted)
```

Change to (note the new `model` argument):

```python
    _print_cost_summary(label, result, model, created, updated, closed, summary, adopted=adopted)
```

The enclosing function already has `model` in scope (it's a function parameter — verify by reading the surrounding function signature; if not, thread it through).

- [ ] **Step 4: Run the ideation_claude tests**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_ideation_claude.py -v
```

Expected: all pass. If any test asserts a specific dollar value from the prior buggy formula, update it to match the corrected formula (or the new "(no price seeded …)" string when the test mocks an unknown model).

- [ ] **Step 5: Commit**

```bash
git add v2/ideation_claude.py
git commit -m "refactor(v2): _print_cost_summary uses pricing.stage_cost_usd, takes model"
```

---

## Task 10: Pricing-coverage meta-test

**Files:**
- Create: `tests/test_pricing_coverage.py`

- [ ] **Step 1: Write the test**

Create `tests/test_pricing_coverage.py`:

```python
"""Meta-test: every hardcoded "claude-*" model literal in v2/ must have a
matching row in db/init/024_session_stage_token_usage.sql's seed.

If this test fails, you've added a new model pin without seeding its
price. Add a row to model_pricing in the migration (or in a follow-up
migration) and re-run."""

import re
from pathlib import Path


CLAUDE_MODEL_RE = re.compile(r'"(claude-[a-z0-9\-]+)"')
V2_DIR = Path(__file__).resolve().parent.parent / "v2"
MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "db" / "init" / "024_session_stage_token_usage.sql"
)


def _hardcoded_models_in_v2() -> set[str]:
    found = set()
    for py in V2_DIR.rglob("*.py"):
        text = py.read_text()
        for m in CLAUDE_MODEL_RE.finditer(text):
            found.add(m.group(1))
    return found


def _seeded_models() -> set[str]:
    text = MIGRATION.read_text()
    return set(re.findall(r"'(claude-[a-z0-9\-]+)'", text))


def test_every_hardcoded_model_is_seeded():
    hardcoded = _hardcoded_models_in_v2()
    seeded = _seeded_models()
    missing = hardcoded - seeded
    assert not missing, (
        f"Models referenced in v2/ but not seeded in model_pricing: {sorted(missing)}. "
        f"Add a row to db/init/024_session_stage_token_usage.sql."
    )
```

- [ ] **Step 2: Run the test**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/test_pricing_coverage.py -v
```

Expected: pass — all current hardcoded pins (`claude-haiku-4-5-20251001`, `claude-opus-4-6`, `claude-sonnet-4-6`) are in the seed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pricing_coverage.py
git commit -m "test(v2): meta-test that every Claude model literal has a price row"
```

---

## Task 11: Mark BUGS.md entry fixed

**Files:**
- Modify: `v2/BUGS.md:354`

- [ ] **Step 1: Update the BUGS.md entry**

Find line 354 in `v2/BUGS.md` (the `_print_cost_summary` hardcoded-pricing entry) and prepend `**FIXED 2026-05-05** — ` to the leading bullet text. If the file uses a different "fixed" convention (check by skimming earlier entries), follow that convention instead.

- [ ] **Step 2: Commit**

```bash
git add v2/BUGS.md
git commit -m "docs: mark _print_cost_summary pricing bug fixed"
```

---

## Task 12: Manual verification against paper DB

**Files:** none (verification only)

- [ ] **Step 1: Run a paper session end-to-end**

```bash
task paper:session:dry-run
```

Expected: session completes; logs show the new `Stage costs (USD):` block at the end with non-zero costs for any stage that hit Claude.

- [ ] **Step 2: Query the view directly**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -c "SELECT session_id, session_date, total_cost_usd FROM session_costs ORDER BY session_id DESC LIMIT 5;" \
    -c "SELECT stage_name, model, input_tokens, output_tokens, cost_usd FROM session_stage_costs WHERE session_id = (SELECT MAX(id) FROM sessions) ORDER BY id;"
```

Expected: the most recent session shows a non-NULL `total_cost_usd` and a per-stage breakdown that matches the log line emitted in Step 1 (within rounding).

- [ ] **Step 3: Apply the migration to prod DB before merging**

When the prod stack is brought up next:

```bash
docker compose cp db/init/024_session_stage_token_usage.sql db:/tmp/024.sql
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/024.sql
```

Expected: no errors. Verify with the same `\d session_stages` and `SELECT * FROM model_pricing;` checks from Task 1.

- [ ] **Step 4: Run the full test suite**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T trading-paper python -m pytest tests/
```

Expected: all tests pass (existing 782 + the new ones from this plan).

---

## Self-review notes

Coverage check against the spec:

- Schema (5 columns + `model_pricing` + 2 views) — Task 1.
- Capture primitive (`UsageAccumulator`, `capture_usage`, `_record_usage`) — Tasks 3, 4.
- `_call_with_retry` instrumentation (sole capture point) — Task 5.
- Per-stage wrapping in session.py — Task 7.
- DB write path — Task 6.
- Cost helper (`pricing.stage_cost_usd`) — Task 2.
- End-of-session log line — Task 8.
- Replace buggy `_print_cost_summary` — Task 9.
- Coverage test for unseeded models — Task 10.
- BUGS.md cleanup — Task 11.
- Manual verification + prod migration — Task 12.

No placeholders. All steps include exact code, exact paths, and exact commands.
