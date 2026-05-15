# LLM Context Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every LLM request/response round-trip for the strategist, executor, and reflection loops into a new `llm_call_contexts` Postgres table, so we can do forensic replay and prompt/context tuning.

**Architecture:** Single capture point inside `_call_with_retry` in `v2/claude_client.py`. A new helper `_record_call_context()` fires from the same `finally` block as the existing `agent_call` telemetry write, gated on `purpose ∈ {EXECUTOR, STRATEGIST_LOOP, REFLECTION_LOOP}` and `session_id is not None`. One row per LLM turn, captures `system`, `messages`, `tools`, `response.content`, tokens, stop_reason, duration. New `insert_llm_call_context()` writer in `v2/database/trading_db.py`. Failures swallowed so logging cannot break a session.

**Tech Stack:** Python 3, psycopg2 (raw SQL + `Json` adapter), pytest, anthropic SDK. Migrations are plain SQL files under `db/migrations/`.

**Spec:** `docs/superpowers/specs/2026-05-13-llm-context-logging-design.md`

---

## Task 1: Add the database migration

**Files:**
- Create: `db/migrations/006_llm_call_contexts.sql`

- [ ] **Step 1: Verify migration numbering**

Run: `ls db/migrations/ | sort`
Expected output ends with `005_drop_audit_tables.sql`. Confirms `006_` is the next number.

- [ ] **Step 2: Create the migration file**

Create `db/migrations/006_llm_call_contexts.sql` with:

```sql
-- db/migrations/006_llm_call_contexts.sql
-- Per-turn LLM request/response log for the strategist, executor, and
-- reflection loops. Gated by AgentPurpose so classifier calls don't
-- flood this table.
-- See docs/superpowers/specs/2026-05-13-llm-context-logging-design.md

CREATE TABLE IF NOT EXISTS llm_call_contexts (
    id                      BIGSERIAL PRIMARY KEY,
    session_id              INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    stage_name              TEXT NOT NULL,
    purpose                 TEXT NOT NULL,
    sequence                INTEGER NOT NULL,
    model                   TEXT NOT NULL,
    system_prompt           TEXT,
    messages                JSONB NOT NULL,
    tool_definitions        JSONB,
    response_content        JSONB,
    input_tokens            INTEGER,
    output_tokens           INTEGER,
    cache_read_tokens       INTEGER,
    cache_creation_tokens   INTEGER,
    stop_reason             TEXT,
    duration_ms             INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, stage_name, purpose, sequence)
);

CREATE INDEX IF NOT EXISTS idx_llm_call_contexts_session_stage
    ON llm_call_contexts (session_id, stage_name);
CREATE INDEX IF NOT EXISTS idx_llm_call_contexts_created_at
    ON llm_call_contexts (created_at);
```

- [ ] **Step 3: Apply migration against the dev DB**

The dev stack auto-applies new files on container startup. Apply by:

```bash
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/migrations/006_llm_call_contexts.sql
```

If the container doesn't mount `db/migrations/` at that path, copy first:

```bash
docker compose cp db/migrations/006_llm_call_contexts.sql db:/tmp/006.sql
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/006.sql
```

Expected: `CREATE TABLE` then `CREATE INDEX` printed twice. No errors.

- [ ] **Step 4: Verify the schema**

Run:
```bash
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d llm_call_contexts"
```

Expected: table with all 16 columns, the UNIQUE constraint on `(session_id, stage_name, purpose, sequence)`, and the two indexes.

- [ ] **Step 5: Commit**

```bash
git add db/migrations/006_llm_call_contexts.sql
git commit -m "feat(db): add llm_call_contexts table for strategist/executor/reflection round-trip logging"
```

---

## Task 2: Write the database insert helper

**Files:**
- Modify: `v2/database/trading_db.py` (append a new function near the other `insert_*` helpers)
- Test: `tests/v2/test_db.py` (append a new test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/v2/test_db.py`:

```python
class TestLlmCallContexts:
    def test_insert_llm_call_context_executes_insert_with_jsonb_payload(self, mock_db, mock_cursor):
        from v2.database.trading_db import insert_llm_call_context

        insert_llm_call_context(
            session_id=42,
            stage_name="trading",
            purpose="executor",
            model="claude-haiku-4-5-20251001",
            system_prompt="you are a trading executor",
            messages=[{"role": "user", "content": "hi"}],
            tool_definitions=None,
            response_content=[{"type": "text", "text": "ok"}],
            input_tokens=120,
            output_tokens=45,
            cache_read_tokens=80,
            cache_creation_tokens=10,
            stop_reason="end_turn",
            duration_ms=987,
        )

        sql = mock_cursor.execute.call_args[0][0]
        params = mock_cursor.execute.call_args[0][1]
        assert "INSERT INTO llm_call_contexts" in sql
        # sequence is computed by a sub-select in the same statement
        assert "MAX(sequence)" in sql
        # JSON-wrapped fields go through psycopg2's Json adapter
        from psycopg2.extras import Json
        assert any(isinstance(p, Json) for p in params)

    def test_insert_llm_call_context_serializes_tool_definitions_when_present(self, mock_db, mock_cursor):
        from v2.database.trading_db import insert_llm_call_context

        tools = [{"name": "get_positions", "input_schema": {"type": "object"}}]
        insert_llm_call_context(
            session_id=42,
            stage_name="ideation",
            purpose="strategist_loop",
            model="claude-opus-4-7",
            system_prompt="strategist system",
            messages=[{"role": "user", "content": "go"}],
            tool_definitions=tools,
            response_content=[{"type": "tool_use", "id": "t1", "name": "get_positions", "input": {}}],
            input_tokens=200,
            output_tokens=80,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            stop_reason="tool_use",
            duration_ms=4200,
        )
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO llm_call_contexts" in sql
        # tool_definitions should appear in the column list of the INSERT
        assert "tool_definitions" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_db.py::TestLlmCallContexts -v`
Expected: FAIL with `ImportError: cannot import name 'insert_llm_call_context'` (or similar).

- [ ] **Step 3: Add the writer to `v2/database/trading_db.py`**

Append near the other `insert_*` helpers (after the news/macro inserts, anywhere — order does not matter):

```python
def insert_llm_call_context(
    session_id: int,
    stage_name: str,
    purpose: str,
    model: str,
    system_prompt: str | None,
    messages: list,
    tool_definitions: list | None,
    response_content: list | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_tokens: int | None,
    cache_creation_tokens: int | None,
    stop_reason: str | None,
    duration_ms: int | None,
) -> None:
    """Persist one LLM request/response round-trip for forensic replay.

    `sequence` is computed in-statement as MAX(sequence)+1 within the
    (session_id, stage_name, purpose) group. The UNIQUE constraint is a
    backstop against the read-then-insert race; in practice there is one
    writer per session so the race does not occur.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_call_contexts (
                session_id, stage_name, purpose, sequence,
                model, system_prompt, messages, tool_definitions,
                response_content, input_tokens, output_tokens,
                cache_read_tokens, cache_creation_tokens,
                stop_reason, duration_ms
            )
            VALUES (
                %s, %s, %s,
                COALESCE(
                    (SELECT MAX(sequence) + 1 FROM llm_call_contexts
                     WHERE session_id = %s AND stage_name = %s AND purpose = %s),
                    0
                ),
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s
            )
            """,
            (
                session_id, stage_name, purpose,
                session_id, stage_name, purpose,
                model, system_prompt, Json(messages),
                Json(tool_definitions) if tool_definitions is not None else None,
                Json(response_content) if response_content is not None else None,
                input_tokens, output_tokens,
                cache_read_tokens, cache_creation_tokens,
                stop_reason, duration_ms,
            ),
        )
```

`Json` and `get_cursor` are already imported at the top of `trading_db.py` (line 8 and line 10 respectively) — no new imports needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_db.py::TestLlmCallContexts -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "feat(db): add insert_llm_call_context writer"
```

---

## Task 3: Define the captured-purpose set and helper signature

**Files:**
- Modify: `v2/claude_client.py` (add a module-level constant and a stub function)
- Test: `tests/v2/test_claude_client_context_logging.py` (new file)

This task introduces the gate and the helper *without* wiring it into `_call_with_retry` yet. Keeps the next task small and focused on the integration.

- [ ] **Step 1: Write the failing tests**

Create `tests/v2/test_claude_client_context_logging.py`:

```python
"""Tests for LLM context logging captured at _call_with_retry.

Spec: docs/superpowers/specs/2026-05-13-llm-context-logging-design.md
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestCapturedPurposes:
    def test_executor_strategist_reflection_are_captured(self):
        from v2.claude_client import _CONTEXT_LOGGED_PURPOSES, AgentPurpose

        assert AgentPurpose.EXECUTOR in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.STRATEGIST_LOOP in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.REFLECTION_LOOP in _CONTEXT_LOGGED_PURPOSES

    def test_classifier_purposes_not_captured(self):
        from v2.claude_client import _CONTEXT_LOGGED_PURPOSES, AgentPurpose

        assert AgentPurpose.CLASSIFIER_NEWS not in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.CLASSIFIER_MACRO not in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.CLASSIFIER_RELEVANCE not in _CONTEXT_LOGGED_PURPOSES


class TestRecordCallContextHelper:
    """`_record_call_context` is the gate + serializer. It must be a no-op
    when session_id is None, when purpose is not captured, or when the DB
    write raises."""

    def test_no_op_when_session_id_is_none(self, monkeypatch):
        from v2.claude_client import _record_call_context

        called = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: called.append(kw),
        )
        _record_call_context(
            session_id=None,
            stage_name="trading",
            purpose="executor",
            create_kwargs={"model": "m", "messages": [], "system": ""},
            message=None,
            duration_ms=10,
        )
        assert called == []

    def test_no_op_when_purpose_not_captured(self, monkeypatch):
        from v2.claude_client import _record_call_context

        called = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: called.append(kw),
        )
        _record_call_context(
            session_id=42,
            stage_name="pipeline",
            purpose="classifier_news",
            create_kwargs={"model": "m", "messages": [], "system": ""},
            message=None,
            duration_ms=10,
        )
        assert called == []

    def test_swallows_db_errors(self, monkeypatch, caplog):
        from v2.claude_client import _record_call_context

        def boom(**_kw):
            raise RuntimeError("db down")
        monkeypatch.setattr("v2.claude_client.insert_llm_call_context", boom)

        message = MagicMock()
        message.content = [SimpleNamespace(type="text", text="ok")]
        message.stop_reason = "end_turn"
        message.usage = SimpleNamespace(
            input_tokens=1, output_tokens=2,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        # Must not raise
        _record_call_context(
            session_id=42,
            stage_name="trading",
            purpose="executor",
            create_kwargs={"model": "m", "messages": [{"role": "user", "content": "hi"}], "system": "sys"},
            message=message,
            duration_ms=10,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_claude_client_context_logging.py -v`
Expected: FAIL on imports — `_CONTEXT_LOGGED_PURPOSES` and `_record_call_context` do not exist yet.

- [ ] **Step 3: Add the constant, import, and helper stub**

In `v2/claude_client.py`:

1. Find the existing `from .telemetry import record_event` line (around line 14) and add the new import right below it:

```python
from .database.trading_db import insert_llm_call_context
```

2. After the `class AgentPurpose:` block (ends around line 40), add the captured-purpose set:

```python
_CONTEXT_LOGGED_PURPOSES = frozenset({
    AgentPurpose.EXECUTOR,
    AgentPurpose.STRATEGIST_LOOP,
    AgentPurpose.REFLECTION_LOOP,
})
```

3. Above `def _call_with_retry(...)` (around line 137), add the helper:

```python
def _serialize_content_blocks(content) -> list:
    """Convert anthropic content blocks to JSON-serializable dicts.

    Real anthropic SDK blocks are pydantic models exposing `.model_dump()`.
    Test fixtures use SimpleNamespace. Fall back to `vars()` for the
    latter so tests don't have to add a model_dump shim everywhere.
    """
    result = []
    for block in content or []:
        if hasattr(block, "model_dump"):
            result.append(block.model_dump())
        else:
            result.append({k: v for k, v in vars(block).items()})
    return result


def _record_call_context(
    *,
    session_id,
    stage_name,
    purpose,
    create_kwargs: dict,
    message,
    duration_ms: int,
) -> None:
    """Persist the LLM round-trip for forensic replay.

    Gated on (a) a real session_id and (b) a captured purpose. Errors
    are logged and swallowed — context logging must never break a
    session.
    """
    if session_id is None:
        return
    if purpose not in _CONTEXT_LOGGED_PURPOSES:
        return
    try:
        usage = getattr(message, "usage", None) if message is not None else None
        response_content = (
            _serialize_content_blocks(message.content)
            if message is not None and getattr(message, "content", None) is not None
            else None
        )
        insert_llm_call_context(
            session_id=session_id,
            stage_name=stage_name or "unknown",
            purpose=purpose,
            model=create_kwargs.get("model"),
            system_prompt=create_kwargs.get("system") if isinstance(create_kwargs.get("system"), str) else None,
            messages=create_kwargs.get("messages") or [],
            tool_definitions=create_kwargs.get("tools"),
            response_content=response_content,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) if usage else None,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None) if usage else None,
            stop_reason=getattr(message, "stop_reason", None) if message else None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        logger.warning(
            "Failed to record llm_call_context (session=%s stage=%s purpose=%s): %s",
            session_id, stage_name, purpose, e,
        )
```

Note on `system_prompt`: the strategist passes `system` as a string in production. If a future call site ever passes a list-of-blocks form, we capture None to avoid storing a malformed value; we can extend later if/when that happens.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_claude_client_context_logging.py -v`
Expected: 4 passed (3 in the new file + the existing ones once Task 4 adds them).

If only the 4 written so far pass, that's the expected state at this point.

- [ ] **Step 5: Commit**

```bash
git add v2/claude_client.py tests/v2/test_claude_client_context_logging.py
git commit -m "feat(claude_client): add _record_call_context helper (not yet wired)"
```

---

## Task 4: Wire `_record_call_context` into `_call_with_retry`

**Files:**
- Modify: `v2/claude_client.py` (the `finally` block of `_call_with_retry`)
- Modify: `tests/v2/test_claude_client_context_logging.py` (append integration tests)

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/v2/test_claude_client_context_logging.py`:

```python
class TestCallWithRetryWritesContext:
    """End-to-end: a `_call_with_retry` for a captured purpose writes one
    llm_call_contexts row alongside the existing `agent_call` telemetry
    event."""

    def _text_block(self, text):
        return SimpleNamespace(type="text", text=text)

    def _tool_use_block(self, tool_id, name, input_data):
        return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=input_data)

    def _make_response(self, content, stop_reason="end_turn",
                       input_tokens=100, output_tokens=50,
                       cache_creation=0, cache_read=0):
        response = MagicMock()
        response.content = content
        response.stop_reason = stop_reason
        response.usage = MagicMock()
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        response.usage.cache_creation_input_tokens = cache_creation
        response.usage.cache_read_input_tokens = cache_read
        return response

    def _make_stream_mock(self, response):
        def stream_factory(**_kwargs):
            ctx = MagicMock()
            ctx.__enter__.return_value = ctx
            ctx.__exit__.return_value = None
            ctx.get_final_message.return_value = response
            return ctx
        client = MagicMock()
        client.messages.stream.side_effect = stream_factory
        return client

    def test_executor_call_writes_context_row(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        response = self._make_response(
            content=[self._text_block("decision json here")],
            stop_reason="end_turn",
            input_tokens=120, output_tokens=45,
            cache_creation=10, cache_read=80,
        )
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            system="executor system prompt",
            messages=[{"role": "user", "content": "executor input json"}],
            session_id=42,
            stage_name="trading",
            purpose="executor",
        )

        assert len(recorded_inserts) == 1
        row = recorded_inserts[0]
        assert row["session_id"] == 42
        assert row["stage_name"] == "trading"
        assert row["purpose"] == "executor"
        assert row["model"] == "claude-haiku-4-5-20251001"
        assert row["system_prompt"] == "executor system prompt"
        assert row["messages"] == [{"role": "user", "content": "executor input json"}]
        assert row["tool_definitions"] is None
        assert row["response_content"] == [{"type": "text", "text": "decision json here"}]
        assert row["input_tokens"] == 120
        assert row["output_tokens"] == 45
        assert row["cache_read_tokens"] == 80
        assert row["cache_creation_tokens"] == 10
        assert row["stop_reason"] == "end_turn"
        assert row["duration_ms"] is not None and row["duration_ms"] >= 0

    def test_strategist_call_writes_context_row_with_tools_and_tool_use_response(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        tools = [{"name": "get_positions", "input_schema": {"type": "object"}}]
        response = self._make_response(
            content=[
                self._text_block("checking positions"),
                self._tool_use_block("t1", "get_positions", {}),
            ],
            stop_reason="tool_use",
        )
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="claude-opus-4-7",
            max_tokens=32000,
            system="strategist system",
            messages=[{"role": "user", "content": "go"}],
            tools=tools,
            session_id=99,
            stage_name="ideation",
            purpose="strategist_loop",
        )

        assert len(recorded_inserts) == 1
        row = recorded_inserts[0]
        assert row["purpose"] == "strategist_loop"
        assert row["tool_definitions"] == tools
        assert row["response_content"] == [
            {"type": "text", "text": "checking positions"},
            {"type": "tool_use", "id": "t1", "name": "get_positions", "input": {}},
        ]
        assert row["stop_reason"] == "tool_use"

    def test_classifier_call_does_not_write_context_row(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        recorded_events = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kw: recorded_events.append(kw),
        )

        response = self._make_response(content=[self._text_block("category")])
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="m",
            max_tokens=200,
            system="classify",
            messages=[{"role": "user", "content": "headline"}],
            session_id=7,
            stage_name="pipeline",
            purpose="classifier_news",
        )

        # No llm_call_contexts row …
        assert recorded_inserts == []
        # … but the telemetry event is still emitted.
        assert len(recorded_events) == 1

    def test_no_session_id_does_not_write_context_row(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        response = self._make_response(content=[self._text_block("hi")])
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="m",
            max_tokens=200,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            session_id=None,
            stage_name="trading",
            purpose="executor",
        )

        assert recorded_inserts == []

    def test_context_logging_failure_does_not_break_caller(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        def boom(**_kw):
            raise RuntimeError("db down")

        monkeypatch.setattr("v2.claude_client.insert_llm_call_context", boom)
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        response = self._make_response(content=[self._text_block("ok")])
        client = self._make_stream_mock(response)

        result = _call_with_retry(
            client,
            model="m",
            max_tokens=200,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            session_id=42,
            stage_name="trading",
            purpose="executor",
        )
        # The caller still gets back a real response.
        assert result is response
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_claude_client_context_logging.py::TestCallWithRetryWritesContext -v`
Expected: FAIL on `test_executor_call_writes_context_row` (and the others) — `recorded_inserts` is empty because `_record_call_context` is not yet called from `_call_with_retry`.

- [ ] **Step 3: Wire the helper into `_call_with_retry`'s finally block**

Find the `finally:` block at the end of `_call_with_retry` (around line 193). It currently ends with the `record_event(...)` call at line 212. Append after that call, *inside the same `finally` block*:

```python
        _record_call_context(
            session_id=session_id,
            stage_name=stage_name,
            purpose=purpose,
            create_kwargs=create_kwargs,
            message=message,
            duration_ms=duration_ms,
        )
```

Order matters: `record_event` runs first so telemetry is durable even if context logging itself blows up unexpectedly (defense in depth — `_record_call_context` already has its own try/except).

- [ ] **Step 4: Run the new tests**

Run: `python3 -m pytest tests/v2/test_claude_client_context_logging.py -v`
Expected: all 9 tests pass (4 from Task 3 + 5 from Task 4).

- [ ] **Step 5: Run the existing `test_claude_client.py` to confirm no regressions**

Run: `python3 -m pytest tests/v2/test_claude_client.py -v`
Expected: all existing tests still pass. The existing tests pass `record_event` mocks but not `insert_llm_call_context` mocks — they should still pass because:
- Tests that use `purpose="classifier_news"` are gated out by `_CONTEXT_LOGGED_PURPOSES`.
- Tests that use `purpose="executor"` or `"strategist_loop"` but `session_id=None` are gated out by the `session_id` check.
- Any test that uses a captured purpose AND a non-None session_id would try to hit the real `insert_llm_call_context`. If such tests exist and start failing, patch `v2.claude_client.insert_llm_call_context` to a no-op for those tests using `monkeypatch.setattr` — do this in the failing test itself, not as a blanket fix.

If a regression appears, fix it in that test before continuing.

- [ ] **Step 6: Commit**

```bash
git add v2/claude_client.py tests/v2/test_claude_client_context_logging.py
git commit -m "feat(claude_client): persist strategist/executor/reflection round-trips to llm_call_contexts"
```

---

## Task 5: Full-suite regression sweep

**Files:** none (read-only validation)

- [ ] **Step 1: Run the full v2 test suite**

Run: `python3 -m pytest tests/v2/ -v`
Expected: all tests pass. Pay particular attention to:
- `tests/v2/test_ideation_claude.py` — exercises strategist loop end-to-end
- `tests/v2/test_executor.py` / `tests/v2/test_agent.py` — exercises executor
- `tests/v2/test_claude_client.py` — exercises `_call_with_retry` directly

If any test fails because it now tries to write a real `insert_llm_call_context` row (the DB is not patched for that test), patch the helper in that test:

```python
monkeypatch.setattr("v2.claude_client.insert_llm_call_context", lambda **kw: None)
```

Do this surgically per-test — do not blanket-patch in `conftest.py`. We want real call paths exercised.

- [ ] **Step 2: Run the rest of the suite**

Run: `python3 -m pytest tests/ -v`
Expected: full suite green. Memory says we have 782 tests; you should see at least that many + the new ones, all passing.

- [ ] **Step 3: Smoke test against the live dev DB**

This exercises the real Postgres write path one time, end-to-end. Skip if not running the docker stack locally.

```bash
docker compose exec trading python3 -c "
from v2.database.trading_db import insert_llm_call_context
from v2.database.connection import get_cursor

# Need a real session_id (FK). Grab the most recent.
with get_cursor() as cur:
    cur.execute('SELECT id FROM sessions ORDER BY id DESC LIMIT 1')
    row = cur.fetchone()
    if row is None:
        print('No sessions exist yet — skipping smoke test')
    else:
        sid = row['id']
        insert_llm_call_context(
            session_id=sid,
            stage_name='smoke_test',
            purpose='executor',
            model='claude-haiku-4-5-20251001',
            system_prompt='smoke test system',
            messages=[{'role': 'user', 'content': 'hi'}],
            tool_definitions=None,
            response_content=[{'type': 'text', 'text': 'ok'}],
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            stop_reason='end_turn',
            duration_ms=100,
        )
        cur.execute(
            \"\"\"SELECT id, sequence, model FROM llm_call_contexts
               WHERE session_id = %s AND stage_name = 'smoke_test'\"\"\",
            (sid,),
        )
        for r in cur.fetchall():
            print(r)
"
```

Expected: prints one row with the model and `sequence=0`.

Clean up:
```bash
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "DELETE FROM llm_call_contexts WHERE stage_name = 'smoke_test'"
```

- [ ] **Step 4: No code changes here — nothing to commit unless you fixed a test in step 1**

If you patched a test in step 1, commit that fix with the message
`test: patch insert_llm_call_context in <test_name> after context-logging wire-in`.

---

## Task 6: One full session against the dev stack

**Files:** none

This is the final integration check: a real strategist+executor+reflection run that produces rows.

- [ ] **Step 1: Confirm the stack is up**

Run: `docker compose ps`
Expected: `trading`, `db`, `dashboard` services in `Up` state.

- [ ] **Step 2: Run a paper-stack dry-run session**

Run: `task paper:session:dry-run`
(Or, equivalent: `docker compose -f docker-compose.yml -f docker-compose.paper.yml --env-file .env.paper exec trading-paper python -m v2.session --dry-run`.)

Expected: the session completes through ideation (strategist), trading (executor), and strategy (reflection) stages without errors. Token counts logged as usual.

- [ ] **Step 3: Inspect the new table**

Run:
```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml --env-file .env.paper exec db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
SELECT session_id, stage_name, purpose, sequence,
       length(system_prompt) AS sys_len,
       jsonb_array_length(messages) AS n_messages,
       jsonb_array_length(response_content) AS n_response_blocks,
       stop_reason, duration_ms
FROM llm_call_contexts
WHERE session_id = (SELECT MAX(id) FROM sessions)
ORDER BY purpose, sequence;
"
```

Expected:
- 1 row for `purpose='executor'`, `sequence=0`, with `n_messages=1`, response blocks present.
- N rows for `purpose='strategist_loop'`, sequences `0..N-1`, with `n_messages` increasing across rows (each turn appends to the conversation).
- M rows for `purpose='reflection_loop'`, same shape.
- No rows for any `classifier_*` purpose.

- [ ] **Step 4: No commits expected here.** This step is validation only. If anything looks wrong, file a follow-up rather than patching in this task — the plan ends.

---

## Self-review notes

- **Spec coverage:** schema (Task 1), capture point in `_call_with_retry` (Task 4), purpose gate (Task 3), session-id gate (Task 3), tools captured (Task 4), response content captured (Tasks 2, 3, 4), failure swallowed (Task 3, 4), tests for all six bullet points in the spec's Testing section (Tasks 2-4), retention deferred per spec.
- **No placeholders:** every step has either a command, code, or a verification step with an expected outcome.
- **Type consistency:** `insert_llm_call_context` signature is identical in Tasks 2 and 3. The set name `_CONTEXT_LOGGED_PURPOSES` is consistent across Tasks 3 and 4. Constants reference `AgentPurpose.EXECUTOR` / `STRATEGIST_LOOP` / `REFLECTION_LOOP` — verified against `v2/claude_client.py:32-40`.
- **Frequent commits:** five commits across six tasks (Task 5 is a regression sweep, Task 6 is validation).
