# Auditor-Visible Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the auditor's known behavioral blind spots by adding a generic event log (`agent_events`), a small instrumentation surface in the three places where the auditor cannot currently see (agentic-loop tool dispatch, evidence shown to Claude, risk-gate rejections), and six new auditor checks that consume those events. After this, the auditor can ask questions like *"is the strategist using the new `get_recent_playbooks` tool?"* and *"is reflection acting on round-trip evidence?"* — both currently unanswerable without manual log inspection.

**Architecture:** One generic JSONB event table (zero-migration to add new event types later), three thin instrumentation hooks (`run_agentic_loop` dispatch, `tool_get_session_summary`, `check_sector_cap_for_buy`), and six new auditor checks added to `v2/audit.py`'s `CHECKS` list. No prompt changes, no business-logic changes, no new agentic loops. Telemetry inserts are best-effort (try/except wrapped) so a logging failure cannot break a session.

**Tech Stack:** PostgreSQL 16 (new migration `026_agent_events.sql`), psycopg2 raw SQL via `get_cursor()`, Python 3.x, pytest with `mock_db`/`mock_cursor` fixtures from `tests/v2/conftest.py:75-104`. New auditor checks follow the existing `Finding`/`check_*` pattern in `v2/audit.py`.

---

## Auditor blind-spot inventory (2026-05-08)

The current auditor (`v2/audit.py`, 11 active checks) reads only persisted state. It can flag:
- Orphan FKs, missing backfill, invalid attribution categories, snapshot gaps, equity drift (Tier 1)
- Five rule-overfitting patterns via Haiku judgment (Tier 2)
- Attribution coverage, stage failure rate, stage stale-running, cost spikes, missing signal_refs (Tier 3)

It cannot see:

| # | Blind spot | Why current auditor misses it |
|---|---|---|
| 1 | Did the strategist call `get_recent_playbooks`? | No record of LLM tool calls anywhere |
| 2 | Did the strategist call `propose_rule` / `retire_rule` this session? | Rule changes land in `strategy_rules` but aren't keyed back to the session that produced them |
| 3 | Are tools failing inside the agentic loop? | Tool errors logged to stderr only |
| 4 | What evidence (round-trips, attribution, etc.) did reflection see? | Computed inside `tool_get_session_summary`, returned as text, never persisted |
| 5 | Is the risk gate (`check_sector_cap_for_buy`) blocking the same ticker repeatedly? | Rejections produce a `[REJECTED: sector ...]` reason in `decisions.reasoning` but no structured event |
| 6 | Are buys being repeatedly rejected by sector cap on the same date? | Same reason as #5 — no structured rejection log |

**Cuts (intentionally not added):**
- Rule citation counts — already computed inline in `check_rule_judgment` and `check_decisions_missing_signal_refs`. Persisting would duplicate cheap regex.
- Stage Claude metadata (stop_reason, max_turns, cache hit ratio) — `session_stages` already has token counts. Adding columns is preferable to events when the data is per-stage-summary, not per-event.
- Prompt versioning / hash — no A/B testing happening; YAGNI until a prompt change correlates with a behavior shift we want to attribute.
- Position / order divergence — already covered by Alpaca sync (per audit design spec §5.3).

The three event types below collapse blind spots 1-6 into the minimum useful taxonomy.

---

## Event taxonomy

| event_type | Fired by | Payload schema (illustrative) |
|---|---|---|
| `tool_invocation` | `run_agentic_loop` per tool dispatch | `{"tool_name": str, "args": dict, "success": bool, "error": str\|null, "duration_ms": int}` |
| `evidence_shown` | `tool_get_session_summary` (initially), extensible | `{"evidence_kind": "round_trips"\|..., "items": list, "summary": dict}` |
| `risk_block` | `_prepare_decision` in `v2/trader.py` when sector cap rejects | `{"ticker": str, "sector": str, "proposed_qty": float, "price": float, "sector_pct_after": float, "cap": float}` |

`session_id` and `stage_name` are first-class columns, not in the payload, so the auditor's joins to `sessions` / `session_stages` stay cheap.

---

## New auditor checks unlocked

| check_code | Tier | Severity rule | What it asks | Event source |
|---|---|---|---|---|
| `STRATEGIST_NOT_USING_REVERSAL_TOOL` | 3 | warn if 3+ consecutive ideation sessions with reversals but no `get_recent_playbooks` call | "Did strategist consult prior playbooks before reversing?" | `tool_invocation` |
| `REFLECTION_INERT_ON_ROUND_TRIPS` | 3 | warn if `evidence_shown.evidence_kind=round_trips` non-empty for 5 sessions and no `propose_rule`/`retire_rule` from reflection in same window | "Is reflection ignoring oscillation signals?" | `evidence_shown` + `tool_invocation` |
| `TOOL_ERROR_RATE` | 3 | warn if any tool >20% error rate over 7d, critical if >50% | "Are tools silently failing?" | `tool_invocation` |
| `RISK_BLOCK_HOTSPOT` | 3 | warn if same ticker triggers ≥3 `risk_block` events in 7d | "Is strategist proposing buys the gate keeps rejecting? (Failure to learn)" | `risk_block` |
| `RISK_BLOCK_BURST` | 3 | warn if ≥5 `risk_block` events on a single date | "Did one session generate a wave of cap-blocked buys?" | `risk_block` |
| `IDEATION_TOOL_DROUGHT` | 3 | warn if any tool listed in expected ideation toolset has 0 calls across last 7 ideation sessions | "Did the strategist stop using a tool entirely?" | `tool_invocation` |

Six checks. Each defends itself against *"what would the auditor flag, and what would I do about it?"*

---

## Files

**New:**
- `db/init/026_agent_events.sql` — `agent_events` table + indexes
- `v2/telemetry.py` — `record_event()` helper + small query helpers used by the new auditor checks
- `tests/v2/test_telemetry.py` — unit tests for helpers

**Modify (production):**
- `v2/claude_client.py:272-443` — extend `run_agentic_loop` signature with `(session_id, stage_name)`; emit `tool_invocation` events
- `v2/ideation_claude.py:219` — pass `session_id`, `stage_name="ideation"` through
- `v2/strategy.py:275-360, :545` — wrapper around `tool_get_session_summary` that emits `evidence_shown` events; pass telemetry args through to `run_agentic_loop`
- `v2/trader.py:~520` — emit `risk_block` event when `check_sector_cap_for_buy` returns a breach
- `v2/audit.py` — add 6 new check functions + register in `CHECKS` list
- `v2/session.py` — pass `session_id` through to ideation + reflection stages (if not already)

**Modify (tests):**
- `tests/v2/test_claude_client.py` — assert tool dispatch emits `tool_invocation` events
- `tests/v2/test_ideation_claude.py` — assert telemetry args threaded through
- `tests/v2/test_strategy.py` — assert reflection emits `evidence_shown` for round-trips
- `tests/v2/test_trader.py` — assert sector-cap rejection emits `risk_block`
- `tests/v2/test_audit.py` — 6 new test classes, one per new check

Total: 1 SQL migration, 1 new module + tests, ~6 production files modified, ~5 test files extended. Estimate: ~450 LOC (≈200 production, ≈250 tests). One new table. Zero changes to existing tables, prompts, or LLM tool definitions.

---

## Task 1: Schema migration

**Files:**
- New: `db/init/026_agent_events.sql`

- [ ] **Step 1.1: Write the migration**

Create `db/init/026_agent_events.sql`:

```sql
-- Generic event log for LLM-side and gate-side observability.
-- One table for all event types (zero-migration extensibility).
-- Auditor reads from this table; nothing else does. Inserts are
-- best-effort from the producer side.
CREATE TABLE IF NOT EXISTS agent_events (
    id           BIGSERIAL PRIMARY KEY,
    session_id   INT REFERENCES sessions(id) ON DELETE CASCADE,
    stage_name   VARCHAR(50),
    event_type   VARCHAR(50) NOT NULL,
    payload      JSONB NOT NULL,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_events_session
    ON agent_events(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_type_time
    ON agent_events(event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_events_stage_type
    ON agent_events(stage_name, event_type);

-- Functional index for tool_name lookups inside payload (covers
-- STRATEGIST_NOT_USING_REVERSAL_TOOL and TOOL_ERROR_RATE checks).
CREATE INDEX IF NOT EXISTS idx_agent_events_tool_name
    ON agent_events ((payload->>'tool_name'))
    WHERE event_type = 'tool_invocation';

-- Functional index for risk_block ticker hotspot detection.
CREATE INDEX IF NOT EXISTS idx_agent_events_ticker
    ON agent_events ((payload->>'ticker'))
    WHERE event_type = 'risk_block';
```

- [ ] **Step 1.2: Apply to paper DB**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper \
  psql -U algo -d trading < db/init/026_agent_events.sql
```

- [ ] **Step 1.3: Verify table + indexes**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper \
  psql -U algo -d trading -c "\d agent_events"
```

Expected: table prints with all 4 indexes.

- [ ] **Step 1.4: Apply to prod DB after Tasks 2-6 are passing in paper**

Defer to Task 8.

---

## Task 2: `v2/telemetry.py` — generic recorder + read helpers

**Files:**
- New: `v2/telemetry.py`
- New: `tests/v2/test_telemetry.py`

The producer side has one function: `record_event(session_id, stage_name, event_type, payload)`. The reader side has small helpers used by the auditor: `count_tool_invocations(session_id, tool_name)`, `recent_evidence_kinds_seen(days, evidence_kind)`, `risk_block_counts_by_ticker(days)`. Keep helpers thin — the auditor checks do their own SQL beyond what these expose.

- [ ] **Step 2.1: Write failing tests for `record_event`**

Create `tests/v2/test_telemetry.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


class TestRecordEvent:
    def test_inserts_row_with_jsonb_payload(self):
        from v2.telemetry import record_event
        cur = MagicMock()
        with patch("v2.telemetry.get_cursor") as gc:
            gc.return_value.__enter__.return_value = cur
            record_event(
                session_id=42, stage_name="ideation",
                event_type="tool_invocation",
                payload={"tool_name": "get_recent_playbooks", "success": True},
            )
        sql, params = cur.execute.call_args[0]
        assert "INSERT INTO agent_events" in sql
        assert params[0] == 42
        assert params[1] == "ideation"
        assert params[2] == "tool_invocation"
        assert "get_recent_playbooks" in params[3]  # JSON-serialized

    def test_noop_when_session_id_none(self):
        from v2.telemetry import record_event
        with patch("v2.telemetry.get_cursor") as gc:
            record_event(None, "ideation", "tool_invocation", {})
        gc.assert_not_called()

    def test_swallows_exceptions(self):
        """Telemetry must never break a session."""
        from v2.telemetry import record_event
        with patch("v2.telemetry.get_cursor", side_effect=RuntimeError("DB down")):
            # Should not raise
            record_event(1, "ideation", "tool_invocation", {})

    def test_serializes_dates_in_payload(self):
        from v2.telemetry import record_event
        from datetime import date
        cur = MagicMock()
        with patch("v2.telemetry.get_cursor") as gc:
            gc.return_value.__enter__.return_value = cur
            record_event(
                session_id=1, stage_name="reflection",
                event_type="evidence_shown",
                payload={"items": [{"ticker": "GOOGL", "first_date": date(2026, 5, 1)}]},
            )
        params = cur.execute.call_args[0][1]
        assert "2026-05-01" in params[3]  # date serialized as ISO string
```

- [ ] **Step 2.2: Run to verify failure**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_telemetry.py -v
```

Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 2.3: Implement `record_event`**

Create `v2/telemetry.py`:

```python
"""Generic agent_events recorder + small auditor-facing query helpers.

`record_event` is a no-op when session_id is None and swallows DB errors:
telemetry must never break a session.
"""
import json
import logging
from datetime import date, datetime
from .database.connection import get_cursor

logger = logging.getLogger(__name__)


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def record_event(session_id, stage_name, event_type, payload):
    if session_id is None:
        return
    try:
        serialized = json.dumps(payload, default=_json_default)
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_events
                    (session_id, stage_name, event_type, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (session_id, stage_name, event_type, serialized),
            )
    except Exception:
        logger.exception("Failed to record agent_event; continuing")
```

- [ ] **Step 2.4: Run tests to verify pass**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_telemetry.py::TestRecordEvent -v
```

Expected: 4 pass.

- [ ] **Step 2.5: Add auditor read helpers**

Append to `v2/telemetry.py`:

```python
def count_tool_invocations_by_session(session_id: int) -> dict[str, int]:
    """Returns {tool_name: count} for a session's ideation+reflection stages."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT payload->>'tool_name' AS tool_name, COUNT(*) AS n
            FROM agent_events
            WHERE session_id = %s AND event_type = 'tool_invocation'
            GROUP BY 1
            """,
            (session_id,),
        )
        return {r["tool_name"]: r["n"] for r in cur.fetchall()}


def session_summary_line(session_id: int) -> str:
    """One-line human-readable summary; logged at end of each session."""
    counts = count_tool_invocations_by_session(session_id)
    if not counts:
        return f"[telemetry] session={session_id} no_tool_events"
    return f"[telemetry] session={session_id} tools={counts}"
```

The auditor checks in Task 7 use richer SQL directly — these helpers exist for `v2/session.py`'s end-of-session log line, nothing more.

- [ ] **Step 2.6: Test the helpers**

Append to `tests/v2/test_telemetry.py`:

```python
class TestCountToolInvocations:
    def test_groups_by_tool_name(self):
        from v2.telemetry import count_tool_invocations_by_session
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"tool_name": "get_recent_playbooks", "n": 1},
            {"tool_name": "write_playbook", "n": 1},
        ]
        with patch("v2.telemetry.get_cursor") as gc:
            gc.return_value.__enter__.return_value = cur
            counts = count_tool_invocations_by_session(42)
        assert counts == {"get_recent_playbooks": 1, "write_playbook": 1}


class TestSessionSummaryLine:
    def test_includes_session_id_and_tool_counts(self):
        from v2.telemetry import session_summary_line
        with patch("v2.telemetry.count_tool_invocations_by_session",
                   return_value={"get_session_summary": 1}):
            line = session_summary_line(7)
        assert "session=7" in line
        assert "get_session_summary" in line

    def test_handles_empty_session(self):
        from v2.telemetry import session_summary_line
        with patch("v2.telemetry.count_tool_invocations_by_session", return_value={}):
            line = session_summary_line(99)
        assert "session=99" in line
        assert "no_tool_events" in line
```

Run: `docker compose run --rm --no-deps trading python -m pytest tests/v2/test_telemetry.py -v` → 7 pass.

---

## Task 3: Wire `tool_invocation` events into `run_agentic_loop`

**Files:**
- Modify: `v2/claude_client.py:272-443`
- Modify: `v2/ideation_claude.py:219`
- Modify: `v2/strategy.py:545`
- Modify: `v2/session.py` — thread `session_id` to ideation + reflection callers if not already
- Modify: `tests/v2/test_claude_client.py`
- Modify: `tests/v2/test_ideation_claude.py`
- Modify: `tests/v2/test_strategy.py`

### Signature change

```python
def run_agentic_loop(
    client, model, system, initial_message, tools, tool_handlers,
    max_turns: int = 20,
    session_id: int | None = None,
    stage_name: str | None = None,
) -> AgenticLoopResult:
```

`session_id=None` propagates through `record_event`'s no-op behavior, so existing tests that don't pass these args continue to work.

### Event emission point

After each tool dispatch in the loop (around `v2/claude_client.py:405-416`), emit:

```python
record_event(
    session_id=session_id,
    stage_name=stage_name or "unknown",
    event_type="tool_invocation",
    payload={
        "tool_name": tool_name,
        "args": tool_input if isinstance(tool_input, dict) else {"_raw": str(tool_input)},
        "success": not result.is_error,
        "error": (result.content if result.is_error else None),
    },
)
```

Capture `time.monotonic()` before and after the handler call to add `duration_ms`.

- [ ] **Step 3.1: Failing test in `tests/v2/test_claude_client.py`**

Find the existing tool-dispatch test class. Append:

```python
class TestRunAgenticLoopTelemetry:
    def test_emits_tool_invocation_event_on_success(self, monkeypatch):
        from v2.claude_client import run_agentic_loop
        recorded = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kw: recorded.append(kw),
        )
        # Use the existing fake-client fixture pattern in this file —
        # configure two-turn response: tool_use → end_turn.
        # ... fixture setup mirroring existing tests ...
        run_agentic_loop(
            client=fake_client, model="m", system="s",
            initial_message="hi", tools=[fake_tool],
            tool_handlers={"my_tool": lambda **k: "ok"},
            session_id=99, stage_name="ideation",
        )
        assert len(recorded) == 1
        assert recorded[0]["session_id"] == 99
        assert recorded[0]["stage_name"] == "ideation"
        assert recorded[0]["event_type"] == "tool_invocation"
        assert recorded[0]["payload"]["tool_name"] == "my_tool"
        assert recorded[0]["payload"]["success"] is True

    def test_emits_tool_invocation_event_on_handler_error(self, monkeypatch):
        # Same setup but handler raises. Expect success=False, error captured.
        ...

    def test_no_event_when_session_id_omitted(self, monkeypatch):
        # session_id defaults to None → record_event called with None,
        # which is itself a no-op. Just verify the call still happens.
        ...
```

Use existing `fake_client` / `_call_with_retry`-mocked fixtures in this file — don't invent a new mock harness.

- [ ] **Step 3.2: Run to verify failure**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_claude_client.py::TestRunAgenticLoopTelemetry -v
```

Expected: 3 fail with `ImportError` or `AttributeError`.

- [ ] **Step 3.3: Wire into `run_agentic_loop`**

In `v2/claude_client.py`:

1. Top of file: `from .telemetry import record_event` and `import time`.
2. Extend `run_agentic_loop` signature (line 272) with `session_id` and `stage_name` kwargs.
3. In the dispatch loop, wrap the handler call with timing and emit on each path:

```python
                started = time.monotonic()
                handler = tool_handlers.get(tool_name)
                if handler is None:
                    result = ToolResult(tool_use_id=tool_use_id,
                                        content=f"Error: Unknown tool '{tool_name}'",
                                        is_error=True)
                else:
                    try:
                        output = handler(**tool_input)
                        result = ToolResult(tool_use_id=tool_use_id, content=str(output))
                    except Exception as e:
                        logger.exception(f"Tool {tool_name} failed")
                        result = ToolResult(tool_use_id=tool_use_id,
                                            content=f"Error: {e}",
                                            is_error=True)
                duration_ms = int((time.monotonic() - started) * 1000)
                record_event(
                    session_id=session_id,
                    stage_name=stage_name or "unknown",
                    event_type="tool_invocation",
                    payload={
                        "tool_name": tool_name,
                        "args": tool_input if isinstance(tool_input, dict) else {"_raw": str(tool_input)},
                        "success": not result.is_error,
                        "error": (result.content if result.is_error else None),
                        "duration_ms": duration_ms,
                    },
                )
```

- [ ] **Step 3.4: Tests pass**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_claude_client.py -v
```

Expected: existing tests pass (signature change is backward-compatible), 3 new pass.

- [ ] **Step 3.5: Thread `session_id` through ideation**

`v2/ideation_claude.py:219` already inside a function that receives `session_id` from `v2/session.py`'s ideation stage. Locate the `run_agentic_loop(...)` call and add `session_id=session_id, stage_name="ideation"` to the kwargs.

If `session_id` is not in scope at the call site, add it to the function signature and update the caller in `v2/session.py`.

Add a test in `tests/v2/test_ideation_claude.py`:

```python
class TestIdeationStageTelemetryWiring:
    def test_passes_session_id_and_stage_name_to_loop(self, monkeypatch):
        captured = {}
        def fake_loop(**kw):
            captured.update(kw)
            return MagicMock(messages=[], turns_used=0,
                             stop_reason="end_turn",
                             input_tokens=0, output_tokens=0,
                             cache_creation_input_tokens=0, cache_read_input_tokens=0)
        monkeypatch.setattr("v2.ideation_claude.run_agentic_loop", fake_loop)
        # ... call run_ideation_stage with session_id=42 ...
        assert captured.get("session_id") == 42
        assert captured.get("stage_name") == "ideation"
```

- [ ] **Step 3.6: Thread `session_id` through reflection**

`v2/strategy.py:545` — same change, `stage_name="reflection"`. Mirror the test in `tests/v2/test_strategy.py`.

- [ ] **Step 3.7: Run full v2 suite**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/ -q --ignore=tests/v2/test_session.py
```

Expected: ≥1118 passed (1112 baseline + ~6 new).

---

## Task 4: Wire `evidence_shown` events for round-trip data

**Files:**
- Modify: `v2/strategy.py:275-360, :545`
- Modify: `tests/v2/test_strategy.py`

`tool_get_session_summary` currently calls `analyze_round_trips(...)` and embeds the result in its return text. We need to also persist that result keyed to the session. Cleanest approach: closure-bind a session-aware wrapper at handler-registration time.

- [ ] **Step 4.1: Failing test**

Append to `tests/v2/test_strategy.py`:

```python
class TestEvidenceShownEvent:
    def test_session_summary_wrapper_emits_event_for_round_trips(self, mock_db):
        from v2.strategy import tool_get_session_summary_with_telemetry
        from v2.patterns import RoundTrip
        from datetime import date

        with patch("v2.strategy.analyze_round_trips") as mock_rt, \
             patch("v2.strategy.record_event") as mock_rec:
            mock_rt.return_value = [
                RoundTrip(ticker="GOOGL", pair_count=7,
                          first_date=date(2026, 4, 17), last_date=date(2026, 5, 8)),
            ]
            tool_get_session_summary_with_telemetry(days=30, session_id=42)
        # First call must be tool_invocation (from run_agentic_loop, not here)
        # OR evidence_shown (from inside this wrapper). Asserting evidence_shown
        # was emitted at least once with the right shape.
        evidence_calls = [
            c for c in mock_rec.call_args_list
            if c.kwargs.get("event_type") == "evidence_shown"
        ]
        assert len(evidence_calls) == 1
        kwargs = evidence_calls[0].kwargs
        assert kwargs["session_id"] == 42
        assert kwargs["payload"]["evidence_kind"] == "round_trips"
        assert kwargs["payload"]["items"][0]["ticker"] == "GOOGL"
        assert kwargs["payload"]["items"][0]["pair_count"] == 7

    def test_no_event_when_round_trips_empty(self):
        from v2.strategy import tool_get_session_summary_with_telemetry
        with patch("v2.strategy.analyze_round_trips", return_value=[]), \
             patch("v2.strategy.record_event") as mock_rec:
            tool_get_session_summary_with_telemetry(days=30, session_id=1)
        # Emit even on empty so the auditor can distinguish "saw nothing"
        # from "didn't run". This is intentional.
        evidence_calls = [
            c for c in mock_rec.call_args_list
            if c.kwargs.get("event_type") == "evidence_shown"
        ]
        assert len(evidence_calls) == 1
        assert evidence_calls[0].kwargs["payload"]["items"] == []

    def test_skips_event_when_session_id_none(self):
        from v2.strategy import tool_get_session_summary_with_telemetry
        with patch("v2.strategy.analyze_round_trips", return_value=[]), \
             patch("v2.strategy.record_event") as mock_rec:
            tool_get_session_summary_with_telemetry(days=30, session_id=None)
        # record_event is called but with session_id=None (no-op).
        evidence_calls = [c for c in mock_rec.call_args_list
                          if c.kwargs.get("event_type") == "evidence_shown"]
        assert len(evidence_calls) == 1
        assert evidence_calls[0].kwargs["session_id"] is None
```

The "emit on empty" decision matters for the `REFLECTION_INERT_ON_ROUND_TRIPS` check — we need to distinguish "reflection ran but saw nothing" from "reflection didn't run."

- [ ] **Step 4.2: Run to verify failure**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_strategy.py::TestEvidenceShownEvent -v
```

Expected: 3 fail with `ImportError`.

- [ ] **Step 4.3: Implement the wrapper**

In `v2/strategy.py`:

1. Add: `from .telemetry import record_event`
2. After `tool_get_session_summary` definition (around line 360), add:

```python
def tool_get_session_summary_with_telemetry(days: int = 30, *, session_id: int | None = None) -> str:
    """Wrapper around tool_get_session_summary that also persists the
    round-trip evidence reflection was shown, keyed to the active session.

    Computes round_trips twice (once inside the inner call, once here) — cheap
    enough not to optimize until we measure.
    """
    output = tool_get_session_summary(days=days)
    round_trips = analyze_round_trips(days=30, gap_days=7, min_pairs=2)
    record_event(
        session_id=session_id,
        stage_name="reflection",
        event_type="evidence_shown",
        payload={
            "evidence_kind": "round_trips",
            "items": [
                {"ticker": rt.ticker, "pair_count": rt.pair_count,
                 "first_date": rt.first_date, "last_date": rt.last_date}
                for rt in round_trips
            ],
            "summary": {"n_tickers": len(round_trips)},
        },
    )
    return output
```

3. At the reflection-stage agentic-loop call (line 545 area), bind a session-specific handler dict:

```python
from functools import partial

session_handlers = {
    **STRATEGY_TOOL_HANDLERS,
    "get_session_summary": partial(tool_get_session_summary_with_telemetry, session_id=session_id),
}

result = run_agentic_loop(
    # ... existing args ...
    tool_handlers=session_handlers,
    session_id=session_id,
    stage_name="reflection",
)
```

- [ ] **Step 4.4: Tests pass**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_strategy.py -v
```

Expected: all strategy tests pass, 3 new pass.

---

## Task 5: Wire `risk_block` events into trader

**Files:**
- Modify: `v2/trader.py:~520` (where `check_sector_cap_for_buy` returns a breach)
- Modify: `tests/v2/test_trader.py`

- [ ] **Step 5.1: Locate the breach handler**

```bash
grep -n "check_sector_cap_for_buy\|REJECTED: sector\|invalid" /home/jay/dev/algo/v2/trader.py | head -20
```

Confirm the file/line where the rejection branch sets `decision.action = "invalid"` or similar.

- [ ] **Step 5.2: Failing test**

Append to `tests/v2/test_trader.py` (or create a new class):

```python
class TestRiskBlockTelemetry:
    def test_emits_risk_block_event_on_sector_cap_breach(self, monkeypatch):
        from v2.trader import _prepare_decision  # or whichever function holds the breach branch
        recorded = []
        monkeypatch.setattr(
            "v2.trader.record_event",
            lambda **kw: recorded.append(kw),
        )
        # Construct a buy that breaches the sector cap. Use existing test
        # fixtures from this file — there are sector-cap-hard-gate tests
        # already at TestSectorCapHardGate that build a breach scenario.
        # ... fixture setup mirroring existing tests ...
        _prepare_decision(action_breaching_sector_cap, session_id=42, ...)
        block_events = [r for r in recorded if r["event_type"] == "risk_block"]
        assert len(block_events) == 1
        payload = block_events[0]["payload"]
        assert payload["ticker"] == "<expected_ticker>"
        assert payload["sector"] == "<expected_sector>"
        assert payload["proposed_qty"] == <expected>
        assert payload["cap"] > 0

    def test_no_event_when_sector_cap_passes(self, monkeypatch):
        # Existing under-cap buy test path. Assert no risk_block event.
        ...
```

Read existing `TestSectorCapHardGate` in `tests/v2/test_trader.py` to copy its breach-fixture shape rather than inventing new mocks.

- [ ] **Step 5.3: Run to verify failure**

Expected: tests fail because no event is emitted.

- [ ] **Step 5.4: Implement**

In `v2/trader.py`:

1. Add: `from .telemetry import record_event`
2. At the `check_sector_cap_for_buy` breach branch (line 520 area), after marking the decision invalid, emit:

```python
        if breach:
            decision.action = "invalid"
            decision.reasoning = f"[REJECTED: sector cap] {breach}"
            record_event(
                session_id=session_id,
                stage_name="trading",
                event_type="risk_block",
                payload={
                    "ticker": decision.ticker,
                    "sector": breach_sector,  # parsed from breach message or available locally
                    "proposed_qty": decision.quantity,
                    "price": decision.price,
                    "sector_pct_after": breach_pct,  # if computed; else omit
                    "cap": MAX_SECTOR_PCT,
                    "reason_text": breach,
                },
            )
```

`session_id` must be in scope at this call site. If `_prepare_decision` doesn't currently receive it, thread it through the function signature from `_execute_decisions` (which knows the active session).

- [ ] **Step 5.5: Tests pass**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_trader.py -v
```

Expected: existing trader tests pass, 2 new pass.

---

## Task 6: Add 6 new auditor checks

**Files:**
- Modify: `v2/audit.py` — append 6 new check functions + register in `CHECKS`
- Modify: `tests/v2/test_audit.py` — 6 new test classes

Each check follows the existing `def check_*(cur) -> list[Finding]` shape. Severity rules and thresholds are documented in the table at the top of this plan; below are the SQL skeletons.

### Check 1: `STRATEGIST_NOT_USING_REVERSAL_TOOL`

```python
REVERSAL_LOOKBACK_SESSIONS = 3

def check_strategist_using_reversal_tool(cur) -> list[Finding]:
    """Strategist has access to get_recent_playbooks (added 2026-05-08) to
    justify reversals. Flag if 3 consecutive ideation sessions had reversals
    (round-trips in the window) but never called the tool."""
    cur.execute("""
        WITH recent_ideation_sessions AS (
            SELECT s.id AS session_id, s.session_date
            FROM sessions s
            JOIN session_stages st ON st.session_id = s.id
                                  AND st.stage_name = 'ideation'
                                  AND st.status = 'completed'
            WHERE s.session_date > CURRENT_DATE - 14
            ORDER BY s.session_date DESC
            LIMIT %s
        ),
        had_round_trips AS (
            SELECT ris.session_id
            FROM recent_ideation_sessions ris
            JOIN agent_events e ON e.session_id = ris.session_id
                               AND e.event_type = 'evidence_shown'
                               AND e.payload->>'evidence_kind' = 'round_trips'
                               AND jsonb_array_length(e.payload->'items') > 0
        ),
        called_tool AS (
            SELECT DISTINCT e.session_id
            FROM agent_events e
            JOIN recent_ideation_sessions ris ON ris.session_id = e.session_id
            WHERE e.event_type = 'tool_invocation'
              AND e.stage_name = 'ideation'
              AND e.payload->>'tool_name' = 'get_recent_playbooks'
        )
        SELECT ris.session_id, ris.session_date
        FROM recent_ideation_sessions ris
        WHERE ris.session_id IN (SELECT session_id FROM had_round_trips)
          AND ris.session_id NOT IN (SELECT session_id FROM called_tool)
    """, (REVERSAL_LOOKBACK_SESSIONS,))
    rows = cur.fetchall()
    if len(rows) < REVERSAL_LOOKBACK_SESSIONS:
        return []
    return [Finding(
        check_code="STRATEGIST_NOT_USING_REVERSAL_TOOL",
        tier=3, severity="warn",
        title=f"{len(rows)} consecutive ideation sessions had round-trip evidence but did not call get_recent_playbooks",
        body=("Phase A wired `get_recent_playbooks` so the strategist could "
              "justify reversals. The tool isn't being invoked despite "
              "round-trip evidence being shown to reflection. The Reversal "
              "Justification rule may be aspirational — investigate prompt "
              "or tool description."),
        affected_count=len(rows),
        evidence={"session_ids": [r["session_id"] for r in rows],
                  "session_dates": [r["session_date"].isoformat() for r in rows]},
        auto_fix=None,
    )]
```

### Check 2: `REFLECTION_INERT_ON_ROUND_TRIPS`

```python
INERT_LOOKBACK_SESSIONS = 5

def check_reflection_inert_on_round_trips(cur) -> list[Finding]:
    """Reflection saw round-trip evidence in N sessions but produced no
    propose_rule or retire_rule tool calls. Flags whether Phase B evidence
    is load-bearing."""
    cur.execute("""
        WITH recent AS (
            SELECT s.id AS session_id
            FROM sessions s
            JOIN session_stages st ON st.session_id = s.id
                                  AND st.stage_name = 'strategy'
                                  AND st.status = 'completed'
            WHERE s.session_date > CURRENT_DATE - 21
            ORDER BY s.session_date DESC
            LIMIT %s
        ),
        had_round_trips AS (
            SELECT r.session_id
            FROM recent r
            JOIN agent_events e ON e.session_id = r.session_id
                               AND e.event_type = 'evidence_shown'
                               AND e.payload->>'evidence_kind' = 'round_trips'
                               AND jsonb_array_length(e.payload->'items') > 0
        ),
        proposed_or_retired AS (
            SELECT DISTINCT r.session_id
            FROM recent r
            JOIN agent_events e ON e.session_id = r.session_id
                               AND e.event_type = 'tool_invocation'
                               AND e.stage_name = 'reflection'
                               AND e.payload->>'tool_name' IN ('propose_rule', 'retire_rule')
        )
        SELECT session_id FROM had_round_trips
        WHERE session_id NOT IN (SELECT session_id FROM proposed_or_retired)
    """, (INERT_LOOKBACK_SESSIONS,))
    inert = [r["session_id"] for r in cur.fetchall()]
    if len(inert) < INERT_LOOKBACK_SESSIONS:
        return []
    return [Finding(
        check_code="REFLECTION_INERT_ON_ROUND_TRIPS",
        tier=3, severity="warn",
        title=f"Reflection saw round-trip evidence in {len(inert)} sessions but proposed/retired no rules",
        body=("Phase B surfaces round-trip evidence into reflection's session "
              "summary. If reflection never proposes or retires rules in "
              "response, the loop's not closing. Manual rule review or prompt "
              "change is warranted."),
        affected_count=len(inert),
        evidence={"session_ids": inert},
        auto_fix=None,
    )]
```

### Check 3: `TOOL_ERROR_RATE`

```python
TOOL_ERROR_WARN_PCT = 0.20
TOOL_ERROR_CRITICAL_PCT = 0.50
TOOL_ERROR_MIN_N = 5  # ignore tools with too few invocations to be meaningful

def check_tool_error_rate(cur) -> list[Finding]:
    """Per-tool error rate over last 7 days. Flags tools that are silently
    failing in production agentic loops."""
    cur.execute("""
        SELECT payload->>'tool_name' AS tool_name,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE (payload->>'success')::boolean = false) AS errors
        FROM agent_events
        WHERE event_type = 'tool_invocation'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
    """, (TOOL_ERROR_MIN_N,))
    flagged = []
    for r in cur.fetchall():
        rate = r["errors"] / r["total"]
        if rate >= TOOL_ERROR_WARN_PCT:
            flagged.append({"tool_name": r["tool_name"],
                            "total": r["total"], "errors": r["errors"],
                            "rate": round(rate, 3)})
    if not flagged:
        return []
    worst = max(f["rate"] for f in flagged)
    sev = "critical" if worst >= TOOL_ERROR_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="TOOL_ERROR_RATE",
        tier=3, severity=sev,
        title=f"{len(flagged)} tool(s) with error rate >= {int(TOOL_ERROR_WARN_PCT*100)}% in last 7d",
        body="Per-tool error counts in agentic loops. Investigate handler exceptions.",
        affected_count=len(flagged),
        evidence={"tools": flagged},
        auto_fix=None,
    )]
```

### Check 4: `RISK_BLOCK_HOTSPOT`

```python
HOTSPOT_MIN_BLOCKS = 3
HOTSPOT_WINDOW_DAYS = 7

def check_risk_block_hotspot(cur) -> list[Finding]:
    """Same ticker hits the sector-cap gate ≥3 times in a week — strategist
    repeatedly proposing buys the gate keeps rejecting."""
    cur.execute("""
        SELECT payload->>'ticker' AS ticker,
               COUNT(*) AS n,
               MIN(occurred_at) AS first_block,
               MAX(occurred_at) AS last_block
        FROM agent_events
        WHERE event_type = 'risk_block'
          AND occurred_at > now() - interval '%s days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        ORDER BY n DESC
    """, (HOTSPOT_WINDOW_DAYS, HOTSPOT_MIN_BLOCKS))
    rows = cur.fetchall()
    if not rows:
        return []
    return [Finding(
        check_code="RISK_BLOCK_HOTSPOT",
        tier=3, severity="warn",
        title=f"{len(rows)} ticker(s) blocked by sector-cap gate >= {HOTSPOT_MIN_BLOCKS} times in {HOTSPOT_WINDOW_DAYS}d",
        body=("Strategist proposing buys that keep failing the sector-cap "
              "gate on the same ticker. Either the strategist isn't seeing "
              "the rejection or it's ignoring it."),
        affected_count=len(rows),
        evidence={"hotspots": [
            {"ticker": r["ticker"], "n": r["n"],
             "first_block": r["first_block"].isoformat(),
             "last_block": r["last_block"].isoformat()}
            for r in rows
        ]},
        auto_fix=None,
    )]
```

### Check 5: `RISK_BLOCK_BURST`

```python
BURST_MIN_BLOCKS = 5

def check_risk_block_burst(cur) -> list[Finding]:
    """≥5 sector-cap rejections on a single date — one session generated a
    wave of cap-blocked buys."""
    cur.execute("""
        SELECT occurred_at::date AS d,
               COUNT(*) AS n,
               array_agg(DISTINCT payload->>'ticker') AS tickers
        FROM agent_events
        WHERE event_type = 'risk_block'
          AND occurred_at > now() - interval '14 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        ORDER BY 1 DESC
    """, (BURST_MIN_BLOCKS,))
    rows = cur.fetchall()
    if not rows:
        return []
    return [Finding(
        check_code="RISK_BLOCK_BURST",
        tier=3, severity="warn",
        title=f"{len(rows)} day(s) with >= {BURST_MIN_BLOCKS} sector-cap rejections",
        body="A single session generated many cap-blocked buys; strategist "
             "may be over-allocating to a sector.",
        affected_count=len(rows),
        evidence={"bursts": [
            {"date": r["d"].isoformat(), "n": r["n"], "tickers": list(r["tickers"])}
            for r in rows
        ]},
        auto_fix=None,
    )]
```

### Check 6: `IDEATION_TOOL_DROUGHT`

```python
EXPECTED_IDEATION_TOOLS = {
    "get_attribution", "write_playbook", "get_theses",
    # `get_recent_playbooks` excluded — covered by STRATEGIST_NOT_USING_REVERSAL_TOOL.
}

def check_ideation_tool_drought(cur) -> list[Finding]:
    """Any tool in the expected ideation toolset has 0 calls across the last
    7 ideation sessions. Detects accidental tool removal or prompt drift."""
    cur.execute("""
        WITH recent AS (
            SELECT s.id AS session_id
            FROM sessions s
            JOIN session_stages st ON st.session_id = s.id
                                  AND st.stage_name = 'ideation'
                                  AND st.status = 'completed'
            WHERE s.session_date > CURRENT_DATE - 14
            ORDER BY s.session_date DESC
            LIMIT 7
        )
        SELECT DISTINCT e.payload->>'tool_name' AS tool_name
        FROM agent_events e
        JOIN recent r ON r.session_id = e.session_id
        WHERE e.event_type = 'tool_invocation'
          AND e.stage_name = 'ideation'
    """)
    seen_tools = {r["tool_name"] for r in cur.fetchall()}
    missing = sorted(EXPECTED_IDEATION_TOOLS - seen_tools)
    if not missing:
        return []
    return [Finding(
        check_code="IDEATION_TOOL_DROUGHT",
        tier=3, severity="warn",
        title=f"{len(missing)} expected ideation tool(s) unused in last 7 sessions",
        body=("Strategist hasn't invoked these tools in the recent window. "
              "Either prompt drift or a real disuse signal — verify the "
              "tool is still in TOOL_DEFINITIONS and the prompt mentions it."),
        affected_count=len(missing),
        evidence={"unused_tools": missing,
                  "expected_set": sorted(EXPECTED_IDEATION_TOOLS)},
        auto_fix=None,
    )]
```

- [ ] **Step 6.1: Failing tests for all 6 checks**

In `tests/v2/test_audit.py`, add 6 test classes (one per check). Each class has at minimum:
- One test with no events → empty findings list
- One test with events that should trip → 1 finding with expected `check_code`
- One test with events below threshold → empty findings list

Use the existing `mock_db` / `mock_cursor` patterns from `tests/v2/conftest.py:75-104`. Mock cursor `fetchall` and `fetchone` to return the row shapes the check expects.

- [ ] **Step 6.2: Run to verify failures**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_audit.py -v
```

Expected: ~18 new tests fail with `AttributeError` (functions don't exist yet).

- [ ] **Step 6.3: Implement the 6 checks**

Append all 6 functions to `v2/audit.py` after `check_theses_missing_signal_refs` (around line 527). Add to `CHECKS` list (line 727):

```python
CHECKS: list = [
    "check_orphan_fks",
    "check_missing_backfill",
    "check_invalid_attribution_categories",
    "check_snapshot_gaps",
    "check_decision_equity_drift",
    "check_attribution_category_coverage",
    "check_stage_failure_rate",
    "check_cost_trend",
    "check_decisions_missing_signal_refs",
    "check_theses_missing_signal_refs",
    "check_strategist_using_reversal_tool",
    "check_reflection_inert_on_round_trips",
    "check_tool_error_rate",
    "check_risk_block_hotspot",
    "check_risk_block_burst",
    "check_ideation_tool_drought",
    "check_rule_judgment",  # keep last (LLM call)
]
```

- [ ] **Step 6.4: Tests pass**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_audit.py -v
```

Expected: all audit tests pass.

- [ ] **Step 6.5: Run full v2 suite**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/ -q --ignore=tests/v2/test_session.py
```

Expected: ≥1140 passed.

---

## Task 7: Session-end summary log

**Files:**
- Modify: `v2/session.py`
- Modify: `tests/v2/test_session.py`

- [ ] **Step 7.1: Add the log line**

At end of `run_session` in `v2/session.py`:

```python
from .telemetry import session_summary_line
logger.info(session_summary_line(session_id))
```

- [ ] **Step 7.2: Test that the log line emits**

In `tests/v2/test_session.py`, add or extend a happy-path test asserting `session_summary_line(session_id)` is called once at end-of-session. Mock the underlying `count_tool_invocations_by_session` call.

- [ ] **Step 7.3: Run session tests**

Pre-existing 10 twitter/bluesky/dashboard failures stay unchanged.

---

## Task 8: Paper validation, then prod migration

**Files:** None modified. Validation only.

- [ ] **Step 8.1: Apply migration to paper DB** (already done in Task 1.2)

- [ ] **Step 8.2: Run paper session in dry-run**

```bash
task paper:session:dry-run
```

- [ ] **Step 8.3: Inspect events**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper \
  psql -U algo -d trading -c "
    SELECT event_type, stage_name, COUNT(*) AS n
    FROM agent_events
    GROUP BY 1, 2
    ORDER BY 1, 2;
  "
```

Expected: rows for `tool_invocation` (ideation, reflection), `evidence_shown` (reflection), and `risk_block` (trading) if any sector breaches occurred.

- [ ] **Step 8.4: Run the paper auditor and confirm new checks fire (or correctly stay quiet)**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T trading-paper \
  python -m v2.audit
```

Expected: no new critical findings if telemetry is producing as expected (since one session won't trip any of the multi-session thresholds). Verify `STRATEGIST_NOT_USING_REVERSAL_TOOL` etc. don't false-fire.

- [ ] **Step 8.5: Apply migration to prod**

```bash
docker compose exec -T db psql -U algo -d trading < db/init/026_agent_events.sql
```

- [ ] **Step 8.6: Wait for next prod session**

After the next prod daily session runs, repeat 8.3-8.4 against prod. Capture:
- Was `get_recent_playbooks` invoked? (`payload->>'tool_name' = 'get_recent_playbooks'` count)
- Did reflection see round-trips? (`evidence_shown` count and `items` length)
- Were there any risk blocks? (`risk_block` count)

These three answers + 5-7 more sessions of telemetry are the inputs for deciding manual retirement of Rule #27 vs. trusting the loop.

---

## Verification checklist

Before declaring this plan complete:

- [ ] `db/init/026_agent_events.sql` applied to paper and prod
- [ ] Full v2 test suite passes (`docker compose run --rm --no-deps trading python -m pytest tests/v2/ -q --ignore=tests/v2/test_session.py`)
- [ ] One paper session has populated `agent_events` with all 3 event types (or `risk_block` legitimately empty)
- [ ] `[telemetry]` log line appears in `trading-paper` logs at session end
- [ ] Paper auditor run completes without new false-positive findings
- [ ] One prod session has produced events; the 6 new auditor checks have run against prod data
- [ ] Decision recorded (in a follow-up note or memo): manually retire Rule #27, or wait for the reflection loop, or extend instrumentation further (rule lifecycle, prompt versions) based on what telemetry actually shows

---

## Out of scope (deferred)

These were considered and explicitly cut. They become justifiable only if telemetry data shows we need them.

- **Rule citation persistence** — `check_rule_judgment` already regex-extracts citations on demand. No new check would benefit.
- **Prompt versioning / hash** — only useful for A/B testing, which isn't happening.
- **Stage-level Claude metadata** (stop_reason, max_turns hit, cache hit ratio) — better as columns on `session_stages` than as events; defer until a check actually wants them.
- **Decision-to-rule mapping table** — same as rule citation persistence; the inline regex covers it.
- **Order outcome events** — already in `decisions.outcome_*`.
- **Theses lifecycle events** — already in `theses.status`/`updated_at`.
