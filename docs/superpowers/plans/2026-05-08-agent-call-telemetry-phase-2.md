# Agent Call Telemetry — Phase 2 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Prerequisite:** `2026-05-08-flip-flop-telemetry.md` (Phase 1) MUST be merged and run in prod for at least one full session before this plan starts. Phase 2 reuses Phase 1's `agent_events` table and `record_event()` helper.

**Goal:** Close the remaining agent-call observability gaps that Phase 1 left open: per-call response metadata (`stop_reason`, `duration_ms`, per-call tokens), agentic-loop recovery events (`max_tokens` retry, context-length aggressive prune), and the executor's raw JSON response (with a schema-drift canary that catches when the LLM emits fields we silently drop). After Phase 2, the auditor can answer *"is the executor being truncated?"* and *"is the LLM trying to tell us something we're not parsing?"* — both currently invisible.

**Architecture:** Single-place instrumentation in `v2/claude_client.py` (`_call_with_retry` and `run_agentic_loop` recovery branches) plus a tight wrapper around the executor's parse step in `v2/agent.py`. Three new event types in the existing `agent_events` table — **zero schema migration**. Five new auditor checks consume them. No prompt changes, no business-logic changes.

**Tech Stack:** Same as Phase 1 — PostgreSQL 16, psycopg2 raw SQL via `get_cursor()`, Python 3.x, pytest with `mock_db`/`mock_cursor` fixtures from `tests/v2/conftest.py:75-104`.

---

## What Phase 1 leaves uncaptured

After Phase 1 ships, `agent_events` contains:
- `tool_invocation` — every tool dispatch in `run_agentic_loop` (ideation + reflection)
- `evidence_shown` — round-trip evidence shown to reflection
- `risk_block` — sector-cap rejections in trader

What's still invisible:

| Gap | Why it matters | Today's coverage |
|---|---|---|
| Per-call `stop_reason` (`end_turn` / `max_tokens` / `max_turns` / `tool_use`) | Catches "executor was truncated 3× this week" — silent quality regression | `logger.info` line in `v2/agent.py:231` only |
| Per-call `duration_ms` | Latency drift detection; correlate cost spikes to specific calls | Nowhere |
| Per-call token counts | Aggregated in `session_stages.*_tokens`; can't attribute spikes to specific calls | Nowhere per-call |
| Agentic-loop `max_tokens` recovery firing | Known degradation mode (`v2/claude_client.py:342`); silent today | `logger.warning` only |
| Agentic-loop context-length aggressive prune firing | Known degradation mode (`v2/claude_client.py:324`); silent today | `logger.warning` only |
| Executor raw JSON response | If LLM adds new fields we don't parse, we never know | Discarded after `json.loads` |
| Executor parse failures | Raw text in exception message but never persisted | Exception only |
| Executor truncation (`max_tokens`) | Raises but loses partial response (`v2/agent.py:238-242`) | Exception only |

The first 5 gaps are universal across every Claude call. The last 3 are executor-specific.

---

## Four new event types

| event_type | Fired by | Payload schema (illustrative) |
|---|---|---|
| `agent_call` | `_call_with_retry` after every Claude API call | `{"model": str, "purpose": str, "stop_reason": str, "duration_ms": int, "input_tokens": int, "output_tokens": int, "cache_read_tokens": int, "cache_creation_tokens": int, "success": bool, "error": str\|null}` |
| `loop_recovery` | `run_agentic_loop` recovery branches | `{"reason": "max_tokens"\|"context_length", "turn": int, "model": str}` |
| `loop_completion` | `run_agentic_loop` once at end (success or max_turns) | `{"stop_reason": str, "turns_used": int, "model": str, "input_tokens": int, "output_tokens": int}` |
| `executor_response` | After parsing (or failing to parse) the executor's response in `v2/agent.py` | `{"parse_succeeded": bool, "stop_reason": str, "decision_count": int, "thesis_invalidation_count": int, "unknown_top_level_keys": list, "unknown_decision_keys": list, "raw_response_text_truncated": str (first 4KB), "error": str\|null}` |

`raw_response_text_truncated` is bounded at 4KB. On parse failure we capture the full raw text via the `error` field. Always-on full-transcript capture is **explicitly out of scope** (see deferred section).

`loop_completion` is the loop-level twin of `agent_call`: `agent_call` fires per API call (per turn), `loop_completion` fires once when the whole agentic loop terminates. It's the only way to distinguish "loop hit max_turns" from "loop ended cleanly with a final tool_use" — both leave per-call `stop_reason='tool_use'` on the last `agent_call`, indistinguishable without loop-level state.

---

## Eight new auditor checks unlocked

| check_code | Tier | Severity rule | What it asks | Source |
|---|---|---|---|---|
| `EXECUTOR_TRUNCATION_RATE` | 3 | warn ≥10%, critical ≥25% in 14d | Executor responses hitting `max_tokens` — input too large or model under-budgeted | `agent_call` (purpose='executor') |
| `EXECUTOR_SCHEMA_DRIFT` | 3 | warn on any unknown key seen ≥3x in 7d | LLM emitting fields we silently drop — schema drift canary | `executor_response` |
| `EXECUTOR_PARSE_FAILURE_RATE` | 3 | warn ≥5%, critical ≥15% in 14d | Executor responses failing JSON parse for reasons OTHER than truncation — malformed output, fenced-code edge cases | `executor_response` |
| `CLASSIFIER_ERROR_RATE` | 3 | warn ≥10%, critical ≥25% in 7d | News/macro classifier silently failing on items | `agent_call` (purpose like 'classifier_%') |
| `AGENT_CALL_ERROR_RATE_BY_PURPOSE` | 3 | warn ≥10%, critical ≥25% in 7d, any purpose with N≥10 | Generic catch-all: any Claude call type whose error rate jumps. Surfaces failures in purposes the specific checks (executor/classifier) don't cover | `agent_call` |
| `LOOP_RECOVERY_BURST` | 3 | warn ≥3 events in 7d for same `reason` | Agentic loop hitting recovery often — prompt or input-size regression | `loop_recovery` |
| `LOOP_MAX_TURNS_HIT` | 3 | warn ≥1 in 7d, critical ≥3 in 7d | Agentic loop terminated by max_turns (not end_turn) — strategist or reflection ran out of budget mid-task | `loop_completion` |
| `CACHE_HIT_RATIO_DEGRADATION` | 3 | info if cache_read share for any purpose dropped ≥30 percentage points vs prior 7d | Cache breakpoint placement broken — cost regression that silently doubles bills | `agent_call` |
| `AGENT_CALL_LATENCY_DRIFT` | 3 | info if any purpose's p95 ≥2× prior 7-day window | Latency spike on a specific Claude call type | `agent_call` |

Each defends itself against *"what would the auditor flag, and what would I do about it?"*

The general check (`AGENT_CALL_ERROR_RATE_BY_PURPOSE`) sits alongside the specific one (`CLASSIFIER_ERROR_RATE`) deliberately: the classifier check has a tighter threshold and a classifier-specific remediation path (signal_attribution coverage degrades downstream); the general check catches everything else without forcing every purpose to get its own bespoke check.

---

## Files

**Modify (production):**
- `v2/claude_client.py` — extend `_call_with_retry` signature with `(session_id, stage_name, purpose)` kwargs + emit `agent_call` event; emit `loop_recovery` events from the two recovery branches in `run_agentic_loop`; emit `loop_completion` event at end of `run_agentic_loop` (success and max_turns paths)
- `v2/agent.py:188-260` — pass telemetry kwargs to `_call_with_retry`; emit `executor_response` event after parse (success or failure); compute `unknown_*_keys` against canonical key sets
- `v2/classifier.py:316, 404, 471` — pass telemetry kwargs to each `_call_with_retry`. Three call sites, three `purpose` values (TBD per site role)
- `v2/audit.py` — append 8 new check functions + register in `CHECKS` list

**Modify (tests):**
- `tests/v2/test_claude_client.py` — assert `agent_call` event emitted with correct payload; assert `loop_recovery` events on truncation/context-length triggers; assert `loop_completion` event emitted with correct `stop_reason` on both clean exit and max_turns paths
- `tests/v2/test_agent.py` — assert `executor_response` event emitted on success and on each failure mode (parse error, max_tokens); assert `unknown_*_keys` populated when LLM returns a field not in the canonical set
- `tests/v2/test_classifier.py` — assert `agent_call` events emitted from each classifier call site with the right `purpose`
- `tests/v2/test_audit.py` — 8 new test classes, one per check (no-op, trip, below-threshold)

Total: 4 production files modified, 4 test files extended, ~340 LOC (≈150 production, ≈190 tests). Zero new files. Zero schema changes. Zero new tables.

---

## Task 1: `_call_with_retry` instrumentation + `agent_call` event

**Files:**
- Modify: `v2/claude_client.py`
- Modify: `tests/v2/test_claude_client.py`

### Signature change

```python
def _call_with_retry(
    client: anthropic.Anthropic,
    *,
    model: str,
    max_tokens: int,
    system,
    messages,
    tools=None,
    session_id: int | None = None,
    stage_name: str | None = None,
    purpose: str | None = None,
) -> Message:
```

`session_id=None` propagates through `record_event`'s no-op behavior (Phase 1 contract); existing callers that don't pass these kwargs continue to work unchanged.

### Event emission

Wrap the actual API call (currently around `_call_with_retry`'s body) with timing and emit on every return path (success, after-retry-success, retry-exhausted-failure):

```python
started = time.monotonic()
success = False
error_msg: str | None = None
response: Message | None = None
try:
    response = client.messages.create(model=model, max_tokens=max_tokens,
                                      system=system, messages=messages,
                                      **({"tools": tools} if tools else {}))
    success = True
    return response
except Exception as e:
    error_msg = f"{type(e).__name__}: {str(e)[:500]}"
    raise
finally:
    duration_ms = int((time.monotonic() - started) * 1000)
    payload = {
        "model": model,
        "purpose": purpose or "unknown",
        "duration_ms": duration_ms,
        "success": success,
        "error": error_msg,
    }
    if response is not None:
        payload.update({
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
            "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        })
    record_event(
        session_id=session_id,
        stage_name=stage_name or "unknown",
        event_type="agent_call",
        payload=payload,
    )
```

The current `_call_with_retry` already has retry-on-overload logic — wrap that whole function so retries count as one event with the final outcome, not one per attempt. (If you want per-retry telemetry, that's a follow-up — for now, one event per logical call is the right granularity.)

- [ ] **Step 1.1: Failing tests for `agent_call` event**

In `tests/v2/test_claude_client.py`, add:

```python
class TestCallWithRetryTelemetry:
    def test_emits_agent_call_event_on_success(self, monkeypatch):
        from v2.claude_client import _call_with_retry
        recorded = []
        monkeypatch.setattr("v2.claude_client.record_event",
                            lambda **kw: recorded.append(kw))
        # Use existing fake-client fixture; configure successful response
        # with usage attribute populated.
        _call_with_retry(
            fake_client_returning_success,
            model="claude-haiku-4-5", max_tokens=4096,
            system=[], messages=[{"role": "user", "content": "hi"}],
            session_id=42, stage_name="trading", purpose="executor",
        )
        assert len(recorded) == 1
        p = recorded[0]["payload"]
        assert p["purpose"] == "executor"
        assert p["model"] == "claude-haiku-4-5"
        assert p["success"] is True
        assert "duration_ms" in p
        assert "stop_reason" in p
        assert p["input_tokens"] >= 0

    def test_emits_agent_call_event_on_failure(self, monkeypatch):
        # Fake client raises a non-retriable exception; assert event has
        # success=False, error captured, no stop_reason.
        ...

    def test_no_event_kwargs_default_none(self, monkeypatch):
        # Existing callers that don't pass session_id should still work;
        # record_event is called with session_id=None (no-op).
        ...
```

- [ ] **Step 1.2: Run to verify failure**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_claude_client.py::TestCallWithRetryTelemetry -v
```

Expected: 3 fail.

- [ ] **Step 1.3: Implement instrumentation**

In `v2/claude_client.py`:
1. Add: `import time` (if not already imported) and `from .telemetry import record_event` (if not already from Phase 1).
2. Extend `_call_with_retry` signature with the three new keyword-only kwargs.
3. Wrap the existing call body per the snippet above.

- [ ] **Step 1.4: Tests pass**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/test_claude_client.py -v
```

Expected: existing tests pass (signature change is backward-compatible), 3 new pass.

---

## Task 2: `loop_recovery` + `loop_completion` events in `run_agentic_loop`

**Files:**
- Modify: `v2/claude_client.py:300-370` (the two recovery branches)
- Modify: `tests/v2/test_claude_client.py`

The two known degradation modes in `run_agentic_loop`:
1. **`max_tokens` recovery** (line 342) — response was truncated; we discard the turn and retry once with a concision nudge.
2. **Context-length aggressive prune** (line 324) — `BadRequestError` matching context-length error; aggressively prune message history and retry once.

Both fire **at most once per loop** (bounded counters), so an event per fire is correct cardinality.

- [ ] **Step 2.1: Failing tests**

```python
class TestLoopRecoveryTelemetry:
    def test_emits_loop_recovery_on_max_tokens(self, monkeypatch):
        # Configure fake client to return stop_reason='max_tokens' on first
        # turn, then end_turn on retry. Assert one loop_recovery event with
        # reason='max_tokens'.
        ...

    def test_emits_loop_recovery_on_context_length_error(self, monkeypatch):
        # Configure fake client to raise context-length error first, then
        # succeed. Assert one loop_recovery event with reason='context_length'.
        ...

    def test_no_recovery_events_on_clean_run(self, monkeypatch):
        # Normal end_turn response — assert zero loop_recovery events
        # (tool_invocation events from Phase 1 still fire).
        ...
```

- [ ] **Step 2.2: Run to verify failure**

Expected: 3 fail.

- [ ] **Step 2.3: Implement**

In `v2/claude_client.py:run_agentic_loop`, at each recovery branch (after the `max_tokens_recoveries += 1` and `context_length_recoveries += 1` increments):

```python
                record_event(
                    session_id=session_id,
                    stage_name=stage_name or "unknown",
                    event_type="loop_recovery",
                    payload={
                        "reason": "max_tokens",  # or "context_length"
                        "turn": turn + 1,
                        "model": model,
                    },
                )
```

- [ ] **Step 2.4: Tests pass**

Expected: 3 new pass.

### Loop completion event

`run_agentic_loop` has two terminal paths:
1. **Clean exit** — the `for turn in range(max_turns):` loop hits `break` on `stop_reason == 'end_turn'` (or any other non-`tool_use` reason).
2. **`max_turns` exhaustion** — the for-else branch fires when no `break` occurred; today logs `"Agentic loop hit max turns"` and sets `stop_reason = "max_turns"` on the result.

Both paths must emit `loop_completion`. Place the emission after the for-else branch but before the function returns, so a single point covers both:

```python
record_event(
    session_id=session_id,
    stage_name=stage_name or "unknown",
    event_type="loop_completion",
    payload={
        "stop_reason": stop_reason,  # "end_turn" | "max_turns" | "tool_use" | ...
        "turns_used": turns_used,
        "model": model,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    },
)
```

- [ ] **Step 2.5: Failing tests for `loop_completion`**

Append to `tests/v2/test_claude_client.py::TestLoopRecoveryTelemetry` (or a new `TestLoopCompletionTelemetry` class):

```python
class TestLoopCompletionTelemetry:
    def test_emits_loop_completion_on_clean_exit(self, monkeypatch):
        # Configure fake client to end_turn on first turn. Assert one
        # loop_completion event with stop_reason='end_turn' and turns_used=1.
        ...

    def test_emits_loop_completion_on_max_turns(self, monkeypatch):
        # Configure fake client to always return tool_use; run with
        # max_turns=2. Assert one loop_completion event with
        # stop_reason='max_turns' and turns_used=2.
        ...

    def test_loop_completion_includes_token_totals(self, monkeypatch):
        # Verify payload carries summed input/output tokens across turns.
        ...
```

- [ ] **Step 2.6: Run to verify failure**

Expected: 3 fail.

- [ ] **Step 2.7: Implement**

In `v2/claude_client.py:run_agentic_loop`, after the for-else branch but before the `return AgenticLoopResult(...)`, add the `record_event` call shown above.

- [ ] **Step 2.8: Tests pass**

Expected: 3 new pass.

---

## Task 3: Thread telemetry kwargs through Tier A call sites

**Files:**
- Modify: `v2/agent.py:212` (executor)
- Modify: `v2/classifier.py:316, 404, 471` (3 classifier call sites)
- Modify: `tests/v2/test_agent.py`, `tests/v2/test_classifier.py`

### Purpose values

Decide and document the `purpose` taxonomy upfront (these become the auditor's filter values):

| Call site | `purpose` |
|---|---|
| `v2/agent.py:212` (executor) | `"executor"` |
| `v2/classifier.py:316` (TBD per site role — read the call to confirm) | `"classifier_news"` or per-role variant |
| `v2/classifier.py:404` | `"classifier_macro"` or per-role variant |
| `v2/classifier.py:471` | `"classifier_relevance"` or per-role variant |

Read the actual context at each line to assign accurate names. Document the final mapping in the Phase 2 plan as it lands.

`stage_name` should be the v2 session stage name (`"trading"`, `"pipeline"`, etc.) so auditor SQL can join cleanly to `session_stages`.

- [ ] **Step 3.1: Add purpose taxonomy as a constant**

In `v2/claude_client.py`, define:

```python
class AgentPurpose:
    EXECUTOR = "executor"
    CLASSIFIER_NEWS = "classifier_news"
    CLASSIFIER_MACRO = "classifier_macro"
    CLASSIFIER_RELEVANCE = "classifier_relevance"
    STRATEGIST_LOOP = "strategist_loop"  # used by agentic-loop calls
    REFLECTION_LOOP = "reflection_loop"
    # Tier B/C purposes added later if needed.
```

Use these constants in call sites, not bare strings, to prevent typos.

- [ ] **Step 3.2: Wire executor**

`v2/agent.py:212`:

```python
    response = _call_with_retry(
        client,
        model=model,
        max_tokens=4096,
        system=cached_system,
        messages=[{"role": "user", "content": input_json}],
        session_id=session_id,
        stage_name="trading",
        purpose=AgentPurpose.EXECUTOR,
    )
```

`session_id` must reach `get_trading_decisions`. Thread it through the function signature from the caller (likely `run_trading_stage` in `v2/session.py`).

- [ ] **Step 3.3: Wire classifier**

For each of the three call sites in `v2/classifier.py`, read surrounding context, decide the right `purpose` constant, thread `session_id`/`stage_name="pipeline"`/`purpose=...` through.

- [ ] **Step 3.4: Update tests**

In `tests/v2/test_agent.py`, add a test asserting that `_call_with_retry` is invoked with the correct telemetry kwargs:

```python
def test_executor_passes_telemetry_kwargs(self, monkeypatch):
    captured = {}
    def fake_call_with_retry(client, **kw):
        captured.update(kw)
        # return mock response
        return MockResponse(...)
    monkeypatch.setattr("v2.agent._call_with_retry", fake_call_with_retry)
    get_trading_decisions(executor_input, session_id=42)
    assert captured["session_id"] == 42
    assert captured["stage_name"] == "trading"
    assert captured["purpose"] == "executor"
```

Mirror this pattern in `tests/v2/test_classifier.py` for each call site.

- [ ] **Step 3.5: Update agentic-loop call sites to use the constants**

Phase 1 wires `stage_name="ideation"` / `"reflection"` to `run_agentic_loop`. Update those to also pass `purpose=AgentPurpose.STRATEGIST_LOOP` / `REFLECTION_LOOP` if you want loop-level rollup separate from stage-level. (Optional — `stage_name` may be sufficient for auditor queries.)

- [ ] **Step 3.6: Run full v2 suite**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/ -q --ignore=tests/v2/test_session.py
```

Expected: all pass.

---

## Task 4: `executor_response` event with schema-drift canary

**Files:**
- Modify: `v2/agent.py:225-260`
- Modify: `tests/v2/test_agent.py`

### Canonical key sets

Define at the top of `v2/agent.py`:

```python
EXECUTOR_KNOWN_TOP_KEYS = {"decisions", "thesis_invalidations", "market_summary", "risk_assessment"}
EXECUTOR_KNOWN_DECISION_KEYS = {
    "playbook_action_id", "ticker", "action", "intent_type",
    "intent_magnitude", "reasoning", "confidence",
    "is_off_playbook", "signal_refs", "thesis_id",
}
```

If the LLM returns a key not in these sets, we capture it in the event and the auditor flags it. This is the schema-drift canary.

### Event emission

Three places to emit:

1. **`max_tokens` truncation** (line 238) — before raising, emit:
```python
record_event(
    session_id=session_id, stage_name="trading",
    event_type="executor_response",
    payload={
        "parse_succeeded": False,
        "stop_reason": response.stop_reason,
        "decision_count": 0,
        "thesis_invalidation_count": 0,
        "unknown_top_level_keys": [],
        "unknown_decision_keys": [],
        "raw_response_text_truncated": response_text[:4096],
        "error": "max_tokens_truncation",
    },
)
```

2. **`JSONDecodeError`** (line 257) — before raising:
```python
record_event(
    session_id=session_id, stage_name="trading",
    event_type="executor_response",
    payload={
        "parse_succeeded": False,
        "stop_reason": response.stop_reason,
        "decision_count": 0, "thesis_invalidation_count": 0,
        "unknown_top_level_keys": [], "unknown_decision_keys": [],
        "raw_response_text_truncated": response_text[:4096],
        "error": f"JSONDecodeError: {e}",
    },
)
```

3. **Successful parse** (after building `AgentResponse`, before return):
```python
unknown_top = sorted(set(data.keys()) - EXECUTOR_KNOWN_TOP_KEYS)
unknown_dec = sorted({
    k for d in data.get("decisions", []) for k in d.keys()
} - EXECUTOR_KNOWN_DECISION_KEYS)
record_event(
    session_id=session_id, stage_name="trading",
    event_type="executor_response",
    payload={
        "parse_succeeded": True,
        "stop_reason": response.stop_reason,
        "decision_count": len(decisions),
        "thesis_invalidation_count": len(thesis_invalidations),
        "unknown_top_level_keys": unknown_top,
        "unknown_decision_keys": unknown_dec,
        "raw_response_text_truncated": response_text[:4096],
        "error": None,
    },
)
```

`session_id` reaches this function via the threading from Task 3.

- [ ] **Step 4.1: Failing tests**

```python
class TestExecutorResponseTelemetry:
    def test_emits_event_on_successful_parse(self, monkeypatch):
        # Configure fake response with valid JSON; assert one event with
        # parse_succeeded=True, decision_count populated.
        ...

    def test_captures_unknown_top_level_key(self, monkeypatch):
        # Fake response has {"decisions": [...], "novel_field": "..."}.
        # Assert unknown_top_level_keys == ["novel_field"].
        ...

    def test_captures_unknown_decision_key(self, monkeypatch):
        # Fake decision has a "confidence_calibration" field.
        # Assert unknown_decision_keys contains it.
        ...

    def test_emits_event_on_max_tokens(self, monkeypatch):
        # Response with stop_reason='max_tokens'; assert event with
        # parse_succeeded=False, error='max_tokens_truncation', raw text
        # truncated to <=4096.
        ...

    def test_emits_event_on_parse_failure(self, monkeypatch):
        # Response is malformed JSON; assert event with
        # parse_succeeded=False, error starts with 'JSONDecodeError'.
        ...

    def test_truncates_raw_response_text_to_4kb(self, monkeypatch):
        # Response is 10KB. Assert raw_response_text_truncated len <= 4096.
        ...
```

- [ ] **Step 4.2: Run to verify failure**

Expected: 6 fail.

- [ ] **Step 4.3: Implement**

Follow the snippets above. Be careful: the existing code raises on `max_tokens` and `JSONDecodeError` — emit before the raise, not after.

- [ ] **Step 4.4: Tests pass**

Expected: 6 new pass.

- [ ] **Step 4.5: Run full v2 suite**

Expected: all pass.

---

## Task 5: Eight new auditor checks

**Files:**
- Modify: `v2/audit.py` — append 8 check functions + register
- Modify: `tests/v2/test_audit.py`

### Check 1: `EXECUTOR_TRUNCATION_RATE`

```python
EXECUTOR_TRUNC_WARN_PCT = 0.10
EXECUTOR_TRUNC_CRITICAL_PCT = 0.25
EXECUTOR_TRUNC_MIN_N = 5

def check_executor_truncation_rate(cur) -> list[Finding]:
    """Executor calls hitting max_tokens. Indicates input context too large
    or max_tokens budget set too low. Both are silent quality regressions."""
    cur.execute("""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (
                 WHERE payload->>'stop_reason' = 'max_tokens'
               ) AS truncated
        FROM agent_events
        WHERE event_type = 'agent_call'
          AND payload->>'purpose' = 'executor'
          AND occurred_at > now() - interval '14 days'
    """)
    r = cur.fetchone()
    if r["total"] < EXECUTOR_TRUNC_MIN_N or not r["truncated"]:
        return []
    rate = r["truncated"] / r["total"]
    if rate < EXECUTOR_TRUNC_WARN_PCT:
        return []
    sev = "critical" if rate >= EXECUTOR_TRUNC_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="EXECUTOR_TRUNCATION_RATE",
        tier=3, severity=sev,
        title=f"Executor truncated on {r['truncated']}/{r['total']} calls (last 14d)",
        body=("Executor responses hit max_tokens. Input context may be too "
              "large; check ExecutorInput field sizes. Or model max_tokens "
              "budget needs raising."),
        affected_count=r["truncated"],
        evidence={"total": r["total"], "truncated": r["truncated"], "rate": round(rate, 3)},
        auto_fix=None,
    )]
```

### Check 2: `EXECUTOR_SCHEMA_DRIFT`

```python
SCHEMA_DRIFT_MIN_OCCURRENCES = 3

def check_executor_schema_drift(cur) -> list[Finding]:
    """LLM emitting JSON keys we don't parse. Canary for executor schema
    drift across model versions or prompt changes."""
    cur.execute("""
        SELECT key, COUNT(*) AS n
        FROM agent_events,
             LATERAL jsonb_array_elements_text(payload->'unknown_top_level_keys') AS t(key)
        WHERE event_type = 'executor_response'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
    """, (SCHEMA_DRIFT_MIN_OCCURRENCES,))
    top_drift = cur.fetchall()

    cur.execute("""
        SELECT key, COUNT(*) AS n
        FROM agent_events,
             LATERAL jsonb_array_elements_text(payload->'unknown_decision_keys') AS t(key)
        WHERE event_type = 'executor_response'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
    """, (SCHEMA_DRIFT_MIN_OCCURRENCES,))
    dec_drift = cur.fetchall()

    if not top_drift and not dec_drift:
        return []
    return [Finding(
        check_code="EXECUTOR_SCHEMA_DRIFT",
        tier=3, severity="warn",
        title=f"Executor LLM emitting unknown JSON keys (top:{len(top_drift)}, decision:{len(dec_drift)})",
        body=("Executor response contains JSON fields not in our canonical "
              "key sets. Either the prompt is requesting new fields the "
              "parser doesn't handle, or the LLM is emitting drift we should "
              "either consume or suppress. Update EXECUTOR_KNOWN_*_KEYS or "
              "the parser in v2/agent.py."),
        affected_count=len(top_drift) + len(dec_drift),
        evidence={
            "top_level_drift": [{"key": r["key"], "n": r["n"]} for r in top_drift],
            "decision_drift": [{"key": r["key"], "n": r["n"]} for r in dec_drift],
        },
        auto_fix=None,
    )]
```

### Check 3: `CLASSIFIER_ERROR_RATE`

```python
CLASSIFIER_WARN_PCT = 0.10
CLASSIFIER_CRITICAL_PCT = 0.25
CLASSIFIER_MIN_N = 10

def check_classifier_error_rate(cur) -> list[Finding]:
    """News/macro classifier failing on items in the pipeline stage.
    Distinct from `INVALID_ATTRIBUTION_CATEGORY` (which catches *what*
    the classifier produced when wrong); this catches *whether* the call
    succeeded at all."""
    cur.execute("""
        SELECT payload->>'purpose' AS purpose,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE (payload->>'success')::boolean = false) AS errors
        FROM agent_events
        WHERE event_type = 'agent_call'
          AND payload->>'purpose' LIKE 'classifier_%%'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
    """, (CLASSIFIER_MIN_N,))
    flagged = []
    for r in cur.fetchall():
        rate = r["errors"] / r["total"] if r["total"] else 0
        if rate >= CLASSIFIER_WARN_PCT:
            flagged.append({"purpose": r["purpose"], "total": r["total"],
                            "errors": r["errors"], "rate": round(rate, 3)})
    if not flagged:
        return []
    worst = max(f["rate"] for f in flagged)
    sev = "critical" if worst >= CLASSIFIER_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="CLASSIFIER_ERROR_RATE",
        tier=3, severity=sev,
        title=f"{len(flagged)} classifier purpose(s) with error rate >= {int(CLASSIFIER_WARN_PCT*100)}% in last 7d",
        body="Classifier calls failing in the pipeline stage. Investigate handler exceptions or API errors.",
        affected_count=len(flagged),
        evidence={"classifiers": flagged},
        auto_fix=None,
    )]
```

### Check 4: `LOOP_RECOVERY_BURST`

```python
RECOVERY_BURST_MIN = 3

def check_loop_recovery_burst(cur) -> list[Finding]:
    """Agentic-loop recovery branches fire often — known degradation
    indicator. Either the prompt is generating responses that exceed
    max_tokens or message history is bloating to context-length limits."""
    cur.execute("""
        SELECT payload->>'reason' AS reason, COUNT(*) AS n
        FROM agent_events
        WHERE event_type = 'loop_recovery'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
    """, (RECOVERY_BURST_MIN,))
    rows = cur.fetchall()
    if not rows:
        return []
    return [Finding(
        check_code="LOOP_RECOVERY_BURST",
        tier=3, severity="warn",
        title=f"{len(rows)} loop-recovery reason(s) firing >= {RECOVERY_BURST_MIN} times in 7d",
        body=("`run_agentic_loop` recovery branches (max_tokens retry / "
              "context-length aggressive prune) firing often. Investigate "
              "prompt size, message history pruning, or model max_tokens."),
        affected_count=sum(r["n"] for r in rows),
        evidence={"recoveries": [{"reason": r["reason"], "n": r["n"]} for r in rows]},
        auto_fix=None,
    )]
```

### Check 5: `AGENT_CALL_LATENCY_DRIFT`

```python
LATENCY_DRIFT_RATIO = 2.0
LATENCY_MIN_N = 10  # both windows

def check_agent_call_latency_drift(cur) -> list[Finding]:
    """p95 duration_ms by purpose — spike detector."""
    cur.execute("""
        WITH recent AS (
            SELECT payload->>'purpose' AS purpose,
                   percentile_cont(0.95) WITHIN GROUP
                       (ORDER BY (payload->>'duration_ms')::int) AS p95,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '7 days'
              AND payload ? 'duration_ms'
            GROUP BY 1
        ),
        prior AS (
            SELECT payload->>'purpose' AS purpose,
                   percentile_cont(0.95) WITHIN GROUP
                       (ORDER BY (payload->>'duration_ms')::int) AS p95,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '14 days'
              AND occurred_at <= now() - interval '7 days'
              AND payload ? 'duration_ms'
            GROUP BY 1
        )
        SELECT r.purpose, r.p95 AS recent_p95, p.p95 AS prior_p95
        FROM recent r JOIN prior p ON p.purpose = r.purpose
        WHERE r.n >= %s AND p.n >= %s
          AND p.p95 > 0
          AND r.p95 >= %s * p.p95
    """, (LATENCY_MIN_N, LATENCY_MIN_N, LATENCY_DRIFT_RATIO))
    rows = cur.fetchall()
    if not rows:
        return []
    return [Finding(
        check_code="AGENT_CALL_LATENCY_DRIFT",
        tier=3, severity="info",
        title=f"{len(rows)} agent purpose(s) with p95 latency >= {LATENCY_DRIFT_RATIO}x prior 7-day window",
        body="Per-purpose p95 latency in last 7d is significantly elevated vs prior 7d.",
        affected_count=len(rows),
        evidence={"drifts": [
            {"purpose": r["purpose"],
             "recent_p95_ms": int(r["recent_p95"]),
             "prior_p95_ms": int(r["prior_p95"]),
             "ratio": round(r["recent_p95"] / r["prior_p95"], 2)}
            for r in rows
        ]},
        auto_fix=None,
    )]
```

### Check 6: `EXECUTOR_PARSE_FAILURE_RATE`

```python
PARSE_FAIL_WARN_PCT = 0.05
PARSE_FAIL_CRITICAL_PCT = 0.15
PARSE_FAIL_MIN_N = 5

def check_executor_parse_failure_rate(cur) -> list[Finding]:
    """Executor responses failing to parse for reasons OTHER than truncation.
    Distinct from EXECUTOR_TRUNCATION_RATE: this catches malformed JSON
    (LLM returned prose, missing braces, fenced-code edge cases) — actual
    schema/format failures rather than budget overruns."""
    cur.execute("""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (
                 WHERE (payload->>'parse_succeeded')::boolean = false
                   AND payload->>'error' NOT LIKE 'max_tokens%%'
               ) AS parse_failed
        FROM agent_events
        WHERE event_type = 'executor_response'
          AND occurred_at > now() - interval '14 days'
    """)
    r = cur.fetchone()
    if r["total"] < PARSE_FAIL_MIN_N or not r["parse_failed"]:
        return []
    rate = r["parse_failed"] / r["total"]
    if rate < PARSE_FAIL_WARN_PCT:
        return []
    sev = "critical" if rate >= PARSE_FAIL_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="EXECUTOR_PARSE_FAILURE_RATE",
        tier=3, severity=sev,
        title=f"Executor JSON parse failed on {r['parse_failed']}/{r['total']} non-truncated calls (last 14d)",
        body=("Executor responses are failing JSON parse for reasons other "
              "than max_tokens truncation. Likely causes: prompt regression "
              "causing prose responses, fenced-code-block edge cases the "
              "stripper misses, or the LLM returning structured output in a "
              "different shape. Inspect raw_response_text_truncated on "
              "recent failures."),
        affected_count=r["parse_failed"],
        evidence={"total": r["total"], "parse_failed": r["parse_failed"], "rate": round(rate, 3)},
        auto_fix=None,
    )]
```

### Check 7: `AGENT_CALL_ERROR_RATE_BY_PURPOSE`

```python
GENERAL_CALL_ERROR_WARN_PCT = 0.10
GENERAL_CALL_ERROR_CRITICAL_PCT = 0.25
GENERAL_CALL_ERROR_MIN_N = 10

def check_agent_call_error_rate_by_purpose(cur) -> list[Finding]:
    """Catch-all error rate by purpose. Sits alongside CLASSIFIER_ERROR_RATE
    (which has tighter thresholds for the classifier-specific failure mode);
    this surfaces problems in any other purpose (executor, social posts,
    premarket, audit_judgment) without forcing each to get its own check."""
    cur.execute("""
        SELECT payload->>'purpose' AS purpose,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE (payload->>'success')::boolean = false) AS errors
        FROM agent_events
        WHERE event_type = 'agent_call'
          AND occurred_at > now() - interval '7 days'
          AND COALESCE(payload->>'purpose', '') NOT LIKE 'classifier_%%'
        GROUP BY 1
        HAVING COUNT(*) >= %s
    """, (GENERAL_CALL_ERROR_MIN_N,))
    flagged = []
    for r in cur.fetchall():
        rate = r["errors"] / r["total"] if r["total"] else 0
        if rate >= GENERAL_CALL_ERROR_WARN_PCT:
            flagged.append({"purpose": r["purpose"], "total": r["total"],
                            "errors": r["errors"], "rate": round(rate, 3)})
    if not flagged:
        return []
    worst = max(f["rate"] for f in flagged)
    sev = "critical" if worst >= GENERAL_CALL_ERROR_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="AGENT_CALL_ERROR_RATE_BY_PURPOSE",
        tier=3, severity=sev,
        title=f"{len(flagged)} agent purpose(s) with error rate >= {int(GENERAL_CALL_ERROR_WARN_PCT*100)}% in last 7d",
        body=("Generic per-purpose error tracking. Investigate the specific "
              "purpose's call site for handler exceptions or upstream API errors."),
        affected_count=len(flagged),
        evidence={"purposes": flagged},
        auto_fix=None,
    )]
```

The `NOT LIKE 'classifier_%%'` clause prevents double-firing alongside `CLASSIFIER_ERROR_RATE` — classifiers have their own dedicated check with a tighter threshold and a more specific remediation path.

### Check 8: `LOOP_MAX_TURNS_HIT`

```python
MAX_TURNS_WARN_N = 1
MAX_TURNS_CRITICAL_N = 3

def check_loop_max_turns_hit(cur) -> list[Finding]:
    """Agentic loops terminating because they hit max_turns rather than
    end_turn. Means strategist or reflection ran out of turn budget mid-task,
    leaving incomplete work (e.g., playbook half-written, theses uncommitted)."""
    cur.execute("""
        SELECT stage_name, COUNT(*) AS n,
               array_agg(DISTINCT session_id ORDER BY session_id DESC) AS session_ids
        FROM agent_events
        WHERE event_type = 'loop_completion'
          AND payload->>'stop_reason' = 'max_turns'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
    """)
    rows = cur.fetchall()
    if not rows:
        return []
    total = sum(r["n"] for r in rows)
    if total < MAX_TURNS_WARN_N:
        return []
    sev = "critical" if total >= MAX_TURNS_CRITICAL_N else "warn"
    return [Finding(
        check_code="LOOP_MAX_TURNS_HIT",
        tier=3, severity=sev,
        title=f"{total} agentic-loop run(s) terminated by max_turns in last 7d",
        body=("`run_agentic_loop` exited because it hit max_turns, not "
              "because Claude returned end_turn. Strategist or reflection "
              "didn't finish its task — playbook may be partial, rules may "
              "not have been proposed/retired. Either the prompt is asking "
              "for too much, the tool surface is too noisy, or max_turns "
              "needs raising."),
        affected_count=total,
        evidence={"by_stage": [
            {"stage_name": r["stage_name"], "n": r["n"],
             "recent_session_ids": list(r["session_ids"])[:5]}
            for r in rows
        ]},
        auto_fix=None,
    )]
```

This is high-signal: a strategist that hit max_turns means it stopped mid-task. Phase 1's `tool_invocation` events would show *which* tool calls happened, but only `loop_completion` reveals that the loop *didn't get to call* the tools it intended to.

### Check 9: `CACHE_HIT_RATIO_DEGRADATION`

```python
CACHE_RATIO_DROP_PCT_POINTS = 0.30  # absolute drop in cache_read share
CACHE_RATIO_MIN_N = 10  # both windows

def check_cache_hit_ratio_degradation(cur) -> list[Finding]:
    """Cache breakpoint placement broken — silent cost regression. The
    Anthropic prompt cache halves API costs; if a refactor moves cache
    breakpoints or breaks ephemeral cache markers, costs silently double.
    No per-stage cost ratio exists today; this catches it at the call level."""
    cur.execute("""
        WITH recent AS (
            SELECT payload->>'purpose' AS purpose,
                   SUM((payload->>'cache_read_tokens')::int) AS cache_read,
                   SUM((payload->>'cache_creation_tokens')::int) AS cache_creation,
                   SUM((payload->>'input_tokens')::int) AS input_tok,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '7 days'
              AND (payload->>'success')::boolean = true
            GROUP BY 1
        ),
        prior AS (
            SELECT payload->>'purpose' AS purpose,
                   SUM((payload->>'cache_read_tokens')::int) AS cache_read,
                   SUM((payload->>'cache_creation_tokens')::int) AS cache_creation,
                   SUM((payload->>'input_tokens')::int) AS input_tok,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '14 days'
              AND occurred_at <= now() - interval '7 days'
              AND (payload->>'success')::boolean = true
            GROUP BY 1
        )
        SELECT r.purpose,
               r.cache_read::float / NULLIF(r.cache_read + r.cache_creation + r.input_tok, 0) AS recent_ratio,
               p.cache_read::float / NULLIF(p.cache_read + p.cache_creation + p.input_tok, 0) AS prior_ratio,
               r.n AS recent_n,
               p.n AS prior_n
        FROM recent r JOIN prior p ON p.purpose = r.purpose
        WHERE r.n >= %s AND p.n >= %s
    """, (CACHE_RATIO_MIN_N, CACHE_RATIO_MIN_N))
    flagged = []
    for r in cur.fetchall():
        recent = r["recent_ratio"] or 0
        prior = r["prior_ratio"] or 0
        if prior - recent >= CACHE_RATIO_DROP_PCT_POINTS:
            flagged.append({
                "purpose": r["purpose"],
                "recent_ratio": round(recent, 3),
                "prior_ratio": round(prior, 3),
                "drop_pct_points": round(prior - recent, 3),
            })
    if not flagged:
        return []
    return [Finding(
        check_code="CACHE_HIT_RATIO_DEGRADATION",
        tier=3, severity="info",
        title=f"{len(flagged)} agent purpose(s) with cache_read share dropped >= {int(CACHE_RATIO_DROP_PCT_POINTS*100)}pp vs prior 7d",
        body=("Cache-read token share dropped sharply for one or more "
              "purposes. Likely cause: a refactor moved/removed an "
              "`ephemeral` cache breakpoint in `cached_system` or tool "
              "definitions. This silently doubles API cost. Inspect recent "
              "changes to system prompts and tool registration."),
        affected_count=len(flagged),
        evidence={"degradations": flagged},
        auto_fix=None,
    )]
```

`info` severity (not `warn`) because cache regressions are cost issues, not correctness issues — they shouldn't gate critical alerting but should surface in the audit dashboard.

- [ ] **Step 5.1: Failing tests for all 8 checks**

In `tests/v2/test_audit.py`, add 8 test classes following the same shape as Phase 1's check tests (no-events → empty findings, events trip → 1 finding, below-threshold → empty findings).

- [ ] **Step 5.2: Run to verify failures**

Expected: ~24 new tests fail with `AttributeError`.

- [ ] **Step 5.3: Implement the 8 checks**

Append all 8 to `v2/audit.py` after the Phase 1 checks. Add to `CHECKS` list:

```python
CHECKS: list = [
    # ... existing entries (Phase 1 + originals) ...
    "check_executor_truncation_rate",
    "check_executor_schema_drift",
    "check_executor_parse_failure_rate",
    "check_classifier_error_rate",
    "check_agent_call_error_rate_by_purpose",
    "check_loop_recovery_burst",
    "check_loop_max_turns_hit",
    "check_cache_hit_ratio_degradation",
    "check_agent_call_latency_drift",
    "check_rule_judgment",  # keep last (LLM call)
]
```

- [ ] **Step 5.4: Tests pass**

Expected: ~24 new pass.

- [ ] **Step 5.5: Run full v2 suite**

```bash
docker compose run --rm --no-deps trading python -m pytest tests/v2/ -q --ignore=tests/v2/test_session.py
```

Expected: all pass.

---

## Task 6: Paper validation, then prod

**Files:** None modified. Validation only.

- [ ] **Step 6.1: Run paper session in dry-run**

```bash
task paper:session:dry-run
```

- [ ] **Step 6.2: Inspect new event types**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper \
  psql -U algo -d trading -c "
    SELECT event_type, payload->>'purpose' AS purpose, COUNT(*) AS n
    FROM agent_events
    WHERE event_type IN ('agent_call', 'loop_recovery', 'loop_completion', 'executor_response')
    GROUP BY 1, 2
    ORDER BY 1, 2;
  " -c "
    SELECT stage_name, payload->>'stop_reason' AS stop_reason, COUNT(*) AS n
    FROM agent_events
    WHERE event_type = 'loop_completion'
    GROUP BY 1, 2;
  "
```

Expected:
- ≥1 row with `event_type='agent_call'`, `purpose='executor'`
- ≥1 row with `event_type='agent_call'`, purpose like `classifier_*`
- ≥1 row with `event_type='executor_response'`, `parse_succeeded=true`
- ≥1 row with `event_type='loop_completion'` per agentic-loop run (ideation + reflection); `stop_reason='end_turn'` on a healthy run
- `loop_recovery` rows only if recovery branches actually fired

- [ ] **Step 6.3: Manually corrupt a test response to verify schema canary**

In a paper-only test session, monkey-patch the executor to return a JSON with an extra field (e.g., `"experimental_score": 0.5` at the top level). Verify:

```sql
SELECT payload->'unknown_top_level_keys'
FROM agent_events
WHERE event_type = 'executor_response'
ORDER BY id DESC LIMIT 1;
```

Expected: `["experimental_score"]`.

This is the schema-drift canary in action.

- [ ] **Step 6.4: Run paper auditor**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T trading-paper \
  python -m v2.audit
```

Expected: no new false-positive findings (one paper session won't trip the multi-session thresholds).

- [ ] **Step 6.5: Apply to prod**

No SQL migration needed (Phase 2 reuses Phase 1's `agent_events` table). Just deploy the code change.

- [ ] **Step 6.6: After 7 days of prod data, run full audit**

```bash
docker compose exec -T trading python -m v2.audit
```

Capture: did any of the 5 new checks fire? What did they reveal?

---

## Out of scope (deferred)

These were considered and explicitly cut. They become justifiable only if telemetry from Phase 1 + Phase 2 reveals a specific need.

- **Tier B call sites** (twitter, bluesky, social_trades, social_weekly, premarket) — ~6 call sites. Could thread `_call_with_retry` kwargs through them in 30-50 LOC, but no auditor check would fire on these that isn't already covered by stage failure rate. Add later if a specific question (e.g., "is the twitter post LLM call frequently truncating?") arises.
- **Tier C audit rule judgment call site** — already tracked via `audit_runs.*_tokens`. Duplicative.
- **Per-retry telemetry inside `_call_with_retry`** — current granularity (one event per logical call) is correct. Per-retry is debug-only.
- **Full agentic-loop transcript capture (`loop_transcript` event_type)** — high payload size (20-100KB), low query value once tool_invocation events exist. Defer until a specific incident makes you wish for it. Add a `--save-transcripts` debug flag instead of always-on logging if needed.
- **Strategist intermediate assistant text between tool calls** — same reasoning as transcript capture. The tool-call args in `tool_invocation` events tell the structural story.
- **Per-call cache hit ratio derived metric** — reading `cache_read_tokens / (cache_read_tokens + cache_creation_tokens + input_tokens)` is fine on demand from `agent_call.payload`; don't materialize.

---

## Verification checklist

- [ ] Phase 1 plan has been merged and run in prod for at least one full session before Phase 2 starts
- [ ] Full v2 test suite passes
- [ ] Paper session has populated `agent_call`, `executor_response`, and (if recovery fires) `loop_recovery` events
- [ ] Schema-drift canary verified manually (Step 6.3)
- [ ] Paper auditor run completes without new false-positive findings
- [ ] After 7 days of prod telemetry, the 5 new audit checks have run against real data and produced expected outputs (or correctly stayed silent)
- [ ] Decision recorded: which auditor finding actually shaped a follow-up action — close the loop on whether the new telemetry was load-bearing
