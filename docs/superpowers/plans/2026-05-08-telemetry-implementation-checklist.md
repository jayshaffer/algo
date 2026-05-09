# Auditor-Visible Telemetry — Implementation Checklist

> **Branch:** `telemetry-phase-1`
> **Design plans:**
> - Phase 1: [`2026-05-08-flip-flop-telemetry.md`](2026-05-08-flip-flop-telemetry.md) — `agent_events` substrate + 3 event types + 6 auditor checks
> - Phase 2: [`2026-05-08-agent-call-telemetry-phase-2.md`](2026-05-08-agent-call-telemetry-phase-2.md) — 4 more event types + 8 more auditor checks
>
> This file is the flat execution checklist. Tick each box as you go. Ship Phase 1 first; Phase 2 only after one prod session validates the substrate.

---

## Phase 1 — Auditor-visible blind-spot telemetry

### P1.T1 — Schema migration (`db/init/026_agent_events.sql`)

- [ ] Create `db/init/026_agent_events.sql` with `agent_events` table (id, session_id FK→sessions, stage_name, event_type, payload JSONB, occurred_at)
- [ ] Add 5 indexes: `(session_id)`, `(event_type, occurred_at DESC)`, `(stage_name, event_type)`, functional `(payload->>'tool_name') WHERE event_type='tool_invocation'`, functional `(payload->>'ticker') WHERE event_type='risk_block'`
- [ ] Apply to paper DB: `docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper psql -U algo -d trading < db/init/026_agent_events.sql`
- [ ] Verify with `\d agent_events` — table + all indexes present
- [ ] Defer prod migration until P1.T8

### P1.T2 — `v2/telemetry.py` recorder + helpers

- [ ] Write failing tests in `tests/v2/test_telemetry.py::TestRecordEvent` (4 cases: insert ok, no-op on session_id=None, swallows exceptions, serializes dates)
- [ ] Run tests, verify ModuleNotFoundError
- [ ] Create `v2/telemetry.py` with `record_event(session_id, stage_name, event_type, payload)` — JSONB insert, best-effort try/except, no-op on `session_id=None`
- [ ] Run tests, verify 4 pass
- [ ] Add tests for `count_tool_invocations_by_session(session_id) -> dict[str,int]` (1 case)
- [ ] Add tests for `session_summary_line(session_id) -> str` (2 cases: populated, empty)
- [ ] Implement both helpers; verify all 7 pass

### P1.T3 — Wire `tool_invocation` events into `run_agentic_loop`

- [ ] Add failing tests in `tests/v2/test_claude_client.py::TestRunAgenticLoopTelemetry` (3 cases: success emits event, handler error emits with success=false, no session_id is no-op)
- [ ] Run tests, verify failures
- [ ] In `v2/claude_client.py`: add `from .telemetry import record_event` and `import time`
- [ ] Extend `run_agentic_loop` signature with `session_id: int | None = None, stage_name: str | None = None`
- [ ] In tool dispatch loop (around line 405-416), wrap handler call with `time.monotonic()` timing; emit `tool_invocation` event with `{tool_name, args, success, error, duration_ms}` after each dispatch
- [ ] Run tests; verify all pass
- [ ] In `v2/ideation_claude.py:219`, pass `session_id=session_id, stage_name="ideation"` to `run_agentic_loop`
- [ ] Thread `session_id` through `run_ideation_stage` signature if needed; update caller in `v2/session.py`
- [ ] Add test in `tests/v2/test_ideation_claude.py` asserting telemetry kwargs reach the loop
- [ ] In `v2/strategy.py:545`, pass `session_id=session_id, stage_name="reflection"` to `run_agentic_loop`
- [ ] Add test in `tests/v2/test_strategy.py` asserting telemetry kwargs reach the loop
- [ ] Run full v2 suite (excluding `test_session.py`); verify ≥1118 passed

### P1.T4 — Wire `evidence_shown` events for round-trip data

- [ ] Add failing tests in `tests/v2/test_strategy.py::TestEvidenceShownEvent` (3 cases: emits with round_trips, emits empty list when no round-trips, no event when session_id=None)
- [ ] Run tests, verify failures
- [ ] In `v2/strategy.py`: add `from .telemetry import record_event`
- [ ] Add `tool_get_session_summary_with_telemetry(days: int = 30, *, session_id: int | None = None) -> str` wrapper that calls inner function then computes `analyze_round_trips(...)` and emits `evidence_shown` event with payload `{evidence_kind: "round_trips", items: [...], summary: {n_tickers}}` (emit even on empty)
- [ ] At reflection-stage agentic-loop call site (line 545 area), bind handler dict via `partial(tool_get_session_summary_with_telemetry, session_id=session_id)` for `get_session_summary` key; pass to `run_agentic_loop`
- [ ] Run tests; verify all pass

### P1.T5 — Wire `risk_block` events into trader

- [ ] Locate `check_sector_cap_for_buy` breach branch in `v2/trader.py` (around line 520)
- [ ] Add failing tests in `tests/v2/test_trader.py::TestRiskBlockTelemetry` (2 cases: emits on breach, no event on under-cap)
- [ ] Run tests, verify failures
- [ ] In `v2/trader.py`: add `from .telemetry import record_event`
- [ ] At breach branch (after marking decision invalid), emit `risk_block` event with `{ticker, sector, proposed_qty, price, sector_pct_after, cap, reason_text}`
- [ ] Thread `session_id` from `_execute_decisions` to `_prepare_decision` via function signature if not already in scope
- [ ] Run tests; verify all pass

### P1.T6 — Six new auditor checks in `v2/audit.py`

- [ ] Add failing tests for `check_strategist_using_reversal_tool` in `tests/v2/test_audit.py` (3 cases: no events, trips, below threshold)
- [ ] Add failing tests for `check_reflection_inert_on_round_trips` (3 cases)
- [ ] Add failing tests for `check_tool_error_rate` (3 cases)
- [ ] Add failing tests for `check_risk_block_hotspot` (3 cases)
- [ ] Add failing tests for `check_risk_block_burst` (3 cases)
- [ ] Add failing tests for `check_ideation_tool_drought` (3 cases)
- [ ] Run tests, verify ~18 failures
- [ ] Implement `check_strategist_using_reversal_tool` (3 consecutive ideation sessions with round-trip evidence but no `get_recent_playbooks` call → warn)
- [ ] Implement `check_reflection_inert_on_round_trips` (5 sessions with round-trip evidence and zero `propose_rule`/`retire_rule` → warn)
- [ ] Implement `check_tool_error_rate` (any tool ≥20% error in 7d → warn, ≥50% → critical, min N=5)
- [ ] Implement `check_risk_block_hotspot` (same ticker ≥3 risk_blocks in 7d → warn)
- [ ] Implement `check_risk_block_burst` (≥5 risk_blocks on a single date in 14d → warn)
- [ ] Implement `check_ideation_tool_drought` (any tool in expected set with 0 calls across last 7 ideation sessions → warn)
- [ ] Add all 6 to `CHECKS` list in `v2/audit.py` (before `check_rule_judgment`, which stays last)
- [ ] Run all audit tests; verify pass
- [ ] Run full v2 suite; verify ≥1140 passed

### P1.T7 — Session-end `[telemetry]` log line

- [ ] In `v2/session.py`: add `from .telemetry import session_summary_line`
- [ ] At end of `run_session` (after all stages), call `logger.info(session_summary_line(session_id))`
- [ ] Add test in `tests/v2/test_session.py` asserting `session_summary_line` is called once with active session_id
- [ ] Run session tests; pre-existing 10 twitter/bluesky/dashboard failures unchanged

### P1.T8 — Paper validation + prod migration

- [ ] Run `task paper:session:dry-run`
- [ ] Inspect events: `SELECT event_type, stage_name, COUNT(*) FROM agent_events GROUP BY 1,2 ORDER BY 1,2;` against paper DB
- [ ] Confirm rows exist for `tool_invocation` (ideation+reflection), `evidence_shown` (reflection); `risk_block` rows only if breaches occurred
- [ ] Confirm `[telemetry]` log line in `docker compose logs trading-paper`
- [ ] Run paper auditor: `docker compose ... exec -T trading-paper python -m v2.audit` — verify no false positives
- [ ] Apply migration to prod DB: `docker compose exec -T db psql -U algo -d trading < db/init/026_agent_events.sql`
- [ ] Wait for next prod daily session
- [ ] Re-inspect events on prod; capture: was `get_recent_playbooks` invoked? round-trip count? risk_block count?

### Phase 1 verification gate

- [ ] Migration applied to paper and prod
- [ ] Full v2 test suite passes
- [ ] Paper session populated `agent_events` with all 3 event types
- [ ] `[telemetry]` log line confirmed
- [ ] Paper auditor run completes without new false positives
- [ ] One prod session has produced events and the 6 new audit checks have run against real data
- [ ] Decision recorded: manually retire Rule #27, wait for the loop, or proceed to Phase 2

---

## Phase 2 — Per-call agent telemetry (after Phase 1 validated for ≥1 prod session)

### P2.T1 — `agent_call` event in `_call_with_retry`

- [ ] Add failing tests in `tests/v2/test_claude_client.py::TestCallWithRetryTelemetry` (3 cases: success, failure, default-None kwargs)
- [ ] Run tests, verify failures
- [ ] Extend `_call_with_retry` signature with keyword-only `session_id`, `stage_name`, `purpose`
- [ ] Define `class AgentPurpose:` constants (`EXECUTOR`, `CLASSIFIER_NEWS`, `CLASSIFIER_MACRO`, `CLASSIFIER_RELEVANCE`, `STRATEGIST_LOOP`, `REFLECTION_LOOP`)
- [ ] Wrap call body with `time.monotonic()` timing; emit `agent_call` event in finally block with `{model, purpose, stop_reason, duration_ms, input/output/cache_* tokens, success, error}`
- [ ] Run tests; verify all pass

### P2.T2 — `loop_recovery` + `loop_completion` events

- [ ] Add failing tests for `TestLoopRecoveryTelemetry` (3 cases: max_tokens, context_length, clean run)
- [ ] Run, verify failures
- [ ] Emit `loop_recovery` event in both recovery branches in `run_agentic_loop` (line 324 context-length, line 342 max_tokens) with `{reason, turn, model}`
- [ ] Tests pass
- [ ] Add failing tests for `TestLoopCompletionTelemetry` (3 cases: clean exit, max_turns, token totals)
- [ ] Run, verify failures
- [ ] Emit `loop_completion` event after for-else but before return; payload `{stop_reason, turns_used, model, input_tokens, output_tokens}`
- [ ] Tests pass

### P2.T3 — Thread telemetry kwargs through Tier A call sites

- [ ] In `v2/agent.py:212` (executor), pass `session_id, stage_name="trading", purpose=AgentPurpose.EXECUTOR` to `_call_with_retry`
- [ ] Thread `session_id` through `get_trading_decisions` from caller (`run_trading_stage` in `v2/session.py`)
- [ ] Add test in `tests/v2/test_agent.py` asserting kwargs reach `_call_with_retry`
- [ ] In `v2/classifier.py:316`, pass telemetry kwargs (purpose TBD per role)
- [ ] In `v2/classifier.py:404`, same
- [ ] In `v2/classifier.py:471`, same
- [ ] Add tests in `tests/v2/test_classifier.py` for each call site
- [ ] Update agentic-loop call sites (ideation, reflection) to also pass `purpose` constants if useful for rollup
- [ ] Run full v2 suite; verify pass

### P2.T4 — `executor_response` event with schema-drift canary

- [ ] In `v2/agent.py`: define `EXECUTOR_KNOWN_TOP_KEYS` and `EXECUTOR_KNOWN_DECISION_KEYS` constants
- [ ] Add failing tests in `tests/v2/test_agent.py::TestExecutorResponseTelemetry` (6 cases: success, unknown top key, unknown decision key, max_tokens path, parse failure path, raw text 4KB cap)
- [ ] Run, verify failures
- [ ] Emit `executor_response` event at successful parse path: `{parse_succeeded: true, stop_reason, decision_count, thesis_invalidation_count, unknown_top_level_keys, unknown_decision_keys, raw_response_text_truncated, error: null}`
- [ ] Emit at `max_tokens` truncation path before raise (with `error: 'max_tokens_truncation'`)
- [ ] Emit at `JSONDecodeError` path before raise (with `error: f'JSONDecodeError: {e}'`)
- [ ] Tests pass

### P2.T5 — Eight new auditor checks

- [ ] Add failing tests for `check_executor_truncation_rate` (3 cases)
- [ ] Add failing tests for `check_executor_schema_drift` (3 cases)
- [ ] Add failing tests for `check_executor_parse_failure_rate` (3 cases)
- [ ] Add failing tests for `check_classifier_error_rate` (3 cases)
- [ ] Add failing tests for `check_agent_call_error_rate_by_purpose` (3 cases)
- [ ] Add failing tests for `check_loop_recovery_burst` (3 cases)
- [ ] Add failing tests for `check_loop_max_turns_hit` (3 cases)
- [ ] Add failing tests for `check_cache_hit_ratio_degradation` (3 cases)
- [ ] Add failing tests for `check_agent_call_latency_drift` (3 cases)
- [ ] Run, verify ~27 failures
- [ ] Implement all 9 check functions in `v2/audit.py`
- [ ] Add to `CHECKS` list (before `check_rule_judgment`)
- [ ] Tests pass

### P2.T6 — Phase 2 paper validation

- [ ] Run paper session in dry-run
- [ ] Inspect new event types: `SELECT event_type, payload->>'purpose', COUNT(*) FROM agent_events WHERE event_type IN ('agent_call','loop_recovery','loop_completion','executor_response') GROUP BY 1,2;`
- [ ] Verify `executor_response` rows present with `parse_succeeded=true`
- [ ] Verify `loop_completion` rows for both ideation and reflection stages
- [ ] Manually corrupt a test response to verify schema canary populates `unknown_top_level_keys`
- [ ] Run paper auditor; verify no false positives
- [ ] Deploy to prod (no migration needed — reuses Phase 1's `agent_events`)
- [ ] After 7 days of prod data, re-run audit; capture which (if any) new checks fired

### Phase 2 verification gate

- [ ] Phase 1 was running in prod for ≥1 session before Phase 2 started
- [ ] Full v2 test suite passes after Phase 2
- [ ] Paper validated for all 4 new event types
- [ ] Schema-drift canary tested manually
- [ ] After 7 days of prod data, decision recorded: which auditor finding shaped a follow-up action — close the loop on whether the new telemetry was load-bearing

---

## Out of scope (deferred from both phases)

- Tier B agent calls (twitter, bluesky, social_trades, social_weekly, premarket) — telemetry kwargs threadable later if a question demands it
- Tier C audit rule-judgment call — already tracked in `audit_runs.*_tokens`
- Per-retry telemetry inside `_call_with_retry`
- Full agentic-loop transcript capture (high payload, low query value)
- Strategist intermediate assistant text between tool calls
- Rule citation persistence (already inline-regex in `check_rule_judgment`)
- Prompt versioning / hash (no A/B testing happening)
- Stage Claude metadata (better as columns on `session_stages` than events)
- `positions` divergence (already covered by Alpaca sync)
