# LLM Context Logging — Design

**Date:** 2026-05-13
**Status:** Draft, pending approval

## Problem

We have no durable record of what context was actually fed into Claude on
the strategist and executor calls. When a decision looks wrong in
hindsight, we can read the resulting decision text and the telemetry
(token counts, stop reasons) but we cannot reconstruct what the model
saw. This blocks two use cases:

- **Forensics** — "On 2026-05-08 the executor took a 30% position in
  GOOGL; what did its input look like?"
- **Prompt/context tuning** — eyeballing what is actually being sent to
  spot bloat, stale fields, or context that drifted out of sync with
  what we think we are sending.

## Scope

In scope: capture the full payload sent to Claude (system prompt + the
messages array as it stood at the moment of the API call) for:

- `AgentPurpose.EXECUTOR` (`v2/agent.py`)
- `AgentPurpose.STRATEGIST_LOOP` (`v2/ideation_claude.py`)
- `AgentPurpose.REFLECTION_LOOP` (`v2/strategy.py`)

Reflection is included because it is structurally identical to the
strategist (agentic loop, multi-turn, reads/writes strategy state) and
sits in the same observability bucket. If we capture strategist context
without reflection we will regret it the first time we want to debug a
bad rule retirement.

Out of scope: classifier (`CLASSIFIER_NEWS`, `CLASSIFIER_MACRO`,
`CLASSIFIER_RELEVANCE`) calls. Classifier runs hundreds of times per
session and is not a target of forensic or tuning interest at this time.
A gate by `purpose` keeps these out of the new table.

## Design

### Capture point

Instrumentation lives in a single place: `_call_with_retry` in
`v2/claude_client.py`. This function already sees:

- `session_id`, `stage_name`, `purpose` (popped from `create_kwargs`)
- The full `create_kwargs` it forwards to `client.messages.stream()` —
  including `system`, `messages`, `tools`, `model`, `max_tokens`
- The resulting `message.usage` (token counts) and `message.stop_reason`
- Duration via the existing monotonic timer

A new helper `_record_call_context()` is called from the same `finally`
block that emits the `agent_call` telemetry event. It is gated on
`purpose` so only EXECUTOR / STRATEGIST_LOOP / REFLECTION_LOOP rows are
written.

Because the strategist and reflection loops invoke `_call_with_retry`
once per agentic turn, this naturally produces **one row per turn**.
Each row captures the cumulative `messages` array as it was sent on
that turn. Earlier rows are prefixes of later rows; the final row in a
session/stage/purpose group contains the complete transcript. This is
what we want: it shows how context grew and where the model chose to
stop or call tools.

### Sequence assignment

`sequence` is a monotonically increasing integer within
`(session_id, stage_name, purpose)`. Computed at insert time via:

```sql
INSERT INTO llm_call_contexts (session_id, stage_name, purpose, sequence, ...)
VALUES (
  $1, $2, $3,
  COALESCE(
    (SELECT MAX(sequence) + 1 FROM llm_call_contexts
     WHERE session_id = $1 AND stage_name = $2 AND purpose = $3),
    0
  ),
  ...
);
```

There is only one writer process per session, so the read-then-insert
race is not a concern in practice. The UNIQUE constraint on
`(session_id, stage_name, purpose, sequence)` is a backstop — if the
race ever happens, the second insert fails with an `IntegrityError`,
which is caught and logged the same way as any other capture failure.

### Failure mode

`_record_call_context()` wraps its DB work in a broad try/except, logs a
warning on failure, and swallows the exception. Logging context must
never break a session. Matches the posture of the existing
`record_event()` telemetry write.

### What is captured

For each captured call:

| Field | Source |
|---|---|
| `session_id` | from telemetry kwargs |
| `stage_name` | from telemetry kwargs |
| `purpose` | from telemetry kwargs |
| `sequence` | computed (see above) |
| `model` | `create_kwargs["model"]` |
| `system_prompt` | `create_kwargs.get("system")` — the actual string as sent, post-assembly of formation_context and any other dynamic suffixes |
| `messages` | `create_kwargs["messages"]` serialized to JSONB. Captures the full conversation state as sent: user/assistant turns, tool_use blocks, tool_result blocks. Cache_control markers are preserved as-is. |
| `tool_definitions` | `create_kwargs.get("tools")` serialized to JSONB. Helps interpret tool_use blocks in the transcript. NULL for executor (no tools). |
| `response_content` | `message.content` serialized to JSONB — the full list of content blocks returned by Claude (text blocks, tool_use blocks, thinking blocks if any). Preserves block types and tool_use IDs so the transcript is reconstructable end-to-end. |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens` | from `message.usage` |
| `stop_reason` | from `message` |
| `duration_ms` | from the monotonic timer |
| `created_at` | `now()` default |

Each row now represents a complete request/response round-trip for one
LLM call. For the strategist/reflection loops, the cumulative
transcript is reconstructable from the rows by concatenating
`messages[N]` (the input at turn N) with `response_content[N]` (the
assistant reply at turn N) — and equivalently `response_content[N]`
should match the trailing assistant content embedded in
`messages[N+1]`. That redundancy is useful as a consistency check.

The executor produces exactly one row whose `response_content` holds
the assistant text containing the decision JSON. That JSON is also
parsed and persisted to the `decisions` table; the raw assistant text
is captured here for forensics on parse failures or for diffing what
the model said against what the parser extracted.

### Schema

New table created by a new migration
(`db/migrations/006_llm_call_contexts.sql`):

```sql
CREATE TABLE llm_call_contexts (
  id BIGSERIAL PRIMARY KEY,
  session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
  stage_name TEXT NOT NULL,
  purpose TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  model TEXT NOT NULL,
  system_prompt TEXT,
  messages JSONB NOT NULL,
  tool_definitions JSONB,
  response_content JSONB,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cache_read_tokens INTEGER,
  cache_creation_tokens INTEGER,
  stop_reason TEXT,
  duration_ms INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (session_id, stage_name, purpose, sequence)
);

CREATE INDEX idx_llm_call_contexts_session_stage
  ON llm_call_contexts (session_id, stage_name);
CREATE INDEX idx_llm_call_contexts_created_at
  ON llm_call_contexts (created_at);
```

`session_id` is nullable to match existing telemetry semantics (calls
made outside a session, e.g. ad-hoc CLI runs, still pass through
`_call_with_retry` with `session_id=None`). In that case the call is
**not** logged — the `_record_call_context` helper also gates on
`session_id is not None`, same as `record_event`.

`ON DELETE CASCADE` ensures that if a `sessions` row is ever deleted
(rare, but possible during cleanup) the associated context rows go with
it.

### Retention

No automatic retention for now. The table grows unbounded; revisit when
it starts to hurt. Rough sizing assuming ~1 session/day:

- Executor: 1 row/session, ~30 KB input + small response = ~35 KB
- Strategist: ~10-30 turns/session, ~50 KB average row (cumulative
  transcript) plus the per-turn response (~2-20 KB of text + tool_use
  blocks). Long-tail ~1.5 MB/session including responses.
- Reflection: similar shape to strategist, smaller

Order-of-magnitude budget: ~1.5 MB/day, ~550 MB/year. Postgres handles
this trivially via TOAST compression on the JSONB column. The
`idx_llm_call_contexts_created_at` index is there so a future retention
sweep can find old rows efficiently.

## Components

### `v2/database/trading_db.py`

Add `insert_llm_call_context(...)` writer function. Follows the existing
raw-SQL + psycopg2 pattern used everywhere else in this module. Returns
nothing useful (fire-and-forget).

### `v2/claude_client.py`

Add `_record_call_context(create_kwargs, message, session_id, stage_name, purpose, duration_ms)`.

- Gates on `session_id is not None AND purpose in {EXECUTOR, STRATEGIST_LOOP, REFLECTION_LOOP}`.
- Calls `insert_llm_call_context(...)` inside try/except. On exception, logs a warning at WARNING level and returns.
- Called from the same `finally` block that calls `record_event`, after the existing telemetry write so a failure in context-logging cannot prevent the telemetry write.

### `db/migrations/006_llm_call_contexts.sql`

The CREATE TABLE + indexes shown above.

## Data flow

```
trader.py → get_trading_decisions() → _call_with_retry(purpose=EXECUTOR)
                                          │
                                          ├─ stream().get_final_message()
                                          ├─ record_event(agent_call)         (existing)
                                          └─ _record_call_context()           (new)
                                                  │
                                                  └─ insert_llm_call_contexts row

ideation_claude.py → run_strategist_loop() → _run_claude_loop()
                                                  │
                                                  └─ for each turn:
                                                       _call_with_retry(purpose=STRATEGIST_LOOP)
                                                          │
                                                          ├─ record_event       (existing)
                                                          └─ _record_call_context (new)

strategy.py → reflection agentic loop → same shape as strategist (REFLECTION_LOOP)

classifier.py / social posts → _call_with_retry(purpose=CLASSIFIER_*) → NOT captured (gated out)
```

## Error handling

| Failure | Behavior |
|---|---|
| DB connection error during insert | Warning logged, exception swallowed, session continues |
| UNIQUE constraint violation on `(session_id, stage_name, purpose, sequence)` race | Caught with the general except; logged at WARNING; session continues. Loses that one row. |
| `session_id is None` (CLI runs, tests) | Skip silently |
| `purpose` not in the captured set | Skip silently |
| JSON serialization fails on `messages` or `tools` | Caught; logged with `purpose` and `stage_name`; session continues. We use `psycopg2`'s `Json` adapter so this should not happen in practice — but if a tool result ever contains something exotic, we want a warning not a crash. |

## Testing

Unit tests in `tests/v2/test_claude_client_context_logging.py`:

1. `_call_with_retry` with `purpose=EXECUTOR` and a non-None `session_id` writes one row matching the expected fields (system, messages, tool_definitions=None, response_content serialized from `message.content`, tokens, duration_ms ≥ 0).
2. `_call_with_retry` with `purpose=STRATEGIST_LOOP` writes a row with the `tools` payload serialized into `tool_definitions` and the response containing `tool_use` blocks serialized into `response_content`.
3. Two sequential calls with the same `(session_id, stage_name, purpose)` produce `sequence=0` then `sequence=1`.
4. `_call_with_retry` with `purpose=CLASSIFIER_NEWS` writes **no** row in `llm_call_contexts` (but still writes the `agent_call` telemetry row).
5. `_call_with_retry` with `session_id=None` writes **no** row.
6. DB failure during `insert_llm_call_context` (mocked to raise) does not surface to the caller; the LLM response is still returned.
7. Migration smoke test: `006_llm_call_contexts.sql` applies cleanly on top of the current schema and creates the expected indexes and constraints.

All tests follow the existing pattern: mock the Claude client and the
`get_cursor()` context manager; assert on the SQL that would have been
executed.

## Risks

**Sensitive content.** The strategist transcript and executor input
include positions, P&L, and reasoning text. This data is already in the
DB elsewhere (`positions`, `decisions`, `strategy_memos`), so the new
table does not expand the blast radius of a DB compromise. No new
exposure surface.

**Row size.** A single JSONB column can hold up to 1 GB. We are nowhere
near that, but worth noting: if a strategist run ever balloons (say, a
tool returns a giant blob), the row size will reflect it. The capture
itself does not impose a size cap. We accept that — capping would defeat
the purpose. If a single row ever exceeds, say, 10 MB, that is a signal
the strategist context is broken and we want to see it, not truncate
it.

**Write amplification.** Each strategist turn is now one additional
write. The agentic loop is already slow (LLM call dominates), so this
is irrelevant in practice.

## Open questions

None — confirmed in brainstorming:

- One row per turn (not per loop): yes
- Capture system prompt as actually sent on that turn: yes
- Capture response content (`message.content` blocks) per turn: yes
- Include `REFLECTION_LOOP` alongside strategist: defaulting to yes per
  the rationale above; flag for user confirmation when reviewing this
  spec.
