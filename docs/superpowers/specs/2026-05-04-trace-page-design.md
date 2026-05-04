---
name: AI-audience methodology — /trace/ pages + redaction
date: 2026-05-04
status: draft
parent: 2026-05-03-ai-audience-pages-design.md
depends_on: 2026-05-03-dashboard-permalinks-design.md
---

# /trace/ tool-call viewer

Sub-spec 4c of the AI-audience methodology phase. Highest risk of the three — redaction is the only thing standing between strategist tool inputs/outputs and the public internet. This sub-spec is intentionally isolated so the redaction logic gets focused review.

## Goal

Two new pages:

1. `/trace/<session_id>/index.html` — linearized view of one session's strategist tool-call sequence, redacted.
2. `/trace/index.html` — listing of all eligible traces, most recent featured at the top. Stable URL for the weekly "trace of the week" social post to link to.

Stage 6 publishes **all eligible traces ever**, matching Spec #1's link-permanence rationale (Cloudflare Pages does full-bundle replacement, so we can't only publish the latest without breaking old links).

## Non-goals

- No real-time trace viewer.
- No trace from non-strategist agentic loops (executor, reflection, classifier). Strategist only — that's the audience-facing surface.
- No prompt source-of-truth viewer. Tool *calls* are public; tool *prompts* are not.
- No diff between traces. Each trace stands alone.

## Architecture

### Schema addition

Add migration `db/migrations/005_strategist_traces.sql` (after the model_usage migration if 4b ships first; otherwise renumber to next available):

```sql
CREATE TABLE strategist_traces (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    role TEXT NOT NULL,              -- 'user' | 'assistant' | 'tool_use' | 'tool_result'
    tool_name TEXT,
    tool_input_redacted JSONB,
    tool_output_redacted JSONB,
    text_content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (session_id, step_number)
);
CREATE INDEX idx_strategist_traces_session ON strategist_traces(session_id);
```

### Capture point

The agentic loop the strategist runs lives in `claude_client.run_agentic_loop`, called from both `ideation_claude` (strategist) and `strategy.py` (reflection). We only want strategist turns. Two options:

1. **Filter by stage contextvar** — reuse 4b's `claude_call_context`; record traces only when `stage == 'ideation'`.
2. **Explicit callback parameter** — `run_agentic_loop` accepts an optional `on_turn` callback; `run_strategist_loop` passes one, `run_strategy_stage` does not.

Recommend Option 2. Explicit > ambient for something that touches a hot path. The callback receives `(step_number, role, tool_name, tool_input, tool_output, text)` and is responsible for redacting + persisting; loop integrity does not depend on it (try/except wrapper around the call).

If 4b already lands and the contextvar exists, the option-2 callback can still read the contextvar to get session_id; that's fine.

### Redaction

`v2/trace_redactor.py` — pure function `redact(value: Any) -> Any`. Walks dicts/lists recursively. Two redaction rules:

1. **Field denylist** (case-insensitive substring match on key): `api_key`, `secret`, `bearer`, `authorization`, `password`, `token` (sub-substring of `tokens`? Yes — be generous; `tokens_used` becomes `[REDACTED]`. Acceptable cost). Replace value with `"[REDACTED]"`.
2. **String regex pass** on every leaf string value: Alpaca order UUID pattern (`[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`), Alpaca account ids if discoverable. Replace match with `[REDACTED-UUID]`.

Redaction failure on any single field defaults to dropping the field entirely (`{"field_name": "[REDACTED — see logs]"}`) rather than publishing it raw.

### Eligibility for publishing

A trace is eligible if `session.complete = true` AND `now - completed_at > 4 hours`. The 4-hour delay gives an operator window to spot bad redactions and pull the trace before it ships. Pull mechanism: a `published = false` flag on `strategist_traces` (or session-level `traces_published_after TIMESTAMPTZ` that the publisher respects).

### Page content

**`/trace/<session_id>/index.html`:**

- Header: session date, model, tool count, total turns.
- Vertical timeline. Each turn is a card:
  - `tool_use` card: tool name + collapsed JSON input (click to expand).
  - `tool_result` card: first 500 chars + `<details>` for full output.
  - Final assistant message: rendered as the produced playbook.
- Truncation: show first 50 turns by default; remaining turns collapsed behind a "Show all N turns" button. (Open question — could go higher if 50 feels arbitrary in practice.)

**`/trace/index.html`:**

- Top: featured (most recent eligible) trace, link + brief metadata.
- Below: chronological list of remaining eligible traces. Date, session id, tool count.

OG image (per-trace): simple Pillow card with session date + tool count. Single `/og/trace.png` for the listing page (since the listing rotates).

### Changes summary

| File | Change |
|---|---|
| `db/migrations/005_strategist_traces.sql` | NEW |
| `v2/trace_redactor.py` | NEW |
| `v2/claude_client.py` | `run_agentic_loop` accepts optional `on_turn` callback |
| `v2/ideation_claude.py` | Pass on_turn callback; redact + persist per turn |
| `v2/database/trading_db.py` | `insert_strategist_trace`, `get_eligible_traces`, `get_trace_by_session` |
| `v2/dashboard_pages.py` | Add `render_trace_page`, `render_trace_index_page` |
| `v2/dashboard_og.py` | Add `render_trace_og`, `render_trace_index_og` |
| `v2/dashboard_publish.py` | Gather + emit `/trace/<id>/` for every eligible session, plus `/trace/index.html` |

### Data flow

```
Each strategist turn:
  └─ run_agentic_loop step
      └─ on_turn(step_number, role, tool_name, raw_input, raw_output, text)
          ├─ redact tool inputs/outputs
          └─ insert_strategist_trace(...)  # try/except — never block strategist

Stage 6 (daily):
  └─ run_dashboard_stage
      ├─ get_eligible_traces() → list of session_ids
      ├─ for each: render_trace_page → /trace/<id>/index.html
      └─ render_trace_index_page → /trace/index.html
```

### Error handling

- Trace insert failure: log + continue. Strategist must never block on telemetry.
- Redaction failure on a field: drop the field with `"[REDACTED — see logs]"`. Never publish raw.
- Empty `strategist_traces` (initial state): emit `/trace/index.html` with a "no traces published yet" placeholder. No `/trace/<id>/` pages emitted.
- Redaction regression discovered post-publish: operator sets `traces_published_after` on the affected session(s) to `NULL` (or sets a `published=false` flag if we choose that schema); next Stage 6 run drops those pages. Cloudflare picks up the deletion.

## Testing

- `tests/test_trace_redactor.py`: denylist substrings replaced; UUID regex catches Alpaca orders; non-targeted fields untouched; nested dicts recursed; redaction failure on one field doesn't poison the rest.
- `tests/test_dashboard_pages.py`: `render_trace_page` produces expected card structure; `render_trace_index_page` features most recent trace; truncation rule applied.
- `tests/test_dashboard_publish.py`: eligibility query picks the right sessions; publish emits one page per eligible session; empty state emits placeholder index.

## Open questions

- Truncation default (50 turns) — calibrate after looking at recent strategist sessions during plan-writing.
- Whether to keep raw (un-redacted) trace records anywhere for operator audit, or rely on the live DB tables (which contain raw tool inputs/outputs already). Default: rely on live tables; `strategist_traces` is redacted-from-the-start.
- File-count ceiling: per-session `/trace/<id>/` pages add ~250/year. Combined with Spec #1's per-trade and per-thesis pages this needs a single shared headroom check before this sub-spec ships — count current emitted pages, project growth rate.
- 4-hour eligibility delay — long enough? Short enough? Shorter = stale-trace risk. Longer = audience misses the freshness window.
