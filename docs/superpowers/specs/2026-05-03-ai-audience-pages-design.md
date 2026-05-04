---
name: AI-audience methodology pages
date: 2026-05-03
status: draft
parent: 2026-05-03-audience-growth-overview.md
depends_on: 2026-05-03-dashboard-permalinks-design.md
---

# AI-audience methodology pages

Phase 3 of the audience-growth strategy. Adds the conversion surface for the AI/agentic-systems-builder audience: a "how it works" page, model & cost transparency, and a sample tool-call trace viewer. None of these are content engines on their own — they're standing pages that posts can reference and visitors can find.

Depends on Spec #1 for the rendering pipeline. No dependency on Specs #2 or #3.

## Goal

Three new pages on the public dashboard:

1. **`/about/index.html`** — methodology page. What the system is, the agent loop, models used, the daily session flow, the data it ingests, the dashboard's relationship to reality.
2. **`/internals/index.html`** — model & cost transparency. Daily Anthropic spend, per-stage model assignment, token counts, latency.
3. **`/trace/<session_id>/index.html`** — sample tool-call trace viewer. One representative session per week (or on-demand), redacted, showing the strategist's actual tool calls in order.

## Non-goals

- No prompt source-of-truth viewer. Prompts evolve fast; pinning them to the public site invites stale-content issues.
- No comparing-to-other-bots framing. Stay grounded in what this system does.
- No real-time / live trace viewer. Static, per-published-session.
- No new ingestion or telemetry pipeline. Use what we already log.

## Architecture

### `/about/index.html`

Static-ish: 80% hand-authored Markdown rendered to HTML at publish time, 20% auto-populated stats injected per publish. No SPA state.

Content sections (proposed):
- **What this is** — one paragraph.
- **The daily loop** — diagram of the 7 session stages.
- **Models** — which Claude model handles which stage (executor: Haiku, ideation/reflection: Sonnet/Opus). Pulled from `agent.DEFAULT_EXECUTOR_MODEL` and `strategy.DEFAULT_REFLECTION_MODEL` so it stays accurate.
- **Data** — what's ingested (Alpaca news, market data, account state) and what isn't.
- **Honesty** — limitations, known issues, what the dashboard does and doesn't reflect.
- **Code** — link to the GitHub repo.

Rendered from `public_dashboard/about.md` via Python's stdlib (no `markdown` library if we can avoid it; use a 30-line subset renderer or accept the dependency — call it in the implementation plan). The few auto-injected fields use the same `string.Template` substitution pattern as Spec #1's other pages.

### `/internals/index.html`

Pulls from a new `model_usage` table that the existing `claude_client._call_with_retry` populates per call.

**Schema addition:**

```sql
CREATE TABLE model_usage (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    stage TEXT,                      -- 'pipeline' | 'ideation' | 'trader' | etc
    model TEXT NOT NULL,             -- 'claude-haiku-4-5-20251001'
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_model_usage_session ON model_usage(session_id);
CREATE INDEX idx_model_usage_created_at ON model_usage(created_at);
```

Wire `claude_client._call_with_retry` to insert a row on each successful call. Stage attribution comes from a `claude_call_context` contextvar set at stage entry — falls back to `'unknown'`.

**Page content:**
- Last-7-days bar chart: cost per day, stacked by stage.
- Last-30-days table: model, total calls, avg input tokens, avg output tokens, total spend (computed from public Anthropic pricing — keep a dict in `v2/pricing.py`, hardcoded but easy to update).
- Per-stage call counts and average latency for the last session.

### `/trace/<session_id>/index.html`

A linearized view of the strategist's tool-call sequence for one session. Already partially observable from `session_stages` and the strategist's logs, but no sanitized public view exists.

**New table:**

```sql
CREATE TABLE strategist_traces (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    role TEXT NOT NULL,              -- 'user' | 'assistant' | 'tool_use' | 'tool_result'
    tool_name TEXT,
    tool_input_redacted JSONB,       -- with PII / api keys / Alpaca credentials stripped
    tool_output_redacted JSONB,
    text_content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Wire into `ideation_claude.run_strategist_loop` to record each turn. Redaction lives in a `v2/trace_redactor.py` — denylist of field names (`api_key`, `secret`, `bearer`, `authorization`) plus a regex for Alpaca order UUIDs.

Stage 6 publishes **all eligible traces ever** (same link-permanence rationale as Spec #1 — Cloudflare Pages does full-bundle replacement, so old `/trace/<id>/` URLs would 404 if we only emitted the latest). Eligibility: `session.complete=true` AND `now - completed_at > 4h`. The session_id is the URL.

A separate `/trace/index.html` lists all eligible traces with the most recent featured at the top — that's the page the weekly "trace of the week" tweet links to, so its URL stays stable while the featured trace rotates.

**Page content:**
- Header: session date, model, tool count.
- Vertical timeline: each turn renders as a card. Tool calls show `tool_name` + collapsed JSON. Tool results show first 500 chars + collapsible "show full result". Final assistant message renders the produced playbook.

### Changes summary

| File | Change |
|---|---|
| `public_dashboard/about.md` | NEW (hand-authored content) |
| `v2/dashboard_pages.py` | Add `render_about_page`, `render_internals_page`, `render_trace_page` |
| `v2/dashboard_publish.py` | Gather usage / trace data; emit pages |
| `v2/claude_client.py` | Insert into `model_usage` per call; read contextvar for stage |
| `v2/ideation_claude.py` | Insert into `strategist_traces` per turn |
| `v2/trace_redactor.py` | NEW redaction logic |
| `v2/pricing.py` | NEW model pricing dict |
| `v2/database/migrations/` | Two new migrations: `model_usage`, `strategist_traces` |

## Data flow

```
Each Claude API call (any stage):
  └─ claude_client._call_with_retry
      ├─ make API call
      └─ insert_model_usage(session_id, stage, model, tokens, latency)

Each strategist turn (ideation only):
  └─ run_strategist_loop step
      ├─ redact tool inputs/outputs
      └─ insert_strategist_trace(session_id, step, role, ...)

Stage 6 (daily):
  └─ run_dashboard_stage
      ├─ ... existing flow ...
      ├─ render_about_page → /about/index.html
      ├─ gather_internals_data → render_internals_page → /internals/index.html
      └─ pick latest stable session → render_trace_page → /trace/<id>/index.html
```

## Error handling

- `model_usage` write failures must NOT abort the API call. Wrap the insert in try/except; log + continue. Telemetry is best-effort.
- `strategist_traces` write failures same — never block the strategist on a logging insert.
- Trace redaction failure on any single field defaults to dropping the field (`"[REDACTED — see logs]"`) rather than publishing it raw.
- If no qualifying session exists for the trace page, emit a placeholder page with "no trace published yet" rather than a 404.

## Testing

- `tests/test_model_usage.py`: insert path; resilient to DB failure; pricing computation.
- `tests/test_trace_redactor.py`: denylist fields stripped; UUID regex catches Alpaca orders; non-targeted fields untouched.
- `tests/test_dashboard_pages.py` extensions: each new render function returns expected sections.
- `tests/test_dashboard_publish.py` extensions: trace selection rule picks the right session; publishes one trace.

## Open questions left for the implementation plan

- Whether to use a Markdown library (`markdown-it-py`) or a hand-rolled subset renderer. The latter avoids a dependency for ~150 lines of code; depends on how rich the about page wants to be.
- Trace data volume — strategist sessions can be hundreds of turns. Truncation rules for the page (probably "show first 50, expand for more").
- Whether `/internals/` should be public from day one or behind a query-param soft-gate. Default: fully public — transparency is the point.
- Pricing dict drift — Anthropic pricing changes occasionally. Add a CI test that asserts the dict matches a checked-in JSON snapshot, so updates are deliberate.
- Initial state: `model_usage` and `strategist_traces` start empty. The first publish after deployment will show sparse charts and a single trace. Acceptable; flag in plan.
- File-count ceiling: each trace adds 1 HTML page. Combined with Spec #1's per-trade and per-thesis pages, the 20,000-file Cloudflare Pages limit is a shared resource — same headroom check applies.
