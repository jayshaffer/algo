---
name: AI-audience methodology — /internals/ page + model-usage telemetry
date: 2026-05-04
status: draft
parent: 2026-05-03-ai-audience-pages-design.md
depends_on: 2026-05-03-dashboard-permalinks-design.md
---

# /internals/ model & cost transparency

Sub-spec 4b of the AI-audience methodology phase. Adds a model-usage telemetry table populated by every Claude API call, plus a public `/internals/index.html` that visualizes it.

This sub-spec instruments a hot path (`claude_client._call_with_retry`) called from 12+ sites — wider blast radius than 4a or 4c. Treat the instrumentation patch as load-bearing.

## Goal

1. Capture per-call model usage in a new `model_usage` table.
2. Publish `/internals/index.html` with a 7-day cost-by-stage chart, a 30-day per-model table, and per-stage call counts + average latency from the most recent session.

## Non-goals

- No real-time / streaming view. Daily refresh is fine.
- No per-call drill-down. Stage- and model-level granularity only.
- No spend-cap or alerting integration in this sub-spec — surfacing the data is enough; alerting can come later if needed.
- No prompt source-of-truth on this page.

## Architecture

### Schema addition

Add migration `db/migrations/004_model_usage.sql` (next available number after `003_thesis_signals.sql`):

```sql
CREATE TABLE model_usage (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    stage TEXT,                      -- 'pipeline' | 'ideation' | 'trader' | etc; 'unknown' fallback
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

### Stage attribution via contextvar

Stage-level attribution (so a row can say "this Haiku call came from the trader") needs an ambient signal `_call_with_retry` can read without changing every callsite signature.

Add `v2/claude_call_context.py` — a single `contextvars.ContextVar[dict]` holding `{"session_id": int | None, "stage": str | None}`. Stage entry points (`run_pipeline_stage`, `run_strategist_loop`, `run_trader_stage`, `run_strategy_stage`, `run_dashboard_stage`, premarket, social_weekly) bind the context once at the top of their function via a helper `bind_claude_call_context(session_id, stage)`. `_call_with_retry` reads it on insert; missing values fall back to `None` / `'unknown'`.

### Pricing

Add `v2/pricing.py` — a single dict mapping model id → `(input_per_mtok, output_per_mtok, cache_write_per_mtok, cache_read_per_mtok)` in USD. Hardcoded; updated manually when Anthropic ships a price change.

CI snapshot test: assert the dict matches a checked-in `tests/fixtures/pricing_snapshot.json` so price-change PRs are deliberate. (Open question: keep snapshot in sync via assertion only, or auto-regenerate? Default: assertion only — one human review per change.)

### Insert path

`_call_with_retry` records on success, after `stream.get_final_message()` returns:

```python
try:
    insert_model_usage(
        session_id=ctx["session_id"],
        stage=ctx["stage"] or "unknown",
        model=create_kwargs["model"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", None),
        cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", None),
        latency_ms=int((t_end - t_start) * 1000),
    )
except Exception:
    logger.exception("model_usage insert failed; continuing")
```

Telemetry write failure must NOT abort the API call. Bare except (logged) is the correct shape here.

### Page content

`/internals/index.html`:

- **Last 7 days, cost by day, stacked by stage** — bar chart (Chart.js stacked bar). Cost computed from `model_usage` rows joined to the `pricing.py` dict.
- **Last 30 days, per-model summary** — table: model, total calls, avg input tokens, avg output tokens, total spend.
- **Most recent session** — per-stage call counts + avg latency.

Rendered server-side via `render_internals_page` and emitted to `<deploy>/internals/index.html` by Stage 6. JSON sidecar `/internals.json` for the chart, fetched by the page's inline JS.

OG image: simple "$X spent over 7 days" text card via Pillow, same pattern as Spec #3's mistakes/attribution OG renderers.

### Changes summary

| File | Change |
|---|---|
| `db/migrations/004_model_usage.sql` | NEW |
| `v2/claude_call_context.py` | NEW (contextvar + binder) |
| `v2/claude_client.py` | Read contextvar; insert per call |
| `v2/database/trading_db.py` | `insert_model_usage`, `get_model_usage_*` query helpers |
| `v2/pricing.py` | NEW model→price dict |
| `v2/dashboard_pages.py` | Add `render_internals_page` |
| `v2/dashboard_og.py` | Add `render_internals_og` |
| `v2/dashboard_publish.py` | Gather usage data; emit `/internals/index.html` + OG + JSON |
| `v2/session.py`, `v2/premarket.py`, `v2/social_*.py`, etc. | Bind `claude_call_context` at stage entry |
| `tests/fixtures/pricing_snapshot.json` | NEW (price-change diff target) |

### Data flow

```
Each Claude API call (any stage):
  └─ claude_client._call_with_retry
      ├─ make API call
      ├─ on success: insert_model_usage(... from contextvar + response.usage)
      └─ on insert failure: log + continue (don't break the call)

Stage 6 (daily):
  └─ run_dashboard_stage
      ├─ gather_internals_data (joins usage × pricing)
      ├─ render_internals_page → /internals/index.html
      ├─ write internals.json
      └─ render_internals_og → /og/internals.png
```

### Error handling

- Insert failure: log + continue. Never raise.
- Pricing dict miss (model id present in `model_usage` but missing from dict): treat cost as `None`, surface as "—" in the table, log a warning. Don't crash publish.
- Empty `model_usage` (initial state): page renders with "no data yet" placeholder copy. Acceptable until the first session writes rows.

## Testing

- `tests/test_model_usage.py`: insert path; resilient to DB failure; stage attribution from contextvar; missing contextvar falls back to `'unknown'`.
- `tests/test_pricing.py`: pricing computation; snapshot diff catches drift.
- `tests/test_dashboard_pages.py`: `render_internals_page` returns expected sections; empty data → placeholder.
- `tests/test_dashboard_publish.py`: publish writes the page + JSON + OG image.

## Open questions

- Pricing snapshot policy (assertion only vs auto-regenerate) — default: assertion only.
- Whether `/internals/` is public from day one or behind a query-param soft-gate. Default: fully public — transparency is the point.
- File-count ceiling: `/internals/` adds 1 page (not per-session). No meaningful pressure on the 20k Cloudflare limit.
- Whether the OG image is worth the implementation cost given how rarely a "/internals/" link gets shared. Default: yes, keep parity with other permalinks; revisit if rendering noise outweighs use.
