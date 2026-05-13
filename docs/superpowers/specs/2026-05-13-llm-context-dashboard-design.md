# LLM Context Dashboard — Design

**Date:** 2026-05-13
**Status:** Draft, pending approval
**Stacks on:** `feat/llm-context-logging` (the `llm_call_contexts` table and writer)

## Problem

The `llm_call_contexts` table now records every LLM round-trip for the
strategist, executor, and reflection loops. Today the only way to read
it is to write ad-hoc SQL. That makes forensic replay slow (re-formatting
JSON in psql) and prompt-tuning blind (no easy way to scan a whole
session's worth of transcripts).

We need a dashboard surface that turns these rows into something
readable.

## Scope

Surface the new table in the legacy `dashboard/` app (Flask, served on
port 3000 / 3001 paper). That dashboard is the existing debug/operations
UI — it already has `/session/<id>`, `/decision/<id>`, `/costs/<id>`,
`/events`. The v2 `dashboard/` is for public-facing publishing and is
not the right place for raw-transcript inspection.

In scope:

- A new "LLM Calls" section on the existing `/session/<id>` page listing
  every captured row for that session.
- A new detail page `/llm-call/<int:id>` rendering a single row with
  structured sections.

Out of scope:

- Search / filter across sessions
- Side-by-side transcript diffing
- Pagination (a session has at most ~50 rows; we render them all)
- Token-level streaming or per-block timing
- Changes to `v2/dashboard/`

## Design

### Routes

**Modified: `/session/<int:session_id>`** (in `dashboard/app.py`)

Pass a new `llm_calls` list to the template. Render a new section at the
bottom of `session_detail.html` titled "LLM Calls". Rows are ordered by
`(purpose, sequence)` so the executor appears first, then strategist
turns in order, then reflection turns. Columns: `purpose`, `sequence`,
`model`, `input_tokens / output_tokens`, `stop_reason`, `duration_ms`,
and a `[view]` link to the detail page.

**New: `/llm-call/<int:id>`**

Renders one row via `llm_call_detail.html`. 404 when no such id. Sections:

1. **Header summary** — links back to the parent session (`/session/<id>`),
   shows stage_name, purpose, sequence, model, duration_ms,
   stop_reason, token totals (input / output / cache_read / cache_creation).
2. **System prompt** — collapsed `<details>` block, contents in `<pre>`.
   Skip the section if `system_prompt` is null.
3. **Tool definitions** — collapsed `<details>` block, pretty-printed
   JSON. Skip when null.
4. **Conversation transcript** — `messages` is a JSONB array. For each
   message, render a role chip (`user` / `assistant`) and then walk the
   content blocks:
   - String content → single `<pre>` block.
   - `text` block → `<pre>` of the text.
   - `tool_use` block → "Tool call: `<name>` (id=`<id>`)" header, then
     collapsed JSON of `input`.
   - `tool_result` block → "Tool result for `<tool_use_id>`" header,
     then collapsed content. Content may be a string or a list of
     blocks; render uniformly.
   - Unknown block type → "Unknown block type: `<type>`" header, then
     collapsed JSON of the whole block (defensive — we should not crash
     on future block types).
5. **Assistant response** — render `response_content` with the same
   per-block walker used in section 4.

The transcript and response sections both reuse one Jinja macro
`render_blocks(blocks)` so the rendering logic lives in one place.

### Queries (in `dashboard/queries.py`)

Two new functions. Pattern matches the existing helpers in the file —
raw SQL via `get_cursor()`, returning dicts.

```python
def get_session_llm_calls(session_id: int) -> list[dict]:
    """Summary projection of llm_call_contexts for one session.

    Excludes the JSONB columns (messages, tool_definitions,
    response_content) because they are large and the list view does not
    need them.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, stage_name, purpose, sequence, model,
                   input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens,
                   stop_reason, duration_ms, created_at
            FROM llm_call_contexts
            WHERE session_id = %s
            ORDER BY
              CASE purpose
                WHEN 'executor' THEN 0
                WHEN 'strategist_loop' THEN 1
                WHEN 'reflection_loop' THEN 2
                ELSE 3
              END,
              sequence
            """,
            (session_id,),
        )
        return cur.fetchall()


def get_llm_call(call_id: int) -> dict | None:
    """Full row including JSONB columns. Returns None when not found."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM llm_call_contexts
            WHERE id = %s
            """,
            (call_id,),
        )
        return cur.fetchone()
```

`get_session_llm_calls` ordering puts executor first, then strategist
turns in sequence, then reflection turns. Stable and readable.

### Templates

**Modified: `dashboard/templates/session_detail.html`**

Append a new section near the end of the page, after the existing
sections (stages, decisions, theses, etc.):

```html
<section class="llm-calls">
  <h2>LLM Calls ({{ llm_calls|length }})</h2>
  {% if llm_calls %}
    <table>
      <thead>
        <tr>
          <th>Purpose</th><th>Seq</th><th>Stage</th><th>Model</th>
          <th>In/Out</th><th>Cache R/W</th><th>Stop</th><th>Dur (ms)</th><th></th>
        </tr>
      </thead>
      <tbody>
        {% for c in llm_calls %}
        <tr>
          <td>{{ c.purpose }}</td>
          <td>{{ c.sequence }}</td>
          <td>{{ c.stage_name }}</td>
          <td>{{ c.model }}</td>
          <td>{{ c.input_tokens }} / {{ c.output_tokens }}</td>
          <td>{{ c.cache_read_tokens or 0 }} / {{ c.cache_creation_tokens or 0 }}</td>
          <td>{{ c.stop_reason or '-' }}</td>
          <td>{{ c.duration_ms }}</td>
          <td><a href="{{ url_for('llm_call_detail', id=c.id) }}">view</a></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p>No LLM calls captured for this session.</p>
  {% endif %}
</section>
```

(Styling follows whatever `base.html` already provides; no new CSS.)

**New: `dashboard/templates/llm_call_detail.html`**

Extends `base.html`. Renders the five sections described above. The
`render_blocks` macro is defined at the top:

```jinja
{% macro render_blocks(blocks) %}
  {% if blocks is string %}
    <pre>{{ blocks }}</pre>
  {% else %}
    {% for block in blocks %}
      {% if block.type == 'text' %}
        <pre>{{ block.text }}</pre>
      {% elif block.type == 'tool_use' %}
        <div class="block tool-use">
          <strong>Tool call:</strong> <code>{{ block.name }}</code>
          <small>(id={{ block.id }})</small>
          <details><summary>input</summary>
            <pre>{{ block.input | tojson(indent=2) }}</pre>
          </details>
        </div>
      {% elif block.type == 'tool_result' %}
        <div class="block tool-result">
          <strong>Tool result</strong>
          <small>(tool_use_id={{ block.tool_use_id }})</small>
          <details><summary>content</summary>
            {{ render_blocks(block.content) }}
          </details>
        </div>
      {% else %}
        <div class="block unknown">
          <strong>Unknown block type:</strong> <code>{{ block.type }}</code>
          <details><summary>raw</summary>
            <pre>{{ block | tojson(indent=2) }}</pre>
          </details>
        </div>
      {% endif %}
    {% endfor %}
  {% endif %}
{% endmacro %}
```

`render_blocks` recurses through `tool_result.content` when that content
is itself a list of blocks (the anthropic SDK shape).

### Navigation

- Session detail page has `[view]` links to `/llm-call/<id>` per row.
- Detail page has `← back to session <session_id>` link at the top.
- No top-nav entry — the data is session-scoped, you discover it via the
  session page.

### Error handling

- `get_llm_call(id)` returns `None` for a missing id → route calls
  `abort(404)`, same pattern used by `/decision/<id>` (`dashboard/app.py:170`).
- `get_session_llm_calls(id)` returns `[]` when the session has no
  captured rows (CLI runs, old sessions before this feature). Template
  renders "No LLM calls captured for this session."
- Malformed JSONB (in theory impossible — we wrote it) is caught at the
  template layer via the `unknown` block fallback. We do not validate
  schemas; the table is append-only and trusted.

## Components

### `dashboard/queries.py`
Add `get_session_llm_calls` and `get_llm_call`. No other changes.

### `dashboard/app.py`
- Modify `session_detail` to fetch and pass `llm_calls`.
- Add `llm_call_detail` route at `/llm-call/<int:id>`.

### `dashboard/templates/session_detail.html`
Append the "LLM Calls" section.

### `dashboard/templates/llm_call_detail.html`
New template.

## Data flow

```
GET /session/42
  → session_detail(42)
    → get_session(42)
    → get_session_llm_calls(42)              [new]
    → other existing fetches
    → render session_detail.html with llm_calls

GET /llm-call/137
  → llm_call_detail(137)
    → get_llm_call(137)                       [new]
    → 404 if None
    → render llm_call_detail.html with row
```

## Testing

Tests live in `tests/test_dashboard.py` (or wherever the dashboard tests
currently sit — confirm path during implementation). Memory notes that
dashboard tests inject a mock via `sys.modules["queries"]` because
`dashboard/app.py` does bare `from queries import ...`.

1. **`get_session_llm_calls` projection** — given mocked rows, returns a
   list of dicts with the expected keys and excludes the JSONB columns.
2. **`get_session_llm_calls` ordering** — given mixed-purpose rows,
   returns executor → strategist_loop (in sequence order) → reflection_loop
   (in sequence order).
3. **`get_llm_call` returns row** — single id lookup returns the full
   row including JSONB.
4. **`get_llm_call` missing id** — returns `None`.
5. **Route `GET /session/<id>`** — response includes the "LLM Calls"
   header.
6. **Route `GET /session/<id>` empty state** — when the session has no
   captured rows, the response includes "No LLM calls captured for this
   session."
7. **Route `GET /llm-call/<id>` happy path** — response contains the
   model, stop_reason, and at least one rendered message.
8. **Route `GET /llm-call/<id>` 404** — non-existent id returns 404.
9. **Template `render_blocks` macro** — given a payload containing a
   `text` block, a `tool_use` block, a `tool_result` block whose
   content is a list of `text` blocks, and an unknown block type, the
   rendered HTML contains all four. (Implemented as a route-level test
   that feeds the mock query a row with all block types.)

## Risks

**Large payloads.** A strategist transcript can be ~1 MB. The detail
page renders the whole thing inline. The browser handles it fine
(modern browsers comfortably render multi-MB DOMs), but the initial
HTML response could be slow over a slow connection. Mitigation: the
list page does not pull JSONB at all, and the detail page collapses all
JSON blocks behind `<details>` so the initial paint is light. We
accept the size — the dashboard is local.

**XSS via tool inputs.** Tool inputs are LLM-controlled strings.
Jinja's autoescape is on by default in Flask; using `{{ ... }}`
and `tojson` is safe. We do not use `|safe` anywhere in this code.

**Schema drift.** If anthropic adds a new content block type, the
`unknown` branch keeps the page rendering. We do not need to update
the template every time a new block type appears.

## Open questions

None.
