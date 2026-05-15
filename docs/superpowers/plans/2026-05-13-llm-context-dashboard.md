# LLM Context Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the `llm_call_contexts` table in the legacy `dashboard/` debug UI: an "LLM Calls" section on `/session/<id>` plus a new `/llm-call/<id>` detail page with structured per-block transcript rendering.

**Architecture:** Two new queries in `dashboard/queries.py`, a new route + template in `dashboard/app.py` + `dashboard/templates/`, and a section appended to the existing `session_detail.html`. The detail template uses a single Jinja macro `render_blocks` to recursively render text / tool_use / tool_result / unknown content blocks. List view excludes JSONB columns so it stays cheap; detail view pulls the full row.

**Tech Stack:** Python 3, Flask, Jinja2, psycopg2 (raw SQL via `get_cursor()`), pytest. Dashboard tests inject a mock `queries` module via `sys.modules` and assert on rendered HTML bytes.

**Spec:** `docs/superpowers/specs/2026-05-13-llm-context-dashboard-design.md`

**Branch:** `feat/llm-context-dashboard` (already checked out, stacks on `feat/llm-context-logging`).

---

## Task 1: Add `get_session_llm_calls` query

**Files:**
- Modify: `dashboard/queries.py` (append a new function near the other `get_session_*` helpers around line 614)
- Test: `tests/test_dashboard_queries.py` (append a new test class)

Strict TDD.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_queries.py`:

```python
class TestGetSessionLlmCalls:
    def test_returns_summary_projection_excluding_jsonb(self, cur):
        from dashboard.queries import get_session_llm_calls
        cur.fetchall.return_value = [
            {
                "id": 1, "stage_name": "trading", "purpose": "executor",
                "sequence": 0, "model": "claude-haiku-4-5-20251001",
                "input_tokens": 120, "output_tokens": 45,
                "cache_read_tokens": 80, "cache_creation_tokens": 10,
                "stop_reason": "end_turn", "duration_ms": 987,
                "created_at": datetime(2026, 5, 13, 16, 30),
            }
        ]
        result = get_session_llm_calls(42)
        assert len(result) == 1
        assert result[0]["model"] == "claude-haiku-4-5-20251001"
        sql = cur.execute.call_args[0][0]
        # The SELECT must list only summary columns; JSONB columns absent.
        assert "messages" not in sql
        assert "tool_definitions" not in sql
        assert "response_content" not in sql
        # And the predicate is on session_id.
        assert "session_id = %s" in sql

    def test_orders_executor_then_strategist_then_reflection(self, cur):
        from dashboard.queries import get_session_llm_calls
        cur.fetchall.return_value = []
        get_session_llm_calls(42)
        sql = cur.execute.call_args[0][0]
        # The CASE expression must put executor=0, strategist_loop=1, reflection_loop=2
        # and then order by sequence.
        assert "executor" in sql
        assert "strategist_loop" in sql
        assert "reflection_loop" in sql
        assert "ORDER BY" in sql
        # Sequence is the tie-breaker.
        assert "sequence" in sql

    def test_passes_session_id_as_param(self, cur):
        from dashboard.queries import get_session_llm_calls
        cur.fetchall.return_value = []
        get_session_llm_calls(99)
        params = cur.execute.call_args[0][1]
        assert params == (99,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestGetSessionLlmCalls -v`
Expected: ImportError on `get_session_llm_calls`.

- [ ] **Step 3: Add the query**

Append to `dashboard/queries.py` after `get_session_events` (around line 617):

```python
def get_session_llm_calls(session_id: int):
    """Return summary rows from llm_call_contexts for one session.

    Excludes the JSONB columns (messages, tool_definitions,
    response_content) because they are large and the list view does not
    need them. Ordering puts the executor row first, then strategist
    loop turns in sequence, then reflection loop turns in sequence.
    """
    with get_cursor() as cur:
        cur.execute("""
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
        """, (session_id,))
        return cur.fetchall()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestGetSessionLlmCalls -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add get_session_llm_calls query"
```

---

## Task 2: Add `get_llm_call` query

**Files:**
- Modify: `dashboard/queries.py` (append after `get_session_llm_calls`)
- Test: `tests/test_dashboard_queries.py` (append a new test class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_queries.py`:

```python
class TestGetLlmCall:
    def test_returns_full_row_including_jsonb(self, cur):
        from dashboard.queries import get_llm_call
        cur.fetchone.return_value = {
            "id": 137, "session_id": 42, "stage_name": "trading",
            "purpose": "executor", "sequence": 0,
            "model": "claude-haiku-4-5-20251001",
            "system_prompt": "you are a trading executor",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_definitions": None,
            "response_content": [{"type": "text", "text": "ok"}],
            "input_tokens": 120, "output_tokens": 45,
            "cache_read_tokens": 80, "cache_creation_tokens": 10,
            "stop_reason": "end_turn", "duration_ms": 987,
            "created_at": datetime(2026, 5, 13, 16, 30),
        }
        result = get_llm_call(137)
        assert result["id"] == 137
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["response_content"] == [{"type": "text", "text": "ok"}]
        sql = cur.execute.call_args[0][0]
        assert "FROM llm_call_contexts" in sql
        assert "id = %s" in sql

    def test_returns_none_when_missing(self, cur):
        from dashboard.queries import get_llm_call
        cur.fetchone.return_value = None
        result = get_llm_call(99999)
        assert result is None

    def test_passes_id_as_param(self, cur):
        from dashboard.queries import get_llm_call
        cur.fetchone.return_value = None
        get_llm_call(137)
        assert cur.execute.call_args[0][1] == (137,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestGetLlmCall -v`
Expected: ImportError on `get_llm_call`.

- [ ] **Step 3: Add the query**

Append to `dashboard/queries.py`:

```python
def get_llm_call(call_id: int):
    """Return a single llm_call_contexts row by id, or None.

    Returns all columns including the JSONB payloads (messages,
    tool_definitions, response_content) — the detail view needs them.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, session_id, stage_name, purpose, sequence, model,
                   system_prompt, messages, tool_definitions,
                   response_content,
                   input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens,
                   stop_reason, duration_ms, created_at
            FROM llm_call_contexts
            WHERE id = %s
        """, (call_id,))
        return cur.fetchone()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestGetLlmCall -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add get_llm_call query for detail view"
```

---

## Task 3: Wire `get_session_llm_calls` into `/session/<id>`

**Files:**
- Modify: `dashboard/app.py` (the `session_detail` function around line 234)
- Modify: `dashboard/app.py` (the top-level `from queries import ...` block around line 10)
- Modify: `dashboard/templates/session_detail.html` (append new section)
- Test: `tests/test_dashboard.py` (extend `TestSessionDetail` class)

- [ ] **Step 1: Write the failing test**

Append to `TestSessionDetail` in `tests/test_dashboard.py` (around line 1247):

```python
    def test_renders_llm_calls_section_when_present(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = []
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        mock_queries.get_session_llm_calls.return_value = [
            {
                "id": 137, "stage_name": "trading", "purpose": "executor",
                "sequence": 0, "model": "claude-haiku-4-5-20251001",
                "input_tokens": 120, "output_tokens": 45,
                "cache_read_tokens": 80, "cache_creation_tokens": 10,
                "stop_reason": "end_turn", "duration_ms": 987,
                "created_at": datetime(2026, 5, 13, 16, 30),
            },
        ]
        resp = client.get("/session/42")
        assert resp.status_code == 200
        assert b"LLM Calls" in resp.data
        assert b"executor" in resp.data
        # The view link points at /llm-call/137
        assert b'href="/llm-call/137"' in resp.data

    def test_renders_empty_state_when_no_llm_calls(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = []
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        mock_queries.get_session_llm_calls.return_value = []
        resp = client.get("/session/42")
        assert resp.status_code == 200
        assert b"No LLM calls captured for this session." in resp.data
```

Also add a `get_session_llm_calls` safe default to the `_reset_query_mocks` autouse fixture in `tests/test_dashboard.py` (around line 108-114 where the other session defaults live). Append:

```python
    mock_queries.get_session_llm_calls.return_value = []
```

This keeps every other `TestSessionDetail` test from failing once the route starts fetching this query.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard.py::TestSessionDetail -v`
Expected: the two new tests fail (HTML doesn't contain "LLM Calls" yet), existing tests still pass.

- [ ] **Step 3: Add the import in `dashboard/app.py`**

In the top-level `from queries import ...` block (around line 10), add `get_llm_call,` and `get_session_llm_calls,` in alphabetical order. After the change, the block contains:

```python
from queries import (
    close_thesis,
    get_agent_event_types,
    ...
    get_llm_call,
    ...
    get_session_llm_calls,
    ...
)
```

(Keep alphabetical order — `get_llm_call` between `get_latest_snapshot` and `get_open_orders`; `get_session_llm_calls` between `get_session_events` and `get_session_memo`.)

- [ ] **Step 4: Fetch llm_calls in `session_detail` and pass to template**

In `dashboard/app.py`, update `session_detail` (around line 234):

```python
@app.route("/session/<int:session_id>")
def session_detail(session_id):
    """Unified per-session view: stages, events, decisions, theses, tweets, memo, LLM calls."""
    session = get_session(session_id)
    if not session:
        abort(404)
    stages = get_session_stage_costs(session_id)
    decisions = get_session_decisions(session_id)
    theses_created = get_session_theses_created(session_id)
    memo = get_session_memo(session_id)
    tweets = get_session_tweets(session_id)
    events = get_session_events(session_id, limit=200)
    llm_calls = get_session_llm_calls(session_id)
    return render_template(
        "session_detail.html",
        session=session,
        stages=stages,
        decisions=decisions,
        theses_created=theses_created,
        memo=memo,
        tweets=tweets,
        events=events,
        llm_calls=llm_calls,
    )
```

- [ ] **Step 5: Add the section to `session_detail.html`**

Find the bottom of the `{% block content %}` body in `dashboard/templates/session_detail.html` (just before the `{% endblock %}` that closes content). Append the new section:

```html
<section class="llm-calls">
  <h2>LLM Calls ({{ llm_calls|length }})</h2>
  {% if llm_calls %}
    <table>
      <thead>
        <tr>
          <th>Purpose</th>
          <th>Seq</th>
          <th>Stage</th>
          <th>Model</th>
          <th>In/Out</th>
          <th>Cache R/W</th>
          <th>Stop</th>
          <th>Dur (ms)</th>
          <th></th>
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

Note: `url_for('llm_call_detail', id=c.id)` refers to a route that is added in Task 4. The session_detail tests in Step 1 above assert on the resulting URL string `/llm-call/137`. Until Task 4 lands, the route does not exist — but `url_for` only resolves at render time, and Flask raises `BuildError` if the endpoint is unknown.

To keep these tests passing before Task 4 wires the route, we temporarily inline the URL with a literal until Task 4 swaps it:

```html
<td><a href="/llm-call/{{ c.id }}">view</a></td>
```

(Task 4 will update this back to `url_for(...)` once the route exists.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard.py::TestSessionDetail -v`
Expected: all 5 tests pass (3 pre-existing + 2 new).

- [ ] **Step 7: Commit**

```bash
git add dashboard/app.py dashboard/templates/session_detail.html tests/test_dashboard.py
git commit -m "feat(dashboard): add LLM Calls section to /session/<id>"
```

---

## Task 4: Add `/llm-call/<id>` route and detail template

**Files:**
- Modify: `dashboard/app.py` (add a new route)
- Create: `dashboard/templates/llm_call_detail.html`
- Modify: `dashboard/templates/session_detail.html` (swap literal URL back to `url_for`)
- Test: `tests/test_dashboard.py` (append a new test class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# LLM call detail
# ---------------------------------------------------------------------------


class TestLlmCallDetail:
    def _make_row(self, **overrides):
        row = {
            "id": 137,
            "session_id": 42,
            "stage_name": "trading",
            "purpose": "executor",
            "sequence": 0,
            "model": "claude-haiku-4-5-20251001",
            "system_prompt": "you are a trading executor",
            "messages": [
                {"role": "user", "content": "executor input json"},
            ],
            "tool_definitions": None,
            "response_content": [
                {"type": "text", "text": "decision json here"},
            ],
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_read_tokens": 80,
            "cache_creation_tokens": 10,
            "stop_reason": "end_turn",
            "duration_ms": 987,
            "created_at": datetime(2026, 5, 13, 16, 30),
        }
        row.update(overrides)
        return row

    def test_returns_200_with_header_fields(self, client):
        mock_queries.get_llm_call.return_value = self._make_row()
        resp = client.get("/llm-call/137")
        assert resp.status_code == 200
        # Header content
        assert b"executor" in resp.data
        assert b"claude-haiku-4-5-20251001" in resp.data
        assert b"end_turn" in resp.data
        # Token totals
        assert b"120" in resp.data
        assert b"45" in resp.data
        # Back link to the parent session
        assert b'href="/session/42"' in resp.data

    def test_renders_system_prompt(self, client):
        mock_queries.get_llm_call.return_value = self._make_row()
        resp = client.get("/llm-call/137")
        assert b"you are a trading executor" in resp.data

    def test_skips_system_prompt_section_when_null(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(system_prompt=None)
        resp = client.get("/llm-call/137")
        # The H3 / summary text should be absent when the value is null.
        assert b"System prompt" not in resp.data

    def test_renders_tool_definitions_when_present(self, client):
        tools = [{"name": "get_positions", "input_schema": {"type": "object"}}]
        mock_queries.get_llm_call.return_value = self._make_row(tool_definitions=tools)
        resp = client.get("/llm-call/137")
        assert b"Tool definitions" in resp.data
        assert b"get_positions" in resp.data

    def test_skips_tool_definitions_when_null(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(tool_definitions=None)
        resp = client.get("/llm-call/137")
        assert b"Tool definitions" not in resp.data

    def test_renders_text_message_in_transcript(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(
            messages=[{"role": "user", "content": "hello executor"}],
        )
        resp = client.get("/llm-call/137")
        assert b"hello executor" in resp.data

    def test_renders_tool_use_and_tool_result_and_unknown_blocks(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "get_positions", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "positions json"}],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "future_block_type", "foo": "bar"},
                    ],
                },
            ],
        )
        resp = client.get("/llm-call/137")
        # tool_use rendered with name and id
        assert b"get_positions" in resp.data
        assert b"t1" in resp.data
        # tool_result content is reachable
        assert b"positions json" in resp.data
        # Unknown block falls back rather than crashing
        assert b"Unknown block type" in resp.data
        assert b"future_block_type" in resp.data

    def test_renders_response_content(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(
            response_content=[{"type": "text", "text": "executor decision json"}],
        )
        resp = client.get("/llm-call/137")
        assert b"executor decision json" in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_llm_call.return_value = None
        resp = client.get("/llm-call/99999")
        assert resp.status_code == 404
```

Also add a `get_llm_call` safe default to the `_reset_query_mocks` autouse fixture (alongside the `get_session_llm_calls` default added in Task 3):

```python
    mock_queries.get_llm_call.return_value = None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard.py::TestLlmCallDetail -v`
Expected: 9 tests fail. Likely error: `BuildError` from `url_for` in session_detail.html if Task 3 used `url_for`; or 404 from missing route on the detail tests if Task 3 used the literal URL. The literal URL approach in Task 3 means these fail with 404.

- [ ] **Step 3: Add the route in `dashboard/app.py`**

Add this route after the `decision_detail` route (around line 194). The endpoint name is `llm_call_detail` (matches the `url_for(...)` we'll switch back to in step 5):

```python
@app.route("/llm-call/<int:id>")
def llm_call_detail(id):
    """Render one LLM round-trip: system prompt, messages, response."""
    row = get_llm_call(id)
    if row is None:
        abort(404)
    return render_template("llm_call_detail.html", row=row)
```

- [ ] **Step 4: Create the detail template**

Create `dashboard/templates/llm_call_detail.html`:

```html
{% extends "base.html" %}

{% block title %}LLM Call #{{ row.id }}{% endblock %}

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

{% block content %}
<a href="{{ url_for('session_detail', session_id=row.session_id) }}">&larr; back to session #{{ row.session_id }}</a>

<h1>LLM Call #{{ row.id }}</h1>

<section class="header">
  <dl>
    <dt>Session</dt><dd><a href="{{ url_for('session_detail', session_id=row.session_id) }}">#{{ row.session_id }}</a></dd>
    <dt>Stage</dt><dd>{{ row.stage_name }}</dd>
    <dt>Purpose</dt><dd>{{ row.purpose }}</dd>
    <dt>Sequence</dt><dd>{{ row.sequence }}</dd>
    <dt>Model</dt><dd>{{ row.model }}</dd>
    <dt>Stop reason</dt><dd>{{ row.stop_reason or '-' }}</dd>
    <dt>Duration</dt><dd>{{ row.duration_ms }} ms</dd>
    <dt>Tokens in / out</dt><dd>{{ row.input_tokens }} / {{ row.output_tokens }}</dd>
    <dt>Cache read / creation</dt><dd>{{ row.cache_read_tokens or 0 }} / {{ row.cache_creation_tokens or 0 }}</dd>
    <dt>Captured at</dt><dd>{{ row.created_at }}</dd>
  </dl>
</section>

{% if row.system_prompt %}
<section class="system-prompt">
  <h2>System prompt</h2>
  <details open>
    <summary>show</summary>
    <pre>{{ row.system_prompt }}</pre>
  </details>
</section>
{% endif %}

{% if row.tool_definitions %}
<section class="tool-definitions">
  <h2>Tool definitions</h2>
  <details>
    <summary>show</summary>
    <pre>{{ row.tool_definitions | tojson(indent=2) }}</pre>
  </details>
</section>
{% endif %}

<section class="transcript">
  <h2>Conversation transcript</h2>
  {% for msg in row.messages %}
    <div class="message role-{{ msg.role }}">
      <h3>{{ msg.role }}</h3>
      {{ render_blocks(msg.content) }}
    </div>
  {% endfor %}
</section>

<section class="response">
  <h2>Assistant response</h2>
  {{ render_blocks(row.response_content) }}
</section>
{% endblock %}
```

- [ ] **Step 5: Swap the literal URL in `session_detail.html` to `url_for`**

In `dashboard/templates/session_detail.html`, find the LLM Calls table row added in Task 3:

```html
<td><a href="/llm-call/{{ c.id }}">view</a></td>
```

Replace with:

```html
<td><a href="{{ url_for('llm_call_detail', id=c.id) }}">view</a></td>
```

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_dashboard.py::TestLlmCallDetail tests/test_dashboard.py::TestSessionDetail -v`
Expected: 9 detail tests pass + 5 session tests pass.

- [ ] **Step 7: Run the broader dashboard suite for regressions**

Run: `python3 -m pytest tests/test_dashboard.py tests/test_dashboard_queries.py -v 2>&1 | tail -30`
Expected: all green. No pre-existing test broken by the `get_session_llm_calls` / `get_llm_call` mocks (their safe defaults are `[]` and `None`).

- [ ] **Step 8: Commit**

```bash
git add dashboard/app.py dashboard/templates/llm_call_detail.html dashboard/templates/session_detail.html tests/test_dashboard.py
git commit -m "feat(dashboard): add /llm-call/<id> detail view"
```

---

## Task 5: Live-stack smoke test

**Files:** none (manual validation)

Validates that the routes render against the real paper DB.

- [ ] **Step 1: Restart the paper dashboard so it picks up code changes**

The paper dashboard service mounts `dashboard/` as a bind mount but Flask in production mode does NOT auto-reload. Check whether the running container needs a restart:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml restart dashboard-paper
```

Wait for it to come up:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml ps dashboard-paper
```

Expected: `Up`.

- [ ] **Step 2: Find a session id that has captured rows**

The migration has been applied to paper DB but no session has run since the wire-in. If a session was run between then and now, find it:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec db-paper bash -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
    SELECT session_id, COUNT(*) AS n_rows
    FROM llm_call_contexts
    GROUP BY session_id
    ORDER BY session_id DESC
    LIMIT 5
  "'
```

If no rows exist yet (no real session has run since the wire-in), insert a synthetic row to validate the dashboard rendering. Pick the latest session id from `sessions` and run:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec trading-paper python3 -c "
from v2.database.trading_db import insert_llm_call_context
from v2.database.connection import get_cursor
with get_cursor() as cur:
    cur.execute('SELECT id FROM sessions ORDER BY id DESC LIMIT 1')
    sid = cur.fetchone()['id']
    insert_llm_call_context(
        session_id=sid, stage_name='trading', purpose='executor',
        model='claude-haiku-4-5-20251001',
        system_prompt='dashboard smoke',
        messages=[{'role': 'user', 'content': 'smoke input'}],
        tool_definitions=None,
        response_content=[{'type': 'text', 'text': 'smoke response'}],
        input_tokens=5, output_tokens=2,
        cache_read_tokens=0, cache_creation_tokens=0,
        stop_reason='end_turn', duration_ms=42,
    )
    print('inserted into session', sid)
"
```

- [ ] **Step 3: Hit the session page**

```bash
curl -s http://127.0.0.1:3001/session/<SID> | grep -E "LLM Calls|llm-call" | head -5
```

Expected: the heading "LLM Calls" appears, and at least one `href="/llm-call/<id>"` link.

- [ ] **Step 4: Hit the detail page**

Find the inserted row id and curl the detail page:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec db-paper bash -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT id FROM llm_call_contexts WHERE stage_name=\"trading\" ORDER BY id DESC LIMIT 1"'
```

```bash
curl -s http://127.0.0.1:3001/llm-call/<ID> | grep -E "smoke|executor|claude-haiku" | head -5
```

Expected: the smoke text, the purpose, and the model are visible.

- [ ] **Step 5: Clean up the smoke row (only if you inserted one in Step 2)**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec db-paper bash -c \
  "psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -c \"DELETE FROM llm_call_contexts WHERE system_prompt = 'dashboard smoke'\""
```

Expected: `DELETE 1`.

- [ ] **Step 6: No commits expected.** This task is validation only.

---

## Self-review notes

- **Spec coverage:** modified `/session/<id>` (Task 3), new `/llm-call/<id>` (Task 4), `get_session_llm_calls` projection excluding JSONB (Task 1), `get_llm_call` full row (Task 2), per-block rendering with `render_blocks` macro covering text / tool_use / tool_result / unknown (Task 4 Step 4), 404 on missing detail (Task 4 Step 1 last test), empty state message (Task 3 Step 1 second test), back-link to session (Task 4 Step 1 first test), live-stack smoke test (Task 5). No spec section unmapped.
- **No placeholders:** every step has actual code or commands with expected output.
- **Type consistency:** `get_session_llm_calls` returns a list of summary dicts (Tasks 1, 3); `get_llm_call` returns a single full row or None (Tasks 2, 4); endpoint name `llm_call_detail` consistent in Tasks 3 (literal URL placeholder), 4 (route + url_for swap-back); template macro `render_blocks` is defined once and used twice (transcript + response) within the same template.
- **Test infra:** dashboard tests run on host Python (no v2 dependency chain) — should not hit the Python 3.10/3.11 issue documented in memory. If they do, the same docker fallback (`docker compose -f docker-compose.yml -f docker-compose.paper.yml exec trading-paper python3 -m pytest tests/test_dashboard.py`) applies, copy `tests/` and `dashboard/` over first.
- **Frequent commits:** 4 commits across Tasks 1-4. Task 5 is validation only.
