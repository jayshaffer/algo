# Strategist Cache-Thrashing Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate prompt-cache thrashing in the strategist's agentic loop by freezing tool-result truncation at addition time, dropping `cache_creation_tokens` ~50–60% per run.

**Architecture:** Two phases. Phase 1 adds `output_chars` to existing `tool_invocation` telemetry and gathers per-tool size data from one paper run. Phase 2 deletes the per-turn `_truncate_old_tool_results` (which mutates message history each turn and breaks the cache prefix), replaces it with a one-shot `_truncate_tool_result_blocks` helper called when tool results are first appended to `messages`, and sets the truncation threshold based on Phase 1 data.

**Tech Stack:** Python 3, anthropic SDK, pytest, PostgreSQL (for telemetry queries), Docker Compose paper pipeline.

**Spec:** `docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md`

---

## File Structure

**Modified:**
- `v2/claude_client.py` — telemetry add (Phase 1) + truncation refactor (Phase 2)
- `tests/v2/test_claude_client.py` — new tests for both phases
- `docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md` — fill in TBD threshold/data sections after Phase 1

**No new files.**

---

## Phase 1 — Measurement

### Task 1: Add `output_chars` to `tool_invocation` telemetry

**Files:**
- Modify: `v2/claude_client.py:498-509`
- Test: `tests/v2/test_claude_client.py` (append new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/v2/test_claude_client.py`:

```python
class TestToolInvocationOutputChars:
    """Phase 1: tool_invocation events must include output_chars so we can
    size-tune the cache-friendly truncation threshold from real data."""

    def test_tool_invocation_event_includes_output_chars(self, monkeypatch):
        """The tool_invocation payload should record len(result.content)."""
        from v2 import claude_client

        captured: list[dict] = []

        def fake_record_event(session_id, stage_name, event_type, payload):
            captured.append({"event_type": event_type, "payload": payload})

        monkeypatch.setattr(claude_client, "record_event", fake_record_event)

        # Two-turn loop: first response calls a tool that returns 1500 chars,
        # second response ends the turn so the loop exits cleanly.
        tool_response = _make_response(
            content=[
                _tool_use_block("call-1", "echo", {"text": "hi"}),
            ],
            stop_reason="tool_use",
        )
        end_response = _make_response(
            content=[_text_block("done")],
            stop_reason="end_turn",
        )
        client = _make_stream_mock([tool_response, end_response])

        big_payload = "x" * 1500

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="go",
            tools=[{"name": "echo", "description": "", "input_schema": {}}],
            tool_handlers={"echo": lambda **_: big_payload},
            max_turns=5,
            session_id=1,
            stage_name="strategist",
        )

        tool_events = [e for e in captured if e["event_type"] == "tool_invocation"]
        assert len(tool_events) == 1, f"expected 1 tool_invocation event, got {len(tool_events)}"
        payload = tool_events[0]["payload"]
        assert "output_chars" in payload, f"output_chars missing from payload: {payload}"
        assert payload["output_chars"] == 1500, (
            f"expected output_chars=1500, got {payload['output_chars']}"
        )
```

If `_make_stream_mock` is not yet imported at the test module level, add it to the existing imports at the top of the file. Check the helpers already defined in `tests/v2/test_claude_client.py` (`_make_response`, `_text_block`, `_tool_use_block`, `_make_stream_mock`) and reuse them — do not re-define.

- [ ] **Step 2: Run test to verify it fails**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py::TestToolInvocationOutputChars::test_tool_invocation_event_includes_output_chars -v
```

Expected: FAIL with `assert "output_chars" in payload` (the field doesn't exist yet).

- [ ] **Step 3: Add `output_chars` to the `tool_invocation` payload**

In `v2/claude_client.py`, modify the `record_event` call inside `run_agentic_loop` (currently at lines 498–509) to add the new field:

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
        "duration_ms": duration_ms,
        "output_chars": len(result.content),
    },
)
```

Only one line is added: `"output_chars": len(result.content),`. Everything else is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py::TestToolInvocationOutputChars::test_tool_invocation_event_includes_output_chars -v
```

Expected: PASS.

- [ ] **Step 5: Run the full claude_client test file to confirm no regressions**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py -v
```

Expected: all tests pass (existing tests + 1 new).

- [ ] **Step 6: Commit**

```
git add v2/claude_client.py tests/v2/test_claude_client.py
git commit -m "$(cat <<'EOF'
feat(telemetry): record output_chars in tool_invocation events

Adds the byte length of each tool_result to the tool_invocation
payload so we can size the cache-friendly truncation threshold
from observed per-tool size distributions. Phase 1 of the
strategist cache-thrashing fix.

Spec: docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Run a paper strategist session and capture per-tool size data

**Files:**
- Modify: `docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md` (fill in TBD section)

- [ ] **Step 1: Bring up the paper stack if it isn't already running**

Run:
```
docker ps --format "{{.Names}}" | grep -q algo-trading-paper-1 && echo "up" || task paper:up
```

Expected: prints `up`, or `paper:up` runs to completion. If `paper:up` runs, wait for `algo-db-paper-1` and `algo-trading-paper-1` containers to show healthy in `docker ps`.

- [ ] **Step 2: Trigger one paper strategist run**

You only need stages that exercise the strategist. The simplest approach is the existing full-paper task; the strategist will run and populate `agent_events`. The `--force` flag is required if a paper session has already run today.

Run:
```
task paper:session -- --force
```

Expected: command runs to completion (~3–6 min). Look for `[Stage 2] Running Claude strategist` and `Strategist Loop Complete` in the output. If the strategist stage fails, do not proceed — debug first.

- [ ] **Step 3: Identify the session_id from the run that just completed**

Run:
```
docker exec algo-db-paper-1 psql -U algo -d trading -c "SELECT id, started_at, status FROM sessions ORDER BY started_at DESC LIMIT 3;"
```

Expected: top row is the run you just triggered. Note the `id` value.

- [ ] **Step 4: Verify `output_chars` is being recorded**

Replace `<SESSION_ID>` with the id from Step 3:
```
docker exec algo-db-paper-1 psql -U algo -d trading -c "SELECT payload->>'tool_name' AS tool, payload->>'output_chars' AS chars FROM agent_events WHERE event_type = 'tool_invocation' AND session_id = <SESSION_ID> LIMIT 5;"
```

Expected: 5 rows with non-null `chars` values. If `chars` is null, the telemetry change from Task 1 didn't ship correctly — re-check the `record_event` payload.

- [ ] **Step 5: Compute the per-tool size distribution**

Run:
```
docker exec algo-db-paper-1 psql -U algo -d trading -c "
SELECT
  payload->>'tool_name' AS tool,
  COUNT(*) AS calls,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY (payload->>'output_chars')::int) AS p50,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY (payload->>'output_chars')::int) AS p90,
  MAX((payload->>'output_chars')::int) AS max
FROM agent_events
WHERE event_type = 'tool_invocation' AND stage_name = 'strategist'
GROUP BY 1 ORDER BY p90 DESC;
"
```

Expected: a row per tool name (`get_news_signals`, `web_search`, `get_market_snapshot`, `create_thesis`, etc.) with `p50`, `p90`, and `max` columns.

- [ ] **Step 6: Pick the threshold value**

Rule: pick a value at or slightly above the **p90 of the heaviest 2–3 tools** (the ones at the top of the sorted result). The intent is that small tools never get truncated and the heaviest tools retain a useful prefix.

Worked example: if `get_news_signals` p90 is 4200 and `web_search` p90 is 6800, pick `8000`. If both top tools p90 under 1500, pick `2000`. Round to a clean number.

Record the chosen value — it goes into the spec next, and into the constant in Task 3.

- [ ] **Step 7: Fill in the TBD section of the spec**

Open `docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md`. Find the block:

```
TBD — fill in after Phase 1 paper session
- p90 of get_news_signals: ____
- p90 of web_search: ____
- p90 of get_market_snapshot: ____
- chosen threshold: ____
```

Replace with the actual numbers from Step 5 + Step 6. If a tool wasn't called this run, write `not exercised`. If a different tool dominated, add it to the list.

- [ ] **Step 8: Commit the spec update**

```
git add docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md
git commit -m "$(cat <<'EOF'
docs(specs): record Phase 1 size data and chosen truncation threshold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Cache fix

### Task 3: Add `_truncate_tool_result_blocks` helper (TDD)

**Files:**
- Modify: `v2/claude_client.py` (add helper near the existing truncation code)
- Test: `tests/v2/test_claude_client.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/v2/test_claude_client.py`:

```python
class TestTruncateToolResultBlocks:
    """Phase 2: pure helper that clips oversized tool_result blocks once,
    at the moment they're appended to messages. Must NOT mutate input."""

    def test_small_tool_result_passes_through_unchanged(self):
        from v2.claude_client import _truncate_tool_result_blocks

        blocks = [
            {"type": "tool_result", "tool_use_id": "x", "content": "short"},
        ]
        out = _truncate_tool_result_blocks(blocks, threshold=2000)
        assert out == blocks

    def test_oversized_tool_result_is_clipped_with_marker(self):
        from v2.claude_client import _truncate_tool_result_blocks

        big = "x" * 5000
        blocks = [
            {"type": "tool_result", "tool_use_id": "x", "content": big},
        ]
        out = _truncate_tool_result_blocks(blocks, threshold=2000)
        assert out[0]["content"] == "x" * 2000 + "...[truncated]"
        assert out[0]["tool_use_id"] == "x"
        assert out[0]["type"] == "tool_result"

    def test_non_tool_result_blocks_pass_through(self):
        from v2.claude_client import _truncate_tool_result_blocks

        blocks = [
            {"type": "text", "text": "x" * 5000},
        ]
        out = _truncate_tool_result_blocks(blocks, threshold=2000)
        assert out == blocks

    def test_does_not_mutate_input(self):
        from v2.claude_client import _truncate_tool_result_blocks

        big = "x" * 5000
        blocks = [
            {"type": "tool_result", "tool_use_id": "x", "content": big},
        ]
        _truncate_tool_result_blocks(blocks, threshold=2000)
        assert blocks[0]["content"] == big, "input must not be mutated"

    def test_empty_list(self):
        from v2.claude_client import _truncate_tool_result_blocks
        assert _truncate_tool_result_blocks([], threshold=2000) == []

    def test_non_string_content_passes_through(self):
        """tool_result content can be a list of blocks (e.g. images). Only
        clip when content is a string longer than threshold."""
        from v2.claude_client import _truncate_tool_result_blocks

        blocks = [
            {"type": "tool_result", "tool_use_id": "x", "content": [{"type": "text", "text": "y" * 5000}]},
        ]
        out = _truncate_tool_result_blocks(blocks, threshold=2000)
        assert out == blocks
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py::TestTruncateToolResultBlocks -v
```

Expected: all 6 tests FAIL with `ImportError: cannot import name '_truncate_tool_result_blocks'`.

- [ ] **Step 3: Implement the helper**

In `v2/claude_client.py`, add the helper (place it directly above the existing `_truncate_old_tool_results` function, around line 281; we'll delete the old one in Task 4):

```python
def _truncate_tool_result_blocks(content_blocks: list[dict], threshold: int) -> list[dict]:
    """Clip oversized tool_result blocks to `threshold` chars + truncation marker.

    Pure function. Returns a new list; does not mutate the input. Called
    once when tool results are appended to `messages`, so the resulting
    blocks are frozen — never modified again. This is what keeps the
    Anthropic prompt-cache prefix byte-stable across turns of the agentic
    loop.

    Only clips `tool_result` blocks whose `content` is a string longer
    than `threshold`. Image/multi-block content (`content` as list) is
    passed through unchanged — string-length truncation isn't meaningful
    for those.
    """
    result = []
    for item in content_blocks:
        if (
            isinstance(item, dict)
            and item.get("type") == "tool_result"
            and isinstance(item.get("content"), str)
            and len(item["content"]) > threshold
        ):
            result.append({
                **item,
                "content": item["content"][:threshold] + "...[truncated]",
            })
        else:
            result.append(item)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py::TestTruncateToolResultBlocks -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```
git add v2/claude_client.py tests/v2/test_claude_client.py
git commit -m "$(cat <<'EOF'
feat(claude_client): add _truncate_tool_result_blocks helper

Pure function that clips oversized tool_result blocks once, at the
moment they enter messages. No history awareness — frozen on
addition. This is the building block for the cache-stability fix;
the swap-in happens in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire the helper into `run_agentic_loop`, delete the old per-turn truncator

**Files:**
- Modify: `v2/claude_client.py`
- Test: `tests/v2/test_claude_client.py`

- [ ] **Step 1: Write a failing prefix-stability test**

Append to `tests/v2/test_claude_client.py`:

```python
class TestMessagePrefixStability:
    """Phase 2: once a tool_result is appended to messages, it must
    never change again. Anthropic's prompt cache is byte-stable over
    the prefix; mutating an old message invalidates the cache from
    that point forward and re-bills as cache_creation."""

    def test_old_tool_results_are_not_modified_across_turns(self):
        """Drive a 5-turn agentic loop where each tool returns a 5000-char
        payload. After the loop, the messages list at index N must equal
        what it was right after turn N appended its tool_result — no later
        turn may have re-truncated it."""
        from v2.claude_client import run_agentic_loop

        big = "x" * 5000

        # Turns 1-4: tool_use response. Turn 5: end_turn.
        responses = []
        for i in range(4):
            responses.append(_make_response(
                content=[_tool_use_block(f"call-{i}", "echo", {"i": i})],
                stop_reason="tool_use",
            ))
        responses.append(_make_response(
            content=[_text_block("done")],
            stop_reason="end_turn",
        ))
        client = _make_stream_mock(responses)

        result = run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="go",
            tools=[{"name": "echo", "description": "", "input_schema": {}}],
            tool_handlers={"echo": lambda **_: big},
            max_turns=10,
        )

        # Inspect the final messages list. Every user message that contains
        # a tool_result should have the SAME truncated form — verifying no
        # mid-conversation re-truncation occurred.
        tool_result_msgs = [
            m for m in result.messages
            if m["role"] == "user" and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 4, f"expected 4 tool_result turns, got {len(tool_result_msgs)}"

        # Every tool_result block should have the same shape: clipped to
        # threshold + marker. None should still hold the full 5000 chars,
        # AND none should be over-truncated to the old 300-char value.
        for msg in tool_result_msgs:
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    content = block["content"]
                    assert content.endswith("...[truncated]"), (
                        f"expected truncation marker, got: {content[-30:]!r}"
                    )
                    # The clipped portion length must equal the threshold
                    # constant. Read it from the module so the test follows
                    # whatever value Phase 1 picked.
                    from v2.claude_client import _TOOL_RESULT_TRUNCATION_THRESHOLD
                    assert len(content) == _TOOL_RESULT_TRUNCATION_THRESHOLD + len("...[truncated]"), (
                        f"unexpected length {len(content)}"
                    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py::TestMessagePrefixStability -v
```

Expected: FAIL — either with `ImportError` for `_TOOL_RESULT_TRUNCATION_THRESHOLD` (constant not defined yet), or with truncation-length mismatch (old code clips to 300 chars, not the new threshold).

- [ ] **Step 3: Add the threshold constant**

In `v2/claude_client.py`, near the top (above the existing `_TRUNCATION_THRESHOLD` line, around line 240), add:

```python
_TOOL_RESULT_TRUNCATION_THRESHOLD = <CHOSEN_VALUE>  # chars; set from Phase 1 size distribution
```

Replace `<CHOSEN_VALUE>` with the integer recorded in the spec at the end of Task 2 (e.g. `2000`, `4000`, `8000` — whatever was chosen). Do not leave a placeholder.

- [ ] **Step 4: Wire the helper into the tool-result append site**

In `v2/claude_client.py`, find this line (currently around line 521):

```python
        messages.append({"role": "user", "content": tool_results})
```

Replace with:

```python
        messages.append({
            "role": "user",
            "content": _truncate_tool_result_blocks(
                tool_results, _TOOL_RESULT_TRUNCATION_THRESHOLD
            ),
        })
```

- [ ] **Step 5: Replace the per-turn truncator call**

In `v2/claude_client.py`, find this line (currently around line 363):

```python
        pruned = _truncate_old_tool_results(messages)
```

Replace with:

```python
        pruned = messages
```

(`pruned` is still the right local name — the variable is consumed two lines later by `_messages_with_cache_breakpoint(pruned)`. Keeping the name avoids a wider diff.)

- [ ] **Step 6: Delete the now-unused `_truncate_old_tool_results` function and constants**

In `v2/claude_client.py`:

1. Delete the constant `_TRUNCATION_THRESHOLD = 300` (currently line 241).
2. Delete the constant `_KEEP_RECENT_EXCHANGES = 3` (currently line 242).
3. Delete the entire function `_truncate_old_tool_results` (currently lines ~281–326). The block starts with `def _truncate_old_tool_results(messages: list[dict]) -> list[dict]:` and ends with `return result` followed by a blank line.

Search the file for any remaining references after deletion:

```
grep -n "_truncate_old_tool_results\|_TRUNCATION_THRESHOLD\|_KEEP_RECENT_EXCHANGES" v2/claude_client.py
```

Expected output: no matches. If any remain, delete those references (they would break the import or refer to the removed symbol).

Also search the test file:

```
grep -n "_truncate_old_tool_results\|_TRUNCATION_THRESHOLD\|_KEEP_RECENT_EXCHANGES" tests/v2/test_claude_client.py
```

Expected output: no matches. (The earlier check confirmed there are no existing tests on these symbols, but verify.)

- [ ] **Step 7: Run the new prefix-stability test to verify it passes**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py::TestMessagePrefixStability -v
```

Expected: PASS.

- [ ] **Step 8: Run the full claude_client test suite to confirm no regressions**

Run:
```
python3 -m pytest tests/v2/test_claude_client.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Run the full project test suite to catch any cross-module regressions**

Run:
```
python3 -m pytest tests/ -q
```

Expected: all tests pass. If anything fails outside `tests/v2/test_claude_client.py`, investigate before proceeding — the change is small enough that broader breakage suggests an import or call-site we missed.

- [ ] **Step 10: Commit**

```
git add v2/claude_client.py tests/v2/test_claude_client.py
git commit -m "$(cat <<'EOF'
fix(claude_client): freeze tool-result truncation at addition

Replaces _truncate_old_tool_results (which re-truncated the
message history every turn, invalidating Anthropic's prompt-cache
prefix on each newly-aged-out message) with a one-shot truncation
applied when tool results first enter messages. The messages list
is now byte-stable from the moment a turn is appended, so previously-
seen content reliably hits the cache as cache_read instead of being
re-billed as cache_creation.

Threshold _TOOL_RESULT_TRUNCATION_THRESHOLD set from observed p90
of the heaviest tools in a paper strategist run (Phase 1 data
captured in spec).

Spec: docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Validate the cost reduction on a paper run

**Files:**
- No code changes; validation only.

- [ ] **Step 1: Record the pre-fix baseline**

Before this branch's work merges, the strategist's pre-fix per-session cost is documented in the spec (avg ~$5.18 across sessions 3251 and 3405). No action needed — just confirm the values are recorded in the spec under "Validation".

- [ ] **Step 2: Trigger one post-fix paper strategist run**

Run:
```
task paper:session -- --force
```

Expected: completes successfully, including the strategist stage. Note the new session's id (top row of `SELECT id ... FROM sessions ORDER BY started_at DESC LIMIT 1` in `algo-db-paper-1`).

- [ ] **Step 3: Compute the post-fix cost**

Replace `<NEW_SESSION_ID>` with the id from Step 2:

```
docker exec algo-db-paper-1 psql -U algo -d trading -c "
WITH pricing AS (
  SELECT 'claude-opus-4-6' AS model, 15.0 AS in_per_m, 75.0 AS out_per_m, 18.75 AS cache_w_per_m, 1.50 AS cache_r_per_m
)
SELECT
  ss.session_id,
  ss.model,
  ss.input_tokens AS in_tok,
  ss.output_tokens AS out_tok,
  ss.cache_creation_tokens AS cw_tok,
  ss.cache_read_tokens AS cr_tok,
  ROUND((ss.input_tokens * p.in_per_m
       + ss.output_tokens * p.out_per_m
       + ss.cache_creation_tokens * p.cache_w_per_m
       + ss.cache_read_tokens * p.cache_r_per_m) / 1e6 :: numeric, 4) AS cost_usd
FROM session_stages ss
JOIN pricing p ON p.model = ss.model
WHERE ss.session_id = <NEW_SESSION_ID> AND ss.stage_name = 'strategist';
"
```

Expected: a single row with `cost_usd` ≤ $3.10. If `cost_usd` is > $3.10, the fix didn't deliver the targeted ≥40% reduction — investigate before declaring done (likely indicates either threshold is too tight forcing extra turns, or there's another driver of cache_creation we missed).

- [ ] **Step 4: Sanity-check the playbook quality**

Replace `<SESSION_DATE>` with today's date in `YYYY-MM-DD` format (or use the date the post-fix session ran):

```
docker exec algo-db-paper-1 psql -U algo -d trading -c "
SELECT id, action, ticker, rationale
FROM playbook_actions
WHERE playbook_id = (SELECT id FROM playbooks WHERE session_date = '<SESSION_DATE>' ORDER BY id DESC LIMIT 1)
ORDER BY id;
"
```

Expected: ≥1 row, with non-empty `rationale`. Eyeball: does the rationale read coherently? If actions are missing or rationale is gibberish/empty, the threshold may be too tight — raise `_TOOL_RESULT_TRUNCATION_THRESHOLD` (e.g., double it), commit, re-run.

- [ ] **Step 5: Record validation result in the spec**

Append a "Validation results" section at the end of `docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md`:

```markdown
## Validation results (post-fix)

- Post-fix session: <NEW_SESSION_ID> on <DATE>
- cache_creation_tokens: <CW_TOK>
- cache_read_tokens: <CR_TOK>
- cost_usd: $<COST>
- Reduction vs pre-fix avg ($5.18): <PCT>%
- Playbook actions: <COUNT>; rationale spot-check: <pass|concerns>
```

Fill in actuals from Step 3 and Step 4.

- [ ] **Step 6: Commit**

```
git add docs/superpowers/specs/2026-05-08-strategist-cache-fix-design.md
git commit -m "$(cat <<'EOF'
docs(specs): record post-fix validation for strategist cache fix

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review checklist (post-implementation)

After Task 5, run through this list:

- All tasks above show green checkboxes.
- `python3 -m pytest tests/v2/test_claude_client.py -v` passes.
- `python3 -m pytest tests/ -q` passes.
- `grep -n "_truncate_old_tool_results\|_TRUNCATION_THRESHOLD\|_KEEP_RECENT_EXCHANGES" v2/ tests/` returns no matches.
- Spec file has Phase 1 size data, chosen threshold value, and post-fix validation results filled in.
- Post-fix `cost_usd` ≤ $3.10 (≥40% reduction from $5.18 baseline).
- Most recent paper playbook has ≥1 action with coherent rationale.

If any item fails, fix before merging.

---

## Out of scope (do NOT do as part of this plan)

- Per-tool truncation thresholds (revisit only if Phase 1 data shows extreme spread).
- Sonnet model swap.
- Trimming `_build_pre_seeded_context`.
- Any change to `_messages_with_cache_breakpoint` or `_aggressive_prune`.
- Touching the executor or strategy reflection stages.
