# Executor model pilot — telemetry + Sonnet swap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the agentic-loop telemetry gap so per-stage Claude cost is measurable, drop the dead executor cache marker, make executor model + max_tokens env-configurable, raise max_tokens, and flip the executor default from Haiku 4.5 to Sonnet 4.6.

**Architecture:** Five small, sequenced changes. Telemetry tasks (1–2) come first so we can measure the swap. Cache cleanup (3) is independent debt. Config knobs (4–5) come next, then the default flip (6). Each task is a self-contained commit with a focused test addition.

**Tech Stack:** Python 3, Anthropic SDK (already installed), pytest, psycopg2 (`agent_events` table for telemetry), docker compose (paper + prod stacks).

**Background — why this matters:**
- The executor (`v2/agent.py`) currently uses Haiku 4.5 with `max_tokens=4096` for a one-shot JSON output that must include reversal-justification reasoning across all playbook actions. The audit module has a standing check (`check_executor_max_tokens_hit`) for this being a known failure mode.
- The agentic loop in `claude_client.py` emits per-turn `agent_call` events from `_call_with_retry`, but `run_agentic_loop` calls it without passing `session_id`, `stage_name`, or `purpose` — so every per-turn event is logged with `stage_name='unknown'`, making per-stage cost queries impossible. The aggregate `loop_completion` event omits cache tokens entirely.
- The executor's `TRADING_SYSTEM_PROMPT` is tagged `cache_control: ephemeral`, but the executor runs once per day and the cache has a 5-minute TTL. Confirmed empirically: `cache_read_tokens=0` in `agent_events`. The marker is dead weight.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `v2/claude_client.py` | Per-turn `_call_with_retry` invocations; `loop_completion` event emission | Plumb `stage_name`/`session_id`/`purpose` from loop to per-call; include cache tokens in `loop_completion` payload |
| `v2/agent.py` | Executor entry point + system prompt | Drop ephemeral cache wrapper; read model from env; read max_tokens from env; raise default |
| `tests/v2/test_claude_client.py` | Existing claude_client tests | Add test asserting per-turn `agent_call` event carries loop's stage_name |
| `tests/v2/test_agent.py` | Existing executor tests | Add tests for env-configured model + max_tokens + cache marker removal |
| (none — no schema changes) | | |

No new files. No new dependencies.

---

## Task 1: Plumb stage_name/session_id/purpose from `run_agentic_loop` into per-turn `_call_with_retry`

**Files:**
- Modify: `v2/claude_client.py:368-375` (the per-turn `_call_with_retry` call inside `run_agentic_loop`)
- Test: `tests/v2/test_claude_client.py` (add a new test class at the bottom of the file)

**Context:** `_call_with_retry` already supports `session_id`, `stage_name`, `purpose` kwargs (it pops them at lines 151-153). `run_agentic_loop` receives `session_id` and `stage_name` as parameters but does not forward them on line 368. Result: per-turn `agent_call` events get `stage_name='unknown'`.

- [ ] **Step 1: Write the failing test**

Add this test class at the bottom of `tests/v2/test_claude_client.py`:

```python
class TestAgenticLoopPerTurnTelemetry:
    """`run_agentic_loop` must forward stage_name/session_id to the per-turn
    `_call_with_retry` so emitted `agent_call` events carry the loop's stage
    instead of defaulting to 'unknown'."""

    def test_per_turn_agent_call_event_carries_loop_stage_name(self, monkeypatch):
        from unittest.mock import MagicMock
        import v2.claude_client as cc

        events = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kwargs: events.append(kwargs),
        )

        # Mock the stream context so _call_with_retry runs its full body
        # (including the finally block that emits agent_call).
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [MagicMock(type="text", text="done")]
        response.usage = MagicMock(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )

        stream_cm = MagicMock()
        stream_cm.__enter__ = MagicMock(return_value=MagicMock(
            get_final_message=MagicMock(return_value=response)
        ))
        stream_cm.__exit__ = MagicMock(return_value=False)

        client = MagicMock()
        client.messages.stream = MagicMock(return_value=stream_cm)

        cc.run_agentic_loop(
            client=client,
            model="claude-test",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=1,
            session_id=42,
            stage_name="reflection",
        )

        agent_calls = [e for e in events if e["event_type"] == "agent_call"]
        assert len(agent_calls) == 1, f"expected 1 agent_call event, got {len(agent_calls)}: {events}"
        assert agent_calls[0]["stage_name"] == "reflection", \
            f"per-turn agent_call should carry loop stage_name, got {agent_calls[0]['stage_name']!r}"
        assert agent_calls[0]["session_id"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_claude_client.py::TestAgenticLoopPerTurnTelemetry -v`
Expected: FAIL with `assert 'unknown' == 'reflection'` (the per-turn event currently goes out with `stage_name='unknown'`).

- [ ] **Step 3: Forward kwargs in `run_agentic_loop`**

Edit `v2/claude_client.py` around line 368. Find this block:

```python
        try:
            response = _call_with_retry(
                client,
                model=model,
                max_tokens=32000,
                system=cached_system,
                tools=cached_tools,
                messages=_messages_with_cache_breakpoint(pruned),
            )
```

Replace with:

```python
        try:
            response = _call_with_retry(
                client,
                model=model,
                max_tokens=32000,
                system=cached_system,
                tools=cached_tools,
                messages=_messages_with_cache_breakpoint(pruned),
                session_id=session_id,
                stage_name=stage_name,
                purpose="agentic_loop",
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_claude_client.py::TestAgenticLoopPerTurnTelemetry -v`
Expected: PASS.

- [ ] **Step 5: Run the full claude_client test file to verify no regression**

Run: `python3 -m pytest tests/v2/test_claude_client.py -v`
Expected: All pre-existing tests still pass. Existing `TestAgentCallTelemetry` tests already pass `stage_name` explicitly to `_call_with_retry`, so they are unaffected.

- [ ] **Step 6: Commit**

```bash
git add v2/claude_client.py tests/v2/test_claude_client.py
git commit -m "$(cat <<'EOF'
fix(telemetry): forward stage_name/session_id from run_agentic_loop to per-turn _call_with_retry

Per-turn agent_call events were being emitted with stage_name='unknown'
because run_agentic_loop didn't pass its own session_id/stage_name into
the per-turn _call_with_retry. Per-stage cost queries were therefore
blind to anything inside the agentic loops (ideation, reflection).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Include cache tokens in `loop_completion` event payload

**Files:**
- Modify: `v2/claude_client.py:529-540` (the `loop_completion` `record_event` call)
- Test: `tests/v2/test_claude_client.py` (extend the test class from Task 1)

**Context:** The loop tracks `total_cache_creation` and `total_cache_read` (lines 407-408) but the `loop_completion` event only logs `input_tokens` and `output_tokens`. You cannot reconstruct true cost from the event alone — cache reads are billed at 10% of input.

- [ ] **Step 1: Write the failing test**

Append to `TestAgenticLoopPerTurnTelemetry` in `tests/v2/test_claude_client.py`:

```python
    def test_loop_completion_event_includes_cache_tokens(self, monkeypatch):
        from unittest.mock import MagicMock
        import v2.claude_client as cc

        events = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kwargs: events.append(kwargs),
        )

        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [MagicMock(type="text", text="done")]
        response.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=300,
        )

        stream_cm = MagicMock()
        stream_cm.__enter__ = MagicMock(return_value=MagicMock(
            get_final_message=MagicMock(return_value=response)
        ))
        stream_cm.__exit__ = MagicMock(return_value=False)

        client = MagicMock()
        client.messages.stream = MagicMock(return_value=stream_cm)

        cc.run_agentic_loop(
            client=client,
            model="claude-test",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=1,
            session_id=1,
            stage_name="reflection",
        )

        completions = [e for e in events if e["event_type"] == "loop_completion"]
        assert len(completions) == 1
        payload = completions[0]["payload"]
        assert payload["cache_creation_tokens"] == 200, \
            f"loop_completion missing cache_creation_tokens: {payload}"
        assert payload["cache_read_tokens"] == 300, \
            f"loop_completion missing cache_read_tokens: {payload}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_claude_client.py::TestAgenticLoopPerTurnTelemetry::test_loop_completion_event_includes_cache_tokens -v`
Expected: FAIL with `KeyError: 'cache_creation_tokens'`.

- [ ] **Step 3: Extend the `loop_completion` payload**

Edit `v2/claude_client.py` around line 529. Find:

```python
    record_event(
        session_id=session_id,
        stage_name=stage_name or "unknown",
        event_type="loop_completion",
        payload={
            "stop_reason": stop_reason,
            "turns_used": turns_used,
            "model": model,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
    )
```

Replace with:

```python
    record_event(
        session_id=session_id,
        stage_name=stage_name or "unknown",
        event_type="loop_completion",
        payload={
            "stop_reason": stop_reason,
            "turns_used": turns_used,
            "model": model,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_creation_tokens": total_cache_creation,
            "cache_read_tokens": total_cache_read,
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_claude_client.py::TestAgenticLoopPerTurnTelemetry -v`
Expected: PASS.

- [ ] **Step 5: Verify no regressions in the full v2 suite touching claude_client**

Run: `python3 -m pytest tests/v2/test_claude_client.py tests/v2/test_strategy.py tests/v2/test_ideation_claude.py -v`
Expected: All pass. Existing tests don't assert on the absence of these new payload keys.

- [ ] **Step 6: Commit**

```bash
git add v2/claude_client.py tests/v2/test_claude_client.py
git commit -m "$(cat <<'EOF'
feat(telemetry): include cache tokens in loop_completion event payload

Cache reads are billed at 10% of input tokens; omitting them from the
aggregate loop_completion event meant per-stage cost queries underread
total spend whenever the prompt cache hit. Add cache_creation_tokens
and cache_read_tokens to match the per-turn agent_call payload shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Drop the executor's dead ephemeral cache wrapper

**Files:**
- Modify: `v2/agent.py:225-243` (the `cached_system` construction and `_call_with_retry` call inside `get_trading_decisions`)
- Test: `tests/v2/test_agent.py` (add a test asserting `system=` is a plain string, not a cache-wrapped list)

**Context:** The executor runs at most once per session (daily). Ephemeral cache TTL is 5 minutes. `cache_read_tokens=0` for every executor call in the `agent_events` table. The wrapper adds zero benefit and complicates the call site. Simplify to a plain string.

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_agent.py` inside `class TestGetTradingDecisions` (after `test_calls_haiku_with_structured_input`):

```python
    def test_does_not_wrap_system_prompt_in_ephemeral_cache(self):
        """Executor runs once per day; 5-minute ephemeral cache never hits.
        The system kwarg should be passed as a plain string, not a
        cache_control-wrapped list."""
        from unittest.mock import MagicMock, patch

        captured = {}

        def fake_call(client, **kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.content = [MagicMock(text='{"decisions":[],"thesis_invalidations":[],"market_summary":"","risk_assessment":""}')]
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock(input_tokens=10, output_tokens=10)
            return resp

        executor_input = ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "50000"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="Neutral", risk_notes="",
        )

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", side_effect=fake_call):
            get_trading_decisions(executor_input)

        assert isinstance(captured["system"], str), \
            f"system should be a plain string (cache wrapper is dead weight for once-per-day executor); got {type(captured['system'])}: {captured['system']!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_agent.py::TestGetTradingDecisions::test_does_not_wrap_system_prompt_in_ephemeral_cache -v`
Expected: FAIL — `system` is currently a list with `cache_control`.

- [ ] **Step 3: Drop the wrapper**

Edit `v2/agent.py`. Find lines 225-233:

```python
    cached_system = [
        {"type": "text", "text": TRADING_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]

    response = _call_with_retry(
        client,
        model=model,
        max_tokens=4096,
        system=cached_system,
```

Replace with:

```python
    response = _call_with_retry(
        client,
        model=model,
        max_tokens=4096,
        system=TRADING_SYSTEM_PROMPT,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_agent.py::TestGetTradingDecisions::test_does_not_wrap_system_prompt_in_ephemeral_cache -v`
Expected: PASS.

- [ ] **Step 5: Run full agent test file**

Run: `python3 -m pytest tests/v2/test_agent.py -v`
Expected: All pre-existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add v2/agent.py tests/v2/test_agent.py
git commit -m "$(cat <<'EOF'
chore(agent): drop dead ephemeral cache wrapper on executor system prompt

The executor runs at most once per session (daily). Ephemeral cache has
a 5-minute TTL, so cache_read_tokens has always been 0 for executor
calls. Removing the wrapper simplifies the call site without changing
cost or behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Make `DEFAULT_EXECUTOR_MODEL` env-configurable

**Files:**
- Modify: `v2/agent.py:28` (module-level constant)
- Test: `tests/v2/test_agent.py` (add a test reading the constant under env override)

**Context:** Today `DEFAULT_EXECUTOR_MODEL = "claude-haiku-4-5-20251001"` is hardcoded. The CLI flag `--executor-model` already overrides at the call site, but ops/cron runs go through `session.py` and would benefit from an env knob (`ALGO_EXECUTOR_MODEL`) so paper and prod can flip independently via `.env` without editing CLI args. Default stays Haiku in this task; Task 6 flips it.

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_agent.py` (top-level test class, place near the existing top of file):

```python
class TestDefaultExecutorModelEnvOverride:
    """ALGO_EXECUTOR_MODEL env var should override the hardcoded default
    so paper/prod can flip via .env without code changes."""

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("ALGO_EXECUTOR_MODEL", "claude-sonnet-4-6")
        # Force re-evaluation of the module-level default
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)
        assert agent_module.DEFAULT_EXECUTOR_MODEL == "claude-sonnet-4-6"

    def test_falls_back_to_haiku_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ALGO_EXECUTOR_MODEL", raising=False)
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)
        assert agent_module.DEFAULT_EXECUTOR_MODEL == "claude-haiku-4-5-20251001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_agent.py::TestDefaultExecutorModelEnvOverride -v`
Expected: `test_env_var_overrides_default` FAILS (env var ignored).

- [ ] **Step 3: Add `import os` to the imports block**

`v2/agent.py` currently imports `json` and `logging` but not `os`. Edit `v2/agent.py` around line 3-4. Find:

```python
import json
import logging
```

Replace with:

```python
import json
import logging
import os
```

- [ ] **Step 4: Read from env in the module-level constant**

Edit `v2/agent.py` line 28. Find:

```python
DEFAULT_EXECUTOR_MODEL = "claude-haiku-4-5-20251001"
```

Replace with:

```python
DEFAULT_EXECUTOR_MODEL = os.environ.get("ALGO_EXECUTOR_MODEL", "claude-haiku-4-5-20251001")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_agent.py::TestDefaultExecutorModelEnvOverride -v`
Expected: PASS.

- [ ] **Step 6: Run full agent test file**

Run: `python3 -m pytest tests/v2/test_agent.py -v`
Expected: All pre-existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add v2/agent.py tests/v2/test_agent.py
git commit -m "$(cat <<'EOF'
feat(agent): make DEFAULT_EXECUTOR_MODEL env-configurable via ALGO_EXECUTOR_MODEL

Enables paper/prod to swap executor model independently via .env without
code edits. Default falls back to Haiku 4.5 when unset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Make executor `max_tokens` env-configurable and raise the default

**Files:**
- Modify: `v2/agent.py` (introduce module-level `EXECUTOR_MAX_TOKENS` constant; replace hardcoded `max_tokens=4096`)
- Test: `tests/v2/test_agent.py` (add tests for env override and the raised default)

**Context:** `max_tokens=4096` is hardcoded inside `get_trading_decisions`. The audit module has `check_executor_max_tokens_hit` because hitting the cap silently drops decisions. Doubling the cap to 8192 buys headroom for multi-action playbooks; making it env-configurable (`ALGO_EXECUTOR_MAX_TOKENS`) lets ops tune without code changes.

- [ ] **Step 1: Write the failing tests**

Add to `tests/v2/test_agent.py`:

```python
class TestExecutorMaxTokensEnvOverride:
    """ALGO_EXECUTOR_MAX_TOKENS env var should override the default cap.
    The default is raised to 8192 to give headroom for multi-action
    playbooks; the audit module's check_executor_max_tokens_hit was
    catching real truncations at 4096."""

    def test_default_is_8192_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ALGO_EXECUTOR_MAX_TOKENS", raising=False)
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)
        assert agent_module.EXECUTOR_MAX_TOKENS == 8192

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("ALGO_EXECUTOR_MAX_TOKENS", "12000")
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)
        assert agent_module.EXECUTOR_MAX_TOKENS == 12000

    def test_max_tokens_passed_to_api_call(self, monkeypatch):
        """The configured value must reach _call_with_retry, not a stale
        constant captured at function-def time."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("ALGO_EXECUTOR_MAX_TOKENS", "9999")
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)

        captured = {}
        def fake_call(client, **kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.content = [MagicMock(text='{"decisions":[],"thesis_invalidations":[],"market_summary":"","risk_assessment":""}')]
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock(input_tokens=10, output_tokens=10)
            return resp

        executor_input = agent_module.ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "50000"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="Neutral", risk_notes="",
        )
        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", side_effect=fake_call):
            agent_module.get_trading_decisions(executor_input)
        assert captured["max_tokens"] == 9999
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_agent.py::TestExecutorMaxTokensEnvOverride -v`
Expected: All three FAIL (`EXECUTOR_MAX_TOKENS` doesn't exist; `max_tokens=4096` is hardcoded).

- [ ] **Step 3: Introduce the constant**

Edit `v2/agent.py`. Find the line (added in Task 4):

```python
DEFAULT_EXECUTOR_MODEL = os.environ.get("ALGO_EXECUTOR_MODEL", "claude-haiku-4-5-20251001")
```

Append a new line directly below it:

```python
EXECUTOR_MAX_TOKENS = int(os.environ.get("ALGO_EXECUTOR_MAX_TOKENS", "8192"))
```

- [ ] **Step 4: Use the constant in the API call**

Inside `get_trading_decisions`, after Task 3's edit, the call site looks like:

```python
    response = _call_with_retry(
        client,
        model=model,
        max_tokens=4096,
        system=TRADING_SYSTEM_PROMPT,
```

Change `max_tokens=4096` to `max_tokens=EXECUTOR_MAX_TOKENS`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_agent.py::TestExecutorMaxTokensEnvOverride -v`
Expected: All three PASS.

- [ ] **Step 6: Run full agent test file**

Run: `python3 -m pytest tests/v2/test_agent.py -v`
Expected: All pre-existing tests pass. Note: pre-existing `test_raises_on_max_tokens` asserts behavior when the API returns `stop_reason='max_tokens'` — unaffected by the cap change.

- [ ] **Step 7: Commit**

```bash
git add v2/agent.py tests/v2/test_agent.py
git commit -m "$(cat <<'EOF'
feat(agent): raise executor max_tokens to 8192, make env-configurable

Audit's check_executor_max_tokens_hit was catching real truncations at
4096 — silently dropping decisions when the playbook had many actions
each requiring reversal-justification reasoning. Double the cap and
expose ALGO_EXECUTOR_MAX_TOKENS for further tuning without code changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Flip executor default model to Sonnet 4.6 (paper first, then prod)

**Files:**
- Modify: `.env.paper` (add `ALGO_EXECUTOR_MODEL=claude-sonnet-4-6`)
- No code changes.

**Context:** The Sonnet flip is a config change. We flip paper first, observe for at least 3 paper sessions, then flip prod in a follow-up. The `agent_events` telemetry from Tasks 1–2 lets us compare cost and quality across the swap.

This task is configuration only; there is no test to write. The verification step is operational, not unit-test-based.

- [ ] **Step 1: Inspect current `.env.paper` to confirm the var isn't already set**

Run: `grep -n "ALGO_EXECUTOR_MODEL" .env.paper || echo "(not set)"`
Expected: `(not set)`. If it is set, stop and ask the user how to proceed.

- [ ] **Step 2: Append the env var to `.env.paper`**

Edit `.env.paper`. Append at the end of the file:

```
# Executor pilot: Sonnet 4.6 instead of Haiku 4.5 for tighter reversal
# reasoning + headroom from the 8192 max_tokens default. Revert by
# removing this line.
ALGO_EXECUTOR_MODEL=claude-sonnet-4-6
```

- [ ] **Step 3: Verify the paper container picks up the new env var**

Run: `docker compose -f docker-compose.yml -f docker-compose.paper.yml run --rm trading-paper python -c "import v2.agent; print(v2.agent.DEFAULT_EXECUTOR_MODEL)"`
Expected output: `claude-sonnet-4-6`

If output is `claude-haiku-4-5-20251001`, the env file wasn't picked up — check `env_file` directive in `docker-compose.paper.yml`.

- [ ] **Step 4: Run a paper dry-run session to confirm end-to-end**

Run: `task paper:session:dry-run`
Expected: Session completes without errors. Inspect the trading stage log for a line like `Haiku tokens -- input: ...` (note: that log line is misnamed; we'll leave it for now since it'd be a separate cosmetic commit).

- [ ] **Step 5: Query telemetry to confirm the model field reflects Sonnet**

Run:
```bash
source .env.paper && docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT stage_name, payload->>'model' AS model, COUNT(*), SUM((payload->>'input_tokens')::bigint) AS in_tok, SUM((payload->>'output_tokens')::bigint) AS out_tok FROM agent_events WHERE stage_name='trading' AND occurred_at > now() - interval '1 day' AND payload ? 'input_tokens' GROUP BY 1, 2;"
```
Expected: `model=claude-sonnet-4-6` for the trading stage, with non-zero token counts.

- [ ] **Step 6: Commit the env-file change**

```bash
git add .env.paper
git commit -m "$(cat <<'EOF'
chore(paper): flip executor to Sonnet 4.6 for pilot

Pilot the executor model rebalance on paper first. Telemetry from
Tasks 1-2 lets us compare cost and reversal-decision quality vs Haiku
baseline. Prod flip is a follow-up after at least 3 clean paper
sessions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Run the full v2 test suite as a final regression gate**

Run: `python3 -m pytest tests/v2/ -v`
Expected: All pass.

---

## Out of scope (deliberately not in this plan)

- **Prod `.env` flip.** Comes in a follow-up after at least 3 clean paper sessions, with a comparison query showing cost delta + decision-quality delta (truncation rate, reversal-justification compliance via existing audit checks).
- **Audit rule-judgment model swap (Haiku → Sonnet).** Separate plan. Different file (`v2/audit.py`), different risk profile (low frequency, but identity-shaping).
- **Reflection model evaluation.** The user pushed back on demoting the strategist; reflection-on-Opus would be a separate experiment if data from this pilot suggests reflection is the next bottleneck.
- **Dead `run_ideation_claude` path removal + shared reversal-justification prompt extraction.** Code-debt plan; orthogonal to model selection.
- **News filter recall instrumentation.** Belongs in its own measurement plan.
