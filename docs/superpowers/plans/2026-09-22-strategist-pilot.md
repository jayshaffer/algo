# Strategist Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run only the strategist stage of the paper session as a headless `claude -p` call under the operator's Claude subscription, in an isolated context, alternating with the API path on a fixed weekday schedule, so the two backends can be compared on the telemetry that already exists before anything else is inverted.

**Architecture:** A host-side bash "sandwich": `v2.session` runs every stage before the strategist on the API path and creates the day's session row; the driver brackets one `claude -p` call with `session_ctl stage-begin` / `stage-context` / `stage-end`; `v2.session --resume` (from the seams plan) runs the executor, reflection and dashboard stages and finalises the row. The strategist's tools come from a stdio MCP server (`v2/mcp_server.py`) that serves the existing strategist registry unchanged. The executor stays on the API path: it is Haiku, it costs cents, and it is the call that becomes orders.

**Tech Stack:** Python 3.12 (container), `mcp` Python SDK, bash (host), Claude Code CLI 2.1.280 headless mode, PostgreSQL 16, pytest (run in docker: `task test INSTANCE=paper -- <args>`).

**Spec:** `docs/superpowers/specs/2026-09-22-claude-session-inversion-design.md` — sections "`v2/mcp_server.py`", "Telemetry", "Failure handling", "Usage limits". Departures from the spec, all deliberate: no skills (the prompt is passed with `--system-prompt` from `v2/prompts/strategist.md`), no `session_ctl plan` / driver loop (the sandwich reuses `run_session`), one stage only, and a two-week A/B before any further stage moves. `docs/superpowers/plans/2026-09-22-session-seams.md` must be merged first.

## Global Constraints

Verified on this host on 2026-09-22 with Claude Code 2.1.280; the smoke script (Task 1) re-checks them after every CLI upgrade:

- `claude --bare -p` returns `"is_error": true, "result": "Not logged in · Please run /login"` with **exit code 0**. Never pass `--bare`, and never trust the exit code alone: the driver reads `is_error` from the result JSON.
- A non-bare `claude -p` under `env -i HOME=… PATH=…` authenticates with the subscription login in `~/.claude/.credentials.json`. `env -i` also guarantees `ANTHROPIC_API_KEY` is absent, which is what keeps the call off the metered API.
- Context isolation recipe, verified to yield a context with no operator instructions: run from a fresh scratch directory (project `CLAUDE.md` lookup walks up parents), `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, `--setting-sources ""`, `--system-prompt <text>`. Without the recipe the global `~/.claude/CLAUDE.md` and plugin reminders reach the model.
- `-p` with no prompt argument reads the prompt from stdin. `--tools ""` is accepted. `--append-system-prompt` works alongside `--system-prompt`. There is no `--max-turns` and no `--max-tool-calls`; the turn cap is the MCP server's counter plus the driver's `timeout`.
- `--output-format json` / `stream-json` result objects carry `is_error`, `result`, `stop_reason`, `num_turns`, `duration_api_ms`, `usage`, and `modelUsage` keyed by full model id (e.g. `claude-haiku-4-5-20251001`).
- `DISABLE_AUTOUPDATER=1` is exported by the driver for the `claude` process; the CLI version is recorded on every `agent_call` event so the telemetry series carries it.
- Tests run in the container: `task test INSTANCE=paper -- tests/v2/test_x.py -v`. No test may reach the network; `tests/v2/conftest.py` must gain patch targets for `v2.session_ctl`.
- Stage names: `strategist` in `session_stages`; `ideation` in `llm_call_contexts` / `agent_events`; purpose `strategist_loop`. Unchanged from the API path so the internal dashboard and the audit checks keep working.
- The money path is untouched. The strategist writes theses and a playbook; the executor (API path, unchanged) turns it into orders.
- `claude-opus-4-8` is the default strategist model and already has a `model_pricing` row. `ALGO_STAGE_MODEL_STRATEGIST` may override it with any full id that has a row (`tests/test_pricing_coverage.py` covers literals in `v2/`, not env values; check `model_pricing` by hand before overriding).
- ruff: line length 140, target py312, isort ordering. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/smoke-claude-headless.sh` (new) | Host guard for the constraints above: subscription auth, isolation, stdin, empty tool list, and (after Task 3) the MCP server appearing in the session's tool list. |
| `v2/ideation_claude.py` (modify) | Extract `build_strategist_system_appendix()` and `build_strategist_initial_message()` from `run_strategist_loop` so the API path and `session_ctl` share one implementation. |
| `v2/session.py` (modify) | Extract `persist_strategist_memo(summary, session_date, session_id)` from `_persist_strategist_memo`. |
| `v2/session_ctl.py` (new) | `status`, `stage-begin`, `stage-context`, `stage-end`: the pilot's gates, validators, usage and telemetry. |
| `v2/mcp_server.py` (new) | stdio MCP server over the strategist registry; tool-call cap; `tools.jsonl`. |
| `v2/prompts/strategist-pilot-note.md` (new) | Short runtime note appended to the system prompt under Claude Code (tool naming, WebSearch cap, how the stage ends). |
| `session-driver.sh` (new) | The sandwich. One function builds the `claude` invocation. |
| `scripts/driver_json.py` (new) | Stdlib JSON helpers for the driver (`field`, `contains`, `last-result`). |
| `v2/requirements.txt` (modify) | Add `mcp>=1.2`. |
| `tests/v2/test_session_ctl.py`, `tests/v2/test_mcp_server.py`, `tests/test_driver_json.py`, `tests/test_session_driver.py` (new) | Unit tests; the driver test stubs `docker` and `claude`. |
| `tests/v2/conftest.py` (modify) | `_SESSION_CTL_DB_PATCH_TARGETS`. |
| `Taskfile.yml`, `crontab`, `instances/example.env`, `CLAUDE.md`, `docs/runbook-recovery.md` (modify) | `session:pilot` and `driver:smoke` targets, the alternating paper schedule, the model knob, docs. |

**Prerequisite:** the seams plan is merged and paper has at least three `completed` sessions on the API path since the credit top-up. Those are the baseline rows the comparison in Task 5 reads.

---

### Task 1: Smoke script for the headless assumptions

**Files:**
- Create: `scripts/smoke-claude-headless.sh`
- Modify: `Taskfile.yml` (add `driver:smoke`)

**Interfaces:**
- `scripts/smoke-claude-headless.sh [--with-mcp <instance>]` — exit 0 when every check passes, 1 with a loud message naming the first failed check. `--with-mcp` is used from Task 4 on; before Task 3 exists it must be omitted.
- Environment: `PINCHY_CLAUDE_BIN` (default `claude`), `PINCHY_SMOKE_MODEL` (default `claude-haiku-4-5-20251001`).
- Checks, in order: `auth` (non-bare under `env -i`, `is_error` false, `modelUsage` non-empty), `bare-is-not-a-fallback` (bare under `env -i` reports not logged in — proves the working path is OAuth, not a stray API key), `isolation` (the probe prompt answers `NONE`), `stdin-and-empty-tools` (prompt from stdin with `--tools ""` answers `ok`), and with `--with-mcp`: `mcp` (the `system`/`init` event of a stream-json run lists `mcp__pinchy__write_playbook`).

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# scripts/smoke-claude-headless.sh [--with-mcp <instance>]
#
# Guard for the strategist pilot's hard assumptions about headless Claude
# Code. Run after every CLI upgrade and from `task driver:smoke`. Each check
# prints PASS/FAIL; the first FAIL exits 1 with the raw output.
#
# Verified on 2026-09-22 with 2.1.280:
#   - `--bare` never reads the subscription login (is_error true, exit 0)
#   - non-bare `-p` under `env -i HOME PATH` authenticates with it
#   - scratch cwd + CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 + --setting-sources ""
#     + --system-prompt gives a context with no operator instructions
set -uo pipefail
CLAUDE_BIN="${PINCHY_CLAUDE_BIN:-claude}"
MODEL="${PINCHY_SMOKE_MODEL:-claude-haiku-4-5-20251001}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_MCP=""
[ "${1:-}" = "--with-mcp" ] && WITH_MCP="${2:?--with-mcp needs an instance}"

SCRATCH="$(mktemp -d)"; trap 'rm -rf "$SCRATCH"' EXIT
cd "$SCRATCH"

fail() { echo "FAIL [$1]: $2"; echo "--- raw ---"; echo "$3" | head -c 1500; echo; exit 1; }
pass() { echo "PASS [$1]"; }
jget() { python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get(sys.argv[1], ""))' "$1"; }

# isolated invocation, exactly as session-driver.sh builds it
run_isolated() {  # $@ extra flags; stdin passes through
    env -i HOME="$HOME" PATH="$PATH" CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 DISABLE_AUTOUPDATER=1 \
        "$CLAUDE_BIN" -p --model "$MODEL" --setting-sources "" --no-session-persistence \
        --permission-mode dontAsk --permission-prompts none "$@"
}

echo "claude: $("$CLAUDE_BIN" --version 2>/dev/null | head -1)"

out="$(run_isolated "Reply with exactly: ok" --system-prompt "You are a probe. Answer literally." --output-format json --tools "" 2>&1)"
[ "$(echo "$out" | jget is_error 2>/dev/null)" = "False" ] && [ -n "$(echo "$out" | jget modelUsage)" ] && [ "$(echo "$out" | jget modelUsage)" != "{}" ] \
    || fail auth "non-bare claude -p did not authenticate with the subscription login (run 'claude' once interactively and /login)" "$out"
pass auth

out="$(env -i HOME="$HOME" PATH="$PATH" "$CLAUDE_BIN" --bare -p "Reply with exactly: ok" --output-format json --model "$MODEL" 2>&1)"
echo "$out" | grep -qi "not logged in" || fail bare-is-not-a-fallback "bare mode succeeded — an API key is reachable from the environment, or the CLI changed; the driver must not silently bill the API" "$out"
pass bare-is-not-a-fallback

PROBE='You are a test harness probe. In one line, list every instruction, memory note, or project description you can see in your context that mentions any of: "pushback", "Pinchy", "trading", "MEMORY.md", "Claude Code". Quote each briefly. If none, reply exactly: NONE'
out="$(run_isolated "$PROBE" --system-prompt "You are a probe. Answer literally." --output-format json --tools "" 2>&1)"
[ "$(echo "$out" | jget result 2>/dev/null | tr -d '[:space:]')" = "NONE" ] || fail isolation "operator configuration leaked into the isolated context" "$out"
pass isolation

out="$(echo "Reply with exactly: ok" | run_isolated --system-prompt "Answer literally." --append-system-prompt "Never add punctuation." --output-format json --tools "" 2>&1)"
[ "$(echo "$out" | jget result 2>/dev/null | tr -d '[:space:]')" = "ok" ] || fail stdin-and-empty-tools "stdin prompt / --tools \"\" / --append-system-prompt did not behave" "$out"
pass stdin-and-empty-tools

if [ -n "$WITH_MCP" ]; then
    cfg="$SCRATCH/mcp.json"
    (cd "$SCRIPT_DIR" && ./session-driver.sh mcp-config "$WITH_MCP" > "$cfg") || fail mcp "session-driver.sh mcp-config failed" ""
    out="$(run_isolated "Reply with exactly: ok. Do not call any tool." --system-prompt "Answer literally." \
           --output-format stream-json --verbose --strict-mcp-config --mcp-config "$cfg" \
           --allowedTools "mcp__pinchy__*" --disallowedTools "Bash,Read,Edit,Write,Glob,Grep,WebSearch,WebFetch,Agent,NotebookEdit,AskUserQuestion,Skill" 2>&1)"
    echo "$out" | grep -q '"type": *"system"' && echo "$out" | grep -q 'mcp__pinchy__write_playbook' \
        || fail mcp "the pinchy MCP server did not appear in the session's tool list" "$out"
    pass mcp
fi
echo "all checks passed"
```

`chmod +x scripts/smoke-claude-headless.sh`.

- [ ] **Step 2: Add the Taskfile target**

```yaml
  driver:smoke:
    desc: Verify headless Claude Code still authenticates with the subscription and isolates its context (run after every CLI upgrade)
    requires: { vars: [INSTANCE] }
    cmds:
      - ./scripts/smoke-claude-headless.sh {{if .WITH_MCP}}--with-mcp {{.INSTANCE}}{{end}}
```

(`WITH_MCP=1 task driver:smoke INSTANCE=paper` runs the MCP check once Task 4 exists.)

- [ ] **Step 3: Run it**

Run: `task driver:smoke INSTANCE=paper`
Expected: four `PASS` lines and `all checks passed`. If `auth` fails, run `claude` interactively and `/login`, then rerun.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke-claude-headless.sh Taskfile.yml
git commit -m "Add the headless Claude Code smoke check for the strategist pilot

Guards subscription auth, non-bare mode, context isolation, stdin
prompts and empty tool lists — the assumptions session-driver.sh rests on.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `session_ctl` — status, stage-begin, stage-context, stage-end

**Files:**
- Modify: `v2/ideation_claude.py:416-481` (`run_strategist_loop`), `v2/session.py:337-364` (`_persist_strategist_memo`)
- Create: `v2/session_ctl.py`, `v2/prompts/strategist-pilot-note.md`
- Modify: `tests/v2/conftest.py:47-57`
- Test: `tests/v2/test_session_ctl.py`, `tests/v2/test_ideation_claude.py`

**Interfaces:**
- Produces in `v2/ideation_claude.py`:
  - `build_strategist_system_appendix(attribution_constraints: str, orphans: list[dict]) -> str` — attribution constraints and formation context joined by a blank line; empty string when both are empty. `run_strategist_loop` appends it to the base prompt with `"\n\n"` exactly as before.
  - `build_strategist_initial_message(orphans: list[dict]) -> str` — the pre-seeded state message (`_build_pre_seeded_context` + orphan block + open watchlist + numbered steps), text unchanged.
- Produces in `v2/session.py`: `persist_strategist_memo(summary: str | None, session_date, session_id: int | None) -> None` (the body of `_persist_strategist_memo`; the old function becomes a one-line wrapper passing `result.strategist_result.final_summary`).
- Produces `python -m v2.session_ctl <cmd>`; every command prints one JSON object (except `stage-context`, which prints text) and exits 0 unless noted:
  - `status` → `{"halted": bool, "session_date": "YYYY-MM-DD", "session_id": int|null, "status": str|null, "completed_stages": [str]}`. `halted` mirrors `ALGO_TRADING_HALTED`; the rest describes the latest `sessions` row for today's market date.
  - `stage-begin --session-id N --stage strategist` → `{"ok": true}`.
  - `stage-context --stage strategist --part {system_appendix,message}` → text. Exit 1 on an exception (the driver fails the stage).
  - `stage-end --session-id N --stage strategist --session-date D [--result PATH] [--transcript PATH] [--tools-log PATH] [--context PATH] [--system-prompt PATH] [--exit-code N] [--error TEXT] [--cli-version V]` → `{"status": "completed"|"failed", "error": str|null, "error_class": str|null}`.
- `tools.jsonl` line format (written by Task 3, read here): `{"tool_name", "args", "success", "error", "duration_ms", "output_chars", "result_prefix"}`; `result_prefix` is the first 200 characters of the tool's string result.
- Python API: `status() -> dict`, `stage_context(part: str) -> str`, `usage_from_result(result: dict | None) -> UsageAccumulator | None`, `classify_failure(*, exit_code, result, error) -> tuple[str, str] | None`, `read_tools_log(path) -> list[dict]`, `messages_from_transcript(lines: list[str], initial_context: str | None) -> tuple[list[dict], list | None, str | None]`, `validate_strategist(session_date) -> str | None`, `end_strategist_stage(**kw) -> dict`.
- Failure classes: `timeout` (exit 124), `usage_limit`, `auth`, `model_error` (from `is_error` + result text), `command` (nonzero exit without a result), `validator` (stage ran but left no playbook or an open watchlist item).

- [ ] **Step 1: Write the failing ideation tests**

```python
# tests/v2/test_ideation_claude.py — append
from unittest.mock import patch


def test_system_appendix_joins_constraints_and_formation():
    from v2.ideation_claude import build_strategist_system_appendix
    with patch("v2.ideation_claude.build_formation_context", return_value="FORMATION"):
        assert build_strategist_system_appendix("CONSTRAINTS", []) == "CONSTRAINTS\n\nFORMATION"
        assert build_strategist_system_appendix("", []) == "FORMATION"
    with patch("v2.ideation_claude.build_formation_context", return_value=""):
        assert build_strategist_system_appendix("", []) == ""


def test_initial_message_contains_state_and_steps():
    from v2.ideation_claude import build_strategist_initial_message
    with patch("v2.ideation_claude._build_pre_seeded_context", return_value="STATE"), \
         patch("v2.ideation_claude.wl.get_open_items", return_value=[]), \
         patch("v2.ideation_claude.wl.format_open_watchlist_items", return_value="WATCHLIST"):
        msg = build_strategist_initial_message([])
    assert msg.startswith("Here is the current state (pre-loaded to save round-trips):\n\nSTATE\n\nWATCHLIST")
    assert "4. Write today's playbook using the write_playbook tool" in msg


def test_run_strategist_loop_uses_the_shared_builders():
    from v2.ideation_claude import run_strategist_loop
    with patch("v2.ideation_claude.get_orphan_positions", return_value=[]), \
         patch("v2.ideation_claude.build_strategist_system_appendix", return_value="APPENDIX"), \
         patch("v2.ideation_claude.build_strategist_initial_message", return_value="MESSAGE"), \
         patch("v2.ideation_claude._run_claude_loop") as loop, \
         patch("v2.ideation_claude.wl.assert_watchlist_resolved"):
        run_strategist_loop(system_prompt="BASE", attribution_constraints="C", session_id=5)
    assert loop.call_args.kwargs["system"] == "BASE\n\nAPPENDIX"
    assert loop.call_args.kwargs["initial_message"] == "MESSAGE"
```

- [ ] **Step 2: Write the failing session_ctl tests**

```python
# tests/v2/test_session_ctl.py
"""session_ctl brackets the one stage the pilot runs under Claude Code. Every
gate, validator and telemetry write for that stage lives here."""
import json
from datetime import date
from unittest.mock import patch

import pytest

from v2 import session_ctl as sc


class TestStatus:
    def test_reports_halt_and_latest_session(self):
        with patch("v2.session_ctl._trading_halted", return_value=False), \
             patch("v2.session_ctl.current_market_date", return_value=date(2026, 9, 22)), \
             patch("v2.session_ctl.get_session_for_date", return_value={"id": 42, "status": "completed"}), \
             patch("v2.session_ctl.get_completed_stages", return_value={"pipeline", "learning"}):
            out = sc.status()
        assert out == {"halted": False, "session_date": "2026-09-22", "session_id": 42, "status": "completed",
                       "completed_stages": ["learning", "pipeline"]}

    def test_no_session_yet(self):
        with patch("v2.session_ctl._trading_halted", return_value=True), \
             patch("v2.session_ctl.current_market_date", return_value=date(2026, 9, 22)), \
             patch("v2.session_ctl.get_session_for_date", return_value=None):
            out = sc.status()
        assert out["halted"] is True and out["session_id"] is None and out["completed_stages"] == []


class TestStageContext:
    def test_message_part_uses_shared_builder(self):
        with patch("v2.session_ctl.get_orphan_positions", return_value=[{"ticker": "X"}]), \
             patch("v2.session_ctl.build_strategist_initial_message", return_value="MSG") as b:
            assert sc.stage_context("message") == "MSG"
        b.assert_called_once_with([{"ticker": "X"}])

    def test_system_appendix_part_uses_constraints(self):
        with patch("v2.session_ctl.get_orphan_positions", return_value=[]), \
             patch("v2.session_ctl.build_attribution_constraints", return_value="C"), \
             patch("v2.session_ctl.build_strategist_system_appendix", return_value="APP") as b:
            assert sc.stage_context("system_appendix") == "APP"
        b.assert_called_once_with("C", [])

    def test_orphan_lookup_failure_degrades_to_empty(self):
        with patch("v2.session_ctl.get_orphan_positions", side_effect=RuntimeError("db")), \
             patch("v2.session_ctl.build_strategist_initial_message", return_value="MSG") as b:
            sc.stage_context("message")
        b.assert_called_once_with([])


RESULT = {
    "type": "result", "is_error": False, "result": "Reviewed theses, adopted the orphan, wrote today's playbook.",
    "stop_reason": "end_turn", "num_turns": 9, "duration_api_ms": 61000,
    "modelUsage": {"claude-opus-4-8": {"inputTokens": 1200, "outputTokens": 800, "cacheCreationInputTokens": 30000,
                                        "cacheReadInputTokens": 400000}},
}


class TestUsageAndFailure:
    def test_usage_from_result_builds_accumulator(self):
        acc = sc.usage_from_result(RESULT)
        assert acc.model == "claude-opus-4-8"
        assert (acc.input_tokens, acc.output_tokens, acc.cache_creation_tokens, acc.cache_read_tokens) == (1200, 800, 30000, 400000)

    def test_usage_from_result_none_when_missing(self):
        assert sc.usage_from_result(None) is None
        assert sc.usage_from_result({"is_error": True}) is None

    def test_classify_timeout(self):
        assert sc.classify_failure(exit_code=124, result=None, error="")[0] == "timeout"

    def test_classify_usage_limit_from_result(self):
        r = {"is_error": True, "result": "You have hit your usage limit for this window"}
        assert sc.classify_failure(exit_code=0, result=r, error=None)[0] == "usage_limit"

    def test_classify_auth(self):
        r = {"is_error": True, "result": "Not logged in · Please run /login"}
        assert sc.classify_failure(exit_code=0, result=r, error=None)[0] == "auth"

    def test_classify_model_error(self):
        r = {"is_error": True, "result": "something else"}
        assert sc.classify_failure(exit_code=0, result=r, error=None) == ("model_error", "something else")

    def test_classify_command_without_result(self):
        assert sc.classify_failure(exit_code=1, result=None, error="docker: boom") == ("command", "docker: boom")

    def test_classify_success(self):
        assert sc.classify_failure(exit_code=0, result=RESULT, error=None) is None


class TestTranscriptAndTools:
    def test_read_tools_log_skips_bad_lines(self, tmp_path):
        p = tmp_path / "tools.jsonl"
        p.write_text('{"tool_name": "get_theses", "success": true}\nnot json\n{"tool_name": "write_playbook", "success": true}\n')
        assert [t["tool_name"] for t in sc.read_tools_log(str(p))] == ["get_theses", "write_playbook"]
        assert sc.read_tools_log(None) == [] and sc.read_tools_log(str(tmp_path / "missing")) == []

    def test_messages_from_transcript(self):
        lines = [
            json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-4-8"}),
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]}}),
            json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "x"}]}}),
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}),
            "garbage",
        ]
        messages, response, model = sc.messages_from_transcript(lines, "CONTEXT")
        assert messages[0] == {"role": "user", "content": "CONTEXT"}
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        assert response == [{"type": "text", "text": "done"}]
        assert model == "claude-opus-4-8"


class TestValidateAndEnd:
    def _paths(self, tmp_path, result=RESULT, tools=True):
        r = tmp_path / "result.json"; r.write_text(json.dumps(result))
        t = tmp_path / "transcript.jsonl"; t.write_text(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}}) + "\n")
        c = tmp_path / "context.md"; c.write_text("CONTEXT")
        s = tmp_path / "system-prompt.md"; s.write_text("SYSTEM")
        tl = tmp_path / "tools.jsonl"
        if tools:
            tl.write_text(json.dumps({"tool_name": "write_playbook", "args": {}, "success": True, "error": None,
                                      "duration_ms": 5, "output_chars": 20, "result_prefix": "Playbook written"}) + "\n")
        return dict(result_path=str(r), transcript_path=str(t), tools_log_path=str(tl), context_path=str(c), system_prompt_path=str(s))

    def test_validate_requires_playbook(self):
        with patch("v2.session_ctl.get_playbook", return_value=None):
            assert "playbook" in sc.validate_strategist(date(2026, 9, 22))
        with patch("v2.session_ctl.get_playbook", return_value={"id": 1}), \
             patch("v2.session_ctl.wl.assert_watchlist_resolved"):
            assert sc.validate_strategist(date(2026, 9, 22)) is None

    def test_validate_requires_watchlist_resolved(self):
        with patch("v2.session_ctl.get_playbook", return_value={"id": 1}), \
             patch("v2.session_ctl.wl.assert_watchlist_resolved", side_effect=RuntimeError("open item 3")):
            assert sc.validate_strategist(date(2026, 9, 22)) == "open item 3"

    def test_end_completed_records_usage_memo_and_telemetry(self, tmp_path):
        with patch("v2.session_ctl.get_playbook", return_value={"id": 1}), \
             patch("v2.session_ctl.wl.assert_watchlist_resolved"), \
             patch("v2.session_ctl.complete_session_stage") as complete, \
             patch("v2.session_ctl.fail_session_stage") as fail, \
             patch("v2.session_ctl.persist_strategist_memo") as memo, \
             patch("v2.session_ctl.insert_llm_call_context") as ctx, \
             patch("v2.session_ctl.record_event") as events, \
             patch("v2.session_ctl.registry_tool_definitions", return_value=[{"name": "write_playbook"}]):
            out = sc.end_strategist_stage(session_id=42, session_date=date(2026, 9, 22), exit_code=0, error=None,
                                          cli_version="2.1.280 (Claude Code)", **self._paths(tmp_path))
        assert out == {"status": "completed", "error": None, "error_class": None}
        fail.assert_not_called()
        usage = complete.call_args.kwargs["usage"]
        assert usage.model == "claude-opus-4-8" and usage.cache_read_tokens == 400000
        memo.assert_called_once_with(RESULT["result"], date(2026, 9, 22), 42)
        kinds = [c.kwargs["event_type"] for c in events.call_args_list]
        assert kinds == ["agent_call", "tool_invocation", "loop_completion"]
        agent_call = events.call_args_list[0].kwargs["payload"]
        assert agent_call["backend"] == "claude_code" and agent_call["cli_version"] == "2.1.280 (Claude Code)"
        assert agent_call["model"] == "claude-opus-4-8" and agent_call["purpose"] == "strategist_loop"
        assert agent_call["cache_read_tokens"] == 400000 and agent_call["success"] is True
        assert events.call_args_list[2].kwargs["payload"]["turns_used"] == 9
        assert all(c.kwargs["stage_name"] == "ideation" for c in events.call_args_list)
        ctx.assert_called_once()
        assert ctx.call_args.kwargs["system_prompt"] == "SYSTEM"
        assert ctx.call_args.kwargs["stage_name"] == "ideation" and ctx.call_args.kwargs["purpose"] == "strategist_loop"
        assert ctx.call_args.kwargs["messages"][0] == {"role": "user", "content": "CONTEXT"}
        assert ctx.call_args.kwargs["tool_definitions"] == [{"name": "write_playbook"}]

    def test_end_validator_failure_fails_stage_and_skips_memo(self, tmp_path):
        with patch("v2.session_ctl.get_playbook", return_value=None), \
             patch("v2.session_ctl.complete_session_stage") as complete, \
             patch("v2.session_ctl.fail_session_stage") as fail, \
             patch("v2.session_ctl.persist_strategist_memo") as memo, \
             patch("v2.session_ctl.insert_llm_call_context"), \
             patch("v2.session_ctl.record_event"), \
             patch("v2.session_ctl.registry_tool_definitions", return_value=[]):
            out = sc.end_strategist_stage(session_id=42, session_date=date(2026, 9, 22), exit_code=0, error=None,
                                          cli_version="x", **self._paths(tmp_path))
        assert out["status"] == "failed" and out["error_class"] == "validator"
        complete.assert_not_called()
        memo.assert_not_called()
        assert fail.call_args.args[:2] == (42, "strategist") and fail.call_args.args[2].startswith("[validator]")
        assert fail.call_args.kwargs["usage"].model == "claude-opus-4-8"

    def test_end_usage_limit_is_classified(self, tmp_path):
        bad = {"type": "result", "is_error": True, "result": "You have hit your usage limit", "modelUsage": {}}
        with patch("v2.session_ctl.fail_session_stage") as fail, \
             patch("v2.session_ctl.complete_session_stage") as complete, \
             patch("v2.session_ctl.insert_llm_call_context"), \
             patch("v2.session_ctl.record_event") as events, \
             patch("v2.session_ctl.registry_tool_definitions", return_value=[]):
            out = sc.end_strategist_stage(session_id=42, session_date=date(2026, 9, 22), exit_code=0, error=None,
                                          cli_version="x", **self._paths(tmp_path, result=bad, tools=False))
        assert out["error_class"] == "usage_limit"
        complete.assert_not_called()
        assert fail.call_args.args[2].startswith("[usage_limit]")
        assert events.call_args_list[0].kwargs["payload"]["success"] is False

    def test_end_without_result_file_is_a_command_failure(self, tmp_path):
        with patch("v2.session_ctl.fail_session_stage") as fail, \
             patch("v2.session_ctl.insert_llm_call_context") as ctx, \
             patch("v2.session_ctl.record_event") as events:
            out = sc.end_strategist_stage(session_id=42, session_date=date(2026, 9, 22), exit_code=2, error="stage-context failed",
                                          cli_version="x", result_path=str(tmp_path / "none"), transcript_path=None,
                                          tools_log_path=None, context_path=None, system_prompt_path=None)
        assert out == {"status": "failed", "error": "[command] stage-context failed", "error_class": "command"}
        ctx.assert_not_called()
        events.assert_not_called()
        fail.assert_called_once()


class TestCli:
    def test_stage_end_cli_prints_json(self, tmp_path, capsys):
        with patch("v2.session_ctl.end_strategist_stage", return_value={"status": "completed", "error": None, "error_class": None}) as end, \
             patch("sys.argv", ["session_ctl", "stage-end", "--session-id", "42", "--stage", "strategist", "--session-date", "2026-09-22",
                                "--result", "/r", "--exit-code", "0", "--cli-version", "2.1.280"]):
            assert sc.main() == 0
        assert json.loads(capsys.readouterr().out) == {"status": "completed", "error": None, "error_class": None}
        assert end.call_args.kwargs["session_date"] == date(2026, 9, 22) and end.call_args.kwargs["cli_version"] == "2.1.280"

    def test_stage_end_rejects_other_stages(self):
        with patch("sys.argv", ["session_ctl", "stage-end", "--session-id", "1", "--stage", "executor", "--session-date", "2026-09-22"]):
            with pytest.raises(SystemExit):
                sc.main()

    def test_stage_begin_cli(self, capsys):
        with patch("v2.session_ctl.insert_session_stage") as ins, \
             patch("sys.argv", ["session_ctl", "stage-begin", "--session-id", "42", "--stage", "strategist"]):
            assert sc.main() == 0
        ins.assert_called_once_with(42, "strategist")
        assert json.loads(capsys.readouterr().out) == {"ok": True}

    def test_stage_context_cli_exit_1_on_failure(self, capsys):
        with patch("v2.session_ctl.stage_context", side_effect=RuntimeError("db down")), \
             patch("sys.argv", ["session_ctl", "stage-context", "--stage", "strategist", "--part", "message"]):
            assert sc.main() == 1
```

- [ ] **Step 3: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_ideation_claude.py -k "appendix or initial_message or shared_builders" tests/v2/test_session_ctl.py -v`
Expected: FAIL (`ImportError` for the new builders; `ModuleNotFoundError: No module named 'v2.session_ctl'`).

- [ ] **Step 4: Extract the shared builders in `ideation_claude.py`**

Replace the body of `run_strategist_loop` between `base_prompt = ...` and `result = _run_claude_loop(` with:

```python
    base_prompt = system_prompt or CLAUDE_SESSION_STRATEGIST_SYSTEM

    # T2.14: fetch orphans once and share with both consumers — formation
    # context (system prompt block) and the orphan-adoption step list.
    try:
        orphans = get_orphan_positions()
    except Exception:
        logger.exception("Failed to fetch orphan positions for strategist setup")
        orphans = []

    appendix = build_strategist_system_appendix(attribution_constraints, orphans)
    if appendix:
        base_prompt = base_prompt + "\n\n" + appendix

    initial_message = build_strategist_initial_message(orphans)
```

and add, above `run_strategist_loop`:

```python
def build_strategist_system_appendix(attribution_constraints: str, orphans: list[dict]) -> str:
    """Text appended to the strategist system prompt: attribution constraints
    (signal performance made enforceable) then the formation context. Shared
    by the API loop and session_ctl so both backends send the same prompt."""
    parts = []
    if attribution_constraints:
        parts.append(attribution_constraints)
    formation_context = build_formation_context(orphans=orphans)
    if formation_context:
        parts.append(formation_context)
    return "\n\n".join(parts)


def build_strategist_initial_message(orphans: list[dict]) -> str:
    """The pre-seeded state message that opens the strategist conversation."""
    pre_seeded = _build_pre_seeded_context()
    orphan_block, adopt_step, step_offset = _build_orphan_block(orphans=orphans)
    if orphan_block:
        pre_seeded = pre_seeded + "\n\n" + orphan_block

    open_watchlist = wl.get_open_items("ideation")
    watchlist_block = wl.format_open_watchlist_items(open_watchlist)

    return f"""Here is the current state (pre-loaded to save round-trips):

{pre_seeded}

{watchlist_block}

Now proceed with your strategist session:
1. Review the data above — do NOT re-fetch portfolio, theses, decisions, attribution, identity, rules, or strategy history
{adopt_step}{2 + step_offset}. Explore market conditions for new opportunities (use web_search, get_market_snapshot, get_news_signals)
{3 + step_offset}. Update or close stale theses, create 2-4 new ones
{4 + step_offset}. Write today's playbook using the write_playbook tool

When you've completed your work, provide a summary of your findings and actions."""
```

(The f-string is the existing text moved verbatim.)

- [ ] **Step 5: Extract `persist_strategist_memo` in `session.py`**

```python
def persist_strategist_memo(summary: str | None, session_date, session_id: int | None) -> None:
    """Save the strategist's final summary as a strategist_notes memo.

    Rejects trivially-short summaries ("all good", "done", "ok") so the
    journal isn't polluted with placeholder rows. ALGO-15: 92% of recent
    strategist_notes had collapsed to the literal "all good" (8 chars),
    erasing run-to-run continuity.
    """
    summary = (summary or "").strip()
    if len(summary) < STRATEGIST_MEMO_MIN_LENGTH:
        logger.warning("Strategist memo skipped: final_summary too short (%d chars): %r", len(summary), summary)
        return
    try:
        state = get_current_strategy_state()
        insert_strategy_memo(
            session_date=session_date,
            memo_type='strategist_notes',
            content=summary,
            strategy_state_id=state['id'] if state else None,
            session_id=session_id,
        )
        logger.info("Strategist summary saved as memo")
    except Exception as e:
        logger.warning("Could not save strategist memo: %s", e)


def _persist_strategist_memo(result: SessionResult, session_date, session_id: int | None = None) -> None:
    if result.strategist_result and result.strategist_result.final_summary:
        persist_strategist_memo(result.strategist_result.final_summary, session_date, session_id)
```

- [ ] **Step 6: Write the pilot note**

```markdown
<!-- v2/prompts/strategist-pilot-note.md — appended to the system prompt only when the strategist runs under Claude Code -->
## Runtime note

You are running headless under Claude Code. Your tools are served by the `pinchy` MCP server and appear as `mcp__pinchy__<name>` (for example `mcp__pinchy__write_playbook`); call them by those names. The tool your instructions call `web_search` is the built-in `WebSearch` tool here; use it at most 6 times. Do not read, write or edit files, and do not run commands. The session state arrives as the first user message. The stage ends when you have called `write_playbook` and resolved every open watchlist item; then reply with a summary of at least 40 characters — it is saved as the strategist memo.
```

- [ ] **Step 7: Write `v2/session_ctl.py`**

```python
# v2/session_ctl.py
"""Session control for the strategist pilot driver (session-driver.sh).

The driver is a bash sandwich: `v2.session` runs every stage but the
strategist on the API path, this module brackets the one stage that runs as
a headless Claude Code call, and `v2.session --resume` finishes the day.
Every decision (gates, validators, telemetry) lives here, in tested Python.

Stage name in session_stages: strategist. Stage name in llm_call_contexts /
agent_events: ideation. Purpose: strategist_loop. All three match the API
path so the internal dashboard and the audit checks keep working.
"""
import argparse
import json
import logging
import os
import re
import sys
from datetime import date

from . import watchlist as wl
from .attribution import build_attribution_constraints
from .claude_client import AgentPurpose, UsageAccumulator
from .database.trading_db import (
    complete_session_stage,
    fail_session_stage,
    get_completed_stages,
    get_playbook,
    get_session_for_date,
    insert_llm_call_context,
    insert_session_stage,
)
from .formation import get_orphan_positions
from .ideation_claude import build_strategist_initial_message, build_strategist_system_appendix
from .session import _trading_halted, current_market_date, persist_strategist_memo
from .telemetry import record_event

logger = logging.getLogger("session_ctl")

STAGE = "strategist"
CONTEXT_STAGE = "ideation"
PURPOSE = AgentPurpose.STRATEGIST_LOOP


def registry_tool_definitions() -> list[dict]:
    """The strategist tool list as served by the MCP server (Task 3)."""
    from .mcp_server import registry_tool_definitions as served
    return served("strategist")


# --- status ------------------------------------------------------------------

def status() -> dict:
    today = current_market_date()
    row = get_session_for_date(today)
    completed = sorted(get_completed_stages(row["id"])) if row else []
    return {
        "halted": _trading_halted(),
        "session_date": today.isoformat(),
        "session_id": row["id"] if row else None,
        "status": row.get("status") if row else None,
        "completed_stages": completed,
    }


# --- stage-context -----------------------------------------------------------

def stage_context(part: str) -> str:
    try:
        orphans = get_orphan_positions()
    except Exception:
        logger.exception("Failed to fetch orphan positions for strategist setup")
        orphans = []
    if part == "message":
        return build_strategist_initial_message(orphans)
    if part == "system_appendix":
        return build_strategist_system_appendix(build_attribution_constraints(), orphans)
    raise ValueError(f"unknown part {part!r}")


# --- stage-end helpers -------------------------------------------------------

def usage_from_result(result: dict | None) -> UsageAccumulator | None:
    """Build the accumulator session_stages expects from claude -p's modelUsage."""
    if not result or not result.get("modelUsage"):
        return None
    acc = UsageAccumulator()
    ranked = sorted(result["modelUsage"].items(),
                    key=lambda kv: -((kv[1].get("inputTokens") or 0) + (kv[1].get("cacheReadInputTokens") or 0)))
    for model, u in ranked:
        acc.add(model, _Usage(u))
    return acc


class _Usage:
    """Duck-typed stand-in for anthropic's Usage object, for UsageAccumulator.add."""

    def __init__(self, u: dict):
        self.input_tokens = u.get("inputTokens") or 0
        self.output_tokens = u.get("outputTokens") or 0
        self.cache_creation_input_tokens = u.get("cacheCreationInputTokens") or 0
        self.cache_read_input_tokens = u.get("cacheReadInputTokens") or 0


_USAGE_LIMIT_RE = re.compile(r"usage limit|rate limit|\b429\b|overloaded", re.I)
_AUTH_RE = re.compile(r"not logged in|/login|authenticat|credential|oauth", re.I)


def classify_failure(*, exit_code: int | None, result: dict | None, error: str | None) -> tuple[str, str] | None:
    """(class, message) for a failed stage, or None when nothing failed.

    The alert text carries the class so a subscription-window hit reads as
    one, not as a generic model error. `validator` is assigned by
    end_strategist_stage after these checks pass.
    """
    if exit_code == 124:
        return "timeout", "claude call exceeded the stage wall-clock timeout"
    if result and result.get("is_error"):
        text = str(result.get("result") or "")
        if _USAGE_LIMIT_RE.search(text):
            return "usage_limit", text[:500]
        if _AUTH_RE.search(text):
            return "auth", text[:500]
        return "model_error", text[:500] or "claude returned is_error"
    if exit_code not in (None, 0):
        return "command", (error or f"exit {exit_code}")[:500]
    if error:
        return "command", error[:500]
    return None


def read_tools_log(path: str | None) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def messages_from_transcript(lines: list[str], initial_context: str | None) -> tuple[list[dict], list | None, str | None]:
    """Rebuild the message list from a stream-json transcript."""
    messages, model, response = [], None, None
    if initial_context:
        messages.append({"role": "user", "content": initial_context})
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            model = ev.get("model") or model
        elif ev.get("type") in ("assistant", "user") and isinstance(ev.get("message"), dict):
            msg = {"role": ev["message"].get("role", ev["type"]), "content": ev["message"].get("content")}
            messages.append(msg)
            if msg["role"] == "assistant":
                response = msg["content"]
    return messages, response, model


def validate_strategist(session_date: date) -> str | None:
    """The post-loop checks the API path enforces: a playbook for the date
    (session._run_strategist_stage) and no open ideation watchlist item
    (ideation_claude.run_strategist_loop)."""
    if get_playbook(session_date) is None:
        return (f"Strategist finished without writing a playbook for {session_date} "
                "(likely hit the tool-call cap or the timeout before calling write_playbook)")
    try:
        wl.assert_watchlist_resolved("ideation")
    except RuntimeError as e:
        return str(e)
    return None


def _read(path: str | None) -> str | None:
    if path and os.path.exists(path):
        with open(path) as fh:
            return fh.read()
    return None


def _write_telemetry(*, session_id: int, result: dict, tools: list[dict], usage, cli_version: str | None,
                     failure: tuple[str, str] | None, transcript: str | None, context: str | None,
                     system_prompt: str | None) -> None:
    model = usage.model if usage else None
    record_event(session_id=session_id, stage_name=CONTEXT_STAGE, event_type="agent_call", payload={
        "model": model, "purpose": PURPOSE, "duration_ms": result.get("duration_api_ms"),
        "success": failure is None, "error": failure[1] if failure else None,
        "stop_reason": result.get("stop_reason"),
        "input_tokens": usage.input_tokens if usage else 0, "output_tokens": usage.output_tokens if usage else 0,
        "cache_creation_tokens": usage.cache_creation_tokens if usage else 0,
        "cache_read_tokens": usage.cache_read_tokens if usage else 0,
        "backend": "claude_code", "cli_version": cli_version,
    })
    for t in tools:
        record_event(session_id=session_id, stage_name=CONTEXT_STAGE, event_type="tool_invocation", payload={
            "tool_name": str(t.get("tool_name", "")).split("__")[-1], "args": t.get("args"), "success": t.get("success"),
            "error": t.get("error"), "duration_ms": t.get("duration_ms"), "output_chars": t.get("output_chars"),
        })
    record_event(session_id=session_id, stage_name=CONTEXT_STAGE, event_type="loop_completion", payload={
        "stop_reason": result.get("stop_reason"), "turns_used": result.get("num_turns"), "model": model,
        "input_tokens": usage.input_tokens if usage else 0, "output_tokens": usage.output_tokens if usage else 0,
        "cache_creation_tokens": usage.cache_creation_tokens if usage else 0,
        "cache_read_tokens": usage.cache_read_tokens if usage else 0,
        "backend": "claude_code",
    })
    messages, response, _ = messages_from_transcript((transcript or "").splitlines(), context)
    try:
        insert_llm_call_context(
            session_id=session_id, stage_name=CONTEXT_STAGE, purpose=PURPOSE, model=model or "unknown",
            system_prompt=system_prompt, messages=messages, tool_definitions=registry_tool_definitions(),
            response_content=response,
            input_tokens=usage.input_tokens if usage else None, output_tokens=usage.output_tokens if usage else None,
            cache_read_tokens=usage.cache_read_tokens if usage else None,
            cache_creation_tokens=usage.cache_creation_tokens if usage else None,
            stop_reason=result.get("stop_reason"), duration_ms=result.get("duration_api_ms"),
        )
    except Exception as e:
        logger.warning("Could not record llm_call_context for strategist: %s", e)


def end_strategist_stage(*, session_id: int, session_date: date, result_path: str | None, transcript_path: str | None,
                         tools_log_path: str | None, context_path: str | None, system_prompt_path: str | None,
                         exit_code: int | None, error: str | None, cli_version: str | None) -> dict:
    result = None
    raw = _read(result_path)
    if raw:
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            error = error or "result.json is not valid JSON"
    usage = usage_from_result(result)
    tools = read_tools_log(tools_log_path)

    failure = classify_failure(exit_code=exit_code, result=result, error=error)
    if failure is None:
        verr = validate_strategist(session_date)
        if verr:
            failure = ("validator", verr)
    if result is not None:
        _write_telemetry(session_id=session_id, result=result, tools=tools, usage=usage, cli_version=cli_version,
                         failure=failure, transcript=_read(transcript_path), context=_read(context_path),
                         system_prompt=_read(system_prompt_path))
    if failure:
        cls, msg = failure
        text = f"[{cls}] {msg}"
        fail_session_stage(session_id, STAGE, text, usage=usage)
        logger.error("Stage strategist failed: %s", text)
        return {"status": "failed", "error": text, "error_class": cls}
    # P2.24 ordering: the memo is written only after the playbook check passed.
    persist_strategist_memo((result or {}).get("result"), session_date, session_id)
    complete_session_stage(session_id, STAGE, usage=usage)
    return {"status": "completed", "error": None, "error_class": None}


# --- CLI ---------------------------------------------------------------------

def _emit(obj) -> None:
    print(json.dumps(obj, default=str))


def main() -> int:
    from .log_config import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(prog="session_ctl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p = sub.add_parser("stage-begin")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--stage", required=True, choices=[STAGE])

    p = sub.add_parser("stage-context")
    p.add_argument("--stage", required=True, choices=[STAGE])
    p.add_argument("--part", required=True, choices=["system_appendix", "message"])

    p = sub.add_parser("stage-end")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--stage", required=True, choices=[STAGE])
    p.add_argument("--session-date", required=True)
    for opt in ("--result", "--transcript", "--tools-log", "--context", "--system-prompt"):
        p.add_argument(opt, default=None)
    p.add_argument("--exit-code", type=int, default=None)
    p.add_argument("--error", default=None)
    p.add_argument("--cli-version", default=None)

    args = parser.parse_args()
    if args.cmd == "status":
        _emit(status())
        return 0
    if args.cmd == "stage-begin":
        insert_session_stage(args.session_id, args.stage)
        _emit({"ok": True})
        return 0
    if args.cmd == "stage-context":
        try:
            print(stage_context(args.part))
        except Exception:
            logger.exception("stage-context failed")
            return 1
        return 0
    if args.cmd == "stage-end":
        _emit(end_strategist_stage(
            session_id=args.session_id, session_date=date.fromisoformat(args.session_date),
            result_path=args.result, transcript_path=args.transcript, tools_log_path=args.tools_log,
            context_path=args.context, system_prompt_path=args.system_prompt,
            exit_code=args.exit_code, error=args.error or None, cli_version=args.cli_version,
        ))
        return 0
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
```

Note: `registry_tool_definitions` imports `v2.mcp_server` lazily; until Task 3 lands, the tests patch it (they do) and the real command would raise `ImportError` inside the `try` around `insert_llm_call_context` — acceptable for one task, since nothing runs the real command before Task 4.

- [ ] **Step 8: Extend the conftest safety net**

```python
# tests/v2/conftest.py — add after _SESSION_DB_PATCH_TARGETS
_SESSION_CTL_DB_PATCH_TARGETS = (
    # v2.session_ctl talks to the DB through these import-site names. Default
    # them to mocks so a test that forgets to patch cannot touch the paper DB.
    "v2.session_ctl.insert_session_stage",
    "v2.session_ctl.complete_session_stage",
    "v2.session_ctl.fail_session_stage",
    "v2.session_ctl.get_completed_stages",
    "v2.session_ctl.get_session_for_date",
    "v2.session_ctl.get_playbook",
    "v2.session_ctl.insert_llm_call_context",
    "v2.session_ctl.record_event",
    "v2.session_ctl.persist_strategist_memo",
    "v2.session_ctl.get_orphan_positions",
    "v2.session_ctl.build_attribution_constraints",
)
```

and inside the `ExitStack` block of `_block_social_llm_and_session_db_calls`, before `yield`:

```python
        for target in _SESSION_CTL_DB_PATCH_TARGETS:
            stack.enter_context(patch(target))
```

- [ ] **Step 9: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_session_ctl.py tests/v2/test_ideation_claude.py tests/v2/test_session.py tests/test_conftest_gate.py -v && task lint`
Expected: PASS; lint clean.

- [ ] **Step 10: Commit**

```bash
git add v2/session_ctl.py v2/ideation_claude.py v2/session.py v2/prompts/strategist-pilot-note.md tests/v2/test_session_ctl.py tests/v2/test_ideation_claude.py tests/v2/conftest.py
git commit -m "Add session_ctl: brackets the strategist stage for the Claude Code pilot

status / stage-begin / stage-context / stage-end. The strategist's prompt
appendix and initial message are shared builders used by both backends.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `v2/mcp_server.py` — stdio MCP server over the strategist registry

**Files:**
- Create: `v2/mcp_server.py`
- Modify: `v2/requirements.txt` (add `mcp>=1.2`)
- Test: `tests/v2/test_mcp_server.py`

**Interfaces:**
- CLI: `python -m v2.mcp_server --registry strategist [--session-id N] [--max-tool-calls N] [--tools-log PATH] [--model NAME]` — speaks MCP over stdio until stdin closes. `--registry` accepts only `strategist` in this plan; the choices list is where reflection/supervisor are added later.
- Python API:
  - `registry_tool_definitions(registry: str) -> list[dict]` — Anthropic-format definitions served for that registry (used by `session_ctl` for `llm_call_contexts.tool_definitions`).
  - `build_registry(registry: str, *, session_id: int | None, model: str) -> tuple[list[dict], dict[str, Callable]]` — definitions + bound handlers, the same binding `ideation_claude._run_claude_loop` uses (`create_thesis` / `adopt_thesis` with `session_id`, `resolve_watchlist_item` via `wl.make_resolve_handler(session_id=…, stage="ideation")`). `web_search` is excluded: Claude Code's `WebSearch` replaces it.
  - `ToolRunner(handlers, *, max_tool_calls: int | None, tools_log: str | None, terminal_tools: set[str])` with `.call(name: str, arguments: dict) -> tuple[str, bool]` returning `(text, is_error)`; appends one `tools.jsonl` line per call (format in Task 2); after `max_tool_calls` calls, every further call returns `("Error: tool-call cap reached (N). Finish now: call resolve_watchlist_item, write_playbook if you have not, then stop.", True)` **except** the terminal tools themselves (`write_playbook`, `resolve_watchlist_item`), which still run.
  - Tool names are served unprefixed (`get_theses`); Claude Code exposes them as `mcp__pinchy__get_theses`.
- Logging goes to stderr only (stdout is the MCP transport).

- [ ] **Step 1: Write the failing tests**

```python
# tests/v2/test_mcp_server.py
"""The MCP server must serve exactly the strategist registry the API loop
uses, bound the same way, plus the tool-call cap that replaces max_turns."""
import json
from unittest.mock import patch

import pytest

from v2 import mcp_server, tools
from v2 import watchlist as wl


def test_strategist_registry_matches_api_loop_minus_web_search():
    defs, handlers = mcp_server.build_registry("strategist", session_id=7, model="claude-opus-4-8")
    expected = [d["name"] for d in tools.TOOL_DEFINITIONS if d["name"] != "web_search"] + [wl.RESOLVE_WATCHLIST_TOOL_DEF["name"]]
    assert [d["name"] for d in defs] == expected
    assert set(handlers) == set(expected)
    assert "web_search" not in handlers


def test_session_bound_handlers_carry_session_id():
    _, handlers = mcp_server.build_registry("strategist", session_id=7, model="")
    assert handlers["create_thesis"].keywords == {"session_id": 7}
    assert handlers["adopt_thesis"].keywords == {"session_id": 7}
    with patch("v2.watchlist.make_resolve_handler") as mk:
        mcp_server.build_registry("strategist", session_id=9, model="")
    mk.assert_called_once_with(session_id=9, stage="ideation")


def test_registry_tool_definitions_are_anthropic_format():
    defs = mcp_server.registry_tool_definitions("strategist")
    assert all({"name", "input_schema"} <= set(d) for d in defs)


def test_unknown_registry_raises():
    with pytest.raises(ValueError):
        mcp_server.build_registry("executor", session_id=None, model="")


def test_to_mcp_tools_renames_input_schema():
    t = mcp_server.to_mcp_tools([{"name": "x", "description": "d", "input_schema": {"type": "object", "properties": {}}}])
    assert t[0].name == "x" and t[0].inputSchema == {"type": "object", "properties": {}}


class TestToolRunner:
    def _runner(self, tmp_path, cap=None):
        def boom(**kw):
            raise RuntimeError("bad")

        handlers = {
            "get_theses": lambda **kw: f"theses {kw}",
            "write_playbook": lambda **kw: "Playbook written",
            "boom": boom,
        }
        r = mcp_server.ToolRunner(handlers, max_tool_calls=cap, tools_log=str(tmp_path / "tools.jsonl"),
                                  terminal_tools={"write_playbook", "resolve_watchlist_item"})
        return r, None

    def test_dispatch_and_log(self, tmp_path):
        r, _ = self._runner(tmp_path)
        text, is_error = r.call("get_theses", {"status": "active"})
        assert (text, is_error) == ("theses {'status': 'active'}", False)
        line = json.loads((tmp_path / "tools.jsonl").read_text().splitlines()[0])
        assert line["tool_name"] == "get_theses" and line["args"] == {"status": "active"} and line["success"] is True
        assert line["output_chars"] == len(text) and line["result_prefix"] == text and line["error"] is None
        assert isinstance(line["duration_ms"], int)

    def test_unknown_tool_and_handler_exception_become_error_text(self, tmp_path):
        r, _ = self._runner(tmp_path)
        assert r.call("nope", {}) == ("Error: Unknown tool 'nope'", True)
        text, is_error = r.call("boom", {})
        assert text == "Error: bad" and is_error
        lines = [json.loads(x) for x in (tmp_path / "tools.jsonl").read_text().splitlines()]
        assert [x["success"] for x in lines] == [False, False] and lines[1]["error"] == "bad"

    def test_cap_blocks_non_terminal_tools_only(self, tmp_path):
        r, _ = self._runner(tmp_path, cap=2)
        r.call("get_theses", {}); r.call("get_theses", {})
        text, is_error = r.call("get_theses", {})
        assert is_error and text.startswith("Error: tool-call cap reached (2). Finish now: call resolve_watchlist_item, write_playbook")
        assert r.call("write_playbook", {}) == ("Playbook written", False)

    def test_no_log_path_is_fine(self):
        r = mcp_server.ToolRunner({"a": lambda **kw: "ok"}, max_tool_calls=None, tools_log=None, terminal_tools=set())
        assert r.call("a", {}) == ("ok", False)
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_mcp_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2.mcp_server'`

- [ ] **Step 3: Add the dependency and rebuild the image**

Append `mcp>=1.2` to `v2/requirements.txt` (above the `# Testing` block). Rebuild: `docker compose -p pinchy-paper --env-file instances/paper.env build trading && task up INSTANCE=paper`. Confirm: `docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading python -c "import mcp; print(mcp.__name__)"` prints `mcp`.

- [ ] **Step 4: Write the server**

```python
# v2/mcp_server.py
"""stdio MCP server that exposes the strategist's tool registry to Claude Code.

Started by session-driver.sh via --mcp-config as
  docker compose -p pinchy-<i> --env-file instances/<i>.env exec -i trading \
      python -m v2.mcp_server --registry strategist --session-id N ...
It serves the same definitions and bound handlers ideation_claude's API loop
uses, counts tool calls for the cap that replaces max_turns, and appends
every call to tools.jsonl for session_ctl stage-end.

Only the strategist registry is served in the pilot. Reflection and
supervisor registries are added here when their stages move.
"""
import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from functools import partial

from . import tools
from . import watchlist as wl
from .tools import reset_session

logger = logging.getLogger("mcp_server")

TERMINAL_TOOLS = {
    "strategist": {"write_playbook", "resolve_watchlist_item"},
}


def build_registry(registry: str, *, session_id: int | None, model: str) -> tuple[list[dict], dict[str, Callable]]:
    """Definitions + handlers with the same session binding as the API loop."""
    if registry == "strategist":
        defs = [d for d in tools.TOOL_DEFINITIONS if d.get("name") != "web_search"] + [wl.RESOLVE_WATCHLIST_TOOL_DEF]
        handlers = {
            **tools.TOOL_HANDLERS,
            "create_thesis": partial(tools.tool_create_thesis, session_id=session_id),
            "adopt_thesis": partial(tools.tool_adopt_thesis, session_id=session_id),
            "resolve_watchlist_item": wl.make_resolve_handler(session_id=session_id, stage="ideation"),
        }
    else:
        raise ValueError(f"unknown registry {registry!r}")
    names = {d["name"] for d in defs}
    return defs, {k: v for k, v in handlers.items() if k in names}


def registry_tool_definitions(registry: str) -> list[dict]:
    return build_registry(registry, session_id=None, model="")[0]


def to_mcp_tools(defs: list[dict]):
    import mcp.types as types
    return [types.Tool(name=d["name"], description=d.get("description", ""), inputSchema=d["input_schema"]) for d in defs]


class ToolRunner:
    """Dispatch + cap + tools.jsonl. Mirrors run_agentic_loop's dispatch contract:
    unknown tool and handler exceptions become error strings, never raises."""

    def __init__(self, handlers: dict[str, Callable], *, max_tool_calls: int | None, tools_log: str | None,
                 terminal_tools: set[str]):
        self.handlers = handlers
        self.max_tool_calls = max_tool_calls
        self.tools_log = tools_log
        self.terminal_tools = set(terminal_tools)
        self.calls = 0

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        started = time.monotonic()
        self.calls += 1
        if self.max_tool_calls and self.calls > self.max_tool_calls and name not in self.terminal_tools:
            terminal = ", ".join(sorted(self.terminal_tools)) or "your terminal tool"
            text = f"Error: tool-call cap reached ({self.max_tool_calls}). Finish now: call {terminal} if you have not, then stop."
            self._log(name, arguments, False, text, started, 0, text)
            return text, True
        handler = self.handlers.get(name)
        if handler is None:
            text = f"Error: Unknown tool '{name}'"
            self._log(name, arguments, False, text, started, 0, text)
            return text, True
        try:
            output = str(handler(**(arguments or {})))
        except Exception as e:  # noqa: BLE001 — same contract as run_agentic_loop
            text = f"Error: {e}"
            self._log(name, arguments, False, str(e), started, 0, text)
            return text, True
        self._log(name, arguments, True, None, started, len(output), output)
        return output, False

    def _log(self, name, arguments, success, error, started, output_chars, result_text):
        if not self.tools_log:
            return
        line = {
            "tool_name": name, "args": arguments, "success": success, "error": error,
            "duration_ms": int((time.monotonic() - started) * 1000), "output_chars": output_chars,
            "result_prefix": (result_text or "")[:200],
        }
        try:
            os.makedirs(os.path.dirname(self.tools_log), exist_ok=True)
            with open(self.tools_log, "a") as fh:
                fh.write(json.dumps(line, default=str) + "\n")
        except OSError:
            logger.exception("could not append to tools log")


def serve(*, mcp_tools, runner: ToolRunner) -> None:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server("pinchy")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return mcp_tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        text, _is_error = await asyncio.to_thread(runner.call, name, arguments or {})
        return [types.TextContent(type="text", text=text)]

    async def _run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")  # stderr
    parser = argparse.ArgumentParser(prog="mcp_server")
    parser.add_argument("--registry", required=True, choices=sorted(TERMINAL_TOOLS))
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--tools-log", default=None)
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    reset_session()
    defs, handlers = build_registry(args.registry, session_id=args.session_id, model=args.model)
    runner = ToolRunner(handlers, max_tool_calls=args.max_tool_calls, tools_log=args.tools_log,
                        terminal_tools=TERMINAL_TOOLS[args.registry])
    serve(mcp_tools=to_mcp_tools(defs), runner=runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Error text starts with `Error:` exactly as the API loop's tool results do, so the model reads it the same way; do not spend time on MCP's `isError` flag.

stdout is the MCP transport. Check that nothing the served modules import prints to stdout: `grep -n "print(" v2/tools.py v2/watchlist.py v2/formation.py v2/context.py` — any hit must become `logger.info`.

- [ ] **Step 5: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_mcp_server.py tests/v2/test_session_ctl.py -v && task lint`
Expected: PASS; lint clean.

- [ ] **Step 6: Manual stdio smoke (paper)**

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading python -m v2.mcp_server --registry strategist 2>/dev/null | head -c 600
```

Expected: a JSON-RPC `initialize` result followed by a `tools/list` result naming `get_portfolio_state` … `write_playbook`, `resolve_watchlist_item`, and no `web_search`.

- [ ] **Step 7: Commit**

```bash
git add v2/mcp_server.py v2/requirements.txt tests/v2/test_mcp_server.py
git commit -m "Add the pinchy MCP server: strategist registry over stdio, call cap, tools.jsonl

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `session-driver.sh`, `scripts/driver_json.py`, driver test, full smoke

**Files:**
- Create: `session-driver.sh`, `scripts/driver_json.py`
- Create: `tests/test_driver_json.py`, `tests/test_session_driver.py`

**Interfaces:**
- `./session-driver.sh <instance> [--resume]` → exit 0 when the day is clean or there was nothing to do (halted, or a session row already exists and `--resume` was not passed); exit 1 when the strategist stage failed or the second `v2.session` half exited nonzero; exit 2 on driver-level faults (no instance, no login, compose failure, no session row after the first half).
- `./session-driver.sh mcp-config <instance> [--session-id N] [--max-tool-calls N] [--tools-log PATH] [--model M]` → prints the `--mcp-config` JSON for the strategist registry (used by the smoke script).
- Environment: `PINCHY_CLAUDE_BIN` (default `claude`), `PINCHY_DOCKER_BIN` (default `docker`), `PINCHY_STRATEGIST_TIMEOUT` (seconds, default `2400`), `PINCHY_STRATEGIST_MAX_TOOL_CALLS` (default `60`). `INSTANCE`, `LOGS_DIR` and `ALGO_STAGE_MODEL_STRATEGIST` (default `claude-opus-4-8`) are read from `instances/<name>.env`.
- Artefacts on the host under `$LOGS_DIR/sessions/<date>/strategist/`: `context.md` (the first user message), `system-appendix.md`, `system-prompt.md` (base prompt + pilot note + appendix, exactly what was sent), `mcp.json`, `transcript.jsonl`, `result.json`, `tools.jsonl`, `stderr.log`. The same directory is `/app/logs/sessions/<date>/strategist/` in the container; container paths are what `stage-end` receives.
- `scripts/driver_json.py` (stdlib only, host python 3.10): `field FILE KEY` (prints the value; strings raw, other types as JSON, empty for null/missing), `contains FILE KEY VALUE` (exit 0 if `VALUE` is in the list at `KEY`), `last-result TRANSCRIPT OUT` (writes the last `"type": "result"` line to `OUT`; exit 1 if none).
- The `claude` invocation is built in one function and must include: `env -i HOME PATH CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 DISABLE_AUTOUPDATER=1`, a scratch cwd, `-p` with the context on stdin, `--model`, `--system-prompt`, `--append-system-prompt`, `--setting-sources ""`, `--no-session-persistence`, `--output-format stream-json --verbose`, `--permission-mode dontAsk --permission-prompts none`, `--strict-mcp-config --mcp-config <file>`, `--allowedTools "mcp__pinchy__*,WebSearch"`, `--disallowedTools` naming every built-in that touches files, shells or delegation, and never `--bare`.

- [ ] **Step 1: Write the failing JSON-helper tests**

```python
# tests/test_driver_json.py
import json
import subprocess
import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parent.parent / "scripts" / "driver_json.py"


def run(*args):
    return subprocess.run([sys.executable, str(HELPER), *args], capture_output=True, text=True)


def test_field_prints_strings_raw_and_others_as_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"s": "hello", "n": 42, "b": True, "l": [1, 2], "z": None}))
    assert run("field", str(p), "s").stdout == "hello\n"
    assert run("field", str(p), "n").stdout == "42\n"
    assert run("field", str(p), "b").stdout == "true\n"
    assert run("field", str(p), "l").stdout == "[1, 2]\n"
    assert run("field", str(p), "z").stdout == "\n"
    assert run("field", str(p), "missing").stdout == "\n"


def test_contains(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"completed_stages": ["learning", "strategist"]}))
    assert run("contains", str(p), "completed_stages", "strategist").returncode == 0
    assert run("contains", str(p), "completed_stages", "executor").returncode == 1
    assert run("contains", str(p), "nope", "x").returncode == 1


def test_last_result(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"system"}\nnot json\n{"type":"result","is_error":false,"result":"first"}\n{"type":"result","is_error":false,"result":"last"}\n')
    out = tmp_path / "r.json"
    assert run("last-result", str(t), str(out)).returncode == 0
    assert json.loads(out.read_text())["result"] == "last"


def test_last_result_missing_exits_1(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"system"}\n')
    assert run("last-result", str(t), str(tmp_path / "r.json")).returncode == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/test_driver_json.py -v`
Expected: FAIL (`FileNotFoundError` for `scripts/driver_json.py`).

- [ ] **Step 3: Write the helper**

```python
#!/usr/bin/env python3
"""JSON helpers for session-driver.sh. Stdlib only — runs on the host's
python3 (3.10) as well as in the container. Keeps jq out of the host deps."""
import json
import sys


def cmd_field(path, key):
    with open(path) as fh:
        value = json.load(fh).get(key)
    if value is None:
        print("")
    elif isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value))
    return 0


def cmd_contains(path, key, value):
    with open(path) as fh:
        items = json.load(fh).get(key) or []
    return 0 if value in items else 1


def cmd_last_result(transcript, out):
    last = None
    with open(transcript) as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and ev.get("type") == "result":
                last = ev
    if last is None:
        return 1
    with open(out, "w") as fh:
        json.dump(last, fh)
    return 0


def main(argv):
    cmd, args = argv[1], argv[2:]
    if cmd == "field":
        return cmd_field(*args)
    if cmd == "contains":
        return cmd_contains(*args)
    if cmd == "last-result":
        return cmd_last_result(*args)
    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

`chmod +x scripts/driver_json.py`.

- [ ] **Step 4: Write the failing driver test**

```python
# tests/test_session_driver.py
"""session-driver.sh is a sandwich: v2.session (API path) for the stages
before the strategist, one claude -p call bracketed by session_ctl, then
v2.session --resume. These tests stub `docker` and `claude` and assert the
sequence, the claude flags, and the exit codes."""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "session-driver.sh"

STATUS_BEFORE = {"halted": False, "session_date": "2026-09-22", "session_id": None, "status": None, "completed_stages": []}
STATUS_AFTER = {"halted": False, "session_date": "2026-09-22", "session_id": 42, "status": "completed",
                "completed_stages": ["learning", "pipeline", "supervisor"]}

DOCKER_STUB = r'''#!/bin/bash
# Fake `docker compose ... exec -T trading <cmd...>` / `docker compose ... up -d`.
LOG="$STUB_DIR/docker.log"
while [ $# -gt 0 ] && [ "$1" != "trading" ] && [ "$1" != "up" ]; do shift; done
[ "$1" = "trading" ] && shift
echo "$*" >> "$LOG"
case "$*" in
  *"session_ctl status"*)
     n=$(cat "$STUB_DIR/status.n" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$STUB_DIR/status.n"
     f="$STUB_DIR/status.$n.json"; [ -f "$f" ] || f="$STUB_DIR/status.2.json"; cat "$f" ;;
  *"session_ctl stage-begin"*) echo '{"ok": true}' ;;
  *"session_ctl stage-context"*"--part message"*) echo "CONTEXT MESSAGE"; exit "${CONTEXT_EXIT:-0}" ;;
  *"session_ctl stage-context"*"--part system_appendix"*) echo "APPENDIX TEXT" ;;
  *"session_ctl stage-end"*) cat "$STUB_DIR/stage-end.json" ;;
  *"v2.session"*) exit "${SESSION_EXIT:-0}" ;;
  "up -d") : ;;
esac
exit 0
'''

CLAUDE_STUB = r'''#!/bin/bash
if [ "$1" = "--version" ]; then echo "9.9.9 (stub)"; exit 0; fi   # not logged: only real calls create claude.log
LOG="$STUB_DIR/claude.log"
printf '%s\n' "$*" >> "$LOG"
env > "$STUB_DIR/claude.env"
pwd > "$STUB_DIR/claude.cwd"
cat > "$STUB_DIR/claude.stdin"
if [ -n "${CLAUDE_FAIL:-}" ]; then echo '{"type":"result","is_error":true,"result":"You have hit your usage limit"}'; exit 0; fi
echo '{"type":"system","subtype":"init","model":"claude-opus-4-8"}'
echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]}}'
echo '{"type":"result","is_error":false,"result":"Playbook written and theses reviewed in detail today.","modelUsage":{"claude-opus-4-8":{"inputTokens":1,"outputTokens":1}}}'
'''


@pytest.fixture
def harness(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    for name, body in (("docker", DOCKER_STUB), ("claude", CLAUDE_STUB)):
        p = stub / name; p.write_text(body); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "status.1.json").write_text(json.dumps(STATUS_BEFORE))
    (stub / "status.2.json").write_text(json.dumps(STATUS_AFTER))
    (stub / "stage-end.json").write_text(json.dumps({"status": "completed", "error": None, "error_class": None}))
    inst = tmp_path / "instances"; inst.mkdir()
    (inst / "t.env").write_text(f"INSTANCE=t\nLOGS_DIR={logs}\nALGO_STAGE_MODEL_STRATEGIST=claude-opus-4-8\n")
    shutil.copy(DRIVER, tmp_path / "session-driver.sh")
    shutil.copytree(REPO / "scripts", tmp_path / "scripts")
    (tmp_path / "v2" / "prompts").mkdir(parents=True)
    (tmp_path / "v2" / "prompts" / "strategist.md").write_text("You are the strategist for an automated trading system.")
    (tmp_path / "v2" / "prompts" / "strategist-pilot-note.md").write_text("## Runtime note\nPILOT NOTE")
    env = dict(os.environ, STUB_DIR=str(stub), PINCHY_DOCKER_BIN=str(stub / "docker"), PINCHY_CLAUDE_BIN=str(stub / "claude"),
               ANTHROPIC_API_KEY="must-not-leak")
    env.pop("CLAUDE_CODE_DISABLE_AUTO_MEMORY", None)

    def run(*args, **extra):
        return subprocess.run(["bash", str(tmp_path / "session-driver.sh"), "t", *args], capture_output=True, text=True,
                              env=dict(env, **extra), cwd=tmp_path)
    return run, stub, logs


def test_halted_exits_zero_without_running_anything(harness):
    run, stub, _ = harness
    (stub / "status.1.json").write_text(json.dumps(dict(STATUS_BEFORE, halted=True)))
    r = run()
    assert r.returncode == 0 and "halted" in r.stdout
    assert "v2.session" not in (stub / "docker.log").read_text()
    assert not (stub / "claude.log").exists()


def test_existing_session_without_resume_is_a_noop(harness):
    run, stub, _ = harness
    (stub / "status.1.json").write_text(json.dumps(STATUS_AFTER))
    r = run()
    assert r.returncode == 0 and "--resume" in r.stdout
    assert "v2.session" not in (stub / "docker.log").read_text()
    assert not (stub / "claude.log").exists()


def test_happy_path_sequences_the_sandwich(harness):
    run, stub, logs = harness
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    docker = (stub / "docker.log").read_text().splitlines()
    first = [i for i, l in enumerate(docker) if "v2.session --skip-ideation --skip-executor --skip-strategy --skip-dashboard" in l]
    begin = [i for i, l in enumerate(docker) if "session_ctl stage-begin --session-id 42 --stage strategist" in l]
    end = [i for i, l in enumerate(docker) if "session_ctl stage-end --session-id 42 --stage strategist --session-date 2026-09-22" in l]
    second = [i for i, l in enumerate(docker) if l.strip() == "python -m v2.session --resume"]
    assert len(first) == len(begin) == len(end) == len(second) == 1
    assert first[0] < begin[0] < end[0] < second[0]
    assert "--resume" not in docker[first[0]]
    claude = (stub / "claude.log").read_text()
    assert "--bare" not in claude
    for flag in ("--model claude-opus-4-8", "--output-format stream-json", "--verbose", "--setting-sources ", "--no-session-persistence",
                 "--permission-mode dontAsk", "--permission-prompts none", "--strict-mcp-config", "--mcp-config",
                 "--allowedTools mcp__pinchy__*,WebSearch", "--disallowedTools", "--system-prompt You are the strategist",
                 "--append-system-prompt ## Runtime note"):
        assert flag in claude, flag
    assert "APPENDIX TEXT" in claude  # appendix rides on --append-system-prompt, after the pilot note
    assert (stub / "claude.stdin").read_text().strip() == "CONTEXT MESSAGE"
    cenv = (stub / "claude.env").read_text()
    assert "ANTHROPIC_API_KEY" not in cenv and "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1" in cenv and "DISABLE_AUTOUPDATER=1" in cenv
    assert (stub / "claude.cwd").read_text().strip() != str(logs.parent)  # scratch cwd, not the repo
    sdir = logs / "sessions" / "2026-09-22" / "strategist"
    assert (sdir / "result.json").exists() and (sdir / "transcript.jsonl").exists()
    assert (sdir / "system-prompt.md").read_text().startswith("You are the strategist")
    assert "PILOT NOTE" in (sdir / "system-prompt.md").read_text() and "APPENDIX TEXT" in (sdir / "system-prompt.md").read_text()
    cfg = json.loads((sdir / "mcp.json").read_text())["mcpServers"]["pinchy"]
    assert cfg["args"][:8] == ["compose", "-p", "pinchy-t", "--env-file", "instances/t.env", "exec", "-i", "trading"]
    assert "--session-id" in cfg["args"] and "42" in cfg["args"] and "--max-tool-calls" in cfg["args"]
    endline = docker[end[0]]
    for part in ("--result /app/logs/sessions/2026-09-22/strategist/result.json", "--transcript /app/logs/sessions/2026-09-22/strategist/transcript.jsonl",
                 "--tools-log /app/logs/sessions/2026-09-22/strategist/tools.jsonl", "--context /app/logs/sessions/2026-09-22/strategist/context.md",
                 "--system-prompt /app/logs/sessions/2026-09-22/strategist/system-prompt.md", "--exit-code 0", "--cli-version 9.9.9 (stub)"):
        assert part in endline, part


def test_resume_passes_through_to_both_halves(harness):
    run, stub, _ = harness
    (stub / "status.1.json").write_text(json.dumps(STATUS_AFTER))
    r = run("--resume")
    assert r.returncode == 0
    docker = (stub / "docker.log").read_text()
    assert "v2.session --resume --skip-ideation --skip-executor --skip-strategy --skip-dashboard" in docker
    assert (stub / "claude.log").exists()


def test_completed_strategist_is_skipped_on_resume(harness):
    run, stub, _ = harness
    done = dict(STATUS_AFTER, completed_stages=STATUS_AFTER["completed_stages"] + ["strategist"])
    (stub / "status.1.json").write_text(json.dumps(done))
    (stub / "status.2.json").write_text(json.dumps(done))
    r = run("--resume")
    assert r.returncode == 0
    assert not (stub / "claude.log").exists()
    assert "stage-begin" not in (stub / "docker.log").read_text()


def test_strategist_failure_still_runs_second_half_and_exits_1(harness):
    run, stub, _ = harness
    (stub / "stage-end.json").write_text(json.dumps({"status": "failed", "error": "[usage_limit] ...", "error_class": "usage_limit"}))
    r = run(CLAUDE_FAIL="1")
    assert r.returncode == 1
    docker = (stub / "docker.log").read_text()
    assert "python -m v2.session --resume" in docker
    assert "usage_limit" in r.stdout


def test_second_half_failure_exits_1(harness):
    run, _, _ = harness
    r = run(SESSION_EXIT="1")
    assert r.returncode == 1


def test_stage_context_failure_fails_stage_without_calling_claude(harness):
    run, stub, _ = harness
    (stub / "stage-end.json").write_text(json.dumps({"status": "failed", "error": "[command] stage-context failed", "error_class": "command"}))
    r = run(CONTEXT_EXIT="1")
    assert r.returncode == 1
    assert not (stub / "claude.log").exists()
    assert "--exit-code 2 --error stage-context failed" in (stub / "docker.log").read_text()


def test_no_session_after_first_half_is_a_driver_fault(harness):
    run, stub, _ = harness
    (stub / "status.2.json").write_text(json.dumps(STATUS_BEFORE))
    r = run()
    assert r.returncode == 2 and "refusing" in r.stdout


def test_mcp_config_subcommand(harness, tmp_path):
    r = subprocess.run(["bash", str(tmp_path / "session-driver.sh"), "mcp-config", "t", "--session-id", "7", "--max-tool-calls", "60"],
                       capture_output=True, text=True, cwd=tmp_path)
    cfg = json.loads(r.stdout)["mcpServers"]["pinchy"]
    assert cfg["command"] == "docker"
    assert cfg["args"][8:12] == ["python", "-m", "v2.mcp_server", "--registry"] and cfg["args"][12] == "strategist"
    assert "7" in cfg["args"] and "60" in cfg["args"]
```

- [ ] **Step 5: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/test_driver_json.py tests/test_session_driver.py -v`
Expected: `test_driver_json.py` PASS; `test_session_driver.py` FAIL (`session-driver.sh` missing → `shutil.copy` raises `FileNotFoundError`).

- [ ] **Step 6: Write `session-driver.sh`**

```bash
#!/bin/bash
#
# session-driver.sh <instance> [--resume]
# session-driver.sh mcp-config <instance> [--session-id N] [--max-tool-calls N] [--tools-log PATH] [--model M]
#
# Strategist pilot: run the daily session with the strategist stage on the
# operator's Claude subscription (headless Claude Code) and every other stage
# on the API path. Runs UNDER cron-wrap.sh (which owns HALT, heartbeat,
# alerts). Sandwich:
#   1. v2.session  --skip-ideation --skip-executor --skip-strategy --skip-dashboard
#   2. session_ctl stage-begin / stage-context → claude -p → session_ctl stage-end
#   3. v2.session --resume
# Nothing here decides anything: v2/session_ctl.py validates the strategist's
# output and writes telemetry; v2/session.py owns the rest.
# Plan: docs/superpowers/plans/2026-09-22-strategist-pilot.md
#
# NEVER pass --bare: bare mode does not read the subscription (OAuth) login.
# NEVER drop `env -i`: it is what keeps ANTHROPIC_API_KEY away from claude.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 2
CLAUDE_BIN="${PINCHY_CLAUDE_BIN:-claude}"
DOCKER_BIN="${PINCHY_DOCKER_BIN:-docker}"
PYJSON="python3 $SCRIPT_DIR/scripts/driver_json.py"
CONTAINER_LOGS=/app/logs
STRATEGIST_TIMEOUT="${PINCHY_STRATEGIST_TIMEOUT:-2400}"
STRATEGIST_MAX_TOOL_CALLS="${PINCHY_STRATEGIST_MAX_TOOL_CALLS:-60}"

DRIVER_LOG=""   # set once LOGS_DIR is known; log() appends there as well as to stdout
log() { local m; m="$(date '+%Y-%m-%d %H:%M:%S') [driver] $*"; echo "$m"; [ -n "$DRIVER_LOG" ] && echo "$m" >> "$DRIVER_LOG"; }
die() { log "FATAL: $*"; exit 2; }

load_instance() {
    INSTANCE="$1"; ENV_FILE="instances/$INSTANCE.env"
    [ -f "$ENV_FILE" ] || die "no such instance: $ENV_FILE"
    grep -qx "INSTANCE=$INSTANCE" "$ENV_FILE" || die "$ENV_FILE must contain the line INSTANCE=$INSTANCE"
    LOGS_DIR="$(grep -E '^LOGS_DIR=' "$ENV_FILE" | tail -1 | cut -d= -f2-)"
    [ -n "$LOGS_DIR" ] || die "LOGS_DIR missing in $ENV_FILE"
    case "$LOGS_DIR" in /*) ;; *) LOGS_DIR="$SCRIPT_DIR/$LOGS_DIR" ;; esac
    STRATEGIST_MODEL="$(grep -E '^ALGO_STAGE_MODEL_STRATEGIST=' "$ENV_FILE" | tail -1 | cut -d= -f2-)"
    STRATEGIST_MODEL="${STRATEGIST_MODEL:-claude-opus-4-8}"
    COMPOSE=("$DOCKER_BIN" compose -p "pinchy-$INSTANCE" --env-file "$ENV_FILE")
}

in_container() { "${COMPOSE[@]}" exec -T trading "$@"; }
ctl() { in_container python -m v2.session_ctl "$@"; }
session() {  # the API-path halves; their output is kept next to the driver's own log (pipefail keeps the exit status)
    in_container python -m v2.session "$@" 2>&1 | tee -a "$DRIVER_LOG"
}
host_path() { echo "${1/#$CONTAINER_LOGS/$LOGS_DIR}"; }   # container path -> host path
field() { $PYJSON field "$1" "$2"; }

mcp_config_json() {
    # optional --session-id/--max-tool-calls/--tools-log/--model, passed through to v2.mcp_server
    python3 - "$INSTANCE" "$@" <<'EOF'
import json, sys
inst, *rest = sys.argv[1:]
args = ["compose", "-p", f"pinchy-{inst}", "--env-file", f"instances/{inst}.env", "exec", "-i", "trading",
        "python", "-m", "v2.mcp_server", "--registry", "strategist"] + rest
print(json.dumps({"mcpServers": {"pinchy": {"command": "docker", "args": args}}}))
EOF
}

# --- the claude invocation (the ONLY place flags live) -----------------------
claude_strategist() {
    # $1 system prompt text  $2 appended system prompt text  $3 mcp config file
    # stdin: the first user message   stdout: stream-json transcript
    local scratch; scratch="$(mktemp -d)"
    ( cd "$scratch" && env -i HOME="$HOME" PATH="$PATH" CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 DISABLE_AUTOUPDATER=1 \
        timeout "$STRATEGIST_TIMEOUT" "$CLAUDE_BIN" -p \
            --model "$STRATEGIST_MODEL" \
            --system-prompt "$1" \
            --append-system-prompt "$2" \
            --setting-sources "" \
            --no-session-persistence \
            --output-format stream-json --verbose \
            --permission-mode dontAsk --permission-prompts none \
            --strict-mcp-config --mcp-config "$3" \
            --allowedTools "mcp__pinchy__*,WebSearch" \
            --disallowedTools "AskUserQuestion,Edit,Write,NotebookEdit,Agent,Task,Bash,Read,Glob,Grep,WebFetch,Skill" )
    local status=$?
    rm -rf "$scratch"
    return $status
}

run_strategist() {
    local cdir="$CONTAINER_LOGS/sessions/$SESSION_DATE/strategist" hdir status err
    hdir="$(host_path "$cdir")"; mkdir -p "$hdir"
    ctl stage-begin --session-id "$SESSION_ID" --stage strategist >/dev/null || log "stage-begin failed (continuing)"
    if ! ctl stage-context --stage strategist --part message > "$hdir/context.md" \
       || ! ctl stage-context --stage strategist --part system_appendix > "$hdir/system-appendix.md"; then
        ctl stage-end --session-id "$SESSION_ID" --stage strategist --session-date "$SESSION_DATE" \
            --exit-code 2 --error "stage-context failed" --cli-version "$CLI_VERSION" > "$TMP/end.json"
        STRATEGIST_STATUS="$(field "$TMP/end.json" status)"
        log "strategist: $STRATEGIST_STATUS $(field "$TMP/end.json" error)"
        return
    fi
    local base append
    base="$(cat "$SCRIPT_DIR/v2/prompts/strategist.md")"
    append="$(cat "$SCRIPT_DIR/v2/prompts/strategist-pilot-note.md")"
    [ -s "$hdir/system-appendix.md" ] && append="$append"$'\n\n'"$(cat "$hdir/system-appendix.md")"
    printf '%s\n\n%s\n' "$base" "$append" > "$hdir/system-prompt.md"   # exactly what is sent; stage-end stores it
    mcp_config_json --session-id "$SESSION_ID" --max-tool-calls "$STRATEGIST_MAX_TOOL_CALLS" \
        --tools-log "$cdir/tools.jsonl" --model "$STRATEGIST_MODEL" > "$hdir/mcp.json"
    claude_strategist "$base" "$append" "$hdir/mcp.json" < "$hdir/context.md" > "$hdir/transcript.jsonl" 2> "$hdir/stderr.log"
    status=$?
    $PYJSON last-result "$hdir/transcript.jsonl" "$hdir/result.json" || rm -f "$hdir/result.json"
    err=""
    if [ "$status" -ne 0 ]; then
        err="$(tail -c 400 "$hdir/stderr.log" 2>/dev/null | tr '\n' ' ')"; err="${err:-exit $status}"
    fi
    ctl stage-end --session-id "$SESSION_ID" --stage strategist --session-date "$SESSION_DATE" \
        --result "$cdir/result.json" --transcript "$cdir/transcript.jsonl" --tools-log "$cdir/tools.jsonl" \
        --context "$cdir/context.md" --system-prompt "$cdir/system-prompt.md" \
        --exit-code "$status" --error "$err" --cli-version "$CLI_VERSION" > "$TMP/end.json"
    STRATEGIST_STATUS="$(field "$TMP/end.json" status)"
    log "strategist: ${STRATEGIST_STATUS:-unknown} $(field "$TMP/end.json" error)"
}

# --- entry -------------------------------------------------------------------
[ $# -ge 1 ] || die "usage: $0 <instance> [--resume] | mcp-config <instance> [...]"
if [ "$1" = "mcp-config" ]; then
    shift; load_instance "$1"; shift; mcp_config_json "$@"; exit 0
fi

load_instance "$1"; shift
RESUME=""
for arg in "$@"; do
    case "$arg" in --resume) RESUME="--resume" ;; *) die "unknown flag $arg (only --resume; use 'task session' for other flags)" ;; esac
done
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$LOGS_DIR"; DRIVER_LOG="$LOGS_DIR/driver.log"
log "start instance=$INSTANCE resume=${RESUME:-no}"

if [ -z "${PINCHY_CLAUDE_BIN:-}" ]; then   # real claude: refuse to start without a subscription login
    [ -f "$HOME/.claude/.credentials.json" ] || die "no Claude Code login at ~/.claude/.credentials.json — run 'claude' once interactively"
fi
CLI_VERSION="$("$CLAUDE_BIN" --version 2>/dev/null | head -1)"
"${COMPOSE[@]}" up -d >/dev/null || die "compose up failed"

ctl status > "$TMP/status.json" || die "session_ctl status failed"
if [ "$(field "$TMP/status.json" halted)" = "true" ]; then log "halted (ALGO_TRADING_HALTED) — nothing to do"; exit 0; fi
if [ -n "$(field "$TMP/status.json" session_id)" ] && [ -z "$RESUME" ]; then
    log "session already exists for $(field "$TMP/status.json" session_date) (status=$(field "$TMP/status.json" status)); re-run with --resume to continue it"
    exit 0
fi

# 1. everything before the strategist, on the API path (creates or reopens the session row)
# shellcheck disable=SC2086
session $RESUME --skip-ideation --skip-executor --skip-strategy --skip-dashboard
log "first half exit=$?"

ctl status > "$TMP/status.json" || die "session_ctl status failed"
SESSION_ID="$(field "$TMP/status.json" session_id)"
SESSION_DATE="$(field "$TMP/status.json" session_date)"
[ -n "$SESSION_ID" ] || die "no session row after the first half — refusing to run the strategist untracked"

# 2. the strategist, under the subscription
STRATEGIST_STATUS=""
if $PYJSON contains "$TMP/status.json" completed_stages strategist; then
    log "strategist already completed in a prior run — skipping"; STRATEGIST_STATUS=completed
else
    run_strategist
fi

# 3. executor, reflection, dashboard — resuming past everything completed above
session --resume
SECOND=$?
log "finished strategist=${STRATEGIST_STATUS:-unknown} second-half exit=$SECOND"
[ "$STRATEGIST_STATUS" = "completed" ] && [ "$SECOND" -eq 0 ] && exit 0
exit 1
```

`chmod +x session-driver.sh`.

- [ ] **Step 7: Run the tests**

Run: `task test INSTANCE=paper -- tests/test_driver_json.py tests/test_session_driver.py -v && task lint`
Expected: PASS; lint clean (ruff ignores shell; `scripts/driver_json.py` must pass).

- [ ] **Step 8: Full smoke, including the MCP server**

Run: `WITH_MCP=1 task driver:smoke INSTANCE=paper`
Expected: five `PASS` lines. If `mcp` fails with the server absent from the tool list, inspect `docker compose -p pinchy-paper --env-file instances/paper.env logs trading` and the stderr of the stream-json run; the usual cause is a `print` to stdout in a served module.

- [ ] **Step 9: One real run, by hand**

On a trading day after the cron's API run has completed (so `--resume` has a row to reopen and every other stage is already `completed`):

```bash
./session-driver.sh paper --resume
```

Expected in `logs/paper/driver.log`: first half logs every stage `SKIPPED (completed in prior run)`; `strategist: completed`; second half skips everything and finalises. Then check:

```sql
SELECT stage_name, status, model, input_tokens, cache_read_tokens, error FROM session_stages WHERE session_id = <id> ORDER BY id;
SELECT payload->>'backend', payload->>'cli_version', payload->>'stop_reason' FROM agent_events WHERE session_id = <id> AND event_type = 'agent_call' AND stage_name = 'ideation';
SELECT count(*) FROM agent_events WHERE session_id = <id> AND event_type = 'tool_invocation' AND stage_name = 'ideation';
SELECT id, length(system_prompt), jsonb_array_length(messages) FROM llm_call_contexts WHERE session_id = <id> AND purpose = 'strategist_loop';
SELECT * FROM session_stage_costs WHERE session_id = <id> AND stage_name = 'strategist';
```

and read `logs/paper/sessions/<date>/strategist/system-prompt.md` once, end to end: it must be the strategist prompt, the runtime note, and the day's appendix, nothing else. Note that this hand run writes a second playbook and memo for the date on top of the API run's; that is expected for the one-off check and is why the comparison schedule in Task 5 never runs both backends on the same day.

- [ ] **Step 10: Commit**

```bash
git add session-driver.sh scripts/driver_json.py tests/test_driver_json.py tests/test_session_driver.py
git commit -m "Add session-driver.sh: the strategist pilot sandwich

v2.session on the API path around one isolated, non-bare claude -p call
for the strategist, then v2.session --resume. Stubbed driver tests pin the
flags, the env isolation and the exit codes.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Schedule, knobs, docs, and the comparison protocol

**Files:**
- Modify: `Taskfile.yml` (add `session:pilot`), `crontab`, `instances/example.env`, `CLAUDE.md`, `docs/runbook-recovery.md`
- Create: `docs/superpowers/specs/2026-09-22-strategist-pilot-comparison.md`

**Interfaces:**
- `task session:pilot INSTANCE=<name> [-- --resume]` runs `./session-driver.sh <name> [--resume]`.
- Crontab: paper runs the API path Mon/Wed/Fri and the pilot Tue/Thu, both under `cron-wrap.sh --instance paper paper-session` so the heartbeat and alerting are shared and the dead-man's switch keeps its daily cadence.
- `ALGO_STAGE_MODEL_STRATEGIST` documented in `instances/example.env` (commented, default `claude-opus-4-8`).

- [ ] **Step 1: Taskfile target**

```yaml
  session:pilot:
    desc: Run the daily session with the strategist on the Claude subscription (strategist pilot; other stages on the API path)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    cmds:
      - ./session-driver.sh {{.INSTANCE}} {{.CLI_ARGS}}
```

- [ ] **Step 2: Crontab**

Replace the single paper session line in `crontab` with:

```
# Paper session (12:30 PM MST / 2:30 PM ET). Strategist pilot A/B, two weeks
# from 2026-09-29: Mon/Wed/Fri on the API path, Tue/Thu on the Claude Code
# driver. Same label so the heartbeat stays daily. Decision protocol:
# docs/superpowers/specs/2026-09-22-strategist-pilot-comparison.md
30 12 * * 1,3,5 cd /home/jay/dev/algo/ && ./cron-wrap.sh --instance paper paper-session task session INSTANCE=paper
30 12 * * 2,4   cd /home/jay/dev/algo/ && ./cron-wrap.sh --instance paper paper-session ./session-driver.sh paper
```

Install with `crontab /home/jay/dev/algo/crontab` and confirm with `crontab -l`.

- [ ] **Step 3: Instance knob**

Append to `instances/example.env`:

```
# Strategist pilot (session-driver.sh): model for the strategist stage when it
# runs under Claude Code. Full id with a model_pricing row. Unset = claude-opus-4-8.
# ALGO_STAGE_MODEL_STRATEGIST=claude-opus-4-8
```

- [ ] **Step 4: CLAUDE.md**

Under "Commands", after the `--resume` example, add:

```bash
# Strategist pilot: strategist on the Claude subscription, everything else on the API
task session:pilot INSTANCE=paper
task session:pilot INSTANCE=paper -- --resume   # continue a day whose session row exists

# After every Claude Code upgrade
WITH_MCP=1 task driver:smoke INSTANCE=paper
```

Under "Key v2 Modules" add:

- **`session_ctl.py`** — Brackets the strategist stage when it runs under Claude Code (`session-driver.sh`): `status`, `stage-begin`, `stage-context`, `stage-end` (validators, usage, telemetry). Plan: `docs/superpowers/plans/2026-09-22-strategist-pilot.md`.
- **`mcp_server.py`** — stdio MCP server serving the strategist tool registry to Claude Code; tool-call cap; `tools.jsonl`.

Under "Environment Variables", host-side section, add: the driver runs `claude` under `env -i`, so nothing from the shell reaches it except `HOME` and `PATH`; `ALGO_STAGE_MODEL_STRATEGIST` is read from the instance env file by the driver.

- [ ] **Step 5: Runbook**

In `docs/runbook-recovery.md` under "Halt / Resume" add a subsection:

```markdown
### Strategist pilot (Claude subscription)

On Tue/Thu the paper strategist runs as `claude -p` under the operator's
Claude Max login (`session-driver.sh`, under `cron-wrap.sh` like every job).
Failure classes in `session_stages.error` for `strategist`:

- `[usage_limit]` — the subscription window was exhausted. Interactive Claude
  Code use in the hours before 12:30 MST shares that window; avoid it on
  pilot days, or rerun later with `task session:pilot INSTANCE=paper -- --resume`.
- `[auth]` — the login expired or was revoked. Run `claude` interactively,
  `/login`, then `task driver:smoke INSTANCE=paper` and rerun with `--resume`.
- `[timeout]` — the 40-minute wall clock fired. Check
  `logs/paper/sessions/<date>/strategist/tools.jsonl` for a loop.
- `[validator]` — the run ended without a playbook or with an open watchlist
  item. The executor ran on the previous playbook; read `transcript.jsonl`.
- `[model_error]` / `[command]` — read `stderr.log` in the same directory.

A failed strategist stage never blocks the executor: the day continues with
the existing playbook, exactly as on the API path. To stop the pilot, restore
the single `task session` crontab line; nothing else needs undoing. After a
Claude Code upgrade run `WITH_MCP=1 task driver:smoke INSTANCE=paper` before
the next pilot day.
```

- [ ] **Step 6: The comparison protocol**

```markdown
# Strategist Pilot — Comparison Protocol

**Window:** two calendar weeks from the first pilot day (Tue 2026-09-29 if the
driver merges by then). Mon/Wed/Fri API, Tue/Thu Claude Code. About six API
and four pilot sessions.

**Do not** use Claude Code interactively between 10:30 and 13:30 MST on
pilot days, so a usage-window hit is a real signal, not self-inflicted.

## What to pull (after the window)

```sql
-- one row per strategist run, both backends
WITH runs AS (
  SELECT s.session_date, st.session_id, st.status, st.error,
         COALESCE(ev.payload->>'backend', 'api') AS backend,
         ev.payload->>'cli_version' AS cli_version,
         st.input_tokens, st.output_tokens, st.cache_creation_tokens, st.cache_read_tokens
  FROM session_stages st
  JOIN sessions s ON s.id = st.session_id
  LEFT JOIN LATERAL (
    SELECT payload FROM agent_events e
    WHERE e.session_id = st.session_id AND e.stage_name = 'ideation' AND e.event_type = 'agent_call'
    ORDER BY e.id DESC LIMIT 1
  ) ev ON true
  WHERE st.stage_name = 'strategist' AND s.session_date >= '2026-09-22'
)
SELECT r.*,
       c.cost_usd AS list_price_usd,
       (SELECT count(*) FROM agent_events e WHERE e.session_id = r.session_id AND e.stage_name = 'ideation' AND e.event_type = 'tool_invocation') AS tool_calls,
       (SELECT count(*) FROM playbook_actions pa JOIN playbooks p ON p.id = pa.playbook_id WHERE p.playbook_date = r.session_date) AS playbook_actions,
       (SELECT count(*) FROM theses t WHERE t.created_at::date = r.session_date) AS theses_created
FROM runs r
LEFT JOIN session_stage_costs c ON c.session_id = r.session_id AND c.stage_name = 'strategist'
ORDER BY r.session_date;
```

Also read, by hand, every pilot-day strategist memo (`strategy_memos` where
`memo_type = 'strategist_notes'`) and each `system-prompt.md` under
`logs/paper/sessions/<date>/strategist/`.

## Decision

Continue to the next stage (reflection) only if **all** hold:

1. Every scheduled pilot run reached `strategist: completed` — no
   `usage_limit`, `auth`, or `timeout` class at all in the window.
2. No memo or playbook on a pilot day mentions files, CLAUDE.md, Claude Code,
   memory, or any tool that is not in the strategist registry. (Harness
   leakage.)
3. Pilot list-price cost per run is within 2× of the API median (cache
   behaviour differs; a large gap means the context is being rebuilt).
4. Pilot tool-call counts and playbook-action counts fall inside the API
   runs' min–max range. Outside it is not automatically bad, but it needs an
   explanation from the transcripts before continuing.

Any failure of 1 or 2 ends the pilot: restore the single crontab line and
write the reason under "Outcome" below. 3 or 4 failing means another two
weeks with the cause addressed, not a stop.

## Outcome

_(filled in after the window)_
```

- [ ] **Step 7: Commit**

```bash
git add Taskfile.yml crontab instances/example.env CLAUDE.md docs/runbook-recovery.md docs/superpowers/specs/2026-09-22-strategist-pilot-comparison.md
git commit -m "Strategist pilot: alternating paper schedule, knob, runbook, comparison protocol

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Rollout (after the branch merges; manual, in this order)

1. `docker compose -p pinchy-paper --env-file instances/paper.env build trading && task up INSTANCE=paper` (the image gains `mcp`).
2. `WITH_MCP=1 task driver:smoke INSTANCE=paper`.
3. Task 4 Step 9's hand run with `--resume` on a day the API cron already completed. Read the five queries and `system-prompt.md`.
4. `crontab /home/jay/dev/algo/crontab`. Confirm the two paper lines with `crontab -l`.
5. Two weeks. Then the comparison protocol's decision, written into its "Outcome" section, and either a follow-up plan for the reflection stage or a one-line crontab revert.

`live` stays on `task session` behind `instances/live.HALT` throughout. Nothing in this plan touches it.
