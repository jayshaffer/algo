# Claude Session Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every LLM stage of the daily session as a headless `claude -p` call under the operator's Claude subscription, with Python reduced to session bookkeeping, an MCP tool server, and the deterministic order path.

**Architecture:** `cron-wrap.sh` invokes a dumb host-side bash driver. The driver asks a new container CLI (`v2/session_ctl.py`) what to run, executes deterministic stages in the container and LLM stages as one `claude -p` per stage on the host, and reports each result back to `session_ctl`, which owns validators, usage, and telemetry. Agentic stages get their tools from a stdio MCP server (`v2/mcp_server.py`) that serves the existing registries unchanged.

**Tech Stack:** Python 3.12 (container), `mcp` Python SDK, bash + `jq` (host), Claude Code CLI 2.1.280 headless mode, PostgreSQL 16, pytest (run in docker: `task test INSTANCE=paper -- <args>`).

**Spec:** `docs/superpowers/specs/2026-09-22-claude-session-inversion-design.md`

## Global Constraints

- Tests run in the container: `task test INSTANCE=paper -- tests/v2/test_x.py -v`. Host python is 3.10 and cannot run the suite.
- No test may reach the network. `tests/v2/conftest.py` patches `v2.session.*` LLM entry points; new modules must patch their own DB and subprocess calls explicitly.
- Every `claude -p` invocation is **non-bare**. Never pass `--bare`; it disables subscription (OAuth) credentials.
- Claude Code 2.1.280 has **no `--max-turns`**. Turn caps are `--max-tool-calls` in the MCP server plus a wall-clock `timeout` in the driver.
- The CLI has `--system-prompt <text>` and `--append-system-prompt-file`, but no `--system-prompt-file`. The driver reads prompt files itself.
- Stage names in `session_stages` are `learning, supervisor, pipeline, strategist, executor, strategy, dashboard`. Stage names in `llm_call_contexts` / `agent_events` are `supervisor, pipeline, ideation, trading, reflection, dashboard_publish`. Keep both namespaces exactly.
- `llm_call_contexts.purpose` values stay `strategist_loop`, `reflection_loop`, `executor`.
- Every model id the driver can emit must be a full id (e.g. `claude-opus-4-8`, not `opus`) and must have a `model_pricing` row (`tests/test_pricing_coverage.py`).
- The money path (`_validate_llm_ids` → `_execute_decisions` → `_handle_thesis_invalidations` → `_log_decisions`) is called, never reimplemented.
- ruff: line length 140, target py312, isort ordering. Pre-commit runs `ruff check .`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `v2/prompts.py` (new) | Load stage prompts from `v2/prompts/*.md` and from skill bodies. Single source of truth for prompt text. |
| `v2/prompts/{classifier,executor,changelog}.md` (new) | Structured-stage system prompts, moved verbatim from Python constants. |
| `.claude/skills/pinchy-{supervisor,strategist,reflection}/SKILL.md` (new) | Agentic-stage skills: preamble + marker + system prompt verbatim. |
| `.claude/skills/pinchy-{session,executor,classify}/SKILL.md` (new) | Interactive wrappers around the driver. |
| `v2/agent.py` (modify) | Split `get_trading_decisions` into serialize / call / `parse_executor_response`; add `executor_output_schema()`, `executor_input_to_dict()`, `executor_input_from_dict()`. |
| `v2/trader.py` (modify) | `prepare_executor_stage()` (steps 1–3 + breaker) and `execute_decisions_payload()` (steps 4–6 from a payload); CLI `--emit-input` / `--decisions-file`. |
| `v2/classifier.py` (modify) | `build_batch_user_message()`, `parse_batch_entries()`, `classification_output_schema()`; `_classify_batch` uses them. |
| `v2/pipeline.py` (modify) | `emit_batches()` / `ingest_batches()`; CLI `--emit-batches` / `--ingest`. |
| `v2/tools.py` (modify) | `tool_get_curated_news` ranks deterministically; `v2/news_filter.py` deleted. |
| `v2/dashboard_publish.py` (modify) | `emit_changelog_commits()`, `changelog_output_schema()`, `run_dashboard_stage(changelog_entries=...)`, new `main()`. |
| `v2/session_ctl.py` (new) | `plan`, `stage-begin`, `stage-context`, `stage-end`, `finalize`, `run-learning`, `prompt`, `schema`. All decisions. |
| `v2/mcp_server.py` (new) | stdio MCP server over the three registries; tool-call cap; `tools.jsonl`; `write_supervisor_memo`. |
| `v2/supervisor.py` (modify) | `write_supervisor_memo` tool def + handler factory; `STRATEGY_SUPERVISOR_SYSTEM` loaded from skill. |
| `session-driver.sh` (new) | Host loop: plan → stages → finalize. One `claude` builder function. |
| `scripts/smoke-claude-headless.sh` (new) | Host guard: non-bare subscription auth still works. |
| `docker-compose.yml` (modify) | Mount `./.claude/skills` read-only into `trading`. |
| `v2/requirements.txt` (modify) | Add `mcp>=1.2`. |
| `Taskfile.yml`, `crontab`, `CLAUDE.md`, `docs/runbook-recovery.md` (modify) | Targets, paper cron line, docs. |

**Prerequisite (manual, before Task 1):** paper sessions have failed daily since 2026-08-25. Run `task session INSTANCE=paper -- --force` by hand and fix whatever fails, so the driver is measured against a working baseline. Not part of this plan's tasks.

---

### Task 1: Prompt loader, prompt files, and skill bodies

**Files:**
- Create: `v2/prompts.py`, `v2/prompts/classifier.md`, `v2/prompts/executor.md`, `v2/prompts/changelog.md`
- Create: `.claude/skills/pinchy-supervisor/SKILL.md`, `.claude/skills/pinchy-strategist/SKILL.md`, `.claude/skills/pinchy-reflection/SKILL.md`
- Modify: `v2/classifier.py:153-186` (`BATCH_CLASSIFICATION_SYSTEM`), `v2/agent.py:179-233` (`TRADING_SYSTEM_PROMPT`), `v2/dashboard_publish.py:1281-1300` (`_changelog_prompt`), `v2/ideation_claude.py:73-146`, `v2/strategy.py:120-185`, `v2/supervisor.py:31-83`
- Modify: `docker-compose.yml:31-35` (trading volumes)
- Test: `tests/v2/test_prompts.py`

**Interfaces:**
- Produces: `v2.prompts.load_prompt(name: str) -> str` (reads `v2/prompts/<name>.md`, stripped), `v2.prompts.load_skill_prompt(stage: str) -> str` (reads `.claude/skills/pinchy-<stage>/SKILL.md`, returns text after the line `<!-- pinchy:system-prompt -->`, stripped), `v2.prompts.SKILL_PROMPT_MARKER = "<!-- pinchy:system-prompt -->"`, `v2.prompts.REPO_ROOT: Path`.
- Later tasks call `load_prompt("executor")`, `load_prompt("classifier")`, `load_prompt("changelog")`, `load_skill_prompt("strategist"|"reflection"|"supervisor")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/v2/test_prompts.py
"""Prompt text lives in files (skills for agentic stages, v2/prompts for
structured stages). These tests pin the loader contract and prompt parity
between the API fallback and the skills the driver runs."""
from pathlib import Path

import pytest

from v2 import prompts


def test_repo_root_contains_skills_and_prompts():
    assert (prompts.REPO_ROOT / ".claude" / "skills").is_dir()
    assert (prompts.REPO_ROOT / "v2" / "prompts").is_dir()


@pytest.mark.parametrize("name", ["classifier", "executor", "changelog"])
def test_load_prompt_returns_nonempty_stripped_text(name):
    text = prompts.load_prompt(name)
    assert text and text == text.strip()


def test_load_prompt_unknown_raises():
    with pytest.raises(FileNotFoundError):
        prompts.load_prompt("does-not-exist")


@pytest.mark.parametrize("stage", ["supervisor", "strategist", "reflection"])
def test_skill_prompt_is_after_marker_and_nonempty(stage):
    skill = prompts.REPO_ROOT / ".claude" / "skills" / f"pinchy-{stage}" / "SKILL.md"
    raw = skill.read_text()
    assert raw.startswith("---\n"), "skill needs YAML frontmatter"
    assert prompts.SKILL_PROMPT_MARKER in raw
    body = prompts.load_skill_prompt(stage)
    assert body == raw.split(prompts.SKILL_PROMPT_MARKER, 1)[1].strip()
    assert len(body) > 500


def test_load_skill_prompt_missing_marker_raises(tmp_path, monkeypatch):
    skills = tmp_path / ".claude" / "skills" / "pinchy-fake"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: x\n---\nno marker here\n")
    monkeypatch.setattr(prompts, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="pinchy:system-prompt"):
        prompts.load_skill_prompt("fake")


def test_api_constants_are_the_skill_bodies():
    from v2 import ideation_claude, strategy, supervisor
    assert ideation_claude.CLAUDE_SESSION_STRATEGIST_SYSTEM == prompts.load_skill_prompt("strategist")
    assert strategy.STRATEGY_REFLECTION_SYSTEM == prompts.load_skill_prompt("reflection")
    assert supervisor.STRATEGY_SUPERVISOR_SYSTEM == prompts.load_skill_prompt("supervisor")


def test_api_constants_are_the_prompt_files():
    from v2 import agent, classifier
    assert agent.TRADING_SYSTEM_PROMPT == prompts.load_prompt("executor")
    assert classifier.BATCH_CLASSIFICATION_SYSTEM == prompts.load_prompt("classifier")


def test_strategist_skill_keeps_required_phrases():
    body = prompts.load_skill_prompt("strategist")
    for phrase in ("write_playbook", "resolve_watchlist_item"):
        assert phrase in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `task test INSTANCE=paper -- tests/v2/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2.prompts'`

- [ ] **Step 3: Write the loader**

```python
# v2/prompts.py
"""Stage prompt text, loaded from files.

Agentic stages (supervisor, strategist, reflection) keep their system prompt
in the skill the Claude Code driver runs (`.claude/skills/pinchy-<stage>/
SKILL.md`, after the marker line). Structured stages keep theirs in
`v2/prompts/<name>.md`. The API fallback in Python loads the same files, so
the two paths cannot drift (tests/v2/test_prompts.py).

REPO_ROOT resolves to the checkout on the host and to /app in the trading
container, where v2/ is mounted at /app/v2 and .claude/skills at
/app/.claude/skills (docker-compose.yml).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "v2" / "prompts"
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
SKILL_PROMPT_MARKER = "<!-- pinchy:system-prompt -->"


def load_prompt(name: str) -> str:
    """System prompt for a structured stage (`v2/prompts/<name>.md`)."""
    return (PROMPTS_DIR / f"{name}.md").read_text().strip()


def load_skill_prompt(stage: str) -> str:
    """System prompt embedded in `.claude/skills/pinchy-<stage>/SKILL.md`.

    Everything after SKILL_PROMPT_MARKER is the prompt; everything before it
    (frontmatter + preamble) is Claude-Code-only instruction.
    """
    path = SKILLS_DIR / f"pinchy-{stage}" / "SKILL.md"
    raw = path.read_text()
    if SKILL_PROMPT_MARKER not in raw:
        raise ValueError(f"{path} has no {SKILL_PROMPT_MARKER} line")
    return raw.split(SKILL_PROMPT_MARKER, 1)[1].strip()
```

- [ ] **Step 4: Move the structured prompts into files**

Create `v2/prompts/classifier.md` with the exact text of `BATCH_CLASSIFICATION_SYSTEM` (`v2/classifier.py:153-184`). Create `v2/prompts/executor.md` with the exact text of `TRADING_SYSTEM_PROMPT` (`v2/agent.py:179-233`). Create `v2/prompts/changelog.md` with the instruction paragraphs of `_changelog_prompt` (`v2/dashboard_publish.py:1293-1305`, everything before the JSON payload is appended), followed by one final line: `Return strict JSON only. Do not wrap the response in markdown.`

Then replace the constants:

```python
# v2/classifier.py — replace the BATCH_CLASSIFICATION_SYSTEM = """...""" block with:
from .prompts import load_prompt

BATCH_CLASSIFICATION_SYSTEM = load_prompt("classifier")
```

```python
# v2/agent.py — replace the TRADING_SYSTEM_PROMPT = """...""" block with:
from .prompts import load_prompt

TRADING_SYSTEM_PROMPT = load_prompt("executor")
```

```python
# v2/dashboard_publish.py — replace _changelog_prompt's literal instruction text:
from .prompts import load_prompt

CHANGELOG_SYSTEM_PROMPT = load_prompt("changelog")


def _changelog_prompt(commits: list[dict]) -> str:
    payload = [
        {
            "sha": c.get("sha"),
            "short_sha": c.get("short_sha"),
            "committed_at": c.get("committed_at"),
            "subject": c.get("subject"),
            "body": c.get("body") or "",
            "files": c.get("files") or [],
        }
        for c in commits
    ]
    return json.dumps(payload, indent=2)
```

and in `summarize_changelog_commits` change `system="Return strict JSON only. Do not wrap the response in markdown."` to `system=CHANGELOG_SYSTEM_PROMPT`. (The instruction text moved into the system prompt; the user message is now the bare payload. Both paths send the same two pieces.)

- [ ] **Step 5: Create the three agentic skills**

Each file: frontmatter, preamble, marker, then the prompt text **verbatim** from the Python constant. Strategist body = the formatted `CLAUDE_SESSION_STRATEGIST_SYSTEM` (the pre-close variant, `v2/ideation_claude.py:141-146`, with `{timing}`-style placeholders already substituted — copy the rendered string, not the template). Reflection body = `STRATEGY_REFLECTION_SYSTEM` (`v2/strategy.py:120-185`). Supervisor body = `STRATEGY_SUPERVISOR_SYSTEM` (`v2/supervisor.py:31-83`).

```markdown
---
name: pinchy-strategist
description: Run the Pinchy strategist stage (thesis management + playbook) for an instance, using the pinchy MCP tools. Usage: /pinchy-strategist <instance>
---

You are running the **strategist** stage of the Pinchy daily session for
instance `$ARGUMENTS`.

Headless runs (session-driver.sh) pipe the session state into this
conversation as the first message and attach the `pinchy` MCP server. If no
state block follows, fetch it yourself:

```
docker compose -p pinchy-$ARGUMENTS --env-file instances/$ARGUMENTS.env \
  exec -T trading python -m v2.session_ctl stage-context strategist
```

and start `claude` with `--mcp-config "$(./session-driver.sh mcp-config $ARGUMENTS strategist)"`.

Rules for this session:
- Use only the `pinchy` MCP tools and WebSearch (at most 6 searches). Do not
  read or edit repository files.
- The stage ends when you have called `write_playbook`. Then reply with a
  summary of at least 40 characters; it is saved as the strategist memo.
- Every open watchlist item in the state block must be resolved with
  `resolve_watchlist_item` before you finish.

<!-- pinchy:system-prompt -->
<verbatim CLAUDE_SESSION_STRATEGIST_SYSTEM text>
```

Reflection preamble differs in two lines: the terminal tool is `write_strategy_memo`, and the context command is `stage-context reflection`. Supervisor preamble: terminal tool is `write_supervisor_memo`; watchlist items are recorded with `record_watchlist_item` before it; no state block is piped (the supervisor pulls everything by tool).

- [ ] **Step 6: Point the Python constants at the skills**

```python
# v2/ideation_claude.py — after the _STRATEGIST_TEMPLATE / CLAUDE_STRATEGIST_SYSTEM block:
from .prompts import load_skill_prompt

# The session (pre-close) variant is the skill body; the post-close variant
# stays a Python template for `run_strategist_session` (unused by session.py).
CLAUDE_SESSION_STRATEGIST_SYSTEM = load_skill_prompt("strategist")
```

```python
# v2/strategy.py — replace STRATEGY_REFLECTION_SYSTEM = """...""" with:
from .prompts import load_skill_prompt

STRATEGY_REFLECTION_SYSTEM = load_skill_prompt("reflection")
```

```python
# v2/supervisor.py — replace STRATEGY_SUPERVISOR_SYSTEM = """...""" with:
from .prompts import load_skill_prompt

STRATEGY_SUPERVISOR_SYSTEM = load_skill_prompt("supervisor")
```

- [ ] **Step 7: Mount the skills into the container**

```yaml
# docker-compose.yml, trading service volumes — add one line:
      - ./.claude/skills:/app/.claude/skills:ro
```

Recreate: `task up INSTANCE=paper`.

- [ ] **Step 8: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_prompts.py tests/v2/test_classifier.py tests/v2/test_agent.py tests/v2/test_strategy.py tests/v2/test_supervisor.py tests/v2/test_ideation_claude.py tests/v2/test_dashboard_publish.py -v`
Expected: all PASS. If a dashboard_publish test asserts the old system string, update it to `CHANGELOG_SYSTEM_PROMPT`.

- [ ] **Step 9: Commit**

```bash
git add v2/prompts.py v2/prompts/ .claude/skills/pinchy-supervisor .claude/skills/pinchy-strategist .claude/skills/pinchy-reflection \
  v2/classifier.py v2/agent.py v2/dashboard_publish.py v2/ideation_claude.py v2/strategy.py v2/supervisor.py docker-compose.yml tests/v2/test_prompts.py
git commit -m "Move stage prompts into skill files and v2/prompts, load them in Python

Skills are the source of truth for the Claude Code driver; the API
fallback loads the same files so the two cannot drift.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 2: Executor seam — parse, schema, serialize, and `trader.py --emit-input` / `--decisions-file`

**Files:**
- Modify: `v2/agent.py:236-440` (`get_trading_decisions`), add helpers
- Modify: `v2/trader.py:1462-1600` (`run_trading_session`, `main`)
- Test: `tests/v2/test_agent.py`, `tests/v2/test_trader.py`

**Interfaces:**
- Produces in `v2/agent.py`:
  - `executor_input_to_dict(executor_input: ExecutorInput) -> dict` (the exact dict `get_trading_decisions` serialises today)
  - `executor_input_from_dict(data: dict) -> ExecutorInput` (inverse; `playbook_actions` rebuilt as `PlaybookAction`, `current_prices` values as `Decimal`)
  - `parse_executor_response(data: dict, *, raw_text: str, stop_reason: str | None, session_id: int | None) -> AgentResponse` (the existing parse/validate/telemetry block, raises `ValueError` on schema failure)
  - `executor_output_schema() -> dict` (JSON Schema built from `VALID_ACTIONS`, `VALID_CONFIDENCES`, `VALID_BUY_INTENTS | VALID_SELL_INTENTS`)
- Produces in `v2/trader.py`:
  - `prepare_executor_stage(*, dry_run: bool, session_id: int | None, session_date: date | None) -> dict` returning `{"skipped": str|None, "executor_input": dict, "account_info": dict, "snapshot_id": int, "positions_synced": int, "orders_synced": int, "session_date": "YYYY-MM-DD", "errors": [str]}`
  - `execute_decisions_payload(decisions: dict, prepared: dict, *, dry_run: bool, session_id: int | None) -> TradingSessionResult`
  - CLI: `python -m v2.trader --emit-input <path> [--dry-run]` (exit 0; writes the prepared dict; exit 3 when `skipped` is set) and `python -m v2.trader --decisions-file <path> --input-file <path> [--dry-run]` (exit 1 when `result.errors`).

- [ ] **Step 1: Write the failing agent tests**

```python
# tests/v2/test_agent.py — append
import json
from decimal import Decimal

import pytest

from v2 import agent
from v2.agent import ExecutorInput, PlaybookAction


def _executor_input():
    return ExecutorInput(
        playbook_actions=[PlaybookAction(**{
            k: v for k, v in dict(
                id=7, ticker="AAPL", action="buy", intent_type="invest_dollar",
                intent_magnitude=Decimal("500"), thesis_id=3, reasoning="r",
                priority=1, status="pending",
            ).items() if k in PlaybookAction.__dataclass_fields__
        })],
        positions=[{"ticker": "AAPL", "shares": 10}],
        account={"portfolio_value": 10000.0, "buying_power": 5000.0},
        attribution_summary={"news_signal:earnings": 0.6},
        recent_outcomes=[],
        market_outlook="calm",
        risk_notes="",
        current_prices={"AAPL": Decimal("190.25")},
    )


def test_executor_input_round_trips_through_dict():
    ei = _executor_input()
    data = agent.executor_input_to_dict(ei)
    json.dumps(data, default=str)  # must be JSON-serialisable with the same default the API path uses
    back = agent.executor_input_from_dict(json.loads(json.dumps(data, default=str)))
    assert back.playbook_actions[0].ticker == "AAPL"
    assert back.current_prices["AAPL"] == Decimal("190.25")
    assert back.account == ei.account


def test_executor_output_schema_pins_enums():
    schema = agent.executor_output_schema()
    dec = schema["properties"]["decisions"]["items"]["properties"]
    assert set(dec["action"]["enum"]) == agent.VALID_ACTIONS
    assert set(dec["confidence"]["enum"]) == agent.VALID_CONFIDENCES
    assert set(dec["intent_type"]["enum"]) == agent.VALID_BUY_INTENTS | agent.VALID_SELL_INTENTS
    assert set(schema["required"]) == {"decisions", "thesis_invalidations", "market_summary", "risk_assessment"}
    assert set(dec.keys()) == agent.EXECUTOR_KNOWN_DECISION_KEYS - {"quantity"}


def test_parse_executor_response_builds_agent_response():
    data = {
        "decisions": [{
            "playbook_action_id": 7, "ticker": " aapl ", "action": "buy",
            "intent_type": "invest_dollar", "intent_magnitude": 500,
            "reasoning": "r", "confidence": "high", "is_off_playbook": False,
            "signal_refs": [], "thesis_id": "3",
        }],
        "thesis_invalidations": [{"thesis_id": 9, "reason": "x"}, {"thesis_id": "bad", "reason": "y"}],
        "market_summary": "m", "risk_assessment": "r",
    }
    with patch("v2.agent.record_event") as ev:
        resp = agent.parse_executor_response(data, raw_text=json.dumps(data), stop_reason="end_turn", session_id=5)
    assert resp.decisions[0].ticker == "AAPL"
    assert resp.decisions[0].intent_magnitude == Decimal("500")
    assert resp.decisions[0].thesis_id == 3
    assert [t.thesis_id for t in resp.thesis_invalidations] == [9]
    payload = ev.call_args.kwargs["payload"]
    assert payload["parse_succeeded"] is True and payload["decision_count"] == 1


def test_parse_executor_response_schema_failure_raises_and_records():
    data = {"decisions": [{"ticker": "AAPL", "action": "short", "confidence": "high"}]}
    with patch("v2.agent.record_event") as ev, pytest.raises(ValueError, match="schema validation"):
        agent.parse_executor_response(data, raw_text="{}", stop_reason="end_turn", session_id=5)
    assert ev.call_args.kwargs["payload"]["parse_succeeded"] is False
```

(`patch` is already imported at the top of `tests/v2/test_agent.py`; if the `PlaybookAction` field set differs from the dict above, the comprehension keeps only real fields — check `v2/agent.py:83` and drop the comprehension if all names match.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `task test INSTANCE=paper -- tests/v2/test_agent.py -v -k "round_trips or schema_pins or parse_executor"`
Expected: FAIL with `AttributeError: module 'v2.agent' has no attribute 'executor_input_to_dict'`

- [ ] **Step 3: Refactor `v2/agent.py`**

Add after the `ExecutorInput` dataclass:

```python
def executor_input_to_dict(executor_input: ExecutorInput) -> dict:
    """The exact payload the executor prompt receives (json.dumps with default=str)."""
    return {
        "playbook_actions": [asdict(a) for a in executor_input.playbook_actions],
        "positions": executor_input.positions,
        "account": executor_input.account,
        "attribution_summary": executor_input.attribution_summary,
        "recent_outcomes": executor_input.recent_outcomes,
        "market_outlook": executor_input.market_outlook,
        "risk_notes": executor_input.risk_notes,
        "current_prices": {k: str(v) for k, v in executor_input.current_prices.items()},
        "strategy_identity": executor_input.strategy_identity,
        "strategy_rules": executor_input.strategy_rules,
        "equity_summary": executor_input.equity_summary,
        "todays_decisions": executor_input.todays_decisions,
        "recent_ticker_decisions": executor_input.recent_ticker_decisions,
        "playbook_action_history": executor_input.playbook_action_history,
    }


def executor_input_from_dict(data: dict) -> ExecutorInput:
    """Inverse of executor_input_to_dict, for the --decisions-file path."""
    fields = {f for f in ExecutorInput.__dataclass_fields__}
    kwargs = {k: v for k, v in data.items() if k in fields}
    action_fields = set(PlaybookAction.__dataclass_fields__)
    kwargs["playbook_actions"] = [
        PlaybookAction(**{k: v for k, v in a.items() if k in action_fields})
        for a in data.get("playbook_actions", [])
    ]
    kwargs["current_prices"] = {k: Decimal(str(v)) for k, v in (data.get("current_prices") or {}).items()}
    return ExecutorInput(**kwargs)


def executor_output_schema() -> dict:
    """JSON Schema for the executor's structured output (claude -p --json-schema).

    Built from the same constants the parser validates against, so a new
    intent or confidence value changes both in one place.
    """
    signal_ref = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["news_signal", "macro_signal", "thesis"]},
            "id": {"type": "integer"},
        },
        "required": ["type", "id"],
    }
    decision = {
        "type": "object",
        "properties": {
            "playbook_action_id": {"type": ["integer", "null"]},
            "ticker": {"type": "string"},
            "action": {"type": "string", "enum": sorted(VALID_ACTIONS)},
            "intent_type": {"type": ["string", "null"], "enum": sorted(VALID_BUY_INTENTS | VALID_SELL_INTENTS) + [None]},
            "intent_magnitude": {"type": ["number", "null"]},
            "reasoning": {"type": "string"},
            "confidence": {"type": "string", "enum": sorted(VALID_CONFIDENCES)},
            "is_off_playbook": {"type": "boolean"},
            "signal_refs": {"type": "array", "items": signal_ref},
            "thesis_id": {"type": ["integer", "null"]},
        },
        "required": ["ticker", "action", "reasoning", "confidence", "is_off_playbook", "signal_refs"],
    }
    return {
        "type": "object",
        "properties": {
            "decisions": {"type": "array", "items": decision},
            "thesis_invalidations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"thesis_id": {"type": "integer"}, "reason": {"type": "string"}},
                    "required": ["thesis_id", "reason"],
                },
            },
            "market_summary": {"type": "string"},
            "risk_assessment": {"type": "string"},
        },
        "required": ["decisions", "thesis_invalidations", "market_summary", "risk_assessment"],
    }
```

Then split `get_trading_decisions`: keep everything up to and including the `max_tokens` truncation check, replace the JSON-parse-and-build block (`v2/agent.py:322-440`) with:

```python
    try:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        record_event(
            session_id=session_id, stage_name="trading", event_type="executor_response",
            payload={
                "parse_succeeded": False, "stop_reason": response.stop_reason,
                "decision_count": 0, "thesis_invalidation_count": 0,
                "unknown_top_level_keys": [], "unknown_decision_keys": [],
                "raw_response_text_truncated": response_text[:EXECUTOR_RAW_TEXT_CAP],
                "error": f"JSONDecodeError: {e}",
            },
        )
        raise ValueError(f"Failed to parse LLM response as JSON: {response_text}") from e

    return parse_executor_response(
        data, raw_text=response_text, stop_reason=response.stop_reason, session_id=session_id,
    )


def parse_executor_response(
    data: dict, *, raw_text: str, stop_reason: str | None, session_id: int | None,
) -> AgentResponse:
    """Validate a decoded executor payload and build the AgentResponse.

    Shared by the API path (get_trading_decisions) and the Claude Code driver
    path (trader.py --decisions-file). Emits the executor_response event on
    both success and schema failure; raises ValueError on failure.
    """
    # <move the existing body from `decisions = []` through `return AgentResponse(...)`
    #  here verbatim, replacing `response.stop_reason` with `stop_reason` and
    #  `response_text` with `raw_text`>
```

Also replace the `input_data = {...}` literal in `get_trading_decisions` with `input_data = executor_input_to_dict(executor_input)`.

- [ ] **Step 4: Run the agent tests**

Run: `task test INSTANCE=paper -- tests/v2/test_agent.py -v`
Expected: PASS (all, including the pre-existing `get_trading_decisions` tests).

- [ ] **Step 5: Write the failing trader tests**

```python
# tests/v2/test_trader.py — append
import json
from datetime import date
from unittest.mock import MagicMock, patch

from v2 import trader
from v2.agent import AgentResponse


def _prepared():
    return {
        "skipped": None,
        "executor_input": {
            "playbook_actions": [], "positions": [], "account": {"portfolio_value": 1000.0},
            "attribution_summary": {}, "recent_outcomes": [], "market_outlook": "", "risk_notes": "",
            "current_prices": {}, "strategy_identity": "", "strategy_rules": [], "equity_summary": {},
            "todays_decisions": [], "recent_ticker_decisions": [], "playbook_action_history": [],
        },
        "account_info": {"portfolio_value": 1000.0, "equity": 1000.0, "last_equity": 1000.0, "buying_power": 500.0},
        "snapshot_id": 1, "positions_synced": 0, "orders_synced": 0,
        "session_date": "2026-09-22", "errors": [],
    }


def test_prepare_executor_stage_market_closed_is_skip():
    with patch("v2.trader._sync_from_alpaca", return_value=(0, 0)), \
         patch("v2.trader.is_market_open", return_value=False):
        out = trader.prepare_executor_stage(dry_run=False, session_id=1, session_date=date(2026, 9, 22))
    assert out["skipped"] == "market_closed"


def test_prepare_executor_stage_breaker_reports_error_and_skips():
    with patch("v2.trader._sync_from_alpaca", return_value=(0, 0)), \
         patch("v2.trader.is_market_open", return_value=True), \
         patch("v2.trader._build_stock_data_client"), \
         patch("v2.trader._snapshot_account", return_value=({"equity": 900.0, "last_equity": 1000.0}, 1)), \
         patch("v2.trader.check_daily_loss_limit", return_value="down 10%"), \
         patch("v2.trader.record_event"):
        out = trader.prepare_executor_stage(dry_run=False, session_id=1, session_date=date(2026, 9, 22))
    assert out["skipped"] == "daily_loss_limit" and out["errors"] == ["down 10%"]


def test_prepare_executor_stage_emits_input():
    ctx = MagicMock()
    with patch("v2.trader._sync_from_alpaca", return_value=(2, 1)), \
         patch("v2.trader.is_market_open", return_value=True), \
         patch("v2.trader._build_stock_data_client"), \
         patch("v2.trader._snapshot_account", return_value=({"equity": 1000.0, "last_equity": 1000.0}, 4)), \
         patch("v2.trader.check_daily_loss_limit", return_value=None), \
         patch("v2.trader._build_executor_context", return_value=ctx), \
         patch("v2.trader.executor_input_to_dict", return_value={"positions": []}):
        out = trader.prepare_executor_stage(dry_run=False, session_id=1, session_date=date(2026, 9, 22))
    assert out["skipped"] is None and out["snapshot_id"] == 4 and out["positions_synced"] == 2
    assert out["executor_input"] == {"positions": []}


def test_execute_decisions_payload_runs_money_path_in_order():
    calls = []
    resp = AgentResponse(decisions=[], thesis_invalidations=[], market_summary="m", risk_assessment="r")
    with patch("v2.trader.parse_executor_response", return_value=resp), \
         patch("v2.trader.check_daily_loss_limit", return_value=None), \
         patch("v2.trader.get_account_info", return_value={"equity": 1.0, "last_equity": 1.0}), \
         patch("v2.trader._validate_llm_ids", side_effect=lambda *a, **k: calls.append("validate")), \
         patch("v2.trader.get_positions", return_value=[]), \
         patch("v2.trader._build_stock_data_client"), \
         patch("v2.trader._execute_decisions", side_effect=lambda *a, **k: (calls.append("execute"), (MagicMock(), [], [], {}))[1]), \
         patch("v2.trader._handle_thesis_invalidations", side_effect=lambda *a, **k: calls.append("invalidate")), \
         patch("v2.trader._log_decisions", side_effect=lambda *a, **k: (calls.append("log"), 0)[1]), \
         patch("v2.trader._log_session_summary"), \
         patch("v2.trader._build_final_result", return_value="final"):
        out = trader.execute_decisions_payload({"decisions": []}, _prepared(), dry_run=True, session_id=1)
    assert out == "final"
    assert calls == ["validate", "execute", "invalidate", "log"]


def test_execute_decisions_payload_rechecks_breaker_before_orders():
    with patch("v2.trader.parse_executor_response") as parse, \
         patch("v2.trader.get_account_info", return_value={"equity": 900.0, "last_equity": 1000.0}), \
         patch("v2.trader.check_daily_loss_limit", return_value="down 10%"), \
         patch("v2.trader.record_event"), \
         patch("v2.trader._execute_decisions") as execute:
        out = trader.execute_decisions_payload({"decisions": []}, _prepared(), dry_run=False, session_id=1)
    assert "down 10%" in out.errors
    parse.assert_not_called()
    execute.assert_not_called()


def test_execute_decisions_payload_schema_failure_is_error_not_exception():
    with patch("v2.trader.get_account_info", return_value={"equity": 1.0, "last_equity": 1.0}), \
         patch("v2.trader.check_daily_loss_limit", return_value=None), \
         patch("v2.trader.parse_executor_response", side_effect=ValueError("bad")), \
         patch("v2.trader._execute_decisions") as execute:
        out = trader.execute_decisions_payload({"decisions": "nope"}, _prepared(), dry_run=True, session_id=1)
    assert any("LLM decision failed" in e for e in out.errors)
    execute.assert_not_called()


def test_cli_emit_input_writes_file_and_exit_codes(tmp_path):
    out = tmp_path / "input.json"
    with patch("v2.trader.prepare_executor_stage", return_value={"skipped": None, "x": 1}), \
         patch("sys.argv", ["trader", "--emit-input", str(out)]):
        assert trader.main() == 0
    assert json.loads(out.read_text()) == {"skipped": None, "x": 1}
    with patch("v2.trader.prepare_executor_stage", return_value={"skipped": "market_closed"}), \
         patch("sys.argv", ["trader", "--emit-input", str(out)]):
        assert trader.main() == 3


def test_cli_decisions_file_exit_one_on_errors(tmp_path):
    d = tmp_path / "d.json"; d.write_text('{"decisions": []}')
    i = tmp_path / "i.json"; i.write_text(json.dumps(_prepared()))
    result = MagicMock(errors=["LLM decision failed: x"])
    with patch("v2.trader.execute_decisions_payload", return_value=result), \
         patch("sys.argv", ["trader", "--decisions-file", str(d), "--input-file", str(i), "--dry-run"]):
        assert trader.main() == 1
```

- [ ] **Step 6: Run the trader tests to verify they fail**

Run: `task test INSTANCE=paper -- tests/v2/test_trader.py -v -k "prepare_executor or decisions_payload or cli_"`
Expected: FAIL with `AttributeError: module 'v2.trader' has no attribute 'prepare_executor_stage'`

- [ ] **Step 7: Implement the trader seam**

In `v2/trader.py`, import `executor_input_from_dict, executor_input_to_dict, parse_executor_response` from `.agent` (extend the existing `from .agent import ...` line) and add `import json` if absent. Then add before `run_trading_session`:

```python
def prepare_executor_stage(
    *, dry_run: bool, session_id: int | None, session_date: date | None,
) -> dict:
    """Steps 1–3 of run_trading_session, for the Claude Code driver.

    Syncs, gates on market hours, snapshots, runs the daily-loss breaker, and
    builds the executor input. Returns a JSON-serialisable dict; `skipped` is
    "market_closed" / "daily_loss_limit" / "no_account" when the stage should
    not call the model at all.
    """
    errors: list[str] = []
    if session_date is None:
        session_date = datetime.now(ZoneInfo("America/New_York")).date()
    out = {
        "skipped": None, "executor_input": None, "account_info": None, "snapshot_id": 0,
        "positions_synced": 0, "orders_synced": 0, "session_date": session_date.isoformat(),
        "errors": errors,
    }
    out["positions_synced"], out["orders_synced"] = _sync_from_alpaca(errors)
    if not dry_run and not is_market_open():
        logger.warning("Market is closed. Skipping trading session (use --dry-run to bypass)")
        out["skipped"] = "market_closed"
        return out
    data_client = _build_stock_data_client()
    account_info, snapshot_id = _snapshot_account(errors)
    out["snapshot_id"] = snapshot_id
    if account_info is None:
        out["skipped"] = "no_account"
        return out
    out["account_info"] = account_info
    loss_breach = check_daily_loss_limit(account_info.get("equity"), account_info.get("last_equity"))
    if loss_breach:
        errors.append(loss_breach)
        logger.error("HALT: %s", loss_breach)
        record_event(session_id=session_id, stage_name="trading", event_type="risk_block",
                     payload={"reason_code": "daily_loss_limit", "reason_text": loss_breach})
        out["skipped"] = "daily_loss_limit"
        return out
    executor_input = _build_executor_context(account_info, data_client, errors)
    out["executor_input"] = executor_input_to_dict(executor_input)
    return out


def execute_decisions_payload(
    decisions: dict, prepared: dict, *, dry_run: bool, session_id: int | None,
) -> TradingSessionResult:
    """Steps 4–6 of run_trading_session from a decoded executor payload.

    The breaker is re-checked against a fresh account read before any order,
    because the model call happened between prepare and execute.
    """
    errors: list[str] = list(prepared.get("errors") or [])
    timestamp = datetime.now()
    session_date = date.fromisoformat(prepared["session_date"])
    account_info = get_account_info()
    loss_breach = check_daily_loss_limit(account_info.get("equity"), account_info.get("last_equity"))
    if loss_breach:
        errors.append(loss_breach)
        logger.error("HALT: %s", loss_breach)
        record_event(session_id=session_id, stage_name="trading", event_type="risk_block",
                     payload={"reason_code": "daily_loss_limit", "reason_text": loss_breach})
        return _empty_result(timestamp, prepared["positions_synced"], prepared["orders_synced"],
                             prepared["snapshot_id"], errors)
    try:
        response = parse_executor_response(
            decisions, raw_text=json.dumps(decisions), stop_reason="end_turn", session_id=session_id,
        )
    except ValueError as e:
        errors.append(f"LLM decision failed: {e}")
        return _empty_result(timestamp, prepared["positions_synced"], prepared["orders_synced"],
                             prepared["snapshot_id"], errors)
    executor_input = executor_input_from_dict(prepared["executor_input"])
    _validate_llm_ids(response, executor_input, session_id=session_id)
    data_client = _build_stock_data_client()
    positions = {p["ticker"]: p["shares"] for p in get_positions()}
    totals, order_ids, order_results, decision_account_states = _execute_decisions(
        response, positions, account_info, data_client, dry_run, errors, session_date, session_id=session_id,
    )
    _handle_thesis_invalidations(response.thesis_invalidations, errors)
    logged_count = _log_decisions(
        response, order_ids, order_results, data_client, account_info, errors, session_date,
        decision_account_states=decision_account_states, session_id=session_id,
    )
    logger.info("Logged %d decisions (%d emitted by executor)", logged_count, len(response.decisions))
    _log_session_summary(response, totals, errors)
    return _build_final_result(
        timestamp, prepared["snapshot_id"], prepared["positions_synced"], prepared["orders_synced"],
        response, totals, errors,
    )
```

Check that `get_account_info` is what `_snapshot_account` uses to read the account (grep `v2/trader.py` for `get_account_info`); if the helper has a different name, use that name in both the code and the tests above.

Replace `main()`:

```python
def main() -> int:
    """CLI entry point for trading agent."""
    import argparse

    from .log_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Run trading agent session")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute real trades")
    parser.add_argument("--model", default=DEFAULT_EXECUTOR_MODEL, help="Claude model to use")
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--emit-input", metavar="PATH",
                        help="Driver mode: run steps 1-3 and write the executor input JSON to PATH")
    parser.add_argument("--decisions-file", metavar="PATH",
                        help="Driver mode: execute a structured decisions payload (needs --input-file)")
    parser.add_argument("--input-file", metavar="PATH", help="The file written by --emit-input")
    args = parser.parse_args()

    if args.emit_input:
        prepared = prepare_executor_stage(dry_run=args.dry_run, session_id=args.session_id, session_date=None)
        with open(args.emit_input, "w") as fh:
            json.dump(prepared, fh, default=str)
        if prepared.get("skipped"):
            logger.warning("Executor stage skipped: %s", prepared["skipped"])
            return 3
        return 0

    if args.decisions_file:
        if not args.input_file:
            parser.error("--decisions-file requires --input-file")
        with open(args.decisions_file) as fh:
            decisions = json.load(fh)
        with open(args.input_file) as fh:
            prepared = json.load(fh)
        result = execute_decisions_payload(decisions, prepared, dry_run=args.dry_run, session_id=args.session_id)
    else:
        result = run_trading_session(dry_run=args.dry_run, model=args.model, session_id=args.session_id)

    if result.errors:
        logger.error("Errors encountered:")
        for error in result.errors:
            logger.error("  - %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

(Keep whatever summary logging the current `main()` does after the error loop; the only behavioural change is the explicit return code, which matches the current `sys.exit(1)`.)

- [ ] **Step 8: Run the trader tests**

Run: `task test INSTANCE=paper -- tests/v2/test_trader.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add v2/agent.py v2/trader.py tests/v2/test_agent.py tests/v2/test_trader.py
git commit -m "Split the executor into prepare / parse / execute seams for the driver

trader.py --emit-input runs sync, gate, snapshot, breaker and context;
--decisions-file runs the unchanged money path on a structured payload.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 3: Classifier seam — batch message, entry parser, schema, and `pipeline.py --emit-batches` / `--ingest`

**Files:**
- Modify: `v2/classifier.py:368-432` (`_classify_batch`), add three functions
- Modify: `v2/pipeline.py` (add `emit_batches`, `ingest_batches`, CLI flags)
- Test: `tests/v2/test_classifier.py`, `tests/v2/test_pipeline.py`

**Interfaces:**
- Produces in `v2/classifier.py`:
  - `build_batch_user_message(headlines: list[str]) -> str` (the numbered, sanitized block)
  - `parse_batch_entries(entries: list, headlines: list[str], published_ats: list[datetime], alpaca_ids: list[str | None], summaries: list[str]) -> list[ClassificationResult]` (index remap + `_build_classification_result`; a missing index is `noise`)
  - `classification_output_schema() -> dict` (object with one required key `classifications`: array of entry objects)
- Produces in `v2/pipeline.py`:
  - `emit_batches(out_dir: str, *, hours: int, limit: int, batch_size: int = 50) -> int` (writes `batch-NN.json` files, returns count)
  - `ingest_batches(in_dir: str, *, session_id: int | None, dry_run: bool) -> PipelineStats` (reads `batch-NN.json` + `batch-NN.result.json`)
  - Batch file shape: `{"items": [{"index": 1, "headline": str, "published_at": iso, "alpaca_id": str|null, "summary": str}], "user_message": str}`. Result file shape: the `claude -p --output-format json` result; `structured_output.classifications` is the entry list. A missing or `is_error` result file marks the whole batch noise.

- [ ] **Step 1: Write the failing classifier tests**

```python
# tests/v2/test_classifier.py — append
from datetime import datetime

from v2 import classifier


def test_build_batch_user_message_numbers_and_sanitizes():
    msg = classifier.build_batch_user_message(["Apple\x00 beats", "Fed holds"])
    assert msg.splitlines() == ['1. "Apple beats"', '2. "Fed holds"']


def test_parse_batch_entries_index_remap_and_noise_default():
    now = datetime(2026, 9, 22, 12, 0)
    entries = [
        {"index": 2, "type": "ticker_specific", "tickers": ["AAPL"], "category": "earnings",
         "sentiment": "bullish", "confidence": "high"},
        {"index": 99, "type": "ticker_specific", "tickers": ["X"], "category": "earnings"},
        "garbage",
    ]
    with patch("v2.classifier._validate_ticker", side_effect=lambda t: t):
        out = classifier.parse_batch_entries(entries, ["h1", "h2"], [now, now], [None, "a2"], ["", "s2"])
    assert out[0].news_type == "noise"
    assert out[1].ticker_signals[0].ticker == "AAPL"
    assert out[1].ticker_signals[0].alpaca_id == "a2" and out[1].ticker_signals[0].summary == "s2"


def test_classification_output_schema_shape():
    schema = classifier.classification_output_schema()
    assert schema["required"] == ["classifications"]
    item = schema["properties"]["classifications"]["items"]
    assert item["required"] == ["index", "type"]
    assert set(item["properties"]["type"]["enum"]) == {"ticker_specific", "macro_political", "sector", "noise"}
```

(`patch` is already imported in `tests/v2/test_classifier.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_classifier.py -v -k "batch_user_message or parse_batch_entries or output_schema"`
Expected: FAIL with `AttributeError: module 'v2.classifier' has no attribute 'build_batch_user_message'`

- [ ] **Step 3: Refactor `v2/classifier.py`**

Add above `_classify_batch`:

```python
def build_batch_user_message(headlines: list[str]) -> str:
    """The numbered headline block the batch classifier prompt expects."""
    return "\n".join(f'{i + 1}. "{_sanitize_headline(h)}"' for i, h in enumerate(headlines))


def parse_batch_entries(
    entries: list,
    headlines: list[str],
    published_ats: list[datetime],
    alpaca_ids: list[str | None],
    summaries: list[str],
) -> list[ClassificationResult]:
    """Map classifier entries back onto headlines by 1-based `index`.

    P2.15: index-based, not positional — the model may reorder or truncate.
    A headline with no entry is noise. Shared by the API path and the
    Claude Code driver path (pipeline.py --ingest).
    """
    by_idx: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if isinstance(idx, int) and 1 <= idx <= len(headlines):
            by_idx[idx - 1] = entry
    results = []
    for i in range(len(headlines)):
        entry = by_idx.get(i)
        if entry is None:
            results.append(ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None))
        else:
            results.append(_build_classification_result(entry, headlines[i], published_ats[i], alpaca_ids[i], summaries[i]))
    return results


def classification_output_schema() -> dict:
    """JSON Schema for one batch's structured output (claude -p --json-schema)."""
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "type": {"type": "string", "enum": ["ticker_specific", "macro_political", "sector", "noise"]},
                        "tickers": {"type": "array", "items": {"type": "string"}},
                        "category": {"type": "string"},
                        "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "affected_sectors": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["index", "type"],
                },
            },
        },
        "required": ["classifications"],
    }
```

Then in `_classify_batch` replace the `headlines_block = ...` expression with `headlines_block = build_batch_user_message(headlines)`, and replace everything from `if alpaca_ids is None:` (after `parsed` is checked to be a list) to the end of the function with:

```python
    if alpaca_ids is None:
        alpaca_ids = [None] * len(headlines)
    if summaries is None:
        summaries = [""] * len(headlines)
    return parse_batch_entries(parsed, headlines, published_ats, alpaca_ids, summaries)
```

- [ ] **Step 4: Run the classifier tests**

Run: `task test INSTANCE=paper -- tests/v2/test_classifier.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing pipeline tests**

```python
# tests/v2/test_pipeline.py — create if absent, else append
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from v2 import pipeline
from v2.news import NewsItem


def _item(i):
    return NewsItem(id=f"id{i}", headline=f"Headline {i}", summary=f"s{i}", author="", source="",
                    symbols=[], published_at=datetime(2026, 9, 22, 12, i), url="")


def test_emit_batches_writes_files_of_batch_size(tmp_path):
    with patch("v2.pipeline.fetch_broad_news", return_value=[_item(i) for i in range(3)]):
        n = pipeline.emit_batches(str(tmp_path), hours=24, limit=300, batch_size=2)
    assert n == 2
    b0 = json.loads((tmp_path / "batch-00.json").read_text())
    assert [it["index"] for it in b0["items"]] == [1, 2]
    assert b0["items"][0]["alpaca_id"] == "id0" and b0["items"][0]["summary"] == "s0"
    assert b0["user_message"].startswith('1. "Headline 0"')
    b1 = json.loads((tmp_path / "batch-01.json").read_text())
    assert len(b1["items"]) == 1


def test_emit_batches_no_news_writes_nothing(tmp_path):
    with patch("v2.pipeline.fetch_broad_news", return_value=[]):
        assert pipeline.emit_batches(str(tmp_path), hours=24, limit=300) == 0
    assert list(tmp_path.iterdir()) == []


def _write_batch(tmp_path, name, items, result=None):
    (tmp_path / f"{name}.json").write_text(json.dumps({"items": items, "user_message": ""}))
    if result is not None:
        (tmp_path / f"{name}.result.json").write_text(json.dumps(result))


def test_ingest_batches_stores_signals_and_marks_missing_results_noise(tmp_path):
    items = [{"index": 1, "headline": "h", "published_at": "2026-09-22T12:00:00", "alpaca_id": "a", "summary": "s"}]
    good = {"is_error": False, "structured_output": {"classifications": [
        {"index": 1, "type": "ticker_specific", "tickers": ["AAPL"], "category": "earnings",
         "sentiment": "bullish", "confidence": "high"}]}}
    _write_batch(tmp_path, "batch-00", items, good)
    _write_batch(tmp_path, "batch-01", items)  # no result file
    with patch("v2.classifier._validate_ticker", side_effect=lambda t: t), \
         patch("v2.pipeline.insert_news_signals_batch", return_value=1) as ins, \
         patch("v2.pipeline.insert_macro_signals_batch", return_value=0):
        stats = pipeline.ingest_batches(str(tmp_path), session_id=1, dry_run=False)
    assert stats.news_fetched == 2 and stats.ticker_signals_stored == 1 and stats.noise_dropped == 1
    assert ins.call_args.args[0][0][0] == "AAPL"


def test_ingest_batches_error_result_is_noise(tmp_path):
    items = [{"index": 1, "headline": "h", "published_at": "2026-09-22T12:00:00", "alpaca_id": None, "summary": ""}]
    _write_batch(tmp_path, "batch-00", items, {"is_error": True, "result": "rate limited"})
    with patch("v2.pipeline.insert_news_signals_batch", return_value=0), \
         patch("v2.pipeline.insert_macro_signals_batch", return_value=0):
        stats = pipeline.ingest_batches(str(tmp_path), session_id=1, dry_run=False)
    assert stats.noise_dropped == 1 and stats.errors == 0


def test_cli_flags(tmp_path):
    with patch("v2.pipeline.emit_batches", return_value=2) as emit, \
         patch("sys.argv", ["pipeline", "--emit-batches", str(tmp_path), "--hours", "6", "--limit", "10"]):
        assert pipeline.main() == 0
    emit.assert_called_once_with(str(tmp_path), hours=6, limit=10)
    stats = pipeline.PipelineStats(1, 0, 0, 0, errors=1)
    with patch("v2.pipeline.ingest_batches", return_value=stats), \
         patch("sys.argv", ["pipeline", "--ingest", str(tmp_path), "--session-id", "4"]):
        assert pipeline.main() == 1
```

- [ ] **Step 6: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_pipeline.py -v`
Expected: FAIL with `AttributeError: module 'v2.pipeline' has no attribute 'emit_batches'`

- [ ] **Step 7: Implement `emit_batches` / `ingest_batches` and the CLI**

```python
# v2/pipeline.py — add imports
import json
import os
from datetime import datetime

from .classifier import build_batch_user_message, classify_news_batch, parse_batch_entries
```

```python
def emit_batches(out_dir: str, *, hours: int, limit: int, batch_size: int = 50) -> int:
    """Driver mode step 1: fetch news and write one JSON file per classifier batch."""
    os.makedirs(out_dir, exist_ok=True)
    news_items = fetch_broad_news(hours=hours, limit=limit)
    count = 0
    for start in range(0, len(news_items), batch_size):
        chunk = news_items[start:start + batch_size]
        payload = {
            "items": [
                {
                    "index": i + 1,
                    "headline": item.headline,
                    "published_at": item.published_at.isoformat(),
                    "alpaca_id": item.id,
                    "summary": item.summary or "",
                }
                for i, item in enumerate(chunk)
            ],
            "user_message": build_batch_user_message([item.headline for item in chunk]),
        }
        with open(os.path.join(out_dir, f"batch-{count:02d}.json"), "w") as fh:
            json.dump(payload, fh)
        count += 1
    logger.info("Emitted %d classifier batch(es) for %d headlines", count, len(news_items))
    return count


def ingest_batches(in_dir: str, *, session_id: int | None, dry_run: bool) -> PipelineStats:
    """Driver mode step 3: read each batch + its claude result and store signals.

    A batch with no result file, an `is_error` result, or no
    `structured_output.classifications` list is marked noise — the same
    outcome as a rate-limited batch on the API path.
    """
    stats = PipelineStats(news_fetched=0, ticker_signals_stored=0, macro_signals_stored=0, noise_dropped=0, errors=0)
    results = []
    names = sorted(n for n in os.listdir(in_dir) if n.startswith("batch-") and n.endswith(".json") and ".result." not in n)
    for name in names:
        with open(os.path.join(in_dir, name)) as fh:
            batch = json.load(fh)
        items = batch["items"]
        stats.news_fetched += len(items)
        headlines = [it["headline"] for it in items]
        published_ats = [datetime.fromisoformat(it["published_at"]) for it in items]
        alpaca_ids = [it.get("alpaca_id") for it in items]
        summaries = [it.get("summary") or "" for it in items]
        entries = None
        result_path = os.path.join(in_dir, name[:-5] + ".result.json")
        if os.path.exists(result_path):
            with open(result_path) as fh:
                result = json.load(fh)
            if not result.get("is_error"):
                entries = (result.get("structured_output") or {}).get("classifications")
        if not isinstance(entries, list):
            logger.error("Batch %s has no usable classifier result — marking noise", name)
            entries = []
        results.extend(parse_batch_entries(entries, headlines, published_ats, alpaca_ids, summaries))
    return _store_results(results, stats, dry_run)
```

Refactor `run_pipeline`'s "Step 3: Store signals" block (from `ticker_signals_batch = []` to `return stats`) into `_store_results(results, stats, dry_run) -> PipelineStats` and have `run_pipeline` call it, so both paths share the insert logic. Then replace `main()`:

```python
def main() -> int:
    """CLI entry point for news pipeline."""
    import argparse

    from .log_config import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Run news processing pipeline")
    parser.add_argument("--hours", type=int, default=24, help="Hours of news to fetch")
    parser.add_argument("--limit", type=int, default=50, help="Max news items")
    parser.add_argument("--dry-run", action="store_true", help="Don't store to database")
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--emit-batches", metavar="DIR", help="Driver mode: fetch news and write classifier batches to DIR")
    parser.add_argument("--ingest", metavar="DIR", help="Driver mode: store signals from DIR's batch + result files")
    args = parser.parse_args()

    if args.emit_batches:
        emit_batches(args.emit_batches, hours=args.hours, limit=args.limit)
        return 0
    if args.ingest:
        stats = ingest_batches(args.ingest, session_id=args.session_id, dry_run=args.dry_run)
    else:
        stats = run_pipeline(hours=args.hours, limit=args.limit, dry_run=args.dry_run, session_id=args.session_id)
    logger.info("Pipeline stats: %s", stats)
    return 1 if stats.errors > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 8: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_pipeline.py tests/v2/test_classifier.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add v2/classifier.py v2/pipeline.py tests/v2/test_classifier.py tests/v2/test_pipeline.py
git commit -m "Expose classifier batches as files for the driver (--emit-batches / --ingest)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Deterministic news curation (remove the in-tool Haiku call)

**Files:**
- Modify: `v2/tools.py:338-395` (`tool_get_curated_news`), remove the `curate_signals` import at `v2/tools.py:30`
- Delete: `v2/news_filter.py`, `tests/v2/test_news_filter.py` (if present; `ls tests/ tests/v2 | grep news_filter`)
- Test: `tests/v2/test_tools.py`

**Interfaces:**
- Produces: `v2.tools.rank_news_candidates(candidates: list[dict], target_n: int) -> list[int]` — ids ordered by confidence (`high` > `medium` > `low`), then `published_at` descending, truncated to `target_n`.
- `tool_get_curated_news` keeps its signature and output format; the docstring changes from "Haiku-filtered" to "ranked by confidence and recency".

- [ ] **Step 1: Write the failing tests**

```python
# tests/v2/test_tools.py — append
from datetime import datetime, timedelta

from v2 import tools


def test_rank_news_candidates_confidence_then_recency():
    t0 = datetime(2026, 9, 22, 12, 0)
    cands = [
        {"id": 1, "confidence": "low", "published_at": t0},
        {"id": 2, "confidence": "high", "published_at": t0 - timedelta(hours=5)},
        {"id": 3, "confidence": "high", "published_at": t0},
        {"id": 4, "confidence": "medium", "published_at": t0},
    ]
    assert tools.rank_news_candidates(cands, target_n=3) == [3, 2, 4]


def test_curated_news_uses_rank_and_caches(mock_db):
    t0 = datetime(2026, 9, 22, 12, 0)
    rows = [
        {"id": i, "ticker": "AAPL", "category": "earnings", "sentiment": "bullish",
         "confidence": "high" if i % 2 else "low", "published_at": t0, "headline": f"h{i}", "summary": "s"}
        for i in range(1, 6)
    ]
    tools.reset_session()
    with patch("v2.tools.get_news_signals", return_value=rows) as get:
        first = tools.tool_get_curated_news(ticker="AAPL", days=7, target_n=2)
        second = tools.tool_get_curated_news(ticker="AAPL", days=7, target_n=2)
    assert first == second
    assert first.splitlines()[0].startswith("[#5]") and len(first.splitlines()) == 2
    assert get.call_count == 2  # rows re-read, ranking cached
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_tools.py -v -k "rank_news or curated_news_uses_rank"`
Expected: FAIL with `AttributeError: module 'v2.tools' has no attribute 'rank_news_candidates'`

- [ ] **Step 3: Implement**

```python
# v2/tools.py — replace the `from .news_filter import curate_signals` import line with nothing, and add near reset_session():
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def rank_news_candidates(candidates: list[dict], target_n: int) -> list[int]:
    """Deterministic stand-in for the retired Haiku relevance filter.

    Confidence first, recency second. The strategist does its own relevance
    judgement over the shortlist; this only keeps the tool result bounded.
    """
    ordered = sorted(
        candidates,
        key=lambda r: (_CONFIDENCE_RANK.get((r.get("confidence") or "low").lower(), 3), -r["published_at"].timestamp()),
    )
    return [r["id"] for r in ordered[:target_n]]
```

In `tool_get_curated_news`, replace the `else:` branch that calls `get_macro_context(days=2)` and `curate_signals(...)` with:

```python
        else:
            selected_ids = rank_news_candidates(candidates, target_n)
```

Update the docstring's first sentence to: `"""Curated news signals — the ~target_n highest-confidence, most recent for the window."""`. Delete `v2/news_filter.py` and its test file. Search for other references: `grep -rn "news_filter\|curate_signals" v2/ tests/ docs/audit-playbook.md` and fix any (the audit playbook may mention it in prose only; leave prose).

- [ ] **Step 4: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_tools.py -v && task lint`
Expected: PASS; lint clean (an unused import would fail ruff).

- [ ] **Step 5: Commit**

```bash
git add -A v2/tools.py v2/news_filter.py tests/v2/test_tools.py tests/v2/test_news_filter.py
git commit -m "Replace the Haiku news curation call with deterministic ranking

A tool that made its own API call would defeat the subscription-only
driver; the strategist judges relevance itself over a bounded shortlist.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 5: Changelog seam — `dashboard_publish.py --emit-changelog-commits` / `--changelog-entries`

**Files:**
- Modify: `v2/dashboard_publish.py:1337-1360` (`summarize_changelog_commits`), `:1664-1745` (`run_dashboard_stage`), add `emit_changelog_commits`, `changelog_output_schema`, `main`
- Test: `tests/v2/test_dashboard_publish.py`

**Interfaces:**
- Produces:
  - `emit_changelog_commits() -> dict` returning `{"publish_sha": str|None, "last_published_sha": str|None, "commits": [dict], "user_message": str}` (`user_message` is `_changelog_prompt(commits)`, i.e. the JSON payload; empty list when no sha)
  - `changelog_output_schema() -> dict` (object with required `entries` array of `{title, summary, bullets[], commit_shas[]}`)
  - `run_dashboard_stage(session_date=None, *, changelog_entries: list[dict] | None = None, changelog_commits: list[dict] | None = None) -> DashboardStageResult`. When `changelog_entries` is not `None`, the stage validates them with `validate_changelog_entries(entries, changelog_commits)` and stores them instead of calling `summarize_changelog_commits`.
  - CLI: `python -m v2.dashboard_publish --emit-changelog-commits <path>` (exit 0; writes the dict) and `python -m v2.dashboard_publish --changelog-entries <path> --changelog-commits <path>` (runs the stage; exit 1 when `result.errors`). With no flags, runs the stage as today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/v2/test_dashboard_publish.py — append
import json
from unittest.mock import MagicMock, patch

from v2 import dashboard_publish as dp


def test_emit_changelog_commits_reads_pointer_and_builds_payload(mock_db):
    commits = [{"sha": "a" * 40, "short_sha": "aaaaaaa", "committed_at": "t", "subject": "s", "body": "", "files": []}]
    with patch("v2.dashboard_publish.get_current_git_sha", return_value="b" * 40), \
         patch("v2.dashboard_publish.get_changelog_pointer", return_value="c" * 40), \
         patch("v2.dashboard_publish.fetch_changelog_commits", return_value=commits) as fetch:
        out = dp.emit_changelog_commits()
    fetch.assert_called_once_with(from_sha="c" * 40, to_sha="b" * 40)
    assert out["publish_sha"] == "b" * 40 and out["commits"] == commits
    assert json.loads(out["user_message"])[0]["sha"] == "a" * 40


def test_emit_changelog_commits_without_sha_is_empty():
    with patch("v2.dashboard_publish.get_current_git_sha", return_value=None):
        out = dp.emit_changelog_commits()
    assert out["commits"] == [] and out["publish_sha"] is None


def test_changelog_output_schema_shape():
    schema = dp.changelog_output_schema()
    assert schema["required"] == ["entries"]
    assert set(schema["properties"]["entries"]["items"]["required"]) == {"title", "summary", "bullets", "commit_shas"}


def test_run_dashboard_stage_uses_supplied_entries_instead_of_llm(mock_db, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_PAGES_PROJECT", "x")
    commits = [{"sha": "a" * 40, "short_sha": "aaaaaaa", "committed_at": "t", "subject": "s", "body": "", "files": []}]
    entries = [{"title": "T", "summary": "S", "bullets": [], "commit_shas": ["a" * 40]}]
    with patch("v2.dashboard_publish.get_net_deposits", return_value=None), \
         patch("v2.dashboard_publish.gather_dashboard_data", return_value={}), \
         patch("v2.dashboard_publish.get_current_git_sha", return_value="b" * 40), \
         patch("v2.dashboard_publish.get_changelog_pointer", return_value=None), \
         patch("v2.dashboard_publish.fetch_changelog_commits", return_value=commits), \
         patch("v2.dashboard_publish.summarize_changelog_commits") as llm, \
         patch("v2.dashboard_publish.store_changelog_entries") as store, \
         patch("v2.dashboard_publish.get_recent_changelog_entries", return_value=[]), \
         patch("v2.dashboard_publish.assemble_deploy_dir"), \
         patch("v2.dashboard_publish.deploy_to_cloudflare"), \
         patch("v2.dashboard_publish.persist_changelog_pointer"):
        result = dp.run_dashboard_stage(changelog_entries=entries, changelog_commits=commits)
    assert result.published
    llm.assert_not_called()
    assert store.call_args.args[1][0]["title"] == "T"


def test_cli_emit_and_consume(tmp_path):
    out = tmp_path / "c.json"
    with patch("v2.dashboard_publish.emit_changelog_commits", return_value={"commits": []}), \
         patch("sys.argv", ["dp", "--emit-changelog-commits", str(out)]):
        assert dp.main() == 0
    assert json.loads(out.read_text()) == {"commits": []}
    e = tmp_path / "e.json"; e.write_text('{"structured_output": {"entries": [{"title": "T"}]}}')
    c = tmp_path / "cc.json"; c.write_text('{"commits": [{"sha": "x"}]}')
    with patch("v2.dashboard_publish.run_dashboard_stage", return_value=MagicMock(errors=["boom"])) as run, \
         patch("sys.argv", ["dp", "--changelog-entries", str(e), "--changelog-commits", str(c)]):
        assert dp.main() == 1
    assert run.call_args.kwargs == {"changelog_entries": [{"title": "T"}], "changelog_commits": [{"sha": "x"}]}
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_dashboard_publish.py -v -k "emit_changelog or output_schema or supplied_entries or cli_emit"`
Expected: FAIL with `AttributeError: module 'v2.dashboard_publish' has no attribute 'emit_changelog_commits'`

- [ ] **Step 3: Implement**

```python
# v2/dashboard_publish.py — add after summarize_changelog_commits
def emit_changelog_commits() -> dict:
    """Driver mode: the commits since the last publish plus the prompt payload."""
    publish_sha = get_current_git_sha()
    if not publish_sha:
        return {"publish_sha": None, "last_published_sha": None, "commits": [], "user_message": ""}
    with get_cursor() as cur:
        last_published_sha = get_changelog_pointer(cur)
    commits = fetch_changelog_commits(from_sha=last_published_sha, to_sha=publish_sha)
    return {
        "publish_sha": publish_sha,
        "last_published_sha": last_published_sha,
        "commits": commits,
        "user_message": _changelog_prompt(commits) if commits else "",
    }


def changelog_output_schema() -> dict:
    """JSON Schema for the changelog summariser's structured output."""
    return {
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "bullets": {"type": "array", "items": {"type": "string"}},
                        "commit_shas": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "summary", "bullets", "commit_shas"],
                },
            },
        },
        "required": ["entries"],
    }
```

Change the `run_dashboard_stage` signature to `def run_dashboard_stage(session_date: date | None = None, *, changelog_entries: list[dict] | None = None, changelog_commits: list[dict] | None = None) -> DashboardStageResult:` and inside the `if publish_sha:` block replace `entries = summarize_changelog_commits(commits)` with:

```python
                if changelog_entries is not None:
                    entries = validate_changelog_entries(changelog_entries, changelog_commits or commits)
                else:
                    entries = summarize_changelog_commits(commits)
```

Append a CLI:

```python
def main() -> int:
    import argparse
    import sys

    from .log_config import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Publish the public dashboard")
    parser.add_argument("--emit-changelog-commits", metavar="PATH",
                        help="Driver mode: write commits since last publish + prompt payload to PATH")
    parser.add_argument("--changelog-entries", metavar="PATH",
                        help="Driver mode: claude result JSON whose structured_output.entries to store")
    parser.add_argument("--changelog-commits", metavar="PATH", help="The file written by --emit-changelog-commits")
    args = parser.parse_args()

    if args.emit_changelog_commits:
        with open(args.emit_changelog_commits, "w") as fh:
            json.dump(emit_changelog_commits(), fh, default=str)
        return 0

    kwargs = {}
    if args.changelog_entries:
        with open(args.changelog_entries) as fh:
            result = json.load(fh)
        kwargs["changelog_entries"] = (result.get("structured_output") or {}).get("entries") or []
        commits = []
        if args.changelog_commits:
            with open(args.changelog_commits) as fh:
                commits = json.load(fh).get("commits") or []
        kwargs["changelog_commits"] = commits
    result = run_dashboard_stage(**kwargs)
    if result.errors:
        for err in result.errors:
            logger.error("  - %s", err)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

Update `Taskfile.yml:263` (`dashboard:publish`) from the `python -c` form to `python -m v2.dashboard_publish`.

- [ ] **Step 4: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_dashboard_publish.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py Taskfile.yml
git commit -m "Let the driver supply changelog entries to the dashboard stage

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `session_ctl` — plan, stage-begin, finalize, run-learning, prompt, schema

**Files:**
- Create: `v2/session_ctl.py`
- Test: `tests/v2/test_session_ctl.py`
- Modify: `tests/v2/conftest.py:33-60` (add a `_SESSION_CTL_DB_PATCH_TARGETS` tuple, see Step 3)

**Interfaces:**
- Produces (all subcommands print one JSON object to stdout; `python -m v2.session_ctl <cmd> ...`):
  - `plan [--force] [--dry-run] [--only STAGE] [--skip-supervisor --skip-pipeline --skip-ideation --skip-executor --skip-strategy --skip-dashboard] [--pipeline-hours N] [--pipeline-limit N]` → `PlanResult` dict:
    ```
    {"session_id": int|null, "session_date": "YYYY-MM-DD", "skip_reason": str|null, "dry_run": bool,
     "stages": [{"name": str, "kind": "python"|"agentic"|"structured"|"batches", "skip": bool, "skip_reason": str|null,
                 "model": str|null, "context_stage": str|null, "registry": str|null, "max_tool_calls": int|null,
                 "timeout_seconds": int, "allowed_tools": str|null, "prompt": str|null, "schema": str|null,
                 "command": str|null, "emit": str|null, "consume": str|null}]}
    ```
    Stage order and fields are fixed by `STAGE_SPECS` below. `command`/`emit`/`consume` are argv strings to run inside the container, with `{session_id}`, `{dir}`, `{file}`, `{out}`, `{dry_run}` placeholders the driver substitutes.
  - `stage-begin --session-id N --stage NAME` → `{"ok": true}`
  - `finalize --session-id N` → `{"exit_code": 0|1, "errors": {stage: error}}`
  - `run-learning` → `{"ok": true}` (runs `run_backfill()` then `compute_signal_attribution()`; exit 1 on exception)
  - `prompt --stage NAME` → prints the system prompt text (not JSON) via `v2.prompts`
  - `schema --stage NAME` → prints the JSON schema for `executor` / `pipeline` / `dashboard`
  - Python API used by tests and by Task 7: `build_plan(...) -> dict`, `resolve_stage_model(stage: str) -> str`, `STAGE_SPECS: list[dict]`, `DEFAULT_MODELS: dict[str, str]`.
- Consumes: `v2.session._trading_halted`, `v2.session._dashboard_publish_enabled`, `v2.session._check_and_record_session`, `v2.session.current_market_date`, `v2.database.trading_db.{get_completed_stages, insert_session_stage, close_orphan_running_stages, complete_session, fail_session, expire_stale_playbook_actions}`, `v2.session._log_session_costs`, `v2.telemetry.session_summary_line`, `v2.agent.executor_output_schema`, `v2.classifier.classification_output_schema`, `v2.dashboard_publish.changelog_output_schema`, `v2.prompts.{load_prompt, load_skill_prompt}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/v2/test_session_ctl.py
"""session_ctl owns every decision the bash driver used to get from
run_session: gates, idempotency, stage order, models, exit codes."""
import json
from datetime import date
from unittest.mock import patch

import pytest

from v2 import session_ctl as sc

STAGE_ORDER = ["learning", "supervisor", "pipeline", "strategist", "executor", "strategy", "dashboard"]


@pytest.fixture
def session_row():
    with patch("v2.session_ctl._check_and_record_session", return_value=(42, set(), None)), \
         patch("v2.session_ctl.get_completed_stages", return_value=set()), \
         patch("v2.session_ctl.expire_stale_playbook_actions", return_value=0), \
         patch("v2.session_ctl.current_market_date", return_value=date(2026, 9, 22)):
        yield


def test_plan_stage_order_and_kinds(session_row, monkeypatch):
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "1")
    plan = sc.build_plan(force=False, dry_run=False, only=None, skips={}, pipeline_hours=24, pipeline_limit=300)
    assert plan["session_id"] == 42 and plan["session_date"] == "2026-09-22" and plan["skip_reason"] is None
    assert [s["name"] for s in plan["stages"]] == STAGE_ORDER
    kinds = {s["name"]: s["kind"] for s in plan["stages"]}
    assert kinds == {"learning": "python", "supervisor": "agentic", "pipeline": "batches", "strategist": "agentic",
                     "executor": "structured", "strategy": "agentic", "dashboard": "structured"}
    assert not any(s["skip"] for s in plan["stages"])


def test_plan_halted_returns_skip_reason_and_no_session(monkeypatch):
    monkeypatch.setenv("ALGO_TRADING_HALTED", "1")
    with patch("v2.session_ctl._check_and_record_session") as check:
        plan = sc.build_plan(force=False, dry_run=False, only=None, skips={}, pipeline_hours=24, pipeline_limit=300)
    assert "ALGO_TRADING_HALTED" in plan["skip_reason"] and plan["stages"] == [] and plan["session_id"] is None
    check.assert_not_called()


def test_plan_idempotent_skip():
    with patch("v2.session_ctl._check_and_record_session", return_value=(None, set(), "Session already exists")), \
         patch("v2.session_ctl.expire_stale_playbook_actions", return_value=0), \
         patch("v2.session_ctl.current_market_date", return_value=date(2026, 9, 22)):
        plan = sc.build_plan(force=False, dry_run=False, only=None, skips={}, pipeline_hours=24, pipeline_limit=300)
    assert plan["skip_reason"] == "Session already exists" and plan["stages"] == []


def test_plan_dry_run_promotes_skips_and_publish_gate(session_row, monkeypatch):
    monkeypatch.delenv("ALGO_DASHBOARD_PUBLISH", raising=False)
    plan = sc.build_plan(force=False, dry_run=True, only=None, skips={}, pipeline_hours=24, pipeline_limit=300)
    skipped = {s["name"]: s["skip_reason"] for s in plan["stages"] if s["skip"]}
    assert set(skipped) == {"strategist", "strategy", "dashboard"}
    assert skipped["dashboard"] == "--skip-dashboard/--dry-run"
    assert plan["dry_run"] is True
    plan2 = sc.build_plan(force=False, dry_run=False, only=None, skips={}, pipeline_hours=24, pipeline_limit=300)
    dash = next(s for s in plan2["stages"] if s["name"] == "dashboard")
    assert dash["skip"] and dash["skip_reason"] == "ALGO_DASHBOARD_PUBLISH not enabled for this instance"


def test_plan_force_resumes_past_completed_stages(monkeypatch):
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "1")
    with patch("v2.session_ctl._check_and_record_session", return_value=(42, set(), None)), \
         patch("v2.session_ctl.get_completed_stages", return_value={"learning", "supervisor"}), \
         patch("v2.session_ctl.expire_stale_playbook_actions", return_value=0), \
         patch("v2.session_ctl.current_market_date", return_value=date(2026, 9, 22)):
        plan = sc.build_plan(force=True, dry_run=False, only=None, skips={}, pipeline_hours=24, pipeline_limit=300)
    skipped = {s["name"]: s["skip_reason"] for s in plan["stages"] if s["skip"]}
    assert skipped == {"learning": "completed in prior run", "supervisor": "completed in prior run"}


def test_plan_only_skips_everything_else(session_row, monkeypatch):
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "1")
    plan = sc.build_plan(force=False, dry_run=True, only="executor", skips={}, pipeline_hours=24, pipeline_limit=300)
    assert [s["name"] for s in plan["stages"] if not s["skip"]] == ["executor"]


def test_plan_explicit_skip_flags(session_row, monkeypatch):
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "1")
    plan = sc.build_plan(force=False, dry_run=False, only=None, skips={"pipeline": True, "executor": True},
                         pipeline_hours=24, pipeline_limit=300)
    assert {s["name"] for s in plan["stages"] if s["skip"]} == {"pipeline", "executor"}


def test_plan_models_are_full_ids_and_env_overridable(session_row, monkeypatch):
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "1")
    monkeypatch.setenv("ALGO_STAGE_MODEL_STRATEGIST", "claude-sonnet-4-6")
    monkeypatch.setenv("ALGO_EXECUTOR_MODEL", "claude-haiku-4-5")
    plan = sc.build_plan(force=False, dry_run=False, only=None, skips={}, pipeline_hours=24, pipeline_limit=300)
    models = {s["name"]: s["model"] for s in plan["stages"]}
    assert models["strategist"] == "claude-sonnet-4-6" and models["executor"] == "claude-haiku-4-5"
    assert models["learning"] is None
    for m in models.values():
        assert m is None or (m.startswith("claude-") and "-" in m[7:])


def test_plan_commands_carry_placeholders(session_row, monkeypatch):
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "1")
    plan = sc.build_plan(force=False, dry_run=False, only=None, skips={}, pipeline_hours=6, pipeline_limit=10)
    by = {s["name"]: s for s in plan["stages"]}
    assert by["learning"]["command"] == "python -m v2.session_ctl run-learning"
    assert by["pipeline"]["emit"] == "python -m v2.pipeline --emit-batches {dir} --hours 6 --limit 10"
    assert by["pipeline"]["consume"] == "python -m v2.pipeline --ingest {dir} --session-id {session_id}"
    assert by["executor"]["emit"] == "python -m v2.trader --emit-input {out} --session-id {session_id}{dry_run}"
    assert by["executor"]["consume"] == "python -m v2.trader --decisions-file {file} --input-file {out} --session-id {session_id}{dry_run}"
    assert by["strategist"]["registry"] == "strategist" and by["strategist"]["allowed_tools"] == "mcp__pinchy__*,WebSearch"
    assert by["strategy"]["registry"] == "reflection" and by["supervisor"]["registry"] == "supervisor"
    assert by["strategist"]["max_tool_calls"] == 60 and by["strategist"]["timeout_seconds"] == 2400


def test_finalize_exit_code_and_error_map():
    with patch("v2.session_ctl.close_orphan_running_stages", return_value=[]), \
         patch("v2.session_ctl.get_stage_errors", return_value={"executor": "boom"}), \
         patch("v2.session_ctl.fail_session") as fail, \
         patch("v2.session_ctl.complete_session") as complete, \
         patch("v2.session_ctl._log_session_costs"), \
         patch("v2.session_ctl.session_summary_line", return_value=""):
        out = sc.finalize(42)
    assert out == {"exit_code": 1, "errors": {"executor": "boom"}}
    fail.assert_called_once_with(42, "executor: boom")
    complete.assert_not_called()


def test_finalize_clean():
    with patch("v2.session_ctl.close_orphan_running_stages", return_value=[]), \
         patch("v2.session_ctl.get_stage_errors", return_value={}), \
         patch("v2.session_ctl.complete_session") as complete, \
         patch("v2.session_ctl._log_session_costs"), \
         patch("v2.session_ctl.session_summary_line", return_value=""):
        assert sc.finalize(42) == {"exit_code": 0, "errors": {}}
    complete.assert_called_once_with(42)


def test_cli_plan_prints_json(capsys, session_row, monkeypatch):
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "1")
    with patch("sys.argv", ["session_ctl", "plan", "--only", "learning"]):
        assert sc.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["session_id"] == 42


def test_cli_stage_begin(capsys):
    with patch("v2.session_ctl.insert_session_stage") as ins, \
         patch("sys.argv", ["session_ctl", "stage-begin", "--session-id", "42", "--stage", "pipeline"]):
        assert sc.main() == 0
    ins.assert_called_once_with(42, "pipeline")


def test_cli_prompt_and_schema(capsys):
    with patch("sys.argv", ["session_ctl", "prompt", "--stage", "executor"]):
        assert sc.main() == 0
    assert "decisions" in capsys.readouterr().out
    with patch("sys.argv", ["session_ctl", "schema", "--stage", "pipeline"]):
        assert sc.main() == 0
    assert json.loads(capsys.readouterr().out)["required"] == ["classifications"]


def test_cli_run_learning_exit_codes(capsys):
    with patch("v2.session_ctl.run_backfill") as bf, patch("v2.session_ctl.compute_signal_attribution") as attr, \
         patch("sys.argv", ["session_ctl", "run-learning"]):
        assert sc.main() == 0
    bf.assert_called_once(); attr.assert_called_once()
    with patch("v2.session_ctl.run_backfill", side_effect=RuntimeError("db down")), \
         patch("sys.argv", ["session_ctl", "run-learning"]):
        assert sc.main() == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_session_ctl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2.session_ctl'`

- [ ] **Step 3: Add the DB safety net for the new module**

```python
# tests/v2/conftest.py — add after _SESSION_DB_PATCH_TARGETS
_SESSION_CTL_DB_PATCH_TARGETS = (
    # v2.session_ctl talks to the DB through these import-site names. Default
    # them to mocks so a test that forgets to patch cannot touch the paper DB.
    "v2.session_ctl.insert_session_stage",
    "v2.session_ctl.complete_session_stage",
    "v2.session_ctl.fail_session_stage",
    "v2.session_ctl.close_orphan_running_stages",
    "v2.session_ctl.complete_session",
    "v2.session_ctl.fail_session",
    "v2.session_ctl.get_completed_stages",
    "v2.session_ctl.get_stage_errors",
    "v2.session_ctl.expire_stale_playbook_actions",
    "v2.session_ctl.run_backfill",
    "v2.session_ctl.compute_signal_attribution",
    "v2.session_ctl._log_session_costs",
)
```

and in `_block_social_llm_and_session_db_calls` add, inside the `ExitStack` block, before `yield`:

```python
        for target in _SESSION_CTL_DB_PATCH_TARGETS:
            stack.enter_context(patch(target))
```

- [ ] **Step 4: Implement `v2/session_ctl.py` (plan / begin / finalize / learning / prompt / schema)**

```python
# v2/session_ctl.py
"""Session control for the Claude Code driver (session-driver.sh).

The bash driver never decides anything. It calls `plan`, runs what the plan
says, reports each stage with `stage-begin` / `stage-end`, and calls
`finalize`. Every gate, validator, and telemetry write lives here so it is
tested Python, not shell. Stage helpers are imported from v2.session (the API
fallback) rather than copied.

Stage names follow session_stages: learning, supervisor, pipeline,
strategist, executor, strategy, dashboard. `context_stage` is the
llm_call_contexts / agent_events name for the same stage.
"""
import argparse
import json
import logging
import os
import sys

from .agent import DEFAULT_EXECUTOR_MODEL, executor_output_schema
from .attribution import compute_signal_attribution
from .backfill import run_backfill
from .classifier import classification_output_schema
from .dashboard_publish import changelog_output_schema
from .database.trading_db import (
    close_orphan_running_stages,
    complete_session,
    complete_session_stage,
    expire_stale_playbook_actions,
    fail_session,
    fail_session_stage,
    get_completed_stages,
    get_stage_errors,
    insert_session_stage,
)
from .prompts import load_prompt, load_skill_prompt
from .session import (
    _check_and_record_session,
    _dashboard_publish_enabled,
    _log_session_costs,
    _trading_halted,
    current_market_date,
)
from .telemetry import session_summary_line

logger = logging.getLogger("session_ctl")

DEFAULT_MODELS = {
    "supervisor": "claude-fable-5",
    "pipeline": "claude-haiku-4-5-20251001",
    "strategist": "claude-opus-4-8",
    "executor": DEFAULT_EXECUTOR_MODEL,
    "strategy": "claude-sonnet-4-6",
    "dashboard": "claude-haiku-4-5-20251001",
}

# One entry per session_stages stage, in run order. Placeholders in the
# argv strings are substituted by the driver: {session_id}, {dir} (batch
# dir), {out} (emitted input file), {file} (claude result file), {dry_run}
# (" --dry-run" or "").
STAGE_SPECS = [
    {"name": "learning", "kind": "python", "context_stage": None, "timeout_seconds": 900,
     "command": "python -m v2.session_ctl run-learning"},
    {"name": "supervisor", "kind": "agentic", "context_stage": "supervisor", "registry": "supervisor",
     "max_tool_calls": 40, "timeout_seconds": 1200, "allowed_tools": "mcp__pinchy__*", "skip_flag": "supervisor"},
    {"name": "pipeline", "kind": "batches", "context_stage": "pipeline", "prompt": "classifier", "schema": "pipeline",
     "timeout_seconds": 300, "skip_flag": "pipeline",
     "emit": "python -m v2.pipeline --emit-batches {dir} --hours {pipeline_hours} --limit {pipeline_limit}",
     "consume": "python -m v2.pipeline --ingest {dir} --session-id {session_id}"},
    {"name": "strategist", "kind": "agentic", "context_stage": "ideation", "registry": "strategist",
     "max_tool_calls": 60, "timeout_seconds": 2400, "allowed_tools": "mcp__pinchy__*,WebSearch", "skip_flag": "ideation"},
    {"name": "executor", "kind": "structured", "context_stage": "trading", "prompt": "executor", "schema": "executor",
     "timeout_seconds": 300, "skip_flag": "executor",
     "emit": "python -m v2.trader --emit-input {out} --session-id {session_id}{dry_run}",
     "consume": "python -m v2.trader --decisions-file {file} --input-file {out} --session-id {session_id}{dry_run}"},
    {"name": "strategy", "kind": "agentic", "context_stage": "reflection", "registry": "reflection",
     "max_tool_calls": 25, "timeout_seconds": 1200, "allowed_tools": "mcp__pinchy__*", "skip_flag": "strategy"},
    {"name": "dashboard", "kind": "structured", "context_stage": "dashboard_publish", "prompt": "changelog",
     "schema": "dashboard", "timeout_seconds": 300, "skip_flag": "dashboard",
     "emit": "python -m v2.dashboard_publish --emit-changelog-commits {out}",
     "consume": "python -m v2.dashboard_publish --changelog-entries {file} --changelog-commits {out}"},
]

_STAGE_FIELDS = ("name", "kind", "context_stage", "registry", "max_tool_calls", "timeout_seconds",
                 "allowed_tools", "prompt", "schema", "command", "emit", "consume")


def resolve_stage_model(stage: str) -> str | None:
    """Full model id for an LLM stage: ALGO_STAGE_MODEL_<STAGE> wins, then the
    executor's own ALGO_EXECUTOR_MODEL knob, then DEFAULT_MODELS."""
    if stage not in DEFAULT_MODELS:
        return None
    override = os.environ.get(f"ALGO_STAGE_MODEL_{stage.upper()}", "").strip()
    if override:
        return override
    if stage == "executor":
        return os.environ.get("ALGO_EXECUTOR_MODEL", "").strip() or DEFAULT_MODELS["executor"]
    return DEFAULT_MODELS[stage]


def build_plan(*, force: bool, dry_run: bool, only: str | None, skips: dict[str, bool],
               pipeline_hours: int, pipeline_limit: int) -> dict:
    """Mirror of run_session's preamble, returning stages instead of running them."""
    plan = {"session_id": None, "session_date": None, "skip_reason": None, "dry_run": dry_run, "stages": []}
    if _trading_halted():
        plan["skip_reason"] = ("ALGO_TRADING_HALTED is set — session skipped "
                               "(halt/resume procedure: docs/runbook-recovery.md)")
        return plan

    skips = dict(skips)
    if dry_run:
        skips.update({"ideation": True, "strategy": True, "dashboard": True})
    if skips.get("dashboard"):
        dashboard_reason = "--skip-dashboard/--dry-run"
    elif not _dashboard_publish_enabled():
        skips["dashboard"] = True
        dashboard_reason = "ALGO_DASHBOARD_PUBLISH not enabled for this instance"
    else:
        dashboard_reason = None

    today = current_market_date()
    plan["session_date"] = today.isoformat()
    try:
        expired = expire_stale_playbook_actions(today)
        if expired:
            logger.info("Expired %d stale pending playbook_actions from prior days", expired)
    except Exception as e:
        logger.warning("expire_stale_playbook_actions failed (non-fatal): %s", e)

    session_id, _, early_error = _check_and_record_session(force, today)
    if early_error:
        plan["skip_reason"] = early_error
        return plan
    plan["session_id"] = session_id

    completed = set()
    if force and session_id is not None:
        try:
            completed = get_completed_stages(session_id)
        except Exception as e:
            logger.warning("Could not read completed stages: %s", e)

    for spec in STAGE_SPECS:
        stage = {k: spec.get(k) for k in _STAGE_FIELDS}
        stage["model"] = resolve_stage_model(spec["name"])
        stage["skip"], stage["skip_reason"] = False, None
        flag = spec.get("skip_flag")
        if only and spec["name"] != only:
            stage["skip"], stage["skip_reason"] = True, f"--only {only}"
        elif spec["name"] in completed:
            stage["skip"], stage["skip_reason"] = True, "completed in prior run"
        elif flag and skips.get(flag):
            stage["skip"] = True
            stage["skip_reason"] = dashboard_reason if spec["name"] == "dashboard" else f"--skip-{flag}"
        for key in ("emit", "consume", "command"):
            if stage.get(key):
                stage[key] = stage[key].replace("{pipeline_hours}", str(pipeline_hours)).replace("{pipeline_limit}", str(pipeline_limit))
        plan["stages"].append(stage)
    return plan


def finalize(session_id: int) -> dict:
    """Mirror of _finalize_session using the session_stages rows as the error source."""
    try:
        swept = close_orphan_running_stages(session_id)
        if swept:
            logger.warning("Closed orphan running stages: %s", swept)
    except Exception as e:
        logger.warning("Could not sweep orphan stages: %s", e)
    errors = get_stage_errors(session_id)
    try:
        if errors:
            fail_session(session_id, "; ".join(f"{k}: {v}" for k, v in errors.items()))
        else:
            complete_session(session_id)
    except Exception as e:
        logger.warning("Could not update session status: %s", e)
    _log_session_costs(session_id)
    try:
        logger.info(session_summary_line(session_id))
    except Exception:
        logger.exception("session_summary_line failed; continuing")
    return {"exit_code": 1 if errors else 0, "errors": errors}


def _emit(obj) -> None:
    print(json.dumps(obj, default=str))


def main() -> int:
    from .log_config import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(prog="session_ctl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default=None)
    for flag in ("supervisor", "pipeline", "ideation", "executor", "strategy", "dashboard"):
        p.add_argument(f"--skip-{flag}", action="store_true")
    p.add_argument("--pipeline-hours", type=int, default=24)
    p.add_argument("--pipeline-limit", type=int, default=300)

    p = sub.add_parser("stage-begin")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--stage", required=True)

    p = sub.add_parser("finalize")
    p.add_argument("--session-id", type=int, required=True)

    sub.add_parser("run-learning")

    p = sub.add_parser("prompt")
    p.add_argument("--stage", required=True, choices=["executor", "pipeline", "dashboard", "strategist", "strategy", "supervisor"])

    p = sub.add_parser("schema")
    p.add_argument("--stage", required=True, choices=["executor", "pipeline", "dashboard"])

    _add_stage_context_parser(sub)  # Task 7
    _add_stage_end_parser(sub)      # Task 7

    args = parser.parse_args()
    if args.cmd == "plan":
        skips = {f: getattr(args, f"skip_{f}") for f in ("supervisor", "pipeline", "ideation", "executor", "strategy", "dashboard")}
        _emit(build_plan(force=args.force, dry_run=args.dry_run, only=args.only, skips=skips,
                         pipeline_hours=args.pipeline_hours, pipeline_limit=args.pipeline_limit))
        return 0
    if args.cmd == "stage-begin":
        insert_session_stage(args.session_id, args.stage)
        _emit({"ok": True})
        return 0
    if args.cmd == "finalize":
        _emit(finalize(args.session_id))
        return 0
    if args.cmd == "run-learning":
        try:
            run_backfill()
            compute_signal_attribution()
        except Exception as e:
            logger.error("Learning refresh failed: %s", e)
            _emit({"ok": False, "error": str(e)})
            return 1
        _emit({"ok": True})
        return 0
    if args.cmd == "prompt":
        skill = {"strategist": "strategist", "strategy": "reflection", "supervisor": "supervisor"}
        file = {"executor": "executor", "pipeline": "classifier", "dashboard": "changelog"}
        print(load_skill_prompt(skill[args.stage]) if args.stage in skill else load_prompt(file[args.stage]))
        return 0
    if args.cmd == "schema":
        _emit({"executor": executor_output_schema, "pipeline": classification_output_schema,
               "dashboard": changelog_output_schema}[args.stage]())
        return 0
    return _run_task7_commands(args)  # Task 7


def _add_stage_context_parser(sub):  # replaced in Task 7
    pass


def _add_stage_end_parser(sub):  # replaced in Task 7
    pass


def _run_task7_commands(args) -> int:  # replaced in Task 7
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
```

Add `get_stage_errors` to `v2/database/trading_db.py` next to `get_completed_stages`:

```python
def get_stage_errors(session_id: int) -> dict[str, str]:
    """{stage_name: error} for every failed stage of a session, in stage-row order."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT stage_name, error FROM session_stages
            WHERE session_id = %s AND status = 'failed'
            ORDER BY id
        """, (session_id,))
        return {row["stage_name"]: row["error"] or "failed" for row in cur.fetchall()}
```

with a test in `tests/v2/test_db.py` following the file's existing pattern (`mock_db` fixture, `mock_cursor.fetchall.return_value = [{"stage_name": "executor", "error": None}]`, assert `{"executor": "failed"}`).

- [ ] **Step 5: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_session_ctl.py tests/v2/test_db.py tests/test_conftest_gate.py -v && task lint`
Expected: PASS; lint clean.

- [ ] **Step 6: Commit**

```bash
git add v2/session_ctl.py v2/database/trading_db.py tests/v2/test_session_ctl.py tests/v2/test_db.py tests/v2/conftest.py
git commit -m "Add session_ctl: plan, stage-begin, finalize, run-learning, prompt, schema

The bash driver asks this CLI what to run; gates, idempotency, stage order
and models are decided here, reusing v2.session's helpers.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 7: `session_ctl stage-context` and `stage-end` — context, validators, usage, telemetry

**Files:**
- Modify: `v2/session_ctl.py` (replace the three Task 6 stubs)
- Modify: `v2/trader.py` `main()` (add `--result-out PATH` to the `--decisions-file` path)
- Test: `tests/v2/test_session_ctl.py`, `tests/v2/test_trader.py`

**Interfaces:**
- `stage-context --stage {strategist,strategy,supervisor} [--session-id N] [--executor-result PATH]` → prints the initial user message text (not JSON). Strategist: attribution constraints + formation context header, pre-seeded sections, orphan block, watchlist block, numbered instructions (same text as `run_strategist_loop` builds, with the system-prompt appendices moved to the top of the message under `=== SESSION CONTEXT (appended to your instructions) ===`). Reflection (`strategy`): `_format_trading_context` of the executor result file if given, revalidation evidence, watchlist block, the five-step instruction. Supervisor: `"Begin your strategy review. Investigate, cite IDs, then write the memo."`.
- `stage-end --session-id N --stage NAME [--result PATH] [--transcript PATH] [--tools-log PATH] [--context PATH] [--input PATH] [--exit-code N] [--error TEXT]` → `{"status": "completed"|"failed", "error": str|null, "error_class": str|null}`. Exit 0 always (the driver reads the JSON).
- `tools.jsonl` line format (written by Task 8's MCP server, read here): `{"tool_name": str, "args": dict, "success": bool, "error": str|null, "duration_ms": int, "output_chars": int, "result_prefix": str}` where `result_prefix` is the first 200 characters of the tool's string result.
- `claude -p --output-format json` result fields used: `is_error`, `result`, `structured_output`, `stop_reason`, `duration_api_ms`, `usage{input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}`, `modelUsage{<model>: {inputTokens, outputTokens, cacheCreationInputTokens, cacheReadInputTokens}}`. For `stream-json` transcripts the last line is the same result object with `"type": "result"`.
- Executor result file (`trader.py --result-out`): `{"decisions_made", "trades_executed", "trades_failed", "total_buy_value", "total_sell_value", "market_summary", "risk_assessment", "errors"}`.
- Python API: `usage_from_result(result: dict) -> UsageAccumulator | None`, `classify_failure(*, exit_code: int | None, result: dict | None, error: str | None) -> tuple[str, str] | None`, `read_tools_log(path) -> list[dict]`, `messages_from_transcript(lines: list[str], initial_context: str | None) -> tuple[list[dict], list | None, str | None]`, `validate_stage(stage, session_id, session_date, result, tools) -> str | None` (returns an error string or None), `end_stage(...)-> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/v2/test_session_ctl.py — append
import json
from types import SimpleNamespace

RESULT = {
    "is_error": False, "result": "Playbook written. Three theses updated, one closed; leaning defensive into CPI.",
    "stop_reason": "end_turn", "duration_api_ms": 1200,
    "usage": {"input_tokens": 10, "output_tokens": 50, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 900},
    "modelUsage": {"claude-opus-4-8": {"inputTokens": 10, "outputTokens": 50, "cacheCreationInputTokens": 100, "cacheReadInputTokens": 900}},
}


def test_usage_from_result_maps_model_usage():
    acc = sc.usage_from_result(RESULT)
    assert acc.model == "claude-opus-4-8" and acc.input_tokens == 10 and acc.output_tokens == 50
    assert acc.cache_creation_tokens == 100 and acc.cache_read_tokens == 900 and not acc.mixed_models


def test_usage_from_result_two_models_marks_mixed_and_picks_biggest():
    r = {"modelUsage": {"claude-haiku-4-5-20251001": {"inputTokens": 5, "outputTokens": 1},
                        "claude-opus-4-8": {"inputTokens": 500, "outputTokens": 9}}}
    acc = sc.usage_from_result(r)
    assert acc.model == "claude-opus-4-8" and acc.mixed_models and acc.input_tokens == 505


def test_usage_from_result_none_without_usage():
    assert sc.usage_from_result({}) is None and sc.usage_from_result(None) is None


@pytest.mark.parametrize("exit_code,result,error,expected", [
    (124, None, None, "timeout"),
    (1, {"is_error": True, "result": "You've hit your usage limit"}, None, "usage_limit"),
    (1, {"is_error": True, "result": "rate limit exceeded (429)"}, None, "usage_limit"),
    (1, {"is_error": True, "result": "Not logged in. Please run /login"}, None, "auth"),
    (1, {"is_error": True, "result": "something odd"}, None, "model_error"),
    (2, None, "stderr tail", "command"),
    (0, {"is_error": False}, None, None),
])
def test_classify_failure(exit_code, result, error, expected):
    out = sc.classify_failure(exit_code=exit_code, result=result, error=error)
    assert (out[0] if out else None) == expected


def test_messages_from_transcript_builds_context_rows():
    lines = [
        json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-4-8"}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "mcp__pinchy__get_theses", "input": {}}]}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[]"}]}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}),
        json.dumps({"type": "result", "is_error": False, "result": "done"}),
    ]
    messages, response, model = sc.messages_from_transcript(lines, initial_context="STATE")
    assert messages[0] == {"role": "user", "content": "STATE"}
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert response == [{"type": "text", "text": "done"}] and model == "claude-opus-4-8"


def test_read_tools_log_skips_bad_lines(tmp_path):
    p = tmp_path / "tools.jsonl"
    p.write_text('{"tool_name": "a", "success": true, "result_prefix": "ok"}\nnot json\n')
    assert sc.read_tools_log(str(p)) == [{"tool_name": "a", "success": True, "result_prefix": "ok"}]


def test_validate_strategist_requires_playbook():
    with patch("v2.session_ctl.get_playbook", return_value=None):
        assert "playbook" in sc.validate_stage("strategist", 42, date(2026, 9, 22), RESULT, [])
    with patch("v2.session_ctl.get_playbook", return_value={"id": 1}):
        assert sc.validate_stage("strategist", 42, date(2026, 9, 22), RESULT, []) is None


def test_validate_reflection_memo_revalidation_watchlist():
    tools_ok = [{"tool_name": "write_strategy_memo", "success": True, "result_prefix": "Memo written (ID: 9)"},
                {"tool_name": "revalidate_rule", "success": True, "result_prefix": "Revalidated rule ID 3: kept with lift condition."}]
    gated = [{"rule_id": 3, "lift_condition": None, "consecutive_session_streak": 5}]
    with patch("v2.session_ctl.get_rule_gate_revalidation_candidates", return_value=[]), \
         patch("v2.session_ctl.wl.assert_watchlist_resolved"):
        assert "memo" in sc.validate_stage("strategy", 42, date(2026, 9, 22), RESULT, [])
        assert sc.validate_stage("strategy", 42, date(2026, 9, 22), RESULT, tools_ok) is None
    with patch("v2.session_ctl.get_rule_gate_revalidation_candidates", return_value=gated), \
         patch("v2.session_ctl.wl.assert_watchlist_resolved"):
        assert "revalidating gated rule(s): 3" in sc.validate_stage("strategy", 42, date(2026, 9, 22), RESULT, tools_ok[:1])
    with patch("v2.session_ctl.get_rule_gate_revalidation_candidates", return_value=[]), \
         patch("v2.session_ctl.wl.assert_watchlist_resolved", side_effect=RuntimeError("unresolved 7")):
        assert "unresolved 7" in sc.validate_stage("strategy", 42, date(2026, 9, 22), RESULT, tools_ok)


def test_validate_supervisor_requires_memo_tool():
    assert "write_supervisor_memo" in sc.validate_stage("supervisor", 42, date(2026, 9, 22), RESULT, [])
    ok = [{"tool_name": "write_supervisor_memo", "success": True, "result_prefix": "Supervisor memo written (ID: 5)"}]
    assert sc.validate_stage("supervisor", 42, date(2026, 9, 22), RESULT, ok) is None


def test_end_stage_success_writes_usage_context_events_and_memo(tmp_path):
    result_path = tmp_path / "result.json"; result_path.write_text(json.dumps(RESULT))
    transcript = tmp_path / "t.jsonl"; transcript.write_text(json.dumps(
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}) + "\n")
    tools = tmp_path / "tools.jsonl"; tools.write_text(json.dumps(
        {"tool_name": "write_playbook", "args": {}, "success": True, "error": None, "duration_ms": 3, "output_chars": 10, "result_prefix": "ok"}) + "\n")
    ctx = tmp_path / "ctx.md"; ctx.write_text("STATE")
    with patch("v2.session_ctl.get_playbook", return_value={"id": 1}), \
         patch("v2.session_ctl.complete_session_stage") as complete, \
         patch("v2.session_ctl.insert_llm_call_context") as llm, \
         patch("v2.session_ctl.record_event") as ev, \
         patch("v2.session_ctl.get_current_strategy_state", return_value={"id": 2}), \
         patch("v2.session_ctl.insert_strategy_memo") as memo, \
         patch("v2.session_ctl.load_skill_prompt", return_value="SYS"), \
         patch("v2.session_ctl.registry_tool_definitions", return_value=[{"name": "write_playbook"}]):
        out = sc.end_stage(session_id=42, stage="strategist", session_date=date(2026, 9, 22), result_path=str(result_path),
                           transcript_path=str(transcript), tools_log_path=str(tools), context_path=str(ctx),
                           input_path=None, exit_code=0, error=None)
    assert out == {"status": "completed", "error": None, "error_class": None}
    assert complete.call_args.args[:2] == (42, "strategist") and complete.call_args.kwargs["usage"].model == "claude-opus-4-8"
    kw = llm.call_args.kwargs
    assert kw["stage_name"] == "ideation" and kw["purpose"] == "strategist_loop" and kw["system_prompt"] == "SYS"
    assert kw["messages"][0] == {"role": "user", "content": "STATE"} and kw["tool_definitions"] == [{"name": "write_playbook"}]
    events = [c.kwargs["event_type"] for c in ev.call_args_list]
    assert events == ["agent_call", "tool_invocation", "loop_completion"]
    assert memo.call_args.kwargs["memo_type"] == "strategist_notes" and memo.call_args.kwargs["session_id"] == 42


def test_end_stage_short_summary_skips_memo(tmp_path):
    r = dict(RESULT, result="all good"); p = tmp_path / "r.json"; p.write_text(json.dumps(r))
    with patch("v2.session_ctl.get_playbook", return_value={"id": 1}), \
         patch("v2.session_ctl.complete_session_stage"), patch("v2.session_ctl.record_event"), \
         patch("v2.session_ctl.insert_strategy_memo") as memo:
        sc.end_stage(session_id=42, stage="strategist", session_date=date(2026, 9, 22), result_path=str(p),
                     transcript_path=None, tools_log_path=None, context_path=None, input_path=None, exit_code=0, error=None)
    memo.assert_not_called()


def test_end_stage_failure_records_class_and_partial_usage(tmp_path):
    r = dict(RESULT, is_error=True, result="You've hit your usage limit"); p = tmp_path / "r.json"; p.write_text(json.dumps(r))
    with patch("v2.session_ctl.fail_session_stage") as fail, patch("v2.session_ctl.record_event"):
        out = sc.end_stage(session_id=42, stage="strategy", session_date=date(2026, 9, 22), result_path=str(p),
                           transcript_path=None, tools_log_path=None, context_path=None, input_path=None, exit_code=1, error=None)
    assert out["status"] == "failed" and out["error_class"] == "usage_limit"
    assert fail.call_args.args[:2] == (42, "strategy") and fail.call_args.args[2].startswith("[usage_limit]")
    assert fail.call_args.kwargs["usage"].model == "claude-opus-4-8"


def test_end_stage_validator_failure_is_failed_stage(tmp_path):
    p = tmp_path / "r.json"; p.write_text(json.dumps(RESULT))
    with patch("v2.session_ctl.get_playbook", return_value=None), \
         patch("v2.session_ctl.fail_session_stage") as fail, patch("v2.session_ctl.record_event"):
        out = sc.end_stage(session_id=42, stage="strategist", session_date=date(2026, 9, 22), result_path=str(p),
                           transcript_path=None, tools_log_path=None, context_path=None, input_path=None, exit_code=0, error=None)
    assert out["error_class"] == "validator" and fail.called


def test_end_stage_executor_skip_exit_3_completes_with_note():
    with patch("v2.session_ctl.complete_session_stage") as complete:
        out = sc.end_stage(session_id=42, stage="executor", session_date=date(2026, 9, 22), result_path=None,
                           transcript_path=None, tools_log_path=None, context_path=None, input_path=None,
                           exit_code=3, error="market_closed")
    assert out["status"] == "completed" and complete.called


def test_end_stage_executor_writes_context_from_input_and_structured_output(tmp_path):
    r = {"is_error": False, "structured_output": {"decisions": []}, "stop_reason": "end_turn", "duration_api_ms": 5,
         "modelUsage": {"claude-haiku-4-5-20251001": {"inputTokens": 1, "outputTokens": 1}}}
    p = tmp_path / "r.json"; p.write_text(json.dumps(r))
    i = tmp_path / "i.json"; i.write_text(json.dumps({"executor_input": {"positions": []}}))
    with patch("v2.session_ctl.complete_session_stage"), patch("v2.session_ctl.record_event"), \
         patch("v2.session_ctl.insert_llm_call_context") as llm, patch("v2.session_ctl.load_prompt", return_value="EXEC"):
        sc.end_stage(session_id=42, stage="executor", session_date=date(2026, 9, 22), result_path=str(p),
                     transcript_path=None, tools_log_path=None, context_path=None, input_path=str(i), exit_code=0, error=None)
    kw = llm.call_args.kwargs
    assert kw["stage_name"] == "trading" and kw["purpose"] == "executor" and kw["system_prompt"] == "EXEC"
    assert json.loads(kw["messages"][0]["content"]) == {"positions": []}
    assert kw["response_content"] == [{"type": "text", "text": json.dumps({"decisions": []})}]


def test_stage_context_strategist_assembles_blocks():
    with patch("v2.session_ctl.build_attribution_constraints", return_value="ATTR"), \
         patch("v2.session_ctl.get_orphan_positions", return_value=[]), \
         patch("v2.session_ctl.build_formation_context", return_value="FORM"), \
         patch("v2.session_ctl._build_pre_seeded_context", return_value="SEED"), \
         patch("v2.session_ctl._build_orphan_block", return_value=("", "", 0)), \
         patch("v2.session_ctl.wl.get_open_items", return_value=[]):
        text = sc.stage_context("strategist", session_id=42, executor_result_path=None)
    assert text.index("ATTR") < text.index("FORM") < text.index("SEED")
    assert "write_playbook" in text and "SUPERVISOR WATCHLIST" in text


def test_stage_context_reflection_uses_executor_result(tmp_path):
    p = tmp_path / "exec.json"
    p.write_text(json.dumps({"decisions_made": 2, "trades_executed": 1, "trades_failed": 0, "total_buy_value": 100.0,
                             "total_sell_value": 0.0, "market_summary": "calm", "risk_assessment": "low", "errors": []}))
    with patch("v2.session_ctl.get_rule_gate_revalidation_candidates", return_value=[]), \
         patch("v2.session_ctl.wl.get_open_items", return_value=[]):
        text = sc.stage_context("strategy", session_id=42, executor_result_path=str(p))
    assert "Decisions made: 2" in text and "RULE REVALIDATION EVIDENCE" in text and "Writing a reflection memo" in text
```

And in `tests/v2/test_trader.py`:

```python
def test_cli_decisions_file_result_out(tmp_path):
    d = tmp_path / "d.json"; d.write_text('{"decisions": []}')
    i = tmp_path / "i.json"; i.write_text(json.dumps(_prepared()))
    o = tmp_path / "o.json"
    result = MagicMock(errors=[], decisions_made=1, trades_executed=1, trades_failed=0,
                       total_buy_value=Decimal("10"), total_sell_value=Decimal("0"), market_summary="m", risk_assessment="r")
    with patch("v2.trader.execute_decisions_payload", return_value=result), \
         patch("sys.argv", ["trader", "--decisions-file", str(d), "--input-file", str(i), "--result-out", str(o)]):
        assert trader.main() == 0
    assert json.loads(o.read_text())["total_buy_value"] == "10"
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_session_ctl.py -v -k "usage_from or classify or transcript or tools_log or validate or end_stage or stage_context"`
Expected: FAIL with `AttributeError: module 'v2.session_ctl' has no attribute 'usage_from_result'`

- [ ] **Step 3: Add `--result-out` to `trader.py`**

In `main()`, add `parser.add_argument("--result-out", metavar="PATH", help="Driver mode: write a summary of the trading result to PATH")` and, right after `result = execute_decisions_payload(...)`:

```python
        if args.result_out:
            with open(args.result_out, "w") as fh:
                json.dump({
                    "decisions_made": result.decisions_made, "trades_executed": result.trades_executed,
                    "trades_failed": result.trades_failed, "total_buy_value": result.total_buy_value,
                    "total_sell_value": result.total_sell_value, "market_summary": result.market_summary,
                    "risk_assessment": result.risk_assessment, "errors": list(result.errors),
                }, fh, default=str)
```

- [ ] **Step 4: Implement stage-context and stage-end in `v2/session_ctl.py`**

Add imports:

```python
import re
from datetime import date
from types import SimpleNamespace

from v2 import watchlist as wl

from .attribution import build_attribution_constraints
from .claude_client import UsageAccumulator
from .database.trading_db import (
    get_current_strategy_state, get_orphan_positions, get_playbook, get_rule_gate_revalidation_candidates,
    insert_llm_call_context, insert_strategy_memo,
)
from .formation import build_formation_context
from .ideation_claude import _build_orphan_block, _build_pre_seeded_context
from .mcp_server import registry_tool_definitions  # Task 8; until then, stub: def registry_tool_definitions(r): return []
from .session import STRATEGIST_MEMO_MIN_LENGTH
from .strategy import _format_rule_revalidation_context, _format_trading_context, _gated_rule_ids
from .telemetry import record_event
```

(Check where `get_orphan_positions` and `get_rule_gate_revalidation_candidates` are imported from in `v2/ideation_claude.py` and `v2/strategy.py` and import from the same module.) Until Task 8 lands, define `def registry_tool_definitions(registry: str) -> list[dict]: return []` at the top of `session_ctl.py` and replace it with the import in Task 8.

```python
CONTEXT_STAGE = {"supervisor": "supervisor", "pipeline": "pipeline", "strategist": "ideation", "executor": "trading",
                 "strategy": "reflection", "dashboard": "dashboard_publish"}
PURPOSE = {"strategist": "strategist_loop", "strategy": "reflection_loop", "executor": "executor",
           "supervisor": "supervisor_loop", "pipeline": "classifier_news", "dashboard": "changelog_summary"}
REGISTRY = {"strategist": "strategist", "strategy": "reflection", "supervisor": "supervisor"}
SKILL = {"strategist": "strategist", "strategy": "reflection", "supervisor": "supervisor"}
CONTEXT_LOGGED = {"strategist", "strategy", "executor"}


# --- stage-context ---------------------------------------------------------

def stage_context(stage: str, *, session_id: int | None, executor_result_path: str | None) -> str:
    if stage == "supervisor":
        return "Begin your strategy review. Investigate, cite IDs, then write the memo."
    if stage == "strategist":
        header = []
        constraints = build_attribution_constraints()
        if constraints:
            header.append(constraints)
        try:
            orphans = get_orphan_positions()
        except Exception:
            logger.exception("Failed to fetch orphan positions")
            orphans = []
        formation = build_formation_context(orphans=orphans)
        if formation:
            header.append(formation)
        pre_seeded = _build_pre_seeded_context()
        orphan_block, adopt_step, step_offset = _build_orphan_block(orphans=orphans)
        if orphan_block:
            pre_seeded += "\n\n" + orphan_block
        watchlist_block = wl.format_open_watchlist_items(wl.get_open_items("ideation"))
        parts = []
        if header:
            parts.append("=== SESSION CONTEXT (appended to your instructions) ===\n" + "\n\n".join(header))
        parts.append(f"""Here is the current state (pre-loaded to save round-trips):

{pre_seeded}

{watchlist_block}

Now proceed with your strategist session:
1. Review the data above — do NOT re-fetch portfolio, theses, decisions, attribution, identity, rules, or strategy history
{adopt_step}{2 + step_offset}. Explore market conditions for new opportunities (use WebSearch, get_market_snapshot, get_news_signals)
{3 + step_offset}. Update or close stale theses, create 2-4 new ones
{4 + step_offset}. Write today's playbook using the write_playbook tool

When you've completed your work, provide a summary of your findings and actions.""")
        return "\n\n".join(parts)
    if stage == "strategy":
        trading_result = None
        if executor_result_path and os.path.exists(executor_result_path):
            with open(executor_result_path) as fh:
                trading_result = SimpleNamespace(**json.load(fh))
        try:
            candidates = get_rule_gate_revalidation_candidates(days=30)
        except Exception as e:
            logger.warning("Could not load rule revalidation evidence: %s", e)
            candidates = []
        parts = []
        trading_context = _format_trading_context(trading_result)
        if trading_context:
            parts += [trading_context, ""]
        parts += [_format_rule_revalidation_context(candidates), "",
                  wl.format_open_watchlist_items(wl.get_open_items("reflection")), "",
                  "Begin your strategy reflection. Start by:\n"
                  "1. Getting the current strategy identity and rules\n"
                  "2. Getting the session summary (recent decisions and attribution)\n"
                  "3. Getting recent strategy memos for context\n"
                  "4. Analyzing what happened and making any necessary updates\n"
                  "5. Writing a reflection memo\n"]
        formation = build_formation_context()
        if formation:
            parts.insert(0, "=== SESSION CONTEXT (appended to your instructions) ===\n" + formation + "\n")
        return "\n".join(parts)
    raise SystemExit(f"stage-context: no context for stage {stage!r}")


# --- stage-end helpers -----------------------------------------------------

def usage_from_result(result: dict | None) -> UsageAccumulator | None:
    """Build the accumulator session_stages expects from claude -p's modelUsage."""
    if not result or not result.get("modelUsage"):
        return None
    acc = UsageAccumulator()
    ranked = sorted(result["modelUsage"].items(),
                    key=lambda kv: -((kv[1].get("inputTokens") or 0) + (kv[1].get("cacheReadInputTokens") or 0)))
    for model, u in ranked:
        acc.add(model, SimpleNamespace(
            input_tokens=u.get("inputTokens") or 0, output_tokens=u.get("outputTokens") or 0,
            cache_creation_input_tokens=u.get("cacheCreationInputTokens") or 0,
            cache_read_input_tokens=u.get("cacheReadInputTokens") or 0,
        ))
    return acc


_USAGE_LIMIT_RE = re.compile(r"usage limit|rate limit|429|overloaded", re.I)
_AUTH_RE = re.compile(r"not logged in|/login|authenticat|credential|oauth", re.I)


def classify_failure(*, exit_code: int | None, result: dict | None, error: str | None) -> tuple[str, str] | None:
    """(class, message) for a failed stage, or None when nothing failed.

    Classes: timeout, usage_limit, auth, model_error, command, validator (the
    last is assigned by end_stage). The alert text carries the class so a
    subscription-window hit reads as one, not as a generic model error.
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


def _tool_ok(tools: list[dict], name: str, prefix: str) -> bool:
    return any(t.get("tool_name", "").endswith(name) and t.get("success") and str(t.get("result_prefix", "")).startswith(prefix)
               for t in tools)


def validate_stage(stage: str, session_id: int | None, session_date: date, result: dict | None, tools: list[dict]) -> str | None:
    """The post-loop checks run_session / the stage modules enforce today."""
    if stage == "strategist":
        if get_playbook(session_date) is None:
            return f"Strategist finished without writing a playbook for {session_date}"
        return None
    if stage == "strategy":
        if not _tool_ok(tools, "write_strategy_memo", "Memo written"):
            return "Reflection finished without writing a memo (write_strategy_memo not called successfully)"
        revalidated = {int(m.group(1)) for t in tools if t.get("success")
                       for m in [re.match(r"Revalidated rule ID (\d+)", str(t.get("result_prefix", "")))] if m}
        gated = _gated_rule_ids(get_rule_gate_revalidation_candidates(days=30))
        missing = sorted(gated - revalidated)
        if missing:
            return "Reflection finished without revalidating gated rule(s): " + ", ".join(map(str, missing))
        try:
            wl.assert_watchlist_resolved("reflection")
        except RuntimeError as e:
            return str(e)
        return None
    if stage == "supervisor":
        if not _tool_ok(tools, "write_supervisor_memo", "Supervisor memo written"):
            return "Supervisor finished without calling write_supervisor_memo"
        return None
    return None


def _persist_strategist_memo(session_id: int | None, session_date: date, summary: str | None) -> None:
    summary = (summary or "").strip()
    if len(summary) < STRATEGIST_MEMO_MIN_LENGTH:
        logger.warning("Strategist memo skipped: final summary too short (%d chars)", len(summary))
        return
    try:
        state = get_current_strategy_state()
        insert_strategy_memo(session_date=session_date, memo_type="strategist_notes", content=summary,
                             strategy_state_id=state["id"] if state else None, session_id=session_id)
    except Exception as e:
        logger.warning("Could not save strategist memo: %s", e)


def _write_telemetry(*, session_id: int, stage: str, result: dict | None, tools: list[dict],
                     transcript_path: str | None, context_path: str | None, input_path: str | None, usage) -> None:
    if session_id is None or result is None:
        return
    context_stage, purpose = CONTEXT_STAGE[stage], PURPOSE[stage]
    model = usage.model if usage else None
    record_event(session_id=session_id, stage_name=context_stage, event_type="agent_call", payload={
        "purpose": purpose, "model": model, "stop_reason": result.get("stop_reason"),
        "duration_ms": result.get("duration_api_ms"), "success": not result.get("is_error"),
        "input_tokens": usage.input_tokens if usage else None, "output_tokens": usage.output_tokens if usage else None,
        "backend": "claude_code",
    })
    for t in tools:
        record_event(session_id=session_id, stage_name=context_stage, event_type="tool_invocation", payload={
            "tool_name": t.get("tool_name", "").split("__")[-1], "args": t.get("args"), "success": t.get("success"),
            "error": t.get("error"), "duration_ms": t.get("duration_ms"), "output_chars": t.get("output_chars"),
        })
    if stage in REGISTRY:
        record_event(session_id=session_id, stage_name=context_stage, event_type="loop_completion", payload={
            "stop_reason": result.get("stop_reason"), "turns_used": len(tools), "model": model,
            "input_tokens": usage.input_tokens if usage else None, "output_tokens": usage.output_tokens if usage else None,
            "cache_creation_input_tokens": usage.cache_creation_tokens if usage else None,
            "cache_read_input_tokens": usage.cache_read_tokens if usage else None,
        })
    if stage not in CONTEXT_LOGGED:
        return
    if stage == "executor":
        with open(input_path) as fh:
            executor_input = json.load(fh).get("executor_input")
        messages = [{"role": "user", "content": json.dumps(executor_input, default=str)}]
        response = [{"type": "text", "text": json.dumps(result.get("structured_output"))}]
        system_prompt, tool_defs = load_prompt("executor"), None
    else:
        lines = open(transcript_path).read().splitlines() if transcript_path and os.path.exists(transcript_path) else []
        initial = open(context_path).read() if context_path and os.path.exists(context_path) else None
        messages, response, _ = messages_from_transcript(lines, initial)
        system_prompt, tool_defs = load_skill_prompt(SKILL[stage]), registry_tool_definitions(REGISTRY[stage])
    try:
        insert_llm_call_context(
            session_id=session_id, stage_name=context_stage, purpose=purpose, model=model or "unknown",
            system_prompt=system_prompt, messages=messages, tool_definitions=tool_defs, response_content=response,
            input_tokens=usage.input_tokens if usage else None, output_tokens=usage.output_tokens if usage else None,
            cache_read_tokens=usage.cache_read_tokens if usage else None,
            cache_creation_tokens=usage.cache_creation_tokens if usage else None,
            stop_reason=result.get("stop_reason"), duration_ms=result.get("duration_api_ms"),
        )
    except Exception as e:
        logger.warning("Could not record llm_call_context for %s: %s", stage, e)


def end_stage(*, session_id: int, stage: str, session_date: date, result_path: str | None, transcript_path: str | None,
              tools_log_path: str | None, context_path: str | None, input_path: str | None,
              exit_code: int | None, error: str | None) -> dict:
    result = None
    if result_path and os.path.exists(result_path):
        with open(result_path) as fh:
            result = json.load(fh)
    usage = usage_from_result(result)
    tools = read_tools_log(tools_log_path)

    if stage == "executor" and exit_code == 3:
        # prepare_executor_stage decided not to call the model (market closed,
        # breaker, no account). Same outcome as today's early return.
        complete_session_stage(session_id, stage, usage=None)
        logger.warning("Executor stage skipped: %s", error)
        return {"status": "completed", "error": None, "error_class": None}

    failure = classify_failure(exit_code=exit_code, result=result, error=error)
    if failure is None:
        verr = validate_stage(stage, session_id, session_date, result, tools)
        if verr:
            failure = ("validator", verr)
    _write_telemetry(session_id=session_id, stage=stage, result=result, tools=tools, transcript_path=transcript_path,
                     context_path=context_path, input_path=input_path, usage=usage)
    if failure:
        cls, msg = failure
        text = f"[{cls}] {msg}"
        fail_session_stage(session_id, stage, text, usage=usage)
        logger.error("Stage %s failed: %s", stage, text)
        return {"status": "failed", "error": text, "error_class": cls}
    if stage == "strategist":
        _persist_strategist_memo(session_id, session_date, (result or {}).get("result"))
    complete_session_stage(session_id, stage, usage=usage)
    return {"status": "completed", "error": None, "error_class": None}
```

Replace the three Task 6 stubs:

```python
def _add_stage_context_parser(sub):
    p = sub.add_parser("stage-context")
    p.add_argument("--stage", required=True, choices=["strategist", "strategy", "supervisor"])
    p.add_argument("--session-id", type=int, default=None)
    p.add_argument("--executor-result", default=None)


def _add_stage_end_parser(sub):
    p = sub.add_parser("stage-end")
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--session-date", default=None)
    for opt in ("--result", "--transcript", "--tools-log", "--context", "--input"):
        p.add_argument(opt, default=None)
    p.add_argument("--exit-code", type=int, default=None)
    p.add_argument("--error", default=None)


def _run_task7_commands(args) -> int:
    if args.cmd == "stage-context":
        print(stage_context(args.stage, session_id=args.session_id, executor_result_path=args.executor_result))
        return 0
    if args.cmd == "stage-end":
        session_date = date.fromisoformat(args.session_date) if args.session_date else current_market_date()
        _emit(end_stage(session_id=args.session_id, stage=args.stage, session_date=session_date, result_path=args.result,
                        transcript_path=args.transcript, tools_log_path=args.tools_log, context_path=args.context,
                        input_path=args.input, exit_code=args.exit_code, error=args.error or None))
        return 0
    raise SystemExit(f"unknown command {args.cmd}")
```

Add `"v2.session_ctl.insert_llm_call_context"`, `"v2.session_ctl.record_event"`, `"v2.session_ctl.insert_strategy_memo"`, `"v2.session_ctl.get_current_strategy_state"`, `"v2.session_ctl.get_playbook"`, `"v2.session_ctl.get_rule_gate_revalidation_candidates"` to `_SESSION_CTL_DB_PATCH_TARGETS` in `tests/v2/conftest.py`.

- [ ] **Step 5: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_session_ctl.py tests/v2/test_trader.py tests/test_conftest_gate.py -v && task lint`
Expected: PASS; lint clean.

- [ ] **Step 6: Commit**

```bash
git add v2/session_ctl.py v2/trader.py tests/v2/test_session_ctl.py tests/v2/test_trader.py tests/v2/conftest.py
git commit -m "session_ctl: stage-context and stage-end with validators, usage and telemetry

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 8: `v2/mcp_server.py` — stdio MCP server over the existing registries

**Files:**
- Create: `v2/mcp_server.py`
- Modify: `v2/requirements.txt` (add `mcp>=1.2`), `v2/supervisor.py` (add `WRITE_SUPERVISOR_MEMO_TOOL_DEF`, `make_write_supervisor_memo_handler`), `v2/session_ctl.py` (swap the `registry_tool_definitions` stub for the import)
- Test: `tests/v2/test_mcp_server.py`, `tests/v2/test_supervisor.py`

**Interfaces:**
- CLI: `python -m v2.mcp_server --registry {strategist,reflection,supervisor} [--session-id N] [--max-tool-calls N] [--tools-log PATH] [--model NAME]` — speaks MCP over stdio until stdin closes.
- Python API:
  - `registry_tool_definitions(registry: str) -> list[dict]` — Anthropic-format definitions served for that registry (used by `session_ctl` for `llm_call_contexts.tool_definitions`).
  - `build_registry(registry: str, *, session_id: int | None, model: str) -> tuple[list[dict], dict[str, Callable]]` — definitions + bound handlers, same binding as `ideation_claude._run_claude_loop`, `strategy.run_strategy_reflection`, `supervisor.run_supervisor`.
  - `ToolRunner(handlers, *, max_tool_calls: int | None, tools_log: str | None)` with `.call(name: str, arguments: dict) -> tuple[str, bool]` returning `(text, is_error)`; appends one `tools.jsonl` line per call (format in Task 7); after `max_tool_calls` successful-or-not calls, every further call returns `("Error: tool-call cap reached (N). Finish now: call <terminal tool> if you have not, then stop.", True)` **except** the terminal tool itself (`write_playbook`, `write_strategy_memo`, `write_supervisor_memo`, `resolve_watchlist_item`), which still runs.
  - Tool names are served unprefixed (`get_theses`); Claude Code exposes them as `mcp__pinchy__get_theses`.
- Supervisor additions in `v2/supervisor.py`:
  - `WRITE_SUPERVISOR_MEMO_TOOL_DEF` — `{"name": "write_supervisor_memo", "input_schema": {"properties": {"content": {"type": "string"}}, "required": ["content"]}}`
  - `make_write_supervisor_memo_handler(buffer: list[dict], *, model: str) -> Callable[[str], str]` — inserts the memo via `_insert_memo(model=model, content=content, status="ok", turns_used=0, tool_calls=[], input_tokens=None, output_tokens=None, cost_usd=None, error_message=None)`, flushes `buffer` through `db.record_watchlist_item(source_memo_id=memo_id, ...)`, returns `f"Supervisor memo written (ID: {memo_id})"`. A second call returns `"Error: memo already written (ID: N)"`.
- `STRATEGY_MUTATOR_NAMES` stays enforced: the supervisor registry must not include any of them (existing test).

- [ ] **Step 1: Write the failing supervisor tests**

```python
# tests/v2/test_supervisor.py — append
def test_write_supervisor_memo_handler_inserts_and_flushes_watchlist():
    buffer = [{"title": "t", "detail": "d", "owner_stage": "ideation"}]
    with patch("v2.supervisor._insert_memo", return_value=77) as ins, \
         patch("v2.supervisor.db.record_watchlist_item") as rec:
        handler = sup.make_write_supervisor_memo_handler(buffer, model="claude-fable-5")
        assert handler("## Memo") == "Supervisor memo written (ID: 77)"
        assert handler("again").startswith("Error: memo already written (ID: 77)")
    assert ins.call_args.kwargs["status"] == "ok" and ins.call_args.kwargs["content"] == "## Memo"
    rec.assert_called_once_with(source_memo_id=77, title="t", detail="d", owner_stage="ideation")
    assert buffer == []


def test_write_supervisor_memo_tool_def_shape():
    assert sup.WRITE_SUPERVISOR_MEMO_TOOL_DEF["name"] == "write_supervisor_memo"
    assert sup.WRITE_SUPERVISOR_MEMO_TOOL_DEF["input_schema"]["required"] == ["content"]
```

- [ ] **Step 2: Write the failing MCP server tests**

```python
# tests/v2/test_mcp_server.py
"""The MCP server must serve exactly the registries the API loops use, bind
session ids the same way, cap tool calls, and log every call for stage-end."""
import json
from unittest.mock import patch

import pytest

from v2 import mcp_server as ms
from v2 import strategy, supervisor, tools
from v2 import watchlist as wl


def _names(defs):
    return sorted(d["name"] for d in defs)


def test_strategist_registry_matches_api_loop_minus_web_search():
    defs, handlers = ms.build_registry("strategist", session_id=1, model="m")
    expected = _names(tools.TOOL_DEFINITIONS + [wl.RESOLVE_WATCHLIST_TOOL_DEF])
    expected.remove("web_search")
    assert _names(defs) == expected and sorted(handlers) == expected


def test_reflection_registry_matches_api_loop():
    defs, handlers = ms.build_registry("reflection", session_id=1, model="m")
    assert _names(defs) == _names(strategy.STRATEGY_TOOL_DEFINITIONS) and sorted(handlers) == _names(defs)


def test_supervisor_registry_adds_memo_tool_and_has_no_mutators():
    defs, handlers = ms.build_registry("supervisor", session_id=None, model="m")
    assert _names(defs) == sorted(_names(supervisor.build_supervisor_tool_defs()) + ["write_supervisor_memo"])
    assert not (set(handlers) & supervisor.STRATEGY_MUTATOR_NAMES)


def test_session_bound_handlers():
    _, handlers = ms.build_registry("strategist", session_id=42, model="m")
    with patch("v2.tools.tool_create_thesis") as create:
        ms.build_registry("strategist", session_id=42, model="m")[1]["create_thesis"](ticker="AAPL")
    assert create.call_args.kwargs["session_id"] == 42
    _, handlers = ms.build_registry("reflection", session_id=42, model="m")
    with patch("v2.strategy.tool_write_strategy_memo") as memo:
        handlers["write_strategy_memo"](memo_type="reflection", content="c")
    assert memo.call_args.kwargs["session_id"] == 42


def test_registry_tool_definitions_is_anthropic_format():
    defs = ms.registry_tool_definitions("reflection")
    assert all("input_schema" in d for d in defs)


def test_mcp_tools_convert_input_schema():
    defs, _ = ms.build_registry("supervisor", session_id=None, model="m")
    mcp_tools = ms.to_mcp_tools(defs)
    assert mcp_tools[0].inputSchema == defs[0]["input_schema"]
    assert {t.name for t in mcp_tools} == {d["name"] for d in defs}


def test_tool_runner_logs_calls_and_stringifies(tmp_path):
    log = tmp_path / "tools.jsonl"
    runner = ms.ToolRunner({"echo": lambda **kw: {"got": kw}, "boom": lambda **kw: 1 / 0},
                           max_tool_calls=None, tools_log=str(log), terminal_tools=set())
    assert runner.call("echo", {"a": 1}) == (str({"got": {"a": 1}}), False)
    text, err = runner.call("boom", {})
    assert err and text.startswith("Error: division by zero")
    assert runner.call("nope", {}) == ("Error: Unknown tool 'nope'", True)
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["tool_name"] for line in lines] == ["echo", "boom", "nope"]
    assert lines[0]["success"] and lines[0]["result_prefix"].startswith("{'got'") and lines[0]["output_chars"] > 0
    assert lines[1]["success"] is False and "division" in lines[1]["error"]
    assert set(lines[0]) == {"tool_name", "args", "success", "error", "duration_ms", "output_chars", "result_prefix"}


def test_tool_runner_cap_blocks_all_but_terminal_tools(tmp_path):
    runner = ms.ToolRunner({"read": lambda **kw: "r", "write_playbook": lambda **kw: "Playbook written"},
                           max_tool_calls=2, tools_log=None, terminal_tools={"write_playbook"})
    runner.call("read", {}); runner.call("read", {})
    text, err = runner.call("read", {})
    assert err and "tool-call cap reached (2)" in text and "write_playbook" in text
    assert runner.call("write_playbook", {}) == ("Playbook written", False)


def test_cli_args_parse_and_main_builds_runner():
    with patch("v2.mcp_server.serve") as serve, \
         patch("sys.argv", ["mcp_server", "--registry", "reflection", "--session-id", "9", "--max-tool-calls", "5", "--tools-log", "/tmp/x"]):
        ms.main()
    kw = serve.call_args.kwargs
    assert kw["runner"].max_tool_calls == 5 and "write_strategy_memo" in kw["runner"].terminal_tools
    assert {t.name for t in kw["mcp_tools"]} >= {"write_strategy_memo", "resolve_watchlist_item"}
```

- [ ] **Step 3: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_mcp_server.py tests/v2/test_supervisor.py -v -k "mcp or write_supervisor_memo"`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2.mcp_server'` (and `AttributeError` for the supervisor handler).

- [ ] **Step 4: Add the dependency and rebuild**

Append `mcp>=1.2` to `v2/requirements.txt`, then `task build INSTANCE=paper && task up INSTANCE=paper`. Verify: `docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading python -c "import mcp.server, mcp.types; print('ok')"`.

- [ ] **Step 5: Supervisor memo tool**

```python
# v2/supervisor.py — after RECORD_WATCHLIST_TOOL_DEF / _make_record_watchlist_handler
WRITE_SUPERVISOR_MEMO_TOOL_DEF: dict = {
    "name": "write_supervisor_memo",
    "description": (
        "Write the final supervisor memo (markdown, with the Watchlist section). Call this exactly once, "
        "after every record_watchlist_item call. It persists the memo and the recorded watchlist items."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"content": {"type": "string", "description": "The full memo in markdown."}},
        "required": ["content"],
    },
}


def make_write_supervisor_memo_handler(buffer: list[dict], *, model: str):
    """Driver-mode terminal tool: insert the memo row, then flush the buffered
    watchlist items against its id (source_memo_id is NOT NULL). Mirrors the
    tail of run_supervisor for the Claude Code path, where the loop's final
    text is not available to Python."""
    state = {"memo_id": None}

    def _handler(content: str) -> str:
        if state["memo_id"] is not None:
            return f"Error: memo already written (ID: {state['memo_id']}); do not call again"
        memo_id = _insert_memo(
            model=model, content=content, status="ok", turns_used=0, tool_calls=[],
            input_tokens=None, output_tokens=None, cost_usd=None, error_message=None,
        )
        state["memo_id"] = memo_id
        while buffer:
            it = buffer.pop(0)
            try:
                db.record_watchlist_item(source_memo_id=memo_id, title=it["title"], detail=it["detail"], owner_stage=it["owner_stage"])
            except Exception:
                log.exception("Failed to persist watchlist item %r", it.get("title"))
        return f"Supervisor memo written (ID: {memo_id})"
    return _handler
```

- [ ] **Step 6: Implement `v2/mcp_server.py`**

```python
# v2/mcp_server.py
"""stdio MCP server that exposes the session's tool registries to Claude Code.

Started by the driver via --mcp-config as
  docker compose -p pinchy-<i> --env-file instances/<i>.env exec -i trading \
      python -m v2.mcp_server --registry strategist --session-id N ...
It serves the same definitions and bound handlers the API loops use
(ideation_claude, strategy, supervisor), counts tool calls for the turn cap
that replaces --max-turns, and appends every call to tools.jsonl for
session_ctl stage-end.
"""
import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from functools import partial

from v2 import watchlist as wl

from . import strategy, supervisor, tools
from .tools import reset_session

logger = logging.getLogger("mcp_server")

TERMINAL_TOOLS = {
    "strategist": {"write_playbook", "resolve_watchlist_item"},
    "reflection": {"write_strategy_memo", "resolve_watchlist_item"},
    "supervisor": {"write_supervisor_memo"},
}


def build_registry(registry: str, *, session_id: int | None, model: str) -> tuple[list[dict], dict[str, Callable]]:
    """Definitions + handlers with the same session binding as the API loops."""
    if registry == "strategist":
        defs = [d for d in tools.TOOL_DEFINITIONS if d.get("name") != "web_search"] + [wl.RESOLVE_WATCHLIST_TOOL_DEF]
        handlers = {
            **tools.TOOL_HANDLERS,
            "create_thesis": partial(tools.tool_create_thesis, session_id=session_id),
            "adopt_thesis": partial(tools.tool_adopt_thesis, session_id=session_id),
            "resolve_watchlist_item": wl.make_resolve_handler(session_id=session_id, stage="ideation"),
        }
    elif registry == "reflection":
        defs = list(strategy.STRATEGY_TOOL_DEFINITIONS)
        handlers = {
            **strategy.STRATEGY_TOOL_HANDLERS,
            "get_session_summary": partial(strategy.tool_get_session_summary_with_telemetry, session_id=session_id),
            "write_strategy_memo": partial(strategy.tool_write_strategy_memo, session_id=session_id),
            "resolve_watchlist_item": wl.make_resolve_handler(session_id=session_id, stage="reflection"),
        }
    elif registry == "supervisor":
        buffer: list[dict] = []
        defs = supervisor.build_supervisor_tool_defs() + [supervisor.WRITE_SUPERVISOR_MEMO_TOOL_DEF]
        handlers = {
            **supervisor.SUPERVISOR_TOOL_HANDLERS,
            "record_watchlist_item": supervisor._make_record_watchlist_handler(buffer),
            "write_supervisor_memo": supervisor.make_write_supervisor_memo_handler(buffer, model=model),
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
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types

    server = Server("pinchy")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return mcp_tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        text, is_error = await asyncio.to_thread(runner.call, name, arguments or {})
        return [types.TextContent(type="text", text=text)]

    async def _run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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

Note on `is_error`: MCP `call_tool` results carry an `isError` flag; with the decorator form above the server returns content only. That is acceptable: the error text starts with `Error:` exactly as the API loop's tool results do, and the model reads it the same way. Do not spend time on `isError` unless the SDK version exposes it trivially.

Logging goes to stderr only (stdout is the MCP transport). Confirm every `print` in imported modules is absent or stderr-bound: `grep -n "print(" v2/tools.py v2/strategy.py v2/supervisor.py v2/watchlist.py` — `_print_cost_summary` in `ideation_claude.py` is not imported here; any `print` found in the served modules must be changed to `logger.info`.

In `v2/session_ctl.py`, replace the `registry_tool_definitions` stub with `from .mcp_server import registry_tool_definitions`.

- [ ] **Step 7: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_mcp_server.py tests/v2/test_supervisor.py tests/v2/test_session_ctl.py -v && task lint`
Expected: PASS; lint clean.

- [ ] **Step 8: Manual stdio smoke (paper)**

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading python -m v2.mcp_server --registry supervisor 2>/dev/null | head -c 600
```

Expected: a JSON-RPC `initialize` result followed by a `tools/list` result naming `get_active_rules` … `write_supervisor_memo`.

- [ ] **Step 9: Commit**

```bash
git add v2/mcp_server.py v2/supervisor.py v2/session_ctl.py v2/requirements.txt tests/v2/test_mcp_server.py tests/v2/test_supervisor.py
git commit -m "Add the pinchy MCP server: existing tool registries over stdio, call cap, tools.jsonl

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 9: `session-driver.sh`, `scripts/driver_json.py`, smoke script, driver test

**Files:**
- Create: `session-driver.sh`, `scripts/driver_json.py`, `scripts/smoke-claude-headless.sh`
- Create: `tests/test_driver_json.py`, `tests/test_session_driver.py`
- Modify: `Taskfile.yml` (add `session:claude`, `driver:smoke` targets)

**Interfaces:**
- `./session-driver.sh <instance> [--force] [--dry-run] [--only STAGE] [--skip-supervisor|--skip-pipeline|--skip-ideation|--skip-executor|--skip-strategy|--skip-dashboard] [--pipeline-hours N] [--pipeline-limit N]` → exit code from `session_ctl finalize` (0 clean, 1 any stage failed, 0 on plan skip). Also `./session-driver.sh mcp-config <instance> <registry> [--session-id N] [--max-tool-calls N] [--tools-log PATH] [--model M]` prints the `--mcp-config` JSON.
- Environment: `PINCHY_CLAUDE_BIN` (default `claude`) and `PINCHY_DOCKER_BIN` (default `docker`) override the binaries so the test can stub them. `LOGS_DIR` and `INSTANCE` come from `instances/<name>.env`.
- Per-stage artefacts on the host under `$LOGS_DIR/sessions/<date>/<stage>/`: `context.md`, `transcript.jsonl`, `result.json`, `tools.jsonl`, `out.json`, `result-out.json`, `batches/`, `stderr.log`. The same directory is `/app/logs/sessions/<date>/<stage>/` inside the container (LOGS_DIR is bind-mounted at `/app/logs`).
- `scripts/driver_json.py` (stdlib only, runs on host python 3.10): `field FILE KEY` (prints the value; strings raw, other types as JSON, empty for null/missing), `stages FILE` (one compact JSON object per line), `last-result TRANSCRIPT OUT` (writes the last `"type": "result"` line to OUT; exit 1 if none), `merge-results OUT FILE...` (sums `modelUsage` and `duration_api_ms`, `is_error` false, `stop_reason` `end_turn`).
- Skill per agentic stage: `strategist → pinchy-strategist`, `strategy → pinchy-reflection`, `supervisor → pinchy-supervisor`.
- The `claude` invocation for structured stages must include `--system-prompt`, `--json-schema`, `--output-format json`, `--permission-mode dontAsk`, `--permission-prompts none`, `--tools ""`, and never `--bare`. For agentic stages: the skill as the prompt, `--output-format stream-json --verbose`, `--allowedTools`, `--strict-mcp-config --mcp-config`, `--disallowedTools "AskUserQuestion,Edit,Write,NotebookEdit,Agent,Bash"`, and never `--bare`.

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


def test_field_prints_raw_strings_and_json_for_others(tmp_path):
    f = tmp_path / "x.json"; f.write_text(json.dumps({"a": "s", "b": [1], "c": None}))
    assert run("field", str(f), "a").stdout == "s\n"
    assert run("field", str(f), "b").stdout == "[1]\n"
    assert run("field", str(f), "c").stdout == "\n" and run("field", str(f), "zz").stdout == "\n"


def test_stages_one_line_each(tmp_path):
    f = tmp_path / "p.json"; f.write_text(json.dumps({"stages": [{"name": "a"}, {"name": "b"}]}))
    lines = run("stages", str(f)).stdout.splitlines()
    assert [json.loads(line)["name"] for line in lines] == ["a", "b"]


def test_last_result_extracts_result_line(tmp_path):
    t = tmp_path / "t.jsonl"; o = tmp_path / "r.json"
    t.write_text('{"type":"assistant"}\n{"type":"result","is_error":false,"result":"x"}\nnot json\n')
    assert run("last-result", str(t), str(o)).returncode == 0
    assert json.loads(o.read_text())["result"] == "x"
    t.write_text('{"type":"assistant"}\n')
    assert run("last-result", str(t), str(o)).returncode == 1


def test_merge_results_sums_usage(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"; o = tmp_path / "o.json"
    a.write_text(json.dumps({"duration_api_ms": 5, "modelUsage": {"m": {"inputTokens": 1, "outputTokens": 2}}}))
    b.write_text(json.dumps({"is_error": True, "duration_api_ms": 7, "modelUsage": {"m": {"inputTokens": 10, "cacheReadInputTokens": 3}}}))
    assert run("merge-results", str(o), str(a), str(b)).returncode == 0
    m = json.loads(o.read_text())
    assert m["is_error"] is False and m["duration_api_ms"] == 12
    assert m["modelUsage"]["m"] == {"inputTokens": 11, "outputTokens": 2, "cacheCreationInputTokens": 0, "cacheReadInputTokens": 3}
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/test_driver_json.py -v`
Expected: FAIL (helper missing; `returncode` 2 / stdout empty).

- [ ] **Step 3: Write `scripts/driver_json.py`**

```python
#!/usr/bin/env python3
"""JSON helpers for session-driver.sh. Stdlib only — runs on the host's
python3 (3.10) as well as in the container. Keeps jq out of the host deps."""
import json
import sys

USAGE_KEYS = ("inputTokens", "outputTokens", "cacheCreationInputTokens", "cacheReadInputTokens")


def cmd_field(path, key):
    with open(path) as fh:
        value = json.load(fh).get(key)
    if value is None:
        print("")
    elif isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value))


def cmd_stages(path):
    with open(path) as fh:
        for stage in json.load(fh).get("stages", []):
            print(json.dumps(stage, separators=(",", ":")))


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


def cmd_merge_results(out, *files):
    usage, duration = {}, 0
    for path in files:
        with open(path) as fh:
            r = json.load(fh)
        duration += int(r.get("duration_api_ms") or 0)
        for model, u in (r.get("modelUsage") or {}).items():
            acc = usage.setdefault(model, dict.fromkeys(USAGE_KEYS, 0))
            for k in USAGE_KEYS:
                acc[k] += int(u.get(k) or 0)
    with open(out, "w") as fh:
        json.dump({"is_error": False, "stop_reason": "end_turn", "duration_api_ms": duration, "modelUsage": usage}, fh)
    return 0


def main(argv):
    cmd, args = argv[1], argv[2:]
    if cmd == "field":
        return cmd_field(*args) or 0
    if cmd == "stages":
        return cmd_stages(*args) or 0
    if cmd == "last-result":
        return cmd_last_result(*args)
    if cmd == "merge-results":
        return cmd_merge_results(*args)
    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Write the failing driver test**

The test stubs `docker` and `claude` with shell scripts on `PATH`-independent env overrides, runs the driver from a temp copy of the repo layout, and asserts the calls.

```python
# tests/test_session_driver.py
"""session-driver.sh is a dumb loop: these tests stub `docker` and `claude`
and assert it sequences plan → stages → finalize exactly as session_ctl says."""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "session-driver.sh"

PLAN = {
    "session_id": 42, "session_date": "2026-09-22", "skip_reason": None, "dry_run": False,
    "stages": [
        {"name": "learning", "kind": "python", "skip": False, "skip_reason": None, "model": None, "context_stage": None,
         "registry": None, "max_tool_calls": None, "timeout_seconds": 900, "allowed_tools": None, "prompt": None,
         "schema": None, "command": "python -m v2.session_ctl run-learning", "emit": None, "consume": None},
        {"name": "supervisor", "kind": "agentic", "skip": True, "skip_reason": "--skip-supervisor", "model": "claude-fable-5",
         "context_stage": "supervisor", "registry": "supervisor", "max_tool_calls": 40, "timeout_seconds": 1200,
         "allowed_tools": "mcp__pinchy__*", "prompt": None, "schema": None, "command": None, "emit": None, "consume": None},
        {"name": "strategist", "kind": "agentic", "skip": False, "skip_reason": None, "model": "claude-opus-4-8",
         "context_stage": "ideation", "registry": "strategist", "max_tool_calls": 60, "timeout_seconds": 2400,
         "allowed_tools": "mcp__pinchy__*,WebSearch", "prompt": None, "schema": None, "command": None, "emit": None, "consume": None},
        {"name": "executor", "kind": "structured", "skip": False, "skip_reason": None, "model": "claude-haiku-4-5-20251001",
         "context_stage": "trading", "registry": None, "max_tool_calls": None, "timeout_seconds": 300, "allowed_tools": None,
         "prompt": "executor", "schema": "executor", "command": None,
         "emit": "python -m v2.trader --emit-input {out} --session-id {session_id}{dry_run}",
         "consume": "python -m v2.trader --decisions-file {file} --input-file {out} --session-id {session_id}{dry_run}"},
    ],
}

DOCKER_STUB = r'''#!/bin/bash
# Fake `docker compose ... exec -T trading <cmd...>`: log the command, answer session_ctl calls.
LOG="$STUB_DIR/docker.log"
args=("$@")
# drop everything up to and including "trading"
while [ $# -gt 0 ] && [ "$1" != "trading" ]; do shift; done
shift
echo "$*" >> "$LOG"
case "$*" in
  *"session_ctl plan"*) cat "$STUB_DIR/plan.json" ;;
  *"session_ctl stage-begin"*) echo '{"ok": true}' ;;
  *"session_ctl stage-context"*) echo "CONTEXT FOR $*" ;;
  *"session_ctl prompt"*) echo "SYSTEM PROMPT" ;;
  *"session_ctl schema"*) echo '{"type":"object"}' ;;
  *"session_ctl stage-end"*) echo '{"status": "completed", "error": null, "error_class": null}' ;;
  *"session_ctl finalize"*) cat "$STUB_DIR/finalize.json" ;;
  *"session_ctl run-learning"*) echo '{"ok": true}' ;;
  *"trader --emit-input"*)
     out=$(echo "$*" | sed -E 's/.*--emit-input ([^ ]+).*/\1/'); host="${out/#\/app\/logs/$LOGS_DIR}"
     mkdir -p "$(dirname "$host")"; echo '{"skipped": null, "executor_input": {"positions": []}, "session_date": "2026-09-22"}' > "$host"
     exit "${EMIT_EXIT:-0}" ;;
  *"trader --decisions-file"*) echo "consumed"; exit 0 ;;
  *"up -d"*) : ;;
esac
exit 0
'''

CLAUDE_STUB = r'''#!/bin/bash
LOG="$STUB_DIR/claude.log"
printf '%s\n' "$*" >> "$LOG"
cat > "$STUB_DIR/claude.stdin.$(date +%s%N)"
if [ -n "${CLAUDE_FAIL:-}" ]; then echo '{"type":"result","is_error":true,"result":"You have hit your usage limit"}'; exit 1; fi
case "$*" in
  *"stream-json"*) echo '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]}}'
                   echo '{"type":"result","is_error":false,"result":"Playbook written and theses reviewed in detail today.","modelUsage":{"claude-opus-4-8":{"inputTokens":1,"outputTokens":1}}}' ;;
  *) echo '{"is_error":false,"structured_output":{"decisions":[]},"modelUsage":{"claude-haiku-4-5-20251001":{"inputTokens":1,"outputTokens":1}}}' ;;
esac
'''


@pytest.fixture
def harness(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    for name, body in (("docker", DOCKER_STUB), ("claude", CLAUDE_STUB)):
        p = stub / name; p.write_text(body); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "plan.json").write_text(json.dumps(PLAN))
    (stub / "finalize.json").write_text(json.dumps({"exit_code": 0, "errors": {}}))
    inst = tmp_path / "instances"; inst.mkdir()
    (inst / "t.env").write_text(f"INSTANCE=t\nLOGS_DIR={logs}\n")
    shutil.copy(DRIVER, tmp_path / "session-driver.sh")
    shutil.copytree(REPO / "scripts", tmp_path / "scripts")
    env = dict(os.environ, STUB_DIR=str(stub), LOGS_DIR=str(logs),
               PINCHY_DOCKER_BIN=str(stub / "docker"), PINCHY_CLAUDE_BIN=str(stub / "claude"))
    def run(*args, **extra):
        e = dict(env, **extra)
        return subprocess.run(["bash", str(tmp_path / "session-driver.sh"), "t", *args], capture_output=True, text=True, env=e, cwd=tmp_path)
    return run, stub, logs


def test_plan_skip_exits_zero_without_claude(harness):
    run, stub, _ = harness
    (stub / "plan.json").write_text(json.dumps(dict(PLAN, stages=[], skip_reason="Session already exists")))
    r = run()
    assert r.returncode == 0 and "Session already exists" in r.stdout + r.stderr
    assert not (stub / "claude.log").exists()


def test_happy_path_sequences_stages(harness):
    run, stub, logs = harness
    r = run("--skip-supervisor")
    assert r.returncode == 0, r.stderr
    docker = (stub / "docker.log").read_text()
    claude = (stub / "claude.log").read_text().splitlines()
    assert "session_ctl plan --skip-supervisor" in docker
    assert docker.index("stage-begin --session-id 42 --stage learning") < docker.index("run-learning") < docker.index("stage-end --session-id 42 --stage learning")
    assert "stage-begin --session-id 42 --stage supervisor" not in docker
    assert len(claude) == 2
    strat, execu = claude
    assert "-p /pinchy-strategist t" in strat and "--model claude-opus-4-8" in strat and "--output-format stream-json" in strat
    assert "--allowedTools mcp__pinchy__*,WebSearch" in strat and "--strict-mcp-config" in strat and "--mcp-config" in strat
    assert "--permission-mode dontAsk" in strat and "--permission-prompts none" in strat and "--bare" not in strat
    assert "--system-prompt SYSTEM PROMPT" in execu and "--json-schema" in execu and "--output-format json" in execu and "--bare" not in execu
    stdins = sorted(stub.glob("claude.stdin.*"))
    assert any("CONTEXT FOR" in p.read_text() for p in stdins)
    assert any(json.loads(p.read_text() or "{}") == {"positions": []} for p in stdins if p.read_text().startswith("{"))
    assert (logs / "sessions" / "2026-09-22" / "strategist" / "result.json").exists()
    assert "trader --decisions-file /app/logs/sessions/2026-09-22/executor/result.json --input-file /app/logs/sessions/2026-09-22/executor/out.json --session-id 42 --result-out /app/logs/sessions/2026-09-22/executor/result-out.json" in docker
    end = [line for line in docker.splitlines() if "stage-end --session-id 42 --stage strategist" in line][0]
    assert "--result /app/logs/sessions/2026-09-22/strategist/result.json" in end and "--transcript" in end and "--tools-log" in end and "--context" in end


def test_claude_failure_continues_and_reports_exit_code(harness):
    run, stub, _ = harness
    (stub / "finalize.json").write_text(json.dumps({"exit_code": 1, "errors": {"strategist": "[usage_limit] ..."}}))
    r = run("--skip-supervisor", CLAUDE_FAIL="1")
    assert r.returncode == 1
    docker = (stub / "docker.log").read_text()
    assert "stage-end --session-id 42 --stage strategist" in docker and "--exit-code 1" in docker
    assert "stage-begin --session-id 42 --stage executor" in docker  # loop continued


def test_executor_emit_exit_3_skips_model_call(harness):
    run, stub, _ = harness
    (stub / "plan.json").write_text(json.dumps(dict(PLAN, stages=[PLAN["stages"][3]])))
    r = run(EMIT_EXIT="3")
    assert r.returncode == 0
    assert not (stub / "claude.log").exists()
    assert "stage-end --session-id 42 --stage executor --session-date 2026-09-22 --exit-code 3" in (stub / "docker.log").read_text()


def test_mcp_config_subcommand(harness, tmp_path):
    r = subprocess.run(["bash", str(tmp_path / "session-driver.sh"), "mcp-config", "t", "reflection", "--session-id", "7",
                        "--max-tool-calls", "25", "--tools-log", "/app/logs/x/tools.jsonl", "--model", "claude-sonnet-4-6"],
                       capture_output=True, text=True, cwd=tmp_path)
    cfg = json.loads(r.stdout)["mcpServers"]["pinchy"]
    assert cfg["command"] == "docker"
    assert cfg["args"][:6] == ["compose", "-p", "pinchy-t", "--env-file", "instances/t.env", "exec"]
    assert "--registry" in cfg["args"] and "reflection" in cfg["args"] and "7" in cfg["args"] and "25" in cfg["args"]
```

- [ ] **Step 5: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/test_session_driver.py -v`
Expected: FAIL (`session-driver.sh` missing → `shutil.copy` raises `FileNotFoundError`).

- [ ] **Step 6: Write `session-driver.sh`**

```bash
#!/bin/bash
#
# session-driver.sh <instance> [--force] [--dry-run] [--only STAGE] [--skip-*...] [--pipeline-hours N] [--pipeline-limit N]
# session-driver.sh mcp-config <instance> <registry> [--session-id N] [--max-tool-calls N] [--tools-log PATH] [--model M]
#
# Host-side driver for the daily session under the operator's Claude
# subscription. Runs UNDER cron-wrap.sh (which owns HALT, heartbeat, alerts).
# It never decides anything: v2/session_ctl.py plans the stages, validates
# their results and writes telemetry; this script only sequences container
# commands and `claude -p` calls. Spec:
# docs/superpowers/specs/2026-09-22-claude-session-inversion-design.md
#
# NEVER pass --bare: bare mode does not read the subscription (OAuth) login.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
CLAUDE_BIN="${PINCHY_CLAUDE_BIN:-claude}"
DOCKER_BIN="${PINCHY_DOCKER_BIN:-docker}"
PYJSON="python3 $SCRIPT_DIR/scripts/driver_json.py"
CONTAINER_LOGS=/app/logs

die() { echo "[$(date -Is)] session-driver: $*" >&2; exit 2; }
log() { echo "[$(date -Is)] session-driver: $*"; }

load_instance() {
    local name="$1"
    ENV_FILE="$SCRIPT_DIR/instances/$name.env"
    [ -f "$ENV_FILE" ] || die "No such instance: instances/$name.env"
    grep -qx "INSTANCE=$name" "$ENV_FILE" || die "instances/$name.env must contain the line INSTANCE=$name"
    INSTANCE="$name"
    LOGS_DIR="$(grep -E '^LOGS_DIR=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
    [ -n "$LOGS_DIR" ] || die "instances/$name.env must set LOGS_DIR"
    case "$LOGS_DIR" in /*) ;; *) LOGS_DIR="$SCRIPT_DIR/$LOGS_DIR" ;; esac
    COMPOSE=("$DOCKER_BIN" compose -p "pinchy-$INSTANCE" --env-file "instances/$INSTANCE.env")
}

in_container() { "${COMPOSE[@]}" exec -T trading "$@"; }
ctl() { in_container python -m v2.session_ctl "$@"; }
host_path() { echo "${1/#$CONTAINER_LOGS/$LOGS_DIR}"; }   # container path -> host path
err_text() {  # $1 status, $2 stderr file -> stderr tail, or "exit N", or "" on success (session_ctl treats "" as no error)
    [ "$1" -ne 0 ] || { echo ""; return; }
    local t; t="$(tail -c 400 "$2" 2>/dev/null | tr '\n' ' ')"; echo "${t:-exit $1}"
}

mcp_config_json() {
    # $1 registry, then optional --session-id/--max-tool-calls/--tools-log/--model
    local registry="$1"; shift
    python3 - "$INSTANCE" "$registry" "$@" <<'EOF'
import json, sys
inst, registry, *rest = sys.argv[1:]
args = ["compose", "-p", f"pinchy-{inst}", "--env-file", f"instances/{inst}.env", "exec", "-i", "trading",
        "python", "-m", "v2.mcp_server", "--registry", registry] + rest
print(json.dumps({"mcpServers": {"pinchy": {"command": "docker", "args": args}}}))
EOF
}

# --- claude invocations (the ONLY place flags live) --------------------------
claude_agentic() {
    # $1 skill  $2 model  $3 allowed_tools  $4 mcp_config_json  $5 timeout  stdin: context  stdout: stream-json
    timeout "$5" "$CLAUDE_BIN" -p "/$1 $INSTANCE" \
        --model "$2" \
        --output-format stream-json --verbose \
        --permission-mode dontAsk --permission-prompts none \
        --allowedTools "$3" \
        --strict-mcp-config --mcp-config "$4" \
        --disallowedTools "AskUserQuestion,Edit,Write,NotebookEdit,Agent,Bash"
}

claude_structured() {
    # $1 model  $2 system prompt text  $3 schema json  $4 timeout  stdin: user message  stdout: json
    local scratch; scratch="$(mktemp -d)"
    ( cd "$scratch" && timeout "$4" "$CLAUDE_BIN" -p \
        --model "$1" \
        --output-format json --json-schema "$3" \
        --system-prompt "$2" \
        --permission-mode dontAsk --permission-prompts none \
        --tools "" )
    local status=$?
    rm -rf "$scratch"
    return $status
}

# --- stage runners -----------------------------------------------------------
sub() { # substitute placeholders in a command template: $1 template, then key=value pairs
    local s="$1"; shift
    for kv in "$@"; do s="${s//\{${kv%%=*}\}/${kv#*=}}"; done
    echo "$s"
}

run_python_stage() {  # $1 stage-json-line
    local cmd; cmd="$(sub "$(field "$1" command)" "session_id=$SESSION_ID")"
    # shellcheck disable=SC2086
    in_container $cmd; local status=$?
    if [ "$status" -ne 0 ]; then
        ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code "$status" --error "exit $status"
    else
        ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code 0
    fi
}

run_agentic_stage() {
    local skill model allowed cap t cdir hdir cfg status
    case "$STAGE" in strategist) skill=pinchy-strategist ;; strategy) skill=pinchy-reflection ;; supervisor) skill=pinchy-supervisor ;; *) die "no skill for $STAGE" ;; esac
    model="$(field "$1" model)"; allowed="$(field "$1" allowed_tools)"; cap="$(field "$1" max_tool_calls)"; t="$(field "$1" timeout_seconds)"
    cdir="$CONTAINER_LOGS/sessions/$SESSION_DATE/$STAGE"; hdir="$(host_path "$cdir")"; mkdir -p "$hdir"
    local exec_result=""; [ -f "$(host_path "$CONTAINER_LOGS/sessions/$SESSION_DATE/executor/result-out.json")" ] && exec_result="--executor-result $CONTAINER_LOGS/sessions/$SESSION_DATE/executor/result-out.json"
    # shellcheck disable=SC2086
    ctl stage-context --stage "$STAGE" --session-id "$SESSION_ID" $exec_result > "$hdir/context.md" || { ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code 2 --error "stage-context failed"; return; }
    cfg="$(mcp_config_json "$(field "$1" registry)" --session-id "$SESSION_ID" --max-tool-calls "$cap" --tools-log "$cdir/tools.jsonl" --model "$model")"
    claude_agentic "$skill" "$model" "$allowed" "$cfg" "$t" < "$hdir/context.md" > "$hdir/transcript.jsonl" 2> "$hdir/stderr.log"; status=$?
    $PYJSON last-result "$hdir/transcript.jsonl" "$hdir/result.json" || rm -f "$hdir/result.json"
    ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code "$status" \
        --result "$cdir/result.json" --transcript "$cdir/transcript.jsonl" --tools-log "$cdir/tools.jsonl" --context "$cdir/context.md" \
        --error "$(err_text "$status" "$hdir/stderr.log")"
}

run_structured_stage() {
    local model prompt schema t cdir hdir emit consume status dry
    model="$(field "$1" model)"; t="$(field "$1" timeout_seconds)"
    cdir="$CONTAINER_LOGS/sessions/$SESSION_DATE/$STAGE"; hdir="$(host_path "$cdir")"; mkdir -p "$hdir"
    dry=""; [ "$DRY_RUN" = "true" ] && dry=" --dry-run"
    emit="$(sub "$(field "$1" emit)" "out=$cdir/out.json" "session_id=$SESSION_ID" "dry_run=$dry")"
    # shellcheck disable=SC2086
    in_container $emit; status=$?
    if [ "$status" -ne 0 ]; then
        local reason; reason="$( [ -f "$hdir/out.json" ] && $PYJSON field "$hdir/out.json" skipped )"
        ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code "$status" --error "${reason:-emit exit $status}"
        return
    fi
    prompt="$(ctl prompt --stage "$(field "$1" prompt)")"; schema="$(ctl schema --stage "$(field "$1" schema)")"
    if [ "$STAGE" = "executor" ]; then
        $PYJSON field "$hdir/out.json" executor_input > "$hdir/stdin.txt"
    else
        $PYJSON field "$hdir/out.json" user_message > "$hdir/stdin.txt"
    fi
    if [ "$STAGE" = "dashboard" ] && [ ! -s "$hdir/stdin.txt" ]; then
        echo '{"is_error": false, "structured_output": {"entries": []}}' > "$hdir/result.json"; status=0
    else
        claude_structured "$model" "$prompt" "$schema" "$t" < "$hdir/stdin.txt" > "$hdir/result.json" 2> "$hdir/stderr.log"; status=$?
    fi
    if [ "$status" -eq 0 ]; then
        consume="$(sub "$(field "$1" consume)" "file=$cdir/result.json" "out=$cdir/out.json" "session_id=$SESSION_ID" "dry_run=$dry")"
        [ "$STAGE" = "executor" ] && consume="$consume --result-out $cdir/result-out.json"
        # shellcheck disable=SC2086
        in_container $consume; status=$?
    fi
    ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code "$status" \
        --result "$cdir/result.json" --input "$cdir/out.json" \
        --error "$(err_text "$status" "$hdir/stderr.log")"
}

run_batches_stage() {
    local model prompt schema t cdir hdir emit consume status f
    model="$(field "$1" model)"; t="$(field "$1" timeout_seconds)"
    cdir="$CONTAINER_LOGS/sessions/$SESSION_DATE/$STAGE"; hdir="$(host_path "$cdir")"; mkdir -p "$hdir/batches"
    emit="$(sub "$(field "$1" emit)" "dir=$cdir/batches" "session_id=$SESSION_ID")"
    # shellcheck disable=SC2086
    in_container $emit; status=$?
    if [ "$status" -ne 0 ]; then ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code "$status" --error "emit exit $status"; return; fi
    prompt="$(ctl prompt --stage "$(field "$1" prompt)")"; schema="$(ctl schema --stage "$(field "$1" schema)")"
    for f in "$hdir"/batches/batch-*.json; do
        [ -e "$f" ] || break
        case "$f" in *.result.json) continue ;; esac
        $PYJSON field "$f" user_message | claude_structured "$model" "$prompt" "$schema" "$t" > "${f%.json}.result.json" 2>> "$hdir/stderr.log" \
            || log "batch $(basename "$f") failed (marked noise by --ingest)"
    done
    # shellcheck disable=SC2046
    $PYJSON merge-results "$hdir/result.json" $(ls "$hdir"/batches/*.result.json 2>/dev/null) 2>/dev/null || echo '{"is_error": false}' > "$hdir/result.json"
    consume="$(sub "$(field "$1" consume)" "dir=$cdir/batches" "session_id=$SESSION_ID")"
    # shellcheck disable=SC2086
    in_container $consume; status=$?
    ctl stage-end --session-id "$SESSION_ID" --stage "$STAGE" --session-date "$SESSION_DATE" --exit-code "$status" --result "$cdir/result.json" \
        --error "$(err_text "$status" "$hdir/stderr.log")"
}

field() { printf '%s' "$1" > "$TMP/stage.json"; $PYJSON field "$TMP/stage.json" "$2"; }

# --- entry ------------------------------------------------------------------
[ $# -ge 1 ] || die "usage: $0 <instance> [flags] | mcp-config <instance> <registry> [...]"
if [ "$1" = "mcp-config" ]; then
    shift; load_instance "$1"; shift; mcp_config_json "$@"; exit 0
fi

load_instance "$1"; shift
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$LOGS_DIR"
exec > >(tee -a "$LOGS_DIR/driver.log") 2>&1
log "start instance=$INSTANCE args=$*"

if [ -z "${PINCHY_CLAUDE_BIN:-}" ]; then   # real claude: refuse to start without a subscription login
    [ -f "$HOME/.claude/.credentials.json" ] || die "no Claude Code login at ~/.claude/.credentials.json — run 'claude' once interactively"
fi
"${COMPOSE[@]}" up -d >/dev/null || die "compose up failed"

ctl plan "$@" > "$TMP/plan.json" || die "session_ctl plan failed"
SKIP="$($PYJSON field "$TMP/plan.json" skip_reason)"
if [ -n "$SKIP" ]; then log "nothing to do: $SKIP"; exit 0; fi
SESSION_ID="$($PYJSON field "$TMP/plan.json" session_id)"
SESSION_DATE="$($PYJSON field "$TMP/plan.json" session_date)"
DRY_RUN="$($PYJSON field "$TMP/plan.json" dry_run)"
[ -n "$SESSION_ID" ] || die "plan returned no session_id (session tracking unavailable) — refusing to run untracked"

while IFS= read -r stage_json; do
    STAGE="$(field "$stage_json" name)"
    if [ "$(field "$stage_json" skip)" = "true" ]; then log "[$STAGE] SKIPPED: $(field "$stage_json" skip_reason)"; continue; fi
    log "[$STAGE] begin"
    ctl stage-begin --session-id "$SESSION_ID" --stage "$STAGE" >/dev/null || log "[$STAGE] stage-begin failed (continuing)"
    case "$(field "$stage_json" kind)" in
        python)     run_python_stage "$stage_json" ;;
        agentic)    run_agentic_stage "$stage_json" ;;
        structured) run_structured_stage "$stage_json" ;;
        batches)    run_batches_stage "$stage_json" ;;
        *) die "unknown stage kind" ;;
    esac
    log "[$STAGE] end"
done < <($PYJSON stages "$TMP/plan.json")

ctl finalize --session-id "$SESSION_ID" > "$TMP/final.json" || die "finalize failed"
EXIT="$($PYJSON field "$TMP/final.json" exit_code)"
log "finished exit=$EXIT errors=$($PYJSON field "$TMP/final.json" errors)"
exit "${EXIT:-1}"
```

`chmod +x session-driver.sh scripts/driver_json.py`. Two things to verify by hand once, then leave a note in the script header: (a) `--tools ""` is accepted by `claude -p` 2.1.280 — if it is rejected, replace it in `claude_structured` with `--disallowedTools "Bash,Read,Edit,Write,Glob,Grep,WebSearch,WebFetch,Agent,NotebookEdit,AskUserQuestion,Skill"`; (b) `-p` with no prompt argument reads the prompt from stdin (the doc says stdin is read; if the CLI insists on a prompt argument, pass `-p "Follow the system prompt."` and keep the payload on stdin).

The `field` helper writes the stage line to a temp file for every lookup; that is fine at this call volume.

- [ ] **Step 7: Write the smoke script**

```bash
#!/bin/bash
# scripts/smoke-claude-headless.sh — guard for the driver's one hard assumption:
# a non-bare `claude -p` under a cron-like environment authenticates with the
# subscription login. Run after every Claude Code upgrade and from
# `task driver:smoke`. Exits 1 with a loud message if the assumption breaks
# (e.g. --bare becoming the -p default).
set -uo pipefail
out="$(env -i HOME="$HOME" PATH="$PATH" claude -p "Reply with exactly the word ok." --output-format json --model claude-haiku-4-5-20251001 2>&1)"
status=$?
if [ $status -ne 0 ]; then echo "FAIL: claude -p exited $status: $out" >&2; exit 1; fi
provider="$(printf '%s' "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((list(d.get("modelUsage",{}).values()) or [{}])[0].get("provider",""), d.get("is_error"))')"
case "$provider" in
  "firstParty False") echo "ok: headless claude uses the subscription login (firstParty)";;
  *) echo "FAIL: unexpected provider/is_error: $provider — did --bare become the default? see spec" >&2; exit 1;;
esac
```

- [ ] **Step 8: Taskfile targets**

```yaml
  session:claude:
    desc: Run the daily session through the Claude Code driver (subscription-billed LLM stages)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    cmds:
      - ./session-driver.sh {{.INSTANCE}} {{.CLI_ARGS}}

  driver:smoke:
    desc: Verify headless claude still authenticates with the subscription login (run after Claude Code upgrades)
    cmds:
      - ./scripts/smoke-claude-headless.sh
```

- [ ] **Step 9: Run the tests and the smoke script**

Run: `task test INSTANCE=paper -- tests/test_driver_json.py tests/test_session_driver.py -v && task driver:smoke`
Expected: PASS; smoke prints `ok: headless claude uses the subscription login (firstParty)`.

- [ ] **Step 10: Hand-run on paper (dry run, then one stage)**

```bash
./session-driver.sh paper --dry-run --force --only executor     # market may be closed → executor exit 3 → completed
./session-driver.sh paper --force --only supervisor              # first real agentic stage: check supervisor_memos and tools.jsonl
```

Inspect `logs/paper/sessions/<date>/supervisor/` and `SELECT * FROM session_stages WHERE session_id = <id>`. Fix the `--tools ""` / stdin-prompt assumptions from Step 6 if either failed, and record the verified form in the script header.

- [ ] **Step 11: Commit**

```bash
git add session-driver.sh scripts/driver_json.py scripts/smoke-claude-headless.sh tests/test_driver_json.py tests/test_session_driver.py Taskfile.yml
git commit -m "Add session-driver.sh: cron-wrap-compatible host loop over session_ctl and claude -p

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 10: Interactive skills, docs, instance knobs, and the paper cron switch

**Files:**
- Create: `.claude/skills/pinchy-session/SKILL.md`, `.claude/skills/pinchy-executor/SKILL.md`, `.claude/skills/pinchy-classify/SKILL.md`
- Modify: `instances/example.env`, `crontab`, `CLAUDE.md`, `docs/runbook-recovery.md`, `docs/audit-playbook.md` (one note), `tests/test_pricing_coverage.py` (no change expected; run it)
- Test: `tests/v2/test_prompts.py` (skill files exist), `tests/test_pricing_coverage.py`

**Interfaces:**
- `/pinchy-session <instance> [driver flags]` — runs `./session-driver.sh` and reports.
- `/pinchy-executor <instance>` and `/pinchy-classify <instance>` — run `./session-driver.sh <instance> --force --only executor --dry-run` / `--only pipeline` and summarise the artefacts.
- New optional knobs in `instances/<name>.env`: `ALGO_STAGE_MODEL_SUPERVISOR`, `ALGO_STAGE_MODEL_STRATEGIST`, `ALGO_STAGE_MODEL_STRATEGY`, `ALGO_STAGE_MODEL_PIPELINE`, `ALGO_STAGE_MODEL_DASHBOARD` (full model ids; executor keeps `ALGO_EXECUTOR_MODEL`).

- [ ] **Step 1: Extend the prompts test to cover the wrapper skills**

```python
# tests/v2/test_prompts.py — append
@pytest.mark.parametrize("skill", ["pinchy-session", "pinchy-executor", "pinchy-classify"])
def test_wrapper_skills_exist_and_name_the_driver(skill):
    raw = (prompts.REPO_ROOT / ".claude" / "skills" / skill / "SKILL.md").read_text()
    assert raw.startswith("---\n") and f"name: {skill}" in raw
    assert "session-driver.sh" in raw
```

Run: `task test INSTANCE=paper -- tests/v2/test_prompts.py -v -k wrapper` → FAIL (files missing).

- [ ] **Step 2: Write the three wrapper skills**

```markdown
---
name: pinchy-session
description: Run the Pinchy daily session for an instance through the Claude Code driver and report the outcome. Usage: /pinchy-session <instance> [--force] [--dry-run] [--only STAGE] [--skip-...]
---

Run the daily session for the instance named in `$ARGUMENTS` (first word; any
further words are driver flags):

```
./session-driver.sh $ARGUMENTS
```

The driver is deterministic; do not intervene while it runs. When it exits:

1. Report the exit code and the `finished exit=... errors=...` line from the
   driver output.
2. For each failed stage, show the `[class] message` from `session_stages`
   (`docker compose -p pinchy-<instance> --env-file instances/<instance>.env exec -T db psql -U algo -d trading -c "select stage_name, status, error from session_stages where session_id = <id> order by id"`)
   and the last 20 lines of `<LOGS_DIR>/sessions/<date>/<stage>/stderr.log`.
3. Do not retry, edit env files, touch HALT files, or run migrations. A retry
   is `/pinchy-session <instance> --force`, and only if the user asks.
```

```markdown
---
name: pinchy-executor
description: Run only the executor stage of the Pinchy session for an instance (dry-run by default) and show the decisions. Usage: /pinchy-executor <instance> [--live]
---

Run the executor stage alone through the driver. Without `--live` it is a dry
run (no orders):

```
./session-driver.sh <instance> --force --only executor --dry-run
```

With `--live`, omit `--dry-run` and confirm with the user first — this places
orders on the instance's Alpaca account.

Afterwards print `<LOGS_DIR>/sessions/<date>/executor/result.json`'s
`structured_output.decisions` as a table (ticker, action, intent, confidence,
reasoning) and the `errors` list from `result-out.json`.
```

```markdown
---
name: pinchy-classify
description: Run only the news classifier stage of the Pinchy session for an instance and summarise what was stored. Usage: /pinchy-classify <instance>
---

```
./session-driver.sh <instance> --force --only pipeline
```

Then report the batch count under `<LOGS_DIR>/sessions/<date>/pipeline/batches/`,
how many `*.result.json` files are present, and the `Pipeline stats:` line from
the container log (`docker compose -p pinchy-<instance> --env-file instances/<instance>.env logs --tail 50 trading`).
```

- [ ] **Step 3: Instance knobs and crontab**

Append to `instances/example.env`:

```
# Claude Code driver (session-driver.sh): per-stage model overrides, full ids.
# Executor keeps ALGO_EXECUTOR_MODEL. Unset = defaults in v2/session_ctl.py.
# ALGO_STAGE_MODEL_SUPERVISOR=claude-fable-5
# ALGO_STAGE_MODEL_STRATEGIST=claude-opus-4-8
# ALGO_STAGE_MODEL_STRATEGY=claude-sonnet-4-6
# ALGO_STAGE_MODEL_PIPELINE=claude-haiku-4-5-20251001
# ALGO_STAGE_MODEL_DASHBOARD=claude-haiku-4-5-20251001
```

In `crontab`, change the paper session line to:

```
30 12 * * 1-5 cd /home/jay/dev/algo/ && ./cron-wrap.sh --instance paper paper-session ./session-driver.sh paper
```

and add above it the comment: `# paper runs LLM stages through the Claude Code driver (subscription); live stays on task session until the hiatus review (docs/runbook-recovery.md).` Do **not** install it yet — that is rollout step 3 below.

- [ ] **Step 4: Docs**

`CLAUDE.md`: under "Commands" add:

```
# Run the daily session through the Claude Code driver (LLM stages on the
# operator's Claude subscription; Python keeps the order path). Same flags
# as `task session`. Spec: docs/superpowers/specs/2026-09-22-claude-session-inversion-design.md
task session:claude INSTANCE=paper -- --force
./session-driver.sh paper --only supervisor        # one stage
task driver:smoke                                  # after Claude Code upgrades: subscription auth still works headless
```

and under "Key v2 Modules" add `session_ctl.py` (driver bookkeeping: plan/stage-end/validators/telemetry), `mcp_server.py` (stdio MCP over the tool registries; `--max-tool-calls` replaces `max_turns`), `prompts.py` (prompt text lives in `.claude/skills/pinchy-*/SKILL.md` and `v2/prompts/*.md`). Add one line to "Environment Variables": the `ALGO_STAGE_MODEL_*` knobs. Add to the halt/resume note: `session-driver.sh` runs under `cron-wrap.sh`, so the HALT sentinels apply unchanged, and `ALGO_TRADING_HALTED` is honoured by `session_ctl plan`.

`docs/runbook-recovery.md`, "Host bootstrap" step 6.5: "**Claude Code login:** the paper session's LLM stages run as headless `claude -p` under the operator's subscription. Log in once interactively (`claude`), confirm `~/.claude/.credentials.json` exists, and run `task driver:smoke`. Cron runs as the same user, so no further setup." Under "Halt / Resume" add: "Interactive Claude Code use in the hour before 12:30 MST shares the subscription usage window with the paper session; a `[usage_limit]` stage failure in the alert means exactly that."

`docs/audit-playbook.md`, `CACHE_HIT_RATIO_DEGRADATION`: add the sentence "Stages run by the Claude Code driver (`agent_events.payload.backend = 'claude_code'`) report Claude Code's cache figures, which are not comparable to the API series; exclude them from the ratio."

- [ ] **Step 5: Run the checks**

Run: `task test INSTANCE=paper -- tests/v2/test_prompts.py tests/test_pricing_coverage.py tests/test_schema_mirror.py -v && task lint`
Expected: PASS.

- [ ] **Step 6: Full suite**

Run: `task test INSTANCE=paper`
Expected: all green. Note the count against the ~1,965 baseline.

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/pinchy-session .claude/skills/pinchy-executor .claude/skills/pinchy-classify instances/example.env crontab CLAUDE.md docs/runbook-recovery.md docs/audit-playbook.md tests/v2/test_prompts.py
git commit -m "Wrapper skills, model knobs, docs and the paper cron line for the Claude Code driver

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Rollout (after the branch merges; manual, in this order)

1. `task up INSTANCE=paper` (picks up the compose mount and the rebuilt image with `mcp`).
2. `task driver:smoke`.
3. On a day the cron has already run: `./session-driver.sh paper --force` and read `logs/paper/driver.log`, `session_stages`, `llm_call_contexts`, and `/costs/<session_id>` on the internal dashboard.
4. `crontab /home/jay/dev/algo/crontab` and commit nothing further (the crontab file is already in the repo from Task 10).
5. After five clean paper sessions and the live hiatus review in `docs/runbook-recovery.md`, switch the `live` line the same way in a separate PR.
6. Follow-up PR: retire the API backend (`run_session`'s LLM branches, `claude_client.run_agentic_loop`'s callers) once nothing depends on it.
