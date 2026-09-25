# Session Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split each LLM-backed session stage into "Python emits the model's input" and "Python consumes the model's output" so every stage is replayable and testable without a model, remove the hidden Haiku call inside the strategist's news tool, and let a session resume past stages a prior run completed.

**Architecture:** Nothing about how sessions run changes. `task session` stays the cron entry and the API backend stays the only backend. Each task adds a file-based seam (`--emit-*` / `--ingest` / `--decisions-file`) beside the existing code path, with the existing path refactored to call the same functions so the two cannot drift. Task 6 adds `--resume`, which the strategist pilot (`docs/superpowers/plans/2026-09-22-strategist-pilot.md`) depends on.

**Tech Stack:** Python 3.12 (container), PostgreSQL 16, pytest (run in docker: `task test INSTANCE=paper -- <args>`), ruff.

**Spec:** `docs/superpowers/specs/2026-09-22-claude-session-inversion-design.md` — sections "Structured stages", "News curation", "Changelog", and the `plan` paragraph on `completed_stages` resume. This plan is phase one of that spec; the inversion itself is deferred to the strategist pilot and is not built here.

## Global Constraints

- Tests run in the container: `task test INSTANCE=paper -- tests/v2/test_x.py -v`. Host python is 3.10 and cannot run the suite.
- No test may reach the network. `tests/v2/conftest.py` patches `v2.session.*` LLM entry points; new import-site names that touch the DB must be added to its patch tuples.
- The money path (`_validate_llm_ids` → `_execute_decisions` → `_handle_thesis_invalidations` → `_log_decisions`) is called, never reimplemented.
- The daily-loss breaker runs once in `prepare_executor_stage` and once more in `execute_decisions_payload`; `run_trading_session` is composed from those two functions so the API path and the file path share one breaker implementation.
- Stage names in `session_stages` are `learning, supervisor, pipeline, strategist, executor, strategy, dashboard`. Stage names in `llm_call_contexts` / `agent_events` are `supervisor, pipeline, ideation, trading, reflection, dashboard_publish`. Keep both namespaces exactly.
- Prompt text moves verbatim. The one permitted edit is the `get_curated_news` sentence in the strategist prompt (Task 4), because Task 4 makes it false.
- Every `claude-*` literal in `v2/` needs a `model_pricing` row (`tests/test_pricing_coverage.py`). This plan adds none.
- ruff: line length 140, target py312, isort ordering. Pre-commit runs `ruff check .`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `v2/prompts.py` (new) | `load_prompt(name)`: read `v2/prompts/<name>.md`. Single source of truth for prompt text. |
| `v2/prompts/{classifier,executor,changelog,strategist}.md` (new) | System prompts moved verbatim from Python constants. `strategist.md` is the pre-close (session) variant. |
| `v2/agent.py` (modify) | Split `get_trading_decisions` into serialize / call / `parse_executor_response`; add `executor_output_schema()`, `executor_input_to_dict()`, `executor_input_from_dict()`. |
| `v2/trader.py` (modify) | `prepare_executor_stage()` (steps 1–3 + breaker) and `execute_decisions_payload()` (steps 4–6 from a payload); `run_trading_session` composed from them; CLI `--emit-input` / `--decisions-file`. |
| `v2/classifier.py` (modify) | `build_batch_user_message()`, `parse_batch_entries()`, `classification_output_schema()`; `_classify_batch` uses them. |
| `v2/pipeline.py` (modify) | `emit_batches()` / `ingest_batches()`; CLI `--emit-batches` / `--ingest`. |
| `v2/tools.py` (modify) | `tool_get_curated_news` ranks deterministically; `v2/news_filter.py` deleted. |
| `v2/dashboard_publish.py` (modify) | `emit_changelog_commits()`, `changelog_output_schema()`, `run_dashboard_stage(changelog_entries=...)`, new `main()`. |
| `v2/session.py` (modify) | `--resume`: reopen the latest session row for the date, skip its completed stages; `_finalize_session` counts failed stage rows from earlier runs. |
| `v2/database/trading_db.py` (modify) | `reopen_session()`, `get_stage_errors()`. |
| `tests/v2/test_prompts.py` (new), `tests/v2/test_pipeline.py` (new if absent) | Prompt parity; batch emit/ingest. |
| `tests/v2/conftest.py` (modify) | New DB import-site names in the patch tuples. |

**Prerequisite (manual, before Task 1):** paper sessions have failed daily since 2026-08-27 because the Anthropic API credit balance is exhausted (`session_stages.error` on `strategist`/`strategy`: "Your credit balance is too low to access the Anthropic API"). Top up credits, run `task session INSTANCE=paper -- --force` by hand, and confirm a `completed` session row. The pilot plan measures against this baseline; without it there is nothing to compare.

---

### Task 1: Prompt files and loader

**Files:**
- Create: `v2/prompts.py`, `v2/prompts/classifier.md`, `v2/prompts/executor.md`, `v2/prompts/changelog.md`, `v2/prompts/strategist.md`
- Modify: `v2/classifier.py:153-186` (`BATCH_CLASSIFICATION_SYSTEM`), `v2/agent.py:179-233` (`TRADING_SYSTEM_PROMPT`), `v2/dashboard_publish.py:1281-1360` (`_changelog_prompt`, `summarize_changelog_commits`), `v2/ideation_claude.py:141-146` (`CLAUDE_SESSION_STRATEGIST_SYSTEM`)
- Test: `tests/v2/test_prompts.py`

**Interfaces:**
- Produces: `v2.prompts.load_prompt(name: str) -> str` (reads `v2/prompts/<name>.md`, stripped; raises `FileNotFoundError` for an unknown name), `v2.prompts.PROMPTS_DIR: Path`.
- Produces: `v2.dashboard_publish.CHANGELOG_SYSTEM_PROMPT: str` (Task 5 uses it).
- `v2/prompts/` lives under `v2/`, which is already mounted at `/app/v2` in the trading container (`docker-compose.yml`). No compose change.
- `v2.ideation_claude._STRATEGIST_TEMPLATE` and `CLAUDE_STRATEGIST_SYSTEM` (the post-close variant used only by `run_strategist_session`) stay in Python. Only the session variant moves to a file; a test asserts the file equals the rendered template so the two cannot drift.

- [ ] **Step 1: Write the failing tests**

```python
# tests/v2/test_prompts.py
"""Stage prompts live in v2/prompts/*.md. The Python constants load them, so
any backend that reads the files gets exactly what the API path sends."""
from pathlib import Path

import pytest

from v2 import prompts

PROMPT_FILES = ("classifier", "executor", "changelog", "strategist")


def test_prompts_dir_is_inside_v2():
    assert prompts.PROMPTS_DIR == Path(prompts.__file__).resolve().parent / "prompts"


@pytest.mark.parametrize("name", PROMPT_FILES)
def test_each_prompt_file_loads_non_empty(name):
    text = prompts.load_prompt(name)
    assert len(text) > 200
    assert text == text.strip()


def test_unknown_prompt_raises():
    with pytest.raises(FileNotFoundError):
        prompts.load_prompt("does-not-exist")


def test_classifier_constant_is_the_file():
    from v2.classifier import BATCH_CLASSIFICATION_SYSTEM
    assert BATCH_CLASSIFICATION_SYSTEM == prompts.load_prompt("classifier")


def test_executor_constant_is_the_file():
    from v2.agent import TRADING_SYSTEM_PROMPT
    assert TRADING_SYSTEM_PROMPT == prompts.load_prompt("executor")


def test_changelog_constant_is_the_file_and_demands_strict_json():
    from v2.dashboard_publish import CHANGELOG_SYSTEM_PROMPT
    assert CHANGELOG_SYSTEM_PROMPT == prompts.load_prompt("changelog")
    assert CHANGELOG_SYSTEM_PROMPT.endswith("Return strict JSON only. Do not wrap the response in markdown.")


def test_strategist_file_equals_rendered_session_template():
    from v2.ideation_claude import _STRATEGIST_TEMPLATE, CLAUDE_SESSION_STRATEGIST_SYSTEM
    rendered = _STRATEGIST_TEMPLATE.format(
        timing="before market close",
        review_scope="the portfolio",
        date_ref="today's",
        executor_note=" that runs immediately after you",
    ).strip()
    assert CLAUDE_SESSION_STRATEGIST_SYSTEM == prompts.load_prompt("strategist")
    assert CLAUDE_SESSION_STRATEGIST_SYSTEM == rendered
```

- [ ] **Step 2: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2.prompts'`

- [ ] **Step 3: Write the loader**

```python
# v2/prompts.py
"""Stage prompt text, loaded from v2/prompts/<name>.md.

The Python constants (BATCH_CLASSIFICATION_SYSTEM, TRADING_SYSTEM_PROMPT,
CHANGELOG_SYSTEM_PROMPT, CLAUDE_SESSION_STRATEGIST_SYSTEM) are loaded from
these files, so any other backend that reads the same files sends exactly
what the API path sends (tests/v2/test_prompts.py).

v2/ is mounted at /app/v2 in the trading container, so the files resolve
there without a compose change.
"""
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """System prompt for a stage (`v2/prompts/<name>.md`), stripped."""
    return (PROMPTS_DIR / f"{name}.md").read_text().strip()
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

and in `summarize_changelog_commits` change `system="Return strict JSON only. Do not wrap the response in markdown."` to `system=CHANGELOG_SYSTEM_PROMPT`. (The instruction text moved into the system prompt; the user message is now the bare payload. Both pieces still reach the model.)

- [ ] **Step 5: Move the session strategist prompt into a file**

Render the session variant once and save it: inside the container run

```bash
docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading \
  python -c 'from v2.ideation_claude import CLAUDE_SESSION_STRATEGIST_SYSTEM as s; print(s)' > v2/prompts/strategist.md
```

(run this **before** editing `ideation_claude.py`; the container sees the host checkout through the `./v2` mount). Inspect the file: it must start with `You are the strategist for an automated trading system. You run before market close to review the portfolio` and contain single-brace JSON examples such as `{"action":"sell","intent_type":"exit_full","intent_magnitude":null}`.

Then in `v2/ideation_claude.py` replace the `CLAUDE_SESSION_STRATEGIST_SYSTEM = _STRATEGIST_TEMPLATE.format(...)` block (lines 141–146) with:

```python
# The session (pre-close) variant is the file; the post-close variant
# CLAUDE_STRATEGIST_SYSTEM stays a rendered template for run_strategist_session.
# tests/v2/test_prompts.py asserts the file equals the rendered template.
CLAUDE_SESSION_STRATEGIST_SYSTEM = load_prompt("strategist")
```

and add `from .prompts import load_prompt` to the module imports.

- [ ] **Step 6: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_prompts.py tests/v2/test_classifier.py tests/v2/test_agent.py tests/v2/test_ideation_claude.py tests/v2/test_dashboard_publish.py -v && task lint`
Expected: all PASS; lint clean. If a dashboard_publish test asserts the old system string, update it to `CHANGELOG_SYSTEM_PROMPT`.

- [ ] **Step 7: Commit**

```bash
git add v2/prompts.py v2/prompts/ v2/classifier.py v2/agent.py v2/dashboard_publish.py v2/ideation_claude.py tests/v2/test_prompts.py
git commit -m "Move stage prompts into v2/prompts/*.md, load them in Python

One file per prompt so any backend reads exactly what the API path sends.
The session strategist prompt is asserted equal to its rendered template.

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
    """Steps 1–3 of run_trading_session, exposed for file-based replay.

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
git commit -m "Split the executor into prepare / parse / execute seams

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
    file-based ingest path (pipeline.py --ingest).
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
git commit -m "Expose classifier batches as files (--emit-batches / --ingest)

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

- [ ] **Step 4: Correct the one prompt sentence this makes false**

`v2/prompts/strategist.md` (Task 1) tells the strategist that `get_curated_news` "returns a Haiku-curated subset of news ranked by relevance to today's market". Replace that clause, and only that clause, so the bullet reads:

```
- Use `get_curated_news` by default for thesis research — it returns the highest-confidence, most recent news for the window, ranked deterministically. Use `get_news_signals` (raw firehose) if you need to look past the filter (e.g., specific ticker deep-dive, or you suspect the filter is missing something). Both return `[#id]`-prefixed lines for signal_refs citation.
```

Make the identical edit in `_STRATEGIST_TEMPLATE` in `v2/ideation_claude.py` (the same bullet, in the template text), otherwise `test_strategist_file_equals_rendered_session_template` fails. This is the only prompt wording change in this plan.

- [ ] **Step 5: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_tools.py tests/v2/test_prompts.py -v && task lint`
Expected: PASS; lint clean (an unused import would fail ruff).

- [ ] **Step 6: Commit**

```bash
git add -A v2/tools.py v2/news_filter.py v2/prompts/strategist.md v2/ideation_claude.py tests/v2/test_tools.py tests/v2/test_news_filter.py
git commit -m "Replace the Haiku news curation call with deterministic ranking

One fewer hidden LLM call inside a tool result; the strategist judges
relevance itself over a bounded, confidence-then-recency shortlist.

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


### Task 6: `--resume` — reopen the day's session and skip its completed stages

**Files:**
- Modify: `v2/database/trading_db.py:896-935` (add `reopen_session`, `get_stage_errors` next to `complete_session` / `fail_session`)
- Modify: `v2/session.py:203-240` (`_check_and_record_session`), `:530-567` (`_finalize_session`), `:567-662` (`run_session`), `:662-701` (`main`)
- Modify: `tests/v2/conftest.py:47-57` (`_SESSION_DB_PATCH_TARGETS`)
- Test: `tests/v2/test_session.py`, `tests/v2/test_db.py`

**Interfaces:**
- Produces in `v2/database/trading_db.py`:
  - `reopen_session(session_id: int) -> None` — `UPDATE sessions SET status = 'running', completed_at = NULL, error = NULL WHERE id = %s`.
  - `get_stage_errors(session_id: int) -> dict[str, str]` — `{stage_name: error}` for every `session_stages` row of the session with `status = 'failed'`, in row order; `error` NULL becomes `"failed"`.
- Produces in `v2/session.py`:
  - `_check_and_record_session(force: bool, session_date, resume: bool = False) -> tuple[int | None, set, str | None]`. With `resume=True` and an existing row for the date: reopens that row, returns its id and its completed stage names, never creates a new row. With `resume=True` and no row: behaves as a normal run. `resume` wins over `force`.
  - `run_session(..., resume: bool = False)` and CLI `--resume`.
  - `_finalize_session` fails the session when any stage row is `failed` and no later row for the same stage is `completed`, even if the current process saw no error (the failure happened in an earlier run of the same session).
- Semantics kept: `--force` alone still creates a fresh row and reruns everything (commit 7939c3d, "per-run sessions"). `--resume` is the only path that reuses a row. The strategist pilot's driver uses `--resume` for both halves of its sandwich.

- [ ] **Step 1: Write the failing DB tests**

```python
# tests/v2/test_db.py — append inside the class that holds test_get_completed_stages
    def test_reopen_session_sets_running_and_clears_error(self, mock_db, mock_cursor):
        from v2.database.trading_db import reopen_session
        reopen_session(session_id=7)
        sql, params = mock_cursor.execute.call_args[0]
        assert "UPDATE sessions" in sql and "'running'" in sql
        assert "completed_at = NULL" in sql and "error = NULL" in sql
        assert params == (7,)

    def test_get_stage_errors_maps_failed_rows(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_stage_errors
        mock_cursor.fetchall.return_value = [
            {"stage_name": "strategist", "error": "no playbook"},
            {"stage_name": "executor", "error": None},
        ]
        assert get_stage_errors(session_id=7) == {"strategist": "no playbook", "executor": "failed"}
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'failed'" in sql

    def test_get_stage_errors_empty(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_stage_errors
        mock_cursor.fetchall.return_value = []
        assert get_stage_errors(session_id=7) == {}
```

- [ ] **Step 2: Write the failing session tests**

```python
# tests/v2/test_session.py — append to the class that holds test_force_creates_new_session_row_when_one_already_exists
    def test_resume_reopens_existing_row_and_returns_completed_stages(self):
        from v2.session import _check_and_record_session
        with patch("v2.session.insert_session_record") as mock_insert, \
             patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "failed"}), \
             patch("v2.session.get_completed_stages", return_value={"learning", "pipeline"}), \
             patch("v2.session.reopen_session") as mock_reopen:
            session_id, completed, err = _check_and_record_session(force=False, session_date=date(2026, 9, 22), resume=True)
        assert session_id == 7
        assert completed == {"learning", "pipeline"}
        assert err is None
        mock_reopen.assert_called_once_with(7)
        mock_insert.assert_not_called()

    def test_resume_without_prior_row_starts_a_normal_session(self):
        from v2.session import _check_and_record_session
        with patch("v2.session.insert_session_record", return_value=12) as mock_insert, \
             patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.reopen_session") as mock_reopen:
            session_id, completed, err = _check_and_record_session(force=False, session_date=date(2026, 9, 22), resume=True)
        assert session_id == 12 and completed == set() and err is None
        mock_insert.assert_called_once()
        mock_reopen.assert_not_called()

    def test_resume_wins_over_force(self):
        from v2.session import _check_and_record_session
        with patch("v2.session.insert_session_record") as mock_insert, \
             patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "completed"}), \
             patch("v2.session.get_completed_stages", return_value=set()), \
             patch("v2.session.reopen_session"):
            session_id, _, _ = _check_and_record_session(force=True, session_date=date(2026, 9, 22), resume=True)
        assert session_id == 7
        mock_insert.assert_not_called()

    def test_resume_skips_completed_stages_and_runs_the_rest(self):
        """A resumed session must not rerun learning/supervisor/pipeline/strategist
        that a prior run completed, but must run the executor and later stages."""
        with patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "running"}), \
             patch("v2.session.get_completed_stages", return_value={"learning", "supervisor", "pipeline", "strategist"}), \
             patch("v2.session.reopen_session"), \
             patch("v2.session.get_stage_errors", return_value={}), \
             patch("v2.session.complete_session") as mock_complete, \
             patch("v2.session.run_backfill") as mock_backfill, \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_supervisor") as mock_supervisor, \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop") as mock_strategist, \
             patch("v2.session.run_trading_session") as mock_trader, \
             patch("v2.session.run_strategy_reflection") as mock_reflection, \
             patch("v2.session.run_dashboard_stage"):
            result = run_session(dry_run=False, resume=True)
        assert result.idempotent_skip is None
        mock_backfill.assert_not_called()
        mock_supervisor.assert_not_called()
        mock_pipeline.assert_not_called()
        mock_strategist.assert_not_called()
        mock_trader.assert_called_once()
        mock_reflection.assert_called_once()
        mock_complete.assert_called_once_with(7)

    def test_finalize_fails_session_when_a_prior_run_left_a_failed_stage(self):
        """The strategist failed in an earlier run of this session (e.g. the
        pilot driver's stage-end); this run saw no error itself. The session
        row must still end up failed, and the exit code must be 1."""
        with patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "running"}), \
             patch("v2.session.get_completed_stages", return_value={"learning", "supervisor", "pipeline"}), \
             patch("v2.session.reopen_session"), \
             patch("v2.session.get_stage_errors", return_value={"strategist": "[validator] no playbook"}), \
             patch("v2.session.fail_session") as mock_fail, \
             patch("v2.session.complete_session") as mock_complete, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):
            result = run_session(dry_run=False, resume=True, skip_ideation=True)
        assert result.prior_stage_errors == {"strategist": "[validator] no playbook"}
        assert result.has_errors is True  # so main() exits 1
        mock_complete.assert_not_called()
        mock_fail.assert_called_once()
        assert "strategist: [validator] no playbook" in mock_fail.call_args[0][1]

    def test_finalize_ignores_failed_stage_that_a_later_row_completed(self):
        """A stage that failed in run 1 and completed in run 2 has two rows;
        only the latest outcome counts."""
        with patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "running"}), \
             patch("v2.session.get_completed_stages", return_value={"strategist"}), \
             patch("v2.session.reopen_session"), \
             patch("v2.session.get_stage_errors", return_value={"strategist": "first attempt"}), \
             patch("v2.session.fail_session") as mock_fail, \
             patch("v2.session.complete_session") as mock_complete, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_supervisor"), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):
            run_session(dry_run=False, resume=True)
        mock_fail.assert_not_called()
        mock_complete.assert_called_once_with(7)

    def test_main_accepts_resume_flag(self):
        import sys
        from v2.session import main
        with patch.object(sys, "argv", ["session", "--resume", "--skip-dashboard"]), \
             patch("v2.session.run_session") as mock_run:
            mock_run.return_value.idempotent_skip = None
            mock_run.return_value.has_errors = False
            main()
        assert mock_run.call_args.kwargs["resume"] is True
```

- [ ] **Step 3: Run to verify failure**

Run: `task test INSTANCE=paper -- tests/v2/test_db.py -k "reopen or stage_errors" tests/v2/test_session.py -k "resume or finalize_fails or finalize_ignores or main_accepts" -v`
Expected: FAIL (`ImportError: cannot import name 'reopen_session'`, `TypeError: unexpected keyword argument 'resume'`).

- [ ] **Step 4: Add the two DB helpers**

```python
# v2/database/trading_db.py — after fail_session
def reopen_session(session_id: int) -> None:
    """Put a session back to running for a --resume run. The stage rows are
    kept: get_completed_stages() decides what the resumed run skips."""
    with get_cursor() as cur:
        cur.execute("""
            UPDATE sessions SET status = 'running', completed_at = NULL, error = NULL
            WHERE id = %s
        """, (session_id,))


def get_stage_errors(session_id: int) -> dict[str, str]:
    """{stage_name: error} for every failed stage row of a session, in row order.
    A stage that failed and was later completed appears here AND in
    get_completed_stages(); callers subtract the latter."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT stage_name, error FROM session_stages
            WHERE session_id = %s AND status = 'failed'
            ORDER BY id
        """, (session_id,))
        return {row["stage_name"]: row["error"] or "failed" for row in cur.fetchall()}
```

- [ ] **Step 5: Wire `--resume` through `session.py`**

Add `get_completed_stages`, `get_stage_errors`, `reopen_session` to the `from .database.trading_db import (...)` block at `v2/session.py:29`.

Replace `_check_and_record_session`:

```python
def _check_and_record_session(force: bool, session_date, resume: bool = False) -> tuple[int | None, set, str | None]:
    """Returns (session_id, completed_stages, early_error).

    Per-run sessions: --force (and a plain run) create a new sessions row and
    completed_stages is empty.

    --resume reopens the latest row for the date instead and returns the
    stages that row already completed, so the run skips them. This is how a
    session split across processes (the strategist pilot's driver) or a
    retry after a mid-run failure continues without redoing finished work.
    With no prior row, --resume is a plain run.

    Idempotency: if neither flag is set and ANY session already exists for
    this date — completed, failed, or running — we skip with early_error
    set. Failed sessions gate too: a cron double-fire after a partial
    failure would otherwise expire the first run's pending playbook actions
    and re-run the strategist, duplicating playbooks/theses for the date.
    """
    if resume:
        try:
            existing = get_session_for_date(session_date)
        except Exception as e:
            logger.warning("Could not look up session to resume: %s — starting a new one", e)
            existing = None
        if existing:
            session_id = existing["id"]
            try:
                completed = get_completed_stages(session_id)
                reopen_session(session_id)
            except Exception as e:
                logger.warning("Could not resume session %s: %s — proceeding without tracking", session_id, e)
                return None, set(), None
            logger.info("Resuming session %d (was %s); completed stages: %s",
                        session_id, existing.get("status"), ", ".join(sorted(completed)) or "none")
            return session_id, completed, None
    elif not force:
        try:
            existing = get_session_for_date(session_date)
            if existing:
                status = existing.get("status")
                if status == "completed":
                    msg = f"Session already completed for {session_date}"
                else:
                    msg = (
                        f"Session already exists for {session_date} "
                        f"(status={status}); re-run with --force to retry"
                    )
                logger.warning("%s. Use --force to override.", msg)
                return None, set(), msg
        except Exception as e:
            logger.warning("Could not check session status: %s — proceeding", e)
    try:
        session_id = insert_session_record(session_date)
        logger.info("Session ID: %d", session_id)
        return session_id, set(), None
    except Exception as e:
        logger.warning("Could not create session record: %s — proceeding without tracking", e)
        return None, set(), None
```

In `_finalize_session`, replace the `try: if result.has_errors: ... else: complete_session(session_id)` block with:

```python
        prior_errors: dict[str, str] = {}
        try:
            completed = get_completed_stages(session_id)
            prior_errors = {s: e for s, e in get_stage_errors(session_id).items() if s not in completed}
        except Exception as e:
            logger.warning("Could not read stage errors: %s", e)
        try:
            in_process = [str(getattr(result, f)) for f in _ERROR_FIELDS if getattr(result, f)]
            # Stage rows failed by another process of this session (a --resume
            # run after the pilot driver's stage-end, or an earlier attempt)
            # count too; otherwise a session with a failed strategist row could
            # end 'completed' because this process only ran the later stages.
            from_rows = [f"{s}: {e}" for s, e in prior_errors.items()
                         if not getattr(result, _STAGE_ERROR_FIELD.get(s, ""), None)]
            if in_process or from_rows:
                fail_session(session_id, "; ".join(in_process + from_rows))
            else:
                complete_session(session_id)
        except Exception as e:
            logger.warning("Could not update session status: %s", e)
```

and, so `main()` exits 1 in that case, set `result.prior_stage_errors = prior_errors` just before the `try`. Add the field to `SessionResult` (import `field` from `dataclasses`):

```python
    prior_stage_errors: dict = field(default_factory=dict)  # failed stage rows from an earlier run of this session

    @property
    def has_errors(self) -> bool:
        return any(getattr(self, f) for f in _ERROR_FIELDS) or bool(self.prior_stage_errors)
```

Add the stage-to-field map next to `_ERROR_FIELDS`:

```python
_STAGE_ERROR_FIELD = {
    "learning": "learning_error", "supervisor": "supervisor_error", "pipeline": "pipeline_error",
    "strategist": "strategist_error", "executor": "trading_error", "strategy": "strategy_error",
    "dashboard": "dashboard_error",
}
```

In `run_session` add `resume: bool = False` after `force: bool = False`, and change the call to `_check_and_record_session(force, today, resume=resume)`. In `main()` add `parser.add_argument("--resume", action="store_true", help="Reopen today's session row and skip stages it already completed")` and pass `resume=args.resume`.

- [ ] **Step 6: Extend the conftest safety net**

```python
# tests/v2/conftest.py — add to _SESSION_DB_PATCH_TARGETS
    "v2.session.get_completed_stages",
    "v2.session.get_stage_errors",
    "v2.session.reopen_session",
```

and after `session_module.close_orphan_running_stages.return_value = []` add:

```python
        session_module.get_completed_stages.return_value = set()
        session_module.get_stage_errors.return_value = {}
```

(Without these defaults every existing session test would see a truthy `MagicMock` from `get_stage_errors` and fail the session.)

- [ ] **Step 7: Run the tests**

Run: `task test INSTANCE=paper -- tests/v2/test_session.py tests/v2/test_db.py tests/test_conftest_gate.py -v && task lint`
Expected: PASS; lint clean.

- [ ] **Step 8: Document the flag**

In `CLAUDE.md` under "Commands", after the `--force` example add:

```bash
# Continue today's session where it stopped: reopens the existing row and
# skips stages it already completed (a --force run would redo everything)
task session INSTANCE=paper -- --resume
```

and in the "v2 Daily Session" paragraph on idempotency append: `--resume` reopens the day's latest session row and skips its completed stages; `--force` starts a fresh row and reruns everything.

- [ ] **Step 9: Commit**

```bash
git add v2/session.py v2/database/trading_db.py tests/v2/test_session.py tests/v2/test_db.py tests/v2/conftest.py CLAUDE.md
git commit -m "Add --resume: reopen the day's session and skip completed stages

--force keeps its per-run semantics. _finalize_session now counts failed
stage rows left by an earlier process of the same session.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Verification and merge

1. `task test INSTANCE=paper` — full suite green; `task lint` clean.
2. `task session INSTANCE=paper -- --force` on a trading day with credits available: a `completed` session row with all seven stage rows; `session_stage_costs` shows the strategist priced.
3. Replay check for the executor seam: `docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading python -m v2.trader --emit-input /app/logs/executor-input.json --dry-run` writes the input; hand-edit a decisions file with an empty `decisions` list and run `--decisions-file … --input-file … --dry-run`; expect exit 0 and no orders.
4. `task session INSTANCE=paper -- --resume` immediately after a completed run: every stage logs `SKIPPED (completed in prior run)`, the session row stays `completed`.
5. Open the PR from this branch against `main`. The strategist pilot plan starts from the merge commit.
