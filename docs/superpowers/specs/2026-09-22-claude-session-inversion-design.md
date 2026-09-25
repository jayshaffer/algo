# Claude Session Inversion — Design Spec

**Date:** 2026-09-22
**Status:** Accepted in two phases (2026-09-22). Phase one, the seams, is
`docs/superpowers/plans/2026-09-22-session-seams.md`; phase two pilots
the strategist only (`docs/superpowers/plans/2026-09-22-strategist-pilot.md`)
before any other stage moves. The full plan
(`2026-09-22-claude-session-inversion.md`) is the reference for later stages.
**Scope:** Run every LLM stage of the daily session as a headless Claude Code
session under the operator's Claude subscription, with Python reduced to
deterministic tooling. Cron and `cron-wrap.sh` stay the trigger.

## Problem

Every LLM call in the v2 session goes through `v2/claude_client.py` against
`ANTHROPIC_API_KEY`, billed per token. A completed paper session costs
roughly $2–9 on the API; the strategist's Opus loop (600–720k input tokens,
mostly cache reads) is nearly all of it. The operator has a Claude Max 5x
subscription that covers Claude Code usage — interactive, headless `claude
-p`, and cloud routines — but not the Python SDK or Agent SDK pointed at
subscription credentials. So the only supported way to run this work on the
subscription is to make Claude Code the thing that reasons, and make the
Python a set of tools it calls.

Constraints that shape the design:

- **Cloud routines cannot reach this host.** They run in Anthropic's sandbox
  with no path to the WSL host, its docker stack, `instances/*.env`, or the
  Postgres volume. Execution stays on the host.
- **`cron-wrap.sh` is non-negotiable.** The 2026-08 audit found the system
  dead for two months because nothing watched the host. HALT sentinels, the
  dead-man's heartbeat, and failure alerting all live in the wrapper. The new
  entry point sits *under* it, never beside it.
- **The money path stays in Python.** Risk sizing, sector cap, churn gate,
  dedup, `client_order_id` idempotency, the daily-loss breaker, and order
  submission are tested Python. No Claude session places or cancels an order.
- **Headless mode must not be `--bare`.** Bare mode never reads OAuth
  credentials. Verified on this host (2026-09-22): a non-bare `claude -p`
  under an empty environment (`env -i HOME=… PATH=…`) authenticates via
  `~/.claude/.credentials.json` and reports `provider: firstParty`. The docs
  note `--bare` "will become the default for `-p` in a future release", so
  the driver pins its mode explicitly and a smoke test guards it.
- **Claude Code 2.1.280 has no `--max-turns`.** Turn caps move server-side.

## Goals

- Every LLM stage runs as a `claude -p` call under the subscription:
  supervisor, classifier, strategist, executor, reflection, changelog.
- Cron path unchanged: `cron-wrap.sh --instance <name> <name>-session <driver>`.
- Session bookkeeping (`sessions`, `session_stages`, idempotency, `--force`,
  halt and publish gates, per-stage validators) keeps its current semantics.
- Cost and trace telemetry survive: `session_stages` token columns, the
  `session_costs` views, `llm_call_contexts`, `agent_events`.
- The same stage skills are usable interactively in a normal Claude Code
  session (`/pinchy-session paper`, `/pinchy-strategist paper`).
- The existing API backend remains callable via `task session` for one
  release as a fallback, sharing prompts with the skills so they cannot drift.

## Non-goals

- Moving the stack off the host, hosted Postgres, or cloud routines.
- A "Claude as operator" layer that triages failures and retries on its own.
  The driver is deterministic; a human reads the alert.
- Per-call granularity in `llm_call_contexts` for agentic stages. One row
  per stage is the new contract for inverted stages.
- Rewriting stage prompts. They move verbatim into skill files; prompt
  changes are a separate concern.
- Replacing `v2/session.py` in this change. It stays as the API fallback
  and is retired in a follow-up once paper has run clean on the driver.

## Architecture

```
cron ──▶ cron-wrap.sh --instance paper paper-session ./session-driver.sh paper
                │  (HALT, heartbeat, alert — unchanged)
                ▼
        session-driver.sh  (host, bash, no decisions)
                │
                ├─ docker compose exec trading python -m v2.session_ctl plan
                │       ◀── JSON: session_id, ordered stages, skips, gates
                │
                └─ for each stage:
                     session_ctl stage-begin <stage>
                     ┌─ deterministic ──▶ docker compose exec trading python -m v2.<module> …
                     └─ llm ────────────▶ claude -p … (host, subscription auth)
                                               │ agentic stages: --mcp-config ──▶ docker compose exec -i trading
                                               │                                  python -m v2.mcp_server --registry <stage>
                                               │ structured stages: --system-prompt <file> --json-schema … < input.json
                                               ▼
                                          result.json (+ transcript.jsonl)
                     session_ctl stage-end <stage> --result result.json [--transcript …]
                             (validators, usage → session_stages, contexts → llm_call_contexts)
                   session_ctl finalize
```

Three new pieces of Python, one new shell script, six skill files, two
compose lines, one crontab line.

### `session-driver.sh` (host)

Usage: `./session-driver.sh <instance> [--force] [--dry-run] [--skip-<stage>…]`.

Deliberately dumb. It sources `instances/<name>.env` (for `INSTANCE`, the
`ALGO_STAGE_MODEL_*` knobs, and nothing else it acts on), builds the
project-scoped compose prefix exactly as `run-docker.sh` does, and then:

1. `plan` — runs `session_ctl plan` with the CLI flags passed through and
   parses the JSON reply. An empty stage list with `skip_reason` set is a
   no-op exit 0 (halted, idempotent skip). This is the only place the driver
   branches.
2. For each stage in order: `stage-begin`, run the stage's command,
   `stage-end`. A nonzero stage command never aborts the loop; it is passed
   to `stage-end --error <text>` and the loop continues.
3. `finalize` — returns the session exit code (0 clean, 1 any stage error),
   which the driver exits with so `cron-wrap.sh` alerts as today.

Every `claude -p` invocation is built by one function so the mode is pinned
in one place:

```
claude -p [--system-prompt "$(cat v2/prompts/<stage>.md)" | "/pinchy-<stage> <instance>"]
    --model "$model"
    --output-format json            # or stream-json for agentic stages
    --permission-mode dontAsk --permission-prompts none
    --allowedTools "$tools" --strict-mcp-config [--mcp-config "$cfg"]
    --disallowedTools "AskUserQuestion,Edit,Write,NotebookEdit,Agent"
```

Structured stages (classifier, executor, changelog) run from a neutral
scratch directory with the tool list emptied (`--tools` with no names;
if the CLI rejects an empty list, every built-in tool is named in
`--disallowedTools`) so no project skills, `CLAUDE.md`, MCP
servers, or file tools leak into a pure classification or decision call.
Agentic stages run from the repo root so the skill and MCP config load.

Each `claude` call runs under `timeout` with a per-stage wall-clock cap
(defaults: supervisor 20m, strategist 40m, reflection 20m, structured
stages 5m each). A timeout is a stage failure like any other.

Logs: the driver appends to `logs/<instance>/session.log` as `task session`
does today, and keeps each stage's `result.json` and transcript under
`logs/<instance>/sessions/<date>/<stage>/` for forensics.

### `v2/session_ctl.py` (container)

Owns every decision the driver used to get from `run_session`. Subcommands,
all printing JSON on stdout:

- `plan [--force] [--dry-run] [--skip-…]` — applies, in this order:
  `ALGO_TRADING_HALTED` (exit 0, `skip_reason`), dry-run promotion of
  skips, the `ALGO_DASHBOARD_PUBLISH` gate, `expire_stale_playbook_actions`,
  then `_check_and_record_session(force, today)`. Returns
  `{session_id, session_date, stages: [{name, kind, command|prompt, model,
  skip, skip_reason}], completed_stages}`. The `completed_stages` set is now
  populated from `get_completed_stages` so a `--force` re-run resumes past
  completed stages instead of redoing them; this activates the resume logic
  that exists today but is never reached in production (`completed_stages`
  is always empty in `run_session`).
- `stage-begin <stage>` — `insert_session_stage(running)`.
- `stage-context <stage>` — prints the stage's input. Strategist: the seven
  pre-seeded sections, orphan block, open watchlist block, attribution
  constraints, formation context. Reflection: trading context, revalidation
  candidates, open watchlist. Executor: `build_executor_input` plus risk
  notes (delegates to `trader.py --emit-input`). Supervisor: none.
- `stage-end <stage> --result <json> [--transcript <jsonl>] [--error <text>]`
  — runs the stage's validator, records usage, writes contexts and events,
  then `complete_session_stage` or `fail_session_stage`. Validators are the
  existing ones, moved: strategist requires `get_playbook(session_date)`
  and persists the strategist memo only after that check (P2.24 ordering);
  reflection requires a memo row for this session and no open reflection
  watchlist items, plus the gated-rule revalidation check; supervisor
  requires the memo id that `write_supervisor_memo` reported in
  `tools.jsonl` to exist in `supervisor_memos`; executor requires a
  `TradingSessionResult` with no `LLM decision failed` error.
- `finalize` — `close_orphan_running_stages`, `complete_session` or
  `fail_session`, `_log_session_costs`, the telemetry summary line. Prints
  `{exit_code}`.

`session_ctl` imports the stage helpers from `session.py` rather than
copying them; `session.py` keeps working for the API path. Model selection
per stage is `ALGO_STAGE_MODEL_<STAGE>` from the instance env with defaults
matching today: supervisor `claude-fable-5`, strategist `claude-opus-4-8`,
reflection `claude-sonnet-4-6`, classifier/executor/changelog
`claude-haiku-4-5-20251001` (executor honours `ALGO_EXECUTOR_MODEL` first).
`plan` emits the resolved model per stage so the driver never guesses.

### `v2/mcp_server.py` (container, stdio)

`python -m v2.mcp_server --registry {strategist,reflection,supervisor}
--session-id N`, started by Claude Code via `--mcp-config` as
`docker compose -p pinchy-<i> --env-file instances/<i>.env exec -i trading …`.
Uses the `mcp` Python SDK (new dependency in `v2/requirements.txt`).

- Tool schemas come straight from the existing definitions:
  `TOOL_DEFINITIONS` (strategist), `STRATEGY_TOOL_DEFINITIONS` (reflection),
  `build_supervisor_tool_defs()` (supervisor), with `input_schema` renamed
  to `inputSchema`. A parity test asserts the served tool list equals the
  registry the API loop uses, minus the exceptions below.
- Handlers are the existing functions. Session-bound ones (`create_thesis`,
  `adopt_thesis`, `write_strategy_memo`, `resolve_watchlist_item`,
  `get_session_summary` for reflection) are bound with `--session-id`
  exactly as `ideation_claude.py` and `strategy.py` do with `partial`.
- `web_search` is not served; the strategist skill uses Claude Code's own
  `WebSearch`, capped at six uses by the skill text (the API tool's
  `max_uses`) and by the server refusing further `get_curated_news` calls
  once the turn cap is near.
- **Turn cap.** The server counts tool calls per process and, past the
  stage's cap (strategist 60, reflection 25, supervisor 40 — tool calls, not
  API turns, so roughly 2× today's `max_turns`), returns an error result
  telling the model to finish with its terminal tool. Wall-clock is the
  driver's backstop.
- **Supervisor memo.** Today the supervisor's memo is the loop's final text
  and its watchlist items are buffered until the memo row exists. The
  server adds `write_supervisor_memo(content)`: inserts the row with
  `status='ok'`, flushes the buffered `record_watchlist_item` calls with
  `source_memo_id`, and returns the id. `stage-end` fails the stage (`[validator]`) if
  the cap fired before the memo was written.
- Every tool call is appended to a per-process JSONL the server writes to
  `/app/logs/…/<stage>/tools.jsonl` (name, args, success, error,
  duration_ms, output_chars) — the same fields `run_agentic_loop` puts in
  `tool_invocation` events. `stage-end` ingests it.
- `reset_session()` runs at server start (clears the curated-news cache).

### Prompts and skills

Six skill files under `.claude/skills/pinchy-<stage>/SKILL.md`:
`pinchy-session` (interactive entry: runs the driver and reports),
`pinchy-supervisor`, `pinchy-strategist`, `pinchy-reflection`, and
`pinchy-executor` / `pinchy-classify` (interactive wrappers around the
structured calls, for debugging a single stage by hand).

The system prompt text for supervisor, strategist, and reflection moves
verbatim from the Python constants into the skill body, below a short
preamble that tells the session how to fetch its context
(`session_ctl stage-context <stage>`) and which tool ends the stage
(`write_playbook`, `write_strategy_memo`, `write_supervisor_memo`). The
Python constants become loaders: `v2/prompts.py: load_stage_prompt(stage)`
reads the skill body from `/app/.claude/skills/pinchy-<stage>/SKILL.md`,
which requires one compose line, `./.claude/skills:/app/.claude/skills:ro`,
on the trading service. A test asserts the API backend's system prompt
equals the skill body after the preamble, so the fallback cannot drift.

Structured-stage prompts (`BATCH_CLASSIFICATION_SYSTEM`,
`TRADING_SYSTEM_PROMPT`, the changelog instruction) move to
`v2/prompts/<stage>.md`, loaded by both the API path and the driver
(`--system-prompt "$(cat …)"`; the CLI has `--system-prompt` and
`--append-system-prompt-file` but no `--system-prompt-file`, so the driver
reads the file itself).

### Structured stages

**Executor.** `trader.py --emit-input` prints `ExecutorInput` as JSON (the
same `json.dumps(input_data, default=str)` the API path sends). The driver
pipes it to `claude -p --model haiku --system-prompt … --json-schema
<executor schema> --tools ""`. The schema is generated from
`EXECUTOR_KNOWN_TOP_KEYS` / `EXECUTOR_KNOWN_DECISION_KEYS` and the enum
constants in `agent.py`, so schema drift is caught where it is defined.
`trader.py --decisions-file <json>` parses with the existing
`_validate_decision_contract` path, then runs `_validate_llm_ids` →
`_execute_decisions` → `_handle_thesis_invalidations` → `_log_decisions` →
`_build_final_result`, i.e. `run_trading_session` from step 4 on. The
daily-loss breaker check at session start runs inside `--decisions-file`
before any order. Malformed output is a stage failure with no re-prompt,
matching today's behaviour.

**Classifier.** `pipeline.py --emit-batches <dir>` fetches news and writes
`batch-NN.json` files of up to 50 sanitized headlines with their alpaca
ids. The driver runs one `claude -p --model haiku --json-schema` per batch.
`pipeline.py --ingest <dir>` runs the existing index remap, category
validation, and ticker resolution, then the batch inserts. A batch whose
call failed is marked noise, as a rate-limited batch is today. Six calls
per session at the default limit of 300.

**News curation.** `tool_get_curated_news` currently calls Haiku through
the API from inside the strategist's tool. It becomes deterministic: rank
candidates by confidence then recency, return `target_n`, keep the `[#id]`
rendering. The strategist does its own relevance judgement; the filter's
own fallback is already "return everything".

**Changelog.** `dashboard_publish.py --emit-changelog-commits` prints the
commit payload; the driver runs one `claude -p --json-schema` call;
`dashboard_publish.py --changelog-entries <json>` validates with
`validate_changelog_entries`, stores, renders, deploys, and advances the
pointer. The dashboard stage's usage is recorded for the first time.

### Telemetry

- **Usage.** `--output-format json` returns `usage` and `modelUsage` with
  input, output, cache-creation, and cache-read tokens per model.
  `stage-end` writes them to `session_stages` through
  `complete_session_stage(usage=…)` by building a `UsageAccumulator` from
  the result. The `session_stage_costs` view then prices them from
  `model_pricing`, showing the list-price equivalent — what the day would
  have cost on the API, which is the number worth tracking now.
  `tests/test_pricing_coverage.py` extends to the model ids the driver can
  emit.
- **Contexts.** Agentic stages run with `--output-format stream-json
  --verbose` captured to `transcript.jsonl`. `stage-end` folds it into one
  `llm_call_contexts` row per stage: system prompt = skill body, messages =
  the reconstructed turn list, `tool_definitions` = the served registry,
  `response_content` = the final assistant message, `stop_reason` from the
  result, `duration_ms` from `duration_api_ms`. Purpose strings stay
  `strategist_loop`, `reflection_loop`, `executor` so the internal
  dashboard's ordering and the `LLM_CONTEXT_MISSING_ROWS_FOR_PURPOSE` audit
  check keep working. `stage_name` values stay `ideation`/`trading`/
  `reflection` to match existing rows.
- **Events.** `stage-end` emits `agent_call` (one per stage),
  `tool_invocation` (from `tools.jsonl`), and `loop_completion`. The
  executor's `executor_response` event is emitted by `--decisions-file`
  where the parse happens.
- **Lost, and accepted.** Per-call rows inside a loop, `loop_recovery`
  events (Claude Code manages its own context), and the API-side
  `cache_hit` ratio semantics. `CACHE_HIT_RATIO_DEGRADATION` in the audit
  playbook gets a note that inverted stages report Claude Code's cache
  figures, which are not comparable to the old series.

### Failure handling

- Stages remain independent. A failed `claude` call (nonzero exit, `is_error`
  in the result, timeout, or a usage-limit rejection) fails only that stage;
  the loop continues; `finalize` returns 1; `cron-wrap.sh` alerts.
- `stage-end` classifies the failure text so the alert says what happened:
  `rate_limit`/`usage_limit` from the result's error category, `timeout`,
  `auth` (credential file missing or rejected — the driver checks
  `~/.claude/.credentials.json` exists before the first LLM stage and fails
  the session early with a clear message), `validator` (stage ran but did
  not leave its artefact), `model_error`.
- Stage order is unchanged (learning → supervisor → pipeline → strategist →
  executor → reflection → dashboard). Reflection is last among the LLM
  stages, so a usage-window cutoff late in a run costs the cheapest stage
  to retry.
- `--force` next day resumes past completed stages (see `plan`). The
  executor's broker-side `client_order_id` and pre-submit dedup remain the
  guard against a double-run placing duplicate orders.
- HALT: both host sentinels are honoured by `cron-wrap.sh` before the
  driver starts; `ALGO_TRADING_HALTED` is honoured by `plan`. Nothing new.

### Usage limits

Max 5x has no published token budget and no pre-flight check. The design
does not pretend otherwise. Mitigations: stage independence (above), the
alert classification so a limit hit reads as a limit hit, and a note in
`docs/runbook-recovery.md` that interactive Claude Code use in the hour
before the 12:30 MST session shares the window with it. If the limit is
hit regularly, the levers are the strategist's context size and model, not
this design.

## Testing

All under the existing docker test invocation; nothing reaches the network
(`tests/v2/conftest.py` gate). New/extended:

- `tests/v2/test_session_ctl.py` — `plan` (halt, dry-run promotion, publish
  gate, idempotency with and without `--force`, completed-stage resume,
  resolved models), `stage-end` (each validator's pass/fail, usage recording
  from a fixture `result.json`, context row from a fixture transcript, error
  classification), `finalize` exit codes.
- `tests/v2/test_mcp_server.py` — served tool list equals each registry;
  session binding; turn cap returns the finish-now error; supervisor memo
  tool flushes the watchlist buffer in order; `tools.jsonl` fields.
- `tests/v2/test_trader.py` — `--decisions-file` reuses the existing
  `AgentResponse` fixtures and asserts the same call sequence as
  `run_trading_session`; the breaker runs first; malformed file fails
  without touching the broker mock.
- `tests/v2/test_pipeline.py` — `--emit-batches` batch shape; `--ingest`
  index remap, invalid category drop, failed-batch-as-noise.
- `tests/v2/test_prompts.py` — each API-path system prompt equals its skill
  body; `v2/prompts/*.md` load; schema generation matches `agent.py` enums.
- `tests/test_pricing_coverage.py` — extended model set.
- `scripts/smoke-claude-headless.sh` — host-side, not in pytest: runs one
  `claude -p` under `env -i` and asserts `provider: firstParty` and a
  non-bare mode, then runs the executor path with `--dry-run` against
  `paper`. This is the guard for the `--bare`-becomes-default change.

## Rollout

1. Land the Python surface (`session_ctl`, `mcp_server`, `trader`/`pipeline`
   flags, prompts loader, compose mount) with `task session` still the cron
   entry. Tests green.
2. Land the driver and skills. Run `./session-driver.sh paper --dry-run` by
   hand, then a full paper run by hand on a day the cron has already run
   (`--force` resumes past completed stages).
3. Switch the paper crontab line to
   `./cron-wrap.sh --instance paper paper-session ./session-driver.sh paper`.
   `live` stays on `task session` behind `instances/live.HALT`.
4. After five clean paper sessions, and after the live hiatus review in the
   runbook, move `live`'s line. Add both to the runbook's halt/resume steps.
5. Follow-up: retire the API backend and `run_session`'s LLM branches;
   remove `anthropic` from the executor/strategist path if nothing else
   needs it. Not this change.

Prerequisite, separate fix: paper sessions have failed nearly every day
since 2026-08-25 and hit exit 201 from 2026-09-16 to 2026-09-22 because the
installed crontab still called the pre-cutover `task paper:session`. The
installed crontab now matches the repo, but today's row is `failed`. Triage
that first so the driver is measured against a working baseline.

## Open questions resolved in this spec

- Where does orchestration live? Host bash loop + container `session_ctl`;
  the loop never decides, it only sequences what `plan` returns.
- Subagents vs. one `claude -p` per stage? One call per stage: exact model
  and usage per stage, fresh context per stage as today, and a usage-limit
  hit fails one stage rather than the run.
- Skill prompts vs. Python prompts? Skills are the source; Python loads
  them through a read-only mount; a test enforces parity.
- What replaces `max_turns`? A tool-call cap in the MCP server plus a
  wall-clock timeout in the driver.
