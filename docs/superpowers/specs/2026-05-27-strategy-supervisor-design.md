# Strategy Supervisor — Design

**Date:** 2026-05-27
**Status:** Design (pre-implementation)

## Summary

A higher-level critic that audits Pinchy's strategy stack — rules, theses, identity, behavior, and the reflection stage itself — from a vantage point above the daily session. The supervisor is **observer-only**, **on-demand**, and produces a **free-form markdown memo** per run, surfaced on the v2 dashboard.

It complements (does not replace) the existing audit-loop in `docs/audit-playbook.md`. That system is integrity- and telemetry-focused, files Jira tickets, and runs on a 24h `/loop`. The supervisor is reasoning-heavy strategic critique, addressed to the human operator, with no autonomous cadence.

## Motivation

The reflection stage (`v2/strategy.py`) updates identity, proposes/retires rules, and writes memos at the end of each session. It works inside one session's view. Patterns that emerge across many sessions — Rule #27 oscillating and driving GOOGL to flip-flop 11x in 22 days; rules churned in and out within a week; identity statements that whipsaw between memos — slip past it. Today those patterns are caught only when the human happens to look.

The supervisor is a deliberate vantage point above the reflection stage. It is granted observer authority only: it writes a memo, it does not mutate strategy state.

## Goals

- Critique four areas: rule coherence & quality, thesis discipline, identity + behavior drift, reflection quality.
- Investigate before opining — the supervisor uses read-only DB tools to verify any pattern it claims, and cites specific IDs and dates in the memo.
- Run on-demand only (CLI / Taskfile target). Not wired into `v2/session.py` and not scheduled.
- Surface memos on the v2 dashboard with permalinks.

## Non-goals

- No write access to strategy state. The supervisor cannot retire rules, close theses, or update identity.
- No automatic cadence. No cron, no `/loop`, no session-stage wiring.
- No structured findings, severity levels, or trend charting in v1. Pure free-form memo. Structured outputs can be a follow-up if the memo proves useful.
- Not feeding critique into the strategist's next-session context. Audience is the human operator only.
- No new dashboard charts, only a new memo page.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ CLI: python -m v2.supervisor                               │
└────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────┐
│ v2/supervisor.py                                           │
│   - builds tool registry (read-only get_* tools)           │
│   - calls claude_client.run_agentic_loop                   │
│   - persists memo + run metadata to supervisor_memos       │
└────────────────────────────────────────────────────────────┘
           │            │
           ▼            ▼
   ┌──────────────┐   ┌──────────────────┐
   │ Anthropic    │   │ PostgreSQL       │
   │ Opus 4.7     │   │ - reads strategy │
   │              │   │   state via tools│
   │              │   │ - INSERT one     │
   │              │   │   supervisor_memo│
   └──────────────┘   └──────────────────┘
                              │
                              ▼
                      ┌──────────────────────┐
                      │ v2/dashboard/        │
                      │   /supervisor        │
                      │   /supervisor/<id>   │
                      └──────────────────────┘
```

## Components

### `v2/supervisor.py`

- `STRATEGY_SUPERVISOR_SYSTEM`: critic system prompt (see below).
- `PROMPT_VERSION`: hand-bumped string. Increment any time the system prompt or tool catalog changes meaningfully.
- `DEFAULT_SUPERVISOR_MODEL = "claude-opus-4-7"`. Overridable via `--model` flag and `ALGO_SUPERVISOR_MODEL` env var (mirroring the executor knob).
- `DEFAULT_MAX_TURNS = 20`.
- `build_supervisor_tools()` → returns the tool-spec list + tool-handler dict. Pulls from helpers already present in `v2/tools.py`, `v2/patterns.py`, `v2/strategy.py`, `v2/attribution.py`. **Never includes** any tool from the strategy mutator set.
- `run_supervisor(model, max_turns, dry_run)` → orchestrates the agentic loop, extracts final text via `claude_client.extract_final_text`, inserts a row into `supervisor_memos`. Returns the inserted row's id.
- `main()` for `python -m v2.supervisor`. Flags: `--model`, `--max-turns`, `--dry-run` (runs the loop but skips the INSERT — useful for prompt iteration).

### System prompt (v1)

```
You are the Strategy Supervisor for Pinchy, an agentic trading system.

Your role is to critique the trading strategy from a senior, skeptical
vantage point. You read state — you do not change it. There are no
write tools available to you.

Your four areas of focus:

1. Rule coherence & quality
   - Do active rules contradict each other?
   - Any rule that oscillates (binds/lifts repeatedly within days)?
   - Any active-but-dormant rule that hasn't bound in 30+ days?
   - Any rule churned out within a week of being added?
   - Is each active rule grounded in evidence or pet theory?

2. Thesis discipline
   - Are theses honored at their stated entry/exit triggers?
   - Any thesis lingering past its entry window with no action?
   - Do any active theses contradict each other?
   - Are closed theses being learned from?

3. Identity + behavior drift
   - Is the strategy identity coherent across recent memos, or whipsawing?
   - Does what the executor actually does (sizing, flip-flops, sector mix,
     round-trip frequency) match the identity?

4. Reflection quality
   - Did the recent reflection stages take action, or coast?
   - Did they ignore obvious problems (flip-flops, dormant rules)?
   - Are memos substantive or vacuous?

Investigate before you opine. Use get_* tools to verify any pattern
you suspect — pull bind histories, decision detail, thesis lineage.
Cite specific rule_ids, thesis_ids, decision_ids, and dates in your
critique. A claim without a citation should not appear in the memo.

Be direct. Don't soften. The point of this role is to surface what
the reflection stage missed. If you find nothing wrong, say so plainly —
do not invent concerns to seem thorough. A short "no major concerns
this week, here's why" memo is more valuable than a padded one.

Output: a single markdown memo with sections matching the four areas
above. Skip a section entirely if you have nothing to say about it.
End with a "Watchlist" section: 1-5 specific things to revisit on
the next supervisor run.
```

### Tool catalog (read-only)

Strategy state:
- `get_strategy_identity()` — current identity text + last 5 versions with timestamps.
- `get_active_rules()` — id, body, status, proposed_at, evidence, lift_condition.
- `get_retired_rules(limit=50)` — as above plus retired_at and retirement reason.
- `get_rule_bind_history(rule_id, days=30)` — count and dates of decisions that cited the rule.

Theses:
- `get_theses(status='all'|'active'|'closed', limit=50)` — id, ticker, hypothesis, entry/exit triggers, status, created_at, closed_at, closure reason.
- `get_thesis_lineage(thesis_id)` — decisions tagged with this thesis with outcomes.

Behavior:
- `get_recent_decisions(days=14)` — compact rows (date, ticker, side, qty, reasoning_excerpt ≤200 chars, signals_referenced, realized_pnl).
- `get_decision_detail(decision_id)` — full reasoning + all referenced signals for one decision.
- `get_flip_flop_report(days=30, min_reversals=3)` — tickers with N+ reversals, with the reversal dates and reasoning excerpts.
- `get_executor_behavior_summary(days=14)` — size-distribution histogram, sector concentration, round-trip count, hold-rate.
- `get_signal_attribution()` — current scores per signal_type / category.

Reflection & sessions:
- `get_session_memos(limit=10)` — last N strategy memos.
- `get_reflection_actions(limit=10)` — per-session: rules proposed, retired, revalidated; identity updated y/n; memo word count.
- `get_session_summary(days=14)` — per-session: decisions count, P&L, stage failures, cost.

14 tools. None of them are write tools.

### Persistence

New table:

```sql
CREATE TABLE supervisor_memos (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    model         TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    content       TEXT,
    status        TEXT NOT NULL,     -- 'ok' | 'max_turns' | 'error'
    turns_used    INT NOT NULL,
    tool_calls    JSONB NOT NULL,    -- [{"name": "get_active_rules", "count": 2}, ...]
    input_tokens  INT,
    output_tokens INT,
    cost_usd      NUMERIC(10,4),
    error_message TEXT
);
CREATE INDEX supervisor_memos_created_idx ON supervisor_memos (created_at DESC);
```

Delivered as `db/migrations/012_supervisor_memos.sql` for existing databases and as a corresponding numbered file in `db/init/` for fresh databases — same two-track convention used throughout the project.

`content` is NULL when `status != 'ok'`. `tool_calls` is a compact summary, not a full transcript.

### Dashboard surface

- `GET /supervisor` — renders the most recent memo. Header shows date, model, prompt version, turns used, cost. Body is the rendered markdown. Sidebar lists the last 10 memos with dates linking to permalinks.
- `GET /supervisor/<id>` — permalink to a specific memo.
- Linked from the existing internals page navigation.
- Markdown rendering uses the project's existing markdown library if present; otherwise add `markdown` to `v2/requirements.txt`.

No charts in v1. Trend charting is a deliberate follow-up if the memos prove useful enough to justify it.

## Data flow

1. Operator invokes `python -m v2.supervisor` (or `task supervise`).
2. `run_supervisor` builds the read-only tool registry and calls `claude_client.run_agentic_loop` with the critic system prompt and `max_turns=20`.
3. Opus 4.7 alternates between tool calls and reasoning until it produces a final text message (or hits the turn cap).
4. The module extracts the final markdown via `extract_final_text`, computes `tool_calls` summary from the loop's recorded calls, captures token usage and cost from the existing `_record_usage` capture, and INSERTs one row into `supervisor_memos`.
5. The dashboard's `/supervisor` page reads the latest memo on its next render.

## Error handling

- **Max turns hit without final text:** row inserted with `status='max_turns'`, `content=NULL`, `turns_used=20`, `error_message='loop did not produce final text within max_turns'`. Dashboard shows "last run incomplete" with link to inspect.
- **Anthropic API failure:** row inserted with `status='error'`, the truncated exception message in `error_message`.
- **DB error during a tool call:** caught by the tool handler, surfaced to the loop as a tool error result (existing `claude_client` machinery handles this), loop continues. No special-casing.
- **`--dry-run`:** loop runs as normal; the final INSERT is skipped and the memo text is printed to stdout.

## Cost & safety controls

- `max_turns=20` hard cap on the agentic loop.
- A unit test asserts that the supervisor's registered tool dict has zero overlap with the strategy mutator set (`tool_propose_rule`, `tool_retire_rule`, `tool_revalidate_rule`, `tool_update_strategy_identity`, `tool_write_strategy_memo`). Defense-in-depth against a future contributor accidentally wiring a write tool into the supervisor.
- No DB-level read-only role introduced. The contract is enforced at the tool registry. (A dedicated read-only role can be added later if warranted; not v1 scope.)
- No cron, no `/loop`, no session-stage wiring. The supervisor only runs when a human runs it.

## Testing

- Unit tests for each `get_*` tool against a seeded test DB, in the existing `tests/` layout. Most underlying SQL already has coverage via the helpers in `v2/tools.py`; add tests for the few tools whose shape is new (e.g. `get_rule_bind_history`, `get_reflection_actions`).
- A `test_supervisor.py` module asserts:
  - The tool dict contains only `get_*` tools.
  - The tool dict has zero overlap with the strategy mutator set.
  - On a mocked agentic loop returning final text, an `ok` row is inserted with the expected metadata.
  - On a mocked max-turns outcome, a `max_turns` row is inserted with `content=NULL`.
  - `--dry-run` does not INSERT.
- One integration test against a fixture DB containing a known oscillating rule, a dormant rule, and a thesis past its trigger window. Assert (loosely) that the memo content references those IDs. Exact prose is not asserted — the test verifies that the supervisor investigates and cites, not that it produces specific words.

## Open questions

None at design time. Resolved during brainstorming:
- Authority: observer only.
- Cadence: on-demand.
- Scope: all four areas (rules, theses, identity+behavior, reflection quality).
- Output: free-form memo per run.
- Audience: human operator via dashboard only.
- Implementation: agentic Opus with read-only DB tools (Approach 2).

## Follow-up candidates (not in scope)

- Structured findings with severity and "still open" tracking across runs.
- Trend widgets on the dashboard (rule churn over time, identity-diff visualization).
- Soft loop: inject the most recent supervisor memo into the next strategist context.
- Weekly cadence via cron once the memo quality is proven.
- Dedicated read-only DB role.
