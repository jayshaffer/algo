# Supervisor Watchlist → Closed-Loop Action — Design

**Date:** 2026-05-27
**Status:** Design (pre-implementation)

## Summary

Close the loop between the strategy supervisor's critique and the daily
session's two acting stages. Today the supervisor (`v2/supervisor.py`)
writes a sharp markdown memo with an explicit watchlist, but **nothing
reads it** — the memo is write-only into `supervisor_memos` and the
dashboard. Separately, the **reflection stage** has settled into a
pattern of long, well-reasoned memos that take essentially no action
(5 consecutive zero-action reflections; the same Rule 43 / GOOGL
auto-lift discrepancy flagged across 3 sessions with no resolution).

This design makes the supervisor emit a **structured watchlist** whose
items each carry an **owning stage**, and makes each owning stage
(strategist/ideation and reflection) **ingest its open items and resolve
them before the stage may complete**. The recurring stale-state problems
(Rule 43, theses 233/267, Rule 49) become the validation that the loop
now works.

This is the explicit follow-up to two non-goals deferred in
`2026-05-27-strategy-supervisor-design.md`:
> "No structured findings… Structured outputs can be a follow-up if the memo proves useful."
> "Not feeding critique into the strategist's next-session context."

Both are promoted to goals here.

## Motivation

The supervisor's most recent memo (memo id 1, 2026-05-28) is accurate —
its claims check out against the live DB: Rule 43 is still `active`
despite its auto-lift clause having fired, and theses 233/267 are still
`active` though three reflections flagged them. The problem is not
diagnosis quality; it is that **diagnosis has no path to action.**

Two breaks in the observe→act loop:

1. **Supervisor → nowhere.** `supervisor_memos` is read only by the
   dashboard. No session stage consumes it.
2. **Reflection's existing forcing function is too narrow.** The gate at
   `v2/strategy.py:782-787` raises `RuntimeError` if reflection finishes
   without revalidating a "gated" rule — but "gated" only means a rule
   with **no `lift_condition`** that keeps getting cited. Rule 43 *has* a
   `lift_condition` (its auto-lift clause), so it is exempt — and nothing
   evaluates whether that condition has been met.

## Goals

- Supervisor emits a structured, machine-readable watchlist alongside its
  markdown memo, each item tagged with an owning stage.
- Each owning stage ingests its open watchlist items into context and
  must resolve each (act or dismiss-with-reason) before completing.
- Items persist across sessions until resolved; unresolved items
  re-surface every run (carry-forward), so a flagged issue cannot be
  silently dropped.
- Respect existing stage boundaries: reflection owns rules + identity;
  the strategist owns theses + playbook.
- The recurring stale items (Rule 43, theses 233/267, Rule 49) get caught
  and resolved by the loop, not by hand.

## Non-goals

- **No deterministic lift-condition evaluator.** We do *not* build code
  that parses free-text `lift_condition` strings and auto-evaluates them
  against data. Approach considered and deferred — the supervisor (Opus)
  remains the judgment layer that decides what belongs on the watchlist.
- **No new dashboard charts.** Surfacing watchlist status on the existing
  supervisor memo page is in scope; new visualizations are not.
- **No autonomous supervisor cadence.** The supervisor stays on-demand
  (`task supervise`). Stages consume the *latest available* watchlist; a
  fresh supervisor run is not required before each session.
- **No deep AVGO playbook re-engineering beyond existing tools.** The
  strategist resolves the AVGO item using the tools it already has
  (re-spec the conditional in the next playbook, or close thesis #257).
  No new playbook-gate machinery.

## Stage ownership

The supervisor critiques the whole system, so its watchlist spans two
stages. Each item is owned by exactly one:

| Watchlist item (from memo 1)        | Owner stage          | Resolution authority |
|-------------------------------------|----------------------|----------------------|
| Retire / re-justify Rule 43         | `reflection`         | `retire_rule` / `revalidate_rule` |
| Rule 49 lifecycle review            | `reflection`         | `retire_rule` / `revalidate_rule` |
| Rule 45→48 / 46→47 churn pattern    | `reflection`         | `amend_rule` (new, Phase 4) |
| Reflection action drought           | `reflection`         | self-correcting via the gate |
| Theses 233/267 stale (Rule 43-grounded) | `ideation`       | `close_thesis` / `update_thesis` |
| AVGO playbook carryover (#257)      | `ideation`           | re-spec playbook conditional / `close_thesis` |

Reflection cannot mutate theses (it has no thesis tools, by design) and
the strategist does not retire rules. Each stage resolves only what it
owns.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ v2/supervisor.py  (Opus, observer of strategy state)           │
│   - writes markdown memo  (unchanged)                          │
│   - NEW: record_watchlist_item(title, detail, owner_stage)     │
│         -> writes ONLY to supervisor_watchlist_items           │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │ supervisor_watchlist_items  (new table)      │
        │  id, source_memo_id, title, detail,          │
        │  owner_stage, status, created_at,            │
        │  resolved_at, resolution, resolution_note,   │
        │  resolved_by_session_id, resolved_by_stage   │
        └────────────────────────────────────────────┘
              │ open items (owner_stage=ideation)   │ open items (owner_stage=reflection)
              ▼                                      ▼
   ┌─────────────────────────┐         ┌─────────────────────────────┐
   │ Stage 2: ideation       │         │ Stage 4: reflection         │
   │ (ideation_claude.py)    │         │ (strategy.py)               │
   │  - ingests its open     │         │  - ingests its open items   │
   │    items into context   │         │    into context             │
   │  - resolve_watchlist_   │         │  - resolve_watchlist_item   │
   │    item tool            │         │  - amend_rule tool (Phase4) │
   │  - GATE: unresolved     │         │  - GATE: unresolved open    │
   │    open items -> raise  │         │    items -> raise           │
   └─────────────────────────┘         └─────────────────────────────┘
```

### Shared building blocks

A single implementation, reused by both stages:

- **`supervisor_watchlist_items` table** (migration).
- **Queries** (in the appropriate queries module):
  - `get_open_watchlist_items(owner_stage) -> list[dict]`
  - `resolve_watchlist_item(item_id, resolution, note, session_id, stage)`
  - `record_watchlist_item(source_memo_id, title, detail, owner_stage)`
- **`resolve_watchlist_item` tool** — one definition + handler, registered
  in *both* stages' tool registries.
- **Context formatter** — `_format_open_watchlist_items(items)`, prepended
  to each stage's initial message (mirrors
  `_format_rule_revalidation_context`).

## Data model

`supervisor_watchlist_items`:

| Column                | Type        | Notes |
|-----------------------|-------------|-------|
| id                    | serial PK   | |
| source_memo_id        | int FK      | → `supervisor_memos(id)` |
| title                 | text        | short item label |
| detail                | text        | the item's body / what to check |
| owner_stage           | text        | CHECK in (`reflection`, `ideation`) |
| status                | text        | CHECK in (`open`, `acted`, `dismissed`), default `open` |
| created_at            | timestamptz | default now() |
| resolved_at           | timestamptz | null until resolved |
| resolution            | text        | `acted` or `dismissed`, null until resolved |
| resolution_note       | text        | required when resolving |
| resolved_by_session_id| int         | which session resolved it |
| resolved_by_stage     | text        | which stage resolved it |

Index on `(owner_stage, status)` for the open-items lookup.

## Behavior

### Supervisor (Phase 1)

The supervisor's prompt already asks it to produce a "Watchlist (next
supervisor run)" section. Add a `record_watchlist_item` write tool scoped
**only** to `supervisor_watchlist_items` — this is a write to
supervision-owned state, not strategy state, so the observer-of-strategy
principle holds. The supervisor classifies each item's `owner_stage` as it
records it (it already reasons about whose job each item is — e.g. memo 1
explicitly says "this is the supervisor's job, not the executor's").

The markdown memo is unchanged; the structured items are the
machine-readable mirror of its watchlist section.

### Stage ingestion + gate (Phase 2)

Both stages follow the same shape, mirroring the existing
`missing_revalidations` gate in `run_strategy_reflection`:

1. At stage start, load `get_open_watchlist_items(owner_stage)` and format
   them into the initial message as **mandatory open items**.
2. Register the `resolve_watchlist_item` tool. The stage must call it for
   each open item — `acted` (with a note describing the strategy/thesis
   action taken) or `dismissed` (with a note justifying no change).
3. After the agentic loop, compute which owned open items were left
   unresolved. If any remain, **raise `RuntimeError`** (same hard-gate
   pattern as `missing_revalidations`). This is the forcing function.

`dismissed` is a legitimate resolution — "nothing to do here" is allowed,
but it must be *stated and reasoned*, which is exactly what five silent
zero-action reflections failed to do.

### Cross-stage ordering

Ideation (stage 2) runs before reflection (stage 4) within a session.
Theses 233/267 are grounded in Rule 43, which reflection retires. Two
acceptable orderings, both fine because items carry forward:

- **Same session:** ideation closes the theses (owner=ideation), reflection
  retires Rule 43 (owner=reflection) later the same session.
- **Across sessions:** reflection retires Rule 43 this session; next
  session ideation sees the now-ungrounded theses item still `open` and
  closes them.

No hard cross-stage dependency is enforced; eventual consistency via
carry-forward is sufficient.

## Phases

1. **Supervisor structured watchlist** — table + migration,
   `record_watchlist_item` tool + handler, prompt update. The supervisor's
   next run populates items.
2. **Stage ingestion + resolution gate** — shared query/tool/formatter;
   wire into both `ideation_claude.py` and `strategy.py`; hard gate on
   unresolved owned items.
3. **Data hygiene (Rule 43, theses 233/267, Rule 49)** — once Phases 1-2
   land and a supervisor run records these, the owning stages resolve
   them. **Both execution paths documented** (decision deferred to
   implementation time, no prod writes during planning):
   - *Loop path:* run supervisor → run ideation + reflection; the gate
     forces resolution.
   - *Manual path:* direct SQL — `retire_rule` Rule 43; `close_thesis`
     233/267 with reason; review Rule 49 (retire if still n<5 / beat-rate
     unrecovered).
4. **Rule churn: amend-in-place** — new `amend_rule(rule_id, new_text,
   new_evidence, reason)` tool that updates a rule's evidence/text without
   retire-and-replace; reflection prompt updated to prefer `amend_rule`
   when only embedded evidence changed (stops the 45→48 / 46→47 churn).
5. **Diagnostics** — investigate the silent stage failures in sessions
   4642 (5/22, 2 failures) and 4644 (5/25, 1 failure), and the
   duplicate-session / cost artifact on 5/22 (two records: $4.42 and
   $0.88). Scoped as **investigate → fix only if a code bug is found**;
   no predetermined fix.

## Testing

Follow existing patterns (pytest in docker; all external deps mocked;
`get_cursor()` mocked):

- Migration applies; `owner_stage` and `status` CHECK constraints reject
  bad values.
- `record_watchlist_item` writes only to `supervisor_watchlist_items` and
  never to strategy-state tables.
- `get_open_watchlist_items(owner_stage)` filters by stage and `open`
  status.
- `resolve_watchlist_item` sets status/resolution/note/resolved_* and is
  idempotent on re-resolution attempts.
- Reflection gate raises `RuntimeError` when a reflection-owned item is
  left open; passes when all are resolved or dismissed.
- Ideation gate raises when an ideation-owned item is left open.
- Carry-forward: an item left `open` in one run re-appears in the next
  run's context.
- `amend_rule` updates in place without creating a new rule row or setting
  `retired_at`.

## Open risks

- **Supervisor cadence vs. session cadence.** If the supervisor hasn't run
  recently, stages consume a stale watchlist. Acceptable for v1 (latest
  available); a future enhancement could wire the supervisor as a session
  stage.
- **Over-gating.** A hard `RuntimeError` could fail a stage if the model
  refuses to resolve an item. Mitigation: `dismissed` is always a valid
  resolution, so the model can always clear the gate by reasoning; the
  gate forces a *decision*, not a specific action.
