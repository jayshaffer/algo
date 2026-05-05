# Session Cost Tracking — Design Spec

**Date:** 2026-05-05
**Status:** Draft, pending implementation
**Scope:** Persist Claude API token usage per session stage so daily session cost is queryable after the fact.

## Problem

The daily session (`v2/session.py`) makes many Claude calls across stages — strategist (Opus), executor (Haiku), reflection (Opus), classifier (Haiku per news item), social posts (Haiku per post). The only place cost is computed is `_print_cost_summary` in `v2/ideation_claude.py:245`, which:

- Covers only the strategist stage.
- Hardcodes Opus 4.6 pricing inline (BUGS.md:354).
- Has a double-subtraction bug — treats `usage.input_tokens` as if it includes cache tokens; it does not.
- Writes to stdout via `print()`, not the logger or DB.

Result: there is no way to answer "how much did the last session cost" after the run, and the printed number is wrong even when captured.

## Goals

- Record per-stage token usage durably in the database.
- Compute cost on read via SQL view, using a maintained per-model price table.
- Replace the buggy `_print_cost_summary` with a correct helper.
- Make zero changes to stage function signatures — instrumentation lives at the `claude_client` boundary.

## Non-goals

- Tracking premarket, weekly, or entertainment Claude calls. The same primitive will support these later; out of scope for v1.
- Per-individual-call audit table. Stage-level granularity is sufficient.
- Public dashboard surface for cost data. Internal/operator-facing only.
- Effective-dated pricing history. A price update retroactively rewrites historical cost. Acceptable for an internal estimate.

## Architecture

### Data model

**Migration:** `db/init/024_session_stage_token_usage.sql`

```sql
ALTER TABLE session_stages
    ADD COLUMN model VARCHAR(64),
    ADD COLUMN input_tokens INT,
    ADD COLUMN output_tokens INT,
    ADD COLUMN cache_creation_tokens INT,
    ADD COLUMN cache_read_tokens INT;

CREATE TABLE model_pricing (
    model VARCHAR(64) PRIMARY KEY,
    input_per_mtok NUMERIC(10,4) NOT NULL,
    output_per_mtok NUMERIC(10,4) NOT NULL,
    cache_creation_per_mtok NUMERIC(10,4) NOT NULL,
    cache_read_per_mtok NUMERIC(10,4) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed both bare aliases and dated pins. Callsites in v2/ pass the dated pin
-- (e.g. classifier uses "claude-haiku-4-5-20251001"); ideation/session pass
-- the bare alias. Both must resolve.
INSERT INTO model_pricing (model, input_per_mtok, output_per_mtok, cache_creation_per_mtok, cache_read_per_mtok) VALUES
    ('claude-opus-4-7',              15.00, 75.00, 18.75, 1.50),
    ('claude-opus-4-6',              15.00, 75.00, 18.75, 1.50),
    ('claude-sonnet-4-6',             3.00, 15.00,  3.75, 0.30),
    ('claude-haiku-4-5',              1.00,  5.00,  1.25, 0.10),
    ('claude-haiku-4-5-20251001',     1.00,  5.00,  1.25, 0.10);

CREATE VIEW session_stage_costs AS
SELECT
    ss.*,
    CASE
        WHEN ss.model IS NULL OR mp.model IS NULL THEN NULL
        ELSE (
            COALESCE(ss.input_tokens, 0)          * mp.input_per_mtok +
            COALESCE(ss.output_tokens, 0)         * mp.output_per_mtok +
            COALESCE(ss.cache_creation_tokens, 0) * mp.cache_creation_per_mtok +
            COALESCE(ss.cache_read_tokens, 0)     * mp.cache_read_per_mtok
        ) / 1000000.0
    END AS cost_usd
FROM session_stages ss
LEFT JOIN model_pricing mp ON mp.model = ss.model;

CREATE VIEW session_costs AS
SELECT
    s.id AS session_id, s.session_date, s.session_type, s.status,
    SUM(sc.cost_usd)                  AS total_cost_usd,
    SUM(sc.input_tokens)              AS total_input_tokens,
    SUM(sc.output_tokens)             AS total_output_tokens,
    SUM(sc.cache_creation_tokens)     AS total_cache_creation_tokens,
    SUM(sc.cache_read_tokens)         AS total_cache_read_tokens
FROM sessions s
LEFT JOIN session_stage_costs sc ON sc.session_id = s.id
GROUP BY s.id, s.session_date, s.session_type, s.status;
```

Stages with no Claude calls (learning, dashboard) and skipped stages leave the columns NULL and contribute NULL cost (skipped by `SUM`'s NULL semantics, so the session total reflects only stages that actually ran a Claude call).

### Code

**`v2/claude_client.py` — context-local accumulator (new):**

```python
@dataclass
class UsageAccumulator:
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    mixed_models: bool = False

    def add(self, model: str, usage):
        if self.model is None:
            self.model = model
        elif self.model != model:
            self.mixed_models = True
        self.input_tokens          += usage.input_tokens or 0
        self.output_tokens         += usage.output_tokens or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens     += getattr(usage, "cache_read_input_tokens", 0) or 0

_current_usage: contextvars.ContextVar[UsageAccumulator | None] = \
    contextvars.ContextVar("_current_usage", default=None)

@contextlib.contextmanager
def capture_usage() -> UsageAccumulator:
    acc = UsageAccumulator()
    token = _current_usage.set(acc)
    try:
        yield acc
    finally:
        _current_usage.reset(token)

def _record_usage(model: str, usage) -> None:
    acc = _current_usage.get()
    if acc is not None:
        acc.add(model, usage)
```

**Single integration point inside `claude_client.py`:**

`_call_with_retry` — after `stream.get_final_message()` returns the assembled `Message`, call `_record_usage(create_kwargs["model"], message.usage)` before returning. This captures every Claude call in the codebase, including each turn of `run_agentic_loop` (which dispatches via `_call_with_retry` at line 246), and every one-shot call (classifier, executor, social, premarket, weekly).

**Important:** `run_agentic_loop` must *not* additionally call `_record_usage` with its aggregated `AgenticLoopResult` totals. Doing so would double-count every agentic-loop token. The existing aggregation inside `run_agentic_loop` stays for its own return value (used by callers that want per-loop totals), but cost capture happens only at the `_call_with_retry` boundary.

**`v2/session.py` — wrap each stage:**

```python
def _run_pipeline_stage(...):
    _start_stage(session_id, "pipeline")
    with capture_usage() as usage:
        try:
            run_pipeline()
            _complete_stage(session_id, "pipeline", usage)
        except Exception as e:
            _fail_stage(session_id, "pipeline", str(e), usage)
            raise  # preserve current behavior
```

`_complete_stage` and `_fail_stage` (in `v2/database/trading_db.py`) gain an optional `usage: UsageAccumulator | None` parameter and write the five new columns when present. Failed stages still record partial usage from any calls that completed before the exception.

**`v2/pricing.py` — new module:**

```python
def stage_cost_usd(model: str, input_t: int, output_t: int,
                   cache_write_t: int, cache_read_t: int) -> float:
    """Look up rates from model_pricing table, return USD."""
```

Used by:
- The end-of-session log line in `session.py`.
- Replaces the inline math inside `_print_cost_summary` in `ideation_claude.py`. The print function itself is deleted; the strategist's existing summary block calls `pricing.stage_cost_usd` instead.

**End-of-session log line** (added at the bottom of `session.py`'s success path):

```
Session 2026-05-05 daily complete in 392.8s
  Stage costs (USD):
    pipeline:    $0.0421  (claude-haiku-4-5)
    strategist:  $1.2840  (claude-opus-4-7)
    executor:    $0.0083  (claude-haiku-4-5)
    strategy:    $0.6210  (claude-opus-4-7)
    twitter:     $0.0012  (claude-haiku-4-5)
    bluesky:     $0.0012  (claude-haiku-4-5)
  Total: $1.9578
```

Implemented as a single `SELECT * FROM session_stage_costs WHERE session_id = ?` in `session.py`, formatted into log output.

### Data flow

```
Claude API call
  └─> _call_with_retry / run_agentic_loop
        └─> _record_usage(model, usage)
              └─> contextvars-bound UsageAccumulator (per stage)

session.py
  └─> capture_usage() opens accumulator
        └─> stage runs, _record_usage calls flow into the accumulator
        └─> _complete_stage / _fail_stage writes columns to session_stages

DB query
  └─> SELECT FROM session_costs (view)
        └─> joins session_stages × model_pricing, computes cost_usd
```

## Cleanups bundled with this work

- Delete `_print_cost_summary` from `v2/ideation_claude.py:245`. Replace its caller with a call into `v2/pricing.py`.
- Mark BUGS.md:354 fixed (`_print_cost_summary` hardcoded pricing) and the related double-subtraction bug.

## Testing

Unit tests:

- `tests/test_usage_accumulator.py` — `add()` sums correctly across calls; `mixed_models` flips on second model; counters initialize to 0; `_record_usage` is a no-op when no `capture_usage` block is active.
- `tests/test_capture_usage.py` — context manager isolation (nested blocks don't bleed; sequential blocks reset cleanly).
- `tests/test_pricing.py` — `stage_cost_usd` matches hand-computed values for each seeded model; raises on unknown model name.
- `tests/test_claude_client_usage.py` — patch `client.messages.stream`, assert `_record_usage` is invoked with the right model and usage object after `_call_with_retry` returns.

Integration tests (DB-backed, fits the existing pattern):

- `tests/test_session_cost_persistence.py` — run a fake stage that calls a mocked Claude inside `capture_usage()`; assert the `session_stages` row has the five new columns populated; assert `SELECT cost_usd FROM session_stage_costs WHERE id = ?` returns the expected value.
- `tests/test_session_costs_view.py` — insert two stage rows with known models and tokens; assert `session_costs` view sums and groups correctly; assert `model IS NULL` rows contribute NULL cost.
- `tests/test_pricing_coverage.py` — grep `v2/` for hardcoded `"claude-*"` model literals; assert each has a row in `model_pricing`. Fails loudly when a new pin is added without a matching seed.

## Migration & rollout

- New file: `db/init/024_session_stage_token_usage.sql`. Additive; no data loss possible.
- `db/init/` runs only on fresh volume create. Both prod and paper DBs already exist, so the migration must be applied manually after deploy:

  ```
  docker compose exec -T db psql -U $POSTGRES_USER -d $POSTGRES_DB \
      -f /docker-entrypoint-initdb.d/024_session_stage_token_usage.sql

  docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper \
      psql -U $POSTGRES_USER -d $POSTGRES_DB \
      -f /docker-entrypoint-initdb.d/024_session_stage_token_usage.sql
  ```

- All code lands in the same PR. Backwards-compatible: stages whose code paths do not yet flow through `capture_usage()` write NULLs and behave exactly as before.

## Risks

- **Pricing drift.** Anthropic price changes require an `UPDATE model_pricing` row. There is no audit/effective-date column, so a price update retroactively rewrites historical cost numbers. Acceptable for an internal estimate; revisit if exact historical reporting matters.
- **Unseeded model pin.** A new dated pin (e.g. `claude-haiku-4-5-20260601`) added to a callsite without a matching `model_pricing` row produces NULL cost for that stage — silent under-reporting, but visible (the row exists with tokens but no cost). Mitigation: a test in `test_pricing_coverage.py` enumerates every hardcoded `claude-*` model literal in `v2/` and asserts each has a `model_pricing` row.
- **Mixed-model stage.** Today no stage uses more than one model, so the `mixed_models` flag is a tripwire only. If it ever fires, follow-up work is to either split the stage or extend the schema to per-model rows.
- **Agentic-loop double-recording.** Avoided by design: only `_call_with_retry` records usage; `run_agentic_loop` does not. A test (`test_claude_client_usage.py`) asserts that running an agentic loop with N turns produces exactly N `_record_usage` calls — no more.

## Open questions

- None.
