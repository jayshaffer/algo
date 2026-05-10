# Haiku News Filter for the Strategist

**Date:** 2026-05-10
**Status:** Design — pending implementation plan
**Owner:** jay

## Problem

The strategist's `get_news_signals` tool returns the entire 7-day news
window — currently 936 signals across 650 distinct tickers (~140K chars
of `[#id] DATE TICKER cat/sent/conf: headline` lines). On any given day
only ~30 of those signals are likely actionable. Asking Opus to scan
the full firehose every session is a quality problem (signal-to-noise
in the context window) and a cost problem (cache_creation on a large
tool result).

A prior attempt at byte-level truncation (the abandoned
`strategist-cache-fix` branch) demonstrated that dumb clipping
introduces a regression: the post-fix run doubled `get_news_signals`
calls and created zero new theses, suggesting the strategist was
working around insufficient news context. A smart filter that ranks
by relevance — rather than clipping by bytes — should both preserve
quality and provide modest cost savings.

## Goal

Curate the news stream the strategist sees by inserting a Haiku-based
relevance filter between the local `news_signals` cache and the
strategist. Specifically:

- Strategist gets a denser, more relevant subset (~30 signals) on its
  default news call, with full article summaries available for each.
- Strategist retains access to the raw firehose via the unchanged
  `get_news_signals` tool when it wants to look past the filter.
- Strategist's prompt nudges it to consider non-news ideation paths
  (market structure, attribution, macro themes, position re-reads) so
  the filter doesn't become a creativity bottleneck.

**Success criteria:**

- Thesis quality measured by playbook action count + rationale depth
  remains at or above current levels across a 5-session paper sample.
- `get_news_signals` re-call frequency does not increase (vs. the
  pre-filter baseline) — i.e., the curated output is sufficient.
- Per-session API cost reduces modestly (~$0.50–$1.00), but quality
  parity is the gating metric, not cost.

## Non-Goals

- Pure cost reduction. Better levers exist (pre-seeded context trim,
  Sonnet swap). Cost gain here is a side effect.
- Replacing the existing news ingestion pipeline. `v2/pipeline.py`
  continues to fetch and classify; this design adds one storage column
  it must populate.
- Filtering other strategist tools (`get_macro_signals`,
  `get_market_snapshot`, etc.). Only news is firehose-shaped; only
  news needs curation.
- Changing the strategist model or its system prompt structure beyond
  one sentence about non-news ideation.

## Approach

Five tightly-coupled components shipped together:

1. **Schema:** add a nullable `summary TEXT` column to `news_signals`.
2. **Ingestion:** `v2/pipeline.py` persists `NewsItem.summary` (already
   plumbed from Alpaca via `v2/news.py`, currently discarded at the
   DB layer).
3. **Backfill:** one-shot `v2/news_backfill.py` re-queries Alpaca for
   the last 7 days, matches by `alpaca_id`, fills `summary` on
   existing rows where it is NULL. Idempotent.
4. **Filter:** `v2/news_filter.py` exposes `curate_signals(signals,
   target_n, regime_context) -> list[int]` — calls Haiku 4.5 with a
   structured prompt asking for the top-N relevant signal IDs.
   Degrades gracefully (returns all input IDs) on any error.
5. **Tool integration:** `v2/tools.py` adds
   `tool_get_curated_news(ticker, days, target_n)` that fetches from
   DB, drops NULL-summary rows, calls the filter, and re-renders the
   selected signals in the existing line format. The existing
   `tool_get_news_signals` (raw firehose) is unchanged.

A light prompt nudge in `v2/ideation_claude.py` encourages
non-news-driven thesis generation alongside the new tool.

## Architecture

### File layout

```
v2/
  news.py                   ← existing; unchanged
  pipeline.py               ← MODIFY: persist summary alongside other fields
  database/trading_db.py    ← MODIFY: schema migration column + insert carries summary
  tools.py                  ← MODIFY: add tool_get_curated_news; register handler
                                      tool_get_news_signals unchanged
  news_filter.py            ← NEW: Haiku filter; ~80 lines
  news_backfill.py          ← NEW: one-shot CLI; ~50 lines
  ideation_claude.py        ← MODIFY: one sentence in system prompt
db/init/
  027_news_signals_summary.sql  ← NEW: ALTER TABLE add summary column
tests/v2/
  test_news_filter.py       ← NEW
  test_news_backfill.py     ← NEW
  test_tools.py             ← MODIFY: add test class for curated tool
  test_pipeline.py          ← MODIFY (or create): summary persistence tests
```

### Components

**Schema (`db/init/027_news_signals_summary.sql`).** Adds
`summary TEXT` column to `news_signals`. Nullable so pre-backfill rows
aren't broken. No index — summary is read as part of the row payload,
not queried directly.

**Ingestion (`v2/pipeline.py`, `v2/database/trading_db.py`).** The
batch-insert SQL gets one extra column. `NewsItem.summary`
(line 66 of `v2/news.py`, already populated from Alpaca's `news.summary`
field) flows through to the DB. `ON CONFLICT DO NOTHING` semantics
preserved — no overwriting of summaries on rerun.

**Backfill (`v2/news_backfill.py`).** Module with a `run()` function and
a `__main__` entry point. Re-fetches the last 7 days via the existing
`fetch_news(hours=168)`, iterates returned items, runs an UPDATE that
sets `summary` only when currently NULL. Idempotent — safe to re-run.

**Filter (`v2/news_filter.py`).** One public function:

```python
def curate_signals(
    signals: list[dict],
    target_n: int,
    regime_context: str,
) -> list[int]:
    """Return the IDs of the top-N most relevant signals for today.

    Degrades gracefully: on any Haiku error, malformed response, or
    all-hallucinated IDs, returns [s['id'] for s in signals] (the
    firehose) so the caller can serve unfiltered output.
    """
```

Internally:
- Builds a prompt with `regime_context` + per-signal block of
  `[#<id>] TICKER cat/sent: HEADLINE\nSUMMARY` (~250 chars per signal).
- Calls Haiku 4.5 via `get_claude_client()` requesting structured JSON
  output: `{"top_ids": [int, int, ...]}`.
- Validates the response: parses JSON; intersects returned IDs with
  the input ID set; if intersection is empty, falls back to all input
  IDs.

**Tool (`v2/tools.py`).**

```python
def tool_get_curated_news(
    ticker: str | None = None,
    days: int = 7,
    target_n: int = 30,
) -> str:
    """Like get_news_signals, but Haiku-curated to ~target_n most
    relevant signals for today. Use this by default; use
    get_news_signals when you need the raw firehose.
    """
```

Behavior:
1. Fetch signals from DB (uses existing `get_news_signals` DB function).
2. Drop rows where `summary` is NULL or empty (transitional handling).
3. If `len(candidates) <= target_n`, skip Haiku entirely — return all.
4. Build `regime_context` via existing `get_macro_context(days=2)`.
5. Check session-local cache by `(ticker, days, target_n)`; cache hit
   returns the cached ID list.
6. Call `curate_signals(...)` to get top-N IDs; cache the result.
7. Re-render selected signals in the existing
   `[#<id>] MM-DD HH:MM TICKER cat/sent/conf: HEADLINE` line format.
   Identical to `tool_get_news_signals` so the strategist's parsing is
   unchanged.
8. Return the formatted string.

Cache is a module-level dict, cleared by the existing
`reset_session()` function called at the start of each strategist loop
(`v2/ideation_claude.py:218`).

**Tool registration.** Add `"get_curated_news": tool_get_curated_news`
to the `TOOL_HANDLERS` dict and the corresponding entry to
`TOOL_DEFINITIONS`. Both tools live side-by-side; strategist can
choose.

**Prompt nudge (`v2/ideation_claude.py`).** Add one sentence to
`CLAUDE_SESSION_STRATEGIST_SYSTEM`:

> News is one input; you can also generate theses from market
> structure (`get_market_snapshot`), attribution patterns
> (`get_signal_attribution`), macro themes (`get_macro_context`,
> `get_macro_signals`), or a fresh read of existing positions.

The tool descriptions in `TOOL_DEFINITIONS` for both
`get_news_signals` and `get_curated_news` are updated to clearly state
that the curated version is the default and the raw firehose is the
opt-out.

## Data Flow

**Strategist makes its first news call:**

```
get_curated_news(ticker=None, days=7, target_n=30)
  ↓
fetch rows from news_signals where published_at > NOW()-7d  (~936 rows)
  ↓
drop rows with NULL/empty summary
  ↓
if len <= 30: skip Haiku, return all rendered
otherwise:
  build regime_context via get_macro_context(days=2)
  cache miss → call curate_signals(rows, 30, regime_context)
    ↓
    Haiku 4.5 with ~58K input tokens + 1K output tokens (~$0.063)
    returns {"top_ids": [...]} → list of int IDs
  cache result by (ticker, days, target_n)
  ↓
render selected rows in [#<id>] DATE TICKER cat/sent/conf: HEADLINE format
return string (~5K chars)
```

**Repeat call same session, same params:** cache hit, skip Haiku.
Returns the same formatted string. Free.

**Ticker-specific call** (`ticker="AAPL"`): SQL narrows to that ticker
(usually <30 rows), Haiku is skipped, all candidates returned in
standard format.

**Strategist explicitly calls the raw firehose:** unchanged behavior
— `get_news_signals(...)` returns the full 936-row listing as it does
today.

**Backfill (one-shot, manual):**

```
python -m v2.news_backfill
  → fetch_news(hours=168) via Alpaca           (existing function)
  → for each NewsItem:
      UPDATE news_signals SET summary = %s
        WHERE alpaca_id = %s AND (summary IS NULL OR summary = '')
  → log "Backfilled N of M rows (skipped K already-populated)"
```

## Failure Modes

| Failure | Behavior |
|---|---|
| Haiku API error or timeout | `curate_signals` returns all input IDs; tool falls back to firehose for that call. Strategist gets data unfiltered. Logs a warning. |
| Haiku returns malformed JSON | Parse failure → return all input IDs. Same fallback path. |
| Haiku returns IDs not in input | Intersect with input IDs; if empty, fall back to firehose. |
| Pre-backfill: all candidates have NULL summary | Drop NULLs → empty candidate set → return the firehose by skipping the filter. |
| Schema migration not yet run | Pipeline insert with `summary=` would fail. **Mitigation:** schema migration must land before the pipeline modification. Plan sequences these correctly. |
| Backfill script fails partway | Re-run is safe (idempotent via NULL guard). |
| `target_n` larger than candidate count | Skip Haiku, return all candidates. Common path for ticker-specific calls. |
| Stale cache between sessions | `reset_session()` clears the module-level cache at the start of each strategist loop. Verified by test. |
| Strategist ignores the curated tool and only uses the raw firehose | No regression — that's the current behavior. The filter is opt-in via tool choice; the prompt nudge encourages but does not force. |

## Risks

1. **Haiku ranking drifts from "good for Opus" over time.** Haiku is
   capable but it's a smaller model; its sense of "newsworthy" may
   diverge from Opus's. Mitigation: the strategist can always fall
   back to `get_news_signals` for the raw firehose; the prompt makes
   this discoverable. If drift becomes a problem, log Haiku's ranking
   and inspect a sample.

2. **Summaries from Alpaca are sometimes empty or boilerplate.** Some
   news sources don't provide useful summaries. Rows with empty
   summaries are dropped from the filter input, which is mostly fine
   (they were noise anyway) but could hide a high-signal headline
   whose summary happens to be missing. Acceptable for v1.

3. **Increased Alpaca API usage during backfill.** The one-shot
   backfill makes one batch fetch covering the last 7 days. Well
   within Alpaca's free-tier rate limits.

4. **Cost accounting.** Haiku adds ~$0.06 per uncached filter call.
   With session-local caching and ticker-specific skip-path, expect
   1–2 actual Haiku calls per strategist session = ~$0.06–$0.12. Net
   savings (less Opus cache_creation on the tool return) should
   recover this 5–10×.

5. **Strategist may not use the new tool.** Without explicit prompt
   guidance, Opus might keep calling `get_news_signals`. Mitigation:
   tool description for the curated tool explicitly says it's the
   default; prompt nudge mentions both tools and what each is for.

## Validation

**Pre-merge:**

- All unit tests pass (see Testing section).
- Manual smoke test on paper DB: schema migrates, backfill runs to
  completion, `SELECT COUNT(*) FROM news_signals WHERE summary IS NOT
  NULL` returns ~936.
- One paper strategist session runs end-to-end with the curated tool
  available. Verify in the session log that at least one
  `get_curated_news` call was made and its output was smaller than the
  raw firehose would have been.

**Post-merge (quality watch over 5 paper sessions):**

| Metric | Pre-filter baseline (playbook 12 from session 241) | Watch-fors |
|---|---|---|
| Playbook actions per run | 5 | ±1; alert if it drops to 0–2 consistently |
| Average reasoning chars per action | 361 | ±20%; alert on large sustained drop |
| `get_news_signals` calls per session | 2 | Should be 0 or 1 with curated as default |
| `get_curated_news` calls per session | (new) | Expect 1–3 |
| Theses created per session | 2 | ±1; alert on persistent 0s |
| Strategist cost per run | $5.18 (avg of prod 3251, 3405) | Expect $4–$5; if higher, regression |

If quality regresses (playbook actions drop, theses dry up, raw news
calls spike), the first lever to pull is `target_n` (raise from 30 →
50 → 80). If the regression persists, revert and reconsider — the
filter is additive and easy to disable by removing the registration
from `TOOL_HANDLERS`.

## Testing

### Schema migration

Manual check only (no unit test per codebase convention):
`psql -c "\d news_signals"` shows the new column.

### Ingestion (`tests/v2/test_pipeline.py`)

- `test_pipeline_persists_summary` — mock Alpaca to return a `NewsItem`
  with `summary="..."`. Run the insert path. Assert the DB row has
  the summary.
- `test_pipeline_handles_empty_summary` — same with `summary=""`.
  Assert empty string is stored.

### Filter (`tests/v2/test_news_filter.py`)

- `test_curate_signals_returns_top_n_ids` — mock the Claude client to
  return `{"top_ids": [3, 7, 12]}`. Call with target_n=3. Assert
  `[3, 7, 12]`.
- `test_curate_signals_drops_hallucinated_ids` — Haiku returns IDs not
  in input. Assert they are excluded.
- `test_curate_signals_falls_back_on_api_error` — mock client raises.
  Assert returns all input IDs.
- `test_curate_signals_falls_back_on_malformed_json` — mock returns
  invalid JSON. Assert returns all input IDs.
- `test_curate_signals_falls_back_on_empty_intersection` — all
  returned IDs hallucinated. Assert returns all input IDs.
- `test_curate_signals_passes_regime_context_to_haiku` — capture the
  prompt sent to the client; assert `regime_context` is in the
  message body.
- `test_curate_signals_input_format_includes_summary` — capture the
  prompt; assert each signal's summary text is present.

### Tool (`tests/v2/test_tools.py`)

- `test_tool_get_curated_news_filters_signals` — mock
  `curate_signals` to return a subset of IDs. Assert the tool output
  contains only those lines and matches the existing
  `[#<id>] DATE TICKER cat/sent/conf: HEADLINE` format byte-for-byte.
- `test_tool_get_curated_news_skips_haiku_when_below_target` —
  provide 10 signals, `target_n=30`. Assert `curate_signals` is NOT
  called and all 10 are returned.
- `test_tool_get_curated_news_drops_null_summary_rows` — provide 5
  signals, 2 with `summary=None`. Assert the 2 NULL rows are excluded
  before the filter.
- `test_tool_get_curated_news_caches_within_session` — call twice with
  same args. Assert `curate_signals` is called once. Call with
  different `target_n` and assert cache miss (new Haiku call).
- `test_tool_get_curated_news_ticker_specific` — `ticker="AAPL"`.
  Assert SQL filter was called with that ticker.
- `test_tool_get_news_signals_unchanged` — existing tests for
  `tool_get_news_signals` must still pass.
- `test_curated_news_cache_resets_with_session` — populate cache, call
  `reset_session()`, identical call re-invokes Haiku.

### Backfill (`tests/v2/test_news_backfill.py`)

- `test_backfill_updates_summary_for_matching_alpaca_id` — mock
  `fetch_news`, seed DB with NULL-summary row matching the
  `alpaca_id`, run backfill, assert row updated.
- `test_backfill_skips_rows_with_existing_summary` — seed DB row with
  `summary='existing'`, run backfill providing different summary,
  assert original `'existing'` preserved (idempotent guard).
- `test_backfill_handles_no_match` — fetch returns items for
  alpaca_ids not in DB. Assert no errors, no rows touched.
- `test_backfill_is_idempotent` — run twice. Assert second run is a
  no-op for rows that were updated on the first pass.

## Out of scope (do NOT do as part of this work)

- Per-tool truncation thresholds inside the agentic loop (this was the
  abandoned approach; not re-relevant once the firehose is curated).
- Filtering `get_macro_signals` or other strategist tools.
- Adjusting Haiku model selection. Use 4.5 (the existing executor
  model) for now; revisit only if cost or latency requires.
- A configuration UI / runtime knobs for `target_n`. Default 30 is
  hardcoded; tunable via the parameter if needed.
- A/B harness comparing curated vs firehose under the same session.
  The post-merge quality watch covers what we care about.
- Body-text storage beyond `summary`. Alpaca's `summary` field is what
  the existing infra provides; pulling full article bodies would
  require additional fetches and storage outside this scope.
- Sentiment/category re-classification by Haiku. Existing pipeline
  classification stays as-is.
