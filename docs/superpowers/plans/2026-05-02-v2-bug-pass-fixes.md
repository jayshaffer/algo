# v2 Bug Pass Fixes — 2026-05-02

> **For agentic workers:** Use `superpowers:executing-plans` to work through this task-by-task. Steps use checkbox (`- [ ]`) syntax. Land each tier as its own commit/PR; do not bundle across tiers.

**Source:** Fresh four-subsystem audit on 2026-05-02 (strategist, executor, learning, pipeline/session). Findings are independent of `v2/BUGS.md`.

**Sequencing rule:** Tier 0 first (live-reachable next session). Tier 1 next (correctness/safety). Tier 2 is opportunistic. Within a tier, items are ordered by blast radius.

---

## Tier 0 — P0, ship before next live session

These are reachable on the next strategist or trader run and silently corrupt data or crash the loop.

### T0.1 — `attribution.py:84` `win_rate_30d` counts NULL as loss
- [x] Patch `compute_signal_attribution` SQL: wrap with `WHEN alpha_30d IS NULL THEN NULL` so NULL propagates through `AVG`
- [x] Apply same fix to `patterns.py:71` (`analyze_signal_categories`)
- [x] Audit `patterns.py` for any other `WHEN col > 0 THEN 1.0 ELSE 0.0` shapes; fix in place (only signal_categories had the bug — sentiment/ticker/confidence funcs only compute win_rate_7d which is filtered NOT NULL)
- [x] One-time recompute: ran `compute_signal_attribution()` against prod + paper DBs (e.g. thesis bucket: 51.7% → 65.2% win_rate_30d after fix)
- [x] Pass NULL through Python upsert (don't coerce to Decimal(0)) so storage preserves "no data" semantic
- [x] Add unit tests: SQL contains the explicit `WHEN ... IS NULL THEN NULL` guard

### T0.2 — `pipeline.py` multi-ticker news drops all but first ticker
- [x] New migration `db/init/021_news_signal_alpaca_id_ticker.sql`: drop `idx_news_signals_alpaca_id`, recreate as `(alpaca_id, ticker) WHERE alpaca_id IS NOT NULL`
- [x] Verified `macro_signals` index stays as `(alpaca_id)` only (one macro per article — no change needed)
- [x] Applied migration to prod + paper DBs
- [x] Live DB sanity test: 3 rows sharing alpaca_id with different tickers all insert; duplicate (alpaca_id, ticker) still blocked
- [x] Regression test in `test_db.py` queries live `pg_indexes` to verify the composite index shape

### T0.3 — `claude_client.py:267-283` max_tokens recovery sends `user → user`
- [x] Fix: before appending the concision-nudge user message, append a synthetic assistant placeholder so role alternation holds
- [x] Regression test: assert `result.messages` has no adjacent same-role entries after a max_tokens recovery
- [x] Confirmed the placeholder gets pruned by `_truncate_old_tool_results` after 3 exchanges (it has no tool_result so it's not eligible for truncation, but it's still a valid alternating message)

### T0.4 — `claude_client.py:135-146` `_aggressive_prune` produces `user → user`
- [x] Added defense-in-depth: walk the pruned list and drop any same-role adjacency (handles both the documented invariant and any future shape drift upstream)
- [x] Regression test: verify pathological input with `user→user` adjacency is corrected, AND canonical alternating input passes through unchanged
- [x] Tool_use/tool_result pairing preserved: in the canonical strategist flow, tail starts at `assistant(tool_use)` followed by `user(tool_result)` so dedup never fires

---

## Tier 1 — P1, ship within the next sprint

Real correctness/safety risks but not single-session blockers.

### T1.1 — `trader.py:369-381` sells don't credit buying_power on local-estimate fallback
- [x] Add `elif decision.action == "sell": buying_power += trade_value` in both real and dry-run branches
- [x] Test: simulate `get_account_info` raising; run a sell; assert returned `buying_power` increased by `trade_value`

### T1.2 — `trader.py:579-586` sector cap not refreshed mid-loop
- [x] After each successful buy fill in `_execute_decisions` (around line 619-622), update `position_values[ticker] = position_values.get(ticker, Decimal(0)) + outcome.trade_value`
- [x] Test: feed 3 buys in same sector that individually fit but cumulatively breach `MAX_SECTOR_PCT`; assert the 3rd is rejected

### T1.3 — `agent.py:259-270` ticker normalization missing
- [x] Normalize at parse boundary: `ticker=(d.get("ticker") or "").strip().upper()`
- [x] Audit `signal_refs` and any other LLM-emitted ticker fields; normalize consistently (added `_norm_ticker` to tools.py and applied to `tool_get_active_theses`, `tool_create_thesis`, `tool_adopt_thesis`, `tool_get_news_signals`, `tool_write_playbook` action items)
- [x] Test: feed `aapl ` (lowercase + trailing space) through the parse path; assert downstream sector lookup hits `SECTOR_MAP["AAPL"]`

### T1.4 — `executor.py:51-61` paper-vs-prod silent fallback
- [x] Add hard assertion at module load: require explicit `ALPACA_PAPER=true|false` env var (skipped when `ALPACA_API_KEY` unset so test imports/non-trading code paths don't crash)
- [x] Cross-check `ALPACA_PAPER` against `ALPACA_BASE_URL` — raise on mismatch (e.g. `paper=true` with prod URL)
- [x] Update `.env`, `.env.example`, `.env.paper`, and CLAUDE.md to document the new requirement; updated `tests/conftest.py::alpaca_env` fixture
- [x] Test: missing `ALPACA_PAPER` raises (RuntimeError); mismatched pair raises with clear message

### T1.5 — `trader.py:693-738` `insert_decision` failure orphans filled order
- [x] Wrap `insert_decision` in a bounded retry (3 attempts, 0.5s/1s exponential backoff) via `_insert_decision_with_retry`
- [x] On final failure: append the full decision payload (ticker, action, qty, price, order_id, reasoning, signal_refs) to `logs/orphan_decisions.jsonl` ONLY when order actually filled (not for unfilled/zero-qty cases)
- [x] Log at ERROR with full payload so an operator can manually reconcile
- [x] Test: mock `insert_decision` to always raise; assert the JSONL fallback file contains the decision; verify unfilled cases don't pollute the log

### T1.6 — `tools.py:343-391` `write_playbook` accepts duplicate-side actions for same ticker
- [x] Pre-insert validation: reject playbook where any (ticker, action) pair appears more than once
- [x] Return a clean tool_result error to the strategist loop so it can retry with deduped actions
- [x] Test: feed a playbook with two buys for AAPL; assert validation error before any DB write; case-insensitive ("aapl" + "AAPL" both blocked after normalization)

### T1.7 — `tools.py:240` `tool_update_thesis` no existence check on add_signal_refs-only path
- [x] Top of `tool_update_thesis`: if `not has_field_updates and not add_signal_refs`, return `"Error: no updates provided"`
- [x] Then call `get_thesis_by_id(thesis_id)`; if None, return `"Error: thesis ID {id} not found"`
- [x] Test: invalid thesis_id with only `add_signal_refs` returns clean error string, not raw psycopg2 error

### T1.8 — Truthy-check bug repeated in 5 sites (P3.41 not propagated)
- [x] `attribution.py:185-186` (`build_attribution_constraints`): replaced `if r.get(...) else 0` with `is not None` checks
- [x] `patterns.py` `analyze_signal_categories` builder
- [x] `patterns.py` `analyze_sentiment_performance` builder
- [x] `patterns.py` `analyze_ticker_performance` builder
- [x] `patterns.py` `analyze_confidence_correlation` builder
- [x] `attribution.py` `_format_attribution_summary` (already partially fixed by P3.41 for bucket gating; remaining truthy formatting checks fixed for consistency)
- [x] Also fixed `generate_pattern_report` rendering — without this fix the dataclass-level `0.0` still renders "N/A" because the report uses the same truthy pattern
- [x] Test: signal_categories/sentiment/ticker/confidence rows with `Decimal(0)` render as "+0.00%" not "N/A" in `generate_pattern_report`; None still renders as "N/A"

### T1.9 — `backfill.py:75,152` calendar vs trading-day cutoff mismatch
- [x] Added `trading_day_cutoff(today, n)` helper (mirror of `trading_day_offset` in reverse)
- [x] `get_decisions_needing_backfill` now uses trading-day cutoff
- [x] Test: from a Friday, calendar-day cutoff would be 7 days back (the prior Friday); trading-day cutoff lands on the Wednesday before — verified

### T1.10 — `backfill.py:170-172` SPY cache stores None on transient failure
- [x] Cache only successful fetches via `_spy_price` helper that gates on `price is not None`
- [x] Added parallel `spy_exit_prices` cache keyed by `exit_date` to eliminate redundant exit-side fetches
- [x] Tests: transient failure does NOT poison cache (refetch on second access); successful fetches cached for subsequent decisions sharing the same date

### T1.11 — `strategy.py` `tool_get_session_summary` missing orphan guards
- [x] Added `LEFT JOIN theses t ON ds.signal_type = 'thesis' AND t.id = ds.signal_id`
- [x] Added WHERE filters for all three signal types (news/macro/thesis), mirroring `attribution.py`
- [x] Test: SQL contains the three orphan-FK guards; orphan thesis labels do not leak through to reflection text

---

## Tier 2 — P2, opportunistic / next cleanup pass

Quality issues; address in batches when convenient.

### T2.1 — `session.py` "session already completed" exits 1
- [x] Added `result.idempotent_skip: str | None` field; not in `_ERROR_FIELDS`
- [x] Idempotent early-return now populates that field; `main()` exits 0 with a "no-op" log line
- [x] Test: idempotent skip leaves has_errors False and routes through the new field

### T2.2 — stage 0 bypasses session_stages tracking
- [x] Wrapped `_run_learning_refresh` with start/complete/fail stage helpers under stage name "learning"
- [x] Resume path skips re-running backfill+attribution if "learning" already completed
- [x] Tests: success and failure each produce a session_stages row; resume skip leaves backfill uncalled

### T2.3 — TZ handling for news/macro published_at
- [x] `news.py:fetch_news` uses `datetime.now(timezone.utc) - timedelta(hours=hours)` for the start cutoff
- [x] New migration `db/init/022_news_macro_published_at_tz.sql` promotes `published_at` to TIMESTAMPTZ on both signal tables (USING ... AT TIME ZONE 'UTC' for existing rows)
- [ ] **Manual step:** apply migration to running prod + paper DBs (denied permission to run ALTER TABLE on prod from this session)

### T2.4 — `get_bar_change(days=1)` returns 0.0 on insufficient data
- [x] Strict `len(symbol_bars) <= days` returns None
- [x] Removed the `change_5d or 0.0` coercion; SectorPerformance.change_5d is now Optional[float]
- [x] `format_market_snapshot` renders missing 5d as "N/A" instead of "0.0%"
- [x] Tests: 1-bar/5-bar responses return None; sectors with no 5d data render as N/A

### T2.5 — `total_pnl_7d` sums percentages
- [x] Renamed dataclass field and SQL alias to `sum_pct_returns_7d`; ORDER BY now `avg_outcome_7d`
- [x] Pattern report header updated to "by avg 7d return" and renders avg per ticker
- [x] Tests: SQL contains `as sum_pct_returns_7d` and `ORDER BY avg_outcome_7d`; existing fixture data flowed through

### T2.6 — strategy asymmetric session caps
- [x] `MAX_PROPOSALS_PER_SESSION = 3`, `_session_proposals` ContextVar mirrors retirements
- [x] `tool_propose_rule` enforces the cap and surfaces a clean message
- [x] System prompt + `reset_session()` updated
- [x] Tests: cap rejection, no DB write past cap, isolation across threads (mirror of retirement test)

### T2.7 — strategy TZ comparison hazard
- [x] Added `_utc_date(dt)` helper that treats naive datetimes as UTC and converts aware ones via `astimezone(UTC)`
- [x] Identity throttle and rule tenure both compute age via `datetime.now(UTC).date() - _utc_date(created_at)`
- [x] Tests: helper covers both naive and aware cases (23:30 ET → May 2 UTC)

### T2.8 — `wait_for_fill` partial-fill semantics
- [x] Confirmed-zero post-cancel re-fetch returns `success=False, filled_qty=Decimal('0'), unknown_partial_fill=False`
- [x] Re-fetch failures set `unknown_partial_fill=True` so the trader treats the result as needing reconciliation
- [x] Trader's `_insert_decision_with_retry` now writes an orphan log entry when `unknown_partial_fill=True` even if `filled_qty` is None
- [x] Tests: confirmed-zero case, unknown-fill case, and orphan-log behavior

### T2.9 — `wait_for_fill` no try/except around get_order_by_id
- [x] Per-iteration try/except with up to 3 consecutive transient errors before breaking the poll loop
- [x] Persistent failures still trigger the cancel + post-cancel re-fetch dance (which then surfaces unknown_partial_fill via T2.8)
- [x] Tests: one transient error then success; persistent errors break loop and attempt cancel

### T2.10 — `trader.py` `date.today()` repeated
- [x] Captured once at the top of `run_trading_session` as `datetime.now(ZoneInfo("America/New_York")).date()`
- [x] Threaded `session_date` through `_execute_decisions`, `_prepare_decision`, `_execute_decision_order`, `_log_decisions`
- [x] Replaces every prior `date.today()` callsite in trader.py — verified with grep
- [x] Test: insert_decision is called with a single distinct `decision_date` across all logged rows

### T2.11 — `sync_orders_from_alpaca` docstring
- [x] Added a multi-paragraph docstring clarifying the function as a transient mirror of Alpaca's open-orders set, not an audit log; calls out that filled orders' history lives in `decisions`

### T2.12 — `count_actions` conflates adopt with create
- [x] `tool_adopt_thesis` now returns `"Adopted thesis ID ..."` instead of reusing the `Created` prefix
- [x] `count_actions` returns a 4-tuple including `adopted`; matching field added to StrategistResult/ClaudeIdeationResult
- [x] `_print_cost_summary` shows `Theses adopted: N` when nonzero
- [x] Tests: distinct adopt/create counters; existing test_adopt_success updated to assert the new prefix

### T2.13 — `get_recent_decisions` strict `>` excludes boundary
- [x] `database/trading_db.py:get_recent_decisions` uses `>=` for the lookback window
- [x] Same fix for `get_account_snapshots` so window math is consistent across the codebase
- [x] Both functions now have a one-liner docstring documenting "inclusive of boundary day"
- [x] Tests: SQL contains the inclusive `>=` form for both functions

### T2.14 — orphan positions fetched twice
- [x] `run_strategist_loop` fetches `get_orphan_positions()` once and passes the list into both `build_formation_context(orphans=...)` and `_build_orphan_block(orphans=...)`
- [x] Both helper functions accept an optional pre-fetched list and fall back to a DB fetch when called standalone
- [x] Test: `get_orphan_positions` is called exactly once per strategist run

---

## Out of scope (noted but not planned)

- **`executor.py:531-539` `calculate_position_size`** — appears to be dead code; verify no callers, then delete in a separate cleanup PR
- **`trader.py:169-177,804` `_build_open_sell_orders` unused** — same; delete or wire into `_precheck_sell_against_alpaca`
- **`agent.py:331-357` SQL-injection allowlist hardening** — not exploitable today; add inline `# noqa` comment when next touched
- **`classifier.py:65` `_validate_ticker` trailing punctuation** — minor, fold into next classifier touch

---

## How to use this plan

- Land **Tier 0 as a single PR** before the next live session. Each task gets a checkbox tick and a brief commit referencing the task ID.
- **Tier 1 in two PRs**: T1.1–T1.7 (executor/strategist/agent), then T1.8–T1.11 (learning).
- **Tier 2 opportunistic**: pick up when adjacent code is touched.
- After each tier ships, update `v2/BUGS.md` if you maintain it as a backlog.
