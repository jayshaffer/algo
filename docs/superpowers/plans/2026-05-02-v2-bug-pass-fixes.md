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

### T2.1 — `session.py:131-132` "session already completed" exits 1
- [ ] Add `result.idempotent_skip: str | None` field; exclude from `_ERROR_FIELDS`
- [ ] Route the early-return through it; `main()` exits 0 with a distinct log line
- [ ] Test: re-run session same day without `--force`; assert exit 0

### T2.2 — `session.py:146-158` stage 0 bypasses session_stages tracking
- [ ] Wrap `_run_learning_refresh` with `_start_stage(session_id, "learning")` / `_complete_stage` / `_fail_stage`
- [ ] Test: learning stage success/failure both produce a `session_stages` row

### T2.3 — `news.py:43,66` + schema TZ handling
- [ ] `news.py:43`: `datetime.now(timezone.utc) - timedelta(...)` instead of naive `datetime.now()`
- [ ] Migration: `ALTER TABLE news_signals ALTER COLUMN published_at TYPE TIMESTAMPTZ USING published_at AT TIME ZONE 'UTC'`
- [ ] Same for `macro_signals.published_at`
- [ ] Test: publish_at round-trips as tz-aware UTC

### T2.4 — `market_data.py:71-96` `get_bar_change(days=1)` returns 0.0 on insufficient data
- [ ] Require `len(symbol_bars) > days` strictly before computing
- [ ] Return `None` otherwise
- [ ] Update `format_market_snapshot` and downstream context builders to render `None` as `"N/A"` (not 0%)
- [ ] Test: single-bar response returns None, not 0.0

### T2.5 — `patterns.py:149,269` `total_pnl_7d` sums percentages
- [ ] Either rename to `sum_pct_returns` and order by `AVG(outcome_7d)`, OR
- [ ] Compute real dollar P&L: `SUM(quantity * price * outcome_7d / 100)`
- [ ] Update report header in `analyze_ticker_performance` to match the chosen semantics
- [ ] Test: ticker leaderboard ordering matches the chosen metric

### T2.6 — `strategy.py` asymmetric session caps
- [ ] Add `MAX_PROPOSALS_PER_SESSION = 3` (mirrors `MAX_RETIREMENTS_PER_SESSION`)
- [ ] Wire through `tool_propose_rule` with the same ContextVar pattern
- [ ] Test: 4th proposal in same session is rejected with clean error

### T2.7 — `strategy.py:134,188` TZ comparison hazard
- [ ] Convert `current["created_at"]` to UTC before `.date()`, OR
- [ ] Use `datetime.now(timezone.utc).date()` consistently
- [ ] Apply to both 3-day identity throttle and 5-day rule-tenure check
- [ ] Test: a created_at at 23:30 ET (= 03:30 UTC next day) compared from 09:00 ET 3 days later behaves consistently

### T2.8 — `executor.py:341-379` `wait_for_fill` partial-fill semantics
- [ ] On post-cancel re-fetch returning partial fill, return `success=True` with `filled_qty=Decimal("0")` only if confirmed zero
- [ ] Otherwise propagate explicit `unknown_partial_fill` marker so trader logic can handle vs treating as failure
- [ ] Test: timeout + re-fetch shows partial fill; assert decision row is logged with the partial qty

### T2.9 — `executor.py:317-339` `wait_for_fill` no try/except around get_order_by_id
- [ ] Wrap with try/except; on transient error sleep + retry up to 3 times before cancelling
- [ ] Test: mock `get_order_by_id` to raise once then succeed; loop continues

### T2.10 — `trader.py:213-223,300,525,709,721` `date.today()` evaluated repeatedly
- [ ] Capture `session_date = datetime.now(ZoneInfo("America/New_York")).date()` once at top of `run_trading_session`
- [ ] Thread through every `date.today()` callsite in trader.py
- [ ] Test: mock the clock to roll past midnight mid-session; assert all decision rows + client_order_ids share one date

### T2.11 — `executor.py:153-186` `sync_orders_from_alpaca` deletes filled orders
- [ ] Decide intent: transient `open_orders` (current behavior) vs audit log
- [ ] If transient, add a docstring clarifying; if audit, query with `status=ALL, after=session_start` and only delete confirmed-not-found

### T2.12 — `ideation_claude.py:161-182` `count_actions` conflates adopt with create
- [ ] Have `tool_adopt_thesis` return `"Adopted thesis ID ..."` (distinct prefix)
- [ ] Add `adopted` counter in `count_actions`; surface separately in the dashboard memo
- [ ] Test: a session with 2 creates + 3 adopts reports `created=2, adopted=3`

### T2.13 — `formation.py:27-32` + DB queries strict `>` excludes boundary day
- [ ] `database/trading_db.py:183-190`: change `>` to `>=` for `get_recent_decisions` window
- [ ] Audit similar `> CURRENT_DATE - INTERVAL` shapes in the same file; fix consistently
- [ ] Document the lookback semantic in a one-liner so it doesn't regress

### T2.14 — `ideation_claude.py:310-337` + `formation.py` orphan positions fetched twice
- [ ] Fetch once in `run_strategist_loop`; pass the list into both `build_formation_context` and `_build_orphan_block`
- [ ] Test: assert `get_orphan_positions` is called exactly once per strategist run

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
