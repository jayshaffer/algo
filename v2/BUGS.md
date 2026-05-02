# v2 Bug Audit — 2026-05-02

Comprehensive bug audit of the v2 codebase. Findings ordered by **(probability of firing) × (blast radius when it fires) × (irreversibility)**.

The trading-money path and the *learning-corruption* path are both top-tier — the latter is silent and compounds over time.

## Progress

- ✅ **P0:** 5/5 fixed
- ☐ **P1:** 0/8 fixed
- ✅ **P2:** 1/12 fixed (P2.18)
- ☐ **P3:** 0/15 fixed

**Tests added so far:** +12 in v2 suite (704 → 716 passing).

---

## P0 — Fix this week

Active wrongness on every relevant run.

### ✅ P0.1 — Sell-decision sign inversion in backfill
- **File:** `v2/backfill.py:90-104, 161`
- **Bug:** `calculate_outcome` computes `((exit_price - entry_price) / entry_price)` and *negates* the result for sells. But a `sell` here closes a long position, not opens a short. Sell at $200; ticker rises to $220 → `outcome_7d = -10%` recorded against the sell decision and rolled into whatever signal motivated the exit.
- **Impact:** Sells of winners scored as losses; sells of losers scored as wins. Inverts the sign of attribution for *every* signal that motivates exits. The learning system is currently gradient-descending in the wrong direction on closes.
- **Reproduction:** Sell AAPL at $200; AAPL is at $220 a week later → `outcome_7d = -10%` against the sell, hurting the signal that prompted it.
- **Fix (2026-05-02):** Negate `benchmark` for sells too in `backfill.py`, so downstream `alpha = outcome - benchmark` measures whether the sell beat the market. Tests: `TestSellAlphaSign` (3 cases) in `tests/v2/test_backfill.py`.
- **Historical re-backfill:** Single `UPDATE decisions SET benchmark_7d = -benchmark_7d, benchmark_30d = -benchmark_30d WHERE action='sell'` ran in prod (38 rows). Followed by `compute_signal_attribution` recompute. Attribution shift: `thesis` bucket went from `-0.82% avg alpha, 46.7% beat rate` (n=30) to `-0.014% avg alpha, 50.0% beat rate` (n=30) — strategist no longer being told its theses are systematically losing to SPY.

### ✅ P0.2 — `float(qty)` casts Decimal share counts
- **File:** `v2/executor.py:204` (market), `:253` (limit), `:445` (`calculate_position_size`)
- **Bug claim:** `qty=float(qty)` causes ULP drift; precheck-trimmed sells get rejected with "Insufficient available shares."
- **Reality (verified during fix):** Audit's specific reproduction does NOT reproduce. `float(Decimal)` is exact for all typical fractional-share quantities; for very-long Decimals (intent-division producing 28 sig figs) `float()` rounds *down*, never *up* — which would make Alpaca *more* likely to accept, not reject. Additionally, alpaca-py's Pydantic model is typed `Optional[float]` and coerces ALL inputs (str, Decimal, float) to `float` before serialization, so `qty=str(qty)` would be cosmetic.
- **Fix shipped anyway as defense-in-depth (2026-05-02):** Quantize qty to 9 decimal places with `ROUND_DOWN` (Alpaca's documented precision limit) before submission. Eliminates any sub-9-decimal ambiguity from internal Decimal math. Applied to `execute_market_order` and `execute_limit_order`. Tests: `TestQuantityPrecision` (3 cases) in `tests/v2/test_executor.py`.
- **Note:** The "Insufficient available shares" failures the audit blamed on this are more likely caused by **P1.7** (sell precheck fails open on transient errors). Promoting P1.7 priority recommended.

### ✅ P0.3 — LLM `category` strings unvalidated → `news_signal:unknown` etc.
- **File:** `v2/classifier.py:155, 163, 332`
- **Bug:** `entry.get("category", "noise")` accepts any string Haiku emits. Macro `category` defaults to `"geopolitical"` when missing; sector defaults to literal string `"sector"` (not in the documented enum).
- **Impact:** This is the *upstream* source of the `news_signal:unknown` orphan bucket already patched downstream on 2026-05-02 (`attribution.py` + `patterns.py`). Any out-of-vocabulary category becomes a junk attribution bucket. Cheap fix, structural cleanup.
- **Reproduction:** Force LLM to emit `category: "earnigns"` (typo) → row stored verbatim, attribution joins on the typo'd category.
- **Fix (2026-05-02):** Defined `VALID_TICKER_CATEGORIES` and `VALID_MACRO_CATEGORIES` frozensets in `classifier.py`. Added `_coerce_category` helper that returns the value if valid or coerces to default + logs warning. Applied to both `_build_classification_result` (batch path) and `classify_ticker_news` (single-ticker path). Tests: 3 new cases in `TestBuildClassificationResult`.
- **Follow-up (P2 candidate):** Add DB CHECK constraint via migration to prevent future regression.

### ✅ P0.4 — Stale `account_equity`/`buying_power` stamped on every decision
- **File:** `v2/trader.py:646-647, 544-546`
- **Bug:** `_refresh_buying_power` updates the *local* `buying_power`/`portfolio_value` variables inside `_execute_decisions` but doesn't thread them back into `account_info`. `_log_decisions` then reads `account_info["portfolio_value"]` and `account_info["buying_power"]` — the snapshot from *before* any trades — and stamps those on every decision row regardless of when in the session the trade executed.
- **Impact:** Corrupts the historical row that strategy reflection learns from. Compounds over months.
- **Reproduction:** Account starts at $100k buying power; session executes 5 buys totaling $30k; all 5 decision rows record `buying_power=100000`.
- **Fix (2026-05-02):** `_execute_decisions` now builds `decision_account_states: dict[int, dict]` keyed by decision index, captured at the top of each loop iteration (state at trade time). Returned as a 4th tuple element and threaded to `_log_decisions`, which uses per-decision state instead of the pre-session snapshot. Test: `test_each_decision_logged_with_account_state_at_trade_time` in `TestBuyingPowerRefresh`.

### ✅ P0.5 — `trim_to_portfolio_pct=0` sells the entire position
- **File:** `v2/intents.py:48-53, 78-82`
- **Bug:** `_require_pct` accepts magnitude=0. For `trim_to_portfolio_pct=0` with `held > 0`, `target_value=0`, sells the entire position.
- **Impact:** One LLM ambiguity → full liquidation. Low frequency but maximum blast radius.
- **Reproduction:** LLM emits `trim_to_portfolio_pct` magnitude=0 thinking it means "no trim"; system sells everything.
- **Fix (2026-05-02):** Reject `magnitude=0` specifically for `trim_to_portfolio_pct` with `IntentError("trim_to_portfolio_pct magnitude must be > 0 — use exit_full to liquidate")`. Forces LLM to be explicit about full exits. Test: `test_trim_to_portfolio_pct_zero_raises`.

---

## P1 — Fix this sprint

Real-money or public-reputation risk; can fire any week.

### ☐ P1.6 — Duplicate-decision dedup runs *after* order submission; no `client_order_id`
- **File:** `v2/trader.py:628-634` (dedup in Step 6, after Step 5 submits orders), `v2/executor.py:202` (no idempotency key on submit)
- **Bug:** `check_decision_exists` runs in `_log_decisions` (Step 6) — *after* `_execute_decisions` (Step 5) has already submitted. Network blip mid-`insert_decision` + operator rerun → duplicate orders submitted with no DB-side guard and no broker-side idempotency.
- **Impact:** Real duplicate orders.
- **Reproduction:** Force `insert_decision` to raise after Alpaca submit succeeds; rerun session; second submit lands a real duplicate order.

### ☐ P1.7 — Sell precheck fails open on transient Alpaca errors *(promoted: actual cause of "Insufficient shares" errors)*
- **File:** `v2/trader.py:223-230`
- **Bug:** `get_live_available_qty` raises on any non-404 APIError (rate limit, transient 5xx, network). Handler logs warning and **returns True** — proceeding without the precheck. The defense against stale state is itself disabled by errors.
- **Impact:** Risk-control bypass precisely when Alpaca is degraded — i.e. when stale state is most likely.
- **Reproduction:** Alpaca returns 503 during precheck; trader proceeds with sell of stale-known qty.

### ☐ P1.8 — Full Alpaca order UUIDs published to `decisions.json` on public site
- **File:** `v2/dashboard_publish.py:189-194`
- **Bug:** Publish pipeline writes full `order_id` for every decision into `decisions.json` uploaded to Cloudflare Pages. Frontend at `public_dashboard/app.js:36-39, 372` only *displays* the truncated form — the unredacted IDs are still in the publicly-fetchable JSON.
- **Impact:** Privacy leak; broker order IDs publicly exposed in bulk.
- **Reproduction:** `curl https://bbottomcap.com/data/decisions.json | jq '.[0].order_id'` returns the full UUID.

### ☐ P1.9 — `tweets` table has no idempotency; double-post risk on rerun
- **File:** `db/init/008_tweets.sql:3-12`, `v2/twitter.py:280-343`, `v2/bluesky.py:325-386`
- **Bug:** No UNIQUE on `(session_date, tweet_type, platform)`; `insert_tweet` blindly INSERTs every call. If `post_tweet` succeeds but `insert_tweet` raises, stage isn't marked failed (`result.tweet_posted = True` is set). Manual rerun → fresh tweet generated and posted again. Twitter `client.create_tweet` has no `client_request_id`.
- **Impact:** Public reputation risk on the Bikini Bottom Capital account.
- **Reproduction:** Stub `insert_tweet` to raise → rerun stage → two tweets land on Twitter.

### ☐ P1.10 — `POST /api/theses/<id>/close` has no auth, listens on 0.0.0.0:3000
- **File:** `v2/dashboard/app.py:124-141`
- **Bug:** Endpoint accepts any caller and mutates DB state. Dashboard listens on `0.0.0.0:3000`.
- **Impact:** If port reachable from outside localhost, anyone can close any active thesis. Severity depends on network exposure — verify Docker port mapping.

### ☐ P1.11 — `paper:session` Taskfile target ships `--force`
- **File:** `Taskfile.yml:104`
- **Bug:** `paper:session` target includes `--force`, which bypasses `_check_and_record_session` idempotency. Prod `session` target does not.
- **Impact:** Every paper rerun duplicates all writes (decisions, tweets, snapshots), polluting the paper learning corpus.
- **Reproduction:** Run `task paper:session` twice in one day → second run re-enters all stages.

### ☐ P1.12 — `--dry-run` only gates the executor; strategist + reflection still write
- **File:** `v2/session.py:268`
- **Bug:** `dry_run` is forwarded only to `run_trading_session`. Strategist still writes theses, playbooks, playbook_actions, thesis_signals, strategy_memos. Reflection still writes rules, memos, identity updates.
- **Impact:** Misleading flag. Users running `--dry-run` to "preview" mutate strategy state.
- **Reproduction:** `python -m v2.session --dry-run` → check `select count(*) from theses where created_at > now() - interval '1 hour'` returns rows.

### ☐ P1.13 — `wait_for_fill` cancel-on-timeout doesn't query for partial fills
- **File:** `v2/executor.py:316-328`; consequence in `v2/trader.py:300-313`
- **Bug:** Timeout path cancels and returns `success=False, filled_qty=None`. Order may have been partially filled before cancel. Trader returns `_DecisionOutcome(False, ...)` and logs decision row with `order_id=None, quantity=requested, price=quote_price`.
- **Impact:** Position sync reconciles shares but the decision-to-fill link is gone — attribution can't see what actually happened.
- **Reproduction:** Order fills 50/100 before 30s timeout; cancel succeeds for the rest; DB has decision with `order_id=None` and the 50 filled shares are detached from any decision row.

---

## P2 — Fix this quarter

Slow learning rot; impact measured in months of bad rule proposals.

### ☐ P2.14 — `decision_signals.signal_id` has no FK constraint
- **File:** `db/init/005_redesign.sql:17-22`; write callers `v2/trader.py:391-413`, `v2/database/trading_db.py:466-482`
- **Bug:** `validate_signal_refs` (Python validator at `v2/agent.py:347`) is the only line of defense. No FK, no DB-level check.
- **Impact:** One regression in the validator (or a future caller bypassing it) and the orphan bucket returns. Structural fix that ends a class of bugs.

### ☐ P2.15 — `_classify_batch` zips Haiku output positionally + pads with `noise`
- **File:** `v2/classifier.py:284-295`
- **Bug:** Loop pads with `noise` when Haiku returns fewer entries than headlines. Pairs `parsed[i]` with `headlines[i]`/`published_ats[i]` purely positionally — assumes Haiku preserves order.
- **Impact:** If Haiku reorders, ticker/sentiment lands on the wrong headline. If Haiku returns short, headlines silently become `noise`. Direct upstream of attribution corruption.

### ☐ P2.16 — Alpaca news `id` captured then discarded
- **File:** `v2/news.py:59-68`, `v2/pipeline.py:50-60`
- **Bug:** `NewsItem.id` is captured from Alpaca but never propagated to the row. `news_signals` has no column for it.
- **Impact:** Dedup falls back to `(ticker, md5(headline), published_at)` from `db/init/014_news_signal_dedup.sql`. 1-second republish jitter creates duplicates; identical content with different timestamps slips through.
- **Fix:** Add `alpaca_id` column with UNIQUE index.

### ☐ P2.17 — `signal_attribution.sample_size` mismatched against 30d metrics
- **File:** `v2/attribution.py:60-67`
- **Bug:** `WHERE outcome_7d IS NOT NULL AND alpha_7d IS NOT NULL` admits decisions that don't yet have 30d outcomes. `COUNT(DISTINCT decision_id)` includes those, but `AVG(alpha_30d)` and `AVG(CASE WHEN alpha_30d > 0 …)` silently drop them via NULL averaging.
- **Impact:** Reported `n` overstates support behind 30d numbers → strategist over-confident in weakly-supported rules.
- **Reproduction:** Backfill 7d but not 30d → row reports `n=30, avg 30d alpha=+0.5%` even though only 5 decisions actually had a 30d outcome.

### ✅ P2.18 — `thesis` bucket has no orphan-FK guard
- **File:** `v2/attribution.py:32-56`, `v2/patterns.py:55-76`
- **Bug:** Orphan-FK guard added 2026-05-02 covers `news_signal` and `macro_signal` (`(ds.signal_type != 'news_signal' OR ns.id IS NOT NULL)`), but no equivalent guard exists on `thesis`. Phantom thesis IDs still inflate the `thesis` aggregate.
- **Impact:** Same shape as the fixed `news_signal:unknown` artifact, just for thesis IDs.
- **Verified scope (2026-05-02):** 6 historical orphan rows in `decision_signals` (`signal_type='thesis', signal_id=0`), all from 2026-02-11–12 (pre-validator era). 5 are HOLDs (already excluded by `action IN ('buy','sell')`); 1 is a real buy (GLD 2026-02-12).
- **Fix (2026-05-02):** Added `LEFT JOIN theses t ON ds.signal_type='thesis' AND t.id=ds.signal_id` and `(ds.signal_type != 'thesis' OR t.id IS NOT NULL)` to both `attribution.py` and `patterns.py`. Tests: `test_excludes_thesis_orphans` in `TestOrphanSignalFiltering` (attribution) and `TestAnalyzeSignalCategories` (patterns).
- **Recompute landed:** Re-ran `compute_signal_attribution` against prod. Thesis bucket shifted from `n=30, alpha=-0.014%, win_rate=50.0%` to `n=29, alpha=-0.066%, win_rate=48.3%`. Small but truthful.

### ☐ P2.19 — `patterns.py` raw outcomes vs `attribution.py` alpha
- **File:** `v2/patterns.py:54-92, 94-124` vs `v2/attribution.py:43-64`
- **Bug:** `analyze_signal_categories` and `analyze_sentiment_performance` use `AVG(d.outcome_7d)` (raw outcome). `compute_signal_attribution` uses alpha vs SPY. Same category, different number, depending on entry point.
- **Impact:** Strategist reads contradictory "STRONG/WEAK" (constraint block) vs "best/worst" (pattern report) labels for the same category. During a bull run, raw outcomes look great while alpha looks poor.

### ☐ P2.20 — No retry/backoff on Anthropic in classifier; batch failure → 50× fan-out
- **File:** `v2/classifier.py:198-216, 244-256, 268-274, 315-326`
- **Bug:** All `client.messages.create` calls wrapped in bare `except Exception` returning `noise`. No 429/529/transient-error handling. Batch JSON failure falls back to N individual calls (up to `batch_size=50`) with no rate-limit handling.
- **Impact:** 50× API request spike on a single batch failure. If original failure was 429, fallback hammers within the same minute.

### ☐ P2.21 — `insert_news_signals_batch` returns `len(signals)` regardless of `ON CONFLICT DO NOTHING` skips
- **File:** `v2/database/trading_db.py:24-33`; consumed at `v2/pipeline.py:64`
- **Bug:** `ON CONFLICT DO NOTHING` may skip rows but function returns `len(signals)` rather than `cur.rowcount`. Stats lie about what was persisted.
- **Impact:** Stage telemetry untrustworthy. Re-runs report identical non-zero `ticker_signals_stored` even when zero rows inserted.

### ☐ P2.22 — `tool_write_playbook` is non-atomic across upsert + delete + N inserts
- **File:** `v2/tools.py:374-403`, `v2/database/trading_db.py:420-429`
- **Bug:** `upsert_playbook` → `delete_playbook_actions` → loop of `insert_playbook_action`, each in its own `get_cursor()` (own connection, own transaction). Mid-loop failure → playbook row exists, old actions gone, only some new actions inserted.
- **Impact:** Executor trades on incomplete intent.
- **Reproduction:** Force `insert_playbook_action` to raise on the third action → DB left with playbook + 2 actions, no rollback.

### ☐ P2.23 — Strategist resume short-circuits without verifying playbook row exists
- **File:** `v2/session.py:209-211, 247-256`; consequence in `v2/context.py:424`
- **Bug:** Resume logic: `"strategist" in completed_stages` → skip. Executor short-circuit at `:247` only fires on `result.strategist_error`. On resume after manual playbook cleanup, `result.strategist_error` is `None` (never ran this invocation), so executor runs `get_pending_playbook_actions(playbook["id"])` against a `None` playbook → `TypeError`.
- **Impact:** Crashes recovery flow on certain partial-failure replays.

### ☐ P2.24 — `_persist_strategist_memo` runs before `write_playbook` validation
- **File:** `v2/session.py:215-230`
- **Bug:** Memo is committed in its own transaction before `get_playbook(session_date) is None` check raises `RuntimeError`. Failure path doesn't call `complete_session_stage`, so next run re-runs the strategist and inserts a *second* memo for the same date.
- **Impact:** Duplicate `strategist_notes` in `strategy_memos` for the same `session_date`. Reflection sees both, double-counting the strategist's voice.

### ☐ P2.25 — `tool_get_session_summary` reads 30d decisions but joins signals only for latest 10
- **File:** `v2/strategy.py:213`
- **Bug:** `decisions` is 30-day list; signal linkage fetched only for first 10. Prompt header says "Decisions ({len(decisions)})" using the full count.
- **Impact:** Reflection LLM infers signal patterns from a sliced sample → biased rule proposals.

---

## P3 — Code quality / latent

Clean up when adjacent code is touched.

### ☐ P3.26 — Daily P&L tweet doesn't subtract deposits; dashboard does
- **File:** `v2/twitter.py:106-111` vs `v2/dashboard_publish.py:311-314`
- **Bug:** Tweet path computes `day_pnl = portfolio - prev` with no deposit adjustment. Dashboard path subtracts `daily_deposit`.
- **Impact:** Disagreeing public numbers for the same day.

### ☐ P3.27 — `INTERVAL '%s days'` parameterization fragile across psycopg drivers
- **File:** `v2/database/dashboard_db.py:24, 34, 49`, `v2/database/trading_db.py:36-50, 77-91`
- **Bug:** `INTERVAL '%s days'` relies on psycopg2 substituting `%s` *inside* a quoted SQL literal. Works on psycopg2; broken on psycopg3. Robust form: `INTERVAL '1 day' * %s`.

### ☐ P3.28 — `_enrich_snapshots_with_deposits` fallback can double-credit early deposits
- **File:** `v2/dashboard_publish.py:99-126`
- **Bug:** First loop credits any deposit dated strictly before a snapshot. Fallback at lines 111-125 says "if first snapshot still has 0 cumulative, credit deposits on/before first snapshot" — but writes `credit` to *every* snapshot, including ones the first loop already credited.
- **Impact:** TWR / Total Return calculations show inflated cost basis and depressed return %.

### ☐ P3.29 — `extract_final_text` returns `None` if final assistant message is tool_use-only
- **File:** `v2/claude_client.py:285-296`
- **Bug:** Iterates `reversed(messages)`, returns first text block. If strategist's final action is `write_playbook` with no trailing text, function returns `None` → memo silently skipped.

### ☐ P3.30 — Sector concentration is advisory-only — no hard gate at order-submit
- **File:** `v2/trader.py:497-548`, `v2/risk.py`
- **Bug:** Sector concentration computed at context-build and injected as a *string warning* into `risk_notes`. Trader never blocks a trade that violates concentration.

### ☐ P3.31 — `_call_with_retry` doesn't catch `BadRequestError`/context-length-exceeded
- **File:** `v2/claude_client.py:21-25`
- **Bug:** Only `RateLimitError`, `InternalServerError`, `APIConnectionError` are retryable. Context-length-exceeded propagates up, captured as `strategist_error` with no graceful degradation.

### ☐ P3.32 — `get_attribution_summary` recomputed 3+ times per session, no memoization
- **File:** `v2/context.py:348-350`, `v2/tools.py:320-323`
- **Bug:** No caching; called from strategist tool loop AND from `build_trading_context`/`build_executor_input`.
- **Impact:** Cost (latency, DB load); not correctness.

### ☐ P3.33 — `_count_actions` substring match treats soft-guard rejections as successful identity updates
- **File:** `v2/strategy.py:356-380`
- **Bug:** `tool_update_strategy_identity`'s soft-guard returns a string containing `"identity was updated"` (line 117). Heuristic `"identity updated" in result_text.lower()` matches — reflection result reports `identity_updated=True` even when the guard rejected the update.

### ☐ P3.34 — `MAX_POSITION_PCT` defined in two places
- **File:** `v2/intents.py:15` ("mirror v2/agent.py")
- **Bug:** Drift risk acknowledged in code comment.

### ☐ P3.35 — `setup_logging` no-ops if root has handlers
- **File:** `v2/log_config.py:17-18`
- **Bug:** `if root.handlers: return` — if any third-party library added a handler before `setup_logging`, per-file handlers and console handler skipped, `os.makedirs(log_dir)` not run. Logs vanish silently.

### ☐ P3.36 — `_session_retirements` module-level global
- **File:** `v2/strategy.py:29-30, 176-185`
- **Bug:** State leaks between concurrent invocations. If paper + prod ever import `strategy.py` in one Python process, retirements clobber each other.

### ☐ P3.37 — v2 Flask dashboard references `templates/` that doesn't exist
- **File:** `v2/dashboard/app.py:32, 44, 51, 62, 78, 93, 117`
- **Bug:** `Flask(__name__)` defaults `template_folder='templates'` (relative). No `v2/dashboard/templates/` directory. Likely dead code or undocumented template path config.

### ☐ P3.38 — `success_rate=None` placeholder in `get_thesis_stats`
- **File:** `v2/database/dashboard_db.py:154`
- **Bug:** Stub returned to v2 dashboard.

### ☐ P3.39 — `subprocess.run(["wrangler", ...])` has no timeout
- **File:** `v2/dashboard_publish.py:443-450`
- **Bug:** No `timeout=` kwarg. A hung `wrangler` blocks the session indefinitely.

### ☐ P3.40 — Haiku ticker extraction has no allowlist
- **File:** `v2/classifier.py:150-159`
- **Bug:** `tickers = entry.get("tickers", [])` iterated raw with only `.upper()` sanitization. Common-word false-positives ("ARE", "IT", "ON", "SO", "GO") become rows. Hallucinated tickers like "FAANG" or "$TSLA" also persist — no FK to a known ticker universe.

---

## Cross-cutting observations

- **Biggest risk to attribution data quality:** P0.1 + P0.3 + P2.14 + P2.15. The orphan-FK problem is structurally unfinished — downstream reads were patched but the write side still depends on a Python validator with a single call site.
- **Biggest risk to public reputation:** P1.8 + P1.9 + P1.10. All cheap to fix.
- **Biggest cost/reliability risk:** P2.20 — no retry/backoff with a 50× fan-out fallback.
- **Paper/prod isolation depends on `.env.paper` *not* setting Twitter/Bluesky/Cloudflare vars** — if those leak via shell or `--env-file` override, paper data publishes as prod.
- **Stage 4 reflection uses 30-day window; Stage 0 attribution uses 90-day window.** Rules "proven" by attribution may be invisible in reflection summary, and vice versa.
- **`_run_learning_refresh` runs before pipeline stage but after executor stage of the *prior* session** — attribution constraints fed to strategist always lag by exactly one session. Worth verifying that's intentional.

## Suggested triage order

P0 work complete. **Recommended next:**

1. **P1.7** (sell precheck fails open) — promoted; this is the actual root cause of the "Insufficient available shares" failures the audit had blamed on P0.2.
2. **P1.8 + P1.9 + P1.10** (public exposure trio) — all cheap, all real-world impact.
3. **P2.14** (FK constraint on `decision_signals.signal_id`) — closes the orphan-FK class permanently.

P1 is largely orthogonal — work can be split.

## Verification suggestion

Several P2/P3 items are flagged based on code reading, not contract verification. Worth running through context7 against current SDK docs:

- **alpaca-py:** `account.cash` nullability (`v2/executor.py:60-71`), `pos.qty` typing in `sync_positions_from_alpaca`, `client.get(...)` private-method use in `get_net_deposits`, `wait_for_fill` status enum coverage (`"canceled"` vs `"cancelled"` vs `"pending_cancel"` vs `"done_for_day"`).
- **anthropic SDK:** which exceptions belong in `RETRYABLE_ERRORS` (`v2/claude_client.py:21-25`); tool_result content-shape contract (`str` vs list-of-blocks for `content`).
- **psycopg2 vs psycopg3:** literal-substitution-inside-quotes for `INTERVAL '%s days'` pattern.
