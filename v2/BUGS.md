# v2 Bug Audit — 2026-05-02

Comprehensive bug audit of the v2 codebase. Findings ordered by **(probability of firing) × (blast radius when it fires) × (irreversibility)**.

The trading-money path and the *learning-corruption* path are both top-tier — the latter is silent and compounds over time.

## Progress

- ✅ **P0:** 5/5 fixed
- ☑️ **P1:** 9/10 fully fixed + 1 partial (P1.9 — app-level guard shipped, DB UNIQUE constraint deferred behind historical-row dedup; P1.6, P1.7, P1.8, P1.10, P1.11, P1.12, P1.13, P1.14 *(post-audit)*, P1.15 *(2nd pass)* fully fixed)
- ✅ **P2:** 13/13 fixed
- ✅ **P3:** 17/17 fixed

**Tests added so far:** +95 in v2 suite (704 → 798 passing, verified 2026-05-02); +2 in v1 dashboard suite for P1.10.

See [**Residual follow-ups**](#residual-follow-ups) for the three deferred items called out inside individual fixes (P0.3 DB CHECK, P1.9 historical dedup + UNIQUE, P3.38 rigorous thesis success metric).

## 2026-05-02 second pass

A second audit pass focused on modules barely covered by the original audit: `formation.py`, `market_data.py`, `entertainment.py`, `learn.py`, `context.py`, `ideation_claude.py`, `agent.py`, `tools.py`, `strategy.py`, `attribution.py`, and `v2/database/*`. Four new findings landed:

- **P1.15** — Stage 4 reflection had no memo-written enforcement (parallel to P2.24 for the strategist).
- **P2.26** — `insert_decision_signals_batch` / `insert_thesis_signals` returned `len()` instead of `cur.rowcount` (parallel to P2.21 for news/macro).
- **P3.41** — Categories with exactly 0 alpha were silently dropped from the strategist-facing summary (truthy check on `Decimal(0)`).
- **P3.42** — `update_strategy_identity`'s rejection text told the LLM to "call again" but the guard fires unconditionally on retry.

Three additional observations were noted but not fixed in this pass — see [Cross-cutting observations](#cross-cutting-observations).

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

### ✅ P1.6 — Duplicate-decision dedup runs *after* order submission; no `client_order_id`
- **File:** `v2/trader.py:628-634` (dedup in Step 6, after Step 5 submits orders), `v2/executor.py:202` (no idempotency key on submit)
- **Bug:** `check_decision_exists` runs in `_log_decisions` (Step 6) — *after* `_execute_decisions` (Step 5) has already submitted. Network blip mid-`insert_decision` + operator rerun → duplicate orders submitted with no DB-side guard and no broker-side idempotency.
- **Impact:** Real duplicate orders.
- **Reproduction:** Force `insert_decision` to raise after Alpaca submit succeeds; rerun session; second submit lands a real duplicate order.
- **Fix (2026-05-02):** Two layers.
  1. **App-level pre-submit dedup.** Moved `check_decision_exists(date.today(), ticker, action)` into `_prepare_decision` in `v2/trader.py` so it fires before `execute_market_order` is called. Existing rows for `(today, ticker, action)` mark the decision invalid and increment `trades_failed` without submitting. The post-log dedup at step 6 stays as belt-and-suspenders. Skipped in dry-run so preview still reflects what would happen.
  2. **Broker-level idempotency.** New `_client_order_id(decision, session_date)` generates a deterministic key — `algo-YYYYMMDD-{b/s}-{TICKER}-{playbook_action_id|"op"}` — passed to `execute_market_order` and `execute_limit_order` (both updated to plumb `client_order_id` to Alpaca's `MarketOrderRequest` / `LimitOrderRequest`). If the prior run's submit succeeded but our DB write itself died (no row to dedup against), the broker rejects the second submit because the same key already mapped to a live order. No duplicate fills.
- **Tests:** `TestPreSubmitDedup` (3 cases), `TestClientOrderIdPlumbing` (2 cases), `TestClientOrderId` in `test_executor.py` (3 cases) verifying both the pass-through and the no-leak path.

### ✅ P1.7 — Sell precheck fails open on transient Alpaca errors *(promoted: actual cause of "Insufficient shares" errors)*
- **File:** `v2/trader.py:223-230`
- **Bug:** `get_live_available_qty` raises on any non-404 APIError (rate limit, transient 5xx, network). Handler logs warning and **returns True** — proceeding without the precheck. The defense against stale state is itself disabled by errors.
- **Impact:** Risk-control bypass precisely when Alpaca is degraded — i.e. when stale state is most likely.
- **Reproduction:** Alpaca returns 503 during precheck; trader proceeds with sell of stale-known qty.
- **Fix (2026-05-02):** Fail closed on any precheck exception. Extracted `_reject` helper inside `_precheck_sell_against_alpaca` and reused it for both the zero-available branch and the new exception branch — sell is marked invalid, playbook action set to `skipped`, `trades_failed` incremented by the caller. Skipping a legit sell is recoverable next session; submitting on stale state during Alpaca degradation is not. Tests: replaced `test_live_availability_check_exception_proceeds` (codified broken behavior) with `test_live_availability_check_exception_skips_sell` and added `test_live_availability_check_exception_updates_playbook_action` in `tests/v2/test_trader.py::TestAlpacaPrecheck`.

### ✅ P1.8 — Full Alpaca order UUIDs published to `decisions.json` on public site
- **File:** `v2/dashboard_publish.py:189-194`
- **Bug:** Publish pipeline writes full `order_id` for every decision into `decisions.json` uploaded to Cloudflare Pages. Frontend at `public_dashboard/app.js:36-39, 372` only *displays* the truncated form — the unredacted IDs are still in the publicly-fetchable JSON.
- **Impact:** Privacy leak; broker order IDs publicly exposed in bulk.
- **Reproduction:** `curl https://bbottomcap.com/data/decisions.json | jq '.[0].order_id'` returns the full UUID.
- **Fix (2026-05-02):** Added `_redact_order_id` helper in `v2/dashboard_publish.py` that mirrors the frontend's `shortOrderId` (8 chars + `...` for ids > 12 chars). Applied in `gather_dashboard_data` when serializing decision rows, so full UUIDs never leave the DB context. Tests: `TestRedactOrderId` (3 cases) + `test_decisions_redact_full_order_uuid` (end-to-end through `gather_dashboard_data`, asserts the full UUID is absent from the JSON-encoded payload). Goes live on next dashboard publish.

### ☑️ P1.9 — `tweets` table has no idempotency; double-post risk on rerun *(partial — app-level guard shipped, DB constraint deferred)*
- **File:** `db/init/008_tweets.sql:3-12`, `v2/twitter.py:280-343`, `v2/bluesky.py:325-386`
- **Bug:** No UNIQUE on `(session_date, tweet_type, platform)`; `insert_tweet` blindly INSERTs every call. If `post_tweet` succeeds but `insert_tweet` raises, stage isn't marked failed (`result.tweet_posted = True` is set). Manual rerun → fresh tweet generated and posted again. Twitter `client.create_tweet` has no `client_request_id`.
- **Impact:** Public reputation risk on the Bikini Bottom Capital account.
- **Reproduction:** Stub `insert_tweet` to raise → rerun stage → two tweets land on Twitter.
- **Fix (2026-05-02):** Two-part app-level guard.
  1. **Pre-stage rerun check.** New `posted_tweet_exists(session_date, tweet_type, platform)` helper in `v2/database/trading_db.py`. `run_twitter_stage` and `run_bluesky_stage` short-circuit before generate/post if a posted=TRUE row already exists for `(session_date, "recap", platform)` — closes the common rerun-after-success case and avoids re-burning Anthropic spend.
  2. **No false-success on DB write failure.** When `post_tweet` succeeds but `insert_tweet` raises (transient DB error), `result.tweet_posted` (and `result.post_posted`) are now set to **False**, not True. Operator monitoring catches the discrepancy (post on platform but no DB row); the audit row in `result.errors` survives. Previously the stage claimed success despite the missing audit record.
- **Tests:** `TestPostedTweetExists` (2 cases), `TestRunTwitterStage::test_skips_when_recap_already_posted`, `test_proceeds_when_pre_check_raises`, `test_db_log_error_does_not_crash` rewritten for new semantics. Parallel `TestRunBlueskyStage` cases.
- **Residual risk + follow-up:** The narrow race "post succeeded, insert failed, then operator reruns before inspecting" still re-posts. Closing it requires either (a) full pre-claim flow (insert posted=FALSE row before posting, update after — invasive refactor) or (b) `client_request_id` (not exposed by tweepy/atproto). Also: the audit recommended a UNIQUE DB constraint on `(session_date, tweet_type, platform)`. Skipped this round because the prod `tweets` table holds existing duplicates from past reruns (e.g. 2026-04-24 has 24 twitter + 30 bluesky duplicate `recap` rows) — adding a UNIQUE index would require historical dedup that loses information about real past posts. Track as a P2 follow-up: dedup historical rows via a triage column (e.g. `superseded_at`) and add a partial UNIQUE on `posted=TRUE`.

### ✅ P1.10 — `POST /api/theses/<id>/close` has no auth, listens on 0.0.0.0:3000
- **File:** `v2/dashboard/app.py:124-141`
- **Bug:** Endpoint accepts any caller and mutates DB state. Dashboard listens on `0.0.0.0:3000`.
- **Impact:** If port reachable from outside localhost, anyone can close any active thesis. Severity depends on network exposure — verify Docker port mapping.
- **Fix (2026-05-02):** Two-layer defense.
  1. **Network-level (verified):** `docker-compose.yml` already binds `127.0.0.1:3000:3000` for the prod dashboard and `127.0.0.1:3001:3000` for the paper dashboard — the dashboard is not reachable from outside the host. This is the primary protection.
  2. **App-level (defense-in-depth):** Added `_require_dashboard_origin()` to both `dashboard/app.py` (v1, the active one) and `v2/dashboard/app.py` — the close endpoint now requires `X-Requested-With: dashboard`. Cross-origin browser POSTs that send form-encoded bodies (which would otherwise bypass the JSON-content-type CORS preflight) cannot add this custom header without triggering a preflight, which the endpoint doesn't satisfy. The frontend `dashboard/templates/theses.html` was updated to send the header on the close fetch.
- **Tests:** Existing 8 cases in `TestApiCloseThesis` updated to send the header; 2 new cases (`test_close_thesis_without_csrf_header_returns_403`, `test_close_thesis_with_wrong_csrf_header_returns_403`) verify the gate fires before any DB work.

### ✅ P1.11 — `paper:session` Taskfile target ships `--force`
- **File:** `Taskfile.yml:104`
- **Bug:** `paper:session` target includes `--force`, which bypasses `_check_and_record_session` idempotency. Prod `session` target does not.
- **Impact:** Every paper rerun duplicates all writes (decisions, tweets, snapshots), polluting the paper learning corpus.
- **Reproduction:** Run `task paper:session` twice in one day → second run re-enters all stages.
- **Fix (2026-05-02):** Removed the implicit `--force` from `paper:session` in `Taskfile.yml`. The flag is still available via `{{.CLI_ARGS}}` for intentional reruns (`task paper:session -- --force`). Aligns paper behavior with prod.

### ✅ P1.12 — `--dry-run` only gates the executor; strategist + reflection still write
- **File:** `v2/session.py:268`
- **Bug:** `dry_run` is forwarded only to `run_trading_session`. Strategist still writes theses, playbooks, playbook_actions, thesis_signals, strategy_memos. Reflection still writes rules, memos, identity updates.
- **Impact:** Misleading flag. Users running `--dry-run` to "preview" mutate strategy state.
- **Reproduction:** `python -m v2.session --dry-run` → check `select count(*) from theses where created_at > now() - interval '1 hour'` returns rows.
- **Fix (2026-05-02):** In `run_session`, when `dry_run=True` we now force `skip_ideation=True`, `skip_strategy=True`, `skip_twitter=True`, `skip_bluesky=True`, `skip_dashboard=True`. The executor still runs in dry mode (preview decisions, no order submission); the pipeline still runs (observing news isn't a strategy mutation). Strategist and reflection are skipped entirely — cheaper than the alternative of suppressing writes inside the LLM tool loop, and cleaner semantics.
- **Tests:** New `TestDryRunSkipPromotion` class with 5 cases (`test_dry_run_skips_strategist`, `_skips_reflection`, `_skips_socials_and_dashboard`, `_still_runs_executor_in_dry_mode`, `_still_runs_pipeline`). The 47 existing `dry_run=True` call sites in `test_session.py` were using the flag as a "trading is mocked anyway, just want safety" marker — bulk-replaced with `dry_run=False` since the trading path is patched in those tests.

### ✅ P1.13 — `wait_for_fill` cancel-on-timeout doesn't query for partial fills
- **File:** `v2/executor.py:316-328`; consequence in `v2/trader.py:300-313`
- **Bug:** Timeout path cancels and returns `success=False, filled_qty=None`. Order may have been partially filled before cancel. Trader returns `_DecisionOutcome(False, ...)` and logs decision row with `order_id=None, quantity=requested, price=quote_price`.
- **Impact:** Position sync reconciles shares but the decision-to-fill link is gone — attribution can't see what actually happened.
- **Reproduction:** Order fills 50/100 before 30s timeout; cancel succeeds for the rest; DB has decision with `order_id=None` and the 50 filled shares are detached from any decision row.
- **Fix (2026-05-02):** After the cancel attempt, re-fetch the order via `get_order_by_id` and inspect `filled_qty`. If > 0, return `OrderResult(success=True, filled_qty=<actual>, filled_avg_price=<actual>, order_id=<orig>)` with the timeout flagged in the error string. The trader's downstream path then logs the decision row with the real order_id and partial qty — preserving the attribution link. If `filled_qty == 0` or the refetch itself raises, fall through to the prior `success=False` timeout path. Tests: `TestWaitForFillPartialFillOnTimeout` (3 cases) — partial-fill success, zero-fill failure preserved, refetch-failure graceful fallback.

### ✅ P1.15 — Stage 4 reflection had no memo-written enforcement; silent journal gaps *(found 2026-05-02 second pass)*
- **File:** `v2/session.py:283-306`, `v2/strategy.py:434-490`
- **Bug:** Strategy reflection's system prompt instructs "Always write a memo: Even if nothing changed, document why" — but nothing structurally enforced it. If the LLM finished the loop without calling `tool_write_strategy_memo`, the stage was marked complete via `_complete_stage(session_id, "strategy")` regardless. Worse: on resume, `"strategy" in completed_stages` short-circuits the stage, so the LLM never gets another chance to fill the gap. The "system's memory" silently loses days, undermining the run-to-run continuity claim.
- **Impact:** Journal gaps that compound. Reflection in subsequent sessions reads incomplete history (`get_recent_strategy_memos`), which biases rule proposals away from sessions whose lessons never got captured.
- **Reproduction:** Mock `run_strategy_reflection` to return `StrategyReflectionResult(memo_written=False, ...)`. Session marks stage complete. Next session's resume skips reflection. No memo for that date in `strategy_memos`.
- **Fix (2026-05-02):** Parallel to P2.24 (strategist's playbook validation): `_run_strategy_stage` in `v2/session.py` now raises `RuntimeError("Reflection finished without writing a memo ...")` when `result.strategy_result.memo_written` is False. The exception propagates into the existing handler, sets `result.strategy_error`, calls `_fail_stage`, and skips `_complete_stage`. Next session re-runs reflection. Tests: `TestReflectionMemoGuard` (2 cases — `test_reflection_stage_fails_when_memo_not_written`, `test_reflection_stage_completes_when_memo_written`).

### ✅ P1.14 — Strategist agentic loop max_tokens cap is too tight; no recovery on truncation *(found post-audit during paper session 2026-05-02)*
- **File:** `v2/claude_client.py:198`
- **Bug:** `max_tokens=4096 if turn == 0 else 2048` capped synthesis turns at 2048 output tokens. The ramp was backwards: synthesis (write_playbook + thesis updates + reasoning) happens on later turns, not the first. When the strategist's turn 4 generated >2048 output tokens (a single `write_playbook` call with rich `priority_actions[].reasoning` + `market_outlook` + `risk_notes` is already ~1-2k tokens, plus optional thesis updates and preamble), Anthropic returned `stop_reason="max_tokens"` with a truncated response, the loop bailed, and no playbook was persisted.
- **Impact:** Strategist work silently dropped → no playbook → executor skipped (P2.23 short-circuit), strategist memo not persisted (P2.24 atomic guard). All earlier tool work in that loop (web searches, news fetches, market snapshots) wasted. Surfaced cleanly because of P2.23/P2.24, but until those landed the same condition was leaving stranded memos and confused executor state.
- **Reproduction:** Paper session 2026-05-02 18:26 → turn 4 took 55s (vs ~5-10s for prior turns), hit `Unexpected stop reason: max_tokens` in `logs_paper/v2_claude_client.log`. v1 (`trading/claude_client.py`) uses flat 4096 and never hit this; v2 introduced the lower 2048 ramp.
- **Fix (2026-05-02):** Two-part — (1) `max_tokens=32000` (Opus 4.x model max) across all turns. Anthropic API requires *some* value; capping below the model max is a self-imposed truncation that buys negligible cost/latency safety for an internal post-market pipeline. Output is metered, so the high cap costs nothing unless Claude actually generates more. (2) Defense-in-depth recovery for the still-possible case of a runaway model: when `stop_reason="max_tokens"` hits for the first time in a loop, discard the truncated assistant turn (don't append to `messages`, since incomplete `tool_use` blocks would break the next API call), inject a user message instructing concision (`"Your previous response was truncated... be more concise. If you have not yet called write_playbook, call it now..."`), and continue. Bounded to 1 retry per loop to prevent runaway costs. Second `max_tokens` falls through to the original bail path. Tests: `TestMaxTokensCap` (2 cases — cap ≥ 32000, cap not lower on later turns) + `TestMaxTokensRecovery` (3 cases — recovery continues with concision prompt, second hit bails, prior tool_results preserved).

---

## P2 — Fix this quarter

Slow learning rot; impact measured in months of bad rule proposals.

### ✅ P2.14 — `decision_signals.signal_id` has no FK constraint
- **File:** `db/init/005_redesign.sql:17-22`; write callers `v2/trader.py:391-413`, `v2/database/trading_db.py:466-482`
- **Bug:** `validate_signal_refs` (Python validator at `v2/agent.py:347`) is the only line of defense. No FK, no DB-level check.
- **Impact:** One regression in the validator (or a future caller bypassing it) and the orphan bucket returns. Structural fix that ends a class of bugs.
- **Fix (2026-05-02):** New migration `db/init/018_decision_signals_fk_trigger.sql` adds a `BEFORE INSERT OR UPDATE` trigger that validates the polymorphic `(signal_type, signal_id)` tuple against the appropriate target table (news_signals / macro_signals / theses) and raises on missing/unknown rows. Postgres has no native polymorphic FK, so a trigger is the structural equivalent. Existing 18 historical orphan rows (11 news_signal, 7 thesis) preserved — the trigger only validates new rows; downstream readers already filter orphans via the LEFT JOIN guard from 2026-05-02. Verified live: bogus inserts now raise `decision_signals: news_signal id X does not exist`.

### ✅ P2.15 — `_classify_batch` zips Haiku output positionally + pads with `noise`
- **File:** `v2/classifier.py:284-295`
- **Bug:** Loop pads with `noise` when Haiku returns fewer entries than headlines. Pairs `parsed[i]` with `headlines[i]`/`published_ats[i]` purely positionally — assumes Haiku preserves order.
- **Impact:** If Haiku reorders, ticker/sentiment lands on the wrong headline. If Haiku returns short, headlines silently become `noise`. Direct upstream of attribution corruption.
- **Fix (2026-05-02):** Switched the contract to **index-based mapping**. Updated `BATCH_CLASSIFICATION_SYSTEM` to require an `"index"` field on each entry (1-based, matches the headline number). `_classify_batch` builds a `by_idx[i] → entry` map from the response, then iterates headlines in order — entries without a valid index are dropped and the corresponding headline becomes explicit noise rather than risking misalignment. Tests: `test_index_based_mapping_handles_reordering` (reverses Haiku's array, asserts AAPL/MSFT/GOOG land on the right headlines) + `test_entry_without_index_skipped` (malformed entry → noise, not mis-attribution) added to `TestClassifyBatch`.

### ✅ P2.16 — Alpaca news `id` captured then discarded
- **File:** `v2/news.py:59-68`, `v2/pipeline.py:50-60`
- **Bug:** `NewsItem.id` is captured from Alpaca but never propagated to the row. `news_signals` has no column for it.
- **Impact:** Dedup falls back to `(ticker, md5(headline), published_at)` from `db/init/014_news_signal_dedup.sql`. 1-second republish jitter creates duplicates; identical content with different timestamps slips through.
- **Fix (2026-05-02):** Migration `019_news_signal_alpaca_id.sql` adds nullable `alpaca_id TEXT` to `news_signals` and `macro_signals` with partial UNIQUE indexes (preserves historical rows without the field). Plumbed `alpaca_id` through `TickerSignal`/`MacroSignal` dataclasses, `_build_classification_result`, `_classify_batch`, `classify_news_batch`, `classify_news`, and `pipeline.py`. `insert_news_signals_batch` / `insert_macro_signals_batch` accept the new tuple shape (with backward-compat padding for legacy callers). The existing dedup index from migration 014 keeps working; alpaca_id is an additional canonical key.

### ✅ P2.17 — `signal_attribution.sample_size` mismatched against 30d metrics
- **File:** `v2/attribution.py:60-67`
- **Bug:** `WHERE outcome_7d IS NOT NULL AND alpha_7d IS NOT NULL` admits decisions that don't yet have 30d outcomes. `COUNT(DISTINCT decision_id)` includes those, but `AVG(alpha_30d)` and `AVG(CASE WHEN alpha_30d > 0 …)` silently drop them via NULL averaging.
- **Impact:** Reported `n` overstates support behind 30d numbers → strategist over-confident in weakly-supported rules.
- **Reproduction:** Backfill 7d but not 30d → row reports `n=30, avg 30d alpha=+0.5%` even though only 5 decisions actually had a 30d outcome.
- **Fix (2026-05-02):** Migration `020_signal_attribution_30d_count.sql` adds `sample_size_30d INT DEFAULT 0`. The aggregation in `compute_signal_attribution` now emits `COUNT(DISTINCT CASE WHEN alpha_30d IS NOT NULL THEN decision_id END) AS sample_size_30d` alongside the 7d-eligible `sample_size`. `upsert_signal_attribution` accepts the new column. `sample_size` semantics unchanged for backward compat (7d-eligible cohort); 30d consumers should read `sample_size_30d`.

### ✅ P2.18 — `thesis` bucket has no orphan-FK guard
- **File:** `v2/attribution.py:32-56`, `v2/patterns.py:55-76`
- **Bug:** Orphan-FK guard added 2026-05-02 covers `news_signal` and `macro_signal` (`(ds.signal_type != 'news_signal' OR ns.id IS NOT NULL)`), but no equivalent guard exists on `thesis`. Phantom thesis IDs still inflate the `thesis` aggregate.
- **Impact:** Same shape as the fixed `news_signal:unknown` artifact, just for thesis IDs.
- **Verified scope (2026-05-02):** 6 historical orphan rows in `decision_signals` (`signal_type='thesis', signal_id=0`), all from 2026-02-11–12 (pre-validator era). 5 are HOLDs (already excluded by `action IN ('buy','sell')`); 1 is a real buy (GLD 2026-02-12).
- **Fix (2026-05-02):** Added `LEFT JOIN theses t ON ds.signal_type='thesis' AND t.id=ds.signal_id` and `(ds.signal_type != 'thesis' OR t.id IS NOT NULL)` to both `attribution.py` and `patterns.py`. Tests: `test_excludes_thesis_orphans` in `TestOrphanSignalFiltering` (attribution) and `TestAnalyzeSignalCategories` (patterns).
- **Recompute landed:** Re-ran `compute_signal_attribution` against prod. Thesis bucket shifted from `n=30, alpha=-0.014%, win_rate=50.0%` to `n=29, alpha=-0.066%, win_rate=48.3%`. Small but truthful.

### ✅ P2.19 — `patterns.py` raw outcomes vs `attribution.py` alpha
- **File:** `v2/patterns.py:54-92, 94-124` vs `v2/attribution.py:43-64`
- **Bug:** `analyze_signal_categories` and `analyze_sentiment_performance` use `AVG(d.outcome_7d)` (raw outcome). `compute_signal_attribution` uses alpha vs SPY. Same category, different number, depending on entry point.
- **Impact:** Strategist reads contradictory "STRONG/WEAK" (constraint block) vs "best/worst" (pattern report) labels for the same category. During a bull run, raw outcomes look great while alpha looks poor.
- **Fix (2026-05-02):** Switched `analyze_signal_categories`, `analyze_sentiment_performance`, and `analyze_confidence_correlation` in `v2/patterns.py` to compute alpha (`d.outcome_7d - d.benchmark_7d`) just like `attribution.py`. WHERE clauses now also require `benchmark_7d IS NOT NULL`. Display labels in `generate_pattern_report` updated to "Signal Category Performance (alpha vs SPY)" / "avg alpha (beat-market rate: …)" so the metric is unambiguous.

### ✅ P2.20 — No retry/backoff on Anthropic in classifier; batch failure → 50× fan-out
- **File:** `v2/classifier.py:198-216, 244-256, 268-274, 315-326`
- **Bug:** All `client.messages.create` calls wrapped in bare `except Exception` returning `noise`. No 429/529/transient-error handling. Batch JSON failure falls back to N individual calls (up to `batch_size=50`) with no rate-limit handling.
- **Impact:** 50× API request spike on a single batch failure. If original failure was 429, fallback hammers within the same minute.
- **Fix (2026-05-02):** All three `messages.create` call sites in `v2/classifier.py` (`classify_news`, `_classify_batch`, `classify_ticker_news`) now go through `claude_client._call_with_retry`, picking up the existing 3-retry / exponential backoff / rate-limit-aware delays. Additionally, the per-headline fallback in `classify_news_batch` now distinguishes `RateLimitError` from other batch failures: a rate-limited batch (already through 3 retries) is marked as all-noise instead of fanning out N more calls in the same minute.

### ✅ P2.21 — `insert_news_signals_batch` returns `len(signals)` regardless of `ON CONFLICT DO NOTHING` skips
- **File:** `v2/database/trading_db.py:24-33`; consumed at `v2/pipeline.py:64`
- **Bug:** `ON CONFLICT DO NOTHING` may skip rows but function returns `len(signals)` rather than `cur.rowcount`. Stats lie about what was persisted.
- **Impact:** Stage telemetry untrustworthy. Re-runs report identical non-zero `ticker_signals_stored` even when zero rows inserted.
- **Fix (2026-05-02):** Both `insert_news_signals_batch` and `insert_macro_signals_batch` now return `cur.rowcount`. Tests updated to set `mock_cursor.rowcount` explicitly; new test `test_insert_news_signals_batch_skipped_rows_reflected` verifies that all-conflicting input returns 0.

### ✅ P2.22 — `tool_write_playbook` is non-atomic across upsert + delete + N inserts
- **File:** `v2/tools.py:374-403`, `v2/database/trading_db.py:420-429`
- **Bug:** `upsert_playbook` → `delete_playbook_actions` → loop of `insert_playbook_action`, each in its own `get_cursor()` (own connection, own transaction). Mid-loop failure → playbook row exists, old actions gone, only some new actions inserted.
- **Impact:** Executor trades on incomplete intent.
- **Reproduction:** Force `insert_playbook_action` to raise on the third action → DB left with playbook + 2 actions, no rollback.
- **Fix (2026-05-02):** New `replace_playbook_actions_atomic` helper in `v2/database/trading_db.py` performs the upsert + decision-clear + delete + N inserts inside a single `get_cursor()` (one connection, one transaction). `tool_write_playbook` now calls this helper instead of the three separate operations. Mid-loop failure rolls everything back; the playbook row + actions are all-or-nothing.

### ✅ P2.23 — Strategist resume short-circuits without verifying playbook row exists
- **File:** `v2/session.py:209-211, 247-256`; consequence in `v2/context.py:424`
- **Bug:** Resume logic: `"strategist" in completed_stages` → skip. Executor short-circuit at `:247` only fires on `result.strategist_error`. On resume after manual playbook cleanup, `result.strategist_error` is `None` (never ran this invocation), so executor runs `get_pending_playbook_actions(playbook["id"])` against a `None` playbook → `TypeError`.
- **Impact:** Crashes recovery flow on certain partial-failure replays.
- **Fix (2026-05-02):** `_run_executor_stage` short-circuit now keys on the playbook itself, not on the strategist error. The condition is `get_playbook(session_date) is None` regardless of `strategist_error` state, with distinct log messages for each cause. New test `test_executor_skipped_on_resume_when_playbook_missing` verifies the resume + manual-cleanup scenario.

### ✅ P2.24 — `_persist_strategist_memo` runs before `write_playbook` validation
- **File:** `v2/session.py:215-230`
- **Bug:** Memo is committed in its own transaction before `get_playbook(session_date) is None` check raises `RuntimeError`. Failure path doesn't call `complete_session_stage`, so next run re-runs the strategist and inserts a *second* memo for the same date.
- **Impact:** Duplicate `strategist_notes` in `strategy_memos` for the same `session_date`. Reflection sees both, double-counting the strategist's voice.
- **Fix (2026-05-02):** Reordered `_run_strategist_stage` to validate the playbook before persisting the memo. Validation failure now raises before any DB write, so the memo isn't committed; rerunning the strategist won't insert a second copy. New test `test_strategist_memo_not_written_when_playbook_missing` asserts the memo write is skipped when the playbook validation fails post-loop.

### ✅ P2.26 — `insert_decision_signals_batch` / `insert_thesis_signals` return `len()` instead of `cur.rowcount` *(found 2026-05-02 second pass)*
- **File:** `v2/database/trading_db.py:512-526` (`insert_thesis_signals`), `v2/database/trading_db.py:552-560` (`insert_decision_signals_batch`)
- **Bug:** Same shape as P2.21 (which fixed the news/macro batches). Both functions execute `INSERT ... ON CONFLICT DO NOTHING` then return `len(rows)` / `len(signals)`. On a rerun where most of the rows already exist, the function reports the input count instead of what was actually inserted.
- **Impact:** Two specific lies:
  1. Strategist sees `_persist_signal_refs` (in `v2/tools.py`) format the message `f" Cited {len(valid)} signal(s)."` — overcounts when refs were already cited.
  2. `decision_signals` insertion telemetry from `trader.py:391-413` reports the requested count even when ON CONFLICT skipped real conflicts (e.g., a duplicate-decision rerun).
- **Fix (2026-05-02):** Both functions now return `cur.rowcount if cur.rowcount is not None else 0`, matching the P2.21 pattern. Tests: `test_insert_thesis_signals_skipped_rows_reflected`, `test_insert_decision_signals_batch_skipped_rows_reflected` — verify rerun on already-conflicting rows reports 0.

### ✅ P2.25 — `tool_get_session_summary` reads 30d decisions but joins signals only for latest 10
- **File:** `v2/strategy.py:213`
- **Bug:** `decisions` is 30-day list; signal linkage fetched only for first 10. Prompt header says "Decisions ({len(decisions)})" using the full count.
- **Impact:** Reflection LLM infers signal patterns from a sliced sample → biased rule proposals.
- **Fix (2026-05-02):** Made the slice explicit. Header now reads `Decisions (latest 10 of N in last Xd):` when truncation applies, so the reflection LLM knows it's reading a recency-biased sample. The display loop and signal-fetch both operate on the same `shown = decisions[:display_limit]` slice, so the count and the rendered rows match.

---

## P3 — Code quality / latent

Clean up when adjacent code is touched.

### ✅ P3.26 — Daily P&L tweet doesn't subtract deposits; dashboard does
- **File:** `v2/twitter.py:106-111` vs `v2/dashboard_publish.py:311-314`
- **Bug:** Tweet path computed `day_pnl = portfolio - prev` with no deposit adjustment; dashboard path subtracted `daily_deposit`. Public numbers disagreed by the size of any cash transfer in the gap day.
- **Fix (2026-05-02):** New `get_daily_deposit(prev_date, today_date)` helper in `v2/executor.py` mirrors the dashboard's `cum[today] − cum[prev]` semantics by querying Alpaca CSD/CSW activities. `_section_book_status` in `v2/twitter.py` subtracts the result from the raw delta and uses `prev + daily_deposit` as the percentage base. On Alpaca lookup failure, falls back to the raw delta so the tweet stage doesn't crash. Tests: `test_day_pnl_subtracts_deposits`, `test_day_pnl_falls_back_when_daily_deposit_fails` in `TestGatherTweetContext`.

### ✅ P3.27 — `INTERVAL '%s days'` parameterization fragile across psycopg drivers
- **File:** `v2/database/dashboard_db.py:24,34,49,61,75,85,95`, `v2/database/trading_db.py:52,58,103,109,138,187`, `v2/patterns.py:77,117,151,185`
- **Bug:** `INTERVAL '%s days'` relied on psycopg2 substituting `%s` *inside* a quoted SQL literal. Works on psycopg2; would break on psycopg3.
- **Fix (2026-05-02):** Replaced all 17 occurrences with `INTERVAL '1 day' * %s` (parameter substituted as a normal value, then multiplied by a true INTERVAL). Smoke-tested against Postgres 16 + psycopg2.

### ✅ P3.28 — `_enrich_snapshots_with_deposits` fallback double-credits early deposits
- **File:** `v2/dashboard_publish.py:99-126`
- **Bug:** First loop credits any deposit dated strictly before a snapshot. Fallback for the "snapshot[0] still 0" case wrote `credit` to *every* snapshot, including ones the first loop already credited — inflating cum_deposits by exactly one extra round for snapshots[1+].
- **Fix (2026-05-02):** Fallback now applies the credit only to snapshots whose date is `<= first_snap_date` (the snapshots the strict-< first loop missed). Tests: `TestEnrichSnapshotsWithDeposits` (3 cases).

### ✅ P3.29 — `extract_final_text` returns `None` if final assistant message is tool_use-only
- **File:** `v2/claude_client.py:308-345`
- **Bug:** `hasattr(block, "text")` was a loose check; the bigger issue was that when *every* assistant message in the loop was tool_use-only, the function returned `None` and the caller's `or "No summary available"` placeholder was stamped as the strategist memo, poisoning reflection's input.
- **Fix (2026-05-02):** Switched to `block.type == "text"` (defensive against future SDK shape changes); concatenated all text blocks in the most-recent assistant message that has any (some responses split synthesis across blocks); fall back to a tool-name summary (`"[no narrative summary; final tool calls: write_playbook]"`) when no assistant message has text. Tests: `TestExtractFinalText` (5 cases).

### ✅ P3.30 — Sector concentration is advisory-only — no hard gate at order-submit
- **File:** `v2/trader.py:495-516`, `v2/risk.py:60-88`
- **Bug:** Sector concentration was injected as a *string warning* into `risk_notes`. Nothing structurally prevented the executor LLM from emitting a buy that ran the book over the cap.
- **Fix (2026-05-02):** New `check_sector_cap_for_buy(ticker, new_qty, price, position_values, portfolio_value, cap)` helper in `v2/risk.py` returns a breach message if a buy would push the ticker's sector over `MAX_SECTOR_PCT`. Wired into `_prepare_decision` after qty resolution: on breach, the decision is marked `invalid` with a `[REJECTED: sector ...]` reason and `trades_failed += 1`. Sells aren't gated (they reduce sector exposure). `_execute_decisions` pre-computes `position_values` once per session and threads it through. Tests: `TestCheckSectorCapForBuy` (5 cases) + `TestSectorCapHardGate` (3 cases — buy rejected, buy under cap allowed, sell not blocked).

### ✅ P3.31 — `_call_with_retry` doesn't catch `BadRequestError`/context-length-exceeded
- **File:** `v2/claude_client.py:115-130, 199-237`
- **Bug:** Only `RateLimitError`, `InternalServerError`, `APIConnectionError` were retryable. A 400 for "prompt is too long" propagated up and failed the strategist stage hard — no graceful degradation despite the loop already having a `_truncate_old_tool_results` pruning helper.
- **Fix (2026-05-02):** Added `_aggressive_prune` (drops everything except the initial user prompt + the most recent exchange) and `_looks_like_context_length_error` (matches "too long" / "context length" / "context_length" in the error string). On the first context-length `BadRequestError` per loop, prune aggressively and continue (retry the same turn). Bounded to 1 recovery to prevent runaway costs; second hit propagates. Unrelated `BadRequestError`s still propagate immediately so we don't burn tokens retrying a known-broken request. Tests: `TestContextLengthRecovery` (3 cases).

### ✅ P3.32 — `get_attribution_summary` recomputed 3+ times per session, no memoization
- **File:** `v2/attribution.py:10-26, 108-129`
- **Bug:** Every call (strategist tool loop + `build_trading_context` + `build_executor_input`) re-read the table and re-formatted the Decimal arithmetic.
- **Fix (2026-05-02):** Module-level `_attribution_summary_cache` populated on first read; `compute_signal_attribution` calls `clear_attribution_summary_cache()` after upsert so the next reader sees fresh numbers. Tests: `test_memoizes_within_process`, `test_recompute_invalidates_memo`. Test isolation: autouse fixture in `tests/v2/conftest.py` clears the cache between tests.

### ✅ P3.33 — `_count_actions` substring match treats soft-guard rejections as successful identity updates
- **File:** `v2/strategy.py:356-389`
- **Bug:** Heuristic `"identity updated" in result_text.lower()` was brittle — the current soft-guard string ("Identity *was* updated...") doesn't actually contain "identity updated" as a substring (the audit's literal claim was wrong) but any future rephrasing without the "was" gap would silently flip `identity_updated=True` on a rejected update.
- **Fix (2026-05-02):** Tightened the match to the unique success-message prefix `"Strategy identity updated to version"`. Test: `test_soft_guard_warning_does_not_count_as_identity_update`, exercising both the actual current warning and a hypothetical rephrasing.

### ✅ P3.34 — `MAX_POSITION_PCT` defined in two places
- **File:** `v2/intents.py:15`, `v2/risk.py:24`
- **Bug:** Constant duplicated; the comment "mirror v2/agent.py" pointed at a sibling that had since been removed (the constant only appears in `agent.py` as a string token inside a prompt). v1's `trading/agent.py` still has its own copy, but v1 is sunset.
- **Fix (2026-05-02):** Moved `MAX_POSITION_PCT = Decimal("0.10")` to `v2/risk.py` (already the home for `MAX_SECTOR_PCT`); `v2/intents.py` now imports it. v1 left untouched.

### ✅ P3.35 — `setup_logging` no-ops if root has handlers
- **File:** `v2/log_config.py:13-58`
- **Bug:** `if root.handlers: return` — any third-party library that registered a root handler before `setup_logging` would cause us to skip console + file handler setup AND `os.makedirs(log_dir)`. Logs would vanish silently.
- **Fix (2026-05-02):** Tag our handlers with a sentinel attribute (`_algo_owned = True`) and only short-circuit when *our* tag is present. Per-file handler installation is now also idempotent (skip the named logger if it already has an algo handler). `os.makedirs` runs unconditionally. Tests: 3 cases in new `tests/v2/test_log_config.py`.

### ✅ P3.36 — `_session_retirements` module-level global
- **File:** `v2/strategy.py:28-49, 184-191`
- **Bug:** Module-level list shared across all imports. If paper + prod ever ran in one Python process, retirements would clobber the same counter and the `MAX_RETIREMENTS_PER_SESSION` cap would behave unpredictably.
- **Fix (2026-05-02):** Replaced the list with a `ContextVar`. `reset_session()` calls `.set([])`; the read path uses a small `_get_session_retirements()` helper that handles the per-context first-access. ContextVar gives each thread / asyncio task / explicit `Context.run` its own copy. Test: `test_session_retirements_isolated_across_threads` (two threads run the cap concurrently; each retires 2 rules and hits its own cap on the 3rd, no leakage).

### ✅ P3.37 — v2 Flask dashboard references `templates/` that doesn't exist
- **File:** `v2/dashboard/app.py`
- **Bug:** `v2/dashboard/templates/` directory exists but is empty; HTML routes (`/`, `/playbook`, `/attribution`, `/signals`, `/theses`, `/decisions`, `/performance`) call `render_template` and would 500 with `TemplateNotFound`. JSON-only routes work today.
- **Fix (2026-05-02):** Documented current state in the module docstring — the v2 dashboard is intended for the eventual cutover from v1; HTML routes need template migration from `dashboard/templates/`. The live operator dashboard remains v1 (`dashboard:3000`); no functional change.

### ✅ P3.38 — `success_rate=None` placeholder in `get_thesis_stats`
- **File:** `v2/database/dashboard_db.py:141-163`, `dashboard/queries.py:319-335`
- **Bug:** Both v1 and v2 returned `'success_rate': None`. Templates rendered as "N/A".
- **Fix (2026-05-02):** Compute execution conversion rate as `executed / (executed + invalidated + expired) * 100`. None when no theses are closed yet (avoids divide-by-zero / artificial 0). The audit's note about a more rigorous "success" (executed + outperformed-SPY via decision_signals join) is preserved as a follow-up; this cheap count is the baseline. Tests: 2 cases in `TestDashboardQueries`.

### ✅ P3.39 — `subprocess.run(["wrangler", ...])` has no timeout
- **File:** `v2/dashboard_publish.py:447-470`
- **Bug:** No `timeout=` kwarg. A hung `wrangler` would block the session indefinitely.
- **Fix (2026-05-02):** `subprocess.run` now passes `timeout=300` (5 min). `subprocess.TimeoutExpired` is caught and converted to `RuntimeError("Wrangler deploy timed out after Xs — bailing rather than blocking the session")` so the operator log clearly identifies the cause. Tests: `test_passes_timeout_to_subprocess`, `test_raises_on_subprocess_timeout`.

### ✅ P3.40 — Haiku ticker extraction has no allowlist
- **File:** `v2/classifier.py:35-83, 184-186`
- **Bug:** `tickers = entry.get("tickers", [])` accepted any string Haiku emitted with only `.upper()` sanitization. `$TSLA`-prefixed strings, group acronyms (FAANG, MAANG, MAGS), and generic acronyms (CEO, GDP, ETF) all became rows in `news_signals` and polluted attribution joins.
- **Fix (2026-05-02):** New `_validate_ticker(raw)` returns the cleaned ticker (`$` stripped, uppercased) or `None`. Filter chain: regex `^[A-Z]{1,5}(\.[A-Z])?$` (catches BRK.B, rejects long alphanumeric junk) + a `_TICKER_BLOCKLIST` of group/agency/economic-indicator acronyms that are never tradable. Did *not* block 2-letter common-word lookalikes (BE, ON, SO, GO, AI) since those *are* real tickers — without an Alpaca-asset allowlist we can't disambiguate. Tests: `TestTickerValidation` (6 cases).

---

### ✅ P3.41 — Zero-alpha categories silently dropped from `get_attribution_summary` *(found 2026-05-02 second pass)*
- **File:** `v2/attribution.py:138-139`
- **Bug:** `outperforming = [r for r in sufficient if r.get("avg_outcome_7d") and float(...) > 0]`. The truthy check `r.get("avg_outcome_7d")` returns `Decimal(0)` (falsy) when alpha is exactly zero, short-circuiting the AND. The row is then absent from BOTH outperforming and underperforming buckets — invisible to the strategist. The `<= 0` bound on underperforming was clearly meant to include zero (mutually exclusive + exhaustive with `> 0`), but the truthy guard prevented it.
- **Impact:** Niche but real — a category that breaks even alpha-wise (n=10+, alpha = 0.0) is *invisible* to the strategist, even though "neutral signal" is real signal.
- **Reproduction:** Pass a row with `avg_outcome_7d = Decimal("0")` and sufficient sample size — neither bucket renders it.
- **Fix (2026-05-02):** Replace truthy check with explicit `is not None` so `Decimal(0)` lands in underperforming where it belongs. None alpha (no data yet) still excluded from both, since None means "no signal" not "neutral signal". Tests: `test_zero_alpha_appears_in_underperforming`, `test_none_alpha_still_excluded` in `TestGetAttributionSummary`.

### ✅ P3.42 — `update_strategy_identity` warning instructs LLM to retry, but guard fires unconditionally *(found 2026-05-02 second pass)*
- **File:** `v2/strategy.py:117-148`
- **Bug:** Identity-update guard returned text ending in `"To proceed anyway, call update_strategy_identity again."`. But the guard is hard, not soft — a retry hits the same `(date.today() - current["created_at"].date()).days < 3` check (and immediately after a successful write the *new* state is 0 days old, so the next call is also rejected). The instruction misled the LLM into wasted turns.
- **Impact:** Cosmetic but burns LLM budget. Worse: if the LLM trusts the instruction it may write the same identity twice in a session, then conclude "the system rejected my update" and write a memo claiming the system ignored a fundamental shift — when it really just throttled the second call.
- **Fix (2026-05-02):** Rewrite the warning to clearly state the rejection is hard ("this update has been rejected" / "Identity updates are gated to one per 3 days") and steer the LLM to write a memo + retry next session. Tests: `test_warning_does_not_promise_retry_will_succeed` in `TestIdentityUpdateGuard` — asserts neither "call update_strategy_identity again" nor "call again" appear in the warning text. The existing hard-gate behavior tests (`test_warns_if_recently_updated`, `test_allows_update_if_not_recent`) still pass unchanged.

## Cross-cutting observations

*(Three of the original six observations have been closed by the audit fixes; struck-through bullets are kept for historical context. The remaining three still apply.)*

- ~~**Biggest risk to attribution data quality:** P0.1 + P0.3 + P2.14 + P2.15.~~ Closed: P0.1 fixed sign inversion, P0.3 added category enum validation at the LLM boundary, P2.14 added the polymorphic FK trigger, P2.15 switched to index-based mapping. Orphan-FK problem is now structurally closed.
- ~~**Biggest risk to public reputation:** P1.8 + P1.9 + P1.10.~~ Closed: P1.8 redacts UUIDs, P1.10 added CSRF gate + verified loopback bind. P1.9 partial — see [Residual follow-ups](#residual-follow-ups).
- ~~**Biggest cost/reliability risk:** P2.20.~~ Closed: classifier now uses `_call_with_retry` and rate-limited batches no longer fan out 50×.
- **Paper/prod isolation depends on `.env.paper` *not* setting Twitter/Bluesky/Cloudflare vars** — if those leak via shell or `--env-file` override, paper data publishes as prod. *Still open — no automated guard.*
- **Stage 4 reflection uses 30-day window; Stage 0 attribution uses 90-day window.** Rules "proven" by attribution may be invisible in reflection summary, and vice versa. *Still open — design tension, not a defect.*
- **`_run_learning_refresh` runs before pipeline stage but after executor stage of the *prior* session** — attribution constraints fed to strategist always lag by exactly one session. *Still open — worth confirming with the project owner whether the lag is intentional.*
- **FIXED 2026-05-05** — **`_print_cost_summary` in `v2/ideation_claude.py` hardcodes Opus 4.6 pricing** — `input_cost = uncached_input * 5 / 1M`, etc. Correct for the default `claude-opus-4-6` strategist model, but if the model is overridden to Haiku ($0.80/M input, $4/M output) or Sonnet ($3/M input, $15/M output), the logged cost is misleading. Internal log only — not visible to dashboards. *Found 2026-05-02 second pass; not fixed because cost log is operator-facing only.* Resolution: `_print_cost_summary` now takes `model` and delegates to `v2/pricing.py::stage_cost_usd`, which reads rates from the `model_pricing` DB table seeded by migration 024. Same formula as the new `session_stage_costs` SQL view.
- **`_check_and_record_session` always inserts a new `sessions` row, even on resume** — if a prior run failed mid-pipeline, its row stays in `status='running'` forever. The new run's row tracks the resumed stages. Over time, `running` rows accumulate as orphans (only `completed` short-circuits at the top). Bloat, not corruption — readers select by `id DESC LIMIT 1`. *Found 2026-05-02 second pass; not fixed in this round.*
- **`v2/entertainment.py` lacks the false-success guard P1.9 added to recap tweets** — if `post_tweet` succeeds but `insert_tweet` raises, `result.posted` stays `True`. Less critical than the recap path because entertainment tweets aren't run as part of the daily session and there's no `posted_tweet_exists`-style rerun guard yet. *Found 2026-05-02 second pass; tracked for the next pass on entertainment if the operator starts using it routinely.*

## Residual follow-ups

Three items were called out as future work inside individual fix descriptions but never promoted to standalone tracked entries. Captured here so they don't get lost.

1. **P0.3 follow-up — DB CHECK constraint for category enums.** P0.3 added Python-side validation in `classifier.py` for `news_signals.category` and `macro_signals.category`. A migration adding `CHECK (category IN (...))` would prevent regression if a future caller bypasses the classifier. Cheap; defer until the next time `classifier.py` is touched.
2. **P1.9 follow-up — historical tweet dedup + partial UNIQUE.** App-level guards landed (pre-stage rerun check + no false-success on DB-write failure), but the audited fix dropped the `UNIQUE (session_date, tweet_type, platform)` index because prod `tweets` has historical duplicates from past reruns (e.g. 2026-04-24: 24 twitter + 30 bluesky duplicate `recap` rows). Path forward: triage column (`superseded_at`) → mark all-but-latest duplicates → add partial UNIQUE on `posted=TRUE AND superseded_at IS NULL`. Closes the residual race ("post succeeded, insert failed, operator reruns before inspecting").
3. **P3.38 follow-up — rigorous thesis success metric.** P3.38 ships an execution-conversion rate (`executed / (executed + invalidated + expired)`). The richer metric the audit suggested — executed-AND-outperformed-SPY via `decision_signals` join — is preserved as a follow-up. Worth doing once attribution windows have enough thesis-tagged decisions to be meaningful (P2.18 cleanup left only ~29 thesis-tagged samples).

## Verification suggestion

Several P2/P3 items were flagged based on code reading, not contract verification. Still worth running through context7 against current SDK docs when next touching these areas:

- **alpaca-py:** `account.cash` nullability (`v2/executor.py:60-71`), `pos.qty` typing in `sync_positions_from_alpaca`, `client.get(...)` private-method use in `get_net_deposits`, `wait_for_fill` status enum coverage (`"canceled"` vs `"cancelled"` vs `"pending_cancel"` vs `"done_for_day"`).
- **anthropic SDK:** which exceptions belong in `RETRYABLE_ERRORS` (`v2/claude_client.py:21-25`); tool_result content-shape contract (`str` vs list-of-blocks for `content`).
- **psycopg2 vs psycopg3:** literal-substitution-inside-quotes for `INTERVAL '%s days'` pattern (P3.27 already migrated to `INTERVAL '1 day' * %s`, but spot-check that no caller reintroduced the old form).
