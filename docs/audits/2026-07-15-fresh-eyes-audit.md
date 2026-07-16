# Pinchy Fresh-Eyes Audit — 2026-07-15

A "new observer" audit run deliberately outside the existing audit machinery
(`docs/audit-playbook.md` deterministic checks + supervisor stage). Four
independent lenses, each run as a separate reviewer with no shared state:

1. **Code correctness** — deep review of the v2 money path
2. **Economics** — read-only SQL against the prod DB (run twice independently;
   the two passes agreed, and the second corrected the first's cost figures
   using accurate model pricing)
3. **Ops resilience** — backups, cron, alerting, kill switches, host risk
4. **Agent/prompt architecture** — the learning-loop design layer

A fifth lens (security & exposure) was started and deliberately cancelled;
it is **not covered** here.

Known/tracked items were excluded by instruction: orphan FKs, backfill gap
counts, rule #27 oscillation as such, the news_signal:unknown artifact, and
the generic init/migrations drift pattern (though a **new concrete instance**
of that drift was found — see 3.3).

---

## Executive verdict

The system didn't crash — it **converged, by design, to doing nothing, and
then went dark**:

1. **The architecture is a ratchet that only tightens.** Only executed trades
   generate outcomes, so inaction is invisible to the learner. The executor
   prompt demands "fresh evidence" that Haiku is structurally never given and
   prescribes HOLD as the fallback. Tiny attribution cohorts (n=3–5) are
   promoted into hard "do not trade this category" constraints that suppress
   the very trading that would grow the cohorts. Result: **zero trades since
   2026-05-13**; June was 12 sessions of pure holds (~$19 API spend, no
   information gained).
2. **Then it went dark.** The 2026-06-15 session exited 1, and the installed
   crontab now has the prod session line **commented out** — an uncommitted,
   undated edit that is the de facto kill switch. No dead-man's switch
   exists, `ALGO_ALERT_WEBHOOK_URL` is set nowhere, so a month of silence
   went unnoticed. *(Post-audit: the owner confirmed the crontab edit was a
   deliberate hiatus. The alerting/dead-man's-switch gaps stand — an
   unintentional stall would have looked identical.)*
3. **The business case fails on arithmetic regardless.** Lifetime gross
   trading P&L ≈ **−$35** (the +691% equity curve is ~87% deposits; true TWR
   +0.75% over 4.3 months). Estimated lifetime API spend: **$150–250+** —
   4–7× the absolute value of everything it ever made or lost trading.
   Per-trade inference (~$7.50) is ~9% of the $80 average trade notional.
   Detecting even a 0.5%/trade edge needs ~1,360 trades; there are 99, and
   accrual has stopped. As sized and paced, the experiment cannot answer its
   own question.

Suggested triage order: **0.1 → 3.1 → 3.3** are same-day items (a decision,
a `pg_dump` cron line, one `task db:migrate`). Tier 1 is a prerequisite for
re-enabling the cron. Tier 2 is the interesting engineering — but 0.2 comes
first, because Tier 2 effort is wasted if the economics stay structurally
negative.

---

## Ranked findings

### Tier 0 — Decisions, not code

| # | Sev | Finding | Lens |
|---|-----|---------|------|
| 0.1 | 🔴 | Installed crontab has the prod session line commented out (uncommitted, undated). **Resolved 2026-07-15: the owner confirmed this is a deliberate hiatus.** Remaining action: make the hiatus visible — commit the crontab state (or document the hiatus + resume procedure) and add a proper halt mechanism (C.5) so future halts aren't ambiguous | ops |
| 0.2 | 🔴 | Structural economics: scale deployed capital ~20–50×, cut strategist cost ~10× (Opus is 94–95% of tracked spend), or reframe as research with an explicit cost budget | econ |
| 0.3 | 🟠 | 3 open positions (~$1,174 cost basis) have been unmonitored by the learning loop for a month. **Resolved 2026-07-15: owner fully liquidated the prod account (positions closed, cash withdrawn) — no live exposure remains.** The prod DB's `positions` table and final snapshots no longer reflect reality | econ |

### Tier 1 — Money-path safety (fix before re-enabling trading)

| # | Sev | Finding | Lens |
|---|-----|---------|------|
| 1.1 | 🟠 | Sell precheck passes when Alpaca reports *no position at all* — stale-state sell can reach the broker, possibly opening a short (Appendix A.1) | code |
| 1.2 | 🟠 | Intra-batch duplicate (ticker, action) decisions all execute; second fill gets no decision row (A.2) | code |
| 1.3 | 🟠 | Broker-rejected/timed-out orders logged as ordinary buy/sell rows — phantom trades in learning data + same-day retry lockout (A.3) | code |
| 1.4 | 🟡 | LLM-authored `thesis_id`/`playbook_action_id` used unvalidated in DB writes; executor can close/invalidate theses it never sees (A.6, D.3) | code+arch |
| 1.5 | 🟡 | Daily-loss breaker's per-fill re-check silently skipped when the post-fill account refresh fails (C.5) | ops |
| 1.6 | 🟡 | `--force` retry deletes already-executed playbook actions and severs decision linkage (A.5) | code |

### Tier 2 — Restore the learning gradient

| # | Sev | Finding | Lens |
|---|-----|---------|------|
| 2.1 | 🔴 | Inaction produces zero learning signal; all-hold is a locally stable fixed point (D.1) | arch |
| 2.2 | 🔴 | Executor "fresh evidence" veto is structurally unsatisfiable; HOLD is the only compliant output (D.2) | arch |
| 2.3 | 🟠 | n=3–5 attribution cohorts become hard prompt constraints and "evidence-based" rules (D.4) | arch+econ |
| 2.4 | 🟠 | Reward signal mislabeled: "P&L" in every prompt is actually 7d price-move; no realized/unrealized P&L anywhere in context (D.5) | arch |
| 2.5 | 🟡 | All 7d/30d outcomes computed from the same-day *in-progress* 3 PM bar, then frozen (A.4) | code |
| 2.6 | 🟡 | Equity reporting conflates deposits with performance; no capital-flows table (B.6) | econ |

### Tier 3 — Ops resilience

| # | Sev | Finding | Lens |
|---|-----|---------|------|
| 3.1 | 🔴 | Zero DB backups — entire learning history is one docker volume inside the WSL2 VHD (C.1) | ops |
| 3.2 | 🟠 | Alerting configured nowhere; paper cron bypasses the alert wrapper; no "session didn't run today" detection (C.2, C.3) | ops |
| 3.3 | 🟠 | Prod DB missing migrations 013–015: Opus priced 3× high, fable-5 supervisor unpriced and untracked, cost ceiling miscounting (B.4) | econ |
| 3.4 | 🟠 | Machine loss is unrecoverable-by-documentation: secrets ×4 vendors, Task Scheduler cron-start trick, true crontab state all host-only (C.4, C.6) | ops |
| 3.5 | 🟡 | All Python deps float (`>=`), base images/wrangler unpinned, no restart policies, unbounded docker json logs (C.7–C.9) | ops |

### Tier 4 — Worth a ticket, not urgent

- Dry-run decisions persist as real buy/sell rows and enter the learning loop (A.11)
- Thesis status taxonomy split: trader writes `'closed'`, tools enum is
  `invalidated|expired|executed`; supervisor's "closed theses" query sees a
  biased subset (D.10)
- Watchlist gate fires *after* the playbook is committed; a failed strategist
  stage still trades, and its memo is skipped on that path (D.7)
- Executor free-text JSON should be a forced tool schema; one malformed
  element currently discards the whole batch (A + D.8)
- Duplicate/aliased strategist tools; `get_session_summary` name collision;
  dict tool outputs serialized as Python repr not JSON (D.9)
- Retroactive tool-result truncation invalidates the prompt cache every turn
  on 25-turn Opus loops (D.12)
- Classifier sees only ≤300-char headlines; fetched article summaries are
  stored but never shown to the model (D.11)
- `wait_for_fill` cancels a live order after ~1.5 s of transient fetch
  errors; deterministic client_order_id then broker-blocks the retry as a
  benign-looking duplicate (A.8)
- Mixed date sources: NY-pinned `session_date` vs container-local (UTC)
  `date.today()` in playbook write/read, snapshots, attribution — dormant at
  the current cron hour, breaks on evening runs (A.9)
- `patterns.analyze_signal_categories` doesn't exclude meta signal types —
  `rule_gate`/`signal_gap` appear as pseudo-signal categories (A.10)
- Partial fills accounted at submitted qty, not filled qty, in in-loop
  accounting (A.7)
- Paper/prod cross-check validates key-vs-URL only; nothing binds account
  type to database (C.10)
- $80 micro-positions are below the measurement floor: a full 1% alpha is
  $0.80 against $7.50 inference and spread (B.5)

### Confirmed sound

Intent resolution math (pure Decimal, correct clamps) · quantity
quantization · benchmark sign-flip for sells · daily-loss breaker and cost
ceiling wiring + unit tests · session idempotency gate · cross-run dedup
stack (DB dedup + deterministic client_order_id + orphan JSONL) ·
paper/prod compose isolation (separate volumes/ports/env/logs; paper lacks
Cloudflare creds) · migration runner (transactional, ON_ERROR_STOP) ·
`run-docker.sh` trap/teardown · host log rotation · market calendar
2024–2027 · claude_client loop recovery + usage accounting · watchlist
forcing functions and supervisor wiring (real, not theater — just late,
see D.7).

---

# Appendix A — Code correctness (v2 money path)

### A.1 Sell precheck passes when Alpaca reports *no position at all* — stale-state sell can reach the broker (and can open an unintended short)
**Severity: high**

`v2/trader.py:275-293`:
```python
available = get_live_available_qty(decision.ticker)   # None == "position does not exist"
...
if available is None or available >= decision.quantity:
    return True
if available <= Decimal("0.0001"):
    return _reject(f"Alpaca reports 0 available shares (DB said {held})")
```
`get_live_available_qty` (`v2/executor.py:174-191`) returns `None` specifically
when Alpaca says the position does not exist; API errors raise and are handled
fail-closed. So "0 shares available" is rejected, but the strictly worse "no
position at all" sails through. Tests cover the exception and trim branches but
never the `None` branch (`tests/v2/test_trader.py:1513-1560`) — an oversight,
not a choice.

**Failure scenario:** Step-1 position sync fails (non-fatal — `_sync_from_alpaca`
at trader.py:114-132 appends an error and continues), leaving a stale DB row for
a position closed in a prior session. Executor emits `exit_full`; qty resolves
from the stale DB `held`; precheck gets `None` and returns True; a market sell
for a non-held symbol is submitted to a live margin account — unintended short,
or broker rejection (which then triggers A.3's phantom decision row).

**Fix:** treat `available is None` like the zero-available branch:
`_reject("Alpaca reports no position (DB said {held})")`.

### A.2 Duplicate same-ticker/same-action decisions within one executor response all execute, but only one gets a decision row
**Severity: high**

All three dedup layers share the same blind spot for intra-batch duplicates:
- Pre-submit DB dedup (`trader.py:823-843`) checks `check_decision_exists`, but
  decision rows are only written in Step 6 (`_log_decisions`,
  trader.py:1341-1348) *after* the execution loop — during Step 5 every
  duplicate passes.
- `client_order_id` (`trader.py:229-239`) is
  `algo-YYYYMMDD-{b/s}-{TICKER}-{playbook_action_id|"op"}` — a playbook buy and
  an off-playbook buy of the same ticker have different keys; Alpaca accepts
  both.
- `tool_write_playbook` rejects duplicate (ticker, action) pairs within the
  playbook (`tools.py:520-540`) but can't prevent playbook + off-playbook
  duplicates in the executor's response.

At logging time the second fill hits the dedup skip in `_log_decisions`
(trader.py:1195-1202) or the partial unique index `idx_decisions_dedup`
(`db/init/016_decision_dedup.sql`) and is silently dropped.

**Failure scenario:** Haiku emits `{buy AAPL, playbook_action_id: 123}` and
`{buy AAPL, is_off_playbook: true}`. Both fill → double position size; the
second fill is an untracked order with no reasoning, no signal links, no
backfilled outcome.

**Fix:** dedupe `response.decisions` on (ticker, action) before the loop (keep
the playbook-backed one), or track an in-memory `submitted` set alongside
`check_decision_exists`.

### A.3 Broker-rejected / timed-out / failed-fill orders are logged as ordinary buy/sell decision rows — phantom trades in the learning data, and same-day retry lockout
**Severity: high**

When `execute_market_order` fails or `wait_for_fill` returns failure
(`trader.py:337-366`), the code appends to `errors` and returns
`executed=False` — but never mutates `decision.action`/`reasoning` (unlike
every pre-submit rejection path, which stamps `[REJECTED: ...]` +
`action="invalid"`). `_log_decisions` (trader.py:1176-1242) then inserts the
decision as a real `buy`/`sell` with quantity, a fresh trade price, and
`order_id=NULL`. Backfill selects `action IN ('buy','sell') AND price IS NOT
NULL` (`backfill.py:100-108`); attribution joins the same rows
(`attribution.py:68`). Nothing filters on order_id.

**Failure scenario:** (a) order submitted after 4 PM ET queues
(`TimeInForce.DAY`), `wait_for_fill` times out at 30 s and cancels — the
never-executed trade is recorded as real, gets 7d/30d outcomes, feeds
attribution, and appears in `todays_decisions` as if it happened. (b) The
phantom row trips pre-submit dedup (`trader.py:824`), so even a `--force`
re-run cannot place the intended trade that day.

**Fix:** on submit/fill failure, stamp the decision like rejection paths
(`action="invalid"`, `[FAILED: ...]`), or persist an `executed` flag /
require `order_id IS NOT NULL` in backfill+attribution eligibility.

### A.4 7d/30d outcomes are systematically computed from the same-day *in-progress* intraday bar, then frozen
**Severity: medium**

`v2/backfill.py:35-52, 94-109, 66-91`. Eligibility is
`date <= trading_day_cutoff(today, N)`; a decision on the cutoff has
`exit_date == today`. Backfill runs daily (Stage 0), so every decision is
backfilled the *first* day it becomes eligible — precisely when
`exit_date == today`. The cron fires 1 PM MST / 3 PM ET (an hour before
close), and `get_price_on_date` returns the first bar with
`bar_date >= target_date` — intraday, today's *partial* bar. The outcome is
never recomputed (`WHERE outcome_Nd IS NULL`).

**Impact:** essentially all `outcome_7d/30d` and SPY benchmarks measure a
3 PM price instead of the close — systematically shortened, noisier windows
baked permanently into `signal_attribution`, rules, and reflection.

**Fix:** require `exit_date < today` for eligibility so outcomes always use a
completed daily bar.

### A.5 `--force` re-run after a partially completed session destroys the day's executed playbook_action rows and severs decision linkage
**Severity: medium**

`_check_and_record_session` always returns `completed_stages = set()`
(`session.py:165-200`), so a `--force` retry re-runs the strategist. The
strategist's `write_playbook` calls `replace_playbook_actions_atomic`
(`trading_db.py:726-759`) which nulls `decisions.playbook_action_id` and
DELETEs **all** `playbook_actions` for the playbook — including rows already
`executed` from the earlier run.

**Failure scenario:** executor fills 3 trades, dashboard stage fails → session
failed → operator retries with `--force` (the documented flow). The 3 executed
action rows vanish; the 3 decision rows lose `playbook_action_id` while
keeping `is_off_playbook=false`; playbook_action_history/outcome_class and
carry-forward context permanently lose the record. Dedup prevents
double-trading; the learning-data damage is silent.

**Fix:** in `replace_playbook_actions_atomic`, only delete actions still
`pending` (preserve executed/failed/skipped + links), or block replacement
when any action for the date is non-pending.

### A.6 LLM-authored `thesis_id` / `playbook_action_id` are used unvalidated in DB writes
**Severity: medium**

- `agent.py:369-380`: both IDs accepted raw — no check that they exist, belong
  to today's playbook, or match the ticker.
- `trader.py:369-374`: `update_playbook_action_status(id, "executed")` — any
  integer marks that arbitrary historical action executed.
- `trader.py:387-405`: full-sell fill calls `close_thesis(thesis_id=...)` — a
  hallucinated/transposed ID closes an unrelated *active* thesis.
- `trader.py:445-455`: `_handle_thesis_invalidations` — executor can
  invalidate any thesis ID with no cross-check; the prompt invites it.

Contrast: `signal_refs` get DB validation; tickers get normalization — these
two ID fields are the only LLM-authored pointers that skip validation.

**Fix:** verify `playbook_action_id` ∈ today's pending actions for that ticker
(data already in `build_executor_input`) and `thesis_id` matches the action's
thesis / an active thesis for that ticker; drop with logged warning otherwise.

### A.7 Partial fills are accounted at submitted quantity, not filled quantity
**Severity: medium-low**

`trader.py:376-377`: `trade_value = fill_price * decision.quantity` even for
timeout-partial fills (`executor.py:469-484`). Inflated `trade_value` drives
sector-cap totals (`trader.py:945-951`), the local buying-power estimate when
post-fill refresh fails (`trader.py:430-442`), session totals; positions-dict
update uses `decision.quantity` outright (`trader.py:957-962`). Thesis-close
and logged-row paths already use `filled_qty` correctly.

**Fix:** compute in-loop accounting from `result.filled_qty` when present.

### A.8 `wait_for_fill` cancels a live order after ~1.5 s of transient fetch errors; the same-day retry is then broker-blocked as a duplicate
**Severity: medium-low**

`executor.py:415-431`: three consecutive `get_order_by_id` failures
(`fetch_retry_limit=3`, `poll_interval=0.5`) → cancel, well before the 30 s
timeout. Deterministic `client_order_id` + Alpaca uniqueness over cancelled
orders → the retry is rejected `duplicate_client_order_id` and treated as
benign ("the winner will mark it executed", trader.py:344-350) — but there is
no winner; the trade silently never happens and the playbook action is never
marked failed.

**Fix:** make the fetch-error budget time-based (poll until timeout through
transient errors); treat duplicate-key rejection as benign only after
confirming the prior order actually filled (lookup by client order id).

### A.9 Mixed date sources: NY-timezone `session_date` vs container-local `date.today()`
**Severity: low (dormant at the current 3 PM ET cron)**

`session.py:54-58` and `trader.py:1274-1275` pin the session to
America/New_York, but `tool_write_playbook` writes with `date.today()`
(`tools.py:617`), `build_executor_input` reads with `date.today()`
(`context.py:513-517`), `todays_decisions` filters on it (`context.py:601`),
`take_account_snapshot` (`executor.py:166`) and the attribution cutoff
(`attribution.py:39`) use it. Containers have no TZ configured → UTC dates.

**Failure scenario:** any run 8 PM–midnight ET (manual retry, rescheduled
cron): the strategist writes a playbook dated *tomorrow* (UTC), stage
validation `get_playbook(session_date)` fails despite the playbook existing,
executor skips; decisions/dedup keys disagree with the playbook date.

**Fix:** thread `session_date` into `build_executor_input` and
`tool_write_playbook`, or centralize on `current_market_date()`.

### A.10 `patterns.analyze_signal_categories` doesn't exclude meta signal types
**Severity: low**

`attribution.py:70` restricts to
`signal_type IN ('news_signal','macro_signal','thesis')`; `patterns.py:77-112`
has no such filter. With `db/init/028_decision_signals_meta_types.sql` +
`_log_signal_links` (trader.py:562-581), every decision citing a rule or
lacking refs emits `rule_gate`/`signal_gap` rows that become pseudo-categories
in the weekly pattern report, duplicating those decisions' outcomes into
meaningless buckets.

**Fix:** add the same `signal_type IN (...)` filter; audit other FK-join
queries in patterns.py.

### A.11 Dry-run decisions are persisted as real buy/sell rows and enter the learning loop
**Severity: low**

In dry run, `_execute_decision_order` returns success with
`order_id="DRY_RUN"` (`executor.py:288-295`) and `_log_decisions` inserts them
indistinguishable from real trades except by `order_id='DRY_RUN'`, which
nothing filters on. A `--dry-run` against the prod DB seeds phantom outcomes
into attribution and blocks that (date, ticker, action) via the dedup index
for the rest of the day.

**Fix:** log dry-run decisions with a distinguishing action/flag, or exclude
`order_id='DRY_RUN'` in backfill/attribution eligibility.

### A — Areas inspected and found sound
- Intent resolution (`v2/intents.py`): pure Decimal math, correct clamps,
  percent-vs-fraction consistent.
- Sell-side benchmark sign-flip; NULL-propagation guards in
  attribution/patterns win-rate SQL.
- Daily-loss breaker math + mid-loop re-check (but see C.5); sector-cap
  projection math incl. T1.2 mid-loop refresh.
- Quantity quantization (ROUND_DOWN 9 dp); Decimal end-to-end in order path.
- Claude client loop: max_tokens/context recovery preserves role alternation;
  usage accumulation + cost ceiling sound.
- `_aggressive_prune`/`_truncate_old_tool_results` mechanics; market calendar
  2024–2027; trading-day window arithmetic (aside from A.4's boundary).
- P1.6 idempotency stack is well-constructed for cross-run cases; the gaps are
  in-batch (A.2) and the cancelled-order duplicate-key interaction (A.8).

---

# Appendix B — Economics (prod DB, read-only)

Two independent passes agreed on all numbers; the second corrected pricing
using the actual (post-migration-014) rates because the prod `model_pricing`
table is stale — DB-priced totals overstate Opus spend 3×. Corrected figures
below. `benchmark_7d/30d` is SPY over the identical trading-day window
(`v2/backfill.py`, `BENCHMARK_TICKER="SPY"`), sign-flipped for sells;
`alpha = outcome − benchmark`.

## B-1. Equity curve (account_snapshots, 2026-02-07 → 2026-06-15, 91 rows, ~4.3 months)

| Metric | Value |
|---|---|
| First → last equity | $1,000.00 → $7,914.79 (headline "+691%" — almost entirely deposits) |
| Detected deposits | +$986.97 (02-12), +$938.12 (03-20), +$5,024.93 (04-24) ≈ $6,950 |
| Deposit-adjusted TWR (ex-deposit days) | **+0.75% total (~+2.1% annualized)** |
| Implied lifetime trading P&L | **≈ −$35** |
| P&L since last deposit (04-24 → 06-15) | −$49.54 |
| Max drawdown | −1.95% post-deposit segment (−5.11% incl. pre-deposit dip) |
| Daily vol (ex-deposit) | ~0.8%/day ≈ 13% annualized |
| Monthly TWR | Feb +0.57%, Mar −2.19%, Apr +3.40%, May −0.31%, Jun (to 6/15) −0.63% |
| Staleness | **Last snapshot 2026-06-15 — 31 days before audit date** |

Recent snapshots: ~84–85% of equity in cash, 3 open positions ($1,174 cost
basis).

## B-2. Benchmark comparison (100% backfill coverage on eligible decisions)

| Cohort | n | avg outcome | avg SPY | avg alpha | sd | beat-SPY |
|---|---|---|---|---|---|---|
| 7d buy | 54 | −1.35% | −0.72% | **−0.64%** | 4.79 | 48.1% |
| 7d sell | 45 | −3.18% | −2.32% | **−0.86%** | 8.30 | 42.2% |
| **7d all** | **99** | −2.18% | −1.44% | **−0.74%** | 6.59 | **45.5%** |
| 30d buy | 52 | +1.04% | +2.71% | **−1.66%** | 9.75 | 55.8% |
| 30d sell | 43 | −5.47% | −7.80% | **+2.33%** | 12.55 | 58.1% |
| **30d all** | **95** | −1.91% | −2.05% | **+0.14%** | 11.22 | 56.8% |

Negative alpha at 7d, ~zero at 30d. The only positive pocket (sell/30d) is
loss avoidance, not gains.

## B-3. API cost drag (corrected pricing)

| Item | Value |
|---|---|
| Tracked window | 2026-05-06 → 2026-06-15 only (token columns empty before) |
| Tracked spend | **$45.00** (May $25.80 / 16 sessions; Jun $19.19 / 11 sessions) |
| Per session | $1.67 avg, $6.01 max; strategist Opus loop ≈ 95% of spend |
| Untracked | ~55 sessions Feb–May 5 (≈$90–120 at like rates); **all 14 supervisor (fable-5) runs have NULL model/tokens — zero cost recorded**; the 24h audit loops |
| Estimated lifetime spend | **~$150–250+** |
| Cost per trade (tracked window) | 6 trades / $45 = **$7.50/trade** vs $80 avg trade notional (~9%) |
| Hurdle | ~$40/mo ÷ $7,915 = 0.5%/mo (~6%/yr) at full deployment; at actual 15% deployment, **~3.4%/month on deployed capital** just to cover API |

Gross P&L ≈ −$35 lifetime vs ~$150–250 inference: **the system has spent
4–7× more on inference than the absolute value of everything it has ever made
or lost trading.**

## B-4. Prod DB missing migrations 013–015 (finding, high)

`schema_migrations` max = `012_supervisor_memos.sql` (2026-05-28);
`db/migrations/013–015` exist on disk, including
`014_model_pricing_fable5_opus4x.sql`. Consequences: `model_pricing` has no
`claude-fable-5` row (supervisor unpriceable), Opus 4.x still at $15/$75 (3×
actual), `ALGO_LOOP_COST_CEILING_USD` prices via this table so it both
over-counts Opus and under-counts fable-5 loops. This contradicts CLAUDE.md's
note that the drift was fixed 2026-06-10, and is the third recurrence of the
documented init/migrations drift failure mode.
**Fix:** `task db:migrate`; make the supervisor record model + tokens into
`session_stages`; add a startup assertion that every model used has a pricing
row.

## B-5. Trade sizing below the measurement floor (finding, medium)

avg trade notional $78–84; CRM 14 trades incl. six March buys of $25–40;
GOOGL 12 trades with 4 buy/sell alternations in 3 weeks (matches rule-#27
oscillation); 20 same-ticker buy↔sell flips within 10 days out of 99 trades;
9 of 19 tickers ≥4 trades; lifetime traded notional ≈ $8k. At $80/trade a
full 1% alpha is $0.80 — unmeasurable against spread and dwarfed by
$7.50/trade inference.
**Fix:** minimum position size (5–10% of equity per thesis) + re-entry
cooldown per ticker.

## B-6. Equity reporting conflates deposits with performance (finding, medium)

Three single-day jumps (+96.9%, +47.3%, +171.0%) total ≈$6,950 of the $6,915
lifetime equity change; no transfers table; `account_snapshots` has no flow
column. Any naive "return since inception" reads +691% when true trading
performance is ≈0% — dashboards, memos, and any LLM context that sees equity
growth are learning from deposits.
**Fix:** capital-flows table (Alpaca transfers API); deposit-adjusted
TWR everywhere equity is surfaced, including the public dashboard.

## B-7. Statistical honesty

- 7d alpha −0.74% ± 1.31 (95% CI −2.04 to +0.56), t ≈ −1.1, p ≈ 0.27.
- 30d alpha +0.14% ± 2.26 (CI −2.11 to +2.40). Pure noise.
- Beat-SPY 45.5% ± 9.8pp (CI 35.7–55.3%) — cannot reject a coin flip.
- Power: sd(alpha)=6.6% → ~1,360 trades to detect a true 0.5%/trade edge at
  80% power; ~340 for a 1% edge. Best-ever pace 43 trades/mo → 8–32 months;
  current pace 0/mo → never. Same-day/same-ticker clustering and overlapping
  windows shrink effective n further.
- 66% of decisions (holds) generate no outcome data — the evaluated sample
  systematically excludes the system's most common action (see D.1).

## B — Verdict

**Profitable net of costs: NO — and "too early" on the edge question by
design, not just time.** Activity decayed Feb 18 → Mar 43 → Apr 25 → May 13 →
Jun 0 trades; last buy 2026-05-11, last sell 2026-05-13; sessions stopped
2026-06-15. As sized and paced, the experiment cannot answer its own
question.

---

# Appendix C — Ops resilience

### C.1 Prod Postgres has zero backups — the entire learning history is one file on one WSL2 VHD
**Severity: critical**

`grep -rn "pg_dump|pg_restore|backup"` hits only `RotatingFileHandler
(backupCount=3)` (`v2/log_config.py:49`, `trading/log_config.py:40`,
`tests/conftest.py:67`). No backup target in Taskfile, crontab, run-docker.sh,
or docs. The only copy of theses/decisions/memos/rules/attribution is docker
volume `algo_postgres_data` inside the WSL2 `ext4.vhdx`. WSL2 VHD corruption
(known failure mode on forced Windows shutdown), accidental
`docker compose down -v`, or disk failure destroys it; Alpaca can reconstruct
fills but none of the LLM reasoning history. No restore path exists to test.
**Fix:** `task db:backup` (`pg_dump -Fc`) + nightly cron, copy off the WSL
filesystem, same for db-paper, rehearse restore (`pg_restore` + `task
db:migrate`).

### C.2 Prod trading is silently disabled: installed crontab has the session line commented out; nothing noticed for a month
**Severity: critical**

`diff <(crontab -l) crontab` → installed line 17 is
`# 0 13 * * 1-5 /home/jay/dev/algo/run-docker.sh trading python -m v2.session`
while the repo file has it active. Last prod session artifact: Jun 15 (exited
1 per `logs/session_failures.log`); last paper session log: Jun 26. Audit
date Jul 15. Git history shows a prior drift repair (`218d6b6 "chore: sync
repo crontab with installed reality"`). No "session didn't run today"
detection exists — all alerting is "ran and failed" (and even that is
unconfigured, C.3).
**Fix:** dead-man's switch (healthchecks.io-style ping-on-success, alert on
silence; or a local check comparing `MAX(sessions.session_date)` to the last
market day). Commit crontab state changes; consider an audit check diffing
`crontab -l` against the repo file.

### C.3 Failure alerting is configured nowhere; the paper cron path bypasses alerting entirely
**Severity: high**

`ALGO_ALERT_WEBHOOK_URL` appears only in `run-docker.sh:25` and docs — absent
from `.env`, `.env.paper`, `.env.example`, and the installed crontab.
`run-docker.sh` executes on the host, so `.env` (env_file feeds containers
only) would never reach it. Paper cron runs `task paper:session` directly —
no `session_failures.log` entry, no webhook; paper failures are 100% silent.
**Fix:** define the var in the crontab itself or a host-side env file sourced
by run-docker.sh; wrap paper sessions in the same failure handling; add the
var to `.env.example`.

### C.4 Session trigger is plain cron on a WSL2 VM manually kept alive — missed windows are skipped, never made up
**Severity: high**

Vanilla cron (no anacron, no systemd timers). Cron runs only because
`start-wsl-cron.bat` (`wsl -u root service cron start`) is wired into Windows
Task Scheduler — documented nowhere but a two-line comment in the .bat.
Host asleep/rebooting/updating at 12:30/13:00 MST → that day's session never
happens, unrecorded and (per C.2) undetected. Task Scheduler entry lost →
system goes permanently quiet.
**Fix:** document the dependency; consider a catch-up wrapper (cron every
30 min during market hours, no-op if today's session row exists — the
idempotency gate makes this safe); C.2's dead-man's switch covers residual
risk.

### C.5 No documented human kill switch; two breaker gaps
**Severity: medium**

The 432dfb5 "kill switches" are automated breakers only:
`check_daily_loss_limit` (`v2/risk.py:114`) and the loop cost ceiling
(`v2/claude_client.py:24,543-562`) — genuinely wired (stage-start halt
`trader.py:1304-1316`, per-fill re-check `trader.py:964-988`) and unit-tested.
But no env flag, sentinel file, or documented one-liner exists for "stop
trading now" — the current de facto halt is the uncommitted crontab edit
(C.2). Gaps: (a) breaker fails open when `last_equity` missing/non-positive
(`risk.py:131-133`, documented choice); (b) the mid-loop re-check is skipped
whenever the post-fill account refresh fails (`_refresh_buying_power` returns
None on exception, `trader.py:427-437`; caller guards `if refreshed_info is
not None`, `trader.py:969`) — precisely the flaky-API moments a breaker
matters. Breaker halts new orders only; never cancels open orders.
**Fix:** explicit switch checked at session start (`ALGO_TRADING_HALTED` or a
`HALT` sentinel file) + documented one-liner and resume path; fail closed (or
retry-then-halt) when the post-fill refresh fails after a real fill.

### C.6 Machine loss is unrecoverable-by-documentation
**Severity: high**

`.env`/`.env.paper` (gitignored) hold Alpaca live keys, Anthropic key,
Cloudflare account/token/project, DB creds — no secrets inventory or setup
checklist anywhere. Host-only state: installed crontab (currently *differs*
from repo copy), Windows Task Scheduler entry, DB volume (C.1). Laptop dies →
code recoverable from GitHub, everything else reconstructed from memory
across four vendors, with no data to restore.
**Fix:** `docs/runbook-recovery.md`: secrets inventory (names + where to
re-issue), host bootstrap (crontab install, Task Scheduler, `task
db:migrate`), restore procedure from C.1's backups; encrypted off-host copy
of env files.

### C.7 Docker container json-file logs are unbounded
**Severity: low**

No `/etc/docker/daemon.json`; `docker info` → json-file driver; no max-size
limits anywhere; neither compose file sets `logging:`. Host-side app logs are
bounded (5 MB × 3; `logs/` 17 MB, `logs_paper/` 1.9 MB); Postgres WAL bounded
by stock `max_wal_size`. Risk is the long-lived paper stack growing inside
the WSL VHD.
**Fix:** `{"log-driver":"json-file","log-opts":{"max-size":"10m","max-file":"3"}}`
in daemon config or `logging:` blocks in compose.

### C.8 Every Python dependency floats; base images/toolchain float too
**Severity: medium**

`v2/requirements.txt` is 100% `>=` (`anthropic>=0.40.0`, `alpaca-py>=0.21.0`,
`psycopg2-binary>=2.9.9`, …). Dockerfile: `python:3.12-slim` (floating),
NodeSource `setup_22.x`, unpinned `npm install -g wrangler`. `postgres:16`
floats minors. No lock file. A rebuild under incident pressure (e.g. while
recovering per C.6) can pull breaking SDK changes — the money-touching
`alpaca-py` deserves pins most.
**Fix:** pip-compile/freeze lock consumed by the Dockerfile; pin
`wrangler@<version>`; consider `postgres:16.x`.

### C.9 No restart policies: the "always up" paper stack does not survive a reboot
**Severity: low**

No `restart:` keys; after the audit-day reboot all six algo containers sat
`Exited (255)`. Mitigated for sessions by `paper:session` → `deps:
[paper:up]` self-heal; dashboards/db stay down until the next weekday cron.
**Fix:** `restart: unless-stopped` on paper services (and prod db/dashboard if
between-session availability is wanted); leave prod `trading` to
run-docker.sh.

### C.10 ALPACA_PAPER cross-check validates key-vs-URL only; nothing binds account to database
**Severity: low**

`v2/executor.py:90-109` asserts `ALPACA_PAPER` agrees with `ALPACA_BASE_URL`
*within one env file*. Paper keys + `ALPACA_PAPER=true` pasted into prod
`.env` passes cleanly and paper fills then pollute the prod learning history —
silent contamination, hard to unwind. (Compose env_file paths are hardcoded,
and `.env.paper` has no Cloudflare creds, so paper can't publish the public
dashboard — good.)
**Fix:** `ALGO_PIPELINE=prod|paper` marker cross-checked against both
`ALPACA_PAPER` and a one-row `pipeline_identity` table written at volume
init.

### C — Areas inspected and found sound
Daily-loss breaker + cost ceiling implementation and tests · session
idempotency gate (idempotent skip exits 0, so cron double-fires neither
duplicate work nor false alerts) · host log rotation · Postgres WAL bounds ·
paper/prod compose separation (services, volumes, ports, env files, log
dirs) · `run-docker.sh` trap/teardown + db healthcheck gating · migration
runner (single transaction per file, ON_ERROR_STOP, atomic
schema_migrations).

---

# Appendix D — Agent architecture & prompt design

### D.1 Inaction produces zero learning signal: the loop has no gradient once it stops trading
**Severity: critical**

- `backfill.py:100-108` filters `action IN ('buy','sell')` — holds never get
  outcomes. `attribution.py:68` — same filter.
- `formation.py:18-31` — formation mode (the only structural pressure to
  trade) deactivates once 5 trades in the trailing 90d have completed
  outcomes, and only re-arms after ~90 days of total silence.
- No counterfactual tracking: a `deferred` playbook action
  (`trader.py:592-599`) gets an outcome_class label but never a "what would
  it have returned" number.

All-hold is a locally stable fixed point: no trades → no new evidence →
prompts demanding "new evidence" before acting (D.2, D.4) keep holding.
There is a dead zone between formation mode and normal operation — enough
historical trades to stay out of formation, zero current trades, no alarm.
This is the direct mechanism behind the observed all-hold collapse since
2026-05-13.
**Fix:** (1) backfill hypothetical outcomes for skipped/deferred actions and
holds on held positions (price path already fetchable via
`get_price_on_date`); surface "cost of inaction" in reflection. (2)
Activity-floor circuit: hold_rate=100% for N consecutive sessions
(`tool_get_executor_behavior_summary` already computes it) → inject a
formation-style block or hard-fail for operator review.

### D.2 The executor prompt demands "fresh evidence" it structurally cannot possess, making HOLD the only compliant output
**Severity: critical**

`agent.py:219-221`: "If uncertain: HOLD"; "REVERSAL JUSTIFICATION: … (b) the
new evidence … If you can't articulate (b), HOLD instead"; "Execute only when
today's playbook action is still valid against … fresh evidence."
But `build_executor_input` (`context.py:512-676`) contains **no news signals,
no macro signals, no thesis text, no entry/exit triggers** — only strategist
reasoning strings, prices, and 7d-old outcome numbers. The identical reversal
gate exists upstream (`ideation_claude.py:126`, strategist Critical Rule 8),
so every trade passes the same test twice — once by Opus with web_search,
again by Haiku with no research tools and no signal feed. Combined with
`strategy_rules` injected as "MUST follow" (`agent.py:193`), many carrying
`lift:UNSET` (`context.py:587`), the compliant response to nearly any
playbook action is defer. No runtime symptom check flags this: parse rates,
tool errors, truncation all look healthy while the executor does exactly what
it was told.
**Fix:** one owner for reversal/carry-forward judgment. Either the executor
becomes mechanical validity checks (position exists, price sane, risk rules)
with the "fresh evidence" language removed, or it gets the same-day signal
digest (`get_ticker_signals_context(days=1)` exists at `context.py:140`, used
only by the dead `build_trading_context` path).

### D.3 The executor invalidates theses it never sees; trader closes them blindly
**Severity: high**

`agent.py:229` — output schema includes `thesis_invalidations`.
`ExecutorInput` carries thesis_id integers but no thesis text, no
`invalidation` criteria, no exit_trigger. `trader.py:445-455` calls
`close_thesis(status="invalidated")` with no cross-check. The learning loop's
most durable state (theses = run-to-run continuity) is writable by the
least-informed, smallest-model stage; a wrong invalidation silently deletes a
trade idea + its signal citations, and the strategist just sees it gone.
**Fix:** include the referenced thesis's `invalidation` text in
`ExecutorInput`, or demote executor invalidations to a `pending_invalidation`
advisory the strategist confirms next session.

### D.4 n=3–5 attribution cohorts are converted into hard constraints and "evidence-based" rules
**Severity: high**

`attribution.py:177-218` — `build_attribution_constraints(min_samples=5)`:
n=5 with avg alpha < −0.5% becomes "CONSTRAINT: Do not create theses
primarily based on WEAK signal categories" appended to the strategist system
prompt (`ideation_claude.py:429-430`). Advisory threshold is n=3
(`attribution.py:145`). `strategy.py:141` tells reflection "Every rule must
cite specific data" with no minimum-n guidance. 60/40 at n=5 is a coin flip,
yet becomes a prohibition — and it self-seals: constraints suppress trading
in "WEAK" categories → cohorts never grow → the label is never falsified.
Combined with D.1, the system codifies noise and starves itself of the data
that could correct it.
**Fix:** raise the constraint threshold to n≥15–20 and/or require the alpha
CI to exclude zero; below threshold emit "INSUFFICIENT DATA — exploratory,
small-size trades allowed" instead of a prohibition; add minimum-n language
to reflection's rule criteria.

### D.5 "Outcomes" are 7-day price moves, not P&L, but every prompt calls them P&L; no realized or unrealized P&L is surfaced anywhere
**Severity: high**

`backfill.py:112-126` — `calculate_outcome` is post-decision price change. A
buy exited at −2% the next day still books the 7d move. `tools.py:1366`
describes it as "7d/30d P&L outcomes" to the strategist and reflection.
Portfolio context (`context.py:43-52`) is "`{ticker}: {shares} @ ${avg_cost}`"
— no current price, no unrealized P&L (positions table stores only
ticker/shares/avg_cost). `get_thesis_lineage` is the only per-thesis outcome
view; the strategist prompt never mentions it. The strategist manages exits
without being shown whether positions are up or down; reflection "reviews
performance" on a metric that is not money. A learning system whose reward
signal is mislabeled to the learner optimizes the wrong thing.
**Fix:** (1) current price + unrealized P&L per position in portfolio
context; (2) realized round-trip P&L per thesis at closure, in
`get_thesis_lineage`, reflection summary, and closure reasons; (3) rename
"P&L outcomes" to "7d/30d post-decision price move (signal quality)".

### D.6 Model-assignment inversion: Haiku vetoes Opus; the highest-leverage learning stage runs on Sonnet with 10 turns
**Severity: medium**

`agent.py:215` — the executor may execute/adjust/skip every playbook action:
full veto/resize authority over Opus's output, exercised by Haiku
(`agent.py:38`) with strategist-grade judgment demanded
(`agent.py:196-229`). Meanwhile reflection — rule lifecycle, identity,
revalidation gates, watchlist resolution, memo — runs on `claude-sonnet-4-6`
with `max_turns=10` (`strategy.py:710`, `session.py:445-449`): 7+ mandatory
tool calls before any thinking, against a hard RuntimeError if a gated
revalidation is missed (`strategy.py:848-853`). The observer-only supervisor
gets the strongest model (fable-5) for a memo nothing programmatically
consumes except watchlist rows. The hardest cognitive tasks are not on the
strongest models.
**Fix:** narrow the executor to mechanical validation (making Haiku
appropriate); move reflection to Opus/fable or raise its max_turns to ~16 and
pre-seed identity/rules/summary in the initial message (the strategist
already does this, `ideation_claude.py:444-464`).

### D.7 Watchlist and playbook "gates" fire after state is already committed; a failed strategist still trades
**Severity: medium**

`ideation_claude.py:466-476` — `wl.assert_watchlist_resolved("ideation")`
runs *after* the loop returns; `write_playbook` has already persisted. The
raise sets `strategist_error` (`session.py:364-367`) but `_run_executor_stage`
gates only on playbook existence (`session.py:379-396`) — the executor trades
on a playbook from a stage marked failed. Same ordering skips
`_persist_strategist_memo` (`session.py:357-362`): the session where the
supervisor flagged problems is exactly the session with no strategist journal
entry. (The supervisor→acting-stage wiring itself is real and enforced:
`watchlist.py:83-95`, resolve handlers `ideation_claude.py:246-249`,
`strategy.py:829-831`, ingestion `ideation_claude.py:449-456`,
`strategy.py:794-801`. Not theater — but enforcement is a post-hoc alarm, not
a gate.)
**Fix:** check open watchlist items *inside* `tool_write_playbook` (reject
the write — same in-loop forcing pattern as signal_refs); persist the
strategist memo before the watchlist assertion.

### D.8 Hand-parsed executor JSON where a forced tool schema would delete the failure class; one bad decision kills the whole batch
**Severity: medium**

`agent.py:322-351` — manual fence stripping + `json.loads`;
`agent.py:354-397` — one invalid decision raises and discards *all* decisions
for the session; `agent.py:301-320` — max_tokens truncation likewise aborts
the stage. Same pattern in `classifier.py:191-200, 394-399` (batch parse
failure → entire batch labeled noise). The executor is the only single-shot
free-text JSON emitter in a codebase that uses tool-use everywhere else, and
its output is the only money-moving artifact.
**Fix:** a single `submit_decisions` tool with enums in `input_schema`,
called with `tool_choice={"type":"tool"}`; validate decisions individually
and drop-with-telemetry rather than raising on the batch.

### D.9 Strategist tool surface bloat: duplicate/aliased tools, one name meaning two things, dict outputs as Python repr
**Severity: medium**

`tools.py:1142-1554` — the strategist receives the entire ~30-tool registry
including supervisor-oriented tools. Direct overlaps: `get_strategy_rules` vs
`get_active_rules` (same handler, `tools.py:1575-1576`), `get_strategy_history`
vs `get_session_memos`, `get_decision_history` vs `get_recent_decisions`,
`get_signal_attribution` duplicated by pre-seeded context (which says "do NOT
re-fetch" while the tools are still offered). Name collision:
`get_session_summary` = cost/failure stats in the strategist/supervisor
registry (`tools.py:1588`) but decisions+attribution+round-trips in reflection
(`strategy.py:562-571`). `claude_client.py:659-661` — tool output is
`str(output)`, so dict/list results reach the model as Python repr, not JSON.
**Fix:** per-role tool sets (supervisor.py already filters); drop aliases;
`json.dumps(output, default=str)`; rename one `get_session_summary`.

### D.10 Thesis status taxonomy is split: trader writes `'closed'`, tools only know `invalidated|expired|executed`
**Severity: medium**

`trader.py:395-399` — position exit calls `close_thesis(status="closed")`.
`tools.py:1295` — close_thesis tool enum `["invalidated","expired","executed"]`;
`tools.py:1489` — `get_theses` filter `all|active|closed` whose SQL
`WHERE status='closed'` matches only the trader-closed subset.
`supervisor.py:52` asks "Are closed theses being learned from?" — its query
silently misses every strategist-closed thesis. Any stage reasoning about
closed theses sees a biased subset (clean exits only, never invalidations).
**Fix:** treat `status != 'active'` as closed in `tool_get_theses` (or filter
on `closed_at IS NOT NULL`); normalize the trader's exit status into the enum
(`executed` fits).

### D.11 Classifier sees only ≤300-char headlines; article summaries are fetched, stored, and never shown to the model
**Severity: low**

`pipeline.py:38-41` pulls summaries from Alpaca; `classifier.py:287, 374-392`
builds prompts from `_sanitize_headline(h)` only (300-char cap,
`classifier.py:60`); summary is attached to the stored signal after
classification. Headline-only sentiment is noisy ("beats estimates but guides
down"), and that noise propagates into the attribution cohorts D.4 hardens
into constraints.
**Fix:** include ~400 chars of summary in the classification message (batch
prompt is already cached; Haiku cost is modest).

### D.12 Per-turn tool-result truncation invalidates the prompt cache prefix every turn
**Severity: low**

`claude_client.py:396-441` — `_truncate_old_tool_results` rewrites one
*additional* older message each turn once >3 tool-result messages exist;
re-applied per turn (`claude_client.py:485-500`). Any byte change at message
k invalidates cache for everything after k — from turn ~5 onward each call
rewrites cache from the earliest newly-truncated message, on 25-turn Opus
loops. Also mutates the model's memory of earlier evidence (300-char stubs)
while the prompt asks it to cite specific signal IDs from that evidence.
**Fix:** truncate deterministically at append time (never retroactively), so
message content is immutable once written and the cache prefix stays stable.

### D — Overall assessment

The plumbing is unusually well-audited — idempotency, orphan handling,
signal-ref validation, watchlist forcing functions, and telemetry are
genuinely enforced in code, and the strategist→playbook→executor→attribution
ID chain is real, not aspirational. The core weakness is at the
learning-loop level, and it coherently explains the all-hold collapse without
any runtime failure: (1) only executed trades generate outcomes, so inaction
is invisible; (2) the executor demands evidence it is never given and
defaults to HOLD, double-gating decisions Opus already justified; (3) tiny
cohorts become hard constraints that suppress the trading that would grow
them; (4) the reward signal is labeled P&L but is a 7-day price-move proxy,
with no real P&L anywhere in context. Each is individually defensible as
conservatism; together they form a ratchet that only tightens. The system
converges — toward inactivity, not better trading. Highest-leverage fixes:
counterfactual outcomes for holds/deferrals (D.1), removing the executor's
"fresh evidence" veto (D.2), minimum-n guards on attribution constraints
(D.4).

---

## Audit housekeeping

- Prod `db` container (`algo-db-1`) was brought up for the economics passes
  (SELECT-only; trading service never started) and left running.
- Security & exposure lens was started and cancelled — **not covered**; run
  separately (e.g. `/security-review`) before the public-repo flip.
- No repo files were modified by any auditor.
