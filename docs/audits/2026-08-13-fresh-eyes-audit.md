# Fresh-Eyes Audit — 2026-08-13

Third full audit of Pinchy (prior: 2026-07-15, on branch `fresh-eyes-audit-2026-07`).
Scope: repo structure, money-path correctness on `main`, database state (prod + paper),
operations, and gap analysis against the July audit. Five parallel read-only
investigations; every finding below was verified against the actual code, DB, or logs.

## Executive summary

**The system has been dead for two months and nothing noticed.** Prod's last session was
2026-06-15 (failed on exhausted Anthropic API credits); paper limped along failing every
LLM stage until 2026-07-21, then the host was powered off from ~Jul 22 until tonight
(2026-08-13 19:56). No alert fired because no alerting is configured anywhere.

**The single biggest finding is process, not code:** `main` has zero commits since
2026-06-10. The entire July audit — all six TDD'd Tier-1 money-path fixes, the operator
kill switch, backup tooling, the recovery runbook, and the report itself — sits on branch
`fresh-eyes-audit-2026-07`, pushed 4 weeks ago, PR never opened, CI never run against it.
It is a clean fast-forward (merge-base = main HEAD). Anyone resuming from `main` gets the
pre-audit money path with all six known ways to lose money incorrectly. A second stranded
branch, `promote-paper-to-prod`, holds the *approved* go-forward plan; it was never
executed either.

**Meta-pattern across three audits:** discovery works, fixes get built well, and then the
last mile — merge, install, configure, monitor — doesn't happen. The July audit's own
same-day fixes (crontab hygiene, backups, alerting) all regressed or never activated.
Until the last-mile problem is fixed, further audit findings are of limited value.

Bright spots, verified: schema migrations are fully applied on both DBs (the July drift is
gone), referential integrity is clean (0 orphans, the 6 historical thesis orphans are
fixed), model pricing rows are correct, ruff is clean, CI was green when it last ran, and
the intent→quantity Decimal path, paper/prod env cross-check, and playbook transaction
handling are sound.

## System status (2026-08-13)

| | Prod | Paper |
|---|---|---|
| Last session | 2026-06-15 (**failed**, API credits) | 2026-07-21 (**failed**, API credits) |
| Sessions total / failed | 69 / 14 | 56 / 26 (46% failure rate) |
| Last decision row | 2026-06-11 | 2026-06-22 |
| Positions in DB | 3 (~$1.2k, stale — account since liquidated per branch HALT file) | 14 |
| Equity | $1,000 → $7,914.79 (~$7,000 of that is deposits; **net trading P&L ≈ −$85**) | $98,895 → $99,751.56 (+0.87%) |
| Attribution last updated | 2026-06-21 | 2026-07-21 |
| Cron | prod session line commented out (installed copy only — uncommitted) | active, will fire next market day the host is on, and fail on credits |
| Alerting | none (`ALGO_ALERT_WEBHOOK_URL` unset) | none, and the paper path bypasses `run-docker.sh` failure logging entirely |
| Public dashboard | frozen at 2026-06-15 | n/a (skipped by design) |
| Backups | one manual dump pair from 2026-07-15, same disk | same |

Root cause chain: strategist stopped emitting playbook actions 2026-05-27 (all-hold
ratchet) → prod effectively stopped trading → Anthropic credits ran out ~2026-06-15 →
every LLM stage 400s → account liquidated (recorded only on the unmerged branch's HALT
file) → host off Jul 22–Aug 13 → nothing alerted at any step.

---

## Status (updated 2026-08-13, branch `fresh-eyes-audit-2026-08`)

Work started against the recommended sequence. This branch is rebased onto
`fresh-eyes-audit-2026-07`, so the July Tier-1 fixes are underneath it and
findings 1.1–1.6 are carried here pending that branch's merge.

| Finding | Status | Where |
|---|---|---|
| 0.1 July branch stranded | **blocked on you** — `gh` token lacks `createPullRequest`; PR body prepared, branch verified fast-forward | — |
| 0.2 hiatus only in local state | **fixed** (July `cf1b5be` + crontab installed) | `crontab`, `HALT` |
| 0.3 no alerting anywhere | **fixed in code, needs your URLs** | `e60c0ba` |
| 2.2 `db/init` can't reproduce live schema | **fixed + enforced** | `213f82c` |
| 2.3 backups stale / in-repo | **fixed** — off-WSL copies verified, cron installed | `1feba0a`, `d03a0e0` |
| hygiene: tracked `settings.local.json`, un-gitignored dirs | **fixed** | `d03a0e0` |
| Stale-but-live weekly `v2.learn` cron | **fixed** — line retired, crontab installed | `crontab` |

Still open and untouched: 0.4, 1.7–1.10, the July carryovers (A.7, A.9), 2.1,
2.4–2.8, and all of Tier 3.

Two things still need you, and nothing below is safe to resume without them:
open the `fresh-eyes-audit-2026-07` PR, and fill in `.env.host` with a real
webhook URL and per-job heartbeat URLs. The alerting code is inert until those
values exist — which is precisely the failure mode this audit is about.

## Tier 0 — before anything else runs

### 0.1 (P0) The July safety layer is stranded on an unmerged branch
`fresh-eyes-audit-2026-07` is 10 commits strictly ahead of `main` (fast-forward, no
conflicts possible — PR #116's "kill switches" are automated breakers that predate the
audit and are deliberately layered under the branch's *operator* kill switch). Contains:
sell-with-no-position rejection (`8291e1d`), intra-batch dedup (`864c263`), broker-reject
stamping (`368a651`), LLM-ID validation (`8ba6ecb`), fail-closed breaker re-check
(`60c7399`), `--force` playbook preservation (`ea9091d`), HALT sentinel +
`ALGO_TRADING_HALTED` (`cf1b5be`), backup tasks + runbook (`785a612`), and the audit
report. **Action: open the PR, let CI run, merge.** Closes findings 1.1–1.6 and most of
Tier 2's backup/kill-switch items in one step.

### 0.2 (P0) The hiatus exists only as uncommitted local state
Repo `crontab` on `main` has the prod session line **active**; only the installed copy
(uncommitted) comments it out. The file's own header says "Install with: `crontab
/home/jay/dev/algo/crontab`" — following it re-arms live trading against a liquidated
account. No HALT file, no `ALGO_TRADING_HALTED` check, no runbook on `main`. (Fixed by
0.1's merge + committing the HALT sentinel.)

### 0.3 (P0) Two-month silent outage: no alerting exists on any path
`ALGO_ALERT_WEBHOOK_URL` unset in both env files; the paper cron (`task paper:session`)
never goes through `run-docker.sh`, so even the failure log doesn't cover it; CLAUDE.md
documents the webhook under container-side knobs but `run-docker.sh` reads it on the
**host**, so the documented configuration path doesn't work. There is no dead-man's
switch, so "host off for 3 weeks" and "credits exhausted for 2 months" are
indistinguishable from "all fine." July finding C.2/C.3, untouched.

### 0.4 (P1) The approved go-forward plan was never executed
`promote-paper-to-prod` (2 commits, 2026-07-15, pushed, no PR): approved design + 565-line
plan to make the paper stack primary — the answer to the July economics finding (0.2). It
references `.env.paper` knobs that are not actually set (`ALGO_EXECUTOR_MODEL` absent) and
an audit doc that lives on the *other* unmerged branch. Decide: execute it, or reject it
explicitly.

---

## Tier 1 — money-path bugs live on `main`

Findings 1.1–1.6 are the July Tier-1 bugs, still live because the fixes are unmerged (see
0.1). New findings 1.7–1.10 need fresh work **on top of** the branch.

- **1.1 (P0)** Sell precheck passes when Alpaca reports *no position at all*
  (`v2/trader.py:282` — `available is None` → allow). With a stale DB position (sync
  failure is non-fatal, see 1.10), an `exit_full` reaches the broker as a market sell of
  an unheld symbol → unintended short on a margin account.
- **1.2 (P1)** Broker-rejected / failed-fill orders are logged as real buy/sell rows
  (`v2/trader.py:337-366`): phantom trades get 7d/30d outcomes, poison attribution, and
  block same-day retries via dedup.
- **1.3 (P1)** Intra-batch duplicate (ticker, action) decisions all execute; dedup reads
  the DB but rows are written after the loop (`v2/trader.py:823` vs `:1341`), and
  broker-side dedup keys on `playbook_action_id` so playbook + off-playbook dupes both
  submit. Two fills, one decision row.
- **1.4 (P1)** LLM-authored `playbook_action_id`/`thesis_id` hit DB writes unvalidated
  (`v2/agent.py:369-380`): can mark arbitrary historical actions executed, close or
  invalidate arbitrary theses, or FK-fail the decision insert *after* a real fill.
- **1.5 (P1)** Mid-loop daily-loss breaker is silently disabled whenever the post-fill
  account refresh fails (`v2/trader.py:971` gates the re-check on `refreshed_info is not
  None`) — fails open exactly when the broker API is degraded.
- **1.6 (P2)** `--force` strategist re-run deletes the day's *executed* playbook actions
  and NULLs decision linkage (`v2/database/trading_db.py:754-759`).
- **1.7 (P1, new)** Session-creation race: check-then-insert with no DB uniqueness
  (`db/init/029` **dropped** `UNIQUE (session_date, session_type)`) and no `flock` in
  `run-docker.sh`/Taskfile. Two concurrent runs both pass the idempotency gate; run B's
  `replace_playbook_actions_atomic` deletes run A's actions mid-execution and re-issued
  action IDs defeat broker-side dedup → double-trading. Fix: restore the unique
  constraint + host-side flock.
- **1.8 (P2, new)** Decisions skipped by a mid-loop `break` (daily-loss halt at
  `v2/trader.py:988`, 10-trade cap at `:910`) are still logged as real buy/sell rows with
  a fresh price — a sibling of 1.2 that the branch fix does **not** cover.
- **1.9 (P2, new)** `wait_for_fill` cancel-failure path (`v2/executor.py:456-493`)
  reports a possibly-still-live DAY order as a clean zero-fill failure; it can fill later
  → real position with no linked record.
- **1.10 (P2, new)** Position-sync failure is non-fatal (`v2/trader.py:114-132`); the
  session then sizes sells and buy caps from stale positions. Should fail closed like the
  account-snapshot path. This is the enabling condition for 1.1.
- **Still open from July (no fix anywhere):** partial fills accounted at submitted
  quantity (A.7, `v2/trader.py:376`); mixed date sources — ET `session_date` vs
  container-UTC `date.today()` in context/tools/snapshots (A.9), which corrupts evening
  `--force` retries.

---

## Tier 2 — reliability, reproducibility, ops

- **2.1 (P0)** Prod executes the host working tree, not an artifact: the image contains
  no code (`Dockerfile` copies only requirements; everything is a `:ro` bind mount), no
  commit SHA is recorded per session, and the checkout at cron time is whatever was left
  — with 125 local branches and 9 worktrees on this machine. Minimum fix: record `git
  rev-parse HEAD` + `git status --porcelain` into the session row and refuse to trade on
  a dirty/non-main tree; better: build code into the image.
- **2.2 (P0)** `db/init/` cannot reproduce the live schema: `thesis_signals`
  (migrations/003) and `llm_call_contexts` (migrations/006) exist **only** in
  `db/migrations/` — no init counterpart. CI seeds Postgres from `db/init/*` only, so CI
  tests a structurally different schema than prod, and any fresh volume / DR restore
  comes up missing tables the strategist reads. Related: migration 014 only `UPDATE`s
  opus-4-8 pricing (init/035 was never mirrored — on any volume seeded before init/035
  the update matches zero rows, reproducing the exact bug it claims to fix), and init
  008–027 (20 files) predate the mirror convention entirely. Add a CI check that the
  mirror is bidirectional.
- **2.3 (P1)** Backups regressed to "one stale dump": exactly one prod+paper pair from
  2026-07-15, on the same WSL vdisk as the volumes, no cron installed (the nightly backup
  lines live only on the unmerged branch). Also: those 26 MB dumps sit **untracked and
  un-gitignored in the repo root** of a repo slated to go public — one `git add -A` from
  publishing full trading history. Same class: `.wrangler/`, `.claude/worktrees/`.
- **2.4 (P1)** No dependency pinning anywhere: three requirements files, all `>=`, no
  lockfile; the container that holds live brokerage credentials can silently pick up new
  major versions of `anthropic`/`alpaca-py` on any rebuild. It also still installs
  `tweepy`/`atproto` for the pipeline deleted in May — and `.env` still carries the
  retired Twitter/Bluesky credentials, which should be **revoked**, not just deleted.
- **2.5 (P1)** Prod and paper share one compose project and network: the `task docker:up`
  gate matches `trading-paper` for `grep trading`, `run-docker.sh`'s EXIT-trap
  `docker compose down` can hit a live paper run, `set -e` can skip `notify_failure`,
  and paper containers can resolve the prod `db` hostname — isolation is one env-file
  string deep.
- **2.6 (P1)** Prod↔paper Taskfile parity is broken: no paper equivalents for `session`
  standalone stages, `test`, `supervise`, etc.; paper mounts omit `tests/`/`pytest.ini`.
  Paper can't fully rehearse prod.
- **2.7 (P1)** `ALGO_*` knobs parse at import time with no validation (`v2/risk.py:39`,
  `v2/agent.py:38-39`, `v2/claude_client.py:24`): a typo like `3%` is an ImportError that
  kills *every* stage, violating the stages-are-independent contract, with no
  stage-failure row.
- **2.8 (P2)** Public dashboard frozen since 2026-06-15 — a "live" page showing
  two-month-old data is a reputational bug for the pending public launch.

---

## Tier 3 — strategy, learning, economics (the open July core)

- **3.1 (P1)** All-hold ratchet **confirmed with mechanism**: 14 consecutive playbooks
  with zero actions 2026-05-27 → 06-15 (last `playbook_actions` row 05-26); June decisions
  are 0 buy / 0 sell / 12 hold, reasoning citing rule gates (Rules 36/41/44). Monthly mix
  decayed Feb 13/5/114 → Jun 0/0/12. Prod rules: 10 active / 39 retired. The ratchet is
  strategist-side (empty playbook), not executor-side. No work on main or any branch.
- **3.2 (P1)** Economics unchanged: net trading P&L ≈ −$85 on ~$8,000 deposited over 4
  months, while API spend ran until the account's credits were exhausted — the system
  literally halted on its own API bill. The approved answer (`promote-paper-to-prod`,
  zero-capital learning) was never executed (see 0.4).
- **3.3 (P2)** Learning loop frozen mid-stride: attribution stale (prod 06-21, paper
  07-21); 4 prod + 12 paper trades past their 30d window will never backfill until
  sessions resume; supervisor watchlist has 3 prod + 5 paper items frozen open, including
  a lapsed July-1 rule-retirement deadline and paper #64 ("rule_gate signal_type defect
  still corrupting attribution") — a supervisor-flagged data defect never resolved.
- **3.4 (P2)** Attribution presents n=1 scores (e.g. guidance −17.80, sample_size 1) to
  the strategist with no minimum-sample gate; 5 of 12 prod categories have n ≤ 2.
- **3.5 (P2)** Flip-flop churn was real but is historical (XLE 5 buy→sell reversals ≤7d,
  GOOGL 4 incl. buy 04-20/sell 04-21/buy 04-22): it stopped only because trading stopped.
  Rule-27-style oscillation remains undressed if trading resumes.
- **3.6 (P2)** 15% of paper decisions are `invalid` from intents resolving to 0 shares
  (`trim_to_portfolio_pct`/`exit_full` on tiny positions) — upstream sizing noise.

---

## Structure & hygiene (condensed)

- Dead v1 (`trading/`, 4.4k LOC, zero v2 importers) is still mounted into the live
  container; `run-docker.sh`'s own usage hint (`python trading/main.py`) runs v1 against
  the live account with none of v2's gates. ~15k LOC of v1 code+tests still gate CI.
- `v2/dashboard/` is dead (self-documented broken, empty templates) while CLAUDE.md says
  "v2 dashboard lives in `v2/dashboard/`" — both compose stacks actually run the v1
  Flask dashboard, which still serves `/tweets`.
- `Dockerfile` CMD targets `v2.main`, which does not exist.
- Committed `.claude/settings.local.json` (gitignore is ineffective on tracked files)
  ships a broad Bash allowlist (`sudo`, `curl`, `pip install`, …) to anyone who clones.
- Pre-commit hook claims to run ruff **and pytest**; it runs only `task lint`.
- Doc drift: README says 11 strategist tools (actual: 32) and "filters by embeddings"
  (actual: Haiku call); `v2/BUGS.md`/`v2/RETRO.md` document deleted modules; two spec
  docs still say "Draft" for long-merged features; CLAUDE.md documents the alert webhook
  in the wrong config layer (see 0.3).
- `v2/learn.py` — the weekly cron job that failed Jun 14 and Jun 21 — has **no tests**.
  Every other v2 module has a matching test file (2,006 test functions on main).
- Cruft: root-owned empty *directory* named `conftest.py` at the pytest rootdir; broken
  py3.10 `.venv` against a py3.12 codebase; 117 merged branches undeleted; stale ollama
  container/volume; May-era PNGs in `logs/`; `session.py` argparse hardcodes a third
  model-selection mechanism (`--model claude-opus-4-8`).
- Stale-but-live weekly cron: Sunday 05:00 `v2.learn` still runs the full prod stack
  against the liquidated account.

## Verified clean

- `schema_migrations`: 15/15 applied on **both** DBs (prod applied 013–015 on 07-16); table
  sets identical; `model_pricing` correct (fable-5 $10/$50, opus-4-x $5/$25).
- Referential integrity: 0 orphan `decision_signals` (all types — the 6 historical thesis
  orphans are gone), 0 orphan playbook_actions/decisions; validation trigger present.
- Money-path items verified sound: intent→quantity resolution (Decimal, clamped,
  ROUND_DOWN), paper/prod env cross-check fails fast, strict executor JSON validation,
  `signal_refs` DB-validated, atomic playbook writes, orphan-decision JSONL fallback,
  cost ceiling, telemetry isolation.
- Ruff clean; CI green on all of its last 10 runs (though none since 06-10).

## Recommended sequence

1. **Same day:** open + merge the `fresh-eyes-audit-2026-07` PR (fast-forward; closes
   1.1–1.6, kill switch, backups, runbook). Commit the hiatus: HALT file + crontab with
   the prod line commented. Add `backups/`, `.wrangler/`, `.claude/worktrees/` to
   `.gitignore` and move the dumps out of the repo. Revoke the retired Twitter/Bluesky
   credentials.
2. **Before any session resumes:** restore Anthropic credits *only after* alerting
   exists — set `ALGO_ALERT_WEBHOOK_URL` on the host, route the paper cron through
   `run-docker.sh` (or give it equivalent failure logging), and add a dead-man's switch
   (e.g. healthchecks.io ping per session). Install the backup cron off-disk. Fix the
   init/migrations mirror both directions + add the CI mirror check; make migration 014
   an upsert.
3. **Next code work (on top of the merged branch):** 1.7 (unique constraint + flock),
   1.8 (stamp halt-skipped decisions), 1.10 (fail-closed position sync), 1.9 (live-order
   flag); record git SHA per session and refuse dirty trees; pin dependencies.
4. **Strategic decision:** execute or explicitly reject `promote-paper-to-prod`. If the
   project resumes, the all-hold ratchet (3.1) and attribution sample gates (3.4) are the
   first strategy-side work — resuming as-is resumes into the same fixed point, now
   unobserved.

## Appendix — key stats

| Metric | Prod | Paper |
|---|---|---|
| Sessions (total/completed/failed) | 69 / 55 / 14 | 56 / 30 / 26 |
| Decisions (buy/sell/hold/invalid/skip) | 340 (54/45/226/6/9) | 89 (31/39/6/13/0) |
| Date range of decisions | 2026-02-07 → 06-11 | 2026-04-09 → 06-22 |
| Theses (total/active) | 246 / 4 | 67 / 18 |
| Playbooks / actions | 89 / 245 (last action 05-26) | 38 / 130 (06-22) |
| Rules active/retired | 10 / 39 | 16 / 10 |
| Watchlist open/acted/dismissed | 3 / 48 / 14 | 5 / 50 / 11 |
| news_signals / macro_signals | 21,852 / 3,412 | 9,820 / 1,529 |
| Equity first → last | $1,000 (02-07) → $7,914.79 (06-15) | $98,895 (04-09) → $99,751.56 (06-22) |
| Net trading P&L | ≈ −$85 (~−1% on ~$8k deposited) | +0.87% |
| Max drawdown (daily closes) | 5.11% | 0.42% |
| Tests / LOC | 2,006 test functions; v2 16.7k LOC, v1 4.4k LOC | |
