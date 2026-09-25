# Subscription Conversion Program Plan

> **For agentic workers:** This is the top-level plan. It sequences two detailed plans and the manual gates between them. Execute it phase by phase, in order, and never skip a gate. Inside a phase, execute the referenced plan with superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking. When a gate says STOP, end your turn and report; do not improvise past it.

**Goal:** Move Pinchy's strategist stage from the pay-per-token Anthropic API onto the operator's Claude Max subscription, safely, with a measured comparison, and without changing how orders are placed.

**Architecture:** Three phases with a hard gate between each. Phase 0 restores a working API baseline. Phase 1 lands file seams and a `--resume` flag with no backend change. Phase 2 pilots only the strategist under headless Claude Code, alternating with the API path on paper for two weeks, then a written go/no-go. The full inversion of the other stages is out of scope for this program.

**Tech Stack:** Python 3.12 in docker, PostgreSQL 16, bash on the WSL host, Claude Code CLI 2.1.280, pytest (`task test INSTANCE=paper -- <args>`), ruff (`task lint`).

**Spec:** `docs/superpowers/specs/2026-09-22-claude-session-inversion-design.md` (status section names the two phase plans).

**Detailed plans this program executes, in order:**
1. `docs/superpowers/plans/2026-09-22-session-seams.md` (Phase 1)
2. `docs/superpowers/plans/2026-09-22-strategist-pilot.md` (Phase 2)

`docs/superpowers/plans/2026-09-22-claude-session-inversion.md` is superseded reference material. Do not execute tasks from it.

## Global Constraints

Read these before every phase. They are repeated in the detailed plans; here they are the short list a worker must never violate.

- **Never pass `--bare` to `claude`.** Bare mode does not read the subscription login. It fails with `"is_error": true` and **exit code 0**, so the exit code of a `claude` call proves nothing. Read `is_error` from the JSON result.
- **Never put `ANTHROPIC_API_KEY` in the environment of a `claude` call.** The driver uses `env -i HOME=… PATH=…` for exactly this reason. Do not "fix" a failing call by exporting the key.
- **The money path is untouched.** `_validate_llm_ids` → `_execute_decisions` → `_handle_thesis_invalidations` → `_log_decisions` in `v2/trader.py` is called, never reimplemented, never bypassed. The executor stays on the API path in this program.
- **`live` is never touched.** Every command in this program uses `INSTANCE=paper`. `instances/live.HALT` stays in place. If a step would touch `live`, STOP.
- **Cron stays under `cron-wrap.sh`.** Never add a bare crontab line.
- **Tests run in docker.** Host python is 3.10 and cannot run the suite. `task test INSTANCE=paper -- tests/...`.
- **No test may reach the network.** New DB-touching import-site names go into the patch tuples in `tests/v2/conftest.py` as each plan specifies. A test that burns real tokens is a plan failure, not a flake.
- **One PR per phase**, from a branch off `main`, opened with `gh pr create`. Do not merge PRs; the operator merges.
- **Commit messages end with** `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **Things only the operator can do** are marked OPERATOR. Ask for them by ending your turn with a one-paragraph request; do not attempt them.

## Roles

| Who | Does |
|---|---|
| Worker (you) | Executes the phase plans task by task, runs the verification commands, opens PRs, reports at gates. |
| OPERATOR | Tops up API credits, merges PRs, runs `claude` interactively to log in, installs the crontab, reads pilot memos, makes the Phase 3 decision. |

---

## Phase 0: Baseline

The paper instance has failed every session since 2026-08-27. The cause is not code: the Anthropic API credit balance is exhausted. Both later phases measure against a working API baseline, so nothing else starts until this phase is green.

- [ ] **Step 0.1: Confirm the current failure is the credit balance**

```bash
docker compose -p pinchy-paper --env-file instances/paper.env exec -T db \
  psql -U "$(grep ^POSTGRES_USER instances/paper.env | cut -d= -f2)" \
       -d "$(grep ^POSTGRES_DB instances/paper.env | cut -d= -f2)" -Atc \
  "SELECT s.session_date, st.stage_name, left(st.error, 80) FROM session_stages st JOIN sessions s ON s.id = st.session_id WHERE st.status='failed' ORDER BY st.id DESC LIMIT 4;"
```

Expected: rows for `strategist` and `strategy` whose error contains `credit balance is too low`. If the error is anything else, STOP and report the text; the baseline problem is different from what this plan assumes.

- [ ] **Step 0.2: OPERATOR — top up API credits**

End your turn with this request: "Phase 0 needs Anthropic API credits on the account behind `ANTHROPIC_API_KEY` in `instances/paper.env`. Roughly $10 per trading day covers a session with headroom. Reply when done."

- [ ] **Step 0.3: Run one paper session by hand**

On a weekday, after the 12:30 MST cron slot has already fired for the day (so the run does not collide with cron):

```bash
task session INSTANCE=paper -- --force
```

Expected: exit 0. Then:

```bash
docker compose -p pinchy-paper --env-file instances/paper.env exec -T db \
  psql -U "$(grep ^POSTGRES_USER instances/paper.env | cut -d= -f2)" \
       -d "$(grep ^POSTGRES_DB instances/paper.env | cut -d= -f2)" -Atc \
  "SELECT s.id, s.status, string_agg(st.stage_name||':'||st.status, ' ' ORDER BY st.id) FROM sessions s JOIN session_stages st ON st.session_id=s.id WHERE s.session_date = CURRENT_DATE GROUP BY s.id ORDER BY s.id DESC LIMIT 1;"
```

Expected: `completed` and every stage `completed` (dashboard may be `completed` or absent depending on `ALGO_DASHBOARD_PUBLISH`). If `strategist` or `strategy` is `failed`, read the error, fix only what the error names, and rerun with `--force`. If you cannot fix it in one attempt, STOP and report.

- [ ] **Step 0.4: Record the baseline cost**

```bash
docker compose -p pinchy-paper --env-file instances/paper.env exec -T db \
  psql -U "$(grep ^POSTGRES_USER instances/paper.env | cut -d= -f2)" \
       -d "$(grep ^POSTGRES_DB instances/paper.env | cut -d= -f2)" -Atc \
  "SELECT session_date, round(sum(cost_usd)::numeric,2) FROM session_stage_costs c JOIN sessions s ON s.id=c.session_id WHERE s.status='completed' GROUP BY session_date ORDER BY session_date DESC LIMIT 5;"
```

Write the numbers into your gate report. Historical completed sessions cost $1.8 to $8.7.

**GATE 0 → 1.** Proceed only when three consecutive cron paper sessions (Mon–Fri) have `status = completed`. Report the three dates and costs, then continue. Phase 1 code work may start on a branch before the three sessions finish, but the Phase 1 PR is not opened until they do.

---

## Phase 1: Seams (no backend change)

Execute `docs/superpowers/plans/2026-09-22-session-seams.md`, Tasks 1 through 6, in order. That plan contains every test, every code block and every commit message. Do not summarise or reorder it.

- [ ] **Step 1.1: Branch**

```bash
git checkout main && git pull && git checkout -b session-seams
```

- [ ] **Step 1.2: Execute the seams plan, one task at a time**

For each task: write the failing tests, run them and confirm they fail for the stated reason, implement, run the task's test command and `task lint`, commit with the given message. After each task, run the full suite once:

```bash
task test INSTANCE=paper && task lint
```

Expected: all green. If a test outside the task's own files breaks, fix the cause in the same commit; do not skip or xfail it.

- [ ] **Step 1.3: Verification runs (from the seams plan's "Verification and merge")**

```bash
# 1. replay check for the executor seam (dry run, no orders)
docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading \
  python -m v2.trader --emit-input /app/logs/executor-input.json --dry-run
echo '{"decisions": [], "thesis_invalidations": [], "market_summary": "replay check", "risk_assessment": "none"}' > logs/paper/decisions-empty.json
docker compose -p pinchy-paper --env-file instances/paper.env exec -T trading \
  python -m v2.trader --decisions-file /app/logs/decisions-empty.json --input-file /app/logs/executor-input.json --dry-run
# 2. resume is a no-op after a completed day
task session INSTANCE=paper -- --resume
```

Expected: the replay exits 0 with no orders; the resume run logs `SKIPPED (completed in prior run)` for every stage and exits 0. The four keys in the decisions file are exactly the ones `executor_output_schema()` marks required (seams plan, Task 2).

- [ ] **Step 1.4: Open the PR**

```bash
git push -u origin session-seams
gh pr create --base main --title "Session seams: file-based stage seams, deterministic news curation, --resume" \
  --body "$(cat <<'EOF'
Phase 1 of docs/superpowers/plans/2026-09-22-subscription-conversion-program.md.
Implements docs/superpowers/plans/2026-09-22-session-seams.md. No backend change.

- Prompts in v2/prompts/*.md, loaded by the Python constants (parity tests)
- Executor / classifier / changelog emit+consume seams
- get_curated_news ranks deterministically (Haiku call removed)
- --resume reopens the day's session row and skips completed stages

Full suite green in docker; replay and resume checks run by hand (see plan).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 1.5: OPERATOR — merge**

End your turn: "Phase 1 PR is open. After merge, the next cron paper session must complete on the merged code before Phase 2 starts."

**GATE 1 → 2.** Proceed only when the Phase 1 PR is merged **and** the next cron paper session on the merged code has `status = completed`. Report the date.

---

## Phase 2: Strategist pilot

Execute `docs/superpowers/plans/2026-09-22-strategist-pilot.md`, Tasks 1 through 5, in order. Its "Global Constraints" section lists the CLI facts verified on this host on 2026-09-22; re-read it before Task 1.

- [ ] **Step 2.1: Branch**

```bash
git checkout main && git pull && git checkout -b strategist-pilot
```

- [ ] **Step 2.2: Confirm the subscription login works before writing any code**

```bash
cd /tmp && env -i HOME="$HOME" PATH="$PATH" claude -p "Reply with exactly: ok" --output-format json --model claude-haiku-4-5-20251001 --tools "" ; cd -
```

Expected: JSON with `"is_error":false` and `"result":"ok"`. If `is_error` is true with "Not logged in", STOP and request: "OPERATOR — run `claude` interactively once and `/login`, then reply." Do not set an API key.

- [ ] **Step 2.3: Execute the pilot plan, one task at a time**

Same discipline as Phase 1. Two tasks have manual checks that need the real stack:
- Task 3 Step 6 (MCP stdio smoke) and Task 4 Step 8 (`WITH_MCP=1 task driver:smoke INSTANCE=paper`) must both pass before Task 4 is committed.
- Task 4 Step 9 (one real run with `--resume`) must be done on a weekday **after** that day's cron API session has completed. It writes a second playbook and memo for the day; that is expected once. Paste the five query results into your report.

After every task:

```bash
task test INSTANCE=paper && task lint
```

- [ ] **Step 2.4: Open the PR**

```bash
git push -u origin strategist-pilot
gh pr create --base main --title "Strategist pilot: strategist on the Claude subscription, A/B against the API path" \
  --body "$(cat <<'EOF'
Phase 2 of docs/superpowers/plans/2026-09-22-subscription-conversion-program.md.
Implements docs/superpowers/plans/2026-09-22-strategist-pilot.md.

- session-driver.sh sandwich: API path → one isolated non-bare `claude -p` for the strategist → `v2.session --resume`
- v2/session_ctl.py brackets the stage (validators, usage, telemetry, memo)
- v2/mcp_server.py serves the strategist registry over stdio with a tool-call cap
- scripts/smoke-claude-headless.sh guards auth, isolation, stdin, empty tools, MCP
- crontab: paper Mon/Wed/Fri API, Tue/Thu pilot, same cron-wrap label
- comparison protocol: docs/superpowers/specs/2026-09-22-strategist-pilot-comparison.md

Executor stays on the API path. live untouched. Hand run results in the PR comments.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2.5: OPERATOR — merge and install the crontab**

End your turn: "Phase 2 PR is open. After merge: (1) `docker compose -p pinchy-paper --env-file instances/paper.env build trading && task up INSTANCE=paper`, (2) `WITH_MCP=1 task driver:smoke INSTANCE=paper`, (3) `crontab /home/jay/dev/algo/crontab` and confirm two paper lines with `crontab -l`. On pilot days (Tue/Thu) avoid interactive Claude Code between 10:30 and 13:30 MST."

**GATE 2 → 3.** The two-week window starts on the first Tuesday after the crontab is installed. Nothing in the repo changes during the window except bug fixes that a failed pilot run names explicitly. A `[usage_limit]` or `[auth]` failure is not a bug to fix; it is data for the decision.

---

## Phase 3: Decision

- [ ] **Step 3.1: Pull the comparison**

Run the query in `docs/superpowers/specs/2026-09-22-strategist-pilot-comparison.md` ("What to pull") against the paper DB and paste the table into the "Outcome" section of that file. Also list, for each pilot day, the path of `logs/paper/sessions/<date>/strategist/system-prompt.md` and the `strategy_memos` row id for its strategist memo.

- [ ] **Step 3.2: Apply the four criteria mechanically**

For each criterion in that file's "Decision" section, write PASS or FAIL with the number that decided it. Criterion 2 (harness leakage) needs the memos read; grep them first for `CLAUDE.md`, `Claude Code`, `memory`, `file`, `Bash`, `Read(`, and list any hit for the operator.

- [ ] **Step 3.3: OPERATOR — decide**

End your turn with the filled-in outcome and one of three recommendations, chosen by the rules in the comparison file:
- **Continue:** all four PASS. Next step is a follow-up plan for the reflection stage, written from the superseded full plan's Task 7 and Task 8 reflection branches.
- **Extend two weeks:** only criterion 3 or 4 FAIL. Name the cause from the transcripts.
- **Stop:** criterion 1 or 2 FAIL. Restore the single crontab line (`30 12 * * 1-5 … task session INSTANCE=paper`), commit the crontab change and the outcome, and leave the code in place for a later attempt.

Do not act on the recommendation yourself.

---

## Reporting format at every gate

One message, in this order, no more than 200 words plus the requested tables:

1. Phase and step reached.
2. Verification commands run and their actual output (paste, do not paraphrase).
3. Anything skipped or failed, with the exact error text.
4. The single OPERATOR action needed, if any.

## Stop conditions (end the turn immediately, report, do not work around)

- Any command would touch `live`, `instances/live.env`, or `instances/live.HALT`.
- A test or script tries to reach the network, or the suite's cost telemetry shows real tokens spent by tests.
- `claude` reports not logged in, or any step suggests exporting `ANTHROPIC_API_KEY` to make `claude` work.
- A verification command's output does not match the plan's "Expected" and the fix is not obvious from the error text after one attempt.
- The detailed plan and this program disagree. Report the disagreement; do not pick one.
