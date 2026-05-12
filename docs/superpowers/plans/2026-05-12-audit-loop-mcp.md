# Audit Loop + Atlassian MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `v2/audit.py` + `v2/audit_jira.py` with a `/loop`-driven, Atlassian-MCP-mediated audit that files Jira tickets for findings and executes human-approved fixes as PRs (code) or two-step DB mutations.

**Architecture:** A long-lived Claude Code `/loop 24h` session that, each tick, reads `docs/audit-playbook.md` and runs two phases: Discovery (port of the deterministic checks from `v2/audit.py` plus an ideation pass) and Execution (work on tickets the human transitioned to `Approved`/`Apply`). All Jira I/O is via the Atlassian MCP. Most of the existing audit machinery (Python module, dashboard pages, DB tables, REST integration, env vars) is deleted.

**Tech Stack:** Markdown (the playbook is the spec of what gets audited), Atlassian MCP, Bash (docker exec psql, gh CLI), Python (only for the throwaway dry-run rehearsal helper), SQL.

**Spec:** `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`

---

## Caveats before starting

Read these before doing any task.

1. **Uncommitted changes to audit files.** The working tree at plan-write time had local edits to `v2/audit.py`, `v2/audit_jira.py`, `tests/v2/test_audit.py`, `tests/v2/test_audit_jira.py`, `Taskfile.yml`, and `db/migrations/001_widen_ticker_columns.sql`. Confirm with the user whether these should be discarded (the plan deletes those files anyway), committed first, or stashed. Do NOT silently overwrite them.

2. **No TDD on the playbook.** The playbook is a markdown document Claude reads inside a /loop. There's no unit test that validates "the audit produces correct findings" — the audit IS the LLM reading the playbook. Verification is by manual rehearsal (Task 11) before going live. This is unusual for our codebase; do not invent fake tests.

3. **Jira workflow is a manual prerequisite.** Task 8 is a checklist the **user** performs in the Jira admin UI. The implementation engineer cannot complete Task 8 on the user's behalf. Block on user confirmation before proceeding to Task 11 (rehearsal).

4. **Don't merge to main until rehearsal passes.** Tasks 1-7 and 9-10 can land via PR. Task 11 (rehearsal) must succeed before Task 12 (start the loop).

---

## File structure

**New files**
- `docs/audit-playbook.md` — the audit's source of truth: deterministic checks, ideation instructions, filing rules.
- `db/migrations/005_drop_audit_tables.sql` — drops `audit_findings`, `audit_runs`, `audit_llm_calls`.

**Deleted files**
- `v2/audit.py`
- `v2/audit_jira.py`
- `tests/v2/test_audit.py`
- `tests/v2/test_audit_jira.py`
- `tests/v2/test_audit_dashboard.py`
- `dashboard/templates/audit.html`
- `dashboard/templates/audit_finding.html`
- `db/init/025_audit_findings.sql` (so fresh DBs don't recreate the tables)

**Modified files**
- `dashboard/app.py` — remove `/audit`, `/audit/findings/<id>`, `/audit/findings/<id>/status` routes (lines 413-436 at plan-write time); remove imports of `get_open_audit_findings`, `get_audit_finding`, `get_recent_audit_runs`, `update_audit_finding_status`.
- `dashboard/queries.py` — remove `get_open_audit_findings`, `get_audit_finding`, `get_recent_audit_runs`, `update_audit_finding_status` (lines ~621-660 at plan-write time).
- `dashboard/templates/base.html` — remove any nav link to `/audit` if present.
- `v2/database/trading_db.py` — remove audit helpers around lines 1074 (`audit_runs`), 1094 (`audit_findings` insert), 1149 (run finalize), 1174-1179 (`audit_llm_calls`), 1216-1254 (read helpers). Use `grep -n "audit_" v2/database/trading_db.py` to locate everything at execution time; line numbers will have drifted.
- `Taskfile.yml` — remove `audit`, `audit:apply`, `paper:audit`, `paper:audit:apply` targets (lines 217-235 at plan-write time).
- `CLAUDE.md` — remove the audit-related env var paragraphs in the "Environment Variables" section (search for `ALGO_AUDIT_FILE_JIRA` and delete the surrounding doc block, plus the `ALGO_EXECUTOR_MAX_TOKENS` reference to `check_executor_max_tokens_hit`, plus the `ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS` paragraph).
- `.env` and `.env.paper` if checked in (they're typically gitignored — check first).

---

## Task graph

```
Task 1 (Catalog checks)
   └─> Task 2 (Author playbook deterministic section)
          └─> Task 3 (Author playbook ideation + filing sections)
                 └─> Task 4 (Commit playbook + spec patch)

Task 5 (Delete v2/audit.py + audit_jira.py + tests)
Task 6 (Delete dashboard audit routes/templates/queries)
Task 7 (Drop audit-related Taskfile targets + CLAUDE.md doc)
   (Tasks 5, 6, 7 are independent of each other and of 1-4; can parallelize.)

Task 8 (USER: configure Jira workflow) — manual, blocks Task 11

Task 9 (Write migration + delete init script)
Task 10 (Apply migration on local dev + paper DBs, verify dashboard still boots)

Task 11 (Rehearsal: dry-run a discovery tick by hand) — depends on 1-10 + Task 8
Task 12 (Start /loop 24h in long-lived session) — depends on Task 11 passing
```

---

## Phase 1: Author the playbook

### Task 1: Catalog the existing checks from v2/audit.py

This task produces a catalog used by Task 2. No file is created yet — the catalog is structured notes the engineer holds in scratch (or in a temp file) to consult when writing the playbook.

**Files:**
- Read: `v2/audit.py`

- [ ] **Step 1: Enumerate every `check_*` function**

Run:
```bash
grep -n "^def check_" v2/audit.py
```

Expected: ~28 functions. Capture the full list.

- [ ] **Step 2: For each check function, extract the canonical SQL and the finding shape**

For every check function from Step 1, read the function body and capture (in a temp file `/tmp/audit-check-catalog.md` or scratch notes):

- **`check_code`** — the string the function passes to `Finding(check_code=...)`. Use grep to find it: `grep -A 30 "^def check_<name>" v2/audit.py | grep "check_code="`.
- **SQL** — the `cur.execute("...")` text in the function body. Copy it verbatim, including parameters.
- **`finding_when`** — the Python predicate the function uses to decide a finding fires (e.g., `if count > 0`, `if ratio > 0.1`). Translate to English.
- **`severity`** — the string passed as `severity=`.
- **`tier`** — the int passed as `tier=`.
- **`title`** — the string passed as `title=`.
- **`body`** — the f-string or templated body.
- **`evidence`** — the dict shape passed as `evidence=`.
- **`auto_fix`** — whether the function declares an `auto_fix` callable. Note YES/NO. (All YES checks become `worktype: db` in the playbook.)

This is mechanical but tedious. Take it function by function. There are ~28; budget ~15 min total.

- [ ] **Step 3: Classify each check's worktype**

For each cataloged check, assign:
- `worktype: db` if the check has an `auto_fix` callable, OR if the natural fix is a SQL mutation.
- `worktype: code` otherwise (the check identifies a code/config issue — fix is a PR).

Examples (from prior knowledge of the file):
- `check_orphan_fks` → `db` (auto_fix nulls bad FKs)
- `check_stage_failure_rate` → `code` (root cause is a code bug)
- `check_executor_truncation_rate` → `code` (raise `ALGO_EXECUTOR_MAX_TOKENS`)
- `check_rule_judgment` → drop entirely; this is the Haiku judgment call on rules, which Claude-in-the-loop replaces by reading the same data directly.
- `check_audit_gaps_opus`, `check_app_improvements_opus` → drop entirely; these are the Opus ideation calls, replaced by Claude-in-the-loop reading the same sources.

- [ ] **Step 4: Mark each check's target environment**

For each check, decide:
- `env: prod` — runs against the live trading DB (default for all today's checks).
- `env: paper` — runs against `db-paper`. (Currently none; the existing audit doesn't run against paper.)
- `env: both` — useful for integrity checks (orphan FKs etc.) that we want on both.

Default to `env: prod` if unsure; the playbook can be amended later.

- [ ] **Step 5: Commit the catalog as a scratch artifact**

This task's output is consumed by Task 2 and then deleted. No commit needed unless you find the catalog useful as a future reference — in which case, save as `docs/audit-check-catalog-snapshot.md` and commit:

```bash
git add docs/audit-check-catalog-snapshot.md
git commit -m "docs: snapshot v2/audit.py checks before deletion"
```

---

### Task 2: Author the playbook's deterministic-checks section

**Files:**
- Create: `docs/audit-playbook.md`

- [ ] **Step 1: Create the file with the header and structure scaffolding**

Write `docs/audit-playbook.md` with exactly this top-level structure (fill in deterministic check entries in Step 2):

```markdown
# Audit Playbook

This file is read by the Claude Code /loop session on each 24h tick. It is the single source of truth for what gets audited. Edit this file to change the audit; no code change required.

> **Spec:** `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`

## How to read this file

When invoked from the /loop:

1. Execute every entry under **Deterministic checks** in order.
2. Run the **Ideation pass** per its instructions.
3. For every finding (deterministic or ideation), apply the **Filing rules**.
4. Then perform **Phase B (Execution)** per the spec.
5. Stop and wait for the next interval.

## Environment

Pick the docker service based on each check's declared `env`:
- `prod` → `docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"`
- `paper` → `docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"`
- `both` → run twice (once each env). File env in the ticket title prefix: `[audit:<category>:<env>] <Title>`.

The `$POSTGRES_USER` and `$POSTGRES_DB` come from `.env` and `.env.paper`; they're already in the trading container's environment.

## Deterministic checks

(filled in by Step 2 of Task 2)

## Ideation pass

(filled in by Step 1 of Task 3)

## Filing rules

(filled in by Step 2 of Task 3)

## Phase B: Execution

(filled in by Step 3 of Task 3)
```

Run:
```bash
git add docs/audit-playbook.md
git commit -m "docs: scaffold audit playbook"
```

- [ ] **Step 2: Port each cataloged check into the playbook**

For each entry from the Task 1 catalog (excluding `check_rule_judgment`, `check_audit_gaps_opus`, `check_app_improvements_opus`), append an entry to the **Deterministic checks** section of `docs/audit-playbook.md` in this exact format:

````markdown
### <check_code>

- **env:** prod | paper | both
- **severity:** critical | warn | info
- **category:** integrity | health | cost | quality | (others from the catalog)
- **worktype:** code | db
- **topic_slug:** kebab-case-derived-from-check_code
- **title_template:** "Concise human title"
- **sql:**
  ```sql
  SELECT ...
  ```
- **finding_when:** "<English predicate, e.g. 'rows returned' or 'count > 10'>"
- **body_template:** "{var1} ...{var2}..." (Python-style format string referencing columns from the SQL result)
- **suggested_fix:** (db worktype only) one of:
  - **sql:** the UPDATE/DELETE/INSERT SQL the Apply step will run. MUST be idempotent and MUST have a non-trivial WHERE clause.
  - **dry_run_probe:** SQL that counts rows the mutation would affect, e.g. `SELECT count(*) FROM ... WHERE ...`
- **suggested_fix:** (code worktype only): "Free-text description of the fix Claude should implement in the PR. Mention the file(s) and the change. Example: 'In `v2/executor.py`, raise the default for `ALGO_EXECUTOR_MAX_TOKENS` from 8192 to 16384; update the docs block in CLAUDE.md.'"
````

Required: **every check from the catalog gets exactly one entry**. Don't skip checks. Don't bundle related checks. If a check looks redundant with another, keep both — it's safer to file two tickets and human-merge them than to silently drop coverage.

Specific notes per check (overrides):
- `check_orphan_fks` — port as ONE check entry; the SQL parameterizes over the four signal_type values it handles today.
- `check_stage_failure_rate` and similar threshold checks — set `finding_when` to the same threshold the Python uses (e.g., `failure_rate > 0.10`).
- For checks whose SQL is complex (e.g., `check_decisions_missing_signal_refs`, `check_theses_missing_signal_refs`), copy the full SQL verbatim into the playbook block. Don't simplify.

- [ ] **Step 3: Verify every check from Task 1's catalog has an entry**

Run:
```bash
grep "^### " docs/audit-playbook.md | wc -l
```

Expected: count matches (cataloged checks − 3 dropped ones).

If the count is off, find the missing checks: paste the catalog's check_codes alongside `grep "^### " docs/audit-playbook.md` output and reconcile.

- [ ] **Step 4: Commit**

```bash
git add docs/audit-playbook.md
git commit -m "docs(audit-playbook): port deterministic checks from v2/audit.py"
```

---

### Task 3: Author the playbook's ideation + filing sections

**Files:**
- Modify: `docs/audit-playbook.md`

- [ ] **Step 1: Write the Ideation pass section**

Replace the placeholder under **## Ideation pass** with this exact content:

````markdown
## Ideation pass

After running every deterministic check, perform one ideation pass. The goal is to surface findings the deterministic checks won't catch — concept-level gaps, cost trends, prompt-engineering hunches, missed audit coverage.

**Sources to read:**

- Last 5 rows of `strategy_memos` (the trading system's session journal):
  ```sql
  SELECT session_id, content, created_at FROM strategy_memos ORDER BY created_at DESC LIMIT 5;
  ```
- Last 14 days of `decisions`:
  ```sql
  SELECT id, symbol, action, qty, reasoning, outcome_summary, created_at
  FROM decisions WHERE created_at > now() - interval '14 days'
  ORDER BY created_at DESC;
  ```
- Recent commits on `main`:
  ```bash
  git log --since="14 days ago" --oneline main
  ```
- The dashboard pages (read via `curl http://localhost:3000/<path>` or by reading the corresponding template + rendering function in `dashboard/`):
  - `/mistakes`
  - `/attribution`
  - `/performance`
  - `/strategy`

**What to look for:**

1. **Audit gaps.** A pattern in the decisions or memos that today's deterministic checks would miss. Example: rule #N has been retired but still appears in playbook actions. Category: `audit_gap`.
2. **App improvements.** Concrete, actionable changes to the trading code or prompts. Example: the executor prompt mentions a tool that has been renamed. Category: `app_improvement`.

**Constraints:**

- Emit at most 3 ideation findings per tick. The cap protects ticket churn.
- Each finding's `topic_slug` must be a kebab-case noun phrase derived from the finding's core idea (e.g., `rule-27-flip-flop`, `executor-token-budget`). The slug + check_code together must dedup the finding across days, so be consistent: if the same idea recurs, use the same slug.
- `check_code` for ideation findings is one of: `ideation_audit_gap` | `ideation_app_improvement`.
- `worktype` for ideation findings defaults to `code`. Override to `db` only when the natural fix is a SQL mutation.
````

- [ ] **Step 2: Write the Filing rules section**

Replace the placeholder under **## Filing rules** with this exact content:

````markdown
## Filing rules

For every finding (deterministic or ideation):

### Fingerprint

```
fingerprint = sha256(check_code + ":" + topic_slug).hexdigest()[:16]
```

A 16-char prefix is enough for label dedup. The full hash is unnecessary.

### Dedup search

Before creating, search Jira via the Atlassian MCP:

```
project = ALGO AND labels = "audit-fingerprint:<fingerprint>"
```

**Note: no `statusCategory` filter.** This is intentional — a closed (Done / Won't Fix) ticket also suppresses re-filing. See the spec's "Failure modes" section.

If any issue comes back, skip this finding. Do not file, comment, or transition.

### Create

If no dedup hit, file via `mcp__atlassian__createJiraIssue`:

- **project:** `ALGO`
- **summary:** `[audit:<category>] <title_template-rendered>` (prepend `:<env>` if env is `both`)
- **issue type:** `Task`
- **labels** (all of):
  - `audit-source:claude`
  - `audit-fingerprint:<fingerprint>`
  - `audit-category:<category>`
  - `audit-worktype:<code|db>`
- **priority:** `Medium` by default. Use `High` for `severity: critical`. Use `Low` for `severity: info`.
- **description (ADF):**
  - Paragraph: rendered body_template
  - Blockquote: top-of-evidence quote (first 500 chars of the SQL result or the ideation source excerpt)
  - Horizontal rule
  - Paragraph: `Filed by /loop audit tick on <YYYY-MM-DD>. Fingerprint: <fingerprint>. Suggested fix:` + (`<suggested_fix_sql>` for db / `<suggested_fix_text>` for code)

### Cap

Max **5** creates per tick across deterministic + ideation. Remaining findings wait until tomorrow.
````

- [ ] **Step 3: Write the Phase B (Execution) section**

The playbook must be self-contained for the /loop. Append after Filing rules:

````markdown
## Phase B: Execution

After Phase A completes, query Jira for actionable tickets:

```
project = ALGO
AND labels = "audit-source:claude"
AND status in ("Approved", "Apply")
```

For each result (cap at **2 per tick**), branch on `(status, audit-worktype)`:

| Status | Worktype | Action |
|---|---|---|
| Approved | code | Transition → `In Progress`. Create branch `audit/<ISSUE_KEY>` from `main`. Implement the change described in the ticket's suggested fix. Run `task test`. Push branch. `gh pr create --base main --title "audit: <ISSUE_KEY> <short title>" --body "Closes <ISSUE_KEY>. <one-line summary>. Generated by /loop audit."`. Transition → `In Review`. Comment on the Jira ticket with the PR URL. **Stop**: human merges + transitions to Done. |
| Approved | db | Transition → `In Progress`. Generate the SQL fix from the ticket's `suggested_fix` block. Run the `dry_run_probe` SQL to get a row count + `EXPLAIN` of the mutation. Post a comment on the ticket with this exact format: `### Dry-run preview\n\nSQL:\n\`\`\`sql\n<mutation SQL>\n\`\`\`\nRows that will be modified: <N>\nEXPLAIN: <plan>\n\nTransition to "Apply" to execute.` Transition → `Pending Apply`. **Stop**: human reviews + transitions to Apply (or back to Approved). |
| Apply | db | Transition → `In Progress`. Read the most recent comment on this ticket authored by the Atlassian-MCP account whose body begins with `### Dry-run preview`. Parse the SQL out of its fenced code block. Execute that exact SQL — do NOT regenerate. Capture rows affected. Comment: `Applied. Rows affected: <N>.` Transition → `Done`. |

**Concurrency:** the transition to `In Progress` is the lock. The Phase B JQL excludes `In Progress`, so a concurrent tick cannot double-work the same ticket.

**Per-tick cap:** 2. Remaining actionable tickets wait for the next tick.

**Failure handling:** if any step throws, do NOT roll back transitions. Leave the ticket in whatever state it landed in (probably `In Progress`). Log the error. The human will inspect and either retry (transition back to `Approved` or `Apply`) or abandon (transition to `Done`).

**Reference:** the design rationale and lifecycle examples live in the spec, `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`. Read it once for context; the table above is the authoritative operational reference.
````

Verify:
```bash
grep -n "^## Phase B" docs/audit-playbook.md
```

Expected: one match.

- [ ] **Step 4: Verify the playbook ends with no placeholders**

Run:
```bash
grep -n "(filled in by\|TODO\|TBD\|FIXME" docs/audit-playbook.md
```

Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add docs/audit-playbook.md
git commit -m "docs(audit-playbook): add ideation, filing rules, and phase B"
```

---

### Task 4: Self-review the playbook against the spec

**Files:**
- Read: `docs/audit-playbook.md`, `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`

- [ ] **Step 1: Compare**

For each section in the spec's "Phase A: Discovery" and "Playbook structure" sections, confirm the playbook implements it:

- [ ] `check_code + topic_slug` fingerprinting → present in Filing rules
- [ ] Dedup search has no `statusCategory` filter → present
- [ ] 5-create cap → present
- [ ] Label scheme matches spec → present
- [ ] Both env modes documented → present
- [ ] Ideation cap of 3 → present
- [ ] Phase B is present in the playbook (the table with status/worktype/action) → confirm via `grep -n "^## Phase B" docs/audit-playbook.md`
- [ ] Phase B per-tick cap of 2 → present

- [ ] **Step 2: Fix any gaps inline. No commit if nothing changes.**

---

## Phase 2: Deletions

These three tasks are independent and can run in parallel.

### Task 5: Delete v2/audit.py, v2/audit_jira.py, and their tests

**Files:**
- Delete: `v2/audit.py`, `v2/audit_jira.py`
- Delete: `tests/v2/test_audit.py`, `tests/v2/test_audit_jira.py`, `tests/v2/test_audit_dashboard.py`

- [ ] **Step 1: Confirm no other code imports these modules**

Run:
```bash
grep -rn "from v2.audit\b\|import v2.audit\b\|from v2 import audit\b\|from v2.audit_jira\b" --include="*.py" . 2>/dev/null | grep -v "\.claude/worktrees/\|tests/v2/test_audit"
```

Expected: no matches outside test files (which are also being deleted).

If there are matches: stop and report to the user. The plan assumed `v2/audit.py` is only referenced by its own tests and `v2/audit_jira.py`.

- [ ] **Step 2: Delete the files**

```bash
git rm v2/audit.py v2/audit_jira.py tests/v2/test_audit.py tests/v2/test_audit_jira.py tests/v2/test_audit_dashboard.py
```

- [ ] **Step 3: Run the test suite to confirm nothing else broke**

```bash
python3 -m pytest tests/ -x -q 2>&1 | tail -30
```

Expected: PASS (count drops by the number of deleted tests; no remaining failures).

If a test fails because it imported one of the deleted symbols: that test is a hidden coupling. Either fix it or include it in this deletion (with note in commit).

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: delete v2/audit.py, v2/audit_jira.py, and their tests"
```

---

### Task 6: Delete dashboard audit routes, templates, queries

**Files:**
- Modify: `dashboard/app.py` (remove audit routes + imports)
- Modify: `dashboard/queries.py` (remove audit query helpers)
- Modify: `dashboard/templates/base.html` (remove nav link to /audit if present)
- Delete: `dashboard/templates/audit.html`, `dashboard/templates/audit_finding.html`
- Modify: `v2/database/trading_db.py` (remove audit DB helpers)

- [ ] **Step 1: Identify all dashboard audit references**

Run:
```bash
grep -n "audit" dashboard/app.py
grep -n "audit" dashboard/queries.py
grep -n "audit_findings\|audit_runs\|audit_llm_calls\|audit_finding\b" v2/database/trading_db.py
grep -rn "audit" dashboard/templates/ 2>/dev/null
```

Capture the line ranges to remove.

- [ ] **Step 2: Edit `dashboard/app.py`**

Remove these (line numbers from plan-write time; will have drifted):
- The four imports at lines 13, 22, 29, 59: `get_audit_finding`, `get_open_audit_findings`, `get_recent_audit_runs`, `update_audit_finding_status`.
- The three routes at lines 413-436: `@app.route("/audit")`, `@app.route("/audit/findings/<int:finding_id>")`, `@app.route("/audit/findings/<int:finding_id>/status", methods=["POST"])`, plus their handler functions `audit_page`, `audit_finding_page`, `audit_finding_status`.

Use `Edit` per chunk (one `Edit` for the imports, one per route). Read the file first to get current line content.

- [ ] **Step 3: Edit `dashboard/queries.py`**

Remove the four functions at lines ~621-660 (plan-write time): `get_open_audit_findings`, `get_audit_finding`, `get_recent_audit_runs`, `update_audit_finding_status`.

- [ ] **Step 4: Delete the templates**

```bash
git rm dashboard/templates/audit.html dashboard/templates/audit_finding.html
```

- [ ] **Step 5: Edit `dashboard/templates/base.html`**

Search for `audit`:
```bash
grep -n "audit" dashboard/templates/base.html
```

If a `<a href="/audit">` nav link exists, remove it (use `Edit`). If none, skip.

- [ ] **Step 6: Edit `v2/database/trading_db.py`**

Remove all audit-related helpers (around lines 1074, 1094, 1117, 1129, 1149, 1174-1216, 1216-1254 at plan-write time). Use `grep -n "audit_" v2/database/trading_db.py` at execution time to enumerate; line numbers will have drifted. The set is:
- Functions that INSERT/UPDATE into `audit_runs`, `audit_findings`, `audit_llm_calls`.
- Functions that SELECT from `audit_findings` or `audit_runs`.

If any non-audit function uses these helpers transitively: stop and report. There shouldn't be any.

- [ ] **Step 7: Run the test suite**

```bash
python3 -m pytest tests/ -x -q 2>&1 | tail -30
```

Expected: PASS. Some tests may need their imports/mocks updated if they referenced `get_open_audit_findings` etc., but at this point those test files were deleted in Task 5.

- [ ] **Step 8: Smoke test the dashboard boots**

If the dashboard is currently running, restart it. Otherwise:
```bash
docker compose up -d dashboard && docker compose logs --tail=50 dashboard
```

Expected: dashboard starts without errors. Curl the home page:
```bash
curl -sSf http://localhost:3000/ -o /dev/null && echo "dashboard OK"
```

Expected: `dashboard OK`. Any 500 means a residual audit reference; grep again and fix.

- [ ] **Step 9: Commit**

```bash
git add dashboard/ v2/database/trading_db.py
git commit -m "refactor: remove audit pages, queries, and DB helpers from dashboard"
```

---

### Task 7: Drop audit Taskfile targets and CLAUDE.md docs

**Files:**
- Modify: `Taskfile.yml`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Remove `Taskfile.yml` audit targets**

Use `Edit` on `Taskfile.yml` to remove the four target blocks (lines 217-235 at plan-write time):
- `audit:`
- `audit:apply:`
- `paper:audit:`
- `paper:audit:apply:`

Verify:
```bash
grep -n "audit" Taskfile.yml
```

Expected: no matches.

- [ ] **Step 2: Update CLAUDE.md**

Open `CLAUDE.md` and remove these blocks:
- The paragraph beginning "**Audit Jira filing** (gated; off by default..." and all the env var bullets under it (`ALGO_AUDIT_FILE_JIRA`, `ALGO_AUDIT_JIRA_MAX_CREATES`, `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_AUDIT_PROJECT_KEY`, `JIRA_AUDIT_ISSUE_TYPE`).
- The `ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS` paragraph in the "Optional knobs" section.
- The reference to `check_executor_max_tokens_hit` in the `ALGO_EXECUTOR_MAX_TOKENS` paragraph (the audit check is being deleted; the env var stays but the doc reference goes).

Add a new short paragraph in its place (under "Environment Variables"):

```markdown
**Audit:** the audit runs as a Claude Code `/loop 24h` session driven by `docs/audit-playbook.md`. It files Jira tickets via the Atlassian MCP (no `JIRA_*` env vars required). See spec `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`.
```

- [ ] **Step 3: Verify**

```bash
grep -n "ALGO_AUDIT\|JIRA_EMAIL\|JIRA_API_TOKEN\|JIRA_AUDIT" CLAUDE.md
```

Expected: no matches (or only references inside the new "Audit" paragraph if its prose mentions them as deleted).

- [ ] **Step 4: Commit**

```bash
git add Taskfile.yml CLAUDE.md
git commit -m "chore: drop audit Taskfile targets and CLAUDE.md env vars"
```

---

## Phase 3: Manual Jira workflow setup

### Task 8: USER configures the ALGO project's Jira workflow

**This task is performed by the user (Jay) in the Jira admin UI. The implementation engineer does NOT perform this task. Block on user confirmation before Task 11.**

- [ ] **Step 1: Add statuses to the ALGO project**

The user adds these statuses (if not already present):

| Status | Category |
|---|---|
| `Approved` | To Do |
| `In Review` | In Progress |
| `Pending Apply` | In Progress |
| `Apply` | To Do |

Default `To Do`, `In Progress`, `Done` should already exist.

- [ ] **Step 2: Wire up transitions**

The user adds these workflow transitions:

| Name | From | To | Performer |
|---|---|---|---|
| Approve | `To Do` | `Approved` | Human |
| Reject | `To Do` | `Done` | Human |
| Start work | `Approved` | `In Progress` | Claude |
| Submit for review | `In Progress` | `In Review` | Claude |
| Submit dry-run | `In Progress` | `Pending Apply` | Claude |
| Approve apply | `Pending Apply` | `Apply` | Human |
| Request revision | `Pending Apply` | `Approved` | Human |
| Start apply | `Apply` | `In Progress` | Claude |
| Mark applied | `In Progress` | `Done` | Claude |
| Resolve post-merge | `In Review` | `Done` | Human |
| Force close | any | `Done` | Human (kill switch) |

- [ ] **Step 3: Verify permissions**

The Atlassian MCP-connected account needs in ALGO: Create Issues, Edit Issues, Add Comments, Transition Issues.

- [ ] **Step 4: User signals task complete to the implementation engineer**

User says explicitly: "Jira workflow setup done, proceed to rehearsal." Do not assume; ask.

---

## Phase 4: Migration

### Task 9: Write the migration to drop audit tables

**Files:**
- Create: `db/migrations/005_drop_audit_tables.sql`
- Delete: `db/init/025_audit_findings.sql`

- [ ] **Step 1: Write the migration**

Create `db/migrations/005_drop_audit_tables.sql`:

```sql
-- 005_drop_audit_tables.sql
--
-- Drops the audit-related tables. The audit is being rewritten as a
-- Claude Code /loop driven by docs/audit-playbook.md; Jira is the
-- canonical store for findings now.
--
-- audit_llm_calls FK-references audit_runs, so drop order matters
-- (or use CASCADE).
--
-- Spec: docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md

DROP TABLE IF EXISTS audit_llm_calls CASCADE;
DROP TABLE IF EXISTS audit_findings CASCADE;
DROP TABLE IF EXISTS audit_runs CASCADE;
```

- [ ] **Step 2: Delete the init script so fresh DBs don't recreate the tables**

```bash
git rm db/init/025_audit_findings.sql
```

If there are any other `db/init/*.sql` files referencing these tables (FKs, views, indexes):
```bash
grep -l "audit_runs\|audit_findings\|audit_llm_calls" db/init/*.sql
```
Stop and report if any exist; they need editing too.

- [ ] **Step 3: Commit**

```bash
git add db/migrations/005_drop_audit_tables.sql db/init/025_audit_findings.sql
git commit -m "feat(db): migration to drop audit_runs/findings/llm_calls"
```

---

### Task 10: Apply the migration to local prod + paper DBs and verify

**Files:**
- No code changes; this is an execution + verification task.

- [ ] **Step 1: Confirm both DBs are running**

```bash
docker compose ps db
docker compose -f docker-compose.yml -f docker-compose.paper.yml ps db-paper
```

Both should be `running`.

- [ ] **Step 2: Apply on prod DB**

If you have a migration runner task (check `Taskfile.yml` for `migrate` or similar), use it. Otherwise:
```bash
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /db/migrations/005_drop_audit_tables.sql
```

(Adjust path if the container mounts migrations differently; check `docker-compose.yml`.)

Expected: `DROP TABLE` x3 (or x3 NOTICE if already absent).

- [ ] **Step 3: Apply on paper DB**

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /db/migrations/005_drop_audit_tables.sql
```

Same expected output.

- [ ] **Step 4: Verify the tables are gone**

```bash
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt" | grep -E "audit_(runs|findings|llm_calls)"
```

Expected: no matches.

- [ ] **Step 5: Restart the dashboard and confirm it still works**

```bash
docker compose restart dashboard && docker compose logs --tail=30 dashboard
curl -sSf http://localhost:3000/ -o /dev/null && echo "OK"
curl -sSf http://localhost:3000/audit && echo "/audit still routes — BUG" || echo "/audit gone — OK"
```

Expected: `OK` for home, the `/audit` curl returns 404 (or connection drops) — the second `echo` is the success branch.

- [ ] **Step 6: No commit (this task only applies the migration; the SQL was committed in Task 9).**

---

## Phase 5: Go live

### Task 11: Rehearsal — dry-run a discovery tick by hand

**Files:**
- Read: `docs/audit-playbook.md`

**This task is performed by the user in a Claude Code session, observing one full discovery cycle WITHOUT MCP create calls. The implementation engineer prepares the rehearsal prompt and walks the user through it.**

- [ ] **Step 1: Prepare the rehearsal prompt**

Have the user open a fresh Claude Code session in `/home/jay/dev/algo` and paste this prompt (do NOT use `/loop` — single one-shot):

```
You are about to rehearse one tick of the audit loop. DO NOT call any MCP
write tool (createJiraIssue, transitionJiraIssue, addCommentToJiraIssue).
Read MCP tools (searchJiraIssuesUsingJql) are allowed.

Read docs/audit-playbook.md and execute Phase A (Discovery) exactly as
specified, with one substitution: instead of calling
mcp__atlassian__createJiraIssue, print the full payload you WOULD have
sent as JSON, with a heading like:

  DRY RUN — would file:
  <JSON>

Run every deterministic check. Run the ideation pass. Apply the 5-create
cap. Report what you found.
```

- [ ] **Step 2: User runs it, reviews output**

The user reads each `DRY RUN — would file:` block. For each:
- Is the title sensible?
- Are the labels correct (especially the worktype)?
- Is the SQL or suggested fix actionable?
- Does the body contain enough evidence to act on?

If problems found: the user edits `docs/audit-playbook.md`, commits, and re-runs the rehearsal.

- [ ] **Step 3: User explicitly confirms rehearsal passed**

User says: "Rehearsal passed, OK to start /loop." Block on this confirmation; do not proceed otherwise.

---

### Task 12: Start the long-lived /loop session

**Files:**
- None.

- [ ] **Step 1: User opens a fresh Claude Code session in a terminal that will stay open**

Suggest `tmux` or `screen` so the session survives terminal-window-close. Example:
```bash
tmux new-session -s audit-loop
cd /home/jay/dev/algo
claude
```

- [ ] **Step 2: User issues the /loop command in that session**

In Claude Code prompt:

```
/loop 24h Read docs/audit-playbook.md and execute one audit tick exactly as specified there (Phase A: Discovery, then Phase B: Execution). Then stop and wait for the next interval.
```

- [ ] **Step 3: User observes the first tick**

Watch the first tick run to completion. Confirm:
- Phase A files at most 5 Jira tickets (initial run may file up to 5; subsequent ticks file only NEW findings due to fingerprint dedup).
- Phase B is a no-op on the first tick (nothing is `Approved` yet).
- Session does not crash.

- [ ] **Step 4: User can now triage tickets in Jira**

The user transitions one ticket to `Approved` over the next ~24h. The next tick (24h later) should pick it up and execute Phase B per spec.

- [ ] **Step 5: No commit. This task is operational, not code.**

---

## Self-review (engineer, before requesting code review)

Run through this checklist after all 12 tasks pass:

- [ ] Spec coverage: every section in `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md` is implemented in either the playbook (Phase A details) or the spec itself (Phase B is in the spec, referenced from the playbook).
- [ ] No placeholders: `grep -rn "TBD\|TODO\|FIXME\|(filled in by" docs/audit-playbook.md` returns nothing.
- [ ] No dangling audit imports: `grep -rn "v2.audit\|v2.audit_jira\|audit_findings\|audit_runs\|audit_llm_calls" --include="*.py" --include="*.html" --include="*.yml" . | grep -v "\.claude/worktrees/\|docs/superpowers/" ` returns nothing.
- [ ] No dangling JIRA env var refs in code: `grep -rn "JIRA_EMAIL\|JIRA_API_TOKEN\|JIRA_AUDIT" --include="*.py" --include="*.yml" .` returns nothing (mentions in `docs/superpowers/` historical specs are OK).
- [ ] `JIRA_BASE_URL` — verify whether anything outside the deleted dashboard audit pages used it. Run `grep -rn "JIRA_BASE_URL" --include="*.py" --include="*.html" .` after Tasks 5-7. If nothing remains, mention in the final commit that this env var can be removed from `.env`.
- [ ] Full test suite: `python3 -m pytest tests/ -q` PASSES with reduced count (deleted tests are gone, no new failures).
- [ ] Dashboard boots and core pages (`/`, `/decisions`, `/theses`) return 200.
- [ ] Jira workflow setup done by user.
- [ ] Rehearsal passed.
- [ ] /loop running in tmux.
