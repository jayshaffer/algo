# Audit Loop + Atlassian MCP

Date: 2026-05-12
Status: Draft

## Summary

Replace the current `v2/audit.py` + `v2/audit_jira.py` pipeline with a
Claude Code `/loop` that owns the entire audit process end-to-end:
discovers findings, files them as Jira tickets via the Atlassian MCP,
and — once a human approves them in Jira — carries out the work
(opening a code PR or applying a reviewed DB mutation).

The audit playbook lives in `docs/audit-playbook.md` so editing the
audit is a documentation change, not a code change.

## Motivation

The current audit has three problems:

1. **Two parallel systems for Jira.** `v2/audit_jira.py` uses
   `JIRA_API_TOKEN` to file tickets via REST; Claude Code has an
   Atlassian MCP server already wired up. Maintaining both is waste.
2. **Discovery and execution are separated.** Today the audit files
   findings; humans (or none) act on them. There is no integrated path
   from "audit identified this" to "fix landed."
3. **The `--apply` auto-fix path is risky.** It runs database mutations
   with no human gate. We want approval before any mutation.

The new shape: Claude finds, the human curates in Jira, Claude
implements what the human approves.

## Non-goals

- No webhook from GitHub PR-merge to Jira-Done. Humans transition Jira
  to Done after merging the PR. Adding a webhook is a follow-up if the
  manual transition becomes friction.
- No retry-on-failure. A ticket stuck in `In Progress` requires manual
  intervention to retry or abandon.
- No auto-merge of PRs. Code changes always wait on human merge.
- No bundling. One ticket maps to one PR or one DB mutation.

## Scheduling model

The loop runs as `/loop 24h` in a long-lived local Claude Code session.
The session must stay alive; if it dies (terminal closed, machine
sleep, reboot), the user restarts it manually with the same command.
This is a known constraint. Host-cron + headless `claude -p` is a
future option if local /loop becomes operationally annoying.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Long-lived Claude Code session                              │
│                                                              │
│   /loop 24h "Read docs/audit-playbook.md and execute one     │
│             audit tick exactly as specified there."          │
│                                                              │
│   Per tick (two phases):                                     │
│     Phase A — Discovery: run checks, file new findings       │
│     Phase B — Execution: work approved tickets                │
│                                                              │
│   Tools used:                                                │
│     - Bash (docker exec psql, gh, git)                       │
│     - Atlassian MCP (searchJiraIssuesUsingJql,                │
│        createJiraIssue, transitionJiraIssue,                  │
│        addCommentToJiraIssue, editJiraIssue)                  │
│     - Read (codebase + playbook)                              │
└───────────────────────────────────────────────────────────┘
```

## New artifacts

- `docs/audit-playbook.md` — the source of truth for *what* gets
  audited. Lists every deterministic check (check_code, SQL, severity,
  worktype, suggested fix) and the ideation instructions.
- `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md` — this
  spec.

## Deletions

- `v2/audit.py` (2,159 lines)
- `v2/audit_jira.py` (159 lines)
- `tests/v2/test_audit.py`
- `tests/v2/test_audit_jira.py`
- `tests/v2/test_audit_dashboard.py`
- Dashboard audit pages (`/audit/`, finding detail pages) and their
  templates. Specific file list deferred to the implementation plan.
- DB migration: drop `audit_findings`, `audit_llm_calls`, and `audit_runs` (the parent table; `audit_llm_calls` FK-references it). Also drop `db/init/025_audit_findings.sql` so fresh DBs don't recreate them.
- Env vars in `.env` / `.env.paper` and docs (`CLAUDE.md`):
  `ALGO_AUDIT_FILE_JIRA`, `ALGO_AUDIT_JIRA_MAX_CREATES`,
  `ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS`, `JIRA_EMAIL`, `JIRA_API_TOKEN`,
  `JIRA_AUDIT_PROJECT_KEY`, `JIRA_AUDIT_ISSUE_TYPE`.
- Taskfile `audit:*` targets. Specific list deferred to plan.
- `JIRA_BASE_URL`: keep IF anything outside the audit dashboard pages
  still uses it. To be confirmed during the implementation plan.

## Phase A: Discovery

Each tick:

1. Read `docs/audit-playbook.md`.
2. Run every deterministic check in the playbook (SQL via
   `docker compose exec db-paper psql` for paper or `db psql` for
   prod — chosen by the playbook's environment instructions).
3. Run the ideation pass: read sources listed in the playbook (recent
   commits, session memos, dashboard, DB state). Identify patterns.
4. For each finding (deterministic or ideation):
   - Compute `fingerprint = sha256(check_code + ":" + topic_slug)`.
     `topic_slug` is kebab-case, derived per playbook convention.
   - Dedup search:
     ```
     JQL: project = ALGO
          AND labels = "audit-fingerprint:<fingerprint>"
     ```
     **No `statusCategory` filter** — we match closed tickets too, so
     that "Won't Fix" and "Done" tickets suppress future re-filings.
     This is the fix to the bug we caught during review of the
     lifecycle.
   - If a ticket exists at any status, skip.
   - Otherwise call MCP `createJiraIssue` with:
     - Summary: `[audit:<category>] <Title>`
     - Description (ADF): the analysis body, an evidence blockquote,
       a footer line with the fingerprint
     - Labels: `audit-source:claude`,
       `audit-fingerprint:<fingerprint>`,
       `audit-category:<category>`,
       `audit-worktype:<code|db>` (set by the playbook for each check)
     - Priority: per playbook (high/medium/low)
5. Cap new creations at 5 per tick. Remaining findings wait until the
   next tick.

## Phase B: Execution

Single JQL pulls everything actionable:

```
project = ALGO
AND labels = "audit-source:claude"
AND status in ("Approved", "Apply")
```

Action depends on `(status, audit-worktype)`:

| Status | Worktype | Claude's action |
|---|---|---|
| Approved | code | Transition → In Progress. Create branch `audit/<ISSUE_KEY>`. Make change. Run `task test`. Push. `gh pr create` linking the Jira issue key in the PR body. Transition → In Review. Comment on Jira with PR URL. (Human merges, then transitions → Done manually.) |
| Approved | db | Transition → In Progress. Generate the SQL fix from the playbook's suggested fix + the current finding evidence. Run `SELECT count(*)` against the WHERE clause + `EXPLAIN` of the mutation. Post a dry-run preview comment with both the SQL and the row count. Transition → Pending Apply. (Human reviews comment, transitions → Apply.) |
| Apply | db | Transition → In Progress. **Re-read the most recent comment on the ticket authored by the Atlassian-MCP account whose body begins with the marker `### Dry-run preview`, parse the SQL out of its fenced code block, and execute that exact SQL** — do NOT regenerate. This prevents drift between approved-SQL and applied-SQL. If the human wants different SQL, they transition back to `Approved` (see Failure modes). Capture actual rows affected. Comment with the result. Transition → Done. |

Per-tick execution cap: 2 tickets. Remaining tickets wait for the
next tick. The cap protects context budget; the loop is daily, so
even 2/day clears modest queues quickly.

## Concurrency

The `In Progress` transition is the lock. The Phase B JQL excludes
`In Progress`, so a second concurrent tick (should not happen with
/loop, but defense-in-depth) cannot pick up the same ticket.

## Failure modes

- **Tick crashes mid-execution.** Ticket left in `In Progress`. No
  automatic retry. Human transitions back to `Approved` (or `Apply`
  for DB tickets that crashed at the apply step) to retry.
- **PR review takes a long time.** Ticket sits in `In Review`
  indefinitely. No action needed from Claude; future ticks ignore
  `In Review`.
- **You want to abandon an approved ticket.** Transition directly to
  `Done` and add a `won't-fix` label (or whatever your project's
  convention is). The fingerprint label on the closed ticket
  suppresses future re-filing regardless.
- **You want different SQL than Claude generated.** Transition
  `Pending Apply` → `Approved`, add a comment with guidance.
  Claude regenerates on the next tick.

## Required Jira workflow setup (manual, one-time)

The ALGO Jira project must support these statuses:

- `To Do` (default)
- `Approved` — your "go" signal
- `In Progress` — Claude's lock
- `In Review` — code PRs awaiting human merge
- `Pending Apply` — DB dry-run posted, awaiting your second approval
- `Apply` — your "execute the SQL" signal
- `Done`

Transitions Claude needs to perform:
- `Approved → In Progress`
- `In Progress → In Review` (code path)
- `In Progress → Pending Apply` (db path)
- `In Progress → Done` (after apply)
- `Apply → In Progress`

Transitions the human performs:
- `To Do → Approved` (curation)
- `To Do → Done` / `Won't Fix` (rejection)
- `Pending Apply → Apply` (second-gate approval for DB)
- `Pending Apply → Approved` (request SQL revisions)
- `In Review → Done` (after PR merge)

If editing the workflow is gnarly, the fallback simplification is to
use labels in place of statuses (`claude-approved`, `claude-apply`)
and lose the board-level visibility of where each ticket is. The
plan should call out which Jira admin permission you need before
starting implementation.

## Playbook structure (`docs/audit-playbook.md`)

```markdown
# Audit Playbook

## How to read this file
(Instructions for Claude: when invoked from the /loop, execute every
check below, then run the ideation pass, then file findings per the
rules in the "Filing" section.)

## Environment
- Use `docker compose exec db-paper psql` for paper-pipeline checks.
- Use `docker compose exec db psql` for prod-pipeline checks.
- Each check declares which env it targets.

## Deterministic checks

### orphan_news_signal_fk
- env: prod, paper
- severity: critical
- category: integrity
- worktype: db
- topic_slug: orphan-news-signal-fk
- sql:
  SELECT count(*) AS n,
         array_agg(ds.id ORDER BY ds.id) FILTER (WHERE ds.id IS NOT NULL) AS sample_ids
  FROM decision_signals ds
  LEFT JOIN news_signals ns ON ds.signal_type='news_signal' AND ds.signal_id=ns.id
  WHERE ds.signal_type='news_signal' AND ds.signal_id IS NOT NULL AND ns.id IS NULL;
- finding_when: n > 0
- title_template: "Orphan FKs in decision_signals (news_signal)"
- body_template: "{n} rows in decision_signals reference news_signals.id values that do not exist. Sample IDs: {sample_ids[:20]}."
- suggested_fix_sql:
  UPDATE decision_signals SET signal_id = NULL
  WHERE signal_type='news_signal'
    AND signal_id IS NOT NULL
    AND signal_id NOT IN (SELECT id FROM news_signals);

### (next check)
...

## Ideation pass

Read these sources and look for patterns:
- Last 5 rows of strategy_memos (DB)
- Last 14 days of decisions (DB)
- Latest dashboard pages: /mistakes, /attribution
- `git log --since="14 days ago" --oneline`

Emit findings with category one of: audit_gap | app_improvement.
Topic slugs: kebab-case, derived from the finding's core noun (e.g.,
"rule-27-flip-flop", "executor-token-budget").

## Filing rules
(consolidated from Phase A of the spec)
```

The exact set of deterministic checks to port from today's `v2/audit.py`
is enumerated in the implementation plan, not here. The plan walks
through each check function in `v2/audit.py`, decides whether it
survives (most will), and writes the playbook entry.

## Security / safety considerations

- **Atlassian MCP credentials.** Already configured for this Claude
  Code installation. No new secrets to introduce; the `JIRA_*` REST
  secrets get deleted.
- **DB mutations.** Two human gates: `Approved` to draft, `Apply` to
  execute. Claude re-runs the previously-approved SQL verbatim from
  the Jira comment — no regeneration on apply.
- **Prod vs paper DB.** Each check declares its target env. The
  playbook's filing rules must include the env in the Jira ticket
  title so you can tell at a glance which DB an Apply will hit.
- **PR target branch.** PRs always target `main`. Never push directly
  to `main`. Always opens a feature branch `audit/<KEY>`.

## Testing

This change deletes most of the audit-related test files. The new
audit lives in a markdown playbook + a Claude prompt; it is not
unit-testable in the conventional sense. The implementation plan
must include:

1. A dry-run rehearsal of one full discovery cycle (no MCP create
   — just produce the JSON of what would be filed) for review
   before going live.
2. A staged rollout: first tick run manually under user observation,
   not unattended.
3. A "kill switch" — the user can remove labels from tickets or
   stop the /loop session at any time.

## Open questions

- Does `JIRA_BASE_URL` survive the deletion sweep, or is everything
  that uses it inside the dashboard's audit pages? To be confirmed
  during plan-writing.
- Does the existing Atlassian MCP have a `searchJiraIssuesUsingJql`
  variant that returns the full label set on hit issues? (Needed for
  dedup to see existing fingerprint labels.) To be confirmed.

## Implementation order

1. Author `docs/audit-playbook.md` (port current checks).
2. Configure the Jira workflow statuses (manual; you do this).
3. Dry-run a discovery tick by hand in Claude Code, observe output.
4. Delete `v2/audit.py`, `v2/audit_jira.py`, their tests, dashboard
   pages, env vars, Taskfile targets.
5. Write the migration that drops `audit_findings` and
   `audit_llm_calls`.
6. Start the long-lived /loop session.
