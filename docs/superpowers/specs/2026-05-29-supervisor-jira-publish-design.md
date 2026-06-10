# Supervisor App-Findings → Jira (direct publish)

**Status:** NOT IMPLEMENTED — design only (verified against code 2026-06-10).
Nothing below exists in `v2/`: there is no `record_app_finding` tool, no
`v2/jira_client.py`, and no `publish_findings()` in `v2/supervisor.py`. Do
not configure `JIRA_*` env vars or expect supervisor-filed tickets until
this spec is built.
**Date:** 2026-05-29
**Related:**
- `docs/superpowers/specs/2026-05-27-strategy-supervisor-design.md` (supervisor)
- `docs/superpowers/specs/2026-05-27-supervisor-watchlist-closed-loop-design.md` (watchlist)
- `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md` (Jira ticket conventions, Phase B execution)

## Problem

The strategy supervisor (`v2/supervisor.py`) produces two outputs today:

1. A free-form markdown memo (`supervisor_memos.content`) — read only by the
   legacy v1 Flask dashboard; **not consumed by any v2 stage**.
2. Structured watchlist items (`supervisor_watchlist_items`) — ingested by
   reflection/ideation under a same-session hard gate. This is a closed loop
   that mutates **strategy state** (rules, theses, identity).

Two classes of supervisor finding have no actuator:

- **Focus area #3 (executor behavior drift)** and **#4 (reflection quality)**
  often require a *code* change (executor sizing logic, prompt hardening), but
  `owner_stage` only routes to `reflection`/`ideation`, neither of which edits
  code. The executor (`trader.py`) ingests no watchlist items at all.
- **Structural findings** (schema/integrity issues, dead-end plumbing) likewise
  need engineering work, not a strategy-state tweak.

These insights currently evaporate into prose nobody acts on.

## Goal

Give app/code/DB-level supervisor findings a real actuator by **filing them as
Jira tickets** that the existing audit Phase B (`/audit-execute`) works
(branch → PR for code, dry-run → apply for DB). Strategy-state findings keep
flowing through the existing watchlist closed loop, unchanged.

## Non-goals

- No change to the watchlist closed loop or to reflection/ideation.
- No new ticket-execution machinery — reuse audit Phase B.
- No drainer command, no queue table. The supervisor publishes directly.

## Key decision: direct publish, not a drainer

The supervisor runs **in-container, via the Anthropic API**, as Stage 0.5 of
the daily session. It has no access to the Atlassian MCP (that lives only in
Claude Code sessions, e.g. the audit `/loop`).

Filing a *structured* finding is a deterministic HTTP POST — no agent judgment
is needed at file time (the supervisor already made the judgment). So the
supervisor publishes findings **directly to the Jira Cloud REST API** at the
end of its stage, the same way `dashboard_publish.py` publishes the dashboard.
No queue, no separate drainer command, no cadence coupling.

**Trade accepted:** this reintroduces a Jira REST credential in `.env` — the
`JIRA_*` secrets the audit-loop migration deliberately removed in favor of the
MCP. The credential is scoped to create + search on the `ALGO` project. For a
single deterministic POST path this is a clean trade; the user approved it.

## Architecture

```
supervisor (in-container, API, read-only tools)
  ├─ markdown memo          → supervisor_memos              (audit trail, dashboard)   [unchanged]
  ├─ record_watchlist_item  → supervisor_watchlist_items    → reflection/ideation       [unchanged]
  └─ record_app_finding     → (in-memory buffer)
                               → publish_findings() at end of Stage 0.5
                                 → Jira REST (create + JQL dedup)
                                   → ALGO board → Phase B (/audit-execute) works it
```

### Components

**1. `record_app_finding` tool (`v2/supervisor.py`)**

Tool def parallel to `RECORD_WATCHLIST_TOOL_DEF`. Params:

| param | type | notes |
|---|---|---|
| `title` | string | short label → ticket summary |
| `detail` | string | what's wrong / what to do → ticket body |
| `category` | string | free-form bucket, e.g. `executor_behavior`, `strategy_quality`, `integrity` → `audit-category:<category>` |
| `worktype` | enum `code`\|`db` | → `audit-worktype:<…>`; drives which Phase B path works it |
| `severity` | enum `info`\|`warn`\|`critical` | → priority (Low/Medium/High) |
| `topic_slug` | string | kebab-case, **stable across runs** — the dedup key |
| `suggested_fix` | string | concrete remediation → ticket footer |

Handler buffers into an in-memory list (same pattern as the watchlist handler);
validates `worktype`/`severity` enums and returns an error string on bad input.
Findings are NOT persisted to a table — Jira is the system of record.

**2. System-prompt routing (`STRATEGY_SUPERVISOR_SYSTEM`)**

Add an explicit boundary so the model routes correctly:

- `record_watchlist_item` → **only** strategy-state changes an acting stage can
  make this session: retire/amend/revalidate a rule, fix a stale/contradictory
  thesis, identity drift.
- `record_app_finding` → anything requiring a **code, config, or DB** change:
  executor-behavior fixes, prompt hardening, schema/data-integrity problems,
  structural dead-ends.

The markdown memo gains an "Engineering Findings" section that must match the
`record_app_finding` calls (mirrors the existing Watchlist match rule).

**3. `v2/jira_client.py` (new, thin)**

Factory `get_jira_client()` mirroring the other client factories. Reads:
`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY` (default
`ALGO`). HTTP Basic auth (email + API token, Jira Cloud standard). Methods:

- `fingerprint_exists(fingerprint) -> bool` — JQL
  `project=<KEY> AND labels="audit-fingerprint:<fp>"` (no status filter, so
  closed Done/Won't-Fix tickets also suppress re-filing — matches audit dedup).
- `create_issue(summary, description_adf, labels, priority, issue_type="Task") -> str`
  — returns the new issue key.

Uses `requests`. All network calls go through this module so tests can patch one
target.

**4. `publish_findings(findings, source_memo_id, dry_run) -> PublishResult`
(`v2/supervisor.py`)**

For each buffered finding:

- `fingerprint = sha256(f"supervisor:{topic_slug}").hexdigest()[:16]`
- `fingerprint_exists(fp)` → skip + log if hit (idempotent across retries and
  recurring weekly findings).
- else `create_issue` with:
  - summary `[supervisor:<category>] <title>`
  - issue type `Task`
  - labels: `audit-source:claude`, `audit-source:supervisor` (provenance),
    `audit-fingerprint:<fp>`, `audit-category:<category>`,
    `audit-worktype:<code|db>`
  - priority: `critical`→High, `warn`→Medium, `info`→Low
  - ADF description: detail paragraph + evidence blockquote (the supervisor's
    cited IDs) + horizontal rule + footer (`Filed by supervisor on <date>.
    Fingerprint: <fp>. Source memo: <id>. Suggested fix: <…>`)

Safety:
- **Cap** `ALGO_SUPERVISOR_JIRA_MAX_CREATES` (default 8) per run.
- Each create wrapped in try/except — log and continue (Stage 0.5 is
  independent; a Jira failure must not break the session). Return counts
  (`created`, `deduped`, `errored`).
- `dry_run=True` → print would-be ticket payloads, POST nothing (the supervisor
  already has a `--dry-run` path; this hooks into it, footer reads
  `memo: dry-run`).

**5. Wiring in `run_supervisor`**

After the memo INSERT and watchlist persistence, if
`ALGO_SUPERVISOR_FILE_JIRA` is enabled, call `publish_findings(buffer,
memo_id, dry_run)`. On the existing dry-run branch, print the buffer instead.

## Configuration

New env vars (`.env`, `.env.paper`, documented in `CLAUDE.md`):

- `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY` (default `ALGO`)
- `ALGO_SUPERVISOR_FILE_JIRA` — master on/off, **default off**. Enable in prod
  `.env` only; paper leaves it off (paper is experimental, shouldn't spam the
  board).
- `ALGO_SUPERVISOR_JIRA_MAX_CREATES` — per-run cap, default `8`.

## Execution (reused, not built)

Filed tickets carry `audit-source:claude`, so the existing audit Phase B
(`/audit-execute`) JQL picks them up once a human transitions them to
`Approved`/`Apply`, and works them via the existing code/db paths. The
`audit-source:supervisor` label distinguishes provenance for reporting. No new
execution code.

## Testing

- `record_app_finding` handler: buffers valid input; rejects bad
  `worktype`/`severity` with an error string.
- `jira_client`: mock `requests`; assert auth header, JQL dedup query shape,
  and create payload (labels, priority mapping, ADF structure).
- `publish_findings`: mock `jira_client`; assert fingerprint computation,
  dedup-skip path, label/priority/summary rendering, per-finding failure
  isolation, cap enforcement, and dry-run posts nothing.
- **Test isolation (critical):** add the Jira publish path to
  `tests/v2/conftest.py::_NETWORK_PATCH_TARGETS`. Per the 2026-05-28 incident,
  any new outbound network call reachable from `run_session` MUST be patched or
  `test_session` makes real calls — now real Jira writes, not just token burn.
- Prompt routing is LLM behavior — not unit-tested beyond asserting the routing
  instructions are present in `STRATEGY_SUPERVISOR_SYSTEM`.

## Rollout

1. Land code with `ALGO_SUPERVISOR_FILE_JIRA` off everywhere.
2. Run `python -m v2.supervisor --dry-run` and inspect the printed would-be
   tickets against a recent week's findings.
3. Enable in prod `.env`; observe the first live run's filed tickets before
   leaving it unattended.
4. Kill switch: set `ALGO_SUPERVISOR_FILE_JIRA=false` (or revoke the token).

## Open questions

- Does the project's Jira instance use a custom issue type / required fields on
  `ALGO` that `create_issue` must populate? Confirm against
  `getJiraProjectIssueTypesMetadata` during implementation.
- Should dedup also match on a coarser key than `topic_slug` to avoid
  near-duplicate tickets when the supervisor rephrases the same issue? Start
  with exact `topic_slug`; revisit if churn appears.
