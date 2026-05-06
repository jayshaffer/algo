# Audit Finding #5 — Cleanup of test pollution in prod DB

**Date:** 2026-05-06
**Status:** Draft
**Type:** One-time data cleanup (no code change)

## Problem

The self-healing audit raised `STAGE_FAILURE_RATE` as **critical** for the
`trade_posts` stage (2 failed of 3 runs = 67%). On inspection the two
"failures" are not real prod failures:

- Both failed `session_stages` rows started at `2026-05-06 21:47:59`
  (about 30 seconds apart): ids `3420` and `3427`.
- Both reference the oldest sessions in the DB — sessions `1` (Mar 23) and
  `2` (Mar 24).
- The error column on both contains the test-guard message from
  `v2/twitter.py:225` ("`get_twitter_client()` must be mocked in tests…").
- Sessions 1 and 2 themselves have `completed_at = 2026-05-06 21:47:59` —
  i.e. their original `completed_at` was overwritten by the same
  test run.

Diagnosis: a test executed today connected to the **prod** database
(instead of a test fixture or paper DB) and wrote real `session_stages`
rows, which then tripped a real audit check.

The actual prod `trade_posts` history is small but clean: one
`completed` run on May 6 at 19:06, and one `running` row stuck since
May 5 (already separately flagged by `STAGE_RUNNING_STALE`).

## Goal

Remove the test-pollution rows so `STAGE_FAILURE_RATE` reflects real
prod behavior, and confirm no other tables were touched by the same
test run. Mark audit finding #5 resolved.

This spec deliberately scopes to **prod-DB cleanup**. Identifying the
offending test and preventing recurrence is a **separate follow-up**
spec — the immediate goal is to stop a misleading critical finding
from sitting open.

## Approach

A short, sequential plan executed against the prod DB
(`docker compose exec -T db psql -U algo -d trading`):

### 1. Sweep all tables for May-6-21:47 test artifacts

The test pollution timestamp is tightly clustered around
`2026-05-06 21:47:59`. Run a targeted sweep against every table that a
session run could plausibly write to, looking for any row whose
created/updated/started/completed timestamp falls in a 5-second window
centered on `21:47:59` and whose foreign-key target is in
`session_id ∈ (1,2)` or otherwise looks anomalous (e.g. very old
`session_date` paired with very recent write time).

Tables to sweep (one query each, recording counts):

- `session_stages` (known: ids 3420, 3427)
- `sessions` — confirm the two `completed_at` overwrites; check no
  new `sessions` row was inserted
- `decisions` — check for `created_at` (if column exists) or row
  inserts referencing session_id 1/2 indirectly via playbook
- `playbooks` / `playbook_actions` — same check
- `theses` / `thesis_signals` — check for `created_at` /
  `updated_at` near the window
- `tweets` — likely empty since the test errored before posting, but
  check
- `decision_signals` — check for inserts in the window

Each query is read-only. Output is a per-table count + sample of
matching ids. The result of the sweep informs which tables need
delete actions in step 2.

### 2. Apply deletes

Based on the sweep results:

- **Definitely delete:** `session_stages` rows `3420` and `3427`.
- **For each other table that returned matches:** decide row-by-row
  whether the row is test pollution (matches the
  test-pollution signature: write timestamp inside the 5s window AND
  references session 1/2 OR has a stale `session_date`/`created_at`
  inconsistency). Delete only the clear test artifacts; flag any
  ambiguous cases for the user before deleting.

All deletes go through a **single transaction** with a
`SAVEPOINT` per table, so a surprise (e.g. a foreign-key violation)
doesn't leave the DB in a half-cleaned state.

### 3. Resolve audit finding #5

Run the audit again (`task audit`) and confirm `STAGE_FAILURE_RATE`
no longer appears on the next run. The existing fingerprint logic in
`v2/audit.py` should auto-supersede the open finding when the
follow-up audit run no longer raises it. If for any reason it stays
open, mark it resolved manually:

```sql
UPDATE audit_findings
   SET status='resolved',
       resolved_at=now(),
       resolved_note='Test pollution cleanup; see spec 2026-05-06-audit-finding-5-cleanup-design.md'
 WHERE id=5;
```

## Out of scope

- Identifying the test that wrote to prod (separate follow-up).
- Adding code-level guardrails to prevent test→prod connections
  (separate follow-up).
- Changes to the audit check itself (the check was correct; the
  data was wrong).
- Restoring the original `completed_at` values on sessions 1 and 2.
  Those are real failed sessions from March; the overwritten
  `completed_at` is cosmetic and not used by any audit check.

## Risks

- **Wrong rows deleted.** Mitigated by: targeted timestamp window
  (5 seconds), required match against test-pollution signature, and
  per-table savepoints. If unsure, ask before deleting.
- **More test artifacts than expected.** If the sweep returns a wide
  spread (dozens of rows across many tables), pause and re-evaluate
  — that suggests broader test→prod connectivity, which warrants
  the prevention follow-up before further cleanup.

## Verification

- `task audit` after cleanup: no `STAGE_FAILURE_RATE` finding for
  `trade_posts`.
- `SELECT * FROM session_stages WHERE id IN (3420, 3427)` returns 0
  rows.
- The sweep queries from step 1, re-run, return 0 matches.
