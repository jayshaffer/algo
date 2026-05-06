# Audit Finding #5 Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove test-pollution rows from the prod database so the audit's `STAGE_FAILURE_RATE` critical finding stops firing, and confirm no other tables were touched by the same test run.

**Architecture:** This is a one-time data cleanup runbook executed against the prod Postgres container. No code changes. All work happens in `psql` against the `trading` database inside `docker compose exec -T db`. Sweep queries are read-only; deletes go through a single transaction with savepoints.

**Tech Stack:** PostgreSQL 16 (in `algo-db-1` container), `docker compose`, `psql -U algo -d trading`.

**Spec:** `docs/superpowers/specs/2026-05-06-audit-finding-5-cleanup-design.md`

---

## Background

The test pollution timestamp is tightly clustered around `2026-05-06 21:47:59`. Two known polluted rows:
- `session_stages.id = 3420` (started `2026-05-06 21:47:59.875349+00`, references session 1)
- `session_stages.id = 3427` (started `2026-05-06 21:47:59.56248+00`, references session 2)

Both contain the test-guard error from `v2/twitter.py:225`. Sessions 1 and 2 also had their `completed_at` overwritten to the same timestamp.

Tables that a session run can write to and have a relevant timestamp column for the sweep:

| Table              | Timestamp column(s)            |
|--------------------|--------------------------------|
| `session_stages`   | `started_at`, `completed_at`   |
| `sessions`         | `started_at`, `completed_at`   |
| `theses`           | `created_at`, `updated_at`     |
| `playbooks`        | `created_at`                   |
| `playbook_actions` | `created_at`                   |
| `tweets`           | `created_at`                   |

`decisions` and `decision_signals` have no timestamp column for the sweep — for those we'll inspect via FK linkage instead.

The sweep window is `[2026-05-06 21:47:55+00, 2026-05-06 21:48:05+00]` — 10 seconds wide centered on the known pollution time.

---

## Task 1: Sweep all timestamped tables for May-6-21:47 anomalies

**Files:** None modified. Read-only queries only.

- [ ] **Step 1: Run the sweep on each table**

Open a psql session:

```bash
docker compose exec -T db psql -U algo -d trading
```

Run each query and record the result (count + ids):

```sql
-- session_stages: known pollution at ids 3420, 3427
SELECT id, session_id, stage_name, status, started_at, completed_at
  FROM session_stages
 WHERE started_at  BETWEEN '2026-05-06 21:47:55+00' AND '2026-05-06 21:48:05+00'
    OR completed_at BETWEEN '2026-05-06 21:47:55+00' AND '2026-05-06 21:48:05+00'
 ORDER BY started_at;

-- sessions: confirm sessions 1, 2 completed_at overwrite; check for new inserts
SELECT id, session_date, status, started_at, completed_at
  FROM sessions
 WHERE started_at  BETWEEN '2026-05-06 21:47:55+00' AND '2026-05-06 21:48:05+00'
    OR completed_at BETWEEN '2026-05-06 21:47:55+00' AND '2026-05-06 21:48:05+00'
 ORDER BY id;

-- theses
SELECT id, ticker, source, status, created_at, updated_at
  FROM theses
 WHERE created_at BETWEEN '2026-05-06 21:47:55' AND '2026-05-06 21:48:05'
    OR updated_at BETWEEN '2026-05-06 21:47:55' AND '2026-05-06 21:48:05'
 ORDER BY id;

-- playbooks
SELECT id, date, created_at
  FROM playbooks
 WHERE created_at BETWEEN '2026-05-06 21:47:55+00' AND '2026-05-06 21:48:05+00'
 ORDER BY id;

-- playbook_actions
SELECT id, playbook_id, ticker, action, status, created_at
  FROM playbook_actions
 WHERE created_at BETWEEN '2026-05-06 21:47:55+00' AND '2026-05-06 21:48:05+00'
 ORDER BY id;

-- tweets
SELECT id, decision_id, platform, posted_at, created_at
  FROM tweets
 WHERE created_at BETWEEN '2026-05-06 21:47:55+00' AND '2026-05-06 21:48:05+00'
 ORDER BY id;
```

Expected baseline result (only the known pollution should appear):
- `session_stages`: 2 rows (3420, 3427).
- `sessions`: 2 rows (id 1, 2 with `completed_at` in window).
- All other tables: **0 rows**.

- [ ] **Step 2: Inspect FK-linked rows for tables without a timestamp**

`decisions` and `decision_signals` lack a written timestamp. Check whether the test pollution created any rows linked to the polluted `session_id`s by examining what the test would have to insert to reach the `trade_posts` stage. Since `trade_posts` is stage 5 and inserts to `tweets` only on success (and the test errored before posting), the most likely linkage is none — but verify:

```sql
-- decisions on the dates of sessions 1 and 2 (Mar 23, Mar 24)
SELECT id, date, ticker, action, playbook_action_id, is_off_playbook
  FROM decisions
 WHERE date IN ('2026-03-23','2026-03-24')
 ORDER BY date, id;

-- recent decision_signals — no direct way to see "added today",
-- so cross-check via the parent decision date
SELECT ds.decision_id, ds.signal_type, ds.signal_id, d.date
  FROM decision_signals ds
  JOIN decisions d ON d.id = ds.decision_id
 WHERE d.date IN ('2026-03-23','2026-03-24')
 ORDER BY ds.decision_id;
```

Compare counts to the user's known March history. If counts look normal (i.e. decisions for those dates exist from the original prod runs and no surprise extras), `decisions`/`decision_signals` were not polluted.

- [ ] **Step 3: Record the sweep result for the user**

Print a short summary (do not delete anything yet):

```
Sweep summary:
  session_stages : 2 polluted rows  → ids: [3420, 3427]
  sessions       : 2 overwritten    → ids: [1, 2]  (completed_at only)
  theses         : <N>  → ids: [...]
  playbooks      : <N>  → ids: [...]
  playbook_actions: <N> → ids: [...]
  tweets         : <N>  → ids: [...]
  decisions/decision_signals: appear normal | suspicious (explain)
```

- [ ] **Step 4: Decision gate**

If the sweep returns ONLY the expected 2 known rows on `session_stages` and the 2 `sessions` overwrites — proceed to Task 2.

If the sweep returns **anything else** (any rows in `theses`, `playbooks`, `playbook_actions`, `tweets`, or unexpected `session_stages` ids) — **STOP** and surface the rows to the user before deleting. The cleanup scope changes when more pollution is found, and the surprise warrants the prevention follow-up before further deletion.

- [ ] **Step 5: Commit nothing**

This task is read-only. No commit.

---

## Task 2: Delete the polluted `session_stages` rows

**Files:** None modified. SQL deletes inside a transaction.

- [ ] **Step 1: Open a transaction with a savepoint**

```sql
BEGIN;
SAVEPOINT before_stage_delete;
```

- [ ] **Step 2: Delete the two known polluted rows**

```sql
DELETE FROM session_stages WHERE id IN (3420, 3427);
```

Expected output: `DELETE 2`.

- [ ] **Step 3: Verify the deletes**

```sql
SELECT id FROM session_stages WHERE id IN (3420, 3427);
```

Expected: 0 rows.

```sql
SELECT id, session_id, stage_name, status, started_at
  FROM session_stages
 WHERE stage_name = 'trade_posts'
 ORDER BY started_at DESC;
```

Expected: 2 rows remain (the May 6 19:06 `completed` row, and the May 5 19:08 `running` row). No other rows.

- [ ] **Step 4: Commit the transaction**

```sql
COMMIT;
```

If anything in steps 2–3 looked wrong: `ROLLBACK TO SAVEPOINT before_stage_delete; ROLLBACK;` and surface to the user.

- [ ] **Step 5: Note — additional deletes only if Task 1 surfaced more rows**

If Task 1's decision gate routed any additional pollution candidates here, repeat steps 1–4 with one DELETE per table inside the same transaction — using a savepoint per table (`SAVEPOINT before_<table>_delete;`) so any per-table failure can be rolled back independently. **Only delete rows that match the test-pollution signature**: write timestamp inside the 10-second window AND linked to session 1/2 OR otherwise inconsistent with normal prod history. Anything ambiguous: surface to the user before deleting.

---

## Task 3: Re-run the audit and verify finding #5 is gone

**Files:** None modified.

- [ ] **Step 1: Run the audit in check mode against prod**

```bash
task audit
```

(Equivalent to `docker compose exec trading python -m v2.audit` — check the `Taskfile.yml` `audit` target if `task` is not on PATH.)

Expected: the audit completes; `STAGE_FAILURE_RATE` does NOT appear in the output's findings list.

- [ ] **Step 2: Confirm the audit superseded the open finding**

```sql
SELECT id, check_code, severity, status, resolved_at, resolved_note
  FROM audit_findings
 WHERE id = 5;
```

Expected: `status = 'superseded'` (the supersession logic in `v2/audit.py` should auto-flip it when the next run no longer raises the same fingerprint).

If `status` is still `open`, mark it resolved manually:

```sql
UPDATE audit_findings
   SET status = 'resolved',
       resolved_at = now(),
       resolved_note = 'Test pollution cleanup; see spec 2026-05-06-audit-finding-5-cleanup-design.md'
 WHERE id = 5;
```

- [ ] **Step 3: Sanity-check no new critical findings**

```sql
SELECT id, check_code, severity, title, created_at
  FROM audit_findings
 WHERE severity = 'critical' AND status = 'open'
 ORDER BY created_at DESC;
```

Expected: only finding #7 (`DECISIONS_NO_SIGNAL_REFS`) — the second of the two original criticals — is still open. The cleanup did not introduce new critical findings.

- [ ] **Step 4: Commit a short note for the audit trail**

This task only changes data, not files in the repo. Skip the git commit step.

If you want a permanent paper trail, append a short bullet to `v2/RETRO.md` (or the equivalent ops log in the repo) describing the cleanup. **Only add if the file already exists** — do not create a new doc.

```bash
# Optional, only if v2/RETRO.md exists:
git status v2/RETRO.md
git add v2/RETRO.md
git commit -m "$(cat <<'EOF'
ops: log cleanup of test-pollution rows in prod (audit finding #5)

Deleted session_stages 3420, 3427 — test-pollution rows from a
test run that connected to prod DB on 2026-05-06 21:47:59.
Audit finding #5 (STAGE_FAILURE_RATE) auto-superseded on next run.

See docs/superpowers/specs/2026-05-06-audit-finding-5-cleanup-design.md.
EOF
)"
```

---

## Done

After Task 3:
- `STAGE_FAILURE_RATE` no longer appears in the audit output.
- Finding #5 is `superseded` or `resolved` in `audit_findings`.
- Only one open critical finding remains (#7), to be addressed by `2026-05-06-audit-finding-7-signal-refs-check-refinement.md`.
