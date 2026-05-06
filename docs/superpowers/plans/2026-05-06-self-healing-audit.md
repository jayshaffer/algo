# Self-Healing Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily auditor (`v2/audit.py`) that detects metadata integrity issues, proposes rule overfitting/contradictions via a single Haiku call, and surfaces application-level regressions — never feeding the strategist.

**Architecture:** Flat module with 11 check functions returning a uniform `Finding` dataclass; runner orchestrates checks with per-check savepoints, fingerprint-based idempotency, and supersession of stale findings; default propose-only with opt-in Tier-1 auto-fix behind `--apply`; surfacing on the existing v1 dashboard; daily cron via Taskfile.

**Tech Stack:** Python 3.11, psycopg2 raw SQL (existing `get_cursor` pattern), Anthropic SDK (`claude_haiku_4_5_20251001`), Flask (v1 `dashboard/app.py`), pytest + MagicMock.

**Spec:** `docs/superpowers/specs/2026-05-06-self-healing-audit-design.md`

**Path corrections vs. spec:**
- Migrations live in `db/init/` (mounted as `docker-entrypoint-initdb.d`), not `db/migrations/`. New file: `db/init/025_audit_findings.sql`.
- Dashboard surface goes into the live v1 dashboard (`dashboard/app.py` on port 3000); the v2 dashboard's HTML routes are dormant per its own module docstring.

---

## File Inventory

**Create:**
- `db/init/025_audit_findings.sql` — schema migration
- `v2/audit.py` — flat module: `Finding` dataclass, 11 check functions, runner, CLI
- `dashboard/templates/audit.html` — Jinja template for `/audit` page
- `dashboard/templates/audit_finding.html` — Jinja template for `/audit/findings/<id>`
- `tests/v2/test_audit.py` — all unit + runner tests
- `tests/v2/test_audit_dashboard.py` — dashboard route tests
- `tests/test_migration_025.py` — migration apply test (follows existing pattern)

**Modify:**
- `v2/database/trading_db.py` — add audit helpers
- `dashboard/app.py` — register `/audit` routes + queries
- `dashboard/queries.py` — add audit-page DB queries
- `Taskfile.yml` — add `audit`, `audit:apply`, `paper:audit`, `paper:audit:apply`
- `README.md` — document cron entry and CLI usage

---

## Task 1: Schema migration

**Files:**
- Create: `db/init/025_audit_findings.sql`
- Test: `tests/test_migration_025.py`

- [ ] **Step 1: Write the migration**

```sql
-- db/init/025_audit_findings.sql
-- Self-healing audit: track integrity, rule-overfitting, and app-issue findings.
-- See docs/superpowers/specs/2026-05-06-self-healing-audit-design.md

CREATE TABLE IF NOT EXISTS audit_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    mode            VARCHAR(16) NOT NULL CHECK (mode IN ('check','apply')),
    total_findings  INTEGER NOT NULL DEFAULT 0,
    auto_fixed      INTEGER NOT NULL DEFAULT 0,
    failed_checks   INTEGER NOT NULL DEFAULT 0,
    model           VARCHAR(64),
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    cache_creation_tokens INTEGER,
    cache_read_tokens     INTEGER
);

CREATE TABLE IF NOT EXISTS audit_findings (
    id              SERIAL PRIMARY KEY,
    audit_run_id    INTEGER NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
    check_code      VARCHAR(64) NOT NULL,
    tier            SMALLINT NOT NULL CHECK (tier IN (1,2,3)),
    severity        VARCHAR(16) NOT NULL CHECK (severity IN ('critical','warn','info')),
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    affected_count  INTEGER NOT NULL DEFAULT 0,
    evidence        JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          VARCHAR(16) NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open','auto_fixed','acknowledged','resolved','superseded')),
    fingerprint     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ,
    resolved_note   TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_findings_status   ON audit_findings(status) WHERE status='open';
CREATE INDEX IF NOT EXISTS idx_audit_findings_run      ON audit_findings(audit_run_id);
CREATE INDEX IF NOT EXISTS idx_audit_findings_code     ON audit_findings(check_code);
CREATE UNIQUE INDEX IF NOT EXISTS uq_audit_findings_open_fingerprint
    ON audit_findings(fingerprint) WHERE status='open';
```

- [ ] **Step 2: Apply to running prod and paper DBs**

```bash
docker compose exec -T db psql -U algo -d trading < db/init/025_audit_findings.sql
docker compose -f docker-compose.yml -f docker-compose.paper.yml \
    exec -T db-paper psql -U algo -d trading < db/init/025_audit_findings.sql
```

Expected: `CREATE TABLE` × 2, `CREATE INDEX` × 4 (per DB).

- [ ] **Step 3: Verify schema applied**

```bash
docker compose exec -T db psql -U algo -d trading -c "\d audit_findings" | head -30
```

Expected output includes the unique partial index `uq_audit_findings_open_fingerprint`.

- [ ] **Step 4: Commit**

```bash
git add db/init/025_audit_findings.sql
git commit -m "feat(db): migration 025 — audit_runs + audit_findings tables"
```

---

## Task 2: trading_db audit helpers

**Files:**
- Modify: `v2/database/trading_db.py` (append a new section)
- Test: `tests/v2/test_audit.py` (new file, helpers section)

- [ ] **Step 1: Write the failing tests for the helpers**

```python
# tests/v2/test_audit.py
"""Tests for v2 self-healing audit module."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# --- DB helper tests (Task 2) ---

class TestAuditDbHelpers:
    @patch("v2.database.trading_db.get_cursor")
    def test_insert_audit_run_returns_id(self, mock_get_cursor):
        cur = MagicMock()
        cur.fetchone.return_value = {"id": 42}
        mock_get_cursor.return_value.__enter__.return_value = cur
        from v2.database.trading_db import insert_audit_run
        run_id = insert_audit_run(mode="check")
        assert run_id == 42
        sql = cur.execute.call_args[0][0]
        assert "audit_runs" in sql
        assert cur.execute.call_args[0][1] == ("check",)

    @patch("v2.database.trading_db.get_cursor")
    def test_insert_audit_finding_skips_when_open_fingerprint_exists(self, mock_get_cursor):
        cur = MagicMock()
        # ON CONFLICT DO NOTHING + RETURNING id → fetchone returns None on conflict
        cur.fetchone.return_value = None
        mock_get_cursor.return_value.__enter__.return_value = cur
        from v2.database.trading_db import insert_audit_finding
        result = insert_audit_finding(
            audit_run_id=1, check_code="X", tier=1, severity="warn",
            title="t", body="b", affected_count=0, evidence={}, fingerprint="abc",
        )
        assert result is None  # signals "already open"

    @patch("v2.database.trading_db.get_cursor")
    def test_insert_audit_finding_returns_id_on_insert(self, mock_get_cursor):
        cur = MagicMock()
        cur.fetchone.return_value = {"id": 7}
        mock_get_cursor.return_value.__enter__.return_value = cur
        from v2.database.trading_db import insert_audit_finding
        result = insert_audit_finding(
            audit_run_id=1, check_code="X", tier=1, severity="warn",
            title="t", body="b", affected_count=0, evidence={}, fingerprint="abc",
        )
        assert result == 7

    @patch("v2.database.trading_db.get_cursor")
    def test_supersede_stale_open_findings_marks_missing_fingerprints(self, mock_get_cursor):
        cur = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = cur
        from v2.database.trading_db import supersede_stale_open_findings
        supersede_stale_open_findings(run_id=99, current_fingerprints={"a", "b"})
        sql = cur.execute.call_args[0][0]
        assert "UPDATE audit_findings" in sql
        assert "superseded" in sql
        assert "fingerprint NOT IN" in sql

    @patch("v2.database.trading_db.get_cursor")
    def test_finalize_audit_run_writes_totals_and_usage(self, mock_get_cursor):
        cur = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = cur
        from v2.database.trading_db import finalize_audit_run
        finalize_audit_run(run_id=1, total_findings=5, auto_fixed=2,
                           failed_checks=0, model="claude-haiku-4-5-20251001",
                           input_tokens=1000, output_tokens=200,
                           cache_creation_tokens=0, cache_read_tokens=0)
        sql, params = cur.execute.call_args[0]
        assert "UPDATE audit_runs" in sql
        assert params[0] == 5  # total_findings

    @patch("v2.database.trading_db.get_cursor")
    def test_try_advisory_lock_returns_bool(self, mock_get_cursor):
        cur = MagicMock()
        cur.fetchone.return_value = {"pg_try_advisory_lock": True}
        mock_get_cursor.return_value.__enter__.return_value = cur
        from v2.database.trading_db import try_advisory_audit_lock
        assert try_advisory_audit_lock() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestAuditDbHelpers -v`
Expected: FAIL with `ImportError: cannot import name 'insert_audit_run' …`

- [ ] **Step 3: Implement the helpers**

Append to `v2/database/trading_db.py`:

```python
# --- Audit (self-healing) ---

# Stable advisory-lock key for the audit runner. Arbitrary 64-bit int; no
# collision possible with other Postgres advisory locks in this codebase
# because no other module uses pg_advisory_lock.
_AUDIT_ADVISORY_LOCK_KEY = 7341984512  # 0x1B58_8C800

def insert_audit_run(mode: str) -> int:
    """Create a new audit_runs row; returns the id."""
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO audit_runs (mode) VALUES (%s) RETURNING id",
            (mode,),
        )
        return cur.fetchone()["id"]


def insert_audit_finding(
    *, audit_run_id: int, check_code: str, tier: int, severity: str,
    title: str, body: str, affected_count: int, evidence: dict,
    fingerprint: str, status: str = "open", resolved_at=None,
    resolved_note=None,
) -> int | None:
    """Insert a finding row. Returns id, or None if an open finding with the
    same fingerprint already exists (idempotent re-emit)."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit_findings
                (audit_run_id, check_code, tier, severity, title, body,
                 affected_count, evidence, status, fingerprint,
                 resolved_at, resolved_note)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (fingerprint) WHERE status='open' DO NOTHING
            RETURNING id
            """,
            (audit_run_id, check_code, tier, severity, title, body,
             affected_count, Json(evidence), status, fingerprint,
             resolved_at, resolved_note),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def supersede_stale_open_findings(*, run_id: int, current_fingerprints: set[str]) -> int:
    """Mark all open findings whose fingerprint is not in current_fingerprints
    as superseded with a reference to this run id. Returns count updated."""
    with get_cursor() as cur:
        if current_fingerprints:
            cur.execute(
                """
                UPDATE audit_findings
                SET status='superseded',
                    resolved_at=now(),
                    resolved_note=%s
                WHERE status='open'
                  AND fingerprint NOT IN %s
                """,
                (f"not detected by run #{run_id}", tuple(current_fingerprints)),
            )
        else:
            cur.execute(
                """
                UPDATE audit_findings
                SET status='superseded',
                    resolved_at=now(),
                    resolved_note=%s
                WHERE status='open'
                """,
                (f"not detected by run #{run_id}",),
            )
        return cur.rowcount


def finalize_audit_run(*, run_id: int, total_findings: int, auto_fixed: int,
                       failed_checks: int, model: str | None = None,
                       input_tokens: int | None = None,
                       output_tokens: int | None = None,
                       cache_creation_tokens: int | None = None,
                       cache_read_tokens: int | None = None) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE audit_runs
            SET completed_at=now(), total_findings=%s, auto_fixed=%s,
                failed_checks=%s, model=%s, input_tokens=%s, output_tokens=%s,
                cache_creation_tokens=%s, cache_read_tokens=%s
            WHERE id=%s
            """,
            (total_findings, auto_fixed, failed_checks, model,
             input_tokens, output_tokens, cache_creation_tokens,
             cache_read_tokens, run_id),
        )


def try_advisory_audit_lock() -> bool:
    """Returns True if the lock was acquired; False if already held."""
    with get_cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_AUDIT_ADVISORY_LOCK_KEY,))
        return bool(cur.fetchone()["pg_try_advisory_lock"])


def release_advisory_audit_lock() -> None:
    with get_cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s)", (_AUDIT_ADVISORY_LOCK_KEY,))


def delete_orphan_decision_signals(signal_type: str, ids: list[int]) -> int:
    """Auto-fix helper: delete orphan decision_signals rows for a given
    signal_type and list of orphan signal_ids. Returns count deleted."""
    if not ids:
        return 0
    with get_cursor() as cur:
        cur.execute(
            "DELETE FROM decision_signals WHERE signal_type=%s AND signal_id = ANY(%s)",
            (signal_type, ids),
        )
        return cur.rowcount


def get_open_audit_findings():
    """Read open findings for the dashboard, ordered by severity then created_at."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, audit_run_id, check_code, tier, severity, title,
                   affected_count, created_at, evidence
            FROM audit_findings
            WHERE status='open'
            ORDER BY
              CASE severity WHEN 'critical' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END,
              tier, created_at DESC
        """)
        return cur.fetchall()


def get_audit_finding(finding_id: int):
    with get_cursor() as cur:
        cur.execute("SELECT * FROM audit_findings WHERE id=%s", (finding_id,))
        return cur.fetchone()


def get_recent_audit_runs(limit: int = 14):
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, started_at, completed_at, mode, total_findings,
                   auto_fixed, failed_checks, model,
                   input_tokens, output_tokens,
                   cache_creation_tokens, cache_read_tokens
            FROM audit_runs ORDER BY started_at DESC LIMIT %s
        """, (limit,))
        return cur.fetchall()


def update_audit_finding_status(finding_id: int, status: str, note: str | None) -> None:
    if status not in ("acknowledged", "resolved"):
        raise ValueError(f"manual status must be acknowledged or resolved, got {status!r}")
    with get_cursor() as cur:
        cur.execute(
            "UPDATE audit_findings SET status=%s, resolved_at=now(), resolved_note=%s WHERE id=%s",
            (status, note, finding_id),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestAuditDbHelpers -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_audit.py
git commit -m "feat(v2): audit DB helpers (insert_audit_run, insert_audit_finding, supersede, finalize, advisory lock)"
```

---

## Task 3: Finding dataclass + fingerprint

**Files:**
- Create: `v2/audit.py` (initial scaffold)
- Test: `tests/v2/test_audit.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestFinding:
    def test_finding_fingerprint_stable_across_dict_ordering(self):
        from v2.audit import Finding
        f1 = Finding(check_code="X", tier=1, severity="warn",
                     title="t", body="b", affected_count=1,
                     evidence={"a": 1, "b": 2}, auto_fix=None)
        f2 = Finding(check_code="X", tier=1, severity="warn",
                     title="t", body="b", affected_count=1,
                     evidence={"b": 2, "a": 1}, auto_fix=None)
        assert f1.fingerprint == f2.fingerprint

    def test_finding_fingerprint_changes_with_evidence(self):
        from v2.audit import Finding
        f1 = Finding(check_code="X", tier=1, severity="warn", title="t",
                     body="b", affected_count=1, evidence={"a": 1},
                     auto_fix=None)
        f2 = Finding(check_code="X", tier=1, severity="warn", title="t",
                     body="b", affected_count=1, evidence={"a": 2},
                     auto_fix=None)
        assert f1.fingerprint != f2.fingerprint

    def test_finding_fingerprint_handles_lists_in_evidence(self):
        from v2.audit import Finding
        f = Finding(check_code="X", tier=1, severity="warn", title="t",
                    body="b", affected_count=2, evidence={"ids": [3, 1, 2]},
                    auto_fix=None)
        # Stable: re-creating with same evidence gives same fingerprint
        assert f.fingerprint == Finding(check_code="X", tier=1, severity="warn",
            title="t", body="b", affected_count=2,
            evidence={"ids": [3, 1, 2]}, auto_fix=None).fingerprint
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestFinding -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v2.audit'`.

- [ ] **Step 3: Create v2/audit.py with the dataclass**

```python
# v2/audit.py
"""Self-healing audit: integrity, rule-overfitting, and app-issue detection.

Daily auditor that runs deterministic SQL checks plus a single Haiku
judgment call over active strategy rules. Default mode is propose-only;
Tier-1 checks declare an auto_fix callable that runs only when --apply is
passed on the CLI.

Design: docs/superpowers/specs/2026-05-06-self-healing-audit-design.md
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

MAX_AUTO_FIX_DEFAULT = 100
RULE_JUDGMENT_MODEL = "claude-haiku-4-5-20251001"
RULE_JUDGMENT_MAX_TOKENS = 4000


@dataclass
class Finding:
    check_code: str
    tier: int                      # 1 | 2 | 3
    severity: str                  # 'critical' | 'warn' | 'info'
    title: str
    body: str
    affected_count: int
    evidence: dict
    auto_fix: Callable | None = None  # cur -> dict (fix evidence)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            {"check_code": self.check_code, "evidence": self.evidence},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class AuditRunSummary:
    run_id: int | None
    findings_emitted: int = 0
    auto_fixed: int = 0
    failed_checks: int = 0
    has_critical_open: bool = False


# Each check's auto-fix accepts the cursor and returns a dict that gets
# merged into the finding's evidence.
AutoFix = Callable[["psycopg2.extensions.cursor"], dict]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestFinding -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): Finding dataclass and fingerprint for audit module"
```

---

## Task 4: `check_orphan_fks` (Tier 1, with auto-fix)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckOrphanFks:
    def _cur_with_results(self, results: dict[tuple, list]):
        """Build a mock cursor that returns specific rows per SQL fragment."""
        cur = MagicMock()
        def execute(sql, params=None):
            # Match by signal_type since the check runs three queries.
            if "news_signal" in sql:
                cur._results = results.get(("news_signal",), [])
            elif "macro_signal" in sql:
                cur._results = results.get(("macro_signal",), [])
            elif "thesis" in sql:
                cur._results = results.get(("thesis",), [])
            else:
                cur._results = []
        cur.execute.side_effect = execute
        cur.fetchall.side_effect = lambda: cur._results
        return cur

    def test_no_orphans_returns_no_findings(self):
        from v2.audit import check_orphan_fks
        cur = self._cur_with_results({})
        assert check_orphan_fks(cur) == []

    def test_news_signal_orphans_emit_finding(self):
        from v2.audit import check_orphan_fks
        cur = self._cur_with_results({
            ("news_signal",): [{"signal_id": 0}, {"signal_id": 99}],
        })
        findings = check_orphan_fks(cur)
        assert len(findings) == 1
        f = findings[0]
        assert f.check_code == "ORPHAN_FK_NEWS_SIGNAL"
        assert f.tier == 1
        assert f.severity == "warn"
        assert f.affected_count == 2
        assert sorted(f.evidence["signal_ids"]) == [0, 99]
        assert f.auto_fix is not None

    def test_thesis_orphans_emit_separate_finding(self):
        from v2.audit import check_orphan_fks
        cur = self._cur_with_results({
            ("thesis",): [{"signal_id": 5}],
        })
        findings = check_orphan_fks(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "ORPHAN_FK_THESIS"

    @patch("v2.database.trading_db.delete_orphan_decision_signals", return_value=2)
    def test_auto_fix_calls_delete_helper(self, mock_delete):
        from v2.audit import check_orphan_fks
        cur = self._cur_with_results({
            ("news_signal",): [{"signal_id": 1}, {"signal_id": 2}],
        })
        finding = check_orphan_fks(cur)[0]
        result = finding.auto_fix(cur)
        mock_delete.assert_called_once_with("news_signal", [1, 2])
        assert result == {"deleted": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckOrphanFks -v`
Expected: FAIL — `ImportError: cannot import name 'check_orphan_fks'`.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 1: orphan FKs in decision_signals -------------------------------

_ORPHAN_FK_QUERIES = {
    "news_signal":  ("ORPHAN_FK_NEWS_SIGNAL",
                     """SELECT DISTINCT signal_id FROM decision_signals
                        WHERE signal_type='news_signal'
                          AND signal_id NOT IN (SELECT id FROM news_signals)"""),
    "macro_signal": ("ORPHAN_FK_MACRO_SIGNAL",
                     """SELECT DISTINCT signal_id FROM decision_signals
                        WHERE signal_type='macro_signal'
                          AND signal_id NOT IN (SELECT id FROM macro_signals)"""),
    "thesis":       ("ORPHAN_FK_THESIS",
                     """SELECT DISTINCT signal_id FROM decision_signals
                        WHERE signal_type='thesis'
                          AND signal_id NOT IN (SELECT id FROM theses)"""),
}


def _make_orphan_autofix(signal_type: str, ids: list[int]):
    def _fix(_cur):
        from v2.database.trading_db import delete_orphan_decision_signals
        deleted = delete_orphan_decision_signals(signal_type, ids)
        return {"deleted": deleted}
    return _fix


def check_orphan_fks(cur) -> list[Finding]:
    """Detect rows in decision_signals whose signal_id no longer matches a
    real news_signals/macro_signals/theses row. Auto-fix deletes them."""
    findings = []
    for signal_type, (code, sql) in _ORPHAN_FK_QUERIES.items():
        cur.execute(sql)
        rows = cur.fetchall()
        if not rows:
            continue
        ids = sorted([r["signal_id"] for r in rows])
        findings.append(Finding(
            check_code=code,
            tier=1,
            severity="warn",
            title=f"{len(ids)} orphan {signal_type} reference(s) in decision_signals",
            body=(f"`decision_signals` rows reference `{signal_type}` ids that no longer "
                  f"exist. Filtered downstream but pollute schema. Auto-fix deletes."),
            affected_count=len(ids),
            evidence={"signal_type": signal_type, "signal_ids": ids},
            auto_fix=_make_orphan_autofix(signal_type, ids),
        ))
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckOrphanFks -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_orphan_fks with delete-orphan auto-fix"
```

---

## Task 5: `check_missing_backfill` (Tier 1, with auto-fix)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckMissingBackfill:
    def test_no_missing_rows_returns_no_findings(self):
        from v2.audit import check_missing_backfill
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_missing_backfill(cur) == []

    def test_missing_7d_emits_warn_finding(self):
        from v2.audit import check_missing_backfill
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [{"id": 11}, {"id": 12}],  # 7d missing
            [],                         # 30d missing
        ]
        findings = check_missing_backfill(cur)
        codes = sorted(f.check_code for f in findings)
        assert codes == ["BACKFILL_GAP_7D"]
        f = findings[0]
        assert f.severity == "warn"
        assert f.affected_count == 2
        assert sorted(f.evidence["decision_ids"]) == [11, 12]

    def test_more_than_25_rows_escalates_to_critical(self):
        from v2.audit import check_missing_backfill
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [{"id": i} for i in range(30)],
            [],
        ]
        findings = check_missing_backfill(cur)
        assert findings[0].severity == "critical"

    @patch("v2.backfill.backfill_decision_outcomes")
    def test_auto_fix_invokes_backfill_per_decision(self, mock_backfill):
        from v2.audit import check_missing_backfill
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [{"id": 5}, {"id": 6}],
            [],
        ]
        finding = check_missing_backfill(cur)[0]
        result = finding.auto_fix(cur)
        assert mock_backfill.call_count == 2
        assert result == {"backfilled_ids": [5, 6]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckMissingBackfill -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 1: missing 7d/30d outcome/benchmark backfill --------------------

def _make_backfill_autofix(decision_ids: list[int]):
    def _fix(_cur):
        from v2.backfill import backfill_decision_outcomes
        for did in decision_ids:
            backfill_decision_outcomes(did)
        return {"backfilled_ids": decision_ids}
    return _fix


def check_missing_backfill(cur) -> list[Finding]:
    """Decisions past the 7d/30d window with NULL outcome_*/benchmark_*.
    Each missing row drops a learning signal; auto-fix re-runs the existing
    backfill function."""
    findings = []
    for window, code in (("7d", "BACKFILL_GAP_7D"), ("30d", "BACKFILL_GAP_30D")):
        days = 7 if window == "7d" else 30
        cur.execute(f"""
            SELECT id FROM decisions
            WHERE action IN ('buy','sell')
              AND date <= now()::date - %s
              AND (outcome_{window} IS NULL OR benchmark_{window} IS NULL)
            ORDER BY id
        """, (days,))
        ids = [r["id"] for r in cur.fetchall()]
        if not ids:
            continue
        severity = "critical" if len(ids) > 25 else "warn"
        findings.append(Finding(
            check_code=code,
            tier=1,
            severity=severity,
            title=f"{len(ids)} decision(s) missing {window} outcome/benchmark backfill",
            body=(f"Decisions older than {window} have NULL outcome_{window} or "
                  f"benchmark_{window}. Auto-fix invokes backfill_decision_outcomes "
                  f"for each."),
            affected_count=len(ids),
            evidence={"decision_ids": ids, "window": window},
            auto_fix=_make_backfill_autofix(ids),
        ))
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckMissingBackfill -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_missing_backfill with re-backfill auto-fix"
```

---

## Task 6: `check_invalid_attribution_categories` (Tier 1, propose-only)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckInvalidAttributionCategories:
    def test_all_valid_returns_no_findings(self):
        from v2.audit import check_invalid_attribution_categories
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"category": "thesis"},
            {"category": "news_signal:earnings"},
        ]
        assert check_invalid_attribution_categories(cur) == []

    def test_typo_category_emits_critical_finding(self):
        from v2.audit import check_invalid_attribution_categories
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"category": "news_signal:earnigns"},   # typo
            {"category": "macro_signal:fed"},
        ]
        findings = check_invalid_attribution_categories(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "INVALID_ATTRIBUTION_CATEGORY"
        assert findings[0].severity == "critical"
        assert findings[0].auto_fix is None
        assert "news_signal:earnigns" in findings[0].evidence["categories"]

    def test_unknown_prefix_treated_as_invalid(self):
        from v2.audit import check_invalid_attribution_categories
        cur = MagicMock()
        cur.fetchall.return_value = [{"category": "weird:thing"}]
        findings = check_invalid_attribution_categories(cur)
        assert findings and findings[0].severity == "critical"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckInvalidAttributionCategories -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 1: invalid signal_attribution categories ------------------------

def check_invalid_attribution_categories(cur) -> list[Finding]:
    """Find signal_attribution.category rows whose suffix isn't in the
    classifier's valid enums. Detects classifier regressions and direct DB
    writes that bypass validation. No auto-fix — investigation needed."""
    from v2.classifier import VALID_TICKER_CATEGORIES, VALID_MACRO_CATEGORIES
    cur.execute("SELECT DISTINCT category FROM signal_attribution")
    cats = [r["category"] for r in cur.fetchall()]
    invalid = []
    for c in cats:
        if c == "thesis":
            continue
        if c.startswith("news_signal:"):
            suffix = c.split(":", 1)[1]
            if suffix not in VALID_TICKER_CATEGORIES:
                invalid.append(c)
        elif c.startswith("macro_signal:"):
            suffix = c.split(":", 1)[1]
            if suffix not in VALID_MACRO_CATEGORIES:
                invalid.append(c)
        else:
            invalid.append(c)  # unknown prefix
    if not invalid:
        return []
    return [Finding(
        check_code="INVALID_ATTRIBUTION_CATEGORY",
        tier=1, severity="critical",
        title=f"{len(invalid)} invalid attribution category value(s)",
        body=("`signal_attribution` contains categories outside the classifier's "
              "valid enums. Classifier regression or direct DB write."),
        affected_count=len(invalid),
        evidence={"categories": sorted(invalid)},
        auto_fix=None,
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckInvalidAttributionCategories -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_invalid_attribution_categories"
```

---

## Task 7: `check_snapshot_gaps` (Tier 1, propose-only)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckSnapshotGaps:
    @patch("v2.market_calendar.is_trading_day")
    def test_no_gaps_returns_no_findings(self, mock_is_td):
        from v2.audit import check_snapshot_gaps
        mock_is_td.return_value = False  # no trading days, trivial pass
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_snapshot_gaps(cur) == []

    @patch("v2.market_calendar.is_trading_day")
    def test_gaps_on_trading_days_emit_finding(self, mock_is_td):
        from v2.audit import check_snapshot_gaps
        from datetime import date
        # Two missing dates; mark them both as trading days.
        missing = [date(2026, 5, 1), date(2026, 5, 2)]
        cur = MagicMock()
        cur.fetchall.return_value = [{"day": d} for d in missing]
        mock_is_td.return_value = True
        findings = check_snapshot_gaps(cur)
        assert len(findings) == 1
        f = findings[0]
        assert f.check_code == "SNAPSHOT_GAP"
        assert f.severity == "warn"
        assert f.auto_fix is None
        assert len(f.evidence["missing_dates"]) == 2

    @patch("v2.market_calendar.is_trading_day")
    def test_skips_holidays_and_weekends(self, mock_is_td):
        from v2.audit import check_snapshot_gaps
        from datetime import date
        cur = MagicMock()
        cur.fetchall.return_value = [{"day": date(2026, 5, 25)}]  # Memorial Day
        mock_is_td.return_value = False  # holiday → not a trading day
        assert check_snapshot_gaps(cur) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckSnapshotGaps -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 1: account_snapshot trading-day gaps ----------------------------

def check_snapshot_gaps(cur) -> list[Finding]:
    """Trading days in last 30d with no account_snapshot row.
    No auto-fix — Alpaca historical equity retrieval is unreliable."""
    from v2.market_calendar import is_trading_day
    cur.execute("""
        WITH d AS (
            SELECT generate_series(now()::date - 30, now()::date - 1, '1 day')::date AS day
        )
        SELECT d.day FROM d
        LEFT JOIN account_snapshots a ON a.date=d.day
        WHERE a.date IS NULL
    """)
    rows = cur.fetchall()
    missing = sorted(r["day"] for r in rows if is_trading_day(r["day"]))
    if not missing:
        return []
    return [Finding(
        check_code="SNAPSHOT_GAP",
        tier=1, severity="warn",
        title=f"{len(missing)} trading day(s) missing account_snapshot in last 30d",
        body=("`account_snapshots` has gaps on trading days, breaking equity-curve "
              "and daily-snapshot dashboards. Investigate stage skipping."),
        affected_count=len(missing),
        evidence={"missing_dates": [d.isoformat() for d in missing]},
        auto_fix=None,
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckSnapshotGaps -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_snapshot_gaps using market_calendar"
```

---

## Task 8: `check_decision_equity_drift` (Tier 1, propose-only)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckDecisionEquityDrift:
    def test_no_drift_returns_no_findings(self):
        from v2.audit import check_decision_equity_drift
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_decision_equity_drift(cur) == []

    def test_drift_emits_critical_finding(self):
        from v2.audit import check_decision_equity_drift
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"id": 50, "decision_equity": 100000, "snapshot_equity": 99500, "delta": -500},
        ]
        findings = check_decision_equity_drift(cur)
        assert len(findings) == 1
        f = findings[0]
        assert f.check_code == "DECISION_EQUITY_DRIFT"
        assert f.severity == "critical"
        assert f.auto_fix is None
        assert 50 in f.evidence["decision_ids"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckDecisionEquityDrift -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 1: decision vs snapshot equity drift (P0.4 regression detector) -

def check_decision_equity_drift(cur) -> list[Finding]:
    """Same-day decisions whose account_equity differs from snapshot
    portfolio_value by > $100. Detects P0.4-style regressions where
    decisions are stamped with stale pre-session equity."""
    cur.execute("""
        SELECT d.id,
               d.account_equity AS decision_equity,
               a.portfolio_value AS snapshot_equity,
               (d.account_equity - a.portfolio_value) AS delta
        FROM decisions d
        JOIN account_snapshots a ON a.date = d.date
        WHERE d.action IN ('buy','sell')
          AND d.date > now()::date - 60
          AND ABS(COALESCE(d.account_equity, 0) - a.portfolio_value) > 100
    """)
    rows = cur.fetchall()
    if not rows:
        return []
    return [Finding(
        check_code="DECISION_EQUITY_DRIFT",
        tier=1, severity="critical",
        title=f"{len(rows)} decision(s) with account_equity drifted from snapshot",
        body=("Decisions in last 60 days have `account_equity` differing from "
              "same-day `account_snapshots.portfolio_value` by > $100. "
              "Suggests stale snapshot logging (P0.4 regression)."),
        affected_count=len(rows),
        evidence={
            "decision_ids": [r["id"] for r in rows],
            "max_delta": max(abs(r["delta"]) for r in rows),
        },
        auto_fix=None,
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckDecisionEquityDrift -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_decision_equity_drift (P0.4 regression detector)"
```

---

## Task 9: `check_attribution_category_coverage` (Tier 3)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckAttributionCategoryCoverage:
    def test_enough_categories_returns_no_findings(self):
        from v2.audit import check_attribution_category_coverage
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"category": f"news_signal:{i}", "sample_size_30d": 10} for i in range(6)
        ]
        assert check_attribution_category_coverage(cur) == []

    def test_below_threshold_emits_finding(self):
        from v2.audit import check_attribution_category_coverage
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"category": "thesis", "sample_size_30d": 30},
            {"category": "news_signal:earnings", "sample_size_30d": 5},
        ]
        findings = check_attribution_category_coverage(cur)
        assert len(findings) == 1
        f = findings[0]
        assert f.check_code == "ATTRIBUTION_COVERAGE_LOW"
        assert f.severity == "warn"
        assert f.auto_fix is None

    def test_categories_below_min_n_excluded(self):
        from v2.audit import check_attribution_category_coverage
        cur = MagicMock()
        # 6 categories but only 1 has sample_size_30d>=3 → coverage low
        cur.fetchall.return_value = [
            {"category": "thesis", "sample_size_30d": 30},
        ] + [
            {"category": f"x:{i}", "sample_size_30d": 1} for i in range(5)
        ]
        findings = check_attribution_category_coverage(cur)
        assert len(findings) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckAttributionCategoryCoverage -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 3: attribution category coverage --------------------------------

ATTRIBUTION_MIN_CATEGORIES = 5
ATTRIBUTION_MIN_N_PER_CATEGORY = 3


def check_attribution_category_coverage(cur) -> list[Finding]:
    """Wiring-degradation early warning. Memory: 'if no new categories appear
    after several sessions, the wiring has regressed.'"""
    cur.execute("SELECT category, COALESCE(sample_size_30d,0) AS sample_size_30d "
                "FROM signal_attribution")
    rows = cur.fetchall()
    qualifying = [r for r in rows if r["sample_size_30d"] >= ATTRIBUTION_MIN_N_PER_CATEGORY]
    if len(qualifying) >= ATTRIBUTION_MIN_CATEGORIES:
        return []
    return [Finding(
        check_code="ATTRIBUTION_COVERAGE_LOW",
        tier=3, severity="warn",
        title=f"Only {len(qualifying)} attribution categories with n_30d≥{ATTRIBUTION_MIN_N_PER_CATEGORY}",
        body=("`signal_attribution` has fewer than the expected number of populated "
              f"categories. Threshold: ≥{ATTRIBUTION_MIN_CATEGORIES} with "
              f"sample_size_30d≥{ATTRIBUTION_MIN_N_PER_CATEGORY}. Likely cause: "
              "signal_refs wiring (strategist→playbook→executor) has regressed."),
        affected_count=len(qualifying),
        evidence={
            "qualifying_categories": [r["category"] for r in qualifying],
            "all_categories": [r["category"] for r in rows],
        },
        auto_fix=None,
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckAttributionCategoryCoverage -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_attribution_category_coverage (wiring regression detector)"
```

---

## Task 10: `check_stage_failure_rate` (Tier 3)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckStageFailureRate:
    def test_no_failures_no_stale_returns_nothing(self):
        from v2.audit import check_stage_failure_rate
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [{"stage_name": "pipeline", "completed": 7, "failed": 0}],
            [],
        ]
        assert check_stage_failure_rate(cur) == []

    def test_failure_rate_above_20pct_warns(self):
        from v2.audit import check_stage_failure_rate
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [{"stage_name": "trade_posts", "completed": 1, "failed": 2}],
            [],
        ]
        findings = check_stage_failure_rate(cur)
        codes = sorted(f.check_code for f in findings)
        assert codes == ["STAGE_FAILURE_RATE"]
        assert findings[0].severity == "critical"  # 2/3 = 67%

    def test_running_stale_24h_emits_finding(self):
        from v2.audit import check_stage_failure_rate
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [{"stage_name": "executor", "completed": 7, "failed": 0}],
            [{"id": 99, "stage_name": "executor"}],
        ]
        findings = check_stage_failure_rate(cur)
        assert any(f.check_code == "STAGE_RUNNING_STALE" for f in findings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckStageFailureRate -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 3: stage failure rate + stale running --------------------------

def check_stage_failure_rate(cur) -> list[Finding]:
    """Per-stage failure rate (last 30d) and stale 'running' rows (>24h)."""
    findings = []

    cur.execute("""
        SELECT stage_name,
               COUNT(*) FILTER (WHERE status='completed') AS completed,
               COUNT(*) FILTER (WHERE status='failed') AS failed
        FROM session_stages
        WHERE started_at > now() - interval '30 days'
        GROUP BY stage_name
    """)
    flagged = []
    for r in cur.fetchall():
        total = r["completed"] + r["failed"]
        if total < 3:
            continue
        rate = r["failed"] / total
        if rate >= 0.20:
            flagged.append({"stage_name": r["stage_name"], "completed": r["completed"],
                            "failed": r["failed"], "rate": round(rate, 3)})
    if flagged:
        worst = max(f["rate"] for f in flagged)
        sev = "critical" if worst >= 0.50 else "warn"
        findings.append(Finding(
            check_code="STAGE_FAILURE_RATE", tier=3, severity=sev,
            title=f"{len(flagged)} stage(s) with failure rate ≥ 20% in last 30d",
            body="See evidence for per-stage rates.",
            affected_count=len(flagged),
            evidence={"stages": flagged},
            auto_fix=None,
        ))

    cur.execute("""
        SELECT id, stage_name FROM session_stages
        WHERE status='running' AND started_at < now() - interval '24 hours'
    """)
    stale = cur.fetchall()
    if stale:
        findings.append(Finding(
            check_code="STAGE_RUNNING_STALE", tier=3, severity="warn",
            title=f"{len(stale)} session_stage row(s) stuck in 'running' >24h",
            body="Stages never marked completed/failed; orphan from interrupted runs.",
            affected_count=len(stale),
            evidence={"stage_ids": [r["id"] for r in stale],
                      "stage_names": sorted({r["stage_name"] for r in stale})},
            auto_fix=None,
        ))
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckStageFailureRate -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_stage_failure_rate + stale-running detector"
```

---

## Task 11: `check_cost_trend` (Tier 3, info)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckCostTrend:
    def test_flat_costs_no_finding(self):
        from v2.audit import check_cost_trend
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"stage_name": "strategist", "recent_tok": 100000, "prior_tok": 100000},
        ]
        assert check_cost_trend(cur) == []

    def test_2x_spike_emits_info(self):
        from v2.audit import check_cost_trend
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"stage_name": "strategist", "recent_tok": 220000, "prior_tok": 100000},
        ]
        findings = check_cost_trend(cur)
        assert len(findings) == 1
        f = findings[0]
        assert f.check_code == "COST_TREND_SPIKE"
        assert f.severity == "info"

    def test_zero_prior_skipped(self):
        from v2.audit import check_cost_trend
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"stage_name": "x", "recent_tok": 100, "prior_tok": 0},
        ]
        assert check_cost_trend(cur) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckCostTrend -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 3: per-stage cost trend (info) ---------------------------------

def check_cost_trend(cur) -> list[Finding]:
    """7d-vs-prior-7d total token usage by stage. Flag stages with ≥2x growth."""
    cur.execute("""
        WITH recent AS (
            SELECT stage_name,
                   SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)
                      +COALESCE(cache_creation_tokens,0)
                      +COALESCE(cache_read_tokens,0)) AS tok
            FROM session_stages
            WHERE started_at > now() - interval '7 days' AND status='completed'
            GROUP BY stage_name
        ),
        prior AS (
            SELECT stage_name,
                   SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)
                      +COALESCE(cache_creation_tokens,0)
                      +COALESCE(cache_read_tokens,0)) AS tok
            FROM session_stages
            WHERE started_at > now() - interval '14 days'
              AND started_at <= now() - interval '7 days'
              AND status='completed'
            GROUP BY stage_name
        )
        SELECT COALESCE(r.stage_name, p.stage_name) AS stage_name,
               COALESCE(r.tok, 0) AS recent_tok,
               COALESCE(p.tok, 0) AS prior_tok
        FROM recent r FULL OUTER JOIN prior p ON r.stage_name = p.stage_name
    """)
    spikes = []
    for r in cur.fetchall():
        if not r["prior_tok"]:
            continue
        if r["recent_tok"] >= 2 * r["prior_tok"]:
            spikes.append({"stage_name": r["stage_name"],
                           "recent_tok": r["recent_tok"],
                           "prior_tok": r["prior_tok"],
                           "ratio": round(r["recent_tok"] / r["prior_tok"], 2)})
    if not spikes:
        return []
    return [Finding(
        check_code="COST_TREND_SPIKE", tier=3, severity="info",
        title=f"{len(spikes)} stage(s) with token usage ≥2× prior 7-day window",
        body="Per-stage 7-day-rolling token totals doubled vs. prior 7-day window.",
        affected_count=len(spikes),
        evidence={"stages": spikes},
        auto_fix=None,
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckCostTrend -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_cost_trend (per-stage 2x spike detector)"
```

---

## Task 12: `check_decisions_missing_signal_refs` (Tier 3)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckDecisionsMissingSignalRefs:
    def test_all_have_refs_no_finding(self):
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 20, "missing": 0, "on_pb_missing": 0}
        cur.fetchall.return_value = []
        assert check_decisions_missing_signal_refs(cur) == []

    def test_warn_below_critical_threshold(self):
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 100, "missing": 8, "on_pb_missing": 5}  # 5% on-pb
        cur.fetchall.return_value = [{"id": 1}]
        findings = check_decisions_missing_signal_refs(cur)
        assert findings[0].severity == "warn"

    def test_critical_when_on_pb_missing_above_10pct(self):
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 29, "missing": 13, "on_pb_missing": 13}
        cur.fetchall.return_value = [{"id": i} for i in range(13)]
        findings = check_decisions_missing_signal_refs(cur)
        assert findings[0].severity == "critical"
        assert findings[0].evidence["on_pb_missing"] == 13
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckDecisionsMissingSignalRefs -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 3: decisions missing signal_refs --------------------------------

def check_decisions_missing_signal_refs(cur) -> list[Finding]:
    cur.execute("""
        SELECT
          COUNT(*) FILTER (WHERE d.action IN ('buy','sell')) AS total,
          COUNT(*) FILTER (WHERE d.action IN ('buy','sell') AND ds.decision_id IS NULL) AS missing,
          COUNT(*) FILTER (WHERE d.action IN ('buy','sell')
                            AND COALESCE(d.is_off_playbook,false)=false
                            AND ds.decision_id IS NULL) AS on_pb_missing
        FROM decisions d
        LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
          ON ds.decision_id = d.id
        WHERE d.date > now()::date - 30
    """)
    summary = cur.fetchone()
    if not summary["missing"]:
        return []

    cur.execute("""
        SELECT d.id FROM decisions d
        LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
          ON ds.decision_id = d.id
        WHERE d.action IN ('buy','sell')
          AND d.date > now()::date - 30
          AND ds.decision_id IS NULL
        ORDER BY d.id
    """)
    ids = [r["id"] for r in cur.fetchall()]

    on_pb_share = (summary["on_pb_missing"] / summary["total"]) if summary["total"] else 0
    severity = "critical" if on_pb_share > 0.10 else "warn"

    return [Finding(
        check_code="DECISIONS_NO_SIGNAL_REFS",
        tier=3, severity=severity,
        title=(f"{summary['missing']} of {summary['total']} recent buy/sell decisions "
               f"have no signal_refs ({summary['on_pb_missing']} on-playbook)"),
        body=("Strategist→playbook→executor signal_refs wiring is being honored "
              "intermittently. Off-playbook trades may legitimately lack refs; "
              "on-playbook gaps indicate degradation."),
        affected_count=summary["missing"],
        evidence={
            "total": summary["total"],
            "missing": summary["missing"],
            "on_pb_missing": summary["on_pb_missing"],
            "decision_ids": ids,
        },
        auto_fix=None,
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckDecisionsMissingSignalRefs -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_decisions_missing_signal_refs (wiring regression)"
```

---

## Task 13: `check_theses_missing_signal_refs` (Tier 3)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckThesesMissingSignalRefs:
    def test_no_recent_theses_no_finding(self):
        from v2.audit import check_theses_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 0, "missing": 0}
        cur.fetchall.return_value = []
        assert check_theses_missing_signal_refs(cur) == []

    def test_below_25pct_missing_no_finding(self):
        from v2.audit import check_theses_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 20, "missing": 4}  # 20%
        cur.fetchall.return_value = [{"id": i} for i in range(4)]
        assert check_theses_missing_signal_refs(cur) == []

    def test_above_threshold_warns(self):
        from v2.audit import check_theses_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 69, "missing": 56}
        cur.fetchall.return_value = [{"id": i} for i in range(56)]
        findings = check_theses_missing_signal_refs(cur)
        assert findings[0].check_code == "THESES_NO_SIGNAL_REFS"
        assert findings[0].severity == "warn"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckThesesMissingSignalRefs -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 3: theses missing signal_refs (Rule #6 drift) -------------------

def check_theses_missing_signal_refs(cur) -> list[Finding]:
    cur.execute("""
        SELECT
          COUNT(*) AS total,
          COUNT(*) FILTER (WHERE ts.thesis_id IS NULL) AS missing
        FROM theses t
        LEFT JOIN (SELECT DISTINCT thesis_id FROM thesis_signals) ts
          ON ts.thesis_id = t.id
        WHERE t.created_at > now() - interval '30 days'
    """)
    s = cur.fetchone()
    if not s["total"] or s["missing"] / s["total"] <= 0.25:
        return []

    cur.execute("""
        SELECT t.id FROM theses t
        LEFT JOIN (SELECT DISTINCT thesis_id FROM thesis_signals) ts
          ON ts.thesis_id = t.id
        WHERE t.created_at > now() - interval '30 days' AND ts.thesis_id IS NULL
        ORDER BY t.id
    """)
    ids = [r["id"] for r in cur.fetchall()]
    return [Finding(
        check_code="THESES_NO_SIGNAL_REFS",
        tier=3, severity="warn",
        title=f"{s['missing']} of {s['total']} recent theses have no signal_refs",
        body=("Strategist is creating theses without citing signals, violating "
              "Rule #6 from the 2026-05-02 wiring fix. Without citations, "
              "downstream attribution receives nothing to score."),
        affected_count=s["missing"],
        evidence={"total": s["total"], "missing": s["missing"], "thesis_ids": ids},
        auto_fix=None,
    )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckThesesMissingSignalRefs -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_theses_missing_signal_refs (Rule #6 drift)"
```

---

## Task 14: `check_rule_judgment` (Tier 2 LLM)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

This is the only check that calls Claude. The LLM is a *proposer* whose findings are validated against the same SQL data we already have.

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCheckRuleJudgment:
    def _stub_inputs(self, cur):
        """Stub the SQL queries the check runs to assemble its prompt."""
        # 1) active rules
        rules = [
            {"id": 27, "rule_text": "During fragile macro windows cap deployment "
                                    "at $500/day. Lifts when binary event resolves "
                                    "and markets confirm direction.",
             "created_at": None},
            {"id": 39, "rule_text": "When attribution shows thesis beat-rate <55% "
                                    "over 20+ samples, require corroboration.",
             "created_at": None},
        ]
        # 2) attribution
        attribution = [
            {"category": "thesis", "sample_size": 30, "sample_size_30d": 23,
             "avg_outcome_7d": -0.4, "win_rate_7d": 0.47,
             "avg_outcome_30d": -0.5, "win_rate_30d": 0.45},
        ]
        # 3) citation counts: rule 27 is cited; rule 39 is dead.
        citations = [{"rule_id": 27, "n": 12}, {"rule_id": 39, "n": 0}]
        # 4) decision-data summary
        summary = {"recent_buys_with_empty_signal_refs": 13,
                   "recent_thesis_only_decisions": 5,
                   "recent_off_playbook_buys": 2}
        cur.fetchall.side_effect = [rules, attribution, citations]
        cur.fetchone.return_value = summary
        return rules, attribution, citations, summary

    @patch("v2.audit._call_rule_judgment_llm")
    def test_drops_dead_rule_when_citations_disagree(self, mock_llm):
        from v2.audit import check_rule_judgment
        cur = MagicMock()
        self._stub_inputs(cur)
        mock_llm.return_value = ({
            "findings": [
                # LLM claims rule 27 is dead even though we counted 12 citations.
                {"check_code": "RULE_DEAD", "rule_id": 27,
                 "title": "Rule 27 not cited", "explanation": "..."},
            ]
        }, {"input_tokens": 800, "output_tokens": 100,
             "cache_creation_tokens": 0, "cache_read_tokens": 0})
        findings = check_rule_judgment(cur)
        # Verifier dropped the unverifiable RULE_DEAD claim.
        assert findings == []

    @patch("v2.audit._call_rule_judgment_llm")
    def test_accepts_unfalsifiable_lift_finding(self, mock_llm):
        from v2.audit import check_rule_judgment
        cur = MagicMock()
        self._stub_inputs(cur)
        mock_llm.return_value = ({
            "findings": [
                {"check_code": "RULE_UNFALSIFIABLE_LIFT", "rule_id": 27,
                 "title": "Rule 27 lift condition has no numeric threshold",
                 "explanation": "Lift clause says 'markets confirm direction "
                                "for 2+ days' with no defined threshold."},
            ]
        }, {"input_tokens": 800, "output_tokens": 100,
             "cache_creation_tokens": 0, "cache_read_tokens": 0})
        findings = check_rule_judgment(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "RULE_UNFALSIFIABLE_LIFT"
        assert findings[0].evidence["rule_id"] == 27

    @patch("v2.audit._call_rule_judgment_llm")
    def test_drops_unknown_rule_id(self, mock_llm):
        from v2.audit import check_rule_judgment
        cur = MagicMock()
        self._stub_inputs(cur)
        mock_llm.return_value = ({
            "findings": [
                {"check_code": "RULE_LOW_N_BACKING", "rule_id": 999,
                 "title": "x", "explanation": "y"},
            ]
        }, {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_tokens": 0, "cache_read_tokens": 0})
        findings = check_rule_judgment(cur)
        assert findings == []

    @patch("v2.audit._call_rule_judgment_llm")
    def test_drops_invalid_check_code(self, mock_llm):
        from v2.audit import check_rule_judgment
        cur = MagicMock()
        self._stub_inputs(cur)
        mock_llm.return_value = ({
            "findings": [{"check_code": "RULE_NOT_VALID", "rule_id": 27,
                          "title": "x", "explanation": "y"}]
        }, {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_tokens": 0, "cache_read_tokens": 0})
        assert check_rule_judgment(cur) == []

    @patch("v2.audit._call_rule_judgment_llm")
    def test_truncates_runaway_output(self, mock_llm):
        from v2.audit import check_rule_judgment
        cur = MagicMock()
        self._stub_inputs(cur)
        many = [{"check_code": "RULE_DEAD", "rule_id": 39,
                 "title": f"t{i}", "explanation": "..."} for i in range(50)]
        mock_llm.return_value = ({"findings": many},
            {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_tokens": 0, "cache_read_tokens": 0})
        findings = check_rule_judgment(cur)
        assert len(findings) == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckRuleJudgment -v`
Expected: FAIL.

- [ ] **Step 3: Implement the check**

Append to `v2/audit.py`:

```python
# --- Tier 2: LLM-judgment check (single Haiku call) ----------------------

VALID_RULE_CHECK_CODES = {
    "RULE_UNFALSIFIABLE_LIFT", "RULE_LOW_N_BACKING", "RULE_RETIRED_BUCKET",
    "RULE_CONTRADICTION", "RULE_DEAD",
}

RULE_JUDGE_SYSTEM = """\
You are an auditor reviewing a trading system's active strategy rules. You
will be given (1) the full text of all active rules, (2) the current
signal_attribution table, (3) per-rule citation counts in recent decisions,
and (4) a summary of recent decision-data anomalies.

Emit findings in JSON. Be conservative — if unsure, omit. Each finding must
include a `check_code` from this fixed set:

  RULE_UNFALSIFIABLE_LIFT — rule's "lift" / "deactivate" / "expires when"
    clause has no numeric criterion (e.g. "markets confirm direction" with no
    defined threshold).
  RULE_LOW_N_BACKING — rule cites attribution data with sample_size < 10, or
    cites a category whose evidence does not support the claim.
  RULE_RETIRED_BUCKET — rule references an attribution category that no
    longer exists in the snapshot.
  RULE_CONTRADICTION — two active rules contradict each other, OR a rule is
    contradicted by observed decision data (e.g., rule says "X must always
    happen" but stats show X happens only 50% of the time).
  RULE_DEAD — rule has zero citations in last 30 days AND its conditions
    appear satisfiable in current state.

Output format:
{
  "findings": [
    {"check_code": "...", "rule_id": <int>, "title": "...",
     "explanation": "...", "evidence_quote": "...",
     "contradicts_rule_id": <int or null>}
  ]
}

Maximum 20 findings. If you cannot find any defensible finding, return
{"findings": []}.
"""


def _build_rule_judgment_prompt(rules, attribution, citation_counts, summary) -> str:
    parts = ["## Active rules\n"]
    for r in rules:
        parts.append(f"### Rule {r['id']}\n{r['rule_text']}\n")
    parts.append("\n## signal_attribution snapshot\n")
    for a in attribution:
        parts.append(
            f"- {a['category']}: n={a['sample_size']} n_30d={a['sample_size_30d']} "
            f"out7={a['avg_outcome_7d']} win7={a['win_rate_7d']} "
            f"out30={a['avg_outcome_30d']} win30={a['win_rate_30d']}"
        )
    parts.append("\n\n## Per-rule citation counts (last 30d)\n")
    for c in citation_counts:
        parts.append(f"- rule {c['rule_id']}: {c['n']} citations")
    parts.append("\n\n## Decision-data summary\n")
    parts.append(json.dumps(summary, indent=2, default=str))
    return "\n".join(parts)


def _call_rule_judgment_llm(prompt: str) -> tuple[dict, dict]:
    """Returns (parsed_json, usage_dict). Separate function for easy stubbing."""
    from anthropic import Anthropic
    import os

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=RULE_JUDGMENT_MODEL,
        max_tokens=RULE_JUDGMENT_MAX_TOKENS,
        system=RULE_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    parsed = _extract_json(text)
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
        "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    }
    return parsed, usage


def _extract_json(text: str) -> dict:
    """Find the first JSON object in text and parse it. Returns {} on failure."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        log.warning("Could not extract JSON from LLM response: %s", text[:200])
        return {}


# Module-level state captured by check_rule_judgment so the runner can read
# usage after the call. Keyed weakly by audit_run_id; reset before each run.
_LAST_RULE_JUDGMENT_USAGE: dict = {}


def get_last_rule_judgment_usage() -> dict:
    return _LAST_RULE_JUDGMENT_USAGE.copy()


def _reset_rule_judgment_usage() -> None:
    _LAST_RULE_JUDGMENT_USAGE.clear()


def check_rule_judgment(cur) -> list[Finding]:
    """Single LLM call surveying active rules for the 5 overfitting patterns."""
    cur.execute("SELECT id, rule_text, created_at FROM strategy_rules WHERE status='active'")
    rules = cur.fetchall()
    if not rules:
        return []
    active_rule_ids = {r["id"] for r in rules}

    cur.execute("SELECT category, sample_size, sample_size_30d, "
                "avg_outcome_7d, win_rate_7d, avg_outcome_30d, win_rate_30d "
                "FROM signal_attribution")
    attribution = cur.fetchall()
    attribution_categories = {a["category"] for a in attribution}

    cur.execute("""
        WITH r AS (SELECT id FROM strategy_rules WHERE status='active')
        SELECT r.id AS rule_id,
               COUNT(*) FILTER (
                 WHERE d.reasoning ~* ('\\\\mrule\\\\s*#?\\\\s*' || r.id || '\\\\M')
                   AND d.date > now()::date - 30
               ) AS n
        FROM r LEFT JOIN decisions d ON true
        GROUP BY r.id
    """)
    citation_counts = cur.fetchall()
    citation_map = {c["rule_id"]: c["n"] for c in citation_counts}

    cur.execute("""
        SELECT
          COUNT(*) FILTER (WHERE d.action IN ('buy','sell')
                            AND d.date > now()::date - 30
                            AND ds.decision_id IS NULL) AS recent_buys_with_empty_signal_refs,
          0 AS recent_thesis_only_decisions,
          COUNT(*) FILTER (WHERE d.action='buy'
                            AND d.date > now()::date - 30
                            AND COALESCE(d.is_off_playbook,false)=true) AS recent_off_playbook_buys
        FROM decisions d
        LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
          ON ds.decision_id = d.id
    """)
    summary = cur.fetchone()

    prompt = _build_rule_judgment_prompt(rules, attribution, citation_counts, summary)
    parsed, usage = _call_rule_judgment_llm(prompt)

    _reset_rule_judgment_usage()
    _LAST_RULE_JUDGMENT_USAGE.update(usage)

    findings: list[Finding] = []
    for item in parsed.get("findings", [])[:20]:
        code = item.get("check_code")
        rid = item.get("rule_id")
        if code not in VALID_RULE_CHECK_CODES:
            continue
        if rid not in active_rule_ids:
            continue
        if code == "RULE_DEAD" and citation_map.get(rid, 0) > 0:
            log.warning("Dropping unverifiable RULE_DEAD for rule_id=%d", rid)
            continue
        if code == "RULE_RETIRED_BUCKET":
            cited = item.get("evidence_quote", "")
            if any(cat in cited for cat in attribution_categories):
                continue
        findings.append(Finding(
            check_code=code, tier=2, severity="warn",
            title=item.get("title", "")[:160],
            body=item.get("explanation", "")[:1200],
            affected_count=1,
            evidence={
                "rule_id": rid,
                "evidence_quote": item.get("evidence_quote", "")[:400],
                "contradicts_rule_id": item.get("contradicts_rule_id"),
            },
            auto_fix=None,
        ))
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCheckRuleJudgment -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): check_rule_judgment (Haiku LLM with deterministic post-validation)"
```

---

## Task 15: Runner orchestration

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestRunner:
    def _patch_db(self):
        """Common patches for runner tests."""
        return ExitStack()

    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_per_check_isolation(self, mock_cur, mock_run, mock_finding,
                                 mock_supersede, mock_finalize, mock_unlock,
                                 mock_lock):
        from v2.audit import run_audit, Finding
        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur

        good_check = MagicMock(__name__="good_check",
                               return_value=[Finding("OK", 1, "warn",
                                                     "t", "b", 1, {"x":1}, None)])
        bad_check = MagicMock(__name__="bad_check",
                              side_effect=RuntimeError("boom"))

        with patch("v2.audit.CHECKS", [bad_check, good_check]):
            summary = run_audit(apply=False)

        assert summary.failed_checks == 1
        # good_check still ran and emitted a finding
        good_call_count = sum(
            1 for c in mock_finding.call_args_list
            if c.kwargs.get("check_code") == "OK"
        )
        assert good_call_count == 1

    @patch("v2.audit.try_advisory_audit_lock", return_value=False)
    def test_advisory_lock_contention_exits_cleanly(self, mock_lock):
        from v2.audit import run_audit
        summary = run_audit(apply=False)
        assert summary.run_id is None
        assert summary.findings_emitted == 0

    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_apply_invokes_auto_fix(self, mock_cur, mock_run, mock_finding,
                                    mock_supersede, mock_finalize, mock_unlock,
                                    mock_lock):
        from v2.audit import run_audit, Finding
        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur
        fix_calls = []
        def fix(c):
            fix_calls.append("ran")
            return {"deleted": 3}
        check = MagicMock(__name__="c",
                          return_value=[Finding("X", 1, "warn", "t", "b", 1,
                                                {"a":1}, fix)])
        with patch("v2.audit.CHECKS", [check]):
            summary = run_audit(apply=True)
        assert summary.auto_fixed == 1
        assert fix_calls == ["ran"]

    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_max_auto_fix_ceiling_escalates(self, mock_cur, mock_run, mock_finding,
                                            mock_supersede, mock_finalize, mock_unlock,
                                            mock_lock):
        from v2.audit import run_audit, Finding
        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur
        n_fixes = []
        def fix(c):
            n_fixes.append(1)
            return {}
        many = [Finding("X", 1, "warn", "t", "b", 1, {"i": i}, fix) for i in range(5)]
        check = MagicMock(__name__="c", return_value=many)
        with patch("v2.audit.CHECKS", [check]):
            summary = run_audit(apply=True, max_auto_fix=3)
        assert summary.auto_fixed == 3
        assert len(n_fixes) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestRunner -v`
Expected: FAIL.

- [ ] **Step 3: Implement the runner**

Append to `v2/audit.py`:

```python
# --- Runner ---------------------------------------------------------------

from v2.database.connection import get_cursor
from v2.database.trading_db import (
    insert_audit_run,
    insert_audit_finding,
    finalize_audit_run,
    supersede_stale_open_findings,
    try_advisory_audit_lock,
    release_advisory_audit_lock,
)


CHECKS: list = [
    # Order matters: cheap & deterministic first; LLM last (cost + dependency
    # on attribution being current).
    "check_orphan_fks",
    "check_missing_backfill",
    "check_invalid_attribution_categories",
    "check_snapshot_gaps",
    "check_decision_equity_drift",
    "check_attribution_category_coverage",
    "check_stage_failure_rate",
    "check_cost_trend",
    "check_decisions_missing_signal_refs",
    "check_theses_missing_signal_refs",
    "check_rule_judgment",
]


def _resolve_checks() -> list:
    import sys
    mod = sys.modules[__name__]
    return [getattr(mod, name) for name in CHECKS] if isinstance(CHECKS[0], str) else CHECKS


def _emit_check_failure(*, run_id: int, check_name: str, exc: Exception):
    import traceback
    insert_audit_finding(
        audit_run_id=run_id,
        check_code="CHECK_FAILED",
        tier=3, severity="warn",
        title=f"Audit check {check_name} raised {type(exc).__name__}",
        body=str(exc)[:1000],
        affected_count=0,
        evidence={"check": check_name, "exception": str(exc),
                  "traceback": traceback.format_exc()[:2000]},
        fingerprint=hashlib.sha256(
            (check_name + str(exc)[:200]).encode("utf-8")
        ).hexdigest(),
    )


def run_audit(apply: bool = False, max_auto_fix: int = MAX_AUTO_FIX_DEFAULT) -> AuditRunSummary:
    if not try_advisory_audit_lock():
        log.warning("Audit already running (advisory lock contention); exiting cleanly")
        return AuditRunSummary(run_id=None)

    summary = AuditRunSummary(run_id=None)
    rule_judgment_usage: dict = {}
    try:
        run_id = insert_audit_run(mode="apply" if apply else "check")
        summary.run_id = run_id
        log.info("Audit run #%d started (mode=%s)", run_id, "apply" if apply else "check")

        current_fingerprints: set[str] = set()
        emitted = auto_fixed = failed_checks = 0

        with get_cursor() as cur:
            for check in _resolve_checks():
                cur.execute("SAVEPOINT audit_check")
                try:
                    findings = check(cur)
                    cur.execute("RELEASE SAVEPOINT audit_check")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT audit_check")
                    cur.execute("RELEASE SAVEPOINT audit_check")
                    log.exception("Audit check %s failed", check.__name__)
                    _emit_check_failure(run_id=run_id, check_name=check.__name__, exc=e)
                    failed_checks += 1
                    continue

                # capture rule-judgment usage if applicable
                if check.__name__ == "check_rule_judgment":
                    rule_judgment_usage = get_last_rule_judgment_usage()

                for f in findings:
                    current_fingerprints.add(f.fingerprint)
                    inserted_id = insert_audit_finding(
                        audit_run_id=run_id,
                        check_code=f.check_code, tier=f.tier, severity=f.severity,
                        title=f.title, body=f.body,
                        affected_count=f.affected_count, evidence=f.evidence,
                        fingerprint=f.fingerprint,
                    )
                    if inserted_id is not None:
                        emitted += 1
                    if f.severity == "critical":
                        summary.has_critical_open = True

                    if apply and f.auto_fix is not None:
                        if auto_fixed >= max_auto_fix:
                            log.error("Auto-fix ceiling %d reached; escalating "
                                      "%s to critical without applying", max_auto_fix, f.check_code)
                            continue
                        cur.execute("SAVEPOINT audit_fix")
                        try:
                            fix_evidence = f.auto_fix(cur)
                            cur.execute("RELEASE SAVEPOINT audit_fix")
                            insert_audit_finding(
                                audit_run_id=run_id,
                                check_code=f.check_code + "_FIXED",
                                tier=f.tier, severity="info",
                                title=f"Auto-fixed: {f.title}",
                                body=f"Applied auto-fix for {f.check_code}.",
                                affected_count=f.affected_count,
                                evidence={**f.evidence, "fix": fix_evidence},
                                fingerprint=f.fingerprint + ":fixed",
                                status="auto_fixed",
                            )
                            auto_fixed += 1
                        except Exception as fx:
                            cur.execute("ROLLBACK TO SAVEPOINT audit_fix")
                            cur.execute("RELEASE SAVEPOINT audit_fix")
                            log.exception("Auto-fix failed for %s", f.check_code)

            supersede_stale_open_findings(run_id=run_id,
                                          current_fingerprints=current_fingerprints)

        finalize_audit_run(
            run_id=run_id,
            total_findings=emitted,
            auto_fixed=auto_fixed,
            failed_checks=failed_checks,
            model=RULE_JUDGMENT_MODEL if rule_judgment_usage else None,
            input_tokens=rule_judgment_usage.get("input_tokens"),
            output_tokens=rule_judgment_usage.get("output_tokens"),
            cache_creation_tokens=rule_judgment_usage.get("cache_creation_tokens"),
            cache_read_tokens=rule_judgment_usage.get("cache_read_tokens"),
        )

        summary.findings_emitted = emitted
        summary.auto_fixed = auto_fixed
        summary.failed_checks = failed_checks
        log.info("Audit run #%d complete: emitted=%d auto_fixed=%d failed_checks=%d",
                 run_id, emitted, auto_fixed, failed_checks)
        return summary
    finally:
        release_advisory_audit_lock()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestRunner -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): audit runner with savepoints, idempotency, supersession, advisory lock"
```

---

## Task 16: CLI entry (`python -m v2.audit`)

**Files:**
- Modify: `v2/audit.py`
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_audit.py`:

```python
class TestCli:
    @patch("v2.audit.run_audit")
    def test_default_is_check_mode(self, mock_run):
        from v2.audit import main
        mock_run.return_value = MagicMock(has_critical_open=False)
        rc = main(argv=[])
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["apply"] is False
        assert rc == 0

    @patch("v2.audit.run_audit")
    def test_apply_flag_passed_through(self, mock_run):
        from v2.audit import main
        mock_run.return_value = MagicMock(has_critical_open=False)
        main(argv=["--apply"])
        assert mock_run.call_args.kwargs["apply"] is True

    @patch("v2.audit.run_audit")
    def test_critical_finding_exit_1(self, mock_run):
        from v2.audit import main
        mock_run.return_value = MagicMock(has_critical_open=True)
        rc = main(argv=[])
        assert rc == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCli -v`
Expected: FAIL — `cannot import name 'main'`.

- [ ] **Step 3: Implement the CLI**

Append to `v2/audit.py`:

```python
# --- CLI ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m v2.audit",
                                     description="Self-healing audit runner")
    parser.add_argument("--apply", action="store_true",
                        help="Apply Tier-1 auto-fixes (default: propose-only)")
    parser.add_argument("--max-auto-fix", type=int, default=MAX_AUTO_FIX_DEFAULT,
                        help=f"Cap on auto-fixes per run (default {MAX_AUTO_FIX_DEFAULT})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        summary = run_audit(apply=args.apply, max_auto_fix=args.max_auto_fix)
    except Exception:
        log.exception("Audit run failed unrecoverably")
        return 2

    if summary.has_critical_open:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_audit.py::TestCli -v`
Expected: 3 PASS.

- [ ] **Step 5: Run the full test file once**

Run: `python3 -m pytest tests/v2/test_audit.py -v`
Expected: all tests pass; total ≥ 35.

- [ ] **Step 6: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(v2): audit CLI entry point with --apply/--max-auto-fix"
```

---

## Task 17: Taskfile targets

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Append targets**

Append to `Taskfile.yml` (under an existing section or create a new one):

```yaml
  # ---------------------------------------------------------------------------
  # Audit
  # ---------------------------------------------------------------------------
  audit:
    desc: Run prod auditor (propose-only)
    cmds:
      - docker compose exec -T trading python -m v2.audit

  audit:apply:
    desc: Run prod auditor and apply Tier-1 auto-fixes
    cmds:
      - docker compose exec -T trading python -m v2.audit --apply

  paper:audit:
    desc: Run paper auditor (propose-only)
    cmds:
      - docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T trading-paper python -m v2.audit

  paper:audit:apply:
    desc: Run paper auditor and apply Tier-1 auto-fixes
    cmds:
      - docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T trading-paper python -m v2.audit --apply
```

- [ ] **Step 2: Smoke-test the propose-only target on prod**

Run: `task audit`
Expected: completes; produces an `audit_runs` row and (per the prod queries
in the spec) several open findings — ORPHAN_FK_NEWS_SIGNAL,
ORPHAN_FK_THESIS, BACKFILL_GAP_30D, ATTRIBUTION_COVERAGE_LOW,
DECISIONS_NO_SIGNAL_REFS, THESES_NO_SIGNAL_REFS, plus rule-judgment
findings.

- [ ] **Step 3: Verify findings landed**

```bash
docker compose exec -T db psql -U algo -d trading \
  -c "SELECT id, status, total_findings, auto_fixed FROM audit_runs ORDER BY id DESC LIMIT 1;"
docker compose exec -T db psql -U algo -d trading \
  -c "SELECT severity, check_code, affected_count, title FROM audit_findings WHERE status='open' ORDER BY severity, tier;"
```

- [ ] **Step 4: Commit**

```bash
git add Taskfile.yml
git commit -m "feat: Taskfile targets for v2 audit (prod and paper)"
```

---

## Task 18: Dashboard `/audit` page

**Files:**
- Modify: `dashboard/app.py`, `dashboard/queries.py`
- Create: `dashboard/templates/audit.html`, `dashboard/templates/audit_finding.html`
- Test: `tests/v2/test_audit_dashboard.py` (new file)

- [ ] **Step 1: Add dashboard queries**

Append to `dashboard/queries.py`:

```python
def get_open_audit_findings():
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, audit_run_id, check_code, tier, severity, title,
                   affected_count, created_at, evidence
            FROM audit_findings WHERE status='open'
            ORDER BY
              CASE severity WHEN 'critical' THEN 0 WHEN 'warn' THEN 1 ELSE 2 END,
              tier, created_at DESC
        """)
        return cur.fetchall()


def get_audit_finding(finding_id: int):
    with get_cursor() as cur:
        cur.execute("SELECT * FROM audit_findings WHERE id=%s", (finding_id,))
        return cur.fetchone()


def get_recent_audit_runs(limit: int = 14):
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, started_at, completed_at, mode, total_findings,
                   auto_fixed, failed_checks, model,
                   input_tokens, output_tokens,
                   cache_creation_tokens, cache_read_tokens
            FROM audit_runs ORDER BY started_at DESC LIMIT %s
        """, (limit,))
        return cur.fetchall()


def update_audit_finding_status(finding_id: int, status: str, note: str | None):
    if status not in ("acknowledged", "resolved"):
        raise ValueError(f"manual status must be acknowledged or resolved, got {status!r}")
    with get_cursor() as cur:
        cur.execute(
            "UPDATE audit_findings SET status=%s, resolved_at=now(), resolved_note=%s WHERE id=%s",
            (status, note, finding_id),
        )
```

- [ ] **Step 2: Add routes to `dashboard/app.py`**

Append (after existing routes):

```python
from queries import (
    get_open_audit_findings, get_audit_finding,
    get_recent_audit_runs, update_audit_finding_status,
)


@app.route("/audit")
def audit_page():
    findings = get_open_audit_findings()
    runs = get_recent_audit_runs()
    return render_template("audit.html", findings=findings, runs=runs)


@app.route("/audit/findings/<int:finding_id>")
def audit_finding_page(finding_id):
    f = get_audit_finding(finding_id)
    if not f:
        return "Not found", 404
    return render_template("audit_finding.html", finding=f)


@app.route("/audit/findings/<int:finding_id>/status", methods=["POST"])
def audit_finding_status(finding_id):
    status = request.form.get("status")
    note = request.form.get("note")
    update_audit_finding_status(finding_id, status, note)
    return redirect(url_for("audit_page"))
```

- [ ] **Step 3: Create the templates**

`dashboard/templates/audit.html`:

```html
<!doctype html>
<html><head><title>Audit</title>
<style>
  body { font-family: monospace; max-width: 1100px; margin: 2em auto; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 0.4em 0.7em; border-bottom: 1px solid #ddd; text-align: left; vertical-align: top; }
  .sev-critical { color: #b00; font-weight: bold; }
  .sev-warn { color: #a60; }
  .sev-info { color: #888; }
  form { display: inline; }
</style></head><body>
<h1>Audit</h1>

<h2>Open findings ({{ findings|length }})</h2>
<table>
  <tr><th>Sev</th><th>Tier</th><th>Code</th><th>Title</th><th>Affected</th><th>Seen</th><th></th></tr>
  {% for f in findings %}
  <tr>
    <td class="sev-{{ f.severity }}">{{ f.severity }}</td>
    <td>{{ f.tier }}</td>
    <td>{{ f.check_code }}</td>
    <td><a href="/audit/findings/{{ f.id }}">{{ f.title }}</a></td>
    <td>{{ f.affected_count }}</td>
    <td>{{ f.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
    <td>
      <form method="post" action="/audit/findings/{{ f.id }}/status">
        <input type="hidden" name="status" value="acknowledged">
        <button type="submit">Ack</button>
      </form>
      <form method="post" action="/audit/findings/{{ f.id }}/status">
        <input type="hidden" name="status" value="resolved">
        <button type="submit">Resolve</button>
      </form>
    </td>
  </tr>
  {% endfor %}
</table>

<h2>Recent runs</h2>
<table>
  <tr><th>#</th><th>Started</th><th>Mode</th><th>Findings</th><th>Auto-fixed</th><th>Failed</th><th>Model</th><th>Input tok</th><th>Output tok</th></tr>
  {% for r in runs %}
  <tr>
    <td>{{ r.id }}</td>
    <td>{{ r.started_at.strftime('%Y-%m-%d %H:%M') }}</td>
    <td>{{ r.mode }}</td>
    <td>{{ r.total_findings }}</td>
    <td>{{ r.auto_fixed }}</td>
    <td>{{ r.failed_checks }}</td>
    <td>{{ r.model or '' }}</td>
    <td>{{ r.input_tokens or '' }}</td>
    <td>{{ r.output_tokens or '' }}</td>
  </tr>
  {% endfor %}
</table>
</body></html>
```

`dashboard/templates/audit_finding.html`:

```html
<!doctype html>
<html><head><title>{{ finding.check_code }}</title>
<style>
  body { font-family: monospace; max-width: 900px; margin: 2em auto; }
  pre { background: #f5f5f5; padding: 0.8em; overflow-x: auto; }
  .sev-critical { color: #b00; }
  .sev-warn { color: #a60; }
  .sev-info { color: #888; }
</style></head><body>
<a href="/audit">&larr; Back to audit</a>
<h1 class="sev-{{ finding.severity }}">{{ finding.title }}</h1>
<p><b>{{ finding.check_code }}</b> · tier {{ finding.tier }} · {{ finding.severity }} ·
   affected {{ finding.affected_count }} · status {{ finding.status }}</p>

<h3>Body</h3>
<pre>{{ finding.body }}</pre>

<h3>Evidence</h3>
<pre>{{ finding.evidence|tojson(indent=2) }}</pre>
</body></html>
```

- [ ] **Step 4: Write the dashboard route tests**

Create `tests/v2/test_audit_dashboard.py`:

```python
"""Tests for the dashboard /audit routes."""

import sys
from datetime import datetime
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_queries():
    """Inject a queries mock the way other dashboard tests do."""
    queries = MagicMock()
    sys.modules["queries"] = queries
    yield queries
    del sys.modules["queries"]


@pytest.fixture
def app_client(mock_queries):
    # Re-import dashboard.app each test to pick up the injected queries.
    if "dashboard.app" in sys.modules:
        del sys.modules["dashboard.app"]
    from dashboard import app as dash_app
    dash_app.app.config["TESTING"] = True
    return dash_app.app.test_client()


def test_audit_page_renders(app_client, mock_queries):
    mock_queries.get_open_audit_findings.return_value = [
        {"id": 1, "audit_run_id": 7, "check_code": "ORPHAN_FK_NEWS_SIGNAL",
         "tier": 1, "severity": "warn", "title": "11 orphan news_signal refs",
         "affected_count": 11, "created_at": datetime(2026, 5, 6, 22, 30),
         "evidence": {}},
    ]
    mock_queries.get_recent_audit_runs.return_value = []
    response = app_client.get("/audit")
    assert response.status_code == 200
    assert b"ORPHAN_FK_NEWS_SIGNAL" in response.data
    assert b"11 orphan news_signal refs" in response.data


def test_audit_finding_detail_renders(app_client, mock_queries):
    mock_queries.get_audit_finding.return_value = {
        "id": 1, "check_code": "X", "title": "t", "tier": 1, "severity": "warn",
        "body": "b", "affected_count": 0, "status": "open", "evidence": {"k": "v"},
    }
    response = app_client.get("/audit/findings/1")
    assert response.status_code == 200
    assert b"\"k\": \"v\"" in response.data


def test_audit_finding_status_post_updates(app_client, mock_queries):
    response = app_client.post("/audit/findings/1/status",
                               data={"status": "acknowledged", "note": "looking"})
    assert response.status_code == 302
    mock_queries.update_audit_finding_status.assert_called_once_with(
        1, "acknowledged", "looking"
    )
```

- [ ] **Step 5: Run dashboard tests**

Run: `python3 -m pytest tests/v2/test_audit_dashboard.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Manual smoke test**

Open `http://127.0.0.1:3000/audit` in a browser and confirm the page lists
findings and that Ack/Resolve buttons mutate state.

- [ ] **Step 7: Commit**

```bash
git add dashboard/app.py dashboard/queries.py dashboard/templates/audit.html \
        dashboard/templates/audit_finding.html tests/v2/test_audit_dashboard.py
git commit -m "feat(dashboard): /audit page surfacing audit findings + ack/resolve actions"
```

---

## Task 19: Cron documentation in README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append a section to README.md**

Append (or insert into the appropriate Operations section):

````markdown
## Self-healing audit

A daily auditor (`v2/audit.py`) detects metadata integrity issues, proposes
rule-overfitting / contradictions via a single Haiku call, and surfaces
application-level regressions. Findings appear at
[http://127.0.0.1:3000/audit](http://127.0.0.1:3000/audit). They never feed
the strategist.

### Manual run

```bash
task audit              # propose-only (recommended default)
task audit:apply        # apply Tier-1 auto-fixes (orphan FK delete, backfill re-run)
task paper:audit        # same against paper stack
```

### Daily cron

Runs propose-only. Add to host crontab (`crontab -e`):

```
MAILTO=you@example.com
30 22 * * * cd /home/jay/dev/algo && /usr/local/bin/task audit >> logs/cron-audit.log 2>&1
```

The auditor exits 0 (clean), 1 (≥1 critical finding open — MAILTO fires), or
2 (run itself failed — MAILTO fires). Advisory-lock contention exits 0.

### Enabling auto-fix in cron

Once you've reviewed propose-only runs for a week and trust the auto-fix
behavior, replace `task audit` with `task audit:apply` in the cron entry.

### CLI options

```
python -m v2.audit                    # propose-only
python -m v2.audit --apply            # apply Tier-1 auto-fixes
python -m v2.audit --max-auto-fix N   # override ceiling (default 100)
```
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README section for self-healing audit (CLI, cron, auto-fix)"
```

---

## Task 20: Migration apply test (optional, follows existing convention)

**Files:**
- Create: `tests/test_migration_025.py`

- [ ] **Step 1: Inspect existing migration tests for the project's pattern**

Run: `ls tests/test_migration*.py 2>/dev/null && head -40 $(ls tests/test_migration*.py 2>/dev/null | head -1)`

If no existing migration test pattern: skip this task. Migration was already
applied and verified manually in Task 1.

If there is a pattern: copy it for migration 025 — apply on a fresh test DB,
assert `audit_runs` and `audit_findings` tables exist with the unique
fingerprint partial index.

- [ ] **Step 2: Run the test (if added)**

Run: `python3 -m pytest tests/test_migration_025.py -v`
Expected: PASS.

- [ ] **Step 3: Commit (if added)**

```bash
git add tests/test_migration_025.py
git commit -m "test: migration 025 apply test"
```

---

## Final Verification

- [ ] **Step 1: Full test suite**

Run: `python3 -m pytest tests/v2/test_audit.py tests/v2/test_audit_dashboard.py -v`
Expected: ≥ 38 tests passing.

- [ ] **Step 2: Smoke run on prod (propose-only)**

Run: `task audit`
Expected: completes; row in `audit_runs`; multiple open findings visible at
`http://127.0.0.1:3000/audit`.

- [ ] **Step 3: Cross-check against the spec's empirical predictions**

The first prod run should surface (per spec §1 and the empirical pass on
2026-05-06): ORPHAN_FK_NEWS_SIGNAL (11), ORPHAN_FK_THESIS (7),
BACKFILL_GAP_30D (13), ATTRIBUTION_COVERAGE_LOW (only 2 categories qualify),
DECISIONS_NO_SIGNAL_REFS (13/29 with on-pb critical),
THESES_NO_SIGNAL_REFS (56/69), STAGE_FAILURE_RATE (trade_posts 50%),
STAGE_RUNNING_STALE (multiple). Plus rule-judgment findings on Rule #27
(unfalsifiable lift) at minimum. If any of these are missing, the
corresponding check has a logic bug.

- [ ] **Step 4: Decide if/when to enable cron auto-fix**

Per the spec rollout plan, propose-only runs daily for 1 week before
cron is flipped to `task audit:apply`.
