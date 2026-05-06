# Audit Finding #7 — Signal-refs Check Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine `check_decisions_missing_signal_refs` and `check_theses_missing_signal_refs` in `v2/audit.py` so adopted theses and no-thesis playbook actions are excluded from the "missing" count, severity is driven only by the `genuinely_missing` bucket, and titles/bodies/evidence accurately describe the residual problem.

**Architecture:** Two pure functions in `v2/audit.py` get rewritten SQL + reclassified output. Tests in `tests/v2/test_audit.py` are updated for the new evidence shape. No schema change. No DB writes (the checks are read-only). No call-site changes — both functions are invoked by name through the module-level check registry at line 687.

**Tech Stack:** Python 3, `psycopg2`, `pytest`, `unittest.mock.MagicMock`, PostgreSQL 16.

**Spec:** `docs/superpowers/specs/2026-05-06-audit-finding-7-signal-refs-check-refinement-design.md`

---

## File Structure

| File | Responsibility | Modification |
|---|---|---|
| `v2/audit.py` | Audit checks; `Finding` dataclass | Modify two functions: `check_decisions_missing_signal_refs` (line 393), `check_theses_missing_signal_refs` (line 445) |
| `tests/v2/test_audit.py` | Unit tests for audit checks | Update `TestCheckDecisionsMissingSignalRefs` (line 424), update `TestCheckThesesMissingSignalRefs` (line 452) |

No new files. No registry changes.

---

## Task 1: Update `TestCheckDecisionsMissingSignalRefs` fixtures and add new test cases

**Files:**
- Modify: `tests/v2/test_audit.py:424-447`

The new check returns evidence with five buckets: `total`, `has_refs`, `excluded_off_playbook`, `excluded_no_thesis`, `excluded_adoption`, `genuinely_missing`. Severity is driven by `genuinely_missing / total > 0.10`. The check returns `[]` (no finding) when `genuinely_missing == 0`.

- [ ] **Step 1: Replace the existing test class with the new fixtures and cases**

Replace lines 422–447 of `tests/v2/test_audit.py` with:

```python
# --- Decisions missing signal_refs tests (refined: bucket by source) ---

class TestCheckDecisionsMissingSignalRefs:
    def _row(self, **overrides):
        """Default fetchone shape for the refined check. All buckets present."""
        base = {
            "total": 0,
            "has_refs": 0,
            "excluded_off_playbook": 0,
            "excluded_no_thesis": 0,
            "excluded_adoption": 0,
            "genuinely_missing": 0,
        }
        base.update(overrides)
        return base

    def test_no_recent_decisions_no_finding(self):
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row()
        cur.fetchall.return_value = []
        assert check_decisions_missing_signal_refs(cur) == []

    def test_all_have_refs_no_finding(self):
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(total=20, has_refs=20)
        cur.fetchall.return_value = []
        assert check_decisions_missing_signal_refs(cur) == []

    def test_all_missing_explained_by_adoption_no_finding(self):
        """13 missing, all adopted theses → fully explained → no finding."""
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(
            total=29, has_refs=16, excluded_adoption=13, genuinely_missing=0,
        )
        cur.fetchall.return_value = []
        assert check_decisions_missing_signal_refs(cur) == []

    def test_all_missing_explained_by_no_thesis_no_finding(self):
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(
            total=29, has_refs=16, excluded_no_thesis=13, genuinely_missing=0,
        )
        cur.fetchall.return_value = []
        assert check_decisions_missing_signal_refs(cur) == []

    def test_warn_below_critical_threshold(self):
        """genuinely_missing/total = 5/100 = 5% → below 10% → warn."""
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(
            total=100, has_refs=90, excluded_adoption=3, excluded_no_thesis=2,
            genuinely_missing=5,
        )
        cur.fetchall.return_value = [{"id": i} for i in range(5)]
        findings = check_decisions_missing_signal_refs(cur)
        assert len(findings) == 1
        assert findings[0].severity == "warn"
        assert findings[0].evidence["genuinely_missing"] == 5

    def test_critical_when_genuinely_missing_above_10pct(self):
        """genuinely_missing/total = 15/100 = 15% → critical."""
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(
            total=100, has_refs=80, excluded_adoption=3, excluded_no_thesis=2,
            genuinely_missing=15,
        )
        cur.fetchall.return_value = [{"id": i} for i in range(15)]
        findings = check_decisions_missing_signal_refs(cur)
        assert findings[0].severity == "critical"
        assert findings[0].evidence["genuinely_missing"] == 15

    def test_evidence_buckets_all_present(self):
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(
            total=29, has_refs=16, excluded_off_playbook=2,
            excluded_no_thesis=4, excluded_adoption=4, genuinely_missing=3,
        )
        cur.fetchall.return_value = [{"id": 100}, {"id": 101}, {"id": 102}]
        findings = check_decisions_missing_signal_refs(cur)
        ev = findings[0].evidence
        assert ev["total"] == 29
        assert ev["has_refs"] == 16
        assert ev["excluded_off_playbook"] == 2
        assert ev["excluded_no_thesis"] == 4
        assert ev["excluded_adoption"] == 4
        assert ev["genuinely_missing"] == 3
        assert ev["decision_ids"] == [100, 101, 102]

    def test_decision_ids_are_genuinely_missing_only(self):
        """The id list query should return only the genuinely_missing bucket."""
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(
            total=29, has_refs=16, excluded_adoption=10, genuinely_missing=3,
        )
        cur.fetchall.return_value = [{"id": 270}, {"id": 271}, {"id": 290}]
        findings = check_decisions_missing_signal_refs(cur)
        # The check must run a second query to fetch ids; assert we got 3, not 13.
        assert findings[0].evidence["decision_ids"] == [270, 271, 290]
        assert len(findings[0].evidence["decision_ids"]) == 3

    def test_check_code_unchanged(self):
        """Same check_code preserves fingerprint history & supersession."""
        from v2.audit import check_decisions_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = self._row(
            total=10, has_refs=5, genuinely_missing=5,
        )
        cur.fetchall.return_value = [{"id": i} for i in range(5)]
        findings = check_decisions_missing_signal_refs(cur)
        assert findings[0].check_code == "DECISIONS_NO_SIGNAL_REFS"
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
docker compose exec -T trading pytest tests/v2/test_audit.py::TestCheckDecisionsMissingSignalRefs -v
```

Expected: most tests FAIL (because the implementation still emits the old keys: `missing`, `on_pb_missing`). Read the failures to confirm they're failing on the right thing (`KeyError` on `genuinely_missing`, or wrong severity, or wrong evidence shape).

- [ ] **Step 3: Commit (test-first)**

```bash
git add tests/v2/test_audit.py
git commit -m "$(cat <<'EOF'
test(audit): rewrite TestCheckDecisionsMissingSignalRefs for bucket-aware check

Adds 5-bucket fixtures (has_refs, excluded_off_playbook,
excluded_no_thesis, excluded_adoption, genuinely_missing). New cases
verify adoption-only and no-thesis-only gaps emit no finding,
severity is driven by genuinely_missing only, and decision_ids in
evidence comes from the genuinely_missing bucket. Tests fail until
the implementation lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement the refined `check_decisions_missing_signal_refs`

**Files:**
- Modify: `v2/audit.py:391-440`

- [ ] **Step 1: Replace the function body with the bucket-aware version**

Replace lines 391–440 of `v2/audit.py` with:

```python
# --- Tier 3: decisions missing signal_refs --------------------------------

def check_decisions_missing_signal_refs(cur) -> list[Finding]:
    """Classify each recent buy/sell decision into one of five buckets and
    flag only when `genuinely_missing` is non-zero. The other buckets are
    legitimate gaps:
      - excluded_off_playbook : is_off_playbook=true
      - excluded_no_thesis    : on-playbook but playbook_action.thesis_id is null
      - excluded_adoption     : on-playbook, thesis exists, source='adoption'
    `genuinely_missing` is on-playbook decisions whose underlying
    ideation thesis (source != 'adoption') has no thesis_signals.
    Severity = critical when genuinely_missing/total > 10%.
    """
    cur.execute("""
        WITH classified AS (
          SELECT
            d.id,
            CASE
              WHEN ds.decision_id IS NOT NULL                       THEN 'has_refs'
              WHEN COALESCE(d.is_off_playbook, false)               THEN 'excluded_off_playbook'
              WHEN d.playbook_action_id IS NULL OR pa.thesis_id IS NULL
                                                                    THEN 'excluded_no_thesis'
              WHEN t.source = 'adoption'                            THEN 'excluded_adoption'
              ELSE 'genuinely_missing'
            END AS bucket
          FROM decisions d
          LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
            ON ds.decision_id = d.id
          LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
          LEFT JOIN theses t            ON t.id  = pa.thesis_id
          WHERE d.action IN ('buy','sell')
            AND d.date > now()::date - 30
        )
        SELECT
          COUNT(*)                                                 AS total,
          COUNT(*) FILTER (WHERE bucket='has_refs')                AS has_refs,
          COUNT(*) FILTER (WHERE bucket='excluded_off_playbook')   AS excluded_off_playbook,
          COUNT(*) FILTER (WHERE bucket='excluded_no_thesis')      AS excluded_no_thesis,
          COUNT(*) FILTER (WHERE bucket='excluded_adoption')       AS excluded_adoption,
          COUNT(*) FILTER (WHERE bucket='genuinely_missing')       AS genuinely_missing
        FROM classified
    """)
    summary = cur.fetchone()
    if not summary["genuinely_missing"]:
        return []

    cur.execute("""
        SELECT d.id
          FROM decisions d
          LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
            ON ds.decision_id = d.id
          LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
          LEFT JOIN theses t            ON t.id  = pa.thesis_id
         WHERE d.action IN ('buy','sell')
           AND d.date > now()::date - 30
           AND ds.decision_id IS NULL
           AND COALESCE(d.is_off_playbook, false) = false
           AND d.playbook_action_id IS NOT NULL
           AND pa.thesis_id IS NOT NULL
           AND COALESCE(t.source, '') <> 'adoption'
         ORDER BY d.id
    """)
    ids = [r["id"] for r in cur.fetchall()]

    share = (summary["genuinely_missing"] / summary["total"]) if summary["total"] else 0
    severity = "critical" if share > 0.10 else "warn"

    return [Finding(
        check_code="DECISIONS_NO_SIGNAL_REFS",
        tier=3, severity=severity,
        title=(f"{summary['genuinely_missing']} of {summary['total']} recent "
               f"buy/sell decisions derived from ideation theses have no "
               f"signal_refs ({summary['excluded_adoption']} adopted, "
               f"{summary['excluded_no_thesis']} no-thesis excluded)"),
        body=("Strategist created theses without citing signals via "
              "`signal_refs`/`add_signal_refs`, so executor decisions "
              "derived from them have nothing to attribute. Fix in "
              "v2/ideation_claude.py prompt or the create_thesis / "
              "update_thesis tool-call validation."),
        affected_count=summary["genuinely_missing"],
        evidence={
            "total": summary["total"],
            "has_refs": summary["has_refs"],
            "excluded_off_playbook": summary["excluded_off_playbook"],
            "excluded_no_thesis": summary["excluded_no_thesis"],
            "excluded_adoption": summary["excluded_adoption"],
            "genuinely_missing": summary["genuinely_missing"],
            "decision_ids": ids,
        },
        auto_fix=None,
    )]
```

- [ ] **Step 2: Run the test class to verify all pass**

```bash
docker compose exec -T trading pytest tests/v2/test_audit.py::TestCheckDecisionsMissingSignalRefs -v
```

Expected: all 9 tests in the class PASS.

- [ ] **Step 3: Commit**

```bash
git add v2/audit.py
git commit -m "$(cat <<'EOF'
feat(audit): bucket DECISIONS_NO_SIGNAL_REFS by source, fire only on real gaps

Previously the check counted all missing signal_refs together and
blamed the executor wiring. Adopted-thesis decisions and no-thesis
playbook actions are legitimate gaps and inflated severity to
critical when the wiring was actually fine.

Reclassifies each decision into has_refs / excluded_off_playbook /
excluded_no_thesis / excluded_adoption / genuinely_missing. Severity
now driven only by genuinely_missing/total > 10%, so the finding
fires when the strategist has actually failed to cite signals on
its own ideation theses. Title and body updated to point at the
strategist (the real layer at fault). decision_ids in evidence is
now the genuinely_missing bucket only — the IDs worth investigating.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update `TestCheckThesesMissingSignalRefs` for the adoption exclusion

**Files:**
- Modify: `tests/v2/test_audit.py:450-474`

The sibling check needs to filter `source != 'adoption'` from both the count query and the id query. The mock-cursor tests can't catch the SQL change directly, so we add an assertion against `cur.execute.call_args_list` to verify the filter is in the SQL string.

- [ ] **Step 1: Add a new test case verifying the SQL filter**

Append to the `TestCheckThesesMissingSignalRefs` class (after line 474):

```python
    def test_sql_filters_out_adoption_source(self):
        """The check must exclude theses with source='adoption' so
        adopted positions don't count against strategist citations."""
        from v2.audit import check_theses_missing_signal_refs
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 0, "missing": 0}
        cur.fetchall.return_value = []
        check_theses_missing_signal_refs(cur)
        # Both the count query and (if reached) the id query must filter adoption.
        executed_sqls = [call.args[0] for call in cur.execute.call_args_list]
        assert any("source" in sql.lower() and "adoption" in sql.lower()
                   for sql in executed_sqls), (
            f"Expected at least one SQL to filter source='adoption'. "
            f"Got: {executed_sqls!r}"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
docker compose exec -T trading pytest tests/v2/test_audit.py::TestCheckThesesMissingSignalRefs::test_sql_filters_out_adoption_source -v
```

Expected: FAIL with the AssertionError "Expected at least one SQL to filter source='adoption'".

- [ ] **Step 3: Commit (test-first)**

```bash
git add tests/v2/test_audit.py
git commit -m "$(cat <<'EOF'
test(audit): require check_theses_missing_signal_refs to exclude adoption source

Adopted theses legitimately have no signals; counting them against
the strategist's citation rate inflates the THESES_NO_SIGNAL_REFS
finding (currently 56/69 = 81% in prod, mostly adopted). Test
asserts the executed SQL contains a source/adoption filter. Fails
until the implementation lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement the adoption exclusion in `check_theses_missing_signal_refs`

**Files:**
- Modify: `v2/audit.py:443-477`

- [ ] **Step 1: Add `source != 'adoption'` to both queries**

Replace lines 443–477 of `v2/audit.py` with:

```python
# --- Tier 3: theses missing signal_refs (Rule #6 drift) -------------------

def check_theses_missing_signal_refs(cur) -> list[Finding]:
    """Adopted theses (`source='adoption'`) legitimately lack signal_refs;
    only count strategist-created theses (typically `source='claude_ideation'`)
    against the citation rate.
    """
    cur.execute("""
        SELECT
          COUNT(*) AS total,
          COUNT(*) FILTER (WHERE ts.thesis_id IS NULL) AS missing
        FROM theses t
        LEFT JOIN (SELECT DISTINCT thesis_id FROM thesis_signals) ts
          ON ts.thesis_id = t.id
        WHERE t.created_at > now() - interval '30 days'
          AND COALESCE(t.source, '') <> 'adoption'
    """)
    s = cur.fetchone()
    if not s["total"] or s["missing"] / s["total"] <= 0.25:
        return []

    cur.execute("""
        SELECT t.id FROM theses t
        LEFT JOIN (SELECT DISTINCT thesis_id FROM thesis_signals) ts
          ON ts.thesis_id = t.id
        WHERE t.created_at > now() - interval '30 days'
          AND COALESCE(t.source, '') <> 'adoption'
          AND ts.thesis_id IS NULL
        ORDER BY t.id
    """)
    ids = [r["id"] for r in cur.fetchall()]
    return [Finding(
        check_code="THESES_NO_SIGNAL_REFS",
        tier=3, severity="warn",
        title=f"{s['missing']} of {s['total']} recent ideation theses have no signal_refs",
        body=("Strategist is creating theses without citing signals, violating "
              "Rule #6 from the 2026-05-02 wiring fix. Without citations, "
              "downstream attribution receives nothing to score. Adopted "
              "theses are excluded from this count by design."),
        affected_count=s["missing"],
        evidence={"total": s["total"], "missing": s["missing"], "thesis_ids": ids},
        auto_fix=None,
    )]
```

- [ ] **Step 2: Run the full test class**

```bash
docker compose exec -T trading pytest tests/v2/test_audit.py::TestCheckThesesMissingSignalRefs -v
```

Expected: all 4 tests PASS (the 3 existing + the new one).

- [ ] **Step 3: Commit**

```bash
git add v2/audit.py
git commit -m "$(cat <<'EOF'
feat(audit): exclude source='adoption' from THESES_NO_SIGNAL_REFS check

Adopted theses are added to the system without signal citations by
design; counting them against the strategist's rate misattributes
adoption-volume to a strategist bug. Adds COALESCE(t.source,'')<>
'adoption' to both the count query and the id query.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Run the full audit test file as a regression check

**Files:** None modified.

- [ ] **Step 1: Run all audit tests**

```bash
docker compose exec -T trading pytest tests/v2/test_audit.py -v
```

Expected: all tests PASS. If any unrelated test fails, investigate (likely a fixture interaction). Do not proceed until green.

- [ ] **Step 2: Run the broader v2 test suite as a sanity check**

```bash
docker compose exec -T trading pytest tests/v2/ -q
```

Expected: full v2 suite green. Audit module changes shouldn't ripple, but the registry at `v2/audit.py:687` is exercised by integration-style tests; this catches any signature drift.

- [ ] **Step 3: No commit**

This task only verifies; nothing changed.

---

## Task 6: Run the audit against prod and verify the finding refines correctly

**Files:** None modified.

- [ ] **Step 1: Run the audit against prod**

```bash
task audit
```

Expected: `task audit` completes; the on-screen output shows either:
- `DECISIONS_NO_SIGNAL_REFS` is no longer present (full count was explained by adoption + no-thesis), OR
- `DECISIONS_NO_SIGNAL_REFS` appears with a smaller `genuinely_missing` count and possibly downgraded `severity` (warn instead of critical).

- [ ] **Step 2: Inspect the new finding state in the DB**

```bash
docker compose exec -T db psql -U algo -d trading -c "
SELECT id, check_code, severity, status, title, evidence
  FROM audit_findings
 WHERE check_code IN ('DECISIONS_NO_SIGNAL_REFS', 'THESES_NO_SIGNAL_REFS')
 ORDER BY id DESC LIMIT 4;
"
```

Expected:
- The original finding #7 has `status='superseded'` (auto-flipped because the fingerprint changed when the evidence keys changed).
- The original finding #8 (THESES_NO_SIGNAL_REFS) is also superseded.
- A new row may exist for each check with the refined evidence shape — inspect to confirm `genuinely_missing` is materially smaller than 13, and `excluded_adoption` + `excluded_no_thesis` together explain most of the original 13.

- [ ] **Step 3: Capture the residual count for the follow-up spec**

If `genuinely_missing > 0` after the refined audit, that's the real strategist citation bug. Record the number and the `decision_ids` from the new evidence — these become the input to a separate follow-up spec to chase the strategist behavior. Do not file it as part of this plan.

If `genuinely_missing == 0`, the original finding was 100% explained by legitimate gaps — no follow-up needed.

- [ ] **Step 4: No commit**

Verification only.

---

## Done

After Task 6:
- Both audit checks correctly exclude adopted theses from their counts.
- Finding #7 either disappears or drops to `warn` with a smaller `genuinely_missing` count.
- Finding #8 likely drops materially (most of 56/69 was probably adoption-source theses).
- Any residual `genuinely_missing > 0` is the genuine strategist-citation bug, captured for a separate follow-up spec.
- Test suite green; no regressions in `tests/v2/`.
