# Opus Audit Ideation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `v2/audit.py` with two daily Opus 4.7 checks that propose audit-gap improvements and app/feature improvements, dedup by topic_slug across days, and file Jira tickets via REST.

**Architecture:** Add two new check functions to the existing audit `CHECKS` list. Persist per-LLM-call token usage in a new `audit_llm_calls` table. Introduce a thin Jira REST client in `v2/audit_jira.py` gated by env vars and a `--file-jira` CLI flag. Ship in three independently mergeable commits: (1) schema + accounting, (2) Opus checks without Jira, (3) Jira filing.

**Tech Stack:** Python 3.10, psycopg2, anthropic SDK, requests (already in deps), pytest, PostgreSQL 16, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-05-12-opus-audit-ideation-design.md`

---

## Test Conventions (IMPORTANT — read before writing any test)

`tests/v2/test_audit.py` and `tests/v2/test_audit_dashboard.py` follow a strict
mock-everything pattern. **No test in this plan should touch a real Postgres
database.** The test code shown in each task's "Step 1: Write the failing test"
demonstrates intent — adapt the mechanism to match these conventions:

1. **Inside a check function** (e.g. `check_audit_llm_cost_trend`):
   ```python
   cur = MagicMock()
   cur.fetchall.side_effect = [<query1_rows>, <query2_rows>, ...]
   cur.fetchone.return_value = <single_row_or_None>
   findings = check_function(cur)
   ```
   The check receives the mock `cur` directly — no `get_cursor` involved.

2. **DB helpers in `v2/database/trading_db.py`** (e.g. `insert_audit_llm_call`):
   ```python
   @patch("v2.database.trading_db.get_cursor")
   def test_insert_audit_llm_call_writes_row(self, mock_get_cursor):
       cur = MagicMock()
       cur.fetchone.return_value = {"id": 42}
       mock_get_cursor.return_value.__enter__.return_value = cur
       from v2.database.trading_db import insert_audit_llm_call
       row_id = insert_audit_llm_call(...)
       assert row_id == 42
       sql = cur.execute.call_args[0][0]
       assert "audit_llm_calls" in sql
   ```

3. **Runner integration tests** (e.g. testing `run_audit` end-to-end): patch
   every trading_db helper the runner imports, plus `v2.audit.get_cursor`. See
   `TestRunner.test_per_check_isolation` at `tests/v2/test_audit.py:1137-1166`
   for the canonical pattern (7 patches stacked). Assertions go on
   `mock_insert_audit_finding.call_args_list` to verify what *would* have been
   written. Assertions on `mock_insert_audit_llm_call` cover LLM-call rows.

4. **Class organization:** Group new tests in `class TestCheckAuditLlmCostTrend`,
   `class TestCheckAuditGapsOpus`, `class TestCheckAppImprovementsOpus`,
   `class TestOpusFingerprintHelpers`, etc. — one class per unit. See the
   existing class layout (`TestCheckRuleJudgment` at line 586, `TestRunner` at
   line 1137) for shape.

5. **Jira tests** (`tests/v2/test_audit_jira.py`): mock `v2.audit_jira.requests.get`
   and `v2.audit_jira.requests.post` with `unittest.mock.patch`. No real HTTP.

When this plan's "Step 1: Write the failing test" snippet uses real DB imports
or fixtures, treat it as a behavior spec and reshape the test to match the
patterns above. Behavior under test stays identical; mechanism follows the
project convention.

---

## File Structure

**Commit 1 — schema + LLM call accounting**

- Create: `db/init/029_audit_llm_calls.sql` — new table.
- Modify: `v2/database/trading_db.py` — add `insert_audit_llm_call`, `cost_trend_by_purpose`.
- Modify: `v2/audit.py` — wire existing `check_rule_judgment` to write a row; add new `check_audit_llm_cost_trend`; remove writes to deprecated `audit_runs.*` token columns (leave columns in place).
- Modify: `tests/v2/test_audit.py` — tests for new helpers + new check.

**Commit 2 — Opus checks, no Jira**

- Modify: `v2/audit.py` — add `OPUS_IDEATION_MODEL`, prompt builders for gaps + improvements, `_call_opus_ideation` LLM helper, two new check functions, append to `CHECKS`.
- Modify: `tests/v2/test_audit.py` — tests for both new checks.
- Modify: `CLAUDE.md` — document new env var `ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS`.

**Commit 3 — Jira filing**

- Create: `v2/audit_jira.py` — REST client + `file_jira_ticket` function.
- Modify: `v2/audit.py` — hook Jira filing into the runner; add `--file-jira` CLI flag.
- Create: `tests/v2/test_audit_jira.py` — full coverage of filing flow.
- Modify: `tests/v2/test_audit.py` — runner-integration test that filing is called.
- Modify: `dashboard/templates/audit_finding.html` — render `evidence.jira` block.
- Modify: `tests/v2/test_audit_dashboard.py` — template renders Jira link.
- Modify: `CLAUDE.md` — document Jira env vars + `--file-jira` flag.

---

## Commit 1: Schema + LLM Call Accounting

### Task 1.1: Create the migration

**Files:**
- Create: `db/init/029_audit_llm_calls.sql`

- [ ] **Step 1: Write the migration**

```sql
-- db/init/029_audit_llm_calls.sql
-- Per-LLM-call token accounting for audit runs.
-- See docs/superpowers/specs/2026-05-12-opus-audit-ideation-design.md

CREATE TABLE IF NOT EXISTS audit_llm_calls (
    id                      SERIAL PRIMARY KEY,
    audit_run_id            INTEGER NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
    purpose                 VARCHAR(64) NOT NULL,
    model                   VARCHAR(128) NOT NULL,
    input_tokens            INTEGER NOT NULL DEFAULT 0,
    output_tokens           INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens       INTEGER NOT NULL DEFAULT 0,
    latency_ms              INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_llm_calls_run     ON audit_llm_calls(audit_run_id);
CREATE INDEX IF NOT EXISTS idx_audit_llm_calls_purpose ON audit_llm_calls(purpose, created_at);
```

- [ ] **Step 2: Apply the migration**

Run: `docker compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB -f /docker-entrypoint-initdb.d/029_audit_llm_calls.sql`

If that fails because the file isn't in the container yet (init scripts only run on first boot), apply by piping:

Run: `docker compose exec -T db psql -U $POSTGRES_USER -d $POSTGRES_DB < db/init/029_audit_llm_calls.sql`

Expected: `CREATE TABLE` and two `CREATE INDEX` lines.

- [ ] **Step 3: Verify table exists**

Run: `docker compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\d audit_llm_calls"`
Expected: Table listing with all columns above.

- [ ] **Step 4: Commit**

```bash
git add db/init/029_audit_llm_calls.sql
git commit -m "feat(audit): add audit_llm_calls table for per-call token accounting"
```

---

### Task 1.2: Add `insert_audit_llm_call` helper

**Files:**
- Modify: `v2/database/trading_db.py` (add after `finalize_audit_run`, around line 1160)
- Test: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_audit.py` (near other DB-helper tests if present; otherwise create a new section at the bottom):

```python
def test_insert_audit_llm_call_writes_row():
    from v2.database.trading_db import insert_audit_run, insert_audit_llm_call
    from v2.database.connection import get_cursor

    run_id = insert_audit_run(mode="check")
    insert_audit_llm_call(
        audit_run_id=run_id,
        purpose="rule_judgment",
        model="claude-haiku-4-5-20251001",
        input_tokens=1000,
        output_tokens=200,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        latency_ms=2345,
    )
    with get_cursor() as cur:
        cur.execute(
            "SELECT purpose, model, input_tokens, output_tokens, latency_ms "
            "FROM audit_llm_calls WHERE audit_run_id=%s",
            (run_id,),
        )
        row = cur.fetchone()
    assert row["purpose"] == "rule_judgment"
    assert row["model"] == "claude-haiku-4-5-20251001"
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 200
    assert row["latency_ms"] == 2345
```

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_insert_audit_llm_call_writes_row -v`
Expected: FAIL — `ImportError: cannot import name 'insert_audit_llm_call'`.

- [ ] **Step 3: Add the helper**

In `v2/database/trading_db.py`, insert after `finalize_audit_run`:

```python
def insert_audit_llm_call(
    *,
    audit_run_id: int,
    purpose: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    latency_ms: int | None = None,
) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO audit_llm_calls
                (audit_run_id, purpose, model, input_tokens, output_tokens,
                 cache_creation_tokens, cache_read_tokens, latency_ms)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (audit_run_id, purpose, model, input_tokens, output_tokens,
             cache_creation_tokens, cache_read_tokens, latency_ms),
        )
        return cur.fetchone()["id"]
```

- [ ] **Step 4: Run test, verify it passes**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_insert_audit_llm_call_writes_row -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_audit.py
git commit -m "feat(audit): add insert_audit_llm_call helper"
```

---

### Task 1.3: Wire `check_rule_judgment` to record its LLM call

**Files:**
- Modify: `v2/audit.py:1501-1596` (the `run_audit` function)
- Modify: `v2/audit.py:1355-1430` (`check_rule_judgment`)

The cleanest place for the write is the `run_audit` runner where we already extract `rule_judgment_usage`. We need (a) latency tracking around the LLM call, (b) the write itself.

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_audit.py`:

```python
def test_run_audit_records_rule_judgment_llm_call(monkeypatch):
    """Existing rule_judgment check now writes to audit_llm_calls."""
    from v2 import audit
    from v2.database.connection import get_cursor

    # Stub the LLM call to avoid network + return canned usage
    def fake_call(prompt):
        usage = {"input_tokens": 1234, "output_tokens": 56,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0}
        return ({"findings": []}, usage)
    monkeypatch.setattr(audit, "_call_rule_judgment_llm", fake_call)

    # Make sure rule_judgment runs by inserting at least one active rule
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO strategy_rules (rule_text, status) "
            "VALUES (%s, 'active') ON CONFLICT DO NOTHING",
            ("Test rule for LLM call accounting",),
        )

    summary = audit.run_audit(apply=False)

    with get_cursor() as cur:
        cur.execute(
            "SELECT purpose, model, input_tokens, output_tokens "
            "FROM audit_llm_calls WHERE audit_run_id=%s",
            (summary.run_id,),
        )
        rows = cur.fetchall()

    assert any(r["purpose"] == "rule_judgment" for r in rows)
    rj_row = next(r for r in rows if r["purpose"] == "rule_judgment")
    assert rj_row["model"] == audit.RULE_JUDGMENT_MODEL
    assert rj_row["input_tokens"] == 1234
    assert rj_row["output_tokens"] == 56
```

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_run_audit_records_rule_judgment_llm_call -v`
Expected: FAIL — no rows in `audit_llm_calls`.

- [ ] **Step 3: Update `_call_rule_judgment_llm` to record latency**

In `v2/audit.py`, replace the existing `_call_rule_judgment_llm` (around line 600) so it also returns latency:

```python
def _call_rule_judgment_llm(prompt: str) -> tuple[dict, dict]:
    """Returns (parsed_json, usage_dict). Separate function for easy stubbing.

    usage_dict includes 'latency_ms' alongside the four token fields.
    """
    import time
    from anthropic import Anthropic
    import os

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.monotonic()
    response = client.messages.create(
        model=RULE_JUDGMENT_MODEL,
        max_tokens=RULE_JUDGMENT_MAX_TOKENS,
        system=RULE_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    parsed = _extract_json(text)
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
        "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "latency_ms": latency_ms,
    }
    return parsed, usage
```

- [ ] **Step 4: Update `run_audit` to write the row**

In `v2/audit.py`, in `run_audit`, immediately after the existing block that updates `rule_judgment_usage`, add the insert. Find this block (around line 1530):

```python
                if check.__name__ == "check_rule_judgment":
                    rule_judgment_usage = get_last_rule_judgment_usage()
```

Replace with:

```python
                if check.__name__ == "check_rule_judgment":
                    rule_judgment_usage = get_last_rule_judgment_usage()
                    if rule_judgment_usage:
                        from v2.database.trading_db import insert_audit_llm_call
                        insert_audit_llm_call(
                            audit_run_id=run_id,
                            purpose="rule_judgment",
                            model=RULE_JUDGMENT_MODEL,
                            input_tokens=rule_judgment_usage.get("input_tokens", 0),
                            output_tokens=rule_judgment_usage.get("output_tokens", 0),
                            cache_creation_tokens=rule_judgment_usage.get("cache_creation_tokens", 0),
                            cache_read_tokens=rule_judgment_usage.get("cache_read_tokens", 0),
                            latency_ms=rule_judgment_usage.get("latency_ms"),
                        )
```

- [ ] **Step 5: Stop writing deprecated `audit_runs` token columns**

In `v2/audit.py`, find `finalize_audit_run` call in `run_audit` (around line 1576). Replace this:

```python
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
```

With:

```python
        finalize_audit_run(
            run_id=run_id,
            total_findings=emitted,
            auto_fixed=auto_fixed,
            failed_checks=failed_checks,
            # LLM accounting moved to audit_llm_calls table; leave legacy
            # token columns NULL on new runs. See spec 2026-05-12.
        )
```

- [ ] **Step 6: Run the test, verify it passes**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_run_audit_records_rule_judgment_llm_call -v`
Expected: PASS.

- [ ] **Step 7: Run the full audit test file to confirm nothing else broke**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -v`
Expected: All pre-existing tests still pass (if any tests assert the old `audit_runs.input_tokens` column, they'll need updating — see Task 1.4).

- [ ] **Step 8: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(audit): record rule_judgment LLM usage in audit_llm_calls"
```

---

### Task 1.4: Fix any tests that asserted the old `audit_runs` token columns

This is a clean-up safety step. If Task 1.3 step 7 turned up failures referencing `audit_runs.input_tokens` etc., those tests need to read from `audit_llm_calls` instead.

- [ ] **Step 1: Find references**

Run: `grep -nE "audit_runs.*(input_tokens|output_tokens|cache_)" tests/`
Expected: zero matches, OR a list of test files needing updates.

- [ ] **Step 2: Update each test (if any)**

For each match, change the assertion to query `audit_llm_calls` for the appropriate `purpose`. If no matches were found, skip to step 3.

- [ ] **Step 3: Run the audit test file**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -v`
Expected: All pass.

- [ ] **Step 4: Commit (if any changes were made)**

```bash
git add tests/
git commit -m "test(audit): migrate token-usage assertions to audit_llm_calls"
```

If step 1 found no matches, skip the commit.

---

### Task 1.5: Add `check_audit_llm_cost_trend`

Compares per-`purpose` token totals last 7d vs prior 7d; flag any purpose whose recent usage is ≥2x prior (and prior was non-zero).

**Files:**
- Modify: `v2/audit.py` — add the new check function and append to `CHECKS`.
- Modify: `tests/v2/test_audit.py` — test the new check.

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_audit.py`:

```python
def test_check_audit_llm_cost_trend_flags_doubled_purpose(monkeypatch):
    """A purpose whose recent 7d tokens are >=2x prior 7d yields a finding."""
    from v2 import audit
    from v2.database.connection import get_cursor
    from v2.database.trading_db import insert_audit_run, insert_audit_llm_call

    # Prior 7d: 100 tokens for 'audit_gaps'
    run_old = insert_audit_run(mode="check")
    with get_cursor() as cur:
        cur.execute(
            "UPDATE audit_runs SET started_at = now() - interval '10 days' WHERE id=%s",
            (run_old,),
        )
    insert_audit_llm_call(
        audit_run_id=run_old, purpose="audit_gaps",
        model="claude-opus-4-7", input_tokens=100, output_tokens=0,
    )
    with get_cursor() as cur:
        cur.execute(
            "UPDATE audit_llm_calls SET created_at = now() - interval '10 days' "
            "WHERE audit_run_id=%s",
            (run_old,),
        )

    # Recent 7d: 500 tokens for 'audit_gaps' (5x)
    run_new = insert_audit_run(mode="check")
    insert_audit_llm_call(
        audit_run_id=run_new, purpose="audit_gaps",
        model="claude-opus-4-7", input_tokens=500, output_tokens=0,
    )

    with get_cursor() as cur:
        findings = audit.check_audit_llm_cost_trend(cur)

    codes = [f.check_code for f in findings]
    assert "AUDIT_LLM_COST_TREND_SPIKE" in codes
    spike_finding = next(f for f in findings if f.check_code == "AUDIT_LLM_COST_TREND_SPIKE")
    purposes = [p["purpose"] for p in spike_finding.evidence["purposes"]]
    assert "audit_gaps" in purposes
```

- [ ] **Step 2: Run test, verify it fails**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_check_audit_llm_cost_trend_flags_doubled_purpose -v`
Expected: FAIL — `AttributeError: module 'v2.audit' has no attribute 'check_audit_llm_cost_trend'`.

- [ ] **Step 3: Add the check**

In `v2/audit.py`, add this function near the existing `check_cost_trend` (around line 397):

```python
def check_audit_llm_cost_trend(cur) -> list[Finding]:
    """Per-purpose 7d-vs-prior-7d audit LLM token usage. Flag >=2x growth."""
    cur.execute("""
        WITH recent AS (
            SELECT purpose,
                   SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)
                      +COALESCE(cache_creation_tokens,0)
                      +COALESCE(cache_read_tokens,0)) AS tok
            FROM audit_llm_calls
            WHERE created_at > now() - interval '7 days'
            GROUP BY purpose
        ),
        prior AS (
            SELECT purpose,
                   SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)
                      +COALESCE(cache_creation_tokens,0)
                      +COALESCE(cache_read_tokens,0)) AS tok
            FROM audit_llm_calls
            WHERE created_at > now() - interval '14 days'
              AND created_at <= now() - interval '7 days'
            GROUP BY purpose
        )
        SELECT COALESCE(r.purpose, p.purpose) AS purpose,
               COALESCE(r.tok, 0) AS recent_tok,
               COALESCE(p.tok, 0) AS prior_tok
        FROM recent r FULL OUTER JOIN prior p ON r.purpose = p.purpose
    """)
    spikes = []
    for r in cur.fetchall():
        if not r["prior_tok"]:
            continue
        if r["recent_tok"] >= 2 * r["prior_tok"]:
            spikes.append({
                "purpose": r["purpose"],
                "recent_tok": int(r["recent_tok"]),
                "prior_tok": int(r["prior_tok"]),
                "ratio": round(r["recent_tok"] / r["prior_tok"], 2),
            })
    if not spikes:
        return []
    return [Finding(
        check_code="AUDIT_LLM_COST_TREND_SPIKE", tier=3, severity="info",
        title=f"{len(spikes)} audit LLM purpose(s) with token usage >=2x prior 7-day window",
        body="Per-purpose 7-day rolling audit LLM token totals doubled vs. prior 7-day window.",
        affected_count=len(spikes),
        evidence={"purposes": spikes},
        auto_fix=None,
    )]
```

- [ ] **Step 4: Register the check**

In `v2/audit.py`, append to the `CHECKS` list (around line 1473), placing it adjacent to `check_cost_trend`:

```python
    "check_audit_llm_cost_trend",
```

The list should now include `"check_cost_trend"` immediately followed by `"check_audit_llm_cost_trend"`.

- [ ] **Step 5: Run test, verify it passes**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_check_audit_llm_cost_trend_flags_doubled_purpose -v`
Expected: PASS.

- [ ] **Step 6: Run the full audit test file**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(audit): add check_audit_llm_cost_trend"
```

---

## Commit 2: Opus Ideation Checks (no Jira yet)

### Task 2.1: Add Opus model constants + shared LLM helper

**Files:**
- Modify: `v2/audit.py` — constants and `_call_opus_ideation`.

- [ ] **Step 1: Add constants**

In `v2/audit.py`, near the existing `RULE_JUDGMENT_MODEL` constant (around line 23), add:

```python
OPUS_IDEATION_MODEL = "claude-opus-4-7"
OPUS_IDEATION_MAX_TOKENS = 4000
OPUS_INPUT_TOKEN_CAP_DEFAULT = 60_000
```

- [ ] **Step 2: Add the shared LLM call helper**

In `v2/audit.py`, near `_call_rule_judgment_llm` (around line 600), add a sibling helper:

```python
def _call_opus_ideation(system: str, prompt: str) -> tuple[dict, dict]:
    """Single Opus call returning (parsed_json, usage_with_latency_ms).

    Caches `system` (via cache_control on the system block) so repeated daily
    runs amortize the system-prompt cost.
    """
    import time
    from anthropic import Anthropic
    import os

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.monotonic()
    response = client.messages.create(
        model=OPUS_IDEATION_MODEL,
        max_tokens=OPUS_IDEATION_MAX_TOKENS,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    parsed = _extract_json(text)
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
        "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "latency_ms": latency_ms,
    }
    return parsed, usage
```

- [ ] **Step 3: Verify the module still imports**

Run: `docker compose exec trading python -c "import v2.audit; print(v2.audit.OPUS_IDEATION_MODEL)"`
Expected: `claude-opus-4-7`

- [ ] **Step 4: Commit**

```bash
git add v2/audit.py
git commit -m "feat(audit): add Opus model constants and _call_opus_ideation helper"
```

---

### Task 2.2: Add `_opus_finding_from_json` parser + fingerprint helper

The two new checks share parsing logic. Build the helper once.

**Files:**
- Modify: `v2/audit.py` — parser + fingerprint helper.
- Modify: `tests/v2/test_audit.py` — tests.

- [ ] **Step 1: Write the failing tests**

Add to `tests/v2/test_audit.py`:

```python
def test_opus_topic_slug_fingerprint_stable():
    from v2.audit import _opus_topic_fingerprint
    fp_a = _opus_topic_fingerprint("AUDIT_GAP", "Add Regime Detector")
    fp_b = _opus_topic_fingerprint("AUDIT_GAP", "add-regime-detector")
    fp_c = _opus_topic_fingerprint("AUDIT_GAP", "  Add Regime Detector!! ")
    assert fp_a == fp_b == fp_c
    assert fp_a != _opus_topic_fingerprint("APP_IMPROVEMENT", "Add Regime Detector")


def test_opus_finding_from_json_valid():
    from v2.audit import _opus_finding_from_json
    item = {
        "topic_slug": "Add Regime Detector",
        "title": "Add a regime-detector module",
        "category": "app_improvement",
        "priority": "high",
        "body": "Detect bull/bear regimes from SPY trend...",
        "evidence_quote": "Recent decisions ignore SPY context.",
    }
    f = _opus_finding_from_json(item, default_category="app_improvement")
    assert f is not None
    assert f.check_code == "APP_IMPROVEMENT"
    assert f.tier == 3 and f.severity == "info"
    assert f.evidence["topic_slug"] == "add-regime-detector"
    assert f.evidence["priority"] == "high"
    assert f.evidence["category"] == "app_improvement"


def test_opus_finding_from_json_missing_required_returns_none():
    from v2.audit import _opus_finding_from_json
    assert _opus_finding_from_json({}, default_category="app_improvement") is None
    assert _opus_finding_from_json(
        {"topic_slug": "x"}, default_category="app_improvement"
    ) is None  # missing title


def test_opus_finding_from_json_invalid_category_falls_back():
    from v2.audit import _opus_finding_from_json
    item = {"topic_slug": "x", "title": "T", "category": "garbage",
            "body": "b", "priority": "medium"}
    f = _opus_finding_from_json(item, default_category="audit_gap")
    assert f.evidence["category"] == "audit_gap"  # bogus value falls back
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -k "opus_topic_slug or opus_finding_from_json" -v`
Expected: 4 FAIL (ImportError).

- [ ] **Step 3: Implement the helpers**

In `v2/audit.py`, near the existing `Finding` dataclass (around line 47), add:

```python
import re as _re

_SLUG_RE = _re.compile(r"[^a-z0-9]+")
_VALID_OPUS_CATEGORIES = {"audit_gap", "app_improvement"}
_VALID_OPUS_PRIORITIES = {"high", "medium", "low"}


def _normalize_slug(s: str) -> str:
    return _SLUG_RE.sub("-", s.lower()).strip("-")


def _opus_topic_fingerprint(check_code: str, topic_slug: str) -> str:
    """Coarse fingerprint: hash(check_code + normalized_slug) only.

    Deliberately ignores evidence prose so daily re-emissions of the same
    underlying issue collapse. See spec 2026-05-12.
    """
    canonical = f"{check_code}:{_normalize_slug(topic_slug)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _opus_finding_from_json(item: dict, *, default_category: str) -> Finding | None:
    """Map one Opus finding dict to a Finding. Returns None on validation failure."""
    slug_raw = item.get("topic_slug") or ""
    title = item.get("title") or ""
    body = item.get("body") or ""
    if not slug_raw or not title:
        return None
    slug = _normalize_slug(slug_raw)
    if not slug:
        return None

    category = item.get("category")
    if category not in _VALID_OPUS_CATEGORIES:
        category = default_category
    priority = item.get("priority")
    if priority not in _VALID_OPUS_PRIORITIES:
        priority = "medium"

    check_code = "AUDIT_GAP" if category == "audit_gap" else "APP_IMPROVEMENT"

    evidence = {
        "topic_slug": slug,
        "category": category,
        "priority": priority,
        "evidence_quote": (item.get("evidence_quote") or "")[:600],
    }
    if category == "audit_gap" and item.get("proposed_check_code"):
        evidence["proposed_check_code"] = str(item["proposed_check_code"])[:64]

    f = Finding(
        check_code=check_code,
        tier=3, severity="info",
        title=title[:200],
        body=body[:2000],
        affected_count=1,
        evidence=evidence,
        auto_fix=None,
    )
    # Override the default fingerprint with the coarse topic-based one.
    # Because Finding.fingerprint is a property over evidence, we monkey-pin
    # the fingerprint by stashing the slug as the only fingerprint-relevant
    # field on a shallow-copy of evidence — but for the Finding model used
    # here, the caller computes the fingerprint via _opus_topic_fingerprint
    # at insert time. We expose it via evidence['_fp_override'] to avoid
    # modifying the Finding dataclass.
    f.evidence["_fp_override"] = _opus_topic_fingerprint(check_code, slug)
    return f
```

NOTE: the existing `Finding.fingerprint` property computes from full evidence. We do not modify the dataclass; instead, the two new check functions will read `evidence.pop("_fp_override")` themselves when emitting the finding for insertion. This keeps the existing Finding behavior untouched.

- [ ] **Step 4: Run tests, verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -k "opus_topic_slug or opus_finding_from_json" -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(audit): add Opus finding parser and topic-slug fingerprint helper"
```

---

### Task 2.3: Add prompt builder for audit-gap check + `check_audit_gaps_opus`

**Files:**
- Modify: `v2/audit.py`
- Modify: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_audit.py`:

```python
def test_check_audit_gaps_opus_emits_findings(monkeypatch):
    from v2 import audit
    from v2.database.connection import get_cursor

    canned = {
        "findings": [
            {
                "topic_slug": "missing-thesis-rotation-check",
                "title": "No check for stale active theses",
                "category": "audit_gap",
                "priority": "medium",
                "body": "Active theses aren't pruned by age...",
                "evidence_quote": "Some theses are >60 days old.",
                "proposed_check_code": "THESIS_STALE",
            }
        ]
    }
    usage = {"input_tokens": 100, "output_tokens": 50,
             "cache_creation_tokens": 0, "cache_read_tokens": 0, "latency_ms": 1000}
    monkeypatch.setattr(audit, "_call_opus_ideation", lambda s, p: (canned, usage))

    with get_cursor() as cur:
        findings = audit.check_audit_gaps_opus(cur)

    assert len(findings) == 1
    f = findings[0]
    assert f.check_code == "AUDIT_GAP"
    assert f.evidence["topic_slug"] == "missing-thesis-rotation-check"
    assert f.evidence["proposed_check_code"] == "THESIS_STALE"
    # Usage stashed for runner pickup
    assert audit.get_last_opus_ideation_usage("audit_gaps")["input_tokens"] == 100


def test_check_audit_gaps_opus_handles_empty(monkeypatch):
    from v2 import audit
    from v2.database.connection import get_cursor

    monkeypatch.setattr(
        audit, "_call_opus_ideation",
        lambda s, p: ({"findings": []}, {"input_tokens": 0, "output_tokens": 0,
                                          "cache_creation_tokens": 0,
                                          "cache_read_tokens": 0, "latency_ms": 100}),
    )
    with get_cursor() as cur:
        findings = audit.check_audit_gaps_opus(cur)
    assert findings == []
```

- [ ] **Step 2: Run, verify fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -k "check_audit_gaps_opus" -v`
Expected: 2 FAIL (AttributeError).

- [ ] **Step 3: Add the prompt builder + check + usage stash**

In `v2/audit.py`, near `_call_opus_ideation`, add:

```python
_LAST_OPUS_IDEATION_USAGE: dict[str, dict] = {}


def get_last_opus_ideation_usage(purpose: str) -> dict:
    return _LAST_OPUS_IDEATION_USAGE.get(purpose, {}).copy()


def _reset_opus_ideation_usage() -> None:
    _LAST_OPUS_IDEATION_USAGE.clear()


OPUS_AUDIT_GAPS_SYSTEM = """\
You are auditing the auditor of an agentic trading system. Given (1) the list
of existing audit check codes, (2) the audit's recent findings by check_code,
(3) a high-level DB schema, and (4) audit cost trend, propose NEW audit
checks that would catch integrity, strategy, or learning-loop problems the
current audit does not cover. Be conservative. Prefer specific checks with
clearly testable conditions over vague ones.

Output JSON only:
{
  "findings": [
    {"topic_slug": "kebab-case-stable-id",
     "title": "short imperative phrasing",
     "category": "audit_gap",
     "priority": "high"|"medium"|"low",
     "body": "1-3 short paragraphs",
     "evidence_quote": "specific motivating data point",
     "proposed_check_code": "PROPOSED_NEW_CODE"}
  ]
}

The topic_slug must be a stable identifier for the underlying gap so that
the same proposal on a future day yields the same slug. Max 10 findings.
Empty findings array is fine if you cannot find anything defensible.
"""


def _build_audit_gaps_prompt(cur) -> str:
    cur.execute("""
        SELECT check_code, COUNT(DISTINCT fingerprint) AS n, severity
        FROM audit_findings
        WHERE created_at > now() - interval '30 days'
        GROUP BY check_code, severity
        ORDER BY n DESC
    """)
    findings_summary = cur.fetchall()

    cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name IN ('decisions','decision_signals','news_signals',
                             'macro_signals','theses','strategy_rules',
                             'agent_events','agent_calls','sessions',
                             'session_stages','signal_attribution')
        ORDER BY table_name, ordinal_position
    """)
    schema_rows = cur.fetchall()
    schema_by_table: dict[str, list[str]] = {}
    for r in schema_rows:
        schema_by_table.setdefault(r["table_name"], []).append(r["column_name"])

    cur.execute("""
        SELECT ar.id, ar.started_at,
               COALESCE(SUM(ac.input_tokens+ac.output_tokens
                           +ac.cache_creation_tokens+ac.cache_read_tokens),0) AS tok,
               ar.total_findings
        FROM audit_runs ar
        LEFT JOIN audit_llm_calls ac ON ac.audit_run_id = ar.id
        WHERE ar.started_at > now() - interval '14 days'
        GROUP BY ar.id
        ORDER BY ar.started_at DESC
        LIMIT 14
    """)
    cost_trend = cur.fetchall()

    parts = ["## Existing audit check codes\n"]
    parts.extend(f"- {name}" for name in CHECKS if isinstance(name, str))
    parts.append("\n## Last 30 days of findings (by check_code)\n")
    for r in findings_summary:
        parts.append(f"- {r['check_code']} [{r['severity']}]: {r['n']} distinct fingerprints")
    parts.append("\n## DB schema (selected tables)\n")
    for t, cols in schema_by_table.items():
        parts.append(f"### {t}\n{', '.join(cols)}\n")
    parts.append("\n## Recent audit runs (last 14, total tokens + finding count)\n")
    for r in cost_trend:
        parts.append(f"- run {r['id']} @ {r['started_at']}: tok={r['tok']} findings={r['total_findings']}")
    return "\n".join(parts)


def check_audit_gaps_opus(cur) -> list[Finding]:
    """Single Opus call proposing new audit checks. Tier 3 / severity info."""
    prompt = _build_audit_gaps_prompt(cur)
    parsed, usage = _call_opus_ideation(OPUS_AUDIT_GAPS_SYSTEM, prompt)
    _LAST_OPUS_IDEATION_USAGE["audit_gaps"] = usage

    findings: list[Finding] = []
    seen_slugs: set[str] = set()
    for item in (parsed.get("findings") or [])[:20]:
        f = _opus_finding_from_json(item, default_category="audit_gap")
        if f is None:
            continue
        slug = f.evidence["topic_slug"]
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        findings.append(f)
        if len(findings) >= 10:
            break
    return findings
```

- [ ] **Step 4: Run tests, verify pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -k "check_audit_gaps_opus" -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(audit): add check_audit_gaps_opus (Opus 4.7 meta-audit)"
```

---

### Task 2.4: Add `check_app_improvements_opus`

Mirrors `check_audit_gaps_opus` with a different system prompt and different DB inputs.

**Files:**
- Modify: `v2/audit.py`
- Modify: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_audit.py`:

```python
def test_check_app_improvements_opus_emits_findings(monkeypatch):
    from v2 import audit
    from v2.database.connection import get_cursor

    canned = {
        "findings": [
            {
                "topic_slug": "add-regime-detector",
                "title": "Add a regime-detector signal",
                "category": "app_improvement",
                "priority": "high",
                "body": "Detect bull/bear regimes from SPY trend...",
                "evidence_quote": "Recent decisions ignore market regime.",
            }
        ]
    }
    usage = {"input_tokens": 200, "output_tokens": 80,
             "cache_creation_tokens": 0, "cache_read_tokens": 0, "latency_ms": 1500}
    monkeypatch.setattr(audit, "_call_opus_ideation", lambda s, p: (canned, usage))

    with get_cursor() as cur:
        findings = audit.check_app_improvements_opus(cur)

    assert len(findings) == 1
    assert findings[0].check_code == "APP_IMPROVEMENT"
    assert findings[0].evidence["topic_slug"] == "add-regime-detector"
    assert audit.get_last_opus_ideation_usage("app_improvements")["input_tokens"] == 200
```

- [ ] **Step 2: Run, verify fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_check_app_improvements_opus_emits_findings -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `v2/audit.py`, after `check_audit_gaps_opus`, add:

```python
OPUS_APP_IMPROVEMENTS_SYSTEM = """\
You are an architect reviewing an agentic trading system. Given recent
session reflections, active and retired rules, signal attribution, recent
decisions and their outcomes, active/closed theses, equity trend, and a
high-level module manifest, propose application-level improvements:
new features, modeling changes, data sources, risk improvements, or UX
changes that would plausibly improve outcomes. Be conservative.

Output JSON only:
{
  "findings": [
    {"topic_slug": "kebab-case-stable-id",
     "title": "short imperative phrasing",
     "category": "app_improvement",
     "priority": "high"|"medium"|"low",
     "body": "1-3 short paragraphs",
     "evidence_quote": "specific motivating data point"}
  ]
}

The topic_slug must be a stable identifier for the underlying idea so the
same proposal on a future day yields the same slug. Max 10 findings.
Empty findings array is fine if you cannot find anything defensible.
"""


def _build_app_improvements_prompt(cur) -> str:
    cur.execute("""
        SELECT id, created_at, body
        FROM strategy_memos
        ORDER BY created_at DESC
        LIMIT 14
    """)
    memos = cur.fetchall()

    cur.execute("""
        SELECT id, status, rule_text, retired_at
        FROM strategy_rules
        WHERE status='active'
           OR (status='retired' AND retired_at > now() - interval '30 days')
        ORDER BY status, id
    """)
    rules = cur.fetchall()

    cur.execute("""
        SELECT category, sample_size, sample_size_30d,
               avg_outcome_7d, win_rate_7d, avg_outcome_30d, win_rate_30d
        FROM signal_attribution
    """)
    attribution = cur.fetchall()

    cur.execute("""
        SELECT id, date, ticker, action, notional,
               outcome_7d_pct, outcome_30d_pct
        FROM decisions
        WHERE date > now()::date - 14
        ORDER BY date DESC, id DESC
        LIMIT 200
    """)
    decisions = cur.fetchall()

    cur.execute("""
        SELECT id, ticker, status, summary, outcome_pct, closed_at
        FROM theses
        WHERE status='active'
           OR (status IN ('closed','retired')
               AND closed_at > now() - interval '30 days')
        ORDER BY status, id DESC
    """)
    theses = cur.fetchall()

    cur.execute("""
        SELECT snapshot_date, equity
        FROM account_snapshots
        WHERE snapshot_date > now()::date - 30
        ORDER BY snapshot_date DESC
    """)
    snapshots = cur.fetchall()

    import os, pathlib
    module_manifest = []
    v2_dir = pathlib.Path(__file__).resolve().parent
    for path in sorted(v2_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            first_lines = path.read_text(encoding="utf-8").splitlines()[:6]
        except OSError:
            continue
        doc = ""
        for ln in first_lines:
            s = ln.strip().strip('"').strip("'")
            if s and not s.startswith("#") and not s.startswith("from") and not s.startswith("import"):
                doc = s
                break
        module_manifest.append((path.name, doc[:160]))

    parts = ["## Recent strategy memos (last 14)\n"]
    for m in memos:
        body_excerpt = (m["body"] or "")[:600]
        parts.append(f"### memo {m['id']} @ {m['created_at']}\n{body_excerpt}\n")
    parts.append("\n## Rules (active + retired-30d)\n")
    for r in rules:
        parts.append(f"- rule {r['id']} [{r['status']}]: {r['rule_text']}")
    parts.append("\n## signal_attribution snapshot\n")
    for a in attribution:
        parts.append(
            f"- {a['category']}: n={a['sample_size']} n30={a['sample_size_30d']} "
            f"out7={a['avg_outcome_7d']} win7={a['win_rate_7d']} "
            f"out30={a['avg_outcome_30d']} win30={a['win_rate_30d']}"
        )
    parts.append("\n## Recent decisions (last 14d, up to 200)\n")
    for d in decisions:
        parts.append(
            f"- {d['date']} {d['ticker']} {d['action']} notional={d['notional']} "
            f"out7={d['outcome_7d_pct']} out30={d['outcome_30d_pct']}"
        )
    parts.append("\n## Theses (active + recently-closed)\n")
    for t in theses:
        parts.append(f"- thesis {t['id']} {t['ticker']} [{t['status']}] out={t['outcome_pct']}: {t['summary']}")
    parts.append("\n## Account snapshots (last 30d)\n")
    for s in snapshots:
        parts.append(f"- {s['snapshot_date']}: equity={s['equity']}")
    parts.append("\n## v2 module manifest\n")
    for name, doc in module_manifest:
        parts.append(f"- {name}: {doc}")

    raw = "\n".join(parts)
    cap_str = os.environ.get("ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS")
    try:
        cap = int(cap_str) if cap_str else OPUS_INPUT_TOKEN_CAP_DEFAULT
    except ValueError:
        cap = OPUS_INPUT_TOKEN_CAP_DEFAULT
    # Rough char-to-token approximation: 4 chars/token.
    max_chars = cap * 4
    if len(raw) > max_chars:
        log.warning("Opus app-improvements prompt truncated: %d > %d chars",
                    len(raw), max_chars)
        raw = raw[:max_chars] + "\n\n[INPUT TRUNCATED]"
    return raw


def check_app_improvements_opus(cur) -> list[Finding]:
    """Single Opus call proposing application-level improvements."""
    prompt = _build_app_improvements_prompt(cur)
    parsed, usage = _call_opus_ideation(OPUS_APP_IMPROVEMENTS_SYSTEM, prompt)
    _LAST_OPUS_IDEATION_USAGE["app_improvements"] = usage

    findings: list[Finding] = []
    seen_slugs: set[str] = set()
    for item in (parsed.get("findings") or [])[:20]:
        f = _opus_finding_from_json(item, default_category="app_improvement")
        if f is None:
            continue
        slug = f.evidence["topic_slug"]
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        findings.append(f)
        if len(findings) >= 10:
            break
    return findings
```

- [ ] **Step 4: Run test, verify pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_check_app_improvements_opus_emits_findings -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(audit): add check_app_improvements_opus (Opus 4.7 ideation)"
```

---

### Task 2.5: Wire both Opus checks into the runner with fingerprint override + LLM accounting

The new checks emit findings with an `evidence['_fp_override']` value. The runner must use that as the fingerprint when present, instead of `Finding.fingerprint`. The runner also must write per-call rows to `audit_llm_calls`.

**Files:**
- Modify: `v2/audit.py:1501-1596` (`run_audit`)
- Modify: `v2/audit.py:1446-1473` (`CHECKS` list)
- Modify: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/v2/test_audit.py`:

```python
def test_run_audit_runs_opus_checks_and_writes_llm_calls(monkeypatch):
    from v2 import audit
    from v2.database.connection import get_cursor

    def fake_opus(system, prompt):
        if "auditing the auditor" in system.lower():
            usage = {"input_tokens": 1000, "output_tokens": 100,
                     "cache_creation_tokens": 0, "cache_read_tokens": 0,
                     "latency_ms": 2000}
            return ({"findings": [{
                "topic_slug": "missing-foo-check",
                "title": "Missing foo check",
                "category": "audit_gap", "priority": "low",
                "body": "...", "evidence_quote": "..."}]}, usage)
        usage = {"input_tokens": 2000, "output_tokens": 200,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 2500}
        return ({"findings": [{
            "topic_slug": "build-bar",
            "title": "Build the bar feature",
            "category": "app_improvement", "priority": "medium",
            "body": "...", "evidence_quote": "..."}]}, usage)

    monkeypatch.setattr(audit, "_call_opus_ideation", fake_opus)
    # Don't actually call Haiku for rule judgment in this test.
    monkeypatch.setattr(audit, "_call_rule_judgment_llm",
                        lambda p: ({"findings": []},
                                   {"input_tokens": 0, "output_tokens": 0,
                                    "cache_creation_tokens": 0,
                                    "cache_read_tokens": 0, "latency_ms": 10}))

    summary = audit.run_audit(apply=False)

    with get_cursor() as cur:
        cur.execute(
            "SELECT purpose, model, input_tokens FROM audit_llm_calls "
            "WHERE audit_run_id=%s ORDER BY purpose",
            (summary.run_id,),
        )
        rows = cur.fetchall()
    purposes = {r["purpose"] for r in rows}
    assert {"audit_gaps", "app_improvements"}.issubset(purposes)
    gap_row = next(r for r in rows if r["purpose"] == "audit_gaps")
    assert gap_row["model"] == audit.OPUS_IDEATION_MODEL
    assert gap_row["input_tokens"] == 1000

    # Findings are inserted with the topic-slug-based fingerprint
    with get_cursor() as cur:
        cur.execute(
            "SELECT check_code, fingerprint, evidence "
            "FROM audit_findings WHERE audit_run_id=%s "
            "AND check_code IN ('AUDIT_GAP','APP_IMPROVEMENT')",
            (summary.run_id,),
        )
        fnds = cur.fetchall()
    assert any(f["check_code"] == "AUDIT_GAP" for f in fnds)
    assert any(f["check_code"] == "APP_IMPROVEMENT" for f in fnds)
    # Fingerprint matches the topic-slug formula
    gap_fnd = next(f for f in fnds if f["check_code"] == "AUDIT_GAP")
    assert gap_fnd["fingerprint"] == audit._opus_topic_fingerprint(
        "AUDIT_GAP", "missing-foo-check"
    )
    assert "_fp_override" not in gap_fnd["evidence"]  # stripped before insert
```

- [ ] **Step 2: Run, verify fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_run_audit_runs_opus_checks_and_writes_llm_calls -v`
Expected: FAIL.

- [ ] **Step 3: Register the checks**

In `v2/audit.py`, append to `CHECKS` (around line 1473):

```python
    "check_audit_gaps_opus",
    "check_app_improvements_opus",
```

- [ ] **Step 4: Update `run_audit` finding-insert path**

In `v2/audit.py`, locate the loop in `run_audit` that inserts findings (around line 1533). Replace the `for f in findings:` block with:

```python
                for f in findings:
                    # Coarse fingerprint for Opus ideation findings (topic-slug-based).
                    # Other checks keep the default content-evidence fingerprint.
                    fp_override = None
                    if isinstance(f.evidence, dict) and "_fp_override" in f.evidence:
                        fp_override = f.evidence.pop("_fp_override")
                    fingerprint = fp_override if fp_override else f.fingerprint
                    current_fingerprints.add(fingerprint)

                    inserted_id = insert_audit_finding(
                        audit_run_id=run_id,
                        check_code=f.check_code, tier=f.tier, severity=f.severity,
                        title=f.title, body=f.body,
                        affected_count=f.affected_count, evidence=f.evidence,
                        fingerprint=fingerprint,
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
                                fingerprint=fingerprint + ":fixed",
                                status="auto_fixed",
                            )
                            auto_fixed += 1
                        except Exception:
                            cur.execute("ROLLBACK TO SAVEPOINT audit_fix")
                            cur.execute("RELEASE SAVEPOINT audit_fix")
                            log.exception("Auto-fix failed for %s", f.check_code)
```

- [ ] **Step 5: Wire LLM-call accounting for Opus**

In `v2/audit.py`, in `run_audit`, replace the existing block that updates `rule_judgment_usage` (from Task 1.3) with a more general block. Find the line that begins `if check.__name__ == "check_rule_judgment":` and expand it to also handle the two Opus checks:

```python
                if check.__name__ == "check_rule_judgment":
                    rule_judgment_usage = get_last_rule_judgment_usage()
                    if rule_judgment_usage:
                        from v2.database.trading_db import insert_audit_llm_call
                        insert_audit_llm_call(
                            audit_run_id=run_id,
                            purpose="rule_judgment",
                            model=RULE_JUDGMENT_MODEL,
                            input_tokens=rule_judgment_usage.get("input_tokens", 0),
                            output_tokens=rule_judgment_usage.get("output_tokens", 0),
                            cache_creation_tokens=rule_judgment_usage.get("cache_creation_tokens", 0),
                            cache_read_tokens=rule_judgment_usage.get("cache_read_tokens", 0),
                            latency_ms=rule_judgment_usage.get("latency_ms"),
                        )
                elif check.__name__ in ("check_audit_gaps_opus", "check_app_improvements_opus"):
                    purpose = "audit_gaps" if check.__name__ == "check_audit_gaps_opus" else "app_improvements"
                    usage = get_last_opus_ideation_usage(purpose)
                    if usage:
                        from v2.database.trading_db import insert_audit_llm_call
                        insert_audit_llm_call(
                            audit_run_id=run_id,
                            purpose=purpose,
                            model=OPUS_IDEATION_MODEL,
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            cache_creation_tokens=usage.get("cache_creation_tokens", 0),
                            cache_read_tokens=usage.get("cache_read_tokens", 0),
                            latency_ms=usage.get("latency_ms"),
                        )
```

- [ ] **Step 6: Reset Opus usage stash at the start of each run**

In `v2/audit.py`, inside `run_audit`, just after the `try:` block opens (around line 1508, before `run_id = insert_audit_run(...)`), add:

```python
        _reset_opus_ideation_usage()
```

- [ ] **Step 7: Run the integration test, verify pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_run_audit_runs_opus_checks_and_writes_llm_calls -v`
Expected: PASS.

- [ ] **Step 8: Run the full audit test file**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py -v`
Expected: All pass.

- [ ] **Step 9: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(audit): wire Opus ideation checks into runner with topic-slug fingerprints"
```

---

### Task 2.6: Document new env var in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (Optional knobs section, near other `ALGO_*` vars)

- [ ] **Step 1: Find the Optional knobs section**

Run: `grep -n "Optional knobs" CLAUDE.md`
Expected: A line number around the env-var documentation block.

- [ ] **Step 2: Add the env var entry**

In `CLAUDE.md`, in the "Optional knobs" bullet list, add:

```markdown
- `ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS` — overrides the input-token cap for each Opus ideation audit check. Defaults to `60000` (≈$0.90 input per call worst-case at Opus 4.7 pricing). The runner truncates the lowest-priority section of the prompt when exceeded and logs a warning.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS"
```

---

## Commit 3: Jira Filing

### Task 3.1: Create `v2/audit_jira.py` skeleton + dedup search

**Files:**
- Create: `v2/audit_jira.py`
- Create: `tests/v2/test_audit_jira.py`

- [ ] **Step 1: Write the failing dedup test**

Create `tests/v2/test_audit_jira.py`:

```python
"""Tests for v2/audit_jira.py — Jira ticket filing for Opus ideation findings."""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def jira_env(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    monkeypatch.setenv("JIRA_AUDIT_PROJECT_KEY", "ALGO")


def test_find_existing_issue_returns_key_on_match(jira_env):
    from v2 import audit_jira

    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"issues": [{"key": "ALGO-7"}]}
    with patch("v2.audit_jira.requests.get", return_value=fake_resp) as mock_get:
        result = audit_jira.find_existing_issue("abc123fingerprint")
    assert result == "ALGO-7"
    assert mock_get.called
    # JQL must include the fingerprint label and the project key
    called_params = mock_get.call_args.kwargs.get("params") or {}
    assert "ALGO" in called_params["jql"]
    assert "audit-fingerprint:abc123fingerprint" in called_params["jql"]


def test_find_existing_issue_returns_none_on_empty(jira_env):
    from v2 import audit_jira

    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"issues": []}
    with patch("v2.audit_jira.requests.get", return_value=fake_resp):
        result = audit_jira.find_existing_issue("nomatch")
    assert result is None
```

- [ ] **Step 2: Run, verify fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit_jira.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v2.audit_jira'`.

- [ ] **Step 3: Create the module**

Create `v2/audit_jira.py`:

```python
# v2/audit_jira.py
"""Jira REST integration for Opus audit ideation findings.

Files tickets for new ideation findings; dedups against existing open issues
by a fingerprint label. Read by v2/audit.py's runner.

Spec: docs/superpowers/specs/2026-05-12-opus-audit-ideation-design.md
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

REQUEST_TIMEOUT_SEC = 15


class JiraConfigMissing(Exception):
    """Raised when required Jira env vars are not all set."""


def _config() -> dict:
    required = ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN",
                "JIRA_AUDIT_PROJECT_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise JiraConfigMissing(f"Missing env vars: {', '.join(missing)}")
    return {
        "base_url": os.environ["JIRA_BASE_URL"].rstrip("/"),
        "email": os.environ["JIRA_EMAIL"],
        "token": os.environ["JIRA_API_TOKEN"],
        "project_key": os.environ["JIRA_AUDIT_PROJECT_KEY"],
        "issue_type": os.environ.get("JIRA_AUDIT_ISSUE_TYPE", "Task"),
    }


def _auth(cfg: dict):
    return (cfg["email"], cfg["token"])


def find_existing_issue(fingerprint: str) -> str | None:
    """JQL-search for an open Jira issue tagged with this fingerprint.

    Returns the issue key (e.g. 'ALGO-7') if found, else None.
    Raises JiraConfigMissing if env not configured.
    """
    cfg = _config()
    jql = (
        f'project = "{cfg["project_key"]}" '
        f'AND labels = "audit-fingerprint:{fingerprint}" '
        f'AND statusCategory != Done'
    )
    resp = requests.get(
        f"{cfg['base_url']}/rest/api/3/search",
        params={"jql": jql, "fields": "summary", "maxResults": 1},
        auth=_auth(cfg),
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    issues = resp.json().get("issues") or []
    return issues[0]["key"] if issues else None
```

- [ ] **Step 4: Run tests, verify pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit_jira.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit_jira.py tests/v2/test_audit_jira.py
git commit -m "feat(audit): scaffold audit_jira module + fingerprint dedup search"
```

---

### Task 3.2: Implement issue creation + `file_jira_ticket`

**Files:**
- Modify: `v2/audit_jira.py`
- Modify: `tests/v2/test_audit_jira.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/v2/test_audit_jira.py`:

```python
def _make_finding(check_code="APP_IMPROVEMENT", category="app_improvement",
                  topic_slug="add-regime-detector", title="Add a regime detector",
                  priority="high", body="Detect bull/bear regimes from SPY."):
    from v2.audit import Finding, _opus_topic_fingerprint
    f = Finding(
        check_code=check_code, tier=3, severity="info",
        title=title, body=body, affected_count=1,
        evidence={
            "topic_slug": topic_slug, "category": category,
            "priority": priority,
            "evidence_quote": "evidence here",
        },
        auto_fix=None,
    )
    # Stash the precomputed fingerprint exactly like the real flow does
    f.evidence["_fp"] = _opus_topic_fingerprint(check_code, topic_slug)
    return f


def test_file_jira_ticket_dedup_hit_skips_create(jira_env):
    from v2 import audit_jira
    f = _make_finding()
    with patch.object(audit_jira, "find_existing_issue", return_value="ALGO-9"), \
         patch.object(audit_jira.requests, "post") as mock_post:
        result = audit_jira.file_jira_ticket(f, fingerprint=f.evidence["_fp"], run_id=1)
    assert result == {"status": "existing", "issue_key": "ALGO-9"}
    assert not mock_post.called


def test_file_jira_ticket_creates_when_no_dedup(jira_env):
    from v2 import audit_jira
    f = _make_finding()
    fake_resp = MagicMock(status_code=201)
    fake_resp.json.return_value = {"key": "ALGO-42"}
    with patch.object(audit_jira, "find_existing_issue", return_value=None), \
         patch.object(audit_jira.requests, "post", return_value=fake_resp) as mock_post:
        result = audit_jira.file_jira_ticket(f, fingerprint=f.evidence["_fp"], run_id=99)
    assert result == {"status": "created", "issue_key": "ALGO-42"}
    payload = json.loads(mock_post.call_args.kwargs["data"])
    fields = payload["fields"]
    assert fields["project"]["key"] == "ALGO"
    assert fields["issuetype"]["name"] == "Task"
    assert "[audit:app_improvement]" in fields["summary"]
    assert "Add a regime detector" in fields["summary"]
    labels = fields["labels"]
    assert f"audit-fingerprint:{f.evidence['_fp']}" in labels
    assert "audit-source:opus-ideation" in labels
    assert "audit-category:app_improvement" in labels
    assert fields["priority"]["name"] == "High"
    assert "run #99" in fields["description"]


def test_file_jira_ticket_500_returns_failed(jira_env):
    from v2 import audit_jira
    f = _make_finding()
    fake_resp = MagicMock(status_code=500, text="boom")
    fake_resp.raise_for_status.side_effect = requests.HTTPError("500")
    import requests
    with patch.object(audit_jira, "find_existing_issue", return_value=None), \
         patch.object(audit_jira.requests, "post", return_value=fake_resp):
        result = audit_jira.file_jira_ticket(f, fingerprint=f.evidence["_fp"], run_id=1)
    assert result["status"] == "failed"
    assert "500" in result["error"]


def test_file_jira_ticket_missing_env_returns_disabled(monkeypatch):
    monkeypatch.delenv("JIRA_BASE_URL", raising=False)
    from v2 import audit_jira
    f = _make_finding()
    result = audit_jira.file_jira_ticket(f, fingerprint="abc", run_id=1)
    assert result == {"status": "disabled", "reason": "config_missing"}
```

Also add `import requests` at the top of the test file so `requests.HTTPError` is in scope:

```python
import requests
```

- [ ] **Step 2: Run, verify fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit_jira.py -v`
Expected: 4 new tests FAIL with `AttributeError: module 'v2.audit_jira' has no attribute 'file_jira_ticket'`.

- [ ] **Step 3: Implement `file_jira_ticket`**

Append to `v2/audit_jira.py`:

```python
import json

PRIORITY_MAP = {"high": "High", "medium": "Medium", "low": "Low"}


def _build_description(finding, run_id: int, fingerprint: str) -> str:
    evidence = finding.evidence or {}
    slug = evidence.get("topic_slug", "")
    quote = evidence.get("evidence_quote", "")
    lines = [finding.body or ""]
    if quote:
        lines.append("")
        lines.append("**Evidence:**")
        lines.append(f"> {quote}")
    lines.append("")
    lines.append("---")
    lines.append(
        f"Filed by audit run #{run_id}. Topic: `{slug}`. Fingerprint: `{fingerprint}`."
    )
    return "\n".join(lines)


def _build_create_payload(finding, run_id: int, fingerprint: str, cfg: dict) -> dict:
    category = (finding.evidence or {}).get("category", "app_improvement")
    priority = (finding.evidence or {}).get("priority", "medium")
    summary = f"[audit:{category}] {finding.title}"[:250]
    fields = {
        "project": {"key": cfg["project_key"]},
        "issuetype": {"name": cfg["issue_type"]},
        "summary": summary,
        "description": _build_description(finding, run_id, fingerprint),
        "labels": [
            f"audit-fingerprint:{fingerprint}",
            "audit-source:opus-ideation",
            f"audit-category:{category}",
        ],
    }
    if priority in PRIORITY_MAP:
        fields["priority"] = {"name": PRIORITY_MAP[priority]}
    return {"fields": fields}


def file_jira_ticket(finding, *, fingerprint: str, run_id: int) -> dict:
    """File a Jira ticket for an Opus ideation finding.

    Returns a dict suitable for stashing in finding.evidence['jira']:
      {"status": "existing"|"created"|"failed"|"disabled", ...}.
    Never raises — all exceptions are caught and recorded in the return dict.
    """
    try:
        cfg = _config()
    except JiraConfigMissing as exc:
        log.info("Jira filing disabled: %s", exc)
        return {"status": "disabled", "reason": "config_missing"}

    try:
        existing = find_existing_issue(fingerprint)
    except Exception as exc:
        log.exception("Jira dedup search failed for %s", fingerprint)
        return {"status": "failed", "error": f"dedup_search: {exc}"}

    if existing:
        return {"status": "existing", "issue_key": existing}

    payload = _build_create_payload(finding, run_id, fingerprint, cfg)
    try:
        resp = requests.post(
            f"{cfg['base_url']}/rest/api/3/issue",
            data=json.dumps(payload),
            auth=_auth(cfg),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.exception("Jira create failed for fingerprint=%s", fingerprint)
        return {"status": "failed", "error": str(exc)}

    body = resp.json() if resp.content else {}
    issue_key = body.get("key") or ""
    return {"status": "created", "issue_key": issue_key}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit_jira.py -v`
Expected: All 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/audit_jira.py tests/v2/test_audit_jira.py
git commit -m "feat(audit): implement file_jira_ticket with dedup, create, and failure paths"
```

---

### Task 3.3: Hook Jira filing into the audit runner + `--file-jira` flag

**Files:**
- Modify: `v2/audit.py` (`run_audit` + `main`)
- Modify: `tests/v2/test_audit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_audit.py`:

```python
def test_run_audit_file_jira_caps_creates(monkeypatch):
    """When --file-jira is on, only ALGO_AUDIT_JIRA_MAX_CREATES tickets are created.

    Beyond the cap, findings still write to audit_findings with status='capped'.
    """
    from v2 import audit
    from v2.database.connection import get_cursor

    # Stub Opus to emit 3 distinct findings via the gap check
    canned = {
        "findings": [
            {"topic_slug": f"slug-{i}", "title": f"T{i}",
             "category": "audit_gap", "priority": "low",
             "body": "b", "evidence_quote": "e"}
            for i in range(3)
        ]
    }
    usage = {"input_tokens": 1, "output_tokens": 1,
             "cache_creation_tokens": 0, "cache_read_tokens": 0, "latency_ms": 1}

    def fake_opus(system, prompt):
        if "auditing the auditor" in system.lower():
            return (canned, usage)
        return ({"findings": []}, usage)

    monkeypatch.setattr(audit, "_call_opus_ideation", fake_opus)
    monkeypatch.setattr(audit, "_call_rule_judgment_llm",
                        lambda p: ({"findings": []},
                                   {"input_tokens": 0, "output_tokens": 0,
                                    "cache_creation_tokens": 0,
                                    "cache_read_tokens": 0, "latency_ms": 1}))
    monkeypatch.setenv("ALGO_AUDIT_JIRA_MAX_CREATES", "2")

    create_calls = []
    def fake_file(finding, *, fingerprint, run_id):
        create_calls.append(fingerprint)
        return {"status": "created", "issue_key": f"ALGO-{len(create_calls)}"}

    from v2 import audit_jira
    monkeypatch.setattr(audit_jira, "file_jira_ticket", fake_file)

    summary = audit.run_audit(apply=False, file_jira=True)

    # Only 2 creates should have happened
    assert len(create_calls) == 2

    with get_cursor() as cur:
        cur.execute(
            "SELECT evidence FROM audit_findings "
            "WHERE audit_run_id=%s AND check_code='AUDIT_GAP'",
            (summary.run_id,),
        )
        evidences = [r["evidence"] for r in cur.fetchall()]
    statuses = sorted(e["jira"]["status"] for e in evidences)
    assert statuses == ["capped", "created", "created"]
```

- [ ] **Step 2: Run, verify fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_run_audit_file_jira_caps_creates -v`
Expected: FAIL — `run_audit() got an unexpected keyword argument 'file_jira'`.

- [ ] **Step 3: Extend `run_audit` signature + Jira hook**

In `v2/audit.py`, update the `run_audit` signature (line 1501):

```python
def run_audit(
    apply: bool = False,
    max_auto_fix: int = MAX_AUTO_FIX_DEFAULT,
    file_jira: bool = False,
) -> AuditRunSummary:
```

At the top of `run_audit`, after `_reset_opus_ideation_usage()`, add Jira state:

```python
        import os as _os
        file_jira_enabled = file_jira or _os.environ.get("ALGO_AUDIT_FILE_JIRA") == "1"
        try:
            jira_create_cap = int(_os.environ.get("ALGO_AUDIT_JIRA_MAX_CREATES", "5"))
        except ValueError:
            jira_create_cap = 5
        jira_created_this_run = 0
```

In the finding-insert loop (within `for f in findings:` after the fingerprint is computed), add Jira filing AFTER the `insert_audit_finding` call. Concretely, replace the `inserted_id = insert_audit_finding(...)` section so the Jira step appends a `jira` key to `evidence` and re-records it. Because the existing `insert_audit_finding` happens before we know the Jira result, the cleanest path is to file Jira FIRST, then insert. Restructure as:

```python
                for f in findings:
                    fp_override = None
                    if isinstance(f.evidence, dict) and "_fp_override" in f.evidence:
                        fp_override = f.evidence.pop("_fp_override")
                    fingerprint = fp_override if fp_override else f.fingerprint
                    current_fingerprints.add(fingerprint)

                    # Jira filing (Opus ideation findings only)
                    is_opus_finding = f.check_code in ("AUDIT_GAP", "APP_IMPROVEMENT")
                    if file_jira_enabled and is_opus_finding:
                        from v2 import audit_jira
                        if jira_created_this_run >= jira_create_cap:
                            f.evidence["jira"] = {"status": "capped"}
                        else:
                            jira_result = audit_jira.file_jira_ticket(
                                f, fingerprint=fingerprint, run_id=run_id,
                            )
                            f.evidence["jira"] = jira_result
                            if jira_result.get("status") == "created":
                                jira_created_this_run += 1
                    elif is_opus_finding:
                        f.evidence["jira"] = {"status": "disabled"}

                    inserted_id = insert_audit_finding(
                        audit_run_id=run_id,
                        check_code=f.check_code, tier=f.tier, severity=f.severity,
                        title=f.title, body=f.body,
                        affected_count=f.affected_count, evidence=f.evidence,
                        fingerprint=fingerprint,
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
                                fingerprint=fingerprint + ":fixed",
                                status="auto_fixed",
                            )
                            auto_fixed += 1
                        except Exception:
                            cur.execute("ROLLBACK TO SAVEPOINT audit_fix")
                            cur.execute("RELEASE SAVEPOINT audit_fix")
                            log.exception("Auto-fix failed for %s", f.check_code)
```

- [ ] **Step 4: Add `--file-jira` to the CLI**

In `v2/audit.py`, in `main()` (around line 1600), add the argument and pass it through:

```python
def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m v2.audit",
                                     description="Self-healing audit runner")
    parser.add_argument("--apply", action="store_true",
                        help="Apply Tier-1 auto-fixes (default: propose-only)")
    parser.add_argument("--max-auto-fix", type=int, default=MAX_AUTO_FIX_DEFAULT,
                        help=f"Cap on auto-fixes per run (default {MAX_AUTO_FIX_DEFAULT})")
    parser.add_argument("--file-jira", action="store_true",
                        help="File Jira tickets for new Opus ideation findings "
                             "(also enabled by ALGO_AUDIT_FILE_JIRA=1)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        summary = run_audit(apply=args.apply, max_auto_fix=args.max_auto_fix,
                            file_jira=args.file_jira)
    except Exception:
        log.exception("Audit run failed unrecoverably")
        return 2

    if summary.has_critical_open:
        return 1
    return 0
```

- [ ] **Step 5: Run the new test, verify pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py::test_run_audit_file_jira_caps_creates -v`
Expected: PASS.

- [ ] **Step 6: Run full audit test file**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py tests/v2/test_audit_jira.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add v2/audit.py tests/v2/test_audit.py
git commit -m "feat(audit): wire --file-jira flag and per-run create cap into runner"
```

---

### Task 3.4: Dashboard template renders the Jira block

**Files:**
- Modify: `dashboard/templates/audit_finding.html`
- Modify: `tests/v2/test_audit_dashboard.py`

- [ ] **Step 1: Inspect the existing template**

Run: `cat dashboard/templates/audit_finding.html`
Look for where `evidence` is rendered. Note the surrounding HTML structure.

- [ ] **Step 2: Write the failing test**

Add to `tests/v2/test_audit_dashboard.py`:

```python
def test_audit_finding_template_renders_jira_link(client, queries_mock):
    """Findings with evidence.jira.issue_key render a Jira link."""
    queries_mock.get_audit_finding.return_value = {
        "id": 1, "check_code": "AUDIT_GAP", "tier": 3, "severity": "info",
        "title": "Add missing X", "body": "...",
        "affected_count": 1, "status": "open", "fingerprint": "abc",
        "audit_run_id": 1, "created_at": "2026-05-12 00:00",
        "resolved_at": None, "resolved_note": None,
        "evidence": {
            "topic_slug": "add-missing-x",
            "jira": {"status": "created", "issue_key": "ALGO-42"},
        },
    }
    # Make sure the audit-run lookup also succeeds
    queries_mock.get_audit_run.return_value = {"id": 1, "started_at": "2026-05-12 00:00"}

    resp = client.get("/audit/findings/1")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "ALGO-42" in html
    # Should link to Jira; test that the issue key is presented as an anchor or readable text
    assert "audit-source:opus-ideation" not in html  # don't dump raw label list
```

NOTE: the exact dashboard test pattern depends on existing scaffolding in `tests/v2/test_audit_dashboard.py`; the assistant implementing should adapt the fixture names (e.g. `client`, `queries_mock`) to match what's already in the file. If the existing tests use a different lookup path (e.g. `/audit/<run_id>/finding/<id>`), use that path.

- [ ] **Step 3: Run, verify fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit_dashboard.py::test_audit_finding_template_renders_jira_link -v`
Expected: FAIL — `ALGO-42` not in rendered HTML.

- [ ] **Step 4: Update the template**

In `dashboard/templates/audit_finding.html`, find where `evidence` is iterated or pretty-printed. Add this block ABOVE the generic evidence rendering (replace any `JIRA_BASE_URL_HERE` with `{{ jira_base_url }}` if the template already has it; otherwise hardcode the issue key as text + use `evidence.jira.issue_key` to compute the URL):

```html
{% if finding.evidence and finding.evidence.jira %}
  <div class="jira-block">
    {% set jira = finding.evidence.jira %}
    {% if jira.issue_key %}
      <strong>Jira:</strong>
      <a href="https://{{ jira_workspace if jira_workspace else 'example.atlassian.net' }}/browse/{{ jira.issue_key }}"
         target="_blank" rel="noopener">{{ jira.issue_key }}</a>
      <span class="jira-status">({{ jira.status }})</span>
    {% else %}
      <strong>Jira:</strong> <span class="jira-status">{{ jira.status }}</span>
      {% if jira.error %}<span class="jira-error">— {{ jira.error }}</span>{% endif %}
    {% endif %}
  </div>
{% endif %}
```

If the dashboard does not currently pass `jira_workspace` to the template, render just the issue key as text — the test only asserts the key is present, not the URL.

- [ ] **Step 5: Run the test, verify pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit_dashboard.py::test_audit_finding_template_renders_jira_link -v`
Expected: PASS.

- [ ] **Step 6: Run the full dashboard test file**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit_dashboard.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add dashboard/templates/audit_finding.html tests/v2/test_audit_dashboard.py
git commit -m "feat(dashboard): render Jira block on audit finding detail page"
```

---

### Task 3.5: Document Jira env vars + `--file-jira` flag in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Locate the env-var list**

Run: `grep -n "ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS\|Optional knobs" CLAUDE.md`

- [ ] **Step 2: Add Jira documentation**

In `CLAUDE.md`, under the "Optional knobs" section, after the existing audit-related env var documentation, add:

```markdown
**Audit Jira filing** (gated; off by default):
- `ALGO_AUDIT_FILE_JIRA=1` — file Jira tickets for new Opus ideation findings on every audit run. Without this flag, the audit runs Opus checks and writes findings to `audit_findings` but does not create tickets. Can also be enabled per-run via the `--file-jira` CLI flag on `python -m v2.audit`.
- `ALGO_AUDIT_JIRA_MAX_CREATES` — per-run cap on new Jira tickets created (default `5`). Dedup hits against existing open issues do not count against the cap. Findings beyond the cap still write to `audit_findings` with `evidence.jira.status = "capped"`.
- `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` — Atlassian credentials. If any is missing, Jira filing is silently disabled and findings record `evidence.jira.status = "disabled"`.
- `JIRA_AUDIT_PROJECT_KEY` — project key tickets are filed against (e.g. `ALGO`).
- `JIRA_AUDIT_ISSUE_TYPE` — Jira issue type for filed tickets (default `Task`).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document Jira audit-filing env vars and --file-jira flag"
```

---

### Task 3.6: Final full-suite run + smoke test

**Files:**
- None — verification only.

- [ ] **Step 1: Run all audit tests**

Run: `docker compose exec trading python -m pytest tests/v2/test_audit.py tests/v2/test_audit_jira.py tests/v2/test_audit_dashboard.py -v`
Expected: All pass.

- [ ] **Step 2: Run the full test suite**

Run: `docker compose exec trading python -m pytest tests/ -x`
Expected: All pass.

- [ ] **Step 3: Smoke-test the audit CLI in dry mode (no Jira filing)**

Stub the Opus call locally is not possible without code change; instead verify the CLI parses the new flag without error:

Run: `docker compose exec trading python -m v2.audit --help`
Expected: Help text includes `--file-jira` flag.

- [ ] **Step 4: No commit needed**

This is a verification-only step. If any step failed, stop and investigate before moving on.

---

## Self-Review

After writing this plan, fresh-eyes review against the spec:

**Spec coverage:**
- ✅ Two Opus checks added with `check_code` `AUDIT_GAP` / `APP_IMPROVEMENT` (Tasks 2.3, 2.4).
- ✅ Tier 3, severity `info`, never critical (in `_opus_finding_from_json`).
- ✅ Inputs match spec for each check (`_build_audit_gaps_prompt`, `_build_app_improvements_prompt`).
- ✅ Token cap with truncation + warning (in `_build_app_improvements_prompt`; gap-check prompt is small enough that truncation is unlikely but the same env var documented). Note: gap-check prompt does NOT enforce the cap — small enough that this is acceptable, but if real-world prompts grow, add the same truncation to it.
- ✅ Output schema with `topic_slug`, `category`, `priority` validated by `_opus_finding_from_json`.
- ✅ Coarse fingerprint = `sha256(check_code + ":" + normalized_slug)` via `_opus_topic_fingerprint`, plumbed through runner.
- ✅ `audit_llm_calls` table + accounting (Tasks 1.1, 1.2, 1.3, 2.5).
- ✅ New `check_audit_llm_cost_trend` (Task 1.5) replaces "rewrite cost_trend" idea — separate check.
- ✅ Jira gates: `--file-jira` flag OR `ALGO_AUDIT_FILE_JIRA=1`, env vars must all be set, per-run cap (Task 3.3).
- ✅ JQL dedup → POST → record outcome in `evidence.jira` (Task 3.2).
- ✅ Failure handling: failed POST → `status=failed`, audit continues (test in Task 3.2).
- ✅ Dashboard renders Jira link (Task 3.4).
- ✅ Three-step rollout matches spec (Commits 1, 2, 3).

**Placeholder scan:** No TBDs, no "implement later". Note in Task 3.4 about adapting fixture names matches existing test scaffolding — that's a realistic adapter instruction, not a placeholder.

**Type consistency:**
- `_opus_finding_from_json` returns `Finding | None` consistently.
- `_opus_topic_fingerprint(check_code, slug)` signature stable across Tasks 2.2, 2.5, 3.2.
- `file_jira_ticket(finding, *, fingerprint, run_id)` signature stable across Tasks 3.2 and 3.3.
- `insert_audit_llm_call` kwargs match between definition (Task 1.2) and callers (Tasks 1.3, 2.5).

**Known soft spots flagged for the implementer:**
- The dashboard template task assumes the existing test scaffolding pattern; adapter notes are inline.
- Gap-check prompt does not enforce the token cap (likely small). Future enhancement, not a blocker.
- The `_fp_override` mechanism uses `evidence` as a side channel because we deliberately do not modify the `Finding` dataclass. This is documented inline. Pop it before insert so it does not persist into the DB row.
