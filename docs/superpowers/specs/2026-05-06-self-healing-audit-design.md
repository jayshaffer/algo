# Self-Healing Audit — Design Spec

**Date:** 2026-05-06
**Status:** Draft, pending implementation plan
**Scope:** v2 daily auditor that backfills metadata, proposes rule overfitting / contradictions, and surfaces application issues. Findings never feed the strategist.

---

## 1. Motivation

The v2 platform has accumulated a recurring class of self-inflicted bugs that
are silent, compounding, and expensive when caught late. Examples already
documented in `v2/BUGS.md` and project memory:

- `news_signal:unknown` orphan-FK artifact promoted to **Rule #25** before
  detection (2026-05-02).
- **Rule #27** (`$500/day macro cap`) carries an unfalsifiable lift condition.
- The 2026-05-02 signal-refs wiring fix is already degrading: 13 of 29 (45%)
  recent buy/sell decisions and 56 of 69 (81%) recent theses are missing
  signal_refs in prod (verified 2026-05-06).
- `account_snapshot` trading-day gaps, missing 30d backfill rows, stale
  `running` session_stage entries — all currently present in prod, none
  surfaced by any existing system.

A daily auditor closes the gap between "we eventually find these by hand" and
"the system tells us." It is **proposer-first**: deterministic checks
auto-fix only the narrowest, safest class of integrity issues; everything
else surfaces to a human via the internal dashboard.

## 2. Non-Goals

- Feeding findings into the strategist or reflection LLM context. Explicit
  exclusion — we do not want the auditor to influence trading decisions.
- Auto-fixing strategy rules, decisions, or any trading-money path.
- Replacing `v2/BUGS.md`. Manual audit retains its place; the auditor catches
  the recurring, mechanically-detectable subset.
- Public dashboard surfacing. The audit page is internal only.

## 3. Architecture

### 3.1 Module layout

```
v2/
├── audit.py              # new — flat module, all checks + runner
├── database/
│   └── trading_db.py     # add: insert_audit_run, insert_audit_finding,
│                         #      get_open_findings, mark_finding_resolved,
│                         #      delete_orphan_decision_signals (auto-fix helper)
└── dashboard/
    └── pages/audit.py    # new — internal /audit page (private, not GH-Pages)

db/migrations/
└── 025_audit_findings.sql  # audit_runs + audit_findings tables

Taskfile.yml: + audit, audit:apply, paper:audit, paper:audit:apply targets
```

Flat-file convention matches existing v2 layout (`learn.py`, `attribution.py`,
`patterns.py`). No package nesting.

### 3.2 Module shape (`v2/audit.py`)

```python
@dataclass
class Finding:
    check_code: str           # e.g. "ORPHAN_FK_NEWS_SIGNAL"
    tier: int                 # 1 | 2 | 3
    severity: str             # "critical" | "warn" | "info"
    title: str
    body: str                 # markdown
    affected_count: int
    evidence: dict            # JSON-serialisable
    auto_fix: Callable | None # set on opt-in checks; None elsewhere

CHECKS: list[Callable[[cursor], list[Finding]]] = [
    check_orphan_fks,
    check_missing_backfill,
    check_invalid_attribution_categories,
    check_snapshot_gaps,
    check_decision_equity_drift,
    check_rule_judgment,                  # LLM, Tier 2
    check_attribution_category_coverage,
    check_stage_failure_rate,
    check_cost_trend,
    check_decisions_missing_signal_refs,
    check_theses_missing_signal_refs,
]

def run_audit(apply: bool=False) -> AuditRunSummary: ...
def main(): ...   # CLI entry: python -m v2.audit [--apply]
```

The runner does not branch on tier for storage; only on `apply` flag plus
presence of `auto_fix`.

### 3.3 Hybrid engine

Tier-1 (integrity) and Tier-3 (operational) checks are pure SQL.
The single Tier-2 check (`check_rule_judgment`) calls Haiku via the existing
`claude_client` and emits multiple findings per call. Cost is bounded
(<1% of one strategist stage per audit run).

## 4. Data Model

Migration `db/migrations/025_audit_findings.sql`:

```sql
CREATE TABLE audit_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    mode            VARCHAR(16) NOT NULL,     -- 'check' | 'apply'
    total_findings  INTEGER NOT NULL DEFAULT 0,
    auto_fixed      INTEGER NOT NULL DEFAULT 0,
    failed_checks   INTEGER NOT NULL DEFAULT 0,
    model           VARCHAR(64),
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    cache_creation_tokens INTEGER,
    cache_read_tokens     INTEGER
);

CREATE TABLE audit_findings (
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

CREATE INDEX idx_audit_findings_status   ON audit_findings(status) WHERE status='open';
CREATE INDEX idx_audit_findings_run      ON audit_findings(audit_run_id);
CREATE INDEX idx_audit_findings_code     ON audit_findings(check_code);
CREATE UNIQUE INDEX uq_audit_findings_open_fingerprint
    ON audit_findings(fingerprint) WHERE status='open';
```

### 4.1 Lifecycle

| Transition | Trigger |
|---|---|
| `→ open` | Check emits a finding whose fingerprint has no existing open row. |
| `open → auto_fixed` | Tier-1 check with auto-fix succeeds while `--apply`. |
| `open → acknowledged` | Operator action via dashboard, with note. |
| `open → resolved` | Operator action via dashboard, with note. |
| `open → superseded` | A subsequent run no longer detects this fingerprint; runner sets `resolved_note='not detected by run #N'`. |

`fingerprint = sha256(check_code + canonical_json(evidence))`. The unique
partial index enforces *at most one open finding per fingerprint*. Re-running
the auditor multiple times in one day is a no-op for already-open findings.

### 4.2 Why JSONB `evidence`

Per-check details vary widely (orphan IDs, rule IDs, decision IDs, gap dates).
JSONB avoids per-check column proliferation and lets the dashboard render
arbitrary structures. Fingerprint is computed from canonicalized evidence so
semantically-identical findings hash identically regardless of dict ordering.

## 5. Check Inventory

A finding's *tier* is its category. Whether a check has an auto-fix is a
**per-check** property declared on the function. Default action is propose;
auto-fix is opt-in and gated by the `--apply` CLI flag.

| Check function | Tier | check_code(s) emitted | Auto-fix? | Severity rule |
|---|---|---|---|---|
| `check_orphan_fks` | 1 | `ORPHAN_FK_NEWS_SIGNAL`, `ORPHAN_FK_MACRO_SIGNAL`, `ORPHAN_FK_THESIS` | ✅ DELETE orphan rows from `decision_signals` | warn |
| `check_missing_backfill` | 1 | `BACKFILL_GAP_7D`, `BACKFILL_GAP_30D` | ✅ Re-invoke `backfill.backfill_decision_outcomes(id)` per row | warn (critical if >25 rows) |
| `check_invalid_attribution_categories` | 1 | `INVALID_ATTRIBUTION_CATEGORY` | ❌ propose only | critical |
| `check_snapshot_gaps` | 1 | `SNAPSHOT_GAP` | ❌ Alpaca historical retrieval is unreliable | warn |
| `check_decision_equity_drift` | 1 | `DECISION_EQUITY_DRIFT` | ❌ bug detector for P0.4-style regression | critical |
| `check_rule_judgment` (LLM) | 2 | `RULE_UNFALSIFIABLE_LIFT`, `RULE_LOW_N_BACKING`, `RULE_RETIRED_BUCKET`, `RULE_CONTRADICTION`, `RULE_DEAD` | ❌ rule edits go through human + reflection | warn |
| `check_attribution_category_coverage` | 3 | `ATTRIBUTION_COVERAGE_LOW` | ❌ wiring regression signal | warn |
| `check_stage_failure_rate` | 3 | `STAGE_FAILURE_RATE`, `STAGE_RUNNING_STALE` | ❌ propose only | warn (critical if rate >50%) |
| `check_cost_trend` | 3 | `COST_TREND_SPIKE` | ❌ propose only | info |
| `check_decisions_missing_signal_refs` | 3 | `DECISIONS_NO_SIGNAL_REFS` | ❌ propose only | critical when on-playbook share >10% |
| `check_theses_missing_signal_refs` | 3 | `THESES_NO_SIGNAL_REFS` | ❌ strategist Rule #6 drift | warn |

Total: **11 functions** emitting up to **19 distinct check_codes**
(3 orphan-FK + 2 backfill + 1 invalid-category + 1 snapshot-gap + 1
equity-drift + 5 rule-judgment + 1 coverage-low + 2 stage-health + 1
cost-trend + 1 decisions-no-refs + 1 theses-no-refs).

`check_invalid_attribution_categories` validates each
`signal_attribution.category` against the classifier's
`VALID_TICKER_CATEGORIES` and `VALID_MACRO_CATEGORIES` frozensets
(`v2/classifier.py`, defined per BUGS.md P0.3 fix). Any category outside
those enums emits a finding.

`check_snapshot_gaps` and any other "trading day" check use the existing
NYSE calendar helper (`v2/market_calendar.py`); weekends and NYSE holidays
are excluded from gap detection.

### 5.1 Auto-fix safety

- Auto-fix runs only when CLI flag `--apply` is set. Default mode is dry-run.
- Each auto-fix is bounded:
  - Orphan-FK delete touches only `decision_signals`.
  - Backfill re-run delegates to the existing tested
    `backfill.backfill_decision_outcomes` function.
- No auto-fix touches trading-money paths (`positions`, `decisions.action`,
  `playbooks`, etc.).
- Each auto-fix records a finding row with `status='auto_fixed'` and the
  affected row IDs/operations in `evidence`. Full forensic trail.
- `MAX_AUTO_FIX = 100` (configurable via `--max-auto-fix`). Exceeded ⇒ check
  emits the finding but performs no fix; severity escalates to critical.

### 5.2 Severity-to-action coupling

| Severity | Action |
|---|---|
| `info` | Surfaced in dashboard. No log alert. |
| `warn` | Surfaced + logged at WARNING. |
| `critical` | Surfaced + logged at ERROR + non-zero exit code (cron MAILTO triggers email). |

### 5.3 Excluded from v1

- Auto-fix for `check_snapshot_gaps` (Alpaca historical equity retrieval is
  unreliable).
- Cross-rule contradiction detection beyond what the LLM catches in one pass.
- Auto-cleanup of `superseded` findings — keep history; manual prune later if
  table size becomes a concern.
- `positions` divergence checks (already covered by existing Alpaca sync).

## 6. LLM Judgment Subsystem (`check_rule_judgment`)

### 6.1 Inputs assembled before the LLM call (pure SQL)

1. All `strategy_rules` rows where `status='active'` — `id`, full `rule_text`,
   `created_at`.
2. Full `signal_attribution` snapshot — every category with `sample_size`,
   `sample_size_30d`, `avg_outcome_7d`, `win_rate_7d`, `avg_outcome_30d`,
   `win_rate_30d`.
3. Per-rule citation counts in `decisions.reasoning` over last 30 days
   (regex `\brule\s*#?\s*<id>\b` per rule id).
4. Decision-data summary for contradiction detection:
   `{recent_buys_with_empty_signal_refs, recent_thesis_only_decisions, recent_off_playbook_buys}`.

These four blobs serialize to roughly 5–15 KB of compact text. Bounded by data
shape (≤30 active rules, ≤50 categories), not LLM judgment.

### 6.2 Single completion, structured output

```python
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["check_code", "rule_id", "title", "explanation"],
                "properties": {
                    "check_code": {"enum": [
                        "RULE_UNFALSIFIABLE_LIFT", "RULE_LOW_N_BACKING",
                        "RULE_RETIRED_BUCKET",     "RULE_CONTRADICTION",
                        "RULE_DEAD",
                    ]},
                    "rule_id": {"type": "integer"},
                    "title": {"type": "string", "maxLength": 160},
                    "explanation": {"type": "string", "maxLength": 1200},
                    "evidence_quote": {"type": "string", "maxLength": 400},
                    "contradicts_rule_id": {"type": ["integer", "null"]},
                }
            }
        }
    }
}
```

### 6.3 Model & call shape

- Model: `claude-haiku-4-5-20251001` (matches executor stage tier).
- `max_tokens=4000` (hard ceiling on cost regardless of model behavior).
- System prompt: defines the 5 check_codes, evidence bar required for each,
  and "be conservative — if unsure, omit." Examples drawn from Rule #25
  (retired-bucket) and Rule #27 (unfalsifiable-lift) histories.
- No tool use. No prompt caching v1.
- Wrapped in `capture_usage` so tokens land on the `audit_runs` row.

### 6.4 Validation pipeline (defensive)

```python
def _parse_rule_judgment_findings(response_text, active_rule_ids,
                                  attribution_categories, citation_counts):
    raw = _extract_json(response_text)
    findings = []
    for item in raw.get("findings", []):
        if item.get("check_code") not in VALID_CHECK_CODES: continue
        if item.get("rule_id") not in active_rule_ids: continue
        # cross-check claimed RULE_DEAD against own citation count
        if item["check_code"] == "RULE_DEAD" and citation_counts[item["rule_id"]] > 0:
            log.warning("LLM asserted RULE_DEAD for cited rule_id=%d, dropping",
                        item["rule_id"])
            continue
        # cross-check RULE_RETIRED_BUCKET against attribution snapshot
        if item["check_code"] == "RULE_RETIRED_BUCKET":
            cited = _extract_category_from_evidence(item.get("evidence_quote",""))
            if cited and cited in attribution_categories: continue
        findings.append(_to_finding(item))
    return findings[:20]  # truncate runaway output
```

The LLM is a **proposer** whose proposals must survive deterministic
validation against the same SQL data we already have. If it claims "rule
cites a retired bucket" but the bucket is in `signal_attribution`, the
finding is dropped.

### 6.5 Cost bounding

- Single call per audit run.
- `max_tokens=4000` × Haiku rate ≈ pennies per run.
- Daily cron ⇒ ≤30 calls/month. Trivial vs. strategist daily spend
  (memory: 431k–551k tokens/day for the strategist alone).

### 6.6 Failure handling

- API failure or malformed JSON: the check function re-raises; the runner
  converts it into a `CHECK_FAILED` finding with the exception in evidence.
  No Tier-2 findings emitted for that run.

## 7. Runner Orchestration & Error Handling

```python
def run_audit(apply: bool = False) -> AuditRunSummary:
    run_id = insert_audit_run(mode='apply' if apply else 'check')
    findings_emitted = auto_fixed = failed_checks = 0
    current_fingerprints: set[str] = set()

    if not _try_advisory_lock():
        log.warning("Audit already running; exiting cleanly")
        return AuditRunSummary(run_id=None, ...)

    with get_cursor() as cur:
        for check in CHECKS:
            try:
                check_findings = check(cur)
            except Exception as e:
                log.exception("Audit check %s failed", check.__name__)
                _emit_check_failure(run_id, check.__name__, e)
                failed_checks += 1
                continue

            for f in check_findings:
                current_fingerprints.add(f.fingerprint)
                outcome = _persist_or_supersede(run_id, f)
                if outcome == 'inserted':
                    findings_emitted += 1
                if apply and f.auto_fix is not None:
                    if auto_fixed >= MAX_AUTO_FIX:
                        _escalate_to_critical(f, "auto-fix ceiling reached")
                        continue
                    try:
                        fix_evidence = f.auto_fix(cur)
                        _mark_auto_fixed(f, fix_evidence)
                        auto_fixed += 1
                    except Exception as e:
                        log.exception("Auto-fix failed for %s", f.check_code)
                        _record_autofix_failure(f, e)

        _supersede_stale_open_findings(run_id, current_fingerprints)

    finalize_audit_run(run_id,
        total_findings=findings_emitted,
        auto_fixed=auto_fixed,
        failed_checks=failed_checks,
        usage=usage_accumulator.totals())

    return AuditRunSummary(run_id, findings_emitted, auto_fixed, failed_checks)
```

### 7.1 Behaviors

- **Single connection, savepoint per check.** The runner opens a
  `SAVEPOINT check_<name>` before each `check(cur)` call and either
  `RELEASE`s it on success or `ROLLBACK TO`s it on failure. A check that
  raises mid-iteration rolls back partially-emitted findings for *that*
  check; previously-completed checks stay committed and the connection
  remains usable for subsequent checks.
- **Per-check isolation.** `try/except` around each `check(cur)` call. A
  failure becomes a `CHECK_FAILED` finding with exception + traceback in
  evidence.
- **Idempotency.** `_persist_or_supersede` uses the unique partial index
  from §4. Insert with already-open fingerprint is a no-op.
- **Stale-finding supersession.** After all checks complete, any open
  finding whose fingerprint isn't in `current_fingerprints` flips to
  `superseded` with a timestamp and note.
- **Auto-fix safety.** `MAX_AUTO_FIX = 100` ceiling, savepoint per fix so
  one failure doesn't taint the run, fix evidence merged into finding row.
- **Concurrency.** Postgres advisory lock at run start prevents overlapping
  cron firings.

### 7.2 Exit codes

| Code | Meaning |
|---|---|
| 0 | Run completed, no critical findings open. |
| 1 | Run completed, ≥1 critical finding open. |
| 2 | Run itself failed unrecoverably (DB connection lost, migration mismatch, finalize_audit_run write failure). Advisory-lock contention is *not* an exit-2 case — it logs and exits 0 since the other run will cover the window. |

Cron's standard MAILTO triggers email on 1 and 2.

### 7.3 Logging

Uses existing `log_config`. INFO for run start/end + per-check headlines;
WARNING for warn findings; ERROR for critical and check failures.

## 8. Cron / Taskfile Integration

### 8.1 Taskfile additions

```yaml
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
  desc: Run paper auditor with Tier-1 auto-fixes
  cmds:
    - docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T trading-paper python -m v2.audit --apply
```

### 8.2 CLI shape

```
python -m v2.audit                       # propose-only
python -m v2.audit --apply               # apply Tier-1 auto-fixes
python -m v2.audit --max-auto-fix N      # override ceiling (default 100)
python -m v2.audit --check ORPHAN_FK_NEWS_SIGNAL  # run a single check (debug)
python -m v2.audit --since 7d            # only emit findings new vs last 7d
```

### 8.3 Cron entry (host crontab, documented in README)

```
# Daily audit, prod, propose-only, after typical post-session window
30 22 * * * cd /home/jay/dev/algo && /usr/local/bin/task audit >> logs/cron-audit.log 2>&1
```

- 22:30 runs after the daily session typically completes.
- Default is propose-only; `task audit:apply` is documented in README but
  not enabled in cron until the operator has triaged a few runs manually.
- No paper cron in v1 — paper audits are user-initiated.

## 9. Dashboard Surfacing

Internal v2 dashboard (port 3000) gets a new private route `/audit/`.

### 9.1 Page structure

1. **Header**: latest run id, started_at, finishing status, total findings
   open, auto-fixed last run.
2. **Open findings table**, grouped by severity → tier → check_code:
   - Columns: severity badge | check_code | title | affected_count |
     first_seen | last_seen | actions
   - Action buttons: `[Acknowledge]` `[Resolve]` POST to
     `/audit/findings/<id>/status`.
3. **Run history** (last 14 runs): id, started_at, total_findings,
   auto_fixed, failed_checks, model + token totals, cost (via
   `pricing.stage_cost_usd`).
4. **Per-finding detail page** (`/audit/findings/<id>/`): full body markdown
   + raw evidence JSON.

### 9.2 Public dashboard untouched

`dashboard_publish.py` is not modified. The audit page lives only on the
internal Flask service bound to `127.0.0.1`. No auth needed.

## 10. Testing Strategy

Tests live in `tests/v2/test_audit.py`. Per existing v2 test conventions
(mocked `get_cursor`, factory fixtures from `conftest.py`).

### 10.1 Per-check unit tests

Each check function gets focused tests inserting a known-bad fixture and
asserting `Finding` shape. Examples:

- `test_orphan_fks_finds_news_signal_orphans` — insert
  decision_signal pointing at non-existent news_signal id; expect 1
  finding with that id in evidence.
- `test_missing_backfill_skips_recent_decisions` — decision dated 2 days
  ago with NULL outcome_7d; expect no finding (still inside window).
- `test_invalid_attribution_categories_detects_typo` — insert
  `signal_attribution` row with category `news_signal:earnigns`; expect 1
  finding.
- `test_decisions_missing_signal_refs_separates_on_off_playbook` — verify
  the on-vs-off split flips severity threshold correctly.

### 10.2 LLM check tests

Stub `claude_client.call_with_retry` to return canned JSON. Cover:

- `test_rule_judgment_drops_unverifiable_dead_rule` — LLM returns
  `RULE_DEAD` for a rule the citation count contradicts; parser drops it.
- `test_rule_judgment_drops_unverifiable_retired_bucket` — LLM cites
  category present in `signal_attribution`; parser drops it.
- `test_rule_judgment_truncates_runaway_output` — LLM returns 50 findings;
  parser keeps 20.
- `test_rule_judgment_validates_against_active_rule_ids` — LLM emits
  rule_id not in active set; parser drops.

### 10.3 Auto-fix tests

Each Tier-1 check with auto-fix gets a test asserting both the fix happens
*and* the finding is recorded with `status='auto_fixed'` plus merged evidence.

### 10.4 Runner tests

- **Idempotency.** Run twice; assert no duplicate open findings.
- **Supersession.** Run with mock check returning a finding, then with check
  returning nothing; assert prior finding flips to `superseded`.
- **Per-check isolation.** Check raises; later checks still run; `CHECK_FAILED`
  finding is recorded.
- **Concurrency.** Two parallel `run_audit()` calls — second exits cleanly via
  advisory lock.
- **Auto-fix ceiling.** Inject 101 fixable findings; assert 100 fixed, 101st
  is escalated to critical.

### 10.5 Migration test

`test_migration_025` follows existing pattern (apply migration on a fresh
test DB, assert tables/indexes exist).

### 10.6 Coverage target

≥30 new tests, maintaining the project's 89% coverage discipline.

## 11. Rollout Plan

1. Migration 025 applied to prod and paper.
2. v2.audit module shipped with all 11 checks, tests passing.
3. First manual run on prod: `task audit` (propose-only). Operator reviews
   findings against `/audit/` dashboard page.
4. After 1 week of clean propose-only runs and manual validation of the
   findings, enable `task audit:apply` in cron.
5. Triage cycle: operator reviews open findings weekly, acknowledges or
   resolves each. Recurring critical findings drive code or rule fixes.

## 12. Open Questions

None at design-spec finalization. Implementation plan will refine
operational details (exact regex for rule citation matching, advisory lock
key allocation, dashboard CSS).
