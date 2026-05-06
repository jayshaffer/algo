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
from dataclasses import dataclass
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
        title=f"Only {len(qualifying)} attribution categories with n_30d>={ATTRIBUTION_MIN_N_PER_CATEGORY}",
        body=("`signal_attribution` has fewer than the expected number of populated "
              f"categories. Threshold: >= {ATTRIBUTION_MIN_CATEGORIES} with "
              f"sample_size_30d >= {ATTRIBUTION_MIN_N_PER_CATEGORY}. Likely cause: "
              "signal_refs wiring (strategist->playbook->executor) has regressed."),
        affected_count=len(qualifying),
        evidence={
            "qualifying_categories": [r["category"] for r in qualifying],
            "all_categories": [r["category"] for r in rows],
        },
        auto_fix=None,
    )]


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
            title=f"{len(flagged)} stage(s) with failure rate >= 20% in last 30d",
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


# --- Tier 3: per-stage cost trend (info) ---------------------------------

def check_cost_trend(cur) -> list[Finding]:
    """7d-vs-prior-7d total token usage by stage. Flag stages with >=2x growth."""
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
        title=f"{len(spikes)} stage(s) with token usage >=2x prior 7-day window",
        body="Per-stage 7-day-rolling token totals doubled vs. prior 7-day window.",
        affected_count=len(spikes),
        evidence={"stages": spikes},
        auto_fix=None,
    )]


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
        body=("Strategist->playbook->executor signal_refs wiring is being honored "
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
