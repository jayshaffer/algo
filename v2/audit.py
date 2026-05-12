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
import re as _re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

MAX_AUTO_FIX_DEFAULT = 100
RULE_JUDGMENT_MODEL = "claude-haiku-4-5-20251001"
RULE_JUDGMENT_MAX_TOKENS = 4000
OPUS_IDEATION_MODEL = "claude-opus-4-7"
OPUS_IDEATION_MAX_TOKENS = 4000
OPUS_INPUT_TOKEN_CAP_DEFAULT = 60_000


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


_SLUG_RE = _re.compile(r"[^a-z0-9]+")
_VALID_OPUS_CATEGORIES = {"audit_gap", "app_improvement"}
_VALID_OPUS_PRIORITIES = {"high", "medium", "low"}


def _normalize_slug(s: str) -> str:
    """Kebab-case normalization: lowercase, alphanumerics + single hyphens."""
    return _SLUG_RE.sub("-", s.lower()).strip("-")


def _opus_topic_fingerprint(check_code: str, topic_slug: str) -> str:
    """Coarse fingerprint: hash(check_code + ":" + normalized_slug) only.

    Deliberately ignores evidence prose so daily re-emissions of the same
    underlying topic collapse to one open finding. See spec
    2026-05-12-opus-audit-ideation-design.md.
    """
    canonical = f"{check_code}:{_normalize_slug(topic_slug)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _opus_finding_from_json(item: dict, *, default_category: str) -> "Finding | None":
    """Map one Opus finding dict to a Finding. Returns None on validation failure.

    Stashes the coarse topic-based fingerprint in evidence['_fp_override'];
    the audit runner pops it and uses it as the insert fingerprint instead
    of the default Finding.fingerprint property.
    """
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

    evidence: dict = {
        "topic_slug": slug,
        "category": category,
        "priority": priority,
        "evidence_quote": (item.get("evidence_quote") or "")[:600],
    }
    if category == "audit_gap" and item.get("proposed_check_code"):
        evidence["proposed_check_code"] = str(item["proposed_check_code"])[:64]

    f = Finding(
        check_code=check_code,
        tier=3,
        severity="info",
        title=title[:200],
        body=body[:2000],
        affected_count=1,
        evidence=evidence,
        auto_fix=None,
    )
    # Coarse fingerprint override — runner reads this instead of f.fingerprint.
    # See spec; ignoring evidence prose is deliberate.
    f.evidence["_fp_override"] = _opus_topic_fingerprint(check_code, slug)
    return f


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
    backfill function.

    Uses trading-day cutoff (not calendar days) so the eligibility predicate
    matches what backfill_decision_outcomes can actually fill — its exit_date
    is N trading days after the decision, so a calendar-day cutoff over-flags
    decisions whose trading-day window hasn't closed yet.
    """
    from datetime import date as _date
    from v2.backfill import trading_day_cutoff
    findings = []
    for window, code in (("7d", "BACKFILL_GAP_7D"), ("30d", "BACKFILL_GAP_30D")):
        days = 7 if window == "7d" else 30
        cutoff = trading_day_cutoff(_date.today(), days)
        cur.execute(f"""
            SELECT id FROM decisions
            WHERE action IN ('buy','sell')
              AND date <= %s
              AND (outcome_{window} IS NULL OR benchmark_{window} IS NULL)
            ORDER BY id
        """, (cutoff,))
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


def check_audit_llm_cost_trend(cur) -> list[Finding]:
    """Per-purpose 7d-vs-prior-7d audit LLM token usage. Flag >=2x growth.

    Reads audit_llm_calls; emits one info finding listing all spiking purposes.
    Separate from check_cost_trend (which reads session_stages).
    """
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


def _call_opus_ideation(system: str, prompt: str) -> tuple[dict, dict]:
    """Single Opus call returning (parsed_json, usage_with_latency_ms).

    Caches the `system` block via cache_control so repeated daily runs
    amortize the system-prompt cost. Used by check_audit_gaps_opus and
    check_app_improvements_opus.
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


def _extract_json(text: str) -> dict:
    """Find the first JSON object in text and parse it. Returns {} on failure."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        log.warning("Could not extract JSON from LLM response: %s", text[:200])
        return {}


_LAST_RULE_JUDGMENT_USAGE: dict = {}


def get_last_rule_judgment_usage() -> dict:
    return _LAST_RULE_JUDGMENT_USAGE.copy()


def _reset_rule_judgment_usage() -> None:
    _LAST_RULE_JUDGMENT_USAGE.clear()


# --- Phase 1: agent_events-driven checks ---------------------------------

# Tools the strategist should be using on a typical ideation run. A persistent
# zero across the last 7 sessions for any of these is a drought. Kept narrow
# on purpose: don't include narrow rare-call tools (close_thesis, adopt_thesis)
# whose absence is normal for many sessions.
EXPECTED_IDEATION_TOOLS = (
    "get_portfolio_state",
    "get_active_theses",
    "get_news_signals",
    "get_signal_attribution",
    "get_recent_playbooks",
    "write_playbook",
)


def check_strategist_using_reversal_tool(cur) -> list[Finding]:
    """Warn when 3 consecutive sessions had round-trip evidence shown but the
    ideation stage never called `get_recent_playbooks`. The reversal tool
    exists specifically so the strategist can sanity-check its own flip-flops
    against prior plays — silence here means the loop isn't using the
    feedback we wired up."""
    cur.execute(
        """
        WITH recent_sessions AS (
            SELECT DISTINCT session_id
            FROM agent_events
            WHERE session_id IS NOT NULL
              AND occurred_at > now() - interval '14 days'
            ORDER BY session_id DESC
            LIMIT 3
        )
        SELECT s.session_id,
               SUM(CASE
                     WHEN e.event_type='evidence_shown'
                      AND e.payload->>'evidence_kind'='round_trips'
                      AND COALESCE(jsonb_array_length(e.payload->'items'), 0) > 0
                     THEN 1 ELSE 0 END) AS reversal_evidence,
               SUM(CASE
                     WHEN e.event_type='tool_invocation'
                      AND e.payload->>'tool_name'='get_recent_playbooks'
                      AND e.stage_name='ideation'
                     THEN 1 ELSE 0 END) AS lookup_calls
        FROM recent_sessions s
        LEFT JOIN agent_events e ON e.session_id = s.session_id
        GROUP BY s.session_id
        ORDER BY s.session_id DESC
        """
    )
    rows = cur.fetchall()
    if len(rows) < 3:
        return []
    if not all(r["reversal_evidence"] >= 1 and r["lookup_calls"] == 0 for r in rows):
        return []
    return [Finding(
        check_code="STRATEGIST_NOT_USING_REVERSAL_TOOL",
        tier=3, severity="warn",
        title="Strategist not consulting prior playbooks despite round-trip evidence",
        body=(
            "3 consecutive sessions showed round-trip evidence to reflection, but the "
            "ideation stage never called `get_recent_playbooks` to compare against prior "
            "plays. The reversal-lookup tool exists for this exact case."
        ),
        affected_count=len(rows),
        evidence={"sessions": [dict(r) for r in rows]},
        auto_fix=None,
    )]


def check_reflection_inert_on_round_trips(cur) -> list[Finding]:
    """Warn when 5 consecutive sessions had round-trip evidence shown to
    reflection AND zero rule actions (propose_rule / retire_rule) from
    reflection in those sessions. Reflection is supposed to convert
    repeated-flip-flop signal into rule changes."""
    cur.execute(
        """
        WITH recent_sessions AS (
            SELECT DISTINCT session_id
            FROM agent_events
            WHERE session_id IS NOT NULL
              AND occurred_at > now() - interval '21 days'
            ORDER BY session_id DESC
            LIMIT 5
        )
        SELECT s.session_id,
               SUM(CASE
                     WHEN e.event_type='evidence_shown'
                      AND e.payload->>'evidence_kind'='round_trips'
                      AND COALESCE(jsonb_array_length(e.payload->'items'), 0) > 0
                     THEN 1 ELSE 0 END) AS reversal_evidence,
               SUM(CASE
                     WHEN e.event_type='tool_invocation'
                      AND e.stage_name='reflection'
                      AND e.payload->>'tool_name' IN ('propose_rule', 'retire_rule')
                     THEN 1 ELSE 0 END) AS rule_actions
        FROM recent_sessions s
        LEFT JOIN agent_events e ON e.session_id = s.session_id
        GROUP BY s.session_id
        ORDER BY s.session_id DESC
        """
    )
    rows = cur.fetchall()
    if len(rows) < 5:
        return []
    if not all(r["reversal_evidence"] >= 1 and r["rule_actions"] == 0 for r in rows):
        return []
    return [Finding(
        check_code="REFLECTION_INERT_ON_ROUND_TRIPS",
        tier=3, severity="warn",
        title="Reflection not acting on round-trip evidence",
        body=(
            "5 consecutive sessions surfaced round-trip evidence to reflection, but no "
            "rules were proposed or retired in any of them. Either the rules already "
            "cover the pattern (and reflection should retire dead rules), or reflection "
            "is ignoring the signal."
        ),
        affected_count=len(rows),
        evidence={"sessions": [dict(r) for r in rows]},
        auto_fix=None,
    )]


def check_tool_error_rate(cur) -> list[Finding]:
    """Warn when any tool's 7-day error rate ≥20% (critical at ≥50%).
    Skips tools with fewer than 5 invocations to avoid small-N noise."""
    cur.execute(
        """
        SELECT payload->>'tool_name' AS tool_name,
               COUNT(*) AS n,
               COUNT(*) FILTER (
                   WHERE COALESCE((payload->>'success')::boolean, true) = false
               ) AS errors
        FROM agent_events
        WHERE event_type = 'tool_invocation'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        """
    )
    rows = cur.fetchall()
    flagged = []
    for r in rows:
        if r["n"] < 5:
            continue
        rate = (r["errors"] or 0) / r["n"]
        if rate >= 0.20:
            flagged.append({
                "tool_name": r["tool_name"], "n": r["n"],
                "errors": r["errors"], "rate": round(rate, 3),
            })
    if not flagged:
        return []
    worst = max(f["rate"] for f in flagged)
    sev = "critical" if worst >= 0.50 else "warn"
    return [Finding(
        check_code="TOOL_ERROR_RATE",
        tier=3, severity=sev,
        title=f"{len(flagged)} tool(s) with error rate >= 20% in last 7d",
        body="Tool handlers are raising exceptions. See evidence for per-tool rates.",
        affected_count=len(flagged),
        evidence={"tools": flagged},
        auto_fix=None,
    )]


def check_risk_block_hotspot(cur) -> list[Finding]:
    """Warn when the same ticker has been blocked by the sector-cap gate
    ≥3 times in 7 days. Indicates the strategist keeps proposing the same
    blocked buy without adapting."""
    cur.execute(
        """
        SELECT payload->>'ticker' AS ticker, COUNT(*) AS n
        FROM agent_events
        WHERE event_type = 'risk_block'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        ORDER BY n DESC
        """
    )
    rows = [r for r in cur.fetchall() if r["n"] >= 3]
    if not rows:
        return []
    return [Finding(
        check_code="RISK_BLOCK_HOTSPOT",
        tier=3, severity="warn",
        title=f"{len(rows)} ticker(s) blocked by sector-cap >= 3x in 7d",
        body=(
            "The same ticker is being repeatedly proposed and blocked by the sector-cap "
            "gate. The strategist isn't reacting to the rejection — review whether the "
            "cap or the thesis needs to change."
        ),
        affected_count=len(rows),
        evidence={"tickers": [{"ticker": r["ticker"], "n": r["n"]} for r in rows]},
        auto_fix=None,
    )]


def check_risk_block_burst(cur) -> list[Finding]:
    """Warn when ≥5 risk_blocks fired on a single calendar date in 14 days.
    A single bad session that hammered the gate."""
    cur.execute(
        """
        SELECT date_trunc('day', occurred_at)::date AS d, COUNT(*) AS n
        FROM agent_events
        WHERE event_type = 'risk_block'
          AND occurred_at > now() - interval '14 days'
        GROUP BY 1
        ORDER BY d DESC
        """
    )
    rows = [r for r in cur.fetchall() if r["n"] >= 5]
    if not rows:
        return []
    return [Finding(
        check_code="RISK_BLOCK_BURST",
        tier=3, severity="warn",
        title=f"{len(rows)} day(s) with >= 5 sector-cap rejections",
        body=(
            "A single session generated 5+ sector-cap rejections. Either the strategist "
            "ignored advisory text, or the cap is wrong for current portfolio shape."
        ),
        affected_count=len(rows),
        evidence={"dates": [{"date": r["d"].isoformat(), "n": r["n"]} for r in rows]},
        auto_fix=None,
    )]


def check_ideation_tool_drought(cur) -> list[Finding]:
    """Warn when any tool in EXPECTED_IDEATION_TOOLS has zero invocations
    across the last 7 ideation sessions. Catches the strategist silently
    dropping a tool — rare and almost always a regression."""
    cur.execute(
        """
        SELECT COUNT(DISTINCT session_id) AS n_sessions
        FROM agent_events
        WHERE event_type = 'tool_invocation'
          AND stage_name = 'ideation'
          AND session_id IS NOT NULL
          AND occurred_at > now() - interval '7 days'
        """
    )
    row = cur.fetchone() or {"n_sessions": 0}
    if (row["n_sessions"] or 0) < 3:
        return []

    cur.execute(
        """
        SELECT payload->>'tool_name' AS tool_name, COUNT(*) AS n
        FROM agent_events
        WHERE event_type = 'tool_invocation'
          AND stage_name = 'ideation'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        """
    )
    seen = {r["tool_name"]: r["n"] for r in cur.fetchall()}
    missing = [t for t in EXPECTED_IDEATION_TOOLS if seen.get(t, 0) == 0]
    if not missing:
        return []
    return [Finding(
        check_code="IDEATION_TOOL_DROUGHT",
        tier=3, severity="warn",
        title=f"{len(missing)} expected ideation tool(s) unused in last 7d",
        body=(
            "One or more tools in the expected ideation toolset went unused across the "
            "last 7 days of ideation sessions. Either prompt drift dropped them or the "
            "loop is silently skipping them — both are regressions."
        ),
        affected_count=len(missing),
        evidence={"missing_tools": missing, "observed_counts": seen},
        auto_fix=None,
    )]


# --- Phase 2: agent_call / loop / executor_response checks ---------------

EXECUTOR_TRUNC_WARN_PCT = 0.10
EXECUTOR_TRUNC_CRITICAL_PCT = 0.25
EXECUTOR_TRUNC_MIN_N = 5


def check_executor_truncation_rate(cur) -> list[Finding]:
    """Executor calls hitting max_tokens. Indicates input context too large
    or max_tokens budget set too low. Both are silent quality regressions."""
    cur.execute(
        """
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (
                 WHERE payload->>'stop_reason' = 'max_tokens'
               ) AS truncated
        FROM agent_events
        WHERE event_type = 'agent_call'
          AND payload->>'purpose' = 'executor'
          AND occurred_at > now() - interval '14 days'
        """
    )
    r = cur.fetchone() or {"total": 0, "truncated": 0}
    if r["total"] < EXECUTOR_TRUNC_MIN_N or not r["truncated"]:
        return []
    rate = r["truncated"] / r["total"]
    if rate < EXECUTOR_TRUNC_WARN_PCT:
        return []
    sev = "critical" if rate >= EXECUTOR_TRUNC_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="EXECUTOR_TRUNCATION_RATE",
        tier=3, severity=sev,
        title=f"Executor truncated on {r['truncated']}/{r['total']} calls (last 14d)",
        body=("Executor responses hit max_tokens. Input context may be too "
              "large; check ExecutorInput field sizes. Or model max_tokens "
              "budget needs raising."),
        affected_count=r["truncated"],
        evidence={"total": r["total"], "truncated": r["truncated"], "rate": round(rate, 3)},
        auto_fix=None,
    )]


SCHEMA_DRIFT_MIN_OCCURRENCES = 3


def check_executor_schema_drift(cur) -> list[Finding]:
    """LLM emitting JSON keys we don't parse. Canary for executor schema
    drift across model versions or prompt changes."""
    cur.execute(
        """
        SELECT key, COUNT(*) AS n
        FROM agent_events,
             LATERAL jsonb_array_elements_text(payload->'unknown_top_level_keys') AS t(key)
        WHERE event_type = 'executor_response'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        """,
        (SCHEMA_DRIFT_MIN_OCCURRENCES,),
    )
    top_drift = cur.fetchall()

    cur.execute(
        """
        SELECT key, COUNT(*) AS n
        FROM agent_events,
             LATERAL jsonb_array_elements_text(payload->'unknown_decision_keys') AS t(key)
        WHERE event_type = 'executor_response'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        """,
        (SCHEMA_DRIFT_MIN_OCCURRENCES,),
    )
    dec_drift = cur.fetchall()

    if not top_drift and not dec_drift:
        return []
    return [Finding(
        check_code="EXECUTOR_SCHEMA_DRIFT",
        tier=3, severity="warn",
        title=f"Executor LLM emitting unknown JSON keys (top:{len(top_drift)}, decision:{len(dec_drift)})",
        body=("Executor response contains JSON fields not in our canonical "
              "key sets. Either the prompt is requesting new fields the "
              "parser doesn't handle, or the LLM is emitting drift we should "
              "either consume or suppress. Update EXECUTOR_KNOWN_*_KEYS or "
              "the parser in v2/agent.py."),
        affected_count=len(top_drift) + len(dec_drift),
        evidence={
            "top_level_drift": [{"key": r["key"], "n": r["n"]} for r in top_drift],
            "decision_drift": [{"key": r["key"], "n": r["n"]} for r in dec_drift],
        },
        auto_fix=None,
    )]


PARSE_FAIL_WARN_PCT = 0.05
PARSE_FAIL_CRITICAL_PCT = 0.15
PARSE_FAIL_MIN_N = 5


def check_executor_parse_failure_rate(cur) -> list[Finding]:
    """Executor responses failing to parse for reasons OTHER than truncation.
    Distinct from EXECUTOR_TRUNCATION_RATE: this catches malformed JSON."""
    cur.execute(
        """
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (
                 WHERE (payload->>'parse_succeeded')::boolean = false
                   AND payload->>'error' NOT LIKE 'max_tokens%%'
               ) AS parse_failed
        FROM agent_events
        WHERE event_type = 'executor_response'
          AND occurred_at > now() - interval '14 days'
        """
    )
    r = cur.fetchone() or {"total": 0, "parse_failed": 0}
    if r["total"] < PARSE_FAIL_MIN_N or not r["parse_failed"]:
        return []
    rate = r["parse_failed"] / r["total"]
    if rate < PARSE_FAIL_WARN_PCT:
        return []
    sev = "critical" if rate >= PARSE_FAIL_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="EXECUTOR_PARSE_FAILURE_RATE",
        tier=3, severity=sev,
        title=f"Executor JSON parse failed on {r['parse_failed']}/{r['total']} non-truncated calls (last 14d)",
        body=("Executor responses are failing JSON parse for reasons other "
              "than max_tokens truncation. Likely causes: prompt regression "
              "causing prose responses, fenced-code-block edge cases the "
              "stripper misses, or the LLM returning structured output in a "
              "different shape. Inspect raw_response_text_truncated on "
              "recent failures."),
        affected_count=r["parse_failed"],
        evidence={"total": r["total"], "parse_failed": r["parse_failed"], "rate": round(rate, 3)},
        auto_fix=None,
    )]


CLASSIFIER_WARN_PCT = 0.10
CLASSIFIER_CRITICAL_PCT = 0.25
CLASSIFIER_MIN_N = 10


def check_classifier_error_rate(cur) -> list[Finding]:
    """News/macro/relevance classifier failing on items in pipeline stage."""
    cur.execute(
        """
        SELECT payload->>'purpose' AS purpose,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE COALESCE((payload->>'success')::boolean, true) = false) AS errors
        FROM agent_events
        WHERE event_type = 'agent_call'
          AND payload->>'purpose' LIKE 'classifier_%%'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        """,
        (CLASSIFIER_MIN_N,),
    )
    flagged = []
    for r in cur.fetchall():
        rate = r["errors"] / r["total"] if r["total"] else 0
        if rate >= CLASSIFIER_WARN_PCT:
            flagged.append({"purpose": r["purpose"], "total": r["total"],
                            "errors": r["errors"], "rate": round(rate, 3)})
    if not flagged:
        return []
    worst = max(f["rate"] for f in flagged)
    sev = "critical" if worst >= CLASSIFIER_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="CLASSIFIER_ERROR_RATE",
        tier=3, severity=sev,
        title=f"{len(flagged)} classifier purpose(s) with error rate >= {int(CLASSIFIER_WARN_PCT*100)}% in last 7d",
        body="Classifier calls failing in the pipeline stage. Investigate handler exceptions or API errors.",
        affected_count=len(flagged),
        evidence={"classifiers": flagged},
        auto_fix=None,
    )]


GENERAL_CALL_ERROR_WARN_PCT = 0.10
GENERAL_CALL_ERROR_CRITICAL_PCT = 0.25
GENERAL_CALL_ERROR_MIN_N = 10


def check_agent_call_error_rate_by_purpose(cur) -> list[Finding]:
    """Catch-all error rate by purpose. Excludes `classifier_*` (covered by
    its own check with tighter thresholds and distinct remediation)."""
    cur.execute(
        """
        SELECT payload->>'purpose' AS purpose,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE COALESCE((payload->>'success')::boolean, true) = false) AS errors
        FROM agent_events
        WHERE event_type = 'agent_call'
          AND occurred_at > now() - interval '7 days'
          AND COALESCE(payload->>'purpose', '') NOT LIKE 'classifier_%%'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        """,
        (GENERAL_CALL_ERROR_MIN_N,),
    )
    flagged = []
    for r in cur.fetchall():
        rate = r["errors"] / r["total"] if r["total"] else 0
        if rate >= GENERAL_CALL_ERROR_WARN_PCT:
            flagged.append({"purpose": r["purpose"], "total": r["total"],
                            "errors": r["errors"], "rate": round(rate, 3)})
    if not flagged:
        return []
    worst = max(f["rate"] for f in flagged)
    sev = "critical" if worst >= GENERAL_CALL_ERROR_CRITICAL_PCT else "warn"
    return [Finding(
        check_code="AGENT_CALL_ERROR_RATE_BY_PURPOSE",
        tier=3, severity=sev,
        title=f"{len(flagged)} agent purpose(s) with error rate >= {int(GENERAL_CALL_ERROR_WARN_PCT*100)}% in last 7d",
        body=("Generic per-purpose error tracking. Investigate the specific "
              "purpose's call site for handler exceptions or upstream API errors."),
        affected_count=len(flagged),
        evidence={"purposes": flagged},
        auto_fix=None,
    )]


RECOVERY_BURST_MIN = 3


def check_loop_recovery_burst(cur) -> list[Finding]:
    """Agentic-loop recovery branches firing often. Either prompts are
    generating responses that exceed max_tokens or message history is
    bloating to context-length limits."""
    cur.execute(
        """
        SELECT payload->>'reason' AS reason, COUNT(*) AS n
        FROM agent_events
        WHERE event_type = 'loop_recovery'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        """,
        (RECOVERY_BURST_MIN,),
    )
    rows = cur.fetchall()
    if not rows:
        return []
    return [Finding(
        check_code="LOOP_RECOVERY_BURST",
        tier=3, severity="warn",
        title=f"{len(rows)} loop-recovery reason(s) firing >= {RECOVERY_BURST_MIN} times in 7d",
        body=("`run_agentic_loop` recovery branches (max_tokens retry / "
              "context-length aggressive prune) firing often. Investigate "
              "prompt size, message history pruning, or model max_tokens."),
        affected_count=sum(r["n"] for r in rows),
        evidence={"recoveries": [{"reason": r["reason"], "n": r["n"]} for r in rows]},
        auto_fix=None,
    )]


MAX_TURNS_WARN_N = 1
MAX_TURNS_CRITICAL_N = 3


def check_loop_max_turns_hit(cur) -> list[Finding]:
    """Agentic loops terminating because they hit max_turns rather than
    end_turn. Means strategist or reflection ran out of turn budget mid-task,
    leaving incomplete work."""
    cur.execute(
        """
        SELECT stage_name, COUNT(*) AS n,
               array_agg(DISTINCT session_id ORDER BY session_id DESC) AS session_ids
        FROM agent_events
        WHERE event_type = 'loop_completion'
          AND payload->>'stop_reason' = 'max_turns'
          AND occurred_at > now() - interval '7 days'
        GROUP BY 1
        """
    )
    rows = cur.fetchall()
    if not rows:
        return []
    total = sum(r["n"] for r in rows)
    if total < MAX_TURNS_WARN_N:
        return []
    sev = "critical" if total >= MAX_TURNS_CRITICAL_N else "warn"
    return [Finding(
        check_code="LOOP_MAX_TURNS_HIT",
        tier=3, severity=sev,
        title=f"{total} agentic-loop run(s) terminated by max_turns in last 7d",
        body=("`run_agentic_loop` exited because it hit max_turns, not "
              "because Claude returned end_turn. Strategist or reflection "
              "didn't finish its task — playbook may be partial, rules may "
              "not have been proposed/retired. Either the prompt is asking "
              "for too much, the tool surface is too noisy, or max_turns "
              "needs raising."),
        affected_count=total,
        evidence={"by_stage": [
            {"stage_name": r["stage_name"], "n": r["n"],
             "recent_session_ids": list(r["session_ids"] or [])[:5]}
            for r in rows
        ]},
        auto_fix=None,
    )]


CACHE_RATIO_DROP_PCT_POINTS = 0.30
CACHE_RATIO_MIN_N = 10


def check_cache_hit_ratio_degradation(cur) -> list[Finding]:
    """Cache breakpoint placement broken — silent cost regression. The
    Anthropic prompt cache halves API costs; if a refactor moves cache
    breakpoints or breaks ephemeral cache markers, costs silently double."""
    cur.execute(
        """
        WITH recent AS (
            SELECT payload->>'purpose' AS purpose,
                   SUM((payload->>'cache_read_tokens')::int) AS cache_read,
                   SUM((payload->>'cache_creation_tokens')::int) AS cache_creation,
                   SUM((payload->>'input_tokens')::int) AS input_tok,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '7 days'
              AND COALESCE((payload->>'success')::boolean, true) = true
            GROUP BY 1
        ),
        prior AS (
            SELECT payload->>'purpose' AS purpose,
                   SUM((payload->>'cache_read_tokens')::int) AS cache_read,
                   SUM((payload->>'cache_creation_tokens')::int) AS cache_creation,
                   SUM((payload->>'input_tokens')::int) AS input_tok,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '14 days'
              AND occurred_at <= now() - interval '7 days'
              AND COALESCE((payload->>'success')::boolean, true) = true
            GROUP BY 1
        )
        SELECT r.purpose,
               r.cache_read::float / NULLIF(r.cache_read + r.cache_creation + r.input_tok, 0) AS recent_ratio,
               p.cache_read::float / NULLIF(p.cache_read + p.cache_creation + p.input_tok, 0) AS prior_ratio,
               r.n AS recent_n,
               p.n AS prior_n
        FROM recent r JOIN prior p ON p.purpose = r.purpose
        WHERE r.n >= %s AND p.n >= %s
        """,
        (CACHE_RATIO_MIN_N, CACHE_RATIO_MIN_N),
    )
    flagged = []
    for r in cur.fetchall():
        recent = r["recent_ratio"] or 0
        prior = r["prior_ratio"] or 0
        if prior - recent >= CACHE_RATIO_DROP_PCT_POINTS:
            flagged.append({
                "purpose": r["purpose"],
                "recent_ratio": round(recent, 3),
                "prior_ratio": round(prior, 3),
                "drop_pct_points": round(prior - recent, 3),
            })
    if not flagged:
        return []
    return [Finding(
        check_code="CACHE_HIT_RATIO_DEGRADATION",
        tier=3, severity="info",
        title=f"{len(flagged)} agent purpose(s) with cache_read share dropped >= {int(CACHE_RATIO_DROP_PCT_POINTS*100)}pp vs prior 7d",
        body=("Cache-read token share dropped sharply for one or more "
              "purposes. Likely cause: a refactor moved/removed an "
              "`ephemeral` cache breakpoint in `cached_system` or tool "
              "definitions. This silently doubles API cost. Inspect recent "
              "changes to system prompts and tool registration."),
        affected_count=len(flagged),
        evidence={"degradations": flagged},
        auto_fix=None,
    )]


LATENCY_DRIFT_RATIO = 2.0
LATENCY_MIN_N = 10


def check_agent_call_latency_drift(cur) -> list[Finding]:
    """p95 duration_ms by purpose — spike detector."""
    cur.execute(
        """
        WITH recent AS (
            SELECT payload->>'purpose' AS purpose,
                   percentile_cont(0.95) WITHIN GROUP
                       (ORDER BY (payload->>'duration_ms')::int) AS p95,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '7 days'
              AND payload ? 'duration_ms'
            GROUP BY 1
        ),
        prior AS (
            SELECT payload->>'purpose' AS purpose,
                   percentile_cont(0.95) WITHIN GROUP
                       (ORDER BY (payload->>'duration_ms')::int) AS p95,
                   COUNT(*) AS n
            FROM agent_events
            WHERE event_type = 'agent_call'
              AND occurred_at > now() - interval '14 days'
              AND occurred_at <= now() - interval '7 days'
              AND payload ? 'duration_ms'
            GROUP BY 1
        )
        SELECT r.purpose, r.p95 AS recent_p95, p.p95 AS prior_p95
        FROM recent r JOIN prior p ON p.purpose = r.purpose
        WHERE r.n >= %s AND p.n >= %s
          AND p.p95 > 0
          AND r.p95 >= %s * p.p95
        """,
        (LATENCY_MIN_N, LATENCY_MIN_N, LATENCY_DRIFT_RATIO),
    )
    rows = cur.fetchall()
    if not rows:
        return []
    return [Finding(
        check_code="AGENT_CALL_LATENCY_DRIFT",
        tier=3, severity="info",
        title=f"{len(rows)} agent purpose(s) with p95 latency >= {LATENCY_DRIFT_RATIO}x prior 7-day window",
        body="Per-purpose p95 latency in last 7d is significantly elevated vs prior 7d.",
        affected_count=len(rows),
        evidence={"drifts": [
            {"purpose": r["purpose"],
             "recent_p95_ms": int(r["recent_p95"]),
             "prior_p95_ms": int(r["prior_p95"]),
             "ratio": round(r["recent_p95"] / r["prior_p95"], 2)}
            for r in rows
        ]},
        auto_fix=None,
    )]


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

    cur.execute(r"""
        WITH r AS (SELECT id FROM strategy_rules WHERE status='active')
        SELECT r.id AS rule_id,
               COUNT(*) FILTER (
                 WHERE d.reasoning ~* ('\mrule\s*#?\s*' || r.id || '\M')
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


# --- Runner ---------------------------------------------------------------

from v2.database.connection import get_cursor
from v2.database.trading_db import (
    insert_audit_run,
    insert_audit_finding,
    insert_audit_llm_call,
    finalize_audit_run,
    supersede_stale_open_findings,
    try_advisory_audit_lock,
    release_advisory_audit_lock,
)


CHECKS: list = [
    "check_orphan_fks",
    "check_missing_backfill",
    "check_invalid_attribution_categories",
    "check_snapshot_gaps",
    "check_decision_equity_drift",
    "check_attribution_category_coverage",
    "check_stage_failure_rate",
    "check_cost_trend",
    "check_audit_llm_cost_trend",
    "check_decisions_missing_signal_refs",
    "check_theses_missing_signal_refs",
    "check_strategist_using_reversal_tool",
    "check_reflection_inert_on_round_trips",
    "check_tool_error_rate",
    "check_risk_block_hotspot",
    "check_risk_block_burst",
    "check_ideation_tool_drought",
    "check_executor_truncation_rate",
    "check_executor_schema_drift",
    "check_executor_parse_failure_rate",
    "check_classifier_error_rate",
    "check_agent_call_error_rate_by_purpose",
    "check_loop_recovery_burst",
    "check_loop_max_turns_hit",
    "check_cache_hit_ratio_degradation",
    "check_agent_call_latency_drift",
    "check_rule_judgment",
]


def _resolve_checks() -> list:
    import sys
    mod = sys.modules[__name__]
    if not CHECKS:
        return []
    return [getattr(mod, name) for name in CHECKS] if isinstance(CHECKS[0], str) else list(CHECKS)


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

                if check.__name__ == "check_rule_judgment":
                    rule_judgment_usage = get_last_rule_judgment_usage()
                    if rule_judgment_usage:
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
                        except Exception:
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
            # LLM accounting moved to audit_llm_calls table; legacy
            # token columns on audit_runs are no longer populated.
            # See spec 2026-05-12-opus-audit-ideation-design.md.
        )

        summary.findings_emitted = emitted
        summary.auto_fixed = auto_fixed
        summary.failed_checks = failed_checks
        log.info("Audit run #%d complete: emitted=%d auto_fixed=%d failed_checks=%d",
                 run_id, emitted, auto_fixed, failed_checks)
        return summary
    finally:
        release_advisory_audit_lock()


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
