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
