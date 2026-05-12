"""Tests for v2 self-healing audit module."""

from contextlib import ExitStack
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

    @patch("v2.database.trading_db.get_cursor")
    def test_insert_audit_llm_call_returns_id(self, mock_get_cursor):
        cur = MagicMock()
        cur.fetchone.return_value = {"id": 42}
        mock_get_cursor.return_value.__enter__.return_value = cur
        from v2.database.trading_db import insert_audit_llm_call
        row_id = insert_audit_llm_call(
            audit_run_id=1,
            purpose="rule_judgment",
            model="claude-haiku-4-5-20251001",
            input_tokens=1000,
            output_tokens=200,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            latency_ms=2345,
        )
        assert row_id == 42
        sql = cur.execute.call_args[0][0]
        assert "audit_llm_calls" in sql
        params = cur.execute.call_args[0][1]
        # Spot-check positional params: purpose, model, input_tokens, latency_ms
        assert "rule_judgment" in params
        assert "claude-haiku-4-5-20251001" in params
        assert 1000 in params
        assert 2345 in params


# --- Finding dataclass tests (Task 3) ---

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
        assert f.fingerprint == Finding(check_code="X", tier=1, severity="warn",
            title="t", body="b", affected_count=2,
            evidence={"ids": [3, 1, 2]}, auto_fix=None).fingerprint


# --- Orphan FK check tests (Task 4) ---

class TestCheckOrphanFks:
    def _cur_with_results(self, results: dict[tuple, list]):
        """Build a mock cursor that returns specific rows per SQL fragment."""
        cur = MagicMock()
        def execute(sql, params=None):
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


# --- Missing-backfill check tests (Task 5) ---

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


# --- Invalid attribution category tests (Task 6) ---

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


# --- Snapshot gap tests (Task 7) ---

class TestCheckSnapshotGaps:
    @patch("v2.market_calendar.is_trading_day")
    def test_no_gaps_returns_no_findings(self, mock_is_td):
        from v2.audit import check_snapshot_gaps
        mock_is_td.return_value = False
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_snapshot_gaps(cur) == []

    @patch("v2.market_calendar.is_trading_day")
    def test_gaps_on_trading_days_emit_finding(self, mock_is_td):
        from v2.audit import check_snapshot_gaps
        from datetime import date
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
        mock_is_td.return_value = False
        assert check_snapshot_gaps(cur) == []


# --- Decision equity drift tests (Task 8) ---

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


# --- Attribution category coverage tests (Task 9) ---

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
        cur.fetchall.return_value = [
            {"category": "thesis", "sample_size_30d": 30},
        ] + [
            {"category": f"x:{i}", "sample_size_30d": 1} for i in range(5)
        ]
        findings = check_attribution_category_coverage(cur)
        assert len(findings) == 1


# --- Stage failure rate tests (Task 10) ---

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


# --- Cost trend tests (Task 11) ---

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


class TestCheckAuditLlmCostTrend:
    def test_flags_doubled_purpose(self):
        from v2.audit import check_audit_llm_cost_trend
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "audit_gaps", "recent_tok": 500, "prior_tok": 100},
            {"purpose": "rule_judgment", "recent_tok": 110, "prior_tok": 100},
        ]
        findings = check_audit_llm_cost_trend(cur)
        assert len(findings) == 1
        f = findings[0]
        assert f.check_code == "AUDIT_LLM_COST_TREND_SPIKE"
        assert f.tier == 3
        assert f.severity == "info"
        purposes = [p["purpose"] for p in f.evidence["purposes"]]
        assert purposes == ["audit_gaps"]
        spike = f.evidence["purposes"][0]
        assert spike["ratio"] == 5.0
        assert spike["recent_tok"] == 500
        assert spike["prior_tok"] == 100

    def test_no_spike_returns_empty(self):
        from v2.audit import check_audit_llm_cost_trend
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "audit_gaps", "recent_tok": 110, "prior_tok": 100},
        ]
        assert check_audit_llm_cost_trend(cur) == []

    def test_zero_prior_skipped(self):
        """Don't divide by zero / don't flag new purposes with no history."""
        from v2.audit import check_audit_llm_cost_trend
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "audit_gaps", "recent_tok": 500, "prior_tok": 0},
        ]
        assert check_audit_llm_cost_trend(cur) == []


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


# --- Theses missing signal_refs tests (Task 13) ---

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


# --- Rule judgment LLM tests (Task 14) ---

class TestCheckRuleJudgment:
    def _stub_inputs(self, cur):
        rules = [
            {"id": 27, "rule_text": "During fragile macro windows cap deployment "
                                    "at $500/day. Lifts when binary event resolves "
                                    "and markets confirm direction.",
             "created_at": None},
            {"id": 39, "rule_text": "When attribution shows thesis beat-rate <55% "
                                    "over 20+ samples, require corroboration.",
             "created_at": None},
        ]
        attribution = [
            {"category": "thesis", "sample_size": 30, "sample_size_30d": 23,
             "avg_outcome_7d": -0.4, "win_rate_7d": 0.47,
             "avg_outcome_30d": -0.5, "win_rate_30d": 0.45},
        ]
        citations = [{"rule_id": 27, "n": 12}, {"rule_id": 39, "n": 0}]
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
                {"check_code": "RULE_DEAD", "rule_id": 27,
                 "title": "Rule 27 not cited", "explanation": "..."},
            ]
        }, {"input_tokens": 800, "output_tokens": 100,
             "cache_creation_tokens": 0, "cache_read_tokens": 0})
        findings = check_rule_judgment(cur)
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


# --- Phase 1 telemetry-driven checks ---


class TestCheckStrategistUsingReversalTool:
    """Warn when 3 consecutive sessions have round-trip evidence but the
    ideation stage never called `get_recent_playbooks` to consult prior
    plays before reversing direction."""

    def test_no_event_data_returns_nothing(self):
        from v2.audit import check_strategist_using_reversal_tool
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_strategist_using_reversal_tool(cur) == []

    def test_three_sessions_with_evidence_no_lookup_warns(self):
        from v2.audit import check_strategist_using_reversal_tool
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"session_id": 100, "reversal_evidence": 1, "lookup_calls": 0},
            {"session_id": 99, "reversal_evidence": 1, "lookup_calls": 0},
            {"session_id": 98, "reversal_evidence": 1, "lookup_calls": 0},
        ]
        findings = check_strategist_using_reversal_tool(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "STRATEGIST_NOT_USING_REVERSAL_TOOL"
        assert findings[0].severity == "warn"

    def test_below_threshold_returns_nothing(self):
        """Even if 2 sessions show the gap, stay quiet — pattern needs 3+."""
        from v2.audit import check_strategist_using_reversal_tool
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"session_id": 100, "reversal_evidence": 1, "lookup_calls": 0},
            {"session_id": 99, "reversal_evidence": 1, "lookup_calls": 1},
            {"session_id": 98, "reversal_evidence": 1, "lookup_calls": 0},
        ]
        assert check_strategist_using_reversal_tool(cur) == []


class TestCheckReflectionInertOnRoundTrips:
    """Warn when 5 consecutive sessions had round-trip evidence shown to
    reflection AND zero `propose_rule` / `retire_rule` invocations from the
    reflection stage in those same sessions."""

    def test_no_event_data_returns_nothing(self):
        from v2.audit import check_reflection_inert_on_round_trips
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_reflection_inert_on_round_trips(cur) == []

    def test_five_sessions_with_evidence_no_rule_actions_warns(self):
        from v2.audit import check_reflection_inert_on_round_trips
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"session_id": s, "reversal_evidence": 1, "rule_actions": 0}
            for s in (200, 199, 198, 197, 196)
        ]
        findings = check_reflection_inert_on_round_trips(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "REFLECTION_INERT_ON_ROUND_TRIPS"
        assert findings[0].severity == "warn"

    def test_any_rule_action_clears_warning(self):
        from v2.audit import check_reflection_inert_on_round_trips
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"session_id": 200, "reversal_evidence": 1, "rule_actions": 0},
            {"session_id": 199, "reversal_evidence": 1, "rule_actions": 1},
            {"session_id": 198, "reversal_evidence": 1, "rule_actions": 0},
            {"session_id": 197, "reversal_evidence": 1, "rule_actions": 0},
            {"session_id": 196, "reversal_evidence": 1, "rule_actions": 0},
        ]
        assert check_reflection_inert_on_round_trips(cur) == []


class TestCheckToolErrorRate:
    """Per-tool 7-day error rate. ≥20% is a warn, ≥50% is critical. Tools
    with fewer than 5 invocations are ignored to avoid noisy small-N alerts."""

    def test_no_tool_calls_returns_nothing(self):
        from v2.audit import check_tool_error_rate
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_tool_error_rate(cur) == []

    def test_high_error_rate_emits_critical(self):
        from v2.audit import check_tool_error_rate
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"tool_name": "write_playbook", "n": 10, "errors": 6},
        ]
        findings = check_tool_error_rate(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "TOOL_ERROR_RATE"
        assert findings[0].severity == "critical"

    def test_low_volume_skipped(self):
        """4 invocations, 3 errors → 75% rate but n<5, ignore."""
        from v2.audit import check_tool_error_rate
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"tool_name": "write_playbook", "n": 4, "errors": 3},
        ]
        assert check_tool_error_rate(cur) == []


class TestCheckRiskBlockHotspot:
    """Same-ticker risk_block ≥3 times in 7 days → warn. Surfaces a
    persistently breached sector-cap that the strategist isn't reacting to."""

    def test_no_blocks_returns_nothing(self):
        from v2.audit import check_risk_block_hotspot
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_risk_block_hotspot(cur) == []

    def test_three_or_more_same_ticker_warns(self):
        from v2.audit import check_risk_block_hotspot
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"ticker": "AAPL", "n": 4},
        ]
        findings = check_risk_block_hotspot(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "RISK_BLOCK_HOTSPOT"
        assert findings[0].severity == "warn"
        assert findings[0].evidence["tickers"][0]["ticker"] == "AAPL"

    def test_below_threshold_skipped(self):
        from v2.audit import check_risk_block_hotspot
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"ticker": "AAPL", "n": 2},
        ]
        assert check_risk_block_hotspot(cur) == []


class TestCheckRiskBlockBurst:
    """≥5 risk_blocks on a single date in 14 days → warn. Catches a single
    bad session that hammered the sector gate."""

    def test_no_blocks_returns_nothing(self):
        from v2.audit import check_risk_block_burst
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_risk_block_burst(cur) == []

    def test_five_or_more_on_one_date_warns(self):
        from v2.audit import check_risk_block_burst
        from datetime import date as _date
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"d": _date(2026, 5, 8), "n": 6},
        ]
        findings = check_risk_block_burst(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "RISK_BLOCK_BURST"
        assert findings[0].severity == "warn"

    def test_below_threshold_skipped(self):
        from v2.audit import check_risk_block_burst
        from datetime import date as _date
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"d": _date(2026, 5, 8), "n": 4},
        ]
        assert check_risk_block_burst(cur) == []


class TestCheckIdeationToolDrought:
    """Any tool in the expected ideation toolset with 0 invocations across
    the last 7 ideation sessions → warn. Flags a tool the strategist has
    silently stopped using."""

    def test_all_expected_tools_used_returns_nothing(self):
        from v2.audit import EXPECTED_IDEATION_TOOLS, check_ideation_tool_drought
        cur = MagicMock()
        cur.fetchone.return_value = {"n_sessions": 7}
        cur.fetchall.return_value = [
            {"tool_name": t, "n": 1} for t in EXPECTED_IDEATION_TOOLS
        ]
        assert check_ideation_tool_drought(cur) == []

    def test_missing_tool_emits_warning(self):
        from v2.audit import EXPECTED_IDEATION_TOOLS, check_ideation_tool_drought
        cur = MagicMock()
        cur.fetchone.return_value = {"n_sessions": 7}
        # Drop `get_recent_playbooks` from observed counts.
        cur.fetchall.return_value = [
            {"tool_name": t, "n": 3}
            for t in EXPECTED_IDEATION_TOOLS if t != "get_recent_playbooks"
        ]
        findings = check_ideation_tool_drought(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "IDEATION_TOOL_DROUGHT"
        assert findings[0].severity == "warn"
        assert "get_recent_playbooks" in findings[0].evidence["missing_tools"]

    def test_too_few_sessions_skipped(self):
        """Window not yet meaningful — fewer than 3 ideation sessions in 7d."""
        from v2.audit import check_ideation_tool_drought
        cur = MagicMock()
        cur.fetchone.return_value = {"n_sessions": 2}
        cur.fetchall.return_value = []
        assert check_ideation_tool_drought(cur) == []


# --- Phase 2 telemetry-driven checks ---


class TestCheckExecutorTruncationRate:
    """Executor calls hitting max_tokens. Warn ≥10%, critical ≥25% in 14d."""

    def test_no_calls_returns_nothing(self):
        from v2.audit import check_executor_truncation_rate
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 0, "truncated": 0}
        assert check_executor_truncation_rate(cur) == []

    def test_high_truncation_rate_critical(self):
        from v2.audit import check_executor_truncation_rate
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 10, "truncated": 4}  # 40%
        findings = check_executor_truncation_rate(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "EXECUTOR_TRUNCATION_RATE"
        assert findings[0].severity == "critical"

    def test_low_volume_skipped(self):
        from v2.audit import check_executor_truncation_rate
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 4, "truncated": 4}  # n<5
        assert check_executor_truncation_rate(cur) == []


class TestCheckExecutorSchemaDrift:
    def test_no_drift_returns_nothing(self):
        from v2.audit import check_executor_schema_drift
        cur = MagicMock()
        cur.fetchall.side_effect = [[], []]
        assert check_executor_schema_drift(cur) == []

    def test_top_level_drift_emits_warning(self):
        from v2.audit import check_executor_schema_drift
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [{"key": "experimental_score", "n": 5}],
            [],
        ]
        findings = check_executor_schema_drift(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "EXECUTOR_SCHEMA_DRIFT"
        assert findings[0].severity == "warn"
        assert findings[0].evidence["top_level_drift"][0]["key"] == "experimental_score"

    def test_decision_drift_emits_warning(self):
        from v2.audit import check_executor_schema_drift
        cur = MagicMock()
        cur.fetchall.side_effect = [
            [],
            [{"key": "confidence_calibration", "n": 4}],
        ]
        findings = check_executor_schema_drift(cur)
        assert len(findings) == 1
        assert "confidence_calibration" in [
            d["key"] for d in findings[0].evidence["decision_drift"]
        ]


class TestCheckExecutorParseFailureRate:
    def test_no_calls_returns_nothing(self):
        from v2.audit import check_executor_parse_failure_rate
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 0, "parse_failed": 0}
        assert check_executor_parse_failure_rate(cur) == []

    def test_high_parse_failure_critical(self):
        from v2.audit import check_executor_parse_failure_rate
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 10, "parse_failed": 2}  # 20%
        findings = check_executor_parse_failure_rate(cur)
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert findings[0].check_code == "EXECUTOR_PARSE_FAILURE_RATE"

    def test_below_warn_threshold_skipped(self):
        from v2.audit import check_executor_parse_failure_rate
        cur = MagicMock()
        cur.fetchone.return_value = {"total": 50, "parse_failed": 1}  # 2%
        assert check_executor_parse_failure_rate(cur) == []


class TestCheckClassifierErrorRate:
    def test_no_calls_returns_nothing(self):
        from v2.audit import check_classifier_error_rate
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_classifier_error_rate(cur) == []

    def test_high_error_rate_critical(self):
        from v2.audit import check_classifier_error_rate
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "classifier_news", "total": 20, "errors": 6},
        ]
        findings = check_classifier_error_rate(cur)
        assert len(findings) == 1
        assert findings[0].severity == "critical"

    def test_below_min_n_skipped(self):
        """SQL HAVING filters n < 10 — defensively also handled in Python."""
        from v2.audit import check_classifier_error_rate
        cur = MagicMock()
        cur.fetchall.return_value = []  # SQL HAVING filtered
        assert check_classifier_error_rate(cur) == []


class TestCheckAgentCallErrorRateByPurpose:
    def test_no_calls_returns_nothing(self):
        from v2.audit import check_agent_call_error_rate_by_purpose
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_agent_call_error_rate_by_purpose(cur) == []

    def test_high_error_rate_critical(self):
        from v2.audit import check_agent_call_error_rate_by_purpose
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "executor", "total": 20, "errors": 6},
        ]
        findings = check_agent_call_error_rate_by_purpose(cur)
        assert len(findings) == 1
        assert findings[0].severity == "critical"

    def test_below_threshold_skipped(self):
        from v2.audit import check_agent_call_error_rate_by_purpose
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "executor", "total": 20, "errors": 1},  # 5%
        ]
        assert check_agent_call_error_rate_by_purpose(cur) == []


class TestCheckLoopRecoveryBurst:
    def test_no_recoveries_returns_nothing(self):
        from v2.audit import check_loop_recovery_burst
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_loop_recovery_burst(cur) == []

    def test_burst_emits_warning(self):
        from v2.audit import check_loop_recovery_burst
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"reason": "max_tokens", "n": 5},
        ]
        findings = check_loop_recovery_burst(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "LOOP_RECOVERY_BURST"

    def test_below_threshold_skipped(self):
        from v2.audit import check_loop_recovery_burst
        cur = MagicMock()
        cur.fetchall.return_value = []  # SQL HAVING filtered n<3
        assert check_loop_recovery_burst(cur) == []


class TestCheckLoopMaxTurnsHit:
    def test_no_max_turns_returns_nothing(self):
        from v2.audit import check_loop_max_turns_hit
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_loop_max_turns_hit(cur) == []

    def test_three_or_more_critical(self):
        from v2.audit import check_loop_max_turns_hit
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"stage_name": "ideation", "n": 3, "session_ids": [101, 102, 103]},
        ]
        findings = check_loop_max_turns_hit(cur)
        assert len(findings) == 1
        assert findings[0].severity == "critical"

    def test_one_emits_warning(self):
        from v2.audit import check_loop_max_turns_hit
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"stage_name": "reflection", "n": 1, "session_ids": [101]},
        ]
        findings = check_loop_max_turns_hit(cur)
        assert len(findings) == 1
        assert findings[0].severity == "warn"


class TestCheckCacheHitRatioDegradation:
    def test_no_data_returns_nothing(self):
        from v2.audit import check_cache_hit_ratio_degradation
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_cache_hit_ratio_degradation(cur) == []

    def test_degradation_emits_info(self):
        from v2.audit import check_cache_hit_ratio_degradation
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "executor", "recent_ratio": 0.10, "prior_ratio": 0.60,
             "recent_n": 50, "prior_n": 50},
        ]
        findings = check_cache_hit_ratio_degradation(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "CACHE_HIT_RATIO_DEGRADATION"
        assert findings[0].severity == "info"

    def test_small_drop_skipped(self):
        from v2.audit import check_cache_hit_ratio_degradation
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "executor", "recent_ratio": 0.55, "prior_ratio": 0.60,
             "recent_n": 50, "prior_n": 50},
        ]
        assert check_cache_hit_ratio_degradation(cur) == []


class TestCheckAgentCallLatencyDrift:
    def test_no_drift_returns_nothing(self):
        from v2.audit import check_agent_call_latency_drift
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_agent_call_latency_drift(cur) == []

    def test_drift_emits_info(self):
        from v2.audit import check_agent_call_latency_drift
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"purpose": "executor", "recent_p95": 8000, "prior_p95": 3000},
        ]
        findings = check_agent_call_latency_drift(cur)
        assert len(findings) == 1
        assert findings[0].check_code == "AGENT_CALL_LATENCY_DRIFT"
        assert findings[0].severity == "info"

    def test_small_drift_skipped(self):
        """SQL filters out r.p95 < 2x p.p95 — defensively returns [] here."""
        from v2.audit import check_agent_call_latency_drift
        cur = MagicMock()
        cur.fetchall.return_value = []
        assert check_agent_call_latency_drift(cur) == []


# --- Runner tests (Task 15) ---

class TestRunner:
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

    @patch("v2.audit.insert_audit_llm_call")
    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_rule_judgment_records_llm_call_row(
        self, mock_cur, mock_run, mock_finding, mock_supersede,
        mock_finalize, mock_unlock, mock_lock, mock_insert_llm,
    ):
        from v2.audit import run_audit, Finding
        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur

        # Fake rule_judgment check that stashes usage like the real one
        from v2 import audit as audit_mod
        def fake_rule_judgment(c):
            audit_mod._LAST_RULE_JUDGMENT_USAGE.clear()
            audit_mod._LAST_RULE_JUDGMENT_USAGE.update({
                "input_tokens": 1234,
                "output_tokens": 56,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "latency_ms": 2000,
            })
            return []
        fake_rule_judgment.__name__ = "check_rule_judgment"

        with patch("v2.audit.CHECKS", [fake_rule_judgment]):
            run_audit(apply=False)

        assert mock_insert_llm.called
        kwargs = mock_insert_llm.call_args.kwargs
        assert kwargs["purpose"] == "rule_judgment"
        assert kwargs["model"] == audit_mod.RULE_JUDGMENT_MODEL
        assert kwargs["input_tokens"] == 1234
        assert kwargs["output_tokens"] == 56
        assert kwargs["latency_ms"] == 2000

        # finalize_audit_run should NOT receive deprecated token kwargs anymore
        finalize_kwargs = mock_finalize.call_args.kwargs
        for k in ("input_tokens", "output_tokens", "cache_creation_tokens",
                 "cache_read_tokens", "model"):
            assert k not in finalize_kwargs, f"{k} should no longer be passed"

    @patch("v2.audit.insert_audit_llm_call")
    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_opus_checks_record_llm_calls_and_use_topic_fingerprint(
        self, mock_cur, mock_run, mock_finding, mock_supersede,
        mock_finalize, mock_unlock, mock_lock, mock_insert_llm,
    ):
        from v2.audit import run_audit, Finding, _opus_topic_fingerprint
        from v2 import audit as audit_mod
        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur

        def fake_gap(c):
            audit_mod._LAST_OPUS_IDEATION_USAGE["audit_gaps"] = {
                "input_tokens": 1000, "output_tokens": 100,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "latency_ms": 2000,
            }
            f = Finding("AUDIT_GAP", 3, "info", "Missing foo check", "...",
                        1, {"topic_slug": "missing-foo", "category": "audit_gap",
                            "priority": "low",
                            "_fp_override": _opus_topic_fingerprint("AUDIT_GAP", "missing-foo")},
                        None)
            return [f]
        fake_gap.__name__ = "check_audit_gaps_opus"

        def fake_app(c):
            audit_mod._LAST_OPUS_IDEATION_USAGE["app_improvements"] = {
                "input_tokens": 2000, "output_tokens": 200,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "latency_ms": 2500,
            }
            f = Finding("APP_IMPROVEMENT", 3, "info", "Build the bar", "...",
                        1, {"topic_slug": "build-bar", "category": "app_improvement",
                            "priority": "medium",
                            "_fp_override": _opus_topic_fingerprint("APP_IMPROVEMENT", "build-bar")},
                        None)
            return [f]
        fake_app.__name__ = "check_app_improvements_opus"

        with patch("v2.audit.CHECKS", [fake_gap, fake_app]):
            run_audit(apply=False)

        # Both Opus purposes got LLM-call rows
        purposes = [c.kwargs["purpose"] for c in mock_insert_llm.call_args_list]
        assert "audit_gaps" in purposes
        assert "app_improvements" in purposes
        gap_call = next(c for c in mock_insert_llm.call_args_list
                        if c.kwargs["purpose"] == "audit_gaps")
        assert gap_call.kwargs["model"] == audit_mod.OPUS_IDEATION_MODEL
        assert gap_call.kwargs["input_tokens"] == 1000
        assert gap_call.kwargs["latency_ms"] == 2000

        # Findings inserted with topic-slug fingerprint and WITHOUT _fp_override
        gap_finding_call = next(
            c for c in mock_finding.call_args_list
            if c.kwargs["check_code"] == "AUDIT_GAP"
        )
        assert gap_finding_call.kwargs["fingerprint"] == _opus_topic_fingerprint(
            "AUDIT_GAP", "missing-foo"
        )
        assert "_fp_override" not in gap_finding_call.kwargs["evidence"]

    @patch("v2.audit.insert_audit_llm_call")
    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_run_audit_resets_opus_usage_at_start(
        self, mock_cur, mock_run, mock_finding, mock_supersede,
        mock_finalize, mock_unlock, mock_lock, mock_insert_llm,
    ):
        """A new run_audit invocation must not see stale usage from a prior call."""
        from v2.audit import run_audit
        from v2 import audit as audit_mod
        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur

        # Pre-populate stale usage from a "previous run"
        audit_mod._LAST_OPUS_IDEATION_USAGE["audit_gaps"] = {"input_tokens": 99999}

        with patch("v2.audit.CHECKS", []):
            run_audit(apply=False)

        # After the run, the stash should have been reset; the prior key gone
        assert "audit_gaps" not in audit_mod._LAST_OPUS_IDEATION_USAGE


# --- CLI tests (Task 16) ---

    @patch("v2.audit.insert_audit_llm_call")
    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_file_jira_off_by_default_records_disabled(
        self, mock_cur, mock_run, mock_finding, mock_supersede,
        mock_finalize, mock_unlock, mock_lock, mock_insert_llm,
    ):
        from v2.audit import run_audit, Finding, _opus_topic_fingerprint
        from v2 import audit as audit_mod
        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur

        def fake_gap(c):
            audit_mod._LAST_OPUS_IDEATION_USAGE["audit_gaps"] = {
                "input_tokens": 1, "output_tokens": 1,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "latency_ms": 1,
            }
            return [Finding(
                "AUDIT_GAP", 3, "info", "t", "b", 1,
                {"topic_slug": "slug-1", "category": "audit_gap", "priority": "low",
                 "_fp_override": _opus_topic_fingerprint("AUDIT_GAP", "slug-1")},
                None,
            )]
        fake_gap.__name__ = "check_audit_gaps_opus"

        with patch("v2.audit.CHECKS", [fake_gap]):
            run_audit(apply=False)  # file_jira defaults False

        gap_call = next(c for c in mock_finding.call_args_list
                        if c.kwargs["check_code"] == "AUDIT_GAP")
        assert gap_call.kwargs["evidence"]["jira"] == {"status": "disabled"}

    @patch("v2.audit_jira.file_jira_ticket")
    @patch("v2.audit.insert_audit_llm_call")
    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_file_jira_caps_creates_at_max(
        self, mock_cur, mock_run, mock_finding, mock_supersede,
        mock_finalize, mock_unlock, mock_lock, mock_insert_llm, mock_file,
        monkeypatch,
    ):
        """When --file-jira and 3 findings + cap=2, file 2, last gets 'capped'."""
        from v2.audit import run_audit, Finding, _opus_topic_fingerprint
        from v2 import audit as audit_mod

        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur

        monkeypatch.setenv("ALGO_AUDIT_JIRA_MAX_CREATES", "2")

        create_count = [0]
        def fake_file(finding, *, fingerprint, run_id):
            create_count[0] += 1
            return {"status": "created", "issue_key": f"ALGO-{create_count[0]}"}
        mock_file.side_effect = fake_file

        def fake_gap(c):
            audit_mod._LAST_OPUS_IDEATION_USAGE["audit_gaps"] = {
                "input_tokens": 1, "output_tokens": 1,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "latency_ms": 1,
            }
            return [
                Finding("AUDIT_GAP", 3, "info", f"T{i}", "b", 1,
                        {"topic_slug": f"slug-{i}", "category": "audit_gap",
                         "priority": "low",
                         "_fp_override": _opus_topic_fingerprint("AUDIT_GAP", f"slug-{i}")},
                        None)
                for i in range(3)
            ]
        fake_gap.__name__ = "check_audit_gaps_opus"

        with patch("v2.audit.CHECKS", [fake_gap]):
            run_audit(apply=False, file_jira=True)

        # Only 2 file_jira_ticket calls were issued
        assert mock_file.call_count == 2

        # 3 findings inserted; last has status=capped, first two have status=created
        gap_calls = [c for c in mock_finding.call_args_list
                     if c.kwargs["check_code"] == "AUDIT_GAP"]
        assert len(gap_calls) == 3
        statuses = [c.kwargs["evidence"]["jira"]["status"] for c in gap_calls]
        assert sorted(statuses) == ["capped", "created", "created"]

    @patch("v2.audit_jira.file_jira_ticket",
           return_value={"status": "existing", "issue_key": "ALGO-9"})
    @patch("v2.audit.insert_audit_llm_call")
    @patch("v2.audit.try_advisory_audit_lock", return_value=True)
    @patch("v2.audit.release_advisory_audit_lock")
    @patch("v2.audit.finalize_audit_run")
    @patch("v2.audit.supersede_stale_open_findings", return_value=0)
    @patch("v2.audit.insert_audit_finding", return_value=1)
    @patch("v2.audit.insert_audit_run", return_value=99)
    @patch("v2.audit.get_cursor")
    def test_file_jira_dedup_hits_do_not_count_against_cap(
        self, mock_cur, mock_run, mock_finding, mock_supersede,
        mock_finalize, mock_unlock, mock_lock, mock_insert_llm, mock_file,
        monkeypatch,
    ):
        """Existing-issue dedup hits don't consume the cap budget."""
        from v2.audit import run_audit, Finding, _opus_topic_fingerprint
        from v2 import audit as audit_mod

        cur = MagicMock()
        mock_cur.return_value.__enter__.return_value = cur
        monkeypatch.setenv("ALGO_AUDIT_JIRA_MAX_CREATES", "1")

        def fake_gap(c):
            audit_mod._LAST_OPUS_IDEATION_USAGE["audit_gaps"] = {
                "input_tokens": 1, "output_tokens": 1,
                "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "latency_ms": 1,
            }
            return [
                Finding("AUDIT_GAP", 3, "info", f"T{i}", "b", 1,
                        {"topic_slug": f"slug-{i}", "category": "audit_gap",
                         "priority": "low",
                         "_fp_override": _opus_topic_fingerprint("AUDIT_GAP", f"slug-{i}")},
                        None)
                for i in range(3)
            ]
        fake_gap.__name__ = "check_audit_gaps_opus"

        with patch("v2.audit.CHECKS", [fake_gap]):
            run_audit(apply=False, file_jira=True)

        # All 3 findings consulted file_jira_ticket (no cap hit)
        assert mock_file.call_count == 3
        gap_calls = [c for c in mock_finding.call_args_list
                     if c.kwargs["check_code"] == "AUDIT_GAP"]
        statuses = [c.kwargs["evidence"]["jira"]["status"] for c in gap_calls]
        assert statuses == ["existing"] * 3


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

    @patch("v2.audit.run_audit")
    def test_cli_file_jira_flag_passed_through(self, mock_run):
        from v2.audit import main
        mock_run.return_value = MagicMock(has_critical_open=False)
        main(argv=["--file-jira"])
        assert mock_run.call_args.kwargs["file_jira"] is True

    @patch("v2.audit.run_audit")
    def test_cli_file_jira_default_false(self, mock_run):
        from v2.audit import main
        mock_run.return_value = MagicMock(has_critical_open=False)
        main(argv=[])
        assert mock_run.call_args.kwargs.get("file_jira", False) is False


class TestOpusIdeationConstants:
    def test_model_constant_is_opus_4_7(self):
        from v2 import audit
        assert audit.OPUS_IDEATION_MODEL == "claude-opus-4-7"
        assert audit.OPUS_IDEATION_MAX_TOKENS == 4000
        assert audit.OPUS_INPUT_TOKEN_CAP_DEFAULT == 60_000

    def test_call_opus_ideation_is_callable(self):
        from v2 import audit
        assert callable(audit._call_opus_ideation)


class TestOpusFindingHelpers:
    def test_topic_slug_normalization_is_stable(self):
        from v2.audit import _opus_topic_fingerprint
        fp_a = _opus_topic_fingerprint("AUDIT_GAP", "Add Regime Detector")
        fp_b = _opus_topic_fingerprint("AUDIT_GAP", "add-regime-detector")
        fp_c = _opus_topic_fingerprint("AUDIT_GAP", "  Add Regime Detector!! ")
        assert fp_a == fp_b == fp_c

    def test_topic_slug_distinguishes_check_codes(self):
        from v2.audit import _opus_topic_fingerprint
        assert _opus_topic_fingerprint("AUDIT_GAP", "x") != _opus_topic_fingerprint("APP_IMPROVEMENT", "x")

    def test_finding_from_json_valid(self):
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
        # Fingerprint override is stashed for runner pickup
        assert "_fp_override" in f.evidence

    def test_finding_from_json_audit_gap_includes_proposed_code(self):
        from v2.audit import _opus_finding_from_json
        item = {
            "topic_slug": "missing-foo",
            "title": "Missing foo check",
            "category": "audit_gap",
            "priority": "medium",
            "body": "...",
            "proposed_check_code": "FOO_MISSING",
        }
        f = _opus_finding_from_json(item, default_category="audit_gap")
        assert f.check_code == "AUDIT_GAP"
        assert f.evidence["proposed_check_code"] == "FOO_MISSING"

    def test_finding_from_json_missing_required_returns_none(self):
        from v2.audit import _opus_finding_from_json
        assert _opus_finding_from_json({}, default_category="app_improvement") is None
        # missing title
        assert _opus_finding_from_json({"topic_slug": "x"}, default_category="app_improvement") is None
        # missing topic_slug
        assert _opus_finding_from_json({"title": "x"}, default_category="app_improvement") is None
        # slug normalizes to empty string
        assert _opus_finding_from_json({"topic_slug": "!!!", "title": "x"},
                                       default_category="app_improvement") is None

    def test_finding_from_json_invalid_category_falls_back(self):
        from v2.audit import _opus_finding_from_json
        item = {"topic_slug": "x", "title": "T", "category": "garbage",
                "body": "b", "priority": "medium"}
        f = _opus_finding_from_json(item, default_category="audit_gap")
        assert f.evidence["category"] == "audit_gap"

    def test_finding_from_json_invalid_priority_defaults_to_medium(self):
        from v2.audit import _opus_finding_from_json
        item = {"topic_slug": "x", "title": "T", "category": "app_improvement",
                "body": "b", "priority": "garbage"}
        f = _opus_finding_from_json(item, default_category="app_improvement")
        assert f.evidence["priority"] == "medium"

    def test_finding_from_json_caps_long_fields(self):
        from v2.audit import _opus_finding_from_json
        item = {"topic_slug": "x", "title": "T" * 500, "category": "app_improvement",
                "body": "B" * 5000, "priority": "high",
                "evidence_quote": "Q" * 5000}
        f = _opus_finding_from_json(item, default_category="app_improvement")
        # Caps from spec: title 200, body 2000, evidence_quote 600
        assert len(f.title) <= 200
        assert len(f.body) <= 2000
        assert len(f.evidence["evidence_quote"]) <= 600


class TestCheckAuditGapsOpus:
    def _stub_cursor(self):
        """A cursor whose fetchall returns the three queries _build_audit_gaps_prompt makes."""
        cur = MagicMock()
        # Query 1: findings summary (per check_code, severity)
        findings_rows = [{"check_code": "ORPHAN_FK_NEWS_SIGNAL", "n": 3, "severity": "critical"}]
        # Query 2: schema columns
        schema_rows = [{"table_name": "decisions", "column_name": "id"},
                       {"table_name": "decisions", "column_name": "ticker"}]
        # Query 3: cost trend
        cost_rows = [{"id": 1, "started_at": "2026-05-11", "tok": 1000, "total_findings": 5}]
        cur.fetchall.side_effect = [findings_rows, schema_rows, cost_rows]
        return cur

    @patch("v2.audit._call_opus_ideation")
    def test_emits_audit_gap_finding(self, mock_call):
        from v2 import audit
        canned = {
            "findings": [
                {
                    "topic_slug": "missing-thesis-rotation-check",
                    "title": "No check for stale active theses",
                    "category": "audit_gap",
                    "priority": "medium",
                    "body": "Active theses are not pruned by age.",
                    "evidence_quote": "Some theses are >60 days old.",
                    "proposed_check_code": "THESIS_STALE",
                }
            ]
        }
        usage = {"input_tokens": 100, "output_tokens": 50,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1000}
        mock_call.return_value = (canned, usage)
        cur = self._stub_cursor()

        findings = audit.check_audit_gaps_opus(cur)

        assert len(findings) == 1
        f = findings[0]
        assert f.check_code == "AUDIT_GAP"
        assert f.evidence["topic_slug"] == "missing-thesis-rotation-check"
        assert f.evidence["proposed_check_code"] == "THESIS_STALE"
        # Usage is stashed under the audit_gaps purpose
        assert audit.get_last_opus_ideation_usage("audit_gaps")["input_tokens"] == 100
        assert audit.get_last_opus_ideation_usage("audit_gaps")["latency_ms"] == 1000

    @patch("v2.audit._call_opus_ideation")
    def test_empty_findings_returns_empty_list(self, mock_call):
        from v2 import audit
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 100}
        mock_call.return_value = ({"findings": []}, usage)
        cur = self._stub_cursor()
        assert audit.check_audit_gaps_opus(cur) == []
        # Even on empty, usage is recorded
        assert audit.get_last_opus_ideation_usage("audit_gaps")["latency_ms"] == 100

    @patch("v2.audit._call_opus_ideation")
    def test_dedup_within_call(self, mock_call):
        """Two findings with same topic_slug → only one Finding returned."""
        from v2 import audit
        canned = {"findings": [
            {"topic_slug": "x", "title": "T1", "category": "audit_gap",
             "priority": "low", "body": "..."},
            {"topic_slug": "X", "title": "T2 dup", "category": "audit_gap",
             "priority": "low", "body": "..."},
        ]}
        usage = {"input_tokens": 1, "output_tokens": 1,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1}
        mock_call.return_value = (canned, usage)
        cur = self._stub_cursor()
        findings = audit.check_audit_gaps_opus(cur)
        assert len(findings) == 1

    @patch("v2.audit._call_opus_ideation")
    def test_caps_at_10_findings(self, mock_call):
        from v2 import audit
        many = [
            {"topic_slug": f"slug-{i}", "title": f"T{i}",
             "category": "audit_gap", "priority": "low", "body": "..."}
            for i in range(15)
        ]
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1}
        mock_call.return_value = ({"findings": many}, usage)
        cur = self._stub_cursor()
        findings = audit.check_audit_gaps_opus(cur)
        assert len(findings) == 10

    @patch("v2.audit._call_opus_ideation")
    def test_invalid_items_silently_dropped(self, mock_call):
        from v2 import audit
        canned = {"findings": [
            {},  # missing everything
            {"topic_slug": "ok", "title": "T", "category": "audit_gap",
             "priority": "low", "body": "..."},
        ]}
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1}
        mock_call.return_value = (canned, usage)
        cur = self._stub_cursor()
        findings = audit.check_audit_gaps_opus(cur)
        assert len(findings) == 1
        assert findings[0].evidence["topic_slug"] == "ok"


class TestCheckAppImprovementsOpus:
    def _stub_cursor(self):
        """Cursor whose fetchall returns the six queries _build_app_improvements_prompt makes."""
        cur = MagicMock()
        memos = [{"id": 1, "created_at": "2026-05-01", "body": "Memo body excerpt..."}]
        rules = [{"id": 27, "status": "active", "rule_text": "Cap deployment...",
                  "retired_at": None}]
        attribution = [{"category": "thesis", "sample_size": 30, "sample_size_30d": 23,
                        "avg_outcome_7d": -0.4, "win_rate_7d": 0.47,
                        "avg_outcome_30d": -0.5, "win_rate_30d": 0.45}]
        decisions = [{"id": 5, "date": "2026-05-10", "ticker": "AAPL", "action": "buy",
                      "notional": 1000, "outcome_7d_pct": 1.2, "outcome_30d_pct": 3.4}]
        theses = [{"id": 1, "ticker": "AAPL", "status": "active", "summary": "Bullish...",
                   "outcome_pct": None, "closed_at": None}]
        snapshots = [{"snapshot_date": "2026-05-10", "equity": 50000.0}]
        cur.fetchall.side_effect = [memos, rules, attribution, decisions, theses, snapshots]
        return cur

    @patch("v2.audit._call_opus_ideation")
    def test_emits_app_improvement_finding(self, mock_call):
        from v2 import audit
        canned = {
            "findings": [{
                "topic_slug": "add-regime-detector",
                "title": "Add a regime-detector signal",
                "category": "app_improvement",
                "priority": "high",
                "body": "Detect bull/bear regimes from SPY trend...",
                "evidence_quote": "Recent decisions ignore market regime.",
            }]
        }
        usage = {"input_tokens": 200, "output_tokens": 80,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1500}
        mock_call.return_value = (canned, usage)
        cur = self._stub_cursor()

        findings = audit.check_app_improvements_opus(cur)

        assert len(findings) == 1
        assert findings[0].check_code == "APP_IMPROVEMENT"
        assert findings[0].evidence["topic_slug"] == "add-regime-detector"
        assert audit.get_last_opus_ideation_usage("app_improvements")["input_tokens"] == 200

    @patch("v2.audit._call_opus_ideation")
    def test_truncates_when_over_cap(self, mock_call, monkeypatch):
        """If prompt is huge, the builder truncates and appends a TRUNCATED marker."""
        from v2 import audit
        # Set the cap to something very small so even the stub triggers truncation
        monkeypatch.setenv("ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS", "10")
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1}
        mock_call.return_value = ({"findings": []}, usage)
        cur = self._stub_cursor()
        audit.check_app_improvements_opus(cur)
        # Verify the call_opus_ideation was passed a truncated prompt
        sent_prompt = mock_call.call_args[0][1]  # second positional arg
        assert "[INPUT TRUNCATED]" in sent_prompt
        # Approx: cap=10 tokens → 40 chars limit + marker
        assert len(sent_prompt) < 200

    @patch("v2.audit._call_opus_ideation")
    def test_caps_at_10_findings(self, mock_call):
        from v2 import audit
        many = [
            {"topic_slug": f"slug-{i}", "title": f"T{i}",
             "category": "app_improvement", "priority": "low", "body": "..."}
            for i in range(15)
        ]
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1}
        mock_call.return_value = ({"findings": many}, usage)
        cur = self._stub_cursor()
        findings = audit.check_app_improvements_opus(cur)
        assert len(findings) == 10

    @patch("v2.audit._call_opus_ideation")
    def test_invalid_items_dropped(self, mock_call):
        from v2 import audit
        canned = {"findings": [
            {"title": "missing slug"},
            {"topic_slug": "ok", "title": "T", "category": "app_improvement",
             "priority": "low", "body": "..."},
        ]}
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_tokens": 0, "cache_read_tokens": 0,
                 "latency_ms": 1}
        mock_call.return_value = (canned, usage)
        cur = self._stub_cursor()
        findings = audit.check_app_improvements_opus(cur)
        assert len(findings) == 1
        assert findings[0].evidence["topic_slug"] == "ok"
