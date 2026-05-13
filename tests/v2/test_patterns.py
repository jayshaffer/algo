"""Tests for v2.patterns — pattern analysis with decision_signals FK."""

from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.patterns import (
    ConfidenceCorrelation,
    SentimentPerformance,
    SignalPerformance,
    TickerPerformance,
    analyze_confidence_correlation,
    analyze_sentiment_performance,
    analyze_signal_categories,
    analyze_ticker_performance,
    generate_pattern_report,
    get_best_performing_signals,
    get_worst_performing_signals,
)


@pytest.fixture
def mock_db():
    """Patch get_cursor as imported into v2.patterns."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0

    @contextmanager
    def _get_cursor():
        yield cursor

    with patch("v2.patterns.get_cursor", _get_cursor):
        yield cursor


class TestAnalyzeSignalCategories:
    def test_uses_decision_signals_fk(self, mock_db):
        """SQL must JOIN through decision_signals, not time-window."""
        mock_db.fetchall.return_value = []
        analyze_signal_categories(days=90)

        sql = mock_db.execute.call_args[0][0]
        # Must use the decision_signals FK table
        assert "decision_signals" in sql.lower()
        assert "ds.decision_id" in sql or "ds.signal_type" in sql
        # Must JOIN decisions through decision_signals, not time-window
        assert "JOIN decisions d ON d.id = ds.decision_id" in sql
        # Must NOT use time-window JOIN pattern from V1
        assert "published_at::date" not in sql
        assert "INTERVAL '7 days'" not in sql

    def test_excludes_orphan_signal_references(self, mock_db):
        """Orphan signal references (signal_type=news_signal but no matching row in news_signals,
        or signal_type=macro_signal with no matching row in macro_signals) must be filtered out.
        These are broken FK references and bucketing them as 'unknown' produces noise."""
        mock_db.fetchall.return_value = []
        analyze_signal_categories(days=90)
        sql = mock_db.execute.call_args[0][0]
        assert "ns.id IS NOT NULL" in sql or "ns.category IS NOT NULL" in sql
        assert "ms.id IS NOT NULL" in sql or "ms.category IS NOT NULL" in sql
        assert "'unknown'" not in sql

    def test_excludes_thesis_orphans(self, mock_db):
        """Same orphan-FK guard for signal_type='thesis'. Without it, pre-validator
        residue (signal_id=0) inflates the thesis bucket — the same artifact
        shape as news_signal:unknown but for theses.
        """
        mock_db.fetchall.return_value = []
        analyze_signal_categories(days=90)
        sql = mock_db.execute.call_args[0][0]
        assert "JOIN theses" in sql
        assert "t.id IS NOT NULL" in sql

    def test_win_rate_30d_sql_guards_null(self, mock_db):
        """Regression: NULL outcome_30d/benchmark_30d (decision too young)
        must propagate through AVG, not collapse to a loss via NULL > 0 → ELSE.
        """
        mock_db.fetchall.return_value = []
        analyze_signal_categories(days=90)
        sql = mock_db.execute.call_args[0][0]
        # The fix: explicit IS NULL guard on 30d operands.
        assert "d.outcome_30d IS NULL OR d.benchmark_30d IS NULL THEN NULL" in sql

    def test_returns_signal_performance_objects(self, mock_db):
        """Returns list of SignalPerformance dataclasses."""
        mock_db.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "total_signals": 10,
                "avg_outcome_7d": Decimal("2.5"),
                "avg_outcome_30d": Decimal("5.0"),
                "win_rate_7d": Decimal("60.0"),
                "win_rate_30d": Decimal("70.0"),
            },
            {
                "category": "macro_signal:fed",
                "total_signals": 5,
                "avg_outcome_7d": Decimal("-1.0"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("40.0"),
                "win_rate_30d": None,
            },
        ]
        results = analyze_signal_categories(days=30)

        assert len(results) == 2
        assert all(isinstance(r, SignalPerformance) for r in results)

        assert results[0].category == "news_signal:earnings"
        assert results[0].total_signals == 10
        assert results[0].avg_outcome_7d == 2.5
        assert results[0].avg_outcome_30d == 5.0
        assert results[0].win_rate_7d == 60.0
        assert results[0].win_rate_30d == 70.0

        assert results[1].category == "macro_signal:fed"
        assert results[1].avg_outcome_30d is None
        assert results[1].win_rate_30d is None

    def test_empty_results(self, mock_db):
        """Returns empty list when no data."""
        mock_db.fetchall.return_value = []
        results = analyze_signal_categories(days=90)
        assert results == []


class TestAnalyzeSentimentPerformance:
    def test_uses_decision_signals_fk(self, mock_db):
        """SQL uses decision_signals table, not time-window JOINs."""
        mock_db.fetchall.return_value = []
        analyze_sentiment_performance(days=90)

        sql = mock_db.execute.call_args[0][0]
        assert "decision_signals" in sql.lower()
        assert "JOIN decisions d ON d.id = ds.decision_id" in sql
        # Must NOT use time-window JOIN pattern from V1
        assert "published_at::date" not in sql

    def test_returns_sentiment_objects(self, mock_db):
        """Returns SentimentPerformance dataclasses."""
        mock_db.fetchall.return_value = [
            {
                "sentiment": "bullish",
                "total_decisions": 15,
                "avg_outcome_7d": Decimal("3.2"),
                "avg_outcome_30d": Decimal("6.1"),
                "win_rate_7d": Decimal("73.0"),
            },
            {
                "sentiment": "bearish",
                "total_decisions": 8,
                "avg_outcome_7d": Decimal("-0.5"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("25.0"),
            },
        ]
        results = analyze_sentiment_performance(days=60)

        assert len(results) == 2
        assert all(isinstance(r, SentimentPerformance) for r in results)

        assert results[0].sentiment == "bullish"
        assert results[0].total_decisions == 15
        assert results[0].avg_outcome_7d == 3.2
        assert results[0].avg_outcome_30d == 6.1
        assert results[0].win_rate_7d == 73.0

        assert results[1].sentiment == "bearish"
        assert results[1].avg_outcome_30d is None


class TestAnalyzeTickerPerformance:
    def test_no_signal_join(self, mock_db):
        """SQL queries decisions directly without decision_signals."""
        mock_db.fetchall.return_value = []
        analyze_ticker_performance(days=90)

        sql = mock_db.execute.call_args[0][0]
        # Should query decisions table directly
        assert "FROM decisions" in sql
        # Should NOT join through decision_signals
        assert "decision_signals" not in sql.lower()
        assert "news_signals" not in sql.lower()
        assert "macro_signals" not in sql.lower()

    def test_returns_ticker_objects(self, mock_db):
        """Returns TickerPerformance dataclasses. T2.5: SUM column renamed
        to `sum_pct_returns_7d` so the misleading `total_pnl_7d` label is
        retired."""
        mock_db.fetchall.return_value = [
            {
                "ticker": "AAPL",
                "total_decisions": 12,
                "buys": 8,
                "sells": 4,
                "avg_outcome_7d": Decimal("1.8"),
                "avg_outcome_30d": Decimal("4.5"),
                "sum_pct_returns_7d": Decimal("21.6"),
            },
            {
                "ticker": "TSLA",
                "total_decisions": 6,
                "buys": 3,
                "sells": 3,
                "avg_outcome_7d": None,
                "avg_outcome_30d": None,
                "sum_pct_returns_7d": None,
            },
        ]
        results = analyze_ticker_performance(days=90)

        assert len(results) == 2
        assert all(isinstance(r, TickerPerformance) for r in results)

        assert results[0].ticker == "AAPL"
        assert results[0].total_decisions == 12
        assert results[0].buys == 8
        assert results[0].sells == 4
        assert results[0].avg_outcome_7d == 1.8
        assert results[0].sum_pct_returns_7d == 21.6

        assert results[1].ticker == "TSLA"
        assert results[1].avg_outcome_7d is None
        assert results[1].sum_pct_returns_7d is None

    def test_orders_by_avg_outcome_not_sum(self, mock_db):
        """T2.5: ORDER BY must be `avg_outcome_7d`, not `total_pnl_7d`/SUM —
        averaging percentage returns is a defensible per-ticker metric;
        summing percentages across heterogeneous notional sizes is not.
        """
        mock_db.fetchall.return_value = []
        analyze_ticker_performance(days=90)
        sql = mock_db.execute.call_args[0][0]
        assert "ORDER BY avg_outcome_7d" in sql
        assert "ORDER BY total_pnl_7d" not in sql
        # The SUM column must be aliased to the new name.
        assert "as sum_pct_returns_7d" in sql


class TestAnalyzeConfidenceCorrelation:
    def test_uses_decision_signals_fk(self, mock_db):
        """SQL uses decision_signals table, not time-window JOINs."""
        mock_db.fetchall.return_value = []
        analyze_confidence_correlation(days=90)

        sql = mock_db.execute.call_args[0][0]
        assert "decision_signals" in sql.lower()
        assert "JOIN decisions d ON d.id = ds.decision_id" in sql
        # Must NOT use time-window JOIN pattern from V1
        assert "published_at::date" not in sql

    def test_returns_correlation_objects(self, mock_db):
        """Returns ConfidenceCorrelation dataclasses."""
        mock_db.fetchall.return_value = [
            {
                "confidence": "high",
                "total_decisions": 20,
                "avg_outcome_7d": Decimal("4.1"),
                "win_rate_7d": Decimal("80.0"),
            },
            {
                "confidence": "medium",
                "total_decisions": 30,
                "avg_outcome_7d": Decimal("1.5"),
                "win_rate_7d": Decimal("55.0"),
            },
            {
                "confidence": "low",
                "total_decisions": 10,
                "avg_outcome_7d": Decimal("-0.8"),
                "win_rate_7d": Decimal("30.0"),
            },
        ]
        results = analyze_confidence_correlation(days=90)

        assert len(results) == 3
        assert all(isinstance(r, ConfidenceCorrelation) for r in results)

        assert results[0].confidence == "high"
        assert results[0].total_decisions == 20
        assert results[0].avg_outcome_7d == 4.1
        assert results[0].win_rate_7d == 80.0

        assert results[2].confidence == "low"
        assert results[2].avg_outcome_7d == -0.8


class TestBestWorstPerforming:
    def test_best_reads_from_signal_attribution(self, mock_db):
        """SQL uses signal_attribution table, not time-window JOINs."""
        mock_db.fetchall.return_value = [
            {"category": "earnings", "avg_outcome": Decimal("5.2"),
             "occurrences": 8, "win_rate_7d": Decimal("75.0")},
        ]
        results = get_best_performing_signals(days=90, min_occurrences=3)

        sql = mock_db.execute.call_args[0][0]
        assert "signal_attribution" in sql.lower()
        # Must NOT use old time-window pattern
        assert "news_signals" not in sql.lower()
        assert "published_at" not in sql.lower()
        # Should order DESC for best
        assert "DESC" in sql

        assert len(results) == 1
        assert results[0]["category"] == "earnings"
        assert results[0]["avg_outcome"] == Decimal("5.2")
        assert results[0]["occurrences"] == 8

    def test_worst_reads_from_signal_attribution(self, mock_db):
        """SQL uses signal_attribution table, not time-window JOINs."""
        mock_db.fetchall.return_value = [
            {"category": "speculation", "avg_outcome": Decimal("-3.1"),
             "occurrences": 5, "win_rate_7d": Decimal("20.0")},
        ]
        results = get_worst_performing_signals(days=90, min_occurrences=3)

        sql = mock_db.execute.call_args[0][0]
        assert "signal_attribution" in sql.lower()
        assert "news_signals" not in sql.lower()
        assert "published_at" not in sql.lower()
        # Should order ASC for worst
        assert "ASC" in sql

        assert len(results) == 1
        assert results[0]["category"] == "speculation"
        assert results[0]["avg_outcome"] == Decimal("-3.1")


class TestGeneratePatternReport:
    def test_generates_report_header(self, mock_db):
        """Report starts with 'Pattern Analysis Report'."""
        # generate_pattern_report calls 6 functions, each calling fetchall once.
        # Provide appropriate data for signal_categories (1st call), empty for rest.
        signal_cat_data = [
            {
                "category": "news_signal:earnings",
                "total_signals": 10,
                "avg_outcome_7d": Decimal("2.5"),
                "avg_outcome_30d": Decimal("5.0"),
                "win_rate_7d": Decimal("60.0"),
                "win_rate_30d": Decimal("70.0"),
            },
        ]
        mock_db.fetchall.side_effect = [
            signal_cat_data,  # analyze_signal_categories
            [],               # analyze_sentiment_performance
            [],               # analyze_ticker_performance
            [],               # analyze_confidence_correlation
            [],               # get_best_performing_signals
            [],               # get_worst_performing_signals
        ]
        report = generate_pattern_report(days=90)

        assert report.startswith("Pattern Analysis Report (90 days)")
        assert "=" * 50 in report
        assert "Signal Category Performance" in report  # P2.19: header now reads "(alpha vs SPY)"
        assert "news_signal:earnings" in report

    def test_empty_data(self, mock_db):
        """Report still contains header when no data."""
        mock_db.fetchall.return_value = []
        report = generate_pattern_report(days=30)

        assert report.startswith("Pattern Analysis Report (30 days)")
        assert "=" * 50 in report

    def test_zero_alpha_renders_as_percent_not_na(self, mock_db):
        """T1.8: a category with avg_outcome_7d=Decimal(0) used to render as
        'N/A' because the truthy check `if x else None` collapsed Decimal(0)
        to None at the dataclass boundary, then the report's truthy check
        rendered it as N/A. Both checks must use `is not None` so a
        genuine 0% measurement stays distinguishable from missing data.
        """
        zero_signal_cat = [
            {
                "category": "news_signal:flat",
                "total_signals": 10,
                "avg_outcome_7d": Decimal("0"),
                "avg_outcome_30d": Decimal("0"),
                "win_rate_7d": Decimal("50.0"),
                "win_rate_30d": Decimal("50.0"),
            },
        ]
        zero_sentiment = [
            {
                "sentiment": "neutral",
                "total_decisions": 10,
                "avg_outcome_7d": Decimal("0"),
                "avg_outcome_30d": Decimal("0"),
                "win_rate_7d": Decimal("50.0"),
            },
        ]
        zero_ticker = [
            {
                "ticker": "FLAT",
                "total_decisions": 5,
                "buys": 3,
                "sells": 2,
                "avg_outcome_7d": Decimal("0"),
                "avg_outcome_30d": Decimal("0"),
                "sum_pct_returns_7d": Decimal("0"),
            },
        ]
        zero_conf = [
            {
                "confidence": "medium",
                "total_decisions": 8,
                "avg_outcome_7d": Decimal("0"),
                "win_rate_7d": Decimal("50.0"),
            },
        ]
        mock_db.fetchall.side_effect = [
            zero_signal_cat,  # analyze_signal_categories
            zero_sentiment,   # analyze_sentiment_performance
            zero_ticker,      # analyze_ticker_performance
            zero_conf,        # analyze_confidence_correlation
            [],               # get_best_performing_signals
            [],               # get_worst_performing_signals
        ]
        report = generate_pattern_report(days=90)

        # Each section must show "+0.00%", not "N/A".
        assert "news_signal:flat: +0.00%" in report, (
            f"Zero-alpha signal category rendered as N/A.\n{report}"
        )
        assert "neutral: +0.00%" in report, (
            f"Zero-alpha sentiment rendered as N/A.\n{report}"
        )
        assert "FLAT: +0.00%" in report, (
            f"Zero-pnl ticker rendered as N/A.\n{report}"
        )
        assert "medium: +0.00%" in report, (
            f"Zero-alpha confidence bucket rendered as N/A.\n{report}"
        )

    def test_none_alpha_still_renders_as_na(self, mock_db):
        """Defense check: rows with None (no data yet) must still render as
        'N/A' — the fix preserves the None vs 0.0 distinction.
        """
        none_signal_cat = [
            {
                "category": "news_signal:nodata",
                "total_signals": 0,
                "avg_outcome_7d": None,
                "avg_outcome_30d": None,
                "win_rate_7d": None,
                "win_rate_30d": None,
            },
        ]
        mock_db.fetchall.side_effect = [
            none_signal_cat,
            [], [], [], [], [],
        ]
        report = generate_pattern_report(days=90)
        assert "news_signal:nodata: N/A" in report


class TestAnalyzeRoundTrips:
    """Tests for analyze_round_trips() — surfaces flip-flop patterns."""

    def test_returns_empty_list_when_no_pairs(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        assert analyze_round_trips(days=30, gap_days=7, min_pairs=2) == []

    def test_returns_round_trip_objects(self, mock_db):
        from datetime import date

        from v2.patterns import RoundTrip, analyze_round_trips
        mock_db.fetchall.return_value = [
            {"ticker": "GOOGL", "pair_count": 11,
             "first_date": date(2026, 4, 15), "last_date": date(2026, 5, 6)},
            {"ticker": "CRM", "pair_count": 9,
             "first_date": date(2026, 3, 10), "last_date": date(2026, 5, 5)},
        ]

        result = analyze_round_trips(days=60, gap_days=14, min_pairs=2)

        assert len(result) == 2
        assert result[0] == RoundTrip(
            ticker="GOOGL", pair_count=11,
            first_date=date(2026, 4, 15), last_date=date(2026, 5, 6),
        )
        assert result[1].ticker == "CRM"

    def test_sql_self_joins_decisions_on_opposite_action(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        analyze_round_trips(days=30, gap_days=7, min_pairs=2)

        sql = mock_db.execute.call_args[0][0]
        assert "decisions" in sql.lower()
        assert "b.action <> a.action" in sql
        assert "action IN ('buy', 'sell')" in sql or "action in ('buy','sell')" in sql.lower()
        assert "GROUP BY" in sql.upper()
        assert "HAVING" in sql.upper()

    def test_passes_window_and_gap_parameters(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        analyze_round_trips(days=45, gap_days=10, min_pairs=3)

        params = mock_db.execute.call_args[0][1]
        assert 45 in params
        assert 10 in params
        assert 3 in params

    def test_default_parameters(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        analyze_round_trips()

        params = mock_db.execute.call_args[0][1]
        assert 30 in params
        assert 7 in params
        assert 2 in params
