"""Tests for v2.patterns — pattern analysis with decision_signals FK."""

import pytest
from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import MagicMock, patch

from v2.patterns import (
    SignalPerformance,
    SentimentPerformance,
    TickerPerformance,
    ConfidenceCorrelation,
    analyze_signal_categories,
    analyze_sentiment_performance,
    analyze_ticker_performance,
    analyze_confidence_correlation,
    get_best_performing_signals,
    get_worst_performing_signals,
    generate_pattern_report,
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
        """Returns TickerPerformance dataclasses."""
        mock_db.fetchall.return_value = [
            {
                "ticker": "AAPL",
                "total_decisions": 12,
                "buys": 8,
                "sells": 4,
                "avg_outcome_7d": Decimal("1.8"),
                "avg_outcome_30d": Decimal("4.5"),
                "total_pnl_7d": Decimal("21.6"),
            },
            {
                "ticker": "TSLA",
                "total_decisions": 6,
                "buys": 3,
                "sells": 3,
                "avg_outcome_7d": None,
                "avg_outcome_30d": None,
                "total_pnl_7d": None,
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
        assert results[0].total_pnl_7d == 21.6

        assert results[1].ticker == "TSLA"
        assert results[1].avg_outcome_7d is None
        assert results[1].total_pnl_7d is None


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
        assert "Signal Category Performance:" in report
        assert "news_signal:earnings" in report

    def test_empty_data(self, mock_db):
        """Report still contains header when no data."""
        mock_db.fetchall.return_value = []
        report = generate_pattern_report(days=30)

        assert report.startswith("Pattern Analysis Report (30 days)")
        assert "=" * 50 in report
