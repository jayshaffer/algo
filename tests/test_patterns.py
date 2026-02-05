"""Tests for trading/patterns.py - Pattern analysis for learning from past decisions."""

from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.patterns import (
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_get_cursor():
    """Patch trading.patterns.get_cursor to yield a mock cursor."""
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None

    @contextmanager
    def _get_cursor():
        yield cursor

    with patch("trading.patterns.get_cursor", _get_cursor):
        yield cursor


# ---------------------------------------------------------------------------
# analyze_signal_categories
# ---------------------------------------------------------------------------

class TestAnalyzeSignalCategories:
    """Tests for analyze_signal_categories()."""

    def test_returns_empty_list_when_no_rows(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        result = analyze_signal_categories(90)
        assert result == []

    def test_returns_signal_performance_objects(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {
                "category": "earnings",
                "total_signals": 10,
                "avg_outcome_7d": Decimal("3.5"),
                "avg_outcome_30d": Decimal("5.2"),
                "win_rate_7d": Decimal("60.0"),
                "win_rate_30d": Decimal("70.0"),
            },
        ]
        result = analyze_signal_categories(90)

        assert len(result) == 1
        assert isinstance(result[0], SignalPerformance)
        assert result[0].category == "earnings"
        assert result[0].total_signals == 10
        assert result[0].avg_outcome_7d == 3.5
        assert result[0].avg_outcome_30d == 5.2
        assert result[0].win_rate_7d == 60.0
        assert result[0].win_rate_30d == 70.0

    def test_handles_none_outcomes(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {
                "category": "merger",
                "total_signals": 2,
                "avg_outcome_7d": None,
                "avg_outcome_30d": None,
                "win_rate_7d": None,
                "win_rate_30d": None,
            },
        ]
        result = analyze_signal_categories(30)

        assert result[0].avg_outcome_7d is None
        assert result[0].avg_outcome_30d is None
        assert result[0].win_rate_7d is None
        assert result[0].win_rate_30d is None

    def test_multiple_categories(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"category": "earnings", "total_signals": 10,
             "avg_outcome_7d": Decimal("3.5"), "avg_outcome_30d": Decimal("5.0"),
             "win_rate_7d": Decimal("65.0"), "win_rate_30d": Decimal("70.0")},
            {"category": "fda", "total_signals": 5,
             "avg_outcome_7d": Decimal("-1.2"), "avg_outcome_30d": Decimal("2.0"),
             "win_rate_7d": Decimal("40.0"), "win_rate_30d": Decimal("55.0")},
        ]
        result = analyze_signal_categories(90)
        assert len(result) == 2
        assert result[0].category == "earnings"
        assert result[1].category == "fda"

    def test_passes_days_parameter_to_sql(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        analyze_signal_categories(45)
        params = mock_get_cursor.execute.call_args[0][1]
        assert params == (45,)


# ---------------------------------------------------------------------------
# analyze_sentiment_performance
# ---------------------------------------------------------------------------

class TestAnalyzeSentimentPerformance:
    """Tests for analyze_sentiment_performance()."""

    def test_returns_empty_list_when_no_rows(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        result = analyze_sentiment_performance(90)
        assert result == []

    def test_returns_sentiment_performance_objects(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {
                "sentiment": "bullish",
                "total_decisions": 15,
                "avg_outcome_7d": Decimal("2.1"),
                "avg_outcome_30d": Decimal("4.0"),
                "win_rate_7d": Decimal("55.0"),
            },
        ]
        result = analyze_sentiment_performance(90)

        assert len(result) == 1
        assert isinstance(result[0], SentimentPerformance)
        assert result[0].sentiment == "bullish"
        assert result[0].total_decisions == 15
        assert result[0].avg_outcome_7d == 2.1

    def test_handles_none_outcomes(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"sentiment": "neutral", "total_decisions": 3,
             "avg_outcome_7d": None, "avg_outcome_30d": None, "win_rate_7d": None},
        ]
        result = analyze_sentiment_performance(90)
        assert result[0].avg_outcome_7d is None
        assert result[0].win_rate_7d is None

    def test_multiple_sentiments(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"sentiment": "bullish", "total_decisions": 10,
             "avg_outcome_7d": Decimal("3.0"), "avg_outcome_30d": Decimal("5.0"),
             "win_rate_7d": Decimal("70.0")},
            {"sentiment": "bearish", "total_decisions": 8,
             "avg_outcome_7d": Decimal("-2.0"), "avg_outcome_30d": Decimal("-1.0"),
             "win_rate_7d": Decimal("30.0")},
            {"sentiment": "neutral", "total_decisions": 5,
             "avg_outcome_7d": Decimal("0.5"), "avg_outcome_30d": Decimal("1.0"),
             "win_rate_7d": Decimal("50.0")},
        ]
        result = analyze_sentiment_performance(90)
        assert len(result) == 3
        sentiments = [r.sentiment for r in result]
        assert "bullish" in sentiments
        assert "bearish" in sentiments
        assert "neutral" in sentiments


# ---------------------------------------------------------------------------
# analyze_ticker_performance
# ---------------------------------------------------------------------------

class TestAnalyzeTickerPerformance:
    """Tests for analyze_ticker_performance()."""

    def test_returns_empty_list_when_no_rows(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        result = analyze_ticker_performance(90)
        assert result == []

    def test_returns_ticker_performance_objects(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {
                "ticker": "AAPL",
                "total_decisions": 8,
                "buys": 6,
                "sells": 2,
                "avg_outcome_7d": Decimal("4.5"),
                "avg_outcome_30d": Decimal("8.0"),
                "total_pnl_7d": Decimal("36.0"),
            },
        ]
        result = analyze_ticker_performance(90)

        assert len(result) == 1
        assert isinstance(result[0], TickerPerformance)
        assert result[0].ticker == "AAPL"
        assert result[0].total_decisions == 8
        assert result[0].buys == 6
        assert result[0].sells == 2
        assert result[0].avg_outcome_7d == 4.5
        assert result[0].total_pnl_7d == 36.0

    def test_handles_none_outcomes(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"ticker": "TSLA", "total_decisions": 1, "buys": 1, "sells": 0,
             "avg_outcome_7d": None, "avg_outcome_30d": None, "total_pnl_7d": None},
        ]
        result = analyze_ticker_performance(90)
        assert result[0].avg_outcome_7d is None
        assert result[0].total_pnl_7d is None

    def test_multiple_tickers(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"ticker": "AAPL", "total_decisions": 5, "buys": 3, "sells": 2,
             "avg_outcome_7d": Decimal("3.0"), "avg_outcome_30d": Decimal("5.0"),
             "total_pnl_7d": Decimal("15.0")},
            {"ticker": "MSFT", "total_decisions": 3, "buys": 2, "sells": 1,
             "avg_outcome_7d": Decimal("1.0"), "avg_outcome_30d": Decimal("2.0"),
             "total_pnl_7d": Decimal("3.0")},
        ]
        result = analyze_ticker_performance(90)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# analyze_confidence_correlation
# ---------------------------------------------------------------------------

class TestAnalyzeConfidenceCorrelation:
    """Tests for analyze_confidence_correlation()."""

    def test_returns_empty_list_when_no_rows(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        result = analyze_confidence_correlation(90)
        assert result == []

    def test_returns_confidence_correlation_objects(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {
                "confidence": "high",
                "total_decisions": 20,
                "avg_outcome_7d": Decimal("5.5"),
                "win_rate_7d": Decimal("75.0"),
            },
        ]
        result = analyze_confidence_correlation(90)

        assert len(result) == 1
        assert isinstance(result[0], ConfidenceCorrelation)
        assert result[0].confidence == "high"
        assert result[0].total_decisions == 20
        assert result[0].avg_outcome_7d == 5.5
        assert result[0].win_rate_7d == 75.0

    def test_handles_none_values(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"confidence": "low", "total_decisions": 2,
             "avg_outcome_7d": None, "win_rate_7d": None},
        ]
        result = analyze_confidence_correlation(90)
        assert result[0].avg_outcome_7d is None
        assert result[0].win_rate_7d is None

    def test_multiple_confidence_levels(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"confidence": "high", "total_decisions": 10,
             "avg_outcome_7d": Decimal("5.0"), "win_rate_7d": Decimal("70.0")},
            {"confidence": "medium", "total_decisions": 15,
             "avg_outcome_7d": Decimal("2.0"), "win_rate_7d": Decimal("55.0")},
            {"confidence": "low", "total_decisions": 8,
             "avg_outcome_7d": Decimal("-1.0"), "win_rate_7d": Decimal("35.0")},
        ]
        result = analyze_confidence_correlation(90)
        assert len(result) == 3
        confidences = [r.confidence for r in result]
        assert confidences == ["high", "medium", "low"]


# ---------------------------------------------------------------------------
# get_best_performing_signals
# ---------------------------------------------------------------------------

class TestGetBestPerformingSignals:
    """Tests for get_best_performing_signals()."""

    def test_returns_empty_list_when_no_rows(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        result = get_best_performing_signals(90, min_occurrences=3)
        assert result == []

    def test_returns_list_of_dicts(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"category": "earnings", "sentiment": "bullish",
             "avg_outcome": Decimal("5.0"), "occurrences": 10},
        ]
        result = get_best_performing_signals(90)

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["category"] == "earnings"
        assert result[0]["sentiment"] == "bullish"

    def test_passes_min_occurrences_to_sql(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        get_best_performing_signals(90, min_occurrences=5)
        params = mock_get_cursor.execute.call_args[0][1]
        assert params == (90, 5)

    def test_default_min_occurrences_is_three(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        get_best_performing_signals(90)
        params = mock_get_cursor.execute.call_args[0][1]
        assert params == (90, 3)


# ---------------------------------------------------------------------------
# get_worst_performing_signals
# ---------------------------------------------------------------------------

class TestGetWorstPerformingSignals:
    """Tests for get_worst_performing_signals()."""

    def test_returns_empty_list_when_no_rows(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        result = get_worst_performing_signals(90, min_occurrences=3)
        assert result == []

    def test_returns_list_of_dicts(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"category": "lawsuit", "sentiment": "bearish",
             "avg_outcome": Decimal("-4.0"), "occurrences": 6},
        ]
        result = get_worst_performing_signals(90)

        assert len(result) == 1
        assert result[0]["category"] == "lawsuit"
        assert result[0]["avg_outcome"] == Decimal("-4.0")

    def test_passes_min_occurrences_to_sql(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        get_worst_performing_signals(60, min_occurrences=7)
        params = mock_get_cursor.execute.call_args[0][1]
        assert params == (60, 7)


# ---------------------------------------------------------------------------
# generate_pattern_report
# ---------------------------------------------------------------------------

class TestGeneratePatternReport:
    """Tests for generate_pattern_report()."""

    @patch("trading.patterns.get_worst_performing_signals")
    @patch("trading.patterns.get_best_performing_signals")
    @patch("trading.patterns.analyze_confidence_correlation")
    @patch("trading.patterns.analyze_ticker_performance")
    @patch("trading.patterns.analyze_sentiment_performance")
    @patch("trading.patterns.analyze_signal_categories")
    def test_report_with_all_data(self, mock_signals, mock_sentiment,
                                   mock_ticker, mock_confidence,
                                   mock_best, mock_worst):
        mock_signals.return_value = [
            SignalPerformance("earnings", 10, 3.5, 5.0, 65.0, 70.0),
        ]
        mock_sentiment.return_value = [
            SentimentPerformance("bullish", 15, 2.1, 4.0, 55.0),
        ]
        mock_ticker.return_value = [
            TickerPerformance("AAPL", 8, 6, 2, 4.5, 8.0, 36.0),
        ]
        mock_confidence.return_value = [
            ConfidenceCorrelation("high", 20, 5.5, 75.0),
        ]
        mock_best.return_value = [
            {"category": "earnings", "sentiment": "bullish",
             "avg_outcome": 6.0, "occurrences": 12},
        ]
        mock_worst.return_value = [
            {"category": "lawsuit", "sentiment": "bearish",
             "avg_outcome": -3.5, "occurrences": 5},
        ]

        report = generate_pattern_report(90)

        assert "Pattern Analysis Report (90 days)" in report
        assert "Signal Category Performance:" in report
        assert "earnings" in report
        assert "Sentiment Performance:" in report
        assert "bullish" in report
        assert "Ticker Performance" in report
        assert "AAPL" in report
        assert "Confidence vs Outcomes:" in report
        assert "high" in report
        assert "Best Performing Signal Combinations:" in report
        assert "Worst Performing Signal Combinations:" in report

    @patch("trading.patterns.get_worst_performing_signals")
    @patch("trading.patterns.get_best_performing_signals")
    @patch("trading.patterns.analyze_confidence_correlation")
    @patch("trading.patterns.analyze_ticker_performance")
    @patch("trading.patterns.analyze_sentiment_performance")
    @patch("trading.patterns.analyze_signal_categories")
    def test_report_with_no_data(self, mock_signals, mock_sentiment,
                                  mock_ticker, mock_confidence,
                                  mock_best, mock_worst):
        mock_signals.return_value = []
        mock_sentiment.return_value = []
        mock_ticker.return_value = []
        mock_confidence.return_value = []
        mock_best.return_value = []
        mock_worst.return_value = []

        report = generate_pattern_report(30)

        assert "Pattern Analysis Report (30 days)" in report
        # Should not contain section headers when no data
        assert "Signal Category Performance:" not in report
        assert "Sentiment Performance:" not in report
        assert "Ticker Performance" not in report
        assert "Confidence vs Outcomes:" not in report

    @patch("trading.patterns.get_worst_performing_signals")
    @patch("trading.patterns.get_best_performing_signals")
    @patch("trading.patterns.analyze_confidence_correlation")
    @patch("trading.patterns.analyze_ticker_performance")
    @patch("trading.patterns.analyze_sentiment_performance")
    @patch("trading.patterns.analyze_signal_categories")
    def test_report_passes_days_to_all_analyzers(self, mock_signals, mock_sentiment,
                                                   mock_ticker, mock_confidence,
                                                   mock_best, mock_worst):
        for m in [mock_signals, mock_sentiment, mock_ticker, mock_confidence]:
            m.return_value = []
        mock_best.return_value = []
        mock_worst.return_value = []

        generate_pattern_report(45)

        mock_signals.assert_called_once_with(45)
        mock_sentiment.assert_called_once_with(45)
        mock_ticker.assert_called_once_with(45)
        mock_confidence.assert_called_once_with(45)
        mock_best.assert_called_once_with(45)
        mock_worst.assert_called_once_with(45)

    @patch("trading.patterns.get_worst_performing_signals")
    @patch("trading.patterns.get_best_performing_signals")
    @patch("trading.patterns.analyze_confidence_correlation")
    @patch("trading.patterns.analyze_ticker_performance")
    @patch("trading.patterns.analyze_sentiment_performance")
    @patch("trading.patterns.analyze_signal_categories")
    def test_report_returns_string(self, mock_signals, mock_sentiment,
                                    mock_ticker, mock_confidence,
                                    mock_best, mock_worst):
        for m in [mock_signals, mock_sentiment, mock_ticker, mock_confidence]:
            m.return_value = []
        mock_best.return_value = []
        mock_worst.return_value = []

        report = generate_pattern_report(90)
        assert isinstance(report, str)

    @patch("trading.patterns.get_worst_performing_signals")
    @patch("trading.patterns.get_best_performing_signals")
    @patch("trading.patterns.analyze_confidence_correlation")
    @patch("trading.patterns.analyze_ticker_performance")
    @patch("trading.patterns.analyze_sentiment_performance")
    @patch("trading.patterns.analyze_signal_categories")
    def test_report_formats_none_outcomes_as_na(self, mock_signals, mock_sentiment,
                                                  mock_ticker, mock_confidence,
                                                  mock_best, mock_worst):
        mock_signals.return_value = [
            SignalPerformance("earnings", 3, None, None, None, None),
        ]
        mock_sentiment.return_value = []
        mock_ticker.return_value = []
        mock_confidence.return_value = []
        mock_best.return_value = []
        mock_worst.return_value = []

        report = generate_pattern_report(90)
        assert "N/A" in report

    @patch("trading.patterns.get_worst_performing_signals")
    @patch("trading.patterns.get_best_performing_signals")
    @patch("trading.patterns.analyze_confidence_correlation")
    @patch("trading.patterns.analyze_ticker_performance")
    @patch("trading.patterns.analyze_sentiment_performance")
    @patch("trading.patterns.analyze_signal_categories")
    def test_report_limits_tickers_to_five(self, mock_signals, mock_sentiment,
                                            mock_ticker, mock_confidence,
                                            mock_best, mock_worst):
        """Only top 5 tickers should be shown."""
        tickers = [
            TickerPerformance(f"TKR{i}", 5, 3, 2, float(i), float(i * 2), float(i * 5))
            for i in range(10)
        ]
        mock_signals.return_value = []
        mock_sentiment.return_value = []
        mock_ticker.return_value = tickers
        mock_confidence.return_value = []
        mock_best.return_value = []
        mock_worst.return_value = []

        report = generate_pattern_report(90)
        # Count ticker lines in the report
        ticker_lines = [l for l in report.split("\n") if l.strip().startswith("TKR")]
        assert len(ticker_lines) == 5

    @patch("trading.patterns.get_worst_performing_signals")
    @patch("trading.patterns.get_best_performing_signals")
    @patch("trading.patterns.analyze_confidence_correlation")
    @patch("trading.patterns.analyze_ticker_performance")
    @patch("trading.patterns.analyze_sentiment_performance")
    @patch("trading.patterns.analyze_signal_categories")
    def test_report_limits_best_worst_to_three(self, mock_signals, mock_sentiment,
                                                 mock_ticker, mock_confidence,
                                                 mock_best, mock_worst):
        """Only top 3 best/worst should be shown."""
        mock_signals.return_value = []
        mock_sentiment.return_value = []
        mock_ticker.return_value = []
        mock_confidence.return_value = []
        mock_best.return_value = [
            {"category": f"cat{i}", "sentiment": "bullish",
             "avg_outcome": 5.0 - i, "occurrences": 10}
            for i in range(5)
        ]
        mock_worst.return_value = [
            {"category": f"bad{i}", "sentiment": "bearish",
             "avg_outcome": -5.0 + i, "occurrences": 10}
            for i in range(5)
        ]

        report = generate_pattern_report(90)
        best_lines = [l for l in report.split("\n")
                      if l.strip().startswith("cat")]
        worst_lines = [l for l in report.split("\n")
                       if l.strip().startswith("bad")]
        assert len(best_lines) == 3
        assert len(worst_lines) == 3
