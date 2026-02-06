"""Tests for trading/learn.py - Learning loop orchestrator."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from trading.learn import (
    LearningResult,
    run_learning_loop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backfill_result(total_filled=5):
    """Create a mock backfill result dict."""
    return {
        "7d": {"decisions_found": 3, "outcomes_filled": 2,
               "skipped_no_price": 1, "errors": 0},
        "30d": {"decisions_found": 4, "outcomes_filled": 3,
               "skipped_no_price": 1, "errors": 0},
        "total_filled": total_filled,
    }


def _make_attribution_results(count=3):
    """Create mock attribution results."""
    return [
        {"category": f"news_signal:cat{i}", "sample_size": 10 - i,
         "avg_outcome_7d": 1.5, "avg_outcome_30d": 3.0,
         "win_rate_7d": 0.6, "win_rate_30d": 0.55}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# LearningResult dataclass
# ---------------------------------------------------------------------------

class TestLearningResult:
    """Tests for the LearningResult dataclass."""

    def test_can_construct_with_all_fields(self):
        ts = datetime(2025, 1, 15, 10, 0, 0)
        result = LearningResult(
            timestamp=ts,
            outcomes_backfilled=5,
            attribution_computed=3,
            pattern_report="test report",
            errors=[],
        )
        assert result.timestamp == ts
        assert result.outcomes_backfilled == 5
        assert result.attribution_computed == 3
        assert result.pattern_report == "test report"
        assert result.errors == []

    def test_can_have_zero_attribution(self):
        result = LearningResult(
            timestamp=datetime.now(),
            outcomes_backfilled=0,
            attribution_computed=0,
            pattern_report="",
            errors=[],
        )
        assert result.attribution_computed == 0

    def test_can_have_errors(self):
        result = LearningResult(
            timestamp=datetime.now(),
            outcomes_backfilled=0,
            attribution_computed=0,
            pattern_report="",
            errors=["Error 1", "Error 2"],
        )
        assert len(result.errors) == 2


# ---------------------------------------------------------------------------
# run_learning_loop - full loop
# ---------------------------------------------------------------------------

class TestRunLearningLoopFull:
    """Tests for run_learning_loop() with all steps enabled."""

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_full_loop_returns_learning_result(self, mock_backfill,
                                                 mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(5)
        mock_patterns.return_value = "Pattern report content"
        mock_attribution.return_value = _make_attribution_results(3)

        result = run_learning_loop(analysis_days=60)

        assert isinstance(result, LearningResult)
        assert result.outcomes_backfilled == 5
        assert result.pattern_report == "Pattern report content"
        assert result.attribution_computed == 3
        assert result.errors == []

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_full_loop_calls_all_steps(self, mock_backfill,
                                        mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_attribution.return_value = _make_attribution_results(2)

        run_learning_loop(analysis_days=60, dry_run=False)

        mock_backfill.assert_called_once_with(dry_run=False)
        mock_patterns.assert_called_once_with(days=60)
        mock_attribution.assert_called_once()

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_dry_run_passed_to_backfill(self, mock_backfill,
                                          mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_attribution.return_value = _make_attribution_results(0)

        run_learning_loop(analysis_days=30, dry_run=True)

        mock_backfill.assert_called_once_with(dry_run=True)

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_analysis_days_passed_to_patterns(self, mock_backfill,
                                                mock_patterns,
                                                mock_attribution):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_attribution.return_value = _make_attribution_results(0)

        run_learning_loop(analysis_days=90)

        mock_patterns.assert_called_once_with(days=90)


# ---------------------------------------------------------------------------
# run_learning_loop - skip_backfill
# ---------------------------------------------------------------------------

class TestRunLearningLoopSkipBackfill:
    """Tests for run_learning_loop() with skip_backfill=True."""

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_skips_backfill_step(self, mock_backfill, mock_patterns, mock_attribution):
        mock_patterns.return_value = "report"
        mock_attribution.return_value = _make_attribution_results(2)

        result = run_learning_loop(analysis_days=60, skip_backfill=True)

        mock_backfill.assert_not_called()
        assert result.outcomes_backfilled == 0

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_still_runs_patterns_and_attribution(self, mock_backfill,
                                                   mock_patterns, mock_attribution):
        mock_patterns.return_value = "report"
        mock_attribution.return_value = _make_attribution_results(2)

        run_learning_loop(analysis_days=60, skip_backfill=True)

        mock_patterns.assert_called_once()
        mock_attribution.assert_called_once()


# ---------------------------------------------------------------------------
# run_learning_loop - skip_attribution
# ---------------------------------------------------------------------------

class TestRunLearningLoopSkipAttribution:
    """Tests for run_learning_loop() with skip_attribution=True."""

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_skips_attribution_step(self, mock_backfill, mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(3)
        mock_patterns.return_value = "report"

        result = run_learning_loop(analysis_days=60, skip_attribution=True)

        mock_attribution.assert_not_called()
        assert result.attribution_computed == 0

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_still_runs_backfill_and_patterns(self, mock_backfill,
                                                mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(2)
        mock_patterns.return_value = "report"

        run_learning_loop(analysis_days=60, skip_attribution=True)

        mock_backfill.assert_called_once()
        mock_patterns.assert_called_once()


# ---------------------------------------------------------------------------
# run_learning_loop - skip both
# ---------------------------------------------------------------------------

class TestRunLearningLoopSkipBoth:
    """Tests for run_learning_loop() with both steps skipped."""

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_only_runs_patterns(self, mock_backfill, mock_patterns, mock_attribution):
        mock_patterns.return_value = "report"

        result = run_learning_loop(
            analysis_days=60,
            skip_backfill=True,
            skip_attribution=True,
        )

        mock_backfill.assert_not_called()
        mock_attribution.assert_not_called()
        mock_patterns.assert_called_once()
        assert result.outcomes_backfilled == 0
        assert result.attribution_computed == 0
        assert result.pattern_report == "report"


# ---------------------------------------------------------------------------
# run_learning_loop - error handling
# ---------------------------------------------------------------------------

class TestRunLearningLoopErrors:
    """Tests for error handling in run_learning_loop()."""

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_backfill_error_captured(self, mock_backfill, mock_patterns, mock_attribution):
        mock_backfill.side_effect = Exception("DB connection failed")
        mock_patterns.return_value = "report"
        mock_attribution.return_value = _make_attribution_results(2)

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 1
        assert "Backfill failed" in result.errors[0]
        assert "DB connection failed" in result.errors[0]
        assert result.outcomes_backfilled == 0

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_pattern_error_captured(self, mock_backfill, mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.side_effect = Exception("SQL syntax error")
        mock_attribution.return_value = _make_attribution_results(2)

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 1
        assert "Pattern analysis failed" in result.errors[0]
        assert result.pattern_report == ""

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_attribution_error_captured(self, mock_backfill, mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = "report"
        mock_attribution.side_effect = Exception("Attribution computation failed")

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 1
        assert "Attribution computation failed" in result.errors[0]
        assert result.attribution_computed == 0

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_all_steps_fail(self, mock_backfill, mock_patterns, mock_attribution):
        mock_backfill.side_effect = Exception("backfill error")
        mock_patterns.side_effect = Exception("patterns error")
        mock_attribution.side_effect = Exception("attribution error")

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 3
        assert result.outcomes_backfilled == 0
        assert result.pattern_report == ""
        assert result.attribution_computed == 0

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_backfill_error_does_not_prevent_patterns(self, mock_backfill,
                                                        mock_patterns, mock_attribution):
        mock_backfill.side_effect = Exception("backfill error")
        mock_patterns.return_value = "patterns ok"
        mock_attribution.return_value = _make_attribution_results(2)

        result = run_learning_loop(analysis_days=60)

        mock_patterns.assert_called_once()
        assert result.pattern_report == "patterns ok"
        assert result.attribution_computed == 2

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_pattern_error_does_not_prevent_attribution(self, mock_backfill,
                                                          mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(3)
        mock_patterns.side_effect = Exception("patterns error")
        mock_attribution.return_value = _make_attribution_results(2)

        result = run_learning_loop(analysis_days=60)

        mock_attribution.assert_called_once()
        assert result.attribution_computed == 2

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_skipped_backfill_error_not_recorded(self, mock_backfill,
                                                   mock_patterns, mock_attribution):
        """Skipped steps should not produce errors even if they would fail."""
        mock_backfill.side_effect = Exception("should not be called")
        mock_patterns.return_value = "report"
        mock_attribution.return_value = _make_attribution_results(2)

        result = run_learning_loop(analysis_days=60, skip_backfill=True)

        assert result.errors == []


# ---------------------------------------------------------------------------
# run_learning_loop - timestamp and defaults
# ---------------------------------------------------------------------------

class TestRunLearningLoopMisc:
    """Miscellaneous tests for run_learning_loop()."""

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_result_has_timestamp(self, mock_backfill, mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_attribution.return_value = _make_attribution_results(0)

        before = datetime.now()
        result = run_learning_loop(analysis_days=60)
        after = datetime.now()

        assert before <= result.timestamp <= after

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_default_parameters(self, mock_backfill, mock_patterns, mock_attribution):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_attribution.return_value = _make_attribution_results(0)

        run_learning_loop()

        # Default analysis_days=60, dry_run=False
        mock_backfill.assert_called_once_with(dry_run=False)
        mock_patterns.assert_called_once_with(days=60)
        mock_attribution.assert_called_once()

    @patch("trading.learn.compute_signal_attribution")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_attribution_count_matches_results(self, mock_backfill, mock_patterns, mock_attribution):
        """attribution_computed should equal the length of attribution results."""
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = "report"
        mock_attribution.return_value = _make_attribution_results(5)

        result = run_learning_loop(analysis_days=60)

        assert result.attribution_computed == 5
