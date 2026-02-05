"""Tests for trading/learn.py - Learning loop orchestrator."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from trading.learn import (
    LearningResult,
    run_learning_loop,
)
from trading.strategy import StrategyRecommendation


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


def _make_strategy():
    """Create a mock StrategyRecommendation."""
    return StrategyRecommendation(
        watchlist=["AAPL", "MSFT"],
        avoid_list=["BAD1"],
        risk_tolerance="moderate",
        focus_sectors=["tech"],
        description="Test strategy",
        reasoning=["reason1", "reason2"],
    )


# ---------------------------------------------------------------------------
# LearningResult dataclass
# ---------------------------------------------------------------------------

class TestLearningResult:
    """Tests for the LearningResult dataclass."""

    def test_can_construct_with_all_fields(self):
        ts = datetime(2025, 1, 15, 10, 0, 0)
        strategy = _make_strategy()
        result = LearningResult(
            timestamp=ts,
            outcomes_backfilled=5,
            pattern_report="test report",
            strategy=strategy,
            errors=[],
        )
        assert result.timestamp == ts
        assert result.outcomes_backfilled == 5
        assert result.pattern_report == "test report"
        assert result.strategy is strategy
        assert result.errors == []

    def test_can_have_none_strategy(self):
        result = LearningResult(
            timestamp=datetime.now(),
            outcomes_backfilled=0,
            pattern_report="",
            strategy=None,
            errors=[],
        )
        assert result.strategy is None

    def test_can_have_errors(self):
        result = LearningResult(
            timestamp=datetime.now(),
            outcomes_backfilled=0,
            pattern_report="",
            strategy=None,
            errors=["Error 1", "Error 2"],
        )
        assert len(result.errors) == 2


# ---------------------------------------------------------------------------
# run_learning_loop - full loop
# ---------------------------------------------------------------------------

class TestRunLearningLoopFull:
    """Tests for run_learning_loop() with all steps enabled."""

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_full_loop_returns_learning_result(self, mock_backfill,
                                                 mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(5)
        mock_patterns.return_value = "Pattern report content"
        mock_strategy.return_value = _make_strategy()

        result = run_learning_loop(analysis_days=60)

        assert isinstance(result, LearningResult)
        assert result.outcomes_backfilled == 5
        assert result.pattern_report == "Pattern report content"
        assert result.strategy is not None
        assert result.errors == []

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_full_loop_calls_all_steps(self, mock_backfill,
                                        mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_strategy.return_value = _make_strategy()

        run_learning_loop(analysis_days=60, dry_run=False)

        mock_backfill.assert_called_once_with(dry_run=False)
        mock_patterns.assert_called_once_with(days=60)
        mock_strategy.assert_called_once_with(days=60, dry_run=False)

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_dry_run_passed_to_backfill_and_strategy(self, mock_backfill,
                                                       mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_strategy.return_value = _make_strategy()

        run_learning_loop(analysis_days=30, dry_run=True)

        mock_backfill.assert_called_once_with(dry_run=True)
        mock_strategy.assert_called_once_with(days=30, dry_run=True)

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_analysis_days_passed_to_patterns_and_strategy(self, mock_backfill,
                                                             mock_patterns,
                                                             mock_strategy):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_strategy.return_value = _make_strategy()

        run_learning_loop(analysis_days=90)

        mock_patterns.assert_called_once_with(days=90)
        mock_strategy.assert_called_once_with(days=90, dry_run=False)


# ---------------------------------------------------------------------------
# run_learning_loop - skip_backfill
# ---------------------------------------------------------------------------

class TestRunLearningLoopSkipBackfill:
    """Tests for run_learning_loop() with skip_backfill=True."""

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_skips_backfill_step(self, mock_backfill, mock_patterns, mock_strategy):
        mock_patterns.return_value = "report"
        mock_strategy.return_value = _make_strategy()

        result = run_learning_loop(analysis_days=60, skip_backfill=True)

        mock_backfill.assert_not_called()
        assert result.outcomes_backfilled == 0

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_still_runs_patterns_and_strategy(self, mock_backfill,
                                                mock_patterns, mock_strategy):
        mock_patterns.return_value = "report"
        mock_strategy.return_value = _make_strategy()

        run_learning_loop(analysis_days=60, skip_backfill=True)

        mock_patterns.assert_called_once()
        mock_strategy.assert_called_once()


# ---------------------------------------------------------------------------
# run_learning_loop - skip_strategy
# ---------------------------------------------------------------------------

class TestRunLearningLoopSkipStrategy:
    """Tests for run_learning_loop() with skip_strategy=True."""

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_skips_strategy_step(self, mock_backfill, mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(3)
        mock_patterns.return_value = "report"

        result = run_learning_loop(analysis_days=60, skip_strategy=True)

        mock_strategy.assert_not_called()
        assert result.strategy is None

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_still_runs_backfill_and_patterns(self, mock_backfill,
                                                mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(2)
        mock_patterns.return_value = "report"

        run_learning_loop(analysis_days=60, skip_strategy=True)

        mock_backfill.assert_called_once()
        mock_patterns.assert_called_once()


# ---------------------------------------------------------------------------
# run_learning_loop - skip both
# ---------------------------------------------------------------------------

class TestRunLearningLoopSkipBoth:
    """Tests for run_learning_loop() with both steps skipped."""

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_only_runs_patterns(self, mock_backfill, mock_patterns, mock_strategy):
        mock_patterns.return_value = "report"

        result = run_learning_loop(
            analysis_days=60,
            skip_backfill=True,
            skip_strategy=True,
        )

        mock_backfill.assert_not_called()
        mock_strategy.assert_not_called()
        mock_patterns.assert_called_once()
        assert result.outcomes_backfilled == 0
        assert result.strategy is None
        assert result.pattern_report == "report"


# ---------------------------------------------------------------------------
# run_learning_loop - error handling
# ---------------------------------------------------------------------------

class TestRunLearningLoopErrors:
    """Tests for error handling in run_learning_loop()."""

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_backfill_error_captured(self, mock_backfill, mock_patterns, mock_strategy):
        mock_backfill.side_effect = Exception("DB connection failed")
        mock_patterns.return_value = "report"
        mock_strategy.return_value = _make_strategy()

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 1
        assert "Backfill failed" in result.errors[0]
        assert "DB connection failed" in result.errors[0]
        assert result.outcomes_backfilled == 0

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_pattern_error_captured(self, mock_backfill, mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.side_effect = Exception("SQL syntax error")
        mock_strategy.return_value = _make_strategy()

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 1
        assert "Pattern analysis failed" in result.errors[0]
        assert result.pattern_report == ""

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_strategy_error_captured(self, mock_backfill, mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = "report"
        mock_strategy.side_effect = Exception("Strategy generation failed")

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 1
        assert "Strategy evolution failed" in result.errors[0]
        assert result.strategy is None

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_all_steps_fail(self, mock_backfill, mock_patterns, mock_strategy):
        mock_backfill.side_effect = Exception("backfill error")
        mock_patterns.side_effect = Exception("patterns error")
        mock_strategy.side_effect = Exception("strategy error")

        result = run_learning_loop(analysis_days=60)

        assert len(result.errors) == 3
        assert result.outcomes_backfilled == 0
        assert result.pattern_report == ""
        assert result.strategy is None

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_backfill_error_does_not_prevent_patterns(self, mock_backfill,
                                                        mock_patterns, mock_strategy):
        mock_backfill.side_effect = Exception("backfill error")
        mock_patterns.return_value = "patterns ok"
        mock_strategy.return_value = _make_strategy()

        result = run_learning_loop(analysis_days=60)

        mock_patterns.assert_called_once()
        assert result.pattern_report == "patterns ok"
        assert result.strategy is not None

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_pattern_error_does_not_prevent_strategy(self, mock_backfill,
                                                       mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(3)
        mock_patterns.side_effect = Exception("patterns error")
        mock_strategy.return_value = _make_strategy()

        result = run_learning_loop(analysis_days=60)

        mock_strategy.assert_called_once()
        assert result.strategy is not None

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_skipped_backfill_error_not_recorded(self, mock_backfill,
                                                   mock_patterns, mock_strategy):
        """Skipped steps should not produce errors even if they would fail."""
        mock_backfill.side_effect = Exception("should not be called")
        mock_patterns.return_value = "report"
        mock_strategy.return_value = _make_strategy()

        result = run_learning_loop(analysis_days=60, skip_backfill=True)

        assert result.errors == []


# ---------------------------------------------------------------------------
# run_learning_loop - timestamp and defaults
# ---------------------------------------------------------------------------

class TestRunLearningLoopMisc:
    """Miscellaneous tests for run_learning_loop()."""

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_result_has_timestamp(self, mock_backfill, mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_strategy.return_value = _make_strategy()

        before = datetime.now()
        result = run_learning_loop(analysis_days=60)
        after = datetime.now()

        assert before <= result.timestamp <= after

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_default_parameters(self, mock_backfill, mock_patterns, mock_strategy):
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = ""
        mock_strategy.return_value = _make_strategy()

        run_learning_loop()

        # Default analysis_days=60, dry_run=False
        mock_backfill.assert_called_once_with(dry_run=False)
        mock_patterns.assert_called_once_with(days=60)
        mock_strategy.assert_called_once_with(days=60, dry_run=False)

    @patch("trading.learn.evolve_strategy")
    @patch("trading.learn.generate_pattern_report")
    @patch("trading.learn.run_backfill")
    def test_dry_run_with_strategy(self, mock_backfill, mock_patterns, mock_strategy):
        """In dry_run mode, strategy object is still returned but not saved."""
        mock_backfill.return_value = _make_backfill_result(0)
        mock_patterns.return_value = "report"
        strategy = _make_strategy()
        mock_strategy.return_value = strategy

        result = run_learning_loop(analysis_days=60, dry_run=True)

        assert result.strategy is strategy
        mock_strategy.assert_called_once_with(days=60, dry_run=True)
