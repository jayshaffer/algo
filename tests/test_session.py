"""Tests for trading/session.py - consolidated daily session orchestrator."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.session import SessionResult, run_session
from trading.pipeline import PipelineStats
from trading.ideation_claude import ClaudeIdeationResult
from trading.trader import TradingSessionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_stats(**overrides):
    defaults = dict(
        news_fetched=50,
        news_filtered=30,
        ticker_signals_stored=20,
        macro_signals_stored=5,
        noise_dropped=10,
        errors=0,
    )
    defaults.update(overrides)
    return PipelineStats(**defaults)


def _make_ideation_result(**overrides):
    defaults = dict(
        timestamp=datetime(2025, 6, 15),
        model="claude-opus-4-6",
        turns_used=5,
        theses_created=2,
        theses_updated=1,
        theses_closed=0,
        final_summary="Good session",
        input_tokens=2000,
        output_tokens=800,
    )
    defaults.update(overrides)
    return ClaudeIdeationResult(**defaults)


def _make_trading_result(**overrides):
    defaults = dict(
        timestamp=datetime(2025, 6, 15),
        account_snapshot_id=1,
        positions_synced=3,
        orders_synced=0,
        decisions_made=2,
        trades_executed=1,
        trades_failed=0,
        total_buy_value=Decimal("500.00"),
        total_sell_value=Decimal("0.00"),
        errors=[],
    )
    defaults.update(overrides)
    return TradingSessionResult(**defaults)


# ---------------------------------------------------------------------------
# SessionResult dataclass
# ---------------------------------------------------------------------------

class TestSessionResult:

    def test_no_errors_by_default(self):
        result = SessionResult()
        assert result.has_errors is False

    def test_has_errors_pipeline(self):
        result = SessionResult(pipeline_error="fetch failed")
        assert result.has_errors is True

    def test_has_errors_strategist(self):
        result = SessionResult(strategist_error="Claude down")
        assert result.has_errors is True

    def test_has_errors_trading(self):
        result = SessionResult(trading_error="Alpaca API error")
        assert result.has_errors is True

    def test_has_errors_multiple(self):
        result = SessionResult(
            pipeline_error="err1",
            strategist_error="err2",
            trading_error="err3",
        )
        assert result.has_errors is True

    def test_skip_flags_default_false(self):
        result = SessionResult()
        assert result.skipped_pipeline is False
        assert result.skipped_ideation is False

    def test_skip_flags_set(self):
        result = SessionResult(skipped_pipeline=True, skipped_ideation=True)
        assert result.skipped_pipeline is True
        assert result.skipped_ideation is True

    def test_duration_default_zero(self):
        result = SessionResult()
        assert result.duration_seconds == 0.0

    def test_results_default_none(self):
        result = SessionResult()
        assert result.pipeline_result is None
        assert result.strategist_result is None
        assert result.trading_result is None


# ---------------------------------------------------------------------------
# Shared fixture for run_session tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session_deps():
    """Mock all external dependencies for run_session."""
    with patch("trading.session.check_dependencies") as mock_deps, \
         patch("trading.session.run_pipeline") as mock_pipeline, \
         patch("trading.session.run_strategist_loop") as mock_strategist, \
         patch("trading.session.run_trading_session") as mock_trading:

        mock_deps.return_value = True
        mock_pipeline.return_value = _make_pipeline_stats()
        mock_strategist.return_value = _make_ideation_result()
        mock_trading.return_value = _make_trading_result()

        yield {
            "check_dependencies": mock_deps,
            "run_pipeline": mock_pipeline,
            "run_strategist_loop": mock_strategist,
            "run_trading_session": mock_trading,
        }


# ---------------------------------------------------------------------------
# run_session — happy path
# ---------------------------------------------------------------------------

class TestRunSession:

    def test_happy_path_all_stages(self, mock_session_deps):
        """All three stages should run and produce results."""
        result = run_session()

        assert result.pipeline_result is not None
        assert result.strategist_result is not None
        assert result.trading_result is not None
        assert result.has_errors is False
        assert result.duration_seconds > 0

    def test_calls_dependency_check(self, mock_session_deps):
        run_session()
        mock_session_deps["check_dependencies"].assert_called_once()

    def test_calls_pipeline(self, mock_session_deps):
        run_session()
        mock_session_deps["run_pipeline"].assert_called_once()

    def test_calls_strategist(self, mock_session_deps):
        run_session()
        mock_session_deps["run_strategist_loop"].assert_called_once()

    def test_calls_trading(self, mock_session_deps):
        run_session()
        mock_session_deps["run_trading_session"].assert_called_once()

    def test_forwards_dry_run(self, mock_session_deps):
        run_session(dry_run=True)
        mock_session_deps["run_trading_session"].assert_called_once_with(
            dry_run=True, model="claude-haiku-4-5-20251001",
        )

    def test_forwards_model(self, mock_session_deps):
        run_session(model="claude-sonnet-4-20250514")
        call_kwargs = mock_session_deps["run_strategist_loop"].call_args
        assert call_kwargs[1]["model"] == "claude-sonnet-4-20250514" or \
               call_kwargs.kwargs.get("model") == "claude-sonnet-4-20250514"

    def test_forwards_executor_model(self, mock_session_deps):
        run_session(executor_model="llama3:8b")
        mock_session_deps["run_trading_session"].assert_called_once_with(
            dry_run=False, model="llama3:8b",
        )

    def test_forwards_max_turns(self, mock_session_deps):
        run_session(max_turns=10)
        call_kwargs = mock_session_deps["run_strategist_loop"].call_args
        assert call_kwargs[1]["max_turns"] == 10 or \
               call_kwargs.kwargs.get("max_turns") == 10

    def test_forwards_pipeline_hours(self, mock_session_deps):
        run_session(pipeline_hours=12)
        call_kwargs = mock_session_deps["run_pipeline"].call_args
        assert call_kwargs[1]["hours"] == 12 or \
               call_kwargs.kwargs.get("hours") == 12

    def test_forwards_pipeline_limit(self, mock_session_deps):
        run_session(pipeline_limit=100)
        call_kwargs = mock_session_deps["run_pipeline"].call_args
        assert call_kwargs[1]["limit"] == 100 or \
               call_kwargs.kwargs.get("limit") == 100


# ---------------------------------------------------------------------------
# run_session — skip flags
# ---------------------------------------------------------------------------

class TestRunSessionSkipFlags:

    def test_skip_pipeline(self, mock_session_deps):
        result = run_session(skip_pipeline=True)
        mock_session_deps["run_pipeline"].assert_not_called()
        assert result.skipped_pipeline is True
        assert result.pipeline_result is None

    def test_skip_ideation(self, mock_session_deps):
        result = run_session(skip_ideation=True)
        mock_session_deps["run_strategist_loop"].assert_not_called()
        assert result.skipped_ideation is True
        assert result.strategist_result is None

    def test_skip_both(self, mock_session_deps):
        result = run_session(skip_pipeline=True, skip_ideation=True)
        mock_session_deps["run_pipeline"].assert_not_called()
        mock_session_deps["run_strategist_loop"].assert_not_called()
        # Trading should still run
        mock_session_deps["run_trading_session"].assert_called_once()

    def test_skip_pipeline_still_runs_strategist(self, mock_session_deps):
        run_session(skip_pipeline=True)
        mock_session_deps["run_strategist_loop"].assert_called_once()

    def test_skip_ideation_still_runs_pipeline(self, mock_session_deps):
        run_session(skip_ideation=True)
        mock_session_deps["run_pipeline"].assert_called_once()


# ---------------------------------------------------------------------------
# run_session — resilience (stage failures)
# ---------------------------------------------------------------------------

class TestRunSessionResilience:

    def test_pipeline_failure_continues(self, mock_session_deps):
        """Pipeline failure should not prevent strategist or trading."""
        mock_session_deps["run_pipeline"].side_effect = Exception("news API down")
        result = run_session()

        assert result.pipeline_error == "news API down"
        assert result.pipeline_result is None
        mock_session_deps["run_strategist_loop"].assert_called_once()
        mock_session_deps["run_trading_session"].assert_called_once()

    def test_strategist_failure_continues(self, mock_session_deps):
        """Strategist failure should not prevent trading."""
        mock_session_deps["run_strategist_loop"].side_effect = Exception("Claude error")
        result = run_session()

        assert result.strategist_error == "Claude error"
        assert result.strategist_result is None
        mock_session_deps["run_trading_session"].assert_called_once()

    def test_trading_failure_captured(self, mock_session_deps):
        """Trading failure should be captured in result."""
        mock_session_deps["run_trading_session"].side_effect = Exception("Alpaca down")
        result = run_session()

        assert result.trading_error == "Alpaca down"
        assert result.trading_result is None

    def test_all_stages_fail(self, mock_session_deps):
        """All stages failing should still produce a result."""
        mock_session_deps["run_pipeline"].side_effect = Exception("e1")
        mock_session_deps["run_strategist_loop"].side_effect = Exception("e2")
        mock_session_deps["run_trading_session"].side_effect = Exception("e3")

        result = run_session()

        assert result.pipeline_error == "e1"
        assert result.strategist_error == "e2"
        assert result.trading_error == "e3"
        assert result.has_errors is True
        assert result.duration_seconds > 0

    def test_dependency_check_failure_nonfatal(self, mock_session_deps):
        """Dependency check failure should not prevent any stage."""
        mock_session_deps["check_dependencies"].return_value = False
        result = run_session()

        mock_session_deps["run_pipeline"].assert_called_once()
        mock_session_deps["run_strategist_loop"].assert_called_once()
        mock_session_deps["run_trading_session"].assert_called_once()
        assert result.has_errors is False

    def test_dependency_check_exception_nonfatal(self, mock_session_deps):
        """Dependency check exception should not prevent any stage."""
        mock_session_deps["check_dependencies"].side_effect = Exception("Ollama unreachable")
        result = run_session()

        mock_session_deps["run_pipeline"].assert_called_once()
        mock_session_deps["run_strategist_loop"].assert_called_once()
        mock_session_deps["run_trading_session"].assert_called_once()
