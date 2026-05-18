"""Tests for 5-stage session orchestrator."""
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.dashboard_publish import DashboardStageResult
from v2.session import SessionResult, run_session
from v2.strategy import StrategyReflectionResult
from v2.trader import TradingSessionResult


@pytest.fixture(autouse=True)
def _bypass_session_idempotency():
    """All tests in this module should exercise run_session as if no prior session exists today.

    The production idempotency check reads the live Postgres `sessions` row for
    today's date; leaving that intact makes tests flaky against shared DB state.

    Also stubs get_playbook so the strategist-stage playbook-check assertion
    (added to catch max_tokens / max_turns early-exits that leave no playbook)
    passes by default. Tests that want to simulate "strategist wrote no
    playbook" patch get_playbook to return None inside the test body.
    """
    with patch("v2.session.get_session_for_date", return_value=None), \
         patch("v2.session.get_playbook", return_value={"id": 1}):
        yield


class TestRunSession:
    def test_stage_0_runs_before_pipeline(self):
        """Learning refresh (backfill + attribution) runs before news pipeline."""
        call_order = []

        with patch("v2.session.run_backfill") as mock_backfill, \
             patch("v2.session.compute_signal_attribution") as mock_attr, \
             patch("v2.session.build_attribution_constraints", return_value="CONSTRAINTS") as mock_constraints, \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session") as mock_trade:

            mock_backfill.side_effect = lambda **kw: call_order.append("backfill")
            mock_attr.side_effect = lambda: call_order.append("attribution") or []
            mock_pipeline.side_effect = lambda **kw: call_order.append("pipeline")
            mock_strat.side_effect = lambda **kw: call_order.append("strategist")
            mock_trade.side_effect = lambda **kw: call_order.append("trader")

            run_session(dry_run=False)

        assert call_order.index("backfill") < call_order.index("pipeline")
        assert call_order.index("attribution") < call_order.index("strategist")

    def test_attribution_constraints_passed_to_strategist(self):
        """Stage 0's constraints should be passed to Stage 2."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value="STRONG: earnings"), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session"):

            run_session(dry_run=False)

        mock_strat.assert_called_once()
        assert "STRONG: earnings" in str(mock_strat.call_args)

    def test_market_date_passed_to_executor(self):
        """One canonical market date should own session and decision rows."""
        market_date = date(2026, 5, 15)
        with patch("v2.session.current_market_date", return_value=market_date), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.insert_session_record", return_value=77) as mock_insert:

            run_session(dry_run=False)

        mock_insert.assert_called_once_with(market_date)
        assert mock_trade.call_args.kwargs["session_date"] == market_date

    def test_stage_0_failure_does_not_block(self):
        """If learning refresh fails, session continues with stale data."""
        with patch("v2.session.run_backfill", side_effect=Exception("DB error")), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"):

            result = run_session(dry_run=False)

        assert result.learning_error is not None
        assert "DB error" in result.learning_error

    def test_stage_0_tracked_in_session_stages(self):
        """T2.2: learning refresh must be tracked as a session_stages row
        so resume + dashboards see it the same way as every other stage.
        """
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.insert_session_record", return_value=42), \
             patch("v2.session.insert_session_stage") as mock_start, \
             patch("v2.session.complete_session_stage") as mock_complete, \
             patch("v2.session.fail_session_stage"):

            run_session(dry_run=False)

        # Both start and complete must be called for the "learning" stage.
        start_stages = [c.args[1] for c in mock_start.call_args_list]
        complete_stages = [c.args[1] for c in mock_complete.call_args_list]
        assert "learning" in start_stages
        assert "learning" in complete_stages

    def test_stage_0_failure_marks_stage_failed(self):
        """T2.2: failure path also lands in session_stages as 'failed'."""
        with patch("v2.session.run_backfill", side_effect=Exception("DB error")), \
             patch("v2.session.compute_signal_attribution"), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.insert_session_record", return_value=42), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.complete_session_stage"), \
             patch("v2.session.fail_session_stage") as mock_fail:

            run_session(dry_run=False)

        # learning stage should be the one marked failed.
        failed_stages = [c.args[1] for c in mock_fail.call_args_list]
        assert "learning" in failed_stages

    def test_idempotent_skip_routes_to_idempotent_field(self):
        """T2.1: when a session is already completed and force=False, the
        early return must populate idempotent_skip — not learning_error.
        That keeps has_errors False so main() exits 0.
        """
        with patch("v2.session.get_session_for_date",
                   return_value={"id": 42, "status": "completed"}), \
             patch("v2.session.run_backfill") as mock_backfill, \
             patch("v2.session.run_pipeline") as mock_pipeline:

            result = run_session(dry_run=False)

        assert result.idempotent_skip is not None
        assert "already completed" in result.idempotent_skip
        assert result.learning_error is None
        assert result.has_errors is False
        # Importantly, no stage work runs after the idempotent skip.
        mock_backfill.assert_not_called()
        mock_pipeline.assert_not_called()

    def test_skip_pipeline(self):
        """Pipeline should be skipped when flag is set."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"):

            result = run_session(dry_run=False, skip_pipeline=True)

        mock_pipeline.assert_not_called()
        assert result.skipped_pipeline is True

    def test_skip_ideation(self):
        """Strategist should be skipped when flag is set."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session"):

            result = run_session(dry_run=False, skip_ideation=True)

        mock_strat.assert_not_called()
        assert result.skipped_ideation is True

    def test_has_errors_property(self):
        """has_errors should return True when any error is set."""
        result = SessionResult(learning_error="test error")
        assert result.has_errors is True

        result = SessionResult()
        assert result.has_errors is False

    def test_all_stages_run(self):
        """All 4 stages should run in order."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session") as mock_trade:

            result = run_session(dry_run=False)

        mock_pipeline.assert_called_once()
        mock_strat.assert_called_once()
        mock_trade.assert_called_once()


class TestSessionResult:
    def test_has_learning_error_field(self):
        result = SessionResult()
        assert hasattr(result, 'learning_error')

    def test_default_values(self):
        result = SessionResult()
        assert result.pipeline_result is None
        assert result.strategist_result is None
        assert result.trading_result is None
        assert result.strategy_result is None
        assert result.learning_error is None
        assert result.pipeline_error is None
        assert result.strategist_error is None
        assert result.trading_error is None
        assert result.strategy_error is None
        assert result.idempotent_skip is None
        assert result.skipped_pipeline is False
        assert result.skipped_ideation is False
        assert result.skipped_strategy is False
        assert result.dashboard_result is None
        assert result.dashboard_error is None
        assert result.skipped_dashboard is False
        assert result.duration_seconds == 0.0

    def test_idempotent_skip_not_an_error(self):
        """T2.1: idempotent_skip is a benign signal, not an error.
        has_errors must remain False so main() exits 0.
        """
        result = SessionResult(idempotent_skip="Session already completed for 2026-05-03")
        assert result.has_errors is False


class TestStage4StrategyReflection:
    def test_stage_4_runs_after_trading(self):
        """Strategy reflection should run after trading session."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection") as mock_reflect:

            mock_trade.side_effect = lambda **kw: call_order.append("trader")
            mock_reflect.side_effect = lambda **kw: call_order.append("reflection")

            run_session(dry_run=False)

        assert call_order.index("trader") < call_order.index("reflection")

    def test_stage_4_result_captured(self):
        """Strategy reflection result should be in SessionResult."""
        mock_reflection_result = StrategyReflectionResult(
            rules_proposed=1, rules_retired=0, identity_updated=True,
            memo_written=True, input_tokens=500, output_tokens=200, turns_used=3,
        )

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection", return_value=mock_reflection_result):

            result = run_session(dry_run=False)

        assert result.strategy_result is not None
        assert result.strategy_result.rules_proposed == 1

    def test_stage_4_failure_does_not_block(self):
        """Strategy reflection failure should be captured but not crash."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection", side_effect=Exception("Opus down")):

            result = run_session(dry_run=False)

        assert result.strategy_error == "Opus down"
        assert result.strategy_result is None

    def test_stage_4_error_in_has_errors(self):
        """Strategy error should be included in has_errors check."""
        result = SessionResult(strategy_error="test")
        assert result.has_errors is True

    def test_skip_strategy_flag(self):
        """Strategy reflection should be skippable."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection") as mock_reflect:

            result = run_session(dry_run=False, skip_strategy=True)

        mock_reflect.assert_not_called()
        assert result.skipped_strategy is True


class TestStage5Dashboard:
    def test_dashboard_runs_after_strategy(self):
        """Dashboard should run after strategy reflection."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection") as mock_reflect, \
             patch("v2.session.run_dashboard_stage") as mock_dashboard:

            mock_reflect.side_effect = lambda **kw: call_order.append("reflection")
            mock_dashboard.side_effect = lambda: call_order.append("dashboard")

            run_session(dry_run=False)

        assert call_order.index("reflection") < call_order.index("dashboard")

    def test_dashboard_result_captured(self):
        """Dashboard result should be in SessionResult."""
        mock_dashboard_result = DashboardStageResult(published=True)

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage", return_value=mock_dashboard_result):

            result = run_session(dry_run=False)

        assert result.dashboard_result is not None
        assert result.dashboard_result.published is True

    def test_dashboard_failure_does_not_block(self):
        """Dashboard failure should be captured but not crash."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage", side_effect=Exception("gh-pages down")):

            result = run_session(dry_run=False)

        assert result.dashboard_error == "gh-pages down"
        assert result.dashboard_result is None

    def test_dashboard_error_in_has_errors(self):
        """Dashboard error should be included in has_errors check."""
        result = SessionResult(dashboard_error="test")
        assert result.has_errors is True

    def test_skip_dashboard_flag(self):
        """Dashboard should be skippable."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage") as mock_dashboard:

            result = run_session(dry_run=False, skip_dashboard=True)

        mock_dashboard.assert_not_called()
        assert result.skipped_dashboard is True


class TestSessionIdempotency:
    def test_blocks_duplicate_session(self):
        """Should refuse to run if a completed session exists for today.
        T2.1: this is a benign skip, not an error — has_errors stays False
        and the result instead carries `idempotent_skip`.
        """
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline:

            mock_get.return_value = {"id": 1, "status": "completed"}

            result = run_session(dry_run=False)

        mock_pipeline.assert_not_called()
        assert result.idempotent_skip is not None
        assert result.has_errors is False

    def test_allows_session_when_none_exists(self):
        """Should proceed normally when no session exists for today."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=1), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        mock_pipeline.assert_called_once()

    def test_allows_session_when_previous_failed(self):
        """Should allow re-run if previous session failed."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.insert_session_record", return_value=2), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "failed"}

            result = run_session(dry_run=False)

        mock_pipeline.assert_called_once()

    def test_force_flag_overrides_idempotency(self):
        """--force should allow running even if completed session exists."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.insert_session_record", return_value=2), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "completed"}

            result = run_session(dry_run=False, force=True)

        mock_pipeline.assert_called_once()

    def test_force_creates_new_session_row_when_one_already_exists(self):
        """Per-run uniqueness: --force inserts a brand-new sessions row
        even when a completed session already exists for today."""

        from v2.session import _check_and_record_session
        with patch("v2.session.insert_session_record", return_value=99) as mock_insert, \
             patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "completed"}):
            session_id, completed, err = _check_and_record_session(force=True, session_date=date(2026, 5, 13))
        assert session_id == 99
        assert completed == set()
        assert err is None
        mock_insert.assert_called_once()

    def test_no_force_skips_when_completed_session_exists(self):

        from v2.session import _check_and_record_session
        with patch("v2.session.insert_session_record") as mock_insert, \
             patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "completed"}):
            session_id, completed, err = _check_and_record_session(force=False, session_date=date(2026, 5, 13))
        assert session_id is None
        assert err is not None
        mock_insert.assert_not_called()

    def test_no_force_creates_new_session_when_prior_was_failed(self):
        """A failed prior session does not gate; a fresh run gets its own
        session row (no resume, no stage skipping)."""

        from v2.session import _check_and_record_session
        with patch("v2.session.insert_session_record", return_value=12) as mock_insert, \
             patch("v2.session.get_session_for_date", return_value={"id": 5, "status": "failed"}):
            session_id, completed, err = _check_and_record_session(force=False, session_date=date(2026, 5, 13))
        assert session_id == 12
        assert completed == set()
        assert err is None
        mock_insert.assert_called_once()


class TestPerStageResume:
    def test_no_resume_all_stages_run_after_prior_failure(self):
        """Per-run sessions: a re-run after a failed prior session does NOT skip
        any stages — every invocation runs every stage from scratch.
        """
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.insert_session_record", return_value=2), \
             patch("v2.session.complete_session"), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.complete_session_stage"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "failed"}

            result = run_session(dry_run=False)

        mock_pipeline.assert_called_once()
        mock_strat.assert_called_once()
        mock_trade.assert_called_once()

    def test_stage_tracking_calls_insert_and_complete(self):
        """Successful stage should call insert_session_stage then complete_session_stage."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.insert_session_stage") as mock_insert_stage, \
             patch("v2.session.complete_session_stage") as mock_complete_stage, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            run_session(dry_run=False)

        # Should have called insert_session_stage for each stage
        stage_names_inserted = [call[0][1] for call in mock_insert_stage.call_args_list]
        assert "pipeline" in stage_names_inserted
        assert "strategist" in stage_names_inserted
        assert "executor" in stage_names_inserted
        # Should have called complete_session_stage for each stage
        stage_names_completed = [call[0][1] for call in mock_complete_stage.call_args_list]
        assert "pipeline" in stage_names_completed
        assert "executor" in stage_names_completed

    def test_failed_stage_calls_fail_session_stage(self):
        """A stage that raises should call fail_session_stage."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.fail_session"), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.complete_session_stage"), \
             patch("v2.session.fail_session_stage") as mock_fail_stage, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline", side_effect=Exception("fetch error")), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        assert result.pipeline_error == "fetch error"
        # fail_session_stage now takes a usage kwarg; assert the positional args match
        matching = [
            c for c in mock_fail_stage.call_args_list
            if c.args[:3] == (5, "pipeline", "fetch error")
        ]
        assert matching, f"Expected fail_session_stage(5, 'pipeline', 'fetch error', ...); got {mock_fail_stage.call_args_list}"

    def test_stage_tracking_failure_does_not_break_session(self):
        """If insert_session_stage raises, the stage should still run."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.insert_session_stage", side_effect=Exception("DB down")), \
             patch("v2.session.complete_session_stage", side_effect=Exception("DB down")), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        # Pipeline should still have run despite tracking failures
        mock_pipeline.assert_called_once()
        # No pipeline_error since the pipeline itself succeeded
        assert result.pipeline_error is None


class TestStrategistMemoPersistence:
    def test_strategist_summary_written_as_memo(self):
        """After Stage 2 completes, the summary should be saved as a strategy memo."""
        mock_ideation_result = MagicMock()
        mock_ideation_result.final_summary = "Strategist reasoning about today's decisions"

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", return_value=mock_ideation_result), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_strategy_memo") as mock_memo, \
             patch("v2.session.get_current_strategy_state", return_value={"id": 1}):

            run_session(dry_run=False)

        mock_memo.assert_called_once()
        assert "strategist_notes" in str(mock_memo.call_args)
        assert "Strategist reasoning" in str(mock_memo.call_args)

    def test_strategist_memo_not_written_on_failure(self):
        """If Stage 2 fails, no memo should be written."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", side_effect=Exception("Opus down")), \
             patch("v2.session.get_playbook", return_value=None), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_strategy_memo") as mock_memo:

            run_session(dry_run=False)

        mock_memo.assert_not_called()

    def test_strategist_memo_not_written_when_playbook_missing(self):
        """P2.24: when run_strategist_loop completes but never wrote a playbook
        (max_tokens / max_turns), the validation must fail BEFORE the memo is
        committed. Otherwise a rerun would insert a duplicate memo on the
        retry, and reflection would double-count the strategist's voice.
        """
        mock_ideation_result = MagicMock()
        mock_ideation_result.final_summary = "Strategist would-be summary"

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", return_value=mock_ideation_result), \
             patch("v2.session.get_playbook", return_value=None), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.get_current_strategy_state", return_value={"id": 1}), \
             patch("v2.session.insert_strategy_memo") as mock_memo:

            result = run_session(dry_run=False)

        mock_memo.assert_not_called()
        assert result.strategist_error is not None
        assert "without writing a playbook" in result.strategist_error

    def test_strategist_summary_memo_save_passes_session_id(self):
        """The strategist-stage path saves the strategist's final summary as a
        memo. Verify that the direct insert_strategy_memo call in session.py
        receives session_id from the caller."""
        mock_ideation_result = MagicMock()
        mock_ideation_result.final_summary = (
            "A sufficiently long strategist summary that passes the min-length guard."
        )

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", return_value=mock_ideation_result), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.get_current_strategy_state", return_value={"id": 1}), \
             patch("v2.session._check_and_record_session", return_value=(7777, set(), None)), \
             patch("v2.session.insert_strategy_memo") as mock_memo:

            run_session(dry_run=False)

        mock_memo.assert_called_once()
        assert mock_memo.call_args.kwargs.get("session_id") == 7777

    def test_strategist_memo_skipped_when_summary_too_short(self):
        """ALGO-15: strategist agentic loops occasionally end with a stub
        final assistant text ("all good", "done"). Persisting those as
        strategist_notes pollutes the journal — recent prod data showed
        103/112 memos collapsed to the literal string "all good".
        Skip the insert when the summary is below the min-length guard."""
        mock_ideation_result = MagicMock()
        mock_ideation_result.final_summary = "all good"

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", return_value=mock_ideation_result), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.get_current_strategy_state", return_value={"id": 1}), \
             patch("v2.session.insert_strategy_memo") as mock_memo:

            run_session(dry_run=False)

        mock_memo.assert_not_called()


class TestReflectionMemoGuard:
    """P1.15: Stage 4 reflection must produce a memo or fail the stage —
    parallel to P2.24 for the strategist. If the LLM finishes the loop
    without calling write_strategy_memo, the journal silently has a hole
    for that session, and on resume the stage is "completed" so the
    LLM never gets another chance to fill it. Failing the stage instead
    leaves it incomplete so the next session retries."""

    def test_reflection_stage_fails_when_memo_not_written(self):
        """If memo_written is False, the stage must not be marked complete
        and an error must be recorded so the next run retries.

        Patches insert_session_record so session tracking is active and
        complete_session_stage observability becomes meaningful (otherwise
        session_id is None and the stage-completion calls short-circuit)."""
        no_memo_result = StrategyReflectionResult(
            rules_proposed=0, rules_retired=0, identity_updated=False,
            memo_written=False, input_tokens=500, output_tokens=200, turns_used=3,
        )

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection", return_value=no_memo_result), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_session_record", return_value=42), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.complete_session_stage") as mock_complete, \
             patch("v2.session.fail_session_stage"), \
             patch("v2.session.complete_session"), \
             patch("v2.session.fail_session"):

            result = run_session(dry_run=False)

        # The strategy stage must NOT be marked complete — otherwise resume
        # would skip the reflection on the next run, and the memo gap stays.
        completed_stage_names = [
            call.args[1] for call in mock_complete.call_args_list
        ]
        assert "strategy" not in completed_stage_names, (
            "Reflection stage was marked complete despite no memo — resume "
            "will skip it next session, locking in the journal hole."
        )
        assert result.strategy_error is not None
        assert "memo" in result.strategy_error.lower()

    def test_reflection_stage_completes_when_memo_written(self):
        """Sanity check: a normal reflection with memo_written=True must
        still mark the stage complete."""
        good_result = StrategyReflectionResult(
            rules_proposed=1, rules_retired=0, identity_updated=False,
            memo_written=True, input_tokens=500, output_tokens=200, turns_used=3,
        )

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection", return_value=good_result), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_session_record", return_value=42), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.complete_session_stage") as mock_complete, \
             patch("v2.session.fail_session_stage"), \
             patch("v2.session.complete_session"), \
             patch("v2.session.fail_session"):

            result = run_session(dry_run=False)

        completed_stage_names = [
            call.args[1] for call in mock_complete.call_args_list
        ]
        assert "strategy" in completed_stage_names
        assert result.strategy_error is None


class TestExecutorPlaybookDependency:
    def test_executor_skipped_when_strategist_fails_and_no_playbook(self):
        """Stage 3 should be skipped if Stage 2 failed and no playbook exists."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", side_effect=Exception("Opus down")), \
             patch("v2.session.get_playbook", return_value=None), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        mock_trade.assert_not_called()
        assert result.skipped_executor is True
        assert result.strategist_error == "Opus down"

    def test_executor_runs_when_strategist_fails_but_playbook_exists(self):
        """Stage 3 should run if Stage 2 failed but a prior playbook exists."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", side_effect=Exception("Opus down")), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        mock_trade.assert_called_once()

class TestStrategistMissingPlaybook:
    """If run_strategist_loop returns without raising but never called
    write_playbook (e.g. it hit max_tokens mid-tool-call), the stage must
    still be recorded as a failure so the executor guard skips.
    """

    def test_missing_playbook_marks_strategist_failed(self):
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.get_playbook", return_value=None), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            mock_strat.return_value = MagicMock(final_summary=None)

            result = run_session(dry_run=False)

        assert result.strategist_error is not None
        assert "no playbook" in result.strategist_error.lower() or "write_playbook" in result.strategist_error
        mock_trade.assert_not_called()
        assert result.skipped_executor is True

    def test_playbook_written_completes_stage_normally(self):
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            mock_strat.return_value = MagicMock(final_summary="all good")

            result = run_session(dry_run=False)

        assert result.strategist_error is None
        mock_trade.assert_called_once()


# ---------------------------------------------------------------------------
# Branch-coverage completion tests (one branch per test)
#
# These cover paths not exercised by the semantic/behavior tests above:
# - exception handlers around DB tracking calls
# - the session_id-is-None branch of per-stage fail paths
# - the final fail_session exception handler
# - main() CLI entry point (both success and --has-errors exit paths)
# ---------------------------------------------------------------------------


class TestSessionIdempotencyExceptionHandler:
    def test_idempotency_check_exception_does_not_block(self):
        """If get_session_for_date raises, session proceeds with a fresh record."""
        with patch("v2.session.get_session_for_date", side_effect=RuntimeError("db dead")), \
             patch("v2.session.insert_session_record", return_value=99), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        # Idempotency exception is swallowed; session ran end-to-end.
        assert not result.has_errors


class TestStageFailFailureSwallowed:
    """Each stage's fail_session_stage call is wrapped in try/except: pass.
    Stages must continue even if the DB tracking itself fails.
    """

    def test_pipeline_fail_session_stage_exception_is_swallowed(self):
        with patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.fail_session"), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.fail_session_stage", side_effect=Exception("tracking down")), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline", side_effect=Exception("pipeline broke")), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        assert result.pipeline_error == "pipeline broke"

    def test_pipeline_failure_without_session_id_takes_short_branch(self):
        """Pipeline raises AND insert_session_record failed (session_id=None).
        Covers the 'if session_id:' False branch in the pipeline except handler.
        """
        with patch("v2.session.insert_session_record", side_effect=Exception("cant track")), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline", side_effect=Exception("pipeline broke")), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        assert result.pipeline_error == "pipeline broke"

    def test_strategist_failure_tracks_fail_session_stage(self):
        with patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.fail_session"), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.fail_session_stage", side_effect=Exception("tracking down")) as mock_fail, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", side_effect=Exception("strat broke")), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        assert result.strategist_error == "strat broke"
        stages_failed = [c.args[1] for c in mock_fail.call_args_list]
        assert "strategist" in stages_failed

    def test_executor_stage_failure_tracks_and_captures(self):
        with patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.fail_session"), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.fail_session_stage", side_effect=Exception("tracking down")) as mock_fail, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session", side_effect=Exception("trader broke")), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        assert result.trading_error == "trader broke"
        stages_failed = [c.args[1] for c in mock_fail.call_args_list]
        assert "executor" in stages_failed

    def test_strategy_stage_failure_tracks_and_captures(self):
        with patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.fail_session"), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.fail_session_stage", side_effect=Exception("tracking down")) as mock_fail, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection", side_effect=Exception("reflection broke")), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        assert result.strategy_error == "reflection broke"
        stages_failed = [c.args[1] for c in mock_fail.call_args_list]
        assert "strategy" in stages_failed

    def test_dashboard_stage_failure_tracks_and_captures(self):
        with patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.fail_session"), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.fail_session_stage", side_effect=Exception("tracking down")) as mock_fail, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage", side_effect=Exception("dash broke")):

            result = run_session(dry_run=False)

        assert result.dashboard_error == "dash broke"
        stages_failed = [c.args[1] for c in mock_fail.call_args_list]
        assert "dashboard" in stages_failed


    def test_executor_failure_without_session_id_takes_short_branch(self):
        """Executor stage raises AND insert_session_record failed (session_id=None).
        Covers the 'if session_id:' False branch in the executor except handler.
        """
        with patch("v2.session.insert_session_record", side_effect=Exception("cant track")), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session", side_effect=Exception("trader broke")), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        assert result.trading_error == "trader broke"


class TestFinalSessionStatusExceptionHandler:
    def test_fail_session_exception_is_swallowed(self):
        """A stage failed so run_session calls fail_session; fail_session itself raises.
        Covers the except Exception around fail_session/complete_session.
        """
        with patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.fail_session", side_effect=Exception("status db down")), \
             patch("v2.session.insert_session_stage"), \
             patch("v2.session.fail_session_stage"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline", side_effect=Exception("pipeline broke")), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        # fail_session itself threw — run_session must still return normally.
        assert result.has_errors


class TestDryRunSkipPromotion:
    """P1.12: --dry-run must skip every stage that would mutate strategy state
    or be visible to the outside world. Previously dry_run was forwarded only
    to the executor, so strategist/reflection still wrote DB rows and the
    Dashboard publisher still posted.
    """

    def _all_mocks(self):
        return [
            patch("v2.session.run_backfill"),
            patch("v2.session.compute_signal_attribution", return_value=[]),
            patch("v2.session.build_attribution_constraints", return_value=""),
            patch("v2.session.run_pipeline"),
            patch("v2.session.run_strategist_loop"),
            patch("v2.session.run_trading_session"),
            patch("v2.session.run_strategy_reflection"),
            patch("v2.session.run_dashboard_stage"),
        ]

    def test_dry_run_skips_strategist(self):
        ctxs = self._all_mocks()
        mocks = [c.start() for c in ctxs]
        try:
            result = run_session(dry_run=True)
            mock_strat = mocks[4]  # run_strategist_loop
            mock_strat.assert_not_called()
            assert result.skipped_ideation is True
        finally:
            for c in ctxs:
                c.stop()

    def test_dry_run_skips_reflection(self):
        ctxs = self._all_mocks()
        mocks = [c.start() for c in ctxs]
        try:
            result = run_session(dry_run=True)
            mock_refl = mocks[6]  # run_strategy_reflection
            mock_refl.assert_not_called()
            assert result.skipped_strategy is True
        finally:
            for c in ctxs:
                c.stop()

    def test_dry_run_skips_dashboard(self):
        ctxs = self._all_mocks()
        mocks = [c.start() for c in ctxs]
        try:
            result = run_session(dry_run=True)
            mock_dash = mocks[7]
            mock_dash.assert_not_called()
            assert result.skipped_dashboard is True
        finally:
            for c in ctxs:
                c.stop()

    def test_dry_run_still_runs_executor_in_dry_mode(self):
        """The executor still runs — but in dry mode (no order submission)."""
        ctxs = self._all_mocks()
        mocks = [c.start() for c in ctxs]
        try:
            run_session(dry_run=True)
            mock_trade = mocks[5]  # run_trading_session
            mock_trade.assert_called_once()
            assert mock_trade.call_args.kwargs.get("dry_run") is True
        finally:
            for c in ctxs:
                c.stop()

    def test_dry_run_still_runs_pipeline(self):
        """News pipeline observes new headlines — not a strategy mutation."""
        ctxs = self._all_mocks()
        mocks = [c.start() for c in ctxs]
        try:
            run_session(dry_run=True)
            mock_pipeline = mocks[3]
            mock_pipeline.assert_called_once()
        finally:
            for c in ctxs:
                c.stop()


class TestCliMain:
    def test_main_forwards_args_to_run_session(self):
        """CLI main() parses argv and invokes run_session with those kwargs."""
        import sys as _sys

        from v2.session import SessionResult, main

        argv = [
            "v2.session", "--dry-run", "--skip-pipeline",
            "--skip-dashboard",
        ]
        with patch.object(_sys, "argv", argv), \
             patch("v2.session.setup_logging"), \
             patch("v2.session.run_session", return_value=SessionResult()) as mock_run:

            main()

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["skip_pipeline"] is True

    def test_main_exits_nonzero_when_session_has_errors(self):
        """CLI main() exits 1 when the SessionResult has errors."""
        import sys as _sys

        from v2.session import SessionResult, main

        failed = SessionResult()
        failed.pipeline_error = "boom"
        with patch.object(_sys, "argv", ["v2.session"]), \
             patch("v2.session.setup_logging"), \
             patch("v2.session.run_session", return_value=failed):

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestStageCaptureUsage:
    def test_run_pipeline_stage_passes_usage_to_complete(self, mock_db, monkeypatch):
        """When the pipeline stage runs, the captured token usage should be
        forwarded to complete_session_stage."""
        from types import SimpleNamespace

        from v2 import session
        from v2.claude_client import _record_usage

        captured = {}

        def fake_complete(session_id, stage_name, usage=None):
            captured["stage"] = stage_name
            captured["model"] = usage.model if usage else None
            captured["input_tokens"] = usage.input_tokens if usage else 0

        monkeypatch.setattr(session, "complete_session_stage", fake_complete)
        monkeypatch.setattr(session, "insert_session_stage", lambda *a, **kw: None)

        def fake_pipeline(*args, **kwargs):
            _record_usage("claude-haiku-4-5-20251001", SimpleNamespace(
                input_tokens=500, output_tokens=200,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
            ))
            return MagicMock()

        monkeypatch.setattr(session, "run_pipeline", fake_pipeline)

        from v2.session import SessionResult, _run_pipeline_stage
        result = SessionResult()
        _run_pipeline_stage(result, session_id=1, completed_stages=set(),
                            skip=False, pipeline_hours=24, pipeline_limit=300)

        assert captured["stage"] == "pipeline"
        assert captured["model"] == "claude-haiku-4-5-20251001"
        assert captured["input_tokens"] == 500


class TestLogSessionCosts:
    def test_log_session_costs_emits_per_stage_breakdown(self, mock_db, mock_cursor, caplog):
        """The end-of-session cost summary should log one line per stage with a
        Claude call, and a Total: line."""
        from v2.session import _log_session_costs

        mock_cursor.fetchall.return_value = [
            {"stage_name": "pipeline",   "model": "claude-haiku-4-5-20251001",
             "cost_usd": 0.04, "input_tokens": 100, "output_tokens": 50,
             "cache_creation_tokens": 0, "cache_read_tokens": 0},
            {"stage_name": "strategist", "model": "claude-opus-4-7",
             "cost_usd": 1.28, "input_tokens": 5000, "output_tokens": 2000,
             "cache_creation_tokens": 100000, "cache_read_tokens": 500000},
            {"stage_name": "dashboard",  "model": None,
             "cost_usd": None, "input_tokens": None, "output_tokens": None,
             "cache_creation_tokens": None, "cache_read_tokens": None},
        ]

        with caplog.at_level("INFO", logger="session"):
            _log_session_costs(session_id=42)

        log_text = "\n".join(caplog.messages)
        assert "pipeline" in log_text
        assert "strategist" in log_text
        assert "$0.04" in log_text
        assert "$1.28" in log_text
        assert "$1.32" in log_text  # Total
        # Stages with NULL cost are omitted from the breakdown
        assert "dashboard" not in log_text


class TestSessionEndTelemetryLog:
    """`_finalize_session` should emit a `[telemetry]` summary line keyed
    on the live session_id so per-session tool-usage rolls up at one
    grep-able location."""

    def test_session_summary_line_called_with_session_id(self):
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.insert_session_record", return_value=77), \
             patch("v2.session.session_summary_line",
                   return_value="[telemetry] session=77 tools={}") as mock_summary:
            run_session(dry_run=False)

        mock_summary.assert_called_once_with(77)

    def test_session_summary_failure_does_not_break_session(self):
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.insert_session_record", return_value=77), \
             patch("v2.session.session_summary_line", side_effect=RuntimeError("DB down")):
            # Should not raise — telemetry never breaks a session.
            result = run_session(dry_run=False)

        assert result is not None


class TestExecutorStageErrors:
    def test_executor_result_errors_fail_stage(self):
        """A returned TradingSessionResult with errors is not a successful stage."""
        failed = TradingSessionResult(
            timestamp=datetime(2026, 5, 15, 10, 0, 0),
            account_snapshot_id=0,
            positions_synced=0,
            orders_synced=0,
            decisions_made=0,
            trades_executed=0,
            trades_failed=0,
            total_buy_value=Decimal("0"),
            total_sell_value=Decimal("0"),
            errors=["Market is closed -- skipped trading"],
        )

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session", return_value=failed), \
             patch("v2.session.insert_session_record", return_value=77), \
             patch("v2.session.complete_session_stage") as mock_complete, \
             patch("v2.session.fail_session_stage") as mock_fail:
            result = run_session(dry_run=False)

        assert "Market is closed" in result.trading_error
        completed = [c.args[1] for c in mock_complete.call_args_list]
        failed_stages = [c.args[1] for c in mock_fail.call_args_list]
        assert "executor" not in completed
        assert "executor" in failed_stages
