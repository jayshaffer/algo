"""Tests for 5-stage session orchestrator."""
from unittest.mock import patch, MagicMock
from datetime import date

from v2.session import run_session, SessionResult
from v2.strategy import StrategyReflectionResult
from v2.twitter import TwitterStageResult
from v2.bluesky import BlueskyStageResult
from v2.dashboard_publish import DashboardStageResult


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

            run_session(dry_run=True)

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

            run_session(dry_run=True)

        mock_strat.assert_called_once()
        assert "STRONG: earnings" in str(mock_strat.call_args)

    def test_stage_0_failure_does_not_block(self):
        """If learning refresh fails, session continues with stale data."""
        with patch("v2.session.run_backfill", side_effect=Exception("DB error")), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"):

            result = run_session(dry_run=True)

        assert result.learning_error is not None
        assert "DB error" in result.learning_error

    def test_skip_pipeline(self):
        """Pipeline should be skipped when flag is set."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"):

            result = run_session(dry_run=True, skip_pipeline=True)

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

            result = run_session(dry_run=True, skip_ideation=True)

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

            result = run_session(dry_run=True)

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
        assert result.skipped_pipeline is False
        assert result.skipped_ideation is False
        assert result.skipped_strategy is False
        assert result.bluesky_result is None
        assert result.bluesky_error is None
        assert result.skipped_bluesky is False
        assert result.dashboard_result is None
        assert result.dashboard_error is None
        assert result.skipped_dashboard is False
        assert result.duration_seconds == 0.0


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

            run_session(dry_run=True)

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

            result = run_session(dry_run=True)

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

            result = run_session(dry_run=True)

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

            result = run_session(dry_run=True, skip_strategy=True)

        mock_reflect.assert_not_called()
        assert result.skipped_strategy is True


class TestStage5Twitter:
    def test_stage_5_runs_after_strategy(self):
        """Twitter should run after strategy reflection."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection") as mock_reflect, \
             patch("v2.session.run_twitter_stage") as mock_twitter:

            mock_reflect.side_effect = lambda **kw: call_order.append("reflection")
            mock_twitter.side_effect = lambda: call_order.append("twitter")

            run_session(dry_run=True)

        assert call_order.index("reflection") < call_order.index("twitter")

    def test_stage_5_result_captured(self):
        """Twitter result should be in SessionResult."""
        mock_twitter_result = TwitterStageResult(tweet_posted=True)

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage", return_value=mock_twitter_result):

            result = run_session(dry_run=True)

        assert result.twitter_result is not None
        assert result.twitter_result.tweet_posted is True

    def test_stage_5_failure_does_not_block(self):
        """Twitter failure should be captured but not crash."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage", side_effect=Exception("Tweepy down")):

            result = run_session(dry_run=True)

        assert result.twitter_error == "Tweepy down"
        assert result.twitter_result is None

    def test_stage_5_error_in_has_errors(self):
        """Twitter error should be included in has_errors check."""
        result = SessionResult(twitter_error="test")
        assert result.has_errors is True

    def test_skip_twitter_flag(self):
        """Twitter should be skippable."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage") as mock_twitter:

            result = run_session(dry_run=True, skip_twitter=True)

        mock_twitter.assert_not_called()
        assert result.skipped_twitter is True


class TestStage5Bluesky:
    def test_bluesky_runs_after_twitter(self):
        """Bluesky should run after Twitter posting."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage") as mock_twitter, \
             patch("v2.session.run_bluesky_stage") as mock_bluesky:

            mock_twitter.side_effect = lambda: call_order.append("twitter")
            mock_bluesky.side_effect = lambda: call_order.append("bluesky")

            run_session(dry_run=True)

        assert call_order.index("twitter") < call_order.index("bluesky")

    def test_bluesky_result_captured(self):
        """Bluesky result should be in SessionResult."""
        mock_bluesky_result = BlueskyStageResult(post_posted=True)

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage", return_value=mock_bluesky_result):

            result = run_session(dry_run=True)

        assert result.bluesky_result is not None
        assert result.bluesky_result.post_posted is True

    def test_bluesky_failure_does_not_block(self):
        """Bluesky failure should be captured but not crash."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage", side_effect=Exception("atproto down")):

            result = run_session(dry_run=True)

        assert result.bluesky_error == "atproto down"
        assert result.bluesky_result is None

    def test_bluesky_error_in_has_errors(self):
        """Bluesky error should be included in has_errors check."""
        result = SessionResult(bluesky_error="test")
        assert result.has_errors is True

    def test_skip_bluesky_flag(self):
        """Bluesky should be skippable."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage") as mock_bluesky:

            result = run_session(dry_run=True, skip_bluesky=True)

        mock_bluesky.assert_not_called()
        assert result.skipped_bluesky is True


class TestStage6Dashboard:
    def test_dashboard_runs_after_bluesky(self):
        """Dashboard should run after Bluesky posting."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage") as mock_bluesky, \
             patch("v2.session.run_dashboard_stage") as mock_dashboard:

            mock_bluesky.side_effect = lambda: call_order.append("bluesky")
            mock_dashboard.side_effect = lambda: call_order.append("dashboard")

            run_session(dry_run=True)

        assert call_order.index("bluesky") < call_order.index("dashboard")

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage", return_value=mock_dashboard_result):

            result = run_session(dry_run=True)

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage", side_effect=Exception("gh-pages down")):

            result = run_session(dry_run=True)

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage") as mock_dashboard:

            result = run_session(dry_run=True, skip_dashboard=True)

        mock_dashboard.assert_not_called()
        assert result.skipped_dashboard is True


class TestSessionIdempotency:
    def test_blocks_duplicate_session(self):
        """Should refuse to run if a completed session exists for today."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline:

            mock_get.return_value = {"id": 1, "status": "completed"}

            result = run_session(dry_run=True)

        mock_pipeline.assert_not_called()
        assert result.has_errors or result.skipped_executor

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "failed"}

            result = run_session(dry_run=True)

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "completed"}

            result = run_session(dry_run=True, force=True)

        mock_pipeline.assert_called_once()


class TestPerStageResume:
    def test_resumes_skipping_completed_stages(self):
        """Re-run should skip stages that completed in a prior run."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.get_completed_stages", return_value={"pipeline", "strategist"}), \
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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "failed"}

            result = run_session(dry_run=True)

        # Pipeline and strategist were completed before — should be skipped
        mock_pipeline.assert_not_called()
        mock_strat.assert_not_called()
        # Executor was not completed — should run
        mock_trade.assert_called_once()

    def test_stage_tracking_calls_insert_and_complete(self):
        """Successful stage should call insert_session_stage then complete_session_stage."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.get_completed_stages", return_value=set()), \
             patch("v2.session.insert_session_stage") as mock_insert_stage, \
             patch("v2.session.complete_session_stage") as mock_complete_stage, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            run_session(dry_run=True)

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
             patch("v2.session.get_completed_stages", return_value=set()), \
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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

        assert result.pipeline_error == "fetch error"
        mock_fail_stage.assert_any_call(5, "pipeline", "fetch error")

    def test_stage_tracking_failure_does_not_break_session(self):
        """If insert_session_stage raises, the stage should still run."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=5), \
             patch("v2.session.complete_session"), \
             patch("v2.session.get_completed_stages", return_value=set()), \
             patch("v2.session.insert_session_stage", side_effect=Exception("DB down")), \
             patch("v2.session.complete_session_stage", side_effect=Exception("DB down")), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_strategy_memo") as mock_memo, \
             patch("v2.session.get_current_strategy_state", return_value={"id": 1}):

            run_session(dry_run=True)

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_strategy_memo") as mock_memo:

            run_session(dry_run=True)

        mock_memo.assert_not_called()


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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

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
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

        mock_trade.assert_called_once()
