"""Consolidated daily session orchestrator.

Runs the full daily pipeline in a single invocation:
  Stage 0: Learning refresh (backfill + attribution)
  Stage 1: News pipeline (fetch, classify, store)
  Stage 2: Claude strategist (thesis management + playbook generation)
  Stage 3: Trading executor (decisions + order execution)
  Stage 4: Strategy reflection (rules, identity, memos)
  Stage 5: Twitter posting (Mr. Krabs voice tweets)
  Stage 5b: Bluesky posting
  Stage 6: Public dashboard publish (GitHub Pages)

Each stage is independent — failures are captured and do not prevent
subsequent stages from running.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date

from .agent import DEFAULT_EXECUTOR_MODEL
from .attribution import build_attribution_constraints, compute_signal_attribution
from .backfill import run_backfill
from .bluesky import BlueskyStageResult, run_bluesky_stage
from .dashboard_publish import DashboardStageResult, run_dashboard_stage
from .database.trading_db import (
    complete_session,
    complete_session_stage,
    fail_session,
    fail_session_stage,
    get_completed_stages,
    get_current_strategy_state,
    get_playbook,
    get_session_for_date,
    insert_session_record,
    insert_session_stage,
    insert_strategy_memo,
)
from .ideation_claude import ClaudeIdeationResult, run_strategist_loop
from .log_config import setup_logging
from .pipeline import PipelineStats, run_pipeline
from .strategy import DEFAULT_REFLECTION_MODEL, StrategyReflectionResult, run_strategy_reflection
from .trader import TradingSessionResult, run_trading_session
from .twitter import TwitterStageResult, run_twitter_stage

logger = logging.getLogger("session")


_ERROR_FIELDS = (
    "learning_error", "pipeline_error", "strategist_error", "trading_error",
    "strategy_error", "twitter_error", "bluesky_error", "dashboard_error",
)


@dataclass
class SessionResult:
    pipeline_result: PipelineStats | None = None
    strategist_result: ClaudeIdeationResult | None = None
    trading_result: TradingSessionResult | None = None
    strategy_result: StrategyReflectionResult | None = None
    twitter_result: TwitterStageResult | None = None
    bluesky_result: BlueskyStageResult | None = None
    dashboard_result: DashboardStageResult | None = None

    learning_error: str | None = None     # V3: Stage 0
    pipeline_error: str | None = None
    strategist_error: str | None = None
    trading_error: str | None = None
    strategy_error: str | None = None
    twitter_error: str | None = None
    bluesky_error: str | None = None
    dashboard_error: str | None = None

    skipped_pipeline: bool = False
    skipped_ideation: bool = False
    skipped_executor: bool = False
    skipped_strategy: bool = False
    skipped_twitter: bool = False
    skipped_bluesky: bool = False
    skipped_dashboard: bool = False
    duration_seconds: float = 0.0

    @property
    def has_errors(self) -> bool:
        return any(getattr(self, f) for f in _ERROR_FIELDS)


def _start_stage(session_id: int | None, stage: str) -> None:
    if session_id is None:
        return
    try:
        insert_session_stage(session_id, stage)
    except Exception:
        pass


def _complete_stage(session_id: int | None, stage: str) -> None:
    if session_id is None:
        return
    try:
        complete_session_stage(session_id, stage)
    except Exception:
        pass


def _fail_stage(session_id: int | None, stage: str, error: str) -> None:
    if session_id is None:
        return
    try:
        fail_session_stage(session_id, stage, error)
    except Exception:
        pass


def _check_and_record_session(force: bool, session_date) -> tuple[int | None, set, str | None]:
    """Returns (session_id, completed_stages, early_error).

    early_error is non-None when the caller should return immediately
    (e.g., session already completed and force=False).
    """
    session_id: int | None = None
    completed_stages: set = set()
    if not force:
        try:
            existing = get_session_for_date(session_date)
            if existing and existing["status"] == "completed":
                logger.warning("Session already completed for %s. Use --force to override.", session_date)
                return None, set(), f"Session already completed for {session_date}"
            if existing:
                completed_stages = get_completed_stages(existing["id"])
                if completed_stages:
                    logger.info("Resuming session — already completed: %s", completed_stages)
        except Exception as e:
            logger.warning("Could not check session status: %s — proceeding", e)
    try:
        session_id = insert_session_record(session_date)
        logger.info("Session ID: %d", session_id)
    except Exception as e:
        logger.warning("Could not create session record: %s — proceeding without tracking", e)
    return session_id, completed_stages, None


def _run_learning_refresh(result: SessionResult) -> str:
    """Stage 0 — returns attribution_constraints (possibly empty)."""
    logger.info("[Stage 0] Refreshing learning data")
    try:
        run_backfill()
        compute_signal_attribution()
        constraints = build_attribution_constraints()
        logger.info("Learning refresh complete")
        return constraints
    except Exception as e:
        result.learning_error = str(e)
        logger.warning("Learning refresh failed: %s — continuing with stale data", e)
        return ""


def _run_pipeline_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
    pipeline_hours: int,
    pipeline_limit: int,
) -> None:
    if skip or "pipeline" in completed_stages:
        logger.info("[Stage 1] News pipeline — SKIPPED%s",
                    " (completed in prior run)" if "pipeline" in completed_stages else "")
        return
    logger.info("[Stage 1] Running news pipeline")
    _start_stage(session_id, "pipeline")
    try:
        result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit)
        _complete_stage(session_id, "pipeline")
    except Exception as e:
        result.pipeline_error = str(e)
        _fail_stage(session_id, "pipeline", str(e))
        logger.error("Pipeline failed: %s — continuing with existing signals", e)


def _persist_strategist_memo(result: SessionResult, session_date) -> None:
    try:
        if result.strategist_result and result.strategist_result.final_summary:
            state = get_current_strategy_state()
            insert_strategy_memo(
                session_date=session_date,
                memo_type='strategist_notes',
                content=result.strategist_result.final_summary,
                strategy_state_id=state['id'] if state else None,
            )
            logger.info("Strategist summary saved as memo")
    except Exception as e:
        logger.warning("Could not save strategist memo: %s", e)


def _run_strategist_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
    model: str,
    max_turns: int,
    attribution_constraints: str,
    session_date,
) -> None:
    if skip or "strategist" in completed_stages:
        logger.info("[Stage 2] Strategist — SKIPPED%s",
                    " (completed in prior run)" if "strategist" in completed_stages else "")
        return
    logger.info("[Stage 2] Running Claude strategist")
    _start_stage(session_id, "strategist")
    try:
        result.strategist_result = run_strategist_loop(
            model=model,
            max_turns=max_turns,
            attribution_constraints=attribution_constraints,
        )
        # P2.24: validate playbook BEFORE persisting the memo. The previous
        # order committed the memo first, then raised on missing playbook;
        # the failure path didn't complete the stage, so the next run re-ran
        # the strategist and inserted a *second* memo for the same date.
        # Validate first → fail fast → no memo on failure → no duplicates.
        if get_playbook(session_date) is None:
            raise RuntimeError(
                f"Strategist finished without writing a playbook for {session_date} "
                "(likely hit max_tokens or max_turns before calling write_playbook)"
            )
        _persist_strategist_memo(result, session_date)
        _complete_stage(session_id, "strategist")
    except Exception as e:
        result.strategist_error = str(e)
        _fail_stage(session_id, "strategist", str(e))
        logger.error("Strategist failed: %s — continuing with existing playbook", e)


def _run_executor_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
    dry_run: bool,
    executor_model: str,
    session_date,
) -> None:
    # P2.23: short-circuit on missing playbook regardless of strategist_error.
    # The previous condition required `result.strategist_error` to flip to
    # skipped — but on a resume run where strategist was completed in a prior
    # invocation, strategist_error is None even if the playbook was manually
    # deleted. The executor would then try to read get_pending_playbook_actions
    # against a None row and crash with TypeError. The truthy check below is
    # the right contract: "if no playbook for today, skip the executor."
    if (
        not skip
        and "executor" not in completed_stages
        and get_playbook(session_date) is None
    ):
        if result.strategist_error:
            logger.warning("Strategist failed and no playbook exists for %s — skipping executor", session_date)
        else:
            logger.warning("No playbook exists for %s (resume + manual cleanup?) — skipping executor", session_date)
        skip = True
        result.skipped_executor = True

    if skip or "executor" in completed_stages:
        logger.info("[Stage 3] Trading executor — SKIPPED%s",
                    " (completed in prior run)" if "executor" in completed_stages else "")
        return

    logger.info("[Stage 3] Running trading session")
    _start_stage(session_id, "executor")
    try:
        result.trading_result = run_trading_session(dry_run=dry_run, model=executor_model)
        _complete_stage(session_id, "executor")
    except Exception as e:
        result.trading_error = str(e)
        _fail_stage(session_id, "executor", str(e))
        logger.error("Trading session failed: %s", e)


def _run_strategy_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
) -> None:
    if skip or "strategy" in completed_stages:
        logger.info("[Stage 4] Strategy reflection — SKIPPED%s",
                    " (completed in prior run)" if "strategy" in completed_stages else "")
        result.skipped_strategy = True
        return
    logger.info("[Stage 4] Running strategy reflection")
    _start_stage(session_id, "strategy")
    try:
        result.strategy_result = run_strategy_reflection(
            model=DEFAULT_REFLECTION_MODEL,
            max_turns=10,
            trading_result=result.trading_result,
        )
        _complete_stage(session_id, "strategy")
    except Exception as e:
        result.strategy_error = str(e)
        _fail_stage(session_id, "strategy", str(e))
        logger.error("Strategy reflection failed: %s", e)


def _run_twitter_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set, skip: bool,
) -> None:
    if skip or "twitter" in completed_stages:
        logger.info("[Stage 5] Twitter posting — SKIPPED%s",
                    " (completed in prior run)" if "twitter" in completed_stages else "")
        result.skipped_twitter = True
        return
    logger.info("[Stage 5] Running Twitter posting")
    _start_stage(session_id, "twitter")
    try:
        result.twitter_result = run_twitter_stage()
        _complete_stage(session_id, "twitter")
    except Exception as e:
        result.twitter_error = str(e)
        _fail_stage(session_id, "twitter", str(e))
        logger.error("Twitter stage failed: %s", e)


def _run_bluesky_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set, skip: bool,
) -> None:
    if skip or "bluesky" in completed_stages:
        logger.info("[Stage 5b] Bluesky posting — SKIPPED%s",
                    " (completed in prior run)" if "bluesky" in completed_stages else "")
        result.skipped_bluesky = True
        return
    logger.info("[Stage 5b] Running Bluesky posting")
    _start_stage(session_id, "bluesky")
    try:
        result.bluesky_result = run_bluesky_stage()
        _complete_stage(session_id, "bluesky")
    except Exception as e:
        result.bluesky_error = str(e)
        _fail_stage(session_id, "bluesky", str(e))
        logger.error("Bluesky stage failed: %s", e)


def _run_dashboard_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set, skip: bool,
) -> None:
    if skip or "dashboard" in completed_stages:
        logger.info("[Stage 6] Dashboard publish — SKIPPED%s",
                    " (completed in prior run)" if "dashboard" in completed_stages else "")
        result.skipped_dashboard = True
        return
    logger.info("[Stage 6] Publishing public dashboard")
    _start_stage(session_id, "dashboard")
    try:
        result.dashboard_result = run_dashboard_stage()
        _complete_stage(session_id, "dashboard")
    except Exception as e:
        result.dashboard_error = str(e)
        _fail_stage(session_id, "dashboard", str(e))
        logger.error("Dashboard publish failed: %s", e)


def _finalize_session(result: SessionResult, session_id: int | None) -> None:
    if session_id:
        try:
            if result.has_errors:
                error_summary = "; ".join(
                    str(getattr(result, f)) for f in _ERROR_FIELDS if getattr(result, f)
                )
                fail_session(session_id, error_summary)
            else:
                complete_session(session_id)
        except Exception as e:
            logger.warning("Could not update session status: %s", e)

    logger.info("=" * 60)
    logger.info("Session complete in %.1fs", result.duration_seconds)
    if result.has_errors:
        for field_name in _ERROR_FIELDS:
            err = getattr(result, field_name)
            if err:
                logger.error("  %s: %s", field_name, err)
    else:
        logger.info("  All stages completed successfully")
    logger.info("=" * 60)


def run_session(
    dry_run: bool = False,
    model: str = "claude-opus-4-6",
    executor_model: str = DEFAULT_EXECUTOR_MODEL,
    max_turns: int = 25,
    skip_pipeline: bool = False,
    skip_ideation: bool = False,
    skip_executor: bool = False,
    skip_strategy: bool = False,
    skip_twitter: bool = False,
    skip_bluesky: bool = False,
    skip_dashboard: bool = False,
    pipeline_hours: int = 24,
    pipeline_limit: int = 300,
    force: bool = False,
) -> SessionResult:
    start = time.monotonic()

    # P1.12: --dry-run was misleading — it only gated the executor's order
    # submission, leaving the strategist (writes theses/playbooks/memos) and
    # reflection (writes rules/memos/identity) and the social/dashboard
    # publishers free to mutate state and post publicly. Promote dry_run to
    # the skip flags for any stage that would otherwise mutate strategy state
    # or be visible to the outside world. Pipeline is left running (it just
    # observes new news; not a strategy mutation).
    if dry_run:
        skip_ideation = True
        skip_strategy = True
        skip_twitter = True
        skip_bluesky = True
        skip_dashboard = True

    result = SessionResult(
        skipped_pipeline=skip_pipeline, skipped_ideation=skip_ideation,
        skipped_executor=skip_executor, skipped_strategy=skip_strategy,
        skipped_twitter=skip_twitter, skipped_bluesky=skip_bluesky,
        skipped_dashboard=skip_dashboard,
    )
    today = date.today()

    session_id, completed_stages, early_error = _check_and_record_session(force, today)
    if early_error:
        result.learning_error = early_error
        result.duration_seconds = time.monotonic() - start
        return result

    attribution_constraints = _run_learning_refresh(result)
    _run_pipeline_stage(result, session_id, completed_stages, skip_pipeline, pipeline_hours, pipeline_limit)
    _run_strategist_stage(
        result, session_id, completed_stages, skip_ideation,
        model, max_turns, attribution_constraints, today,
    )
    _run_executor_stage(result, session_id, completed_stages, skip_executor, dry_run, executor_model, today)
    _run_strategy_stage(result, session_id, completed_stages, skip_strategy)
    _run_twitter_stage_wrapper(result, session_id, completed_stages, skip_twitter)
    _run_bluesky_stage_wrapper(result, session_id, completed_stages, skip_bluesky)
    _run_dashboard_stage_wrapper(result, session_id, completed_stages, skip_dashboard)

    result.duration_seconds = time.monotonic() - start
    _finalize_session(result, session_id)
    return result


def main():
    """CLI entry point for consolidated daily session."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run consolidated daily trading session")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--executor-model", default=DEFAULT_EXECUTOR_MODEL)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-ideation", action="store_true")
    parser.add_argument("--skip-executor", action="store_true")
    parser.add_argument("--skip-strategy", action="store_true")
    parser.add_argument("--skip-twitter", action="store_true")
    parser.add_argument("--skip-bluesky", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--pipeline-hours", type=int, default=24)
    parser.add_argument("--force", action="store_true", help="Override session idempotency check")

    args = parser.parse_args()
    result = run_session(
        dry_run=args.dry_run, model=args.model, executor_model=args.executor_model,
        max_turns=args.max_turns, skip_pipeline=args.skip_pipeline,
        skip_ideation=args.skip_ideation, skip_executor=args.skip_executor,
        skip_strategy=args.skip_strategy,
        skip_twitter=args.skip_twitter, skip_bluesky=args.skip_bluesky,
        skip_dashboard=args.skip_dashboard,
        pipeline_hours=args.pipeline_hours,
        force=args.force,
    )
    if result.has_errors:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
