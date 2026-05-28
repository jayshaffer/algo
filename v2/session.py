"""Consolidated daily session orchestrator.

Runs the full daily pipeline in a single invocation:
  Stage 0: Learning refresh (backfill + attribution)
  Stage 1: News pipeline (fetch, classify, store)
  Stage 2: Claude strategist (thesis management + playbook generation)
  Stage 3: Trading executor (decisions + order execution)
  Stage 4: Strategy reflection (rules, identity, memos)
  Stage 5: Public dashboard publish

Each stage is independent — failures are captured and do not prevent
subsequent stages from running.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .agent import DEFAULT_EXECUTOR_MODEL
from .attribution import build_attribution_constraints, compute_signal_attribution
from .backfill import run_backfill
from .claude_client import capture_usage
from .dashboard_publish import DashboardStageResult, run_dashboard_stage
from .database.trading_db import (
    close_orphan_running_stages,
    complete_session,
    complete_session_stage,
    expire_stale_playbook_actions,
    fail_session,
    fail_session_stage,
    get_current_strategy_state,
    get_playbook,
    get_session_for_date,
    insert_session_record,
    insert_session_stage,
    insert_strategy_memo,
    mark_session_market_closed,
)
from .ideation_claude import ClaudeIdeationResult, run_strategist_loop
from .log_config import setup_logging
from .pipeline import PipelineStats, run_pipeline
from .strategy import DEFAULT_REFLECTION_MODEL, StrategyReflectionResult, run_strategy_reflection
from .supervisor import run_supervisor
from .telemetry import session_summary_line
from .trader import TradingSessionResult, run_trading_session

logger = logging.getLogger("session")


MARKET_TIMEZONE = ZoneInfo("America/New_York")


def current_market_date():
    return datetime.now(MARKET_TIMEZONE).date()


_ERROR_FIELDS = (
    "learning_error", "supervisor_error", "pipeline_error", "strategist_error",
    "trading_error", "strategy_error", "dashboard_error",
)


@dataclass
class SessionResult:
    pipeline_result: PipelineStats | None = None
    strategist_result: ClaudeIdeationResult | None = None
    trading_result: TradingSessionResult | None = None
    strategy_result: StrategyReflectionResult | None = None
    dashboard_result: DashboardStageResult | None = None

    learning_error: str | None = None     # V3: Stage 0
    supervisor_error: str | None = None   # Stage 0.5
    pipeline_error: str | None = None
    strategist_error: str | None = None
    trading_error: str | None = None
    strategy_error: str | None = None
    dashboard_error: str | None = None

    supervisor_memo_id: int | None = None  # Stage 0.5
    skipped_supervisor: bool = False

    # T2.1: distinct from errors — set when the session was already completed
    # for the day and force=False. The caller exits 0 with a clear log line
    # rather than treating "nothing to do" as a failure.
    idempotent_skip: str | None = None

    skipped_pipeline: bool = False
    skipped_ideation: bool = False
    skipped_executor: bool = False
    skipped_strategy: bool = False
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


def _complete_stage(session_id: int | None, stage: str, usage=None) -> None:
    if session_id is None:
        return
    try:
        complete_session_stage(session_id, stage, usage=usage)
    except Exception:
        pass


def _fail_stage(session_id: int | None, stage: str, error: str, usage=None) -> None:
    if session_id is None:
        return
    try:
        fail_session_stage(session_id, stage, error, usage=usage)
    except Exception:
        pass


def _log_session_costs(session_id: int | None) -> None:
    """Emit a per-stage cost breakdown to the session logger.

    Reads from the session_stage_costs view (defined in
    db/init/024_session_stage_token_usage.sql). Stages with NULL cost
    (no Claude calls or unseeded model) are omitted from the breakdown
    but contribute NULL to the SUM (i.e. nothing).
    """
    if session_id is None:
        return
    try:
        from v2.database.trading_db import get_cursor
        with get_cursor() as cur:
            cur.execute("""
                SELECT stage_name, model, cost_usd
                FROM session_stage_costs
                WHERE session_id = %s
                ORDER BY id
            """, (session_id,))
            rows = cur.fetchall()
    except Exception as e:
        logger.warning("Could not load session costs: %s", e)
        return

    priced = [r for r in rows if r["cost_usd"] is not None]
    if not priced:
        return

    logger.info("Stage costs (USD):")
    for r in priced:
        logger.info("  %-12s $%.4f  (%s)", r["stage_name"] + ":", float(r["cost_usd"]), r["model"])
    total = sum(float(r["cost_usd"]) for r in priced)
    logger.info("  Total: $%.4f", total)


def _check_and_record_session(force: bool, session_date) -> tuple[int | None, set, str | None]:
    """Returns (session_id, completed_stages, early_error).

    Per-run sessions: every invocation creates a new sessions row.
    completed_stages is always the empty set — no resume across runs.

    Idempotency: if force=False and a session of session_type='daily'
    is already 'completed' for this date, we skip with early_error set.
    --force bypasses that gate.
    """
    if not force:
        try:
            existing = get_session_for_date(session_date)
            if existing and existing["status"] == "completed":
                logger.warning("Session already completed for %s. Use --force to override.", session_date)
                return None, set(), f"Session already completed for {session_date}"
        except Exception as e:
            logger.warning("Could not check session status: %s — proceeding", e)
    try:
        session_id = insert_session_record(session_date)
        logger.info("Session ID: %d", session_id)
        return session_id, set(), None
    except Exception as e:
        logger.warning("Could not create session record: %s — proceeding without tracking", e)
        return None, set(), None


def _run_learning_refresh(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
) -> str:
    """Stage 0 — returns attribution_constraints (possibly empty).

    T2.2: tracked as a session_stages row so resumes skip a completed
    learning refresh and the dashboard can see learning success/failure
    alongside other stages.
    """
    if "learning" in completed_stages:
        logger.info("[Stage 0] Learning refresh — SKIPPED (completed in prior run)")
        # Recompute constraints from the existing attribution table; it's
        # cheap and gives the strategist the same context it would have
        # had on the original run.
        return build_attribution_constraints()
    logger.info("[Stage 0] Refreshing learning data")
    _start_stage(session_id, "learning")
    with capture_usage() as usage:
        try:
            run_backfill()
            compute_signal_attribution()
            constraints = build_attribution_constraints()
            _complete_stage(session_id, "learning", usage=usage)
            logger.info("Learning refresh complete")
            return constraints
        except Exception as e:
            result.learning_error = str(e)
            _fail_stage(session_id, "learning", str(e), usage=usage)
            logger.warning("Learning refresh failed: %s — continuing with stale data", e)
            return ""


def _run_supervisor_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
) -> None:
    """Stage 0.5 — observer-only critic that records the watchlist the acting
    stages must resolve. Runs after the learning refresh (fresh attribution)
    and before ideation. Independent: a failure here does not abort the session;
    acting stages then resolve any items still open from a prior run.

    Cost is recorded in supervisor_memos by run_supervisor itself. We do not
    wrap this in capture_usage: run_supervisor's own capture_usage swaps the
    active accumulator, so an outer one would capture nothing.
    """
    if skip or "supervisor" in completed_stages:
        logger.info(
            "[Stage 0.5] Supervisor — SKIPPED%s",
            " (completed in prior run)" if "supervisor" in completed_stages else "",
        )
        result.skipped_supervisor = True
        return
    logger.info("[Stage 0.5] Running strategy supervisor")
    _start_stage(session_id, "supervisor")
    try:
        result.supervisor_memo_id = run_supervisor()
        _complete_stage(session_id, "supervisor")
    except Exception as e:
        result.supervisor_error = str(e)
        _fail_stage(session_id, "supervisor", str(e))
        logger.error(
            "Supervisor failed: %s — continuing; acting stages use existing watchlist", e
        )


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
    with capture_usage() as usage:
        try:
            result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit, session_id=session_id)
            _complete_stage(session_id, "pipeline", usage=usage)
        except Exception as e:
            result.pipeline_error = str(e)
            _fail_stage(session_id, "pipeline", str(e), usage=usage)
            logger.error("Pipeline failed: %s — continuing with existing signals", e)


STRATEGIST_MEMO_MIN_LENGTH = 40


def _persist_strategist_memo(result: SessionResult, session_date, session_id: int | None = None) -> None:
    try:
        if not (result.strategist_result and result.strategist_result.final_summary):
            return
        summary = result.strategist_result.final_summary.strip()
        # Reject trivially-short summaries ("all good", "done", "ok") so the
        # journal isn't polluted with placeholder rows. ALGO-15: 92% of recent
        # strategist_notes had collapsed to the literal "all good" (8 chars),
        # erasing run-to-run continuity. If the model didn't produce a real
        # summary, recording that is more honest than recording the stub.
        if len(summary) < STRATEGIST_MEMO_MIN_LENGTH:
            logger.warning(
                "Strategist memo skipped: final_summary too short (%d chars): %r",
                len(summary), summary,
            )
            return
        state = get_current_strategy_state()
        insert_strategy_memo(
            session_date=session_date,
            memo_type='strategist_notes',
            content=summary,
            strategy_state_id=state['id'] if state else None,
            session_id=session_id,
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
    with capture_usage() as usage:
        try:
            result.strategist_result = run_strategist_loop(
                model=model,
                max_turns=max_turns,
                attribution_constraints=attribution_constraints,
                session_id=session_id,
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
            _persist_strategist_memo(result, session_date, session_id=session_id)
            _complete_stage(session_id, "strategist", usage=usage)
        except Exception as e:
            result.strategist_error = str(e)
            _fail_stage(session_id, "strategist", str(e), usage=usage)
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
    with capture_usage() as usage:
        try:
            result.trading_result = run_trading_session(
                dry_run=dry_run,
                model=executor_model,
                session_id=session_id,
                session_date=session_date,
            )
            if result.trading_result.market_closed:
                if session_id is not None:
                    try:
                        mark_session_market_closed(session_id)
                    except Exception as e:
                        logger.warning("Could not mark session market_closed: %s", e)
                _complete_stage(session_id, "executor", usage=usage)
                return
            if result.trading_result.errors:
                raise RuntimeError("; ".join(result.trading_result.errors))
            _complete_stage(session_id, "executor", usage=usage)
        except Exception as e:
            result.trading_error = str(e)
            _fail_stage(session_id, "executor", str(e), usage=usage)
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
    with capture_usage() as usage:
        try:
            result.strategy_result = run_strategy_reflection(
                model=DEFAULT_REFLECTION_MODEL,
                max_turns=10,
                trading_result=result.trading_result,
                session_id=session_id,
            )
            # P1.15: parallel to P2.24 for the strategist — the reflection LLM is
            # instructed to "always write a memo" but nothing structurally enforced
            # it. Silent omissions left journal gaps and, worse, marked the stage
            # complete so resume would skip the reflection on the next run instead
            # of retrying. Treat a missing memo as a stage failure: the stage stays
            # incomplete and the next session re-runs reflection.
            if not result.strategy_result.memo_written:
                raise RuntimeError(
                    "Reflection finished without writing a memo "
                    "(LLM did not call write_strategy_memo)"
                )
            _complete_stage(session_id, "strategy", usage=usage)
        except Exception as e:
            result.strategy_error = str(e)
            _fail_stage(session_id, "strategy", str(e), usage=usage)
            logger.error("Strategy reflection failed: %s", e)


def _run_dashboard_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set, skip: bool,
) -> None:
    if skip or "dashboard" in completed_stages:
        logger.info("[Stage 5] Dashboard publish — SKIPPED%s",
                    " (completed in prior run)" if "dashboard" in completed_stages else "")
        result.skipped_dashboard = True
        return
    logger.info("[Stage 5] Publishing public dashboard")
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
            swept = close_orphan_running_stages(session_id)
            if swept:
                logger.warning("Closed orphan running stages: %s", swept)
        except Exception as e:
            logger.warning("Could not sweep orphan stages: %s", e)
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

    _log_session_costs(session_id)
    if session_id:
        try:
            logger.info(session_summary_line(session_id))
        except Exception:
            logger.exception("session_summary_line failed; continuing")
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
    model: str = "claude-opus-4-8",
    executor_model: str = DEFAULT_EXECUTOR_MODEL,
    max_turns: int = 25,
    skip_supervisor: bool = False,
    skip_pipeline: bool = False,
    skip_ideation: bool = False,
    skip_executor: bool = False,
    skip_strategy: bool = False,
    skip_dashboard: bool = False,
    pipeline_hours: int = 24,
    pipeline_limit: int = 300,
    force: bool = False,
) -> SessionResult:
    start = time.monotonic()

    # P1.12: --dry-run was misleading — it only gated the executor's order
    # submission, leaving the strategist (writes theses/playbooks/memos) and
    # reflection (writes rules/memos/identity) and the dashboard publisher
    # free to mutate state and publish publicly. Promote dry_run to the skip
    # flags for any stage that would otherwise mutate strategy state or be
    # visible to the outside world. Pipeline is left running (it just
    # observes new news; not a strategy mutation).
    if dry_run:
        skip_ideation = True
        skip_strategy = True
        skip_dashboard = True

    result = SessionResult(
        skipped_supervisor=skip_supervisor,
        skipped_pipeline=skip_pipeline, skipped_ideation=skip_ideation,
        skipped_executor=skip_executor, skipped_strategy=skip_strategy,
        skipped_dashboard=skip_dashboard,
    )
    today = current_market_date()

    try:
        expired = expire_stale_playbook_actions(today)
        if expired:
            logger.info("Expired %d stale pending playbook_actions from prior days", expired)
    except Exception as e:
        logger.warning("expire_stale_playbook_actions failed (non-fatal): %s", e)

    session_id, completed_stages, early_error = _check_and_record_session(force, today)
    if early_error:
        # T2.1: idempotent skip is not a failure. main() reads this field
        # and exits 0 with a "nothing to do" log line.
        result.idempotent_skip = early_error
        result.duration_seconds = time.monotonic() - start
        return result

    try:
        attribution_constraints = _run_learning_refresh(result, session_id, completed_stages)
        _run_supervisor_stage(result, session_id, completed_stages, skip_supervisor)
        _run_pipeline_stage(result, session_id, completed_stages, skip_pipeline, pipeline_hours, pipeline_limit)
        _run_strategist_stage(
            result, session_id, completed_stages, skip_ideation,
            model, max_turns, attribution_constraints, today,
        )
        _run_executor_stage(result, session_id, completed_stages, skip_executor, dry_run, executor_model, today)
        _run_strategy_stage(result, session_id, completed_stages, skip_strategy)
        _run_dashboard_stage_wrapper(result, session_id, completed_stages, skip_dashboard)
    finally:
        result.duration_seconds = time.monotonic() - start
        _finalize_session(result, session_id)
    return result


def main():
    """CLI entry point for consolidated daily session."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run consolidated daily trading session")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--executor-model", default=DEFAULT_EXECUTOR_MODEL)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--skip-supervisor", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-ideation", action="store_true")
    parser.add_argument("--skip-executor", action="store_true")
    parser.add_argument("--skip-strategy", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--pipeline-hours", type=int, default=24)
    parser.add_argument("--force", action="store_true", help="Override session idempotency check")

    args = parser.parse_args()
    result = run_session(
        dry_run=args.dry_run, model=args.model, executor_model=args.executor_model,
        max_turns=args.max_turns, skip_supervisor=args.skip_supervisor,
        skip_pipeline=args.skip_pipeline,
        skip_ideation=args.skip_ideation, skip_executor=args.skip_executor,
        skip_strategy=args.skip_strategy,
        skip_dashboard=args.skip_dashboard,
        pipeline_hours=args.pipeline_hours,
        force=args.force,
    )
    if result.idempotent_skip:
        # T2.1: nothing to do, not a failure. Exit 0 so cron/Taskfile
        # callers don't alert on a re-run.
        logger.info("Session is a no-op: %s", result.idempotent_skip)
        sys.exit(0)
    if result.has_errors:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
