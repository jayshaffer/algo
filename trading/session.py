"""Consolidated daily session orchestrator.

Runs the full daily pipeline in a single invocation:
  1. News pipeline (fetch, filter, classify, store)
  2. Claude strategist (thesis management + playbook generation)
  3. Trading executor (decisions + order execution)

Each stage is independent — failures are captured and do not prevent
subsequent stages from running.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from .log_config import setup_logging
from .pipeline import PipelineStats, run_pipeline, check_dependencies
from .ideation_claude import ClaudeIdeationResult, run_strategist_loop
from .agent import DEFAULT_EXECUTOR_MODEL
from .trader import TradingSessionResult, run_trading_session
from .twitter import TwitterStageResult, run_twitter_stage

logger = logging.getLogger("session")


@dataclass
class SessionResult:
    """Combined result of a full daily session."""

    pipeline_result: Optional[PipelineStats] = None
    strategist_result: Optional[ClaudeIdeationResult] = None
    trading_result: Optional[TradingSessionResult] = None
    twitter_result: Optional[TwitterStageResult] = None

    pipeline_error: Optional[str] = None
    strategist_error: Optional[str] = None
    trading_error: Optional[str] = None
    twitter_error: Optional[str] = None

    skipped_pipeline: bool = False
    skipped_ideation: bool = False
    skipped_twitter: bool = False

    duration_seconds: float = 0.0

    @property
    def has_errors(self) -> bool:
        return any([self.pipeline_error, self.strategist_error, self.trading_error])


def run_session(
    dry_run: bool = False,
    model: str = "claude-opus-4-6",
    executor_model: str = DEFAULT_EXECUTOR_MODEL,
    max_turns: int = 25,
    skip_pipeline: bool = False,
    skip_ideation: bool = False,
    skip_twitter: bool = False,
    pipeline_hours: int = 24,
    pipeline_limit: int = 300,
) -> SessionResult:
    """
    Run the consolidated daily session.

    Stages:
      1. Dependency check (warn-only)
      2. News pipeline
      3. Claude strategist loop
      4. Trading session

    Each stage is wrapped in try/except. Failure at any stage does not
    prevent subsequent stages from running — they degrade gracefully
    using existing DB data.

    Args:
        dry_run: If True, don't execute real trades
        model: Claude model for strategist
        executor_model: Claude model for trading decisions
        max_turns: Max conversation turns for strategist
        skip_pipeline: Skip the news pipeline stage
        skip_ideation: Skip the strategist stage
        pipeline_hours: Hours of news to fetch
        pipeline_limit: Max news items to fetch

    Returns:
        SessionResult with per-stage results and errors
    """
    start = time.monotonic()
    result = SessionResult(
        skipped_pipeline=skip_pipeline,
        skipped_ideation=skip_ideation,
        skipped_twitter=skip_twitter,
    )

    logger.info(
        "Starting consolidated session (dry_run=%s, model=%s, executor_model=%s)",
        dry_run, model, executor_model,
    )

    # Stage 0: Dependency check (non-fatal)
    try:
        if not check_dependencies():
            logger.warning("Dependency check failed — continuing anyway")
    except Exception as e:
        logger.warning("Dependency check error: %s — continuing anyway", e)

    # Stage 1: News pipeline
    if skip_pipeline:
        logger.info("[Stage 1] News pipeline — SKIPPED")
    else:
        logger.info("[Stage 1] Running news pipeline")
        try:
            result.pipeline_result = run_pipeline(
                hours=pipeline_hours,
                limit=pipeline_limit,
            )
            logger.info(
                "Pipeline complete — fetched=%d, stored=%d ticker + %d macro",
                result.pipeline_result.news_fetched,
                result.pipeline_result.ticker_signals_stored,
                result.pipeline_result.macro_signals_stored,
            )
        except Exception as e:
            result.pipeline_error = str(e)
            logger.error("Pipeline failed: %s — continuing with existing signals", e)

    # Stage 2: Claude strategist
    if skip_ideation:
        logger.info("[Stage 2] Strategist — SKIPPED")
    else:
        logger.info("[Stage 2] Running Claude strategist")
        try:
            result.strategist_result = run_strategist_loop(
                model=model,
                max_turns=max_turns,
            )
            logger.info(
                "Strategist complete — turns=%d, created=%d, updated=%d, closed=%d",
                result.strategist_result.turns_used,
                result.strategist_result.theses_created,
                result.strategist_result.theses_updated,
                result.strategist_result.theses_closed,
            )
        except Exception as e:
            result.strategist_error = str(e)
            logger.error("Strategist failed: %s — continuing with existing playbook", e)

    # Stage 3: Trading session
    logger.info("[Stage 3] Running trading session")
    try:
        result.trading_result = run_trading_session(
            dry_run=dry_run,
            model=executor_model,
        )
        logger.info(
            "Trading complete — decisions=%d, executed=%d, failed=%d",
            result.trading_result.decisions_made,
            result.trading_result.trades_executed,
            result.trading_result.trades_failed,
        )
    except Exception as e:
        result.trading_error = str(e)
        logger.error("Trading session failed: %s", e)

    # Stage 4: Twitter (Bikini Bottom Capital)
    if skip_twitter:
        logger.info("[Stage 4] Twitter — SKIPPED")
    else:
        logger.info("[Stage 4] Posting to Twitter")
        try:
            result.twitter_result = run_twitter_stage()
        except Exception as e:
            result.twitter_error = str(e)
            logger.error("Twitter stage failed: %s", e)

    result.duration_seconds = time.monotonic() - start

    # Summary
    logger.info("=" * 60)
    logger.info("Session complete in %.1fs", result.duration_seconds)
    if result.has_errors:
        if result.pipeline_error:
            logger.error("  Pipeline error: %s", result.pipeline_error)
        if result.strategist_error:
            logger.error("  Strategist error: %s", result.strategist_error)
        if result.trading_error:
            logger.error("  Trading error: %s", result.trading_error)
    else:
        logger.info("  All stages completed successfully")
    if result.twitter_error:
        logger.warning("  Twitter error: %s", result.twitter_error)
    logger.info("=" * 60)

    return result


def main():
    """CLI entry point for consolidated daily session."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run consolidated daily trading session")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute real trades")
    parser.add_argument("--model", default="claude-opus-4-6", help="Claude model for strategist")
    parser.add_argument("--executor-model", default=DEFAULT_EXECUTOR_MODEL, help="Claude model for trading")
    parser.add_argument("--max-turns", type=int, default=25, help="Max strategist conversation turns")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip news pipeline stage")
    parser.add_argument("--skip-ideation", action="store_true", help="Skip strategist stage")
    parser.add_argument("--skip-twitter", action="store_true", help="Skip Twitter posting stage")
    parser.add_argument("--pipeline-hours", type=int, default=24, help="Hours of news to fetch")

    args = parser.parse_args()

    result = run_session(
        dry_run=args.dry_run,
        model=args.model,
        executor_model=args.executor_model,
        max_turns=args.max_turns,
        skip_pipeline=args.skip_pipeline,
        skip_ideation=args.skip_ideation,
        skip_twitter=args.skip_twitter,
        pipeline_hours=args.pipeline_hours,
    )

    if result.has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
