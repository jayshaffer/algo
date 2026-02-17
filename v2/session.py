"""Consolidated daily session orchestrator.

Runs the full daily pipeline in a single invocation:
  Stage 0: Learning refresh (backfill + attribution)
  Stage 1: News pipeline (fetch, classify, store)
  Stage 2: Claude strategist (thesis management + playbook generation)
  Stage 3: Trading executor (decisions + order execution)
  Stage 4: Strategy reflection (rules, identity, memos)
  Stage 5: Twitter posting (Mr. Krabs voice tweets)

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
from .backfill import run_backfill
from .attribution import compute_signal_attribution, build_attribution_constraints
from .pipeline import PipelineStats, run_pipeline
from .ideation_claude import ClaudeIdeationResult, run_strategist_loop
from .agent import DEFAULT_EXECUTOR_MODEL
from .trader import TradingSessionResult, run_trading_session
from .strategy import StrategyReflectionResult, run_strategy_reflection
from .twitter import TwitterStageResult, run_twitter_stage

logger = logging.getLogger("session")


@dataclass
class SessionResult:
    pipeline_result: Optional[PipelineStats] = None
    strategist_result: Optional[ClaudeIdeationResult] = None
    trading_result: Optional[TradingSessionResult] = None
    strategy_result: Optional[StrategyReflectionResult] = None
    twitter_result: Optional[TwitterStageResult] = None

    learning_error: Optional[str] = None     # V3: Stage 0
    pipeline_error: Optional[str] = None
    strategist_error: Optional[str] = None
    trading_error: Optional[str] = None
    strategy_error: Optional[str] = None
    twitter_error: Optional[str] = None

    skipped_pipeline: bool = False
    skipped_ideation: bool = False
    skipped_executor: bool = False
    skipped_strategy: bool = False
    skipped_twitter: bool = False
    duration_seconds: float = 0.0

    @property
    def has_errors(self) -> bool:
        return any([self.learning_error, self.pipeline_error,
                    self.strategist_error, self.trading_error,
                    self.strategy_error, self.twitter_error])


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
    pipeline_hours: int = 24,
    pipeline_limit: int = 300,
) -> SessionResult:
    start = time.monotonic()
    result = SessionResult(skipped_pipeline=skip_pipeline, skipped_ideation=skip_ideation, skipped_executor=skip_executor, skipped_strategy=skip_strategy, skipped_twitter=skip_twitter)

    # Stage 0: Refresh learning data
    attribution_constraints = ""
    logger.info("[Stage 0] Refreshing learning data")
    try:
        run_backfill()
        compute_signal_attribution()
        attribution_constraints = build_attribution_constraints()
        logger.info("Learning refresh complete")
    except Exception as e:
        result.learning_error = str(e)
        logger.warning("Learning refresh failed: %s — continuing with stale data", e)

    # Stage 1: News pipeline
    if skip_pipeline:
        logger.info("[Stage 1] News pipeline — SKIPPED")
    else:
        logger.info("[Stage 1] Running news pipeline")
        try:
            result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit)
        except Exception as e:
            result.pipeline_error = str(e)
            logger.error("Pipeline failed: %s — continuing with existing signals", e)

    # Stage 2: Claude strategist (receives attribution constraints)
    if skip_ideation:
        logger.info("[Stage 2] Strategist — SKIPPED")
    else:
        logger.info("[Stage 2] Running Claude strategist")
        try:
            result.strategist_result = run_strategist_loop(
                model=model,
                max_turns=max_turns,
                attribution_constraints=attribution_constraints,
            )
        except Exception as e:
            result.strategist_error = str(e)
            logger.error("Strategist failed: %s — continuing with existing playbook", e)

    # Stage 3: Trading session
    if skip_executor:
        logger.info("[Stage 3] Trading executor — SKIPPED")
    else:
        logger.info("[Stage 3] Running trading session")
        try:
            result.trading_result = run_trading_session(dry_run=dry_run, model=executor_model)
        except Exception as e:
            result.trading_error = str(e)
            logger.error("Trading session failed: %s", e)

    # Stage 4: Strategy reflection
    if skip_strategy:
        logger.info("[Stage 4] Strategy reflection — SKIPPED")
        result.skipped_strategy = True
    else:
        logger.info("[Stage 4] Running strategy reflection")
        try:
            result.strategy_result = run_strategy_reflection(model=model, max_turns=10)
        except Exception as e:
            result.strategy_error = str(e)
            logger.error("Strategy reflection failed: %s", e)

    # Stage 5: Twitter posting
    if skip_twitter:
        logger.info("[Stage 5] Twitter posting — SKIPPED")
        result.skipped_twitter = True
    else:
        logger.info("[Stage 5] Running Twitter posting")
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
        for field_name in ["learning_error", "pipeline_error", "strategist_error", "trading_error", "strategy_error", "twitter_error"]:
            err = getattr(result, field_name)
            if err:
                logger.error("  %s: %s", field_name, err)
    else:
        logger.info("  All stages completed successfully")
    logger.info("=" * 60)

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
    parser.add_argument("--pipeline-hours", type=int, default=24)

    args = parser.parse_args()
    result = run_session(
        dry_run=args.dry_run, model=args.model, executor_model=args.executor_model,
        max_turns=args.max_turns, skip_pipeline=args.skip_pipeline,
        skip_ideation=args.skip_ideation, skip_executor=args.skip_executor,
        skip_strategy=args.skip_strategy,
        skip_twitter=args.skip_twitter, pipeline_hours=args.pipeline_hours,
    )
    if result.has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
