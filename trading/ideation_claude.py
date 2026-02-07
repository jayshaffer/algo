"""Claude-based ideation module for autonomous trade idea generation."""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

from .backfill import run_backfill
from .attribution import compute_signal_attribution, get_attribution_summary
from .claude_client import get_claude_client, run_agentic_loop, extract_final_text
from .tools import TOOL_DEFINITIONS, TOOL_HANDLERS, reset_session

logger = logging.getLogger(__name__)


CLAUDE_IDEATION_SYSTEM = """You are an autonomous investment research agent. Your job is to generate and manage trade theses using your knowledge of markets, companies, and current conditions.

## Your Research Process

1. **Understand Current State**: Start by getting the portfolio state and active theses to understand what you're working with.

2. **Review Existing Theses**: For each active thesis, evaluate whether:
   - The thesis remains valid (keep it as-is)
   - Conditions have changed requiring updates (update the thesis)
   - The invalidation criteria have been met (close as invalidated)
   - The thesis has aged out without being acted upon (close as expired)

3. **Explore New Opportunities**: Use the market snapshot to identify interesting sectors or movers. Apply your knowledge of market dynamics, company fundamentals, and macro trends.

4. **Generate New Theses**: For promising opportunities, create well-reasoned theses with:
   - Clear directional view (long/short/avoid)
   - Specific entry and exit triggers (price levels, events, metrics)
   - Defined invalidation criteria

## Critical Rules

1. **No Duplicate Theses**: Do not create theses for tickers that already have active theses or are currently in the portfolio.

2. **Quality Over Quantity**: Generate 2-4 high-quality theses rather than many low-quality ones. Each thesis should be actionable and well-reasoned.

3. **Be Specific**: Entry/exit triggers should be actionable (e.g., "Price breaks above $150 with volume", "Q2 earnings miss by >10%") not vague ("if things improve").

4. **Self-Critique**: Before creating a thesis, consider: What could I be missing? Is my confidence calibrated? What's the bear case?

## Tool Usage Guidelines

- Use `web_search` to research current market conditions, company news, earnings, analyst opinions, and sector trends. Search for specific tickers, events, or themes you want to investigate. This is your primary research tool for generating well-informed theses.
- Use `get_market_snapshot` to identify sectors/stocks worth researching
- Use `get_portfolio_state` to understand current holdings and capital
- Always check `get_active_theses` before creating new ones to avoid duplicates
- Use `get_news_signals` to check recent news for specific tickers you're researching
- Use `get_macro_context` to understand broader market environment

Begin your research session now. When you've completed your analysis and thesis management, provide a brief summary of your findings and actions."""


_STRATEGIST_TEMPLATE = """You are the strategist for an automated trading system. You run {timing} to review {review_scope}, manage theses, and write {date_ref} playbook for the executor{executor_note}.

## Your Daily Process

1. **Review**: Check the portfolio state, recent decision outcomes, and signal attribution scores. Identify what's working and what isn't.

2. **Thesis Management**: For each active thesis:
   - If still valid, keep it
   - If conditions changed, update it
   - If invalidation criteria met, close it
   - If aged out without action, close as expired

3. **Research & New Theses**: Use web search to investigate opportunities. Create 2-4 new well-reasoned theses with specific entry/exit triggers.

4. **Write Playbook**: This is your primary output. Use the `write_playbook` tool to create {date_ref} trading plan with:
   - Market outlook
   - Priority actions (specific trades the executor should consider)
   - Watch list (tickers to monitor)
   - Risk notes (warnings, upcoming events)

## Tool Usage
- Use `get_signal_attribution` to see which signal types have been historically predictive
- Use `get_decision_history` to review recent trading performance
- Use `web_search` to research current market conditions and companies
- Use `get_market_snapshot` to see sector performance and movers
- Use thesis tools to manage trade ideas
- Use `write_playbook` to write {date_ref} plan (REQUIRED — always write a playbook)

## Critical Rules
1. **Always write a playbook** — the executor depends on it
2. **Quality theses only** — specific entry/exit triggers, not vague ideas
3. **Learn from attribution** — weight signal types that have been predictive
4. **No duplicate theses** — check existing before creating
5. **Fractional shares supported** — the executor can buy/sell fractional shares. Size playbook actions by dollar amount (e.g. "$500 of AMZN" → max_quantity: 2.5 at ~$200/share) rather than forcing round lots."""

CLAUDE_STRATEGIST_SYSTEM = _STRATEGIST_TEMPLATE.format(
    timing="after market close",
    review_scope="results",
    date_ref="tomorrow's",
    executor_note="",
)

CLAUDE_SESSION_STRATEGIST_SYSTEM = _STRATEGIST_TEMPLATE.format(
    timing="before market close",
    review_scope="the portfolio",
    date_ref="today's",
    executor_note=" that runs immediately after you",
)


@dataclass
class StrategistResult:
    """Result of a strategist session."""
    timestamp: datetime
    model: str
    turns_used: int
    outcomes_backfilled: int
    attribution_computed: int
    theses_created: int
    theses_updated: int
    theses_closed: int
    final_summary: str
    input_tokens: int
    output_tokens: int


@dataclass
class ClaudeIdeationResult:
    """Result of a Claude ideation session."""

    timestamp: datetime
    model: str
    turns_used: int
    theses_created: int
    theses_updated: int
    theses_closed: int
    final_summary: str
    input_tokens: int
    output_tokens: int


def count_actions(messages: list[dict]) -> tuple[int, int, int]:
    """Count thesis actions from tool results in conversation."""
    created = 0
    updated = 0
    closed = 0

    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        result = item.get("content", "")
                        if isinstance(result, str):
                            if "Created thesis ID" in result:
                                created += 1
                            elif "Updated thesis ID" in result:
                                updated += 1
                            elif "Closed thesis ID" in result:
                                closed += 1

    return created, updated, closed


def _run_claude_loop(
    system: str,
    initial_message: str,
    model: str,
    max_turns: int,
    label: str,
) -> ClaudeIdeationResult:
    """Run a Claude agentic loop and return a standardized result.

    Shared core for both ideation and strategist modes.
    """
    timestamp = datetime.now()
    logger.info("Starting %s (model=%s, max_turns=%d)", label, model, max_turns)

    reset_session()
    client = get_claude_client()

    result = run_agentic_loop(
        client=client,
        model=model,
        system=system,
        initial_message=initial_message,
        tools=TOOL_DEFINITIONS,
        tool_handlers=TOOL_HANDLERS,
        max_turns=max_turns,
    )

    created, updated, closed = count_actions(result.messages)
    summary = extract_final_text(result.messages) or "No summary available"

    _print_cost_summary(label, result, created, updated, closed, summary)

    return ClaudeIdeationResult(
        timestamp=timestamp,
        model=model,
        turns_used=result.turns_used,
        theses_created=created,
        theses_updated=updated,
        theses_closed=closed,
        final_summary=summary,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _print_cost_summary(label, result, created, updated, closed, summary):
    """Print token usage and cost estimate for an agentic loop result."""
    uncached_input = result.input_tokens - result.cache_creation_input_tokens - result.cache_read_input_tokens
    input_cost = uncached_input * 5 / 1_000_000
    cache_write_cost = result.cache_creation_input_tokens * 6.25 / 1_000_000
    cache_read_cost = result.cache_read_input_tokens * 0.50 / 1_000_000
    output_cost = result.output_tokens * 25 / 1_000_000
    total_cost = input_cost + cache_write_cost + cache_read_cost + output_cost

    print("\n" + "=" * 60)
    print(f"{label} Complete")
    print("=" * 60)
    print(f"  Turns used: {result.turns_used}")
    print(f"  Stop reason: {result.stop_reason}")
    print(f"  Theses created: {created}")
    print(f"  Theses updated: {updated}")
    print(f"  Theses closed: {closed}")
    print(f"\nToken usage:")
    print(f"  Input tokens: {result.input_tokens:,}")
    if result.cache_read_input_tokens:
        print(f"  Cache read tokens: {result.cache_read_input_tokens:,}")
        print(f"  Cache write tokens: {result.cache_creation_input_tokens:,}")
    print(f"  Output tokens: {result.output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.4f}")
    print(f"\nSummary:\n{summary[:1000]}{'...' if len(summary) > 1000 else ''}")


def run_ideation_claude(
    model: str = "claude-opus-4-6",
    max_turns: int = 20,
) -> ClaudeIdeationResult:
    """Run an agentic ideation session with Claude.

    Lets Claude autonomously drive the research process using tools.
    """
    initial_message = """Begin your research session. Start by:
1. Getting the current portfolio state and active theses
2. Reviewing each active thesis and determining if updates are needed
3. Exploring market conditions for new opportunities
4. Creating 2-4 new theses based on your analysis

When you've completed your research, provide a summary of your findings and actions."""

    return _run_claude_loop(
        system=CLAUDE_IDEATION_SYSTEM,
        initial_message=initial_message,
        model=model,
        max_turns=max_turns,
        label="Claude Ideation Session",
    )


def run_strategist_loop(
    model: str = "claude-opus-4-6",
    max_turns: int = 25,
    system_prompt: str = None,
) -> ClaudeIdeationResult:
    """Run only the Claude strategist agentic loop.

    Reads existing attribution from DB via get_attribution_summary().
    Does NOT run backfill or compute attribution — those are learning-system
    operations handled separately.
    """
    attribution_summary = get_attribution_summary()

    initial_message = f"""Begin your strategist session. Here is the current signal attribution data:

{attribution_summary}

Start by:
1. Getting the current portfolio state and active theses
2. Reviewing recent decision history and outcomes
3. Reviewing each active thesis and determining if updates are needed
4. Exploring market conditions for new opportunities
5. Creating 2-4 new theses based on your analysis
6. Writing today's playbook using the write_playbook tool

When you've completed your work, provide a summary of your findings and actions."""

    return _run_claude_loop(
        system=system_prompt or CLAUDE_SESSION_STRATEGIST_SYSTEM,
        initial_message=initial_message,
        model=model,
        max_turns=max_turns,
        label="Strategist Loop",
    )


def run_strategist_session(
    model: str = "claude-opus-4-6",
    max_turns: int = 25,
    dry_run: bool = False,
) -> StrategistResult:
    """
    Run the full strategist session.

    1. Backfill decision outcomes
    2. Compute signal attribution
    3. Run Claude agentic loop (thesis management + playbook generation)

    Args:
        model: Claude model to use
        max_turns: Maximum conversation turns
        dry_run: If True, skip backfill DB writes

    Returns:
        StrategistResult with session details
    """
    timestamp = datetime.now()
    print(f"[{timestamp.isoformat()}] Starting strategist session")
    print(f"  Model: {model}")
    print(f"  Max turns: {max_turns}")

    # Step 1: Backfill outcomes
    print("\n[Step 1] Backfilling decision outcomes...")
    try:
        backfill_result = run_backfill(dry_run=dry_run)
        outcomes_backfilled = backfill_result["total_filled"]
    except Exception as e:
        print(f"  Backfill error (continuing): {e}")
        outcomes_backfilled = 0

    # Step 2: Compute signal attribution
    print("\n[Step 2] Computing signal attribution...")
    try:
        attribution_results = compute_signal_attribution()
        attribution_computed = len(attribution_results)
    except Exception as e:
        print(f"  Attribution error (continuing): {e}")
        attribution_computed = 0

    # Step 3: Run Claude agentic loop (delegates to run_strategist_loop)
    print("\n[Step 3] Running Claude strategist loop...")
    loop_result = run_strategist_loop(
        model=model,
        max_turns=max_turns,
        system_prompt=CLAUDE_STRATEGIST_SYSTEM,
    )

    print(f"  Outcomes backfilled: {outcomes_backfilled}")
    print(f"  Attribution categories: {attribution_computed}")

    return StrategistResult(
        timestamp=timestamp,
        model=model,
        turns_used=loop_result.turns_used,
        outcomes_backfilled=outcomes_backfilled,
        attribution_computed=attribution_computed,
        theses_created=loop_result.theses_created,
        theses_updated=loop_result.theses_updated,
        theses_closed=loop_result.theses_closed,
        final_summary=loop_result.final_summary,
        input_tokens=loop_result.input_tokens,
        output_tokens=loop_result.output_tokens,
    )


def main():
    """CLI entry point for Claude ideation/strategist."""
    parser = argparse.ArgumentParser(description="Run Claude strategist session")
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Claude model to use",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=25,
        help="Maximum conversation turns (default: 25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip database writes for backfill",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy ideation mode (no backfill/attribution/playbook)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.legacy:
        run_ideation_claude(model=args.model, max_turns=args.max_turns)
    else:
        run_strategist_session(model=args.model, max_turns=args.max_turns, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
