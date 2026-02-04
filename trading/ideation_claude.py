"""Claude-based ideation module for autonomous trade idea generation."""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

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

- Use `get_market_snapshot` to identify sectors/stocks worth researching
- Use `get_portfolio_state` to understand current holdings and capital
- Always check `get_active_theses` before creating new ones to avoid duplicates
- Use `get_macro_context` to understand broader market environment

Begin your research session now. When you've completed your analysis and thesis management, provide a brief summary of your findings and actions."""


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


def run_ideation_claude(
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 20,
) -> ClaudeIdeationResult:
    """
    Run an agentic ideation session with Claude.

    Unlike the Ollama version which uses a single LLM call with pre-fetched context,
    this version lets Claude autonomously drive the research process using tools.

    Args:
        model: Claude model to use
        max_turns: Maximum conversation turns

    Returns:
        ClaudeIdeationResult with session details
    """
    timestamp = datetime.now()
    print(f"[{timestamp.isoformat()}] Starting Claude ideation session")
    print(f"  Model: {model}")
    print(f"  Max turns: {max_turns}")

    # Reset session state (clears valid doc IDs from previous runs)
    reset_session()

    # Get Claude client
    client = get_claude_client()

    # Initial prompt to kick off the agent
    initial_message = """Begin your research session. Start by:
1. Getting the current portfolio state and active theses
2. Reviewing each active thesis and determining if updates are needed
3. Exploring market conditions for new opportunities
4. Creating 2-4 new theses based on your analysis

When you've completed your research, provide a summary of your findings and actions."""

    # Run agentic loop
    print("\n[Running agentic loop...]")
    result = run_agentic_loop(
        client=client,
        model=model,
        system=CLAUDE_IDEATION_SYSTEM,
        initial_message=initial_message,
        tools=TOOL_DEFINITIONS,
        tool_handlers=TOOL_HANDLERS,
        max_turns=max_turns,
    )

    # Extract results
    created, updated, closed = count_actions(result.messages)
    summary = extract_final_text(result.messages) or "No summary available"

    # Calculate cost estimate
    input_cost = result.input_tokens * 3 / 1_000_000  # $3 per 1M tokens
    output_cost = result.output_tokens * 15 / 1_000_000  # $15 per 1M tokens
    total_cost = input_cost + output_cost

    # Print summary
    print("\n" + "=" * 60)
    print("Claude Ideation Session Complete")
    print("=" * 60)
    print(f"  Turns used: {result.turns_used}")
    print(f"  Stop reason: {result.stop_reason}")
    print(f"  Theses created: {created}")
    print(f"  Theses updated: {updated}")
    print(f"  Theses closed: {closed}")
    print(f"\nToken usage:")
    print(f"  Input tokens: {result.input_tokens:,}")
    print(f"  Output tokens: {result.output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.4f}")
    print(f"\nSummary:\n{summary[:1000]}{'...' if len(summary) > 1000 else ''}")

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


def main():
    """CLI entry point for Claude ideation."""
    parser = argparse.ArgumentParser(description="Run Claude ideation session")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum conversation turns (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_ideation_claude(
        model=args.model,
        max_turns=args.max_turns,
    )

    if result.theses_created == 0 and result.theses_updated == 0 and result.theses_closed == 0:
        print("\nNo thesis changes made.")


if __name__ == "__main__":
    main()
