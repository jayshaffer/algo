"""Claude-based ideation module for autonomous trade idea generation."""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime

from .claude_client import extract_final_text, get_claude_client, run_agentic_loop
from .context import get_equity_summary
from .formation import build_formation_context, get_orphan_positions
from .pricing import UnknownModelError, stage_cost_usd
from .tools import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    reset_session,
    tool_get_active_theses,
    tool_get_decision_history,
    tool_get_portfolio_state,
    tool_get_signal_attribution,
    tool_get_strategy_history,
    tool_get_strategy_identity,
    tool_get_strategy_rules,
)

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
- Use `get_news_signals` to check recent news for specific tickers you're researching (each line is prefixed with `[#id]` — those IDs are what you cite via `signal_refs` on `create_thesis` / `update_thesis`)
- Use `get_macro_context` for a category-level summary of the macro environment, and `get_macro_signals` to get a list of macro signals with IDs for citation

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
- Use `get_strategy_identity` to understand the system's evolving trading identity
- Use `get_strategy_rules` to see accumulated trading rules from past performance
- Use `get_strategy_history` to review recent strategy reflection memos
- Use `get_signal_attribution` to see which signal types have been historically predictive
- Use `get_decision_history` to review recent trading performance
- Use `get_recent_playbooks` to see what you planned in recent sessions. Always check this BEFORE writing today's playbook — if today's plan reverses a recent action on the same ticker, you must justify the reversal in your reasoning.
- Use `web_search` to research current market conditions and companies
- Use `get_market_snapshot` to see sector performance and movers
- Use thesis tools to manage trade ideas
- Use `adopt_thesis` to attach a thesis to any held position that lacks one (orphan positions). Every held position MUST have an active thesis — otherwise exits can't be reasoned about and outcomes can't be learned from.
- Use `write_playbook` to write {date_ref} plan (REQUIRED — always write a playbook)

## Critical Rules
1. **Always write a playbook** — the executor depends on it
2. **No orphan positions** — if a held position has no active thesis, adopt one via `adopt_thesis` before writing the playbook
3. **Quality theses only** — specific entry/exit triggers, not vague ideas
4. **Learn from attribution** — weight signal types that have been predictive
5. **No duplicate theses** — check existing before creating
6. **Cite the evidence on every thesis you create or update.** Pass `signal_refs` (or `add_signal_refs` on update) with the news/macro signal IDs that justify it. IDs come from `get_news_signals` and `get_macro_signals` (each line is prefixed `[#id]`). The executor and attribution loop trace trades back to these IDs — if you skip them, the trade is invisible to the learning system. Adopted positions may legitimately have no current signals (omit the field).
7. **You author INTENTS, not share counts.** The trader resolves your intents to exact shares against live portfolio state. Never compute shares yourself — you will drift. Pick from:
   - SELLS: exit_full (null magnitude) · exit_partial_pct (0-100) · exit_dollar (dollars) · trim_to_portfolio_pct (target % of portfolio)
   - BUYS: invest_dollar (dollars) · invest_portfolio_pct (0-100) · invest_buying_power_pct (0-100) · add_to_target_pct (target % of portfolio, for adding to existing positions)

   Example: "exit AMZN fully" → {{"action":"sell","intent_type":"exit_full","intent_magnitude":null}}
   Example: "put $500 into AMD" → {{"action":"buy","intent_type":"invest_dollar","intent_magnitude":500}}
   Example: "take half off SPY" → {{"action":"sell","intent_type":"exit_partial_pct","intent_magnitude":50}}
8. **Reversal Justification.** Before adding a buy/sell action to today's playbook, check `get_recent_playbooks` for the same ticker. If today's action is the opposite of what you wrote in the past 7 days, your action's `reasoning` must explicitly cite (a) the prior playbook date and action, and (b) the new evidence — fundamentals shift, catalyst resolution, price level reached — that justifies reversing. Re-narrating the same fundamentals from a different angle is not new evidence; if you cannot articulate (b), do not propose the action."""

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
    # T2.12: distinct from `theses_created` so adoption-heavy sessions
    # don't read as bursts of new idea generation.
    theses_adopted: int = 0
    final_summary: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ClaudeIdeationResult:
    """Result of a Claude ideation session."""

    timestamp: datetime
    model: str
    turns_used: int
    theses_created: int
    theses_updated: int
    theses_closed: int
    theses_adopted: int = 0
    final_summary: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def count_actions(messages: list[dict]) -> tuple[int, int, int, int]:
    """Count thesis actions from tool results in conversation.

    T2.12: returns a 4-tuple (created, updated, closed, adopted). Previously
    `count_actions` returned 3 values and `Adopted` was a `Created`
    prefix, so adoption-heavy sessions inflated `created` and the
    dashboard couldn't separate "the system found new ideas" from
    "the system tidied orphan positions."
    """
    created = 0
    updated = 0
    closed = 0
    adopted = 0

    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        result = item.get("content", "")
                        if isinstance(result, str):
                            if "Adopted thesis ID" in result:
                                adopted += 1
                            elif "Created thesis ID" in result:
                                created += 1
                            elif "Updated thesis ID" in result:
                                updated += 1
                            elif "Closed thesis ID" in result:
                                closed += 1

    return created, updated, closed, adopted


def _run_claude_loop(
    system: str,
    initial_message: str,
    model: str,
    max_turns: int,
    label: str,
    session_id: int | None = None,
    stage_name: str | None = None,
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
        session_id=session_id,
        stage_name=stage_name,
    )

    created, updated, closed, adopted = count_actions(result.messages)
    summary = extract_final_text(result.messages) or "No summary available"

    _print_cost_summary(label, result, model, created, updated, closed, summary, adopted=adopted)

    return ClaudeIdeationResult(
        timestamp=timestamp,
        model=model,
        turns_used=result.turns_used,
        theses_created=created,
        theses_updated=updated,
        theses_closed=closed,
        theses_adopted=adopted,
        final_summary=summary,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _print_cost_summary(label, result, model, created, updated, closed, summary, adopted=0):
    """Print token usage and estimated USD cost for an agentic loop result.

    Uses v2.pricing.stage_cost_usd, which reads rates from the
    model_pricing DB table — single source of truth, matches the SQL
    cost view exactly. Prior version hardcoded Opus pricing inline and
    miscomputed uncached input by double-subtracting cache tokens.
    """
    try:
        cost = stage_cost_usd(
            model=model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_creation_tokens=result.cache_creation_input_tokens,
            cache_read_tokens=result.cache_read_input_tokens,
        )
        cost_str = f"${cost:.4f}"
    except UnknownModelError:
        cost_str = f"(no price seeded for model {model})"

    print("\n" + "=" * 60)
    print(f"{label} Complete")
    print("=" * 60)
    print(f"  Turns used: {result.turns_used}")
    print(f"  Stop reason: {result.stop_reason}")
    print(f"  Theses created: {created}")
    print(f"  Theses updated: {updated}")
    print(f"  Theses closed: {closed}")
    if adopted:
        print(f"  Theses adopted: {adopted}")
    print("\nToken usage:")
    print(f"  Input tokens: {result.input_tokens:,}")
    if result.cache_read_input_tokens:
        print(f"  Cache read tokens: {result.cache_read_input_tokens:,}")
        print(f"  Cache write tokens: {result.cache_creation_input_tokens:,}")
    print(f"  Output tokens: {result.output_tokens:,}")
    print(f"  Estimated cost: {cost_str}")
    print(f"\nSummary:\n{summary[:1000]}{'...' if len(summary) > 1000 else ''}")


def run_ideation_claude(
    model: str = "claude-opus-4-6",
    max_turns: int = 20,
    session_id: int | None = None,
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
        session_id=session_id,
        stage_name="ideation",
    )


def _build_pre_seeded_context() -> str:
    """Build the pre-seeded portfolio/strategy context block for the strategist."""
    sections = [
        ("Portfolio", tool_get_portfolio_state),
        ("Active Theses", tool_get_active_theses),
        ("Decision History", tool_get_decision_history),
        ("Signal Attribution", tool_get_signal_attribution),
        ("Strategy Identity", tool_get_strategy_identity),
        ("Strategy Rules", tool_get_strategy_rules),
        ("Strategy History", tool_get_strategy_history),
    ]
    parts = []
    for name, fetch in sections:
        try:
            parts.append(f"=== {name} ===\n{fetch()}")
        except Exception:
            parts.append(f"=== {name} ===\n(unavailable)")

    try:
        equity = get_equity_summary(days=30)
        if equity:
            parts.append(f"=== Account History ===\n{equity}")
    except Exception:
        pass

    return "\n\n".join(parts)


def _build_orphan_block(orphans: list[dict] | None = None) -> tuple[str, str, int]:
    """Return (orphan_block, adopt_step, step_offset). Empty strings / 0 when no orphans.

    T2.14: callers can pass a pre-fetched orphans list to share state
    with `build_formation_context`. The function still falls back to a
    DB fetch if nothing is passed.
    """
    if orphans is None:
        try:
            orphans = get_orphan_positions()
        except Exception:
            logger.exception("Failed to fetch orphan positions for initial message")
            orphans = []

    if not orphans:
        return "", "", 0

    orphan_lines = "\n".join(
        f"- {p['ticker']}: {p['shares']} shares @ ${float(p['avg_cost']):.2f} avg"
        for p in orphans
    )
    orphan_block = f"""
=== ORPHAN POSITIONS (no active thesis) ===
{orphan_lines}

These are held positions with no thesis attached. For EACH orphan, call the \
`adopt_thesis` tool to attach a direction, rationale, and exit/invalidation \
triggers. Do this BEFORE writing the playbook — orphans without theses cannot \
be reasoned about or learned from."""
    adopt_step = (
        "2. Adopt every orphan position listed above via `adopt_thesis` — "
        "this is a prerequisite for the playbook.\n"
    )
    return orphan_block, adopt_step, 1


def run_strategist_loop(
    model: str = "claude-opus-4-6",
    max_turns: int = 25,
    system_prompt: str = None,
    attribution_constraints: str = "",
    session_id: int | None = None,
) -> ClaudeIdeationResult:
    """Run the strategist agentic loop.

    attribution_constraints is appended to the system prompt, making
    signal performance data enforceable rather than advisory.
    """
    base_prompt = system_prompt or CLAUDE_SESSION_STRATEGIST_SYSTEM
    if attribution_constraints:
        base_prompt = base_prompt + "\n\n" + attribution_constraints

    # T2.14: fetch orphans once and share with both consumers — formation
    # context (system prompt block) and the orphan-adoption step list.
    try:
        orphans = get_orphan_positions()
    except Exception:
        logger.exception("Failed to fetch orphan positions for strategist setup")
        orphans = []

    formation_context = build_formation_context(orphans=orphans)
    if formation_context:
        base_prompt = base_prompt + "\n\n" + formation_context

    pre_seeded = _build_pre_seeded_context()
    orphan_block, adopt_step, step_offset = _build_orphan_block(orphans=orphans)
    if orphan_block:
        pre_seeded = pre_seeded + "\n\n" + orphan_block

    initial_message = f"""Here is the current state (pre-loaded to save round-trips):

{pre_seeded}

Now proceed with your strategist session:
1. Review the data above — do NOT re-fetch portfolio, theses, decisions, attribution, identity, rules, or strategy history
{adopt_step}{2 + step_offset}. Explore market conditions for new opportunities (use web_search, get_market_snapshot, get_news_signals)
{3 + step_offset}. Update or close stale theses, create 2-4 new ones
{4 + step_offset}. Write today's playbook using the write_playbook tool

When you've completed your work, provide a summary of your findings and actions."""

    return _run_claude_loop(
        system=base_prompt,
        initial_message=initial_message,
        model=model,
        max_turns=max_turns,
        label="Strategist Loop",
        session_id=session_id,
        stage_name="ideation",
    )


def run_strategist_session(
    model: str = "claude-opus-4-6",
    max_turns: int = 25,
    attribution_constraints: str = "",
) -> ClaudeIdeationResult:
    """Entry point for strategist. No longer runs backfill/attribution internally."""
    return run_strategist_loop(
        model=model,
        max_turns=max_turns,
        attribution_constraints=attribution_constraints,
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
        run_strategist_session(model=args.model, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
