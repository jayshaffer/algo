"""Strategy supervisor: observer-only critic of Pinchy's strategy stack.

Runs a one-shot agentic loop with a read-only tool registry and persists
a single markdown memo to `supervisor_memos`. No writes to strategy state.

CLI: `python -m v2.supervisor [--model MODEL] [--max-turns N] [--dry-run]`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from collections import Counter

from v2 import claude_client
from v2 import tools as v2_tools
from v2.database.connection import get_cursor
from v2.pricing import UnknownModelError, stage_cost_usd

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1.0.0"
DEFAULT_SUPERVISOR_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TURNS = 20

STRATEGY_SUPERVISOR_SYSTEM = """\
You are the Strategy Supervisor for Pinchy, an agentic trading system.

Your role is to critique the trading strategy from a senior, skeptical
vantage point. You read state — you do not change it. There are no
write tools available to you.

Your four areas of focus:

1. Rule coherence & quality
   - Do active rules contradict each other?
   - Any rule that oscillates (binds/lifts repeatedly within days)?
   - Any active-but-dormant rule that hasn't bound in 30+ days?
   - Any rule churned out within a week of being added?
   - Is each active rule grounded in evidence or pet theory?

2. Thesis discipline
   - Are theses honored at their stated entry/exit triggers?
   - Any thesis lingering past its entry window with no action?
   - Do any active theses contradict each other?
   - Are closed theses being learned from?

3. Identity + behavior drift
   - Is the strategy identity coherent across recent memos, or whipsawing?
   - Does what the executor actually does (sizing, flip-flops, sector mix,
     round-trip frequency) match the identity?

4. Reflection quality
   - Did the recent reflection stages take action, or coast?
   - Did they ignore obvious problems (flip-flops, dormant rules)?
   - Are memos substantive or vacuous?

Investigate before you opine. Use get_* tools to verify any pattern
you suspect — pull bind histories, decision detail, thesis lineage.
Cite specific rule_ids, thesis_ids, decision_ids, and dates in your
critique. A claim without a citation should not appear in the memo.

Be direct. Don't soften. The point of this role is to surface what
the reflection stage missed. If you find nothing wrong, say so plainly —
do not invent concerns to seem thorough. A short "no major concerns
this week, here's why" memo is more valuable than a padded one.

Output: a single markdown memo with sections matching the four areas
above. Skip a section entirely if you have nothing to say about it.
End with a "Watchlist" section: 1-5 specific things to revisit on
the next supervisor run.
"""


# Mutator tool names — must NEVER appear in the supervisor's registered handlers.
# Source of truth for the mutator-overlap defense test in Task 4.1.
STRATEGY_MUTATOR_NAMES: frozenset[str] = frozenset({
    "propose_rule",
    "retire_rule",
    "revalidate_rule",
    "update_strategy_identity",
    "write_strategy_memo",
})


# Claude-facing tool name → Python handler.
# Names are the spec's `get_*` form. Some entries reuse existing tools.py handlers
# under a renamed key (e.g. spec's "get_active_rules" → existing tool_get_strategy_rules).
SUPERVISOR_TOOL_HANDLERS: dict = {
    # Strategy state
    "get_strategy_identity_with_history": v2_tools.tool_get_strategy_identity_with_history,
    "get_active_rules": v2_tools.tool_get_strategy_rules,
    "get_retired_rules": v2_tools.tool_get_retired_rules,
    "get_rule_bind_history": v2_tools.tool_get_rule_bind_history,
    # Theses
    "get_theses": v2_tools.tool_get_theses,
    "get_thesis_lineage": v2_tools.tool_get_thesis_lineage,
    # Behavior
    "get_recent_decisions": v2_tools.tool_get_recent_decisions,
    "get_decision_detail": v2_tools.tool_get_decision_detail,
    "get_flip_flop_report": v2_tools.tool_get_flip_flop_report,
    "get_executor_behavior_summary": v2_tools.tool_get_executor_behavior_summary,
    "get_signal_attribution": v2_tools.tool_get_signal_attribution,
    # Reflection / sessions
    "get_session_memos": v2_tools.tool_get_session_memos,
    "get_reflection_actions": v2_tools.tool_get_reflection_actions,
    "get_session_summary": v2_tools.tool_get_session_summary_window,
}


def build_supervisor_tool_defs() -> list[dict]:
    """Filter the registered TOOL_DEFS in v2.tools to just the supervisor's set.

    Looks up the canonical v2.tools registry name (`TOOL_DEFINITIONS` in this
    codebase, NOT `TOOL_DEFS`). If the name differs, the import below will
    raise on first call and the cause will be obvious from the traceback.
    """
    wanted = set(SUPERVISOR_TOOL_HANDLERS.keys())
    return [td for td in v2_tools.TOOL_DEFINITIONS if td.get("name") in wanted]


def _summarize_tool_calls(messages: list) -> list[dict]:
    """Walk assistant messages and count tool_use blocks by name."""
    counter: Counter = Counter()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                counter[block.get("name", "?")] += 1
    return [{"name": name, "count": count} for name, count in counter.most_common()]


def _insert_memo(
    *,
    model: str,
    content: str | None,
    status: str,
    turns_used: int,
    tool_calls: list[dict],
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
    error_message: str | None,
) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO supervisor_memos
              (model, prompt_version, content, status, turns_used,
               tool_calls, input_tokens, output_tokens, cost_usd, error_message)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                model,
                PROMPT_VERSION,
                content,
                status,
                turns_used,
                json.dumps(tool_calls),
                input_tokens,
                output_tokens,
                cost_usd,
                error_message,
            ),
        )
        row = cur.fetchone()
    # RealDictCursor returns dict-shaped rows
    return row["id"] if isinstance(row, dict) else row[0]


def _compute_cost_safely(model: str, usage) -> float | None:
    """Return cost in USD, or None if the model isn't priced."""
    try:
        return stage_cost_usd(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            cache_read_tokens=usage.cache_read_tokens,
        )
    except UnknownModelError:
        log.warning("model %r missing from model_pricing; cost_usd=NULL", model)
        return None


def run_supervisor(
    model: str = DEFAULT_SUPERVISOR_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    dry_run: bool = False,
) -> int | None:
    """Run one supervisor pass. Returns inserted memo id, or None if dry_run."""
    client = claude_client.get_claude_client()
    tool_defs = build_supervisor_tool_defs()
    initial = "Begin your strategy review. Investigate, cite IDs, then write the memo."

    try:
        with claude_client.capture_usage() as usage:
            result = claude_client.run_agentic_loop(
                client=client,
                model=model,
                system=STRATEGY_SUPERVISOR_SYSTEM,
                initial_message=initial,
                tools=tool_defs,
                tool_handlers=SUPERVISOR_TOOL_HANDLERS,
                max_turns=max_turns,
                stage_name="supervisor",
                purpose="supervisor_loop",
            )
    except Exception as exc:  # noqa: BLE001 — top-level guard for the agentic loop
        log.exception("supervisor loop failed")
        if dry_run:
            raise
        return _insert_memo(
            model=model,
            content=None,
            status="error",
            turns_used=0,
            tool_calls=[],
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )

    final_text = claude_client.extract_final_text(result.messages)
    tool_calls = _summarize_tool_calls(result.messages)
    cost_usd = _compute_cost_safely(model, usage)

    if result.stop_reason == "max_turns":
        status = "max_turns"
        # extract_final_text may have returned a synthetic fallback string —
        # store it so the operator can see what was attempted, but mark the
        # status so the dashboard banner triggers.
        content = final_text
        error_message = "loop hit max_turns before producing a final memo"
    elif final_text is None:
        # Belt-and-suspenders: extract_final_text shouldn't return None given
        # its synthetic fallback, but guard anyway.
        status = "max_turns"
        content = None
        error_message = "loop did not produce final text"
    else:
        status = "ok"
        content = final_text
        error_message = None

    if dry_run:
        print("=== SUPERVISOR MEMO (dry-run, not persisted) ===")
        print(content or f"[no final text — status={status}]")
        return None

    return _insert_memo(
        model=model,
        content=content,
        status=status,
        turns_used=result.turns_used,
        tool_calls=tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd,
        error_message=error_message,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pinchy strategy supervisor (observer-only)")
    parser.add_argument(
        "--model",
        default=os.environ.get("ALGO_SUPERVISOR_MODEL", DEFAULT_SUPERVISOR_MODEL),
        help=f"Anthropic model id (default: env ALGO_SUPERVISOR_MODEL or {DEFAULT_SUPERVISOR_MODEL!r})",
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the loop, print the memo, skip the INSERT",
    )
    args = parser.parse_args()
    try:
        memo_id = run_supervisor(model=args.model, max_turns=args.max_turns, dry_run=args.dry_run)
    except Exception:
        traceback.print_exc()
        return 2
    if memo_id is not None:
        print(f"supervisor_memo_id={memo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
