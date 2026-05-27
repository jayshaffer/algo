"""Strategy supervisor: observer-only critic of Pinchy's strategy stack.

Runs a one-shot agentic loop with a read-only tool registry and persists
a single markdown memo to `supervisor_memos`. No writes to strategy state.

CLI: `python -m v2.supervisor [--model MODEL] [--max-turns N] [--dry-run]`
(CLI wiring is added in Task 3.2.)
"""

from __future__ import annotations

import logging

from v2 import tools as v2_tools

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
    "get_strategy_identity": v2_tools.tool_get_strategy_identity,
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
