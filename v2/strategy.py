"""Strategy reflection stage (Stage 4).

Runs after the trading session to reflect on performance,
update the system's evolving trading identity, manage accumulated
rules, and write session memos.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .claude_client import get_claude_client, run_agentic_loop, extract_final_text
from .tools import tool_get_strategy_identity, tool_get_strategy_rules, tool_get_strategy_history
from .attribution import get_attribution_summary
from .database.trading_db import (
    get_current_strategy_state,
    clear_current_strategy_state,
    insert_strategy_state,
    insert_strategy_rule,
    retire_strategy_rule,
    insert_strategy_memo,
    get_recent_strategy_memos,
    get_active_strategy_rules,
    get_recent_decisions,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyReflectionResult:
    rules_proposed: int
    rules_retired: int
    identity_updated: bool
    memo_written: bool
    input_tokens: int
    output_tokens: int
    turns_used: int


STRATEGY_REFLECTION_SYSTEM = """You are the strategy reflection agent for an autonomous trading system. You have just completed a trading session. Your job is to review what happened, update the system's evolving trading identity, and manage its accumulated rules.

You are NOT making trades. You are reflecting on performance and shaping who this system is as a trader.

## Your Process

1. **Review Current State**: Get the current strategy identity, active rules, and recent memos to understand where the system is.

2. **Analyze Session**: Get the session summary to see what happened today — decisions made, outcomes of past decisions, and signal attribution scores.

3. **Update Rules**: Based on attribution data and decision outcomes:
   - Propose new rules when patterns emerge (e.g., a signal type consistently wins or loses)
   - Retire rules that are no longer supported by data
   - Rules should be specific and evidence-based

4. **Update Identity**: If the system's behavior or performance suggests a shift in trading style, update the identity. The identity should reflect what the system IS, not what it aspires to be.

5. **Write Memo**: Always write a reflection memo summarizing what you observed and any changes you made. This is the system's memory.

## Critical Rules

1. **Evidence-based**: Every rule must cite specific data (win rates, sample sizes, outcome averages)
2. **First session**: If no identity exists yet, bootstrap one from attribution data and today's results
3. **Concise memos**: Memos should be 2-4 paragraphs, focused on actionable observations
4. **Always write a memo**: Even if nothing changed, document why
5. **Don't over-rotate**: A single bad session doesn't warrant major strategy changes. Look for patterns across multiple sessions."""


# --- Write Tool Handlers ---

def tool_update_strategy_identity(
    identity_text: str,
    risk_posture: str,
    sector_biases: dict,
    preferred_signals: list,
    avoided_signals: list,
) -> str:
    """Update the system's strategy identity (creates new versioned row)."""
    logger.info("Updating strategy identity")
    current = get_current_strategy_state()
    new_version = (current["version"] + 1) if current else 1

    clear_current_strategy_state()
    state_id = insert_strategy_state(
        identity_text=identity_text,
        risk_posture=risk_posture,
        sector_biases=sector_biases,
        preferred_signals=preferred_signals,
        avoided_signals=avoided_signals,
        version=new_version,
    )
    return f"Strategy identity updated to version {new_version} (ID: {state_id})"


def tool_propose_rule(
    rule_text: str,
    category: str,
    direction: str,
    confidence: float,
    supporting_evidence: str,
) -> str:
    """Propose a new strategy rule."""
    logger.info(f"Proposing rule: {rule_text[:50]}...")
    rule_id = insert_strategy_rule(
        rule_text=rule_text,
        category=category,
        direction=direction,
        confidence=confidence,
        supporting_evidence=supporting_evidence,
    )
    return f"Created rule ID {rule_id}: {rule_text}"


def tool_retire_rule(rule_id: int, reason: str) -> str:
    """Retire a strategy rule."""
    logger.info(f"Retiring rule {rule_id}: {reason}")
    success = retire_strategy_rule(rule_id=rule_id)
    if success:
        return f"Retired rule ID {rule_id}. Reason: {reason}"
    return f"Error: Rule ID {rule_id} not found or already retired"


def tool_write_strategy_memo(memo_type: str, content: str) -> str:
    """Write a strategy reflection memo."""
    logger.info(f"Writing strategy memo ({memo_type})")
    current = get_current_strategy_state()
    state_id = current["id"] if current else None
    memo_id = insert_strategy_memo(
        session_date=date.today(),
        memo_type=memo_type,
        content=content,
        strategy_state_id=state_id,
    )
    return f"Memo written (ID: {memo_id})"


def tool_get_session_summary(days: int = 30) -> str:
    """Get today's session summary — recent decisions, outcomes, attribution."""
    logger.info("Getting session summary")
    lines = []

    decisions = get_recent_decisions(days=days)
    if decisions:
        lines.append(f"Recent decisions ({len(decisions)}):")
        for d in decisions[:10]:
            outcome_7d = f"{d['outcome_7d']:+.2f}%" if d.get("outcome_7d") is not None else "pending"
            outcome_30d = f"{d['outcome_30d']:+.2f}%" if d.get("outcome_30d") is not None else "pending"
            lines.append(
                f"  [{d['date']}] {d['action'].upper()} {d['ticker']}: "
                f"7d={outcome_7d}, 30d={outcome_30d}"
            )
    else:
        lines.append("No recent decisions.")

    lines.append("")
    lines.append("Signal Attribution:")
    lines.append(get_attribution_summary())

    return "\n".join(lines)


# --- Tool Definitions ---

STRATEGY_TOOL_DEFINITIONS = [
    {
        "name": "get_strategy_identity",
        "description": "Get the system's current strategy identity.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_rules",
        "description": "Get all active strategy rules.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_history",
        "description": "Get recent strategy reflection memos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of memos (default: 5)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_session_summary",
        "description": (
            "Get a summary of recent trading activity — decisions, outcomes, "
            "and signal attribution scores. Use this to understand what happened."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Look back period (default: 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "update_strategy_identity",
        "description": (
            "Update the system's trading identity. Creates a new versioned identity. "
            "Use this to reflect changes in trading style, risk posture, or signal preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identity_text": {"type": "string", "description": "Who is this system as a trader?"},
                "risk_posture": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive"],
                },
                "sector_biases": {
                    "type": "object",
                    "description": "Sector biases, e.g. {\"tech\": \"overweight\", \"energy\": \"avoid\"}",
                },
                "preferred_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Signal types to favor",
                },
                "avoided_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Signal types to avoid",
                },
            },
            "required": ["identity_text", "risk_posture", "sector_biases", "preferred_signals", "avoided_signals"],
        },
    },
    {
        "name": "propose_rule",
        "description": (
            "Propose a new strategy rule based on observed patterns. "
            "Rules are either constraints (avoid X) or preferences (favor X)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_text": {"type": "string", "description": "Human-readable rule"},
                "category": {"type": "string", "description": "Domain, e.g. news_signal:legal, position_sizing"},
                "direction": {"type": "string", "enum": ["constraint", "preference"]},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "supporting_evidence": {"type": "string", "description": "Data backing this rule"},
            },
            "required": ["rule_text", "category", "direction", "confidence", "supporting_evidence"],
        },
    },
    {
        "name": "retire_rule",
        "description": "Retire a strategy rule that is no longer supported by data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "integer", "description": "ID of rule to retire"},
                "reason": {"type": "string", "description": "Why this rule is being retired"},
            },
            "required": ["rule_id", "reason"],
        },
    },
    {
        "name": "write_strategy_memo",
        "description": (
            "Write a strategy reflection memo. Always write one at the end of reflection. "
            "Types: reflection (general), rule_change (when rules changed), identity_update (when identity changed)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memo_type": {
                    "type": "string",
                    "enum": ["reflection", "rule_change", "identity_update"],
                },
                "content": {"type": "string", "description": "The memo content (2-4 paragraphs)"},
            },
            "required": ["memo_type", "content"],
        },
    },
]

STRATEGY_TOOL_HANDLERS = {
    "get_strategy_identity": tool_get_strategy_identity,
    "get_strategy_rules": tool_get_strategy_rules,
    "get_strategy_history": tool_get_strategy_history,
    "get_session_summary": tool_get_session_summary,
    "update_strategy_identity": tool_update_strategy_identity,
    "propose_rule": tool_propose_rule,
    "retire_rule": tool_retire_rule,
    "write_strategy_memo": tool_write_strategy_memo,
}


def _count_actions(messages: list[dict]) -> tuple[int, int, bool, bool]:
    """Count strategy actions from tool results."""
    proposed = 0
    retired = 0
    identity_updated = False
    memo_written = False

    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        result_text = item.get("content", "")
                        if isinstance(result_text, str):
                            if "Created rule ID" in result_text:
                                proposed += 1
                            elif "Retired rule ID" in result_text:
                                retired += 1
                            elif "identity updated" in result_text.lower():
                                identity_updated = True
                            elif "Memo written" in result_text:
                                memo_written = True

    return proposed, retired, identity_updated, memo_written


def run_strategy_reflection(
    model: str = "claude-opus-4-6",
    max_turns: int = 10,
) -> StrategyReflectionResult:
    """Run the strategy reflection stage (Stage 4)."""
    logger.info("Starting strategy reflection (model=%s, max_turns=%d)", model, max_turns)

    client = get_claude_client()

    result = run_agentic_loop(
        client=client,
        model=model,
        system=STRATEGY_REFLECTION_SYSTEM,
        initial_message=(
            "Begin your strategy reflection. Start by:\n"
            "1. Getting the current strategy identity and rules\n"
            "2. Getting the session summary (recent decisions and attribution)\n"
            "3. Getting recent strategy memos for context\n"
            "4. Analyzing what happened and making any necessary updates\n"
            "5. Writing a reflection memo\n"
        ),
        tools=STRATEGY_TOOL_DEFINITIONS,
        tool_handlers=STRATEGY_TOOL_HANDLERS,
        max_turns=max_turns,
    )

    proposed, retired, identity_updated, memo_written = _count_actions(result.messages)

    logger.info(
        "Strategy reflection complete: %d rules proposed, %d retired, "
        "identity_updated=%s, memo_written=%s",
        proposed, retired, identity_updated, memo_written,
    )

    return StrategyReflectionResult(
        rules_proposed=proposed,
        rules_retired=retired,
        identity_updated=identity_updated,
        memo_written=memo_written,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        turns_used=result.turns_used,
    )
