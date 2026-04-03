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
5. **Don't over-rotate**: A single bad session doesn't warrant major strategy changes. Look for patterns across multiple sessions.

## Identity vs. Memos

The strategy identity describes WHO this system is as a trader — its style, risk philosophy, signal preferences, and core beliefs. It should be stable across sessions and NOT reference specific session numbers, individual trades, or recent events.

Session-specific observations belong in memos, not the identity.

Only update the identity when the system's fundamental character has genuinely shifted (e.g., from momentum trader to value trader, or from aggressive to conservative). Cosmetic updates ("in its 36th session") are not identity changes.

A good identity reads like a bio. A bad identity reads like a session log.

## Rule Management

Before proposing a new rule:
1. Check if an existing active rule covers the same pattern
2. If so, update the existing rule's confidence, scope, or evidence rather than creating a new one
3. Only create a new rule if the pattern is genuinely distinct from all existing rules"""


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

    # Soft guard: warn if updated within last 3 days
    if current and (date.today() - current["created_at"].date()).days < 3:
        return (
            f"Warning: Identity was updated within the last 3 days "
            f"(v{current['version']} on {current['created_at'].date()}). "
            f"Consider writing a memo instead unless the system's fundamental "
            f"character has changed. To proceed anyway, call update_strategy_identity again."
        )

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
    success = retire_strategy_rule(rule_id=rule_id, reason=reason)
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
    """Get today's session summary — recent decisions, outcomes, attribution, signal linkage."""
    logger.info("Getting session summary")
    lines = []

    decisions = get_recent_decisions(days=days)
    if decisions:
        # Fetch signal linkage for these decisions
        from .database.connection import get_cursor
        decision_ids = [d["id"] for d in decisions[:10]]
        signal_map: dict[int, list[str]] = {}
        if decision_ids:
            with get_cursor() as cur:
                cur.execute("""
                    SELECT ds.decision_id, ds.signal_type,
                           COALESCE(ns.category, ms.category, '') AS signal_category
                    FROM decision_signals ds
                    LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
                    LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
                    WHERE ds.decision_id = ANY(%s)
                """, (decision_ids,))
                for row in cur.fetchall():
                    did = row["decision_id"]
                    label = f"{row['signal_type']}"
                    if row["signal_category"]:
                        label += f":{row['signal_category']}"
                    signal_map.setdefault(did, []).append(label)

        lines.append(f"Decisions ({len(decisions)}):")
        for d in decisions[:10]:
            outcome_7d = f"{d['outcome_7d']:+.1f}%" if d.get("outcome_7d") is not None else "-"
            outcome_30d = f"{d['outcome_30d']:+.1f}%" if d.get("outcome_30d") is not None else "-"
            off_pb = " [OFF-PLAYBOOK]" if d.get("is_off_playbook") else ""
            signals = signal_map.get(d["id"], [])
            signal_str = f" signals=[{', '.join(signals)}]" if signals else " signals=[]"
            lines.append(
                f"  {d['date']} {d['action'].upper()} {d['ticker']} "
                f"7d:{outcome_7d} 30d:{outcome_30d}{off_pb}{signal_str}"
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
        "description": "Current trading identity, risk posture, biases.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_rules",
        "description": "Active strategy rules (constraints/preferences).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_history",
        "description": "Recent strategy reflection memos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Count (default: 5)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_session_summary",
        "description": "Recent decisions, outcomes, and signal attribution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback days (default: 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "update_strategy_identity",
        "description": "Update trading identity (creates new version).",
        "input_schema": {
            "type": "object",
            "properties": {
                "identity_text": {"type": "string"},
                "risk_posture": {"type": "string", "enum": ["conservative", "moderate", "aggressive"]},
                "sector_biases": {"type": "object"},
                "preferred_signals": {"type": "array", "items": {"type": "string"}},
                "avoided_signals": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["identity_text", "risk_posture", "sector_biases", "preferred_signals", "avoided_signals"],
        },
    },
    {
        "name": "propose_rule",
        "description": "Propose new rule (constraint or preference) with evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_text": {"type": "string"},
                "category": {"type": "string", "description": "e.g. news_signal:legal, position_sizing"},
                "direction": {"type": "string", "enum": ["constraint", "preference"]},
                "confidence": {"type": "number", "description": "0.0-1.0"},
                "supporting_evidence": {"type": "string"},
            },
            "required": ["rule_text", "category", "direction", "confidence", "supporting_evidence"],
        },
    },
    {
        "name": "retire_rule",
        "description": "Retire a rule no longer supported by data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["rule_id", "reason"],
        },
    },
    {
        "name": "write_strategy_memo",
        "description": "Write reflection memo. REQUIRED at end of every session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memo_type": {"type": "string", "enum": ["reflection", "rule_change", "identity_update"]},
                "content": {"type": "string", "description": "2-4 paragraphs"},
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


DEFAULT_REFLECTION_MODEL = "claude-sonnet-4-6"


def _format_trading_context(trading_result) -> str:
    """Format TradingSessionResult into a context block for the reflection agent."""
    if not trading_result:
        return ""
    lines = [
        "TODAY'S TRADING SESSION:",
        f"  Decisions made: {trading_result.decisions_made}",
        f"  Trades executed: {trading_result.trades_executed}",
        f"  Trades failed: {trading_result.trades_failed}",
        f"  Total buy value: ${float(trading_result.total_buy_value):,.2f}",
        f"  Total sell value: ${float(trading_result.total_sell_value):,.2f}",
    ]
    if trading_result.market_summary:
        lines.append(f"  Executor's market summary: {trading_result.market_summary}")
    if trading_result.risk_assessment:
        lines.append(f"  Executor's risk assessment: {trading_result.risk_assessment}")
    if trading_result.errors:
        lines.append(f"  Errors ({len(trading_result.errors)}):")
        for err in trading_result.errors[:5]:
            lines.append(f"    - {err}")
    return "\n".join(lines)


def run_strategy_reflection(
    model: str = DEFAULT_REFLECTION_MODEL,
    max_turns: int = 10,
    trading_result=None,
) -> StrategyReflectionResult:
    """Run the strategy reflection stage (Stage 4)."""
    logger.info("Starting strategy reflection (model=%s, max_turns=%d)", model, max_turns)

    client = get_claude_client()

    trading_context = _format_trading_context(trading_result)
    initial_parts = []
    if trading_context:
        initial_parts.append(trading_context)
        initial_parts.append("")
    initial_parts.append(
        "Begin your strategy reflection. Start by:\n"
        "1. Getting the current strategy identity and rules\n"
        "2. Getting the session summary (recent decisions and attribution)\n"
        "3. Getting recent strategy memos for context\n"
        "4. Analyzing what happened and making any necessary updates\n"
        "5. Writing a reflection memo\n"
    )

    result = run_agentic_loop(
        client=client,
        model=model,
        system=STRATEGY_REFLECTION_SYSTEM,
        initial_message="\n".join(initial_parts),
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
