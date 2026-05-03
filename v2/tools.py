"""Tool definitions and implementations for Claude ideation agent."""

import logging
from datetime import date, datetime
from decimal import Decimal

from .attribution import get_attribution_summary
from .context import get_macro_context, get_portfolio_context
from .agent import validate_signal_refs
from .database.trading_db import (
    close_thesis,
    get_active_strategy_rules,
    get_active_theses,
    get_current_strategy_state,
    get_macro_signals,
    get_news_signals,
    get_positions,
    get_recent_decisions,
    get_recent_strategy_memos,
    get_thesis_by_id,
    insert_thesis,
    insert_thesis_signals,
    replace_playbook_actions_atomic,
    update_thesis,
)
from .executor import get_account_info
from .market_data import format_market_snapshot, get_market_snapshot

logger = logging.getLogger(__name__)


def reset_session():
    """Reset session state. Call at start of each ideation run."""
    logger.info("Session state reset")


def _norm_ticker(ticker: str | None) -> str | None:
    """T1.3: canonicalize an LLM-emitted ticker (uppercase, strip whitespace).

    The strategist has been observed emitting "aapl" or " AAPL "; without
    normalization, downstream lookups (DB filters, position dicts, sector map)
    miss. None passes through so optional-filter handlers stay None.
    """
    if ticker is None:
        return None
    cleaned = ticker.strip().upper()
    return cleaned or None


# --- Tool Handlers ---


def tool_get_market_snapshot() -> str:
    """Get current market state."""
    logger.info("Getting market snapshot")
    try:
        snapshot = get_market_snapshot()
        return format_market_snapshot(snapshot)
    except Exception as e:
        logger.exception("Failed to get market snapshot")
        return f"Error getting market snapshot: {e}"


def tool_get_portfolio_state() -> str:
    """Get portfolio positions and account info."""
    logger.info("Getting portfolio state")
    try:
        account_info = get_account_info()
        return get_portfolio_context(account_info)
    except Exception as e:
        logger.exception("Failed to get portfolio state")
        return f"Error getting portfolio state: {e}"


def tool_get_active_theses(ticker: str | None = None) -> str:
    """Get active theses."""
    ticker = _norm_ticker(ticker)
    logger.info(f"Getting active theses (ticker filter: {ticker})")
    theses = get_active_theses(ticker=ticker)

    if not theses:
        return "No active theses."

    lines = []
    for t in theses:
        age_days = (datetime.now() - t["created_at"]).days
        lines.append(
            f"#{t['id']} {t['ticker']} {t['direction']} {t['confidence']} {age_days}d | "
            f"{t['thesis'][:120]}"
        )
        parts = []
        if t['entry_trigger']:
            parts.append(f"entry:{t['entry_trigger']}")
        if t['exit_trigger']:
            parts.append(f"exit:{t['exit_trigger']}")
        if t['invalidation']:
            parts.append(f"invalidate:{t['invalidation']}")
        if parts:
            lines.append(f"  {' | '.join(parts)}")

    return "\n".join(lines)


def _persist_signal_refs(thesis_id: int, signal_refs: list[dict] | None) -> str:
    """Validate, persist, and return a human-readable note for tool output.

    Returns a status fragment describing how many refs were stripped (if any),
    so the strategist sees its own miscites within the agentic loop and can
    correct on the next turn. Empty string if no refs given or all valid.
    """
    if not signal_refs:
        return ""
    submitted = len(signal_refs)
    valid = validate_signal_refs(signal_refs)
    insert_thesis_signals(thesis_id, valid)
    stripped = submitted - len(valid)
    if stripped:
        return (
            f" Note: {stripped} of {submitted} signal_refs were stripped as invalid "
            f"(IDs not found in DB)."
        )
    return f" Cited {len(valid)} signal(s)."


def tool_create_thesis(
    ticker: str,
    direction: str,
    thesis: str,
    entry_trigger: str,
    exit_trigger: str,
    invalidation: str,
    confidence: str,
    signal_refs: list[dict] | None = None,
) -> str:
    """Create a new thesis.

    signal_refs: list of {"type": "news_signal"|"macro_signal"|"thesis", "id": int}
    citing the evidence that justified this thesis. IDs must come from
    get_news_signals / get_macro_signals / get_active_theses output. Invalid
    IDs are stripped at write time and reported back in the tool result.
    """
    ticker = _norm_ticker(ticker) or ""
    logger.info(f"Creating thesis for {ticker} ({direction})")

    # Check for duplicates
    existing = get_active_theses(ticker=ticker)
    if existing:
        return (
            f"Error: Active thesis already exists for {ticker} "
            f"(ID {existing[0]['id']}). Update or close it first."
        )

    positions = {p["ticker"] for p in get_positions()}
    if ticker in positions:
        return f"Error: {ticker} is already in the portfolio. Cannot create thesis."

    # Create the thesis
    thesis_id = insert_thesis(
        ticker=ticker,
        direction=direction,
        thesis=thesis,
        entry_trigger=entry_trigger,
        exit_trigger=exit_trigger,
        invalidation=invalidation,
        confidence=confidence,
        source="claude_ideation",
    )

    note = _persist_signal_refs(thesis_id, signal_refs)
    logger.info(f"Created thesis ID {thesis_id} for {ticker}")
    return (
        f"Created thesis ID {thesis_id} for {ticker} "
        f"({direction}, {confidence} confidence).{note}"
    )


def tool_adopt_thesis(
    ticker: str,
    direction: str,
    thesis: str,
    exit_trigger: str,
    invalidation: str,
    confidence: str,
    signal_refs: list[dict] | None = None,
) -> str:
    """Adopt an existing portfolio position by creating a thesis for it.

    Unlike create_thesis, this REQUIRES the ticker to already be in the portfolio.
    Used to bring orphan positions under thesis management.
    """
    ticker = _norm_ticker(ticker) or ""
    logger.info(f"Adopting thesis for existing position {ticker} ({direction})")

    existing = get_active_theses(ticker=ticker)
    if existing:
        return (
            f"Error: Active thesis already exists for {ticker} "
            f"(ID {existing[0]['id']}). Update it instead."
        )

    positions = {p["ticker"] for p in get_positions()}
    if ticker not in positions:
        return f"Error: {ticker} is not in the portfolio. Use create_thesis for new ideas."

    thesis_id = insert_thesis(
        ticker=ticker,
        direction=direction,
        thesis=thesis,
        entry_trigger="Already held — adopted into thesis management",
        exit_trigger=exit_trigger,
        invalidation=invalidation,
        confidence=confidence,
        source="adoption",
    )

    note = _persist_signal_refs(thesis_id, signal_refs)
    logger.info(f"Adopted position {ticker} as thesis ID {thesis_id}")
    return (
        f"Created thesis ID {thesis_id} for {ticker} "
        f"(adopted existing position, {direction}, {confidence} confidence).{note}"
    )


def tool_update_thesis(
    thesis_id: int,
    thesis: str | None = None,
    entry_trigger: str | None = None,
    exit_trigger: str | None = None,
    invalidation: str | None = None,
    confidence: str | None = None,
    add_signal_refs: list[dict] | None = None,
) -> str:
    """Update an existing thesis.

    add_signal_refs appends new signal citations to the thesis (idempotent).
    Use it as evidence accumulates over the life of the thesis.
    """
    logger.info(f"Updating thesis ID {thesis_id}")

    has_field_updates = any(
        v is not None for v in (thesis, entry_trigger, exit_trigger, invalidation, confidence)
    )

    # T1.7: enforce input shape and existence BEFORE any DB write. The
    # add_signal_refs-only path previously skipped existence checks; an invalid
    # thesis_id would crash _persist_signal_refs with a raw FK error instead of
    # surfacing a clean tool_result the strategist can act on.
    if not has_field_updates and not add_signal_refs:
        return "Error: no updates provided (must set at least one field or add_signal_refs)"
    if get_thesis_by_id(thesis_id) is None:
        return f"Error: thesis ID {thesis_id} not found"

    if has_field_updates:
        update_thesis(
            thesis_id=thesis_id,
            thesis=thesis,
            entry_trigger=entry_trigger,
            exit_trigger=exit_trigger,
            invalidation=invalidation,
            confidence=confidence,
        )

    note = _persist_signal_refs(thesis_id, add_signal_refs)

    return f"Updated thesis ID {thesis_id}.{note}"


def tool_close_thesis(thesis_id: int, status: str, reason: str) -> str:
    """Close a thesis."""
    logger.info(f"Closing thesis ID {thesis_id} with status {status}")

    success = close_thesis(thesis_id=thesis_id, status=status, reason=reason)

    if success:
        return f"Closed thesis ID {thesis_id} with status '{status}'"
    else:
        return f"Error: Thesis ID {thesis_id} not found"


def tool_get_news_signals(ticker: str = None, days: int = 7) -> str:
    """Get recent ticker-specific news signals.

    Each line is prefixed with [#<id>] so the strategist can cite the signal
    by ID via signal_refs on create_thesis / update_thesis. IDs that don't
    appear in tool output should not be cited — they will be stripped by
    validation at thesis-creation time.
    """
    ticker = _norm_ticker(ticker)
    logger.info(f"Getting news signals (ticker: {ticker}, days: {days})")
    signals = get_news_signals(ticker=ticker, days=days)

    if not signals:
        if ticker:
            return f"No news signals for {ticker} in the last {days} days."
        return f"No news signals in the last {days} days."

    lines = []
    for s in signals:
        date_str = s["published_at"].strftime("%m-%d %H:%M")
        headline = s["headline"][:60]
        lines.append(
            f"[#{s['id']}] {date_str} {s['ticker']} "
            f"{s['category']}/{s['sentiment']}/{s['confidence']}: {headline}"
        )

    return "\n".join(lines)


def tool_get_macro_context(days: int = 7) -> str:
    """Get macro economic context."""
    logger.info(f"Getting macro context (last {days} days)")
    return get_macro_context(days=days)


def tool_get_macro_signals(days: int = 7) -> str:
    """Get recent macro signals as a list, one per line, with IDs.

    Parallels tool_get_news_signals — use this when you need IDs to cite on
    a thesis via signal_refs. tool_get_macro_context is the higher-level
    summary by category; this tool returns the raw list with IDs.
    """
    logger.info(f"Getting macro signals (last {days} days)")
    signals = get_macro_signals(days=days)

    if not signals:
        return f"No macro signals in the last {days} days."

    lines = []
    for s in signals:
        date_str = s["published_at"].strftime("%m-%d %H:%M")
        headline = s["headline"][:80]
        lines.append(
            f"[#{s['id']}] {date_str} {s['category']}/{s['sentiment']}: {headline}"
        )

    return "\n".join(lines)


def tool_get_signal_attribution() -> str:
    """Get signal attribution scores."""
    logger.info("Getting signal attribution")
    return get_attribution_summary()


def tool_get_decision_history(days: int = 30) -> str:
    """Get recent decisions with outcomes."""
    logger.info(f"Getting decision history ({days} days)")
    decisions = get_recent_decisions(days=days)

    if not decisions:
        return f"No decisions in the last {days} days."

    lines = []
    for d in decisions:
        outcome_7d = f"{d['outcome_7d']:+.1f}%" if d.get("outcome_7d") is not None else "-"
        outcome_30d = f"{d['outcome_30d']:+.1f}%" if d.get("outcome_30d") is not None else "-"
        lines.append(
            f"{d['date']} {d['action'].upper()} {d['ticker']} 7d:{outcome_7d} 30d:{outcome_30d} {d['reasoning'][:60]}"
        )

    return "\n".join(lines)


def tool_write_playbook(
    market_outlook: str,
    priority_actions: list,
    watch_list: list,
    risk_notes: str,
) -> str:
    """Write today's playbook to the database.

    V3 behavior:
    - Validates no conflicting actions (buy + sell same ticker)
    - Writes playbook row via upsert_playbook()
    - Deletes old playbook_actions for this playbook
    - Inserts new playbook_actions rows for each priority action
    - Returns confirmation with playbook ID and action count
    """
    logger.info("Writing playbook")

    # T1.3: normalize ticker on every action so persisted playbook rows are
    # canonical TICKER, not "aapl"/"AAPL "/etc. Mutates in place so the
    # downstream insert sees the cleaned values too.
    for action in priority_actions:
        norm = _norm_ticker(action.get("ticker"))
        if norm:
            action["ticker"] = norm

    # Validate no conflicting actions (buy + sell same ticker)
    # AND no duplicate (ticker, action) pairs. T1.6: a playbook with two buys
    # for AAPL was previously accepted; the executor then evaluated both and
    # could double-up size against the same idea. Rejecting at write-time
    # forces the strategist to dedupe in its own loop.
    actions_by_ticker: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for action in priority_actions:
        ticker = action.get("ticker")
        act = action.get("action")
        if ticker in actions_by_ticker and actions_by_ticker[ticker] != act:
            return (
                f"Error: Conflicting actions for {ticker} — "
                f"cannot {actions_by_ticker[ticker]} and {act} the same ticker."
            )
        pair = (ticker, act)
        if pair in seen_pairs:
            return (
                f"Error: Duplicate (ticker, action) pair for {ticker} {act}. "
                f"Each (ticker, action) must appear at most once per playbook."
            )
        seen_pairs.add(pair)
        actions_by_ticker[ticker] = act

    try:
        playbook_date = date.today()
        # P2.22: single transaction for upsert + delete + N inserts. The
        # previous flow had three separate connections, so a mid-loop failure
        # left the playbook row + a subset of actions with no rollback.
        playbook_id, action_count = replace_playbook_actions_atomic(
            playbook_date=playbook_date,
            market_outlook=market_outlook,
            priority_actions=priority_actions,
            watch_list=watch_list,
            risk_notes=risk_notes,
            actions=priority_actions,
        )
        return (
            f"Playbook written for {playbook_date} "
            f"(ID: {playbook_id}, {action_count} actions)"
        )
    except Exception as e:
        logger.exception("Failed to write playbook")
        return f"Error writing playbook: {e}"


def tool_get_strategy_identity() -> str:
    """Get the system's current strategy identity."""
    logger.info("Getting strategy identity")
    state = get_current_strategy_state()
    if state is None:
        return "No strategy identity established yet. This is the first session."

    lines = [
        f"Strategy Identity (v{state['version']}, updated {state['created_at'].strftime('%Y-%m-%d')}):",
        f"  Identity: {state['identity_text']}",
        f"  Risk Posture: {state['risk_posture']}",
        f"  Sector Biases: {state['sector_biases']}",
        f"  Preferred Signals: {state['preferred_signals']}",
        f"  Avoided Signals: {state['avoided_signals']}",
    ]
    return "\n".join(lines)


def tool_get_strategy_rules() -> str:
    """Get all active strategy rules."""
    logger.info("Getting strategy rules")
    rules = get_active_strategy_rules()
    if not rules:
        return "No active strategy rules yet."

    lines = []
    for r in rules:
        evidence = f" | {r['supporting_evidence'][:80]}" if r.get("supporting_evidence") else ""
        lines.append(
            f"#{r['id']} {r['direction']}/{r['category']} conf:{r['confidence']}: "
            f"{r['rule_text']}{evidence}"
        )
    return "\n".join(lines)


def tool_get_strategy_history(n: int = 5, full_recent: int = 2) -> str:
    """Get recent strategy memos. Last `full_recent` shown in full, older truncated to 300 chars."""
    logger.info(f"Getting strategy history (last {n})")
    memos = get_recent_strategy_memos(n=n)
    if not memos:
        return "No strategy memos yet. This is the first session."

    lines = []
    for i, m in enumerate(memos):
        if i < full_recent:
            content = m['content']
        else:
            content = m['content'][:300] + "..." if len(m['content']) > 300 else m['content']
        lines.append(f"[{m['session_date']}] {m['memo_type']}: {content}")
    return "\n".join(lines)


# --- Tool Definitions for Claude ---

TOOL_DEFINITIONS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 6,
    },
    {
        "name": "get_market_snapshot",
        "description": "Current market: sectors, indices, movers, unusual volume.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_portfolio_state",
        "description": "Positions, open orders, cash, buying power.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_active_theses",
        "description": "Active trade theses with direction, triggers, invalidation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Filter by ticker"},
            },
            "required": [],
        },
    },
    {
        "name": "create_thesis",
        "description": (
            "Create trade thesis. Rejects if ticker has active thesis or is held. "
            "IMPORTANT: Thesis text is NARRATIVE only. Do NOT include current share counts, entry prices, P&L, or any numeric state in the `thesis`, `entry_trigger`, `exit_trigger`, or `invalidation` fields — those are computed from the positions table at read time. Numeric state you embed here will drift and cause incorrect decisions. "
            "Cite the news/macro signals that justify the thesis via `signal_refs` so the executor and attribution can trace the trade back to the evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "direction": {"type": "string", "enum": ["long", "short", "avoid"]},
                "thesis": {"type": "string", "description": "Core reasoning"},
                "entry_trigger": {"type": "string", "description": "Entry conditions"},
                "exit_trigger": {"type": "string", "description": "Exit conditions"},
                "invalidation": {"type": "string", "description": "What proves thesis wrong"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "signal_refs": {
                    "type": "array",
                    "description": (
                        "Signals supporting this thesis. IDs MUST come from "
                        "get_news_signals / get_macro_signals / get_active_theses output. "
                        "Invalid IDs are silently dropped at write time and reported back "
                        "in the tool result."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["news_signal", "macro_signal", "thesis"],
                            },
                            "id": {"type": "integer"},
                        },
                        "required": ["type", "id"],
                    },
                },
            },
            "required": [
                "ticker", "direction", "thesis",
                "entry_trigger", "exit_trigger", "invalidation", "confidence",
            ],
        },
    },
    {
        "name": "adopt_thesis",
        "description": (
            "Adopt an existing portfolio position by creating a thesis. Use for orphan "
            "positions (held but no thesis). REQUIRES ticker to be in portfolio. "
            "Cite supporting signals via `signal_refs` if available — adopted positions "
            "may legitimately have none, in which case omit the field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "direction": {"type": "string", "enum": ["long", "short", "avoid"]},
                "thesis": {"type": "string", "description": "Why you believe in this position"},
                "exit_trigger": {"type": "string", "description": "When to exit"},
                "invalidation": {"type": "string", "description": "What proves thesis wrong"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "signal_refs": {
                    "type": "array",
                    "description": "Optional supporting signals (same shape as create_thesis).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["news_signal", "macro_signal", "thesis"],
                            },
                            "id": {"type": "integer"},
                        },
                        "required": ["type", "id"],
                    },
                },
            },
            "required": ["ticker", "direction", "thesis", "exit_trigger", "invalidation", "confidence"],
        },
    },
    {
        "name": "update_thesis",
        "description": (
            "Update thesis fields and/or append new signal citations. "
            "IMPORTANT: Thesis text is NARRATIVE only. Do NOT include current share counts, entry prices, P&L, or any numeric state in the `thesis`, `entry_trigger`, `exit_trigger`, or `invalidation` fields — those are computed from the positions table at read time. Numeric state you embed here will drift and cause incorrect decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thesis_id": {"type": "integer"},
                "thesis": {"type": "string"},
                "entry_trigger": {"type": "string"},
                "exit_trigger": {"type": "string"},
                "invalidation": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "add_signal_refs": {
                    "type": "array",
                    "description": (
                        "Signals to append to the thesis as new evidence accumulates. "
                        "Idempotent — previously cited signals are not duplicated."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["news_signal", "macro_signal", "thesis"],
                            },
                            "id": {"type": "integer"},
                        },
                        "required": ["type", "id"],
                    },
                },
            },
            "required": ["thesis_id"],
        },
    },
    {
        "name": "close_thesis",
        "description": "Close thesis as invalidated, expired, or executed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thesis_id": {"type": "integer"},
                "status": {"type": "string", "enum": ["invalidated", "expired", "executed"]},
                "reason": {"type": "string"},
            },
            "required": ["thesis_id", "status", "reason"],
        },
    },
    {
        "name": "get_news_signals",
        "description": "Recent ticker news signals: headlines, sentiment, category, confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Filter by ticker"},
                "days": {"type": "integer", "description": "Lookback days (default: 7)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_macro_context",
        "description": "Macro signals: Fed, trade, geopolitical, sector trends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback days (default: 7)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_macro_signals",
        "description": (
            "Recent macro signals as a list with IDs. Use this to find macro signal "
            "IDs to cite via signal_refs on create_thesis / update_thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback days (default: 7)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_signal_attribution",
        "description": "Win rates by signal type from historical outcomes.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_decision_history",
        "description": "Recent decisions with 7d/30d P&L outcomes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback days (default: 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "write_playbook",
        "description": (
            "Write today's playbook for the executor. REQUIRED every session. "
            "You author intents — NOT share counts. The trader resolves intents "
            "to exact shares against live portfolio state at execution time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "market_outlook": {"type": "string"},
                "priority_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "action": {"type": "string", "enum": ["buy", "sell"]},
                            "thesis_id": {"type": "integer"},
                            "reasoning": {"type": "string"},
                            "intent_type": {
                                "type": "string",
                                "enum": [
                                    "exit_full", "exit_partial_pct", "exit_dollar", "trim_to_portfolio_pct",
                                    "invest_dollar", "invest_portfolio_pct", "invest_buying_power_pct", "add_to_target_pct",
                                ],
                            },
                            "intent_magnitude": {"type": "number"},
                            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                        "required": ["ticker", "action", "reasoning", "confidence", "intent_type"],
                    },
                },
                "watch_list": {"type": "array", "items": {"type": "string"}},
                "risk_notes": {"type": "string"},
            },
            "required": ["market_outlook", "priority_actions", "watch_list", "risk_notes"],
        },
    },
    {
        "name": "get_strategy_identity",
        "description": "Trading identity: risk posture, sector biases, signal preferences.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_rules",
        "description": "Active strategy rules (constraints and preferences).",
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
]


TOOL_HANDLERS = {
    "get_market_snapshot": tool_get_market_snapshot,
    "get_portfolio_state": tool_get_portfolio_state,
    "get_active_theses": tool_get_active_theses,
    "create_thesis": tool_create_thesis,
    "adopt_thesis": tool_adopt_thesis,
    "update_thesis": tool_update_thesis,
    "close_thesis": tool_close_thesis,
    "get_news_signals": tool_get_news_signals,
    "get_macro_context": tool_get_macro_context,
    "get_macro_signals": tool_get_macro_signals,
    "get_signal_attribution": tool_get_signal_attribution,
    "get_decision_history": tool_get_decision_history,
    "write_playbook": tool_write_playbook,
    "get_strategy_identity": tool_get_strategy_identity,
    "get_strategy_rules": tool_get_strategy_rules,
    "get_strategy_history": tool_get_strategy_history,
}
