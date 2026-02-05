"""Tool definitions and implementations for Claude ideation agent."""

import logging
from datetime import datetime
from typing import Optional

from .market_data import get_market_snapshot, format_market_snapshot
from .context import get_portfolio_context, get_macro_context
from .executor import get_account_info
from .db import (
    get_active_theses,
    get_news_signals,
    insert_thesis,
    update_thesis,
    close_thesis,
    get_positions,
)

logger = logging.getLogger(__name__)


def reset_session():
    """Reset session state. Call at start of each ideation run."""
    logger.info("Session state reset")


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


def tool_get_active_theses(ticker: Optional[str] = None) -> str:
    """Get active theses."""
    logger.info(f"Getting active theses (ticker filter: {ticker})")
    theses = get_active_theses(ticker=ticker)

    if not theses:
        return "No active theses."

    lines = []
    for t in theses:
        age_days = (datetime.now() - t["created_at"]).days
        lines.append(
            f"ID {t['id']}: {t['ticker']} ({t['direction']}) - "
            f"{t['confidence']} confidence, {age_days}d old"
        )
        lines.append(f"  Thesis: {t['thesis']}")
        lines.append(f"  Entry: {t['entry_trigger'] or 'N/A'}")
        lines.append(f"  Exit: {t['exit_trigger'] or 'N/A'}")
        lines.append(f"  Invalidation: {t['invalidation'] or 'N/A'}")
        lines.append("")

    return "\n".join(lines)


def tool_create_thesis(
    ticker: str,
    direction: str,
    thesis: str,
    entry_trigger: str,
    exit_trigger: str,
    invalidation: str,
    confidence: str,
) -> str:
    """Create a new thesis."""
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

    logger.info(f"Created thesis ID {thesis_id} for {ticker}")
    return f"Created thesis ID {thesis_id} for {ticker} ({direction}, {confidence} confidence)"


def tool_update_thesis(
    thesis_id: int,
    thesis: Optional[str] = None,
    entry_trigger: Optional[str] = None,
    exit_trigger: Optional[str] = None,
    invalidation: Optional[str] = None,
    confidence: Optional[str] = None,
) -> str:
    """Update an existing thesis."""
    logger.info(f"Updating thesis ID {thesis_id}")

    success = update_thesis(
        thesis_id=thesis_id,
        thesis=thesis,
        entry_trigger=entry_trigger,
        exit_trigger=exit_trigger,
        invalidation=invalidation,
        confidence=confidence,
    )

    if success:
        return f"Updated thesis ID {thesis_id}"
    else:
        return f"Error: Thesis ID {thesis_id} not found or no updates provided"


def tool_close_thesis(thesis_id: int, status: str, reason: str) -> str:
    """Close a thesis."""
    logger.info(f"Closing thesis ID {thesis_id} with status {status}")

    success = close_thesis(thesis_id=thesis_id, status=status, reason=reason)

    if success:
        return f"Closed thesis ID {thesis_id} with status '{status}'"
    else:
        return f"Error: Thesis ID {thesis_id} not found"


def tool_get_news_signals(ticker: str = None, days: int = 7) -> str:
    """Get recent ticker-specific news signals."""
    logger.info(f"Getting news signals (ticker: {ticker}, days: {days})")
    signals = get_news_signals(ticker=ticker, days=days)

    if not signals:
        if ticker:
            return f"No news signals for {ticker} in the last {days} days."
        return f"No news signals in the last {days} days."

    lines = []
    for s in signals:
        date_str = s["published_at"].strftime("%Y-%m-%d %H:%M")
        headline = s["headline"][:80] + "..." if len(s["headline"]) > 80 else s["headline"]
        lines.append(
            f"- [{date_str}] {s['ticker']} ({s['category']}, {s['sentiment']}, "
            f"confidence: {s['confidence']}): {headline}"
        )

    return "\n".join(lines)


def tool_get_macro_context(days: int = 7) -> str:
    """Get macro economic context."""
    logger.info(f"Getting macro context (last {days} days)")
    return get_macro_context(days=days)


# --- Tool Definitions for Claude ---

TOOL_DEFINITIONS = [
    {
        "name": "get_market_snapshot",
        "description": (
            "Get current market state including sector performance, "
            "major indices, top gainers/losers, and unusual volume stocks. "
            "Use this to identify sectors or stocks worth researching."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_portfolio_state",
        "description": (
            "Get current portfolio positions, open orders, cash balance, "
            "and buying power. Use this to understand current holdings "
            "and available capital."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_active_theses",
        "description": (
            "Get all active trade theses, optionally filtered by ticker. "
            "Returns thesis details including direction, confidence, "
            "entry/exit triggers, and invalidation criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Filter to specific ticker (optional)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_thesis",
        "description": (
            "Create a new trade thesis. Will reject if ticker already has an active "
            "thesis or is in the portfolio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "direction": {
                    "type": "string",
                    "enum": ["long", "short", "avoid"],
                    "description": "Trade direction",
                },
                "thesis": {
                    "type": "string",
                    "description": "Core reasoning for the trade",
                },
                "entry_trigger": {
                    "type": "string",
                    "description": "Specific entry conditions (price levels, events)",
                },
                "exit_trigger": {
                    "type": "string",
                    "description": "Target or stop conditions",
                },
                "invalidation": {
                    "type": "string",
                    "description": "What would prove the thesis wrong",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level",
                },
            },
            "required": [
                "ticker",
                "direction",
                "thesis",
                "entry_trigger",
                "exit_trigger",
                "invalidation",
                "confidence",
            ],
        },
    },
    {
        "name": "update_thesis",
        "description": (
            "Update an existing thesis with new information. "
            "Only provide fields you want to change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thesis_id": {"type": "integer", "description": "ID of thesis to update"},
                "thesis": {"type": "string", "description": "Updated reasoning"},
                "entry_trigger": {"type": "string", "description": "Updated entry trigger"},
                "exit_trigger": {"type": "string", "description": "Updated exit trigger"},
                "invalidation": {"type": "string", "description": "Updated invalidation criteria"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Updated confidence level",
                },
            },
            "required": ["thesis_id"],
        },
    },
    {
        "name": "close_thesis",
        "description": (
            "Close a thesis that is no longer valid. "
            "Use 'invalidated' if the invalidation criteria were met, "
            "'expired' if the thesis aged out without action, "
            "'executed' if it was acted upon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thesis_id": {"type": "integer", "description": "ID of thesis to close"},
                "status": {
                    "type": "string",
                    "enum": ["invalidated", "expired", "executed"],
                    "description": "Reason for closure",
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation for closure",
                },
            },
            "required": ["thesis_id", "status", "reason"],
        },
    },
    {
        "name": "get_news_signals",
        "description": (
            "Get recent ticker-specific news signals including headlines, "
            "sentiment (bullish/bearish/neutral), category (news/earnings/product), "
            "and confidence scores. Optionally filter by ticker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Filter to a specific ticker (optional)",
                },
                "days": {
                    "type": "integer",
                    "description": "Look back period in days (default: 7)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_macro_context",
        "description": (
            "Get recent macro economic signals including Fed policy, "
            "trade news, geopolitical events, and sector trends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look back period in days (default: 7)",
                },
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
    "update_thesis": tool_update_thesis,
    "close_thesis": tool_close_thesis,
    "get_news_signals": tool_get_news_signals,
    "get_macro_context": tool_get_macro_context,
}
