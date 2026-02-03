"""Context builder for trading agent - aggregates signals into compressed format."""

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from .db import (
    get_positions,
    get_open_orders,
    get_news_signals,
    get_macro_signals,
    get_recent_decisions,
    get_account_snapshots,
    get_active_theses,
)


def get_portfolio_context(account_info: dict) -> str:
    """
    Build portfolio context section.

    Args:
        account_info: Dict with cash, portfolio_value, buying_power from Alpaca

    Returns:
        Formatted portfolio context string
    """
    lines = ["Current Portfolio:"]

    # Get positions from database
    positions = get_positions()

    if positions:
        for pos in positions:
            ticker = pos["ticker"]
            shares = pos["shares"]
            avg_cost = pos["avg_cost"]
            lines.append(f"- {ticker}: {shares} shares @ ${avg_cost:.2f} avg")
    else:
        lines.append("- No open positions")

    # Add open orders
    orders = get_open_orders()
    if orders:
        lines.append("")
        lines.append("Open Orders:")
        for order in orders:
            ticker = order["ticker"]
            side = order["side"].upper()
            qty = order["qty"]
            filled = order["filled_qty"]
            order_type = order["order_type"]
            status = order["status"]

            order_desc = f"- {side} {qty} {ticker} ({order_type})"
            if order.get("limit_price"):
                order_desc += f" @ ${float(order['limit_price']):.2f}"
            if filled and filled > 0:
                order_desc += f" [{filled}/{qty} filled]"
            order_desc += f" - {status}"
            lines.append(order_desc)

    # Add cash/buying power
    cash = account_info.get("cash", 0)
    buying_power = account_info.get("buying_power", 0)
    lines.append(f"- Cash: ${float(cash):,.2f}")
    lines.append(f"- Buying Power: ${float(buying_power):,.2f}")

    return "\n".join(lines)


def get_macro_context(days: int = 7) -> str:
    """
    Build macro context section from recent macro signals.

    Args:
        days: How many days of signals to include

    Returns:
        Formatted macro context string
    """
    signals = get_macro_signals(days=days)

    if not signals:
        return "Macro Context:\n- No significant macro news"

    # Group by category
    by_category = defaultdict(list)
    for signal in signals:
        cat = signal["category"]
        by_category[cat].append(signal)

    lines = ["Macro Context:"]

    # Summarize each category
    category_labels = {
        "fed": "Fed",
        "trade": "Trade",
        "regulation": "Regulation",
        "geopolitical": "Geopolitical",
        "fiscal": "Fiscal",
        "election": "Election",
        "sector": "Sector",
    }

    for cat, cat_signals in by_category.items():
        label = category_labels.get(cat, cat.title())

        # Get most recent signal for summary
        latest = cat_signals[0]
        sentiment = latest["sentiment"]
        headline = latest["headline"][:60] + "..." if len(latest["headline"]) > 60 else latest["headline"]

        # Count sentiment distribution
        sentiments = [s["sentiment"] for s in cat_signals]
        bullish = sentiments.count("bullish")
        bearish = sentiments.count("bearish")

        sentiment_note = f"({sentiment})"
        if len(cat_signals) > 1:
            sentiment_note = f"({bullish} bullish, {bearish} bearish)"

        lines.append(f"- {label}: {headline} {sentiment_note}")

    return "\n".join(lines)


def get_ticker_signals_context(days: int = 1) -> str:
    """
    Build today's ticker signals section.

    Args:
        days: How many days of signals (default 1 for today)

    Returns:
        Formatted signals context string
    """
    signals = get_news_signals(days=days)

    if not signals:
        return "Today's Signals:\n- No ticker-specific signals"

    # Group by ticker
    by_ticker = defaultdict(list)
    for signal in signals:
        by_ticker[signal["ticker"]].append(signal)

    lines = ["Today's Signals:"]

    for ticker, ticker_signals in by_ticker.items():
        sentiments = [s["sentiment"] for s in ticker_signals]
        bullish = sentiments.count("bullish")
        bearish = sentiments.count("bearish")
        neutral = sentiments.count("neutral")

        # Build summary
        parts = []
        if bullish:
            categories = [s["category"] for s in ticker_signals if s["sentiment"] == "bullish"]
            parts.append(f"{bullish} bullish ({', '.join(set(categories))})")
        if bearish:
            categories = [s["category"] for s in ticker_signals if s["sentiment"] == "bearish"]
            parts.append(f"{bearish} bearish ({', '.join(set(categories))})")
        if neutral:
            parts.append(f"{neutral} neutral")

        lines.append(f"- {ticker}: {', '.join(parts)}")

    return "\n".join(lines)


def get_signal_trend_context(days: int = 7) -> str:
    """
    Build 7-day signal trend section.

    Args:
        days: Days to analyze

    Returns:
        Formatted trend context string
    """
    signals = get_news_signals(days=days)

    if not signals:
        return "7-Day Signal Trend:\n- No recent signals"

    # Group by ticker
    by_ticker = defaultdict(list)
    for signal in signals:
        by_ticker[signal["ticker"]].append(signal)

    lines = ["7-Day Signal Trend:"]

    for ticker, ticker_signals in by_ticker.items():
        sentiments = [s["sentiment"] for s in ticker_signals]
        bullish = sentiments.count("bullish")
        bearish = sentiments.count("bearish")

        if bullish == 0 and bearish == 0:
            lines.append(f"- {ticker}: neutral, no significant news")
        else:
            lines.append(f"- {ticker}: {bullish} bullish, {bearish} bearish")

    return "\n".join(lines)


def get_decision_outcomes_context(days: int = 30) -> str:
    """
    Build recent decision outcomes section.

    Args:
        days: Days of decisions to include

    Returns:
        Formatted outcomes context string
    """
    decisions = get_recent_decisions(days=days)

    if not decisions:
        return "Recent Decision Outcomes:\n- No recent decisions"

    lines = ["Recent Decision Outcomes:"]

    # Only show decisions with outcomes
    with_outcomes = [d for d in decisions if d.get("outcome_7d") is not None]

    if not with_outcomes:
        lines.append("- Decisions pending outcome measurement")
        return "\n".join(lines)

    for decision in with_outcomes[:5]:  # Last 5 with outcomes
        action = decision["action"].upper()
        ticker = decision["ticker"]
        outcome_7d = decision["outcome_7d"]
        decision_date = decision["date"]

        outcome_str = f"+{outcome_7d:.1f}%" if outcome_7d >= 0 else f"{outcome_7d:.1f}%"
        lines.append(f"- {decision_date} {action} {ticker}: {outcome_str} (7d)")

    return "\n".join(lines)


def get_strategy_context() -> str:
    """
    Build current strategy section.

    Returns:
        Formatted strategy context string
    """
    from .db import get_cursor

    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM strategy
            ORDER BY date DESC
            LIMIT 1
        """)
        strategy = cur.fetchone()

    if not strategy:
        return "Current Strategy:\nNo strategy defined. Default: conservative, broad market focus."

    lines = ["Current Strategy:"]

    if strategy.get("description"):
        lines.append(strategy["description"])

    if strategy.get("risk_tolerance"):
        lines.append(f"Risk tolerance: {strategy['risk_tolerance']}")

    if strategy.get("focus_sectors"):
        sectors = ", ".join(strategy["focus_sectors"])
        lines.append(f"Focus sectors: {sectors}")

    if strategy.get("watchlist"):
        tickers = ", ".join(strategy["watchlist"])
        lines.append(f"Watchlist: {tickers}")

    return "\n".join(lines)


def get_theses_context() -> str:
    """
    Build active theses section for trading context.

    Returns:
        Formatted theses context string
    """
    from datetime import datetime

    theses = get_active_theses()

    if not theses:
        return "Active Theses:\n- No active trade theses"

    lines = ["Active Theses:"]

    for thesis in theses:
        ticker = thesis["ticker"]
        direction = thesis["direction"]
        confidence = thesis["confidence"]
        age_days = (datetime.now() - thesis["created_at"]).days

        lines.append(f"- {ticker} ({direction}, {confidence} confidence, {age_days}d old)")
        lines.append(f"  Thesis: {thesis['thesis']}")

        if thesis["entry_trigger"]:
            lines.append(f"  Entry trigger: {thesis['entry_trigger']}")

        if thesis["exit_trigger"]:
            lines.append(f"  Exit trigger: {thesis['exit_trigger']}")

        if thesis["invalidation"]:
            lines.append(f"  Invalidation: {thesis['invalidation']}")

    return "\n".join(lines)


def build_trading_context(account_info: dict) -> str:
    """
    Build complete compressed context for trading agent.

    Args:
        account_info: Dict with cash, portfolio_value, buying_power from Alpaca

    Returns:
        Complete formatted context string
    """
    sections = [
        get_portfolio_context(account_info),
        "",
        get_macro_context(days=7),
        "",
        get_theses_context(),
        "",
        get_ticker_signals_context(days=1),
        "",
        get_signal_trend_context(days=7),
        "",
        get_decision_outcomes_context(days=30),
        "",
        get_strategy_context(),
    ]

    return "\n".join(sections)
