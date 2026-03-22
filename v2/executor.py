"""Trade execution and position management via Alpaca API."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

logger = logging.getLogger(__name__)

from .database.trading_db import (
    upsert_position,
    delete_position,
    insert_account_snapshot,
    get_positions as db_get_positions,
    upsert_open_order,
    delete_open_order,
    get_open_orders as db_get_open_orders,
)


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: Optional[str]
    filled_qty: Optional[Decimal]
    filled_avg_price: Optional[Decimal]
    error: Optional[str]


def get_trading_client() -> TradingClient:
    """Create Alpaca trading client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    paper = "paper" in base_url
    return TradingClient(api_key, secret_key, paper=paper)


def get_account_info() -> dict:
    """Get current account information from Alpaca."""
    client = get_trading_client()
    account = client.get_account()

    return {
        "account_id": account.id,
        "status": account.status,
        "cash": Decimal(account.cash),
        "portfolio_value": Decimal(account.portfolio_value),
        "buying_power": Decimal(account.buying_power),
        "long_market_value": Decimal(account.long_market_value),
        "short_market_value": Decimal(account.short_market_value),
        "equity": Decimal(account.equity),
        "daytrade_count": account.daytrade_count,
        "pattern_day_trader": account.pattern_day_trader,
    }


def is_market_open() -> bool:
    """Check if the market is currently open via Alpaca clock API."""
    client = get_trading_client()
    clock = client.get_clock()
    return clock.is_open


def take_account_snapshot() -> int:
    """Take a daily snapshot of account state and store in database."""
    info = get_account_info()

    return insert_account_snapshot(
        snapshot_date=date.today(),
        cash=info["cash"],
        portfolio_value=info["portfolio_value"],
        buying_power=info["buying_power"],
        long_market_value=info["long_market_value"],
        short_market_value=info["short_market_value"],
    )


def sync_positions_from_alpaca() -> int:
    """Sync positions from Alpaca to local database."""
    client = get_trading_client()
    positions = client.get_all_positions()

    db_positions = {p["ticker"]: p for p in db_get_positions()}

    synced = 0
    alpaca_tickers = set()

    for pos in positions:
        ticker = pos.symbol
        alpaca_tickers.add(ticker)

        upsert_position(
            ticker=ticker,
            shares=Decimal(pos.qty),
            avg_cost=Decimal(pos.avg_entry_price),
        )
        synced += 1

    # Remove positions no longer in Alpaca
    for ticker in db_positions:
        if ticker not in alpaca_tickers:
            delete_position(ticker)

    return synced


def sync_orders_from_alpaca() -> int:
    """Sync open orders from Alpaca to local database."""
    client = get_trading_client()
    orders = client.get_orders()

    db_orders = {o["order_id"]: o for o in db_get_open_orders()}

    synced = 0
    alpaca_order_ids = set()

    for order in orders:
        order_id = str(order.id)
        alpaca_order_ids.add(order_id)

        upsert_open_order(
            order_id=order_id,
            ticker=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            qty=Decimal(str(order.qty)),
            filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else Decimal(0),
            limit_price=Decimal(str(order.limit_price)) if order.limit_price else None,
            stop_price=Decimal(str(order.stop_price)) if order.stop_price else None,
            status=order.status.value,
            submitted_at=order.submitted_at,
        )
        synced += 1

    # Remove orders no longer in Alpaca
    for order_id in db_orders:
        if order_id not in alpaca_order_ids:
            delete_open_order(order_id)

    return synced


def execute_market_order(
    ticker: str,
    side: str,
    qty: Decimal,
    dry_run: bool = False
) -> OrderResult:
    """Execute a market order."""
    if dry_run:
        return OrderResult(
            success=True,
            order_id="DRY_RUN",
            filled_qty=qty,
            filled_avg_price=None,
            error=None,
        )

    try:
        client = get_trading_client()

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=float(qty),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = client.submit_order(order_request)

        return OrderResult(
            success=True,
            order_id=str(order.id),
            filled_qty=Decimal(order.filled_qty) if order.filled_qty else None,
            filled_avg_price=Decimal(order.filled_avg_price) if order.filled_avg_price else None,
            error=None,
        )

    except Exception as e:
        return OrderResult(
            success=False,
            order_id=None,
            filled_qty=None,
            filled_avg_price=None,
            error=str(e),
        )


def execute_limit_order(
    ticker: str,
    side: str,
    qty: Decimal,
    limit_price: Decimal,
    dry_run: bool = False
) -> OrderResult:
    """Execute a limit order."""
    if dry_run:
        return OrderResult(
            success=True,
            order_id="DRY_RUN",
            filled_qty=qty,
            filled_avg_price=limit_price,
            error=None,
        )

    try:
        client = get_trading_client()

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        order_request = LimitOrderRequest(
            symbol=ticker,
            qty=float(qty),
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=float(limit_price),
        )

        order = client.submit_order(order_request)

        return OrderResult(
            success=True,
            order_id=str(order.id),
            filled_qty=Decimal(order.filled_qty) if order.filled_qty else None,
            filled_avg_price=Decimal(order.filled_avg_price) if order.filled_avg_price else None,
            error=None,
        )

    except Exception as e:
        return OrderResult(
            success=False,
            order_id=None,
            filled_qty=None,
            filled_avg_price=None,
            error=str(e),
        )


def wait_for_fill(
    order_id: str,
    timeout_seconds: float = 30,
    poll_interval: float = 0.5,
) -> OrderResult:
    """Poll Alpaca until order is filled, cancelled, or timeout."""
    import time

    client = get_trading_client()
    terminal_failures = {"canceled", "cancelled", "expired", "rejected", "suspended"}
    start = time.monotonic()

    while time.monotonic() - start < timeout_seconds:
        order = client.get_order_by_id(order_id)
        status = order.status.value if hasattr(order.status, "value") else str(order.status)

        if status == "filled":
            return OrderResult(
                success=True,
                order_id=order_id,
                filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else None,
                filled_avg_price=Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
                error=None,
            )

        if status in terminal_failures:
            return OrderResult(
                success=False,
                order_id=order_id,
                filled_qty=None,
                filled_avg_price=None,
                error=f"Order {status}",
            )

        time.sleep(poll_interval)

    # Timeout — attempt to cancel the orphaned order
    try:
        client.cancel_order_by_id(order_id)
        logger.warning("Order %s timed out after %.0fs — cancelled", order_id, timeout_seconds)
    except Exception as cancel_err:
        logger.error("Order %s timed out and cancel failed: %s — order may still be live", order_id, cancel_err)

    return OrderResult(
        success=False,
        order_id=order_id,
        filled_qty=None,
        filled_avg_price=None,
        error=f"Timeout waiting for fill after {timeout_seconds}s (cancel attempted)",
    )


def get_latest_price(
    ticker: str,
    max_age_seconds: int = 60,
    max_spread_pct: Decimal = Decimal("0.05"),
) -> Optional[Decimal]:
    """Get latest quote price for a ticker with staleness and spread validation.

    Args:
        ticker: Stock ticker symbol
        max_age_seconds: Reject quotes older than this (seconds). 0 to disable.
        max_spread_pct: Reject quotes with bid-ask spread wider than this fraction. 0 to disable.

    Returns:
        Ask price as Decimal, or None if quote is stale, wide-spread, zero, or unavailable.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockLatestQuoteRequest(symbol_or_symbols=ticker)

    try:
        quotes = client.get_stock_latest_quote(request)
        quote = quotes[ticker]

        ask = Decimal(str(quote.ask_price))
        if ask == 0:
            return None

        # Staleness check
        if max_age_seconds > 0 and hasattr(quote, "timestamp") and quote.timestamp:
            from datetime import timezone
            age = (datetime.now(timezone.utc) - quote.timestamp).total_seconds()
            if age > max_age_seconds:
                logger.warning("%s: quote is %.0fs old (max %ds) — rejecting", ticker, age, max_age_seconds)
                return None

        # Spread check
        bid = Decimal(str(quote.bid_price)) if hasattr(quote, "bid_price") else Decimal(0)
        if max_spread_pct > 0 and bid > 0:
            spread_pct = (ask - bid) / bid
            if spread_pct > max_spread_pct:
                logger.warning("%s: spread %.2f%% exceeds max %.2f%% — rejecting", ticker, float(spread_pct) * 100, float(max_spread_pct) * 100)
                return None

        return ask
    except Exception:
        return None


def get_net_deposits() -> Decimal:
    """Get total net cash deposited into the account from Alpaca activities.

    Sums all CSD (cash deposit, positive) and CSW (cash withdrawal, negative)
    activities to determine total capital contributed.
    """
    client = get_trading_client()
    total = Decimal("0")
    page_token = None

    while True:
        params = {"activity_types": "CSD,CSW", "page_size": 100, "direction": "asc"}
        if page_token:
            params["page_token"] = page_token

        activities = client.get("/account/activities", params)
        if not activities:
            break

        for a in activities:
            total += Decimal(str(a["net_amount"]))

        if len(activities) < 100:
            break
        page_token = activities[-1]["id"]

    return total


def calculate_position_size(
    buying_power: Decimal,
    price: Decimal,
    risk_pct: float = 0.05
) -> float:
    """Calculate position size based on risk percentage."""
    amount_to_risk = buying_power * Decimal(str(risk_pct))
    shares = float(amount_to_risk / price)
    return round(max(shares, 0), 4)
