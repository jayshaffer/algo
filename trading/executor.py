"""Trade execution and position management via Alpaca API."""

import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .db import (
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
    """
    Sync positions from Alpaca to local database.

    Returns:
        Number of positions synced
    """
    client = get_trading_client()
    positions = client.get_all_positions()

    # Get current DB positions for comparison
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
    """
    Sync open orders from Alpaca to local database.

    Returns:
        Number of orders synced
    """
    client = get_trading_client()
    orders = client.get_orders()

    # Get current DB orders for comparison
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
    side: str,  # "buy" or "sell"
    qty: Decimal,
    dry_run: bool = False
) -> OrderResult:
    """
    Execute a market order.

    Args:
        ticker: Stock symbol
        side: "buy" or "sell"
        qty: Number of shares
        dry_run: If True, simulate without executing

    Returns:
        OrderResult with execution details
    """
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
    """
    Execute a limit order.

    Args:
        ticker: Stock symbol
        side: "buy" or "sell"
        qty: Number of shares
        limit_price: Limit price
        dry_run: If True, simulate without executing

    Returns:
        OrderResult with execution details
    """
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


def get_latest_price(ticker: str) -> Optional[Decimal]:
    """Get latest quote price for a ticker."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockLatestQuoteRequest(symbol_or_symbols=ticker)

    try:
        quotes = client.get_stock_latest_quote(request)
        quote = quotes[ticker]
        # Use ask price for buys, could use mid for more accuracy
        return Decimal(str(quote.ask_price))
    except Exception:
        return None


def calculate_position_size(
    buying_power: Decimal,
    price: Decimal,
    risk_pct: float = 0.05
) -> int:
    """
    Calculate position size based on risk percentage.

    Args:
        buying_power: Available buying power
        price: Current stock price
        risk_pct: Percentage of buying power to risk (default 5%)

    Returns:
        Number of shares to buy (whole shares only)
    """
    amount_to_risk = buying_power * Decimal(str(risk_pct))
    shares = int(amount_to_risk / price)
    return max(shares, 0)
