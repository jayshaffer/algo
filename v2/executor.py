"""Trade execution and position management via Alpaca API."""

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import ROUND_DOWN, Decimal

# Alpaca documents fractional-share precision at 9 decimals. Quantize ROUND_DOWN
# before submit so a precheck-trimmed sell can never overshoot qty_available
# from sub-9-decimal noise produced by intent division.
_QTY_PRECISION = Decimal("0.000000001")


def _quantize_qty(qty: Decimal) -> Decimal:
    return qty.quantize(_QTY_PRECISION, rounding=ROUND_DOWN)

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from .database.trading_db import (
    delete_open_order,
    delete_position,
    insert_account_snapshot,
    upsert_open_order,
    upsert_position,
)
from .database.trading_db import (
    get_open_orders as db_get_open_orders,
)
from .database.trading_db import (
    get_positions as db_get_positions,
)

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: str | None
    filled_qty: Decimal | None
    filled_avg_price: Decimal | None
    error: str | None
    # T2.8: when wait_for_fill cancels a timed-out order and the post-cancel
    # re-fetch fails, we don't actually know whether a partial fill landed.
    # Setting `unknown_partial_fill=True` lets the trader treat this as a
    # "needs reconciliation" case rather than ambiguously False (= clean
    # failure) or True (= clean fill).
    unknown_partial_fill: bool = False
    # Set when Alpaca rejected the submission because the deterministic
    # client_order_id already existed broker-side (concurrent run lost the
    # dedup race). Distinct from a real execution failure so the trader
    # can log it as a benign skip rather than an error.
    duplicate_client_order_id: bool = False


def _is_duplicate_client_order_id_error(err: Exception) -> bool:
    """Detect Alpaca's response when a client_order_id collides with an
    already-submitted order. Alpaca returns HTTP 422 with a message that
    includes the literal substring "client_order_id must be unique"."""
    msg = str(err).lower()
    return "client_order_id" in msg and "unique" in msg


def _validate_alpaca_env() -> None:
    """T1.4: enforce explicit paper/prod selection.

    Pre-fix behavior silently defaulted ALPACA_BASE_URL to paper when unset and
    derived `paper=True` from the URL substring — meaning a prod key with a
    missing/wrong URL would route to the paper endpoint (Alpaca then 401s, but
    the failure is opaque). Worse, `paper=true` paired with a prod URL would
    submit live orders against a paper-flagged client.

    Skips validation when ALPACA_API_KEY is unset so module imports in
    non-trading test contexts (CLI tools that don't need Alpaca, conftest
    collection before fixtures fire) don't crash.
    """
    if not os.environ.get("ALPACA_API_KEY"):
        return

    paper_raw = os.environ.get("ALPACA_PAPER")
    if paper_raw is None:
        raise RuntimeError(
            "ALPACA_PAPER env var is required (true|false). "
            "Set explicitly to prevent silent paper/prod misrouting."
        )
    paper_raw = paper_raw.strip().lower()
    if paper_raw not in ("true", "false"):
        raise RuntimeError(
            f"ALPACA_PAPER must be 'true' or 'false', got: {paper_raw!r}"
        )
    paper = paper_raw == "true"

    base_url = os.environ.get("ALPACA_BASE_URL")
    if base_url is None:
        raise RuntimeError("ALPACA_BASE_URL is required when ALPACA_API_KEY is set")
    url_is_paper = "paper" in base_url
    if paper != url_is_paper:
        raise RuntimeError(
            f"ALPACA_PAPER={paper_raw} disagrees with ALPACA_BASE_URL={base_url!r}. "
            "Set both consistently to avoid routing prod orders to paper or vice versa."
        )


_validate_alpaca_env()


def get_trading_client() -> TradingClient:
    """Create Alpaca trading client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    # Re-validate on every client construction; cheap, and catches monkeypatched
    # env in tests that bypassed the module-load gate.
    _validate_alpaca_env()
    paper = os.environ.get("ALPACA_PAPER", "").strip().lower() == "true"
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


def get_live_available_qty(ticker: str) -> Decimal | None:
    """Return Alpaca's live qty_available for a ticker, or None if not held.

    qty_available = held_qty - qty locked by pending sell orders. This is the
    true ceiling for a new sell submission; checking it immediately before
    submit prevents 'Insufficient available shares' failures from stale DB
    state or untracked open orders.
    """
    from alpaca.common.exceptions import APIError
    client = get_trading_client()
    try:
        pos = client.get_open_position(ticker)
    except APIError as e:
        msg = str(e).lower()
        if "position does not exist" in msg or "not found" in msg or "404" in msg:
            return None
        raise
    return Decimal(pos.qty_available)


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
    """Sync the local `open_orders` table to Alpaca's set of OPEN orders.

    This table is a transient mirror of "what is currently working at the
    broker," not an audit log. `client.get_orders()` returns only open
    orders by default, so any local row whose order_id no longer appears
    in Alpaca's list is treated as resolved (filled, cancelled, expired,
    or rejected) and deleted here. Filled orders have already produced a
    `decisions` row when they came back through `wait_for_fill`, so this
    delete does not lose audit information — the historical record lives
    in `decisions`.

    T2.11: previously the docstring left the intent ambiguous, which
    invited future readers to add audit-log expectations to a function
    that's structurally a mirror.
    """
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
    dry_run: bool = False,
    simulated_price: Decimal = None,
    client_order_id: str | None = None,
) -> OrderResult:
    """Execute a market order.

    `client_order_id` (P1.6): if provided, Alpaca treats a duplicate submission
    with the same id as already-submitted (broker-side idempotency). Combined
    with the pre-submit DB dedup in trader.py, this prevents real duplicate
    orders even when our DB write fails between submission and rerun.
    """
    if dry_run:
        return OrderResult(
            success=True,
            order_id="DRY_RUN",
            filled_qty=qty,
            filled_avg_price=simulated_price,
            error=None,
        )

    try:
        client = get_trading_client()

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        order_kwargs = dict(
            symbol=ticker,
            qty=float(_quantize_qty(qty)),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        if client_order_id:
            order_kwargs["client_order_id"] = client_order_id
        order_request = MarketOrderRequest(**order_kwargs)

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
            duplicate_client_order_id=_is_duplicate_client_order_id_error(e),
        )


def execute_limit_order(
    ticker: str,
    side: str,
    qty: Decimal,
    limit_price: Decimal,
    dry_run: bool = False,
    client_order_id: str | None = None,
) -> OrderResult:
    """Execute a limit order.

    See `execute_market_order` for `client_order_id` semantics (P1.6).
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

        order_kwargs = dict(
            symbol=ticker,
            qty=float(_quantize_qty(qty)),
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=float(limit_price),
        )
        if client_order_id:
            order_kwargs["client_order_id"] = client_order_id
        order_request = LimitOrderRequest(**order_kwargs)

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
            duplicate_client_order_id=_is_duplicate_client_order_id_error(e),
        )


def wait_for_fill(
    order_id: str,
    timeout_seconds: float = 30,
    poll_interval: float = 0.5,
) -> OrderResult:
    """Poll Alpaca until order is filled, cancelled, or timeout.

    T2.8: post-cancel re-fetch failures used to silently degrade to a
    clean "Timeout" failure, hiding the possibility of a partial fill
    that we couldn't read. The result now carries `unknown_partial_fill`
    when the post-cancel state is genuinely unknown.

    T2.9: get_order_by_id is a network call; transient hiccups must not
    abort the polling loop. Each polling iteration retries the fetch a
    small number of times before treating the error as fatal.
    """
    import time

    client = get_trading_client()
    terminal_failures = {"canceled", "cancelled", "expired", "rejected", "suspended"}
    start = time.monotonic()
    consecutive_fetch_errors = 0
    fetch_retry_limit = 3

    while time.monotonic() - start < timeout_seconds:
        try:
            order = client.get_order_by_id(order_id)
        except Exception as fetch_err:
            consecutive_fetch_errors += 1
            if consecutive_fetch_errors >= fetch_retry_limit:
                logger.error(
                    "Order %s fetch failed %d times in a row: %s",
                    order_id, consecutive_fetch_errors, fetch_err,
                )
                # Fall through to the cancel + re-fetch dance below.
                break
            logger.warning(
                "Order %s fetch transient error (%d/%d): %s — retrying",
                order_id, consecutive_fetch_errors, fetch_retry_limit, fetch_err,
            )
            time.sleep(poll_interval)
            continue
        consecutive_fetch_errors = 0
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

    # P1.13: re-fetch after cancel to catch partial fills that landed before
    # the cancel took effect. The previous code returned filled_qty=None and
    # the decision row in trader.py was logged with order_id=None — the link
    # between the actual fill (visible in position sync) and the decision row
    # was lost, breaking attribution for that trade.
    try:
        order = client.get_order_by_id(order_id)
        filled_qty = Decimal(str(order.filled_qty)) if order.filled_qty else Decimal("0")
        if filled_qty > 0:
            logger.warning(
                "Order %s timed out with partial fill: %s shares @ %s",
                order_id, filled_qty, order.filled_avg_price,
            )
            return OrderResult(
                success=True,
                order_id=order_id,
                filled_qty=filled_qty,
                filled_avg_price=(
                    Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None
                ),
                error=f"Timeout with partial fill ({filled_qty} shares); order cancelled",
            )
        # T2.8: re-fetch succeeded and confirmed zero fill. This is an
        # actual clean failure.
        return OrderResult(
            success=False,
            order_id=order_id,
            filled_qty=Decimal("0"),
            filled_avg_price=None,
            error=f"Timeout waiting for fill after {timeout_seconds}s (cancel attempted)",
        )
    except Exception as fetch_err:
        # T2.8: post-cancel re-fetch failed. We do NOT know whether a
        # partial fill landed; mark the result so the trader can avoid
        # treating this as a clean miss (and skip orphan logging that
        # would assume nothing filled).
        logger.error(
            "Order %s post-cancel re-fetch failed: %s — partial fill unknown",
            order_id, fetch_err,
        )
        return OrderResult(
            success=False,
            order_id=order_id,
            filled_qty=None,
            filled_avg_price=None,
            error=(
                f"Timeout waiting for fill after {timeout_seconds}s "
                f"(cancel attempted; post-cancel re-fetch failed: {fetch_err}) "
                "— partial-fill state unknown"
            ),
            unknown_partial_fill=True,
        )


def get_latest_price(
    ticker: str,
    max_age_seconds: int = 60,
    max_spread_pct: Decimal = Decimal("0.05"),
    client: StockHistoricalDataClient = None,
) -> Decimal | None:
    """Get latest quote price for a ticker with staleness and spread validation.

    Args:
        ticker: Stock ticker symbol
        max_age_seconds: Reject quotes older than this (seconds). 0 to disable.
        max_spread_pct: Reject quotes with bid-ask spread wider than this fraction. 0 to disable.
        client: Optional pre-created StockHistoricalDataClient to reuse.

    Returns:
        Ask price as Decimal, or None if quote is stale, wide-spread, zero, or unavailable.
    """
    if client is None:
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


def get_latest_trade_price(
    ticker: str,
    client: StockHistoricalDataClient = None,
) -> Decimal | None:
    """Get the last tape-print (trade) price for a ticker.

    Use this for reference pricing (e.g. logging a HOLD decision, valuing
    positions) where we want *a* current price but don't need the tight
    bid/ask spread validation required before submitting an order. The free
    IEX feed produces wide quote spreads near the close as market makers
    withdraw, which causes get_latest_price to reject quotes even though
    the last trade print is fine — so paths that don't need spread
    discipline should call this instead.

    For order submission, keep using get_latest_price so stale/wide quotes
    still block bad fills.
    """
    if client is None:
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret_key)
    request = StockLatestTradeRequest(symbol_or_symbols=ticker)

    try:
        trades = client.get_stock_latest_trade(request)
        trade = trades[ticker]
        price = Decimal(str(trade.price))
        if price == 0:
            return None
        return price
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


def get_daily_deposit(prev_date, today_date) -> Decimal:
    """Sum cash deposits/withdrawals dated in [prev_date, today_date).

    Matches the dashboard's daily_deposit semantics: a deposit dated D shows
    up in cumulative_deposits for snapshots strictly after D, so the
    differential between two adjacent snapshots is the deposits that
    "happened" in the intervening day.
    """
    client = get_trading_client()
    total = Decimal("0")
    page_token = None
    prev_str = str(prev_date)
    today_str = str(today_date)

    while True:
        params = {"activity_types": "CSD,CSW", "page_size": 100, "direction": "asc"}
        if page_token:
            params["page_token"] = page_token

        activities = client.get("/account/activities", params)
        if not activities:
            break

        for a in activities:
            d = str(a["date"])
            if prev_str <= d < today_str:
                total += Decimal(str(a["net_amount"]))

        if len(activities) < 100:
            break
        page_token = activities[-1]["id"]

    return total
