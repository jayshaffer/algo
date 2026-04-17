"""SPY benchmark data fetching and alpha computation for the local dashboard."""

import logging
import os
import time
from datetime import date, datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)


def _twr(window):
    """Time-weighted return (%) chaining sub-period returns, netting out deposits.

    Each sub-period: growth *= V_end / (V_prev + deposit_delta), where
    deposit_delta = cumulative_deposits[i] - cumulative_deposits[i-1].
    """
    growth = 1.0
    for i in range(1, len(window)):
        prev_val = float(window[i - 1]["portfolio_value"])
        prev_dep = float(window[i - 1].get("cumulative_deposits", prev_val))
        curr_dep = float(window[i].get("cumulative_deposits", prev_dep))
        deposit = curr_dep - prev_dep
        start_capital = prev_val + deposit
        if start_capital > 0:
            growth *= float(window[i]["portfolio_value"]) / start_capital
    return (growth - 1) * 100


def compute_alpha(snapshots, benchmark):
    """Compute portfolio alpha vs SPY benchmark.

    When snapshots carry `cumulative_deposits`, portfolio return is computed as
    time-weighted return (TWR) — netting out deposit effects. Otherwise falls
    back to simple `(end - start) / start`.

    Returns {"portfolio_return", "spy_return", "alpha"} or None.
    """
    if not snapshots or len(snapshots) < 2 or not benchmark:
        return None

    spy_map = {b["date"]: b["close"] for b in benchmark}

    first_idx = None
    for i, snap in enumerate(snapshots):
        if str(snap["date"]) in spy_map:
            first_idx = i
            break

    last_idx = None
    for i in range(len(snapshots) - 1, -1, -1):
        if str(snapshots[i]["date"]) in spy_map:
            last_idx = i
            break

    if first_idx is None or last_idx is None or first_idx >= last_idx:
        return None

    spy_start = spy_map[str(snapshots[first_idx]["date"])]
    spy_end = spy_map[str(snapshots[last_idx]["date"])]
    if spy_start == 0:
        return None
    spy_return = ((spy_end - spy_start) / spy_start) * 100

    window = snapshots[first_idx:last_idx + 1]
    if any("cumulative_deposits" in s for s in window):
        portfolio_return = _twr(window)
    else:
        port_start = float(window[0]["portfolio_value"])
        port_end = float(window[-1]["portfolio_value"])
        if port_start == 0:
            return None
        portfolio_return = ((port_end - port_start) / port_start) * 100

    return {
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
        "alpha": portfolio_return - spy_return,
    }


def enrich_snapshots_with_deposits(snapshots, deposit_history):
    """Return a new list of snapshots with `cumulative_deposits` added to each.

    - With no deposit history, uses the first snapshot's portfolio_value as the
      deposit base (no deposits inflate returns, TWR degrades to simple return).
    - Deposits are credited to the first snapshot on or after the day they settle.
    - Deposits on or before the first snapshot are credited to all rows.
    Does not mutate the input.
    """
    if not snapshots:
        return []

    result = [dict(s) for s in snapshots]

    if not deposit_history:
        base = float(result[0]["portfolio_value"])
        for s in result:
            s["cumulative_deposits"] = base
        return result

    sorted_deposits = sorted(deposit_history, key=lambda x: str(x["date"]))
    cum = 0.0
    dep_idx = 0
    for s in result:
        snap_date = str(s["date"])
        while dep_idx < len(sorted_deposits) and str(sorted_deposits[dep_idx]["date"]) <= snap_date:
            cum += float(sorted_deposits[dep_idx]["amount"])
            dep_idx += 1
        s["cumulative_deposits"] = cum

    return result


_TTL_SECONDS = 900  # 15 minutes
# Cache reuse is within-day only: the key is (start, end) and both ends roll
# forward daily as new snapshots land, so cross-day reuse is not a goal. The
# TTL exists to prevent hammering Alpaca on repeated refreshes within a session.
_cache: dict[tuple[date, date], tuple[float, list[dict]]] = {}


def _clear_cache():
    _cache.clear()


def get_spy_benchmark(start: date, end: date) -> list[dict]:
    """Fetch SPY daily bars from Alpaca with in-memory TTL cache.

    Returns list of {"date": "YYYY-MM-DD", "close": float}, or [] on error.
    """
    key = (start, end)
    now = time.time()

    cached = _cache.get(key)
    if cached and cached[0] > now:
        return cached[1]

    try:
        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret_key)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.max.time()),
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request)
        spy_bars = list(bars["SPY"])

        if not spy_bars:
            return []

        result = [
            {"date": bar.timestamp.strftime("%Y-%m-%d"), "close": float(bar.close)}
            for bar in spy_bars
        ]
        _cache[key] = (now + _TTL_SECONDS, result)
        return result
    except Exception:
        logger.warning("Failed to fetch SPY benchmark data", exc_info=True)
        return []


_deposit_cache: tuple[float, list[dict]] | None = None


def _clear_deposit_cache():
    global _deposit_cache
    _deposit_cache = None


def get_deposit_history() -> list[dict]:
    """Fetch cash deposit/withdrawal history from Alpaca with TTL cache.

    Returns list of {"date": "YYYY-MM-DD", "amount": float} sorted by date,
    or [] on error.
    """
    global _deposit_cache
    now = time.time()
    if _deposit_cache and _deposit_cache[0] > now:
        return _deposit_cache[1]

    try:
        api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        secret_key = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        paper = "paper" in base_url
        client = TradingClient(api_key, secret_key, paper=paper)

        history = []
        page_token = None
        while True:
            params = {"activity_types": "CSD,CSW", "page_size": 100, "direction": "asc"}
            if page_token:
                params["page_token"] = page_token
            activities = client.get("/account/activities", params)
            if not activities:
                break
            for a in activities:
                history.append({
                    "date": str(a["date"]),
                    "amount": float(a["net_amount"]),
                })
            if len(activities) < 100:
                break
            page_token = activities[-1]["id"]

        _deposit_cache = (now + _TTL_SECONDS, history)
        return history
    except Exception:
        logger.warning("Failed to fetch deposit history", exc_info=True)
        return []
