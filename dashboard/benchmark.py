"""SPY benchmark data fetching and alpha computation for the local dashboard."""

import logging
import os
import time
from datetime import date, datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

logger = logging.getLogger(__name__)


def compute_alpha(snapshots, benchmark):
    """Compute portfolio alpha vs SPY benchmark.

    Returns {"portfolio_return", "spy_return", "alpha"} or None.
    """
    if not snapshots or len(snapshots) < 2 or not benchmark:
        return None

    spy_map = {b["date"]: b["close"] for b in benchmark}

    spy_start = None
    port_start = None
    for snap in snapshots:
        date_str = str(snap["date"])
        if date_str in spy_map:
            spy_start = spy_map[date_str]
            port_start = float(snap["portfolio_value"])
            break

    spy_end = None
    port_end = None
    for snap in reversed(snapshots):
        date_str = str(snap["date"])
        if date_str in spy_map:
            spy_end = spy_map[date_str]
            port_end = float(snap["portfolio_value"])
            break

    if spy_start is None or spy_end is None or spy_start == spy_end:
        return None
    if port_start is None or port_start == 0:
        return None

    portfolio_return = ((port_end - port_start) / port_start) * 100
    spy_return = ((spy_end - spy_start) / spy_start) * 100
    alpha = portfolio_return - spy_return

    return {
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
        "alpha": alpha,
    }


_TTL_SECONDS = 900  # 15 minutes
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
