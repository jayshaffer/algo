# Finnhub Market Data Migration

**Date:** 2026-02-03
**Status:** Approved

## Overview

Replace Alpaca market data calls with Finnhub (free tier) while keeping Alpaca for news and trading operations.

## Scope

**Replace with Finnhub:**
- Historical bars/candles
- Latest quotes

**Keep as Alpaca:**
- News (news.py, ingest.py)
- Trading operations (orders, positions, account)

## Rate Limiting Strategy

Finnhub free tier: 60 calls/minute

Approach: Accept slower refreshes with built-in delays (~1.1s between requests). Full market snapshot (~74 symbols) takes ~80 seconds. Acceptable for once-per-session usage.

## Files

### New File: `trading/finnhub_client.py`

```python
import os
import time
import requests

class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self):
        self.api_key = os.environ["FINNHUB_API_KEY"]
        self.last_request = 0
        self.min_interval = 1.1  # ~55 req/min, safe under 60 limit

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()

    def quote(self, symbol: str) -> dict:
        """Get current quote. Returns {c, h, l, o, pc, t}"""
        self._rate_limit()
        resp = requests.get(f"{self.BASE_URL}/quote",
            params={"symbol": symbol, "token": self.api_key})
        resp.raise_for_status()
        return resp.json()

    def candles(self, symbol: str, resolution: str,
                from_ts: int, to_ts: int) -> dict:
        """Get OHLCV candles. Resolution: 1, 5, 15, 30, 60, D, W, M"""
        self._rate_limit()
        resp = requests.get(f"{self.BASE_URL}/stock/candle",
            params={"symbol": symbol, "resolution": resolution,
                    "from": from_ts, "to": to_ts, "token": self.api_key})
        resp.raise_for_status()
        return resp.json()
```

### Modified: `trading/market_data.py`

Remove Alpaca imports and `get_data_client()`. Replace `get_bars_batch()`:

```python
def get_bars_batch(symbols: list[str], days: int = 30) -> dict:
    """Fetch daily bars for multiple symbols from Finnhub."""
    client = FinnhubClient()
    now = int(time.time())
    start = now - (days * 86400)

    results = {}
    for symbol in symbols:
        data = client.candles(symbol, "D", start, now)
        if data.get("s") == "ok":
            results[symbol] = [
                {"date": ts, "open": o, "high": h, "low": l,
                 "close": c, "volume": v}
                for ts, o, h, l, c, v in zip(
                    data["t"], data["o"], data["h"],
                    data["l"], data["c"], data["v"])
            ]
    return results
```

Functions that depend on `get_bars_batch()` remain unchanged in logic.

### Modified: `trading/executor.py`

Replace `get_latest_price()`:

```python
def get_latest_price(ticker: str) -> float:
    """Get latest price for order sizing."""
    client = FinnhubClient()
    quote = client.quote(ticker)
    return quote["c"]
```

### Modified: `trading/backfill.py`

Replace `get_price_on_date()`:

```python
def get_price_on_date(ticker: str, date: datetime.date) -> float | None:
    """Get closing price for a specific date."""
    client = FinnhubClient()
    start = int(datetime.combine(date, time.min).timestamp())
    end = start + 86400
    data = client.candles(ticker, "D", start, end)
    if data.get("s") == "ok" and data["c"]:
        return data["c"][0]
    return None
```

### Modified: `trading/main.py`

Replace `get_quote()`:

```python
def get_quote(ticker: str) -> dict:
    """Test connectivity with a quote fetch."""
    client = FinnhubClient()
    return client.quote(ticker)
```

### Modified: `.env.example`

Add:
```
FINNHUB_API_KEY=your_finnhub_api_key
```

## Implementation Order

1. Create `finnhub_client.py`
2. Update `market_data.py`
3. Update `executor.py`
4. Update `backfill.py`
5. Update `main.py`
6. Update `.env.example`
7. Test with dry run
