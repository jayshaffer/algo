"""Finnhub API client with rate limiting."""

import os
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class FinnhubClient:
    """
    Finnhub API client with built-in rate limiting.

    Free tier: 60 calls/minute
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY must be set")
        self.last_request = 0.0
        self.min_interval = 1.1  # ~55 req/min, safe under 60 limit

    def _rate_limit(self):
        """Wait if needed to respect rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request = time.time()

    def quote(self, symbol: str) -> dict:
        """
        Get current quote for a symbol.

        Returns:
            dict with keys: c (current), h (high), l (low), o (open),
                           pc (previous close), t (timestamp)
        """
        self._rate_limit()
        resp = requests.get(
            f"{self.BASE_URL}/quote",
            params={"symbol": symbol, "token": self.api_key},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def candles(
        self,
        symbol: str,
        resolution: str,
        from_ts: int,
        to_ts: int
    ) -> dict:
        """
        Get OHLCV candles for a symbol.

        Args:
            symbol: Stock symbol
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M)
            from_ts: Start timestamp (Unix seconds)
            to_ts: End timestamp (Unix seconds)

        Returns:
            dict with keys: s (status), t (timestamps), o, h, l, c, v
            Status "ok" means data returned, "no_data" means no data
        """
        self._rate_limit()
        resp = requests.get(
            f"{self.BASE_URL}/stock/candle",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "from": from_ts,
                "to": to_ts,
                "token": self.api_key
            },
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
