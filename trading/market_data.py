"""Market data fetching for ideation context."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from .finnhub_client import FinnhubClient

logger = logging.getLogger(__name__)


# Sector ETF proxies
SECTOR_ETFS = {
    "tech": "XLK",
    "finance": "XLF",
    "healthcare": "XLV",
    "energy": "XLE",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communications": "XLC",
}

# Major indices
INDEX_ETFS = ["SPY", "QQQ", "IWM"]


@dataclass
class SectorPerformance:
    """Sector performance data."""
    sector: str
    etf: str
    change_1d: float
    change_5d: float


@dataclass
class StockMover:
    """A stock with unusual movement."""
    ticker: str
    price: Decimal
    change_pct: float
    volume: int
    avg_volume: int


@dataclass
class MarketSnapshot:
    """Complete market snapshot for ideation."""
    timestamp: datetime
    sectors: list[SectorPerformance]
    indices: dict[str, float]  # ETF -> 1d change
    gainers: list[StockMover]
    losers: list[StockMover]
    unusual_volume: list[StockMover]


@dataclass
class _Bar:
    """Internal bar representation matching Finnhub candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def get_bars_batch(
    symbols: list[str],
    days: int
) -> dict[str, list[_Bar]]:
    """
    Fetch daily bars for multiple symbols from Finnhub.

    Args:
        symbols: List of tickers to fetch
        days: Number of days of history to fetch

    Returns:
        Dict mapping symbol to list of _Bar objects
    """
    client = FinnhubClient()
    now = int(time.time())
    start = now - ((days + 5) * 86400)  # Extra buffer for weekends

    results = {}
    failed_count = 0

    for symbol in symbols:
        try:
            data = client.candles(symbol, "D", start, now)
            if data.get("s") == "ok" and data.get("t"):
                bars = [
                    _Bar(
                        timestamp=datetime.fromtimestamp(ts),
                        open=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=v
                    )
                    for ts, o, h, l, c, v in zip(
                        data["t"], data["o"], data["h"],
                        data["l"], data["c"], data["v"]
                    )
                ]
                results[symbol] = bars
            else:
                logger.debug(f"No data for {symbol}: {data.get('s')}")
                failed_count += 1
        except Exception as e:
            logger.warning(f"Failed to fetch bars for {symbol}: {e}")
            failed_count += 1

    if failed_count > 0:
        logger.info(f"Fetched bars for {len(results)}/{len(symbols)} symbols ({failed_count} failed)")

    return results


def get_bar_change(symbol: str, days: int) -> Optional[float]:
    """Get percentage change over N days for a symbol."""
    bars_dict = get_bars_batch([symbol], days)
    symbol_bars = bars_dict.get(symbol, [])

    if len(symbol_bars) < days:
        return None

    recent = symbol_bars[-1]
    past = symbol_bars[-days - 1] if len(symbol_bars) > days else symbol_bars[0]

    change = ((recent.close - past.close) / past.close) * 100
    return round(change, 2)


def get_sector_performance() -> list[SectorPerformance]:
    """Get performance for all sector ETFs."""
    sectors = []
    etfs = list(SECTOR_ETFS.values())

    # Fetch 5 days of bars for all ETFs (covers both 1d and 5d)
    bars_dict = get_bars_batch(etfs, 5)

    for sector, etf in SECTOR_ETFS.items():
        symbol_bars = bars_dict.get(etf, [])
        if len(symbol_bars) < 2:
            logger.warning(f"Insufficient bars for sector ETF {etf}")
            continue

        # Calculate 1-day change
        recent = symbol_bars[-1]
        yesterday = symbol_bars[-2]
        change_1d = round(((recent.close - yesterday.close) / yesterday.close) * 100, 2)

        # Calculate 5-day change if we have enough data
        change_5d = 0.0
        if len(symbol_bars) >= 6:
            past = symbol_bars[-6]
            change_5d = round(((recent.close - past.close) / past.close) * 100, 2)

        sectors.append(SectorPerformance(
            sector=sector,
            etf=etf,
            change_1d=change_1d,
            change_5d=change_5d,
        ))

    return sectors


def get_index_levels() -> dict[str, float]:
    """Get 1-day change for major indices."""
    indices = {}
    bars_dict = get_bars_batch(INDEX_ETFS, 2)  # Need 2 days for 1d change

    for etf in INDEX_ETFS:
        symbol_bars = bars_dict.get(etf, [])
        if len(symbol_bars) < 2:
            logger.warning(f"Insufficient bars for index ETF {etf}")
            continue

        recent = symbol_bars[-1]
        yesterday = symbol_bars[-2]
        change = round(((recent.close - yesterday.close) / yesterday.close) * 100, 2)
        indices[etf] = change

    return indices


@dataclass
class _StockMetrics:
    """Internal metrics for a stock from bar data."""
    ticker: str
    price: Decimal
    change_pct: float
    volume: int
    avg_volume: int
    volume_ratio: float


def _compute_universe_metrics(
    universe: list[str],
) -> list[_StockMetrics]:
    """
    Compute metrics for all stocks in universe.

    This fetches 30 days of data to support both short-term (movers) and
    longer-term (unusual volume) calculations.
    """
    # Fetch 30 days of bars for all symbols
    bars_dict = get_bars_batch(universe, 30)

    metrics = []
    failed_count = 0

    for ticker in universe:
        symbol_bars = bars_dict.get(ticker, [])

        if len(symbol_bars) < 2:
            failed_count += 1
            continue

        today = symbol_bars[-1]
        yesterday = symbol_bars[-2]

        change_pct = ((today.close - yesterday.close) / yesterday.close) * 100

        # Calculate average volume (excluding today)
        volumes = [b.volume for b in symbol_bars[:-1]]
        avg_volume = int(sum(volumes) / len(volumes)) if volumes else 0

        volume_ratio = today.volume / avg_volume if avg_volume > 0 else 0.0

        metrics.append(_StockMetrics(
            ticker=ticker,
            price=Decimal(str(today.close)),
            change_pct=round(change_pct, 2),
            volume=today.volume,
            avg_volume=avg_volume,
            volume_ratio=volume_ratio,
        ))

    if failed_count > 0:
        logger.warning(f"Failed to get data for {failed_count}/{len(universe)} symbols")

    return metrics


def get_top_movers(
    universe: list[str],
    top_n: int = 10,
    _metrics: Optional[list[_StockMetrics]] = None,
) -> tuple[list[StockMover], list[StockMover]]:
    """
    Get top gainers and losers from a universe of stocks.

    Args:
        universe: List of tickers to scan
        top_n: Number of stocks to return for each category
        _metrics: Pre-computed metrics (internal use for batching)

    Returns:
        Tuple of (gainers, losers)
    """
    if _metrics is None:
        _metrics = _compute_universe_metrics(universe)

    # Sort by change percentage
    sorted_metrics = sorted(_metrics, key=lambda m: m.change_pct, reverse=True)

    def to_mover(m: _StockMetrics) -> StockMover:
        return StockMover(
            ticker=m.ticker,
            price=m.price,
            change_pct=m.change_pct,
            volume=m.volume,
            avg_volume=m.avg_volume,
        )

    gainers = [to_mover(m) for m in sorted_metrics[:top_n]]
    losers = [to_mover(m) for m in sorted_metrics[-top_n:][::-1]]

    return gainers, losers


def get_unusual_volume(
    universe: list[str],
    threshold: float = 2.0,
    top_n: int = 10,
    _metrics: Optional[list[_StockMetrics]] = None,
) -> list[StockMover]:
    """
    Find stocks with unusual volume (above threshold of average).

    Args:
        universe: List of tickers to scan
        threshold: Volume multiple (default 2x)
        top_n: Maximum stocks to return
        _metrics: Pre-computed metrics (internal use for batching)

    Returns:
        List of stocks with unusual volume
    """
    if _metrics is None:
        _metrics = _compute_universe_metrics(universe)

    # Filter by threshold and sort by volume ratio
    unusual = [
        m for m in _metrics
        if m.volume_ratio >= threshold
    ]
    unusual.sort(key=lambda m: m.volume_ratio, reverse=True)

    return [
        StockMover(
            ticker=m.ticker,
            price=m.price,
            change_pct=m.change_pct,
            volume=m.volume,
            avg_volume=m.avg_volume,
        )
        for m in unusual[:top_n]
    ]


def get_default_universe() -> list[str]:
    """Get a default universe of liquid stocks to scan."""
    # S&P 100 components (subset for efficiency)
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK.B",
        "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "PEP", "KO", "COST", "AVGO", "LLY", "WMT", "MCD", "CSCO", "ACN",
        "TMO", "ABT", "DHR", "VZ", "ADBE", "CRM", "CMCSA", "NEE", "NKE",
        "TXN", "PM", "RTX", "HON", "INTC", "QCOM", "UNP", "IBM", "AMGN",
        "LOW", "CAT", "BA", "GE", "AMD", "SBUX", "MDLZ", "ISRG", "BKNG",
    ]


def get_market_snapshot(universe: Optional[list[str]] = None) -> MarketSnapshot:
    """
    Build complete market snapshot for ideation.

    Note: With Finnhub rate limits (60/min), this takes ~80 seconds
    for the full universe. Run once per trading session.

    Args:
        universe: Optional list of tickers to scan. Uses default if not provided.

    Returns:
        MarketSnapshot with sectors, indices, movers, and unusual volume
    """
    if universe is None:
        universe = get_default_universe()

    logger.info(f"Fetching market snapshot for {len(universe)} symbols (this may take a while)...")

    sectors = get_sector_performance()
    indices = get_index_levels()

    # Compute universe metrics once and share between movers and volume
    metrics = _compute_universe_metrics(universe)
    gainers, losers = get_top_movers(universe, _metrics=metrics)
    unusual_volume = get_unusual_volume(universe, _metrics=metrics)

    return MarketSnapshot(
        timestamp=datetime.now(),
        sectors=sectors,
        indices=indices,
        gainers=gainers,
        losers=losers,
        unusual_volume=unusual_volume,
    )


def format_market_snapshot(snapshot: MarketSnapshot) -> str:
    """Format market snapshot as text for LLM context."""
    lines = [f"Market Snapshot ({snapshot.timestamp.strftime('%Y-%m-%d %H:%M')}):", ""]

    # Indices
    lines.append("Major Indices (1d change):")
    for etf, change in snapshot.indices.items():
        sign = "+" if change >= 0 else ""
        lines.append(f"  {etf}: {sign}{change:.2f}%")
    lines.append("")

    # Sector performance
    lines.append("Sector Performance:")
    for sector in sorted(snapshot.sectors, key=lambda s: s.change_1d, reverse=True):
        sign_1d = "+" if sector.change_1d >= 0 else ""
        sign_5d = "+" if sector.change_5d >= 0 else ""
        lines.append(f"  {sector.sector}: {sign_1d}{sector.change_1d:.1f}% (1d), {sign_5d}{sector.change_5d:.1f}% (5d)")
    lines.append("")

    # Top gainers
    lines.append("Top Gainers:")
    for stock in snapshot.gainers[:5]:
        lines.append(f"  {stock.ticker}: +{stock.change_pct:.1f}% @ ${stock.price:.2f}")
    lines.append("")

    # Top losers
    lines.append("Top Losers:")
    for stock in snapshot.losers[:5]:
        lines.append(f"  {stock.ticker}: {stock.change_pct:.1f}% @ ${stock.price:.2f}")
    lines.append("")

    # Unusual volume
    if snapshot.unusual_volume:
        lines.append("Unusual Volume (2x+ avg):")
        for stock in snapshot.unusual_volume[:5]:
            ratio = stock.volume / stock.avg_volume if stock.avg_volume else 0
            sign = "+" if stock.change_pct >= 0 else ""
            lines.append(f"  {stock.ticker}: {ratio:.1f}x volume, {sign}{stock.change_pct:.1f}%")

    return "\n".join(lines)
