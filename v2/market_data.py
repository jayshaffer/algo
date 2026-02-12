"""Market data fetching for ideation context."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame


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
    indices: dict[str, float]
    gainers: list[StockMover]
    losers: list[StockMover]
    unusual_volume: list[StockMover]


def get_data_client() -> StockHistoricalDataClient:
    """Create Alpaca data client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    return StockHistoricalDataClient(api_key, secret_key)


def get_bar_change(client: StockHistoricalDataClient, symbol: str, days: int) -> Optional[float]:
    """Get percentage change over N days for a symbol."""
    end = datetime.now()
    start = end - timedelta(days=days + 5)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(request)
        symbol_bars = list(bars[symbol])

        if len(symbol_bars) < days:
            return None

        recent = symbol_bars[-1]
        past = symbol_bars[-days - 1] if len(symbol_bars) > days else symbol_bars[0]

        change = ((recent.close - past.close) / past.close) * 100
        return round(change, 2)
    except Exception:
        return None


def get_sector_performance(client: StockHistoricalDataClient) -> list[SectorPerformance]:
    """Get performance for all sector ETFs."""
    sectors = []

    for sector, etf in SECTOR_ETFS.items():
        change_1d = get_bar_change(client, etf, 1)
        change_5d = get_bar_change(client, etf, 5)

        if change_1d is not None:
            sectors.append(SectorPerformance(
                sector=sector,
                etf=etf,
                change_1d=change_1d,
                change_5d=change_5d or 0.0,
            ))

    return sectors


def get_index_levels(client: StockHistoricalDataClient) -> dict[str, float]:
    """Get 1-day change for major indices."""
    indices = {}

    for etf in INDEX_ETFS:
        change = get_bar_change(client, etf, 1)
        if change is not None:
            indices[etf] = change

    return indices


def get_top_movers(
    client: StockHistoricalDataClient,
    universe: list[str],
    top_n: int = 10
) -> tuple[list[StockMover], list[StockMover]]:
    """Get top gainers and losers from a universe of stocks."""
    movers = []
    end = datetime.now()
    start = end - timedelta(days=10)

    for ticker in universe:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(request)
            symbol_bars = list(bars[ticker])

            if len(symbol_bars) < 2:
                continue

            today = symbol_bars[-1]
            yesterday = symbol_bars[-2]

            change_pct = ((today.close - yesterday.close) / yesterday.close) * 100

            volumes = [b.volume for b in symbol_bars[:-1]]
            avg_volume = sum(volumes) // len(volumes) if volumes else 0

            movers.append(StockMover(
                ticker=ticker,
                price=Decimal(str(today.close)),
                change_pct=round(change_pct, 2),
                volume=today.volume,
                avg_volume=avg_volume,
            ))
        except Exception:
            continue

    movers.sort(key=lambda m: m.change_pct, reverse=True)

    gainers = movers[:top_n]
    losers = movers[-top_n:][::-1]

    return gainers, losers


def get_unusual_volume(
    client: StockHistoricalDataClient,
    universe: list[str],
    threshold: float = 2.0,
    top_n: int = 10
) -> list[StockMover]:
    """Find stocks with unusual volume (above threshold of average)."""
    unusual = []
    end = datetime.now()
    start = end - timedelta(days=30)

    for ticker in universe:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(request)
            symbol_bars = list(bars[ticker])

            if len(symbol_bars) < 5:
                continue

            today = symbol_bars[-1]

            volumes = [b.volume for b in symbol_bars[:-1]]
            avg_volume = sum(volumes) // len(volumes) if volumes else 0

            if avg_volume == 0:
                continue

            volume_ratio = today.volume / avg_volume

            if volume_ratio >= threshold:
                yesterday = symbol_bars[-2]
                change_pct = ((today.close - yesterday.close) / yesterday.close) * 100

                unusual.append(StockMover(
                    ticker=ticker,
                    price=Decimal(str(today.close)),
                    change_pct=round(change_pct, 2),
                    volume=today.volume,
                    avg_volume=avg_volume,
                ))
        except Exception:
            continue

    unusual.sort(key=lambda m: m.volume / m.avg_volume if m.avg_volume else 0, reverse=True)

    return unusual[:top_n]


def get_default_universe() -> list[str]:
    """Get a default universe of liquid stocks to scan."""
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK.B",
        "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "PEP", "KO", "COST", "AVGO", "LLY", "WMT", "MCD", "CSCO", "ACN",
        "TMO", "ABT", "DHR", "VZ", "ADBE", "CRM", "CMCSA", "NEE", "NKE",
        "TXN", "PM", "RTX", "HON", "INTC", "QCOM", "UNP", "IBM", "AMGN",
        "LOW", "CAT", "BA", "GE", "AMD", "SBUX", "MDLZ", "ISRG", "BKNG",
    ]


def get_market_snapshot(universe: Optional[list[str]] = None) -> MarketSnapshot:
    """Build complete market snapshot for ideation."""
    client = get_data_client()

    if universe is None:
        universe = get_default_universe()

    sectors = get_sector_performance(client)
    indices = get_index_levels(client)
    gainers, losers = get_top_movers(client, universe)
    unusual_volume = get_unusual_volume(client, universe)

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

    lines.append("Major Indices (1d change):")
    for etf, change in snapshot.indices.items():
        sign = "+" if change >= 0 else ""
        lines.append(f"  {etf}: {sign}{change:.2f}%")
    lines.append("")

    lines.append("Sector Performance:")
    for sector in sorted(snapshot.sectors, key=lambda s: s.change_1d, reverse=True):
        sign_1d = "+" if sector.change_1d >= 0 else ""
        sign_5d = "+" if sector.change_5d >= 0 else ""
        lines.append(f"  {sector.sector}: {sign_1d}{sector.change_1d:.1f}% (1d), {sign_5d}{sector.change_5d:.1f}% (5d)")
    lines.append("")

    lines.append("Top Gainers:")
    for stock in snapshot.gainers[:5]:
        lines.append(f"  {stock.ticker}: +{stock.change_pct:.1f}% @ ${stock.price:.2f}")
    lines.append("")

    lines.append("Top Losers:")
    for stock in snapshot.losers[:5]:
        lines.append(f"  {stock.ticker}: {stock.change_pct:.1f}% @ ${stock.price:.2f}")
    lines.append("")

    if snapshot.unusual_volume:
        lines.append("Unusual Volume (2x+ avg):")
        for stock in snapshot.unusual_volume[:5]:
            ratio = stock.volume / stock.avg_volume if stock.avg_volume else 0
            sign = "+" if stock.change_pct >= 0 else ""
            lines.append(f"  {stock.ticker}: {ratio:.1f}x volume, {sign}{stock.change_pct:.1f}%")

    return "\n".join(lines)
