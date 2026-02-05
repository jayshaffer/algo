"""Tests for trading/market_data.py - market data fetching and formatting."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.market_data import (
    SECTOR_ETFS,
    INDEX_ETFS,
    SectorPerformance,
    StockMover,
    MarketSnapshot,
    get_bar_change,
    get_sector_performance,
    get_index_levels,
    get_top_movers,
    get_unusual_volume,
    get_default_universe,
    get_market_snapshot,
    format_market_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(close: float, volume: int = 100000) -> MagicMock:
    """Create a mock bar with close price and volume."""
    bar = MagicMock()
    bar.close = close
    bar.volume = volume
    return bar


def _make_snapshot(
    sectors=None,
    indices=None,
    gainers=None,
    losers=None,
    unusual_volume=None,
    timestamp=None,
) -> MarketSnapshot:
    """Create a MarketSnapshot with sensible defaults."""
    return MarketSnapshot(
        timestamp=timestamp or datetime(2025, 6, 15, 14, 30),
        sectors=sectors or [],
        indices=indices or {},
        gainers=gainers or [],
        losers=losers or [],
        unusual_volume=unusual_volume or [],
    )


# ---------------------------------------------------------------------------
# Constants / structure tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module-level constants."""

    def test_sector_etfs_is_dict(self):
        assert isinstance(SECTOR_ETFS, dict)

    def test_sector_etfs_has_expected_sectors(self):
        expected = {"tech", "finance", "healthcare", "energy"}
        assert expected.issubset(set(SECTOR_ETFS.keys()))

    def test_sector_etfs_values_are_strings(self):
        for sector, etf in SECTOR_ETFS.items():
            assert isinstance(etf, str), f"SECTOR_ETFS[{sector}] is not a string"

    def test_index_etfs_is_list(self):
        assert isinstance(INDEX_ETFS, list)

    def test_index_etfs_contains_spy(self):
        assert "SPY" in INDEX_ETFS

    def test_index_etfs_contains_qqq(self):
        assert "QQQ" in INDEX_ETFS

    def test_index_etfs_contains_iwm(self):
        assert "IWM" in INDEX_ETFS


# ---------------------------------------------------------------------------
# get_default_universe
# ---------------------------------------------------------------------------

class TestGetDefaultUniverse:

    def test_returns_non_empty_list(self):
        universe = get_default_universe()
        assert len(universe) > 0

    def test_returns_list_of_strings(self):
        universe = get_default_universe()
        assert all(isinstance(t, str) for t in universe)

    def test_contains_major_tickers(self):
        universe = get_default_universe()
        for ticker in ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]:
            assert ticker in universe, f"{ticker} should be in default universe"

    def test_no_duplicates(self):
        universe = get_default_universe()
        assert len(universe) == len(set(universe)), "Universe should not contain duplicates"


# ---------------------------------------------------------------------------
# get_bar_change
# ---------------------------------------------------------------------------

class TestGetBarChange:

    def test_returns_float_on_success(self):
        """get_bar_change should return a rounded float percent change."""
        client = MagicMock()
        bars = {
            "AAPL": [_make_bar(100.0), _make_bar(105.0)]
        }
        client.get_stock_bars.return_value = bars

        result = get_bar_change(client, "AAPL", 1)
        assert result == 5.0

    def test_returns_none_when_insufficient_bars(self):
        """Should return None when fewer bars than requested days."""
        client = MagicMock()
        # Only 1 bar but we need at least 'days' bars
        bars = {"AAPL": [_make_bar(100.0)]}
        client.get_stock_bars.return_value = bars

        result = get_bar_change(client, "AAPL", 5)
        assert result is None

    def test_returns_none_on_exception(self):
        """Should return None when the API raises an exception."""
        client = MagicMock()
        client.get_stock_bars.side_effect = Exception("API error")

        result = get_bar_change(client, "AAPL", 1)
        assert result is None

    def test_negative_change(self):
        """Correctly calculates negative percent change."""
        client = MagicMock()
        bars = {"AAPL": [_make_bar(200.0), _make_bar(190.0)]}
        client.get_stock_bars.return_value = bars

        result = get_bar_change(client, "AAPL", 1)
        assert result == -5.0

    def test_multiday_change(self):
        """Calculates change over multiple days using correct indices."""
        client = MagicMock()
        # 6 bars, days=5 -> uses bars[-6] as past and bars[-1] as recent
        bars = {"SPY": [
            _make_bar(400.0),
            _make_bar(402.0),
            _make_bar(404.0),
            _make_bar(406.0),
            _make_bar(408.0),
            _make_bar(420.0),
        ]}
        client.get_stock_bars.return_value = bars

        result = get_bar_change(client, "SPY", 5)
        # (420 - 400) / 400 * 100 = 5.0
        assert result == 5.0

    def test_result_rounded_to_two_decimals(self):
        """Result should be rounded to 2 decimal places."""
        client = MagicMock()
        bars = {"X": [_make_bar(100.0), _make_bar(100.0), _make_bar(103.333)]}
        client.get_stock_bars.return_value = bars

        result = get_bar_change(client, "X", 2)
        # (103.333 - 100) / 100 * 100 = 3.333
        assert result == 3.33


# ---------------------------------------------------------------------------
# get_sector_performance
# ---------------------------------------------------------------------------

class TestGetSectorPerformance:

    @patch("trading.market_data.get_bar_change")
    def test_returns_list_of_sector_performance(self, mock_get_bar):
        mock_get_bar.return_value = 1.5
        client = MagicMock()

        result = get_sector_performance(client)
        assert all(isinstance(s, SectorPerformance) for s in result)
        assert len(result) == len(SECTOR_ETFS)

    @patch("trading.market_data.get_bar_change")
    def test_skips_sectors_with_none_1d(self, mock_get_bar):
        """Sectors with None 1d change are excluded."""
        mock_get_bar.return_value = None
        client = MagicMock()

        result = get_sector_performance(client)
        assert len(result) == 0

    @patch("trading.market_data.get_bar_change")
    def test_5d_none_defaults_to_zero(self, mock_get_bar):
        """When 5d is None, change_5d should default to 0.0."""
        def side_effect(client, symbol, days):
            if days == 1:
                return 2.0
            return None

        mock_get_bar.side_effect = side_effect
        client = MagicMock()

        result = get_sector_performance(client)
        for sp in result:
            assert sp.change_5d == 0.0


# ---------------------------------------------------------------------------
# get_index_levels
# ---------------------------------------------------------------------------

class TestGetIndexLevels:

    @patch("trading.market_data.get_bar_change")
    def test_returns_dict_of_changes(self, mock_get_bar):
        mock_get_bar.return_value = 0.5
        client = MagicMock()

        result = get_index_levels(client)
        assert isinstance(result, dict)
        for etf in INDEX_ETFS:
            assert etf in result

    @patch("trading.market_data.get_bar_change")
    def test_skips_none_values(self, mock_get_bar):
        mock_get_bar.return_value = None
        client = MagicMock()

        result = get_index_levels(client)
        assert result == {}


# ---------------------------------------------------------------------------
# format_market_snapshot
# ---------------------------------------------------------------------------

class TestFormatMarketSnapshot:

    def test_basic_format(self):
        """Format snapshot with minimal data into expected text."""
        snapshot = _make_snapshot(
            indices={"SPY": 1.25, "QQQ": -0.50},
            timestamp=datetime(2025, 6, 15, 14, 30),
        )
        text = format_market_snapshot(snapshot)

        assert "Market Snapshot (2025-06-15 14:30):" in text
        assert "SPY: +1.25%" in text
        assert "QQQ: -0.50%" in text

    def test_sectors_sorted_by_1d_desc(self):
        """Sectors should be sorted by 1d change descending."""
        snapshot = _make_snapshot(
            sectors=[
                SectorPerformance("energy", "XLE", -1.0, -2.0),
                SectorPerformance("tech", "XLK", 2.5, 4.0),
                SectorPerformance("finance", "XLF", 0.5, 1.0),
            ],
        )
        text = format_market_snapshot(snapshot)
        tech_pos = text.index("tech:")
        finance_pos = text.index("finance:")
        energy_pos = text.index("energy:")
        assert tech_pos < finance_pos < energy_pos

    def test_gainers_section(self):
        """Gainers should be formatted with + sign."""
        mover = StockMover(
            ticker="NVDA", price=Decimal("900.50"),
            change_pct=5.2, volume=5000000, avg_volume=2000000,
        )
        snapshot = _make_snapshot(gainers=[mover])
        text = format_market_snapshot(snapshot)

        assert "Top Gainers:" in text
        assert "NVDA: +5.2%" in text
        assert "$900.50" in text

    def test_losers_section(self):
        """Losers section should be present."""
        mover = StockMover(
            ticker="BA", price=Decimal("175.00"),
            change_pct=-3.1, volume=3000000, avg_volume=2000000,
        )
        snapshot = _make_snapshot(losers=[mover])
        text = format_market_snapshot(snapshot)

        assert "Top Losers:" in text
        assert "BA: -3.1%" in text

    def test_unusual_volume_section(self):
        """Unusual volume stocks should include volume ratio."""
        mover = StockMover(
            ticker="GME", price=Decimal("25.00"),
            change_pct=12.0, volume=50000000, avg_volume=10000000,
        )
        snapshot = _make_snapshot(unusual_volume=[mover])
        text = format_market_snapshot(snapshot)

        assert "Unusual Volume" in text
        assert "GME: 5.0x volume" in text

    def test_no_unusual_volume_section_when_empty(self):
        """If unusual_volume list is empty, section should not appear."""
        snapshot = _make_snapshot(unusual_volume=[])
        text = format_market_snapshot(snapshot)

        assert "Unusual Volume" not in text

    def test_only_first_five_gainers_shown(self):
        """At most 5 gainers should be listed."""
        movers = [
            StockMover(f"T{i}", Decimal("100"), float(i), 1000, 500)
            for i in range(10)
        ]
        snapshot = _make_snapshot(gainers=movers)
        text = format_market_snapshot(snapshot)

        gainers_section = text.split("Top Gainers:")[1].split("Top Losers:")[0]
        # Count lines that look like stock entries (start with "  T")
        stock_lines = [l for l in gainers_section.strip().split("\n") if l.strip().startswith("T")]
        assert len(stock_lines) == 5

    def test_negative_index_shows_no_plus(self):
        """Negative index change should not have a + sign."""
        snapshot = _make_snapshot(indices={"IWM": -2.33})
        text = format_market_snapshot(snapshot)
        assert "IWM: -2.33%" in text


# ---------------------------------------------------------------------------
# get_market_snapshot (integration-style with mocks)
# ---------------------------------------------------------------------------

class TestGetMarketSnapshot:

    @patch("trading.market_data.get_unusual_volume", return_value=[])
    @patch("trading.market_data.get_top_movers", return_value=([], []))
    @patch("trading.market_data.get_index_levels", return_value={"SPY": 0.5})
    @patch("trading.market_data.get_sector_performance", return_value=[])
    @patch("trading.market_data.get_data_client")
    def test_returns_market_snapshot(
        self, mock_client, mock_sectors, mock_indices, mock_movers, mock_vol
    ):
        result = get_market_snapshot(["AAPL"])

        assert isinstance(result, MarketSnapshot)
        assert result.indices == {"SPY": 0.5}

    @patch("trading.market_data.get_unusual_volume", return_value=[])
    @patch("trading.market_data.get_top_movers", return_value=([], []))
    @patch("trading.market_data.get_index_levels", return_value={})
    @patch("trading.market_data.get_sector_performance", return_value=[])
    @patch("trading.market_data.get_data_client")
    @patch("trading.market_data.get_default_universe", return_value=["AAPL", "MSFT"])
    def test_uses_default_universe_when_none(
        self, mock_universe, mock_client, mock_sectors, mock_indices,
        mock_movers, mock_vol
    ):
        get_market_snapshot(None)
        mock_universe.assert_called_once()


# ---------------------------------------------------------------------------
# get_top_movers
# ---------------------------------------------------------------------------

class TestGetTopMovers:

    def test_returns_gainers_and_losers_tuple(self):
        """get_top_movers returns a tuple of (gainers, losers)."""
        client = MagicMock()
        # Two bars per ticker: yesterday and today
        client.get_stock_bars.side_effect = lambda req: {
            req.symbol_or_symbols: [
                _make_bar(100.0, 10000),
                _make_bar(110.0, 15000),
            ]
        }

        gainers, losers = get_top_movers(client, ["AAPL"], top_n=5)
        assert isinstance(gainers, list)
        assert isinstance(losers, list)

    def test_skips_tickers_with_single_bar(self):
        """Tickers with fewer than 2 bars should be skipped."""
        client = MagicMock()
        client.get_stock_bars.return_value = {"AAPL": [_make_bar(100.0)]}

        gainers, losers = get_top_movers(client, ["AAPL"])
        assert len(gainers) == 0
        assert len(losers) == 0

    def test_skips_on_exception(self):
        """Exception for a ticker should be caught and ticker skipped."""
        client = MagicMock()
        client.get_stock_bars.side_effect = Exception("timeout")

        gainers, losers = get_top_movers(client, ["AAPL", "MSFT"])
        assert gainers == []
        assert losers == []


# ---------------------------------------------------------------------------
# get_unusual_volume
# ---------------------------------------------------------------------------

class TestGetUnusualVolume:

    def test_returns_stocks_above_threshold(self):
        """Stocks with volume >= threshold * avg should be returned."""
        client = MagicMock()
        # 6 bars: 5 history + 1 today (today volume = 3x avg)
        history = [_make_bar(100.0, 1000) for _ in range(5)]
        today = _make_bar(105.0, 3000)
        client.get_stock_bars.return_value = {"GME": history + [today]}

        result = get_unusual_volume(client, ["GME"], threshold=2.0)
        assert len(result) == 1
        assert result[0].ticker == "GME"

    def test_excludes_stocks_below_threshold(self):
        """Stocks with volume below threshold should be excluded."""
        client = MagicMock()
        history = [_make_bar(100.0, 1000) for _ in range(5)]
        today = _make_bar(105.0, 1500)  # 1.5x, below 2.0 threshold
        client.get_stock_bars.return_value = {"X": history + [today]}

        result = get_unusual_volume(client, ["X"], threshold=2.0)
        assert len(result) == 0

    def test_skips_tickers_with_too_few_bars(self):
        """Tickers with fewer than 5 bars should be skipped."""
        client = MagicMock()
        client.get_stock_bars.return_value = {"X": [_make_bar(100.0) for _ in range(3)]}

        result = get_unusual_volume(client, ["X"])
        assert result == []
