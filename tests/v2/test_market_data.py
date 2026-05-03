"""Tests for v2/market_data.py."""

from unittest.mock import MagicMock, patch

import pytest

from v2.market_data import (
    MarketSnapshot,
    SectorPerformance,
    StockMover,
    format_market_snapshot,
    get_bar_change,
    get_sector_performance,
)


def _bar(close: float):
    b = MagicMock()
    b.close = close
    return b


def _bars_response(symbol: str, bars: list):
    response = MagicMock()
    response.__getitem__ = lambda self, key: bars if key == symbol else []
    return response


class TestGetBarChange:
    """T2.4: with `days=N`, the function must require strictly more than N bars
    so a real "past" bar exists. Otherwise the previous `len < days` guard
    let `days=1` with only 1 bar collapse `recent` and `past` to the same row,
    silently returning 0.0% instead of None.
    """

    def test_one_bar_with_days_one_returns_none(self):
        client = MagicMock()
        client.get_stock_bars.return_value = _bars_response("AAPL", [_bar(100.0)])
        assert get_bar_change(client, "AAPL", 1) is None

    def test_two_bars_with_days_one_returns_pct(self):
        client = MagicMock()
        client.get_stock_bars.return_value = _bars_response("AAPL", [_bar(100.0), _bar(110.0)])
        assert get_bar_change(client, "AAPL", 1) == pytest.approx(10.0)

    def test_six_bars_with_days_five_returns_pct(self):
        client = MagicMock()
        bars = [_bar(100.0)] + [_bar(105.0)] * 4 + [_bar(110.0)]
        client.get_stock_bars.return_value = _bars_response("AAPL", bars)
        assert get_bar_change(client, "AAPL", 5) == pytest.approx(10.0)

    def test_five_bars_with_days_five_returns_none(self):
        """`days=5` requires at least 6 bars (one for recent, one for past
        5 days ago)."""
        client = MagicMock()
        bars = [_bar(100.0)] * 5
        client.get_stock_bars.return_value = _bars_response("AAPL", bars)
        assert get_bar_change(client, "AAPL", 5) is None


class TestGetSectorPerformance:
    """T2.4: when 5d data is unavailable, change_5d stays None — no longer
    silently coerced to 0.0.
    """

    @patch("v2.market_data.get_bar_change")
    def test_change_5d_none_propagates(self, mock_change):
        # 1d=+1.5%, 5d=None
        mock_change.side_effect = [1.5, None] * 11
        result = get_sector_performance(MagicMock())
        assert all(s.change_5d is None for s in result)
        assert all(s.change_1d == 1.5 for s in result)

    @patch("v2.market_data.get_bar_change")
    def test_skips_sectors_with_no_1d(self, mock_change):
        # All 1d returns None → no sectors emitted.
        mock_change.return_value = None
        result = get_sector_performance(MagicMock())
        assert result == []


class TestFormatMarketSnapshot:
    """T2.4: rendering must show 'N/A' for missing 5d data, not '0.0%'."""

    def test_renders_none_change_5d_as_na(self):
        snapshot = MarketSnapshot(
            timestamp=__import__("datetime").datetime(2026, 5, 3, 12, 0),
            sectors=[
                SectorPerformance("tech", "XLK", 1.5, None),
                SectorPerformance("energy", "XLE", -0.5, 2.3),
            ],
            indices={"SPY": 0.4},
            gainers=[],
            losers=[],
            unusual_volume=[],
        )
        output = format_market_snapshot(snapshot)
        assert "tech: +1.5% (1d), N/A (5d)" in output
        assert "energy: -0.5% (1d), +2.3% (5d)" in output
