"""Tests for v2 executor functions."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

from v2.executor import OrderResult


class TestIsMarketOpen:
    @patch("v2.executor.get_trading_client")
    def test_returns_true_when_open(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_client.return_value.get_clock.return_value = mock_clock

        from v2.executor import is_market_open
        assert is_market_open() is True

    @patch("v2.executor.get_trading_client")
    def test_returns_false_when_closed(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_client.return_value.get_clock.return_value = mock_clock

        from v2.executor import is_market_open
        assert is_market_open() is False


class TestGetLatestPrice:
    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_price_for_fresh_quote(self, mock_data_client_cls):
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.25
        mock_quote.bid_price = 150.00
        mock_quote.timestamp = datetime.now(timezone.utc)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price == Decimal("150.25")

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_for_stale_quote(self, mock_data_client_cls):
        """Quote older than max_age_seconds should return None."""
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.25
        mock_quote.bid_price = 150.00
        mock_quote.timestamp = datetime.now(timezone.utc) - timedelta(seconds=120)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL", max_age_seconds=60)

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_for_wide_spread(self, mock_data_client_cls):
        """Quote with bid-ask spread > max_spread_pct should return None."""
        mock_quote = MagicMock()
        mock_quote.ask_price = 160.0  # 10% above bid
        mock_quote.bid_price = 145.0
        mock_quote.timestamp = datetime.now(timezone.utc)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL", max_spread_pct=Decimal("0.05"))

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_for_zero_price(self, mock_data_client_cls):
        mock_quote = MagicMock()
        mock_quote.ask_price = 0
        mock_quote.bid_price = 0
        mock_quote.timestamp = datetime.now(timezone.utc)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_on_api_error(self, mock_data_client_cls):
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.side_effect = Exception("API error")
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_staleness_check_skipped_when_no_timestamp(self, mock_data_client_cls):
        """If quote has no timestamp attr, skip staleness check (backwards compat)."""
        mock_quote = MagicMock(spec=["ask_price", "bid_price"])
        mock_quote.ask_price = 150.25
        mock_quote.bid_price = 150.00
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price == Decimal("150.25")
