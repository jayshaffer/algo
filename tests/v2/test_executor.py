"""Tests for v2 executor functions."""

import time
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


class TestGetLatestPriceClientReuse:
    @patch("v2.executor.StockHistoricalDataClient")
    def test_uses_provided_client(self, mock_client_cls):
        from v2.executor import get_latest_price
        external_client = MagicMock()
        quote = MagicMock()
        quote.ask_price = 150.0
        quote.bid_price = 149.5
        quote.timestamp = None
        external_client.get_stock_latest_quote.return_value = {"AAPL": quote}
        result = get_latest_price("AAPL", client=external_client)
        assert result == Decimal("150.0")
        mock_client_cls.assert_not_called()

    @patch("v2.executor.StockHistoricalDataClient")
    def test_creates_client_when_none_provided(self, mock_client_cls):
        """When no client is provided, it should create one (existing behavior)."""
        from v2.executor import get_latest_price
        mock_quote = MagicMock()
        mock_quote.ask_price = 100.0
        mock_quote.bid_price = 99.5
        mock_quote.timestamp = None
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"TSLA": mock_quote}
        mock_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            result = get_latest_price("TSLA")

        assert result == Decimal("100.0")
        mock_client_cls.assert_called_once()


class TestWaitForFill:
    @patch("v2.executor.get_trading_client")
    def test_returns_filled_order(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "filled"
        mock_order.filled_qty = "2.5"
        mock_order.filled_avg_price = "150.25"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is True
        assert result.filled_qty == Decimal("2.5")
        assert result.filled_avg_price == Decimal("150.25")

    @patch("v2.executor.get_trading_client")
    def test_returns_error_on_timeout(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "accepted"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert "timeout" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_returns_error_on_cancelled(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "canceled"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is False
        assert "canceled" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_returns_error_on_rejected(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "rejected"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is False
        assert "rejected" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_polls_until_filled(self, mock_client):
        """Should poll multiple times until order status becomes filled."""
        pending = MagicMock()
        pending.status.value = "accepted"

        filled = MagicMock()
        filled.status.value = "filled"
        filled.filled_qty = "5"
        filled.filled_avg_price = "100.00"

        mock_client.return_value.get_order_by_id.side_effect = [pending, pending, filled]

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is True
        assert mock_client.return_value.get_order_by_id.call_count == 3

    @patch("v2.executor.get_trading_client")
    def test_partially_filled_waits(self, mock_client):
        """Partially filled should keep polling."""
        partial = MagicMock()
        partial.status.value = "partially_filled"

        filled = MagicMock()
        filled.status.value = "filled"
        filled.filled_qty = "10"
        filled.filled_avg_price = "200.00"

        mock_client.return_value.get_order_by_id.side_effect = [partial, filled]

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is True
        assert result.filled_qty == Decimal("10")


class TestWaitForFillCancellation:
    @patch("v2.executor.get_trading_client")
    def test_timeout_cancels_order(self, mock_client):
        """On timeout, cancel_order_by_id should be called to prevent ghost fills."""
        mock_order = MagicMock()
        mock_order.status.value = "accepted"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-abc", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert "cancel attempted" in result.error.lower()
        mock_client.return_value.cancel_order_by_id.assert_called_once_with("order-abc")

    @patch("v2.executor.get_trading_client")
    def test_timeout_cancel_failure_still_returns_timeout(self, mock_client):
        """If cancel fails, should still return timeout error (not raise)."""
        mock_order = MagicMock()
        mock_order.status.value = "accepted"
        mock_client.return_value.get_order_by_id.return_value = mock_order
        mock_client.return_value.cancel_order_by_id.side_effect = Exception("API error")

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-abc", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert "cancel attempted" in result.error.lower()
        mock_client.return_value.cancel_order_by_id.assert_called_once_with("order-abc")
