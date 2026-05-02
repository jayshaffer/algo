"""Tests for v2 executor functions."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch


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
        mock_quote.timestamp = datetime.now(UTC)
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
        mock_quote.timestamp = datetime.now(UTC) - timedelta(seconds=120)
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
        mock_quote.timestamp = datetime.now(UTC)
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
        mock_quote.timestamp = datetime.now(UTC)
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


class TestGetLatestTradePrice:
    """Trade-price lookup path used for reference pricing (logging HOLDs,
    valuing positions for sector concentration). Unlike get_latest_price,
    this does NOT enforce a bid-ask spread — the free IEX feed produces wide
    quote spreads near the close, but the last-trade print remains accurate.
    """

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_trade_price(self, mock_data_client_cls):
        mock_trade = MagicMock()
        mock_trade.price = 178.12
        mock_client = MagicMock()
        mock_client.get_stock_latest_trade.return_value = {"CRM": mock_trade}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_trade_price
            price = get_latest_trade_price("CRM")

        assert price == Decimal("178.12")

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_for_zero_price(self, mock_data_client_cls):
        mock_trade = MagicMock()
        mock_trade.price = 0
        mock_client = MagicMock()
        mock_client.get_stock_latest_trade.return_value = {"CRM": mock_trade}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_trade_price
            price = get_latest_trade_price("CRM")

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_on_api_error(self, mock_data_client_cls):
        mock_client = MagicMock()
        mock_client.get_stock_latest_trade.side_effect = Exception("API error")
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_trade_price
            price = get_latest_trade_price("CRM")

        assert price is None

    def test_does_not_reject_wide_spread(self):
        """Regression: trade-price lookup should ignore bid-ask spread entirely.

        This is the whole point of having a separate helper — the IEX-only free
        feed gives wide spreads near close but the last trade is fine, and paths
        that only need reference pricing (HOLD logging, position valuation)
        shouldn't be gated on quote quality.
        """
        from v2.executor import get_latest_trade_price
        mock_trade = MagicMock()
        mock_trade.price = 100.0
        external_client = MagicMock()
        external_client.get_stock_latest_trade.return_value = {"ANET": mock_trade}

        price = get_latest_trade_price("ANET", client=external_client)

        assert price == Decimal("100.0")
        # Quote API must not be consulted — the whole premise is to bypass it.
        external_client.get_stock_latest_quote.assert_not_called()

    def test_uses_provided_client(self):
        from v2.executor import get_latest_trade_price
        mock_trade = MagicMock()
        mock_trade.price = 42.0
        external_client = MagicMock()
        external_client.get_stock_latest_trade.return_value = {"SPY": mock_trade}

        with patch("v2.executor.StockHistoricalDataClient") as mock_cls:
            result = get_latest_trade_price("SPY", client=external_client)

        assert result == Decimal("42.0")
        mock_cls.assert_not_called()


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
        mock_order.filled_qty = None
        mock_order.filled_avg_price = None
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
        mock_order.filled_qty = None
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order
        mock_client.return_value.cancel_order_by_id.side_effect = Exception("API error")

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-abc", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert "cancel attempted" in result.error.lower()
        mock_client.return_value.cancel_order_by_id.assert_called_once_with("order-abc")


class TestWaitForFillPartialFillOnTimeout:
    """P1.13: timeout must capture any qty that filled before cancel landed,
    so the decision-to-fill link survives. Previously the executor returned
    `filled_qty=None` and the decision row was logged with `order_id=None` —
    position sync saw the shares but attribution couldn't trace them.
    """

    @patch("v2.executor.get_trading_client")
    def test_timeout_with_partial_fill_returns_success(self, mock_client):
        """Order timed out with 50/100 filled before cancel → return success
        with the partial qty so the decision row records the actual fill."""
        # Same mock returned for both poll-loop reads and the post-cancel refetch.
        mock_order = MagicMock()
        mock_order.status.value = "accepted"  # poll loop sees still-active order
        mock_order.filled_qty = "50"
        mock_order.filled_avg_price = "150.25"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-abc", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is True
        assert result.filled_qty == Decimal("50")
        assert result.filled_avg_price == Decimal("150.25")
        assert result.order_id == "order-abc"
        assert "partial fill" in result.error.lower()
        mock_client.return_value.cancel_order_by_id.assert_called_once_with("order-abc")

    @patch("v2.executor.get_trading_client")
    def test_timeout_with_zero_fill_returns_failure(self, mock_client):
        """No fill before cancel → preserve the prior failure semantics so
        the trader marks the trade as failed and doesn't log a phantom row."""
        mock_order = MagicMock()
        mock_order.status.value = "accepted"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-abc", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert result.filled_qty is None
        assert "cancel attempted" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_post_cancel_refetch_failure_falls_through_to_timeout(self, mock_client):
        """If the post-cancel refetch raises (rate limit, network), don't crash —
        return the standard timeout failure. We didn't see a fill we can prove."""
        mock_pending = MagicMock()
        mock_pending.status.value = "accepted"
        mock_pending.filled_qty = None
        mock_pending.filled_avg_price = None
        # Tie the refetch failure to the cancel happening — once cancel fires,
        # any subsequent get_order_by_id call (the post-cancel refetch) raises.
        cancelled = []
        def cancel_se(_):
            cancelled.append(True)
        def get_order_se(_):
            if cancelled:
                raise RuntimeError("rate limit")
            return mock_pending
        mock_client.return_value.cancel_order_by_id.side_effect = cancel_se
        mock_client.return_value.get_order_by_id.side_effect = get_order_se

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-abc", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert result.filled_qty is None
        assert "cancel attempted" in result.error.lower()


class TestQuantityPrecision:
    """Defensive: qty must be quantized to ≤9 decimals (Alpaca's documented limit)
    and rounded DOWN, so we never overshoot a precheck-trimmed sell qty."""

    @patch("v2.executor.get_trading_client")
    def test_market_order_qty_quantized_to_nine_decimals_round_down(self, mock_client):
        mock_order = MagicMock(id="ord-1", filled_qty="0", filled_avg_price=None)
        mock_client.return_value.submit_order.return_value = mock_order

        from v2.executor import execute_market_order
        execute_market_order("AAPL", "buy", Decimal("0.123456789999"))

        request = mock_client.return_value.submit_order.call_args.args[0]
        # 12 decimals → quantized to 9, rounded DOWN → 0.123456789
        assert request.qty == 0.123456789

    @patch("v2.executor.get_trading_client")
    def test_market_order_qty_below_nine_decimals_passes_through(self, mock_client):
        mock_order = MagicMock(id="ord-1", filled_qty="0", filled_avg_price=None)
        mock_client.return_value.submit_order.return_value = mock_order

        from v2.executor import execute_market_order
        execute_market_order("AAPL", "buy", Decimal("2.5"))

        request = mock_client.return_value.submit_order.call_args.args[0]
        assert request.qty == 2.5

    @patch("v2.executor.get_trading_client")
    def test_limit_order_qty_quantized_to_nine_decimals_round_down(self, mock_client):
        mock_order = MagicMock(id="ord-1", filled_qty="0", filled_avg_price=None)
        mock_client.return_value.submit_order.return_value = mock_order

        from v2.executor import execute_limit_order
        execute_limit_order("AAPL", "sell", Decimal("0.987654321999"), Decimal("100.50"))

        request = mock_client.return_value.submit_order.call_args.args[0]
        assert request.qty == 0.987654321


class TestClientOrderId:
    """P1.6: client_order_id is plumbed to Alpaca for broker-side idempotency."""

    @patch("v2.executor.get_trading_client")
    def test_market_order_passes_client_order_id(self, mock_client):
        mock_order = MagicMock(id="ord-1", filled_qty="0", filled_avg_price=None)
        mock_client.return_value.submit_order.return_value = mock_order

        from v2.executor import execute_market_order
        execute_market_order("AAPL", "buy", Decimal("1"), client_order_id="algo-20260502-b-AAPL-42")
        request = mock_client.return_value.submit_order.call_args.args[0]
        assert request.client_order_id == "algo-20260502-b-AAPL-42"

    @patch("v2.executor.get_trading_client")
    def test_market_order_omits_client_order_id_when_none(self, mock_client):
        mock_order = MagicMock(id="ord-1", filled_qty="0", filled_avg_price=None)
        mock_client.return_value.submit_order.return_value = mock_order

        from v2.executor import execute_market_order
        execute_market_order("AAPL", "buy", Decimal("1"))
        request = mock_client.return_value.submit_order.call_args.args[0]
        # alpaca-py defaults unset client_order_id to None — verify we don't set it.
        assert request.client_order_id is None

    @patch("v2.executor.get_trading_client")
    def test_limit_order_passes_client_order_id(self, mock_client):
        mock_order = MagicMock(id="ord-1", filled_qty="0", filled_avg_price=None)
        mock_client.return_value.submit_order.return_value = mock_order

        from v2.executor import execute_limit_order
        execute_limit_order("AAPL", "sell", Decimal("1"), Decimal("150"),
                            client_order_id="algo-20260502-s-AAPL-7")
        request = mock_client.return_value.submit_order.call_args.args[0]
        assert request.client_order_id == "algo-20260502-s-AAPL-7"


class TestDryRunPrice:
    def test_dry_run_order_uses_simulated_price(self):
        from v2.executor import execute_market_order
        result = execute_market_order("AAPL", "buy", Decimal("5"), dry_run=True, simulated_price=Decimal("150.00"))
        assert result.filled_avg_price == Decimal("150.00")

    def test_dry_run_without_simulated_price(self):
        from v2.executor import execute_market_order
        result = execute_market_order("AAPL", "buy", Decimal("5"), dry_run=True)
        assert result.filled_avg_price is None  # backwards compatible
