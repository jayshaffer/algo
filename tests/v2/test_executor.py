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


class TestGetLatestPriceWithReason:
    """ALGO-14: rejection paths return a structured reason string so the
    executor can surface it (instead of the misleading "no price available")
    and postmortems don't have to replay Alpaca state."""

    @patch("v2.executor.StockHistoricalDataClient")
    def test_success_returns_price_and_none_reason(self, mock_data_client_cls):
        from v2.executor import get_latest_price_with_reason
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.0
        mock_quote.bid_price = 149.5
        mock_quote.timestamp = datetime.now(UTC)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            price, reason = get_latest_price_with_reason("AAPL")
        assert price == Decimal("150.0")
        assert reason is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_stale_quote_reason_names_age_and_max(self, mock_data_client_cls):
        from v2.executor import get_latest_price_with_reason
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.0
        mock_quote.bid_price = 149.5
        mock_quote.timestamp = datetime.now(UTC) - timedelta(seconds=120)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            price, reason = get_latest_price_with_reason("AAPL", max_age_seconds=60)
        assert price is None
        assert reason is not None
        assert "stale" in reason
        assert "60s" in reason  # max
        assert "120s" in reason or "119s" in reason or "121s" in reason  # observed age

    @patch("v2.executor.StockHistoricalDataClient")
    def test_wide_spread_reason_names_spread_and_max(self, mock_data_client_cls):
        from v2.executor import get_latest_price_with_reason
        mock_quote = MagicMock()
        mock_quote.ask_price = 160.0  # ~10.3% above bid
        mock_quote.bid_price = 145.0
        mock_quote.timestamp = datetime.now(UTC)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            price, reason = get_latest_price_with_reason("AAPL", max_spread_pct=Decimal("0.05"))
        assert price is None
        assert reason is not None
        assert "spread" in reason
        assert "5.00%" in reason  # max
        assert "10.34%" in reason  # observed (15/145)
        assert "bid=145" in reason and "ask=160" in reason

    @patch("v2.executor.StockHistoricalDataClient")
    def test_zero_ask_reason(self, mock_data_client_cls):
        from v2.executor import get_latest_price_with_reason
        mock_quote = MagicMock()
        mock_quote.ask_price = 0
        mock_quote.bid_price = 0
        mock_quote.timestamp = datetime.now(UTC)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            price, reason = get_latest_price_with_reason("AAPL")
        assert price is None
        assert reason == "quote ask is zero"

    @patch("v2.executor.StockHistoricalDataClient")
    def test_api_error_reason_includes_exception_msg(self, mock_data_client_cls):
        from v2.executor import get_latest_price_with_reason
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.side_effect = Exception("alpaca 500")
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            price, reason = get_latest_price_with_reason("AAPL")
        assert price is None
        assert reason is not None
        assert "API error" in reason
        assert "alpaca 500" in reason

    @patch("v2.executor.StockHistoricalDataClient")
    def test_get_latest_price_still_returns_price_only(self, mock_data_client_cls):
        """Backward compat: the legacy get_latest_price() signature is preserved
        for callers that don't care about the reason (e.g. v2/context.py)."""
        from v2.executor import get_latest_price
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.0
        mock_quote.bid_price = 149.5
        mock_quote.timestamp = datetime.now(UTC)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            price = get_latest_price("AAPL")
        assert price == Decimal("150.0")


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
        """No fill before cancel → success=False so the trader marks the
        trade as failed. T2.8: the post-cancel re-fetch confirmed zero, so
        filled_qty is Decimal('0') (not None) and unknown_partial_fill
        stays False — this is a clean miss, not an ambiguous outcome.
        """
        mock_order = MagicMock()
        mock_order.status.value = "accepted"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-abc", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert result.filled_qty == Decimal("0")
        assert result.unknown_partial_fill is False
        assert "cancel attempted" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_timeout_with_unknown_post_cancel_state(self, mock_client):
        """T2.8: if post-cancel re-fetch fails, we don't know whether a
        partial fill landed. The result must carry `unknown_partial_fill=True`
        so the trader treats the order as needing reconciliation, not as a
        clean miss.
        """
        # First call: poll loop, "accepted". Second call: post-cancel
        # re-fetch, raises.
        accepted = MagicMock()
        accepted.status.value = "accepted"
        accepted.filled_qty = None
        accepted.filled_avg_price = None
        mock_client.return_value.get_order_by_id.side_effect = [
            accepted, accepted, accepted, accepted, accepted,  # poll iterations
            Exception("network blip"),  # post-cancel re-fetch
        ]

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-xyz", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert result.unknown_partial_fill is True
        assert "partial-fill state unknown" in result.error.lower()


class TestWaitForFillTransientFetchRetry:
    """T2.9: transient errors fetching order state must not abort the
    poll loop. The function should swallow brief failures and retry up
    to a small limit before treating the situation as fatal.
    """

    @patch("v2.executor.get_trading_client")
    def test_one_transient_error_then_success(self, mock_client):
        filled = MagicMock()
        filled.status.value = "filled"
        filled.filled_qty = "5"
        filled.filled_avg_price = "100.00"
        mock_client.return_value.get_order_by_id.side_effect = [
            Exception("blip"),
            filled,
        ]

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-1", timeout_seconds=5, poll_interval=0.01)

        assert result.success is True
        assert result.filled_qty == Decimal("5")
        # Both calls happened: the poll loop retried after the transient.
        assert mock_client.return_value.get_order_by_id.call_count == 2

    @patch("v2.executor.get_trading_client")
    def test_persistent_errors_break_poll_loop_and_attempt_cancel(self, mock_client):
        """3 consecutive errors break the loop, cancel is attempted, then the
        post-cancel re-fetch path runs. With the post-cancel re-fetch also
        erroring, the result must carry `unknown_partial_fill=True`.
        """
        mock_client.return_value.get_order_by_id.side_effect = Exception("persistent")

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-2", timeout_seconds=5, poll_interval=0.01)

        # Cancel was still attempted.
        mock_client.return_value.cancel_order_by_id.assert_called_once_with("order-2")
        assert result.success is False
        assert result.unknown_partial_fill is True

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

    @patch("v2.executor.get_trading_client")
    def test_market_order_flags_duplicate_client_order_id(self, mock_client):
        """Concurrent run race: Alpaca rejects a re-submitted client_order_id
        with HTTP 422. The executor should flag this distinctly so the trader
        can log it as a benign skip rather than a real execution error."""
        mock_client.return_value.submit_order.side_effect = Exception(
            "client_order_id must be unique"
        )

        from v2.executor import execute_market_order
        result = execute_market_order(
            "AAPL", "buy", Decimal("1"), client_order_id="algo-20260502-b-AAPL-42"
        )
        assert result.success is False
        assert result.duplicate_client_order_id is True

    @patch("v2.executor.get_trading_client")
    def test_market_order_does_not_flag_other_errors_as_duplicate(self, mock_client):
        mock_client.return_value.submit_order.side_effect = Exception(
            "insufficient buying power"
        )

        from v2.executor import execute_market_order
        result = execute_market_order(
            "AAPL", "buy", Decimal("1"), client_order_id="algo-20260502-b-AAPL-42"
        )
        assert result.success is False
        assert result.duplicate_client_order_id is False


class TestAlpacaEnvValidation:
    """T1.4: ALPACA_PAPER must be explicit and consistent with ALPACA_BASE_URL.

    Pre-fix the executor silently defaulted ALPACA_BASE_URL to paper and
    derived `paper=True` from URL substring search. A prod key with a missing
    URL silently routed to paper; `paper=true` in code with a prod URL would
    cause submitting live orders against a paper-flagged client.
    """

    def test_missing_alpaca_paper_raises_clear_error(self, monkeypatch):
        from v2.executor import _validate_alpaca_env
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.delenv("ALPACA_PAPER", raising=False)
        import pytest
        with pytest.raises(RuntimeError, match="ALPACA_PAPER env var is required"):
            _validate_alpaca_env()

    def test_invalid_alpaca_paper_value_raises(self, monkeypatch):
        from v2.executor import _validate_alpaca_env
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.setenv("ALPACA_PAPER", "yes")
        import pytest
        with pytest.raises(RuntimeError, match="must be 'true' or 'false'"):
            _validate_alpaca_env()

    def test_paper_true_with_prod_url_raises(self, monkeypatch):
        from v2.executor import _validate_alpaca_env
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        monkeypatch.setenv("ALPACA_PAPER", "true")
        import pytest
        with pytest.raises(RuntimeError, match="disagrees with ALPACA_BASE_URL"):
            _validate_alpaca_env()

    def test_paper_false_with_paper_url_raises(self, monkeypatch):
        from v2.executor import _validate_alpaca_env
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.setenv("ALPACA_PAPER", "false")
        import pytest
        with pytest.raises(RuntimeError, match="disagrees with ALPACA_BASE_URL"):
            _validate_alpaca_env()

    def test_consistent_paper_passes(self, monkeypatch):
        from v2.executor import _validate_alpaca_env
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.setenv("ALPACA_PAPER", "true")
        _validate_alpaca_env()  # should not raise

    def test_consistent_prod_passes(self, monkeypatch):
        from v2.executor import _validate_alpaca_env
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        monkeypatch.setenv("ALPACA_PAPER", "false")
        _validate_alpaca_env()  # should not raise

    def test_validation_skipped_when_no_api_key(self, monkeypatch):
        """Tests / non-trading code paths that import executor without
        configuring Alpaca must not be punished for it."""
        from v2.executor import _validate_alpaca_env
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_PAPER", raising=False)
        _validate_alpaca_env()  # should not raise


class TestDryRunPrice:
    def test_dry_run_order_uses_simulated_price(self):
        from v2.executor import execute_market_order
        result = execute_market_order("AAPL", "buy", Decimal("5"), dry_run=True, simulated_price=Decimal("150.00"))
        assert result.filled_avg_price == Decimal("150.00")

    def test_dry_run_without_simulated_price(self):
        from v2.executor import execute_market_order
        result = execute_market_order("AAPL", "buy", Decimal("5"), dry_run=True)
        assert result.filled_avg_price is None  # backwards compatible
