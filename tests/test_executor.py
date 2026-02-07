"""Tests for trading/executor.py - trade execution and position management."""

from decimal import Decimal
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

import pytest

from trading.executor import (
    OrderResult,
    get_trading_client,
    get_account_info,
    take_account_snapshot,
    sync_positions_from_alpaca,
    sync_orders_from_alpaca,
    execute_market_order,
    execute_limit_order,
    get_latest_price,
    calculate_position_size,
)
from tests.conftest import make_position_row


# ---------------------------------------------------------------------------
# OrderResult dataclass
# ---------------------------------------------------------------------------


class TestOrderResult:
    def test_success_result(self):
        r = OrderResult(
            success=True,
            order_id="abc-123",
            filled_qty=Decimal("10"),
            filled_avg_price=Decimal("150.50"),
            error=None,
        )
        assert r.success is True
        assert r.order_id == "abc-123"
        assert r.error is None

    def test_failure_result(self):
        r = OrderResult(
            success=False,
            order_id=None,
            filled_qty=None,
            filled_avg_price=None,
            error="Insufficient funds",
        )
        assert r.success is False
        assert r.error == "Insufficient funds"


# ---------------------------------------------------------------------------
# get_trading_client
# ---------------------------------------------------------------------------


class TestGetTradingClient:
    def test_raises_without_keys(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            get_trading_client()

    def test_raises_with_only_api_key(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "key")
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ValueError):
            get_trading_client()

    @patch("trading.executor.TradingClient")
    def test_creates_paper_client(self, mock_tc, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        get_trading_client()
        mock_tc.assert_called_once_with("key", "secret", paper=True)

    @patch("trading.executor.TradingClient")
    def test_creates_live_client(self, mock_tc, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

        get_trading_client()
        mock_tc.assert_called_once_with("key", "secret", paper=False)


# ---------------------------------------------------------------------------
# get_account_info
# ---------------------------------------------------------------------------


class TestGetAccountInfo:
    @patch("trading.executor.get_trading_client")
    def test_returns_decimal_values(self, mock_get_client, mock_trading_client):
        mock_get_client.return_value = mock_trading_client
        info = get_account_info()

        assert info["account_id"] == "test-account"
        assert info["status"] == "ACTIVE"
        assert isinstance(info["cash"], Decimal)
        assert info["cash"] == Decimal("100000.00")
        assert isinstance(info["buying_power"], Decimal)
        assert info["buying_power"] == Decimal("200000.00")
        assert info["portfolio_value"] == Decimal("150000.00")
        assert info["equity"] == Decimal("150000.00")
        assert info["daytrade_count"] == 0
        assert info["pattern_day_trader"] is False


# ---------------------------------------------------------------------------
# take_account_snapshot
# ---------------------------------------------------------------------------


class TestTakeAccountSnapshot:
    @patch("trading.executor.insert_account_snapshot", return_value=42)
    @patch("trading.executor.get_account_info")
    def test_calls_insert_with_correct_values(self, mock_info, mock_insert):
        mock_info.return_value = {
            "cash": Decimal("100000"),
            "portfolio_value": Decimal("150000"),
            "buying_power": Decimal("200000"),
            "long_market_value": Decimal("50000"),
            "short_market_value": Decimal("0"),
        }
        result = take_account_snapshot()

        assert result == 42
        mock_insert.assert_called_once()
        call_kwargs = mock_insert.call_args.kwargs
        assert call_kwargs["cash"] == Decimal("100000")
        assert call_kwargs["portfolio_value"] == Decimal("150000")


# ---------------------------------------------------------------------------
# sync_positions_from_alpaca
# ---------------------------------------------------------------------------


class TestSyncPositionsFromAlpaca:
    @patch("trading.executor.delete_position")
    @patch("trading.executor.upsert_position")
    @patch("trading.executor.db_get_positions")
    @patch("trading.executor.get_trading_client")
    def test_syncs_new_positions(
        self, mock_get_client, mock_db_positions, mock_upsert, mock_delete
    ):
        # Setup Alpaca positions
        pos = MagicMock()
        pos.symbol = "AAPL"
        pos.qty = "10"
        pos.avg_entry_price = "150.00"
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [pos]
        mock_get_client.return_value = mock_client

        mock_db_positions.return_value = []

        result = sync_positions_from_alpaca()

        assert result == 1
        mock_upsert.assert_called_once_with(
            ticker="AAPL",
            shares=Decimal("10"),
            avg_cost=Decimal("150.00"),
        )
        mock_delete.assert_not_called()

    @patch("trading.executor.delete_position")
    @patch("trading.executor.upsert_position")
    @patch("trading.executor.db_get_positions")
    @patch("trading.executor.get_trading_client")
    def test_removes_stale_db_positions(
        self, mock_get_client, mock_db_positions, mock_upsert, mock_delete
    ):
        """Positions in DB but not in Alpaca should be deleted."""
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_get_client.return_value = mock_client

        mock_db_positions.return_value = [
            make_position_row(ticker="MSFT"),
        ]

        result = sync_positions_from_alpaca()

        assert result == 0
        mock_delete.assert_called_once_with("MSFT")

    @patch("trading.executor.delete_position")
    @patch("trading.executor.upsert_position")
    @patch("trading.executor.db_get_positions")
    @patch("trading.executor.get_trading_client")
    def test_updates_existing_and_removes_stale(
        self, mock_get_client, mock_db_positions, mock_upsert, mock_delete
    ):
        """Alpaca has AAPL, DB has AAPL + MSFT => upsert AAPL, delete MSFT."""
        pos = MagicMock()
        pos.symbol = "AAPL"
        pos.qty = "20"
        pos.avg_entry_price = "155.00"
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [pos]
        mock_get_client.return_value = mock_client

        mock_db_positions.return_value = [
            make_position_row(ticker="AAPL"),
            make_position_row(ticker="MSFT"),
        ]

        result = sync_positions_from_alpaca()

        assert result == 1
        mock_upsert.assert_called_once()
        mock_delete.assert_called_once_with("MSFT")


# ---------------------------------------------------------------------------
# sync_orders_from_alpaca
# ---------------------------------------------------------------------------


class TestSyncOrdersFromAlpaca:
    @patch("trading.executor.delete_open_order")
    @patch("trading.executor.upsert_open_order")
    @patch("trading.executor.db_get_open_orders")
    @patch("trading.executor.get_trading_client")
    def test_syncs_orders(
        self, mock_get_client, mock_db_orders, mock_upsert, mock_delete
    ):
        order = MagicMock()
        order.id = "order-1"
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.qty = "10"
        order.filled_qty = "0"
        order.limit_price = None
        order.stop_price = None
        order.status.value = "new"
        order.submitted_at = datetime(2025, 1, 15, 10, 0, 0)

        mock_client = MagicMock()
        mock_client.get_orders.return_value = [order]
        mock_get_client.return_value = mock_client
        mock_db_orders.return_value = []

        result = sync_orders_from_alpaca()
        assert result == 1
        mock_upsert.assert_called_once()
        mock_delete.assert_not_called()

    @patch("trading.executor.delete_open_order")
    @patch("trading.executor.upsert_open_order")
    @patch("trading.executor.db_get_open_orders")
    @patch("trading.executor.get_trading_client")
    def test_removes_stale_db_orders(
        self, mock_get_client, mock_db_orders, mock_upsert, mock_delete
    ):
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        mock_get_client.return_value = mock_client
        mock_db_orders.return_value = [{"order_id": "old-order-1"}]

        result = sync_orders_from_alpaca()
        assert result == 0
        mock_delete.assert_called_once_with("old-order-1")

    @patch("trading.executor.delete_open_order")
    @patch("trading.executor.upsert_open_order")
    @patch("trading.executor.db_get_open_orders")
    @patch("trading.executor.get_trading_client")
    def test_handles_limit_and_stop_prices(
        self, mock_get_client, mock_db_orders, mock_upsert, mock_delete
    ):
        order = MagicMock()
        order.id = "order-lim"
        order.symbol = "TSLA"
        order.side.value = "sell"
        order.order_type.value = "limit"
        order.qty = "5"
        order.filled_qty = "2"
        order.limit_price = "250.00"
        order.stop_price = "240.00"
        order.status.value = "partially_filled"
        order.submitted_at = datetime(2025, 1, 15, 10, 0, 0)

        mock_client = MagicMock()
        mock_client.get_orders.return_value = [order]
        mock_get_client.return_value = mock_client
        mock_db_orders.return_value = []

        result = sync_orders_from_alpaca()
        assert result == 1
        call_kwargs = mock_upsert.call_args.kwargs
        assert call_kwargs["ticker"] == "TSLA"
        assert call_kwargs["limit_price"] == Decimal("250.00")
        assert call_kwargs["stop_price"] == Decimal("240.00")
        assert call_kwargs["filled_qty"] == Decimal("2")


# ---------------------------------------------------------------------------
# execute_market_order
# ---------------------------------------------------------------------------


class TestExecuteMarketOrder:
    def test_dry_run_returns_success(self):
        result = execute_market_order(
            ticker="AAPL", side="buy", qty=Decimal("10"), dry_run=True
        )
        assert result.success is True
        assert result.order_id == "DRY_RUN"
        assert result.filled_qty == Decimal("10")
        assert result.error is None

    def test_dry_run_sell(self):
        result = execute_market_order(
            ticker="MSFT", side="sell", qty=Decimal("5"), dry_run=True
        )
        assert result.success is True
        assert result.order_id == "DRY_RUN"

    @patch("trading.executor.get_trading_client")
    def test_live_buy_success(self, mock_get_client):
        mock_order = MagicMock()
        mock_order.id = "live-order-1"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.25"
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_get_client.return_value = mock_client

        result = execute_market_order(
            ticker="AAPL", side="buy", qty=Decimal("10"), dry_run=False
        )
        assert result.success is True
        assert result.order_id == "live-order-1"
        assert result.filled_qty == Decimal("10")
        assert result.filled_avg_price == Decimal("150.25")

    @patch("trading.executor.get_trading_client")
    def test_live_sell_success(self, mock_get_client):
        mock_order = MagicMock()
        mock_order.id = "sell-order-1"
        mock_order.filled_qty = "5"
        mock_order.filled_avg_price = "160.00"
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_get_client.return_value = mock_client

        result = execute_market_order(
            ticker="AAPL", side="sell", qty=Decimal("5"), dry_run=False
        )
        assert result.success is True

    @patch("trading.executor.get_trading_client")
    def test_live_order_exception_returns_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("API timeout")
        mock_get_client.return_value = mock_client

        result = execute_market_order(
            ticker="AAPL", side="buy", qty=Decimal("10"), dry_run=False
        )
        assert result.success is False
        assert result.order_id is None
        assert "API timeout" in result.error

    @patch("trading.executor.get_trading_client")
    def test_live_order_no_fill_info(self, mock_get_client):
        """Order submitted but not yet filled."""
        mock_order = MagicMock()
        mock_order.id = "pending-order"
        mock_order.filled_qty = None
        mock_order.filled_avg_price = None
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_get_client.return_value = mock_client

        result = execute_market_order(
            ticker="AAPL", side="buy", qty=Decimal("10"), dry_run=False
        )
        assert result.success is True
        assert result.filled_qty is None
        assert result.filled_avg_price is None


# ---------------------------------------------------------------------------
# execute_limit_order
# ---------------------------------------------------------------------------


class TestExecuteLimitOrder:
    def test_dry_run_returns_success_with_limit_price(self):
        result = execute_limit_order(
            ticker="AAPL",
            side="buy",
            qty=Decimal("10"),
            limit_price=Decimal("145.00"),
            dry_run=True,
        )
        assert result.success is True
        assert result.order_id == "DRY_RUN"
        assert result.filled_qty == Decimal("10")
        assert result.filled_avg_price == Decimal("145.00")

    @patch("trading.executor.get_trading_client")
    def test_live_limit_order_success(self, mock_get_client):
        mock_order = MagicMock()
        mock_order.id = "limit-order-1"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "144.50"
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        mock_get_client.return_value = mock_client

        result = execute_limit_order(
            ticker="AAPL",
            side="buy",
            qty=Decimal("10"),
            limit_price=Decimal("145.00"),
            dry_run=False,
        )
        assert result.success is True
        assert result.order_id == "limit-order-1"

    @patch("trading.executor.get_trading_client")
    def test_live_limit_order_exception(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("Rejected")
        mock_get_client.return_value = mock_client

        result = execute_limit_order(
            ticker="AAPL",
            side="sell",
            qty=Decimal("5"),
            limit_price=Decimal("200.00"),
            dry_run=False,
        )
        assert result.success is False
        assert "Rejected" in result.error


# ---------------------------------------------------------------------------
# get_latest_price
# ---------------------------------------------------------------------------


class TestGetLatestPrice:
    def test_returns_decimal_price(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

        mock_quote = MagicMock()
        mock_quote.ask_price = 155.25
        mock_hist_client = MagicMock()
        mock_hist_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        with patch("alpaca.data.historical.StockHistoricalDataClient", return_value=mock_hist_client):
            price = get_latest_price("AAPL")

        assert price == Decimal("155.25")

    def test_returns_none_on_exception(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")

        mock_hist_client = MagicMock()
        mock_hist_client.get_stock_latest_quote.side_effect = Exception("No data")

        with patch("alpaca.data.historical.StockHistoricalDataClient", return_value=mock_hist_client):
            price = get_latest_price("BADTICKER")

        assert price is None


# ---------------------------------------------------------------------------
# calculate_position_size
# ---------------------------------------------------------------------------


class TestCalculatePositionSize:
    def test_basic_calculation(self):
        shares = calculate_position_size(
            buying_power=Decimal("100000"),
            price=Decimal("150"),
            risk_pct=0.05,
        )
        # 100000 * 0.05 = 5000 / 150 = 33.3333
        assert shares == pytest.approx(33.3333, abs=0.0001)

    def test_small_buying_power(self):
        shares = calculate_position_size(
            buying_power=Decimal("100"),
            price=Decimal("150"),
            risk_pct=0.05,
        )
        # 100 * 0.05 = 5 / 150 = 0.0333
        assert shares == pytest.approx(0.0333, abs=0.0001)

    def test_returns_float(self):
        shares = calculate_position_size(
            buying_power=Decimal("10000"),
            price=Decimal("33.33"),
        )
        assert isinstance(shares, float)

    def test_zero_buying_power(self):
        shares = calculate_position_size(
            buying_power=Decimal("0"),
            price=Decimal("150"),
        )
        assert shares == 0

    def test_high_risk_percentage(self):
        shares = calculate_position_size(
            buying_power=Decimal("100000"),
            price=Decimal("100"),
            risk_pct=0.50,
        )
        # 100000 * 0.50 = 50000 / 100 = 500
        assert shares == 500.0

    def test_very_expensive_stock(self):
        shares = calculate_position_size(
            buying_power=Decimal("10000"),
            price=Decimal("5000"),
            risk_pct=0.05,
        )
        # 10000 * 0.05 = 500 / 5000 = 0.1
        assert shares == 0.1

    def test_cheap_stock_many_shares(self):
        shares = calculate_position_size(
            buying_power=Decimal("100000"),
            price=Decimal("1.00"),
            risk_pct=0.05,
        )
        # 100000 * 0.05 = 5000 / 1 = 5000
        assert shares == 5000.0

    def test_never_negative(self):
        """Even with unusual inputs, result should be >= 0."""
        shares = calculate_position_size(
            buying_power=Decimal("0"),
            price=Decimal("100"),
            risk_pct=0.0,
        )
        assert shares >= 0

    def test_default_risk_pct(self):
        shares = calculate_position_size(
            buying_power=Decimal("100000"),
            price=Decimal("100"),
        )
        # Default 5%: 100000 * 0.05 = 5000 / 100 = 50
        assert shares == 50.0
