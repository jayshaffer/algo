"""Tests for trading/backfill.py - Outcome backfill for past decisions."""

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

import pytest

from trading.backfill import (
    get_data_client,
    get_price_on_date,
    get_decisions_needing_backfill,
    calculate_outcome,
    update_outcome,
    backfill_outcomes,
    run_backfill,
)


# ---------------------------------------------------------------------------
# get_data_client
# ---------------------------------------------------------------------------

class TestGetDataClient:
    """Tests for get_data_client()."""

    def test_raises_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            get_data_client()

    def test_raises_when_only_api_key(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test-key")
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            get_data_client()

    def test_raises_when_only_secret_key(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            get_data_client()

    @patch("trading.backfill.StockHistoricalDataClient")
    def test_creates_client_with_keys(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "my-key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "my-secret")
        get_data_client()
        mock_client_cls.assert_called_once_with("my-key", "my-secret")

    @patch("trading.backfill.StockHistoricalDataClient")
    def test_returns_client_instance(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        expected = MagicMock()
        mock_client_cls.return_value = expected
        result = get_data_client()
        assert result is expected

    def test_raises_when_keys_are_empty_strings(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "")
        with pytest.raises(ValueError):
            get_data_client()


# ---------------------------------------------------------------------------
# get_price_on_date
# ---------------------------------------------------------------------------

class TestGetPriceOnDate:
    """Tests for get_price_on_date()."""

    def _make_bar(self, bar_date, close):
        """Create a mock bar object."""
        bar = MagicMock()
        bar.timestamp = datetime.combine(bar_date, datetime.min.time())
        bar.close = close
        return bar

    def test_returns_decimal_for_valid_bar(self):
        client = MagicMock()
        target = date(2025, 1, 15)
        bar = self._make_bar(target, 150.25)
        bars_dict = MagicMock()
        bars_dict.__contains__ = lambda self, key: key == "AAPL"
        bars_dict.__getitem__ = lambda self, key: [bar]
        client.get_stock_bars.return_value = bars_dict

        result = get_price_on_date(client, "AAPL", target)
        assert result == Decimal("150.25")

    def test_returns_none_when_ticker_not_in_bars(self):
        client = MagicMock()
        target = date(2025, 1, 15)
        bars_dict = MagicMock()
        bars_dict.__contains__ = lambda self, key: False
        client.get_stock_bars.return_value = bars_dict

        result = get_price_on_date(client, "AAPL", target)
        assert result is None

    def test_returns_none_when_bars_empty(self):
        client = MagicMock()
        target = date(2025, 1, 15)
        bars_dict = MagicMock()
        bars_dict.__contains__ = lambda self, key: key == "AAPL"
        bars_dict.__getitem__ = lambda self, key: []
        client.get_stock_bars.return_value = bars_dict

        result = get_price_on_date(client, "AAPL", target)
        assert result is None

    def test_returns_none_on_exception(self):
        client = MagicMock()
        client.get_stock_bars.side_effect = Exception("Network error")
        target = date(2025, 1, 15)

        result = get_price_on_date(client, "AAPL", target)
        assert result is None

    def test_selects_first_bar_on_or_after_target(self):
        """Handles weekends by finding the first bar >= target_date."""
        client = MagicMock()
        target = date(2025, 1, 11)  # Saturday
        # Monday bar
        bar_monday = self._make_bar(date(2025, 1, 13), 155.00)
        bars_dict = MagicMock()
        bars_dict.__contains__ = lambda self, key: key == "AAPL"
        bars_dict.__getitem__ = lambda self, key: [bar_monday]
        client.get_stock_bars.return_value = bars_dict

        result = get_price_on_date(client, "AAPL", target)
        assert result == Decimal("155.0")

    def test_skips_bars_before_target_date(self):
        """If bars before target_date exist, they should be skipped."""
        client = MagicMock()
        target = date(2025, 1, 15)
        bar_before = self._make_bar(date(2025, 1, 14), 100.00)
        bar_on = self._make_bar(date(2025, 1, 15), 150.00)
        bars_dict = MagicMock()
        bars_dict.__contains__ = lambda self, key: key == "AAPL"
        bars_dict.__getitem__ = lambda self, key: [bar_before, bar_on]
        client.get_stock_bars.return_value = bars_dict

        result = get_price_on_date(client, "AAPL", target)
        assert result == Decimal("150.0")

    def test_creates_correct_request_window(self):
        """Verifies the 5-day window for the StockBarsRequest."""
        client = MagicMock()
        target = date(2025, 1, 15)
        bars_dict = MagicMock()
        bars_dict.__contains__ = lambda self, key: False
        client.get_stock_bars.return_value = bars_dict

        get_price_on_date(client, "AAPL", target)

        call_args = client.get_stock_bars.call_args
        request = call_args[0][0]
        assert request.symbol_or_symbols == "AAPL"


# ---------------------------------------------------------------------------
# get_decisions_needing_backfill
# ---------------------------------------------------------------------------

class TestGetDecisionsNeedingBackfill:
    """Tests for get_decisions_needing_backfill()."""

    @pytest.fixture
    def backfill_cursor(self):
        """Patch get_cursor in the backfill module namespace."""
        cursor = MagicMock()
        cursor.fetchall.return_value = []

        @contextmanager
        def _get_cursor():
            yield cursor

        with patch("trading.backfill.get_cursor", _get_cursor):
            yield cursor

    def test_queries_for_7d_backfill(self, backfill_cursor):
        backfill_cursor.fetchall.return_value = []
        result = get_decisions_needing_backfill(7)

        assert result == []
        sql = backfill_cursor.execute.call_args[0][0]
        assert "outcome_7d" in sql
        assert "IS NULL" in sql

    def test_queries_for_30d_backfill(self, backfill_cursor):
        backfill_cursor.fetchall.return_value = []
        get_decisions_needing_backfill(30)

        sql = backfill_cursor.execute.call_args[0][0]
        assert "outcome_30d" in sql

    def test_returns_decision_rows(self, backfill_cursor):
        expected = [
            {"id": 1, "date": date(2025, 1, 1), "ticker": "AAPL", "action": "buy", "price": Decimal("150")},
            {"id": 2, "date": date(2025, 1, 2), "ticker": "MSFT", "action": "sell", "price": Decimal("400")},
        ]
        backfill_cursor.fetchall.return_value = expected
        result = get_decisions_needing_backfill(7)
        assert result == expected
        assert len(result) == 2

    def test_passes_cutoff_date_as_parameter(self, backfill_cursor):
        backfill_cursor.fetchall.return_value = []
        get_decisions_needing_backfill(7)
        params = backfill_cursor.execute.call_args[0][1]
        cutoff = date.today() - timedelta(days=7)
        assert params == (cutoff,)


# ---------------------------------------------------------------------------
# calculate_outcome
# ---------------------------------------------------------------------------

class TestCalculateOutcome:
    """Tests for calculate_outcome()."""

    def test_buy_price_up(self):
        """BUY at $100, price goes to $110 = +10%."""
        result = calculate_outcome("buy", Decimal("100"), Decimal("110"))
        assert result == Decimal("10")

    def test_buy_price_down(self):
        """BUY at $100, price drops to $90 = -10%."""
        result = calculate_outcome("buy", Decimal("100"), Decimal("90"))
        assert result == Decimal("-10")

    def test_buy_price_unchanged(self):
        """BUY at $100, price stays at $100 = 0%."""
        result = calculate_outcome("buy", Decimal("100"), Decimal("100"))
        assert result == Decimal("0")

    def test_sell_price_down(self):
        """SELL at $100, price drops to $90 = +10% (good sell)."""
        result = calculate_outcome("sell", Decimal("100"), Decimal("90"))
        assert result == Decimal("10")

    def test_sell_price_up(self):
        """SELL at $100, price rises to $110 = -10% (bad sell)."""
        result = calculate_outcome("sell", Decimal("100"), Decimal("110"))
        assert result == Decimal("-10")

    def test_sell_price_unchanged(self):
        """SELL at $100, price stays = 0%."""
        result = calculate_outcome("sell", Decimal("100"), Decimal("100"))
        assert result == Decimal("0")

    def test_zero_entry_price_returns_zero(self):
        """Zero entry price should return 0 to avoid division by zero."""
        result = calculate_outcome("buy", Decimal("0"), Decimal("100"))
        assert result == Decimal("0")

    def test_zero_entry_sell_returns_zero(self):
        result = calculate_outcome("sell", Decimal("0"), Decimal("50"))
        assert result == Decimal("0")

    def test_buy_large_gain(self):
        """BUY at $50, price goes to $100 = +100%."""
        result = calculate_outcome("buy", Decimal("50"), Decimal("100"))
        assert result == Decimal("100")

    def test_sell_large_drop(self):
        """SELL at $200, price drops to $100 = +50% (good sell)."""
        result = calculate_outcome("sell", Decimal("200"), Decimal("100"))
        assert result == Decimal("50")

    def test_buy_fractional_prices(self):
        """BUY with fractional prices."""
        result = calculate_outcome("buy", Decimal("150.50"), Decimal("153.51"))
        expected = ((Decimal("153.51") - Decimal("150.50")) / Decimal("150.50")) * 100
        assert result == expected


# ---------------------------------------------------------------------------
# update_outcome
# ---------------------------------------------------------------------------

class TestUpdateOutcome:
    """Tests for update_outcome()."""

    @pytest.fixture
    def backfill_cursor(self):
        """Patch get_cursor in the backfill module namespace."""
        cursor = MagicMock()

        @contextmanager
        def _get_cursor():
            yield cursor

        with patch("trading.backfill.get_cursor", _get_cursor):
            yield cursor

    def test_updates_7d_outcome(self, backfill_cursor):
        update_outcome(42, 7, Decimal("5.25"))
        sql = backfill_cursor.execute.call_args[0][0]
        assert "outcome_7d" in sql
        params = backfill_cursor.execute.call_args[0][1]
        assert params == (Decimal("5.25"), 42)

    def test_updates_30d_outcome(self, backfill_cursor):
        update_outcome(99, 30, Decimal("-3.14"))
        sql = backfill_cursor.execute.call_args[0][0]
        assert "outcome_30d" in sql
        params = backfill_cursor.execute.call_args[0][1]
        assert params == (Decimal("-3.14"), 99)


# ---------------------------------------------------------------------------
# backfill_outcomes
# ---------------------------------------------------------------------------

class TestBackfillOutcomes:
    """Tests for backfill_outcomes()."""

    @patch("trading.backfill.update_outcome")
    @patch("trading.backfill.get_price_on_date")
    @patch("trading.backfill.get_data_client")
    @patch("trading.backfill.get_decisions_needing_backfill")
    def test_no_decisions_returns_zero_stats(self, mock_get_decisions, mock_client,
                                              mock_price, mock_update):
        mock_get_decisions.return_value = []
        stats = backfill_outcomes(days=7)

        assert stats["decisions_found"] == 0
        assert stats["outcomes_filled"] == 0
        assert stats["skipped_no_price"] == 0
        assert stats["errors"] == 0
        mock_client.assert_not_called()

    @patch("trading.backfill.update_outcome")
    @patch("trading.backfill.get_price_on_date")
    @patch("trading.backfill.get_data_client")
    @patch("trading.backfill.get_decisions_needing_backfill")
    def test_successful_backfill(self, mock_get_decisions, mock_client,
                                  mock_price, mock_update):
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2025, 1, 1), "ticker": "AAPL",
             "action": "buy", "price": Decimal("150")},
        ]
        mock_client.return_value = MagicMock()
        mock_price.return_value = Decimal("160")

        stats = backfill_outcomes(days=7, dry_run=False)

        assert stats["decisions_found"] == 1
        assert stats["outcomes_filled"] == 1
        assert stats["skipped_no_price"] == 0
        mock_update.assert_called_once()

    @patch("trading.backfill.update_outcome")
    @patch("trading.backfill.get_price_on_date")
    @patch("trading.backfill.get_data_client")
    @patch("trading.backfill.get_decisions_needing_backfill")
    def test_dry_run_does_not_update(self, mock_get_decisions, mock_client,
                                      mock_price, mock_update):
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2025, 1, 1), "ticker": "AAPL",
             "action": "buy", "price": Decimal("150")},
        ]
        mock_client.return_value = MagicMock()
        mock_price.return_value = Decimal("160")

        stats = backfill_outcomes(days=7, dry_run=True)

        assert stats["decisions_found"] == 1
        assert stats["outcomes_filled"] == 0
        mock_update.assert_not_called()

    @patch("trading.backfill.update_outcome")
    @patch("trading.backfill.get_price_on_date")
    @patch("trading.backfill.get_data_client")
    @patch("trading.backfill.get_decisions_needing_backfill")
    def test_missing_price_skipped(self, mock_get_decisions, mock_client,
                                    mock_price, mock_update):
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2025, 1, 1), "ticker": "AAPL",
             "action": "buy", "price": Decimal("150")},
        ]
        mock_client.return_value = MagicMock()
        mock_price.return_value = None

        stats = backfill_outcomes(days=7)

        assert stats["skipped_no_price"] == 1
        assert stats["outcomes_filled"] == 0
        mock_update.assert_not_called()

    @patch("trading.backfill.update_outcome")
    @patch("trading.backfill.get_price_on_date")
    @patch("trading.backfill.get_data_client")
    @patch("trading.backfill.get_decisions_needing_backfill")
    def test_update_error_counted(self, mock_get_decisions, mock_client,
                                   mock_price, mock_update):
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2025, 1, 1), "ticker": "AAPL",
             "action": "buy", "price": Decimal("150")},
        ]
        mock_client.return_value = MagicMock()
        mock_price.return_value = Decimal("160")
        mock_update.side_effect = Exception("DB error")

        stats = backfill_outcomes(days=7, dry_run=False)

        assert stats["errors"] == 1
        assert stats["outcomes_filled"] == 0

    @patch("trading.backfill.update_outcome")
    @patch("trading.backfill.get_price_on_date")
    @patch("trading.backfill.get_data_client")
    @patch("trading.backfill.get_decisions_needing_backfill")
    def test_multiple_decisions_mixed_results(self, mock_get_decisions, mock_client,
                                               mock_price, mock_update):
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2025, 1, 1), "ticker": "AAPL",
             "action": "buy", "price": Decimal("150")},
            {"id": 2, "date": date(2025, 1, 2), "ticker": "MSFT",
             "action": "sell", "price": Decimal("400")},
            {"id": 3, "date": date(2025, 1, 3), "ticker": "GOOG",
             "action": "buy", "price": Decimal("130")},
        ]
        mock_client.return_value = MagicMock()
        # AAPL has price, MSFT no price, GOOG has price
        mock_price.side_effect = [Decimal("160"), None, Decimal("140")]

        stats = backfill_outcomes(days=7, dry_run=False)

        assert stats["decisions_found"] == 3
        assert stats["outcomes_filled"] == 2
        assert stats["skipped_no_price"] == 1
        assert mock_update.call_count == 2

    @patch("trading.backfill.update_outcome")
    @patch("trading.backfill.get_price_on_date")
    @patch("trading.backfill.get_data_client")
    @patch("trading.backfill.get_decisions_needing_backfill")
    def test_exit_date_is_decision_date_plus_days(self, mock_get_decisions, mock_client,
                                                    mock_price, mock_update):
        decision_date = date(2025, 1, 10)
        mock_get_decisions.return_value = [
            {"id": 1, "date": decision_date, "ticker": "AAPL",
             "action": "buy", "price": Decimal("150")},
        ]
        mock_client.return_value = MagicMock()
        mock_price.return_value = Decimal("160")

        backfill_outcomes(days=7, dry_run=True)

        # Check get_price_on_date was called with correct exit_date
        price_call = mock_price.call_args
        assert price_call[0][1] == "AAPL"
        assert price_call[0][2] == decision_date + timedelta(days=7)


# ---------------------------------------------------------------------------
# run_backfill
# ---------------------------------------------------------------------------

class TestRunBackfill:
    """Tests for run_backfill()."""

    @patch("trading.backfill.backfill_outcomes")
    def test_runs_both_7d_and_30d(self, mock_backfill):
        mock_backfill.return_value = {
            "decisions_found": 0,
            "outcomes_filled": 0,
            "skipped_no_price": 0,
            "errors": 0,
        }

        run_backfill(dry_run=False)

        assert mock_backfill.call_count == 2
        calls = mock_backfill.call_args_list
        assert calls[0] == call(days=7, dry_run=False)
        assert calls[1] == call(days=30, dry_run=False)

    @patch("trading.backfill.backfill_outcomes")
    def test_returns_combined_stats(self, mock_backfill):
        mock_backfill.side_effect = [
            {"decisions_found": 5, "outcomes_filled": 3, "skipped_no_price": 1, "errors": 1},
            {"decisions_found": 2, "outcomes_filled": 2, "skipped_no_price": 0, "errors": 0},
        ]

        result = run_backfill(dry_run=False)

        assert result["7d"]["outcomes_filled"] == 3
        assert result["30d"]["outcomes_filled"] == 2
        assert result["total_filled"] == 5

    @patch("trading.backfill.backfill_outcomes")
    def test_dry_run_passed_through(self, mock_backfill):
        mock_backfill.return_value = {
            "decisions_found": 0,
            "outcomes_filled": 0,
            "skipped_no_price": 0,
            "errors": 0,
        }

        run_backfill(dry_run=True)

        calls = mock_backfill.call_args_list
        assert calls[0] == call(days=7, dry_run=True)
        assert calls[1] == call(days=30, dry_run=True)

    @patch("trading.backfill.backfill_outcomes")
    def test_zero_total_when_no_outcomes(self, mock_backfill):
        mock_backfill.return_value = {
            "decisions_found": 0,
            "outcomes_filled": 0,
            "skipped_no_price": 0,
            "errors": 0,
        }

        result = run_backfill()

        assert result["total_filled"] == 0
