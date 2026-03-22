"""Tests for v2 outcome backfill with trading-day offsets."""

from datetime import date
from decimal import Decimal
from unittest.mock import patch, MagicMock

from v2.backfill import trading_day_offset, calculate_outcome, backfill_outcomes


class TestTradingDayOffset:
    def test_friday_7_trading_days(self):
        friday = date(2026, 3, 13)
        result = trading_day_offset(friday, 7)
        # 7 trading days from Fri: Mon=1,Tue=2,Wed=3,Thu=4,Fri=5,Mon=6,Tue=7
        assert result == date(2026, 3, 24)

    def test_monday_7_trading_days(self):
        monday = date(2026, 3, 16)
        result = trading_day_offset(monday, 7)
        # Tue=1,Wed=2,Thu=3,Fri=4,Mon=5,Tue=6,Wed=7
        assert result == date(2026, 3, 25)

    def test_zero_offset_from_saturday(self):
        saturday = date(2026, 3, 14)
        result = trading_day_offset(saturday, 0)
        assert result == date(2026, 3, 16)  # Monday

    def test_zero_offset_from_weekday(self):
        wednesday = date(2026, 3, 18)
        result = trading_day_offset(wednesday, 0)
        assert result == date(2026, 3, 18)

    def test_one_trading_day_from_friday(self):
        friday = date(2026, 3, 13)
        result = trading_day_offset(friday, 1)
        assert result == date(2026, 3, 16)  # Monday

    def test_30_trading_days(self):
        monday = date(2026, 3, 2)
        result = trading_day_offset(monday, 30)
        # 30 trading days = 6 weeks = 42 calendar days
        assert result == date(2026, 4, 13)


class TestCalculateOutcome:
    def test_buy_positive(self):
        result = calculate_outcome("buy", Decimal("100"), Decimal("110"))
        assert result == Decimal("10")

    def test_sell_positive(self):
        result = calculate_outcome("sell", Decimal("100"), Decimal("90"))
        assert result == Decimal("10")

    def test_zero_entry_price(self):
        result = calculate_outcome("buy", Decimal("0"), Decimal("10"))
        assert result == Decimal("0")


class TestBackfillOutcomes:
    @patch("v2.backfill.update_outcome")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    def test_uses_trading_days_for_exit_date(
        self, mock_get_decisions, mock_client, mock_price, mock_update
    ):
        """Verify backfill uses trading_day_offset instead of calendar days."""
        mock_get_decisions.return_value = [
            {
                "id": 1,
                "date": date(2026, 3, 13),  # Friday
                "ticker": "AAPL",
                "action": "buy",
                "price": Decimal("150.00"),
            }
        ]
        mock_client.return_value = MagicMock()
        mock_price.return_value = Decimal("155.00")

        backfill_outcomes(days=7, dry_run=False)

        # Should call get_price_on_date with trading day offset (March 24),
        # not calendar offset (March 20, which is a Friday but wrong date)
        call_args = mock_price.call_args
        assert call_args[0][1] == "AAPL"
        assert call_args[0][2] == date(2026, 3, 24)


class TestBackfillNoPrice:
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.update_outcome")
    def test_no_price_records_sentinel(self, mock_update, mock_get_price, mock_get_decisions, mock_client):
        mock_client.return_value = MagicMock()
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2026, 1, 1), "ticker": "DELIST",
             "action": "buy", "price": Decimal("50.00")},
        ]
        mock_get_price.return_value = None

        stats = backfill_outcomes(days=7)

        mock_update.assert_called_once()
        args = mock_update.call_args[0]
        assert args[0] == 1  # decision_id
        assert args[1] == 7  # days
        assert args[2] == Decimal("-100")  # sentinel

    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.update_outcome")
    def test_no_price_dry_run_does_not_update(self, mock_update, mock_get_price, mock_get_decisions, mock_client):
        mock_client.return_value = MagicMock()
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2026, 1, 1), "ticker": "DELIST",
             "action": "buy", "price": Decimal("50.00")},
        ]
        mock_get_price.return_value = None

        stats = backfill_outcomes(days=7, dry_run=True)

        mock_update.assert_not_called()
