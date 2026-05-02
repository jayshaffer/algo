"""Tests for v2 outcome backfill with trading-day offsets."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

from v2.backfill import backfill_outcomes, calculate_outcome, trading_day_offset


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
        # Find the AAPL call (not the SPY benchmark calls)
        aapl_calls = [c for c in mock_price.call_args_list if c[0][1] == "AAPL"]
        assert len(aapl_calls) == 1
        assert aapl_calls[0][0][2] == date(2026, 3, 24)


class TestSellAlphaSign:
    """For sell decisions, both outcome AND benchmark must be sign-flipped so
    downstream alpha = outcome - benchmark correctly measures whether the sell
    signal beat the market. Without this flip, every sell during a bull market
    gets wrongly attributed as a large loss, inverting the gradient on every
    signal that motivates exits."""

    @patch("v2.backfill.update_outcome")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    def test_sell_signal_wrong_when_stock_outperforms_market(
        self, mock_get_decisions, mock_client, mock_price, mock_update,
    ):
        """Sell at $100, stock rises to $110 (+10%), SPY rises $400 → $420 (+5%).
        Sell signal was WRONG (stock outperformed market by 5%).
        Expect alpha = outcome - benchmark = -10 - (-5) = -5 → sell missed 5% of relative upside.
        """
        mock_get_decisions.return_value = [{
            "id": 1, "date": date(2026, 3, 13), "ticker": "AAPL",
            "action": "sell", "price": Decimal("100"),
        }]
        mock_client.return_value = MagicMock()

        def price(_client, ticker, dt):
            if ticker == "AAPL":
                return Decimal("110")
            if ticker == "SPY":
                return Decimal("400") if dt == date(2026, 3, 13) else Decimal("420")
        mock_price.side_effect = price

        backfill_outcomes(days=7, dry_run=False)

        outcome, benchmark = mock_update.call_args.args[2:4]
        assert outcome == Decimal("-10")
        assert benchmark == Decimal("-5")  # benchmark must be negated for sells
        assert outcome - benchmark == Decimal("-5")

    @patch("v2.backfill.update_outcome")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    def test_sell_signal_right_when_stock_underperforms_market(
        self, mock_get_decisions, mock_client, mock_price, mock_update,
    ):
        """Sell at $100, stock drops to $90 (-10%), SPY rises $400 → $420 (+5%).
        Sell signal was RIGHT (stock underperformed market by 15%).
        Expect alpha = +10 - (-5) = +15 → sell beat market by 15%.
        """
        mock_get_decisions.return_value = [{
            "id": 1, "date": date(2026, 3, 13), "ticker": "AAPL",
            "action": "sell", "price": Decimal("100"),
        }]
        mock_client.return_value = MagicMock()

        def price(_client, ticker, dt):
            if ticker == "AAPL":
                return Decimal("90")
            if ticker == "SPY":
                return Decimal("400") if dt == date(2026, 3, 13) else Decimal("420")
        mock_price.side_effect = price

        backfill_outcomes(days=7, dry_run=False)

        outcome, benchmark = mock_update.call_args.args[2:4]
        assert outcome == Decimal("10")
        assert benchmark == Decimal("-5")
        assert outcome - benchmark == Decimal("15")

    @patch("v2.backfill.update_outcome")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    def test_buy_benchmark_unchanged(
        self, mock_get_decisions, mock_client, mock_price, mock_update,
    ):
        """Buys keep benchmark sign positive. Buy at $100, stock to $110 (+10%),
        SPY to $420 (+5%). Buy was RIGHT, alpha = +10 - +5 = +5.
        """
        mock_get_decisions.return_value = [{
            "id": 1, "date": date(2026, 3, 13), "ticker": "AAPL",
            "action": "buy", "price": Decimal("100"),
        }]
        mock_client.return_value = MagicMock()

        def price(_client, ticker, dt):
            if ticker == "AAPL":
                return Decimal("110")
            if ticker == "SPY":
                return Decimal("400") if dt == date(2026, 3, 13) else Decimal("420")
        mock_price.side_effect = price

        backfill_outcomes(days=7, dry_run=False)

        outcome, benchmark = mock_update.call_args.args[2:4]
        assert outcome == Decimal("10")
        assert benchmark == Decimal("5")  # buys keep benchmark positive


class TestBackfillNoPrice:
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.update_outcome")
    def test_no_price_skips_decision(self, mock_update, mock_get_price, mock_get_decisions, mock_client):
        mock_client.return_value = MagicMock()
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2026, 1, 1), "ticker": "DELIST",
             "action": "buy", "price": Decimal("50.00")},
        ]
        mock_get_price.return_value = None

        stats = backfill_outcomes(days=7)

        mock_update.assert_not_called()
        assert stats["skipped_no_price"] == 1

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
