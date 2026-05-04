"""Tests for v2 outcome backfill with trading-day offsets."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

from v2.backfill import (
    backfill_outcomes,
    calculate_outcome,
    get_decisions_needing_backfill,
    trading_day_cutoff,
    trading_day_offset,
)


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
        # 30 trading days from Mon 3/2: 6 weeks of trading days (4/13)
        # plus one extra calendar day because Good Friday 2026-04-03 is
        # an NYSE holiday inside the window.
        assert result == date(2026, 4, 14)

    def test_skips_christmas_day(self):
        """+1 trading day from Wed 2025-12-24 lands on Fri 2025-12-26
        because Christmas (Thu 2025-12-25) is an NYSE holiday."""
        wednesday = date(2025, 12, 24)
        result = trading_day_offset(wednesday, 1)
        assert result == date(2025, 12, 26)

    def test_zero_offset_advances_off_holiday(self):
        """+0 from a market holiday (Christmas Thu 2025-12-25) should
        return the next trading day (Fri 2025-12-26)."""
        christmas = date(2025, 12, 25)
        result = trading_day_offset(christmas, 0)
        assert result == date(2025, 12, 26)


class TestTradingDayCutoff:
    def test_seven_trading_days_back_from_friday(self):
        """T1.9: 7 trading days back from Friday spans 9 calendar days
        (skipping the weekend), not 7. The previous calendar-day cutoff
        let a Friday-decision become eligible on the next Friday — only
        5 trading days closed, not 7.
        """
        friday = date(2026, 3, 20)
        result = trading_day_cutoff(friday, 7)
        # Going back: Thu=1, Wed=2, Tue=3, Mon=4, Fri(prev)=5, Thu=6, Wed=7
        assert result == date(2026, 3, 11)

    def test_seven_trading_days_back_from_monday(self):
        """7 trading days back from Monday lands on the Thursday before."""
        monday = date(2026, 3, 23)
        result = trading_day_cutoff(monday, 7)
        # Sun(weekend), Sat(weekend), Fri=1, Thu=2, Wed=3, Tue=4, Mon=5, Fri(prev)=6, Thu=7
        assert result == date(2026, 3, 12)

    def test_thirty_trading_days_back(self):
        """30 trading days back ≈ 6 weeks of calendar days, plus any
        NYSE holidays inside the window."""
        friday = date(2026, 5, 1)
        result = trading_day_cutoff(friday, 30)
        # Good Friday 2026-04-03 is inside the window, so the cutoff
        # extends one trading day further back.
        assert result == date(2026, 3, 19)


class TestGetDecisionsNeedingBackfillCutoff:
    """T1.9: cutoff must be trading-day aware so a decision made 7 calendar
    days ago on a Friday is NOT yet eligible (only 5 trading days closed),
    while one made 10 calendar days ago (= 7 trading days) IS eligible.
    """

    def test_cutoff_is_trading_days_not_calendar_days(self):
        from contextlib import contextmanager

        cursor = MagicMock()
        cursor.fetchall.return_value = []

        @contextmanager
        def _get_cursor():
            yield cursor

        # Anchor "today" to a Friday so the bug shape is testable.
        with patch("v2.backfill.get_cursor", _get_cursor), \
             patch("v2.backfill.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 20)
            # Allow real date(...) construction.
            mock_date.side_effect = lambda *a, **k: date(*a, **k)

            get_decisions_needing_backfill(days_threshold=7)

        # Calendar-day cutoff would be 2026-03-13 (the prior Friday).
        # Trading-day cutoff is 2026-03-11 (Wednesday before).
        # Therefore a decision made on 2026-03-13 must NOT be eligible.
        passed_cutoff = cursor.execute.call_args[0][1][0]
        assert passed_cutoff == date(2026, 3, 11), (
            f"Expected trading-day cutoff 2026-03-11, got {passed_cutoff}"
        )
        assert passed_cutoff < date(2026, 3, 13), (
            "A decision made 7 calendar days ago on a Friday must not be eligible — "
            "the trading-day cutoff has to be earlier than the calendar-day cutoff."
        )


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


class TestSpyBenchmarkCache:
    """T1.10: SPY price fetches must only cache successful results.
    Caching None poisons every later decision sharing that date.
    """

    @patch("v2.backfill.update_outcome")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    def test_transient_spy_failure_is_not_cached(
        self, mock_get_decisions, mock_client, mock_price, mock_update,
    ):
        """First SPY fetch returns None (transient failure); a second
        decision sharing the same date must trigger a re-fetch, not see
        the cached None.
        """
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2026, 3, 13), "ticker": "AAPL",
             "action": "buy", "price": Decimal("100")},
            {"id": 2, "date": date(2026, 3, 13), "ticker": "MSFT",
             "action": "buy", "price": Decimal("200")},
        ]
        mock_client.return_value = MagicMock()

        # SPY entry on 2026-03-13: first call None, second call $400.
        # SPY exit on 2026-03-24: always $420.
        spy_entry_calls = {"count": 0}

        def price(_client, ticker, dt):
            if ticker == "AAPL":
                return Decimal("110")
            if ticker == "MSFT":
                return Decimal("220")
            if ticker == "SPY":
                if dt == date(2026, 3, 13):
                    spy_entry_calls["count"] += 1
                    return None if spy_entry_calls["count"] == 1 else Decimal("400")
                return Decimal("420")
            return None

        mock_price.side_effect = price

        backfill_outcomes(days=7, dry_run=False)

        # The first decision had no benchmark; second must have one because
        # the cache was not poisoned with the failed first fetch.
        first_call = mock_update.call_args_list[0]
        second_call = mock_update.call_args_list[1]
        assert first_call.args[3] is None, "First decision must have no benchmark"
        assert second_call.args[3] is not None, (
            "Second decision must have a benchmark — cache was poisoned by the None fetch"
        )
        # SPY entry was called twice (refetch on cache miss).
        assert spy_entry_calls["count"] == 2

    @patch("v2.backfill.update_outcome")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    def test_successful_spy_fetch_is_cached(
        self, mock_get_decisions, mock_client, mock_price, mock_update,
    ):
        """Two decisions sharing entry+exit dates should each only trigger
        one fetch per (date) pair on a successful path.
        """
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2026, 3, 13), "ticker": "AAPL",
             "action": "buy", "price": Decimal("100")},
            {"id": 2, "date": date(2026, 3, 13), "ticker": "MSFT",
             "action": "buy", "price": Decimal("200")},
        ]
        mock_client.return_value = MagicMock()

        spy_entry_calls = {"count": 0}
        spy_exit_calls = {"count": 0}

        def price(_client, ticker, dt):
            if ticker == "AAPL":
                return Decimal("110")
            if ticker == "MSFT":
                return Decimal("220")
            if ticker == "SPY":
                if dt == date(2026, 3, 13):
                    spy_entry_calls["count"] += 1
                    return Decimal("400")
                spy_exit_calls["count"] += 1
                return Decimal("420")
            return None

        mock_price.side_effect = price

        backfill_outcomes(days=7, dry_run=False)

        assert spy_entry_calls["count"] == 1, "SPY entry should be cached after first hit"
        assert spy_exit_calls["count"] == 1, "SPY exit should be cached after first hit"


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
