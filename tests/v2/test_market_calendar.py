"""Tests for the shared NYSE trading-day calendar."""

from datetime import date

from v2.market_calendar import NYSE_HOLIDAYS, is_trading_day


class TestIsTradingDay:
    def test_weekday_non_holiday_is_trading_day(self):
        # 2026-05-04 is a Monday, not a holiday
        assert is_trading_day(date(2026, 5, 4)) is True

    def test_saturday_is_not_trading_day(self):
        assert is_trading_day(date(2026, 5, 9)) is False

    def test_sunday_is_not_trading_day(self):
        assert is_trading_day(date(2026, 5, 10)) is False

    def test_known_holiday_is_not_trading_day(self):
        # 2026-07-03 is observed July 4th holiday (the actual 4th is Saturday)
        assert is_trading_day(date(2026, 7, 3)) is False

    def test_christmas_2026_is_not_trading_day(self):
        assert is_trading_day(date(2026, 12, 25)) is False


class TestNyseHolidaysCoverage:
    def test_includes_2026_holidays(self):
        # Spot-check key 2026 holidays
        assert date(2026, 1, 1) in NYSE_HOLIDAYS  # New Year's
        assert date(2026, 7, 3) in NYSE_HOLIDAYS  # July 4th observed
        assert date(2026, 12, 25) in NYSE_HOLIDAYS  # Christmas

    def test_includes_2027_holidays(self):
        assert date(2027, 1, 1) in NYSE_HOLIDAYS
