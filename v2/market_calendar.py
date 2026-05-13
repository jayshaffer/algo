"""NYSE trading-day calendar — shared weekend/holiday predicate.

Maintained manually because adding pandas_market_calendars is overkill
for ~10 dates a year. Sources: https://www.nyse.com/markets/hours-calendars
(verify yearly). Half-days (early closes) are NOT included — they still
produce a daily bar.
"""

from datetime import date

NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20),
    date(2025, 2, 17), date(2025, 4, 18), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
})


def is_trading_day(d: date) -> bool:
    """True if `d` is a weekday and not in the NYSE holiday list."""
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS
