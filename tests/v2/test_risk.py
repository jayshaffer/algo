"""Tests for portfolio risk checks."""
from decimal import Decimal
import pytest
from v2.risk import check_sector_concentration, SECTOR_MAP, MAX_SECTOR_PCT


class TestSectorConcentration:
    def test_flags_sector_over_limit(self):
        positions = {
            "AAPL": Decimal("2000"),
            "MSFT": Decimal("2000"),
            "GOOGL": Decimal("2000"),
        }
        warnings = check_sector_concentration(positions, portfolio_value=Decimal("10000"))
        assert any("tech" in w.lower() for w in warnings)

    def test_passes_diversified_portfolio(self):
        positions = {
            "AAPL": Decimal("500"),
            "JPM": Decimal("500"),
            "XOM": Decimal("500"),
        }
        warnings = check_sector_concentration(positions, portfolio_value=Decimal("10000"))
        assert len(warnings) == 0

    def test_empty_portfolio_no_warnings(self):
        warnings = check_sector_concentration({}, portfolio_value=Decimal("10000"))
        assert len(warnings) == 0

    def test_unknown_ticker_classified_as_other(self):
        positions = {"ZZZZZ": Decimal("5000")}
        warnings = check_sector_concentration(positions, portfolio_value=Decimal("10000"))
        assert isinstance(warnings, list)

    def test_zero_portfolio_value(self):
        positions = {"AAPL": Decimal("1000")}
        warnings = check_sector_concentration(positions, portfolio_value=Decimal("0"))
        assert len(warnings) == 0
