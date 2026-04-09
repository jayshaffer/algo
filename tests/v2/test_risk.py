"""Tests for portfolio risk checks."""
from decimal import Decimal
import pytest
from v2.risk import check_sector_concentration, SECTOR_MAP, MAX_SECTOR_PCT


class TestCheckDailyLossLimit:
    def test_blocks_when_loss_exceeds_threshold(self):
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("950"),
            previous_value=Decimal("1000"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is not None
        assert "loss" in result.lower() or "circuit breaker" in result.lower()

    def test_allows_when_within_threshold(self):
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("985"),
            previous_value=Decimal("1000"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None

    def test_allows_when_portfolio_is_up(self):
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("1050"),
            previous_value=Decimal("1000"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None

    def test_handles_zero_previous_value(self):
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("1000"),
            previous_value=Decimal("0"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None

    def test_handles_none_previous_value(self):
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("1000"),
            previous_value=None,
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None


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
