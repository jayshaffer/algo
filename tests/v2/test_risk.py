"""Tests for portfolio risk checks."""
from decimal import Decimal

from v2.risk import check_sector_cap_for_buy, check_sector_concentration


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


class TestCheckSectorCapForBuy:
    """P3.30: hard pre-submit gate. The advisory text in risk_notes is
    fed to the executor LLM, but nothing structurally prevents the LLM
    from ignoring it — the gate enforces the same threshold so a single
    LLM lapse can't run the book over the cap."""

    def test_blocks_buy_that_pushes_sector_over_cap(self):
        # Tech is already at 39% ($3,900 of $10,000); buying $200 more (2%)
        # would push to 41%, over the default 40% cap.
        position_values = {
            "AAPL": Decimal("1300"),
            "MSFT": Decimal("1300"),
            "GOOGL": Decimal("1300"),
        }
        breach = check_sector_cap_for_buy(
            ticker="NVDA",
            new_qty=Decimal("1"),
            price=Decimal("200"),
            position_values=position_values,
            portfolio_value=Decimal("10000"),
        )
        assert breach is not None
        assert "tech" in breach.lower()
        assert "41%" in breach or "exceed" in breach.lower()

    def test_allows_buy_under_cap(self):
        position_values = {"AAPL": Decimal("1000")}
        breach = check_sector_cap_for_buy(
            ticker="MSFT",
            new_qty=Decimal("1"),
            price=Decimal("100"),
            position_values=position_values,
            portfolio_value=Decimal("10000"),
        )
        assert breach is None

    def test_allows_buy_in_unrelated_sector_when_other_sector_is_concentrated(self):
        """A tech-heavy book should still allow buys in other sectors;
        the cap is per-sector, not portfolio-wide."""
        position_values = {
            "AAPL": Decimal("3500"),
            "MSFT": Decimal("500"),
        }
        breach = check_sector_cap_for_buy(
            ticker="JPM",  # finance
            new_qty=Decimal("1"),
            price=Decimal("100"),
            position_values=position_values,
            portfolio_value=Decimal("10000"),
        )
        assert breach is None

    def test_zero_portfolio_value_returns_none(self):
        breach = check_sector_cap_for_buy(
            ticker="AAPL",
            new_qty=Decimal("1"),
            price=Decimal("100"),
            position_values={},
            portfolio_value=Decimal("0"),
        )
        assert breach is None

    def test_unknown_ticker_uses_other_bucket(self):
        """A ticker not in SECTOR_MAP falls into the 'other' bucket;
        cap still applies but won't conflate with named sectors."""
        position_values = {"WEIRD": Decimal("3500")}
        breach = check_sector_cap_for_buy(
            ticker="UNCLASSIFIED",
            new_qty=Decimal("1"),
            price=Decimal("1000"),
            position_values=position_values,
            portfolio_value=Decimal("10000"),
        )
        assert breach is not None
        assert "other" in breach.lower()
