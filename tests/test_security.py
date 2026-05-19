"""Tests for security hardening features."""

from decimal import Decimal

from tests.conftest import make_trading_decision
from trading.agent import MAX_POSITION_PCT, validate_decision


class TestPositionSizeCap:
    """Tests for MAX_POSITION_PCT enforcement in validate_decision."""

    def test_max_position_pct_is_ten_percent(self):
        assert Decimal("0.10") == MAX_POSITION_PCT

    def test_rejects_buy_exceeding_cap(self):
        """A buy > 10% of portfolio should be rejected."""
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=100)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
            portfolio_value=Decimal("100000"),
        )
        assert is_valid is False
        assert "Position size" in reason

    def test_allows_buy_within_cap(self):
        """A buy <= 10% of portfolio should pass."""
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=5)
        is_valid, _ = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
            portfolio_value=Decimal("100000"),
        )
        assert is_valid is True

    def test_allows_buy_at_exact_cap(self):
        """A buy at exactly 10% should pass."""
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
        is_valid, _ = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("100"),
            positions={},
            portfolio_value=Decimal("10000"),
        )
        assert is_valid is True

    def test_skips_cap_when_no_portfolio_value(self):
        """Without portfolio_value, the cap is not applied (backward compat)."""
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=100)
        is_valid, _ = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is True

    def test_sell_not_affected_by_cap(self):
        """Sell orders should not be affected by position size cap."""
        decision = make_trading_decision(action="sell", ticker="AAPL", quantity=100)
        is_valid, _ = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={"AAPL": Decimal("100")},
            portfolio_value=Decimal("10000"),
        )
        assert is_valid is True


class TestTradeSessionLimit:
    """Tests that trade limit is wired correctly in trader.py."""

    def test_trade_limit_exists_in_v1_trader(self):
        """Verify max_trades_per_session is set in trading/trader.py."""
        with open("trading/trader.py") as f:
            source = f.read()
        assert "max_trades_per_session" in source
        assert "trades_executed >= max_trades_per_session" in source

    def test_trade_limit_exists_in_v2_trader(self):
        """Verify max_trades_per_session is set in v2/trader.py."""
        with open("v2/trader.py") as f:
            source = f.read()
        assert "max_trades_per_session" in source
        assert "trades_executed >= max_trades_per_session" in source
