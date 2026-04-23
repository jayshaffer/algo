"""Tests for intent-based sizing resolver."""

from decimal import Decimal

import pytest

from v2.intents import (
    MAX_POSITION_PCT,
    BuyIntent,
    IntentError,
    SellIntent,
    resolve_buy_intent,
    resolve_sell_intent,
)


class TestResolveSellIntent:
    def test_exit_full_returns_held_shares(self):
        intent = SellIntent(type="exit_full", magnitude=None)
        shares = resolve_sell_intent(
            intent, held=Decimal("1.0"), price=Decimal("232.34"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("1.0")

    def test_exit_full_with_zero_holding_returns_zero(self):
        intent = SellIntent(type="exit_full", magnitude=None)
        shares = resolve_sell_intent(
            intent, held=Decimal("0"), price=Decimal("232.34"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("0")

    def test_exit_partial_pct_returns_fraction_of_held(self):
        intent = SellIntent(type="exit_partial_pct", magnitude=Decimal("50"))
        shares = resolve_sell_intent(
            intent, held=Decimal("2.0"), price=Decimal("100"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("1.0")

    def test_exit_partial_pct_100_equals_exit_full(self):
        intent = SellIntent(type="exit_partial_pct", magnitude=Decimal("100"))
        shares = resolve_sell_intent(
            intent, held=Decimal("2.0"), price=Decimal("100"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("2.0")

    def test_exit_partial_pct_invalid_magnitude_raises(self):
        intent = SellIntent(type="exit_partial_pct", magnitude=Decimal("150"))
        with pytest.raises(IntentError, match="between 0 and 100"):
            resolve_sell_intent(
                intent, held=Decimal("2.0"), price=Decimal("100"),
                portfolio_value=Decimal("10000"),
            )

    def test_exit_dollar_divides_by_price(self):
        intent = SellIntent(type="exit_dollar", magnitude=Decimal("500"))
        shares = resolve_sell_intent(
            intent, held=Decimal("10"), price=Decimal("100"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("5")

    def test_exit_dollar_clamped_to_held_shares(self):
        intent = SellIntent(type="exit_dollar", magnitude=Decimal("5000"))
        shares = resolve_sell_intent(
            intent, held=Decimal("10"), price=Decimal("100"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("10")

    def test_trim_to_portfolio_pct_computes_delta(self):
        # Held $3000 worth, want down to 10% of $10k = $1000 → sell $2000 = 20 shares
        intent = SellIntent(type="trim_to_portfolio_pct", magnitude=Decimal("10"))
        shares = resolve_sell_intent(
            intent, held=Decimal("30"), price=Decimal("100"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("20")

    def test_trim_to_portfolio_pct_already_below_target_returns_zero(self):
        intent = SellIntent(type="trim_to_portfolio_pct", magnitude=Decimal("20"))
        shares = resolve_sell_intent(
            intent, held=Decimal("5"), price=Decimal("100"),
            portfolio_value=Decimal("10000"),
        )
        assert shares == Decimal("0")

    def test_unknown_intent_raises(self):
        intent = SellIntent(type="exit_someday", magnitude=None)
        with pytest.raises(IntentError, match="unknown sell intent"):
            resolve_sell_intent(
                intent, held=Decimal("1"), price=Decimal("100"),
                portfolio_value=Decimal("10000"),
            )


class TestResolveBuyIntent:
    def test_invest_dollar_divides_by_price(self):
        intent = BuyIntent(type="invest_dollar", magnitude=Decimal("500"))
        shares = resolve_buy_intent(
            intent, held=Decimal("0"), price=Decimal("200"),
            portfolio_value=Decimal("10000"), buying_power=Decimal("5000"),
        )
        assert shares == Decimal("2.5")

    def test_invest_dollar_clamped_to_buying_power(self):
        intent = BuyIntent(type="invest_dollar", magnitude=Decimal("10000"))
        shares = resolve_buy_intent(
            intent, held=Decimal("0"), price=Decimal("100"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("5000"),
        )
        assert shares == Decimal("50")

    def test_invest_dollar_clamped_to_max_position_pct(self):
        # $100k portfolio, MAX_POSITION_PCT=10% → max $10k position
        intent = BuyIntent(type="invest_dollar", magnitude=Decimal("20000"))
        shares = resolve_buy_intent(
            intent, held=Decimal("0"), price=Decimal("100"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("50000"),
        )
        assert shares == Decimal("100")  # $10k / $100 = 100 shares

    def test_invest_dollar_cap_accounts_for_existing_holding(self):
        # Already hold $5k of the ticker; MAX_POSITION_PCT=10% of $100k=$10k.
        # Remaining headroom = $5k → 50 shares @ $100.
        intent = BuyIntent(type="invest_dollar", magnitude=Decimal("20000"))
        shares = resolve_buy_intent(
            intent, held=Decimal("50"), price=Decimal("100"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("50000"),
        )
        assert shares == Decimal("50")

    def test_invest_portfolio_pct(self):
        intent = BuyIntent(type="invest_portfolio_pct", magnitude=Decimal("2"))
        # $100k * 2% = $2000; @$200 = 10 shares
        shares = resolve_buy_intent(
            intent, held=Decimal("0"), price=Decimal("200"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("50000"),
        )
        assert shares == Decimal("10")

    def test_invest_buying_power_pct(self):
        intent = BuyIntent(type="invest_buying_power_pct", magnitude=Decimal("5"))
        # $50k * 5% = $2500; @$100 = 25 shares
        shares = resolve_buy_intent(
            intent, held=Decimal("0"), price=Decimal("100"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("50000"),
        )
        assert shares == Decimal("25")

    def test_add_to_target_pct_computes_delta(self):
        # Target 5% of $100k = $5000; already hold $2000 (20 @ $100) → add $3000 = 30 shares
        intent = BuyIntent(type="add_to_target_pct", magnitude=Decimal("5"))
        shares = resolve_buy_intent(
            intent, held=Decimal("20"), price=Decimal("100"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("50000"),
        )
        assert shares == Decimal("30")

    def test_add_to_target_pct_already_at_or_above_returns_zero(self):
        intent = BuyIntent(type="add_to_target_pct", magnitude=Decimal("5"))
        # Already hold $10k → above target → no buy
        shares = resolve_buy_intent(
            intent, held=Decimal("100"), price=Decimal("100"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("50000"),
        )
        assert shares == Decimal("0")

    def test_zero_buying_power_returns_zero(self):
        intent = BuyIntent(type="invest_dollar", magnitude=Decimal("500"))
        shares = resolve_buy_intent(
            intent, held=Decimal("0"), price=Decimal("100"),
            portfolio_value=Decimal("100000"), buying_power=Decimal("0"),
        )
        assert shares == Decimal("0")

    def test_zero_price_raises(self):
        intent = BuyIntent(type="invest_dollar", magnitude=Decimal("500"))
        with pytest.raises(IntentError, match="price must be positive"):
            resolve_buy_intent(
                intent, held=Decimal("0"), price=Decimal("0"),
                portfolio_value=Decimal("100000"), buying_power=Decimal("5000"),
            )

    def test_unknown_intent_raises(self):
        intent = BuyIntent(type="invest_someday", magnitude=Decimal("100"))
        with pytest.raises(IntentError, match="unknown buy intent"):
            resolve_buy_intent(
                intent, held=Decimal("0"), price=Decimal("100"),
                portfolio_value=Decimal("100000"), buying_power=Decimal("5000"),
            )
