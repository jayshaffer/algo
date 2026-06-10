"""Tests for portfolio risk checks."""
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

from v2.risk import (
    check_churn_gate,
    check_daily_loss_limit,
    check_sector_cap_for_buy,
    check_sector_concentration,
    count_round_trip_pairs,
)


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


class TestDailyLossLimit:
    """Kill switch: halt trading when equity has dropped more than the
    configured percentage versus the previous close (Alpaca last_equity)."""

    def test_within_limit_returns_none(self):
        # -1% vs a 3% default limit
        assert check_daily_loss_limit(Decimal("99000"), Decimal("100000")) is None

    def test_breach_returns_message(self):
        # -4% vs a 3% default limit
        msg = check_daily_loss_limit(Decimal("96000"), Decimal("100000"))
        assert msg is not None
        assert "daily loss" in msg.lower()

    def test_breach_exactly_at_limit(self):
        msg = check_daily_loss_limit(Decimal("97000"), Decimal("100000"))
        assert msg is not None

    def test_gain_returns_none(self):
        assert check_daily_loss_limit(Decimal("105000"), Decimal("100000")) is None

    def test_missing_inputs_skip_check(self):
        assert check_daily_loss_limit(None, Decimal("100000")) is None
        assert check_daily_loss_limit(Decimal("100000"), None) is None
        assert check_daily_loss_limit(Decimal("100000"), Decimal("0")) is None

    def test_custom_limit_override(self):
        # -2% breaches a 1% limit even though it passes the 3% default
        msg = check_daily_loss_limit(
            Decimal("98000"), Decimal("100000"), limit_pct=Decimal("1"),
        )
        assert msg is not None

    def test_nonpositive_limit_disables_check(self):
        assert check_daily_loss_limit(
            Decimal("50000"), Decimal("100000"), limit_pct=Decimal("0"),
        ) is None


def _patch_decision_rows(rows):
    """Patch the connection-level get_cursor used by count_round_trip_pairs."""
    cursor = MagicMock()
    cursor.fetchall.return_value = rows

    @contextmanager
    def _get_cursor():
        yield cursor

    return patch("v2.database.connection.get_cursor", _get_cursor)


def _row(row_id, day, action):
    return {"id": row_id, "date": date(2026, 6, day), "action": action}


class TestCountRoundTripPairs:
    """Rule 43 greedy pair matching: opposing buy/sell decisions within
    CHURN_PAIR_WINDOW_DAYS of each other count as one round-trip pair."""

    def test_buy_then_sell_within_window_is_one_pair(self):
        rows = [_row(1, 1, "buy"), _row(2, 3, "sell")]
        with _patch_decision_rows(rows):
            assert count_round_trip_pairs("AAPL", reference_date=date(2026, 6, 15)) == 1

    def test_opposing_actions_outside_pair_window_do_not_pair(self):
        rows = [_row(1, 1, "buy"), _row(2, 10, "sell")]  # 9 days apart > 7
        with _patch_decision_rows(rows):
            assert count_round_trip_pairs("AAPL", reference_date=date(2026, 6, 15)) == 0

    def test_same_action_decisions_do_not_pair(self):
        rows = [_row(1, 1, "buy"), _row(2, 2, "buy"), _row(3, 3, "buy")]
        with _patch_decision_rows(rows):
            assert count_round_trip_pairs("AAPL", reference_date=date(2026, 6, 15)) == 0

    def test_greedy_match_consumes_each_decision_once(self):
        # buy(d1) pairs with sell(d3); the second buy(d2) is left unconsumed.
        rows = [_row(1, 1, "buy"), _row(2, 2, "buy"), _row(3, 3, "sell")]
        with _patch_decision_rows(rows):
            assert count_round_trip_pairs("AAPL", reference_date=date(2026, 6, 15)) == 1

    def test_sell_first_pairs_with_later_buy(self):
        rows = [_row(1, 1, "sell"), _row(2, 4, "buy")]
        with _patch_decision_rows(rows):
            assert count_round_trip_pairs("AAPL", reference_date=date(2026, 6, 15)) == 1

    def test_alternating_decisions_count_multiple_pairs(self):
        rows = []
        for i in range(6):
            rows.append(_row(2 * i + 1, 2 * i + 1, "buy"))
            rows.append(_row(2 * i + 2, 2 * i + 2, "sell"))
        with _patch_decision_rows(rows):
            assert count_round_trip_pairs("AAPL", reference_date=date(2026, 6, 20)) == 6

    def test_no_decisions_returns_zero(self):
        with _patch_decision_rows([]):
            assert count_round_trip_pairs("AAPL", reference_date=date(2026, 6, 15)) == 0


class TestCheckChurnGate:
    def test_blocks_at_threshold(self):
        rows = []
        for i in range(6):
            rows.append(_row(2 * i + 1, 2 * i + 1, "buy"))
            rows.append(_row(2 * i + 2, 2 * i + 2, "sell"))
        with _patch_decision_rows(rows):
            msg = check_churn_gate("AAPL", reference_date=date(2026, 6, 20))
        assert msg is not None
        assert "Rule 43" in msg
        assert "AAPL" in msg

    def test_passes_below_threshold(self):
        rows = []
        for i in range(5):
            rows.append(_row(2 * i + 1, 2 * i + 1, "buy"))
            rows.append(_row(2 * i + 2, 2 * i + 2, "sell"))
        with _patch_decision_rows(rows):
            assert check_churn_gate("AAPL", reference_date=date(2026, 6, 20)) is None
