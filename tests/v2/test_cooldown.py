"""Tests for v2/cooldown.py — structural cooldown enforcement."""

from datetime import date, timedelta
from unittest.mock import patch

import pytest


def _make_decision(ticker, action, days_ago, playbook_action_id=1):
    """Helper to create a decision dict at a specific age."""
    return {
        "ticker": ticker,
        "action": action,
        "date": date.today() - timedelta(days=days_ago),
        "playbook_action_id": playbook_action_id,
    }


class TestAddBusinessDays:
    def test_weekday_to_weekday(self):
        from v2.cooldown import _add_business_days
        # Monday + 3 business days = Thursday
        monday = date(2026, 3, 30)  # Monday
        assert _add_business_days(monday, 3) == date(2026, 4, 2)  # Thursday

    def test_friday_spans_weekend(self):
        from v2.cooldown import _add_business_days
        # Friday + 1 business day = Monday
        friday = date(2026, 3, 27)  # Friday
        assert _add_business_days(friday, 1) == date(2026, 3, 30)  # Monday

    def test_friday_plus_3(self):
        from v2.cooldown import _add_business_days
        # Friday + 3 business days = Wednesday
        friday = date(2026, 3, 27)  # Friday
        assert _add_business_days(friday, 3) == date(2026, 4, 1)  # Wednesday

    def test_zero_days(self):
        from v2.cooldown import _add_business_days
        monday = date(2026, 3, 30)
        assert _add_business_days(monday, 0) == monday


class TestGetTickerCooldowns:
    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_hold_blocks_sell_for_3_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=1)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "hold", days_ago=1),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert cooldowns["AAPL"].proposed_action == "sell"
        assert "Rule #20" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_hold_sell_allowed_after_cooldown(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=5)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "hold", days_ago=5),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" not in cooldowns

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_buy_blocks_sell_for_3_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=1)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "buy", days_ago=1),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert cooldowns["AAPL"].proposed_action == "sell"
        assert "Rule #3" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_sell_blocks_buy_for_5_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=2)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "sell", days_ago=2),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert cooldowns["AAPL"].proposed_action == "buy"
        assert "Rule #6" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_post_accumulation_sell_blocks_buy_for_10_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        all_dates = {date.today() - timedelta(days=i) for i in range(10)}
        mock_dates.return_value = all_dates
        mock_decisions.return_value = [
            _make_decision("AAPL", "sell", days_ago=2),
            _make_decision("AAPL", "buy", days_ago=3),
            _make_decision("AAPL", "buy", days_ago=4),
            _make_decision("AAPL", "buy", days_ago=5),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert "Rule #10" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_fallback_hold_does_not_create_cooldown(self, mock_dates, mock_decisions):
        """HOLDs from sessions without a playbook should not create cooldowns."""
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = set()  # No playbooks
        mock_decisions.return_value = [
            _make_decision("AAPL", "hold", days_ago=1, playbook_action_id=None),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" not in cooldowns


class TestCheckCooldown:
    def test_blocked_action(self):
        from v2.cooldown import check_cooldown, CooldownViolation
        cooldowns = {
            "AAPL": CooldownViolation(
                ticker="AAPL", proposed_action="sell", blocking_action="hold",
                blocking_date=date.today(), cooldown_expires=date.today() + timedelta(days=3),
                rule="test rule",
            )
        }
        blocked, reason = check_cooldown("AAPL", "sell", cooldowns)
        assert blocked is True
        assert "test rule" in reason

    def test_different_action_not_blocked(self):
        from v2.cooldown import check_cooldown, CooldownViolation
        cooldowns = {
            "AAPL": CooldownViolation(
                ticker="AAPL", proposed_action="sell", blocking_action="hold",
                blocking_date=date.today(), cooldown_expires=date.today() + timedelta(days=3),
                rule="test rule",
            )
        }
        blocked, reason = check_cooldown("AAPL", "buy", cooldowns)
        assert blocked is False

    def test_unknown_ticker_not_blocked(self):
        from v2.cooldown import check_cooldown
        blocked, reason = check_cooldown("MSFT", "sell", {})
        assert blocked is False


class TestFormatCooldownMap:
    def test_formats_for_executor(self):
        from v2.cooldown import format_cooldown_map, CooldownViolation
        cooldowns = {
            "AAPL": CooldownViolation(
                ticker="AAPL", proposed_action="sell", blocking_action="hold",
                blocking_date=date(2026, 4, 1), cooldown_expires=date(2026, 4, 4),
                rule="HOLD→SELL 3-day lockout",
            )
        }
        result = format_cooldown_map(cooldowns)
        assert result == {"AAPL": "HOLD→SELL 3-day lockout (expires 2026-04-04)"}
