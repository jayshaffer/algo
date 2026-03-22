"""Tests for v2 executor functions."""

from decimal import Decimal
from unittest.mock import patch, MagicMock

from v2.executor import OrderResult


class TestIsMarketOpen:
    @patch("v2.executor.get_trading_client")
    def test_returns_true_when_open(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_client.return_value.get_clock.return_value = mock_clock

        from v2.executor import is_market_open
        assert is_market_open() is True

    @patch("v2.executor.get_trading_client")
    def test_returns_false_when_closed(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_client.return_value.get_clock.return_value = mock_clock

        from v2.executor import is_market_open
        assert is_market_open() is False
