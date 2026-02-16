"""Tests for the Twitter integration (Bikini Bottom Capital) on v2 pipeline."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from v2.database.trading_db import insert_tweet, get_tweets_for_date
from v2.twitter import gather_tweet_context


class TestInsertTweet:
    """Verify insert_tweet issues correct SQL."""

    def test_insert_tweet_basic(self, mock_db):
        mock_db.fetchone.return_value = {"id": 1}
        result = insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="recap",
            tweet_text="Ahoy! Great day for me treasure!",
        )
        assert result == 1
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO tweets" in sql
        assert "RETURNING id" in sql
        params = mock_db.execute.call_args[0][1]
        assert params == (date(2026, 2, 15), "recap", "Ahoy! Great day for me treasure!", None, False, None)

    def test_insert_tweet_with_all_fields(self, mock_db):
        mock_db.fetchone.return_value = {"id": 1}
        result = insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="trade",
            tweet_text="Bought more $AAPL!",
            tweet_id="123456789",
            posted=True,
            error=None,
        )
        assert result == 1
        params = mock_db.execute.call_args[0][1]
        assert params[3] == "123456789"
        assert params[4] is True

    def test_insert_tweet_with_error(self, mock_db):
        mock_db.fetchone.return_value = {"id": 1}
        insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="recap",
            tweet_text="Test tweet",
            error="Rate limit exceeded",
        )
        params = mock_db.execute.call_args[0][1]
        assert params[5] == "Rate limit exceeded"


class TestGetTweetsForDate:
    """Verify get_tweets_for_date issues correct SQL."""

    def test_returns_tweets(self, mock_db):
        mock_db.fetchall.return_value = [
            {"id": 1, "tweet_text": "Ahoy!", "tweet_type": "recap"},
            {"id": 2, "tweet_text": "Bought $AAPL!", "tweet_type": "trade"},
        ]
        result = get_tweets_for_date(date(2026, 2, 15))
        assert len(result) == 2
        assert result[0]["tweet_text"] == "Ahoy!"
        sql = mock_db.execute.call_args[0][0]
        assert "session_date = %s" in sql
        assert "ORDER BY created_at" in sql

    def test_returns_empty_list(self, mock_db):
        mock_db.fetchall.return_value = []
        result = get_tweets_for_date(date(2026, 2, 15))
        assert result == []


class TestGatherTweetContext:
    """Verify gather_tweet_context builds context string from DB data."""

    def test_full_context(self, mock_db):
        mock_db.fetchall.side_effect = [
            [{"ticker": "AAPL", "action": "buy", "quantity": Decimal("10"), "price": Decimal("185.50"), "reasoning": "Earnings beat"}],
            [{"ticker": "AAPL", "shares": Decimal("50"), "avg_cost": Decimal("150.00")}],
            [{"ticker": "NVDA", "direction": "long", "thesis": "AI demand", "confidence": "high"}],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("150000"), "cash": Decimal("100000"), "buying_power": Decimal("200000")},
            {"content": "Stay bullish on tech"},
        ]
        context = gather_tweet_context(date(2026, 2, 15))
        assert "BUY 10 AAPL @ $185.50" in context
        assert "Earnings beat" in context
        assert "AAPL: 50 shares @ $150.00" in context
        assert "NVDA (long, high): AI demand" in context
        assert "portfolio=$150000" in context
        assert "Stay bullish on tech" in context

    def test_decision_without_price(self, mock_db):
        mock_db.fetchall.side_effect = [
            [{"ticker": "AAPL", "action": "sell", "quantity": Decimal("5"), "price": None, "reasoning": "Taking profits"}],
            [], [],
        ]
        mock_db.fetchone.side_effect = [None, None]
        context = gather_tweet_context(date(2026, 2, 15))
        assert "SELL 5 AAPL: Taking profits" in context
        assert "@" not in context

    def test_empty_data(self, mock_db):
        mock_db.fetchall.side_effect = [[], [], []]
        mock_db.fetchone.side_effect = [None, None]
        context = gather_tweet_context(date(2026, 2, 15))
        assert context == "No trading activity today."

    def test_partial_data(self, mock_db):
        mock_db.fetchall.side_effect = [
            [],
            [{"ticker": "MSFT", "shares": Decimal("20"), "avg_cost": Decimal("400.00")}],
            [],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("100000")},
            None,
        ]
        context = gather_tweet_context(date(2026, 2, 15))
        assert "MSFT: 20 shares" in context
        assert "portfolio=$100000" in context
        assert "DECISIONS" not in context
        assert "THESES" not in context
        assert "STRATEGY MEMO" not in context

    def test_defaults_to_today(self, mock_db):
        mock_db.fetchall.side_effect = [[], [], []]
        mock_db.fetchone.side_effect = [None, None]
        gather_tweet_context()
        first_call_params = mock_db.execute.call_args_list[0][0][1]
        assert first_call_params == (date.today(),)
