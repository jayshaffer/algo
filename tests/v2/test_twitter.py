"""Tests for the Twitter integration (Bikini Bottom Capital) on v2 pipeline."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from v2.database.trading_db import insert_tweet, get_tweets_for_date


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
