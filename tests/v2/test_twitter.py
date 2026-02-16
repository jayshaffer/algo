"""Tests for the Twitter integration (Bikini Bottom Capital) on v2 pipeline."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.database.trading_db import insert_tweet, get_tweets_for_date
from v2.twitter import (
    gather_tweet_context,
    generate_tweets,
    get_twitter_client,
    post_tweets,
    MR_KRABS_SYSTEM_PROMPT,
)


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


class TestGenerateTweets:
    """Verify generate_tweets calls Ollama and processes response."""

    @patch("v2.twitter.chat_json")
    def test_generates_tweets(self, mock_chat):
        mock_chat.return_value = {
            "tweets": [
                {"text": "Ahoy! Great day for me treasure! $AAPL up big!", "type": "recap"},
                {"text": "Bought more $NVDA. AI is the future, me boy!", "type": "trade"},
            ]
        }
        result = generate_tweets("test context")
        assert len(result) == 2
        assert result[0]["text"] == "Ahoy! Great day for me treasure! $AAPL up big!"
        assert result[0]["type"] == "recap"
        assert result[1]["type"] == "trade"
        call_kwargs = mock_chat.call_args
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", call_kwargs[1].get("system", ""))
        assert call_kwargs.kwargs.get("model") == "qwen2.5:14b"

    @patch("v2.twitter.chat_json")
    def test_custom_model(self, mock_chat):
        mock_chat.return_value = {"tweets": [{"text": "Test", "type": "recap"}]}
        generate_tweets("context", model="llama3:8b")
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs.get("model") == "llama3:8b"

    @patch("v2.twitter.chat_json")
    def test_truncates_long_tweets(self, mock_chat):
        long_text = "A" * 300
        mock_chat.return_value = {"tweets": [{"text": long_text, "type": "recap"}]}
        result = generate_tweets("context")
        assert len(result) == 1
        assert len(result[0]["text"]) == 280
        assert result[0]["text"].endswith("...")

    @patch("v2.twitter.chat_json")
    def test_handles_empty_response(self, mock_chat):
        mock_chat.return_value = {}
        result = generate_tweets("context")
        assert result == []

    @patch("v2.twitter.chat_json")
    def test_handles_malformed_tweets(self, mock_chat):
        mock_chat.return_value = {
            "tweets": [
                {"text": "Good tweet", "type": "recap"},
                {"no_text_key": "bad"},
                "not a dict",
            ]
        }
        result = generate_tweets("context")
        assert len(result) == 1
        assert result[0]["text"] == "Good tweet"

    @patch("v2.twitter.chat_json")
    def test_handles_chat_json_exception(self, mock_chat):
        mock_chat.side_effect = ValueError("Bad JSON")
        result = generate_tweets("context")
        assert result == []

    @patch("v2.twitter.chat_json")
    def test_default_type_is_commentary(self, mock_chat):
        mock_chat.return_value = {"tweets": [{"text": "Just a thought about the market"}]}
        result = generate_tweets("context")
        assert result[0]["type"] == "commentary"

    @patch("v2.twitter.chat_json")
    def test_tweets_not_a_list(self, mock_chat):
        mock_chat.return_value = {"tweets": "not a list"}
        result = generate_tweets("context")
        assert result == []


class TestGetTwitterClient:
    """Verify get_twitter_client credential handling."""

    def test_returns_client_with_creds(self, monkeypatch):
        monkeypatch.setenv("TWITTER_API_KEY", "key")
        monkeypatch.setenv("TWITTER_API_SECRET", "secret")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "token")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN_SECRET", "token_secret")
        mock_tweepy = MagicMock()
        mock_client = MagicMock()
        mock_tweepy.Client.return_value = mock_client
        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = get_twitter_client()
        assert client is mock_client
        mock_tweepy.Client.assert_called_once_with(
            consumer_key="key", consumer_secret="secret",
            access_token="token", access_token_secret="token_secret",
        )

    def test_returns_none_without_creds(self, monkeypatch):
        monkeypatch.delenv("TWITTER_API_KEY", raising=False)
        monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN_SECRET", raising=False)
        client = get_twitter_client()
        assert client is None

    def test_returns_none_with_partial_creds(self, monkeypatch):
        monkeypatch.setenv("TWITTER_API_KEY", "key")
        monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN_SECRET", raising=False)
        client = get_twitter_client()
        assert client is None


class TestPostTweets:
    """Verify post_tweets calls tweepy and handles errors."""

    @patch("v2.twitter.get_twitter_client")
    def test_posts_successfully(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = {"id": "12345"}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        tweets = [{"text": "Hello from Bikini Bottom!", "type": "recap"}]
        results = post_tweets(tweets)
        assert len(results) == 1
        assert results[0]["posted"] is True
        assert results[0]["tweet_id"] == "12345"
        assert results[0]["error"] is None
        assert results[0]["type"] == "recap"

    @patch("v2.twitter.get_twitter_client")
    def test_handles_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = Exception("Rate limit")
        mock_get_client.return_value = mock_client
        tweets = [{"text": "Tweet 1", "type": "recap"}]
        results = post_tweets(tweets)
        assert len(results) == 1
        assert results[0]["posted"] is False
        assert "Rate limit" in results[0]["error"]

    @patch("v2.twitter.get_twitter_client")
    def test_continues_after_failure(self, mock_get_client):
        mock_client = MagicMock()
        error_response = Exception("Rate limit")
        success_response = MagicMock()
        success_response.data = {"id": "99999"}
        mock_client.create_tweet.side_effect = [error_response, success_response]
        mock_get_client.return_value = mock_client
        tweets = [
            {"text": "Tweet 1", "type": "recap"},
            {"text": "Tweet 2", "type": "trade"},
        ]
        results = post_tweets(tweets)
        assert len(results) == 2
        assert results[0]["posted"] is False
        assert results[1]["posted"] is True

    @patch("v2.twitter.get_twitter_client")
    def test_no_credentials(self, mock_get_client):
        mock_get_client.return_value = None
        tweets = [{"text": "Tweet", "type": "recap"}]
        results = post_tweets(tweets)
        assert len(results) == 1
        assert results[0]["posted"] is False
        assert "No Twitter credentials" in results[0]["error"]
