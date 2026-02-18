"""Tests for the Twitter integration (Bikini Bottom Capital) on v2 pipeline."""

import os
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.database.trading_db import insert_tweet, get_tweets_for_date
from v2.twitter import (
    gather_tweet_context,
    generate_tweet,
    get_twitter_client,
    post_tweet,
    MR_KRABS_SYSTEM_PROMPT,
    run_twitter_stage,
    TwitterStageResult,
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
        assert params == (date(2026, 2, 15), "recap", "Ahoy! Great day for me treasure!", None, False, None, "twitter")

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

    def test_insert_tweet_with_platform(self, mock_db):
        mock_db.fetchone.return_value = {"id": 1}
        result = insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="recap",
            tweet_text="Ahoy!",
            platform="bluesky",
        )
        assert result == 1
        params = mock_db.execute.call_args[0][1]
        assert "bluesky" in params


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
            [
                {"date": date(2026, 2, 15), "portfolio_value": Decimal("150000"), "cash": Decimal("100000"), "buying_power": Decimal("200000"), "long_market_value": Decimal("50000")},
                {"date": date(2026, 2, 14), "portfolio_value": Decimal("148000"), "cash": Decimal("100000"), "buying_power": Decimal("200000"), "long_market_value": Decimal("48000")},
            ],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "date": date(2026, 1, 1)},
            {"content": "Stay bullish on tech"},
        ]
        context = gather_tweet_context(date(2026, 2, 15))
        assert "BUY 10 AAPL @ $185.50" in context
        assert "Earnings beat" in context
        assert "AAPL: 50 shares @ $150.00" in context
        assert "NVDA (long, high): AI demand" in context
        assert "Portfolio value: $150,000.00" in context
        assert "Today's P&L: +$2,000.00 (+1.35%)" in context
        assert "Total return: +$50,000.00 (+50.00%) since 2026-01-01" in context
        assert "Stay bullish on tech" in context

    def test_decision_without_price(self, mock_db):
        mock_db.fetchall.side_effect = [
            [{"ticker": "AAPL", "action": "sell", "quantity": Decimal("5"), "price": None, "reasoning": "Taking profits"}],
            [], [], [],
        ]
        mock_db.fetchone.side_effect = [None]
        context = gather_tweet_context(date(2026, 2, 15))
        assert "SELL 5 AAPL: Taking profits" in context
        assert "@" not in context

    def test_empty_data(self, mock_db):
        mock_db.fetchall.side_effect = [[], [], [], []]
        mock_db.fetchone.side_effect = [None]
        context = gather_tweet_context(date(2026, 2, 15))
        assert context == "No trading activity today."

    def test_partial_data(self, mock_db):
        mock_db.fetchall.side_effect = [
            [],
            [{"ticker": "MSFT", "shares": Decimal("20"), "avg_cost": Decimal("400.00")}],
            [],
            [{"date": date(2026, 2, 15), "portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("100000"), "long_market_value": Decimal("50000")}],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "date": date(2026, 2, 15)},
            None,
        ]
        context = gather_tweet_context(date(2026, 2, 15))
        assert "MSFT: 20 shares" in context
        assert "Portfolio value: $100,000.00" in context
        assert "DECISIONS" not in context
        assert "THESES" not in context
        assert "STRATEGY MEMO" not in context

    def test_defaults_to_today(self, mock_db):
        mock_db.fetchall.side_effect = [[], [], [], []]
        mock_db.fetchone.side_effect = [None]
        gather_tweet_context()
        first_call_params = mock_db.execute.call_args_list[0][0][1]
        assert first_call_params == (date.today(),)


def _make_claude_response(json_data):
    """Helper: build a mock Claude API response containing JSON text."""
    import json as _json
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=_json.dumps(json_data))]
    return mock_resp


class TestGenerateTweet:
    """Verify generate_tweet calls Claude and processes response."""

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_generates_tweet(self, mock_get_client, mock_retry):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_retry.return_value = _make_claude_response({
            "text": "Ahoy! Great day for me treasure! $AAPL up big!",
        })
        result = generate_tweet("test context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day for me treasure! $AAPL up big!"
        assert result["type"] == "recap"
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_custom_model(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Test"})
        generate_tweet("context", model="claude-sonnet-4-5-20250929")
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs.get("model") == "claude-sonnet-4-5-20250929"

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_preserves_long_tweets(self, mock_get_client, mock_retry):
        long_text = "A" * 300
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": long_text})
        result = generate_tweet("context")
        assert result is not None
        assert result["text"] == long_text

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_tweet("context")
        assert result is None

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_handles_non_string_text(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": 123})
        result = generate_tweet("context")
        assert result is None

    @patch("v2.twitter.get_claude_client")
    def test_handles_api_exception(self, mock_get_client):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_tweet("context")
        assert result is None

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_handles_markdown_fenced_json(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        fenced = '```json\n{"text": "Ahoy!"}\n```'
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_retry.return_value = mock_resp
        result = generate_tweet("context")
        assert result is not None
        assert result["text"] == "Ahoy!"

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_generate_tweet_appends_dashboard_url(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Ahoy! Great day!"})
        with patch.dict(os.environ, {"DASHBOARD_URL": "https://example.github.io"}):
            result = generate_tweet("context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day!\nhttps://example.github.io"

    @patch("v2.twitter._call_with_retry")
    @patch("v2.twitter.get_claude_client")
    def test_generate_tweet_no_url_when_not_set(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Ahoy! Great day!"})
        os.environ.pop("DASHBOARD_URL", None)
        result = generate_tweet("context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day!"


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


class TestPostTweet:
    """Verify post_tweet calls tweepy and handles errors."""

    @patch("v2.twitter.get_twitter_client")
    def test_posts_successfully(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = {"id": "12345"}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client
        tweet = {"text": "Hello from Bikini Bottom!", "type": "recap"}
        result = post_tweet(tweet)
        assert result["posted"] is True
        assert result["tweet_id"] == "12345"
        assert result["error"] is None
        assert result["type"] == "recap"

    @patch("v2.twitter.get_twitter_client")
    def test_handles_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = Exception("Rate limit")
        mock_get_client.return_value = mock_client
        tweet = {"text": "Tweet 1", "type": "recap"}
        result = post_tweet(tweet)
        assert result["posted"] is False
        assert "Rate limit" in result["error"]

    @patch("v2.twitter.get_twitter_client")
    def test_no_credentials(self, mock_get_client):
        mock_get_client.return_value = None
        tweet = {"text": "Tweet", "type": "recap"}
        result = post_tweet(tweet)
        assert result["posted"] is False
        assert "No Twitter credentials" in result["error"]


class TestRunTwitterStage:
    """Verify run_twitter_stage orchestration."""

    @patch("v2.twitter.insert_tweet")
    @patch("v2.twitter.post_tweet")
    @patch("v2.twitter.generate_tweet")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_happy_path(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "Today we bought AAPL"
        mock_generate.return_value = {"text": "Ahoy! Bought $AAPL!", "type": "recap"}
        mock_post.return_value = {
            "text": "Ahoy! Bought $AAPL!", "type": "recap", "posted": True, "tweet_id": "111", "error": None,
        }
        result = run_twitter_stage(date(2026, 2, 15))
        assert result.tweet_posted is True
        assert result.skipped is False
        assert result.errors == []
        mock_insert.assert_called_once()

    @patch("v2.twitter.get_twitter_client")
    def test_skips_without_credentials(self, mock_client):
        mock_client.return_value = None
        result = run_twitter_stage(date(2026, 2, 15))
        assert result.skipped is True
        assert result.tweet_posted is False

    @patch("v2.twitter.insert_tweet")
    @patch("v2.twitter.post_tweet")
    @patch("v2.twitter.generate_tweet")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_post_failure(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Tweet 1", "type": "recap"}
        mock_post.return_value = {
            "text": "Tweet 1", "type": "recap", "posted": False, "tweet_id": None, "error": "Rate limit",
        }
        result = run_twitter_stage(date(2026, 2, 15))
        assert result.tweet_posted is False

    @patch("v2.twitter.generate_tweet")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_no_tweet_generated(self, mock_client, mock_context, mock_generate):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "No trading activity today."
        mock_generate.return_value = None
        result = run_twitter_stage(date(2026, 2, 15))
        assert result.tweet_posted is False

    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_context_error_handled(self, mock_client, mock_context):
        mock_client.return_value = MagicMock()
        mock_context.side_effect = Exception("DB connection failed")
        result = run_twitter_stage(date(2026, 2, 15))
        assert len(result.errors) == 1
        assert "Context gathering failed" in result.errors[0]

    @patch("v2.twitter.insert_tweet")
    @patch("v2.twitter.post_tweet")
    @patch("v2.twitter.generate_tweet")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_db_log_error_does_not_crash(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Tweet", "type": "recap"}
        mock_post.return_value = {
            "text": "Tweet", "type": "recap", "posted": True, "tweet_id": "111", "error": None,
        }
        mock_insert.side_effect = Exception("DB write failed")
        result = run_twitter_stage(date(2026, 2, 15))
        assert result.tweet_posted is True
        assert len(result.errors) == 1
        assert "Failed to log tweet" in result.errors[0]


class TestTwitterStageResult:
    """Verify dataclass defaults."""

    def test_defaults(self):
        r = TwitterStageResult()
        assert r.tweet_posted is False
        assert r.skipped is False
        assert r.errors == []

    def test_mutable_default(self):
        r1 = TwitterStageResult()
        r2 = TwitterStageResult()
        r1.errors.append("test")
        assert r2.errors == []
