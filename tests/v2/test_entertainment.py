"""Tests for the entertainment tweet pipeline."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.entertainment import (
    gather_market_context,
    generate_entertainment_tweets,
    EntertainmentResult,
    run_entertainment_pipeline,
    ENTERTAINMENT_SYSTEM_PROMPT,
)


class TestGatherMarketContext:
    """Verify gather_market_context assembles news + market data."""

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_full_context(self, mock_news, mock_snapshot, mock_format):
        mock_news.return_value = [
            MagicMock(headline="NVDA surges on AI demand", symbols=["NVDA"]),
            MagicMock(headline="Fed holds rates steady", symbols=[]),
        ]
        mock_snapshot.return_value = MagicMock()
        mock_format.return_value = "Market Snapshot (2026-02-16 10:00):\n  SPY: +0.50%"
        context = gather_market_context()
        assert "NVDA surges on AI demand" in context
        assert "Fed holds rates steady" in context
        assert "SPY: +0.50%" in context

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_news_only(self, mock_news, mock_snapshot, mock_format):
        mock_news.return_value = [
            MagicMock(headline="Apple announces new product", symbols=["AAPL"]),
        ]
        mock_snapshot.side_effect = Exception("API down")
        context = gather_market_context()
        assert "Apple announces new product" in context
        assert "MARKET DATA" not in context

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_market_data_only(self, mock_news, mock_snapshot, mock_format):
        mock_news.side_effect = Exception("API down")
        mock_snapshot.return_value = MagicMock()
        mock_format.return_value = "Market Snapshot:\n  SPY: -1.20%"
        context = gather_market_context()
        assert "SPY: -1.20%" in context
        assert "NEWS HEADLINES" not in context

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_both_fail_returns_fallback(self, mock_news, mock_snapshot, mock_format):
        mock_news.side_effect = Exception("API down")
        mock_snapshot.side_effect = Exception("API down")
        context = gather_market_context()
        assert context == "No market data available."

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_custom_news_limit(self, mock_news, mock_snapshot, mock_format):
        mock_news.return_value = []
        mock_snapshot.side_effect = Exception("skip")
        gather_market_context(news_limit=10)
        mock_news.assert_called_once_with(hours=24, limit=10)


def _make_claude_response(json_data):
    """Helper: build a mock Claude API response."""
    import json as _json
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=_json.dumps(json_data))]
    return mock_resp


class TestGenerateEntertainmentTweets:
    """Verify entertainment tweet generation via Claude."""

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_generates_tweets(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({
            "tweets": [
                {"text": "Squidward says the market is overvalued. Squidward also eats his cereal dry.", "type": "entertainment"},
            ]
        })
        result = generate_entertainment_tweets("some market context")
        assert len(result) == 1
        assert result[0]["type"] == "entertainment"
        call_kwargs = mock_retry.call_args
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        assert "entertainment" in call_kwargs.kwargs.get("system", "").lower()

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_entertainment_tweets("context")
        assert result == []

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_api_exception(self, mock_get_client, mock_retry):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_entertainment_tweets("context")
        assert result == []

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_markdown_fenced_json(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        fenced = '```json\n{"tweets": [{"text": "Ahoy!", "type": "entertainment"}]}\n```'
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_retry.return_value = mock_resp
        result = generate_entertainment_tweets("context")
        assert len(result) == 1

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_default_type_is_entertainment(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"tweets": [{"text": "Just vibes"}]})
        result = generate_entertainment_tweets("context")
        assert result[0]["type"] == "entertainment"


class TestEntertainmentResult:
    """Verify dataclass defaults."""

    def test_defaults(self):
        r = EntertainmentResult()
        assert r.tweets_generated == 0
        assert r.tweets_posted == 0
        assert r.tweets_failed == 0
        assert r.skipped is False
        assert r.errors == []

    def test_mutable_default(self):
        r1 = EntertainmentResult()
        r2 = EntertainmentResult()
        r1.errors.append("test")
        assert r2.errors == []


class TestRunEntertainmentPipeline:
    """Verify end-to-end orchestration."""

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweets")
    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_happy_path(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "NEWS: NVDA up 5%"
        mock_generate.return_value = [
            {"text": "Arg! $NVDA making me money!", "type": "entertainment"},
        ]
        mock_post.return_value = [
            {"text": "Arg! $NVDA making me money!", "type": "entertainment", "posted": True, "tweet_id": "111", "error": None},
        ]
        result = run_entertainment_pipeline()
        assert result.tweets_generated == 1
        assert result.tweets_posted == 1
        assert result.tweets_failed == 0
        assert result.errors == []
        mock_insert.assert_called_once()

    @patch("v2.entertainment.get_twitter_client")
    def test_skips_without_credentials(self, mock_client):
        mock_client.return_value = None
        result = run_entertainment_pipeline()
        assert result.skipped is True
        assert result.tweets_generated == 0

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweets")
    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_post_failures_counted(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = [
            {"text": "Tweet 1", "type": "entertainment"},
            {"text": "Tweet 2", "type": "entertainment"},
        ]
        mock_post.return_value = [
            {"text": "Tweet 1", "type": "entertainment", "posted": True, "tweet_id": "111", "error": None},
            {"text": "Tweet 2", "type": "entertainment", "posted": False, "tweet_id": None, "error": "Rate limit"},
        ]
        result = run_entertainment_pipeline()
        assert result.tweets_posted == 1
        assert result.tweets_failed == 1

    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_no_tweets_generated(self, mock_client, mock_context, mock_generate):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "No market data available."
        mock_generate.return_value = []
        result = run_entertainment_pipeline()
        assert result.tweets_generated == 0
        assert result.tweets_posted == 0

    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_context_error_handled(self, mock_client, mock_context):
        mock_client.return_value = MagicMock()
        mock_context.side_effect = Exception("Total failure")
        result = run_entertainment_pipeline()
        assert len(result.errors) == 1
        assert "Context gathering failed" in result.errors[0]

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweets")
    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_db_log_error_does_not_crash(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = [{"text": "Tweet", "type": "entertainment"}]
        mock_post.return_value = [
            {"text": "Tweet", "type": "entertainment", "posted": True, "tweet_id": "111", "error": None},
        ]
        mock_insert.side_effect = Exception("DB write failed")
        result = run_entertainment_pipeline()
        assert result.tweets_posted == 1
        assert len(result.errors) == 1
        assert "Failed to log tweet" in result.errors[0]
