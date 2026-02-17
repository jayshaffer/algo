"""Tests for the entertainment tweet pipeline."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.entertainment import (
    gather_market_context,
    generate_entertainment_tweet,
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


class TestGenerateEntertainmentTweet:
    """Verify entertainment tweet generation via Claude."""

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_generates_tweet(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({
            "text": "Squidward says the market is overvalued. Squidward also eats his cereal dry.",
        })
        result = generate_entertainment_tweet("some market context")
        assert result is not None
        assert result["type"] == "entertainment"
        assert "Squidward" in result["text"]
        call_kwargs = mock_retry.call_args
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_entertainment_tweet("context")
        assert result is None

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_api_exception(self, mock_get_client, mock_retry):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_entertainment_tweet("context")
        assert result is None

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_markdown_fenced_json(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        fenced = '```json\n{"text": "Ahoy!"}\n```'
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_retry.return_value = mock_resp
        result = generate_entertainment_tweet("context")
        assert result is not None
        assert result["text"] == "Ahoy!"

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_type_is_always_entertainment(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Just vibes"})
        result = generate_entertainment_tweet("context")
        assert result["type"] == "entertainment"


class TestEntertainmentResult:
    """Verify dataclass defaults."""

    def test_defaults(self):
        r = EntertainmentResult()
        assert r.posted is False
        assert r.skipped is False
        assert r.tweet_id is None
        assert r.error is None
        assert r.bluesky_posted is False
        assert r.bluesky_post_id is None
        assert r.bluesky_error is None


class TestRunEntertainmentPipeline:
    """Verify end-to-end orchestration."""

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_happy_path(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "NEWS: NVDA up 5%"
        mock_generate.return_value = {"text": "Arg! $NVDA making me money!", "type": "entertainment"}
        mock_post.return_value = {
            "text": "Arg! $NVDA making me money!", "type": "entertainment",
            "posted": True, "tweet_id": "111", "error": None,
        }
        result = run_entertainment_pipeline()
        assert result.posted is True
        assert result.tweet_id == "111"
        assert result.error is None
        mock_insert.assert_called_once()

    @patch("v2.entertainment.get_twitter_client")
    def test_skips_without_credentials(self, mock_client):
        mock_client.return_value = None
        result = run_entertainment_pipeline()
        assert result.skipped is True
        assert result.posted is False

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_post_failure(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Tweet", "type": "entertainment"}
        mock_post.return_value = {
            "text": "Tweet", "type": "entertainment",
            "posted": False, "tweet_id": None, "error": "Rate limit",
        }
        result = run_entertainment_pipeline()
        assert result.posted is False
        assert result.error == "Rate limit"

    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_no_tweet_generated(self, mock_client, mock_context, mock_generate):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "No market data available."
        mock_generate.return_value = None
        result = run_entertainment_pipeline()
        assert result.posted is False
        assert result.error is None

    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_context_error_handled(self, mock_client, mock_context):
        mock_client.return_value = MagicMock()
        mock_context.side_effect = Exception("Total failure")
        result = run_entertainment_pipeline()
        assert "Context gathering failed" in result.error

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_db_log_error_does_not_crash(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Tweet", "type": "entertainment"}
        mock_post.return_value = {
            "text": "Tweet", "type": "entertainment",
            "posted": True, "tweet_id": "111", "error": None,
        }
        mock_insert.side_effect = Exception("DB write failed")
        result = run_entertainment_pipeline()
        assert result.posted is True


class TestEntertainmentBluesky:
    """Verify Bluesky integration in entertainment pipeline."""

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_to_bluesky")
    @patch("v2.entertainment.generate_bluesky_entertainment_post")
    @patch("v2.entertainment.get_bluesky_client")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_posts_to_both_platforms(self, mock_tw_client, mock_context, mock_tw_gen,
                                     mock_tw_post, mock_bs_client, mock_bs_gen,
                                     mock_bs_post, mock_insert):
        mock_tw_client.return_value = MagicMock()
        mock_bs_client.return_value = MagicMock()
        mock_context.return_value = "NEWS: NVDA up 5%"
        mock_tw_gen.return_value = {"text": "Twitter tweet!", "type": "entertainment"}
        mock_tw_post.return_value = {
            "text": "Twitter tweet!", "type": "entertainment",
            "posted": True, "tweet_id": "tw-111", "error": None,
        }
        mock_bs_gen.return_value = {"text": "Bluesky post!", "type": "entertainment"}
        mock_bs_post.return_value = {
            "text": "Bluesky post!", "type": "entertainment",
            "posted": True, "post_id": "at://abc/123", "error": None,
        }
        result = run_entertainment_pipeline()
        assert result.posted is True
        assert result.bluesky_posted is True
        assert result.bluesky_post_id == "at://abc/123"
        assert mock_insert.call_count == 2  # one for Twitter, one for Bluesky

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.get_bluesky_client")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_bluesky_skipped_without_credentials(self, mock_tw_client, mock_context,
                                                   mock_tw_gen, mock_tw_post,
                                                   mock_bs_client, mock_insert):
        mock_tw_client.return_value = MagicMock()
        mock_bs_client.return_value = None
        mock_context.return_value = "NEWS"
        mock_tw_gen.return_value = {"text": "Tweet", "type": "entertainment"}
        mock_tw_post.return_value = {
            "text": "Tweet", "type": "entertainment",
            "posted": True, "tweet_id": "111", "error": None,
        }
        result = run_entertainment_pipeline()
        assert result.posted is True
        assert result.bluesky_posted is False

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_to_bluesky")
    @patch("v2.entertainment.generate_bluesky_entertainment_post")
    @patch("v2.entertainment.get_bluesky_client")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_bluesky_failure_does_not_block_twitter(self, mock_tw_client, mock_context,
                                                      mock_tw_gen, mock_tw_post,
                                                      mock_bs_client, mock_bs_gen,
                                                      mock_bs_post, mock_insert):
        mock_tw_client.return_value = MagicMock()
        mock_bs_client.return_value = MagicMock()
        mock_context.return_value = "NEWS"
        mock_tw_gen.return_value = {"text": "Tweet", "type": "entertainment"}
        mock_tw_post.return_value = {
            "text": "Tweet", "type": "entertainment",
            "posted": True, "tweet_id": "111", "error": None,
        }
        mock_bs_gen.side_effect = Exception("Claude down")
        result = run_entertainment_pipeline()
        assert result.posted is True
        assert result.bluesky_posted is False
        assert "Bluesky generation failed" in result.bluesky_error
