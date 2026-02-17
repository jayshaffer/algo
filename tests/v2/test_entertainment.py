"""Tests for the entertainment tweet pipeline."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.entertainment import (
    gather_market_context,
    generate_entertainment_tweets,
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
