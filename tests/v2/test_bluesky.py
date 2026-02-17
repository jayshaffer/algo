"""Tests for the Bluesky integration (Bikini Bottom Capital) on v2 pipeline."""

from datetime import date
from unittest.mock import MagicMock, patch, call

import pytest

from v2.bluesky import (
    get_bluesky_client, post_to_bluesky, generate_bluesky_post,
    BLUESKY_SYSTEM_PROMPT, run_bluesky_stage, BlueskyStageResult,
    generate_bluesky_entertainment_post, BLUESKY_ENTERTAINMENT_SYSTEM_PROMPT,
)


class TestGetBlueskyClient:
    """Verify get_bluesky_client credential handling."""

    def test_returns_client_with_creds(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "test.bsky.social")
        monkeypatch.setenv("BLUESKY_APP_PASSWORD", "app-password-123")
        mock_atproto = MagicMock()
        mock_client = MagicMock()
        mock_atproto.Client.return_value = mock_client
        with patch.dict("sys.modules", {"atproto": mock_atproto}):
            client = get_bluesky_client()
        assert client is mock_client
        mock_client.login.assert_called_once_with("test.bsky.social", "app-password-123")

    def test_returns_none_without_creds(self, monkeypatch):
        monkeypatch.delenv("BLUESKY_HANDLE", raising=False)
        monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
        client = get_bluesky_client()
        assert client is None

    def test_returns_none_with_partial_creds(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "test.bsky.social")
        monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
        client = get_bluesky_client()
        assert client is None

    def test_returns_none_on_login_failure(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "test.bsky.social")
        monkeypatch.setenv("BLUESKY_APP_PASSWORD", "bad-password")
        mock_atproto = MagicMock()
        mock_client = MagicMock()
        mock_client.login.side_effect = Exception("Invalid credentials")
        mock_atproto.Client.return_value = mock_client
        with patch.dict("sys.modules", {"atproto": mock_atproto}):
            client = get_bluesky_client()
        assert client is None


class TestPostToBluesky:
    """Verify post_to_bluesky calls atproto and handles errors."""

    @patch("v2.bluesky.get_bluesky_client")
    def test_posts_successfully(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.uri = "at://did:plc:abc/app.bsky.feed.post/123"
        mock_client.send_post.return_value = mock_response
        mock_get_client.return_value = mock_client
        post = {"text": "Hello from Bikini Bottom!", "type": "recap"}
        result = post_to_bluesky(post)
        assert result["posted"] is True
        assert result["post_id"] == "at://did:plc:abc/app.bsky.feed.post/123"
        assert result["error"] is None
        assert result["type"] == "recap"

    @patch("v2.bluesky.get_bluesky_client")
    def test_handles_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.send_post.side_effect = Exception("Rate limit")
        mock_get_client.return_value = mock_client
        post = {"text": "Post text", "type": "recap"}
        result = post_to_bluesky(post)
        assert result["posted"] is False
        assert "Rate limit" in result["error"]

    @patch("v2.bluesky.get_bluesky_client")
    def test_no_credentials(self, mock_get_client):
        mock_get_client.return_value = None
        post = {"text": "Post text", "type": "recap"}
        result = post_to_bluesky(post)
        assert result["posted"] is False
        assert "No Bluesky credentials" in result["error"]


def _make_claude_response(json_data):
    """Helper: build a mock Claude API response containing JSON text."""
    import json as _json
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=_json.dumps(json_data))]
    return mock_resp


class TestGenerateBlueskyPost:
    """Verify generate_bluesky_post calls Claude and processes response."""

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_generates_post(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({
            "text": "Ahoy! Great day for me treasure! $AAPL up big!",
        })
        result = generate_bluesky_post("test context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day for me treasure! $AAPL up big!"
        assert result["type"] == "recap"
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        assert "300" in call_kwargs.kwargs.get("system", "")

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_custom_model(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Test"})
        generate_bluesky_post("context", model="claude-sonnet-4-5-20250929")
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs.get("model") == "claude-sonnet-4-5-20250929"

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_bluesky_post("context")
        assert result is None

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_non_string_text(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": 123})
        result = generate_bluesky_post("context")
        assert result is None

    @patch("v2.bluesky.get_claude_client")
    def test_handles_api_exception(self, mock_get_client):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_bluesky_post("context")
        assert result is None

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_markdown_fenced_json(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        fenced = '```json\n{"text": "Ahoy!"}\n```'
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_retry.return_value = mock_resp
        result = generate_bluesky_post("context")
        assert result is not None
        assert result["text"] == "Ahoy!"


class TestBlueskyStageResult:
    """Verify dataclass defaults."""

    def test_defaults(self):
        r = BlueskyStageResult()
        assert r.post_posted is False
        assert r.skipped is False
        assert r.errors == []

    def test_mutable_default(self):
        r1 = BlueskyStageResult()
        r2 = BlueskyStageResult()
        r1.errors.append("test")
        assert r2.errors == []


class TestRunBlueskyStage:
    """Verify run_bluesky_stage orchestration."""

    @patch("v2.bluesky.insert_tweet")
    @patch("v2.bluesky.post_to_bluesky")
    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_happy_path(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "Today we bought AAPL"
        mock_generate.return_value = {"text": "Ahoy! Bought $AAPL!", "type": "recap"}
        mock_post.return_value = {
            "text": "Ahoy! Bought $AAPL!", "type": "recap", "posted": True,
            "post_id": "at://did:plc:abc/app.bsky.feed.post/123", "error": None,
        }
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is True
        assert result.skipped is False
        assert result.errors == []
        mock_insert.assert_called_once()
        # Verify platform='bluesky' is passed to insert_tweet
        call_kwargs = mock_insert.call_args
        assert call_kwargs.kwargs.get("platform") == "bluesky" or \
               (call_kwargs[1].get("platform") == "bluesky" if len(call_kwargs) > 1 else False)

    @patch("v2.bluesky.get_bluesky_client")
    def test_skips_without_credentials(self, mock_client):
        mock_client.return_value = None
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.skipped is True
        assert result.post_posted is False

    @patch("v2.bluesky.insert_tweet")
    @patch("v2.bluesky.post_to_bluesky")
    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_post_failure(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Post text", "type": "recap"}
        mock_post.return_value = {
            "text": "Post text", "type": "recap", "posted": False, "post_id": None, "error": "Rate limit",
        }
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is False

    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_no_post_generated(self, mock_client, mock_context, mock_generate):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "No trading activity today."
        mock_generate.return_value = None
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is False

    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_context_error_handled(self, mock_client, mock_context):
        mock_client.return_value = MagicMock()
        mock_context.side_effect = Exception("DB connection failed")
        result = run_bluesky_stage(date(2026, 2, 15))
        assert len(result.errors) == 1
        assert "Context gathering failed" in result.errors[0]

    @patch("v2.bluesky.insert_tweet")
    @patch("v2.bluesky.post_to_bluesky")
    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_db_log_error_does_not_crash(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Post", "type": "recap"}
        mock_post.return_value = {
            "text": "Post", "type": "recap", "posted": True,
            "post_id": "at://did:plc:abc/app.bsky.feed.post/123", "error": None,
        }
        mock_insert.side_effect = Exception("DB write failed")
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is True
        assert len(result.errors) == 1
        assert "Failed to log" in result.errors[0]


class TestGenerateBlueskyEntertainmentPost:
    """Verify entertainment post generation via Claude for Bluesky."""

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_generates_post(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({
            "text": "Squidward says the market is overvalued.",
        })
        result = generate_bluesky_entertainment_post("some market context")
        assert result is not None
        assert result["type"] == "entertainment"
        assert "Squidward" in result["text"]
        call_kwargs = mock_retry.call_args
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        assert "300" in call_kwargs.kwargs.get("system", "")

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_bluesky_entertainment_post("context")
        assert result is None

    @patch("v2.bluesky.get_claude_client")
    def test_handles_api_exception(self, mock_get_client):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_bluesky_entertainment_post("context")
        assert result is None
