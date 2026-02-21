"""Tests for the Bluesky integration (Bikini Bottom Capital) on v2 pipeline."""

import os
from datetime import date
from unittest.mock import MagicMock, patch, call

import pytest

from v2.bluesky import (
    get_bluesky_client, post_to_bluesky, generate_bluesky_post,
    BLUESKY_SYSTEM_PROMPT, run_bluesky_stage, BlueskyStageResult,
    generate_bluesky_entertainment_post, BLUESKY_ENTERTAINMENT_SYSTEM_PROMPT,
    BLUESKY_GRAPHEME_LIMIT, _grapheme_length, _grapheme_truncate,
    _dashboard_url_suffix, _condense_post, _enforce_limit,
    _build_link_facets, _DASHBOARD_LINK_TEXT,
    _PROMPT_BUFFER, _SHORTEN_MAX_RETRIES,
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
        mock_client.send_post.assert_called_once_with(text="Hello from Bikini Bottom!", facets=None)

    @patch("v2.bluesky._build_link_facets")
    @patch("v2.bluesky.get_bluesky_client")
    def test_posts_with_facets_when_dashboard_url(self, mock_get_client, mock_build):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.uri = "at://did:plc:abc/app.bsky.feed.post/456"
        mock_client.send_post.return_value = mock_response
        mock_get_client.return_value = mock_client
        mock_facets = [MagicMock()]
        mock_build.return_value = mock_facets
        post = {"text": "Ahoy!\nportfolio", "type": "recap", "dashboard_url": "https://example.github.io"}
        result = post_to_bluesky(post)
        assert result["posted"] is True
        mock_build.assert_called_once_with("Ahoy!\nportfolio", _DASHBOARD_LINK_TEXT, "https://example.github.io")
        mock_client.send_post.assert_called_once_with(text="Ahoy!\nportfolio", facets=mock_facets)

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
        os.environ.pop("DASHBOARD_URL", None)
        result = generate_bluesky_post("test context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day for me treasure! $AAPL up big!"
        assert result["type"] == "recap"
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        expected_limit = BLUESKY_GRAPHEME_LIMIT - _PROMPT_BUFFER
        assert f"{expected_limit} characters" in call_kwargs.kwargs.get("system", "")

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_prompt_limit_reduced_for_dashboard_link(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Ahoy!"})
        url = "https://example.github.io/dash"
        with patch.dict(os.environ, {"DASHBOARD_URL": url}):
            generate_bluesky_post("context")
        call_kwargs = mock_retry.call_args
        system = call_kwargs.kwargs.get("system", "")
        # Limit should be 300 minus the link suffix length ("\nportfolio") minus prompt buffer
        expected_limit = BLUESKY_GRAPHEME_LIMIT - _grapheme_length(f"\n{_DASHBOARD_LINK_TEXT}") - _PROMPT_BUFFER
        assert f"{expected_limit} characters" in system

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

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_generate_bluesky_post_appends_dashboard_link(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Ahoy! Great day!"})
        with patch.dict(os.environ, {"DASHBOARD_URL": "https://example.github.io"}):
            result = generate_bluesky_post("context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day!\nportfolio"
        assert result["dashboard_url"] == "https://example.github.io"

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_generate_bluesky_post_no_url_when_not_set(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Ahoy! Great day!"})
        os.environ.pop("DASHBOARD_URL", None)
        result = generate_bluesky_post("context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day!"


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
        # First call is the generation call
        call_kwargs = mock_retry.call_args_list[0]
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        assert "270 characters" in call_kwargs.kwargs.get("system", "")

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

    @patch("v2.bluesky._enforce_limit")
    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_enforces_limit_on_long_post(self, mock_get_client, mock_retry, mock_enforce):
        mock_get_client.return_value = MagicMock()
        long_text = "A" * 350
        mock_retry.return_value = _make_claude_response({"text": long_text})
        mock_enforce.return_value = "A" * BLUESKY_GRAPHEME_LIMIT
        result = generate_bluesky_entertainment_post("context")
        assert result is not None
        mock_enforce.assert_called_once()
        assert _grapheme_length(result["text"]) == BLUESKY_GRAPHEME_LIMIT


class TestGraphemeHelpers:
    """Verify grapheme length and truncation helpers."""

    def test_length_ascii(self):
        assert _grapheme_length("hello") == 5

    def test_length_emoji(self):
        # Family emoji is one grapheme cluster
        assert _grapheme_length("\U0001F468\u200D\U0001F469\u200D\U0001F467") == 1

    def test_truncate_under_limit(self):
        assert _grapheme_truncate("short", 300) == "short"

    def test_truncate_at_limit(self):
        text = "A" * 300
        assert _grapheme_truncate(text, 300) == text

    def test_truncate_over_limit(self):
        text = "B" * 350
        result = _grapheme_truncate(text, 300)
        assert _grapheme_length(result) == 300

    def test_truncate_preserves_grapheme_clusters(self):
        # 299 A's + a multi-codepoint emoji = 300 graphemes
        emoji = "\U0001F468\u200D\U0001F469\u200D\U0001F467"  # family emoji, 1 grapheme
        text = "A" * 299 + emoji
        assert _grapheme_length(text) == 300
        truncated = _grapheme_truncate(text, 300)
        assert truncated == text


class TestCondensePost:
    """Verify _condense_post asks LLM to shorten text."""

    @patch("v2.bluesky._call_with_retry")
    def test_returns_shortened_text(self, mock_retry):
        mock_retry.return_value = _make_claude_response({"text": "shorter post"})
        result = _condense_post(MagicMock(), "A" * 350, 300, "claude-haiku-4-5-20251001")
        assert result == "shorter post"
        call_kwargs = mock_retry.call_args
        assert "300 characters" in call_kwargs.kwargs["system"]

    @patch("v2.bluesky._call_with_retry")
    def test_returns_none_on_malformed_response(self, mock_retry):
        mock_retry.return_value = _make_claude_response({"wrong_key": "nope"})
        result = _condense_post(MagicMock(), "A" * 350, 300, "claude-haiku-4-5-20251001")
        assert result is None

    @patch("v2.bluesky._call_with_retry")
    def test_returns_none_on_exception(self, mock_retry):
        mock_retry.side_effect = Exception("API down")
        result = _condense_post(MagicMock(), "A" * 350, 300, "claude-haiku-4-5-20251001")
        assert result is None

    @patch("v2.bluesky._call_with_retry")
    def test_handles_fenced_json(self, mock_retry):
        fenced = '```json\n{"text": "short"}\n```'
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_retry.return_value = mock_resp
        result = _condense_post(MagicMock(), "A" * 350, 300, "claude-haiku-4-5-20251001")
        assert result == "short"


class TestEnforceLimit:
    """Verify _enforce_limit retry and truncation logic."""

    def test_returns_text_under_limit(self):
        text = "A" * 250
        result = _enforce_limit(MagicMock(), text, 300, "model")
        assert result == text

    @patch("v2.bluesky._condense_post")
    def test_condenses_once_when_successful(self, mock_condense):
        short = "B" * 280
        mock_condense.return_value = short
        result = _enforce_limit(MagicMock(), "A" * 350, 300, "model")
        assert result == short
        assert mock_condense.call_count == 1

    @patch("v2.bluesky._condense_post")
    def test_retries_condense_up_to_max(self, mock_condense):
        # First condense returns still-too-long, second returns OK
        mock_condense.side_effect = ["C" * 320, "C" * 290]
        result = _enforce_limit(MagicMock(), "A" * 350, 300, "model")
        assert result == "C" * 290
        assert mock_condense.call_count == 2

    @patch("v2.bluesky._condense_post")
    def test_truncates_as_last_resort(self, mock_condense):
        mock_condense.return_value = None  # condense keeps failing
        result = _enforce_limit(MagicMock(), "A" * 350, 300, "model")
        assert _grapheme_length(result) == 300

    @patch("v2.bluesky._condense_post")
    def test_truncates_when_condense_still_too_long(self, mock_condense):
        mock_condense.return_value = "D" * 320  # still over after all retries
        result = _enforce_limit(MagicMock(), "A" * 350, 300, "model")
        assert _grapheme_length(result) == 300


class TestRecapPostTruncation:
    """Verify recap posts are enforced to grapheme limit."""

    @patch("v2.bluesky._enforce_limit")
    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_enforces_limit_on_long_recap(self, mock_get_client, mock_retry, mock_enforce):
        mock_get_client.return_value = MagicMock()
        long_text = "X" * 350
        mock_retry.return_value = _make_claude_response({"text": long_text})
        mock_enforce.return_value = "X" * BLUESKY_GRAPHEME_LIMIT
        os.environ.pop("DASHBOARD_URL", None)
        result = generate_bluesky_post("context")
        assert result is not None
        mock_enforce.assert_called_once_with(
            mock_get_client.return_value, long_text, BLUESKY_GRAPHEME_LIMIT, "claude-haiku-4-5-20251001",
        )
        assert _grapheme_length(result["text"]) == BLUESKY_GRAPHEME_LIMIT

    @patch("v2.bluesky._enforce_limit")
    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_enforces_body_limit_before_dashboard_link(self, mock_get_client, mock_retry, mock_enforce):
        mock_get_client.return_value = MagicMock()
        body = "A" * 290
        mock_retry.return_value = _make_claude_response({"text": body})
        url = "https://example.github.io/dashboard"
        suffix_len = _grapheme_length(f"\n{_DASHBOARD_LINK_TEXT}")
        body_limit = BLUESKY_GRAPHEME_LIMIT - suffix_len
        # _enforce_limit shortens the body to fit
        mock_enforce.return_value = "A" * body_limit
        with patch.dict(os.environ, {"DASHBOARD_URL": url}):
            result = generate_bluesky_post("context")
        assert result is not None
        mock_enforce.assert_called_once_with(
            mock_get_client.return_value, body, body_limit, "claude-haiku-4-5-20251001",
        )
        assert result["text"] == "A" * body_limit + f"\n{_DASHBOARD_LINK_TEXT}"
        assert result["dashboard_url"] == url
        assert _grapheme_length(result["text"]) <= BLUESKY_GRAPHEME_LIMIT


class TestBuildLinkFacets:
    """Verify _build_link_facets byte-offset calculation."""

    def test_builds_facet_with_correct_byte_offsets(self):
        text = "Ahoy!\nportfolio"
        url = "https://example.github.io"
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"atproto": mock_models}):
            result = _build_link_facets(text, "portfolio", url)
        assert result is not None
        assert len(result) == 1
        # "portfolio" starts at byte 6 ("\nportfolio" after "Ahoy!")
        mock_models.models.AppBskyRichtextFacet.ByteSlice.assert_called_once_with(
            byte_start=6, byte_end=15,
        )
        mock_models.models.AppBskyRichtextFacet.Link.assert_called_once_with(uri=url)

    def test_returns_none_when_link_text_not_found(self):
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"atproto": mock_models}):
            result = _build_link_facets("No link here", "portfolio", "https://example.com")
        assert result is None

    def test_returns_none_without_atproto(self):
        # Simulate ImportError for atproto
        with patch.dict("sys.modules", {"atproto": None}):
            result = _build_link_facets("text\nportfolio", "portfolio", "https://example.com")
        assert result is None

    def test_handles_multibyte_characters(self):
        # Emoji before the link text shifts byte offsets
        text = "\U0001F4B0\nportfolio"  # money bag emoji (4 UTF-8 bytes)
        mock_models = MagicMock()
        with patch.dict("sys.modules", {"atproto": mock_models}):
            result = _build_link_facets(text, "portfolio", "https://example.com")
        assert result is not None
        # \U0001F4B0 = 4 bytes, \n = 1 byte, so "portfolio" starts at byte 5
        mock_models.models.AppBskyRichtextFacet.ByteSlice.assert_called_once_with(
            byte_start=5, byte_end=14,
        )
