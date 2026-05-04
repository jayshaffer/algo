"""Tests for v2/premarket.py — pre-market social post pipeline."""

import json
from datetime import date
from unittest.mock import MagicMock, patch


class TestGatherPremarketContext:
    def test_assembles_active_theses_and_latest_memo(self, mock_db, mock_cursor):
        from v2.premarket import gather_premarket_context

        mock_cursor.fetchall.side_effect = [
            [
                {"ticker": "NVDA", "direction": "long", "thesis": "AI demand",
                 "confidence": "high"},
                {"ticker": "TSLA", "direction": "long", "thesis": "EV cycle",
                 "confidence": "medium"},
            ],
        ]
        mock_cursor.fetchone.side_effect = [
            {"content": "Yesterday: NVDA broke out, holding overnight."},
        ]

        ctx = gather_premarket_context(today=date(2026, 5, 4))

        assert "ACTIVE THESES" in ctx
        assert "NVDA" in ctx
        assert "TSLA" in ctx
        assert "STRATEGY MEMO" in ctx
        assert "broke out" in ctx

    def test_handles_no_theses_no_memo(self, mock_db, mock_cursor):
        from v2.premarket import gather_premarket_context

        mock_cursor.fetchall.side_effect = [[]]
        mock_cursor.fetchone.side_effect = [None]

        ctx = gather_premarket_context(today=date(2026, 5, 4))
        assert isinstance(ctx, str)
        assert len(ctx) > 0


def _make_claude_response(json_data: dict):
    """Helper: shape a MagicMock the way `_call_with_retry` returns one."""
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(json_data))]
    return response


class TestGeneratePremarketPost:
    @patch("v2.premarket._call_with_retry")
    @patch("v2.premarket.get_claude_client")
    def test_generates_text(self, mock_get_client, mock_retry):
        from v2.premarket import generate_premarket_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Watching $NVDA into open. AI demand still pulling."}
        )

        result = generate_premarket_post("ctx")
        assert result is not None
        assert "$NVDA" in result["text"]
        assert result["type"] == "premarket"

    @patch("v2.premarket._call_with_retry", side_effect=Exception("API down"))
    @patch("v2.premarket.get_claude_client")
    def test_llm_failure_returns_none(self, mock_get_client, mock_retry):
        from v2.premarket import generate_premarket_post

        mock_get_client.return_value = MagicMock()

        assert generate_premarket_post("ctx") is None


class TestRunPremarketStage:
    @patch("v2.premarket.is_trading_day", return_value=False)
    def test_skipped_on_weekend(self, mock_is_td):
        from v2.premarket import run_premarket_stage

        result = run_premarket_stage(today=date(2026, 5, 9))  # Saturday
        assert result.skipped is True
        assert "weekend" in (result.skip_reason or "").lower() or \
               "trading day" in (result.skip_reason or "").lower()

    @patch("v2.premarket.is_trading_day", return_value=True)
    @patch("v2.premarket.posted_tweet_exists", return_value=True)
    @patch("v2.premarket.get_twitter_client")
    @patch("v2.premarket.get_bluesky_client")
    def test_skipped_when_already_posted_today(
        self, mock_bs, mock_tw, mock_dedup, mock_is_td,
    ):
        """Idempotent: if both platforms already posted today, the whole
        stage short-circuits."""
        from v2.premarket import run_premarket_stage

        mock_tw.return_value = object()
        mock_bs.return_value = object()

        result = run_premarket_stage(today=date(2026, 5, 4))
        assert result.skipped is True

    @patch("v2.premarket.is_trading_day", return_value=True)
    @patch("v2.premarket.insert_tweet", return_value=1)
    @patch("v2.premarket.posted_tweet_exists", return_value=False)
    @patch("v2.premarket.post_to_bluesky")
    @patch("v2.premarket.post_tweet")
    @patch("v2.premarket.generate_premarket_post")
    @patch("v2.premarket.gather_premarket_context", return_value="ctx")
    @patch("v2.premarket.get_twitter_client")
    @patch("v2.premarket.get_bluesky_client")
    def test_posts_to_both_platforms_on_trading_day(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_gen,
        mock_post_tw, mock_post_bs, mock_dedup, mock_insert, mock_is_td,
    ):
        from v2.premarket import run_premarket_stage

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        mock_gen.return_value = {"text": "Watching $NVDA", "type": "premarket"}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "premarket", "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "premarket", "error": None}

        result = run_premarket_stage(today=date(2026, 5, 4))

        assert result.skipped is False
        assert result.twitter_posted is True
        assert result.bluesky_posted is True
        assert mock_insert.call_count == 2
