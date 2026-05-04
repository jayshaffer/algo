"""Tests for v2/social_weekly.py — weekly mistakes + attribution social posts."""

import json
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch


def _make_claude_response(json_data: dict):
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(json_data))]
    return response


class TestGatherMistakesContext:
    def test_returns_top_loser_and_retired_rule(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_mistakes_context

        with patch("v2.social_weekly.get_closed_losers", return_value=[
                    {"id": 1, "ticker": "TSLA", "action": "buy",
                     "quantity": 5, "price": Decimal("200"),
                     "outcome_30d": Decimal("-12.5"),
                     "reasoning": "EV cycle"}]), \
             patch("v2.social_weekly.get_retired_rules", return_value=[
                    {"rule_text": "Cap macro at $500/day",
                     "retirement_reason": "stale"}]):
            ctx = gather_mistakes_context(today=date(2026, 5, 8))

        assert "TSLA" in ctx
        assert "-12.5" in ctx
        assert "Cap macro" in ctx

    def test_handles_empty_data(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_mistakes_context
        with patch("v2.social_weekly.get_closed_losers", return_value=[]), \
             patch("v2.social_weekly.get_retired_rules", return_value=[]):
            ctx = gather_mistakes_context(today=date(2026, 5, 8))
        assert ctx == ""


class TestGenerateMistakesPost:
    @patch("v2.social_weekly._call_with_retry")
    @patch("v2.social_weekly.get_claude_client")
    def test_generates_text(self, mock_get_client, mock_retry):
        from v2.social_weekly import generate_mistakes_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Worst trade this week: $TSLA -12.5%. Reason was thin."}
        )

        post = generate_mistakes_post("ctx", dashboard_base_url="https://example.com")

        assert post is not None
        assert "TSLA" in post["text"]
        assert "https://example.com/mistakes/" in post["text"]
        assert post["type"] == "weekly_mistakes"

    @patch("v2.social_weekly._call_with_retry", side_effect=Exception("API down"))
    @patch("v2.social_weekly.get_claude_client")
    def test_llm_failure_returns_none(self, mock_get_client, mock_retry):
        from v2.social_weekly import generate_mistakes_post

        mock_get_client.return_value = MagicMock()
        assert generate_mistakes_post("ctx", dashboard_base_url="") is None


class TestGatherAttributionContext:
    def test_summarizes_top_and_bottom(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_attribution_context

        with patch("v2.social_weekly.get_signal_attribution", return_value=[
                    {"category": "earnings", "sample_size": 30,
                     "avg_outcome_30d": Decimal("3.4")},
                    {"category": "fed", "sample_size": 12,
                     "avg_outcome_30d": Decimal("-1.2")},
                    {"category": "macro", "sample_size": 9,
                     "avg_outcome_30d": Decimal("0.8")},
                ]):
            ctx = gather_attribution_context()

        assert "earnings" in ctx
        assert "fed" in ctx

    def test_handles_no_attribution(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_attribution_context

        with patch("v2.social_weekly.get_signal_attribution", return_value=[]):
            ctx = gather_attribution_context()
        assert ctx == ""


class TestGenerateAttributionPost:
    @patch("v2.social_weekly._call_with_retry")
    @patch("v2.social_weekly.get_claude_client")
    def test_generates_text(self, mock_get_client, mock_retry):
        from v2.social_weekly import generate_attribution_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Earnings signals predicted (+3.4%, n=30); fed news was noise."}
        )
        post = generate_attribution_post("ctx", dashboard_base_url="https://example.com")
        assert post is not None
        assert "Earnings" in post["text"]
        assert "https://example.com/attribution/" in post["text"]
        assert post["type"] == "weekly_attribution"


class TestRunMistakesPost:
    @patch("v2.social_weekly.is_trading_day", return_value=False)
    def test_skipped_on_weekend(self, mock_is_td):
        from v2.social_weekly import run_mistakes_post

        result = run_mistakes_post(today=date(2026, 5, 9))  # Saturday
        assert result.skipped is True

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.gather_mistakes_context", return_value="")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_skipped_when_no_data(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_is_td,
    ):
        from v2.social_weekly import run_mistakes_post

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()

        result = run_mistakes_post(today=date(2026, 5, 8))
        assert result.skipped is True
        assert "no data" in (result.skip_reason or "").lower()

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.posted_tweet_exists", return_value=True)
    @patch("v2.social_weekly.gather_mistakes_context", return_value="ctx")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_skipped_when_already_posted(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_dedup, mock_is_td,
    ):
        from v2.social_weekly import run_mistakes_post

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        result = run_mistakes_post(today=date(2026, 5, 8))
        assert result.skipped is True

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.insert_tweet", return_value=1)
    @patch("v2.social_weekly.posted_tweet_exists", return_value=False)
    @patch("v2.social_weekly.post_to_bluesky")
    @patch("v2.social_weekly.post_tweet")
    @patch("v2.social_weekly.generate_mistakes_post")
    @patch("v2.social_weekly.gather_mistakes_context", return_value="ctx")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_posts_to_both_platforms(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_gen,
        mock_post_tw, mock_post_bs, mock_dedup, mock_insert, mock_is_td,
    ):
        from v2.social_weekly import run_mistakes_post

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        mock_gen.return_value = {"text": "x", "type": "weekly_mistakes"}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "weekly_mistakes",
                                     "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "weekly_mistakes",
                                     "error": None}

        result = run_mistakes_post(today=date(2026, 5, 8))
        assert result.skipped is False
        assert result.twitter_posted is True
        assert result.bluesky_posted is True
        assert mock_insert.call_count == 2


class TestRunAttributionPost:
    @patch("v2.social_weekly.is_trading_day", return_value=False)
    def test_skipped_on_weekend(self, mock_is_td):
        from v2.social_weekly import run_attribution_post

        result = run_attribution_post(today=date(2026, 5, 9))
        assert result.skipped is True

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.gather_attribution_context", return_value="")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_skipped_when_no_data(
        self, mock_bs, mock_tw, mock_ctx, mock_is_td,
    ):
        from v2.social_weekly import run_attribution_post

        mock_tw.return_value = object()
        mock_bs.return_value = object()
        result = run_attribution_post(today=date(2026, 5, 8))
        assert result.skipped is True
