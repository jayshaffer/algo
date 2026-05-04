"""Tests for v2/social_trades.py — per-trade social post pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_claude_response(json_data: dict):
    """Helper: shape a MagicMock the way `_call_with_retry` returns one."""
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(json_data))]
    return response


class TestGenerateTradePost:
    """Pure path: decision + optional thesis → post body. URL appended
    deterministically after generation, not by the LLM.

    Patches `v2.social_trades.get_claude_client` and `v2.social_trades._call_with_retry`
    directly (the names imported into social_trades), not the originals in
    v2.claude_client — `from x import y` binds y by reference at module load,
    so patching `v2.claude_client.get_claude_client` doesn't reach social_trades.
    """

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_generates_text_and_appends_trade_url(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Bought 12 $NVDA — AI demand still pulling."}
        )

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "AI tailwind",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(
            decision=decision,
            dashboard_base_url="https://dash.example.com",
        )

        assert result is not None
        assert "Bought 12 $NVDA" in result["text"]
        assert result["text"].endswith("https://dash.example.com/trade/99/")
        assert result["decision_id"] == 99
        assert result["type"] == "trade"

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_appends_thesis_url_when_thesis_present(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Bought 12 $NVDA — backing the AI thesis."}
        )

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "AI tailwind",
                    "thesis_id": 42, "thesis_text": "AI demand pulling",
                    "thesis_direction": "long", "is_off_playbook": False}

        result = generate_trade_post(
            decision=decision,
            dashboard_base_url="https://dash.example.com",
        )

        assert "/trade/99/" in result["text"]
        assert "/thesis/42/" in result["text"]

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_no_dashboard_base_url_skips_url_append(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Bought 12 $NVDA."})

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "x",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(decision=decision, dashboard_base_url="")
        assert "http" not in result["text"]

    @patch("v2.social_trades._call_with_retry", side_effect=Exception("API outage"))
    @patch("v2.social_trades.get_claude_client")
    def test_llm_failure_returns_none(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "x",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(decision=decision, dashboard_base_url="")
        assert result is None

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_llm_returns_malformed_json_returns_none(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="not json at all")]
        mock_retry.return_value = bad_response

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "x",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(decision=decision, dashboard_base_url="")
        assert result is None


class TestRunTradePostsStage:
    """End-to-end stage orchestrator. All external calls mocked."""

    def _decision(self, decision_id: int, ticker: str = "NVDA"):
        return {
            "id": decision_id, "ticker": ticker, "action": "buy",
            "quantity": 10, "price": 500.0, "reasoning": "AI tailwind",
            "thesis_id": None, "thesis_text": None,
            "thesis_direction": None, "is_off_playbook": False,
        }

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client")
    @patch("v2.social_trades.get_twitter_client")
    def test_posts_one_per_decision_to_each_platform(
        self, mock_tw_client, mock_bs_client, mock_select,
        mock_gen, mock_post_tw, mock_post_bs,
        mock_dedup, mock_insert,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        mock_select.return_value = [self._decision(1), self._decision(2, "TSLA")]
        mock_gen.side_effect = [
            {"text": "Bought 10 $NVDA", "type": "trade", "decision_id": 1},
            {"text": "Bought 10 $TSLA", "type": "trade", "decision_id": 2},
        ]
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "trade", "error": None}

        result = run_trade_posts_stage(date(2026, 5, 4))

        assert mock_insert.call_count == 4
        assert result.posts_attempted == 2
        assert result.posts_succeeded_twitter == 2
        assert result.posts_succeeded_bluesky == 2
        assert result.skipped is False

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists")
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    @patch("v2.social_trades.get_twitter_client")
    def test_per_decision_dedup_skips_already_posted(
        self, mock_tw_client, mock_bs_client, mock_select,
        mock_gen, mock_post_tw, mock_post_bs,
        mock_dedup, mock_insert,
    ):
        """If decision 2 was already posted to Twitter, skip the post for it
        but still attempt decision 1."""
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        mock_select.return_value = [self._decision(1), self._decision(2)]
        mock_dedup.side_effect = lambda decision_id, platform: (
            decision_id == 2 and platform == "twitter"
        )
        mock_gen.return_value = {"text": "x", "type": "trade", "decision_id": 1}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}

        result = run_trade_posts_stage(date(2026, 5, 4))

        assert mock_post_tw.call_count == 1
        assert result.posts_skipped_dedup == 1

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    @patch("v2.social_trades.get_twitter_client")
    def test_one_decision_failure_does_not_drop_others(
        self, mock_tw_client, mock_bs_client, mock_select,
        mock_gen, mock_post_tw, mock_post_bs,
        mock_dedup, mock_insert,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        mock_select.return_value = [self._decision(1), self._decision(2)]
        mock_gen.side_effect = [None, {"text": "x", "type": "trade", "decision_id": 2}]
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}

        result = run_trade_posts_stage(date(2026, 5, 4))

        assert mock_post_tw.call_count == 1
        assert result.posts_succeeded_twitter == 1
        assert result.posts_failed >= 1

    @patch("v2.social_trades.select_postable_decisions_for_date", return_value=[])
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    @patch("v2.social_trades.get_twitter_client")
    def test_no_decisions_falls_through_to_quiet_day_handler(
        self, mock_tw_client, mock_bs_client, mock_select,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        with patch("v2.social_trades._post_quiet_day_recap", return_value=None) as mock_quiet:
            result = run_trade_posts_stage(date(2026, 5, 4))

        assert result.posts_attempted == 0
        mock_quiet.assert_called_once()

    @patch("v2.social_trades.get_twitter_client", return_value=None)
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    def test_skipped_when_no_credentials_on_either_platform(
        self, mock_bs, mock_tw,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        result = run_trade_posts_stage(date(2026, 5, 4))
        assert result.skipped is True
