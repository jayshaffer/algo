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


class TestFetchOgImage:
    """`_fetch_og_image` returns the PNG bytes for /og/trade/<id>.png on the
    public dashboard, or None on any failure. Failures must not raise — a
    missing card is degraded gracefully to a no-image post."""

    @patch("v2.social_trades.requests.get")
    def test_returns_png_bytes_on_success(self, mock_get):
        from v2.social_trades import _fetch_og_image

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"\x89PNG\r\n\x1a\n" + b"data"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = _fetch_og_image(decision_id=328, dashboard_base_url="https://bbottomcap.com")

        assert result == b"\x89PNG\r\n\x1a\ndata"
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://bbottomcap.com/og/trade/328.png"
        assert kwargs.get("timeout") is not None

    @patch("v2.social_trades.requests.get")
    def test_returns_none_on_http_error(self, mock_get):
        from v2.social_trades import _fetch_og_image

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        result = _fetch_og_image(decision_id=999, dashboard_base_url="https://bbottomcap.com")
        assert result is None

    @patch("v2.social_trades.requests.get")
    def test_returns_none_on_network_exception(self, mock_get):
        from v2.social_trades import _fetch_og_image

        mock_get.side_effect = Exception("Connection refused")
        result = _fetch_og_image(decision_id=1, dashboard_base_url="https://bbottomcap.com")
        assert result is None

    def test_returns_none_when_no_dashboard_base_url(self):
        from v2.social_trades import _fetch_og_image

        result = _fetch_og_image(decision_id=1, dashboard_base_url="")
        assert result is None

    @patch("v2.social_trades.requests.get")
    def test_strips_trailing_slash_from_base_url(self, mock_get):
        from v2.social_trades import _fetch_og_image

        mock_response = MagicMock()
        mock_response.content = b"png"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        _fetch_og_image(decision_id=42, dashboard_base_url="https://bbottomcap.com/")
        assert mock_get.call_args.args[0] == "https://bbottomcap.com/og/trade/42.png"


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


class TestBlueskyExternalCardWiring:
    """`run_trade_posts_stage` should enrich the bluesky post_body with an
    `external_card` so post_to_bluesky can attach a link preview. The card
    URI/title/description are computed from the decision; the thumbnail is
    the pre-rendered OG PNG fetched from the public dashboard. Twitter's
    post_body must NOT carry the card (would either be ignored or
    surface as accidental coupling later)."""

    def _decision(self, decision_id: int = 11, ticker: str = "AMZN"):
        return {
            "id": decision_id, "ticker": ticker, "action": "buy",
            "quantity": 1.1, "price": 271.60, "reasoning": "Q1 solid",
            "thesis_id": None, "thesis_text": None,
            "thesis_direction": None, "is_off_playbook": False,
        }

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades._fetch_og_image")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client")
    @patch("v2.social_trades.get_twitter_client")
    def test_bluesky_post_body_carries_external_card_with_thumb(
        self, mock_tw_client, mock_bs_client, mock_select, mock_fetch,
        mock_gen, mock_post_tw, mock_post_bs, mock_dedup, mock_insert,
        monkeypatch,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        monkeypatch.setenv("DASHBOARD_URL", "https://bbottomcap.com")

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        mock_select.return_value = [self._decision(11, "AMZN")]
        mock_gen.return_value = {"text": "Bought 1.1 $AMZN", "type": "trade",
                                 "decision_id": 11}
        mock_fetch.return_value = b"\x89PNGfake"
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "trade", "error": None}

        run_trade_posts_stage(date(2026, 5, 4))

        # Twitter receives a post_body WITHOUT external_card
        tw_body = mock_post_tw.call_args.args[0]
        assert "external_card" not in tw_body

        # Bluesky receives a post_body WITH external_card
        bs_body = mock_post_bs.call_args.args[0]
        card = bs_body.get("external_card")
        assert card is not None, "bluesky post_body missing external_card"
        assert card["uri"] == "https://bbottomcap.com/trade/11/"
        assert "AMZN" in card["title"]
        assert "AMZN" in card["description"]
        assert card["thumb_png"] == b"\x89PNGfake"

        mock_fetch.assert_called_once_with(11, "https://bbottomcap.com")

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades._fetch_og_image")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client")
    @patch("v2.social_trades.get_twitter_client", return_value=None)
    def test_card_attached_without_thumb_when_image_fetch_fails(
        self, mock_tw_client, mock_bs_client, mock_select, mock_fetch,
        mock_gen, mock_post_bs, mock_dedup, mock_insert, monkeypatch,
    ):
        """If the OG image fetch fails, still attach the card (text-only) —
        a card without an image is better than a bare URL with no preview."""
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        monkeypatch.setenv("DASHBOARD_URL", "https://bbottomcap.com")

        mock_bs_client.return_value = object()
        mock_select.return_value = [self._decision(11, "AMZN")]
        mock_gen.return_value = {"text": "Bought", "type": "trade", "decision_id": 11}
        mock_fetch.return_value = None
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "trade", "error": None}

        run_trade_posts_stage(date(2026, 5, 4))

        bs_body = mock_post_bs.call_args.args[0]
        card = bs_body.get("external_card")
        assert card is not None
        assert "thumb_png" not in card
        assert card["uri"] == "https://bbottomcap.com/trade/11/"

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades._fetch_og_image")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client")
    @patch("v2.social_trades.get_twitter_client", return_value=None)
    def test_no_card_and_no_fetch_when_dashboard_url_unset(
        self, mock_tw_client, mock_bs_client, mock_select, mock_fetch,
        mock_gen, mock_post_bs, mock_dedup, mock_insert, monkeypatch,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        monkeypatch.delenv("DASHBOARD_URL", raising=False)

        mock_bs_client.return_value = object()
        mock_select.return_value = [self._decision(11, "AMZN")]
        mock_gen.return_value = {"text": "Bought", "type": "trade", "decision_id": 11}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "trade", "error": None}

        run_trade_posts_stage(date(2026, 5, 4))

        bs_body = mock_post_bs.call_args.args[0]
        assert "external_card" not in bs_body
        mock_fetch.assert_not_called()

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades._fetch_og_image")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    @patch("v2.social_trades.get_twitter_client")
    def test_no_image_fetch_when_bluesky_disabled(
        self, mock_tw_client, mock_bs_client, mock_select, mock_fetch,
        mock_gen, mock_post_tw, mock_dedup, mock_insert, monkeypatch,
    ):
        """Don't waste an HTTP call fetching the OG PNG when there's no
        Bluesky client to consume it."""
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        monkeypatch.setenv("DASHBOARD_URL", "https://bbottomcap.com")

        mock_tw_client.return_value = object()
        mock_select.return_value = [self._decision(11, "AMZN")]
        mock_gen.return_value = {"text": "Bought", "type": "trade", "decision_id": 11}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}

        run_trade_posts_stage(date(2026, 5, 4))

        mock_fetch.assert_not_called()


class TestQuietDayFallback:
    @patch("v2.social_trades.is_trading_day", return_value=False)
    def test_skips_quiet_day_recap_on_non_trading_day(self, mock_is_td):
        """Weekends/holidays produce no post — even the quiet-day recap is muted."""
        from datetime import date
        from v2.social_trades import _post_quiet_day_recap, TradePostsStageResult

        result = TradePostsStageResult()
        _post_quiet_day_recap(date(2026, 5, 9), object(), object(), result)  # Saturday
        assert result.quiet_day_recap_posted is False

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_bluesky_post")
    @patch("v2.social_trades.generate_tweet")
    @patch("v2.social_trades.gather_tweet_context", return_value="ctx")
    @patch("v2.social_trades.is_trading_day", return_value=True)
    def test_posts_recap_on_trading_day_no_decisions(
        self, mock_is_td, mock_ctx, mock_gen_tw, mock_gen_bs,
        mock_post_tw, mock_post_bs, mock_dedup, mock_insert,
    ):
        from datetime import date
        from v2.social_trades import _post_quiet_day_recap, TradePostsStageResult

        mock_gen_tw.return_value = {"text": "Quiet day.", "type": "recap"}
        mock_gen_bs.return_value = {"text": "Quiet day.", "type": "recap"}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "Quiet day.", "type": "recap", "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "Quiet day.", "type": "recap", "error": None}

        result = TradePostsStageResult()
        _post_quiet_day_recap(date(2026, 5, 4), object(), object(), result)

        assert result.quiet_day_recap_posted is True
        mock_post_tw.assert_called_once()
        mock_post_bs.assert_called_once()

    @patch("v2.social_trades.posted_tweet_exists", return_value=True)
    @patch("v2.social_trades.gather_tweet_context", return_value="ctx")
    @patch("v2.social_trades.is_trading_day", return_value=True)
    def test_dedup_blocks_recap_when_already_posted(
        self, mock_is_td, mock_ctx, mock_dedup,
    ):
        """Existing rerun guard: if a recap was already posted today on
        either platform, don't repost on that platform."""
        from datetime import date
        from v2.social_trades import _post_quiet_day_recap, TradePostsStageResult

        result = TradePostsStageResult()
        with patch("v2.social_trades.post_tweet") as mock_post_tw, \
             patch("v2.social_trades.post_to_bluesky") as mock_post_bs:
            _post_quiet_day_recap(date(2026, 5, 4), object(), object(), result)

        mock_post_tw.assert_not_called()
        mock_post_bs.assert_not_called()
