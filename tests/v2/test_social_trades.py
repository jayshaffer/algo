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
