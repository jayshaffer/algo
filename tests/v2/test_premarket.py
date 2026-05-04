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
