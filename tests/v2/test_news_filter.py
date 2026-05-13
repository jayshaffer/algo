"""Tests for v2/news_filter.py — Haiku relevance filter."""

import json
from unittest.mock import MagicMock, patch


def _signal(id_: int, ticker: str = "AAPL", summary: str = "summary text") -> dict:
    """Build a minimal signal dict matching the news_signals row shape."""
    return {
        "id": id_,
        "ticker": ticker,
        "headline": f"headline {id_}",
        "category": "momentum",
        "sentiment": "bullish",
        "summary": summary,
    }


def _mock_haiku_response(json_payload: dict) -> MagicMock:
    """Build a fake anthropic Message with one text block."""
    block = MagicMock()
    block.text = json.dumps(json_payload)
    msg = MagicMock()
    msg.content = [block]
    return msg


class TestCurateSignals:
    @patch("v2.news_filter._call_haiku")
    def test_returns_top_n_ids(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [3, 7, 12]})
        signals = [_signal(i) for i in [3, 5, 7, 9, 12]]

        result = curate_signals(signals, target_n=3, regime_context="risk-on")
        assert result == [3, 7, 12]

    @patch("v2.news_filter._call_haiku")
    def test_drops_hallucinated_ids(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [3, 999]})
        signals = [_signal(i) for i in [3, 5]]

        result = curate_signals(signals, target_n=5, regime_context="x")
        assert result == [3]

    @patch("v2.news_filter._call_haiku")
    def test_falls_back_on_api_error(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.side_effect = RuntimeError("Haiku 500")
        signals = [_signal(i) for i in [1, 2, 3]]

        result = curate_signals(signals, target_n=2, regime_context="x")
        assert result == [1, 2, 3], "API error must degrade to firehose (all input IDs)"

    @patch("v2.news_filter._call_haiku")
    def test_falls_back_on_malformed_json(self, mock_call):
        from v2.news_filter import curate_signals

        block = MagicMock()
        block.text = "not json at all"
        msg = MagicMock()
        msg.content = [block]
        mock_call.return_value = msg

        signals = [_signal(i) for i in [1, 2]]
        result = curate_signals(signals, target_n=1, regime_context="x")
        assert result == [1, 2]

    @patch("v2.news_filter._call_haiku")
    def test_falls_back_on_empty_intersection(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [999, 1000]})
        signals = [_signal(i) for i in [1, 2]]

        result = curate_signals(signals, target_n=1, regime_context="x")
        assert result == [1, 2]

    @patch("v2.news_filter._call_haiku")
    def test_passes_regime_context_to_haiku(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [1]})
        signals = [_signal(1)]

        curate_signals(signals, target_n=1, regime_context="VIX 12, risk-on tape")

        sent_kwargs = mock_call.call_args.kwargs
        sent_messages = sent_kwargs.get("messages") or mock_call.call_args.args[1]
        prompt_text = json.dumps(sent_messages)
        assert "VIX 12, risk-on tape" in prompt_text

    @patch("v2.news_filter._call_haiku")
    def test_input_includes_summary_text(self, mock_call):
        from v2.news_filter import curate_signals

        mock_call.return_value = _mock_haiku_response({"top_ids": [1]})
        signals = [_signal(1, summary="Foxconn deal pushes AAPL +5%")]

        curate_signals(signals, target_n=1, regime_context="x")

        sent_kwargs = mock_call.call_args.kwargs
        sent_messages = sent_kwargs.get("messages") or mock_call.call_args.args[1]
        prompt_text = json.dumps(sent_messages)
        assert "Foxconn deal pushes AAPL +5%" in prompt_text

    @patch("v2.news_filter._call_haiku")
    def test_empty_input_returns_empty(self, mock_call):
        from v2.news_filter import curate_signals

        result = curate_signals([], target_n=30, regime_context="x")
        assert result == []
        mock_call.assert_not_called()
