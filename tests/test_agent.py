"""Tests for trading/agent.py - LLM integration and decision validation."""

import json
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from trading.agent import (
    TradingDecision,
    ThesisInvalidation,
    AgentResponse,
    get_ollama_client,
    get_trading_decisions,
    format_decisions_for_logging,
    validate_decision,
)
from tests.conftest import make_trading_decision, make_agent_response


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestTradingDecisionDataclass:
    def test_creates_with_all_fields(self):
        d = TradingDecision(
            action="buy",
            ticker="AAPL",
            quantity=10,
            reasoning="Strong earnings",
            confidence="high",
            thesis_id=42,
        )
        assert d.action == "buy"
        assert d.ticker == "AAPL"
        assert d.quantity == 10
        assert d.reasoning == "Strong earnings"
        assert d.confidence == "high"
        assert d.thesis_id == 42

    def test_thesis_id_defaults_to_none(self):
        d = TradingDecision(
            action="hold",
            ticker="MSFT",
            quantity=None,
            reasoning="No action",
            confidence="low",
        )
        assert d.thesis_id is None


class TestThesisInvalidationDataclass:
    def test_creates_correctly(self):
        inv = ThesisInvalidation(thesis_id=7, reason="Revenue declined")
        assert inv.thesis_id == 7
        assert inv.reason == "Revenue declined"


class TestAgentResponseDataclass:
    def test_creates_with_all_fields(self):
        dec = make_trading_decision()
        inv = ThesisInvalidation(thesis_id=1, reason="test")
        resp = AgentResponse(
            decisions=[dec],
            thesis_invalidations=[inv],
            market_summary="Bullish",
            risk_assessment="Low",
        )
        assert len(resp.decisions) == 1
        assert len(resp.thesis_invalidations) == 1
        assert resp.market_summary == "Bullish"
        assert resp.risk_assessment == "Low"


# ---------------------------------------------------------------------------
# get_ollama_client
# ---------------------------------------------------------------------------


class TestGetOllamaClient:
    def test_uses_env_variable(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://custom:9999")
        with patch("trading.agent.ollama.Client") as mock_client:
            get_ollama_client()
            mock_client.assert_called_once_with(host="http://custom:9999")

    def test_defaults_to_localhost(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_URL", raising=False)
        with patch("trading.agent.ollama.Client") as mock_client:
            get_ollama_client()
            mock_client.assert_called_once_with(host="http://localhost:11434")


# ---------------------------------------------------------------------------
# get_trading_decisions
# ---------------------------------------------------------------------------


def _make_ollama_chat_response(data: dict) -> dict:
    """Helper to build a mock ollama chat response."""
    return {"message": {"content": json.dumps(data)}}


def _sample_llm_data(
    decisions=None,
    thesis_invalidations=None,
    market_summary="All good",
    risk_assessment="Low risk",
):
    if decisions is None:
        decisions = [
            {
                "action": "buy",
                "ticker": "AAPL",
                "quantity": 5,
                "reasoning": "Looks great",
                "confidence": "high",
                "thesis_id": None,
            }
        ]
    if thesis_invalidations is None:
        thesis_invalidations = []
    return {
        "decisions": decisions,
        "thesis_invalidations": thesis_invalidations,
        "market_summary": market_summary,
        "risk_assessment": risk_assessment,
    }


class TestGetTradingDecisions:
    @patch("trading.agent.get_ollama_client")
    def test_parses_valid_json_response(self, mock_get_client):
        data = _sample_llm_data()
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_ollama_chat_response(data)
        mock_get_client.return_value = mock_client

        resp = get_trading_decisions("some context")

        assert isinstance(resp, AgentResponse)
        assert len(resp.decisions) == 1
        assert resp.decisions[0].action == "buy"
        assert resp.decisions[0].ticker == "AAPL"
        assert resp.decisions[0].quantity == 5
        assert resp.market_summary == "All good"
        assert resp.risk_assessment == "Low risk"

    @patch("trading.agent.get_ollama_client")
    def test_parses_json_wrapped_in_markdown_code_block(self, mock_get_client):
        data = _sample_llm_data()
        raw = "```json\n" + json.dumps(data) + "\n```"
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": raw}}
        mock_get_client.return_value = mock_client

        resp = get_trading_decisions("ctx")
        assert len(resp.decisions) == 1
        assert resp.decisions[0].ticker == "AAPL"

    @patch("trading.agent.get_ollama_client")
    def test_parses_json_wrapped_in_generic_code_block(self, mock_get_client):
        data = _sample_llm_data()
        raw = "```\n" + json.dumps(data) + "\n```"
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": raw}}
        mock_get_client.return_value = mock_client

        resp = get_trading_decisions("ctx")
        assert len(resp.decisions) == 1

    @patch("trading.agent.get_ollama_client")
    def test_raises_valueerror_on_invalid_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "not json at all"}}
        mock_get_client.return_value = mock_client

        with pytest.raises(ValueError, match="Failed to parse LLM response"):
            get_trading_decisions("ctx")

    @patch("trading.agent.get_ollama_client")
    def test_empty_decisions_array(self, mock_get_client):
        data = _sample_llm_data(decisions=[])
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_ollama_chat_response(data)
        mock_get_client.return_value = mock_client

        resp = get_trading_decisions("ctx")
        assert resp.decisions == []
        assert resp.market_summary == "All good"

    @patch("trading.agent.get_ollama_client")
    def test_thesis_invalidations_parsed(self, mock_get_client):
        data = _sample_llm_data(
            thesis_invalidations=[
                {"thesis_id": 42, "reason": "Revenue dropped"},
                {"thesis_id": 99, "reason": "Market shifted"},
            ]
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_ollama_chat_response(data)
        mock_get_client.return_value = mock_client

        resp = get_trading_decisions("ctx")
        assert len(resp.thesis_invalidations) == 2
        assert resp.thesis_invalidations[0].thesis_id == 42
        assert resp.thesis_invalidations[1].reason == "Market shifted"

    @patch("trading.agent.get_ollama_client")
    def test_missing_optional_fields_use_defaults(self, mock_get_client):
        """If the LLM omits optional fields, defaults are applied."""
        data = {
            "decisions": [{"ticker": "TSLA"}],
            "thesis_invalidations": [],
        }
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_ollama_chat_response(data)
        mock_get_client.return_value = mock_client

        resp = get_trading_decisions("ctx")
        d = resp.decisions[0]
        assert d.action == "hold"  # default
        assert d.confidence == "low"  # default
        assert d.reasoning == ""  # default
        assert resp.market_summary == ""  # default
        assert resp.risk_assessment == ""  # default

    @patch("trading.agent.get_ollama_client")
    def test_custom_model_passed_to_chat(self, mock_get_client):
        data = _sample_llm_data()
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_ollama_chat_response(data)
        mock_get_client.return_value = mock_client

        get_trading_decisions("ctx", model="llama3:8b")
        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") == "llama3:8b"

    @patch("trading.agent.get_ollama_client")
    def test_multiple_decisions_parsed(self, mock_get_client):
        data = _sample_llm_data(decisions=[
            {"action": "buy", "ticker": "AAPL", "quantity": 5, "reasoning": "r1", "confidence": "high"},
            {"action": "sell", "ticker": "MSFT", "quantity": 3, "reasoning": "r2", "confidence": "medium"},
            {"action": "hold", "ticker": "GOOG", "quantity": None, "reasoning": "r3", "confidence": "low"},
        ])
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_ollama_chat_response(data)
        mock_get_client.return_value = mock_client

        resp = get_trading_decisions("ctx")
        assert len(resp.decisions) == 3
        assert resp.decisions[0].action == "buy"
        assert resp.decisions[1].action == "sell"
        assert resp.decisions[2].action == "hold"


# ---------------------------------------------------------------------------
# format_decisions_for_logging
# ---------------------------------------------------------------------------


class TestFormatDecisionsForLogging:
    def test_basic_response(self):
        resp = make_agent_response()
        result = format_decisions_for_logging(resp)

        assert result["market_summary"] == "Market is bullish"
        assert result["risk_assessment"] == "Low risk environment"
        assert result["decision_count"] == 1
        assert result["thesis_invalidation_count"] == 0

    def test_with_invalidations(self):
        resp = make_agent_response(
            thesis_invalidations=[
                ThesisInvalidation(thesis_id=1, reason="r1"),
                ThesisInvalidation(thesis_id=2, reason="r2"),
            ]
        )
        result = format_decisions_for_logging(resp)
        assert result["thesis_invalidation_count"] == 2

    def test_empty_decisions(self):
        resp = make_agent_response(decisions=[])
        result = format_decisions_for_logging(resp)
        assert result["decision_count"] == 0

    def test_multiple_decisions_counted(self):
        resp = make_agent_response(
            decisions=[
                make_trading_decision(ticker="AAPL"),
                make_trading_decision(ticker="MSFT"),
                make_trading_decision(ticker="GOOG"),
            ]
        )
        result = format_decisions_for_logging(resp)
        assert result["decision_count"] == 3

    def test_returns_dict(self):
        resp = make_agent_response()
        result = format_decisions_for_logging(resp)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# validate_decision
# ---------------------------------------------------------------------------


class TestValidateDecision:
    # --- hold ---
    def test_hold_always_valid(self):
        decision = make_trading_decision(action="hold", quantity=None)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("0"),
            current_price=Decimal("100"),
            positions={},
        )
        assert is_valid is True
        assert "Hold" in reason

    def test_hold_valid_even_with_no_buying_power(self):
        decision = make_trading_decision(action="hold")
        is_valid, _ = validate_decision(
            decision, Decimal("0"), Decimal("999"), {}
        )
        assert is_valid is True

    # --- buy: valid ---
    def test_buy_with_sufficient_buying_power(self):
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is True
        assert "validated" in reason.lower()

    def test_buy_exact_buying_power(self):
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
        is_valid, _ = validate_decision(
            decision,
            buying_power=Decimal("1500"),  # exactly 10 * 150
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is True

    # --- buy: invalid ---
    def test_buy_insufficient_buying_power(self):
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("100"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is False
        assert "Insufficient buying power" in reason

    def test_buy_zero_quantity(self):
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=0)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is False
        assert "positive quantity" in reason.lower()

    def test_buy_negative_quantity(self):
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=-5)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is False
        assert "positive quantity" in reason.lower()

    def test_buy_none_quantity(self):
        decision = make_trading_decision(action="buy", ticker="AAPL", quantity=None)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is False

    # --- sell: valid ---
    def test_sell_with_sufficient_shares(self):
        decision = make_trading_decision(action="sell", ticker="AAPL", quantity=5)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={"AAPL": Decimal("10")},
        )
        assert is_valid is True
        assert "validated" in reason.lower()

    def test_sell_all_shares(self):
        decision = make_trading_decision(action="sell", ticker="AAPL", quantity=10)
        is_valid, _ = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={"AAPL": Decimal("10")},
        )
        assert is_valid is True

    # --- sell: invalid ---
    def test_sell_insufficient_shares(self):
        decision = make_trading_decision(action="sell", ticker="AAPL", quantity=20)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={"AAPL": Decimal("10")},
        )
        assert is_valid is False
        assert "Insufficient shares" in reason

    def test_sell_ticker_not_held(self):
        decision = make_trading_decision(action="sell", ticker="AAPL", quantity=5)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is False
        assert "Insufficient shares" in reason

    def test_sell_zero_quantity(self):
        decision = make_trading_decision(action="sell", ticker="AAPL", quantity=0)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={"AAPL": Decimal("10")},
        )
        assert is_valid is False
        assert "positive quantity" in reason.lower()

    def test_sell_none_quantity(self):
        decision = make_trading_decision(action="sell", ticker="AAPL", quantity=None)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={"AAPL": Decimal("10")},
        )
        assert is_valid is False

    # --- unknown action ---
    def test_unknown_action(self):
        decision = make_trading_decision(action="short", ticker="AAPL", quantity=5)
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("50000"),
            current_price=Decimal("150"),
            positions={},
        )
        assert is_valid is False
        assert "Unknown action" in reason
