"""Tests for executor LLM integration."""
from decimal import Decimal
from unittest.mock import patch, MagicMock

from v2.agent import (
    ExecutorInput,
    ExecutorDecision,
    PlaybookAction,
    AgentResponse,
    ThesisInvalidation,
    get_trading_decisions,
    validate_decision,
    DEFAULT_EXECUTOR_MODEL,
)


class TestExecutorContracts:
    def test_executor_input_serializable(self):
        inp = ExecutorInput(
            playbook_actions=[PlaybookAction(
                id=1, ticker="AAPL", action="buy", thesis_id=1,
                reasoning="Entry hit", confidence="high",
                max_quantity=Decimal("5"), priority=1,
            )],
            positions=[{"ticker": "MSFT", "shares": "10"}],
            account={"cash": "50000", "buying_power": "50000"},
            attribution_summary={"news_signal:earnings": {"win_rate_7d": 0.6, "sample_size": 20}},
            recent_outcomes=[],
            market_outlook="Bullish",
            risk_notes="Fed meeting tomorrow",
        )
        assert inp.playbook_actions[0].ticker == "AAPL"

    def test_executor_decision_has_playbook_action_id(self):
        d = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=2.5, reasoning="Entry hit", confidence="high",
            is_off_playbook=False,
        )
        assert d.playbook_action_id == 1
        assert d.is_off_playbook is False

    def test_off_playbook_decision(self):
        d = ExecutorDecision(
            playbook_action_id=None, ticker="NVDA", action="buy",
            quantity=1.0, reasoning="Urgent opportunity", confidence="medium",
            is_off_playbook=True,
        )
        assert d.playbook_action_id is None
        assert d.is_off_playbook is True

    def test_signal_refs_default_to_empty_list(self):
        d = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=1.0, reasoning="Test", confidence="high",
            is_off_playbook=False,
        )
        assert d.signal_refs == []


class TestGetTradingDecisions:
    def test_calls_haiku_with_structured_input(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"decisions":[],"thesis_invalidations":[],"market_summary":"Quiet day","risk_assessment":"Low"}')]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        executor_input = ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "50000"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="Neutral", risk_notes="",
        )

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response):
            response = get_trading_decisions(executor_input)

        assert isinstance(response, AgentResponse)
        assert response.market_summary == "Quiet day"

    def test_parses_decisions_with_playbook_action_id(self):
        json_response = '{"decisions":[{"playbook_action_id":1,"ticker":"AAPL","action":"buy","quantity":2.5,"reasoning":"Entry hit","confidence":"high","is_off_playbook":false,"signal_refs":[{"type":"news_signal","id":5}]}],"thesis_invalidations":[],"market_summary":"Active day","risk_assessment":"Medium"}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_response)]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        executor_input = ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "50000"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="Neutral", risk_notes="",
        )

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response):
            response = get_trading_decisions(executor_input)

        assert len(response.decisions) == 1
        d = response.decisions[0]
        assert d.playbook_action_id == 1
        assert d.ticker == "AAPL"
        assert d.is_off_playbook is False
        assert len(d.signal_refs) == 1

    def test_raises_on_max_tokens(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"decisions":[]')]
        mock_response.stop_reason = "max_tokens"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=4096)

        executor_input = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response):
            import pytest
            with pytest.raises(ValueError, match="truncated"):
                get_trading_decisions(executor_input)

    def test_strips_markdown_fences(self):
        json_response = '```json\n{"decisions":[],"thesis_invalidations":[],"market_summary":"Test","risk_assessment":"Low"}\n```'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_response)]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        executor_input = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response):
            response = get_trading_decisions(executor_input)
        assert response.market_summary == "Test"


class TestValidateDecision:
    def test_buy_valid(self):
        d = ExecutorDecision(playbook_action_id=1, ticker="AAPL", action="buy",
                             quantity=2.0, reasoning="Buy", confidence="high",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {})
        assert valid is True

    def test_buy_exceeds_buying_power(self):
        d = ExecutorDecision(playbook_action_id=1, ticker="AAPL", action="buy",
                             quantity=1000, reasoning="Buy", confidence="high",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("100"), Decimal("150"), {})
        assert valid is False
        assert "buying power" in reason.lower()

    def test_sell_exceeds_held_shares(self):
        d = ExecutorDecision(playbook_action_id=1, ticker="AAPL", action="sell",
                             quantity=100, reasoning="Sell", confidence="high",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {"AAPL": Decimal("10")})
        assert valid is False
        assert "shares" in reason.lower()

    def test_hold_always_valid(self):
        d = ExecutorDecision(playbook_action_id=None, ticker="AAPL", action="hold",
                             quantity=None, reasoning="Wait", confidence="low",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {})
        assert valid is True

    def test_unknown_action_invalid(self):
        d = ExecutorDecision(playbook_action_id=None, ticker="AAPL", action="short",
                             quantity=5, reasoning="Test", confidence="low",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {})
        assert valid is False

    def test_buy_no_quantity_invalid(self):
        d = ExecutorDecision(playbook_action_id=1, ticker="AAPL", action="buy",
                             quantity=None, reasoning="Buy", confidence="high",
                             is_off_playbook=False)
        valid, reason = validate_decision(d, Decimal("50000"), Decimal("150"), {})
        assert valid is False


class TestFormatDecisionsForLogging:
    def test_returns_dict(self):
        from v2.agent import format_decisions_for_logging
        response = AgentResponse(
            decisions=[ExecutorDecision(
                playbook_action_id=1, ticker="AAPL", action="buy",
                quantity=2.5, reasoning="Test", confidence="high",
                is_off_playbook=False,
            )],
            thesis_invalidations=[],
            market_summary="Test summary",
            risk_assessment="Low",
        )
        result = format_decisions_for_logging(response)
        assert result["decision_count"] == 1
        assert result["market_summary"] == "Test summary"
