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


class TestValidateSignalRefs:
    def test_valid_news_signal_passes(self, mock_db, mock_cursor):
        """Existing news_signal ID should pass validation."""
        mock_cursor.fetchone.return_value = {"id": 5}

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "news_signal", "id": 5}])
        assert valid == [{"type": "news_signal", "id": 5}]

    def test_invalid_signal_id_stripped(self, mock_db, mock_cursor):
        """Non-existent signal ID should be stripped."""
        mock_cursor.fetchone.return_value = None

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "news_signal", "id": 99999}])
        assert valid == []

    def test_invalid_signal_type_stripped(self, mock_db, mock_cursor):
        """Unknown signal type should be stripped."""
        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "invalid_type", "id": 1}])
        assert valid == []

    def test_mixed_valid_and_invalid(self, mock_db, mock_cursor):
        """Should keep valid refs and strip invalid ones."""
        call_count = [0]
        def mock_fetchone():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"id": 1}
            return None

        mock_cursor.fetchone.side_effect = mock_fetchone

        from v2.agent import validate_signal_refs
        refs = [
            {"type": "news_signal", "id": 1},
            {"type": "news_signal", "id": 99999},
        ]
        valid = validate_signal_refs(refs)
        assert len(valid) == 1
        assert valid[0]["id"] == 1

    def test_empty_refs_returns_empty(self, mock_db, mock_cursor):
        from v2.agent import validate_signal_refs
        assert validate_signal_refs([]) == []

    def test_thesis_type_validated(self, mock_db, mock_cursor):
        """thesis signal type should also be validated."""
        mock_cursor.fetchone.return_value = {"id": 3}

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "thesis", "id": 3}])
        assert valid == [{"type": "thesis", "id": 3}]


class TestExecutorInputPrices:
    def test_executor_input_has_current_prices(self):
        inp = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
            current_prices={"AAPL": Decimal("175.50")},
        )
        assert inp.current_prices["AAPL"] == Decimal("175.50")

    def test_executor_input_defaults_empty_prices(self):
        inp = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )
        assert inp.current_prices == {}


class TestValidateDecisionTotalExposure:
    def test_buy_rejected_when_total_exposure_exceeds_cap(self):
        from tests.v2.conftest import make_trading_decision
        decision = make_trading_decision(ticker="AAPL", action="buy", quantity=3.0)
        positions = {"AAPL": Decimal("5.33")}  # 5.33 * 150 = ~$800 = 8%
        is_valid, reason = validate_decision(
            decision, buying_power=Decimal("5000"), current_price=Decimal("150"),
            positions=positions, portfolio_value=Decimal("10000"),
        )
        assert not is_valid
        assert "exposure" in reason.lower() or "position" in reason.lower()

    def test_buy_allowed_when_total_exposure_under_cap(self):
        from tests.v2.conftest import make_trading_decision
        decision = make_trading_decision(ticker="AAPL", action="buy", quantity=1.0)
        positions = {"AAPL": Decimal("3.33")}  # 3.33 * 150 = ~$500 = 5%
        is_valid, reason = validate_decision(
            decision, buying_power=Decimal("5000"), current_price=Decimal("150"),
            positions=positions, portfolio_value=Decimal("10000"),
        )
        assert is_valid

    def test_buy_new_ticker_still_uses_new_cost_only(self):
        from tests.v2.conftest import make_trading_decision
        decision = make_trading_decision(ticker="MSFT", action="buy", quantity=2.0)
        positions = {"AAPL": Decimal("10")}
        is_valid, reason = validate_decision(
            decision, buying_power=Decimal("5000"), current_price=Decimal("200"),
            positions=positions, portfolio_value=Decimal("10000"),
        )
        assert is_valid  # $400 / $10000 = 4% < 10%


class TestValidateDecisionPendingSells:
    def test_sell_rejected_when_pending_orders_consume_shares(self):
        from tests.v2.conftest import make_trading_decision
        decision = make_trading_decision(ticker="AAPL", action="sell", quantity=8.0)
        positions = {"AAPL": Decimal("10")}
        open_sell_orders = {"AAPL": Decimal("5")}
        is_valid, reason = validate_decision(
            decision, buying_power=Decimal("5000"), current_price=Decimal("150"),
            positions=positions, portfolio_value=Decimal("10000"),
            open_sell_orders=open_sell_orders,
        )
        assert not is_valid
        assert "pending" in reason.lower() or "available" in reason.lower()

    def test_sell_allowed_after_accounting_for_pending(self):
        from tests.v2.conftest import make_trading_decision
        decision = make_trading_decision(ticker="AAPL", action="sell", quantity=4.0)
        positions = {"AAPL": Decimal("10")}
        open_sell_orders = {"AAPL": Decimal("5")}
        is_valid, reason = validate_decision(
            decision, buying_power=Decimal("5000"), current_price=Decimal("150"),
            positions=positions, portfolio_value=Decimal("10000"),
            open_sell_orders=open_sell_orders,
        )
        assert is_valid

    def test_sell_works_with_no_pending_orders(self):
        from tests.v2.conftest import make_trading_decision
        decision = make_trading_decision(ticker="AAPL", action="sell", quantity=5.0)
        positions = {"AAPL": Decimal("10")}
        is_valid, reason = validate_decision(
            decision, buying_power=Decimal("5000"), current_price=Decimal("150"),
            positions=positions, portfolio_value=Decimal("10000"),
        )
        assert is_valid
