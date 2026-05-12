"""Tests for executor LLM integration."""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.agent import (
    AgentResponse,
    ExecutorDecision,
    ExecutorInput,
    PlaybookAction,
    TRADING_SYSTEM_PROMPT,
    get_trading_decisions,
)


class TestExecutorContracts:
    def test_executor_input_serializable(self):
        inp = ExecutorInput(
            playbook_actions=[PlaybookAction(
                id=1, ticker="AAPL", action="buy", thesis_id=1,
                reasoning="Entry hit", confidence="high",
                intent_type="invest_dollar", intent_magnitude=Decimal("500"),
                priority=1,
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
            intent_type="invest_dollar", intent_magnitude=Decimal("500"),
            reasoning="Entry hit", confidence="high",
            is_off_playbook=False,
        )
        assert d.playbook_action_id == 1
        assert d.is_off_playbook is False

    def test_off_playbook_decision(self):
        d = ExecutorDecision(
            playbook_action_id=None, ticker="NVDA", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("200"),
            reasoning="Urgent opportunity", confidence="medium",
            is_off_playbook=True,
        )
        assert d.playbook_action_id is None
        assert d.is_off_playbook is True

    def test_playbook_action_carries_signal_refs(self):
        """Phase 4: signals justify the action all the way from the strategist
        to decision_signals. The executor copies them verbatim — it does not
        invent IDs (which would be stripped by validate_signal_refs anyway)."""
        a = PlaybookAction(
            id=1, ticker="AAPL", action="buy", thesis_id=10,
            reasoning="Entry hit", confidence="high",
            intent_type="invest_dollar", intent_magnitude=Decimal("500"),
            priority=1,
            signal_refs=[{"type": "news_signal", "id": 100}],
        )
        assert a.signal_refs == [{"type": "news_signal", "id": 100}]

    def test_playbook_action_signal_refs_default_empty(self):
        a = PlaybookAction(
            id=1, ticker="AAPL", action="buy", thesis_id=None,
            reasoning="...", confidence="medium",
            intent_type="exit_full", intent_magnitude=None,
            priority=1,
        )
        assert a.signal_refs == []


class TestExecutorInput:
    def test_executor_input_has_recent_ticker_decisions(self):
        from v2.agent import ExecutorInput
        ei = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
            recent_ticker_decisions=[
                {"ticker": "GOOGL", "date": "2026-05-04", "action": "sell",
                 "quantity": 0.17, "price": 383.02, "reasoning": "near PT"},
            ],
        )
        assert len(ei.recent_ticker_decisions) == 1
        assert ei.recent_ticker_decisions[0]["ticker"] == "GOOGL"

    def test_executor_input_defaults_empty_recent_ticker_decisions(self):
        from v2.agent import ExecutorInput
        ei = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )
        assert ei.recent_ticker_decisions == []


class TestTradingSystemPrompt:
    """Phase 4: the executor prompt must instruct the LLM to copy signal_refs
    verbatim from the playbook action, not invent them."""

    def test_prompt_directs_copy_verbatim(self):
        from v2.agent import TRADING_SYSTEM_PROMPT
        # Some assertion that the prompt prescribes passthrough behaviour.
        # We're not pinning exact wording, just that the contract is documented.
        assert "verbatim" in TRADING_SYSTEM_PROMPT.lower() or \
               "copy" in TRADING_SYSTEM_PROMPT.lower() and "signal_refs" in TRADING_SYSTEM_PROMPT
        # And that the schema example uses an integer ID, not the placeholder null
        # we used pre-fix (which the LLM was copying as 0).
        assert '"id": 1234' in TRADING_SYSTEM_PROMPT or '"id": <integer>' in TRADING_SYSTEM_PROMPT

    def test_signal_refs_default_to_empty_list(self):
        d = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            intent_type="invest_dollar", intent_magnitude=Decimal("200"),
            reasoning="Test", confidence="high",
            is_off_playbook=False,
        )
        assert d.signal_refs == []


class TestExecutorPromptReversalGuidance:
    def test_prompt_describes_recent_ticker_decisions_input(self):
        from v2.agent import TRADING_SYSTEM_PROMPT
        assert "recent_ticker_decisions" in TRADING_SYSTEM_PROMPT
        assert "past 7 days" in TRADING_SYSTEM_PROMPT.lower() or \
               "7 days" in TRADING_SYSTEM_PROMPT

    def test_prompt_has_reversal_justification_rule(self):
        from v2.agent import TRADING_SYSTEM_PROMPT
        text = TRADING_SYSTEM_PROMPT.lower()
        assert "reversal" in text, "executor must be told to justify reversals"
        assert "new evidence" in text, "rule must require new evidence, not re-narration"

    def test_input_json_includes_recent_ticker_decisions(self):
        from v2.agent import ExecutorInput, get_trading_decisions
        ei = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
            recent_ticker_decisions=[
                {"ticker": "GOOGL", "date": "2026-05-04", "action": "sell",
                 "quantity": 0.17, "price": 383.02, "reasoning": "trim"},
            ],
        )

        captured = {}
        def fake_call(client, **kwargs):
            captured["messages"] = kwargs["messages"]
            resp = MagicMock()
            resp.content = [MagicMock(text='{"decisions": [], "thesis_invalidations": [], "market_summary": "", "risk_assessment": ""}')]
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock(input_tokens=1, output_tokens=1)
            return resp

        with patch("v2.agent._call_with_retry", side_effect=fake_call), \
             patch("v2.agent.get_claude_client"):
            get_trading_decisions(ei)

        sent_json = captured["messages"][0]["content"]
        assert "recent_ticker_decisions" in sent_json
        assert "GOOGL" in sent_json


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

    def test_does_not_wrap_system_prompt_in_ephemeral_cache(self):
        """Executor runs once per day; 5-minute ephemeral cache never hits.
        The system kwarg should be passed as a plain string, not a
        cache_control-wrapped list."""
        captured = {}

        def fake_call(client, **kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.content = [MagicMock(text='{"decisions":[],"thesis_invalidations":[],"market_summary":"","risk_assessment":""}')]
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock(input_tokens=10, output_tokens=10)
            return resp

        executor_input = ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "50000"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="Neutral", risk_notes="",
        )

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", side_effect=fake_call):
            get_trading_decisions(executor_input)

        assert isinstance(captured["system"], str), \
            f"system should be a plain string (cache wrapper is dead weight for once-per-day executor); got {type(captured['system'])}: {captured['system']!r}"
        assert captured["system"] == TRADING_SYSTEM_PROMPT, \
            "system kwarg should be exactly TRADING_SYSTEM_PROMPT (not a truncation or substitute)"

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

    def test_normalizes_ticker_at_parse_boundary(self):
        """T1.3: 'aapl ' (lowercase + trailing space) emitted by the LLM must
        land as 'AAPL' so SECTOR_MAP / position dict keys / exchange-side checks
        all match. Pre-fix this routed through the entire executor pipeline as
        'aapl ' and missed every downstream lookup.
        """
        from v2.risk import SECTOR_MAP

        json_response = (
            '{"decisions":[{"playbook_action_id":1,"ticker":"aapl ",'
            '"action":"buy","intent_type":"invest_dollar","intent_magnitude":500,'
            '"reasoning":"r","confidence":"high","is_off_playbook":false,'
            '"signal_refs":[],"thesis_id":null}],'
            '"thesis_invalidations":[],"market_summary":"","risk_assessment":""}'
        )
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

        assert response.decisions[0].ticker == "AAPL"
        # Sector lookup hits — proves normalization is real, not just .upper()
        assert SECTOR_MAP.get(response.decisions[0].ticker) == "tech"

    def test_normalizes_blank_ticker_to_empty_string(self):
        """Defensive: missing/None ticker stays as empty string, not 'NONE'."""
        json_response = (
            '{"decisions":[{"playbook_action_id":1,"ticker":null,'
            '"action":"hold","intent_type":null,"intent_magnitude":null,'
            '"reasoning":"r","confidence":"low","is_off_playbook":false,'
            '"signal_refs":[],"thesis_id":null}],'
            '"thesis_invalidations":[],"market_summary":"","risk_assessment":""}'
        )
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

        assert response.decisions[0].ticker == ""

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


class TestFormatDecisionsForLogging:
    def test_returns_dict(self):
        from v2.agent import format_decisions_for_logging
        response = AgentResponse(
            decisions=[ExecutorDecision(
                playbook_action_id=1, ticker="AAPL", action="buy",
                intent_type="invest_dollar", intent_magnitude=Decimal("500"),
                reasoning="Test", confidence="high",
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
        mock_cursor.fetchall.return_value = [{"id": 5}]

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "news_signal", "id": 5}])
        assert valid == [{"type": "news_signal", "id": 5}]

    def test_invalid_signal_id_stripped(self, mock_db, mock_cursor):
        """Non-existent signal ID should be stripped."""
        mock_cursor.fetchall.return_value = []

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "news_signal", "id": 99999}])
        assert valid == []

    def test_invalid_signal_type_stripped(self, mock_db, mock_cursor):
        """Unknown signal type should be stripped."""
        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "invalid_type", "id": 1}])
        assert valid == []

    def test_mixed_valid_and_invalid(self, mock_db, mock_cursor):
        """Should keep valid refs and strip invalid ones (batched)."""
        mock_cursor.fetchall.return_value = [{"id": 1}]

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
        mock_cursor.fetchall.return_value = [{"id": 3}]

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


class TestValidateSignalRefsBatch:
    def test_batch_validates_in_single_query_per_type(self, mock_db):
        """Should batch-validate all refs of same type in one query, not N+1."""
        refs = [
            {"type": "news_signal", "id": 1},
            {"type": "news_signal", "id": 2},
            {"type": "news_signal", "id": 99},  # doesn't exist
            {"type": "thesis", "id": 5},
        ]
        # news_signals query returns ids 1 and 2 (not 99)
        # theses query returns id 5
        mock_db.fetchall.side_effect = [
            [{"id": 1}, {"id": 2}],  # news_signals batch
            [{"id": 5}],              # theses batch
        ]

        from v2.agent import validate_signal_refs
        result = validate_signal_refs(refs)

        assert len(result) == 3  # 1, 2, and 5 valid; 99 stripped
        # Should have made exactly 2 queries (one per signal type), not 4
        assert mock_db.execute.call_count == 2

    def test_returns_empty_for_empty_input(self, mock_db):
        from v2.agent import validate_signal_refs
        result = validate_signal_refs([])
        assert result == []
        assert mock_db.execute.call_count == 0

    def test_strips_unknown_signal_types(self, mock_db):
        refs = [{"type": "unknown_type", "id": 1}]
        from v2.agent import validate_signal_refs
        result = validate_signal_refs(refs)
        assert result == []
        assert mock_db.execute.call_count == 0


class TestExecutorResponseTelemetry:
    """`executor_response` event captures parse outcome + schema-drift
    canary. Emitted on every parse path: success, max_tokens truncation,
    and JSONDecodeError. `unknown_*_keys` flags fields the LLM produced
    that we silently drop."""

    @staticmethod
    def _empty_executor_input():
        return ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "0"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )

    def test_emits_event_on_successful_parse(self):
        recorded = []
        json_resp = '{"decisions":[],"thesis_invalidations":[],"market_summary":"","risk_assessment":""}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_resp)]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response), \
             patch("v2.agent.record_event", side_effect=lambda **kw: recorded.append(kw)):
            get_trading_decisions(self._empty_executor_input(), session_id=42)

        events = [r for r in recorded if r["event_type"] == "executor_response"]
        assert len(events) == 1
        p = events[0]["payload"]
        assert p["parse_succeeded"] is True
        assert p["stop_reason"] == "end_turn"
        assert p["decision_count"] == 0
        assert p["unknown_top_level_keys"] == []
        assert p["unknown_decision_keys"] == []
        assert p["error"] is None

    def test_captures_unknown_top_level_key(self):
        recorded = []
        json_resp = '{"decisions":[],"thesis_invalidations":[],"market_summary":"","risk_assessment":"","experimental_score":0.5}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_resp)]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response), \
             patch("v2.agent.record_event", side_effect=lambda **kw: recorded.append(kw)):
            get_trading_decisions(self._empty_executor_input(), session_id=42)

        ev = [r for r in recorded if r["event_type"] == "executor_response"][0]
        assert ev["payload"]["unknown_top_level_keys"] == ["experimental_score"]

    def test_captures_unknown_decision_key(self):
        recorded = []
        json_resp = (
            '{"decisions":[{"ticker":"AAPL","action":"hold","reasoning":"x",'
            '"confidence":"low","confidence_calibration":0.7}],'
            '"thesis_invalidations":[],"market_summary":"","risk_assessment":""}'
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_resp)]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response), \
             patch("v2.agent.record_event", side_effect=lambda **kw: recorded.append(kw)):
            get_trading_decisions(self._empty_executor_input(), session_id=42)

        ev = [r for r in recorded if r["event_type"] == "executor_response"][0]
        assert "confidence_calibration" in ev["payload"]["unknown_decision_keys"]

    def test_emits_event_on_max_tokens(self):
        recorded = []
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="partial...")]
        mock_response.stop_reason = "max_tokens"
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=4096)

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response), \
             patch("v2.agent.record_event", side_effect=lambda **kw: recorded.append(kw)):
            with pytest.raises(ValueError):
                get_trading_decisions(self._empty_executor_input(), session_id=42)

        ev = [r for r in recorded if r["event_type"] == "executor_response"][0]
        assert ev["payload"]["parse_succeeded"] is False
        assert ev["payload"]["error"] == "max_tokens_truncation"

    def test_emits_event_on_parse_failure(self):
        recorded = []
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json {{{")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response), \
             patch("v2.agent.record_event", side_effect=lambda **kw: recorded.append(kw)):
            with pytest.raises(ValueError):
                get_trading_decisions(self._empty_executor_input(), session_id=42)

        ev = [r for r in recorded if r["event_type"] == "executor_response"][0]
        assert ev["payload"]["parse_succeeded"] is False
        assert ev["payload"]["error"].startswith("JSONDecodeError")

    def test_truncates_raw_response_text_to_4kb(self):
        recorded = []
        big = "x" * 10000
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=big)]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)

        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", return_value=mock_response), \
             patch("v2.agent.record_event", side_effect=lambda **kw: recorded.append(kw)):
            with pytest.raises(ValueError):
                get_trading_decisions(self._empty_executor_input(), session_id=42)

        ev = [r for r in recorded if r["event_type"] == "executor_response"][0]
        assert len(ev["payload"]["raw_response_text_truncated"]) <= 4096


class TestExecutorTelemetryWiring:
    """`get_trading_decisions` must pass session_id/stage_name/purpose to
    `_call_with_retry` so Phase 2's `agent_call` event carries the right
    context. `purpose='executor'` is the auditor's filter for executor-level
    truncation, parse, and latency checks."""

    def test_passes_telemetry_kwargs_to_call_with_retry(self):
        captured = {}

        def fake_call(client, **kw):
            captured.update(kw)
            resp = MagicMock()
            resp.content = [MagicMock(text='{"decisions": [], "thesis_invalidations": [], "market_summary": "", "risk_assessment": ""}')]
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock(input_tokens=1, output_tokens=1)
            return resp

        ei = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )
        with patch("v2.agent._call_with_retry", side_effect=fake_call), \
             patch("v2.agent.get_claude_client"):
            get_trading_decisions(ei, session_id=42)

        assert captured.get("session_id") == 42
        assert captured.get("stage_name") == "trading"
        assert captured.get("purpose") == "executor"


class TestDefaultExecutorModelEnvOverride:
    """ALGO_EXECUTOR_MODEL env var should override the hardcoded default
    so paper/prod can flip via .env without code changes."""

    def test_env_var_overrides_default(self, monkeypatch):
        import importlib
        import v2.agent as agent_module

        monkeypatch.setenv("ALGO_EXECUTOR_MODEL", "claude-sonnet-4-6")
        # Force re-evaluation of the module-level default
        importlib.reload(agent_module)
        assert agent_module.DEFAULT_EXECUTOR_MODEL == "claude-sonnet-4-6"

        # Clean up: reload with env unset to restore original state
        monkeypatch.delenv("ALGO_EXECUTOR_MODEL", raising=False)
        importlib.reload(agent_module)

    def test_falls_back_to_haiku_when_env_unset(self, monkeypatch):
        import importlib
        import v2.agent as agent_module

        monkeypatch.delenv("ALGO_EXECUTOR_MODEL", raising=False)
        # Force re-evaluation of the module-level default
        importlib.reload(agent_module)
        assert agent_module.DEFAULT_EXECUTOR_MODEL == "claude-haiku-4-5-20251001"

        # Already in the correct state, no need to clean up


class TestExecutorMaxTokensEnvOverride:
    """ALGO_EXECUTOR_MAX_TOKENS env var should override the default cap.
    The default is raised to 8192 to give headroom for multi-action
    playbooks; the audit module's check_executor_max_tokens_hit was
    catching real truncations at 4096."""

    def test_default_is_8192_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ALGO_EXECUTOR_MAX_TOKENS", raising=False)
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)
        assert agent_module.EXECUTOR_MAX_TOKENS == 8192

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("ALGO_EXECUTOR_MAX_TOKENS", "12000")
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)
        assert agent_module.EXECUTOR_MAX_TOKENS == 12000

    def test_max_tokens_passed_to_api_call(self, monkeypatch):
        """The configured value must reach _call_with_retry, not a stale
        constant captured at function-def time."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("ALGO_EXECUTOR_MAX_TOKENS", "9999")
        import importlib
        import v2.agent as agent_module
        importlib.reload(agent_module)

        captured = {}
        def fake_call(client, **kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.content = [MagicMock(text='{"decisions":[],"thesis_invalidations":[],"market_summary":"","risk_assessment":""}')]
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock(input_tokens=10, output_tokens=10)
            return resp

        executor_input = agent_module.ExecutorInput(
            playbook_actions=[], positions=[], account={"cash": "50000"},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="Neutral", risk_notes="",
        )
        with patch("v2.agent.get_claude_client", return_value=MagicMock()), \
             patch("v2.agent._call_with_retry", side_effect=fake_call):
            agent_module.get_trading_decisions(executor_input)
        assert captured["max_tokens"] == 9999
