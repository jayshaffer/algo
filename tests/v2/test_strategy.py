"""Tests for v2/strategy.py — strategy reflection stage (Stage 4)."""

from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from tests.v2.conftest import (
    make_strategy_state_row,
    make_strategy_rule_row,
    make_strategy_memo_row,
    make_decision_row,
)


class TestStrategyReflectionResult:
    def test_dataclass_fields(self):
        from v2.strategy import StrategyReflectionResult
        result = StrategyReflectionResult(
            rules_proposed=2,
            rules_retired=1,
            identity_updated=True,
            memo_written=True,
            input_tokens=1000,
            output_tokens=500,
            turns_used=3,
        )
        assert result.rules_proposed == 2
        assert result.rules_retired == 1
        assert result.identity_updated is True
        assert result.memo_written is True
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.turns_used == 3


class TestToolUpdateStrategyIdentity:
    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_updates_existing_identity(self, mock_get, mock_clear, mock_insert):
        from v2.strategy import tool_update_strategy_identity
        from datetime import timedelta
        mock_get.return_value = make_strategy_state_row(
            version=2, created_at=datetime.now() - timedelta(days=5),
        )
        mock_insert.return_value = 3

        result = tool_update_strategy_identity(
            identity_text="Contrarian trader",
            risk_posture="aggressive",
            sector_biases={"energy": "overweight"},
            preferred_signals=["macro"],
            avoided_signals=["legal"],
        )
        mock_clear.assert_called_once()
        mock_insert.assert_called_once()
        assert "version 3" in result

    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_bootstraps_first_identity(self, mock_get, mock_clear, mock_insert):
        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = None
        mock_insert.return_value = 1

        result = tool_update_strategy_identity(
            identity_text="New trader",
            risk_posture="conservative",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )
        mock_insert.assert_called_once()
        assert "version 1" in result


class TestToolProposeRule:
    @patch("v2.strategy.insert_strategy_rule")
    def test_creates_rule(self, mock_insert):
        from v2.strategy import tool_propose_rule
        mock_insert.return_value = 5

        result = tool_propose_rule(
            rule_text="Favor earnings signals",
            category="news_signal:earnings",
            direction="preference",
            confidence=0.7,
            supporting_evidence="62% win rate",
        )
        mock_insert.assert_called_once()
        assert "5" in result
        assert "Created rule" in result


class TestToolRetireRule:
    @patch("v2.strategy.retire_strategy_rule")
    @patch("v2.database.trading_db.get_strategy_rule")
    def test_retires_existing_rule(self, mock_get_rule, mock_retire):
        from datetime import datetime, timedelta
        from v2.strategy import tool_retire_rule, reset_session
        mock_get_rule.return_value = {
            "id": 1, "status": "active",
            "created_at": datetime.now() - timedelta(days=10),
        }
        mock_retire.return_value = True
        reset_session()
        result = tool_retire_rule(rule_id=1, reason="No longer predictive")
        assert "Retired" in result

    @patch("v2.strategy.retire_strategy_rule")
    @patch("v2.database.trading_db.get_strategy_rule")
    def test_handles_not_found(self, mock_get_rule, mock_retire):
        from v2.strategy import tool_retire_rule
        mock_get_rule.return_value = None
        mock_retire.return_value = False
        result = tool_retire_rule(rule_id=999, reason="test")
        assert "not found" in result.lower() or "Error" in result

    @patch("v2.strategy.retire_strategy_rule")
    @patch("v2.database.trading_db.get_strategy_rule")
    def test_passes_reason_to_db(self, mock_get_rule, mock_retire):
        """retire_rule should pass the reason to the database function."""
        from datetime import datetime, timedelta
        from v2.strategy import tool_retire_rule, reset_session
        mock_get_rule.return_value = {
            "id": 5, "status": "active",
            "created_at": datetime.now() - timedelta(days=10),
        }
        mock_retire.return_value = True
        reset_session()
        tool_retire_rule(rule_id=5, reason="Superseded by structural enforcement")
        mock_retire.assert_called_once_with(rule_id=5, reason="Superseded by structural enforcement")


class TestToolWriteStrategyMemo:
    @patch("v2.strategy.insert_strategy_memo")
    @patch("v2.strategy.get_current_strategy_state")
    def test_writes_memo(self, mock_state, mock_insert):
        from v2.strategy import tool_write_strategy_memo
        mock_state.return_value = make_strategy_state_row()
        mock_insert.return_value = 1

        result = tool_write_strategy_memo(
            memo_type="reflection",
            content="Session showed strong momentum plays.",
        )
        mock_insert.assert_called_once()
        assert "Memo written" in result

    @patch("v2.strategy.insert_strategy_memo")
    @patch("v2.strategy.get_current_strategy_state")
    def test_handles_no_strategy_state(self, mock_state, mock_insert):
        from v2.strategy import tool_write_strategy_memo
        mock_state.return_value = None
        mock_insert.return_value = 1

        result = tool_write_strategy_memo(
            memo_type="reflection",
            content="First session reflection.",
        )
        # strategy_state_id should be None
        call_kwargs = mock_insert.call_args
        assert call_kwargs.kwargs.get("strategy_state_id") is None or \
               call_kwargs[1].get("strategy_state_id") is None or \
               call_kwargs[0][3] is None  # positional


class TestToolGetSessionSummary:
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_returns_summary_with_decisions(self, mock_decisions, mock_attr, mock_db, mock_cursor):
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = [make_decision_row()]
        mock_attr.return_value = "Attribution data here"
        # Signal linkage query
        mock_cursor.fetchall.return_value = [
            {"decision_id": 1, "signal_type": "news_signal", "signal_category": "earnings"},
        ]

        result = tool_get_session_summary()
        assert "Decisions" in result
        assert "AAPL" in result
        assert "Attribution" in result
        assert "news_signal:earnings" in result

    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_handles_no_decisions(self, mock_decisions, mock_attr):
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = []
        mock_attr.return_value = "No data"

        result = tool_get_session_summary()
        assert "No recent decisions" in result


class TestStrategyToolDefinitions:
    def test_all_tools_defined(self):
        from v2.strategy import STRATEGY_TOOL_DEFINITIONS
        names = [d.get("name") for d in STRATEGY_TOOL_DEFINITIONS]
        assert "get_strategy_identity" in names
        assert "get_strategy_rules" in names
        assert "get_strategy_history" in names
        assert "get_session_summary" in names
        assert "update_strategy_identity" in names
        assert "propose_rule" in names
        assert "retire_rule" in names
        assert "write_strategy_memo" in names

    def test_handlers_match_definitions(self):
        from v2.strategy import STRATEGY_TOOL_DEFINITIONS, STRATEGY_TOOL_HANDLERS
        for defn in STRATEGY_TOOL_DEFINITIONS:
            name = defn["name"]
            assert name in STRATEGY_TOOL_HANDLERS, f"Missing handler for {name}"


class TestCountActions:
    def test_counts_proposed_rules(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Created rule ID 1: test"},
                {"type": "tool_result", "content": "Created rule ID 2: test2"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert proposed == 2

    def test_counts_retired_rules(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Retired rule ID 1. Reason: stale"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert retired == 1

    def test_detects_identity_update(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Strategy identity updated to version 2 (ID: 3)"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert identity_updated is True

    def test_detects_memo_written(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Memo written (ID: 1)"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert memo_written is True

    def test_empty_messages(self):
        from v2.strategy import _count_actions
        proposed, retired, identity_updated, memo_written = _count_actions([])
        assert proposed == 0
        assert retired == 0
        assert identity_updated is False
        assert memo_written is False


class TestRunStrategyReflection:
    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    def test_returns_reflection_result(self, mock_loop, mock_client):
        from v2.strategy import run_strategy_reflection, StrategyReflectionResult
        from v2.claude_client import AgenticLoopResult

        mock_loop.return_value = AgenticLoopResult(
            messages=[
                {"role": "user", "content": "Begin reflection"},
                {"role": "assistant", "content": [MagicMock(text="Done reflecting", type="text")]},
            ],
            turns_used=2,
            stop_reason="end_turn",
            input_tokens=1000,
            output_tokens=500,
        )

        result = run_strategy_reflection()
        assert isinstance(result, StrategyReflectionResult)
        assert result.turns_used == 2
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    def test_passes_system_prompt_and_tools(self, mock_loop, mock_client):
        from v2.strategy import run_strategy_reflection, STRATEGY_REFLECTION_SYSTEM, STRATEGY_TOOL_DEFINITIONS, STRATEGY_TOOL_HANDLERS
        from v2.claude_client import AgenticLoopResult

        mock_loop.return_value = AgenticLoopResult(
            messages=[],
            turns_used=1,
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
        )

        run_strategy_reflection(model="claude-opus-4-6", max_turns=5)

        call_kwargs = mock_loop.call_args
        assert call_kwargs.kwargs["system"] == STRATEGY_REFLECTION_SYSTEM
        assert call_kwargs.kwargs["tools"] == STRATEGY_TOOL_DEFINITIONS
        assert call_kwargs.kwargs["tool_handlers"] == STRATEGY_TOOL_HANDLERS
        assert call_kwargs.kwargs["max_turns"] == 5
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"

    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    def test_counts_actions_from_messages(self, mock_loop, mock_client):
        from v2.strategy import run_strategy_reflection
        from v2.claude_client import AgenticLoopResult

        mock_loop.return_value = AgenticLoopResult(
            messages=[
                {"role": "user", "content": [
                    {"type": "tool_result", "content": "Created rule ID 1: test"},
                    {"type": "tool_result", "content": "Retired rule ID 2. Reason: stale"},
                    {"type": "tool_result", "content": "Strategy identity updated to version 2 (ID: 3)"},
                    {"type": "tool_result", "content": "Memo written (ID: 1)"},
                ]},
                {"role": "assistant", "content": [MagicMock(text="Done", type="text")]},
            ],
            turns_used=3,
            stop_reason="end_turn",
            input_tokens=2000,
            output_tokens=800,
        )

        result = run_strategy_reflection()
        assert result.rules_proposed == 1
        assert result.rules_retired == 1
        assert result.identity_updated is True
        assert result.memo_written is True


class TestRuleTenureGuard:
    def test_cannot_retire_rule_before_min_tenure(self, mock_db):
        """Rules must be active for at least 5 days before retirement."""
        from datetime import datetime, timedelta
        mock_db.fetchone.return_value = {
            "id": 1, "rule_text": "Test rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=2),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        from v2.strategy import tool_retire_rule
        result = tool_retire_rule(rule_id=1, reason="Not working")
        assert "too new" in result.lower() or "minimum tenure" in result.lower()

    def test_can_retire_rule_after_min_tenure(self, mock_db):
        """Rules active for >= 5 days can be retired normally."""
        from datetime import datetime, timedelta
        mock_db.fetchone.return_value = {
            "id": 1, "rule_text": "Old rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=6),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        mock_db.rowcount = 1
        from v2.strategy import tool_retire_rule, reset_session
        reset_session()
        result = tool_retire_rule(rule_id=1, reason="Data no longer supports it")
        assert "retired" in result.lower()


class TestRetirementCap:
    def test_max_retirements_per_session(self, mock_db):
        """At most 2 rules can be retired per session to prevent mass purges."""
        from datetime import datetime, timedelta
        old_rule = {
            "id": 1, "rule_text": "Old rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=30),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        mock_db.fetchone.return_value = old_rule
        mock_db.rowcount = 1
        from v2.strategy import tool_retire_rule, reset_session
        reset_session()
        r1 = tool_retire_rule(rule_id=1, reason="Data changed")
        assert "retired" in r1.lower()
        r2 = tool_retire_rule(rule_id=2, reason="No longer valid")
        assert "retired" in r2.lower()
        r3 = tool_retire_rule(rule_id=3, reason="Also bad")
        assert "limit" in r3.lower() or "maximum" in r3.lower()


class TestIdentityUpdateGuard:
    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_warns_if_recently_updated(self, mock_get, mock_clear, mock_insert):
        """Should return warning if identity was updated within 3 days."""
        from v2.strategy import tool_update_strategy_identity
        from datetime import timezone
        mock_get.return_value = make_strategy_state_row(
            version=5,
            created_at=datetime.now(timezone.utc),
        )

        result = tool_update_strategy_identity(
            identity_text="New identity",
            risk_posture="aggressive",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )

        assert "Warning" in result
        mock_clear.assert_not_called()
        mock_insert.assert_not_called()

    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_allows_update_if_not_recent(self, mock_get, mock_clear, mock_insert):
        """Should allow update if identity hasn't been updated in >3 days."""
        from v2.strategy import tool_update_strategy_identity
        from datetime import timedelta, timezone
        mock_get.return_value = make_strategy_state_row(
            version=5,
            created_at=datetime.now(timezone.utc) - timedelta(days=5),
        )
        mock_insert.return_value = 6

        result = tool_update_strategy_identity(
            identity_text="New identity",
            risk_posture="moderate",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )

        assert "version 6" in result
        mock_clear.assert_called_once()
        mock_insert.assert_called_once()
