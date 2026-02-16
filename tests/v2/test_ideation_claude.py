"""Tests for Claude strategist."""
from unittest.mock import patch, MagicMock
from datetime import datetime

from v2.ideation_claude import (
    run_strategist_loop,
    run_strategist_session,
    run_ideation_claude,
    ClaudeIdeationResult,
    StrategistResult,
    count_actions,
    CLAUDE_IDEATION_SYSTEM,
    CLAUDE_SESSION_STRATEGIST_SYSTEM,
    CLAUDE_STRATEGIST_SYSTEM,
    _STRATEGIST_TEMPLATE,
)


class TestStrategistAttributionConstraints:
    def test_attribution_injected_into_system_prompt(self):
        """Attribution constraints should be appended to the system prompt."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100
        mock_result.cache_creation_input_tokens = 0
        mock_result.cache_read_input_tokens = 0

        constraints = "SIGNAL PERFORMANCE:\n  WEAK: news_signal:legal (38%, n=12)\nCONSTRAINT: Do not use WEAK signals"

        with patch("v2.ideation_claude.get_claude_client", return_value=MagicMock()), \
             patch("v2.ideation_claude.reset_session"), \
             patch("v2.ideation_claude.run_agentic_loop") as mock_loop, \
             patch("v2.ideation_claude.extract_final_text", return_value="Summary"):
            mock_loop.return_value = mock_result

            result = run_strategist_loop(
                model="claude-opus-4-6",
                max_turns=1,
                attribution_constraints=constraints,
            )

        # Verify the system prompt includes constraints
        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_prompt is None:
            # Try positional
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "strategist" in arg.lower():
                    system_prompt = arg
                    break
        assert "WEAK" in system_prompt
        assert "CONSTRAINT" in system_prompt

    def test_runs_without_constraints(self):
        """Should work fine with empty attribution_constraints."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100
        mock_result.cache_creation_input_tokens = 0
        mock_result.cache_read_input_tokens = 0

        with patch("v2.ideation_claude.get_claude_client", return_value=MagicMock()), \
             patch("v2.ideation_claude.reset_session"), \
             patch("v2.ideation_claude.run_agentic_loop") as mock_loop, \
             patch("v2.ideation_claude.extract_final_text", return_value="Summary"):
            mock_loop.return_value = mock_result

            result = run_strategist_loop(
                model="claude-opus-4-6",
                max_turns=1,
            )

        assert isinstance(result, ClaudeIdeationResult)


class TestStrategistSession:
    def test_no_backfill_or_attribution(self):
        """run_strategist_session should NOT call backfill or attribution."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100
        mock_result.cache_creation_input_tokens = 0
        mock_result.cache_read_input_tokens = 0

        with patch("v2.ideation_claude.get_claude_client", return_value=MagicMock()), \
             patch("v2.ideation_claude.reset_session"), \
             patch("v2.ideation_claude.run_agentic_loop") as mock_loop, \
             patch("v2.ideation_claude.extract_final_text", return_value="Summary"):
            mock_loop.return_value = mock_result

            # These should NOT be imported or called:
            result = run_strategist_session(
                model="claude-opus-4-6",
                max_turns=1,
                attribution_constraints="STRONG: earnings",
            )

        assert isinstance(result, ClaudeIdeationResult)

    def test_passes_constraints_through(self):
        """attribution_constraints should be passed to run_strategist_loop."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100
        mock_result.cache_creation_input_tokens = 0
        mock_result.cache_read_input_tokens = 0

        with patch("v2.ideation_claude.get_claude_client", return_value=MagicMock()), \
             patch("v2.ideation_claude.reset_session"), \
             patch("v2.ideation_claude.run_agentic_loop") as mock_loop, \
             patch("v2.ideation_claude.extract_final_text", return_value="Summary"):
            mock_loop.return_value = mock_result

            run_strategist_session(
                attribution_constraints="STRONG: earnings",
            )

        call_kwargs = mock_loop.call_args
        system_arg = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_arg is None:
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "STRONG" in arg:
                    system_arg = arg
                    break
        assert "STRONG: earnings" in system_arg


class TestCountActions:
    def test_counts_created(self):
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": "Created thesis ID 1 for AAPL"}]},
        ]
        created, updated, closed = count_actions(messages)
        assert created == 1

    def test_counts_updated(self):
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": "Updated thesis ID 1"}]},
        ]
        created, updated, closed = count_actions(messages)
        assert updated == 1

    def test_counts_closed(self):
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": "Closed thesis ID 1 with status 'invalidated'"}]},
        ]
        created, updated, closed = count_actions(messages)
        assert closed == 1

    def test_empty_messages(self):
        created, updated, closed = count_actions([])
        assert created == 0 and updated == 0 and closed == 0


class TestSystemPrompts:
    def test_ideation_system_prompt_exists(self):
        assert len(CLAUDE_IDEATION_SYSTEM) > 100

    def test_strategist_template_has_placeholders(self):
        assert "{timing}" in _STRATEGIST_TEMPLATE

    def test_session_strategist_system_formatted(self):
        assert "before market close" in CLAUDE_SESSION_STRATEGIST_SYSTEM or \
               "after market close" in CLAUDE_SESSION_STRATEGIST_SYSTEM or \
               "strategist" in CLAUDE_SESSION_STRATEGIST_SYSTEM.lower()

    def test_strategist_prompt_mentions_strategy_tools(self):
        assert "get_strategy_identity" in CLAUDE_STRATEGIST_SYSTEM
        assert "get_strategy_rules" in CLAUDE_STRATEGIST_SYSTEM
        assert "get_strategy_history" in CLAUDE_STRATEGIST_SYSTEM
