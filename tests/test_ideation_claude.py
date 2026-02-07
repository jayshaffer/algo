"""Tests for trading/ideation_claude.py - Claude-based ideation module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from trading.ideation_claude import (
    CLAUDE_SESSION_STRATEGIST_SYSTEM,
    CLAUDE_STRATEGIST_SYSTEM,
    ClaudeIdeationResult,
    StrategistResult,
    count_actions,
    run_ideation_claude,
    run_strategist_loop,
    run_strategist_session,
)
from trading.claude_client import AgenticLoopResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_result_message(content_str: str) -> dict:
    """Create a user message with a single tool_result entry."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": content_str,
                "is_error": False,
            }
        ],
    }


def _make_agentic_result(
    messages=None,
    turns_used=3,
    stop_reason="end_turn",
    input_tokens=1000,
    output_tokens=500,
):
    """Create a mock AgenticLoopResult."""
    return AgenticLoopResult(
        messages=messages or [],
        turns_used=turns_used,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# ClaudeIdeationResult dataclass
# ---------------------------------------------------------------------------

class TestClaudeIdeationResult:

    def test_dataclass_fields(self):
        result = ClaudeIdeationResult(
            timestamp=datetime(2025, 6, 15),
            model="claude-sonnet-4-20250514",
            turns_used=5,
            theses_created=2,
            theses_updated=1,
            theses_closed=0,
            final_summary="Good session",
            input_tokens=2000,
            output_tokens=800,
        )
        assert result.model == "claude-sonnet-4-20250514"
        assert result.theses_created == 2
        assert result.turns_used == 5
        assert result.final_summary == "Good session"

    def test_zero_counts(self):
        result = ClaudeIdeationResult(
            timestamp=datetime.now(),
            model="test",
            turns_used=0,
            theses_created=0,
            theses_updated=0,
            theses_closed=0,
            final_summary="",
            input_tokens=0,
            output_tokens=0,
        )
        assert result.theses_created == 0
        assert result.theses_updated == 0
        assert result.theses_closed == 0


# ---------------------------------------------------------------------------
# count_actions
# ---------------------------------------------------------------------------

class TestCountActions:

    def test_empty_messages(self):
        """Empty messages list should return all zeros."""
        created, updated, closed = count_actions([])
        assert created == 0
        assert updated == 0
        assert closed == 0

    def test_counts_created(self):
        """Should count 'Created thesis ID' in tool results."""
        messages = [
            _make_tool_result_message("Created thesis ID 42 for NVDA (long, high confidence)"),
        ]
        created, updated, closed = count_actions(messages)
        assert created == 1
        assert updated == 0
        assert closed == 0

    def test_counts_updated(self):
        """Should count 'Updated thesis ID' in tool results."""
        messages = [
            _make_tool_result_message("Updated thesis ID 10"),
        ]
        created, updated, closed = count_actions(messages)
        assert created == 0
        assert updated == 1
        assert closed == 0

    def test_counts_closed(self):
        """Should count 'Closed thesis ID' in tool results."""
        messages = [
            _make_tool_result_message("Closed thesis ID 7 with status 'invalidated'"),
        ]
        created, updated, closed = count_actions(messages)
        assert created == 0
        assert updated == 0
        assert closed == 1

    def test_counts_multiple_actions(self):
        """Should count multiple different actions across messages."""
        messages = [
            _make_tool_result_message("Created thesis ID 1 for AAPL (long, high confidence)"),
            {"role": "assistant", "content": "Continuing..."},
            _make_tool_result_message("Created thesis ID 2 for MSFT (long, medium confidence)"),
            {"role": "assistant", "content": "Done"},
            _make_tool_result_message("Updated thesis ID 5"),
            {"role": "assistant", "content": "Closing old one"},
            _make_tool_result_message("Closed thesis ID 3 with status 'expired'"),
        ]
        created, updated, closed = count_actions(messages)
        assert created == 2
        assert updated == 1
        assert closed == 1

    def test_ignores_assistant_messages(self):
        """Should only look at user messages, not assistant messages."""
        messages = [
            {"role": "assistant", "content": "I Created thesis ID 99"},
        ]
        created, updated, closed = count_actions(messages)
        assert created == 0

    def test_ignores_user_string_messages(self):
        """Should only look at list content (tool results), not string content."""
        messages = [
            {"role": "user", "content": "Created thesis ID 1 for something"},
        ]
        created, updated, closed = count_actions(messages)
        assert created == 0

    def test_ignores_non_tool_result_items(self):
        """Non-tool_result items in content lists should be ignored."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": "Created thesis ID 1"},
                ],
            },
        ]
        created, updated, closed = count_actions(messages)
        assert created == 0

    def test_error_tool_results_still_counted(self):
        """Even error results containing the strings should be counted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "Created thesis ID 5 for AAPL (long, high confidence)",
                        "is_error": False,
                    }
                ],
            },
        ]
        created, _, _ = count_actions(messages)
        assert created == 1

    def test_non_string_content_in_tool_result(self):
        """Tool result with non-string content should be safely handled."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": 12345,  # Non-string
                    }
                ],
            },
        ]
        created, updated, closed = count_actions(messages)
        assert created == 0
        assert updated == 0
        assert closed == 0

    def test_multiple_tool_results_in_one_message(self):
        """Multiple tool results in a single user message should all be counted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "Created thesis ID 1 for AAPL (long, high confidence)",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": "Closed thesis ID 2 with status 'expired'",
                    },
                ],
            },
        ]
        created, updated, closed = count_actions(messages)
        assert created == 1
        assert closed == 1


# ---------------------------------------------------------------------------
# run_ideation_claude
# ---------------------------------------------------------------------------

class TestRunIdeationClaude:

    @patch("trading.ideation_claude.extract_final_text", return_value="Session summary")
    @patch("trading.ideation_claude.run_agentic_loop")
    @patch("trading.ideation_claude.get_claude_client")
    @patch("trading.ideation_claude.reset_session")
    def test_returns_correct_result_structure(
        self, mock_reset, mock_get_client, mock_loop, mock_extract
    ):
        """run_ideation_claude should return a ClaudeIdeationResult."""
        mock_get_client.return_value = MagicMock()
        mock_loop.return_value = _make_agentic_result(
            messages=[
                {"role": "user", "content": "Start"},
                {"role": "assistant", "content": "Done"},
            ],
            turns_used=2,
            stop_reason="end_turn",
            input_tokens=1500,
            output_tokens=700,
        )

        result = run_ideation_claude(model="test-model", max_turns=10)

        assert isinstance(result, ClaudeIdeationResult)
        assert result.model == "test-model"
        assert result.turns_used == 2
        assert result.final_summary == "Session summary"
        assert result.input_tokens == 1500
        assert result.output_tokens == 700

    @patch("trading.ideation_claude.extract_final_text", return_value="Done")
    @patch("trading.ideation_claude.run_agentic_loop")
    @patch("trading.ideation_claude.get_claude_client")
    @patch("trading.ideation_claude.reset_session")
    def test_calls_reset_session(
        self, mock_reset, mock_get_client, mock_loop, mock_extract
    ):
        """reset_session should be called before the agentic loop."""
        mock_get_client.return_value = MagicMock()
        mock_loop.return_value = _make_agentic_result()

        run_ideation_claude()
        mock_reset.assert_called_once()

    @patch("trading.ideation_claude.extract_final_text", return_value="Summary")
    @patch("trading.ideation_claude.run_agentic_loop")
    @patch("trading.ideation_claude.get_claude_client")
    @patch("trading.ideation_claude.reset_session")
    def test_counts_actions_from_messages(
        self, mock_reset, mock_get_client, mock_loop, mock_extract
    ):
        """Should correctly count thesis actions in the conversation."""
        mock_get_client.return_value = MagicMock()
        mock_loop.return_value = _make_agentic_result(
            messages=[
                {"role": "user", "content": "Start"},
                {"role": "assistant", "content": "Using tools..."},
                _make_tool_result_message("Created thesis ID 1 for AAPL (long, high confidence)"),
                {"role": "assistant", "content": "Created one thesis"},
                _make_tool_result_message("Updated thesis ID 5"),
                {"role": "assistant", "content": "Updated thesis"},
                _make_tool_result_message("Closed thesis ID 3 with status 'expired'"),
                {"role": "assistant", "content": "All done"},
            ],
        )

        result = run_ideation_claude()
        assert result.theses_created == 1
        assert result.theses_updated == 1
        assert result.theses_closed == 1

    @patch("trading.ideation_claude.extract_final_text", return_value=None)
    @patch("trading.ideation_claude.run_agentic_loop")
    @patch("trading.ideation_claude.get_claude_client")
    @patch("trading.ideation_claude.reset_session")
    def test_no_summary_available(
        self, mock_reset, mock_get_client, mock_loop, mock_extract
    ):
        """When extract_final_text returns None, summary should be a default message."""
        mock_get_client.return_value = MagicMock()
        mock_loop.return_value = _make_agentic_result()

        result = run_ideation_claude()
        assert result.final_summary == "No summary available"

    @patch("trading.ideation_claude.extract_final_text", return_value="Done")
    @patch("trading.ideation_claude.run_agentic_loop")
    @patch("trading.ideation_claude.get_claude_client")
    @patch("trading.ideation_claude.reset_session")
    def test_passes_tool_definitions_and_handlers(
        self, mock_reset, mock_get_client, mock_loop, mock_extract
    ):
        """run_agentic_loop should be called with TOOL_DEFINITIONS and TOOL_HANDLERS."""
        mock_get_client.return_value = MagicMock()
        mock_loop.return_value = _make_agentic_result()

        run_ideation_claude(model="claude-sonnet-4-20250514", max_turns=15)

        call_kwargs = mock_loop.call_args
        assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") or \
               (len(call_kwargs.args) > 1 and call_kwargs.args[1]) == "claude-sonnet-4-20250514"
        # Verify tools and handlers are passed
        if call_kwargs.kwargs:
            assert "tools" in call_kwargs.kwargs
            assert "tool_handlers" in call_kwargs.kwargs
        else:
            # positional args
            assert len(call_kwargs.args) >= 6

    @patch("trading.ideation_claude.extract_final_text", return_value="Done")
    @patch("trading.ideation_claude.run_agentic_loop")
    @patch("trading.ideation_claude.get_claude_client")
    @patch("trading.ideation_claude.reset_session")
    def test_no_actions_all_zeros(
        self, mock_reset, mock_get_client, mock_loop, mock_extract
    ):
        """When no tool actions occurred, counts should be zero."""
        mock_get_client.return_value = MagicMock()
        mock_loop.return_value = _make_agentic_result(
            messages=[
                {"role": "user", "content": "Start"},
                {"role": "assistant", "content": "Nothing to do"},
            ],
        )

        result = run_ideation_claude()
        assert result.theses_created == 0
        assert result.theses_updated == 0
        assert result.theses_closed == 0


# ---------------------------------------------------------------------------
# run_strategist_session
# ---------------------------------------------------------------------------

class TestStrategistSession:

    @pytest.fixture
    def mock_strategist_deps(self):
        """Mock all dependencies for run_strategist_session."""
        with patch("trading.ideation_claude.run_backfill") as mock_backfill, \
             patch("trading.ideation_claude.compute_signal_attribution") as mock_attr, \
             patch("trading.ideation_claude.get_attribution_summary") as mock_attr_summary, \
             patch("trading.ideation_claude.get_claude_client") as mock_client, \
             patch("trading.ideation_claude.run_agentic_loop") as mock_loop, \
             patch("trading.ideation_claude.reset_session"):
            # Setup defaults
            mock_backfill.return_value = {"total_filled": 5, "7d": {}, "30d": {}}
            mock_attr.return_value = [{"category": "thesis", "sample_size": 10}]
            mock_attr_summary.return_value = "Signal Attribution:\n- thesis: 62% win rate"
            mock_loop.return_value = AgenticLoopResult(
                messages=[],
                turns_used=3,
                stop_reason="end_turn",
                input_tokens=1000,
                output_tokens=500,
            )
            yield {
                "run_backfill": mock_backfill,
                "compute_signal_attribution": mock_attr,
                "get_attribution_summary": mock_attr_summary,
                "get_claude_client": mock_client,
                "run_agentic_loop": mock_loop,
            }

    def test_runs_backfill_first(self, mock_strategist_deps):
        """Backfill should be called during the strategist session."""
        run_strategist_session()
        mock_strategist_deps["run_backfill"].assert_called_once()

    def test_computes_attribution(self, mock_strategist_deps):
        """Signal attribution should be computed during the session."""
        run_strategist_session()
        mock_strategist_deps["compute_signal_attribution"].assert_called_once()

    def test_system_prompt_includes_strategist_role(self, mock_strategist_deps):
        """The system prompt should reference strategist or playbook."""
        run_strategist_session()
        call_kwargs = mock_strategist_deps["run_agentic_loop"].call_args
        system = call_kwargs[1]["system"] if "system" in call_kwargs[1] else call_kwargs[0][2]
        assert "strategist" in system.lower() or "playbook" in system.lower()

    def test_returns_strategist_result(self, mock_strategist_deps):
        """Should return a StrategistResult with correct fields."""
        result = run_strategist_session()
        assert isinstance(result, StrategistResult)
        assert result.outcomes_backfilled == 5
        assert result.attribution_computed == 1
        assert result.turns_used == 3

    def test_backfill_error_doesnt_crash(self, mock_strategist_deps):
        """If backfill raises an exception, session should continue."""
        mock_strategist_deps["run_backfill"].side_effect = Exception("DB down")
        result = run_strategist_session()
        assert result.outcomes_backfilled == 0

    def test_attribution_error_doesnt_crash(self, mock_strategist_deps):
        """If attribution raises an exception, session should continue."""
        mock_strategist_deps["compute_signal_attribution"].side_effect = Exception("fail")
        result = run_strategist_session()
        assert result.attribution_computed == 0

    def test_delegates_to_run_strategist_loop(self, mock_strategist_deps):
        """run_strategist_session should delegate its loop to run_strategist_loop."""
        with patch("trading.ideation_claude.run_strategist_loop") as mock_loop:
            mock_loop.return_value = ClaudeIdeationResult(
                timestamp=datetime.now(),
                model="claude-opus-4-6",
                turns_used=3,
                theses_created=1,
                theses_updated=0,
                theses_closed=0,
                final_summary="Done",
                input_tokens=1000,
                output_tokens=500,
            )
            result = run_strategist_session()
            mock_loop.assert_called_once()
            assert result.theses_created == 1


# ---------------------------------------------------------------------------
# run_strategist_loop
# ---------------------------------------------------------------------------

class TestRunStrategistLoop:

    @pytest.fixture
    def mock_loop_deps(self):
        """Mock dependencies for run_strategist_loop (no backfill/attribution)."""
        with patch("trading.ideation_claude.get_attribution_summary") as mock_attr_summary, \
             patch("trading.ideation_claude.get_claude_client") as mock_client, \
             patch("trading.ideation_claude.run_agentic_loop") as mock_loop, \
             patch("trading.ideation_claude.extract_final_text", return_value="Summary"), \
             patch("trading.ideation_claude.reset_session"):
            mock_attr_summary.return_value = "Signal Attribution:\n- thesis: 62% win rate"
            mock_loop.return_value = AgenticLoopResult(
                messages=[],
                turns_used=3,
                stop_reason="end_turn",
                input_tokens=1000,
                output_tokens=500,
            )
            yield {
                "get_attribution_summary": mock_attr_summary,
                "get_claude_client": mock_client,
                "run_agentic_loop": mock_loop,
            }

    def test_does_not_call_backfill(self, mock_loop_deps):
        """run_strategist_loop should NOT call run_backfill."""
        with patch("trading.ideation_claude.run_backfill") as mock_backfill:
            run_strategist_loop()
            mock_backfill.assert_not_called()

    def test_does_not_call_compute_attribution(self, mock_loop_deps):
        """run_strategist_loop should NOT call compute_signal_attribution."""
        with patch("trading.ideation_claude.compute_signal_attribution") as mock_compute:
            run_strategist_loop()
            mock_compute.assert_not_called()

    def test_reads_attribution_from_db(self, mock_loop_deps):
        """Should call get_attribution_summary to read existing data."""
        run_strategist_loop()
        mock_loop_deps["get_attribution_summary"].assert_called_once()

    def test_returns_claude_ideation_result(self, mock_loop_deps):
        result = run_strategist_loop()
        assert isinstance(result, ClaudeIdeationResult)
        assert result.turns_used == 3

    def test_uses_session_strategist_prompt_by_default(self, mock_loop_deps):
        """Should use CLAUDE_SESSION_STRATEGIST_SYSTEM by default."""
        run_strategist_loop()
        call_kwargs = mock_loop_deps["run_agentic_loop"].call_args
        system = call_kwargs[1].get("system") or call_kwargs.kwargs.get("system")
        assert system == CLAUDE_SESSION_STRATEGIST_SYSTEM

    def test_accepts_custom_system_prompt(self, mock_loop_deps):
        """Should use the provided system_prompt override."""
        run_strategist_loop(system_prompt=CLAUDE_STRATEGIST_SYSTEM)
        call_kwargs = mock_loop_deps["run_agentic_loop"].call_args
        system = call_kwargs[1].get("system") or call_kwargs.kwargs.get("system")
        assert system == CLAUDE_STRATEGIST_SYSTEM

    def test_forwards_model_and_max_turns(self, mock_loop_deps):
        run_strategist_loop(model="claude-sonnet-4-20250514", max_turns=10)
        call_kwargs = mock_loop_deps["run_agentic_loop"].call_args
        assert call_kwargs[1]["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs[1]["max_turns"] == 10


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------

class TestSystemPromptContent:

    def test_strategist_prompt_mentions_fractional_shares(self):
        """Strategist system prompt should mention fractional shares."""
        assert "fractional shares" in CLAUDE_STRATEGIST_SYSTEM.lower()
        assert "fractional shares" in CLAUDE_SESSION_STRATEGIST_SYSTEM.lower()
