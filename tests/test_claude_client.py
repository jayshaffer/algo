"""Tests for trading/claude_client.py - Claude API client and agentic loop."""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from trading.claude_client import (
    ToolResult,
    AgenticLoopResult,
    get_claude_client,
    run_agentic_loop,
    extract_final_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_block(text: str) -> MagicMock:
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, input_data: dict) -> MagicMock:
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _make_response(content, stop_reason="end_turn", input_tokens=100,
                   output_tokens=50):
    """Create a mock Claude messages.create response."""
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestToolResult:

    def test_defaults(self):
        tr = ToolResult(tool_use_id="abc", content="result")
        assert tr.tool_use_id == "abc"
        assert tr.content == "result"
        assert tr.is_error is False

    def test_error_flag(self):
        tr = ToolResult(tool_use_id="abc", content="fail", is_error=True)
        assert tr.is_error is True


class TestAgenticLoopResult:

    def test_fields(self):
        result = AgenticLoopResult(
            messages=[{"role": "user", "content": "hi"}],
            turns_used=3,
            stop_reason="end_turn",
            input_tokens=500,
            output_tokens=200,
        )
        assert result.turns_used == 3
        assert result.stop_reason == "end_turn"
        assert result.input_tokens == 500
        assert result.output_tokens == 200


# ---------------------------------------------------------------------------
# get_claude_client
# ---------------------------------------------------------------------------

class TestGetClaudeClient:

    def test_raises_without_api_key(self, monkeypatch):
        """Should raise ValueError when ANTHROPIC_API_KEY is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_claude_client()

    @patch("trading.claude_client.anthropic.Anthropic")
    def test_creates_client_with_key(self, mock_anthropic, claude_env):
        """Should create Anthropic client with the env var key."""
        client = get_claude_client()
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @patch("trading.claude_client.anthropic.Anthropic")
    def test_returns_anthropic_instance(self, mock_anthropic, claude_env):
        """Should return the client instance."""
        mock_anthropic.return_value = MagicMock()
        client = get_claude_client()
        assert client is mock_anthropic.return_value


# ---------------------------------------------------------------------------
# run_agentic_loop
# ---------------------------------------------------------------------------

class TestRunAgenticLoop:

    def _make_client(self, responses):
        """Create a mock client that returns a sequence of responses."""
        client = MagicMock()
        client.messages.create.side_effect = responses
        return client

    def test_end_turn_stops_loop(self):
        """Loop should stop on end_turn stop_reason."""
        response = _make_response(
            content=[_make_text_block("Done!")],
            stop_reason="end_turn",
        )
        client = self._make_client([response])

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="You are helpful.",
            initial_message="Hello",
            tools=[],
            tool_handlers={},
            max_turns=10,
        )

        assert result.stop_reason == "end_turn"
        assert result.turns_used == 1
        assert client.messages.create.call_count == 1

    def test_tool_use_executes_handler(self):
        """Tool use blocks should execute the corresponding handler."""
        tool_response = _make_response(
            content=[_make_tool_use_block("tool-1", "get_data", {"key": "val"})],
            stop_reason="tool_use",
        )
        final_response = _make_response(
            content=[_make_text_block("Here is the data.")],
            stop_reason="end_turn",
        )
        client = self._make_client([tool_response, final_response])

        handler = MagicMock(return_value="data result")

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Get data",
            tools=[{"name": "get_data"}],
            tool_handlers={"get_data": handler},
            max_turns=10,
        )

        handler.assert_called_once_with(key="val")
        assert result.stop_reason == "end_turn"
        assert result.turns_used == 2

    def test_unknown_tool_returns_error(self):
        """Unknown tool names should produce an error tool result."""
        tool_response = _make_response(
            content=[_make_tool_use_block("tool-1", "nonexistent", {})],
            stop_reason="tool_use",
        )
        final_response = _make_response(
            content=[_make_text_block("Ok")],
            stop_reason="end_turn",
        )
        client = self._make_client([tool_response, final_response])

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Do something",
            tools=[],
            tool_handlers={},
            max_turns=10,
        )

        # The tool result should contain the error message
        tool_result_msg = result.messages[2]  # user msg with tool results
        assert tool_result_msg["role"] == "user"
        content = tool_result_msg["content"]
        assert isinstance(content, list)
        assert content[0]["is_error"] is True
        assert "Unknown tool" in content[0]["content"]

    def test_handler_exception_returns_error(self):
        """Handler exceptions should be caught and returned as error results."""
        tool_response = _make_response(
            content=[_make_tool_use_block("tool-1", "failing_tool", {})],
            stop_reason="tool_use",
        )
        final_response = _make_response(
            content=[_make_text_block("Noted")],
            stop_reason="end_turn",
        )
        client = self._make_client([tool_response, final_response])

        handler = MagicMock(side_effect=RuntimeError("DB connection failed"))

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Try this",
            tools=[{"name": "failing_tool"}],
            tool_handlers={"failing_tool": handler},
            max_turns=10,
        )

        tool_result_msg = result.messages[2]
        content = tool_result_msg["content"]
        assert content[0]["is_error"] is True
        assert "DB connection failed" in content[0]["content"]

    def test_max_turns_hit(self):
        """Loop should stop when max_turns is reached."""
        # Always return tool_use to force another turn
        tool_response = _make_response(
            content=[_make_tool_use_block("t", "tool", {})],
            stop_reason="tool_use",
        )
        client = MagicMock()
        client.messages.create.return_value = tool_response

        handler = MagicMock(return_value="ok")

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Loop forever",
            tools=[{"name": "tool"}],
            tool_handlers={"tool": handler},
            max_turns=3,
        )

        assert result.stop_reason == "max_turns"
        assert result.turns_used == 3
        assert client.messages.create.call_count == 3

    def test_token_counting(self):
        """Input and output tokens should be summed across turns."""
        resp1 = _make_response(
            content=[_make_tool_use_block("t", "tool", {})],
            stop_reason="tool_use",
            input_tokens=100,
            output_tokens=50,
        )
        resp2 = _make_response(
            content=[_make_text_block("Done")],
            stop_reason="end_turn",
            input_tokens=200,
            output_tokens=75,
        )
        client = self._make_client([resp1, resp2])
        handler = MagicMock(return_value="result")

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Go",
            tools=[{"name": "tool"}],
            tool_handlers={"tool": handler},
            max_turns=10,
        )

        assert result.input_tokens == 300
        assert result.output_tokens == 125

    def test_messages_contain_initial_user_message(self):
        """Messages list should start with the initial user message."""
        response = _make_response(
            content=[_make_text_block("Hi")],
            stop_reason="end_turn",
        )
        client = self._make_client([response])

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Hello there",
            tools=[],
            tool_handlers={},
        )

        assert result.messages[0] == {"role": "user", "content": "Hello there"}

    def test_multiple_tool_calls_in_one_turn(self):
        """Multiple tool_use blocks in a single response should all be handled."""
        tool_response = _make_response(
            content=[
                _make_tool_use_block("t1", "tool_a", {"x": 1}),
                _make_tool_use_block("t2", "tool_b", {"y": 2}),
            ],
            stop_reason="tool_use",
        )
        final_response = _make_response(
            content=[_make_text_block("All done")],
            stop_reason="end_turn",
        )
        client = self._make_client([tool_response, final_response])

        handler_a = MagicMock(return_value="result_a")
        handler_b = MagicMock(return_value="result_b")

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Do both",
            tools=[],
            tool_handlers={"tool_a": handler_a, "tool_b": handler_b},
            max_turns=10,
        )

        handler_a.assert_called_once_with(x=1)
        handler_b.assert_called_once_with(y=2)

        # Tool results message should have 2 results
        tool_result_msg = result.messages[2]
        assert len(tool_result_msg["content"]) == 2

    def test_unexpected_stop_reason(self):
        """An unexpected stop_reason should end the loop with that reason."""
        response = _make_response(
            content=[_make_text_block("Hmm")],
            stop_reason="max_tokens",
        )
        client = self._make_client([response])

        result = run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Long response",
            tools=[],
            tool_handlers={},
        )

        assert result.stop_reason == "max_tokens"
        assert result.turns_used == 1

    def test_tool_handler_receives_correct_kwargs(self):
        """Tool handler should receive the tool input as keyword arguments."""
        tool_response = _make_response(
            content=[_make_tool_use_block("t1", "search", {"query": "AAPL", "limit": 5})],
            stop_reason="tool_use",
        )
        final_response = _make_response(
            content=[_make_text_block("Found results")],
            stop_reason="end_turn",
        )
        client = self._make_client([tool_response, final_response])

        handler = MagicMock(return_value="results")

        run_agentic_loop(
            client=client,
            model="test-model",
            system="sys",
            initial_message="Search for AAPL",
            tools=[],
            tool_handlers={"search": handler},
        )

        handler.assert_called_once_with(query="AAPL", limit=5)


# ---------------------------------------------------------------------------
# extract_final_text
# ---------------------------------------------------------------------------

class TestExtractFinalText:

    def test_extracts_text_from_list_content(self):
        """Should extract .text from content blocks in last assistant message."""
        text_block = _make_text_block("Final answer")
        messages = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": [text_block]},
        ]
        assert extract_final_text(messages) == "Final answer"

    def test_extracts_string_content(self):
        """Should handle string content in assistant messages."""
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "String response"},
        ]
        assert extract_final_text(messages) == "String response"

    def test_returns_none_for_empty_messages(self):
        """Should return None when messages list is empty."""
        assert extract_final_text([]) is None

    def test_returns_none_for_no_assistant_messages(self):
        """Should return None when there are no assistant messages."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        assert extract_final_text(messages) is None

    def test_returns_last_assistant_message(self):
        """Should return text from the LAST assistant message, not the first."""
        block1 = _make_text_block("First response")
        block2 = _make_text_block("Final response")
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": [block1]},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": [block2]},
        ]
        assert extract_final_text(messages) == "Final response"

    def test_skips_tool_use_blocks(self):
        """Should find text blocks even when tool_use blocks are present."""
        tool_block = _make_tool_use_block("t1", "search", {})
        text_block = _make_text_block("Answer with tools")
        messages = [
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": [tool_block, text_block]},
        ]
        # tool_block doesn't have .text attr (well it does on MagicMock, but
        # the real code checks hasattr(block, "text") which will be true on
        # a mock - so the first block with .text is the tool_block mock).
        # Let's use a proper object without .text for tool block.
        tool_block_proper = MagicMock(spec=["type", "id", "name", "input"])
        tool_block_proper.type = "tool_use"

        messages2 = [
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": [tool_block_proper, text_block]},
        ]
        result = extract_final_text(messages2)
        assert result == "Answer with tools"

    def test_content_list_without_text_blocks(self):
        """When content is a list but no blocks have .text, try next assistant msg."""
        # Block without text attribute
        block = MagicMock(spec=["type", "id", "name", "input"])
        block.type = "tool_use"

        earlier_text = _make_text_block("Earlier answer")
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": [earlier_text]},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": [block]},
        ]
        # Last assistant has no text blocks, so it should fall through to
        # the earlier assistant message
        result = extract_final_text(messages)
        assert result == "Earlier answer"

    def test_mixed_user_tool_result_messages(self):
        """extract_final_text should skip user messages with tool results."""
        text_block = _make_text_block("Summary")
        messages = [
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": [text_block]},
            {"role": "user", "content": [{"type": "tool_result", "content": "data"}]},
            {"role": "assistant", "content": [_make_text_block("Final summary")]},
        ]
        assert extract_final_text(messages) == "Final summary"
