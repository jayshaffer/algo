"""Tests for v2/claude_client.py - agentic loop, max_tokens cap, and recovery."""

from unittest.mock import MagicMock

import pytest

from v2.claude_client import run_agentic_loop


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(tool_id: str, name: str, input_data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _make_response(
    content,
    stop_reason="end_turn",
    input_tokens=100,
    output_tokens=50,
    cache_creation=0,
    cache_read=0,
):
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.usage.cache_creation_input_tokens = cache_creation
    response.usage.cache_read_input_tokens = cache_read
    return response


class TestMaxTokensCap:
    """The per-turn output cap must be the model's documented max so truncation
    only happens on genuinely pathological generations. Historical regression:
    the cap was 2048 on turns >0, which truncated real strategist runs and
    dropped the playbook silently (paper run 2026-05-02)."""

    def test_cap_is_at_model_max(self):
        """max_tokens passed to the API must be >= 32000 (Opus 4.x model max)
        so the strategist's synthesis turn cannot be truncated by the cap
        we choose ourselves."""
        response = _make_response(
            content=[_text_block("Done")],
            stop_reason="end_turn",
        )
        client = MagicMock()
        client.messages.create.return_value = response

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=3,
        )

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] >= 32000, (
            "Per-turn max_tokens must be >= 32000 (model max) — anything lower "
            "is a self-imposed truncation we have no business choosing"
        )

    def test_cap_is_consistent_across_turns(self):
        """The cap should not be lower on later turns. Synthesis happens
        late (turn 3-5), not early. The previous ramp (4096 first / 2048 after)
        was backwards and caused the 2026-05-02 paper-session failure."""
        tool_resp = _make_response(
            content=[_tool_use_block("t1", "tool", {})],
            stop_reason="tool_use",
        )
        end_resp = _make_response(
            content=[_text_block("Done")],
            stop_reason="end_turn",
        )
        client = MagicMock()
        client.messages.create.side_effect = [tool_resp, end_resp]

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[{"name": "tool"}],
            tool_handlers={"tool": MagicMock(return_value="ok")},
            max_turns=3,
        )

        first_cap = client.messages.create.call_args_list[0].kwargs["max_tokens"]
        second_cap = client.messages.create.call_args_list[1].kwargs["max_tokens"]
        assert second_cap >= first_cap, (
            f"Later-turn cap ({second_cap}) must not be smaller than first-turn ({first_cap})"
        )


class TestMaxTokensRecovery:
    """When the API returns stop_reason='max_tokens', the loop should
    discard the truncated assistant response, inject a concision prompt,
    and retry once before giving up. Otherwise a single oversized turn
    drops the playbook and skips the executor downstream."""

    def test_recovery_continues_loop_with_concision_prompt(self):
        """First max_tokens hit: drop the truncated turn, add a concision
        user message, continue. Loop proceeds to end_turn on the retry."""
        truncated = _make_response(
            content=[_text_block("Long preamble that ran out of...")],
            stop_reason="max_tokens",
        )
        recovered = _make_response(
            content=[_text_block("Done concisely.")],
            stop_reason="end_turn",
        )
        client = MagicMock()
        client.messages.create.side_effect = [truncated, recovered]

        result = run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=5,
        )

        assert result.stop_reason == "end_turn"
        assert client.messages.create.call_count == 2

        # The truncated assistant response should NOT be in the final messages.
        assistant_texts = [
            block.text
            for m in result.messages
            if m["role"] == "assistant"
            for block in m["content"]
            if hasattr(block, "text")
        ]
        assert "Long preamble that ran out of..." not in assistant_texts
        assert "Done concisely." in assistant_texts

        # A concision-instruction user message should have been injected.
        user_strings = [
            m["content"]
            for m in result.messages
            if m["role"] == "user" and isinstance(m["content"], str)
        ]
        assert any("concise" in s.lower() or "truncat" in s.lower() for s in user_strings), (
            "Expected a concision/truncation instruction injected after max_tokens"
        )

    def test_second_max_tokens_bails(self):
        """If max_tokens hits again after recovery, give up — don't loop
        forever burning tokens."""
        first_truncated = _make_response(
            content=[_text_block("first overflow")],
            stop_reason="max_tokens",
        )
        second_truncated = _make_response(
            content=[_text_block("still too long")],
            stop_reason="max_tokens",
        )
        client = MagicMock()
        client.messages.create.side_effect = [first_truncated, second_truncated]

        result = run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=10,
        )

        assert result.stop_reason == "max_tokens"
        assert client.messages.create.call_count == 2  # one retry only

    def test_recovery_preserves_prior_tool_results(self):
        """Recovery must not erase tool_results from earlier successful turns —
        only the most recent (truncated) assistant response."""
        tool_resp = _make_response(
            content=[_tool_use_block("t1", "fetch", {})],
            stop_reason="tool_use",
        )
        truncated = _make_response(
            content=[_text_block("synthesis ran out")],
            stop_reason="max_tokens",
        )
        end_resp = _make_response(
            content=[_text_block("Done")],
            stop_reason="end_turn",
        )
        client = MagicMock()
        client.messages.create.side_effect = [tool_resp, truncated, end_resp]

        handler = MagicMock(return_value="fetched data")

        result = run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[{"name": "fetch"}],
            tool_handlers={"fetch": handler},
            max_turns=5,
        )

        assert result.stop_reason == "end_turn"
        # The original tool_use exchange should still be present.
        tool_use_present = any(
            m["role"] == "assistant"
            and isinstance(m["content"], list)
            and any(getattr(b, "type", None) == "tool_use" for b in m["content"])
            for m in result.messages
        )
        assert tool_use_present, "Tool-use turn before max_tokens must be preserved"

        tool_result_present = any(
            m["role"] == "user"
            and isinstance(m["content"], list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
            for m in result.messages
        )
        assert tool_result_present, "Tool-result message must be preserved"
