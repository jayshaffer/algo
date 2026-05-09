"""Tests for v2/claude_client.py - agentic loop, max_tokens cap, and recovery."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from v2.claude_client import extract_final_text, run_agentic_loop


def _make_bad_request_error(message: str) -> anthropic.BadRequestError:
    """Build a real anthropic.BadRequestError instance for tests. The SDK's
    constructor requires an httpx.Response; we synthesize a 400."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=400, request=request)
    return anthropic.BadRequestError(message, response=response, body=None)


def _text_block(text: str) -> SimpleNamespace:
    """A text content block. SimpleNamespace mirrors real anthropic content
    blocks: only the attributes set are present, so `hasattr(block, "text")`
    returns False on tool_use blocks (unlike bare MagicMock, which auto-creates
    every attribute on access)."""
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(tool_id: str, name: str, input_data: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=input_data)


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


def _make_stream_mock(responses):
    """Build a MagicMock whose `.messages.stream(**kwargs)` returns a
    context manager whose `get_final_message()` yields the next response
    from `responses`. Mirrors the production pattern:

        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    `responses` may be a single Message-like (single call) or a list/iterable
    (sequential calls / side_effect). To raise an exception on a given call,
    include the exception instance in the list — it will be raised in place
    of returning a Message, matching how anthropic SDK errors surface.
    """
    if not isinstance(responses, list):
        responses = [responses]
    iterator = iter(responses)

    def stream_factory(**_kwargs):
        try:
            nxt = next(iterator)
        except StopIteration:
            raise AssertionError("stream_factory called more times than mocked responses")
        if isinstance(nxt, BaseException):
            raise nxt
        ctx = MagicMock()
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = None
        ctx.get_final_message.return_value = nxt
        return ctx

    client = MagicMock()
    client.messages.stream.side_effect = stream_factory
    return client


class TestStreamingIsUsed:
    """The strategist's per-turn cap is the model's documented max (32000).
    The Anthropic Python SDK refuses non-streaming `messages.create()` calls
    whose max_tokens × estimated time-per-token exceeds 10 minutes. With
    32000 + Opus, that heuristic trips on turn 1, before any tokens are
    generated, and the whole stage bails. Streaming has no such guard, so
    the loop MUST use `client.messages.stream(...)` rather than
    `client.messages.create(...)`. Regression source: paper run 2026-05-02."""

    def test_loop_calls_stream_not_create(self):
        response = _make_response(
            content=[_text_block("Done")],
            stop_reason="end_turn",
        )
        client = _make_stream_mock(response)

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=3,
        )

        assert client.messages.stream.called, (
            "Loop must use client.messages.stream(...) — non-streaming "
            "messages.create() trips the SDK's 10-minute guard at "
            "max_tokens=32000 (paper run 2026-05-02)."
        )
        assert not client.messages.create.called, (
            "Loop must not call messages.create() — that path hits the "
            "10-minute non-streaming guard."
        )


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
        client = _make_stream_mock(response)

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=3,
        )

        call_kwargs = client.messages.stream.call_args.kwargs
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
        client = _make_stream_mock([tool_resp, end_resp])

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[{"name": "tool"}],
            tool_handlers={"tool": MagicMock(return_value="ok")},
            max_turns=3,
        )

        first_cap = client.messages.stream.call_args_list[0].kwargs["max_tokens"]
        second_cap = client.messages.stream.call_args_list[1].kwargs["max_tokens"]
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
        client = _make_stream_mock([truncated, recovered])

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
        assert client.messages.stream.call_count == 2

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
        client = _make_stream_mock([first_truncated, second_truncated])

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
        assert client.messages.stream.call_count == 2  # one retry only

    def test_recovery_preserves_role_alternation(self):
        """Bug regression: the prior turn already ends with a `user` message
        (initial prompt on turn 1, tool_results on later turns). Appending the
        concision-nudge user message without a synthetic assistant in between
        produces user→user adjacency, which the Anthropic API rejects.
        Verify no two consecutive messages share the same role after recovery.
        """
        truncated = _make_response(
            content=[_text_block("Long preamble that ran out of...")],
            stop_reason="max_tokens",
        )
        recovered = _make_response(
            content=[_text_block("Done concisely.")],
            stop_reason="end_turn",
        )
        client = _make_stream_mock([truncated, recovered])

        result = run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[],
            tool_handlers={},
            max_turns=5,
        )

        # Inspect the messages list as it stood when the second API call was
        # made — that's `result.messages` minus the final assistant turn that
        # was appended after the second response returned.
        # Easier check: the messages list has no two adjacent same-role
        # entries at any point.
        roles = [m["role"] for m in result.messages]
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1], (
                f"adjacent same-role messages at index {i - 1}/{i}: {roles}"
            )

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
        client = _make_stream_mock([tool_resp, truncated, end_resp])

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


class TestContextLengthRecovery:
    """When the API rejects a request for being too long, prune aggressively
    and retry once. Without recovery, a single bloated context fails the
    whole strategist stage with no graceful path back."""

    def test_recovers_from_context_length_error(self):
        """First call raises BadRequestError('prompt is too long...'); the
        loop should drop most of the message history and retry, then
        complete normally."""
        # Build a fake history with several exchanges so aggressive prune
        # can drop something.
        long_history_sentinel = "old_tool_result_that_should_be_pruned"
        first = _tool_use_block("t1", "fetch", {})
        second = _tool_use_block("t2", "fetch", {})

        recovered = _make_response(
            content=[_text_block("Done after pruning.")],
            stop_reason="end_turn",
        )
        # Sequence: BadRequestError → success on retry.
        client = _make_stream_mock([
            _make_bad_request_error("prompt is too long: 250000 tokens > 200000 maximum"),
            recovered,
        ])

        result = run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message=long_history_sentinel,
            tools=[],
            tool_handlers={},
            max_turns=5,
        )

        assert result.stop_reason == "end_turn"
        assert client.messages.stream.call_count == 2

    def test_second_context_length_error_propagates(self):
        """If pruning didn't help (still over the limit), don't loop —
        let the error propagate so the stage fails clearly."""
        client = _make_stream_mock([
            _make_bad_request_error("prompt is too long"),
            _make_bad_request_error("prompt is too long"),
        ])

        with pytest.raises(anthropic.BadRequestError):
            run_agentic_loop(
                client=client,
                model="m",
                system="sys",
                initial_message="hi",
                tools=[],
                tool_handlers={},
                max_turns=5,
            )

        # One initial attempt + one recovery attempt — no third try.
        assert client.messages.stream.call_count == 2

    def test_aggressive_prune_preserves_role_alternation(self):
        """Bug regression: `_aggressive_prune` returns
        `[messages[0], *messages[-4:]]`. Under the loop's normal alternating
        invariant the slice is safe, but if any upstream perturbation puts a
        `user` at the head of the tail (matching the prepended initial
        prompt), the API rejects same-role adjacency. Verify the prune
        always emits an alternating sequence regardless of input shape."""
        from v2.claude_client import _aggressive_prune

        # Pathological input: tail starts with `user`, matching messages[0].
        bad_input = [
            {"role": "user", "content": "init"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u2"},
            # Extra user tail that would collide with messages[0]:
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
        ]
        out = _aggressive_prune(bad_input)
        roles = [m["role"] for m in out]
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1], (
                f"adjacent same-role messages at index {i - 1}/{i}: {roles}"
            )

        # Also verify the canonical alternating shape passes through unchanged.
        good_input = [
            {"role": "user", "content": "init"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u3"},
        ]
        out = _aggressive_prune(good_input)
        roles = [m["role"] for m in out]
        assert roles == ["user", "assistant", "user", "assistant", "user"]

    def test_unrelated_bad_request_error_is_not_retried(self):
        """A 400 that isn't a context-length issue (e.g. invalid tool
        schema) should propagate immediately — retrying with the same
        prompt won't fix it and burns tokens."""
        client = _make_stream_mock([
            _make_bad_request_error("invalid request: tool 'foo' is not defined"),
        ])

        with pytest.raises(anthropic.BadRequestError):
            run_agentic_loop(
                client=client,
                model="m",
                system="sys",
                initial_message="hi",
                tools=[],
                tool_handlers={},
                max_turns=5,
            )

        assert client.messages.stream.call_count == 1


class TestExtractFinalText:
    """`extract_final_text` produces the strategist memo body. When a
    tool-driven loop ends without a trailing narrative, the function used to
    return None — the caller's `or "No summary available"` then stored a junk
    placeholder as the strategist memo and reflection lost the signal."""

    def test_returns_text_from_final_assistant_when_only_text(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [_text_block("synthesis text")]},
        ]
        assert extract_final_text(msgs) == "synthesis text"

    def test_walks_back_when_most_recent_assistant_is_tool_only(self):
        """If the last assistant turn is pure tool_use (loop ended on
        tool_use without an end_turn synthesis), use the most recent
        assistant text block we can find."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [_text_block("preamble"), _tool_use_block("t1", "write_playbook", {})]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            {"role": "assistant", "content": [_tool_use_block("t2", "write_playbook", {})]},
        ]
        assert extract_final_text(msgs) == "preamble"

    def test_returns_tool_summary_fallback_when_no_assistant_text(self):
        """When every assistant message is tool_use-only, fall back to a
        tool-name summary so the strategist memo still records what the
        agent did. Must be non-None so the caller's `or 'No summary
        available'` placeholder doesn't poison the memo."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [_tool_use_block("t1", "start_thesis", {})]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            {"role": "assistant", "content": [_tool_use_block("t2", "write_playbook", {})]},
        ]
        result = extract_final_text(msgs)
        assert result is not None, (
            "Should not return None when assistant messages exist — caller "
            "falls back to a meaningless placeholder string"
        )
        assert "write_playbook" in result, (
            "Tool-summary fallback should at least name the final tool calls"
        )

    def test_returns_none_when_no_assistant_messages_at_all(self):
        """If literally no assistant message has been added (loop crashed
        before first response), None is the right answer."""
        msgs = [{"role": "user", "content": "hi"}]
        assert extract_final_text(msgs) is None

    def test_concatenates_multiple_text_blocks_in_final_message(self):
        """Some responses split synthesis across multiple text blocks.
        Returning only the first loses the rest."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    _text_block("Part one. "),
                    _text_block("Part two."),
                ],
            },
        ]
        result = extract_final_text(msgs)
        assert "Part one." in result and "Part two." in result, (
            f"Expected both text blocks concatenated, got: {result!r}"
        )


class TestRunAgenticLoopTelemetry:
    """`run_agentic_loop` must emit a `tool_invocation` event after each tool
    dispatch, capturing tool name, args, success, error, and duration_ms.
    `session_id=None` is a no-op (handled inside `record_event`)."""

    def test_emits_tool_invocation_event_on_success(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kw: recorded.append(kw),
        )
        tool_resp = _make_response(
            content=[_tool_use_block("t1", "my_tool", {"x": 1})],
            stop_reason="tool_use",
        )
        end_resp = _make_response(
            content=[_text_block("Done")],
            stop_reason="end_turn",
        )
        client = _make_stream_mock([tool_resp, end_resp])

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[{"name": "my_tool"}],
            tool_handlers={"my_tool": lambda **k: "ok"},
            max_turns=3,
            session_id=99,
            stage_name="ideation",
        )

        assert len(recorded) == 1
        ev = recorded[0]
        assert ev["session_id"] == 99
        assert ev["stage_name"] == "ideation"
        assert ev["event_type"] == "tool_invocation"
        assert ev["payload"]["tool_name"] == "my_tool"
        assert ev["payload"]["success"] is True
        assert ev["payload"]["error"] is None
        assert "duration_ms" in ev["payload"]

    def test_emits_tool_invocation_event_on_handler_error(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kw: recorded.append(kw),
        )

        def bad_handler(**_):
            raise RuntimeError("boom")

        tool_resp = _make_response(
            content=[_tool_use_block("t1", "my_tool", {})],
            stop_reason="tool_use",
        )
        end_resp = _make_response(
            content=[_text_block("Done")],
            stop_reason="end_turn",
        )
        client = _make_stream_mock([tool_resp, end_resp])

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[{"name": "my_tool"}],
            tool_handlers={"my_tool": bad_handler},
            max_turns=3,
            session_id=42,
            stage_name="reflection",
        )

        assert len(recorded) == 1
        ev = recorded[0]
        assert ev["payload"]["success"] is False
        assert "boom" in (ev["payload"]["error"] or "")

    def test_no_session_id_still_calls_record_event_as_noop(self, monkeypatch):
        """Without session_id the loop still calls record_event; record_event
        itself no-ops on session_id=None. We only need to confirm the call
        path is unchanged."""
        recorded = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kw: recorded.append(kw),
        )
        tool_resp = _make_response(
            content=[_tool_use_block("t1", "my_tool", {})],
            stop_reason="tool_use",
        )
        end_resp = _make_response(
            content=[_text_block("Done")],
            stop_reason="end_turn",
        )
        client = _make_stream_mock([tool_resp, end_resp])

        run_agentic_loop(
            client=client,
            model="m",
            system="sys",
            initial_message="hi",
            tools=[{"name": "my_tool"}],
            tool_handlers={"my_tool": lambda **k: "ok"},
            max_turns=3,
        )

        assert len(recorded) == 1
        assert recorded[0]["session_id"] is None
        assert recorded[0]["stage_name"] == "unknown"
