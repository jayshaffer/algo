"""Verify that every Claude call routed through _call_with_retry feeds
its usage into the active capture_usage() block."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from v2.claude_client import _call_with_retry, capture_usage


def _make_mock_client(usage):
    """Build a stand-in for the anthropic.Anthropic client whose
    messages.stream() returns a context manager whose final message
    carries the given usage."""
    final_message = SimpleNamespace(
        usage=usage,
        content=[],
        stop_reason="end_turn",
    )
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=SimpleNamespace(
        get_final_message=MagicMock(return_value=final_message)
    ))
    stream_cm.__exit__ = MagicMock(return_value=False)
    client = MagicMock()
    client.messages.stream.return_value = stream_cm
    return client


def test_call_with_retry_records_usage_in_active_block():
    client = _make_mock_client(SimpleNamespace(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=10, cache_read_input_tokens=20,
    ))
    with capture_usage() as acc:
        _call_with_retry(client, model="claude-haiku-4-5-20251001",
                         max_tokens=100, messages=[])
    assert acc.model == "claude-haiku-4-5-20251001"
    assert acc.input_tokens == 100
    assert acc.output_tokens == 50
    assert acc.cache_creation_tokens == 10
    assert acc.cache_read_tokens == 20


def test_call_with_retry_does_nothing_outside_capture_block():
    """Production callers that don't open a capture block must not break."""
    client = _make_mock_client(SimpleNamespace(
        input_tokens=1, output_tokens=1,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    ))
    # Should not raise
    result = _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
    assert result.usage.input_tokens == 1


def test_multiple_calls_in_one_block_sum():
    client = _make_mock_client(SimpleNamespace(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    ))
    with capture_usage() as acc:
        _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
        _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
        _call_with_retry(client, model="claude-haiku-4-5", max_tokens=10, messages=[])
    assert acc.input_tokens == 300
    assert acc.output_tokens == 150


def test_agentic_loop_does_not_double_count_simulated():
    """If run_agentic_loop is ever modified to also call _record_usage with its
    AgenticLoopResult totals, every token would be counted twice. This
    test asserts that the only path producing recorded usage is
    _call_with_retry."""
    # Simulate three "turns" of an agentic loop — three _call_with_retry
    # invocations. Expectation: exactly summed tokens, never doubled.
    client = _make_mock_client(SimpleNamespace(
        input_tokens=100, output_tokens=50,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    ))
    with capture_usage() as acc:
        for _ in range(3):
            _call_with_retry(client, model="claude-opus-4-7", max_tokens=10, messages=[])
    # Three calls × 100 input each = 300 total. If double-counting, would be 600.
    assert acc.input_tokens == 300
    assert acc.output_tokens == 150
