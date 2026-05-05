"""Tests for capture_usage context manager + _record_usage helper."""

from types import SimpleNamespace

from v2.claude_client import _record_usage, capture_usage


def _usage(input_t=0, output_t=0):
    return SimpleNamespace(
        input_tokens=input_t,
        output_tokens=output_t,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def test_record_usage_outside_block_is_noop():
    """Calls without an active capture_usage() block do nothing — no error,
    no global state mutation."""
    _record_usage("claude-haiku-4-5", _usage(input_t=100))
    # Re-entering a capture block must start fresh
    with capture_usage() as acc:
        assert acc.input_tokens == 0


def test_capture_usage_collects_within_block():
    with capture_usage() as acc:
        _record_usage("claude-haiku-4-5", _usage(input_t=100, output_t=50))
        _record_usage("claude-haiku-4-5", _usage(input_t=200, output_t=80))
    assert acc.input_tokens == 300
    assert acc.output_tokens == 130
    assert acc.model == "claude-haiku-4-5"


def test_sequential_blocks_are_isolated():
    with capture_usage() as a:
        _record_usage("claude-haiku-4-5", _usage(input_t=100))
    with capture_usage() as b:
        _record_usage("claude-opus-4-7", _usage(input_t=200))
    assert a.input_tokens == 100
    assert a.model == "claude-haiku-4-5"
    assert b.input_tokens == 200
    assert b.model == "claude-opus-4-7"


def test_nested_blocks_inner_collects_only_inner_calls():
    """The contextvars-based scoping means inner block sees only its own
    calls; outer block's own calls (before and after) accumulate to outer."""
    with capture_usage() as outer:
        _record_usage("claude-opus-4-7", _usage(input_t=10))
        with capture_usage() as inner:
            _record_usage("claude-haiku-4-5", _usage(input_t=100))
        _record_usage("claude-opus-4-7", _usage(input_t=20))
    assert inner.input_tokens == 100
    assert inner.model == "claude-haiku-4-5"
    assert outer.input_tokens == 30
    assert outer.model == "claude-opus-4-7"


def test_block_resets_after_exception():
    try:
        with capture_usage() as _:
            _record_usage("claude-haiku-4-5", _usage(input_t=100))
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # New block sees fresh state
    with capture_usage() as fresh:
        assert fresh.input_tokens == 0
