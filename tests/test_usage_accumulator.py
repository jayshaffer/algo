"""Unit tests for UsageAccumulator — sums token counts across calls."""

from types import SimpleNamespace

from v2.claude_client import UsageAccumulator


def _usage(input_t=0, output_t=0, cache_create=0, cache_read=0):
    return SimpleNamespace(
        input_tokens=input_t,
        output_tokens=output_t,
        cache_creation_input_tokens=cache_create,
        cache_read_input_tokens=cache_read,
    )


def test_initial_state_is_zero():
    acc = UsageAccumulator()
    assert acc.model is None
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
    assert acc.cache_creation_tokens == 0
    assert acc.cache_read_tokens == 0
    assert acc.mixed_models is False


def test_add_records_first_model():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=100, output_t=50))
    assert acc.model == "claude-haiku-4-5"
    assert acc.input_tokens == 100
    assert acc.output_tokens == 50
    assert acc.mixed_models is False


def test_add_sums_across_calls_same_model():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=100, output_t=50, cache_create=10, cache_read=20))
    acc.add("claude-haiku-4-5", _usage(input_t=200, output_t=80, cache_create=5,  cache_read=15))
    assert acc.input_tokens == 300
    assert acc.output_tokens == 130
    assert acc.cache_creation_tokens == 15
    assert acc.cache_read_tokens == 35
    assert acc.mixed_models is False


def test_add_flips_mixed_models_on_second_model():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=100))
    acc.add("claude-opus-4-7",  _usage(input_t=200))
    assert acc.mixed_models is True
    # First model is preserved as the recorded model
    assert acc.model == "claude-haiku-4-5"
    assert acc.input_tokens == 300


def test_add_handles_missing_cache_attrs():
    """Older API responses may omit cache_creation_input_tokens / cache_read_input_tokens."""
    acc = UsageAccumulator()
    bare = SimpleNamespace(input_tokens=100, output_tokens=50)
    acc.add("claude-haiku-4-5", bare)
    assert acc.cache_creation_tokens == 0
    assert acc.cache_read_tokens == 0


def test_add_handles_none_token_values():
    acc = UsageAccumulator()
    acc.add("claude-haiku-4-5", _usage(input_t=None, output_t=None))
    assert acc.input_tokens == 0
    assert acc.output_tokens == 0
