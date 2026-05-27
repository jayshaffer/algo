"""Unit tests for v2.supervisor.

Tests four run_supervisor outcome paths with mocked Anthropic + DB.
Task 4.1 will add mutator-overlap defense tests to this same module.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from v2 import supervisor as sup


@pytest.fixture
def fake_loop_result():
    """A successful agentic loop result with one tool use and a final text."""
    return MagicMock(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Begin..."}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "get_active_rules", "input": {}, "id": "t1"},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[]"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "## Watchlist\n- Nothing major"}]},
        ],
        turns_used=2,
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


@pytest.fixture
def fake_usage():
    return MagicMock(
        model="claude-opus-4-7",
        input_tokens=100,
        output_tokens=50,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )


@pytest.fixture
def patched_supervisor(mock_db, mock_cursor, fake_usage):
    """Patch get_claude_client + capture_usage + stage_cost_usd.

    The agentic loop and the fetched memo id are still parametric — callers
    set them via the returned context's attributes.
    """
    mock_cursor.fetchone.return_value = {"id": 42}

    with patch.object(sup.claude_client, "get_claude_client"), \
         patch.object(sup.claude_client, "capture_usage") as cap, \
         patch.object(sup, "stage_cost_usd", return_value=0.0123):
        cap.return_value.__enter__.return_value = fake_usage
        cap.return_value.__exit__.return_value = False
        yield {"mock_cursor": mock_cursor, "cap": cap}


def _last_insert_params(mock_cursor):
    """Return the params of the most recent INSERT against supervisor_memos."""
    for call in reversed(mock_cursor.execute.call_args_list):
        sql = call.args[0]
        if "INSERT INTO supervisor_memos" in sql:
            return call.args[1]
    pytest.fail("no INSERT INTO supervisor_memos in execute calls")


def test_run_supervisor_inserts_ok_row(patched_supervisor, fake_loop_result):
    with patch.object(sup.claude_client, "run_agentic_loop", return_value=fake_loop_result):
        memo_id = sup.run_supervisor(dry_run=False)
    assert memo_id == 42
    params = _last_insert_params(patched_supervisor["mock_cursor"])
    # Positional order: model, prompt_version, content, status, turns_used,
    #                   tool_calls, input_tokens, output_tokens, cost_usd, error_message
    assert params[0] == sup.DEFAULT_SUPERVISOR_MODEL
    assert params[1] == sup.PROMPT_VERSION
    assert "Watchlist" in params[2]
    assert params[3] == "ok"
    assert params[4] == 2  # turns_used
    tool_calls = json.loads(params[5])
    assert tool_calls == [{"name": "get_active_rules", "count": 1}]
    assert params[6] == 100  # input_tokens
    assert params[7] == 50   # output_tokens
    assert params[8] == 0.0123  # cost_usd
    assert params[9] is None  # error_message


def test_run_supervisor_max_turns_inserts_null_content(patched_supervisor):
    no_final = MagicMock(
        messages=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "get_theses", "input": {}, "id": "x"},
            ]},
            # No final text-block assistant message → extract_final_text returns None
        ],
        turns_used=20,
        stop_reason="max_turns",
        input_tokens=400,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    with patch.object(sup.claude_client, "run_agentic_loop", return_value=no_final), \
         patch.object(sup.claude_client, "extract_final_text", return_value=None):
        memo_id = sup.run_supervisor(dry_run=False)
    assert memo_id == 42
    params = _last_insert_params(patched_supervisor["mock_cursor"])
    assert params[3] == "max_turns"
    assert params[2] is None  # content NULL
    assert params[4] == 20
    assert params[9] == "loop did not produce final text within max_turns"


def test_run_supervisor_dry_run_does_not_insert(patched_supervisor, fake_loop_result, capsys):
    with patch.object(sup.claude_client, "run_agentic_loop", return_value=fake_loop_result):
        memo_id = sup.run_supervisor(dry_run=True)
    assert memo_id is None
    out = capsys.readouterr().out
    assert "Watchlist" in out
    # No INSERT should have happened.
    for call in patched_supervisor["mock_cursor"].execute.call_args_list:
        assert "INSERT INTO supervisor_memos" not in call.args[0]


def test_run_supervisor_api_error_inserts_error_row(patched_supervisor):
    with patch.object(sup.claude_client, "run_agentic_loop", side_effect=RuntimeError("api 500")):
        memo_id = sup.run_supervisor(dry_run=False)
    assert memo_id == 42
    params = _last_insert_params(patched_supervisor["mock_cursor"])
    assert params[3] == "error"
    assert params[2] is None  # content NULL
    assert params[4] == 0    # turns_used
    assert "RuntimeError: api 500" in params[9]


def test_run_supervisor_dry_run_reraises_on_error(patched_supervisor):
    with patch.object(sup.claude_client, "run_agentic_loop", side_effect=RuntimeError("api 500")), \
         pytest.raises(RuntimeError, match="api 500"):
        sup.run_supervisor(dry_run=True)
    # No INSERT should have happened (the dry-run path doesn't insert).
    for call in patched_supervisor["mock_cursor"].execute.call_args_list:
        assert "INSERT INTO supervisor_memos" not in call.args[0]
