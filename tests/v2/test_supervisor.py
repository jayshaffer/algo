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
            # No final text-block assistant message → extract_final_text returns a synthetic fallback
        ],
        turns_used=20,
        stop_reason="max_turns",
        input_tokens=400,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    # Don't patch extract_final_text — let it run real against the loop messages.
    # status='max_turns' is driven by stop_reason, not by extract_final_text returning None.
    with patch.object(sup.claude_client, "run_agentic_loop", return_value=no_final):
        memo_id = sup.run_supervisor(dry_run=False)
    assert memo_id == 42
    params = _last_insert_params(patched_supervisor["mock_cursor"])
    assert params[3] == "max_turns"
    # content may be a synthetic fallback string or None — both are acceptable
    assert params[4] == 20
    assert params[9] == "loop hit max_turns before producing a final memo"


def test_run_supervisor_max_turns_with_synthetic_text_still_max_turns(patched_supervisor):
    """A loop result with stop_reason='max_turns' AND a final assistant text block
    (synthetic or otherwise) must still be recorded as status='max_turns', not 'ok'.
    """
    result_with_text = MagicMock(
        messages=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "get_theses", "input": {}, "id": "x"},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "[]"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "## Partial analysis\nRan out of turns."},
            ]},
        ],
        turns_used=20,
        stop_reason="max_turns",
        input_tokens=400,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    with patch.object(sup.claude_client, "run_agentic_loop", return_value=result_with_text):
        memo_id = sup.run_supervisor(dry_run=False)
    assert memo_id == 42
    params = _last_insert_params(patched_supervisor["mock_cursor"])
    # Despite having final text, stop_reason='max_turns' must set status='max_turns'
    assert params[3] == "max_turns"
    assert params[9] == "loop hit max_turns before producing a final memo"


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


def test_supervisor_tools_have_zero_overlap_with_mutators():
    """Defense-in-depth: supervisor MUST NOT include any strategy mutator tool.

    If this test fails, a future contributor has wired a write-capable tool
    into the supervisor handlers. The supervisor is observer-only.
    """
    from v2.supervisor import STRATEGY_MUTATOR_NAMES, SUPERVISOR_TOOL_HANDLERS
    overlap = set(SUPERVISOR_TOOL_HANDLERS.keys()) & set(STRATEGY_MUTATOR_NAMES)
    assert overlap == set(), (
        f"Supervisor MUST NOT include strategy mutator tools. Overlap: {overlap}"
    )


def test_supervisor_tools_are_all_get_prefixed():
    """Every supervisor tool name must start with `get_` — naming hygiene
    that also serves as a quick smell-test for an accidentally-wired
    mutator (which would start with `propose_`, `retire_`, etc.).
    """
    from v2.supervisor import SUPERVISOR_TOOL_HANDLERS
    bad = [name for name in SUPERVISOR_TOOL_HANDLERS if not name.startswith("get_")]
    assert bad == [], f"Non-get_ tools in supervisor registry: {bad}"


def test_mutator_set_matches_strategy_module_exports():
    """Sync check: STRATEGY_MUTATOR_NAMES must match v2/strategy.py's actual
    mutator-tool function exports. If a new mutator is added to strategy.py
    without updating STRATEGY_MUTATOR_NAMES, the overlap-defense test silently
    weakens — this catches that drift.
    """
    import v2.strategy as strat
    from v2.supervisor import STRATEGY_MUTATOR_NAMES
    expected_python_names = {
        "tool_propose_rule",
        "tool_retire_rule",
        "tool_revalidate_rule",
        "tool_update_strategy_identity",
        "tool_write_strategy_memo",
    }
    actual_python_names = {name for name in dir(strat) if name in expected_python_names}
    # The 5 expected mutators must all exist in v2.strategy
    missing = expected_python_names - actual_python_names
    assert missing == set(), f"Expected mutator functions missing from v2.strategy: {missing}"
    # And STRATEGY_MUTATOR_NAMES must be the same 5 minus the `tool_` prefix.
    expected_claude_names = {n.removeprefix("tool_") for n in expected_python_names}
    assert set(STRATEGY_MUTATOR_NAMES) == expected_claude_names, (
        f"STRATEGY_MUTATOR_NAMES drift. "
        f"In supervisor but not strategy: {set(STRATEGY_MUTATOR_NAMES) - expected_claude_names}. "
        f"In strategy but not supervisor: {expected_claude_names - set(STRATEGY_MUTATOR_NAMES)}."
    )


def test_record_watchlist_handler_buffers_and_validates():
    buf = []
    handler = sup._make_record_watchlist_handler(buf)
    out = handler(title="Retire Rule 43", detail="auto-lift fired", owner_stage="reflection")
    assert buf == [{"title": "Retire Rule 43", "detail": "auto-lift fired",
                    "owner_stage": "reflection"}]
    assert "recorded" in out.lower()
    err = handler(title="x", detail="y", owner_stage="trading")
    assert "Error" in err
    assert len(buf) == 1  # bad item not buffered


def test_record_watchlist_tool_in_supervisor_defs():
    names = {d["name"] for d in sup.build_supervisor_tool_defs()}
    assert "record_watchlist_item" in names


def test_run_supervisor_persists_buffered_items(fake_loop_result, fake_usage):
    def fake_loop(*args, **kwargs):
        handler = kwargs["tool_handlers"]["record_watchlist_item"]
        handler(title="Close thesis 267", detail="Rule 43 grounded", owner_stage="ideation")
        return fake_loop_result
    with patch("v2.supervisor.claude_client.run_agentic_loop", side_effect=fake_loop), \
         patch("v2.supervisor.claude_client.capture_usage") as cu, \
         patch("v2.supervisor.claude_client.get_claude_client"), \
         patch("v2.supervisor.claude_client.extract_final_text", return_value="## Watchlist\n- x"), \
         patch("v2.supervisor._compute_cost_safely", return_value=0.5), \
         patch("v2.supervisor._insert_memo", return_value=99) as ins, \
         patch("v2.supervisor.db.record_watchlist_item", return_value=1) as rec:
        cu.return_value.__enter__.return_value = fake_usage
        memo_id = sup.run_supervisor()
    assert memo_id == 99
    ins.assert_called_once()
    rec.assert_called_once_with(
        source_memo_id=99, title="Close thesis 267",
        detail="Rule 43 grounded", owner_stage="ideation",
    )


def test_run_supervisor_dry_run_does_not_persist_items(fake_loop_result, fake_usage):
    def fake_loop(*args, **kwargs):
        kwargs["tool_handlers"]["record_watchlist_item"](
            title="t", detail="d", owner_stage="reflection")
        return fake_loop_result
    with patch("v2.supervisor.claude_client.run_agentic_loop", side_effect=fake_loop), \
         patch("v2.supervisor.claude_client.capture_usage") as cu, \
         patch("v2.supervisor.claude_client.get_claude_client"), \
         patch("v2.supervisor.claude_client.extract_final_text", return_value="memo"), \
         patch("v2.supervisor._compute_cost_safely", return_value=0.0), \
         patch("v2.supervisor.db.record_watchlist_item") as rec:
        cu.return_value.__enter__.return_value = fake_usage
        sup.run_supervisor(dry_run=True)
    rec.assert_not_called()
