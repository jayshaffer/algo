"""Unit tests for the supervisor watchlist DB helpers and shared module."""

from unittest.mock import MagicMock, patch

import pytest

from v2 import watchlist as wl
from v2.database import trading_db as db


@pytest.fixture
def fake_cursor():
    cur = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cur)
    cm.__exit__ = MagicMock(return_value=False)
    return cm, cur


def test_record_watchlist_item_inserts_and_returns_id(fake_cursor):
    cm, cur = fake_cursor
    cur.fetchone.return_value = {"id": 7}
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        item_id = db.record_watchlist_item(
            source_memo_id=3, title="Retire Rule 43",
            detail="auto-lift fired", owner_stage="reflection",
        )
    assert item_id == 7
    sql = cur.execute.call_args[0][0]
    assert "INSERT INTO supervisor_watchlist_items" in sql


def test_record_watchlist_item_rejects_bad_owner_stage(fake_cursor):
    cm, _ = fake_cursor
    with patch("v2.database.trading_db.get_cursor", return_value=cm), pytest.raises(ValueError):
        db.record_watchlist_item(
            source_memo_id=3, title="x", detail="y", owner_stage="trading",
        )


@pytest.mark.real_watchlist_db
def test_get_open_watchlist_items_filters_by_stage(fake_cursor):
    cm, cur = fake_cursor
    cur.fetchall.return_value = [{"id": 1, "title": "t", "detail": "d"}]
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        rows = db.get_open_watchlist_items("ideation")
    assert rows == [{"id": 1, "title": "t", "detail": "d"}]
    args = cur.execute.call_args[0]
    assert "status = 'open'" in args[0]
    assert args[1] == ("ideation",)


def test_resolve_watchlist_item_scopes_to_owner_stage(fake_cursor):
    cm, cur = fake_cursor
    cur.rowcount = 1
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        ok = db.resolve_watchlist_item(
            item_id=5, resolution="acted", note="retired Rule 43",
            session_id=42, stage="reflection",
        )
    assert ok is True
    sql, params = cur.execute.call_args[0]
    assert "owner_stage = %s" in sql
    assert "status = 'open'" in sql
    assert params == ("acted", "retired Rule 43", 42, "reflection", 5, "reflection")


def test_resolve_watchlist_item_rejects_bad_resolution(fake_cursor):
    cm, _ = fake_cursor
    with patch("v2.database.trading_db.get_cursor", return_value=cm), pytest.raises(ValueError):
        db.resolve_watchlist_item(
            item_id=5, resolution="maybe", note="x",
            session_id=1, stage="reflection",
        )


def test_resolve_watchlist_item_returns_false_when_no_row(fake_cursor):
    cm, cur = fake_cursor
    cur.rowcount = 0
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        ok = db.resolve_watchlist_item(
            item_id=999, resolution="dismissed", note="n",
            session_id=1, stage="ideation",
        )
    assert ok is False


def test_amend_strategy_rule_updates_in_place(fake_cursor):
    cm, cur = fake_cursor
    cur.rowcount = 1
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        ok = db.amend_strategy_rule(
            rule_id=48, new_rule_text="updated text",
            new_evidence="beat-rate now 61% over 24 samples",
        )
    assert ok is True
    sql = cur.execute.call_args[0][0]
    assert "UPDATE strategy_rules" in sql
    assert "retired_at" not in sql  # amend must NOT retire


def test_resolve_tool_def_shape():
    d = wl.RESOLVE_WATCHLIST_TOOL_DEF
    assert d["name"] == "resolve_watchlist_item"
    props = d["input_schema"]["properties"]
    assert set(props) == {"item_id", "resolution", "note"}
    assert props["resolution"]["enum"] == ["acted", "dismissed"]
    assert set(d["input_schema"]["required"]) == {"item_id", "resolution", "note"}


def test_make_resolve_handler_binds_session_and_stage():
    with patch("v2.watchlist.db.resolve_watchlist_item", return_value=True) as m:
        handler = wl.make_resolve_handler(session_id=42, stage="reflection")
        out = handler(item_id=5, resolution="acted", note="retired Rule 43")
    m.assert_called_once_with(
        item_id=5, resolution="acted", note="retired Rule 43",
        session_id=42, stage="reflection",
    )
    assert "5" in out and "acted" in out


def test_make_resolve_handler_reports_miss():
    with patch("v2.watchlist.db.resolve_watchlist_item", return_value=False):
        handler = wl.make_resolve_handler(session_id=1, stage="ideation")
        out = handler(item_id=99, resolution="dismissed", note="n")
    assert "Error" in out or "not found" in out.lower()


def test_format_open_watchlist_items_empty():
    assert "no open" in wl.format_open_watchlist_items([]).lower()


def test_format_open_watchlist_items_lists_each():
    items = [
        {"id": 1, "title": "Retire Rule 43", "detail": "auto-lift fired"},
        {"id": 2, "title": "Close thesis 267", "detail": "Rule 43 grounded"},
    ]
    out = wl.format_open_watchlist_items(items)
    assert "item 1" in out and "Retire Rule 43" in out
    assert "item 2" in out and "Close thesis 267" in out
    assert "resolve_watchlist_item" in out


def test_assert_watchlist_resolved_raises_when_open_remain():
    with (
        patch("v2.watchlist.db.get_open_watchlist_items",
              return_value=[{"id": 3, "title": "x", "detail": "y"}]),
        pytest.raises(RuntimeError) as exc,
    ):
        wl.assert_watchlist_resolved("reflection")
    assert "3" in str(exc.value)


def test_assert_watchlist_resolved_passes_when_clear():
    with patch("v2.watchlist.db.get_open_watchlist_items", return_value=[]):
        wl.assert_watchlist_resolved("ideation")  # no raise


def test_get_open_items_delegates_to_db():
    with patch("v2.watchlist.db.get_open_watchlist_items", return_value=[{"id": 1}]) as m:
        out = wl.get_open_items("reflection")
    m.assert_called_once_with("reflection")
    assert out == [{"id": 1}]
