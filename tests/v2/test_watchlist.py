"""Unit tests for the supervisor watchlist DB helpers and shared module."""

from unittest.mock import MagicMock, patch

import pytest

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
