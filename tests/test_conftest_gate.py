"""Tests for the conftest integration-logging gate.

The fixture used to gate on substring-checking the markexpr ('integration'
in markexpr), which spuriously fired on `-m "not integration"` because
the substring is present in both. The current gate inspects the items
pytest has already filtered for us, so any markexpr permutation works.
"""

from unittest.mock import MagicMock

from tests.conftest import _has_integration_tests


def _fake_session(items):
    session = MagicMock()
    session.items = items
    return session


def _fake_item(has_marker: bool):
    item = MagicMock()
    item.get_closest_marker.return_value = MagicMock() if has_marker else None
    return item


def test_returns_false_when_no_items_collected():
    """Empty test session — nothing to log for."""
    assert _has_integration_tests(_fake_session([])) is False


def test_returns_false_when_no_items_have_integration_marker():
    """Default `-m "not integration"` run filters integration tests out
    before the fixture sees session.items, so no item has the marker."""
    session = _fake_session([_fake_item(False), _fake_item(False)])
    assert _has_integration_tests(session) is False


def test_returns_true_when_any_item_has_integration_marker():
    """`-m integration` run leaves only marked items collected."""
    session = _fake_session([_fake_item(False), _fake_item(True)])
    assert _has_integration_tests(session) is True


def test_marker_lookup_uses_get_closest_marker():
    """We rely on `get_closest_marker('integration')` because it walks
    class/module-level markers too — `pytest.mark.integration` applied
    at the class level wouldn't show up in `item.own_markers`."""
    item = _fake_item(True)
    _has_integration_tests(_fake_session([item]))
    item.get_closest_marker.assert_called_with("integration")
