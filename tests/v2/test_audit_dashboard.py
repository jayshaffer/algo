"""Tests for the dashboard /audit routes."""

import sys
from datetime import datetime
from unittest.mock import MagicMock

import pytest

# dashboard/app.py uses bare top-level imports (`from queries import ...`,
# `from benchmark import ...`). Inject mocks for both before importing.
mock_queries = MagicMock()
sys.modules["queries"] = mock_queries

mock_benchmark = MagicMock()
mock_benchmark.get_spy_benchmark.return_value = []
mock_benchmark.compute_alpha.return_value = None
mock_benchmark.get_deposit_history.return_value = []
mock_benchmark.enrich_snapshots_with_deposits.side_effect = lambda snaps, deps: list(snaps)
sys.modules["benchmark"] = mock_benchmark

from dashboard.app import app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_audit_query_mocks():
    """Reset child mocks individually (don't replace parent — bound refs)."""
    for attr in (
        "get_open_audit_findings", "get_audit_finding",
        "get_recent_audit_runs", "update_audit_finding_status",
    ):
        child = getattr(mock_queries, attr, None)
        if isinstance(child, MagicMock):
            child.reset_mock()
            child.side_effect = None


@pytest.fixture
def app_client():
    app.config["TESTING"] = True
    return app.test_client()


def test_audit_page_renders(app_client):
    mock_queries.get_open_audit_findings.return_value = [
        {"id": 1, "audit_run_id": 7, "check_code": "ORPHAN_FK_NEWS_SIGNAL",
         "tier": 1, "severity": "warn", "title": "11 orphan news_signal refs",
         "affected_count": 11, "created_at": datetime(2026, 5, 6, 22, 30),
         "evidence": {}},
    ]
    mock_queries.get_recent_audit_runs.return_value = []
    response = app_client.get("/audit")
    assert response.status_code == 200
    assert b"ORPHAN_FK_NEWS_SIGNAL" in response.data
    assert b"11 orphan news_signal refs" in response.data


def test_audit_finding_detail_renders(app_client):
    mock_queries.get_audit_finding.return_value = {
        "id": 1, "check_code": "X", "title": "t", "tier": 1, "severity": "warn",
        "body": "b", "affected_count": 0, "status": "open", "evidence": {"k": "v"},
    }
    response = app_client.get("/audit/findings/1")
    assert response.status_code == 200
    assert b"\"k\": \"v\"" in response.data


def test_audit_finding_status_post_updates(app_client):
    response = app_client.post("/audit/findings/1/status",
                               data={"status": "acknowledged", "note": "looking"})
    assert response.status_code == 302
    mock_queries.update_audit_finding_status.assert_called_once_with(
        1, "acknowledged", "looking"
    )
