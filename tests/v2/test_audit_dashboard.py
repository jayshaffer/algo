"""Tests for the dashboard /audit routes.

Coexists with tests/test_dashboard.py which also imports dashboard.app —
both files must reach the SAME queries mock that app.py bound at first
import. We do that by patching the names already bound on dashboard.app,
not by injecting a fresh sys.modules["queries"].
"""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Inject benchmark mock if no other test file beat us to it (dashboard.app
# imports from `benchmark` at module top).
if "benchmark" not in sys.modules:
    mock_benchmark = MagicMock()
    mock_benchmark.get_spy_benchmark.return_value = []
    mock_benchmark.compute_alpha.return_value = None
    mock_benchmark.get_deposit_history.return_value = []
    mock_benchmark.enrich_snapshots_with_deposits.side_effect = lambda snaps, deps: list(snaps)
    sys.modules["benchmark"] = mock_benchmark

# Also inject a queries mock if it's not already there (tests/test_dashboard.py
# usually wins — but if this file runs alone, we still need one).
if "queries" not in sys.modules:
    sys.modules["queries"] = MagicMock()

from dashboard.app import app  # noqa: E402


@pytest.fixture
def app_client():
    app.config["TESTING"] = True
    return app.test_client()


def test_audit_page_renders(app_client):
    findings = [
        {"id": 1, "audit_run_id": 7, "check_code": "ORPHAN_FK_NEWS_SIGNAL",
         "tier": 1, "severity": "warn", "title": "11 orphan news_signal refs",
         "affected_count": 11, "created_at": datetime(2026, 5, 6, 22, 30),
         "evidence": {}},
    ]
    with patch("dashboard.app.get_open_audit_findings", return_value=findings), \
         patch("dashboard.app.get_recent_audit_runs", return_value=[]):
        response = app_client.get("/audit")
    assert response.status_code == 200
    assert b"ORPHAN_FK_NEWS_SIGNAL" in response.data
    assert b"11 orphan news_signal refs" in response.data


def test_audit_finding_detail_renders(app_client):
    finding = {
        "id": 1, "check_code": "X", "title": "t", "tier": 1, "severity": "warn",
        "body": "b", "affected_count": 0, "status": "open", "evidence": {"k": "v"},
    }
    with patch("dashboard.app.get_audit_finding", return_value=finding):
        response = app_client.get("/audit/findings/1")
    assert response.status_code == 200
    assert b"\"k\": \"v\"" in response.data


def test_audit_finding_detail_renders_jira_link(app_client):
    """Findings with evidence.jira.issue_key render a clickable Jira link."""
    finding = {
        "id": 1, "check_code": "APP_IMPROVEMENT", "title": "Add regime detector",
        "tier": 3, "severity": "info", "body": "...",
        "affected_count": 1, "status": "open",
        "evidence": {
            "topic_slug": "add-regime-detector",
            "category": "app_improvement",
            "jira": {"status": "created", "issue_key": "ALGO-42"},
        },
    }
    with patch("dashboard.app.get_audit_finding", return_value=finding):
        response = app_client.get("/audit/findings/1")
    assert response.status_code == 200
    assert b"ALGO-42" in response.data
    # Anchor target should reference the issue key
    assert b"/browse/ALGO-42" in response.data


def test_audit_finding_detail_renders_jira_disabled_status(app_client):
    """Findings with evidence.jira but no issue_key show status text."""
    finding = {
        "id": 2, "check_code": "AUDIT_GAP", "title": "Missing foo",
        "tier": 3, "severity": "info", "body": "...",
        "affected_count": 1, "status": "open",
        "evidence": {"topic_slug": "missing-foo",
                     "jira": {"status": "disabled"}},
    }
    with patch("dashboard.app.get_audit_finding", return_value=finding):
        response = app_client.get("/audit/findings/2")
    assert response.status_code == 200
    assert b"disabled" in response.data


def test_audit_finding_status_post_updates(app_client):
    with patch("dashboard.app.update_audit_finding_status") as mock_update:
        response = app_client.post("/audit/findings/1/status",
                                   data={"status": "acknowledged", "note": "looking"})
    assert response.status_code == 302
    mock_update.assert_called_once_with(1, "acknowledged", "looking")
