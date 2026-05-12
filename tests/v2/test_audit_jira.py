"""Tests for v2/audit_jira.py — Jira ticket filing for Opus ideation findings."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def jira_env(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    monkeypatch.setenv("JIRA_AUDIT_PROJECT_KEY", "ALGO")


class TestJiraConfig:
    def test_config_raises_when_any_var_missing(self, monkeypatch):
        monkeypatch.delenv("JIRA_BASE_URL", raising=False)
        monkeypatch.setenv("JIRA_EMAIL", "x")
        monkeypatch.setenv("JIRA_API_TOKEN", "y")
        monkeypatch.setenv("JIRA_AUDIT_PROJECT_KEY", "Z")
        from v2.audit_jira import _config, JiraConfigMissing
        with pytest.raises(JiraConfigMissing) as excinfo:
            _config()
        assert "JIRA_BASE_URL" in str(excinfo.value)

    def test_config_returns_all_settings(self, jira_env):
        from v2.audit_jira import _config
        cfg = _config()
        assert cfg["base_url"] == "https://example.atlassian.net"
        assert cfg["email"] == "user@example.com"
        assert cfg["token"] == "tok"
        assert cfg["project_key"] == "ALGO"
        assert cfg["issue_type"] == "Task"  # default

    def test_config_honors_issue_type_override(self, jira_env, monkeypatch):
        monkeypatch.setenv("JIRA_AUDIT_ISSUE_TYPE", "Story")
        from v2.audit_jira import _config
        assert _config()["issue_type"] == "Story"

    def test_config_strips_trailing_slash_from_base_url(self, monkeypatch):
        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net/")
        monkeypatch.setenv("JIRA_EMAIL", "u")
        monkeypatch.setenv("JIRA_API_TOKEN", "t")
        monkeypatch.setenv("JIRA_AUDIT_PROJECT_KEY", "ALGO")
        from v2.audit_jira import _config
        assert _config()["base_url"] == "https://example.atlassian.net"


class TestFindExistingIssue:
    def test_returns_key_on_match(self, jira_env):
        from v2 import audit_jira
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"issues": [{"key": "ALGO-7"}]}
        with patch("v2.audit_jira.requests.get", return_value=fake_resp) as mock_get:
            result = audit_jira.find_existing_issue("abc123fingerprint")
        assert result == "ALGO-7"
        params = mock_get.call_args.kwargs["params"]
        # JQL contains project key and fingerprint-label clause
        assert 'project = "ALGO"' in params["jql"]
        assert 'labels = "audit-fingerprint:abc123fingerprint"' in params["jql"]
        assert "statusCategory != Done" in params["jql"]

    def test_returns_none_on_empty(self, jira_env):
        from v2 import audit_jira
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"issues": []}
        with patch("v2.audit_jira.requests.get", return_value=fake_resp):
            assert audit_jira.find_existing_issue("nomatch") is None

    def test_uses_basic_auth(self, jira_env):
        from v2 import audit_jira
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"issues": []}
        with patch("v2.audit_jira.requests.get", return_value=fake_resp) as mock_get:
            audit_jira.find_existing_issue("fp")
        assert mock_get.call_args.kwargs["auth"] == ("user@example.com", "tok")

    def test_raises_on_5xx(self, jira_env):
        """Per spec, JQL search failures bubble up to caller (file_jira_ticket
        catches them downstream)."""
        from v2 import audit_jira
        import requests as _requests
        fake_resp = MagicMock(status_code=500)
        fake_resp.raise_for_status.side_effect = _requests.HTTPError("500")
        with patch("v2.audit_jira.requests.get", return_value=fake_resp):
            with pytest.raises(_requests.HTTPError):
                audit_jira.find_existing_issue("fp")


def _make_finding(check_code="APP_IMPROVEMENT", category="app_improvement",
                  topic_slug="add-regime-detector", title="Add a regime detector",
                  priority="high", body="Detect bull/bear regimes from SPY."):
    from v2.audit import Finding, _opus_topic_fingerprint
    f = Finding(
        check_code=check_code, tier=3, severity="info",
        title=title, body=body, affected_count=1,
        evidence={
            "topic_slug": topic_slug, "category": category,
            "priority": priority,
            "evidence_quote": "evidence here",
        },
        auto_fix=None,
    )
    # Pre-compute the fingerprint as the runner would
    f._test_fingerprint = _opus_topic_fingerprint(check_code, topic_slug)
    return f


class TestFileJiraTicket:
    def test_dedup_hit_skips_create(self, jira_env):
        import json
        from v2 import audit_jira
        f = _make_finding()
        with patch.object(audit_jira, "find_existing_issue", return_value="ALGO-9"), \
             patch.object(audit_jira.requests, "post") as mock_post:
            result = audit_jira.file_jira_ticket(f, fingerprint=f._test_fingerprint, run_id=1)
        assert result == {"status": "existing", "issue_key": "ALGO-9"}
        assert not mock_post.called

    def test_creates_when_no_dedup(self, jira_env):
        import json
        from v2 import audit_jira
        f = _make_finding()
        fake_resp = MagicMock(status_code=201)
        fake_resp.json.return_value = {"key": "ALGO-42"}
        fake_resp.raise_for_status.return_value = None
        with patch.object(audit_jira, "find_existing_issue", return_value=None), \
             patch.object(audit_jira.requests, "post", return_value=fake_resp) as mock_post:
            result = audit_jira.file_jira_ticket(f, fingerprint=f._test_fingerprint, run_id=99)
        assert result == {"status": "created", "issue_key": "ALGO-42"}

        payload = json.loads(mock_post.call_args.kwargs["data"])
        fields = payload["fields"]
        assert fields["project"]["key"] == "ALGO"
        assert fields["issuetype"]["name"] == "Task"
        assert fields["summary"].startswith("[audit:app_improvement]")
        assert "Add a regime detector" in fields["summary"]
        labels = fields["labels"]
        assert f"audit-fingerprint:{f._test_fingerprint}" in labels
        assert "audit-source:opus-ideation" in labels
        assert "audit-category:app_improvement" in labels
        assert fields["priority"]["name"] == "High"
        assert "run #99" in fields["description"]
        # Description includes evidence quote and fingerprint
        assert "evidence here" in fields["description"]
        assert f._test_fingerprint in fields["description"]

    def test_summary_capped_at_250(self, jira_env):
        import json
        from v2 import audit_jira
        f = _make_finding(title="X" * 500)
        fake_resp = MagicMock(status_code=201)
        fake_resp.json.return_value = {"key": "ALGO-1"}
        fake_resp.raise_for_status.return_value = None
        with patch.object(audit_jira, "find_existing_issue", return_value=None), \
             patch.object(audit_jira.requests, "post", return_value=fake_resp) as mock_post:
            audit_jira.file_jira_ticket(f, fingerprint=f._test_fingerprint, run_id=1)
        payload = json.loads(mock_post.call_args.kwargs["data"])
        assert len(payload["fields"]["summary"]) <= 250

    def test_priority_omitted_when_invalid(self, jira_env):
        """If finding's priority isn't in PRIORITY_MAP, omit the priority field
        rather than send something Jira will reject."""
        import json
        from v2 import audit_jira
        f = _make_finding(priority="garbage")
        fake_resp = MagicMock(status_code=201)
        fake_resp.json.return_value = {"key": "ALGO-1"}
        fake_resp.raise_for_status.return_value = None
        with patch.object(audit_jira, "find_existing_issue", return_value=None), \
             patch.object(audit_jira.requests, "post", return_value=fake_resp) as mock_post:
            audit_jira.file_jira_ticket(f, fingerprint=f._test_fingerprint, run_id=1)
        payload = json.loads(mock_post.call_args.kwargs["data"])
        assert "priority" not in payload["fields"]

    def test_500_returns_failed(self, jira_env):
        import requests as _req
        from v2 import audit_jira
        f = _make_finding()
        fake_resp = MagicMock(status_code=500, text="boom")
        fake_resp.raise_for_status.side_effect = _req.HTTPError("500")
        with patch.object(audit_jira, "find_existing_issue", return_value=None), \
             patch.object(audit_jira.requests, "post", return_value=fake_resp):
            result = audit_jira.file_jira_ticket(f, fingerprint=f._test_fingerprint, run_id=1)
        assert result["status"] == "failed"
        assert "500" in result["error"]

    def test_dedup_search_failure_returns_failed(self, jira_env):
        """If the JQL dedup search itself fails, we should still not raise."""
        from v2 import audit_jira
        f = _make_finding()
        with patch.object(audit_jira, "find_existing_issue", side_effect=RuntimeError("oops")):
            result = audit_jira.file_jira_ticket(f, fingerprint=f._test_fingerprint, run_id=1)
        assert result["status"] == "failed"
        assert "dedup_search" in result["error"]
        assert "oops" in result["error"]

    def test_missing_env_returns_disabled(self, monkeypatch):
        monkeypatch.delenv("JIRA_BASE_URL", raising=False)
        from v2 import audit_jira
        f = _make_finding()
        result = audit_jira.file_jira_ticket(f, fingerprint="abc", run_id=1)
        assert result == {"status": "disabled", "reason": "config_missing"}
