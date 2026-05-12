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
