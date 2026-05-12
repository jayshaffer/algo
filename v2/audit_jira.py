# v2/audit_jira.py
"""Jira REST integration for Opus audit ideation findings.

Files tickets for new ideation findings; dedups against existing open issues
by a fingerprint label. Read by v2/audit.py's runner.

Spec: docs/superpowers/specs/2026-05-12-opus-audit-ideation-design.md
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

REQUEST_TIMEOUT_SEC = 15


class JiraConfigMissing(Exception):
    """Raised when one or more required Jira env vars are not set."""


def _config() -> dict:
    """Read Jira config from env. Raises JiraConfigMissing if anything's missing."""
    required = ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN",
                "JIRA_AUDIT_PROJECT_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise JiraConfigMissing(f"Missing env vars: {', '.join(missing)}")
    return {
        "base_url": os.environ["JIRA_BASE_URL"].rstrip("/"),
        "email": os.environ["JIRA_EMAIL"],
        "token": os.environ["JIRA_API_TOKEN"],
        "project_key": os.environ["JIRA_AUDIT_PROJECT_KEY"],
        "issue_type": os.environ.get("JIRA_AUDIT_ISSUE_TYPE", "Task"),
    }


def _auth(cfg: dict) -> tuple[str, str]:
    return (cfg["email"], cfg["token"])


def find_existing_issue(fingerprint: str) -> str | None:
    """JQL-search for an open Jira issue tagged with this audit fingerprint.

    Returns the issue key (e.g. 'ALGO-7') if found, else None.
    Raises JiraConfigMissing if env not configured; raises requests.HTTPError
    on non-2xx responses (callers catch this).
    """
    cfg = _config()
    jql = (
        f'project = "{cfg["project_key"]}" '
        f'AND labels = "audit-fingerprint:{fingerprint}" '
        f'AND statusCategory != Done'
    )
    resp = requests.get(
        f"{cfg['base_url']}/rest/api/3/search",
        params={"jql": jql, "fields": "summary", "maxResults": 1},
        auth=_auth(cfg),
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    issues = resp.json().get("issues") or []
    return issues[0]["key"] if issues else None
