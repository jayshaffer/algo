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


import json

PRIORITY_MAP = {"high": "High", "medium": "Medium", "low": "Low"}


def _build_description(finding, run_id: int, fingerprint: str) -> str:
    evidence = finding.evidence or {}
    slug = evidence.get("topic_slug", "")
    quote = evidence.get("evidence_quote", "")
    lines = [finding.body or ""]
    if quote:
        lines.append("")
        lines.append("**Evidence:**")
        lines.append(f"> {quote}")
    lines.append("")
    lines.append("---")
    lines.append(
        f"Filed by audit run #{run_id}. Topic: `{slug}`. Fingerprint: `{fingerprint}`."
    )
    return "\n".join(lines)


def _build_create_payload(finding, run_id: int, fingerprint: str, cfg: dict) -> dict:
    category = (finding.evidence or {}).get("category", "app_improvement")
    priority = (finding.evidence or {}).get("priority", "medium")
    summary = f"[audit:{category}] {finding.title}"[:250]
    fields: dict = {
        "project": {"key": cfg["project_key"]},
        "issuetype": {"name": cfg["issue_type"]},
        "summary": summary,
        "description": _build_description(finding, run_id, fingerprint),
        "labels": [
            f"audit-fingerprint:{fingerprint}",
            "audit-source:opus-ideation",
            f"audit-category:{category}",
        ],
    }
    if priority in PRIORITY_MAP:
        fields["priority"] = {"name": PRIORITY_MAP[priority]}
    return {"fields": fields}


def file_jira_ticket(finding, *, fingerprint: str, run_id: int) -> dict:
    """File a Jira ticket for an Opus ideation finding.

    Returns a status dict suitable for stashing in finding.evidence['jira']:
        {"status": "existing"|"created"|"failed"|"disabled", ...}.
    Never raises — all exceptions are caught and recorded.
    """
    try:
        cfg = _config()
    except JiraConfigMissing as exc:
        log.info("Jira filing disabled: %s", exc)
        return {"status": "disabled", "reason": "config_missing"}

    try:
        existing = find_existing_issue(fingerprint)
    except Exception as exc:
        log.exception("Jira dedup search failed for %s", fingerprint)
        return {"status": "failed", "error": f"dedup_search: {exc}"}

    if existing:
        return {"status": "existing", "issue_key": existing}

    payload = _build_create_payload(finding, run_id, fingerprint, cfg)
    try:
        resp = requests.post(
            f"{cfg['base_url']}/rest/api/3/issue",
            data=json.dumps(payload),
            auth=_auth(cfg),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.exception("Jira create failed for fingerprint=%s", fingerprint)
        return {"status": "failed", "error": str(exc)}

    body = resp.json() if resp.content else {}
    issue_key = body.get("key") or ""
    return {"status": "created", "issue_key": issue_key}
