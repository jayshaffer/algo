---
description: Run one Phase A audit discovery tick (deterministic checks + ideation + Jira filing)
---

Read `docs/audit-playbook.md`. It is the catalog of audit checks and the ideation-pass instructions.

## Step 1 — Run checks

Execute every entry under **Deterministic checks** in order. Honor the `env` field for each (`prod` / `paper` / `both`; for `both`, run twice and tag env in the title prefix).

Then run the **Ideation pass** per its instructions. Cap ideation findings at 3 per invocation.

## Step 2 — File findings

For every finding (deterministic or ideation), apply these rules:

### Fingerprint

```
fingerprint = sha256(check_code + ":" + topic_slug).hexdigest()[:16]
```

### Dedup search

Before creating, search Jira via the Atlassian MCP:

```
project = ALGO AND labels = "audit-fingerprint:<fingerprint>"
```

No `statusCategory` filter — a closed (Done / Won't Fix) ticket also suppresses re-filing. If any issue comes back, skip this finding. Do not file, comment, or transition.

### Create

If no dedup hit, file via `mcp__atlassian__createJiraIssue`:

- **project:** `ALGO`
- **summary:** `[audit:<category>] <title_template-rendered>` (prepend `:<env>` if env is `both`)
- **issue type:** `Task`
- **labels** (all of):
  - `audit-source:claude`
  - `audit-fingerprint:<fingerprint>`
  - `audit-category:<category>`
  - `audit-worktype:<code|db>`
- **priority:** `Medium` by default. `High` for `severity: critical`. `Low` for `severity: info`.
- **description (ADF):**
  - Paragraph: rendered body_template
  - Blockquote: top-of-evidence quote (first 500 chars of the SQL result or the ideation source excerpt)
  - Horizontal rule
  - Paragraph: `Filed by /loop audit tick on <YYYY-MM-DD>. Fingerprint: <fingerprint>. Suggested fix:` + (`<suggested_fix_sql>` for db / `<suggested_fix_text>` for code)

### Cap

Max **5** creates per invocation across deterministic + ideation. Remaining findings wait until the next invocation.

## Reference

Design rationale and lifecycle examples: `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`.
