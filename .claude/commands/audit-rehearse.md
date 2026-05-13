---
description: Dry-run one audit Phase A discovery tick (no Jira writes — prints would-be payloads)
---

You are about to rehearse one tick of audit discovery. DO NOT call any MCP write tool (`mcp__atlassian__createJiraIssue`, `mcp__atlassian__transitionJiraIssue`, `mcp__atlassian__addCommentToJiraIssue`). Read MCP tools (`mcp__atlassian__searchJiraIssuesUsingJql`) are allowed.

Follow the instructions in `.claude/commands/audit-discover.md` with one substitution: instead of calling `mcp__atlassian__createJiraIssue`, print the full payload you WOULD have sent as JSON, with a heading like:

```
DRY RUN — would file:
<JSON>
```

Run every deterministic check. Run the ideation pass. Apply the 5-create cap. Report what you found.
