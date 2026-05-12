# Opus audit ideation — design

**Date:** 2026-05-12
**Status:** Approved (pre-implementation)
**Owner:** jay

## Goal

Extend `v2/audit.py` with two new daily LLM checks powered by Claude Opus 4.7
that propose, respectively, **gaps in the audit itself** and
**application/feature improvements** to the trading system. Proposals are
filed as Jira tickets so the user can triage them as a project manager.

This is additive to the existing audit. The existing 26 deterministic +
rule-judgment Haiku checks remain unchanged.

## Non-goals

- Replacing the existing Haiku rule-judgment check.
- Auto-implementing proposals (Opus only proposes; user picks up tickets).
- Cross-session memory for Opus (each call is stateless; daily inputs
  carry whatever continuity is needed).
- A new "Proposals" tab in the dashboard — Jira is the triage surface.

## Architecture

Two new check functions in `v2/audit.py`, appended to the `CHECKS` list:

| Check | Tier | Severity | Model |
|---|---|---|---|
| `check_audit_gaps_opus` | 3 | info | `claude-opus-4-7` |
| `check_app_improvements_opus` | 3 | info | `claude-opus-4-7` |

Both checks emit `Finding` objects into `audit_findings` via the existing
runner. They never produce `critical` findings — these are proposals, not
integrity violations, and must not gate the session.

A new module `v2/audit_jira.py` handles ticket filing via Jira REST. It is
called inline from the runner after each Opus finding is inserted, and is
gated by both a CLI flag (`--file-jira`) and presence of four env vars. Jira
failures are non-fatal and recorded in the finding's `evidence.jira` blob.

Prompt caching is used for the stable portions of each prompt (system prompt,
existing CHECK code list, DB schema summary). Variable inputs (recent
findings, session memos) sit outside the cache breakpoint. This buys a ~10x
cost reduction on the cached portions after the first call of the day.

Failure isolation already exists: the runner wraps each check in
`SAVEPOINT audit_check`, so a 5xx from Anthropic or Jira does not roll back
other checks' findings.

## Inputs

### `check_audit_gaps_opus`

The prompt assembles:

- The full `CHECKS` list (check function names) — what already exists.
- The 25+ `check_code` values currently in use, extracted from finding sources.
- Last 30 days of `audit_findings` aggregated: per `check_code`, count of
  distinct fingerprints and severity mix. Tells Opus what's been firing —
  clues to neighboring gaps.
- High-level DB schema: table names + column names for `decisions`,
  `decision_signals`, `news_signals`, `theses`, `strategy_rules`,
  `agent_events`, `agent_calls`, `sessions`, `session_stages`. Queried from
  `information_schema.columns`.
- Audit cost trend: last 14 audit runs' token usage and finding counts.

### `check_app_improvements_opus`

The prompt assembles:

- Last 14 `strategy_memos` (the system's own session reflection journal).
- Active strategy rules + rules retired in the last 30 days.
- Full `signal_attribution` snapshot.
- Last 14 days of decisions: ticker, side, notional, 7d/30d outcome,
  win/loss.
- Active theses + theses closed in the last 30 days with outcomes.
- Account snapshots last 30 days (equity trend).
- Module manifest: `v2/*.py` filenames + first-line docstring. **Not** the
  source code — keeps tokens bounded and avoids Opus over-anchoring on
  current implementation details. (Tunable knob: a future version may include
  a few key orchestrator modules verbatim if proposals are too vague.)

### Token budget

Both checks share `ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS` (default `60_000`). If a
call's assembled prompt would exceed the cap, the runner truncates the
lowest-priority section (oldest decisions / oldest memos), logs an
`OPUS_INPUT_TRUNCATED` warning finding, and proceeds with the smaller prompt.
At Opus pricing this is roughly $0.90 input per call worst-case, $1.80/day
total before output, before cache discounts.

## Output

Both checks share a single structured output shape (enforced via JSON parse +
schema validation):

```json
{
  "findings": [
    {
      "topic_slug": "kebab-case-stable-id",
      "title": "short, imperative phrasing",
      "category": "audit_gap",
      "priority": "high" | "medium" | "low",
      "body": "one to three short paragraphs",
      "evidence_quote": "specific data point that motivated this",
      "proposed_check_code": "OPTIONAL_CHECK_CODE"
    }
  ]
}
```

`category` is `audit_gap` for `check_audit_gaps_opus` and `app_improvement`
for `check_app_improvements_opus`. `proposed_check_code` is only used by the
gap check.

`topic_slug` is the load-bearing field for dedup. The system prompt
instructs Opus that it is a **stable identifier** and the same underlying
issue must produce the same slug on re-emission. Slugs are normalized
(lowercase, kebab-case, alphanumerics + hyphens only) before fingerprinting.

The system prompt instructs Opus to emit at most 10 findings. The runner
enforces a hard cap of 20 findings per call before parsing further — any
excess is dropped and a warning is logged. Empty `findings` array is the
expected default on days where Opus finds nothing defensible.

## Fingerprint & dedup

Existing audit checks hash `check_code + evidence`. Opus checks deliberately
diverge:

```
fingerprint = sha256(check_code + ":" + normalized_topic_slug)
```

The evidence prose is **not** part of the fingerprint, because Opus's wording
drifts day-to-day even when the underlying gap is the same. Without coarse
fingerprinting, the dashboard would clutter with daily near-duplicates.

This divergence is documented inline in `v2/audit.py` so future-me does not
"fix" it.

## Jira filing

A new module `v2/audit_jira.py` exposes `file_jira_ticket(finding, run_id)`.
The audit runner calls it once per Opus finding when filing is enabled.

### Gates (all must be true)

- `--file-jira` CLI flag present, OR `ALGO_AUDIT_FILE_JIRA=1`.
- All four of `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`,
  `JIRA_AUDIT_PROJECT_KEY` are set.
- Per-run create cap not yet reached: `ALGO_AUDIT_JIRA_MAX_CREATES`
  (default `5`). Dedup hits ("existing" status) do not count against the
  cap — only successful POSTs do.

### Flow per finding

1. Compute fingerprint (already done by `Finding`).
2. JQL search:
   `project = "$KEY" AND labels = "audit-fingerprint:<hash>" AND statusCategory != Done`.
   Single REST GET.
3. If an open issue exists → `evidence.jira = {"status": "existing", "issue_key": "..."}`.
   Skip create.
4. If no match → POST a new issue:
   - **Project:** `$JIRA_AUDIT_PROJECT_KEY`.
   - **Issue type:** `JIRA_AUDIT_ISSUE_TYPE` (default `Task`).
   - **Summary:** `[audit:<category>] <title>` (truncated to 250 chars).
   - **Description:** Opus body + an `evidence_quote` block + auto-footer:
     `Filed by audit run #N on YYYY-MM-DD. Topic: <topic_slug>. Fingerprint: <hash>.`
   - **Labels:** `audit-fingerprint:<hash>`,
     `audit-source:opus-ideation`,
     `audit-category:<audit_gap|app_improvement>`.
   - **Priority:** `high→High`, `medium→Medium`, `low→Low`. Omitted if the
     Jira project rejects the Priority field.
5. Record outcome in `evidence.jira`:
   `{"status": "created"|"failed"|"capped", "issue_key": "...", "error": "..."}`.

### Failure handling

Any non-2xx response, timeout, or unexpected exception is caught at the
`file_jira_ticket` boundary. The finding is still inserted into
`audit_findings` with the failure recorded in `evidence.jira`. The audit run
continues.

Once per-run cap is reached, remaining findings get
`evidence.jira = {"status": "capped"}` and no Jira call is made.

## LLM cost accounting

A new table replaces the per-run LLM columns currently on `audit_runs`:

```sql
CREATE TABLE audit_llm_calls (
  id              SERIAL PRIMARY KEY,
  audit_run_id    INTEGER NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
  purpose         TEXT NOT NULL,         -- 'rule_judgment' | 'audit_gaps' | 'app_improvements'
  model           TEXT NOT NULL,
  input_tokens    INTEGER NOT NULL DEFAULT 0,
  output_tokens   INTEGER NOT NULL DEFAULT 0,
  cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
  latency_ms      INTEGER,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_audit_llm_calls_run ON audit_llm_calls(audit_run_id);
```

Up to three rows per audit run (one Haiku rule-judgment + two Opus). The
existing `audit_runs.input_tokens` / `output_tokens` / `cache_*` columns are
left in place but stop being written on new runs.

A new check `check_audit_llm_cost_trend` is added (separate from the existing
`check_cost_trend`, which aggregates `session_stages` and is unrelated). The
new check compares last-7d vs prior-7d total tokens per `purpose` and emits
an info finding when any purpose's usage doubles. This is what catches an
Opus regression — an Opus spike cannot hide behind summed Haiku usage because
the breakdown is per `purpose`.

No backfill — historic per-call breakdown is unrecoverable.

## Configuration surface

New env vars, all documented in `CLAUDE.md`'s "Optional knobs" section:

| Var | Default | Purpose |
|---|---|---|
| `ALGO_AUDIT_OPUS_MAX_INPUT_TOKENS` | `60000` | Hard cap on prompt size per Opus call. |
| `ALGO_AUDIT_FILE_JIRA` | unset | If `1`, file Jira tickets without `--file-jira`. |
| `ALGO_AUDIT_JIRA_MAX_CREATES` | `5` | Per-run cap on new Jira tickets. |
| `JIRA_BASE_URL` | unset | e.g. `https://<workspace>.atlassian.net`. |
| `JIRA_EMAIL` | unset | Jira account email. |
| `JIRA_API_TOKEN` | unset | API token from id.atlassian.com. |
| `JIRA_AUDIT_PROJECT_KEY` | unset | e.g. `ALGO`. |
| `JIRA_AUDIT_ISSUE_TYPE` | `Task` | Issue type for filed tickets. |

Constants in `v2/audit.py`:

- `OPUS_IDEATION_MODEL = "claude-opus-4-7"`.
- `OPUS_IDEATION_MAX_TOKENS = 4000` (output cap).

## Testing

### `tests/v2/test_audit.py` additions

- `_call_opus_audit_gaps` / `_call_opus_app_improvements` stubbed via
  monkeypatch returning canned JSON. Verify parse → `Finding` mapping,
  including malformed JSON → empty list, missing required fields → finding
  skipped (logged, not raised).
- Truncation: input above `MAX_INPUT_TOKENS` triggers section drop +
  `OPUS_INPUT_TRUNCATED` warning finding.
- Fingerprint collision: two findings same `topic_slug` → identical
  fingerprint → second handled by `supersede_stale_open_findings` as today.
- `audit_llm_calls` rows: three when all LLM checks fire; one when only
  rule-judgment fires; zero when `ANTHROPIC_API_KEY` unset.

### `tests/v2/test_audit_jira.py` (new)

- Mock `requests.get` / `requests.post`. Verify:
  - JQL dedup hit → no POST, `evidence.jira.status == "existing"`.
  - Dedup miss → POST issued with expected payload, `status == "created"`.
  - 5xx on POST → `status == "failed"`, audit run continues.
- Cap behavior: 6 findings with cap=5 → 5 created, 1 capped, no exception.
- Missing env var → Jira step skipped silently, finding still written with
  `evidence.jira.status == "disabled"`.

### `tests/v2/test_audit_dashboard.py` additions

- Verify `audit_finding.html` renders the `evidence.jira` block when present
  (existing-issue link, created-issue link, failed-status badge).

## Dashboard changes

Minimal:

- `dashboard/templates/audit.html`: a small banner at the top of any finding
  whose `evidence.jira.issue_key` is set, linking to the Jira issue.
- `dashboard/templates/audit_finding.html`: render `evidence.jira` if
  present.
- No new route, no new tab. Triage UX lives in Jira.

## Rollout

Three independently shippable commits:

### Commit 1 — schema migration + LLM call accounting

- Add `db/init/026_audit_llm_calls.sql`.
- New helpers in `v2/database/trading_db.py`: `insert_audit_llm_call`,
  and a query that aggregates per-`purpose` 7d-vs-prior-7d token totals.
- Wire the existing Haiku `check_rule_judgment` to write a row.
- Stop writing `audit_runs.input_tokens` / etc. (leave the columns in place).
- Add new check `check_audit_llm_cost_trend` to `CHECKS` (does not modify
  the existing `check_cost_trend`, which reads `session_stages` and is
  unrelated).
- No new user-facing behavior — but new visibility on per-call costs.

### Commit 2 — Opus checks, no Jira

- `OPUS_IDEATION_MODEL`, prompt builders, JSON parsers, Finding mappers.
- Append `check_audit_gaps_opus` and `check_app_improvements_opus` to
  `CHECKS`.
- Findings get `evidence.jira = {"status": "disabled"}`.
- Lets the LLM piece run for several days against real data so we can
  eyeball whether outputs are useful before any side effects fire.

### Commit 3 — Jira filing

- `v2/audit_jira.py` with REST client + filing flow.
- `--file-jira` CLI flag + `ALGO_AUDIT_FILE_JIRA` env var.
- Dashboard template tweak for `evidence.jira` rendering.
- User flips on once the Opus output quality is validated.

## Open risks

- **Opus output drift over time.** Even with `topic_slug` dedup, slug
  formulation may drift (e.g. `add-regime-detector` vs `regime-detection-check`).
  Mitigation: include the existing open Jira issues' summaries in the prompt so
  Opus is anchored. Defer to a follow-up if it bites.
- **JQL injection via fingerprint.** Fingerprint is a hex sha256, so this is
  not a real risk — but the JQL builder must still escape the project key and
  label values defensively.
- **Cost spike on big sessions.** With caching and a 60k cap the worst case
  is roughly $5/day across both calls. If that's wrong, the
  `audit_llm_calls`-backed cost-trend check will surface it within days.
- **Module manifest is shallow.** Opus may propose features that already
  exist in modules it can't see. If proposals are consistently shallow after
  a week, the design says to pass key orchestrator source verbatim — that is
  the documented escape hatch, not a redesign.
