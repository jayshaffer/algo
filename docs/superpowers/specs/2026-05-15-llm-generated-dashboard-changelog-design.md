# LLM-Generated Public Dashboard Changelog

**Date:** 2026-05-15
**Status:** Design proposal.

## Goal

Publish a readable public changelog on the dashboard without making visitors
read raw commit subjects, and without scanning the full git history on every
dashboard publish.

The changelog should be generated from repository history, summarized by an
LLM into public-facing notes, and backed by durable database rows so every
dashboard deploy can render a stable changelog cheaply.

## Non-Goals

- Do not run an unbounded `git log` during normal dashboard publishing.
- Do not make the dashboard publish fail just because changelog summarization
  fails.
- Do not publish every commit as a separate user-facing entry.
- Do not advance the changelog pointer until the Cloudflare Pages deploy has
  succeeded.
- Do not rely only on LLM prose as the audit trail. Raw commit facts must also
  be stored.

## Data Model

### `dashboard_publish_state`

Stores small persistent publish pointers.

```sql
CREATE TABLE IF NOT EXISTS dashboard_publish_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

The changelog uses:

| key | value |
|-----|-------|
| `changelog_last_published_sha` | Full git SHA last successfully included in a deployed dashboard |

### `dashboard_changelog_commits`

Stores raw git facts. This is the audit trail.

```sql
CREATE TABLE IF NOT EXISTS dashboard_changelog_commits (
    sha TEXT PRIMARY KEY,
    short_sha TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    subject TEXT NOT NULL,
    body TEXT,
    files JSONB NOT NULL DEFAULT '[]'::jsonb,
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### `dashboard_changelog_entries`

Stores public LLM-written changelog entries.

```sql
CREATE TABLE IF NOT EXISTS dashboard_changelog_entries (
    id BIGSERIAL PRIMARY KEY,
    from_sha TEXT,
    to_sha TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    bullets JSONB NOT NULL DEFAULT '[]'::jsonb,
    commit_shas JSONB NOT NULL DEFAULT '[]'::jsonb,
    model TEXT,
    prompt_version TEXT NOT NULL DEFAULT 'public_changelog_v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dashboard_changelog_entries_created_at
    ON dashboard_changelog_entries(created_at DESC);
```

`commit_shas` lets the public entry point back to the raw rows. The dashboard
can render each entry with a small SHA list or an expandable "commits" detail.

## Normal Publish Flow

`run_dashboard_stage()` should do the following:

1. Resolve current `HEAD`:

   ```bash
   git rev-parse HEAD
   ```

2. Read `changelog_last_published_sha` from `dashboard_publish_state`.

3. Fetch only the incremental range:

   ```bash
   git log --no-merges \
     --pretty=format:%cI%x1f%H%x1f%h%x1f%s%x1f%b \
     <last_sha>..<head_sha>
   ```

4. For each new commit, optionally fetch touched files:

   ```bash
   git show --name-only --pretty=format: <sha>
   ```

5. Insert raw commits into `dashboard_changelog_commits` with
   `ON CONFLICT (sha) DO NOTHING`.

6. If there are new commits, call the LLM summarizer. The LLM produces one to
   three public entries, grouped by meaning rather than one entry per commit.

7. Store LLM entries in `dashboard_changelog_entries`.

8. Render `/changelog/` from stored `dashboard_changelog_entries`. Include SHA
   pointers to the backing commits.

9. Deploy to Cloudflare Pages.

10. Only after deploy succeeds, update:

    ```sql
    INSERT INTO dashboard_publish_state (key, value, updated_at)
    VALUES ('changelog_last_published_sha', <head_sha>, NOW())
    ON CONFLICT (key) DO UPDATE SET
        value = EXCLUDED.value,
        updated_at = EXCLUDED.updated_at;
    ```

## First-Run Bootstrap

The first publish should not dump the entire repository history into the public
changelog.

Use a one-time bootstrap command, separate from the normal publish loop:

```bash
python -m scripts.bootstrap_dashboard_changelog --since 2026-05-01 --max-commits 300
```

Required controls:

- Require one of `--since`, `--from-sha`, or `--max-commits`.
- Dry-run by default and write a JSON preview.
- Support `--apply` to insert rows and set the pointer.
- Chunk large history before sending it to the LLM.
- Store raw commits and generated entries.
- Set `changelog_last_published_sha` to current `HEAD` only after the DB insert
  succeeds.

Recommended bootstrap behavior:

1. Read bounded historical commits.
2. Store raw commit rows.
3. Send chunks to the LLM for candidate entries.
4. Run a final consolidation pass that deduplicates and rewrites the entries.
5. Store final entries.
6. Set the pointer to current `HEAD`.

This gives the public dashboard an initial curated history while keeping the
daily publish loop incremental.

## LLM Prompt Contract

The summarizer should be strict JSON output, validated before storage.

Prompt:

```text
You are writing a public changelog for Pinchy, an AI trading dashboard.

Audience: people following a public AI-operated trading account.

Create public product notes from these repository commits.
Group related commits into meaningful entries.
Ignore internal-only cleanup unless it changes reliability, transparency,
safety, publishing, dashboard UX, or trading behavior.
Do not invent user-visible behavior.
Do not mention raw implementation details unless they explain a user-facing
change or safety improvement.

Return JSON only:
{
  "entries": [
    {
      "title": "Short public title",
      "summary": "One sentence summary",
      "bullets": ["Specific readable note", "Another note"],
      "commit_shas": ["full_sha"]
    }
  ]
}
```

Commit payload:

```json
[
  {
    "sha": "abc123...",
    "short_sha": "abc1234",
    "committed_at": "2026-05-15T14:37:49-06:00",
    "subject": "Validate executor response schema",
    "body": "",
    "files": ["v2/executor.py", "tests/v2/test_executor.py"]
  }
]
```

Validation:

- `entries` must be a list.
- Each entry must have `title`, `summary`, `bullets`, and `commit_shas`.
- `commit_shas` must be a subset of the commits in the batch.
- Empty or invalid LLM output falls back to raw commit table rendering.

## Public Rendering

The changelog page should render entries, not raw commits as the primary
experience.

Recommended shape:

| Date | Update | Backing commits |
|------|--------|-----------------|
| 2026-05-15 | Safer execution checks | `5776fb2` |

Each update should show:

- Title.
- One-sentence summary.
- Bullets, if present.
- Small SHA pointers to the commits behind the entry.

If no summarized entries exist, fall back to a simple raw commit table:

| Date | SHA | Change |
|------|-----|--------|
| 2026-05-15 | `5776fb2` | Validate executor response schema |

## Failure Behavior

- If git is unavailable, publish the dashboard with the last stored changelog.
- If the LLM call fails, store raw commits and render fallback raw commit rows.
- If storing changelog rows fails, log the error and publish the rest of the
  dashboard.
- If Cloudflare deploy fails, do not update `changelog_last_published_sha`.
- If pointer update fails after successful deploy, return `published=True` with
  a non-fatal error so the next run can retry the range. Raw commit inserts must
  be idempotent.

## Implementation Plan

1. Add/extend migrations for `dashboard_publish_state`,
   `dashboard_changelog_commits`, and `dashboard_changelog_entries`.
2. Add git helpers:
   - `get_current_git_sha()`
   - `fetch_changelog_commits(from_sha, to_sha)`
   - `fetch_commit_files(sha)`
3. Add DB helpers:
   - `get_changelog_pointer(cur)`
   - `store_changelog_commits(cur, commits)`
   - `store_changelog_entries(cur, entries, from_sha, to_sha, model)`
   - `get_recent_changelog_entries(cur)`
   - `update_changelog_pointer(cur, sha)`
4. Add `summarize_changelog_commits(commits)` using the existing LLM client
   path.
5. Wire the normal publish loop to store raw commits, summarize new commits,
   render stored entries, deploy, and then update the pointer.
6. Add a one-time bootstrap command with dry-run and apply modes.
7. Update `/changelog/` rendering to prefer summarized entries and fall back to
   raw commit rows.

## Test Coverage

- Git range helper uses `<last_sha>..<head_sha>` when the pointer exists.
- First-run bootstrap requires a bound (`--since`, `--from-sha`, or
  `--max-commits`).
- Raw commit inserts are idempotent.
- LLM output validation rejects malformed JSON and unknown SHAs.
- Normal publish updates the pointer only after `deploy_to_cloudflare()`
  succeeds.
- Deploy failure leaves the pointer unchanged.
- Existing stored entries render when there are no new commits.
- LLM failure falls back to raw commit rendering.
