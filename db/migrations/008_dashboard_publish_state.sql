-- db/migrations/008_dashboard_publish_state.sql
-- Persistent dashboard publish state for incremental public changelog ranges.

CREATE TABLE IF NOT EXISTS dashboard_publish_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dashboard_changelog_commits (
    sha TEXT PRIMARY KEY,
    short_sha TEXT NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL,
    subject TEXT NOT NULL,
    body TEXT,
    files JSONB NOT NULL DEFAULT '[]'::jsonb,
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dashboard_changelog_commits_committed_at
    ON dashboard_changelog_commits(committed_at DESC);

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
