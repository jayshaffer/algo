-- db/migrations/004_audit_llm_calls.sql
-- Per-LLM-call token accounting for audit runs.
-- See docs/superpowers/specs/2026-05-12-opus-audit-ideation-design.md
--
-- mirror-check: skip — superseded by 005_drop_audit_tables.sql. Kept only as
-- applied history. It references audit_runs, a table no longer created
-- anywhere, so it cannot be replayed; db/check_mirror.sh skips it rather than
-- reporting an unfixable non-idempotency.

CREATE TABLE IF NOT EXISTS audit_llm_calls (
    id                      SERIAL PRIMARY KEY,
    audit_run_id            INTEGER NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
    purpose                 VARCHAR(64) NOT NULL,
    model                   VARCHAR(128) NOT NULL,
    input_tokens            INTEGER NOT NULL DEFAULT 0,
    output_tokens           INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens       INTEGER NOT NULL DEFAULT 0,
    latency_ms              INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_llm_calls_run     ON audit_llm_calls(audit_run_id);
CREATE INDEX IF NOT EXISTS idx_audit_llm_calls_purpose ON audit_llm_calls(purpose, created_at);
