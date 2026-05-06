-- db/init/025_audit_findings.sql
-- Self-healing audit: track integrity, rule-overfitting, and app-issue findings.
-- See docs/superpowers/specs/2026-05-06-self-healing-audit-design.md

CREATE TABLE IF NOT EXISTS audit_runs (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    mode            VARCHAR(16) NOT NULL CHECK (mode IN ('check','apply')),
    total_findings  INTEGER NOT NULL DEFAULT 0,
    auto_fixed      INTEGER NOT NULL DEFAULT 0,
    failed_checks   INTEGER NOT NULL DEFAULT 0,
    model           VARCHAR(64),
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    cache_creation_tokens INTEGER,
    cache_read_tokens     INTEGER
);

CREATE TABLE IF NOT EXISTS audit_findings (
    id              SERIAL PRIMARY KEY,
    audit_run_id    INTEGER NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
    check_code      VARCHAR(64) NOT NULL,
    tier            SMALLINT NOT NULL CHECK (tier IN (1,2,3)),
    severity        VARCHAR(16) NOT NULL CHECK (severity IN ('critical','warn','info')),
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    affected_count  INTEGER NOT NULL DEFAULT 0,
    evidence        JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          VARCHAR(16) NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open','auto_fixed','acknowledged','resolved','superseded')),
    fingerprint     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ,
    resolved_note   TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_findings_status   ON audit_findings(status) WHERE status='open';
CREATE INDEX IF NOT EXISTS idx_audit_findings_run      ON audit_findings(audit_run_id);
CREATE INDEX IF NOT EXISTS idx_audit_findings_code     ON audit_findings(check_code);
CREATE UNIQUE INDEX IF NOT EXISTS uq_audit_findings_open_fingerprint
    ON audit_findings(fingerprint) WHERE status='open';
