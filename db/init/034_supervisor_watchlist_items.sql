-- 034_supervisor_watchlist_items.sql
-- Structured watchlist emitted by the strategy supervisor. Each item is
-- owned by exactly one acting stage, which must resolve it (acted/dismissed)
-- before that stage may complete. See spec
-- docs/superpowers/specs/2026-05-27-supervisor-watchlist-closed-loop-design.md

CREATE TABLE IF NOT EXISTS supervisor_watchlist_items (
    id                      SERIAL PRIMARY KEY,
    source_memo_id          INT NOT NULL REFERENCES supervisor_memos(id),
    title                   TEXT NOT NULL,
    detail                  TEXT NOT NULL,
    owner_stage             TEXT NOT NULL CHECK (owner_stage IN ('ideation', 'reflection')),
    status                  TEXT NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open', 'acted', 'dismissed')),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at             TIMESTAMPTZ,
    resolution_note         TEXT,
    resolved_by_session_id  INT REFERENCES sessions(id),
    resolved_by_stage       TEXT CHECK (resolved_by_stage IS NULL OR resolved_by_stage IN ('ideation', 'reflection'))
);

CREATE INDEX IF NOT EXISTS idx_watchlist_open_by_stage
    ON supervisor_watchlist_items (owner_stage, status);

COMMENT ON TABLE supervisor_watchlist_items IS
    'Structured supervisor watchlist. One row per item the supervisor flags; resolved by the owning acting stage.';
COMMENT ON COLUMN supervisor_watchlist_items.owner_stage IS
    'Which acting stage must resolve this: ideation (theses/playbook) or reflection (rules/identity).';
COMMENT ON COLUMN supervisor_watchlist_items.status IS
    'open until resolved; acted = a change was made; dismissed = reasoned no-op.';
