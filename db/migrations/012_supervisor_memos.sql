-- 012_supervisor_memos.sql (mirror of db/init/033)
-- Strategy supervisor memos. Observer-only critic output, one row per run.

CREATE TABLE IF NOT EXISTS supervisor_memos (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    content         TEXT,
    status          TEXT NOT NULL CHECK (status IN ('ok', 'max_turns', 'error')),
    turns_used      INT NOT NULL,
    tool_calls      JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_tokens    INT,
    output_tokens   INT,
    cost_usd        NUMERIC(10, 4),
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_supervisor_memos_created_at
    ON supervisor_memos (created_at DESC);

COMMENT ON TABLE supervisor_memos IS
    'Strategy supervisor critic output. One row per `python -m v2.supervisor` run.';
COMMENT ON COLUMN supervisor_memos.content IS
    'Markdown body. NULL when status != ok.';
COMMENT ON COLUMN supervisor_memos.tool_calls IS
    'Compact per-tool call counts, e.g. [{"name":"get_active_rules","count":2}, ...]';
