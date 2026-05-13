-- db/migrations/006_llm_call_contexts.sql
-- Per-turn LLM request/response log for the strategist, executor, and
-- reflection loops. Gated by AgentPurpose so classifier calls don't
-- flood this table.
-- See docs/superpowers/specs/2026-05-13-llm-context-logging-design.md

CREATE TABLE IF NOT EXISTS llm_call_contexts (
    id                      BIGSERIAL PRIMARY KEY,
    session_id              INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    stage_name              TEXT NOT NULL,
    purpose                 TEXT NOT NULL,
    sequence                INTEGER NOT NULL,
    model                   TEXT NOT NULL,
    system_prompt           TEXT,
    messages                JSONB NOT NULL,
    tool_definitions        JSONB,
    response_content        JSONB,
    input_tokens            INTEGER,
    output_tokens           INTEGER,
    cache_read_tokens       INTEGER,
    cache_creation_tokens   INTEGER,
    stop_reason             TEXT,
    duration_ms             INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, stage_name, purpose, sequence)
);

CREATE INDEX IF NOT EXISTS idx_llm_call_contexts_session_stage
    ON llm_call_contexts (session_id, stage_name);
CREATE INDEX IF NOT EXISTS idx_llm_call_contexts_created_at
    ON llm_call_contexts (created_at);
