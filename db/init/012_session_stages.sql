CREATE TABLE IF NOT EXISTS session_stages (
    id SERIAL PRIMARY KEY,
    session_id INT REFERENCES sessions(id),
    stage_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'running',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT,
    UNIQUE (session_id, stage_name)
);
CREATE INDEX IF NOT EXISTS idx_session_stages_session ON session_stages(session_id);
