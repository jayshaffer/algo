CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL,
    session_type VARCHAR(50) NOT NULL DEFAULT 'daily',
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT,
    UNIQUE (session_date, session_type)
);
