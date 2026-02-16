-- 007_strategy.sql: Persistent strategy layer tables

CREATE TABLE IF NOT EXISTS strategy_state (
    id SERIAL PRIMARY KEY,
    identity_text TEXT NOT NULL,
    risk_posture VARCHAR(20) NOT NULL DEFAULT 'moderate',
    sector_biases JSONB NOT NULL DEFAULT '{}',
    preferred_signals JSONB NOT NULL DEFAULT '[]',
    avoided_signals JSONB NOT NULL DEFAULT '[]',
    version INT NOT NULL DEFAULT 1,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_strategy_state_current ON strategy_state(is_current) WHERE is_current = TRUE;

CREATE TABLE IF NOT EXISTS strategy_rules (
    id SERIAL PRIMARY KEY,
    rule_text TEXT NOT NULL,
    category VARCHAR(64) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence DECIMAL NOT NULL DEFAULT 0.5,
    supporting_evidence TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    retired_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_strategy_rules_status ON strategy_rules(status);

CREATE TABLE IF NOT EXISTS strategy_memos (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL,
    memo_type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    strategy_state_id INT REFERENCES strategy_state(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_strategy_memos_date ON strategy_memos(session_date DESC);
