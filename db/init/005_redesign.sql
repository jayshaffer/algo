-- Redesign migration: playbooks, decision_signals, signal_attribution

-- New: Daily trading plan from Claude strategist to Ollama executor
CREATE TABLE IF NOT EXISTS playbooks (
    id              SERIAL PRIMARY KEY,
    date            DATE UNIQUE,
    market_outlook  TEXT,
    priority_actions JSONB,
    watch_list      TEXT[],
    risk_notes      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_playbooks_date ON playbooks(date);

-- New: Links decisions to the signals/theses that motivated them
CREATE TABLE IF NOT EXISTS decision_signals (
    decision_id  INT REFERENCES decisions(id),
    signal_type  TEXT NOT NULL,        -- 'news_signal', 'macro_signal', 'thesis'
    signal_id    INT NOT NULL,
    PRIMARY KEY (decision_id, signal_type, signal_id)
);

CREATE INDEX idx_decision_signals_signal ON decision_signals(signal_type, signal_id);

-- New: Precomputed scores showing which signal types are predictive
CREATE TABLE IF NOT EXISTS signal_attribution (
    id              SERIAL PRIMARY KEY,
    category        TEXT UNIQUE NOT NULL,
    sample_size     INT,
    avg_outcome_7d  NUMERIC(8,4),
    avg_outcome_30d NUMERIC(8,4),
    win_rate_7d     NUMERIC(5,4),
    win_rate_30d    NUMERIC(5,4),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Drop tables that are no longer needed
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS strategy CASCADE;
