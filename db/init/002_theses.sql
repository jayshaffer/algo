-- Theses table for tracking trade ideas

CREATE TABLE theses (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(128) NOT NULL,
    direction VARCHAR(10) NOT NULL,      -- long, short, avoid
    thesis TEXT NOT NULL,                 -- core reasoning
    entry_trigger TEXT,                   -- what would trigger entry
    exit_trigger TEXT,                    -- what would trigger exit
    invalidation TEXT,                    -- what would prove thesis wrong
    confidence VARCHAR(10),               -- high, medium, low
    source VARCHAR(20) DEFAULT 'ideation', -- ideation, manual
    status VARCHAR(20) DEFAULT 'active',  -- active, executed, invalidated, expired
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    close_reason TEXT
);

CREATE INDEX idx_theses_ticker ON theses(ticker);
CREATE INDEX idx_theses_status ON theses(status);
CREATE INDEX idx_theses_created_at ON theses(created_at);
