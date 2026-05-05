-- Per-stage Claude API token usage + USD cost views.
-- See docs/superpowers/specs/2026-05-05-session-cost-tracking-design.md

ALTER TABLE session_stages
    ADD COLUMN model VARCHAR(64),
    ADD COLUMN input_tokens INT,
    ADD COLUMN output_tokens INT,
    ADD COLUMN cache_creation_tokens INT,
    ADD COLUMN cache_read_tokens INT;

CREATE TABLE model_pricing (
    model VARCHAR(64) PRIMARY KEY,
    input_per_mtok NUMERIC(10,4) NOT NULL,
    output_per_mtok NUMERIC(10,4) NOT NULL,
    cache_creation_per_mtok NUMERIC(10,4) NOT NULL,
    cache_read_per_mtok NUMERIC(10,4) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed both bare aliases and dated pins. Callsites pass the dated pin
-- (e.g. classifier uses "claude-haiku-4-5-20251001"); ideation/session
-- pass the bare alias. Both must resolve.
INSERT INTO model_pricing (model, input_per_mtok, output_per_mtok, cache_creation_per_mtok, cache_read_per_mtok) VALUES
    ('claude-opus-4-7',              15.00, 75.00, 18.75, 1.50),
    ('claude-opus-4-6',              15.00, 75.00, 18.75, 1.50),
    ('claude-sonnet-4-6',             3.00, 15.00,  3.75, 0.30),
    ('claude-haiku-4-5',              1.00,  5.00,  1.25, 0.10),
    ('claude-haiku-4-5-20251001',     1.00,  5.00,  1.25, 0.10);

CREATE VIEW session_stage_costs AS
SELECT
    ss.*,
    CASE
        WHEN ss.model IS NULL OR mp.model IS NULL THEN NULL
        ELSE (
            COALESCE(ss.input_tokens, 0)          * mp.input_per_mtok +
            COALESCE(ss.output_tokens, 0)         * mp.output_per_mtok +
            COALESCE(ss.cache_creation_tokens, 0) * mp.cache_creation_per_mtok +
            COALESCE(ss.cache_read_tokens, 0)     * mp.cache_read_per_mtok
        ) / 1000000.0
    END AS cost_usd
FROM session_stages ss
LEFT JOIN model_pricing mp ON mp.model = ss.model;

CREATE VIEW session_costs AS
SELECT
    s.id AS session_id, s.session_date, s.session_type, s.status,
    SUM(sc.cost_usd)              AS total_cost_usd,
    SUM(sc.input_tokens)          AS total_input_tokens,
    SUM(sc.output_tokens)         AS total_output_tokens,
    SUM(sc.cache_creation_tokens) AS total_cache_creation_tokens,
    SUM(sc.cache_read_tokens)     AS total_cache_read_tokens
FROM sessions s
LEFT JOIN session_stage_costs sc ON sc.session_id = s.id
GROUP BY s.id, s.session_date, s.session_type, s.status;
