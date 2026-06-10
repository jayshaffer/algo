-- 036_model_pricing_fable_5.sql
-- Seed pricing for claude-fable-5 (the supervisor model, pinned in v2/ as of
-- this change). Without this row the session_stage_costs view returns NULL for
-- any fable-5 stage and the cost is silently dropped from the session summary.
-- Rates per claude.com/pricing: $10 in / $50 out per MTok, $12.50 5m cache
-- write, $1.00 cache read. Idempotent so it can run against live prod + paper.

INSERT INTO model_pricing (model, input_per_mtok, output_per_mtok, cache_creation_per_mtok, cache_read_per_mtok)
VALUES ('claude-fable-5', 10.00, 50.00, 12.50, 1.00)
ON CONFLICT (model) DO NOTHING;
