-- 035_model_pricing_opus_4_8.sql
-- Seed pricing for claude-opus-4-8 (the strategist model, pinned in v2/ as of
-- this change). Without this row the session_stage_costs view returns NULL for
-- any opus-4-8 stage and the cost is silently dropped from the session summary.
-- Same rate as opus-4-6/4-7. Idempotent so it can run against live prod + paper.

INSERT INTO model_pricing (model, input_per_mtok, output_per_mtok, cache_creation_per_mtok, cache_read_per_mtok)
VALUES ('claude-opus-4-8', 15.00, 75.00, 18.75, 1.50)
ON CONFLICT (model) DO NOTHING;
