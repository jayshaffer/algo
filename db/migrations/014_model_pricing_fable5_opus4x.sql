-- 014_model_pricing_fable5_opus4x.sql (mirror of db/init/036 + 037)
-- These two init files were added 2026-06 without db/migrations mirrors, so
-- long-lived volumes never got them: as of 2026-06-10 prod had no
-- claude-fable-5 pricing row (supervisor stage costs silently dropped via
-- UnknownModelError) and still carried the stale $15/$75 Opus 4.x rates
-- (every strategist stage cost overstated ~3x). Idempotent.

-- db/init/036: seed pricing for claude-fable-5 (supervisor model).
-- Rates per claude.com/pricing: $10 in / $50 out per MTok, $12.50 5m cache
-- write, $1.00 cache read.
INSERT INTO model_pricing (model, input_per_mtok, output_per_mtok, cache_creation_per_mtok, cache_read_per_mtok)
VALUES ('claude-fable-5', 10.00, 50.00, 12.50, 1.00)
ON CONFLICT (model) DO NOTHING;

-- db/init/037: correct stale Opus 4.x pricing. Opus 4.5 and later are
-- $5 in / $25 out, $6.25 5m cache write, $0.50 cache read. Because
-- session_stage_costs is a view over model_pricing, this retroactively
-- corrects historical opus stage costs.
UPDATE model_pricing
   SET input_per_mtok          = 5.00,
       output_per_mtok         = 25.00,
       cache_creation_per_mtok = 6.25,
       cache_read_per_mtok     = 0.50
 WHERE model IN ('claude-opus-4-8', 'claude-opus-4-7', 'claude-opus-4-6');
