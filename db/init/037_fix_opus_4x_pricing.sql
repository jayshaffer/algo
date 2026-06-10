-- 037_fix_opus_4x_pricing.sql
-- Correct stale pricing for the Claude Opus 4.x line. The 024 seed (and the
-- 035 follow-up for opus-4-8) used the old Opus 4 / 4.1 rate of $15 in / $75
-- out per MTok. Per claude.com/pricing, Opus 4.5 and later are $5 in / $25 out,
-- with $6.25 5m cache write and $0.50 cache read. Only the deprecated Opus 4
-- and 4.1 remain at $15/$75 (not seeded here).
--
-- Because session_stage_costs is a view that recomputes from model_pricing,
-- this retroactively corrects every historical opus stage cost (previously
-- overstated ~3x). UPDATE rather than INSERT since live DBs already hold the
-- stale rows. Idempotent.

UPDATE model_pricing
   SET input_per_mtok          = 5.00,
       output_per_mtok         = 25.00,
       cache_creation_per_mtok = 6.25,
       cache_read_per_mtok     = 0.50
 WHERE model IN ('claude-opus-4-8', 'claude-opus-4-7', 'claude-opus-4-6');
