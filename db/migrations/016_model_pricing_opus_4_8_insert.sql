-- 016_model_pricing_opus_4_8_insert.sql (mirror of db/init/035, corrected by 037)
--
-- 014 claimed to fix missing/stale pricing on long-lived volumes but mirrored
-- only db/init/036 + 037. db/init/035 — the file that INSERTs the
-- claude-opus-4-8 row at all — was never mirrored, and 014's opus correction is
-- a bare UPDATE. On any volume created before init/035 existed there is no
-- opus-4-8 row for that UPDATE to match, so it silently affects zero rows and
-- the volume ends up with no opus-4-8 pricing whatsoever: session_stage_costs
-- is a view over model_pricing, so every strategist stage cost is dropped
-- rather than mispriced. 014 reproduced the exact failure it was written to
-- fix (audit 2.2).
--
-- Insert at the corrected rate directly (init/035 seeds $15/$75 and init/037
-- then corrects it to $5/$25; there is no reason to replay that detour here).
-- DO UPDATE rather than DO NOTHING so the row converges to the right rate
-- whatever state the volume is in.

INSERT INTO model_pricing (model, input_per_mtok, output_per_mtok, cache_creation_per_mtok, cache_read_per_mtok)
VALUES ('claude-opus-4-8', 5.00, 25.00, 6.25, 0.50)
ON CONFLICT (model) DO UPDATE
    SET input_per_mtok          = EXCLUDED.input_per_mtok,
        output_per_mtok         = EXCLUDED.output_per_mtok,
        cache_creation_per_mtok = EXCLUDED.cache_creation_per_mtok,
        cache_read_per_mtok     = EXCLUDED.cache_read_per_mtok;
