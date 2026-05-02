-- 020_signal_attribution_30d_count.sql: P2.17 — separate 30d sample count.
-- The existing `sample_size` counts decisions with 7d outcomes; AVG(alpha_30d)
-- silently excludes NULL 30d outcomes, leading to "n=30, +0.5% avg 30d alpha"
-- where only a handful of decisions actually had 30d outcomes. Track the 30d
-- count separately so consumers don't conflate the two.

ALTER TABLE signal_attribution
ADD COLUMN IF NOT EXISTS sample_size_30d INT DEFAULT 0;
