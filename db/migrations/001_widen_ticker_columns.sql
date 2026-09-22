-- Widen ticker columns from VARCHAR(10) to VARCHAR(128)
-- Mirror of db/init/001_schema.sql, which already declares VARCHAR(128).
--
-- Guarded per column: an unconditional ALTER ... TYPE is a no-op semantically
-- when the column is already VARCHAR(128), but Postgres still rewrites the
-- table and REBUILDS its indexes, and a rebuilt partial index re-renders its
-- predicate expression (idx_decisions_dedup on decisions). That made this
-- migration produce a textually different schema from an identical starting
-- point, which db/check_mirror.sh reads as drift. Skipping the ALTER when the
-- column is already wide makes the migration genuinely idempotent.

DO $$
DECLARE
  t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY['news_signals', 'positions', 'open_orders', 'decisions', 'theses', 'documents']
  LOOP
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public'
        AND table_name = t
        AND column_name = 'ticker'
        AND character_maximum_length IS DISTINCT FROM 128
    ) THEN
      EXECUTE format('ALTER TABLE %I ALTER COLUMN ticker TYPE VARCHAR(128)', t);
    END IF;
  END LOOP;
END $$;
