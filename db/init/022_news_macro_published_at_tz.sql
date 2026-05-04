-- T2.3: news_signals.published_at and macro_signals.published_at were stored
-- as TIMESTAMP (naive). Alpaca returns aware UTC, the news fetcher now uses
-- aware UTC, but Postgres silently dropped the offset on insert. Promote both
-- columns to TIMESTAMPTZ so the data round-trips with offset intact.
--
-- Existing rows are reinterpreted as UTC (they were always UTC in practice
-- — the bug was the type, not the values).

ALTER TABLE news_signals
    ALTER COLUMN published_at TYPE TIMESTAMPTZ
    USING published_at AT TIME ZONE 'UTC';

ALTER TABLE macro_signals
    ALTER COLUMN published_at TYPE TIMESTAMPTZ
    USING published_at AT TIME ZONE 'UTC';
