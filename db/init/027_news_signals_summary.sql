-- db/init/027_news_signals_summary.sql
-- Adds a nullable `summary` column to news_signals so the Haiku
-- relevance filter has more than 60 chars of headline to rank on.
-- Nullable so pre-backfill rows aren't broken and existing inserts
-- without the column keep working until pipeline is updated.
-- See docs/superpowers/specs/2026-05-10-news-filter-design.md

ALTER TABLE news_signals
    ADD COLUMN IF NOT EXISTS summary TEXT;
