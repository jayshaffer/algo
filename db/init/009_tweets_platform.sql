-- 009_tweets_platform.sql: Add platform column to tweets table
ALTER TABLE tweets ADD COLUMN IF NOT EXISTS platform TEXT NOT NULL DEFAULT 'twitter';
CREATE INDEX IF NOT EXISTS idx_tweets_platform ON tweets(platform);
