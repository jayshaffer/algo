-- 019_news_signal_alpaca_id.sql: P2.16 — persist Alpaca's news id for dedup.
-- Existing dedup on (ticker, md5(headline), published_at) lets near-duplicate
-- timestamps slip through. Alpaca's news id is the canonical identity; storing
-- it enables exact dedup. Partial UNIQUE so historical rows without alpaca_id
-- aren't blocked.

ALTER TABLE news_signals  ADD COLUMN IF NOT EXISTS alpaca_id TEXT;
ALTER TABLE macro_signals ADD COLUMN IF NOT EXISTS alpaca_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_news_signals_alpaca_id
ON news_signals(alpaca_id) WHERE alpaca_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_macro_signals_alpaca_id
ON macro_signals(alpaca_id) WHERE alpaca_id IS NOT NULL;
