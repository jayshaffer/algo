-- 021_news_signal_alpaca_id_ticker.sql: fix multi-ticker news drop.
--
-- Migration 019 created `idx_news_signals_alpaca_id` as UNIQUE on (alpaca_id)
-- alone. But a single Alpaca news article (one alpaca_id) routinely produces
-- multiple ticker signals — e.g. "AAPL/MSFT/NVDA all rally". The classifier
-- emits N rows sharing one alpaca_id, then `ON CONFLICT DO NOTHING` silently
-- drops all but the first. We were losing 2/3+ of multi-ticker news coverage.
--
-- Macro signals are unaffected (one macro signal per article — the existing
-- (alpaca_id) UNIQUE is still correct there).

DROP INDEX IF EXISTS idx_news_signals_alpaca_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_news_signals_alpaca_id_ticker
ON news_signals(alpaca_id, ticker) WHERE alpaca_id IS NOT NULL;
