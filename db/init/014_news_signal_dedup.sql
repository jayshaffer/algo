CREATE UNIQUE INDEX IF NOT EXISTS idx_news_signals_dedup
    ON news_signals (ticker, md5(headline), published_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_macro_signals_dedup
    ON macro_signals (md5(headline), published_at);
