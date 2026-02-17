-- 008_tweets.sql: Tweet log for Bikini Bottom Capital

CREATE TABLE IF NOT EXISTS tweets (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    session_date DATE NOT NULL,
    tweet_type TEXT NOT NULL,
    tweet_text TEXT NOT NULL,
    tweet_id TEXT,
    posted BOOLEAN DEFAULT FALSE,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_tweets_session_date ON tweets(session_date DESC);
