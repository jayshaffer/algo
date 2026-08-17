-- 010_sessions_market_closed.sql (mirror of db/init/031)

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS market_closed BOOLEAN NOT NULL DEFAULT FALSE;
