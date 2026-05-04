-- 023_tweets_decision_id.sql: link tweets back to source decisions.
--
-- Today's rerun guard posted_tweet_exists(session_date, type, platform)
-- assumes one recap tweet per day. The new live-trade pipeline posts
-- one tweet per significant decision, so we need to dedup per-decision
-- per-platform. Add a nullable FK so existing recap rows aren't broken
-- and the new trade rows carry their source decision id.
--
-- Index supports the new query
--   SELECT 1 FROM tweets WHERE decision_id = $1 AND platform = $2
-- which fires once per candidate decision in run_trade_posts_stage.

ALTER TABLE tweets
    ADD COLUMN IF NOT EXISTS decision_id INTEGER
    REFERENCES decisions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_tweets_decision_id_platform
    ON tweets(decision_id, platform) WHERE decision_id IS NOT NULL;
