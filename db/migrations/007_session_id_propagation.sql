-- 007_session_id_propagation.sql — per-run session IDs: thread sessions.id
-- into every table the dashboard currently joins to sessions by date, then
-- drop the per-date uniqueness constraint so multiple session rows can
-- coexist for one date.
--
-- Mirrors db/init/029_session_id_propagation.sql for in-place application to
-- existing dbs whose data volumes predate that init file.

ALTER TABLE decisions
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);
ALTER TABLE theses
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);
ALTER TABLE strategy_memos
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);
ALTER TABLE tweets
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);

CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_theses_session ON theses(session_id);
CREATE INDEX IF NOT EXISTS idx_strategy_memos_session ON strategy_memos(session_id);
CREATE INDEX IF NOT EXISTS idx_tweets_session ON tweets(session_id);

-- Backfill. UNIQUE (session_date, session_type) guarantees at most one
-- daily session per date pre-migration, so this is deterministic.
UPDATE decisions d
SET session_id = s.id
FROM sessions s
WHERE d.session_id IS NULL
  AND s.session_date = d.date
  AND s.session_type = 'daily';

UPDATE theses t
SET session_id = s.id
FROM sessions s
WHERE t.session_id IS NULL
  AND s.session_date = t.created_at::date
  AND s.session_type = 'daily';

UPDATE strategy_memos m
SET session_id = s.id
FROM sessions s
WHERE m.session_id IS NULL
  AND s.session_date = m.session_date
  AND s.session_type = 'daily';

UPDATE tweets tw
SET session_id = s.id
FROM sessions s
WHERE tw.session_id IS NULL
  AND s.session_date = tw.session_date
  AND s.session_type = 'daily';

ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_session_date_session_type_key;
