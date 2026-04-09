-- Prevent duplicate trade decisions (same ticker, action, date)
-- Allow hold decisions to be duplicated (they're not actionable)
CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_dedup
ON decisions (date, ticker, action)
WHERE action IN ('buy', 'sell');
