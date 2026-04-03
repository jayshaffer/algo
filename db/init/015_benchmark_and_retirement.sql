-- 015: Add benchmark columns for SPY-relative attribution + rule retirement reason

ALTER TABLE decisions ADD COLUMN IF NOT EXISTS benchmark_7d DECIMAL;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS benchmark_30d DECIMAL;

ALTER TABLE strategy_rules ADD COLUMN IF NOT EXISTS retirement_reason TEXT;
