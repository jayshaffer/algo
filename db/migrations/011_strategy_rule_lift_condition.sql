-- 011_strategy_rule_lift_condition.sql (mirror of db/init/032)
-- Add explicit lift criteria for execution-gating strategy rules.

ALTER TABLE strategy_rules
    ADD COLUMN IF NOT EXISTS lift_condition TEXT;
