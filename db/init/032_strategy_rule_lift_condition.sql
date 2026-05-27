-- 032_strategy_rule_lift_condition.sql
-- Rules that gate execution need an explicit condition for when the gate lifts.

ALTER TABLE strategy_rules
    ADD COLUMN IF NOT EXISTS lift_condition TEXT;
