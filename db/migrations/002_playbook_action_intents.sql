-- Replace LLM-authored share count with intent + magnitude.
-- Mirror of db/init/006_v3.sql, which declares playbook_actions with
-- intent_type/intent_magnitude and no max_quantity.
-- The trader resolves these to exact shares against live portfolio state
-- at execution time; see v2/intents.py.

ALTER TABLE playbook_actions
    ADD COLUMN IF NOT EXISTS intent_type VARCHAR(32),
    ADD COLUMN IF NOT EXISTS intent_magnitude DECIMAL;

ALTER TABLE playbook_actions
    DROP COLUMN IF EXISTS max_quantity;
