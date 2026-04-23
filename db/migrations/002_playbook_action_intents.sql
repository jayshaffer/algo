-- Replace LLM-authored share count with intent + magnitude.
-- The trader resolves these to exact shares against live portfolio state
-- at execution time; see v2/intents.py.

ALTER TABLE playbook_actions
    ADD COLUMN IF NOT EXISTS intent_type VARCHAR(32),
    ADD COLUMN IF NOT EXISTS intent_magnitude DECIMAL;

ALTER TABLE playbook_actions
    DROP COLUMN IF EXISTS max_quantity;
