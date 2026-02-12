-- V3: Structured playbook actions + decision linkage
CREATE TABLE IF NOT EXISTS playbook_actions (
    id SERIAL PRIMARY KEY,
    playbook_id INT REFERENCES playbooks(id),
    ticker VARCHAR(128) NOT NULL,
    action VARCHAR(10) NOT NULL,
    thesis_id INT REFERENCES theses(id),
    reasoning TEXT,
    confidence VARCHAR(10),
    max_quantity DECIMAL,
    priority INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_playbook_actions_playbook_id ON playbook_actions(playbook_id);

ALTER TABLE decisions ADD COLUMN IF NOT EXISTS playbook_action_id INT REFERENCES playbook_actions(id);
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS is_off_playbook BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_decision_signals_decision_id ON decision_signals(decision_id);
