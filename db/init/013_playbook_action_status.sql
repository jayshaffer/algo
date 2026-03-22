ALTER TABLE playbook_actions ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending';
CREATE INDEX IF NOT EXISTS idx_playbook_actions_status ON playbook_actions(status);
