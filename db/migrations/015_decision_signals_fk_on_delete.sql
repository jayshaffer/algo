-- 015_decision_signals_fk_on_delete.sql (mirror of db/init/038)
-- Add ON DELETE behavior to the FKs that previously defaulted to NO ACTION
-- with no cleanup path. Nothing deletes these parents today, but orphaned
-- decision_signals rows are exactly the shape that produced the phantom
-- "news_signal:unknown" attribution bucket (see 018/028 for the insert-side
-- guards); make the delete side structurally safe too.
--
-- Idempotent
-- (drop + re-add by the deterministic auto-generated constraint names).

-- decision_signals.decision_id: signal links are meaningless without their
-- decision; cascade.
ALTER TABLE decision_signals DROP CONSTRAINT IF EXISTS decision_signals_decision_id_fkey;
ALTER TABLE decision_signals ADD CONSTRAINT decision_signals_decision_id_fkey
    FOREIGN KEY (decision_id) REFERENCES decisions(id) ON DELETE CASCADE;

-- playbook_actions.playbook_id: actions are meaningless without their
-- playbook; cascade. Note decisions.playbook_action_id keeps its default
-- NO ACTION on purpose — a playbook whose actions produced real decisions
-- cannot be deleted without first dealing with the audit trail.
ALTER TABLE playbook_actions DROP CONSTRAINT IF EXISTS playbook_actions_playbook_id_fkey;
ALTER TABLE playbook_actions ADD CONSTRAINT playbook_actions_playbook_id_fkey
    FOREIGN KEY (playbook_id) REFERENCES playbooks(id) ON DELETE CASCADE;

-- playbook_actions.thesis_id: keep the action row (part of the planning
-- audit trail) but null the pointer if its thesis is ever deleted.
ALTER TABLE playbook_actions DROP CONSTRAINT IF EXISTS playbook_actions_thesis_id_fkey;
ALTER TABLE playbook_actions ADD CONSTRAINT playbook_actions_thesis_id_fkey
    FOREIGN KEY (thesis_id) REFERENCES theses(id) ON DELETE SET NULL;
