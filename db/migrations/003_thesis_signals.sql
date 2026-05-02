-- Persist the news/macro/thesis signals that justified each thesis.
-- Closes the gap that previously left the executor inventing signal_refs:
-- the strategist now records its citations on thesis creation, and those
-- IDs flow forward to playbook_actions and decision_signals.
-- See v2/tools.py:tool_create_thesis and v2/context.py:build_executor_input.

CREATE TABLE IF NOT EXISTS thesis_signals (
    thesis_id INT NOT NULL REFERENCES theses(id) ON DELETE CASCADE,
    signal_type VARCHAR(20) NOT NULL,
    signal_id INT NOT NULL,
    PRIMARY KEY (thesis_id, signal_type, signal_id)
);

CREATE INDEX IF NOT EXISTS idx_thesis_signals_thesis ON thesis_signals(thesis_id);
