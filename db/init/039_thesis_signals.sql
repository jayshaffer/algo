-- 039_thesis_signals.sql — mirror of db/migrations/003_thesis_signals.sql.
--
-- thesis_signals existed only in db/migrations/ from 2026-05 until the
-- 2026-08-13 audit (finding 2.2). db/init/ runs only on a fresh volume, so
-- every fresh volume — including CI, which seeds from db/init/* alone, and any
-- DR restore — came up without a table the strategist writes on every thesis
-- creation and the executor reads through signal_refs. CI was therefore
-- testing a structurally different schema from production.
--
-- Persist the news/macro/thesis signals that justified each thesis. Closes the
-- gap that previously left the executor inventing signal_refs: the strategist
-- records its citations on thesis creation, and those IDs flow forward to
-- playbook_actions and decision_signals.
-- See v2/tools.py:tool_create_thesis and v2/context.py:build_executor_input.

CREATE TABLE IF NOT EXISTS thesis_signals (
    thesis_id INT NOT NULL REFERENCES theses(id) ON DELETE CASCADE,
    signal_type VARCHAR(20) NOT NULL,
    signal_id INT NOT NULL,
    PRIMARY KEY (thesis_id, signal_type, signal_id)
);

CREATE INDEX IF NOT EXISTS idx_thesis_signals_thesis ON thesis_signals(thesis_id);
