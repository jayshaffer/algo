-- 018_decision_signals_fk_trigger.sql: P2.14 — DB-level guard for the
-- polymorphic decision_signals.signal_id reference.
--
-- decision_signals.signal_id refers to one of three tables depending on
-- signal_type ('news_signal' → news_signals, 'macro_signal' → macro_signals,
-- 'thesis' → theses). Postgres has no native polymorphic FK, so we use a
-- BEFORE INSERT/UPDATE trigger that validates the (signal_type, signal_id)
-- tuple against the appropriate target table.
--
-- Existing orphan rows (18 historical rows as of 2026-05-02: 11 news_signal,
-- 7 thesis) are preserved — the trigger only validates new rows. Downstream
-- readers (attribution.py, patterns.py) already filter orphans via LEFT JOIN
-- guards added earlier on 2026-05-02. The point of this trigger is to prevent
-- new orphans from regressing the data quality contract.

CREATE OR REPLACE FUNCTION validate_decision_signal_fk()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.signal_type = 'news_signal' THEN
        IF NOT EXISTS (SELECT 1 FROM news_signals WHERE id = NEW.signal_id) THEN
            RAISE EXCEPTION 'decision_signals: news_signal id % does not exist', NEW.signal_id;
        END IF;
    ELSIF NEW.signal_type = 'macro_signal' THEN
        IF NOT EXISTS (SELECT 1 FROM macro_signals WHERE id = NEW.signal_id) THEN
            RAISE EXCEPTION 'decision_signals: macro_signal id % does not exist', NEW.signal_id;
        END IF;
    ELSIF NEW.signal_type = 'thesis' THEN
        IF NOT EXISTS (SELECT 1 FROM theses WHERE id = NEW.signal_id) THEN
            RAISE EXCEPTION 'decision_signals: thesis id % does not exist', NEW.signal_id;
        END IF;
    ELSE
        RAISE EXCEPTION 'decision_signals: unknown signal_type %', NEW.signal_type;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS decision_signals_validate_fk ON decision_signals;
CREATE TRIGGER decision_signals_validate_fk
BEFORE INSERT OR UPDATE ON decision_signals
FOR EACH ROW EXECUTE FUNCTION validate_decision_signal_fk();
