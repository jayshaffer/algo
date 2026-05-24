-- db/migrations/009_decision_signals_meta_types.sql
-- Mirror of db/init/028_decision_signals_meta_types.sql for existing volumes.
-- Extends validate_decision_signal_fk() to accept 'rule_gate' and 'signal_gap'
-- so the trader can write rule-driven HOLDs and signal-gap markers without
-- the FK trigger rejecting them. Idempotent (CREATE OR REPLACE + DROP IF EXISTS).

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
    ELSIF NEW.signal_type = 'rule_gate' THEN
        IF NOT EXISTS (SELECT 1 FROM strategy_rules WHERE id = NEW.signal_id) THEN
            RAISE EXCEPTION 'decision_signals: rule_gate id % (strategy_rules) does not exist', NEW.signal_id;
        END IF;
    ELSIF NEW.signal_type = 'signal_gap' THEN
        -- Marker row; no parent table to validate against.
        NULL;
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
