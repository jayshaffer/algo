-- 028_decision_signals_meta_types.sql — extend the decision_signals
-- FK trigger to accept two meta signal types emitted by the trader at
-- decision-write time:
--
--   * 'rule_gate'  — references strategy_rules.id; emitted when a HOLD
--                    decision was driven by a named rule gate (Rule 27
--                    macro cap, Rule 29 stabilization, Rule 41, etc.).
--                    Closes audit finding 91 (RULE_DEAD on Rule 42).
--
--   * 'signal_gap' — flag row with signal_id = the decision's own id
--                    (self-reference), emitted when a buy/sell decision
--                    has no other signal_refs (or only a thesis ref).
--                    Closes audit finding 89 (RULE_DEAD on Rule 32).
--
-- For 'signal_gap' we accept any signal_id ≥ 0 without checking a
-- parent table — there is no parent; the row exists purely as a
-- count-able marker so attribution / patterns / reflection can see
-- the gap.

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
