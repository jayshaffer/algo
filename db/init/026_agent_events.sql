-- db/init/026_agent_events.sql
-- Generic event log for LLM-side and gate-side observability.
-- One table for all event types (zero-migration extensibility).
-- Auditor reads from this table; nothing else does. Inserts are
-- best-effort from the producer side.
-- See docs/superpowers/plans/2026-05-08-flip-flop-telemetry.md

CREATE TABLE IF NOT EXISTS agent_events (
    id           BIGSERIAL PRIMARY KEY,
    session_id   INT REFERENCES sessions(id) ON DELETE CASCADE,
    stage_name   VARCHAR(50),
    event_type   VARCHAR(50) NOT NULL,
    payload      JSONB NOT NULL,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_events_session
    ON agent_events(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_type_time
    ON agent_events(event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_events_stage_type
    ON agent_events(stage_name, event_type);

-- Functional index for tool_name lookups inside payload (covers
-- STRATEGIST_NOT_USING_REVERSAL_TOOL and TOOL_ERROR_RATE checks).
CREATE INDEX IF NOT EXISTS idx_agent_events_tool_name
    ON agent_events ((payload->>'tool_name'))
    WHERE event_type = 'tool_invocation';

-- Functional index for risk_block ticker hotspot detection.
CREATE INDEX IF NOT EXISTS idx_agent_events_ticker
    ON agent_events ((payload->>'ticker'))
    WHERE event_type = 'risk_block';
