"""Generic agent_events recorder + small auditor-facing query helpers.

`record_event` is a no-op when session_id is None and swallows DB errors:
telemetry must never break a session.
"""
import json
import logging
from datetime import date, datetime

from .database.connection import get_cursor

logger = logging.getLogger(__name__)


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def record_event(session_id, stage_name, event_type, payload):
    if session_id is None:
        return
    try:
        serialized = json.dumps(payload, default=_json_default)
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_events
                    (session_id, stage_name, event_type, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (session_id, stage_name, event_type, serialized),
            )
    except Exception:
        logger.exception("Failed to record agent_event; continuing")


def count_tool_invocations_by_session(session_id: int) -> dict[str, int]:
    """Returns {tool_name: count} for a session's tool_invocation events."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT payload->>'tool_name' AS tool_name, COUNT(*) AS n
            FROM agent_events
            WHERE session_id = %s AND event_type = 'tool_invocation'
            GROUP BY 1
            """,
            (session_id,),
        )
        return {r["tool_name"]: r["n"] for r in cur.fetchall()}


def session_summary_line(session_id: int) -> str:
    """One-line human-readable summary; logged at end of each session."""
    counts = count_tool_invocations_by_session(session_id)
    if not counts:
        return f"[telemetry] session={session_id} no_tool_events"
    return f"[telemetry] session={session_id} tools={counts}"
