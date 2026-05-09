"""Tests for v2.telemetry — agent_events recorder + auditor read helpers."""

from datetime import date
from unittest.mock import MagicMock, patch


class TestRecordEvent:
    def test_inserts_row_with_jsonb_payload(self):
        from v2.telemetry import record_event

        cur = MagicMock()
        with patch("v2.telemetry.get_cursor") as gc:
            gc.return_value.__enter__.return_value = cur
            record_event(
                session_id=42,
                stage_name="ideation",
                event_type="tool_invocation",
                payload={"tool_name": "get_recent_playbooks", "success": True},
            )
        sql, params = cur.execute.call_args[0]
        assert "INSERT INTO agent_events" in sql
        assert params[0] == 42
        assert params[1] == "ideation"
        assert params[2] == "tool_invocation"
        assert "get_recent_playbooks" in params[3]

    def test_noop_when_session_id_none(self):
        from v2.telemetry import record_event

        with patch("v2.telemetry.get_cursor") as gc:
            record_event(None, "ideation", "tool_invocation", {})
        gc.assert_not_called()

    def test_swallows_exceptions(self):
        """Telemetry must never break a session."""
        from v2.telemetry import record_event

        with patch("v2.telemetry.get_cursor", side_effect=RuntimeError("DB down")):
            record_event(1, "ideation", "tool_invocation", {})

    def test_serializes_dates_in_payload(self):
        from v2.telemetry import record_event

        cur = MagicMock()
        with patch("v2.telemetry.get_cursor") as gc:
            gc.return_value.__enter__.return_value = cur
            record_event(
                session_id=1,
                stage_name="reflection",
                event_type="evidence_shown",
                payload={"items": [{"ticker": "GOOGL", "first_date": date(2026, 5, 1)}]},
            )
        params = cur.execute.call_args[0][1]
        assert "2026-05-01" in params[3]


class TestCountToolInvocations:
    def test_groups_by_tool_name(self):
        from v2.telemetry import count_tool_invocations_by_session

        cur = MagicMock()
        cur.fetchall.return_value = [
            {"tool_name": "get_recent_playbooks", "n": 1},
            {"tool_name": "write_playbook", "n": 1},
        ]
        with patch("v2.telemetry.get_cursor") as gc:
            gc.return_value.__enter__.return_value = cur
            counts = count_tool_invocations_by_session(42)
        assert counts == {"get_recent_playbooks": 1, "write_playbook": 1}


class TestSessionSummaryLine:
    def test_includes_session_id_and_tool_counts(self):
        from v2.telemetry import session_summary_line

        with patch(
            "v2.telemetry.count_tool_invocations_by_session",
            return_value={"get_session_summary": 1},
        ):
            line = session_summary_line(7)
        assert "session=7" in line
        assert "get_session_summary" in line

    def test_handles_empty_session(self):
        from v2.telemetry import session_summary_line

        with patch(
            "v2.telemetry.count_tool_invocations_by_session", return_value={}
        ):
            line = session_summary_line(99)
        assert "session=99" in line
        assert "no_tool_events" in line
