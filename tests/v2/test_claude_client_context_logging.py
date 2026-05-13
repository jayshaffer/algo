"""Tests for LLM context logging captured at _call_with_retry.

Spec: docs/superpowers/specs/2026-05-13-llm-context-logging-design.md
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestCapturedPurposes:
    def test_executor_strategist_reflection_are_captured(self):
        from v2.claude_client import _CONTEXT_LOGGED_PURPOSES, AgentPurpose

        assert AgentPurpose.EXECUTOR in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.STRATEGIST_LOOP in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.REFLECTION_LOOP in _CONTEXT_LOGGED_PURPOSES

    def test_classifier_purposes_not_captured(self):
        from v2.claude_client import _CONTEXT_LOGGED_PURPOSES, AgentPurpose

        assert AgentPurpose.CLASSIFIER_NEWS not in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.CLASSIFIER_MACRO not in _CONTEXT_LOGGED_PURPOSES
        assert AgentPurpose.CLASSIFIER_RELEVANCE not in _CONTEXT_LOGGED_PURPOSES


class TestRecordCallContextHelper:
    """`_record_call_context` is the gate + serializer. It must be a no-op
    when session_id is None, when purpose is not captured, or when the DB
    write raises."""

    def test_no_op_when_session_id_is_none(self, monkeypatch):
        from v2.claude_client import _record_call_context

        called = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: called.append(kw),
        )
        _record_call_context(
            session_id=None,
            stage_name="trading",
            purpose="executor",
            create_kwargs={"model": "m", "messages": [], "system": ""},
            message=None,
            duration_ms=10,
        )
        assert called == []

    def test_no_op_when_purpose_not_captured(self, monkeypatch):
        from v2.claude_client import _record_call_context

        called = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: called.append(kw),
        )
        _record_call_context(
            session_id=42,
            stage_name="pipeline",
            purpose="classifier_news",
            create_kwargs={"model": "m", "messages": [], "system": ""},
            message=None,
            duration_ms=10,
        )
        assert called == []

    def test_swallows_db_errors(self, monkeypatch, caplog):
        from v2.claude_client import _record_call_context

        def boom(**_kw):
            raise RuntimeError("db down")
        monkeypatch.setattr("v2.claude_client.insert_llm_call_context", boom)

        message = MagicMock()
        message.content = [SimpleNamespace(type="text", text="ok")]
        message.stop_reason = "end_turn"
        message.usage = SimpleNamespace(
            input_tokens=1, output_tokens=2,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        # Must not raise
        _record_call_context(
            session_id=42,
            stage_name="trading",
            purpose="executor",
            create_kwargs={"model": "m", "messages": [{"role": "user", "content": "hi"}], "system": "sys"},
            message=message,
            duration_ms=10,
        )


class TestCallWithRetryWritesContext:
    """End-to-end: a `_call_with_retry` for a captured purpose writes one
    llm_call_contexts row alongside the existing `agent_call` telemetry
    event."""

    def _text_block(self, text):
        return SimpleNamespace(type="text", text=text)

    def _tool_use_block(self, tool_id, name, input_data):
        return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=input_data)

    def _make_response(self, content, stop_reason="end_turn",
                       input_tokens=100, output_tokens=50,
                       cache_creation=0, cache_read=0):
        response = MagicMock()
        response.content = content
        response.stop_reason = stop_reason
        response.usage = MagicMock()
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        response.usage.cache_creation_input_tokens = cache_creation
        response.usage.cache_read_input_tokens = cache_read
        return response

    def _make_stream_mock(self, response):
        def stream_factory(**_kwargs):
            ctx = MagicMock()
            ctx.__enter__.return_value = ctx
            ctx.__exit__.return_value = None
            ctx.get_final_message.return_value = response
            return ctx
        client = MagicMock()
        client.messages.stream.side_effect = stream_factory
        return client

    def test_executor_call_writes_context_row(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        response = self._make_response(
            content=[self._text_block("decision json here")],
            stop_reason="end_turn",
            input_tokens=120, output_tokens=45,
            cache_creation=10, cache_read=80,
        )
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            system="executor system prompt",
            messages=[{"role": "user", "content": "executor input json"}],
            session_id=42,
            stage_name="trading",
            purpose="executor",
        )

        assert len(recorded_inserts) == 1
        row = recorded_inserts[0]
        assert row["session_id"] == 42
        assert row["stage_name"] == "trading"
        assert row["purpose"] == "executor"
        assert row["model"] == "claude-haiku-4-5-20251001"
        assert row["system_prompt"] == "executor system prompt"
        assert row["messages"] == [{"role": "user", "content": "executor input json"}]
        assert row["tool_definitions"] is None
        assert row["response_content"] == [{"type": "text", "text": "decision json here"}]
        assert row["input_tokens"] == 120
        assert row["output_tokens"] == 45
        assert row["cache_read_tokens"] == 80
        assert row["cache_creation_tokens"] == 10
        assert row["stop_reason"] == "end_turn"
        assert row["duration_ms"] is not None and row["duration_ms"] >= 0

    def test_strategist_call_writes_context_row_with_tools_and_tool_use_response(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        tools = [{"name": "get_positions", "input_schema": {"type": "object"}}]
        response = self._make_response(
            content=[
                self._text_block("checking positions"),
                self._tool_use_block("t1", "get_positions", {}),
            ],
            stop_reason="tool_use",
        )
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="claude-opus-4-7",
            max_tokens=32000,
            system="strategist system",
            messages=[{"role": "user", "content": "go"}],
            tools=tools,
            session_id=99,
            stage_name="ideation",
            purpose="strategist_loop",
        )

        assert len(recorded_inserts) == 1
        row = recorded_inserts[0]
        assert row["purpose"] == "strategist_loop"
        assert row["tool_definitions"] == tools
        assert row["response_content"] == [
            {"type": "text", "text": "checking positions"},
            {"type": "tool_use", "id": "t1", "name": "get_positions", "input": {}},
        ]
        assert row["stop_reason"] == "tool_use"

    def test_classifier_call_does_not_write_context_row(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        recorded_events = []
        monkeypatch.setattr(
            "v2.claude_client.record_event",
            lambda **kw: recorded_events.append(kw),
        )

        response = self._make_response(content=[self._text_block("category")])
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="m",
            max_tokens=200,
            system="classify",
            messages=[{"role": "user", "content": "headline"}],
            session_id=7,
            stage_name="pipeline",
            purpose="classifier_news",
        )

        assert recorded_inserts == []
        assert len(recorded_events) == 1

    def test_no_session_id_does_not_write_context_row(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        response = self._make_response(content=[self._text_block("hi")])
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="m",
            max_tokens=200,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            session_id=None,
            stage_name="trading",
            purpose="executor",
        )

        assert recorded_inserts == []

    def test_strategist_list_form_system_is_captured_as_text(self, monkeypatch):
        """run_agentic_loop wraps the system prompt in a cache-control list:
            [{"type": "text", "text": "...", "cache_control": {...}}]
        The captured system_prompt must extract the text, not store NULL."""
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        cached_system = [
            {"type": "text", "text": "strategist instructions",
             "cache_control": {"type": "ephemeral"}}
        ]
        response = self._make_response(content=[self._text_block("ok")])
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="claude-opus-4-7",
            max_tokens=32000,
            system=cached_system,
            messages=[{"role": "user", "content": "go"}],
            session_id=99,
            stage_name="ideation",
            purpose="strategist_loop",
        )

        assert len(recorded_inserts) == 1
        assert recorded_inserts[0]["system_prompt"] == "strategist instructions"

    def test_multi_block_list_form_system_is_joined(self, monkeypatch):
        """List-form with two text blocks: joined with \\n separator."""
        from v2.claude_client import _call_with_retry

        recorded_inserts = []
        monkeypatch.setattr(
            "v2.claude_client.insert_llm_call_context",
            lambda **kw: recorded_inserts.append(kw),
        )
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        cached_system = [
            {"type": "text", "text": "base prompt"},
            {"type": "text", "text": "appended context", "cache_control": {"type": "ephemeral"}},
        ]
        response = self._make_response(content=[self._text_block("ok")])
        client = self._make_stream_mock(response)

        _call_with_retry(
            client,
            model="claude-opus-4-7",
            max_tokens=32000,
            system=cached_system,
            messages=[{"role": "user", "content": "go"}],
            session_id=99,
            stage_name="ideation",
            purpose="strategist_loop",
        )

        assert len(recorded_inserts) == 1
        assert recorded_inserts[0]["system_prompt"] == "base prompt\nappended context"

    def test_context_logging_failure_does_not_break_caller(self, monkeypatch):
        from v2.claude_client import _call_with_retry

        def boom(**_kw):
            raise RuntimeError("db down")

        monkeypatch.setattr("v2.claude_client.insert_llm_call_context", boom)
        monkeypatch.setattr("v2.claude_client.record_event", lambda **kw: None)

        response = self._make_response(content=[self._text_block("ok")])
        client = self._make_stream_mock(response)

        result = _call_with_retry(
            client,
            model="m",
            max_tokens=200,
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            session_id=42,
            stage_name="trading",
            purpose="executor",
        )
        assert result is response
