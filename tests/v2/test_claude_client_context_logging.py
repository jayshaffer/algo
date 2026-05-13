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
