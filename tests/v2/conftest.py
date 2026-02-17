"""Shared test fixtures and factory functions for v2 module."""

import os
import pytest
from contextlib import contextmanager
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import MagicMock, patch


# --- Core DB Fixtures ---

@pytest.fixture
def mock_cursor():
    """Create a mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    """Patch get_cursor to yield a mock cursor."""
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("v2.database.connection.get_cursor", _get_cursor), \
         patch("v2.database.trading_db.get_cursor", _get_cursor), \
         patch("v2.database.dashboard_db.get_cursor", _get_cursor), \
         patch("v2.twitter.get_cursor", _get_cursor), \
         patch("v2.entertainment.get_cursor", _get_cursor), \
         patch("v2.database.connection.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor


# --- Claude Fixtures ---

@pytest.fixture
def mock_claude_client():
    """Mock Claude API client."""
    client = MagicMock()
    with patch("v2.claude_client.get_claude_client", return_value=client):
        yield client


# --- Factory Functions ---

def make_news_item(**kwargs):
    """Create a news item dict like what Alpaca returns."""
    defaults = {
        "headline": "Test headline",
        "published_at": datetime.now(),
        "source": "test",
        "url": "https://example.com",
        "symbols": ["AAPL"],
    }
    defaults.update(kwargs)

    class NewsItem:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    return NewsItem(**defaults)


def make_trading_decision(**kwargs):
    """Create an ExecutorDecision for testing."""
    from v2.agent import ExecutorDecision
    defaults = {
        "playbook_action_id": 1,
        "ticker": "AAPL",
        "action": "buy",
        "quantity": 2.5,
        "reasoning": "Entry trigger hit",
        "confidence": "high",
        "is_off_playbook": False,
        "signal_refs": [{"type": "news_signal", "id": 1}],
        "thesis_id": 1,
    }
    defaults.update(kwargs)
    return ExecutorDecision(**defaults)


def make_agent_response(**kwargs):
    """Create an AgentResponse for testing."""
    from v2.agent import AgentResponse
    defaults = {
        "decisions": [],
        "thesis_invalidations": [],
        "market_summary": "Test summary",
        "risk_assessment": "Low risk",
    }
    defaults.update(kwargs)
    return AgentResponse(**defaults)


def make_ticker_signal(**kwargs):
    """Create a TickerSignal for testing."""
    from v2.classifier import TickerSignal
    defaults = {
        "ticker": "AAPL",
        "headline": "Apple reports earnings",
        "category": "earnings",
        "sentiment": "bullish",
        "confidence": "high",
        "published_at": datetime.now(),
    }
    defaults.update(kwargs)
    return TickerSignal(**defaults)


def make_position_row(**kwargs):
    defaults = {"ticker": "AAPL", "shares": Decimal("10"), "avg_cost": Decimal("150.00")}
    defaults.update(kwargs)
    return defaults


def make_thesis_row(**kwargs):
    defaults = {
        "id": 1, "ticker": "AAPL", "direction": "long", "confidence": "high",
        "thesis": "Strong earnings growth", "entry_trigger": "Price > $150",
        "exit_trigger": "Price > $180", "invalidation": "Earnings miss",
        "status": "active", "source_signals": None,
        "created_at": datetime.now(), "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_decision_row(**kwargs):
    defaults = {
        "id": 1, "date": date.today(), "ticker": "AAPL", "action": "buy",
        "quantity": Decimal("5"), "price": Decimal("150"), "reasoning": "Test",
        "signals_used": {}, "account_equity": Decimal("100000"),
        "buying_power": Decimal("50000"), "outcome_7d": Decimal("2.5"),
        "outcome_30d": None, "playbook_action_id": None, "is_off_playbook": False,
    }
    defaults.update(kwargs)
    return defaults


def make_snapshot_row(**kwargs):
    defaults = {
        "id": 1, "portfolio_value": Decimal("100000"),
        "cash": Decimal("50000"), "buying_power": Decimal("50000"),
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_news_signal_row(**kwargs):
    defaults = {
        "id": 1, "ticker": "AAPL", "headline": "Test", "category": "earnings",
        "sentiment": "bullish", "confidence": "high", "published_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_macro_signal_row(**kwargs):
    defaults = {
        "id": 1, "headline": "Fed holds rates", "category": "fed",
        "affected_sectors": ["finance"], "sentiment": "neutral",
        "published_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_playbook_action_row(**kwargs):
    """Create a playbook action dict like what DB returns."""
    defaults = {
        "id": 1, "playbook_id": 1, "ticker": "AAPL", "action": "buy",
        "thesis_id": 1, "reasoning": "Entry trigger hit", "confidence": "high",
        "max_quantity": Decimal("5"), "priority": 1, "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_strategy_state_row(**kwargs):
    """Create a strategy_state dict like what DB returns."""
    defaults = {
        "id": 1,
        "identity_text": "Momentum-focused trader favoring earnings signals",
        "risk_posture": "moderate",
        "sector_biases": {"tech": "overweight"},
        "preferred_signals": ["earnings", "fed"],
        "avoided_signals": ["legal"],
        "version": 1,
        "is_current": True,
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_strategy_rule_row(**kwargs):
    """Create a strategy_rules dict like what DB returns."""
    defaults = {
        "id": 1,
        "rule_text": "Fade legal news signals — 38% win rate over 12 trades",
        "category": "news_signal:legal",
        "direction": "constraint",
        "confidence": Decimal("0.80"),
        "supporting_evidence": "Historical win rate below 40% across 12 decisions",
        "status": "active",
        "created_at": datetime.now(),
        "retired_at": None,
    }
    defaults.update(kwargs)
    return defaults


def make_strategy_memo_row(**kwargs):
    """Create a strategy_memos dict like what DB returns."""
    defaults = {
        "id": 1,
        "session_date": date.today(),
        "memo_type": "reflection",
        "content": "Today's session showed strong performance in tech earnings plays.",
        "strategy_state_id": 1,
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults
