"""Shared test fixtures and factory functions for v2 module."""

from contextlib import ExitStack, contextmanager
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.attribution import clear_attribution_summary_cache


@pytest.fixture(autouse=True)
def _isolate_attribution_summary_cache():
    """`get_attribution_summary` is memoized per process. Without clearing
    between tests, an earlier mocked result leaks into later tests."""
    clear_attribution_summary_cache()
    yield
    clear_attribution_summary_cache()

# --- Network safety net ---
#
# Background: tests in this directory have historically called
# ``v2.session.run_session`` without patching every outbound I/O function.
# When prod credentials are present in the container env, the unmocked paths
# burn Anthropic tokens.
#
# This autouse fixture pre-patches the session hooks that make network calls.
# Tests that want finer-grained behaviour still win — their own ``@patch``
# decorators override these defaults for the duration of the test.
# Targets whose default patched return is MagicMock (structure doesn't matter
# to callers, they just need a truthy mock).
_NETWORK_PATCH_TARGETS = (
    # Session stage wrappers (import-site names used by v2.session.run_session)
    "v2.session.run_strategy_reflection",
    "v2.session.run_dashboard_stage",
    "v2.session.run_pipeline",
    "v2.session.run_strategist_loop",
    "v2.session.run_trading_session",
    "v2.session.run_backfill",
    "v2.session.compute_signal_attribution",
)

# Client factories must default to returning None so the production
# "no credentials → skip" path runs unless a test explicitly wires up a client.
_NETWORK_PATCH_NONE_TARGETS = ()

_SESSION_DB_PATCH_TARGETS = (
    # Default session tracking to "unavailable" in tests. Individual tests
    # that assert tracking behavior patch these import-site names themselves,
    # and their patches override this fixture while it is active.
    "v2.session.get_session_for_date",
    "v2.session.insert_session_record",
    "v2.session.insert_session_stage",
    "v2.session.complete_session_stage",
    "v2.session.fail_session_stage",
    "v2.session.close_orphan_running_stages",
    "v2.session.complete_session",
    "v2.session.fail_session",
)


@pytest.fixture(autouse=True)
def _block_social_llm_and_session_db_calls(monkeypatch):
    """Defensive net: tests must not reach real posting/LLM/session DB code."""
    monkeypatch.delenv("ALGO_ENABLE_TRADE_POSTS", raising=False)
    with ExitStack() as stack:
        for target in _NETWORK_PATCH_TARGETS:
            stack.enter_context(patch(target))
        for target in _NETWORK_PATCH_NONE_TARGETS:
            stack.enter_context(patch(target, return_value=None))
        for target in _SESSION_DB_PATCH_TARGETS:
            stack.enter_context(patch(target))
        # Returning None makes run_session proceed without telemetry tracking,
        # matching the production fallback when session-row creation fails.
        import v2.session as session_module
        session_module.get_session_for_date.return_value = None
        session_module.insert_session_record.return_value = None
        session_module.close_orphan_running_stages.return_value = []
        yield


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
         patch("v2.dashboard_publish.get_cursor", _get_cursor), \
         patch("v2.tools.get_cursor", _get_cursor), \
         patch("alpaca.data.historical.StockHistoricalDataClient"), \
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
        "intent_type": "invest_dollar",
        "intent_magnitude": Decimal("500"),
        "reasoning": "Entry trigger hit",
        "confidence": "high",
        "is_off_playbook": False,
        "signal_refs": [{"type": "news_signal", "id": 1}],
        "thesis_id": 1,
        "quantity": None,
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
        "intent_type": "invest_dollar", "intent_magnitude": Decimal("500"),
        "priority": 1, "created_at": datetime.now(),
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
        "retirement_reason": None,
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
