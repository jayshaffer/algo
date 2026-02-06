"""Shared test fixtures for the trading platform test suite."""

import json
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from trading.news import NewsItem
from trading.filter import FilteredNewsItem
from trading.classifier import TickerSignal, MacroSignal, ClassificationResult
from trading.agent import TradingDecision, ThesisInvalidation, AgentResponse


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cursor():
    """Mock database cursor that tracks executed SQL."""
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 1}
    cursor.fetchall.return_value = []
    cursor.rowcount = 1
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    """Patch get_cursor to yield a mock cursor."""
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("trading.db.get_cursor", _get_cursor), \
         patch("trading.db.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor


# ---------------------------------------------------------------------------
# Ollama / LLM fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embed():
    """Mock ollama embed to return a fixed 768-dim vector."""
    vec = [0.1] * 768
    with patch("trading.ollama.embed", return_value=vec) as m:
        m.return_vec = vec
        yield m


@pytest.fixture
def mock_embed_batch():
    """Mock ollama embed_batch to return fixed vectors."""
    def _embed_batch(texts, model="nomic-embed-text"):
        return [[0.1] * 768 for _ in texts]

    with patch("trading.ollama.embed_batch", side_effect=_embed_batch) as m:
        yield m


@pytest.fixture
def mock_chat():
    """Mock ollama chat to return a fixed response."""
    with patch("trading.ollama.chat", return_value="mock response") as m:
        yield m


@pytest.fixture
def mock_chat_json():
    """Mock ollama chat_json to return a fixed dict."""
    with patch("trading.ollama.chat_json", return_value={}) as m:
        yield m


# ---------------------------------------------------------------------------
# Alpaca fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_trading_client():
    """Mock Alpaca TradingClient."""
    client = MagicMock()
    # Account
    account = MagicMock()
    account.id = "test-account"
    account.status = "ACTIVE"
    account.cash = "100000.00"
    account.portfolio_value = "150000.00"
    account.buying_power = "200000.00"
    account.long_market_value = "50000.00"
    account.short_market_value = "0.00"
    account.equity = "150000.00"
    account.daytrade_count = 0
    account.pattern_day_trader = False
    client.get_account.return_value = account
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    return client


@pytest.fixture
def alpaca_env(monkeypatch):
    """Set Alpaca environment variables for tests."""
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


@pytest.fixture
def db_env(monkeypatch):
    """Set database environment variable."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")


@pytest.fixture
def ollama_env(monkeypatch):
    """Set Ollama environment variable."""
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")


@pytest.fixture
def claude_env(monkeypatch):
    """Set Claude API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def make_news_item(**kwargs):
    """Create a NewsItem with sensible defaults."""
    defaults = {
        "id": "test-news-1",
        "headline": "AAPL beats Q3 earnings estimates",
        "summary": "Apple reported better than expected earnings",
        "author": "Test Author",
        "source": "test",
        "symbols": ["AAPL"],
        "published_at": datetime(2025, 1, 15, 10, 0, 0),
        "url": "https://example.com/news/1",
    }
    defaults.update(kwargs)
    return NewsItem(**defaults)


def make_trading_decision(**kwargs):
    """Create a TradingDecision with sensible defaults."""
    defaults = {
        "action": "buy",
        "ticker": "AAPL",
        "quantity": 10,
        "reasoning": "Strong earnings beat",
        "confidence": "high",
        "thesis_id": None,
        "signal_refs": [],
    }
    defaults.update(kwargs)
    return TradingDecision(**defaults)


def make_agent_response(**kwargs):
    """Create an AgentResponse with sensible defaults."""
    defaults = {
        "decisions": [make_trading_decision()],
        "thesis_invalidations": [],
        "market_summary": "Market is bullish",
        "risk_assessment": "Low risk environment",
    }
    defaults.update(kwargs)
    return AgentResponse(**defaults)


def make_ticker_signal(**kwargs):
    """Create a TickerSignal with sensible defaults."""
    defaults = {
        "ticker": "AAPL",
        "headline": "AAPL beats Q3 earnings",
        "category": "earnings",
        "sentiment": "bullish",
        "confidence": "high",
        "published_at": datetime(2025, 1, 15, 10, 0, 0),
    }
    defaults.update(kwargs)
    return TickerSignal(**defaults)


def make_position_row(**kwargs):
    """Create a position dict like what DB returns."""
    defaults = {
        "id": 1,
        "ticker": "AAPL",
        "shares": Decimal("10"),
        "avg_cost": Decimal("150.00"),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_thesis_row(**kwargs):
    """Create a thesis dict like what DB returns."""
    defaults = {
        "id": 1,
        "ticker": "AAPL",
        "direction": "long",
        "thesis": "Strong fundamentals",
        "entry_trigger": "Price drops to $140",
        "exit_trigger": "Price hits $180",
        "invalidation": "Revenue declines 2 quarters",
        "confidence": "high",
        "source": "ideation",
        "status": "active",
        "created_at": datetime.now() - timedelta(days=5),
        "updated_at": datetime.now(),
        "closed_at": None,
        "close_reason": None,
    }
    defaults.update(kwargs)
    return defaults


def make_decision_row(**kwargs):
    """Create a decision dict like what DB returns."""
    defaults = {
        "id": 1,
        "date": date.today() - timedelta(days=10),
        "ticker": "AAPL",
        "action": "buy",
        "quantity": Decimal("10"),
        "price": Decimal("150.00"),
        "reasoning": "Earnings beat",
        "signals_used": {"market_summary": "bullish"},
        "account_equity": Decimal("100000"),
        "buying_power": Decimal("50000"),
        "outcome_7d": Decimal("2.5"),
        "outcome_30d": Decimal("5.0"),
    }
    defaults.update(kwargs)
    return defaults


def make_snapshot_row(**kwargs):
    """Create an account snapshot dict like what DB returns."""
    defaults = {
        "id": 1,
        "date": date.today(),
        "cash": Decimal("100000"),
        "portfolio_value": Decimal("150000"),
        "buying_power": Decimal("200000"),
        "long_market_value": Decimal("50000"),
        "short_market_value": Decimal("0"),
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_news_signal_row(**kwargs):
    """Create a news signal dict like what DB returns."""
    defaults = {
        "id": 1,
        "ticker": "AAPL",
        "headline": "AAPL Q3 earnings beat",
        "category": "earnings",
        "sentiment": "bullish",
        "confidence": "high",
        "published_at": datetime.now() - timedelta(hours=2),
        "processed_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_macro_signal_row(**kwargs):
    """Create a macro signal dict like what DB returns."""
    defaults = {
        "id": 1,
        "headline": "Fed holds rates steady",
        "category": "fed",
        "affected_sectors": ["finance", "tech"],
        "sentiment": "bullish",
        "published_at": datetime.now() - timedelta(hours=3),
        "processed_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_playbook_row(**kwargs):
    """Create a playbook dict like what DB returns."""
    defaults = {
        "id": 1,
        "date": date.today(),
        "market_outlook": "Bullish on tech, cautious on energy",
        "priority_actions": [
            {"ticker": "NVDA", "action": "buy", "thesis_id": 1, "reasoning": "Entry trigger hit", "max_quantity": 5, "confidence": 0.8}
        ],
        "watch_list": ["AAPL", "MSFT", "GOOGL"],
        "risk_notes": "Fed meeting tomorrow, watch for volatility",
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_attribution_row(**kwargs):
    """Create a signal attribution dict like what DB returns."""
    defaults = {
        "id": 1,
        "category": "news:earnings",
        "sample_size": 20,
        "avg_outcome_7d": Decimal("1.50"),
        "avg_outcome_30d": Decimal("3.20"),
        "win_rate_7d": Decimal("0.62"),
        "win_rate_30d": Decimal("0.55"),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults
