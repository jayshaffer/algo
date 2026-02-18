"""Tests for dashboard/app.py - Flask dashboard routes."""

import json
import sys
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    make_position_row,
    make_snapshot_row,
    make_thesis_row,
    make_decision_row,
    make_news_signal_row,
    make_macro_signal_row,
    make_playbook_row,
    make_attribution_row,
    make_open_order_row,
    make_playbook_action_row,
)

# Inject a mock queries module before importing dashboard.app,
# since app.py does `from queries import ...` (top-level import).
mock_queries = MagicMock()
sys.modules["queries"] = mock_queries

from dashboard.app import app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_query_mocks():
    """Reset call history and return values before each test.

    We must NOT call mock_queries.reset_mock() because `from queries import
    close_thesis` in dashboard/app.py binds the attribute at import time.
    reset_mock() would replace child mocks, leaving app.py holding a stale
    reference.  Instead we reset each attribute individually.
    """
    for attr in dir(mock_queries):
        child = getattr(mock_queries, attr, None)
        if isinstance(child, MagicMock) and not attr.startswith("_"):
            child.reset_mock()
            child.side_effect = None
    # Set safe defaults so template rendering doesn't blow up
    mock_queries.get_positions.return_value = []
    mock_queries.get_latest_snapshot.return_value = None
    mock_queries.get_open_orders.return_value = []
    mock_queries.get_today_playbook.return_value = None
    mock_queries.get_playbook_actions.return_value = []
    mock_queries.get_signal_attribution.return_value = []
    mock_queries.get_recent_ticker_signals.return_value = []
    mock_queries.get_recent_macro_signals.return_value = []
    mock_queries.get_signal_summary.return_value = []
    mock_queries.get_thesis_stats.return_value = {
        "active": 0,
        "executed": 0,
        "invalidated": 0,
        "expired": 0,
        "success_rate": None,
        "confidence_dist": {},
    }
    mock_queries.get_theses.return_value = []
    mock_queries.get_recent_decisions.return_value = []
    mock_queries.get_decision_stats.return_value = {
        "total_decisions": 0,
        "buys": 0,
        "sells": 0,
        "holds": 0,
        "avg_outcome_7d": None,
        "avg_outcome_30d": None,
    }
    mock_queries.get_decision_signal_refs_batch.return_value = {}
    mock_queries.get_equity_curve.return_value = []
    mock_queries.get_performance_metrics.return_value = None
    mock_queries.close_thesis.return_value = True
    yield


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"status": "healthy"}


# ---------------------------------------------------------------------------
# Portfolio page
# ---------------------------------------------------------------------------


class TestPortfolioPage:
    """Tests for GET / (portfolio)."""

    def test_portfolio_renders_empty(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        mock_queries.get_positions.assert_called_once()
        mock_queries.get_latest_snapshot.assert_called_once()
        mock_queries.get_today_playbook.assert_called_once()

    def test_portfolio_passes_data_to_template(self, client):
        positions = [make_position_row(ticker="AAPL"), make_position_row(ticker="TSLA", id=2)]
        snapshot = make_snapshot_row()
        playbook = make_playbook_row()

        mock_queries.get_positions.return_value = positions
        mock_queries.get_latest_snapshot.return_value = snapshot
        mock_queries.get_today_playbook.return_value = playbook

        resp = client.get("/")
        assert resp.status_code == 200

    def test_portfolio_with_none_snapshot(self, client):
        mock_queries.get_latest_snapshot.return_value = None
        resp = client.get("/")
        assert resp.status_code == 200

    def test_portfolio_renders_open_orders(self, client):
        orders = [
            make_open_order_row(ticker="AAPL", side="buy", status="new"),
            make_open_order_row(id=2, ticker="TSLA", side="sell", status="partially_filled"),
        ]
        mock_queries.get_open_orders.return_value = orders

        resp = client.get("/")
        assert resp.status_code == 200
        assert b"AAPL" in resp.data
        assert b"TSLA" in resp.data
        assert b"BUY" in resp.data
        assert b"SELL" in resp.data
        mock_queries.get_open_orders.assert_called_once()

    def test_portfolio_empty_open_orders(self, client):
        mock_queries.get_open_orders.return_value = []
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"No open orders" in resp.data


# ---------------------------------------------------------------------------
# Playbook page
# ---------------------------------------------------------------------------


class TestPlaybookPage:
    """Tests for GET /playbook."""

    def test_playbook_renders_with_data(self, client):
        playbook = make_playbook_row()
        mock_queries.get_today_playbook.return_value = playbook

        resp = client.get("/playbook")
        assert resp.status_code == 200
        mock_queries.get_today_playbook.assert_called_once()

    def test_playbook_renders_empty(self, client):
        mock_queries.get_today_playbook.return_value = None

        resp = client.get("/playbook")
        assert resp.status_code == 200
        assert b"No playbook for today" in resp.data

    def test_playbook_renders_structured_actions(self, client):
        playbook = make_playbook_row()
        mock_queries.get_today_playbook.return_value = playbook
        actions = [
            make_playbook_action_row(ticker="NVDA", confidence="high", priority=1),
            make_playbook_action_row(id=2, ticker="AAPL", confidence="medium", priority=2, thesis_id=None),
        ]
        mock_queries.get_playbook_actions.return_value = actions

        resp = client.get("/playbook")
        assert resp.status_code == 200
        assert b"NVDA" in resp.data
        assert b"AAPL" in resp.data
        # Confidence badges
        assert b">high<" in resp.data
        assert b">medium<" in resp.data
        mock_queries.get_playbook_actions.assert_called_once_with(playbook['id'])

    def test_playbook_empty_actions(self, client):
        playbook = make_playbook_row()
        mock_queries.get_today_playbook.return_value = playbook
        mock_queries.get_playbook_actions.return_value = []

        resp = client.get("/playbook")
        assert resp.status_code == 200
        assert b"No priority actions" in resp.data

    def test_playbook_null_playbook_no_actions_query(self, client):
        mock_queries.get_today_playbook.return_value = None

        resp = client.get("/playbook")
        assert resp.status_code == 200
        mock_queries.get_playbook_actions.assert_not_called()


# ---------------------------------------------------------------------------
# Attribution page
# ---------------------------------------------------------------------------


class TestAttributionPage:
    """Tests for GET /attribution."""

    def test_attribution_renders_with_data(self, client):
        scores = [make_attribution_row(), make_attribution_row(category="macro:fed", id=2)]
        mock_queries.get_signal_attribution.return_value = scores

        resp = client.get("/attribution")
        assert resp.status_code == 200
        mock_queries.get_signal_attribution.assert_called_once()

    def test_attribution_renders_empty(self, client):
        mock_queries.get_signal_attribution.return_value = []

        resp = client.get("/attribution")
        assert resp.status_code == 200
        assert b"No attribution data available" in resp.data


# ---------------------------------------------------------------------------
# Signals page
# ---------------------------------------------------------------------------


class TestSignalsPage:
    """Tests for GET /signals."""

    def test_signals_renders(self, client):
        resp = client.get("/signals")
        assert resp.status_code == 200
        mock_queries.get_recent_ticker_signals.assert_called_once_with(days=7, limit=50)
        mock_queries.get_recent_macro_signals.assert_called_once_with(days=7, limit=20)
        mock_queries.get_signal_summary.assert_called_once_with(days=7)

    def test_signals_with_data(self, client):
        mock_queries.get_recent_ticker_signals.return_value = [make_news_signal_row()]
        mock_queries.get_recent_macro_signals.return_value = [make_macro_signal_row()]
        mock_queries.get_signal_summary.return_value = [
            {"ticker": "AAPL", "total": 5, "bullish": 3, "bearish": 1, "neutral": 1}
        ]

        resp = client.get("/signals")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Theses page
# ---------------------------------------------------------------------------


class TestThesesPage:
    """Tests for GET /theses."""

    def test_theses_default_params(self, client):
        resp = client.get("/theses")
        assert resp.status_code == 200
        mock_queries.get_thesis_stats.assert_called_once()
        mock_queries.get_theses.assert_called_once_with("active", "newest")

    def test_theses_with_status_filter(self, client):
        resp = client.get("/theses?status=invalidated")
        assert resp.status_code == 200
        mock_queries.get_theses.assert_called_once_with("invalidated", "newest")

    def test_theses_with_sort_by(self, client):
        resp = client.get("/theses?sort=confidence")
        assert resp.status_code == 200
        mock_queries.get_theses.assert_called_once_with("active", "confidence")

    def test_theses_with_both_params(self, client):
        resp = client.get("/theses?status=all&sort=ticker")
        assert resp.status_code == 200
        mock_queries.get_theses.assert_called_once_with("all", "ticker")


# ---------------------------------------------------------------------------
# Decisions page
# ---------------------------------------------------------------------------


class TestDecisionsPage:
    """Tests for GET /decisions."""

    def test_decisions_renders(self, client):
        resp = client.get("/decisions")
        assert resp.status_code == 200
        mock_queries.get_recent_decisions.assert_called_once_with(days=30, limit=50)
        mock_queries.get_decision_stats.assert_called_once_with(days=30)

    def test_decisions_with_data(self, client):
        mock_queries.get_recent_decisions.return_value = [make_decision_row()]
        mock_queries.get_decision_stats.return_value = {
            "total_decisions": 10,
            "buys": 5,
            "sells": 3,
            "holds": 2,
            "avg_outcome_7d": Decimal("1.5"),
            "avg_outcome_30d": Decimal("3.0"),
        }

        resp = client.get("/decisions")
        assert resp.status_code == 200

    def test_decisions_off_playbook_badge(self, client):
        decision = make_decision_row(id=10, is_off_playbook=True)
        mock_queries.get_recent_decisions.return_value = [decision]

        resp = client.get("/decisions")
        assert resp.status_code == 200
        assert b"Off-Playbook" in resp.data

    def test_decisions_playbook_badge(self, client):
        decision = make_decision_row(id=10, playbook_action_id=5)
        mock_queries.get_recent_decisions.return_value = [decision]

        resp = client.get("/decisions")
        assert resp.status_code == 200
        assert b"Playbook" in resp.data

    def test_decisions_signal_refs_rendering(self, client):
        decision = make_decision_row(id=10)
        mock_queries.get_recent_decisions.return_value = [decision]
        mock_queries.get_decision_signal_refs_batch.return_value = {
            10: [
                {"signal_type": "news_signal", "signal_id": 1, "label": "AAPL beats earnings"},
                {"signal_type": "thesis", "signal_id": 2, "label": "Strong fundamentals"},
            ]
        }

        resp = client.get("/decisions")
        assert resp.status_code == 200
        assert b"AAPL beats earnings" in resp.data
        assert b"Strong fundamentals" in resp.data
        mock_queries.get_decision_signal_refs_batch.assert_called_once_with([10])

    def test_decisions_empty_signal_refs(self, client):
        decision = make_decision_row(id=10)
        mock_queries.get_recent_decisions.return_value = [decision]
        mock_queries.get_decision_signal_refs_batch.return_value = {}

        resp = client.get("/decisions")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Performance page
# ---------------------------------------------------------------------------


class TestPerformancePage:
    """Tests for GET /performance."""

    def test_performance_renders_empty(self, client):
        resp = client.get("/performance")
        assert resp.status_code == 200
        mock_queries.get_equity_curve.assert_called_once_with(days=90)
        mock_queries.get_performance_metrics.assert_called_once_with(days=30)

    def test_performance_with_equity_data(self, client):
        mock_queries.get_equity_curve.return_value = [
            {
                "date": date(2025, 1, 1),
                "portfolio_value": Decimal("100000"),
                "cash": Decimal("50000"),
                "buying_power": Decimal("80000"),
            },
            {
                "date": date(2025, 1, 2),
                "portfolio_value": Decimal("101000"),
                "cash": Decimal("49000"),
                "buying_power": Decimal("79000"),
            },
        ]
        mock_queries.get_performance_metrics.return_value = {
            "start_value": 100000.0,
            "end_value": 101000.0,
            "pnl": 1000.0,
            "pnl_pct": 1.0,
            "start_date": date(2025, 1, 1),
            "end_date": date(2025, 1, 2),
        }

        resp = client.get("/performance")
        assert resp.status_code == 200

    def test_performance_with_none_equity_curve(self, client):
        mock_queries.get_equity_curve.return_value = None
        resp = client.get("/performance")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API: close thesis
# ---------------------------------------------------------------------------


class TestApiCloseThesis:
    """Tests for POST /api/theses/<id>/close."""

    def test_close_thesis_invalidated(self, client):
        mock_queries.close_thesis.return_value = True
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated", "reason": "Revenue declined"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        mock_queries.close_thesis.assert_called_once_with(1, "invalidated", "Revenue declined")

    def test_close_thesis_expired(self, client):
        mock_queries.close_thesis.return_value = True
        resp = client.post(
            "/api/theses/5/close",
            json={"status": "expired", "reason": "Thesis aged out"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        mock_queries.close_thesis.assert_called_once_with(5, "expired", "Thesis aged out")

    def test_close_thesis_invalid_status_returns_400(self, client):
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "active"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "Invalid status" in data["error"]
        mock_queries.close_thesis.assert_not_called()

    def test_close_thesis_missing_status_returns_400(self, client):
        resp = client.post(
            "/api/theses/1/close",
            json={"reason": "some reason"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "Invalid status" in data["error"]

    def test_close_thesis_not_found_returns_404(self, client):
        mock_queries.close_thesis.return_value = False
        resp = client.post(
            "/api/theses/999/close",
            json={"status": "invalidated"},
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert "not found" in data["error"].lower() or "already closed" in data["error"].lower()

    def test_close_thesis_exception_returns_500(self, client):
        mock_queries.close_thesis.side_effect = RuntimeError("DB connection failed")
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated"},
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["error"] == "Internal server error"

    def test_close_thesis_empty_reason_becomes_none(self, client):
        mock_queries.close_thesis.return_value = True
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated", "reason": "   "},
        )
        assert resp.status_code == 200
        mock_queries.close_thesis.assert_called_once_with(1, "invalidated", None)

    def test_close_thesis_no_body(self, client):
        resp = client.post(
            "/api/theses/1/close",
            content_type="application/json",
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# API: portfolio
# ---------------------------------------------------------------------------


class TestApiPortfolio:
    """Tests for GET /api/portfolio."""

    def test_api_portfolio_empty(self, client):
        resp = client.get("/api/portfolio")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["positions"] == []
        assert data["snapshot"] is None

    def test_api_portfolio_with_data(self, client):
        pos = {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0}
        snap = {"cash": 100000, "portfolio_value": 150000}

        mock_queries.get_positions.return_value = [pos]
        mock_queries.get_latest_snapshot.return_value = snap

        resp = client.get("/api/portfolio")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["positions"]) == 1
        assert data["snapshot"]["cash"] == 100000


# ---------------------------------------------------------------------------
# API: signals
# ---------------------------------------------------------------------------


class TestApiSignals:
    """Tests for GET /api/signals."""

    def test_api_signals_empty(self, client):
        resp = client.get("/api/signals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ticker_signals"] == []
        assert data["macro_signals"] == []

    def test_api_signals_with_data(self, client):
        ts = {"ticker": "AAPL", "sentiment": "bullish"}
        ms = {"headline": "Fed holds", "sentiment": "bullish"}

        mock_queries.get_recent_ticker_signals.return_value = [ts]
        mock_queries.get_recent_macro_signals.return_value = [ms]

        resp = client.get("/api/signals")
        assert resp.status_code == 200
        mock_queries.get_recent_ticker_signals.assert_called_once_with(days=7, limit=50)
        mock_queries.get_recent_macro_signals.assert_called_once_with(days=7, limit=20)

    def test_api_signals_with_none_returns(self, client):
        mock_queries.get_recent_ticker_signals.return_value = None
        mock_queries.get_recent_macro_signals.return_value = None

        resp = client.get("/api/signals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ticker_signals"] == []
        assert data["macro_signals"] == []
