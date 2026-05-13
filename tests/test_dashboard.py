"""Tests for dashboard/app.py - Flask dashboard routes."""

import sys
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tests.conftest import (
    make_agent_event_row,
    make_attribution_row,
    make_decision_row,
    make_macro_signal_row,
    make_news_signal_row,
    make_open_order_row,
    make_playbook_action_row,
    make_playbook_row,
    make_position_row,
    make_session_row,
    make_snapshot_row,
    make_strategy_memo_row,
    make_strategy_rule_row,
    make_strategy_state_row,
    make_thesis_row,
    make_tweet_row,
)

# Inject a mock queries module before importing dashboard.app,
# since app.py does `from queries import ...` (top-level import).
mock_queries = MagicMock()
sys.modules["queries"] = mock_queries

mock_benchmark = MagicMock()
mock_benchmark.get_spy_benchmark.return_value = []
mock_benchmark.compute_alpha.return_value = None
mock_benchmark.get_deposit_history.return_value = []
mock_benchmark.enrich_snapshots_with_deposits.side_effect = lambda snaps, deps: list(snaps)
sys.modules["benchmark"] = mock_benchmark

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
    mock_queries.get_current_strategy.return_value = None
    mock_queries.get_strategy_rules.return_value = []
    mock_queries.get_strategy_memos.return_value = []
    mock_queries.get_recent_tweets.return_value = []
    mock_queries.get_decision.return_value = None
    mock_queries.get_decision_signals_full.return_value = []
    mock_queries.get_decision_tweets.return_value = []
    mock_queries.get_playbook_action.return_value = None
    mock_queries.lookup_session_id_by_date.return_value = None
    mock_queries.get_thesis.return_value = None
    mock_queries.get_thesis_decisions.return_value = []
    mock_queries.get_thesis_playbook_actions.return_value = []
    mock_queries.get_ticker_position.return_value = None
    mock_queries.get_ticker_theses.return_value = []
    mock_queries.get_ticker_decisions.return_value = []
    mock_queries.get_ticker_signals.return_value = []
    mock_queries.get_ticker_open_orders.return_value = []
    mock_queries.get_ticker_attribution.return_value = []
    mock_queries.get_session.return_value = None
    mock_queries.get_session_stage_costs.return_value = []
    mock_queries.get_session_decisions.return_value = []
    mock_queries.get_session_theses_created.return_value = []
    mock_queries.get_session_memo.return_value = None
    mock_queries.get_session_tweets.return_value = []
    mock_queries.get_session_events.return_value = []
    mock_queries.get_session_llm_calls.return_value = []
    mock_queries.get_llm_call.return_value = None
    mock_benchmark.get_spy_benchmark.reset_mock()
    mock_benchmark.compute_alpha.reset_mock()
    mock_benchmark.get_deposit_history.reset_mock()
    mock_benchmark.enrich_snapshots_with_deposits.reset_mock()
    mock_benchmark.get_spy_benchmark.return_value = []
    mock_benchmark.compute_alpha.return_value = None
    mock_benchmark.get_deposit_history.return_value = []
    mock_benchmark.enrich_snapshots_with_deposits.side_effect = lambda snaps, deps: list(snaps)
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

    def test_portfolio_computes_alpha_when_equity_data_available(self, client):
        mock_queries.get_equity_curve.return_value = [
            make_snapshot_row(date=date(2026, 1, 2), portfolio_value=Decimal("100000")),
            make_snapshot_row(date=date(2026, 1, 6), portfolio_value=Decimal("105000")),
        ]
        mock_benchmark.get_spy_benchmark.return_value = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 505.0},
        ]
        mock_benchmark.compute_alpha.return_value = {
            "portfolio_return": 5.0,
            "spy_return": 1.0,
            "alpha": 4.0,
        }

        resp = client.get("/")
        assert resp.status_code == 200
        assert b"+4.00%" in resp.data

    def test_portfolio_handles_empty_benchmark(self, client):
        mock_queries.get_equity_curve.return_value = []
        mock_benchmark.get_spy_benchmark.return_value = []
        mock_benchmark.compute_alpha.return_value = None

        resp = client.get("/")
        assert resp.status_code == 200
        assert b"mdash" in resp.data

    def test_position_ticker_links_to_ticker_page(self, client):
        mock_queries.get_positions.return_value = [make_position_row(ticker="AAPL")]
        mock_queries.get_open_orders.return_value = []
        resp = client.get("/")
        assert b'href="/ticker/AAPL"' in resp.data

    def test_open_order_ticker_links(self, client):
        mock_queries.get_positions.return_value = []
        mock_queries.get_open_orders.return_value = [make_open_order_row(ticker="TSLA")]
        resp = client.get("/")
        assert b'href="/ticker/TSLA"' in resp.data


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

    def test_playbook_links_ticker_and_thesis(self, client):
        mock_queries.get_today_playbook.return_value = make_playbook_row()
        mock_queries.get_playbook_actions.return_value = [
            make_playbook_action_row(ticker="NVDA", thesis_id=7),
        ]
        resp = client.get("/playbook")
        assert b'href="/ticker/NVDA"' in resp.data
        assert b'href="/thesis/7"' in resp.data


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

    def test_attribution_categories_link_and_filter(self, client):
        mock_queries.get_signal_attribution.return_value = [
            make_attribution_row(category="news:earnings"),
        ]
        resp = client.get("/attribution")
        # Category cell should be a link that filters the same page
        assert b'href="/attribution?category=news%3Aearnings"' in resp.data or \
               b'href="/attribution?category=news:earnings"' in resp.data
        # Anchor target for cross-page links
        assert b'id="cat-news:earnings"' in resp.data

    def test_attribution_accepts_category_filter(self, client):
        mock_queries.get_signal_attribution.return_value = []
        resp = client.get("/attribution?category=news:earnings")
        assert resp.status_code == 200
        mock_queries.get_signal_attribution.assert_called_with(category="news:earnings")


# ---------------------------------------------------------------------------
# Signals page
# ---------------------------------------------------------------------------


class TestSignalsPage:
    """Tests for GET /signals."""

    def test_signals_renders(self, client):
        resp = client.get("/signals")
        assert resp.status_code == 200
        mock_queries.get_recent_ticker_signals.assert_called_once_with(days=7, limit=50, category=None)
        mock_queries.get_recent_macro_signals.assert_called_once_with(days=7, limit=20, category=None)
        mock_queries.get_signal_summary.assert_called_once_with(days=7)

    def test_signals_with_data(self, client):
        mock_queries.get_recent_ticker_signals.return_value = [make_news_signal_row()]
        mock_queries.get_recent_macro_signals.return_value = [make_macro_signal_row()]
        mock_queries.get_signal_summary.return_value = [
            {"ticker": "AAPL", "total": 5, "bullish": 3, "bearish": 1, "neutral": 1}
        ]

        resp = client.get("/signals")
        assert resp.status_code == 200

    def test_signals_links_tickers_and_categories(self, client):
        mock_queries.get_recent_ticker_signals.return_value = [
            make_news_signal_row(ticker="AAPL", category="earnings"),
        ]
        mock_queries.get_recent_macro_signals.return_value = []
        mock_queries.get_signal_summary.return_value = [
            {"ticker": "AAPL", "bullish": 2, "bearish": 1, "neutral": 0},
        ]
        resp = client.get("/signals")
        assert b'href="/ticker/AAPL"' in resp.data
        assert b'href="/attribution?category=news%3Aearnings"' in resp.data or \
               b'href="/attribution?category=news:earnings"' in resp.data

    def test_signals_category_filter_param(self, client):
        mock_queries.get_recent_ticker_signals.return_value = []
        mock_queries.get_recent_macro_signals.return_value = []
        mock_queries.get_signal_summary.return_value = []
        resp = client.get("/signals?category=earnings")
        assert resp.status_code == 200
        mock_queries.get_recent_ticker_signals.assert_called_with(days=7, limit=50, category="earnings")


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

    def test_theses_link_ticker_and_detail(self, client):
        mock_queries.get_theses.return_value = [make_thesis_row(id=11, ticker="MSFT")]
        resp = client.get("/theses")
        assert b'href="/ticker/MSFT"' in resp.data
        assert b'href="/thesis/11"' in resp.data


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

    def test_decisions_link_ticker_signal_refs_and_detail(self, client):
        mock_queries.get_recent_decisions.return_value = [
            make_decision_row(id=20, ticker="GOOG", playbook_action_id=5),
        ]
        mock_queries.get_decision_signal_refs_batch.return_value = {
            20: [
                {"signal_type": "news_signal", "signal_id": 100, "label": "Big news"},
                {"signal_type": "thesis", "signal_id": 7, "label": "Strong thesis"},
            ],
        }
        resp = client.get("/decisions")
        assert b'href="/ticker/GOOG"' in resp.data
        assert b'href="/decision/20"' in resp.data
        # news signal links into decision detail anchor
        assert b'href="/decision/20#signal-news-100"' in resp.data
        # thesis ref links directly to thesis detail
        assert b'href="/thesis/7"' in resp.data


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
# Performance page — benchmark integration
# ---------------------------------------------------------------------------


class TestPerformancePageBenchmark:
    """Tests for SPY benchmark on /performance."""

    def test_performance_passes_benchmark_data(self, client):
        eq = [
            make_snapshot_row(date=date(2026, 1, 2), portfolio_value=Decimal("100000")),
            make_snapshot_row(date=date(2026, 1, 6), portfolio_value=Decimal("105000")),
        ]
        mock_queries.get_equity_curve.return_value = eq
        mock_queries.get_performance_metrics.return_value = {
            "start_value": 100000,
            "end_value": 105000,
            "pnl": 5000,
            "pnl_pct": 5.0,
            "start_date": date(2026, 1, 2),
            "end_date": date(2026, 1, 6),
        }
        mock_benchmark.get_spy_benchmark.return_value = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 505.0},
        ]
        mock_benchmark.compute_alpha.return_value = {
            "portfolio_return": 5.0,
            "spy_return": 1.0,
            "alpha": 4.0,
        }

        resp = client.get("/performance")
        assert resp.status_code == 200
        assert b"+4.00%" in resp.data
        assert b"S&amp;P 500" in resp.data or b"S&P 500" in resp.data

    def test_performance_no_benchmark_degrades(self, client):
        mock_queries.get_equity_curve.return_value = []
        mock_queries.get_performance_metrics.return_value = None
        mock_benchmark.get_spy_benchmark.return_value = []
        mock_benchmark.compute_alpha.return_value = None

        resp = client.get("/performance")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API: close thesis
# ---------------------------------------------------------------------------


class TestApiCloseThesis:
    """Tests for POST /api/theses/<id>/close."""

    # P1.10: dashboard mutating endpoints require X-Requested-With: dashboard
    # as defense-in-depth against cross-origin form-based CSRF.
    HEADERS = {"X-Requested-With": "dashboard"}

    def test_close_thesis_invalidated(self, client):
        mock_queries.close_thesis.return_value = True
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated", "reason": "Revenue declined"},
            headers=self.HEADERS,
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
            headers=self.HEADERS,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        mock_queries.close_thesis.assert_called_once_with(5, "expired", "Thesis aged out")

    def test_close_thesis_invalid_status_returns_400(self, client):
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "active"},
            headers=self.HEADERS,
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
            headers=self.HEADERS,
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "Invalid status" in data["error"]

    def test_close_thesis_not_found_returns_404(self, client):
        mock_queries.close_thesis.return_value = False
        resp = client.post(
            "/api/theses/999/close",
            json={"status": "invalidated"},
            headers=self.HEADERS,
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert "not found" in data["error"].lower() or "already closed" in data["error"].lower()

    def test_close_thesis_exception_returns_500(self, client):
        mock_queries.close_thesis.side_effect = RuntimeError("DB connection failed")
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated"},
            headers=self.HEADERS,
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["error"] == "Internal server error"

    def test_close_thesis_empty_reason_becomes_none(self, client):
        mock_queries.close_thesis.return_value = True
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated", "reason": "   "},
            headers=self.HEADERS,
        )
        assert resp.status_code == 200
        mock_queries.close_thesis.assert_called_once_with(1, "invalidated", None)

    def test_close_thesis_no_body(self, client):
        resp = client.post(
            "/api/theses/1/close",
            content_type="application/json",
            headers=self.HEADERS,
        )
        assert resp.status_code == 400

    def test_close_thesis_without_csrf_header_returns_403(self, client):
        """P1.10: missing X-Requested-With must reject before any DB work."""
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated"},
        )
        assert resp.status_code == 403
        mock_queries.close_thesis.assert_not_called()

    def test_close_thesis_with_wrong_csrf_header_returns_403(self, client):
        """P1.10: a wrong header value must also reject."""
        resp = client.post(
            "/api/theses/1/close",
            json={"status": "invalidated"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 403
        mock_queries.close_thesis.assert_not_called()


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


# ---------------------------------------------------------------------------
# Strategy page
# ---------------------------------------------------------------------------


class TestStrategyPage:
    """Tests for GET /strategy."""

    def test_strategy_renders_with_data(self, client):
        state = make_strategy_state_row()
        rules = [
            make_strategy_rule_row(),
            make_strategy_rule_row(id=2, rule_text="Avoid buying before Fed meetings", direction="short", confidence=Decimal("0.60")),
        ]
        memos = [make_strategy_memo_row()]

        mock_queries.get_current_strategy.return_value = state
        mock_queries.get_strategy_rules.return_value = rules
        mock_queries.get_strategy_memos.return_value = memos

        resp = client.get("/strategy")
        assert resp.status_code == 200
        assert b"Momentum-focused" in resp.data
        assert b"moderate" in resp.data
        assert b"Buy on earnings beat" in resp.data
        assert b"Avoid buying before Fed" in resp.data
        assert b"reflection" in resp.data
        mock_queries.get_current_strategy.assert_called_once()
        mock_queries.get_strategy_rules.assert_called_once_with(status='active')
        mock_queries.get_strategy_memos.assert_called_once_with(days=30)

    def test_strategy_empty_state(self, client):
        resp = client.get("/strategy")
        assert resp.status_code == 200
        assert b"No strategy state found" in resp.data

    def test_strategy_no_rules(self, client):
        mock_queries.get_current_strategy.return_value = make_strategy_state_row()
        mock_queries.get_strategy_rules.return_value = []

        resp = client.get("/strategy")
        assert resp.status_code == 200
        assert b"No active rules" in resp.data

    def test_strategy_no_memos(self, client):
        mock_queries.get_current_strategy.return_value = make_strategy_state_row()

        resp = client.get("/strategy")
        assert resp.status_code == 200
        assert b"No memos in the last 30 days" in resp.data

    def test_strategy_memo_links_to_session(self, client):
        memo_date = date(2026, 5, 11)
        mock_queries.get_current_strategy.return_value = make_strategy_state_row()
        mock_queries.get_strategy_rules.return_value = []
        mock_queries.get_strategy_memos.return_value = [
            make_strategy_memo_row(session_date=memo_date),
        ]
        mock_queries.lookup_session_id_by_date.return_value = 42
        resp = client.get("/strategy")
        assert b'href="/session/42"' in resp.data

    def test_strategy_memo_without_session(self, client):
        mock_queries.get_current_strategy.return_value = make_strategy_state_row()
        mock_queries.get_strategy_rules.return_value = []
        mock_queries.get_strategy_memos.return_value = [make_strategy_memo_row()]
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/strategy")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tweets page
# ---------------------------------------------------------------------------


class TestTweetsPage:
    """Tests for GET /tweets."""

    def test_tweets_renders_with_data(self, client):
        tweets = [
            make_tweet_row(),
            make_tweet_row(id=2, tweet_type="trade_alert", posted=False, error=None),
        ]
        mock_queries.get_recent_tweets.return_value = tweets

        resp = client.get("/tweets")
        assert resp.status_code == 200
        assert b"Markets closed green" in resp.data
        assert b"Posted" in resp.data
        assert b"Pending" in resp.data
        mock_queries.get_recent_tweets.assert_called_once_with(days=30, limit=50)

    def test_tweets_empty_state(self, client):
        resp = client.get("/tweets")
        assert resp.status_code == 200
        assert b"No tweets yet" in resp.data

    def test_tweets_error_display(self, client):
        tweets = [make_tweet_row(posted=False, error="Rate limit exceeded")]
        mock_queries.get_recent_tweets.return_value = tweets

        resp = client.get("/tweets")
        assert resp.status_code == 200
        assert b"Failed" in resp.data
        assert b"Rate limit exceeded" in resp.data

    def test_tweets_link_session_and_decision(self, client):
        mock_queries.get_recent_tweets.return_value = [
            {**make_tweet_row(id=1), "decision_id": 11, "session_id": 42},
        ]
        resp = client.get("/tweets")
        assert b'href="/session/42"' in resp.data
        assert b'href="/decision/11"' in resp.data


# ---------------------------------------------------------------------------
# Signals: summary column
# ---------------------------------------------------------------------------


class TestSignalsSummaryColumn:
    """Regression: news_signals.summary must render on /signals."""

    def test_signals_renders_summary_when_present(self, client):
        mock_queries.get_recent_ticker_signals.return_value = [
            make_news_signal_row(summary="Filtered summary text exposed to UI")
        ]

        resp = client.get("/signals")
        assert resp.status_code == 200
        assert b"Filtered summary text exposed to UI" in resp.data

    def test_signals_handles_missing_summary(self, client):
        # summary can be NULL for old rows; template must not 500.
        row = make_news_signal_row()
        row["summary"] = None
        mock_queries.get_recent_ticker_signals.return_value = [row]

        resp = client.get("/signals")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Costs page
# ---------------------------------------------------------------------------


class TestCostsPage:
    """Tests for GET /costs and GET /costs/<session_id>."""

    def test_costs_renders_empty(self, client):
        mock_queries.get_recent_session_costs.return_value = []

        resp = client.get("/costs")
        assert resp.status_code == 200
        assert b"No session cost data yet" in resp.data
        mock_queries.get_recent_session_costs.assert_called_once_with(limit=30)

    def test_costs_renders_with_data(self, client):
        mock_queries.get_recent_session_costs.return_value = [
            {
                "session_id": 42,
                "session_date": date(2026, 5, 9),
                "session_type": "daily",
                "status": "completed",
                "total_cost_usd": Decimal("1.2345"),
                "total_input_tokens": 12345,
                "total_output_tokens": 678,
                "total_cache_creation_tokens": 9000,
                "total_cache_read_tokens": 1000,
                "started_at": datetime(2026, 5, 9, 21, 0, 0),
                "completed_at": datetime(2026, 5, 9, 21, 5, 0),
            }
        ]

        resp = client.get("/costs")
        assert resp.status_code == 200
        assert b"$1.2345" in resp.data
        assert b"12,345" in resp.data
        assert b"/costs/42" in resp.data

    def test_costs_session_renders_stages(self, client):
        mock_queries.get_session_stage_costs.return_value = [
            {
                "id": 1,
                "stage_name": "ideation",
                "status": "completed",
                "started_at": datetime(2026, 5, 9, 21, 1, 0),
                "completed_at": datetime(2026, 5, 9, 21, 2, 0),
                "model": "claude-opus-4-7",
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 200,
                "cost_usd": Decimal("0.5000"),
            },
            {
                "id": 2,
                "stage_name": "trading",
                "status": "completed",
                "started_at": datetime(2026, 5, 9, 21, 3, 0),
                "completed_at": datetime(2026, 5, 9, 21, 4, 0),
                "model": "claude-haiku-4-5",
                "input_tokens": 200,
                "output_tokens": 100,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 50,
                "cost_usd": Decimal("0.1000"),
            },
        ]

        resp = client.get("/costs/42")
        assert resp.status_code == 200
        assert b"ideation" in resp.data
        assert b"trading" in resp.data
        assert b"$0.6000" in resp.data  # sum of stage costs
        mock_queries.get_session_stage_costs.assert_called_once_with(42)

    def test_costs_session_404_when_no_stages(self, client):
        mock_queries.get_session_stage_costs.return_value = []

        resp = client.get("/costs/999")
        assert resp.status_code == 404

    def test_session_date_links_to_session_detail(self, client):
        mock_queries.get_recent_session_costs.return_value = [{
            "session_id": 42,
            "session_date": date(2026, 5, 11),
            "session_type": "daily",
            "status": "completed",
            "total_cost_usd": Decimal("0.50"),
            "total_input_tokens": 1000,
            "total_output_tokens": 200,
            "total_cache_creation_tokens": 0,
            "total_cache_read_tokens": 0,
            "started_at": datetime(2026, 5, 11, 10, 0),
            "completed_at": datetime(2026, 5, 11, 10, 5),
        }]
        resp = client.get("/costs")
        assert b'href="/session/42"' in resp.data


# ---------------------------------------------------------------------------
# Events page (agent_events viewer)
# ---------------------------------------------------------------------------


class TestEventsPage:
    """Tests for GET /events."""

    def test_events_renders_empty(self, client):
        mock_queries.get_recent_agent_events.return_value = []
        mock_queries.get_agent_event_types.return_value = []

        resp = client.get("/events")
        assert resp.status_code == 200
        assert b"No events match" in resp.data
        mock_queries.get_recent_agent_events.assert_called_once_with(
            limit=200, event_type=None, session_id=None
        )

    def test_events_renders_with_data(self, client):
        mock_queries.get_recent_agent_events.return_value = [
            {
                "id": 1,
                "session_id": 42,
                "stage_name": "trading",
                "event_type": "risk_block",
                "payload": {"ticker": "AAPL", "reason": "max_position"},
                "occurred_at": datetime(2026, 5, 9, 21, 5, 0),
            }
        ]
        mock_queries.get_agent_event_types.return_value = [
            {"event_type": "tool_invocation", "n": 50},
            {"event_type": "risk_block", "n": 3},
        ]

        resp = client.get("/events")
        assert resp.status_code == 200
        assert b"risk_block" in resp.data
        assert b"tool_invocation" in resp.data
        assert b"AAPL" in resp.data

    def test_events_filter_by_type(self, client):
        mock_queries.get_recent_agent_events.return_value = []
        mock_queries.get_agent_event_types.return_value = []

        resp = client.get("/events?type=tool_invocation")
        assert resp.status_code == 200
        mock_queries.get_recent_agent_events.assert_called_once_with(
            limit=200, event_type="tool_invocation", session_id=None
        )

    def test_events_filter_by_session(self, client):
        mock_queries.get_recent_agent_events.return_value = []
        mock_queries.get_agent_event_types.return_value = []

        resp = client.get("/events?session=42")
        assert resp.status_code == 200
        mock_queries.get_recent_agent_events.assert_called_once_with(
            limit=200, event_type=None, session_id=42
        )

    def test_event_session_links_to_session_detail(self, client):
        mock_queries.get_recent_agent_events.return_value = [make_agent_event_row(session_id=42)]
        mock_queries.get_agent_event_types.return_value = []
        resp = client.get("/events")
        assert b'href="/session/42"' in resp.data


# ---------------------------------------------------------------------------
# Decision detail
# ---------------------------------------------------------------------------


class TestDecisionDetail:
    def test_renders_200_with_decision(self, client):
        mock_queries.get_decision.return_value = make_decision_row(id=5, ticker="AAPL")
        mock_queries.get_decision_signals_full.return_value = []
        mock_queries.get_decision_tweets.return_value = []
        mock_queries.get_playbook_action.return_value = None
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/decision/5")
        assert resp.status_code == 200
        assert b"AAPL" in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_decision.return_value = None
        resp = client.get("/decision/999")
        assert resp.status_code == 404

    def test_renders_signal_refs(self, client):
        mock_queries.get_decision.return_value = make_decision_row(id=5)
        mock_queries.get_decision_signals_full.return_value = [
            {
                "signal_type": "news_signal", "signal_id": 100,
                "news_headline": "Big earnings beat",
                "news_summary": "Apple report", "news_category": "earnings",
                "news_sentiment": "bullish", "news_confidence": "high",
                "news_published_at": datetime(2026, 5, 11, 10, 0),
                "news_ticker": "AAPL",
                "macro_headline": None, "macro_category": None,
                "macro_affected_sectors": None, "macro_sentiment": None,
                "macro_published_at": None,
                "thesis_text": None, "thesis_ticker": None,
                "thesis_direction": None, "thesis_status": None,
            },
        ]
        mock_queries.get_decision_tweets.return_value = []
        mock_queries.get_playbook_action.return_value = None
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/decision/5")
        assert resp.status_code == 200
        assert b"Big earnings beat" in resp.data
        # Anchor target for inbound links from /decisions
        assert b'id="signal-news-100"' in resp.data

    def test_links_to_session_when_session_exists(self, client):
        mock_queries.get_decision.return_value = make_decision_row(id=5)
        mock_queries.get_decision_signals_full.return_value = []
        mock_queries.get_decision_tweets.return_value = []
        mock_queries.get_playbook_action.return_value = None
        mock_queries.lookup_session_id_by_date.return_value = 42
        resp = client.get("/decision/5")
        assert b'href="/session/42"' in resp.data


# ---------------------------------------------------------------------------
# Thesis detail
# ---------------------------------------------------------------------------


class TestThesisDetail:
    def test_renders_200(self, client):
        mock_queries.get_thesis.return_value = make_thesis_row(id=3, ticker="NVDA")
        mock_queries.get_thesis_decisions.return_value = []
        mock_queries.get_thesis_playbook_actions.return_value = []
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/thesis/3")
        assert resp.status_code == 200
        assert b"NVDA" in resp.data
        # Ticker link present
        assert b'href="/ticker/NVDA"' in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_thesis.return_value = None
        resp = client.get("/thesis/999")
        assert resp.status_code == 404

    def test_shows_linked_decisions(self, client):
        mock_queries.get_thesis.return_value = make_thesis_row(id=3, ticker="NVDA")
        mock_queries.get_thesis_decisions.return_value = [
            make_decision_row(id=10, ticker="NVDA", action="buy"),
        ]
        mock_queries.get_thesis_playbook_actions.return_value = []
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/thesis/3")
        assert b'href="/decision/10"' in resp.data


# ---------------------------------------------------------------------------
# Ticker overview
# ---------------------------------------------------------------------------


class TestTickerOverview:
    def test_renders_with_position(self, client):
        mock_queries.get_ticker_position.return_value = make_position_row(ticker="AAPL")
        mock_queries.get_ticker_theses.return_value = []
        mock_queries.get_ticker_decisions.return_value = []
        mock_queries.get_ticker_signals.return_value = []
        mock_queries.get_ticker_open_orders.return_value = []
        mock_queries.get_ticker_attribution.return_value = []
        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        assert b"AAPL" in resp.data

    def test_renders_when_never_traded(self, client):
        """Ticker page renders even for symbols never traded — shows empty state, not 404."""
        mock_queries.get_ticker_position.return_value = None
        mock_queries.get_ticker_theses.return_value = []
        mock_queries.get_ticker_decisions.return_value = []
        mock_queries.get_ticker_signals.return_value = []
        mock_queries.get_ticker_open_orders.return_value = []
        mock_queries.get_ticker_attribution.return_value = []
        resp = client.get("/ticker/ZZZZ")
        assert resp.status_code == 200
        assert b"ZZZZ" in resp.data

    def test_links_to_decisions_and_theses(self, client):
        mock_queries.get_ticker_position.return_value = None
        mock_queries.get_ticker_theses.return_value = [make_thesis_row(id=7, ticker="AAPL")]
        mock_queries.get_ticker_decisions.return_value = [make_decision_row(id=11, ticker="AAPL")]
        mock_queries.get_ticker_signals.return_value = []
        mock_queries.get_ticker_open_orders.return_value = []
        mock_queries.get_ticker_attribution.return_value = []
        resp = client.get("/ticker/AAPL")
        assert b'href="/thesis/7"' in resp.data
        assert b'href="/decision/11"' in resp.data


# ---------------------------------------------------------------------------
# Session detail
# ---------------------------------------------------------------------------


class TestSessionDetail:
    def test_renders_200(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = []
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        resp = client.get("/session/42")
        assert resp.status_code == 200
        assert b"Session #42" in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_session.return_value = None
        resp = client.get("/session/999")
        assert resp.status_code == 404

    def test_renders_decisions_with_links(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = [make_decision_row(id=11, ticker="AAPL")]
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        resp = client.get("/session/42")
        assert b'href="/decision/11"' in resp.data
        assert b'href="/ticker/AAPL"' in resp.data

    def test_renders_llm_calls_section_when_present(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = []
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        mock_queries.get_session_llm_calls.return_value = [
            {
                "id": 137, "stage_name": "trading", "purpose": "executor",
                "sequence": 0, "model": "claude-haiku-4-5-20251001",
                "input_tokens": 120, "output_tokens": 45,
                "cache_read_tokens": 80, "cache_creation_tokens": 10,
                "stop_reason": "end_turn", "duration_ms": 987,
                "created_at": datetime(2026, 5, 13, 16, 30),
            },
        ]
        resp = client.get("/session/42")
        assert resp.status_code == 200
        assert b"LLM Calls" in resp.data
        assert b"executor" in resp.data
        assert b'href="/llm-call/137"' in resp.data

    def test_renders_empty_state_when_no_llm_calls(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = []
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        mock_queries.get_session_llm_calls.return_value = []
        resp = client.get("/session/42")
        assert resp.status_code == 200
        assert b"No LLM calls captured for this session." in resp.data


# ---------------------------------------------------------------------------
# LLM call detail
# ---------------------------------------------------------------------------


class TestLlmCallDetail:
    def _make_row(self, **overrides):
        row = {
            "id": 137,
            "session_id": 42,
            "stage_name": "trading",
            "purpose": "executor",
            "sequence": 0,
            "model": "claude-haiku-4-5-20251001",
            "system_prompt": "you are a trading executor",
            "messages": [
                {"role": "user", "content": "executor input json"},
            ],
            "tool_definitions": None,
            "response_content": [
                {"type": "text", "text": "decision json here"},
            ],
            "input_tokens": 120,
            "output_tokens": 45,
            "cache_read_tokens": 80,
            "cache_creation_tokens": 10,
            "stop_reason": "end_turn",
            "duration_ms": 987,
            "created_at": datetime(2026, 5, 13, 16, 30),
        }
        row.update(overrides)
        return row

    def test_returns_200_with_header_fields(self, client):
        mock_queries.get_llm_call.return_value = self._make_row()
        resp = client.get("/llm-call/137")
        assert resp.status_code == 200
        assert b"executor" in resp.data
        assert b"claude-haiku-4-5-20251001" in resp.data
        assert b"end_turn" in resp.data
        assert b"120" in resp.data
        assert b"45" in resp.data
        assert b'href="/session/42"' in resp.data

    def test_renders_system_prompt(self, client):
        mock_queries.get_llm_call.return_value = self._make_row()
        resp = client.get("/llm-call/137")
        assert b"you are a trading executor" in resp.data

    def test_skips_system_prompt_section_when_null(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(system_prompt=None)
        resp = client.get("/llm-call/137")
        assert b"System prompt" not in resp.data

    def test_renders_tool_definitions_when_present(self, client):
        tools = [{"name": "get_positions", "input_schema": {"type": "object"}}]
        mock_queries.get_llm_call.return_value = self._make_row(tool_definitions=tools)
        resp = client.get("/llm-call/137")
        assert b"Tool definitions" in resp.data
        assert b"get_positions" in resp.data

    def test_skips_tool_definitions_when_null(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(tool_definitions=None)
        resp = client.get("/llm-call/137")
        assert b"Tool definitions" not in resp.data

    def test_renders_text_message_in_transcript(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(
            messages=[{"role": "user", "content": "hello executor"}],
        )
        resp = client.get("/llm-call/137")
        assert b"hello executor" in resp.data

    def test_renders_tool_use_and_tool_result_and_unknown_blocks(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "get_positions", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "positions json"}],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "future_block_type", "foo": "bar"},
                    ],
                },
            ],
        )
        resp = client.get("/llm-call/137")
        assert b"get_positions" in resp.data
        assert b"t1" in resp.data
        assert b"positions json" in resp.data
        assert b"Unknown block type" in resp.data
        assert b"future_block_type" in resp.data

    def test_renders_response_content(self, client):
        mock_queries.get_llm_call.return_value = self._make_row(
            response_content=[{"type": "text", "text": "executor decision json"}],
        )
        resp = client.get("/llm-call/137")
        assert b"executor decision json" in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_llm_call.return_value = None
        resp = client.get("/llm-call/99999")
        assert resp.status_code == 404
