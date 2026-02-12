"""Tests for v2.context — prose context builders and build_executor_input."""

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest


class TestBuildExecutorInput:
    def test_returns_executor_input(self, mock_db, mock_cursor):
        mock_cursor.fetchone.side_effect = [
            {"id": 1, "market_outlook": "Bullish", "risk_notes": "Fed", "priority_actions": [], "watch_list": []},
        ]
        mock_cursor.fetchall.side_effect = [
            # playbook_actions
            [{"id": 1, "ticker": "AAPL", "action": "buy", "thesis_id": 1,
              "reasoning": "Entry", "confidence": "high", "max_quantity": Decimal("5"), "priority": 1}],
            # positions
            [{"ticker": "MSFT", "shares": Decimal("10"), "avg_cost": Decimal("300")}],
            # recent decisions
            [],
            # attribution
            [],
        ]

        from v2.context import build_executor_input
        from v2.agent import ExecutorInput
        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        assert isinstance(result, ExecutorInput)
        assert len(result.playbook_actions) == 1
        assert result.playbook_actions[0].ticker == "AAPL"
        assert result.market_outlook == "Bullish"

    def test_no_playbook_returns_empty_actions(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.side_effect = [
            # positions
            [],
            # recent decisions
            [],
            # attribution
            [],
        ]

        from v2.context import build_executor_input
        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        assert result.playbook_actions == []
        assert "No playbook" in result.market_outlook

    def test_attribution_summary_built(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.side_effect = [
            # positions
            [],
            # recent decisions
            [],
            # attribution rows
            [
                {"category": "news_signal:earnings", "win_rate_7d": Decimal("0.65"), "sample_size": 10},
                {"category": "macro_signal:fed", "win_rate_7d": Decimal("0.40"), "sample_size": 5},
            ],
        ]

        from v2.context import build_executor_input
        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        assert "news_signal:earnings" in result.attribution_summary
        assert result.attribution_summary["news_signal:earnings"]["win_rate_7d"] == 0.65
        assert result.attribution_summary["news_signal:earnings"]["sample_size"] == 10

    def test_recent_outcomes_limited_to_10(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        decisions = [
            {"date": date(2026, 1, i + 1), "ticker": "AAPL", "action": "buy",
             "outcome_7d": Decimal("1.5")}
            for i in range(15)
        ]
        mock_cursor.fetchall.side_effect = [
            # positions
            [],
            # recent decisions
            decisions,
            # attribution
            [],
        ]

        from v2.context import build_executor_input
        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        assert len(result.recent_outcomes) == 10

    def test_recent_outcomes_filters_none_outcome(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        decisions = [
            {"date": date(2026, 1, 1), "ticker": "AAPL", "action": "buy", "outcome_7d": Decimal("1.5")},
            {"date": date(2026, 1, 2), "ticker": "MSFT", "action": "buy", "outcome_7d": None},
            {"date": date(2026, 1, 3), "ticker": "GOOG", "action": "sell", "outcome_7d": Decimal("-0.5")},
        ]
        mock_cursor.fetchall.side_effect = [
            [],  # positions
            decisions,
            [],  # attribution
        ]

        from v2.context import build_executor_input
        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        assert len(result.recent_outcomes) == 2
        tickers = [o["ticker"] for o in result.recent_outcomes]
        assert "MSFT" not in tickers

    def test_playbook_action_defaults(self, mock_db, mock_cursor):
        """Test that missing optional fields get default values."""
        mock_cursor.fetchone.side_effect = [
            {"id": 1, "market_outlook": "Neutral", "risk_notes": "", "priority_actions": [], "watch_list": []},
        ]
        mock_cursor.fetchall.side_effect = [
            # playbook_actions - minimal fields
            [{"id": 5, "ticker": "TSLA", "action": "sell"}],
            # positions
            [],
            # recent decisions
            [],
            # attribution
            [],
        ]

        from v2.context import build_executor_input
        result = build_executor_input(
            account_info={"cash": Decimal("50000"), "buying_power": Decimal("50000"), "portfolio_value": Decimal("100000")},
        )

        action = result.playbook_actions[0]
        assert action.ticker == "TSLA"
        assert action.reasoning == ""
        assert action.confidence == "medium"
        assert action.max_quantity is None
        assert action.priority == 99


class TestGetPortfolioContext:
    def test_no_positions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_portfolio_context
        result = get_portfolio_context({"cash": Decimal("50000"), "buying_power": Decimal("50000")})
        assert "No open positions" in result
        assert "$50,000.00" in result

    def test_with_positions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [{"ticker": "AAPL", "shares": Decimal("10"), "avg_cost": Decimal("150.00")}],
            [],  # open orders
        ]
        from v2.context import get_portfolio_context
        result = get_portfolio_context({"cash": Decimal("50000"), "buying_power": Decimal("50000")})
        assert "AAPL" in result
        assert "10" in result

    def test_with_open_orders(self, mock_db, mock_cursor):
        mock_cursor.fetchall.side_effect = [
            [{"ticker": "AAPL", "shares": Decimal("10"), "avg_cost": Decimal("150.00")}],
            [{"ticker": "MSFT", "side": "buy", "qty": 5, "filled_qty": 2,
              "order_type": "limit", "status": "partially_filled", "limit_price": Decimal("300.00")}],
        ]
        from v2.context import get_portfolio_context
        result = get_portfolio_context({"cash": Decimal("50000"), "buying_power": Decimal("50000")})
        assert "MSFT" in result
        assert "Open Orders" in result
        assert "BUY" in result
        assert "partially_filled" in result

    def test_default_cash_and_buying_power(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_portfolio_context
        result = get_portfolio_context({})
        assert "Cash: $0.00" in result
        assert "Buying Power: $0.00" in result


class TestGetMacroContext:
    def test_no_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_macro_context
        result = get_macro_context(days=7)
        assert "No significant macro news" in result

    def test_with_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"category": "fed", "headline": "Fed holds rates steady", "sentiment": "neutral", "affected_sectors": ["finance"]},
        ]
        from v2.context import get_macro_context
        result = get_macro_context(days=7)
        assert "Fed" in result

    def test_multiple_signals_same_category(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"category": "fed", "headline": "Fed holds rates", "sentiment": "bullish", "affected_sectors": ["finance"]},
            {"category": "fed", "headline": "Fed minutes dovish", "sentiment": "bearish", "affected_sectors": ["finance"]},
        ]
        from v2.context import get_macro_context
        result = get_macro_context(days=7)
        assert "1 bullish" in result
        assert "1 bearish" in result

    def test_long_headline_truncated(self, mock_db, mock_cursor):
        long_headline = "A" * 100
        mock_cursor.fetchall.return_value = [
            {"category": "trade", "headline": long_headline, "sentiment": "neutral", "affected_sectors": []},
        ]
        from v2.context import get_macro_context
        result = get_macro_context(days=7)
        assert "..." in result

    def test_unknown_category_titlecased(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"category": "crypto", "headline": "Bitcoin surges", "sentiment": "bullish", "affected_sectors": []},
        ]
        from v2.context import get_macro_context
        result = get_macro_context(days=7)
        assert "Crypto" in result


class TestGetTickerSignalsContext:
    def test_no_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_ticker_signals_context
        result = get_ticker_signals_context(days=1)
        assert "No ticker-specific signals" in result

    def test_with_bullish_signal(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"ticker": "AAPL", "sentiment": "bullish", "category": "earnings"},
        ]
        from v2.context import get_ticker_signals_context
        result = get_ticker_signals_context(days=1)
        assert "AAPL" in result
        assert "1 bullish" in result
        assert "earnings" in result

    def test_mixed_sentiments(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"ticker": "AAPL", "sentiment": "bullish", "category": "earnings"},
            {"ticker": "AAPL", "sentiment": "bearish", "category": "analyst"},
            {"ticker": "AAPL", "sentiment": "neutral", "category": "general"},
        ]
        from v2.context import get_ticker_signals_context
        result = get_ticker_signals_context(days=1)
        assert "1 bullish" in result
        assert "1 bearish" in result
        assert "1 neutral" in result


class TestGetSignalTrendContext:
    def test_no_signals(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_signal_trend_context
        result = get_signal_trend_context(days=7)
        assert "No recent signals" in result

    def test_with_trend(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"ticker": "AAPL", "sentiment": "bullish"},
            {"ticker": "AAPL", "sentiment": "bullish"},
            {"ticker": "AAPL", "sentiment": "bearish"},
        ]
        from v2.context import get_signal_trend_context
        result = get_signal_trend_context(days=7)
        assert "AAPL" in result
        assert "2 bullish" in result
        assert "1 bearish" in result

    def test_all_neutral(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"ticker": "MSFT", "sentiment": "neutral"},
        ]
        from v2.context import get_signal_trend_context
        result = get_signal_trend_context(days=7)
        assert "neutral, no significant news" in result


class TestGetDecisionOutcomesContext:
    def test_no_decisions(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_decision_outcomes_context
        result = get_decision_outcomes_context(days=30)
        assert "No recent decisions" in result

    def test_decisions_pending(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"ticker": "AAPL", "action": "buy", "outcome_7d": None, "date": date(2026, 1, 1)},
        ]
        from v2.context import get_decision_outcomes_context
        result = get_decision_outcomes_context(days=30)
        assert "pending outcome" in result

    def test_with_outcomes(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"ticker": "AAPL", "action": "buy", "outcome_7d": Decimal("2.5"), "date": date(2026, 1, 1)},
            {"ticker": "MSFT", "action": "sell", "outcome_7d": Decimal("-1.2"), "date": date(2026, 1, 2)},
        ]
        from v2.context import get_decision_outcomes_context
        result = get_decision_outcomes_context(days=30)
        assert "BUY AAPL" in result
        assert "+2.5%" in result
        assert "SELL MSFT" in result
        assert "-1.2%" in result

    def test_limits_to_5(self, mock_db, mock_cursor):
        decisions = [
            {"ticker": f"T{i}", "action": "buy", "outcome_7d": Decimal("1.0"), "date": date(2026, 1, i + 1)}
            for i in range(10)
        ]
        mock_cursor.fetchall.return_value = decisions
        from v2.context import get_decision_outcomes_context
        result = get_decision_outcomes_context(days=30)
        # Should only have 5 decision lines plus the header
        decision_lines = [l for l in result.split("\n") if l.startswith("- 2026")]
        assert len(decision_lines) == 5


class TestGetThesesContext:
    def test_no_theses(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_theses_context
        result = get_theses_context()
        assert "No active trade theses" in result

    def test_with_thesis(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {
                "ticker": "AAPL", "direction": "long", "confidence": "high",
                "thesis": "Strong earnings growth expected",
                "entry_trigger": "Price > $150", "exit_trigger": "Price > $200",
                "invalidation": "Earnings miss",
                "created_at": datetime(2026, 1, 1),
            },
        ]
        from v2.context import get_theses_context
        result = get_theses_context()
        assert "AAPL" in result
        assert "long" in result
        assert "high confidence" in result
        assert "Strong earnings growth" in result
        assert "Entry trigger" in result
        assert "Exit trigger" in result
        assert "Invalidation" in result

    def test_thesis_without_triggers(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {
                "ticker": "GOOG", "direction": "short", "confidence": "medium",
                "thesis": "Overvalued",
                "entry_trigger": None, "exit_trigger": None,
                "invalidation": None,
                "created_at": datetime(2026, 2, 1),
            },
        ]
        from v2.context import get_theses_context
        result = get_theses_context()
        assert "GOOG" in result
        assert "Entry trigger" not in result
        assert "Exit trigger" not in result
        assert "Invalidation" not in result


class TestGetPlaybookContext:
    def test_no_playbook(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.context import get_playbook_context
        result = get_playbook_context(date(2026, 1, 1))
        assert "No playbook available" in result
        assert "conservative mode" in result

    def test_with_playbook(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {
            "market_outlook": "Bullish momentum",
            "priority_actions": [
                {"ticker": "AAPL", "action": "buy", "reasoning": "Breakout", "confidence": "high",
                 "max_quantity": 10, "thesis_id": 1},
            ],
            "watch_list": ["MSFT", "GOOG"],
            "risk_notes": "Watch Fed meeting",
        }
        from v2.context import get_playbook_context
        result = get_playbook_context(date(2026, 1, 1))
        assert "Bullish momentum" in result
        assert "AAPL" in result
        assert "BUY" in result
        assert "Breakout" in result
        assert "Watch List" in result
        assert "MSFT" in result
        assert "Risk Notes" in result
        assert "Fed meeting" in result

    def test_playbook_without_actions(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {
            "market_outlook": "Sideways",
            "priority_actions": None,
            "watch_list": None,
            "risk_notes": None,
        }
        from v2.context import get_playbook_context
        result = get_playbook_context(date(2026, 1, 1))
        assert "Sideways" in result
        assert "Priority Actions" not in result
        assert "Watch List" not in result
        assert "Risk Notes" not in result


class TestGetAttributionContext:
    def test_delegates_to_get_attribution_summary(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.context import get_attribution_context
        result = get_attribution_context()
        assert "Signal Attribution" in result
        assert "No attribution data" in result

    def test_with_attribution_data(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [
            {"category": "news_signal:earnings", "win_rate_7d": Decimal("0.65"),
             "avg_outcome_7d": Decimal("2.1"), "sample_size": 10},
        ]
        from v2.context import get_attribution_context
        result = get_attribution_context()
        assert "Predictive" in result
        assert "news_signal:earnings" in result


class TestBuildTradingContext:
    def test_builds_complete_context(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None  # no playbook
        mock_cursor.fetchall.return_value = []
        from v2.context import build_trading_context
        result = build_trading_context({"cash": Decimal("50000"), "buying_power": Decimal("50000")})
        assert isinstance(result, str)
        assert "Portfolio" in result or "playbook" in result.lower()

    def test_includes_all_sections(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None  # no playbook
        mock_cursor.fetchall.return_value = []
        from v2.context import build_trading_context
        result = build_trading_context({"cash": Decimal("50000"), "buying_power": Decimal("50000")})
        # Check that major sections appear in the output
        assert "Playbook" in result
        assert "Portfolio" in result
        assert "Macro Context" in result
        assert "Signal" in result
        assert "Decision" in result
        assert "Attribution" in result
