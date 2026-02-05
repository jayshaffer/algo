"""Tests for trading/context.py - context builder for trading agent."""

from datetime import datetime, timedelta, date
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from trading.context import (
    get_portfolio_context,
    get_macro_context,
    get_ticker_signals_context,
    get_signal_trend_context,
    get_decision_outcomes_context,
    get_strategy_context,
    get_theses_context,
    build_trading_context,
)
from tests.conftest import (
    make_position_row,
    make_thesis_row,
    make_decision_row,
    make_news_signal_row,
    make_macro_signal_row,
)


# ---------------------------------------------------------------------------
# get_portfolio_context
# ---------------------------------------------------------------------------


class TestGetPortfolioContext:
    @patch("trading.context.get_open_orders", return_value=[])
    @patch("trading.context.get_positions", return_value=[])
    def test_no_positions(self, mock_pos, mock_orders):
        account = {"cash": Decimal("100000"), "buying_power": Decimal("200000")}
        result = get_portfolio_context(account)

        assert "Current Portfolio:" in result
        assert "No open positions" in result
        assert "$100,000.00" in result
        assert "$200,000.00" in result

    @patch("trading.context.get_open_orders", return_value=[])
    @patch("trading.context.get_positions")
    def test_with_positions(self, mock_pos, mock_orders):
        mock_pos.return_value = [
            make_position_row(ticker="AAPL", shares=Decimal("10"), avg_cost=Decimal("150.00")),
            make_position_row(ticker="MSFT", shares=Decimal("5"), avg_cost=Decimal("400.00")),
        ]
        account = {"cash": Decimal("50000"), "buying_power": Decimal("100000")}
        result = get_portfolio_context(account)

        assert "AAPL: 10 shares @ $150.00 avg" in result
        assert "MSFT: 5 shares @ $400.00 avg" in result
        assert "No open positions" not in result

    @patch("trading.context.get_open_orders")
    @patch("trading.context.get_positions", return_value=[])
    def test_with_open_orders(self, mock_pos, mock_orders):
        mock_orders.return_value = [
            {
                "ticker": "GOOG",
                "side": "buy",
                "qty": Decimal("3"),
                "filled_qty": Decimal("0"),
                "order_type": "limit",
                "status": "new",
                "limit_price": Decimal("140.00"),
            },
        ]
        account = {"cash": Decimal("50000"), "buying_power": Decimal("100000")}
        result = get_portfolio_context(account)

        assert "Open Orders:" in result
        assert "BUY" in result
        assert "GOOG" in result
        assert "$140.00" in result

    @patch("trading.context.get_open_orders")
    @patch("trading.context.get_positions", return_value=[])
    def test_partially_filled_order(self, mock_pos, mock_orders):
        mock_orders.return_value = [
            {
                "ticker": "TSLA",
                "side": "sell",
                "qty": Decimal("10"),
                "filled_qty": Decimal("4"),
                "order_type": "market",
                "status": "partially_filled",
                "limit_price": None,
            },
        ]
        account = {"cash": Decimal("50000"), "buying_power": Decimal("100000")}
        result = get_portfolio_context(account)

        assert "4/10 filled" in result


# ---------------------------------------------------------------------------
# get_macro_context
# ---------------------------------------------------------------------------


class TestGetMacroContext:
    @patch("trading.context.get_macro_signals", return_value=[])
    def test_no_macro_signals(self, mock_signals):
        result = get_macro_context()
        assert "Macro Context:" in result
        assert "No significant macro news" in result

    @patch("trading.context.get_macro_signals")
    def test_with_signals(self, mock_signals):
        mock_signals.return_value = [
            make_macro_signal_row(headline="Fed holds rates steady", category="fed", sentiment="bullish"),
        ]
        result = get_macro_context()

        assert "Macro Context:" in result
        assert "Fed" in result
        assert "Fed holds rates steady" in result
        assert "bullish" in result

    @patch("trading.context.get_macro_signals")
    def test_multiple_signals_same_category_show_distribution(self, mock_signals):
        mock_signals.return_value = [
            make_macro_signal_row(category="trade", sentiment="bullish"),
            make_macro_signal_row(category="trade", sentiment="bearish"),
        ]
        result = get_macro_context()

        assert "1 bullish" in result
        assert "1 bearish" in result

    @patch("trading.context.get_macro_signals")
    def test_long_headline_truncated(self, mock_signals):
        long_headline = "A" * 100
        mock_signals.return_value = [
            make_macro_signal_row(headline=long_headline, category="fed"),
        ]
        result = get_macro_context()
        # Headlines > 60 chars get truncated with "..."
        assert "..." in result

    @patch("trading.context.get_macro_signals")
    def test_unknown_category_uses_title_case(self, mock_signals):
        mock_signals.return_value = [
            make_macro_signal_row(category="custom_cat", sentiment="neutral"),
        ]
        result = get_macro_context()
        assert "Custom_Cat" in result or "custom_cat" in result.lower()


# ---------------------------------------------------------------------------
# get_ticker_signals_context
# ---------------------------------------------------------------------------


class TestGetTickerSignalsContext:
    @patch("trading.context.get_news_signals", return_value=[])
    def test_no_signals(self, mock_signals):
        result = get_ticker_signals_context()
        assert "Today's Signals:" in result
        assert "No ticker-specific signals" in result

    @patch("trading.context.get_news_signals")
    def test_with_signals(self, mock_signals):
        mock_signals.return_value = [
            make_news_signal_row(ticker="AAPL", sentiment="bullish", category="earnings"),
        ]
        result = get_ticker_signals_context()

        assert "Today's Signals:" in result
        assert "AAPL" in result
        assert "1 bullish" in result
        assert "earnings" in result

    @patch("trading.context.get_news_signals")
    def test_mixed_sentiments_for_ticker(self, mock_signals):
        mock_signals.return_value = [
            make_news_signal_row(ticker="AAPL", sentiment="bullish", category="earnings"),
            make_news_signal_row(ticker="AAPL", sentiment="bearish", category="analyst"),
            make_news_signal_row(ticker="AAPL", sentiment="neutral", category="sector"),
        ]
        result = get_ticker_signals_context()

        assert "1 bullish" in result
        assert "1 bearish" in result
        assert "1 neutral" in result

    @patch("trading.context.get_news_signals")
    def test_multiple_tickers(self, mock_signals):
        mock_signals.return_value = [
            make_news_signal_row(ticker="AAPL", sentiment="bullish", category="earnings"),
            make_news_signal_row(ticker="MSFT", sentiment="bearish", category="analyst"),
        ]
        result = get_ticker_signals_context()

        assert "AAPL" in result
        assert "MSFT" in result


# ---------------------------------------------------------------------------
# get_signal_trend_context
# ---------------------------------------------------------------------------


class TestGetSignalTrendContext:
    @patch("trading.context.get_news_signals", return_value=[])
    def test_no_signals(self, mock_signals):
        result = get_signal_trend_context()
        assert "7-Day Signal Trend:" in result
        assert "No recent signals" in result

    @patch("trading.context.get_news_signals")
    def test_with_signals(self, mock_signals):
        mock_signals.return_value = [
            make_news_signal_row(ticker="AAPL", sentiment="bullish"),
            make_news_signal_row(ticker="AAPL", sentiment="bullish"),
            make_news_signal_row(ticker="AAPL", sentiment="bearish"),
        ]
        result = get_signal_trend_context()

        assert "7-Day Signal Trend:" in result
        assert "AAPL" in result
        assert "2 bullish" in result
        assert "1 bearish" in result

    @patch("trading.context.get_news_signals")
    def test_neutral_only(self, mock_signals):
        mock_signals.return_value = [
            make_news_signal_row(ticker="GOOG", sentiment="neutral"),
        ]
        result = get_signal_trend_context()

        assert "GOOG" in result
        assert "neutral" in result


# ---------------------------------------------------------------------------
# get_decision_outcomes_context
# ---------------------------------------------------------------------------


class TestGetDecisionOutcomesContext:
    @patch("trading.context.get_recent_decisions", return_value=[])
    def test_no_decisions(self, mock_decisions):
        result = get_decision_outcomes_context()
        assert "Recent Decision Outcomes:" in result
        assert "No recent decisions" in result

    @patch("trading.context.get_recent_decisions")
    def test_decisions_without_outcomes(self, mock_decisions):
        mock_decisions.return_value = [
            make_decision_row(outcome_7d=None, outcome_30d=None),
        ]
        result = get_decision_outcomes_context()
        assert "pending outcome measurement" in result

    @patch("trading.context.get_recent_decisions")
    def test_decisions_with_positive_outcomes(self, mock_decisions):
        mock_decisions.return_value = [
            make_decision_row(
                ticker="AAPL",
                action="buy",
                outcome_7d=Decimal("2.5"),
                date=date(2025, 1, 5),
            ),
        ]
        result = get_decision_outcomes_context()

        assert "BUY AAPL" in result
        assert "+2.5%" in result

    @patch("trading.context.get_recent_decisions")
    def test_decisions_with_negative_outcomes(self, mock_decisions):
        mock_decisions.return_value = [
            make_decision_row(
                ticker="TSLA",
                action="sell",
                outcome_7d=Decimal("-3.2"),
                date=date(2025, 1, 5),
            ),
        ]
        result = get_decision_outcomes_context()

        assert "SELL TSLA" in result
        assert "-3.2%" in result

    @patch("trading.context.get_recent_decisions")
    def test_limits_to_5_outcomes(self, mock_decisions):
        mock_decisions.return_value = [
            make_decision_row(
                ticker=f"T{i}",
                outcome_7d=Decimal("1.0"),
                date=date(2025, 1, i + 1),
            )
            for i in range(8)
        ]
        result = get_decision_outcomes_context()

        # Should only show 5 lines with outcomes
        outcome_lines = [line for line in result.split("\n") if "(7d)" in line]
        assert len(outcome_lines) == 5


# ---------------------------------------------------------------------------
# get_strategy_context
# ---------------------------------------------------------------------------


class TestGetStrategyContext:
    @patch("trading.db.get_cursor")
    def test_no_strategy(self, mock_get_cursor):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = get_strategy_context()
        assert "No strategy defined" in result
        assert "conservative" in result

    @patch("trading.db.get_cursor")
    def test_with_strategy(self, mock_get_cursor):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "description": "Aggressive growth strategy",
            "risk_tolerance": "high",
            "focus_sectors": ["tech", "healthcare"],
            "watchlist": ["AAPL", "MSFT", "AMZN"],
        }
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = get_strategy_context()

        assert "Current Strategy:" in result
        assert "Aggressive growth strategy" in result
        assert "high" in result
        assert "tech" in result
        assert "AAPL" in result

    @patch("trading.db.get_cursor")
    def test_strategy_with_partial_fields(self, mock_get_cursor):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "description": "Balanced approach",
            "risk_tolerance": None,
            "focus_sectors": None,
            "watchlist": None,
        }
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = get_strategy_context()

        assert "Balanced approach" in result
        assert "Focus sectors" not in result
        assert "Watchlist" not in result


# ---------------------------------------------------------------------------
# get_theses_context
# ---------------------------------------------------------------------------


class TestGetThesesContext:
    @patch("trading.context.get_active_theses", return_value=[])
    def test_no_theses(self, mock_theses):
        result = get_theses_context()
        assert "Active Theses:" in result
        assert "No active trade theses" in result

    @patch("trading.context.get_active_theses")
    def test_with_theses(self, mock_theses):
        mock_theses.return_value = [
            make_thesis_row(
                ticker="AAPL",
                direction="long",
                confidence="high",
                thesis="Strong fundamentals",
                entry_trigger="Price drops to $140",
                exit_trigger="Price hits $180",
                invalidation="Revenue declines 2 quarters",
                created_at=datetime.now() - timedelta(days=5),
            ),
        ]
        result = get_theses_context()

        assert "Active Theses:" in result
        assert "AAPL" in result
        assert "long" in result
        assert "high confidence" in result
        assert "5d old" in result
        assert "Strong fundamentals" in result
        assert "Price drops to $140" in result
        assert "Price hits $180" in result
        assert "Revenue declines 2 quarters" in result

    @patch("trading.context.get_active_theses")
    def test_thesis_without_triggers(self, mock_theses):
        mock_theses.return_value = [
            make_thesis_row(
                ticker="GOOG",
                entry_trigger=None,
                exit_trigger=None,
                invalidation=None,
            ),
        ]
        result = get_theses_context()

        assert "GOOG" in result
        assert "Entry trigger" not in result
        assert "Exit trigger" not in result
        assert "Invalidation" not in result

    @patch("trading.context.get_active_theses")
    def test_multiple_theses(self, mock_theses):
        mock_theses.return_value = [
            make_thesis_row(ticker="AAPL"),
            make_thesis_row(ticker="MSFT", direction="short"),
        ]
        result = get_theses_context()

        assert "AAPL" in result
        assert "MSFT" in result


# ---------------------------------------------------------------------------
# build_trading_context
# ---------------------------------------------------------------------------


class TestBuildTradingContext:
    @patch("trading.context.get_strategy_context", return_value="Current Strategy:\nDefault")
    @patch("trading.context.get_decision_outcomes_context", return_value="Recent Decision Outcomes:\n- None")
    @patch("trading.context.get_signal_trend_context", return_value="7-Day Signal Trend:\n- None")
    @patch("trading.context.get_ticker_signals_context", return_value="Today's Signals:\n- None")
    @patch("trading.context.get_theses_context", return_value="Active Theses:\n- None")
    @patch("trading.context.get_macro_context", return_value="Macro Context:\n- None")
    @patch("trading.context.get_portfolio_context", return_value="Current Portfolio:\n- None")
    def test_combines_all_sections(
        self, mock_port, mock_macro, mock_theses, mock_ticker,
        mock_trend, mock_outcomes, mock_strategy
    ):
        account = {"cash": Decimal("100000"), "buying_power": Decimal("200000")}
        result = build_trading_context(account)

        assert "Current Portfolio:" in result
        assert "Macro Context:" in result
        assert "Active Theses:" in result
        assert "Today's Signals:" in result
        assert "7-Day Signal Trend:" in result
        assert "Recent Decision Outcomes:" in result
        assert "Current Strategy:" in result

    @patch("trading.context.get_strategy_context", return_value="Strategy: test")
    @patch("trading.context.get_decision_outcomes_context", return_value="Outcomes: test")
    @patch("trading.context.get_signal_trend_context", return_value="Trend: test")
    @patch("trading.context.get_ticker_signals_context", return_value="Signals: test")
    @patch("trading.context.get_theses_context", return_value="Theses: test")
    @patch("trading.context.get_macro_context", return_value="Macro: test")
    @patch("trading.context.get_portfolio_context", return_value="Portfolio: test")
    def test_passes_account_info_to_portfolio(
        self, mock_port, mock_macro, mock_theses, mock_ticker,
        mock_trend, mock_outcomes, mock_strategy
    ):
        account = {"cash": Decimal("50000"), "buying_power": Decimal("100000")}
        build_trading_context(account)
        mock_port.assert_called_once_with(account)

    @patch("trading.context.get_strategy_context", return_value="Strategy")
    @patch("trading.context.get_decision_outcomes_context", return_value="Outcomes")
    @patch("trading.context.get_signal_trend_context", return_value="Trend")
    @patch("trading.context.get_ticker_signals_context", return_value="Signals")
    @patch("trading.context.get_theses_context", return_value="Theses")
    @patch("trading.context.get_macro_context", return_value="Macro")
    @patch("trading.context.get_portfolio_context", return_value="Portfolio")
    def test_sections_separated_by_blank_lines(
        self, mock_port, mock_macro, mock_theses, mock_ticker,
        mock_trend, mock_outcomes, mock_strategy
    ):
        account = {"cash": Decimal("50000"), "buying_power": Decimal("100000")}
        result = build_trading_context(account)

        # Sections separated by empty lines create "\n\n" sequences
        assert "\n\n" in result
