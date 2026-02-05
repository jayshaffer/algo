"""Tests for trading/strategy.py - Strategy evolution based on learning."""

from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch, call

import pytest

from trading.strategy import (
    StrategyRecommendation,
    get_current_strategy,
    save_strategy,
    calculate_overall_performance,
    determine_risk_tolerance,
    build_watchlist,
    build_avoid_list,
    identify_focus_sectors,
    generate_strategy_recommendation,
    evolve_strategy,
)
from trading.patterns import TickerPerformance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_get_cursor():
    """Patch trading.strategy.get_cursor to yield a mock cursor."""
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None

    @contextmanager
    def _get_cursor():
        yield cursor

    with patch("trading.strategy.get_cursor", _get_cursor):
        yield cursor


# ---------------------------------------------------------------------------
# get_current_strategy
# ---------------------------------------------------------------------------

class TestGetCurrentStrategy:
    """Tests for get_current_strategy()."""

    def test_returns_none_when_no_strategy(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = None
        result = get_current_strategy()
        assert result is None

    def test_returns_dict_when_strategy_exists(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {
            "date": date(2025, 1, 15),
            "description": "Momentum strategy",
            "watchlist": ["AAPL", "MSFT"],
            "risk_tolerance": "moderate",
            "focus_sectors": ["tech"],
        }
        result = get_current_strategy()

        assert isinstance(result, dict)
        assert result["description"] == "Momentum strategy"
        assert result["risk_tolerance"] == "moderate"
        assert result["watchlist"] == ["AAPL", "MSFT"]

    def test_queries_latest_strategy(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = None
        get_current_strategy()

        sql = mock_get_cursor.execute.call_args[0][0]
        assert "ORDER BY date DESC" in sql
        assert "LIMIT 1" in sql


# ---------------------------------------------------------------------------
# save_strategy
# ---------------------------------------------------------------------------

class TestSaveStrategy:
    """Tests for save_strategy()."""

    def test_inserts_and_returns_id(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {"id": 42}
        result = save_strategy(
            description="Test strategy",
            watchlist=["AAPL"],
            risk_tolerance="moderate",
            focus_sectors=["tech"],
        )
        assert result == 42

    def test_passes_correct_parameters(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {"id": 1}
        save_strategy(
            description="Conservative approach",
            watchlist=["MSFT", "GOOG"],
            risk_tolerance="conservative",
            focus_sectors=["tech", "finance"],
        )

        params = mock_get_cursor.execute.call_args[0][1]
        assert params[0] == date.today()
        assert params[1] == "Conservative approach"
        assert params[2] == ["MSFT", "GOOG"]
        assert params[3] == "conservative"
        assert params[4] == ["tech", "finance"]


# ---------------------------------------------------------------------------
# calculate_overall_performance
# ---------------------------------------------------------------------------

class TestCalculateOverallPerformance:
    """Tests for calculate_overall_performance()."""

    def test_returns_performance_dict(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {
            "total_trades": 20,
            "avg_return": Decimal("2.5"),
            "win_rate": Decimal("0.65"),
            "total_return": Decimal("50.0"),
        }
        result = calculate_overall_performance(30)

        assert result["total_trades"] == 20
        assert result["avg_return"] == 2.5
        assert result["win_rate"] == 0.65
        assert result["total_return"] == 50.0

    def test_handles_none_values(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {
            "total_trades": None,
            "avg_return": None,
            "win_rate": None,
            "total_return": None,
        }
        result = calculate_overall_performance(30)

        assert result["total_trades"] == 0
        assert result["avg_return"] == 0
        assert result["win_rate"] == 0
        assert result["total_return"] == 0

    def test_passes_days_parameter(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {
            "total_trades": 0, "avg_return": None,
            "win_rate": None, "total_return": None,
        }
        calculate_overall_performance(60)
        params = mock_get_cursor.execute.call_args[0][1]
        assert params == (60,)


# ---------------------------------------------------------------------------
# determine_risk_tolerance
# ---------------------------------------------------------------------------

class TestDetermineRiskTolerance:
    """Tests for determine_risk_tolerance()."""

    def test_aggressive_high_win_rate_high_return(self):
        perf = {"win_rate": 0.7, "avg_return": 3.0}
        assert determine_risk_tolerance(perf) == "aggressive"

    def test_aggressive_boundary_win_rate_61_return_3(self):
        perf = {"win_rate": 0.61, "avg_return": 2.1}
        assert determine_risk_tolerance(perf) == "aggressive"

    def test_not_aggressive_when_win_rate_exactly_60(self):
        """win_rate > 0.6 is required, not >=."""
        perf = {"win_rate": 0.6, "avg_return": 3.0}
        assert determine_risk_tolerance(perf) == "moderate"

    def test_not_aggressive_when_return_exactly_2(self):
        """avg_return > 2 is required, not >=."""
        perf = {"win_rate": 0.7, "avg_return": 2.0}
        assert determine_risk_tolerance(perf) == "moderate"

    def test_moderate_good_win_rate_positive_return(self):
        perf = {"win_rate": 0.55, "avg_return": 1.0}
        assert determine_risk_tolerance(perf) == "moderate"

    def test_moderate_boundary_win_rate_51_return_01(self):
        perf = {"win_rate": 0.51, "avg_return": 0.1}
        assert determine_risk_tolerance(perf) == "moderate"

    def test_not_moderate_when_win_rate_exactly_50(self):
        """win_rate > 0.5 is required for moderate."""
        perf = {"win_rate": 0.5, "avg_return": 1.0}
        assert determine_risk_tolerance(perf) == "conservative"

    def test_not_moderate_when_return_exactly_0(self):
        """avg_return > 0 is required for moderate."""
        perf = {"win_rate": 0.55, "avg_return": 0.0}
        assert determine_risk_tolerance(perf) == "conservative"

    def test_conservative_low_win_rate(self):
        perf = {"win_rate": 0.3, "avg_return": -2.0}
        assert determine_risk_tolerance(perf) == "conservative"

    def test_conservative_negative_return(self):
        perf = {"win_rate": 0.55, "avg_return": -0.5}
        assert determine_risk_tolerance(perf) == "conservative"

    def test_conservative_zero_trades(self):
        perf = {"win_rate": 0, "avg_return": 0}
        assert determine_risk_tolerance(perf) == "conservative"

    def test_aggressive_needs_both_conditions(self):
        """High win rate alone is not enough for aggressive."""
        perf = {"win_rate": 0.8, "avg_return": 1.5}
        assert determine_risk_tolerance(perf) == "moderate"


# ---------------------------------------------------------------------------
# build_watchlist
# ---------------------------------------------------------------------------

class TestBuildWatchlist:
    """Tests for build_watchlist()."""

    @patch("trading.strategy.analyze_ticker_performance")
    def test_returns_tickers_with_positive_outcomes(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance("AAPL", 5, 3, 2, 4.5, 8.0, 22.5),
            TickerPerformance("MSFT", 3, 2, 1, 2.0, 3.0, 6.0),
            TickerPerformance("TSLA", 4, 2, 2, -1.5, -2.0, -6.0),
        ]
        result = build_watchlist(60)

        assert "AAPL" in result
        assert "MSFT" in result
        assert "TSLA" not in result

    @patch("trading.strategy.analyze_ticker_performance")
    def test_filters_below_min_decisions(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance("AAPL", 1, 1, 0, 10.0, 15.0, 10.0),  # Only 1 decision
            TickerPerformance("MSFT", 3, 2, 1, 2.0, 3.0, 6.0),
        ]
        result = build_watchlist(60)

        assert "AAPL" not in result
        assert "MSFT" in result

    @patch("trading.strategy.analyze_ticker_performance")
    def test_respects_max_tickers(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance(f"TKR{i}", 5, 3, 2, float(i + 1), float(i + 2), float(i * 5))
            for i in range(20)
        ]
        result = build_watchlist(60, max_tickers=3)
        assert len(result) <= 3

    @patch("trading.strategy.analyze_ticker_performance")
    def test_returns_empty_when_no_good_tickers(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance("BAD1", 5, 3, 2, -2.0, -1.0, -10.0),
            TickerPerformance("BAD2", 3, 2, 1, -3.0, -4.0, -9.0),
        ]
        result = build_watchlist(60)
        assert result == []

    @patch("trading.strategy.analyze_ticker_performance")
    def test_filters_none_outcomes(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance("NONE", 5, 3, 2, None, None, None),
        ]
        result = build_watchlist(60)
        assert result == []

    @patch("trading.strategy.analyze_ticker_performance")
    def test_default_max_tickers_is_10(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance(f"T{i}", 5, 3, 2, float(i + 1), float(i), float(i * 5))
            for i in range(15)
        ]
        result = build_watchlist(60)
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# build_avoid_list
# ---------------------------------------------------------------------------

class TestBuildAvoidList:
    """Tests for build_avoid_list()."""

    @patch("trading.strategy.analyze_ticker_performance")
    def test_returns_tickers_with_avg_below_neg_2(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance("BAD", 5, 3, 2, -3.0, -5.0, -15.0),
            TickerPerformance("OKAY", 3, 2, 1, -1.0, -0.5, -3.0),
        ]
        result = build_avoid_list(60)

        assert "BAD" in result
        assert "OKAY" not in result  # avg > -2

    @patch("trading.strategy.analyze_ticker_performance")
    def test_filters_below_min_decisions(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance("BAD1", 1, 1, 0, -5.0, -8.0, -5.0),  # Only 1 decision
            TickerPerformance("BAD2", 3, 2, 1, -3.0, -5.0, -9.0),
        ]
        result = build_avoid_list(60)

        assert "BAD1" not in result
        assert "BAD2" in result

    @patch("trading.strategy.analyze_ticker_performance")
    def test_respects_max_tickers(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance(f"BAD{i}", 5, 3, 2, -5.0 - i, -8.0, float(-25 - i * 5))
            for i in range(10)
        ]
        result = build_avoid_list(60, max_tickers=2)
        assert len(result) <= 2

    @patch("trading.strategy.analyze_ticker_performance")
    def test_returns_empty_when_no_bad_tickers(self, mock_analyze):
        mock_analyze.return_value = [
            TickerPerformance("GOOD", 5, 3, 2, 4.0, 6.0, 20.0),
        ]
        result = build_avoid_list(60)
        assert result == []

    @patch("trading.strategy.analyze_ticker_performance")
    def test_boundary_exactly_neg_2_not_avoided(self, mock_analyze):
        """avg_outcome_7d < -2 is required, not <=."""
        mock_analyze.return_value = [
            TickerPerformance("EDGE", 3, 2, 1, -2.0, -3.0, -6.0),
        ]
        result = build_avoid_list(60)
        assert "EDGE" not in result


# ---------------------------------------------------------------------------
# identify_focus_sectors
# ---------------------------------------------------------------------------

class TestIdentifyFocusSectors:
    """Tests for identify_focus_sectors()."""

    def test_returns_bullish_sectors_in_priority_order(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"sector": "finance"},
            {"sector": "tech"},
            {"sector": "energy"},
        ]
        result = identify_focus_sectors(60)
        # Priority order: tech, finance, healthcare, energy, consumer, defense
        assert result == ["tech", "finance", "energy"]

    def test_limits_to_three_sectors(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"sector": "tech"},
            {"sector": "finance"},
            {"sector": "healthcare"},
            {"sector": "energy"},
            {"sector": "consumer"},
        ]
        result = identify_focus_sectors(60)
        assert len(result) <= 3

    def test_returns_tech_default_when_no_bullish_sectors(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        result = identify_focus_sectors(60)
        assert result == ["tech"]

    def test_unknown_sectors_not_included(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = [
            {"sector": "crypto"},  # Not in sector_priority list
            {"sector": "ai"},
        ]
        result = identify_focus_sectors(60)
        assert result == ["tech"]  # Fallback default

    def test_passes_days_to_sql(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []
        identify_focus_sectors(45)
        params = mock_get_cursor.execute.call_args[0][1]
        assert params == (45,)


# ---------------------------------------------------------------------------
# generate_strategy_recommendation
# ---------------------------------------------------------------------------

class TestGenerateStrategyRecommendation:
    """Tests for generate_strategy_recommendation()."""

    @patch("trading.strategy.get_worst_performing_signals")
    @patch("trading.strategy.get_best_performing_signals")
    @patch("trading.strategy.identify_focus_sectors")
    @patch("trading.strategy.build_avoid_list")
    @patch("trading.strategy.build_watchlist")
    @patch("trading.strategy.calculate_overall_performance")
    def test_returns_strategy_recommendation(self, mock_perf, mock_watchlist,
                                               mock_avoid, mock_sectors,
                                               mock_best, mock_worst):
        mock_perf.return_value = {
            "total_trades": 20, "avg_return": 2.5,
            "win_rate": 0.65, "total_return": 50.0,
        }
        mock_watchlist.return_value = ["AAPL", "MSFT"]
        mock_avoid.return_value = ["BAD1"]
        mock_sectors.return_value = ["tech", "finance"]
        mock_best.return_value = [
            {"category": "earnings", "sentiment": "bullish", "avg_outcome": 5.0},
        ]
        mock_worst.return_value = [
            {"category": "lawsuit", "sentiment": "bearish", "avg_outcome": -3.0},
        ]

        result = generate_strategy_recommendation(60)

        assert isinstance(result, StrategyRecommendation)
        assert result.watchlist == ["AAPL", "MSFT"]
        assert result.avoid_list == ["BAD1"]
        assert result.risk_tolerance == "aggressive"
        assert result.focus_sectors == ["tech", "finance"]
        assert len(result.reasoning) > 0

    @patch("trading.strategy.get_worst_performing_signals")
    @patch("trading.strategy.get_best_performing_signals")
    @patch("trading.strategy.identify_focus_sectors")
    @patch("trading.strategy.build_avoid_list")
    @patch("trading.strategy.build_watchlist")
    @patch("trading.strategy.calculate_overall_performance")
    def test_description_momentum_when_winning(self, mock_perf, mock_watchlist,
                                                 mock_avoid, mock_sectors,
                                                 mock_best, mock_worst):
        mock_perf.return_value = {
            "total_trades": 10, "avg_return": 1.0,
            "win_rate": 0.55, "total_return": 10.0,
        }
        mock_watchlist.return_value = ["AAPL"]
        mock_avoid.return_value = []
        mock_sectors.return_value = ["tech"]
        mock_best.return_value = []
        mock_worst.return_value = []

        result = generate_strategy_recommendation(60)
        assert "Momentum" in result.description

    @patch("trading.strategy.get_worst_performing_signals")
    @patch("trading.strategy.get_best_performing_signals")
    @patch("trading.strategy.identify_focus_sectors")
    @patch("trading.strategy.build_avoid_list")
    @patch("trading.strategy.build_watchlist")
    @patch("trading.strategy.calculate_overall_performance")
    def test_description_conservative_when_losing(self, mock_perf, mock_watchlist,
                                                    mock_avoid, mock_sectors,
                                                    mock_best, mock_worst):
        mock_perf.return_value = {
            "total_trades": 10, "avg_return": -1.0,
            "win_rate": 0.4, "total_return": -10.0,
        }
        mock_watchlist.return_value = []
        mock_avoid.return_value = []
        mock_sectors.return_value = ["tech"]
        mock_best.return_value = []
        mock_worst.return_value = []

        result = generate_strategy_recommendation(60)
        assert "Conservative" in result.description


# ---------------------------------------------------------------------------
# evolve_strategy
# ---------------------------------------------------------------------------

class TestEvolveStrategy:
    """Tests for evolve_strategy()."""

    @patch("trading.strategy.save_strategy")
    @patch("trading.strategy.generate_strategy_recommendation")
    def test_dry_run_does_not_save(self, mock_generate, mock_save):
        mock_generate.return_value = StrategyRecommendation(
            watchlist=["AAPL"],
            avoid_list=[],
            risk_tolerance="moderate",
            focus_sectors=["tech"],
            description="Test",
            reasoning=["reason1"],
        )

        result = evolve_strategy(days=60, dry_run=True)

        assert isinstance(result, StrategyRecommendation)
        mock_save.assert_not_called()

    @patch("trading.strategy.save_strategy")
    @patch("trading.strategy.generate_strategy_recommendation")
    def test_saves_when_not_dry_run(self, mock_generate, mock_save):
        rec = StrategyRecommendation(
            watchlist=["AAPL", "MSFT"],
            avoid_list=["BAD1"],
            risk_tolerance="aggressive",
            focus_sectors=["tech", "finance"],
            description="Momentum strategy",
            reasoning=["r1", "r2"],
        )
        mock_generate.return_value = rec
        mock_save.return_value = 99

        result = evolve_strategy(days=60, dry_run=False)

        assert result is rec
        mock_save.assert_called_once_with(
            description="Momentum strategy",
            watchlist=["AAPL", "MSFT"],
            risk_tolerance="aggressive",
            focus_sectors=["tech", "finance"],
        )

    @patch("trading.strategy.save_strategy")
    @patch("trading.strategy.generate_strategy_recommendation")
    def test_passes_days_to_generate(self, mock_generate, mock_save):
        mock_generate.return_value = StrategyRecommendation(
            watchlist=[], avoid_list=[], risk_tolerance="conservative",
            focus_sectors=["tech"], description="Test", reasoning=[],
        )

        evolve_strategy(days=90, dry_run=True)

        mock_generate.assert_called_once_with(90)

    @patch("trading.strategy.save_strategy")
    @patch("trading.strategy.generate_strategy_recommendation")
    def test_returns_recommendation(self, mock_generate, mock_save):
        rec = StrategyRecommendation(
            watchlist=["GOOG"],
            avoid_list=[],
            risk_tolerance="moderate",
            focus_sectors=["tech"],
            description="Test",
            reasoning=["reason"],
        )
        mock_generate.return_value = rec

        result = evolve_strategy(days=60, dry_run=True)

        assert result is rec
        assert result.watchlist == ["GOOG"]
