"""Tests for dashboard/benchmark.py — SPY benchmark and alpha computation."""

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from dashboard.benchmark import compute_alpha


class TestComputeAlpha:
    """Tests for compute_alpha()."""

    def test_happy_path_aligned_dates(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("101000")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("102000")},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-03", "close": 505.0},
            {"date": "2026-01-06", "close": 510.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        assert result is not None
        # portfolio: (102000 - 100000) / 100000 * 100 = 2.0%
        assert result["portfolio_return"] == pytest.approx(2.0)
        # spy: (510 - 500) / 500 * 100 = 2.0%
        assert result["spy_return"] == pytest.approx(2.0)
        assert result["alpha"] == pytest.approx(0.0)

    def test_positive_alpha(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("105000")},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 505.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        # portfolio: 5%, spy: 1%
        assert result["alpha"] == pytest.approx(4.0)

    def test_negative_alpha(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("99000")},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 510.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        # portfolio: -1%, spy: 2%
        assert result["alpha"] == pytest.approx(-3.0)

    def test_misaligned_dates_weekend(self):
        """Snapshot starts on Monday, SPY has no weekend data."""
        snapshots = [
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("100000")},  # Sat
            {"date": date(2026, 1, 5), "portfolio_value": Decimal("100000")},  # Mon
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("102000")},  # Tue
        ]
        benchmark = [
            {"date": "2026-01-05", "close": 500.0},
            {"date": "2026-01-06", "close": 510.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        assert result is not None
        # Uses first overlapping date (Jan 5): portfolio 100000->102000 = 2%
        # SPY 500->510 = 2%
        assert result["alpha"] == pytest.approx(0.0)

    def test_empty_benchmark_returns_none(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("101000")},
        ]
        assert compute_alpha(snapshots, []) is None

    def test_single_snapshot_returns_none(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
        ]
        benchmark = [{"date": "2026-01-02", "close": 500.0}]
        assert compute_alpha(snapshots, benchmark) is None

    def test_flat_spy_returns_portfolio_return_as_alpha(self):
        """When SPY is flat, alpha should equal portfolio return."""
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("103000")},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 500.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        assert result is not None
        assert result["portfolio_return"] == pytest.approx(3.0)
        assert result["spy_return"] == pytest.approx(0.0)
        assert result["alpha"] == pytest.approx(3.0)

    def test_no_overlapping_dates_returns_none(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("101000")},
        ]
        benchmark = [
            {"date": "2026-01-10", "close": 500.0},
            {"date": "2026-01-11", "close": 505.0},
        ]
        assert compute_alpha(snapshots, benchmark) is None

    def test_none_inputs(self):
        assert compute_alpha(None, [{"date": "2026-01-02", "close": 500.0}]) is None
        assert compute_alpha([], None) is None
        assert compute_alpha(None, None) is None


from dashboard.benchmark import _clear_cache, get_spy_benchmark


class TestGetSpyBenchmark:
    """Tests for get_spy_benchmark() with in-memory TTL cache."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        _clear_cache()
        yield
        _clear_cache()

    def _make_mock_bar(self, dt_str, close):
        bar = MagicMock()
        bar.timestamp = datetime.strptime(dt_str, "%Y-%m-%d")
        bar.close = close
        return bar

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_fetches_from_alpaca(self, mock_client_cls):
        bars = [
            self._make_mock_bar("2026-01-02", 500.0),
            self._make_mock_bar("2026-01-03", 505.0),
        ]
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": bars}
        mock_client_cls.return_value = mock_client

        result = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        assert len(result) == 2
        assert result[0] == {"date": "2026-01-02", "close": 500.0}
        assert result[1] == {"date": "2026-01-03", "close": 505.0}
        mock_client.get_stock_bars.assert_called_once()

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_cache_hit_skips_alpaca(self, mock_client_cls):
        bars = [self._make_mock_bar("2026-01-02", 500.0)]
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": bars}
        mock_client_cls.return_value = mock_client

        result1 = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))
        result2 = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        assert result1 == result2
        assert mock_client.get_stock_bars.call_count == 1

    @patch("dashboard.benchmark.time")
    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_cache_expires_after_ttl(self, mock_client_cls, mock_time):
        bars = [self._make_mock_bar("2026-01-02", 500.0)]
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": bars}
        mock_client_cls.return_value = mock_client

        mock_time.time.return_value = 1000.0
        get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        mock_time.time.return_value = 1000.0 + 901  # Past 900s TTL
        get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        assert mock_client.get_stock_bars.call_count == 2

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_alpaca_error_returns_empty(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("connection refused")
        result = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))
        assert result == []

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_empty_bars_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": []}
        mock_client_cls.return_value = mock_client
        result = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))
        assert result == []


from dashboard.benchmark import (
    _clear_deposit_cache,
    enrich_snapshots_with_deposits,
    get_deposit_history,
)


class TestEnrichSnapshotsWithDeposits:
    """Tests for enrich_snapshots_with_deposits()."""

    def test_empty_snapshots_returns_empty(self):
        assert enrich_snapshots_with_deposits([], []) == []

    def test_no_deposits_uses_first_value_as_base(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("1000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("1100")},
        ]
        result = enrich_snapshots_with_deposits(snapshots, [])
        assert result[0]["cumulative_deposits"] == 1000.0
        assert result[1]["cumulative_deposits"] == 1000.0

    def test_deposit_after_first_snapshot_appears_in_later_row(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("1000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("1050")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("2100")},
        ]
        deposits = [{"date": "2026-01-05", "amount": 1000.0}]
        result = enrich_snapshots_with_deposits(snapshots, deposits)
        # Pre-deposit rows: nothing yet
        assert result[0]["cumulative_deposits"] == 0.0
        assert result[1]["cumulative_deposits"] == 0.0
        # Jan 5 deposit visible from Jan 6 onward
        assert result[2]["cumulative_deposits"] == 1000.0

    def test_deposit_on_first_snapshot_date_credited_same_day(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("1000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("1050")},
        ]
        deposits = [{"date": "2026-01-02", "amount": 1000.0}]
        result = enrich_snapshots_with_deposits(snapshots, deposits)
        assert result[0]["cumulative_deposits"] == 1000.0
        assert result[1]["cumulative_deposits"] == 1000.0

    def test_deposit_before_window_credited_to_all(self):
        """Initial funding that predates the window should appear in row[0]."""
        snapshots = [
            {"date": date(2026, 2, 7), "portfolio_value": Decimal("1000")},
            {"date": date(2026, 2, 9), "portfolio_value": Decimal("1010")},
        ]
        deposits = [{"date": "2026-01-15", "amount": 1000.0}]
        result = enrich_snapshots_with_deposits(snapshots, deposits)
        assert result[0]["cumulative_deposits"] == 1000.0
        assert result[1]["cumulative_deposits"] == 1000.0

    def test_non_mutating(self):
        """Should not mutate the input snapshots."""
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("1000")},
        ]
        enrich_snapshots_with_deposits(snapshots, [])
        assert "cumulative_deposits" not in snapshots[0]


class TestComputeAlphaWithDeposits:
    """Tests for compute_alpha() TWR path when snapshots have cumulative_deposits."""

    def test_twr_nets_out_deposit(self):
        """A mid-window deposit should not inflate portfolio return."""
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("1000"),
             "cumulative_deposits": 1000.0},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("1100"),
             "cumulative_deposits": 1000.0},
            # Deposit of 1000 credited here: value jumps 1100 -> 2100
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("2100"),
             "cumulative_deposits": 2000.0},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-03", "close": 500.0},
            {"date": "2026-01-06", "close": 500.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        # Sub-period 1: 1100/1000 = 1.10
        # Sub-period 2: 2100/(1100+1000) = 2100/2100 = 1.00 (deposit netted)
        # Total growth: 1.10, return: 10%
        assert result["portfolio_return"] == pytest.approx(10.0)
        assert result["spy_return"] == pytest.approx(0.0)
        assert result["alpha"] == pytest.approx(10.0)

    def test_twr_without_deposits_matches_simple_return(self):
        """When cumulative_deposits is flat, TWR should equal simple end/start return."""
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("1000"),
             "cumulative_deposits": 1000.0},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("1200"),
             "cumulative_deposits": 1000.0},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("1100"),
             "cumulative_deposits": 1000.0},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 500.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        # 1100 / 1000 = 1.10 -> 10%
        assert result["portfolio_return"] == pytest.approx(10.0)

    def test_twr_with_withdrawal(self):
        """A mid-window withdrawal should also be netted out (negative delta)."""
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("2000"),
             "cumulative_deposits": 2000.0},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("2200"),
             "cumulative_deposits": 2000.0},
            # Withdrawal of 500: value drops 2200 -> 1700
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("1700"),
             "cumulative_deposits": 1500.0},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 500.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        # Sub 1: 2200/2000 = 1.10
        # Sub 2: 1700/(2200 + (-500)) = 1700/1700 = 1.00
        # Total: 1.10, return: 10%
        assert result["portfolio_return"] == pytest.approx(10.0)


class TestGetDepositHistory:
    """Tests for get_deposit_history() with in-memory TTL cache."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        _clear_deposit_cache()
        yield
        _clear_deposit_cache()

    @patch("dashboard.benchmark.TradingClient")
    def test_fetches_from_alpaca(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.return_value = [
            {"date": "2026-01-02", "net_amount": "1000.00", "id": "a1"},
            {"date": "2026-03-20", "net_amount": "500.00", "id": "a2"},
        ]
        mock_client_cls.return_value = mock_client

        result = get_deposit_history()
        assert len(result) == 2
        assert result[0] == {"date": "2026-01-02", "amount": 1000.0}
        assert result[1] == {"date": "2026-03-20", "amount": 500.0}

    @patch("dashboard.benchmark.TradingClient")
    def test_cache_hit_skips_alpaca(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.return_value = [
            {"date": "2026-01-02", "net_amount": "1000.00", "id": "a1"},
        ]
        mock_client_cls.return_value = mock_client

        get_deposit_history()
        get_deposit_history()
        # Two get_deposit_history calls, but only one Alpaca call
        # (pagination loop calls client.get once per page; one page of data here)
        assert mock_client.get.call_count == 1

    @patch("dashboard.benchmark.TradingClient")
    def test_alpaca_error_returns_empty(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("auth failed")
        assert get_deposit_history() == []

    @patch("dashboard.benchmark.TradingClient")
    def test_empty_response_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.return_value = []
        mock_client_cls.return_value = mock_client
        assert get_deposit_history() == []
