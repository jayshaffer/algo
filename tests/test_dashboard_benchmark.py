"""Tests for dashboard/benchmark.py — SPY benchmark and alpha computation."""

from datetime import date
from decimal import Decimal

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
