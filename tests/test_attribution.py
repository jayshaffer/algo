"""Tests for signal attribution computation."""

from decimal import Decimal
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest

from trading.attribution import compute_signal_attribution, get_attribution_summary


@pytest.fixture
def mock_cursor():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("trading.attribution.get_cursor", _get_cursor), \
         patch("trading.attribution.upsert_signal_attribution") as mock_upsert, \
         patch("trading.attribution.get_signal_attribution") as mock_get_attr, \
         patch("trading.db.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        mock_get_attr.return_value = mock_cursor.fetchall.return_value
        mock_cursor._mock_upsert = mock_upsert
        mock_cursor._mock_get_attr = mock_get_attr
        yield mock_cursor


class TestComputeAttribution:
    def test_computes_from_decision_signals(self, mock_db):
        """Attribution query joins decision_signals with decisions."""
        mock_db.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 15,
                "avg_outcome_7d": Decimal("2.10"),
                "avg_outcome_30d": Decimal("4.30"),
                "win_rate_7d": Decimal("0.67"),
                "win_rate_30d": Decimal("0.60"),
            },
            {
                "category": "thesis",
                "sample_size": 8,
                "avg_outcome_7d": Decimal("3.50"),
                "avg_outcome_30d": Decimal("5.10"),
                "win_rate_7d": Decimal("0.75"),
                "win_rate_30d": Decimal("0.63"),
            },
        ]

        results = compute_signal_attribution()
        assert len(results) == 2

        # Verify SQL references decision_signals
        sql = mock_db.execute.call_args_list[0][0][0]
        assert "decision_signals" in sql
        assert "decisions" in sql

    def test_empty_data_returns_empty(self, mock_db):
        mock_db.fetchall.return_value = []
        results = compute_signal_attribution()
        assert results == []

    def test_upserts_results(self, mock_db):
        """After computing, results are upserted to signal_attribution."""
        mock_db.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("1.5"),
                "avg_outcome_30d": Decimal("3.0"),
                "win_rate_7d": Decimal("0.60"),
                "win_rate_30d": Decimal("0.55"),
            },
        ]

        compute_signal_attribution()

        # Should have called upsert_signal_attribution once for the one result
        mock_db._mock_upsert.assert_called_once_with(
            category="news_signal:earnings",
            sample_size=10,
            avg_outcome_7d=Decimal("1.5"),
            avg_outcome_30d=Decimal("3.0"),
            win_rate_7d=Decimal("0.60"),
            win_rate_30d=Decimal("0.55"),
        )

    def test_upserts_multiple_results(self, mock_db):
        """Multiple categories each get their own upsert call."""
        mock_db.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 15,
                "avg_outcome_7d": Decimal("2.10"),
                "avg_outcome_30d": Decimal("4.30"),
                "win_rate_7d": Decimal("0.67"),
                "win_rate_30d": Decimal("0.60"),
            },
            {
                "category": "thesis",
                "sample_size": 8,
                "avg_outcome_7d": Decimal("3.50"),
                "avg_outcome_30d": Decimal("5.10"),
                "win_rate_7d": Decimal("0.75"),
                "win_rate_30d": Decimal("0.63"),
            },
        ]

        compute_signal_attribution()
        assert mock_db._mock_upsert.call_count == 2

    def test_handles_none_outcomes(self, mock_db):
        """None outcomes are replaced with Decimal(0) for upsert."""
        mock_db.fetchall.return_value = [
            {
                "category": "macro_signal:fed",
                "sample_size": 3,
                "avg_outcome_7d": Decimal("1.0"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": None,
            },
        ]

        compute_signal_attribution()

        mock_db._mock_upsert.assert_called_once_with(
            category="macro_signal:fed",
            sample_size=3,
            avg_outcome_7d=Decimal("1.0"),
            avg_outcome_30d=Decimal(0),
            win_rate_7d=Decimal("0.50"),
            win_rate_30d=Decimal(0),
        )


class TestGetAttributionSummary:
    def test_formats_summary(self, mock_db):
        mock_db._mock_get_attr.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 15,
                "avg_outcome_7d": Decimal("2.10"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("0.67"),
                "win_rate_30d": None,
                "updated_at": None,
            },
        ]

        summary = get_attribution_summary()
        assert "news_signal:earnings" in summary
        assert "67" in summary  # win rate percentage

    def test_no_data_returns_message(self, mock_db):
        mock_db._mock_get_attr.return_value = []
        summary = get_attribution_summary()
        assert "No attribution data" in summary

    def test_separates_predictive_and_weak(self, mock_db):
        """Signals with >50% win rate are predictive, rest are weak."""
        mock_db._mock_get_attr.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("2.50"),
                "avg_outcome_30d": Decimal("5.00"),
                "win_rate_7d": Decimal("0.70"),
                "win_rate_30d": Decimal("0.60"),
                "updated_at": None,
            },
            {
                "category": "macro_signal:fed",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("-0.50"),
                "avg_outcome_30d": Decimal("0.10"),
                "win_rate_7d": Decimal("0.40"),
                "win_rate_30d": Decimal("0.45"),
                "updated_at": None,
            },
        ]

        summary = get_attribution_summary()
        assert "Predictive signal types:" in summary
        assert "Weak/non-predictive signal types:" in summary
        assert "news_signal:earnings" in summary
        assert "macro_signal:fed" in summary

    def test_formats_return_with_sign(self, mock_db):
        """Avg return should have +/- sign in output."""
        mock_db._mock_get_attr.return_value = [
            {
                "category": "thesis",
                "sample_size": 5,
                "avg_outcome_7d": Decimal("-1.20"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("0.30"),
                "win_rate_30d": None,
                "updated_at": None,
            },
        ]

        summary = get_attribution_summary()
        assert "-1.20" in summary
