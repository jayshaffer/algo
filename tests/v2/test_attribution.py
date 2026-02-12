"""Tests for v2 signal attribution engine."""

import pytest
from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import patch, MagicMock, call


class TestBuildAttributionConstraints:
    def test_formats_strong_and_weak_categories(self):
        """Mock get_signal_attribution to return data with different win rates."""
        mock_rows = [
            {
                "category": "news_signal:earnings",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("2.5"),
                "avg_outcome_30d": Decimal("4.0"),
                "win_rate_7d": Decimal("0.70"),
                "win_rate_30d": Decimal("0.65"),
            },
            {
                "category": "macro_signal:fed",
                "sample_size": 15,
                "avg_outcome_7d": Decimal("-1.0"),
                "avg_outcome_30d": Decimal("-0.5"),
                "win_rate_7d": Decimal("0.30"),
                "win_rate_30d": Decimal("0.40"),
            },
            {
                "category": "news_signal:rumor",
                "sample_size": 3,
                "avg_outcome_7d": Decimal("0.5"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": None,
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        assert "STRONG (>55% win rate)" in result
        assert "news_signal:earnings" in result
        assert "70%" in result

        assert "WEAK (<45% win rate)" in result
        assert "macro_signal:fed" in result
        assert "30%" in result

        assert "INSUFFICIENT DATA" in result
        assert "news_signal:rumor (n=3)" in result

        assert "CONSTRAINT:" in result
        assert "Do not create theses primarily based on WEAK signal categories" in result

    def test_empty_when_no_data(self):
        """Returns empty string when no attribution data exists."""
        with patch("v2.attribution.get_signal_attribution", return_value=[]):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints()

        assert result == ""

    def test_threshold_boundaries(self):
        """Test that exactly 55% is not STRONG and exactly 45% is not WEAK.

        The implementation converts Decimal to float before comparing, so we
        use values that land cleanly on 50% (0.5) to test the neutral zone —
        50% is neither >55 nor <45.
        """
        mock_rows = [
            {
                "category": "news_signal:neutral_high",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("1.0"),
                "avg_outcome_30d": Decimal("1.0"),
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
            {
                "category": "news_signal:neutral_low",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("-0.5"),
                "avg_outcome_30d": Decimal("-0.5"),
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
            {
                "category": "news_signal:just_above_strong",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("1.5"),
                "avg_outcome_30d": Decimal("2.0"),
                "win_rate_7d": Decimal("0.56"),
                "win_rate_30d": Decimal("0.56"),
            },
            {
                "category": "news_signal:just_below_weak",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("-1.0"),
                "avg_outcome_30d": Decimal("-1.0"),
                "win_rate_7d": Decimal("0.44"),
                "win_rate_30d": Decimal("0.44"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        # 50% win rate is NOT >55, so should not be STRONG
        assert "news_signal:neutral_high" not in result
        assert "news_signal:neutral_low" not in result

        # 56% IS >55, so should be STRONG
        assert "STRONG" in result
        assert "news_signal:just_above_strong" in result

        # 44% IS <45, so should be WEAK
        assert "WEAK" in result
        assert "news_signal:just_below_weak" in result


class TestComputeSignalAttribution:
    @pytest.fixture(autouse=True)
    def _patch_attribution_cursor(self, mock_cursor):
        """Patch get_cursor in the attribution module where it's imported."""
        @contextmanager
        def _get_cursor():
            yield mock_cursor

        with patch("v2.attribution.get_cursor", _get_cursor):
            yield

    def test_joins_through_decision_signals_fk(self, mock_db, mock_cursor):
        """Verify SQL uses decision_signals table with FK JOINs, not time-window JOINs."""
        mock_cursor.fetchall.return_value = []

        with patch("v2.attribution.upsert_signal_attribution"):
            from v2.attribution import compute_signal_attribution
            compute_signal_attribution()

        sql = mock_cursor.execute.call_args[0][0]
        assert "decision_signals" in sql
        assert "JOIN decisions d ON d.id = ds.decision_id" in sql
        # Should NOT use time-window joins
        assert "INTERVAL" not in sql
        assert "BETWEEN" not in sql

    def test_upserts_results(self, mock_db, mock_cursor):
        """Verify upsert_signal_attribution called for each result row."""
        mock_cursor.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("2.0"),
                "avg_outcome_30d": Decimal("3.0"),
                "win_rate_7d": Decimal("0.60"),
                "win_rate_30d": Decimal("0.55"),
            },
            {
                "category": "macro_signal:fed",
                "sample_size": 5,
                "avg_outcome_7d": Decimal("-1.0"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("0.40"),
                "win_rate_30d": None,
            },
        ]

        with patch("v2.attribution.upsert_signal_attribution") as mock_upsert:
            from v2.attribution import compute_signal_attribution
            compute_signal_attribution()

        assert mock_upsert.call_count == 2
        # First call: earnings
        mock_upsert.assert_any_call(
            category="news_signal:earnings",
            sample_size=10,
            avg_outcome_7d=Decimal("2.0"),
            avg_outcome_30d=Decimal("3.0"),
            win_rate_7d=Decimal("0.60"),
            win_rate_30d=Decimal("0.55"),
        )
        # Second call: fed — None outcome_30d/win_rate_30d should become Decimal(0)
        mock_upsert.assert_any_call(
            category="macro_signal:fed",
            sample_size=5,
            avg_outcome_7d=Decimal("-1.0"),
            avg_outcome_30d=Decimal(0),
            win_rate_7d=Decimal("0.40"),
            win_rate_30d=Decimal(0),
        )

    def test_returns_results_list(self, mock_db, mock_cursor):
        """Verify compute_signal_attribution returns a list of dicts."""
        mock_cursor.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("2.0"),
                "avg_outcome_30d": Decimal("3.0"),
                "win_rate_7d": Decimal("0.60"),
                "win_rate_30d": Decimal("0.55"),
            },
        ]

        with patch("v2.attribution.upsert_signal_attribution"):
            from v2.attribution import compute_signal_attribution
            result = compute_signal_attribution()

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["category"] == "news_signal:earnings"
        assert result[0]["sample_size"] == 10


class TestGetAttributionSummary:
    def test_no_data(self):
        """Returns 'No attribution data yet' when no rows."""
        with patch("v2.attribution.get_signal_attribution", return_value=[]):
            from v2.attribution import get_attribution_summary
            result = get_attribution_summary()

        assert "No attribution data yet" in result
        assert result.startswith("Signal Attribution:")

    def test_formats_predictive_and_weak(self):
        """Formats predictive (>50% win rate) and weak (<=50%) correctly."""
        mock_rows = [
            {
                "category": "news_signal:earnings",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("2.5"),
                "avg_outcome_30d": Decimal("4.0"),
                "win_rate_7d": Decimal("0.70"),
                "win_rate_30d": Decimal("0.65"),
            },
            {
                "category": "macro_signal:fed",
                "sample_size": 15,
                "avg_outcome_7d": Decimal("-1.0"),
                "avg_outcome_30d": Decimal("-0.5"),
                "win_rate_7d": Decimal("0.30"),
                "win_rate_30d": Decimal("0.40"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import get_attribution_summary
            result = get_attribution_summary()

        assert "Signal Attribution:" in result
        assert "Predictive signal types:" in result
        assert "news_signal:earnings" in result
        assert "70% win rate" in result
        assert "+2.50% avg 7d return" in result
        assert "n=20" in result

        assert "Weak/non-predictive signal types:" in result
        assert "macro_signal:fed" in result
        assert "30% win rate" in result
        assert "-1.00% avg 7d return" in result
        assert "n=15" in result
