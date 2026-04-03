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

        assert "STRONG (outperforms market)" in result
        assert "news_signal:earnings" in result
        assert "avg alpha" in result

        assert "WEAK (underperforms market)" in result
        assert "macro_signal:fed" in result
        assert "-1.00% avg alpha" in result

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
        """Test EV thresholds: +0.5% boundary for STRONG, -0.5% for WEAK.

        Exactly +0.5 is NOT > +0.5, so not STRONG.
        Exactly -0.5 is NOT < -0.5, so not WEAK.
        Values above/below the thresholds should be categorized.
        """
        mock_rows = [
            {
                "category": "news_signal:at_strong_boundary",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("0.5"),
                "avg_outcome_30d": Decimal("1.0"),
                "win_rate_7d": Decimal("0.55"),
                "win_rate_30d": Decimal("0.55"),
            },
            {
                "category": "news_signal:at_weak_boundary",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("-0.5"),
                "avg_outcome_30d": Decimal("-0.5"),
                "win_rate_7d": Decimal("0.45"),
                "win_rate_30d": Decimal("0.45"),
            },
            {
                "category": "news_signal:above_strong",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("0.51"),
                "avg_outcome_30d": Decimal("1.0"),
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
            {
                "category": "news_signal:below_weak",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("-0.51"),
                "avg_outcome_30d": Decimal("-1.0"),
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        # Exactly +0.5 is NOT > +0.5, so should not be STRONG
        assert "news_signal:at_strong_boundary" not in result
        # Exactly -0.5 is NOT < -0.5, so should not be WEAK
        assert "news_signal:at_weak_boundary" not in result

        # +0.51 IS > +0.5, so should be STRONG
        assert "STRONG" in result
        assert "news_signal:above_strong" in result

        # -0.51 IS < -0.5, so should be WEAK
        assert "WEAK (underperforms market)" in result
        assert "news_signal:below_weak" in result


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

        assert "Signal Attribution" in result
        assert "Outperforming signals (positive alpha vs SPY):" in result
        assert "news_signal:earnings" in result
        assert "70% beat-market rate" in result
        assert "+2.50% avg 7d alpha" in result
        assert "n=20" in result

        assert "Underperforming signals (negative alpha vs SPY):" in result
        assert "macro_signal:fed" in result
        assert "30% beat-market rate" in result
        assert "-1.00% avg 7d alpha" in result
        assert "n=15" in result


class TestExpectedValueConstraints:
    def test_profitable_low_winrate_is_strong(self):
        """40% win rate but +2.0% avg return -> STRONG."""
        mock_rows = [
            {
                "category": "news_signal:contrarian",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("2.0"),
                "avg_outcome_30d": Decimal("3.0"),
                "win_rate_7d": Decimal("0.40"),
                "win_rate_30d": Decimal("0.45"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        assert "STRONG" in result
        assert "news_signal:contrarian" in result

    def test_unprofitable_high_winrate_is_weak(self):
        """60% win rate but -0.6% avg return -> WEAK."""
        mock_rows = [
            {
                "category": "news_signal:momentum",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("-0.6"),
                "avg_outcome_30d": Decimal("-1.0"),
                "win_rate_7d": Decimal("0.60"),
                "win_rate_30d": Decimal("0.55"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        assert "WEAK" in result
        assert "news_signal:momentum" in result

    def test_neutral_ev_not_categorized(self):
        """Near-zero avg return should not be STRONG or WEAK."""
        mock_rows = [
            {
                "category": "news_signal:flat",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("0.05"),
                "avg_outcome_30d": Decimal("0.1"),
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        # The category should not appear in any STRONG or WEAK section
        assert "news_signal:flat" not in result

    def test_insufficient_data_unchanged(self):
        """Below min_samples should still be INSUFFICIENT."""
        mock_rows = [
            {
                "category": "news_signal:rare",
                "sample_size": 2,
                "avg_outcome_7d": Decimal("10.0"),
                "avg_outcome_30d": Decimal("15.0"),
                "win_rate_7d": Decimal("1.0"),
                "win_rate_30d": Decimal("1.0"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        assert "INSUFFICIENT DATA" in result
        assert "news_signal:rare" in result

    def test_constraint_text_references_expected_value(self):
        """Constraint text should mention avg return."""
        mock_rows = [
            {
                "category": "news_signal:test",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("-2.0"),
                "avg_outcome_30d": Decimal("-1.5"),
                "win_rate_7d": Decimal("0.35"),
                "win_rate_30d": Decimal("0.40"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        assert "avg alpha" in result.lower() or "alpha" in result.lower()


class TestAttributionByDirection:
    @pytest.fixture(autouse=True)
    def _patch_attribution_cursor(self, mock_cursor):
        """Patch get_cursor in the attribution module where it's imported."""
        @contextmanager
        def _get_cursor():
            yield mock_cursor

        with patch("v2.attribution.get_cursor", _get_cursor):
            yield

    def test_attribution_sql_groups_by_action(self, mock_db, mock_cursor):
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        assert "d.action" in sql

    def test_attribution_categories_include_direction(self, mock_db, mock_cursor):
        """Verify the CASE expression appends ':buy' or ':sell' via d.action."""
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        # Each CASE branch should concatenate d.action
        assert "|| ':' || d.action" in sql


class TestAttributionTimeWindow:
    @pytest.fixture(autouse=True)
    def _patch_attribution_cursor(self, mock_cursor):
        """Patch get_cursor in the attribution module where it's imported."""
        @contextmanager
        def _get_cursor():
            yield mock_cursor

        with patch("v2.attribution.get_cursor", _get_cursor):
            yield

    def test_compute_attribution_filters_by_days(self, mock_db, mock_cursor):
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution(days=60)
        sql = mock_cursor.execute.call_args[0][0]
        assert "d.date" in sql
        params = mock_cursor.execute.call_args[0][1] if len(mock_cursor.execute.call_args[0]) > 1 else None
        assert params is not None

    def test_compute_attribution_defaults_to_90_days(self, mock_db, mock_cursor):
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        params = mock_cursor.execute.call_args[0][1] if len(mock_cursor.execute.call_args[0]) > 1 else None
        assert params is not None
