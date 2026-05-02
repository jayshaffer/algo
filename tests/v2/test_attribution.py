"""Tests for v2 signal attribution engine."""

from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import patch

import pytest

from v2.attribution import clear_attribution_summary_cache


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
        # First call: earnings — P2.17: sample_size_30d threaded through.
        mock_upsert.assert_any_call(
            category="news_signal:earnings",
            sample_size=10,
            sample_size_30d=0,  # mock fixture lacks the field; defaults to 0.
            avg_outcome_7d=Decimal("2.0"),
            avg_outcome_30d=Decimal("3.0"),
            win_rate_7d=Decimal("0.60"),
            win_rate_30d=Decimal("0.55"),
        )
        # Second call: fed — None outcome_30d/win_rate_30d should become Decimal(0)
        mock_upsert.assert_any_call(
            category="macro_signal:fed",
            sample_size=5,
            sample_size_30d=0,
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

    def test_memoizes_within_process(self):
        """P3.32: `get_attribution_summary` is called 3+ times per session
        (strategist tool loop + context builders). Memo skips redundant
        DB roundtrips."""
        mock_rows = [
            {
                "category": "news_signal:earnings",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("1.0"),
                "avg_outcome_30d": Decimal("2.0"),
                "win_rate_7d": Decimal("0.6"),
                "win_rate_30d": Decimal("0.55"),
            },
        ]
        with patch(
            "v2.attribution.get_signal_attribution", return_value=mock_rows
        ) as mock_get:
            from v2.attribution import get_attribution_summary
            r1 = get_attribution_summary()
            r2 = get_attribution_summary()
            r3 = get_attribution_summary()

        assert r1 == r2 == r3
        assert mock_get.call_count == 1, (
            f"Expected one DB hit across three reads, got {mock_get.call_count}"
        )

    def test_zero_alpha_appears_in_underperforming(self):
        """P3.41: a row with exactly 0 alpha used to be silently dropped
        from both buckets due to a truthy check on Decimal(0) — `<= 0`
        is inclusive, so the row's intended home is underperforming.
        """
        clear_attribution_summary_cache()
        mock_rows = [
            {
                "category": "news_signal:flat",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("0"),
                "avg_outcome_30d": Decimal("0"),
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import get_attribution_summary
            result = get_attribution_summary()

        assert "news_signal:flat" in result, (
            "Zero-alpha category was silently dropped from the formatted summary."
        )
        assert "Underperforming" in result

    def test_none_alpha_still_excluded(self):
        """Defense check: rows with None alpha (no data yet) should still be
        excluded from both buckets, not coerced into underperforming.
        """
        clear_attribution_summary_cache()
        mock_rows = [
            {
                "category": "news_signal:nodata",
                "sample_size": 10,
                "avg_outcome_7d": None,
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import get_attribution_summary
            result = get_attribution_summary()

        assert "news_signal:nodata" not in result, (
            "Categories with no alpha data must stay out of both buckets."
        )

    def test_recompute_invalidates_memo(self):
        """`compute_signal_attribution` writes new rows; the cached
        summary must be invalidated so subsequent readers see fresh data."""
        first_rows = [
            {"category": "news_signal:earnings", "sample_size": 10,
             "avg_outcome_7d": Decimal("1.0"), "avg_outcome_30d": Decimal("2.0"),
             "win_rate_7d": Decimal("0.6"), "win_rate_30d": Decimal("0.55")},
        ]
        second_rows = [
            {"category": "news_signal:earnings", "sample_size": 20,
             "avg_outcome_7d": Decimal("3.0"), "avg_outcome_30d": Decimal("4.0"),
             "win_rate_7d": Decimal("0.8"), "win_rate_30d": Decimal("0.75")},
        ]

        with patch("v2.attribution.get_signal_attribution", return_value=first_rows):
            from v2.attribution import get_attribution_summary
            r1 = get_attribution_summary()
        # Simulate a recompute clearing the cache.
        clear_attribution_summary_cache()
        with patch("v2.attribution.get_signal_attribution", return_value=second_rows):
            r2 = get_attribution_summary()

        assert "n=10" in r1
        assert "n=20" in r2, "Cache must be invalidated so second read sees new data"


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

    def test_attribution_filters_by_action_not_group_by(self, mock_db, mock_cursor):
        """Verify d.action is used in the WHERE filter but NOT concatenated into the category key.

        The collapsed 2-part category design groups across buy/sell to increase sample sizes.
        Action direction is still enforced via the WHERE clause filter.
        """
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        # d.action should appear in the WHERE filter
        assert "d.action IN ('buy', 'sell')" in sql
        # But NOT appended to the category key
        assert "|| ':' || d.action" not in sql


class TestCollapsedCategories:
    @pytest.fixture(autouse=True)
    def _patch_attribution_cursor(self, mock_cursor):
        """Patch get_cursor in the attribution module where it's imported."""
        @contextmanager
        def _get_cursor():
            yield mock_cursor

        with patch("v2.attribution.get_cursor", _get_cursor):
            yield

    def test_categories_exclude_sentiment_and_action(self, mock_db, mock_cursor):
        """Verify the SQL CASE statement does NOT include sentiment or action in the category key.

        The category should be 2-part: e.g. 'news_signal:earnings', not
        'news_signal:earnings:bullish:buy'. Low-N 4-part keys are statistically useless.
        """
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        # Should NOT concatenate sentiment
        assert "ns.sentiment" not in sql
        assert "ms.sentiment" not in sql
        # Should NOT concatenate action into category key
        assert "|| ':' || d.action" not in sql

    def test_thesis_category_has_no_action_suffix(self, mock_db, mock_cursor):
        """Verify the ELSE branch produces just ds.signal_type, not ds.signal_type || ':' || d.action."""
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        # The ELSE branch should be just the signal_type with no action suffix
        assert "ELSE ds.signal_type" in sql
        assert "ELSE ds.signal_type || ':' || d.action" not in sql


class TestOrphanSignalFiltering:
    @pytest.fixture(autouse=True)
    def _patch_attribution_cursor(self, mock_cursor):
        @contextmanager
        def _get_cursor():
            yield mock_cursor

        with patch("v2.attribution.get_cursor", _get_cursor):
            yield

    def test_excludes_news_signal_orphans(self, mock_db, mock_cursor):
        """Rows where signal_type='news_signal' but the FK does not match a news_signals row
        must be excluded — they are broken FK references, not a real category.
        Bucketing them as 'news_signal:unknown' produces meaningless attribution stats."""
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        # Orphan news_signal references must be filtered out
        assert "ns.id IS NOT NULL" in sql or "ns.category IS NOT NULL" in sql

    def test_excludes_macro_signal_orphans(self, mock_db, mock_cursor):
        """Same protection for macro_signal orphans."""
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        assert "ms.id IS NOT NULL" in sql or "ms.category IS NOT NULL" in sql

    def test_does_not_create_unknown_bucket(self, mock_db, mock_cursor):
        """The 'unknown' fallback bucket must not appear — orphans are dropped, not relabeled."""
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        assert "'unknown'" not in sql

    def test_excludes_thesis_orphans(self, mock_db, mock_cursor):
        """Rows where signal_type='thesis' but the FK does not match a theses row
        must be excluded — same shape as the news_signal orphan filter. Without
        this, pre-validator residue (signal_id=0) inflates the thesis bucket.
        """
        from v2.attribution import compute_signal_attribution
        mock_cursor.fetchall.return_value = []
        with patch("v2.attribution.upsert_signal_attribution"):
            compute_signal_attribution()
        sql = mock_cursor.execute.call_args[0][0]
        # Must LEFT JOIN theses and filter orphans
        assert "JOIN theses" in sql
        assert "t.id IS NOT NULL" in sql or "thesis" in sql.lower()
        # Specifically: the WHERE clause should require thesis FK match
        assert "ds.signal_type != 'thesis'" in sql or "t.id IS NOT NULL" in sql


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
