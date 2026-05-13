"""Tests for v2/news_backfill.py — one-shot summary backfill from Alpaca."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch


class TestNewsBackfill:
    def _news_item(self, id_: str, summary: str):
        from v2.news import NewsItem
        return NewsItem(
            id=id_,
            headline=f"headline {id_}",
            summary=summary,
            author="x",
            source="Reuters",
            symbols=["AAPL"],
            published_at=datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
            url="https://example.com",
        )

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_updates_summary_for_matching_alpaca_id(self, mock_fetch, mock_get_cursor):
        from v2.news_backfill import run

        mock_fetch.return_value = [self._news_item("alp-1", "Foxconn deal pushes AAPL")]
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert mock_cur.execute.called
        executed_sql, executed_params = mock_cur.execute.call_args[0]
        assert "UPDATE news_signals" in executed_sql
        assert "summary IS NULL" in executed_sql or "summary IS NULL OR summary = ''" in executed_sql
        assert "Foxconn deal pushes AAPL" in executed_params
        assert "alp-1" in executed_params
        assert stats["updated"] == 1

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_idempotent_skips_already_populated_rows(self, mock_fetch, mock_get_cursor):
        from v2.news_backfill import run

        mock_fetch.return_value = [self._news_item("alp-1", "new summary")]
        # Simulate the WHERE summary IS NULL guard: no row updated.
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert stats["updated"] == 0
        assert stats["skipped_or_no_match"] == 1

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_skips_items_with_empty_summary(self, mock_fetch, mock_get_cursor):
        """Don't push empty strings into the column."""
        from v2.news_backfill import run

        mock_fetch.return_value = [self._news_item("alp-1", "")]
        mock_cur = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert not mock_cur.execute.called, "must not UPDATE for items with empty summary"
        assert stats["updated"] == 0
        assert stats["skipped_or_no_match"] == 1

    @patch("v2.news_backfill.get_cursor")
    @patch("v2.news_backfill.fetch_news")
    def test_no_news_fetched_is_clean_exit(self, mock_fetch, mock_get_cursor):
        from v2.news_backfill import run

        mock_fetch.return_value = []
        mock_cur = MagicMock()
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        stats = run(hours=168)

        assert stats == {"fetched": 0, "updated": 0, "skipped_or_no_match": 0}
        mock_cur.execute.assert_not_called()
