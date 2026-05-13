"""Regression test: get_news_signals must return the summary column."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch


class TestGetNewsSignalsReturnsSummary:
    @patch("v2.database.trading_db.get_cursor")
    def test_get_news_signals_includes_summary_in_each_row(self, mock_get_cursor):
        from v2.database.trading_db import get_news_signals

        # Mock the cursor context manager + fetchall to return rows that
        # include the new summary key.
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "id": 1,
                "ticker": "AAPL",
                "headline": "AAPL hits ATH",
                "category": "momentum",
                "sentiment": "bullish",
                "confidence": "high",
                "published_at": datetime(2026, 5, 9, 12, 0, tzinfo=UTC),
                "alpaca_id": "alp-123",
                "summary": "Apple closed at $300 after the Foxconn deal.",
                "processed_at": datetime(2026, 5, 9, 12, 5),
            },
        ]
        mock_get_cursor.return_value.__enter__.return_value = mock_cur

        rows = get_news_signals(days=7)

        assert len(rows) == 1
        assert "summary" in rows[0], f"summary missing from row keys: {list(rows[0].keys())}"
        assert rows[0]["summary"] == "Apple closed at $300 after the Foxconn deal."
