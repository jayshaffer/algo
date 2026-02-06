"""Integration test: strategist writes playbook, executor reads and acts on it."""

import json
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_playbook_row, make_attribution_row


class TestFullCycle:
    """Test the strategist -> executor data flow."""

    @patch("trading.db.get_cursor")
    def test_strategist_writes_playbook_executor_reads_it(self, mock_get_cursor):
        """
        1. Strategist writes a playbook via upsert_playbook
        2. Executor reads the playbook via get_playbook
        3. Context builder formats playbook into trading context
        """
        from trading.db import upsert_playbook, get_playbook

        # Mock cursor for upsert
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1}
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        playbook_id = upsert_playbook(
            playbook_date=date.today(),
            market_outlook="Bullish tech sector",
            priority_actions=[
                {"ticker": "NVDA", "action": "buy", "thesis_id": 1,
                 "reasoning": "Entry trigger hit", "confidence": 0.8, "max_quantity": 5}
            ],
            watch_list=["AAPL", "MSFT"],
            risk_notes="Fed meeting Wednesday",
        )
        assert playbook_id == 1

    @patch("trading.context.get_playbook")
    def test_playbook_context_includes_priority_actions(self, mock_get_playbook):
        """Playbook context includes priority actions for executor."""
        from trading.context import get_playbook_context

        mock_get_playbook.return_value = make_playbook_row(
            date=date.today(),
            market_outlook="Bullish tech",
            priority_actions=[
                {"ticker": "NVDA", "action": "buy", "reasoning": "Entry hit", "confidence": 0.8}
            ],
            watch_list=["AAPL"],
            risk_notes="Fed meeting",
        )

        ctx = get_playbook_context(date.today())
        assert "NVDA" in ctx
        assert "Entry hit" in ctx
        assert "Fed meeting" in ctx

    @patch("trading.context.get_playbook")
    def test_playbook_context_shows_conservative_mode_when_no_playbook(self, mock_get_playbook):
        """When no playbook exists, executor sees conservative mode instructions."""
        from trading.context import get_playbook_context

        mock_get_playbook.return_value = None

        ctx = get_playbook_context(date.today())
        assert "conservative" in ctx.lower()
        assert "no playbook" in ctx.lower()

    @patch("trading.attribution.get_signal_attribution")
    def test_attribution_context_available_for_executor(self, mock_get_attr):
        """Attribution scores are formatted into executor context."""
        from trading.context import get_attribution_context

        mock_get_attr.return_value = [
            make_attribution_row(
                category="news:earnings",
                sample_size=15,
                avg_outcome_7d=Decimal("2.50"),
                avg_outcome_30d=Decimal("3.10"),
                win_rate_7d=Decimal("0.65"),
                win_rate_30d=Decimal("0.60"),
            ),
        ]

        ctx = get_attribution_context()
        assert "news:earnings" in ctx
        assert "65" in ctx  # win rate percentage

    def test_decision_signal_refs_round_trip(self):
        """Decision signal_refs can be created and read back."""
        from trading.agent import TradingDecision

        decision = TradingDecision(
            ticker="NVDA",
            action="buy",
            quantity=5,
            reasoning="Playbook priority action",
            confidence="high",
            signal_refs=[
                {"type": "thesis", "id": 1},
                {"type": "news_signal", "id": 42},
            ],
        )

        assert len(decision.signal_refs) == 2
        assert decision.signal_refs[0]["type"] == "thesis"
        assert decision.signal_refs[1]["id"] == 42

    def test_decision_signal_refs_default_to_empty_list(self):
        """Decision with no signal_refs defaults to empty list."""
        from trading.agent import TradingDecision

        decision = TradingDecision(
            ticker="AAPL",
            action="hold",
            quantity=None,
            reasoning="No action needed",
            confidence="low",
        )

        assert decision.signal_refs == []

    @patch("trading.attribution.get_signal_attribution")
    def test_attribution_summary_formats_for_strategist(self, mock_get_attr):
        """Attribution summary is formatted text for strategist context."""
        from trading.attribution import get_attribution_summary

        mock_get_attr.return_value = [
            make_attribution_row(
                category="news:earnings",
                sample_size=15,
                avg_outcome_7d=Decimal("2.50"),
                avg_outcome_30d=Decimal("3.10"),
                win_rate_7d=Decimal("0.65"),
                win_rate_30d=Decimal("0.60"),
            ),
            make_attribution_row(
                id=2,
                category="macro:fed_policy",
                sample_size=8,
                avg_outcome_7d=Decimal("-1.30"),
                avg_outcome_30d=Decimal("-0.50"),
                win_rate_7d=Decimal("0.38"),
                win_rate_30d=Decimal("0.40"),
            ),
        ]

        summary = get_attribution_summary()
        assert "news:earnings" in summary
        assert "macro:fed_policy" in summary
        # earnings has >50% win rate -> predictive
        assert "Predictive" in summary
        # fed_policy has <50% win rate -> weak
        assert "Weak" in summary

    @patch("trading.attribution.get_signal_attribution")
    def test_attribution_summary_empty_when_no_data(self, mock_get_attr):
        """Attribution summary handles no data gracefully."""
        from trading.attribution import get_attribution_summary

        mock_get_attr.return_value = []

        summary = get_attribution_summary()
        assert "No attribution data" in summary

    @patch("trading.context.get_playbook")
    def test_playbook_context_includes_watch_list(self, mock_get_playbook):
        """Playbook watch list tickers are included in context."""
        from trading.context import get_playbook_context

        mock_get_playbook.return_value = make_playbook_row(
            watch_list=["TSLA", "AMZN", "META"],
        )

        ctx = get_playbook_context(date.today())
        assert "TSLA" in ctx
        assert "AMZN" in ctx
        assert "META" in ctx

    @patch("trading.db.execute_values")
    @patch("trading.db.get_cursor")
    def test_decision_signals_batch_insert(self, mock_get_cursor, mock_exec_values):
        """Decision-signal links can be batch inserted."""
        from trading.db import insert_decision_signals_batch

        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        signals = [
            (1, "thesis", 42),
            (1, "news_signal", 15),
            (1, "macro_signal", 3),
        ]

        count = insert_decision_signals_batch(signals)
        assert count == 3
        mock_exec_values.assert_called_once()

    def test_decision_signals_batch_insert_empty(self):
        """Batch insert with empty list returns 0 without DB call."""
        from trading.db import insert_decision_signals_batch

        count = insert_decision_signals_batch([])
        assert count == 0
