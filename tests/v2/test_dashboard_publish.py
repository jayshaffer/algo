"""Tests for dashboard data gathering module."""

import json
from datetime import date, datetime
from decimal import Decimal

from v2.dashboard_publish import _DecimalEncoder, gather_dashboard_data


class TestDecimalEncoder:
    def test_encodes_decimal_as_float(self):
        result = json.dumps({"value": Decimal("123.45")}, cls=_DecimalEncoder)
        assert result == '{"value": 123.45}'

    def test_encodes_date_as_iso(self):
        result = json.dumps({"d": date(2025, 1, 15)}, cls=_DecimalEncoder)
        assert result == '{"d": "2025-01-15"}'

    def test_encodes_datetime_as_iso(self):
        dt = datetime(2025, 1, 15, 10, 30, 0)
        result = json.dumps({"dt": dt}, cls=_DecimalEncoder)
        assert result == '{"dt": "2025-01-15T10:30:00"}'

    def test_raises_for_unsupported_type(self):
        import pytest
        with pytest.raises(TypeError):
            json.dumps({"x": object()}, cls=_DecimalEncoder)


class TestGatherDashboardData:
    def test_returns_all_sections(self, mock_db):
        """Happy path: all sections populated with data."""
        session_date = date(2025, 6, 15)

        # Set up fetchall side_effect for sequential calls:
        # 1. snapshots, 2. positions, 3. decisions, 4. theses
        mock_db.fetchall.side_effect = [
            # snapshots (last 90 days)
            [
                {"date": date(2025, 6, 14), "portfolio_value": Decimal("99000"),
                 "cash": Decimal("49000"), "buying_power": Decimal("49000")},
                {"date": date(2025, 6, 15), "portfolio_value": Decimal("100000"),
                 "cash": Decimal("50000"), "buying_power": Decimal("50000")},
            ],
            # positions
            [
                {"ticker": "AAPL", "shares": Decimal("10"),
                 "avg_cost": Decimal("150.00"), "updated_at": datetime(2025, 6, 15)},
            ],
            # decisions
            [
                {"id": 1, "date": date(2025, 6, 15), "ticker": "AAPL", "action": "buy",
                 "quantity": Decimal("5"), "price": Decimal("150.00"),
                 "reasoning": "Strong momentum", "outcome_7d": Decimal("2.5"),
                 "outcome_30d": None, "order_id": "abc123"},
            ],
            # theses
            [
                {"id": 1, "ticker": "AAPL", "direction": "long", "confidence": "high",
                 "thesis": "Growth story", "entry_trigger": "Above 150",
                 "exit_trigger": "Above 180", "created_at": datetime(2025, 6, 10)},
            ],
        ]

        # Set up fetchone side_effect for sequential calls:
        # 5. latest snapshot, 6. first snapshot, 7. previous snapshot
        mock_db.fetchone.side_effect = [
            # latest snapshot
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"),
             "long_market_value": Decimal("50000")},
            # first snapshot
            {"portfolio_value": Decimal("90000"), "date": date(2025, 1, 1)},
            # previous snapshot
            {"portfolio_value": Decimal("99000")},
        ]

        result = gather_dashboard_data(session_date)

        # Verify all top-level keys present
        assert set(result.keys()) == {"summary", "snapshots", "positions", "decisions", "theses"}

        # Verify snapshots
        assert len(result["snapshots"]) == 2
        assert result["snapshots"][0]["date"] == date(2025, 6, 14)

        # Verify positions
        assert len(result["positions"]) == 1
        assert result["positions"][0]["ticker"] == "AAPL"

        # Verify decisions
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["action"] == "buy"

        # Verify theses
        assert len(result["theses"]) == 1
        assert result["theses"][0]["direction"] == "long"

        # Verify summary
        summary = result["summary"]
        assert summary["portfolio_value"] == Decimal("100000")
        assert summary["cash"] == Decimal("50000")
        assert summary["invested"] == Decimal("50000")
        assert summary["positions_count"] == 1
        assert summary["last_updated"] == "2025-06-15"
        assert summary["inception_date"] == date(2025, 1, 1)

        # Daily P&L: 100000 - 99000 = 1000
        assert summary["daily_pnl"] == Decimal("1000")
        assert float(summary["daily_pnl_pct"]) > 0

        # Total P&L: 100000 - 90000 = 10000
        assert summary["total_pnl"] == Decimal("10000")
        assert float(summary["total_pnl_pct"]) > 0

    def test_empty_database(self, mock_db):
        """Handles empty DB gracefully with empty lists and minimal summary."""
        session_date = date(2025, 6, 15)

        # All fetchall calls return empty
        mock_db.fetchall.side_effect = [[], [], [], []]

        # All fetchone calls return None
        mock_db.fetchone.side_effect = [None, None, None]

        result = gather_dashboard_data(session_date)

        assert result["snapshots"] == []
        assert result["positions"] == []
        assert result["decisions"] == []
        assert result["theses"] == []

        summary = result["summary"]
        assert summary["portfolio_value"] == 0
        assert summary["cash"] == 0
        assert summary["invested"] == 0
        assert summary["positions_count"] == 0
        assert summary["daily_pnl"] == 0
        assert summary["total_pnl"] == 0
        assert summary["inception_date"] is None
        assert summary["last_updated"] == "2025-06-15"

    def test_no_previous_snapshot(self, mock_db):
        """First day of trading: latest exists but no previous snapshot."""
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [
            [{"date": date(2025, 6, 15), "portfolio_value": Decimal("100000"),
              "cash": Decimal("100000"), "buying_power": Decimal("100000")}],
            [],  # no positions
            [],  # no decisions
            [],  # no theses
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("100000"),
             "long_market_value": None},
            {"portfolio_value": Decimal("100000"), "date": date(2025, 6, 15)},
            None,  # no previous snapshot
        ]

        result = gather_dashboard_data(session_date)
        summary = result["summary"]

        assert summary["portfolio_value"] == Decimal("100000")
        assert summary["daily_pnl"] == Decimal("0")
        assert summary["daily_pnl_pct"] == Decimal("0")
        assert summary["total_pnl"] == Decimal("0")
        assert summary["invested"] == Decimal("0")

    def test_json_serializable_with_encoder(self, mock_db):
        """Full result is JSON-serializable via _DecimalEncoder."""
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [
            [{"date": date(2025, 6, 15), "portfolio_value": Decimal("100000"),
              "cash": Decimal("50000"), "buying_power": Decimal("50000")}],
            [{"ticker": "AAPL", "shares": Decimal("10"),
              "avg_cost": Decimal("150.00"), "updated_at": datetime(2025, 6, 15)}],
            [],
            [],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"),
             "long_market_value": Decimal("50000")},
            {"portfolio_value": Decimal("95000"), "date": date(2025, 1, 1)},
            {"portfolio_value": Decimal("99000")},
        ]

        result = gather_dashboard_data(session_date)

        # Should not raise
        output = json.dumps(result, cls=_DecimalEncoder)
        parsed = json.loads(output)
        assert parsed["summary"]["portfolio_value"] == 100000.0
        assert parsed["positions"][0]["ticker"] == "AAPL"

    def test_query_count(self, mock_db):
        """Verifies exactly 7 queries are executed (4 fetchall + 3 fetchone)."""
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [[], [], [], []]
        mock_db.fetchone.side_effect = [None, None, None]

        gather_dashboard_data(session_date)

        assert mock_db.execute.call_count == 7
