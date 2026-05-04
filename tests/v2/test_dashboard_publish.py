"""Tests for dashboard data gathering module."""

import json
import os
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.dashboard_publish import (
    _build_summary,
    _DecimalEncoder,
    _enrich_snapshots_with_deposits,
    _enrich_snapshots_with_twr_value,
    _redact_order_id,
    assemble_deploy_dir,
    deploy_to_cloudflare,
    fetch_spy_benchmark,
    gather_dashboard_data,
    gather_trade_detail,
    run_dashboard_stage,
    write_json_files,
)
from v2.executor import get_net_deposits


class TestRedactOrderId:
    """P1.8: full Alpaca UUIDs must not appear in published decisions.json."""

    def test_full_uuid_truncated(self):
        full = "6f1eef85-83f0-4b6a-b8e6-37cba3d3318e"
        assert _redact_order_id(full) == "6f1eef85..."

    def test_short_id_unchanged(self):
        # Short ids stay as-is (also matches frontend `shortOrderId` boundary at len > 12).
        assert _redact_order_id("abc123") == "abc123"

    def test_none_passes_through(self):
        assert _redact_order_id(None) is None


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


@patch("v2.dashboard_publish.fetch_spy_benchmark", return_value=[])
class TestGatherDashboardData:
    def test_returns_all_sections(self, mock_benchmark, mock_db):
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
        assert set(result.keys()) == {"summary", "snapshots", "positions", "decisions", "theses", "benchmark"}

        # Verify snapshots
        assert len(result["snapshots"]) == 2
        assert result["snapshots"][0]["date"] == date(2025, 6, 14)

        # Verify positions
        assert len(result["positions"]) == 1
        assert result["positions"][0]["ticker"] == "AAPL"

        # Verify decisions
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["action"] == "buy"
        # P1.8: order_id stays short here (input is "abc123") — full UUID
        # truncation is exercised in test_decisions_redact_full_order_uuid.
        assert result["decisions"][0]["order_id"] == "abc123"

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

        # Total P&L: fallback to first snapshot (100000 - 90000 = 10000)
        assert summary["total_pnl"] == Decimal("10000")
        assert float(summary["total_pnl_pct"]) > 0

    def test_total_return_uses_net_deposits(self, mock_benchmark, mock_db):
        """When net_deposits provided, total return = portfolio_value - net_deposits."""
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [[], [], [], []]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("105000"), "cash": Decimal("50000"),
             "long_market_value": Decimal("55000")},
            {"portfolio_value": Decimal("80000"), "date": date(2025, 1, 1)},
            {"portfolio_value": Decimal("104000")},
        ]

        # Deposited 100k total (initial 80k + 20k infusion)
        result = gather_dashboard_data(session_date, net_deposits=Decimal("100000"))
        summary = result["summary"]

        # Investment return: 105000 - 100000 = 5000 (not 105000 - 80000 = 25000)
        assert summary["total_pnl"] == Decimal("5000")
        assert summary["total_pnl_pct"] == Decimal("5")  # 5000/100000 * 100

    def test_total_return_fallback_without_net_deposits(self, mock_benchmark, mock_db):
        """Without net_deposits, falls back to first snapshot comparison."""
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [[], [], [], []]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("105000"), "cash": Decimal("50000"),
             "long_market_value": Decimal("55000")},
            {"portfolio_value": Decimal("80000"), "date": date(2025, 1, 1)},
            {"portfolio_value": Decimal("104000")},
        ]

        result = gather_dashboard_data(session_date)
        summary = result["summary"]

        # Fallback: 105000 - 80000 = 25000
        assert summary["total_pnl"] == Decimal("25000")

    def test_empty_database(self, mock_benchmark, mock_db):
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

        assert result["benchmark"] == []
        mock_benchmark.assert_not_called()

    def test_no_previous_snapshot(self, mock_benchmark, mock_db):
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

    def test_json_serializable_with_encoder(self, mock_benchmark, mock_db):
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

    def test_query_count(self, mock_benchmark, mock_db):
        """Verifies exactly 7 queries are executed (4 fetchall + 3 fetchone)."""
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [[], [], [], []]
        mock_db.fetchone.side_effect = [None, None, None]

        gather_dashboard_data(session_date)

        assert mock_db.execute.call_count == 7

    @patch("v2.dashboard_publish.get_deposit_history")
    def test_daily_pnl_excludes_same_day_deposit_end_to_end(self, mock_history, mock_benchmark, mock_db):
        """Reproduces 2026-04-24: a $5000 deposit arrives between prev and latest.

        The summary card must show trading P&L only, not deposit + trading.
        """
        session_date = date(2026, 4, 24)
        mock_history.return_value = [
            {"date": "2026-02-05", "amount": Decimal("1000")},
            {"date": "2026-02-11", "amount": Decimal("1000")},
            {"date": "2026-03-18", "amount": Decimal("1000")},
            {"date": "2026-04-23", "amount": Decimal("5000")},
        ]
        mock_db.fetchall.side_effect = [
            [
                {"date": date(2026, 4, 23), "portfolio_value": Decimal("2939.40"),
                 "cash": Decimal("1917.42"), "buying_power": Decimal("1917.42")},
                {"date": date(2026, 4, 24), "portfolio_value": Decimal("7964.33"),
                 "cash": Decimal("6810.20"), "buying_power": Decimal("6810.20")},
            ],
            [], [], [],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("7964.33"), "cash": Decimal("6810.20"),
             "long_market_value": Decimal("1154.13")},
            {"portfolio_value": Decimal("1000"), "date": date(2026, 2, 5)},
            {"portfolio_value": Decimal("2939.40")},
        ]

        result = gather_dashboard_data(session_date, net_deposits=Decimal("8000"))
        summary = result["summary"]

        # Without the deposit adjustment this would be $5024.93 (+170%).
        # With it: $7964.33 - $2939.40 - $5000 = $24.93, ~0.31%.
        assert summary["daily_pnl"] == Decimal("24.93")
        assert Decimal("0.3") < summary["daily_pnl_pct"] < Decimal("0.4")

        # Equity chart: snapshots carry twr_value for deposit-neutral plotting.
        snaps = result["snapshots"]
        assert snaps[0]["twr_value"] == snaps[0]["portfolio_value"]
        # Day-2 twr_value reflects only trading gain, not the $5000 deposit.
        assert Decimal("2945") < snaps[1]["twr_value"] < Decimal("2955")

    def test_decisions_redact_full_order_uuid(self, mock_benchmark, mock_db):
        """P1.8: full Alpaca UUIDs in decision rows must be truncated in output."""
        session_date = date(2025, 6, 15)
        full_uuid = "6f1eef85-83f0-4b6a-b8e6-37cba3d3318e"
        mock_db.fetchall.side_effect = [
            [],  # snapshots
            [],  # positions
            [{"id": 1, "date": session_date, "ticker": "AAPL", "action": "buy",
              "quantity": Decimal("5"), "price": Decimal("150"),
              "reasoning": "x", "outcome_7d": None, "outcome_30d": None,
              "order_id": full_uuid}],
            [],  # theses
        ]
        mock_db.fetchone.side_effect = [None, None, None]

        result = gather_dashboard_data(session_date)

        assert result["decisions"][0]["order_id"] == "6f1eef85..."
        # Full UUID never appears anywhere in the published payload.
        assert full_uuid not in json.dumps(result, cls=_DecimalEncoder)

    def test_includes_benchmark_key(self, mock_benchmark, mock_db):
        """gather_dashboard_data includes 'benchmark' key from fetch_spy_benchmark."""
        mock_benchmark.return_value = [{"date": "2025-06-15", "close": 540.0}]
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [
            [{"date": date(2025, 6, 14), "portfolio_value": Decimal("99000"),
              "cash": Decimal("49000"), "buying_power": Decimal("49000")},
             {"date": date(2025, 6, 15), "portfolio_value": Decimal("100000"),
              "cash": Decimal("50000"), "buying_power": Decimal("50000")}],
            [], [], [],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"),
             "long_market_value": Decimal("50000")},
            {"portfolio_value": Decimal("90000"), "date": date(2025, 1, 1)},
            {"portfolio_value": Decimal("99000")},
        ]

        result = gather_dashboard_data(session_date)

        assert "benchmark" in result
        assert result["benchmark"] == [{"date": "2025-06-15", "close": 540.0}]
        mock_benchmark.assert_called_once_with(date(2025, 6, 14), date(2025, 6, 15))


class TestBuildSummary:
    def test_net_deposits_used_over_first_snapshot(self):
        """net_deposits takes precedence over first snapshot for total return."""
        latest = {"portfolio_value": Decimal("110000"), "cash": Decimal("40000"),
                  "long_market_value": Decimal("70000")}
        first = {"portfolio_value": Decimal("50000"), "date": date(2025, 1, 1)}
        previous = {"portfolio_value": Decimal("109000")}

        summary = _build_summary(latest, first, previous, 3, date(2025, 6, 15),
                                 net_deposits=Decimal("100000"))

        # Investment return: 110000 - 100000 = 10000 (10%)
        assert summary["total_pnl"] == Decimal("10000")
        assert summary["total_pnl_pct"] == Decimal("10")
        # inception_date still comes from first snapshot
        assert summary["inception_date"] == date(2025, 1, 1)

    def test_zero_net_deposits_uses_fallback(self):
        """Zero net_deposits triggers fallback to first snapshot."""
        latest = {"portfolio_value": Decimal("110000"), "cash": Decimal("40000"),
                  "long_market_value": Decimal("70000")}
        first = {"portfolio_value": Decimal("100000"), "date": date(2025, 1, 1)}
        previous = {"portfolio_value": Decimal("109000")}

        summary = _build_summary(latest, first, previous, 3, date(2025, 6, 15),
                                 net_deposits=Decimal("0"))

        # Fallback: 110000 - 100000 = 10000
        assert summary["total_pnl"] == Decimal("10000")

    def test_daily_pnl_excludes_same_day_deposit(self):
        """daily_deposit is subtracted from the raw portfolio delta.

        Reproduces 2026-04-24 incident: a $5000 deposit landed between the
        previous snapshot ($2939.40) and latest ($7964.33). Raw delta shows
        +$5024.93 (+170%) which is mostly cash-in, not trading P&L.
        """
        latest = {"portfolio_value": Decimal("7964.33"), "cash": Decimal("6810.20"),
                  "long_market_value": Decimal("1154.13")}
        first = {"portfolio_value": Decimal("1000"), "date": date(2026, 2, 5)}
        previous = {"portfolio_value": Decimal("2939.40")}

        summary = _build_summary(latest, first, previous, 4, date(2026, 4, 24),
                                 net_deposits=Decimal("8000"),
                                 daily_deposit=Decimal("5000"))

        # Daily P&L excludes the deposit: 7964.33 - 2939.40 - 5000 = 24.93
        assert summary["daily_pnl"] == Decimal("24.93")
        # Pct uses (prev + deposit) as base: 24.93 / 7939.40 ≈ 0.314%
        assert Decimal("0.3") < summary["daily_pnl_pct"] < Decimal("0.4")

    def test_daily_pnl_default_daily_deposit_is_zero(self):
        """daily_deposit defaults to 0 — no change for deposit-free days."""
        latest = {"portfolio_value": Decimal("101000"), "cash": Decimal("50000"),
                  "long_market_value": Decimal("51000")}
        first = {"portfolio_value": Decimal("100000"), "date": date(2025, 1, 1)}
        previous = {"portfolio_value": Decimal("100000")}

        summary = _build_summary(latest, first, previous, 3, date(2025, 6, 15))

        assert summary["daily_pnl"] == Decimal("1000")


class TestEnrichSnapshotsWithDeposits:
    """`_enrich_snapshots_with_deposits` credits a deposit dated D to
    snapshots dated strictly after D. The fallback adds deposits dated
    on-or-before the first snapshot — but only the first snapshot was
    actually missing them. Earlier code added the fallback credit to
    *every* snapshot, double-counting cumulative_deposits for the entire
    series and depressing TWR/Total Return %."""

    def test_no_fallback_double_credit_for_later_snapshots(self):
        """P3.28: deposit dated == first_snap_date.

        First loop credits it to snapshots > first_snap_date (correct).
        Fallback credits it to snapshot[0] (also correct).
        Bug was: fallback also credited it to snapshots[1+], inflating
        cumulative_deposits by exactly one extra deposit cycle.
        """
        snapshots = [
            {"date": date(2026, 1, 1), "portfolio_value": Decimal("1000")},
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("1010")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("1020")},
        ]
        deposits = [{"date": "2026-01-01", "amount": Decimal("1000")}]

        _enrich_snapshots_with_deposits(snapshots, deposits)

        # Correct: first snapshot includes the same-day deposit (so
        # cumulative_deposits matches portfolio_value); later snapshots
        # see exactly the same $1000, not double.
        assert snapshots[0]["cumulative_deposits"] == Decimal("1000")
        assert snapshots[1]["cumulative_deposits"] == Decimal("1000"), (
            f"Expected $1000, got {snapshots[1]['cumulative_deposits']} — fallback double-credited"
        )
        assert snapshots[2]["cumulative_deposits"] == Decimal("1000"), (
            f"Expected $1000, got {snapshots[2]['cumulative_deposits']} — fallback double-credited"
        )

    def test_fallback_credits_only_first_when_multiple_pre_first_deposits(self):
        """Two deposits dated before the first snapshot — both should
        roll into the first snapshot's cumulative; later snapshots
        should reflect that same total exactly once."""
        snapshots = [
            {"date": date(2026, 1, 5), "portfolio_value": Decimal("1500")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("1510")},
        ]
        deposits = [
            {"date": "2026-01-01", "amount": Decimal("1000")},
            {"date": "2026-01-03", "amount": Decimal("500")},
        ]

        _enrich_snapshots_with_deposits(snapshots, deposits)

        assert snapshots[0]["cumulative_deposits"] == Decimal("1500")
        assert snapshots[1]["cumulative_deposits"] == Decimal("1500"), (
            f"Got {snapshots[1]['cumulative_deposits']} — should not be doubled"
        )

    def test_no_fallback_when_first_snapshot_already_credited(self):
        """If the first deposit is dated *after* the first snapshot, the
        fallback should not fire and cumulative_deposits should reflect
        the first-loop result exactly."""
        snapshots = [
            {"date": date(2026, 1, 1), "portfolio_value": Decimal("1000")},
            {"date": date(2026, 1, 5), "portfolio_value": Decimal("2010")},
        ]
        deposits = [{"date": "2026-01-03", "amount": Decimal("1000")}]

        _enrich_snapshots_with_deposits(snapshots, deposits)

        # First snapshot precedes any deposit → cum stays 0, fallback
        # rule doesn't apply (first_dep_date > first_snap_date).
        assert snapshots[0]["cumulative_deposits"] == Decimal("0")
        assert snapshots[1]["cumulative_deposits"] == Decimal("1000")


class TestEnrichSnapshotsWithTwrValue:
    """twr_value is the first snapshot's value compounded by daily TWR growth.

    Gives the JS an already-normalized equity line that sits on the same
    dollar axis as SPY without deposit cliffs.
    """

    def test_empty_list_is_noop(self):
        snapshots: list[dict] = []
        _enrich_snapshots_with_twr_value(snapshots)
        assert snapshots == []

    def test_first_snapshot_twr_value_equals_portfolio_value(self):
        snapshots = [
            {"date": date(2025, 1, 1), "portfolio_value": Decimal("1000"),
             "cumulative_deposits": Decimal("1000")},
        ]
        _enrich_snapshots_with_twr_value(snapshots)
        assert snapshots[0]["twr_value"] == Decimal("1000")

    def test_no_deposits_growth_matches_portfolio_value(self):
        """With no deposits, twr_value tracks portfolio_value exactly."""
        snapshots = [
            {"date": date(2025, 1, 1), "portfolio_value": Decimal("1000"),
             "cumulative_deposits": Decimal("1000")},
            {"date": date(2025, 1, 2), "portfolio_value": Decimal("1100"),
             "cumulative_deposits": Decimal("1000")},
        ]
        _enrich_snapshots_with_twr_value(snapshots)
        assert snapshots[1]["twr_value"] == Decimal("1100")

    def test_deposit_does_not_inflate_twr_value(self):
        """2026-04-24 scenario: $5000 deposit makes portfolio jump $2939→$7964,
        but twr_value should show only the real ~0.3% trading gain."""
        snapshots = [
            {"date": date(2026, 4, 23), "portfolio_value": Decimal("2939.40"),
             "cumulative_deposits": Decimal("3000")},
            {"date": date(2026, 4, 24), "portfolio_value": Decimal("7964.33"),
             "cumulative_deposits": Decimal("8000")},
        ]
        _enrich_snapshots_with_twr_value(snapshots)

        # Day-1 twr_value = 2939.40 (starts at portfolio_value)
        # Day-2 growth factor = 7964.33 / (2939.40 + 5000) ≈ 1.003140
        # Day-2 twr_value ≈ 2939.40 * 1.003140 ≈ 2948.63
        assert snapshots[0]["twr_value"] == Decimal("2939.40")
        assert Decimal("2945") < snapshots[1]["twr_value"] < Decimal("2955")

    def test_withdrawal_does_not_deflate_twr_value(self):
        """Withdrawals decrease cumulative_deposits; twr_value should be
        unaffected by the withdrawal itself (only by trading P&L)."""
        snapshots = [
            {"date": date(2025, 1, 1), "portfolio_value": Decimal("10000"),
             "cumulative_deposits": Decimal("10000")},
            # Withdraw $2000; portfolio_value drops to $8000, same trading result
            {"date": date(2025, 1, 2), "portfolio_value": Decimal("8000"),
             "cumulative_deposits": Decimal("8000")},
        ]
        _enrich_snapshots_with_twr_value(snapshots)
        # growth = 8000 / (10000 - 2000) = 1.0 → no change
        assert snapshots[1]["twr_value"] == Decimal("10000")

    def test_zero_start_capital_is_skipped(self):
        """Guard: don't divide by zero if prev_value + deposit == 0."""
        snapshots = [
            {"date": date(2025, 1, 1), "portfolio_value": Decimal("0"),
             "cumulative_deposits": Decimal("0")},
            {"date": date(2025, 1, 2), "portfolio_value": Decimal("100"),
             "cumulative_deposits": Decimal("100")},
        ]
        _enrich_snapshots_with_twr_value(snapshots)
        # First day's twr_value = 0 (portfolio_value). Second day seeded from that.
        assert snapshots[0]["twr_value"] == Decimal("0")
        # When start capital is 0, we just carry previous twr_value forward.
        assert snapshots[1]["twr_value"] == Decimal("0")


class TestWriteJsonFiles:
    def _sample_data(self):
        return {
            "summary": {"portfolio_value": 100000, "cash": 50000},
            "snapshots": [{"date": "2025-06-15", "value": 100000}],
            "positions": [{"ticker": "AAPL", "shares": 10}],
            "decisions": [{"action": "buy", "ticker": "AAPL"}],
            "theses": [{"ticker": "AAPL", "direction": "long"}],
        }

    def test_writes_all_files(self, tmp_path):
        """All 6 JSON files are written with correct content."""
        data = self._sample_data()
        data["benchmark"] = [{"date": "2025-06-15", "close": 540.0}]
        result = write_json_files(data, str(tmp_path))

        assert len(result) == 6
        for key in ("summary", "snapshots", "positions", "decisions", "theses", "benchmark"):
            file_path = tmp_path / "data" / f"{key}.json"
            assert file_path.exists()
            with open(file_path) as f:
                content = json.load(f)
            assert content == data[key]

    def test_creates_data_dir_if_missing(self, tmp_path):
        """data/ directory is created automatically."""
        data = self._sample_data()
        data_dir = tmp_path / "data"
        assert not data_dir.exists()

        write_json_files(data, str(tmp_path))

        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_uses_decimal_encoder(self, tmp_path):
        """Decimal values are serialized as floats."""
        data = {
            "summary": {"portfolio_value": Decimal("100000.50")},
            "snapshots": [],
            "positions": [],
            "decisions": [],
            "theses": [],
        }

        write_json_files(data, str(tmp_path))

        with open(tmp_path / "data" / "summary.json") as f:
            content = json.load(f)
        assert content["portfolio_value"] == 100000.50
        assert isinstance(content["portfolio_value"], float)

    def test_writes_benchmark_file(self, tmp_path):
        """benchmark.json is written when benchmark key present."""
        data = self._sample_data()
        data["benchmark"] = [{"date": "2025-06-15", "close": 540.0}]
        result = write_json_files(data, str(tmp_path))

        benchmark_path = tmp_path / "data" / "benchmark.json"
        assert benchmark_path.exists()
        with open(benchmark_path) as f:
            content = json.load(f)
        assert content == [{"date": "2025-06-15", "close": 540.0}]


class TestRunDashboardStage:
    @patch("v2.dashboard_publish.deploy_to_cloudflare", return_value=True)
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_happy_path(self, mock_deposits, mock_gather, mock_assemble, mock_deploy):
        """Full pipeline runs and returns published=True."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage(session_date=date(2025, 6, 15))

        assert result.published is True
        assert result.skipped is False
        assert result.errors == []
        mock_deposits.assert_called_once()
        mock_gather.assert_called_once_with(date(2025, 6, 15), net_deposits=Decimal("100000"))
        mock_assemble.assert_called_once()
        mock_deploy.assert_called_once()

    def test_skipped_when_no_project_set(self):
        """Returns skipped=True when CLOUDFLARE_PAGES_PROJECT not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLOUDFLARE_PAGES_PROJECT", None)
            result = run_dashboard_stage()

        assert result.skipped is True
        assert result.published is False
        assert result.errors == []

    @patch("v2.dashboard_publish.gather_dashboard_data", side_effect=Exception("DB down"))
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_gather_error(self, mock_deposits, mock_gather):
        """Error in gather step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Data gathering failed" in result.errors[0]

    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.assemble_deploy_dir", side_effect=Exception("Disk full"))
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_assemble_error(self, mock_deposits, mock_assemble, mock_gather):
        """Error in assemble step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Deploy assembly failed" in result.errors[0]

    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.deploy_to_cloudflare", side_effect=RuntimeError("Auth failed"))
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_deploy_error(self, mock_deposits, mock_deploy, mock_assemble, mock_gather):
        """Error in deploy step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Cloudflare deploy failed" in result.errors[0]

    @patch("v2.dashboard_publish.deploy_to_cloudflare", return_value=True)
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.get_net_deposits", side_effect=Exception("Alpaca down"))
    def test_continues_when_net_deposits_fails(self, mock_deposits, mock_gather, mock_assemble, mock_deploy):
        """Pipeline continues with net_deposits=None if Alpaca call fails."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage(session_date=date(2025, 6, 15))

        assert result.published is True
        assert result.errors == []
        mock_gather.assert_called_once_with(date(2025, 6, 15), net_deposits=None)


class TestGetNetDeposits:
    @patch("v2.executor.get_trading_client")
    def test_sums_deposits_and_withdrawals(self, mock_get_client):
        """Sums CSD (positive) and CSW (negative) activities."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = [
            {"net_amount": 50000.0, "id": "1"},
            {"net_amount": 50000.0, "id": "2"},
            {"net_amount": -5000.0, "id": "3"},
        ]

        result = get_net_deposits()

        assert result == Decimal("95000")
        mock_client.get.assert_called_once_with(
            "/account/activities",
            {"activity_types": "CSD,CSW", "page_size": 100, "direction": "asc"},
        )

    @patch("v2.executor.get_trading_client")
    def test_handles_empty_activities(self, mock_get_client):
        """Returns zero when no transfer activities exist."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = []

        result = get_net_deposits()

        assert result == Decimal("0")

    @patch("v2.executor.get_trading_client")
    def test_paginates_when_needed(self, mock_get_client):
        """Fetches multiple pages when activities exceed page_size."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First page: 100 items (triggers pagination)
        page1 = [{"net_amount": 1000.0, "id": str(i)} for i in range(100)]
        # Second page: fewer than 100 items (last page)
        page2 = [{"net_amount": 2000.0, "id": "200"}]

        mock_client.get.side_effect = [page1, page2]

        result = get_net_deposits()

        assert result == Decimal("102000")  # 100 * 1000 + 1 * 2000
        assert mock_client.get.call_count == 2
        # Second call should include page_token from last item of first page
        second_call_params = mock_client.get.call_args_list[1][0][1]
        assert second_call_params["page_token"] == "99"

    @patch("v2.executor.get_trading_client")
    def test_handles_none_response(self, mock_get_client):
        """Returns zero when API returns None."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = None

        result = get_net_deposits()

        assert result == Decimal("0")


class TestDeployToCloudflare:
    @patch("v2.dashboard_publish.subprocess.run")
    def test_deploys_successfully(self, mock_run):
        """Runs wrangler pages deploy with correct args."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Published!")
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dashboard"}):
            result = deploy_to_cloudflare("/tmp/deploy")

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "wrangler"
        assert call_args[1:3] == ["pages", "deploy"]
        assert "/tmp/deploy" in call_args
        assert "--project-name" in call_args
        assert "my-dashboard" in call_args

    @patch("v2.dashboard_publish.subprocess.run")
    def test_raises_on_wrangler_failure(self, mock_run):
        """RuntimeError raised when wrangler exits non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Auth failed")
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dashboard"}), pytest.raises(RuntimeError, match="Auth failed"):
            deploy_to_cloudflare("/tmp/deploy")

    def test_raises_when_project_not_set(self):
        """RuntimeError raised when CLOUDFLARE_PAGES_PROJECT missing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLOUDFLARE_PAGES_PROJECT", None)
            with pytest.raises(RuntimeError, match="CLOUDFLARE_PAGES_PROJECT"):
                deploy_to_cloudflare("/tmp/deploy")

    @patch("v2.dashboard_publish.subprocess.run")
    def test_passes_timeout_to_subprocess(self, mock_run):
        """P3.39: a hung wrangler must not block the session forever.
        subprocess.run should be invoked with a bounded timeout so the
        stage fails loudly instead of stalling indefinitely."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Published!")
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dashboard"}):
            deploy_to_cloudflare("/tmp/deploy")

        kwargs = mock_run.call_args.kwargs
        assert kwargs.get("timeout") is not None, (
            "subprocess.run must be called with a bounded timeout — a hung "
            "wrangler would otherwise block the session forever"
        )
        assert kwargs["timeout"] >= 60, (
            f"timeout={kwargs['timeout']} is too tight for a real deploy"
        )

    @patch("v2.dashboard_publish.subprocess.run")
    def test_raises_on_subprocess_timeout(self, mock_run):
        """If wrangler hangs past the timeout, surface a RuntimeError with
        a clear message — don't propagate the raw TimeoutExpired so the
        session log says what actually happened."""
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="wrangler", timeout=300)
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dashboard"}):
            with pytest.raises(RuntimeError, match="time"):
                deploy_to_cloudflare("/tmp/deploy")


class TestAssembleDeployDir:
    def _sample_data(self):
        return {
            "summary": {"portfolio_value": 100000},
            "snapshots": [{"date": "2025-06-15"}],
            "positions": [{"ticker": "AAPL"}],
            "decisions": [{"action": "buy"}],
            "theses": [{"direction": "long"}],
        }

    def test_copies_static_assets(self, tmp_path):
        """index.html, styles.css, app.js are copied to deploy dir."""
        # Create fake static assets
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
        (assets_dir / "index.html").write_text("<html>test</html>")
        (assets_dir / "styles.css").write_text("body { color: red; }")
        (assets_dir / "app.js").write_text("console.log('hi');")
        (assets_dir / "README.md").write_text("Docs")  # Should NOT be copied

        deploy_dir = tmp_path / "deploy"
        assemble_deploy_dir(self._sample_data(), str(deploy_dir), str(assets_dir))

        assert (deploy_dir / "index.html").exists()
        assert (deploy_dir / "styles.css").exists()
        assert (deploy_dir / "app.js").exists()
        assert not (deploy_dir / "README.md").exists()

    def test_writes_json_data_files(self, tmp_path):
        """data/*.json files are written correctly."""
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
        (assets_dir / "index.html").write_text("<html>")
        (assets_dir / "styles.css").write_text("")
        (assets_dir / "app.js").write_text("")

        deploy_dir = tmp_path / "deploy"
        data = self._sample_data()
        assemble_deploy_dir(data, str(deploy_dir), str(assets_dir))

        for key in ("summary", "snapshots", "positions", "decisions", "theses"):
            json_path = deploy_dir / "data" / f"{key}.json"
            assert json_path.exists()
            with open(json_path) as f:
                assert json.load(f) == data[key]

    def test_creates_deploy_dir_if_missing(self, tmp_path):
        """Deploy directory is created automatically."""
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
        (assets_dir / "index.html").write_text("<html>")
        (assets_dir / "styles.css").write_text("")
        (assets_dir / "app.js").write_text("")

        deploy_dir = tmp_path / "deploy" / "nested"
        assert not deploy_dir.exists()

        assemble_deploy_dir(self._sample_data(), str(deploy_dir), str(assets_dir))

        assert deploy_dir.exists()


class TestGatherTradeDetail:
    def test_returns_decision_thesis_position(self, mock_db):
        mock_db.fetchone.side_effect = [
            # decision row
            {"id": 42, "date": date(2026, 5, 3), "ticker": "NVDA", "action": "buy",
             "quantity": Decimal("12"), "price": Decimal("450.25"),
             "reasoning": "AI capex", "outcome_7d": None, "outcome_30d": None,
             "thesis_id": 7, "order_id": "abc12345-...-uuid"},
            # thesis row
            {"id": 7, "ticker": "NVDA", "direction": "long", "thesis": "AI capex",
             "entry_trigger": "<$440", "exit_trigger": "$520", "invalidation": "no",
             "confidence": "high", "status": "active"},
            # position row (may be None if closed)
            {"ticker": "NVDA", "shares": Decimal("12"), "avg_cost": Decimal("450.25")},
        ]

        result = gather_trade_detail(mock_db, decision_id=42)

        assert result["decision"]["id"] == 42
        assert result["thesis"]["id"] == 7
        assert result["position"]["ticker"] == "NVDA"
        # Order ID truncated even on detail page
        assert "..." in result["decision"]["order_id"]

    def test_returns_none_when_decision_missing(self, mock_db):
        mock_db.fetchone.side_effect = [None]
        result = gather_trade_detail(mock_db, decision_id=999)
        assert result is None

    def test_no_thesis_when_thesis_id_null(self, mock_db):
        mock_db.fetchone.side_effect = [
            {"id": 42, "date": date(2026, 5, 3), "ticker": "NVDA", "action": "buy",
             "quantity": Decimal("12"), "price": Decimal("450.25"),
             "reasoning": "x", "outcome_7d": None, "outcome_30d": None,
             "thesis_id": None, "order_id": None},
            None,  # position lookup
        ]
        result = gather_trade_detail(mock_db, decision_id=42)
        assert result["thesis"] is None


class TestFetchSpyBenchmark:
    @patch("v2.dashboard_publish.StockHistoricalDataClient")
    def test_returns_spy_bars_for_date_range(self, mock_client_cls):
        """Fetches SPY daily bars and returns [{date, close}, ...]."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Simulate Alpaca bar objects
        bar1 = MagicMock()
        bar1.timestamp = datetime(2025, 6, 14, 4, 0)
        bar1.close = 540.50
        bar2 = MagicMock()
        bar2.timestamp = datetime(2025, 6, 15, 4, 0)
        bar2.close = 542.00

        mock_bars = MagicMock()
        mock_bars.__getitem__ = MagicMock(return_value=[bar1, bar2])
        mock_client.get_stock_bars.return_value = mock_bars

        result = fetch_spy_benchmark(date(2025, 6, 14), date(2025, 6, 15))

        assert len(result) == 2
        assert result[0] == {"date": "2025-06-14", "close": 540.50}
        assert result[1] == {"date": "2025-06-15", "close": 542.00}

    @patch("v2.dashboard_publish.StockHistoricalDataClient")
    def test_returns_empty_list_on_api_error(self, mock_client_cls):
        """Returns [] if Alpaca API call fails."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_stock_bars.side_effect = Exception("API down")

        result = fetch_spy_benchmark(date(2025, 6, 14), date(2025, 6, 15))

        assert result == []

    @patch("v2.dashboard_publish.StockHistoricalDataClient")
    def test_returns_empty_list_when_no_bars(self, mock_client_cls):
        """Returns [] if no bars returned."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_bars = MagicMock()
        mock_bars.__getitem__ = MagicMock(return_value=[])
        mock_client.get_stock_bars.return_value = mock_bars

        result = fetch_spy_benchmark(date(2025, 6, 14), date(2025, 6, 15))

        assert result == []
