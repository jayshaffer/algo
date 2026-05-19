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
    _enrich_snapshots_with_spy_match,
    _enrich_snapshots_with_twr_value,
    _redact_order_id,
    assemble_deploy_dir,
    deploy_to_cloudflare,
    fetch_spy_benchmark,
    gather_all_pages_data,
    gather_dashboard_data,
    gather_thesis_detail,
    gather_trade_detail,
    generate_changelog_entries,
    get_changelog_pointer,
    get_current_git_sha,
    get_recent_changelog_entries,
    group_changelog_commits,
    persist_changelog_pointer,
    read_changelog_pointer,
    run_dashboard_stage,
    store_changelog_commits,
    store_changelog_entries,
    summarize_changelog_commits,
    update_changelog_pointer,
    validate_changelog_entries,
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
        # 1. snapshots, 2. positions, 3. decisions, 4. theses,
        # 5. all-decision-ids, 6. all-thesis-ids
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
            # all-decision-ids (gather_all_pages_data)
            [],
            # all-thesis-ids (gather_all_pages_data)
            [],
            # get_closed_losers
            [],
            # get_retired_rules
            [],
            # get_signal_attribution
            [],
            # get_recent_strategy_memos
            [],
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
        assert set(result.keys()) == {
            "summary", "snapshots", "positions", "decisions", "theses",
            "benchmark", "mistakes", "attribution", "memos", "performance",
            "_pages",
        }

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

        mock_db.fetchall.side_effect = [[], [], [], [], [], []]
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

    def test_summary_aliases_total_return_pct(self, mock_benchmark, mock_db):
        """Summary exposes total_return_pct as an alias of total_pnl_pct.

        The new dashboard pages read `total_return_pct` (per spec); the
        legacy publisher emitted `total_pnl_pct`. Both keys must be present
        and equal so the hero "all time" stat renders.
        """
        session_date = date(2025, 6, 15)
        mock_db.fetchall.side_effect = [[], [], [], [], [], []]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("105000"), "cash": Decimal("50000"),
             "long_market_value": Decimal("55000")},
            {"portfolio_value": Decimal("80000"), "date": date(2025, 1, 1)},
            {"portfolio_value": Decimal("104000")},
        ]

        result = gather_dashboard_data(session_date, net_deposits=Decimal("100000"))
        summary = result["summary"]

        assert "total_return_pct" in summary
        assert summary["total_return_pct"] == summary["total_pnl_pct"]

    @patch("v2.dashboard_publish.get_deposit_history")
    def test_summary_includes_vs_spy_pct(self, mock_history, mock_benchmark, mock_db):
        """vs_spy_pct = portfolio total return − deposit-matched SPY return.

        Setup: $1000 deposited on 2025-06-14 when SPY=$500 → 2 SPY shares.
        Latest portfolio = $1100 (+10%). SPY closes at $550 on 2025-06-15
        → shadow value 2 × $550 = $1100, also +10%. Spread = 0%.
        Bumping portfolio to $1200 (+20%) yields vs_spy_pct = +10%.
        """
        session_date = date(2025, 6, 15)
        mock_history.return_value = [
            {"date": "2025-06-14", "amount": Decimal("1000")},
        ]
        mock_benchmark.return_value = [
            {"date": "2025-06-14", "close": 500.0},
            {"date": "2025-06-15", "close": 550.0},
        ]
        mock_db.fetchall.side_effect = [
            [
                {"date": date(2025, 6, 14), "portfolio_value": Decimal("1000"),
                 "cash": Decimal("1000"), "buying_power": Decimal("1000")},
                {"date": date(2025, 6, 15), "portfolio_value": Decimal("1200"),
                 "cash": Decimal("1200"), "buying_power": Decimal("1200")},
            ],
            [], [], [], [], [],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("1200"), "cash": Decimal("1200"),
             "long_market_value": Decimal("0")},
            {"portfolio_value": Decimal("1000"), "date": date(2025, 6, 14)},
            {"portfolio_value": Decimal("1000")},
        ]

        result = gather_dashboard_data(session_date, net_deposits=Decimal("1000"))
        summary = result["summary"]

        # Portfolio: (1200 - 1000)/1000 = +20%
        # SPY shadow: 2 shares × $550 = $1100 → (1100 - 1000)/1000 = +10%
        # Spread: +10%
        assert summary["vs_spy_pct"] is not None
        assert abs(float(summary["vs_spy_pct"]) - 10.0) < 0.01

    def test_summary_vs_spy_pct_none_when_no_benchmark(self, mock_benchmark, mock_db):
        """vs_spy_pct is None when SPY benchmark fetch returns nothing.

        Without SPY closes the shadow portfolio can't be priced, so the
        renderer must show '—' rather than fabricate a comparison.
        """
        session_date = date(2025, 6, 15)
        mock_benchmark.return_value = []
        mock_db.fetchall.side_effect = [
            [{"date": date(2025, 6, 15), "portfolio_value": Decimal("1100"),
              "cash": Decimal("1100"), "buying_power": Decimal("1100")}],
            [], [], [], [], [],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("1100"), "cash": Decimal("1100"),
             "long_market_value": Decimal("0")},
            {"portfolio_value": Decimal("1000"), "date": date(2025, 6, 15)},
            None,
        ]

        result = gather_dashboard_data(session_date, net_deposits=Decimal("1000"))

        assert result["summary"]["vs_spy_pct"] is None

    def test_total_return_fallback_without_net_deposits(self, mock_benchmark, mock_db):
        """Without net_deposits, falls back to first snapshot comparison."""
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [[], [], [], [], [], []]
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
        mock_db.fetchall.side_effect = [[], [], [], [], [], []]

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
            [],  # all-decision-ids
            [],  # all-thesis-ids
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
            [],  # all-decision-ids
            [],  # all-thesis-ids
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
        """Verifies exactly 13 queries are executed.

        Breakdown: 4 fetchall (snapshots/positions/decisions/theses)
        + 3 fetchone (latest/first/previous snapshots)
        + 2 pages (all-decision-ids/all-thesis-ids)
        + 4 new helpers (get_closed_losers/get_retired_rules/
          get_signal_attribution/get_recent_strategy_memos)
        = 13 total
        """
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [[], [], [], [], [], [], [], [], [], []]
        mock_db.fetchone.side_effect = [None, None, None]

        gather_dashboard_data(session_date)

        assert mock_db.execute.call_count == 13

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
            [], [], [], [], [],
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
            [],  # all-decision-ids
            [],  # all-thesis-ids
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
            [], [], [], [], [],
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


class TestEnrichSnapshotsWithSpyMatch:
    """`spy_value_if_deposited` mirrors the cumulative_deposits crediting rule:
    a deposit becomes part of the shadow SPY portfolio on the same snapshot
    where it lands in cumulative_deposits, valued at the SPY close on the
    deposit's own date."""

    def test_empty_snapshots_is_noop(self):
        snapshots: list[dict] = []
        _enrich_snapshots_with_spy_match(snapshots, [], [])
        assert snapshots == []

    def test_no_deposits_or_benchmark_sets_none(self):
        snapshots = [{"date": date(2025, 1, 1), "portfolio_value": Decimal("1000")}]
        _enrich_snapshots_with_spy_match(snapshots, [], [])
        assert snapshots[0]["spy_value_if_deposited"] is None

    def test_single_deposit_grows_with_spy(self):
        """Deposit on day 1 buys SPY at $400; on day 2 SPY = $440 → +10%."""
        snapshots = [
            {"date": date(2025, 1, 1), "portfolio_value": Decimal("1000")},
            {"date": date(2025, 1, 2), "portfolio_value": Decimal("1050")},
        ]
        deposits = [{"date": "2025-01-01", "amount": Decimal("1000")}]
        benchmark = [
            {"date": date(2025, 1, 1), "close": 400.0},
            {"date": date(2025, 1, 2), "close": 440.0},
        ]
        _enrich_snapshots_with_spy_match(snapshots, deposits, benchmark)
        # Day 1 (== first_snap_date): deposit credited under fix-up rule
        # shares = 1000 / 400 = 2.5; value = 2.5 * 400 = 1000
        assert snapshots[0]["spy_value_if_deposited"] == 1000.00
        # Day 2: same shares, new SPY price → 2.5 * 440 = 1100
        assert snapshots[1]["spy_value_if_deposited"] == 1100.00

    def test_mid_window_deposit_credited_after_its_date(self):
        """Mirror of cumulative_deposits crediting: deposit dated D shows up
        on snapshot D+1 (or first snapshot strictly after D)."""
        snapshots = [
            {"date": date(2025, 1, 1), "portfolio_value": Decimal("1000")},
            {"date": date(2025, 1, 2), "portfolio_value": Decimal("1050")},
            {"date": date(2025, 1, 3), "portfolio_value": Decimal("2100")},
        ]
        deposits = [
            {"date": "2025-01-01", "amount": Decimal("1000")},
            {"date": "2025-01-02", "amount": Decimal("1000")},
        ]
        benchmark = [
            {"date": date(2025, 1, 1), "close": 400.0},
            {"date": date(2025, 1, 2), "close": 410.0},
            {"date": date(2025, 1, 3), "close": 420.0},
        ]
        _enrich_snapshots_with_spy_match(snapshots, deposits, benchmark)

        # Day 1: only the Jan-1 deposit credited (fix-up). 1000/400 * 400 = 1000.
        assert snapshots[0]["spy_value_if_deposited"] == 1000.00
        # Day 2: still only Jan-1 deposit (Jan-2 NOT < Jan-2). 2.5 * 410 = 1025.
        assert snapshots[1]["spy_value_if_deposited"] == 1025.00
        # Day 3: both deposits credited. 2.5 * 420 + (1000/410) * 420 ≈ 1050 + 1024.39
        v = snapshots[2]["spy_value_if_deposited"]
        assert 2070 < v < 2080

    def test_falls_back_to_prior_close_when_snapshot_date_not_a_trading_day(self):
        """Snapshot dates may include non-trading days (holiday/weekend writes)
        that have no SPY close — fall back to the most recent prior close."""
        snapshots = [
            {"date": date(2025, 1, 2), "portfolio_value": Decimal("1000")},
            {"date": date(2025, 1, 4), "portfolio_value": Decimal("1100")},  # Saturday
        ]
        deposits = [{"date": "2025-01-02", "amount": Decimal("1000")}]
        benchmark = [
            {"date": date(2025, 1, 2), "close": 400.0},
            {"date": date(2025, 1, 3), "close": 410.0},
        ]
        _enrich_snapshots_with_spy_match(snapshots, deposits, benchmark)
        # Day 2 (Sat): no SPY close — fall back to Jan-3 (Fri). 2.5 * 410 = 1025.
        assert snapshots[1]["spy_value_if_deposited"] == 1025.00


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
    @patch("v2.dashboard_publish.persist_changelog_pointer")
    @patch("v2.dashboard_publish.get_recent_changelog_entries",
           return_value=[{"date": "2026-05-15", "title": "Change",
                          "summary": "Readable update", "commit_shas": ["abc1234"]}])
    @patch("v2.dashboard_publish.store_changelog_entries")
    @patch("v2.dashboard_publish.summarize_changelog_commits",
           return_value=[{"title": "Change", "summary": "Readable update",
                          "bullets": [], "commit_shas": ["abc1234"]}])
    @patch("v2.dashboard_publish.store_changelog_commits")
    @patch("v2.dashboard_publish.fetch_changelog_commits",
           return_value=[{"sha": "abc1234", "short_sha": "abc1234",
                          "committed_at": "2026-05-15T00:00:00+00:00",
                          "subject": "Change"}])
    @patch("v2.dashboard_publish.get_changelog_pointer", return_value="oldsha")
    @patch("v2.dashboard_publish.get_cursor")
    @patch("v2.dashboard_publish.get_current_git_sha", return_value="newsha")
    @patch("v2.dashboard_publish.deploy_to_cloudflare", return_value=True)
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_happy_path(self, mock_deposits, mock_gather, mock_assemble,
                        mock_deploy, mock_sha, mock_cursor, mock_pointer,
                        mock_fetch, mock_store_commits, mock_summarize,
                        mock_store_entries, mock_recent, mock_persist):
        """Full pipeline runs and returns published=True."""
        cur = MagicMock()
        mock_cursor.return_value.__enter__.return_value = cur

        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage(session_date=date(2025, 6, 15))

        assert result.published is True
        assert result.skipped is False
        assert result.errors == []
        mock_deposits.assert_called_once()
        mock_gather.assert_called_once_with(date(2025, 6, 15), net_deposits=Decimal("100000"))
        mock_assemble.assert_called_once()
        mock_deploy.assert_called_once()
        mock_sha.assert_called_once()
        mock_pointer.assert_called_once_with(cur)
        mock_fetch.assert_called_once_with(from_sha="oldsha", to_sha="newsha")
        mock_store_commits.assert_called_once_with(cur, mock_fetch.return_value)
        mock_summarize.assert_called_once_with(mock_fetch.return_value)
        mock_store_entries.assert_called_once_with(
            cur,
            mock_summarize.return_value,
            from_sha="oldsha",
            to_sha="newsha",
            model="claude-haiku-4-5",
        )
        mock_recent.assert_called_once_with(cur)
        assert mock_assemble.call_args.args[0]["changelog"] == [
            {"date": "2026-05-15", "title": "Change",
             "summary": "Readable update", "commit_shas": ["abc1234"]}
        ]
        mock_persist.assert_called_once_with("newsha")

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
    @patch("v2.dashboard_publish.get_current_git_sha", return_value=None)
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_assemble_error(self, mock_deposits, mock_sha,
                                    mock_assemble, mock_gather):
        """Error in assemble step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Deploy assembly failed" in result.errors[0]

    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.deploy_to_cloudflare", side_effect=RuntimeError("Auth failed"))
    @patch("v2.dashboard_publish.get_current_git_sha", return_value=None)
    @patch("v2.dashboard_publish.get_net_deposits", return_value=Decimal("100000"))
    def test_handles_deploy_error(self, mock_deposits, mock_sha, mock_deploy,
                                  mock_assemble, mock_gather):
        """Error in deploy step is captured."""
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dash"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Cloudflare deploy failed" in result.errors[0]

    @patch("v2.dashboard_publish.deploy_to_cloudflare", return_value=True)
    @patch("v2.dashboard_publish.assemble_deploy_dir", return_value="/tmp/deploy")
    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.get_current_git_sha", return_value=None)
    @patch("v2.dashboard_publish.get_net_deposits", side_effect=Exception("Alpaca down"))
    def test_continues_when_net_deposits_fails(self, mock_deposits, mock_sha,
                                               mock_gather, mock_assemble,
                                               mock_deploy):
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
        with patch.dict(os.environ, {"CLOUDFLARE_PAGES_PROJECT": "my-dashboard"}), pytest.raises(RuntimeError, match="time"):
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
        """styles.css and app.js are copied to deploy dir."""
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
        (assets_dir / "styles.css").write_text("body { color: red; }")
        (assets_dir / "app.js").write_text("console.log('hi');")
        (assets_dir / "README.md").write_text("Docs")  # Should NOT be copied

        deploy_dir = tmp_path / "deploy"
        assemble_deploy_dir(self._sample_data(), str(deploy_dir), str(assets_dir))

        assert (deploy_dir / "styles.css").exists()
        assert (deploy_dir / "app.js").exists()
        assert not (deploy_dir / "README.md").exists()
        # index.html is now produced by emit_homepage, only when base_url is set
        assert not (deploy_dir / "index.html").exists()

    def test_writes_json_data_files(self, tmp_path):
        """data/*.json files are written correctly."""
        assets_dir = tmp_path / "public_dashboard"
        assets_dir.mkdir()
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
        (assets_dir / "styles.css").write_text("")
        (assets_dir / "app.js").write_text("")

        deploy_dir = tmp_path / "deploy" / "nested"
        assert not deploy_dir.exists()

        assemble_deploy_dir(self._sample_data(), str(deploy_dir), str(assets_dir))

        assert deploy_dir.exists()


class TestAssembleDeployDirEndToEnd:
    def _stub_trade_detail(self, decision_id):
        return {"decision": {
            "id": decision_id, "date": date(2026, 5, 3), "ticker": "NVDA",
            "action": "buy", "quantity": Decimal("1"), "price": Decimal("100"),
            "reasoning": "x",
            "outcome_7d": None, "outcome_30d": None, "thesis_id": None,
            "order_id": None,
        }, "thesis": None, "position": None}

    def _stub_thesis_detail(self, thesis_id):
        return {"thesis": {
            "id": thesis_id, "ticker": "NVDA", "direction": "long",
            "confidence": "high", "thesis": "x",
            "entry_trigger": None, "exit_trigger": None, "invalidation": None,
            "status": "active",
        }, "decisions": [], "position": None}

    def test_emits_static_assets_json_pages_and_og(self, mock_db, tmp_path):
        from unittest.mock import patch as _patch

        from v2.dashboard_publish import assemble_deploy_dir

        # Set up a fake assets dir with the static files assemble_deploy_dir copies.
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "styles.css").write_text("body{}")
        (assets / "app.js").write_text("// app")

        deploy = tmp_path / "deploy"

        data = {
            "summary": {"portfolio_value": 100, "daily_pnl": 0,
                        "daily_pnl_pct": 0, "last_updated": "2026-05-03"},
            "snapshots": [], "positions": [], "decisions": [], "theses": [],
            "benchmark": [],
            "_pages": {"decision_ids": [1], "thesis_ids": [7]},
        }

        with _patch("v2.dashboard_publish.gather_trade_detail",
                    side_effect=lambda cur, did: self._stub_trade_detail(did)), \
             _patch("v2.dashboard_publish.gather_thesis_detail",
                    side_effect=lambda cur, tid: self._stub_thesis_detail(tid)):
            assemble_deploy_dir(
                data, deploy_dir=str(deploy), assets_dir=str(assets),
                base_url="https://example.com",
            )

        # Static assets present
        assert (deploy / "index.html").is_file()
        assert (deploy / "styles.css").is_file()
        # JSON files
        assert (deploy / "data" / "summary.json").is_file()
        # Per-trade and per-thesis HTML
        assert (deploy / "trade" / "1" / "index.html").is_file()
        assert (deploy / "thesis" / "7" / "index.html").is_file()
        # OG images
        assert (deploy / "og" / "trade" / "1.png").is_file()
        assert (deploy / "og" / "thesis" / "7.png").is_file()
        # Homepage now rendered by emit_homepage with full meta block
        assert '<meta property="og:title"' in (deploy / "index.html").read_text()
        # Homepage OG image is emitted unconditionally
        assert (deploy / "og" / "home.png").is_file()
        assert (deploy / "og" / "home.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


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

    def test_signal_refs_hydrated_from_decision_signals(self, mock_db):
        """gather_trade_detail should hydrate news/macro/thesis citations
        with parent fields and tag each row with its signal_type."""
        mock_db.fetchone.side_effect = [
            {"id": 42, "date": date(2026, 5, 3), "ticker": "NVDA", "action": "buy",
             "quantity": Decimal("12"), "price": Decimal("450.25"),
             "reasoning": "x", "outcome_7d": None, "outcome_30d": None,
             "thesis_id": None, "order_id": None},
            None,  # position lookup
        ]
        mock_db.fetchall.side_effect = [
            [{"signal_id": 100, "ticker": "NVDA", "headline": "AI capex surge",
              "category": "earnings", "sentiment": "bullish",
              "published_at": datetime(2026, 5, 1, 14, 30)}],
            [{"signal_id": 200, "headline": "Fed pauses",
              "category": "rate_decision", "sentiment": "dovish",
              "affected_sectors": "tech,growth",
              "published_at": datetime(2026, 5, 2, 9, 0)}],
            [{"signal_id": 7, "ticker": "NVDA", "direction": "long",
              "thesis": "AI capex acceleration",
              "confidence": "high", "status": "active"}],
        ]

        result = gather_trade_detail(mock_db, decision_id=42)
        refs = result["signal_refs"]
        assert len(refs) == 3
        types = [r["signal_type"] for r in refs]
        assert types == ["news_signal", "macro_signal", "thesis"]
        assert refs[0]["headline"] == "AI capex surge"
        assert refs[1]["affected_sectors"] == "tech,growth"
        assert refs[2]["signal_id"] == 7


class TestGatherThesisDetail:
    def test_returns_thesis_with_decisions_and_position(self, mock_db):
        mock_db.fetchone.side_effect = [
            {"id": 7, "ticker": "NVDA", "direction": "long", "thesis": "AI",
             "entry_trigger": "<$440", "exit_trigger": "$520", "invalidation": "no",
             "confidence": "high", "status": "active"},
            {"ticker": "NVDA", "shares": Decimal("12"), "avg_cost": Decimal("450")},
        ]
        mock_db.fetchall.side_effect = [
            [
                {"id": 42, "date": date(2026, 5, 3), "ticker": "NVDA", "action": "buy",
                 "quantity": Decimal("12"), "price": Decimal("450.25"),
                 "outcome_7d": None, "outcome_30d": None},
            ],
            [],  # news signal refs
            [],  # macro signal refs
            [],  # thesis signal refs
        ]

        result = gather_thesis_detail(mock_db, thesis_id=7)
        assert result["thesis"]["id"] == 7
        assert len(result["decisions"]) == 1
        assert result["position"]["ticker"] == "NVDA"
        assert result["signal_refs"] == []

    def test_returns_none_when_missing(self, mock_db):
        mock_db.fetchone.side_effect = [None]
        result = gather_thesis_detail(mock_db, thesis_id=999)
        assert result is None

    def test_signal_refs_hydrated_from_thesis_signals(self, mock_db):
        """gather_thesis_detail should hydrate citations from thesis_signals."""
        mock_db.fetchone.side_effect = [
            {"id": 7, "ticker": "NVDA", "direction": "long", "thesis": "AI",
             "entry_trigger": "<$440", "exit_trigger": "$520", "invalidation": "no",
             "confidence": "high", "status": "active"},
            None,  # position
        ]
        mock_db.fetchall.side_effect = [
            [],  # decisions
            [{"signal_id": 100, "ticker": "NVDA", "headline": "AI capex surge",
              "category": "earnings", "sentiment": "bullish",
              "published_at": datetime(2026, 5, 1, 14, 30)}],
            [],  # macro
            [],  # thesis citations
        ]

        result = gather_thesis_detail(mock_db, thesis_id=7)
        refs = result["signal_refs"]
        assert len(refs) == 1
        assert refs[0]["signal_type"] == "news_signal"
        assert refs[0]["headline"] == "AI capex surge"


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




class TestGatherAllPagesData:
    def test_returns_all_decision_and_thesis_ids(self, mock_db):
        # Even decisions outside the homepage 30-day window must be returned —
        # link permanence is a hard requirement (Cloudflare full-bundle replace).
        mock_db.fetchall.side_effect = [
            [{"id": 1}, {"id": 42}, {"id": 99}],   # decisions
            [{"id": 7}, {"id": 8}],                 # theses
        ]

        result = gather_all_pages_data(mock_db)

        assert result["decision_ids"] == [1, 42, 99]
        assert result["thesis_ids"] == [7, 8]
        assert result["ticker_symbols"] == []

    def test_includes_closed_theses_not_just_active(self, mock_db):
        mock_db.fetchall.side_effect = [
            [{"id": 1}],
            # All statuses returned — caller doesn't filter by status.
            [{"id": 7}, {"id": 8}, {"id": 9}],
        ]
        result = gather_all_pages_data(mock_db)
        assert result["thesis_ids"] == [7, 8, 9]

    def test_empty_db_returns_empty_lists(self, mock_db):
        mock_db.fetchall.side_effect = [[], []]
        result = gather_all_pages_data(mock_db)
        assert result == {"decision_ids": [], "thesis_ids": [], "ticker_symbols": []}


class TestEmitOgImages:
    def test_writes_png_files(self, mock_db, tmp_path):
        from unittest.mock import patch as _patch

        from v2.dashboard_publish import emit_og_images

        def _stub_trade(cur, did):
            return {"decision": {
                "id": did, "ticker": "NVDA", "action": "buy",
                "quantity": Decimal("12"), "price": Decimal("450"),
                "date": date(2026, 5, 3),
            }, "thesis": None, "position": None}

        def _stub_thesis(cur, tid):
            return {"thesis": {
                "id": tid, "ticker": "NVDA", "direction": "long",
                "confidence": "high", "thesis": "x",
            }, "decisions": [], "position": None}

        with _patch("v2.dashboard_publish.gather_trade_detail", side_effect=_stub_trade), \
             _patch("v2.dashboard_publish.gather_thesis_detail", side_effect=_stub_thesis):
            stats = emit_og_images(
                mock_db,
                decision_ids=[1],
                thesis_ids=[7],
                deploy_dir=str(tmp_path),
            )

        trade_png = tmp_path / "og" / "trade" / "1.png"
        thesis_png = tmp_path / "og" / "thesis" / "7.png"
        assert trade_png.is_file()
        assert thesis_png.is_file()
        assert trade_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert stats["trades_written"] == 1
        assert stats["theses_written"] == 1

    def test_isolates_per_image_failures(self, mock_db, tmp_path):
        from unittest.mock import patch as _patch

        from v2.dashboard_publish import emit_og_images

        def trade_side(cur, did):
            if did == 2:
                raise RuntimeError("boom")
            return {"decision": {
                "id": did, "ticker": "NVDA", "action": "buy",
                "quantity": Decimal("12"), "price": Decimal("450"),
                "date": date(2026, 5, 3),
            }, "thesis": None, "position": None}

        def thesis_side(cur, tid):
            return {"thesis": {
                "id": tid, "ticker": "NVDA", "direction": "long",
                "confidence": "high", "thesis": "x",
            }, "decisions": [], "position": None}

        with _patch("v2.dashboard_publish.gather_trade_detail", side_effect=trade_side), \
             _patch("v2.dashboard_publish.gather_thesis_detail", side_effect=thesis_side):
            stats = emit_og_images(
                mock_db,
                decision_ids=[1, 2, 3],
                thesis_ids=[7],
                deploy_dir=str(tmp_path),
            )

        assert (tmp_path / "og" / "trade" / "1.png").is_file()
        assert not (tmp_path / "og" / "trade" / "2.png").exists()
        assert (tmp_path / "og" / "trade" / "3.png").is_file()
        assert (tmp_path / "og" / "thesis" / "7.png").is_file()
        assert stats["trades_written"] == 2
        assert stats["theses_written"] == 1
        assert stats["failed"] == 1


class TestEmitDetailPages:
    def _stub_trade_detail(self, decision_id):
        return {
            "decision": {
                "id": decision_id, "date": date(2026, 5, 3), "ticker": "NVDA",
                "action": "buy", "quantity": Decimal("12"),
                "price": Decimal("450"), "reasoning": "x",
                "outcome_7d": None, "outcome_30d": None, "thesis_id": None,
                "order_id": None,
            },
            "thesis": None,
            "position": None,
        }

    def _stub_thesis_detail(self, thesis_id):
        return {
            "thesis": {
                "id": thesis_id, "ticker": "NVDA", "direction": "long",
                "thesis": "AI capex", "entry_trigger": None,
                "exit_trigger": None, "invalidation": None,
                "confidence": "high", "status": "active",
            },
            "decisions": [],
            "position": None,
        }

    def test_emits_one_html_per_trade_and_thesis(self, mock_db, tmp_path):
        from unittest.mock import patch as _patch

        from v2.dashboard_publish import emit_detail_pages

        with _patch("v2.dashboard_publish.gather_trade_detail",
                    side_effect=lambda cur, did: self._stub_trade_detail(did)), \
             _patch("v2.dashboard_publish.gather_thesis_detail",
                    side_effect=lambda cur, tid: self._stub_thesis_detail(tid)), \
             _patch("v2.dashboard_publish.gather_ticker_detail",
                    return_value={
                        "ticker": "NVDA",
                        "decisions": [{"id": 1, "date": date(2026, 5, 4),
                                       "action": "buy", "quantity": 1,
                                       "price": Decimal("100"),
                                       "reasoning": "x"}],
                        "theses": [],
                        "position": None,
                    }):
            stats = emit_detail_pages(
                mock_db,
                decision_ids=[1, 2],
                thesis_ids=[7],
                deploy_dir=str(tmp_path),
                base_url="https://example.com",
                ticker_symbols=["NVDA"],
            )

        assert (tmp_path / "trade" / "1" / "index.html").is_file()
        assert (tmp_path / "trade" / "2" / "index.html").is_file()
        assert (tmp_path / "thesis" / "7" / "index.html").is_file()
        assert (tmp_path / "ticker" / "NVDA" / "index.html").is_file()
        assert stats["trades_written"] == 2
        assert stats["theses_written"] == 1
        assert stats["tickers_written"] == 1
        assert stats["failed"] == 0

    def test_isolates_per_page_failures(self, mock_db, tmp_path):
        from unittest.mock import patch as _patch

        from v2.dashboard_publish import emit_detail_pages

        def trade_side_effect(cur, did):
            if did == 2:
                raise RuntimeError("simulated render failure")
            return self._stub_trade_detail(did)

        with _patch("v2.dashboard_publish.gather_trade_detail",
                    side_effect=trade_side_effect), \
             _patch("v2.dashboard_publish.gather_thesis_detail",
                    side_effect=lambda cur, tid: self._stub_thesis_detail(tid)):
            stats = emit_detail_pages(
                mock_db,
                decision_ids=[1, 2, 3],
                thesis_ids=[7],
                deploy_dir=str(tmp_path),
                base_url="https://example.com",
            )

        # 1 and 3 succeed; 2 fails but doesn't abort the run.
        assert (tmp_path / "trade" / "1" / "index.html").is_file()
        assert not (tmp_path / "trade" / "2" / "index.html").exists()
        assert (tmp_path / "trade" / "3" / "index.html").is_file()
        assert (tmp_path / "thesis" / "7" / "index.html").is_file()
        assert stats["trades_written"] == 2
        assert stats["theses_written"] == 1
        assert stats["failed"] == 1


class TestGatherDashboardDataMistakesAttribution:
    def test_includes_mistakes_and_attribution_keys(self, mock_db, mock_cursor):
        from datetime import date

        # gather_dashboard_data executes many cursor calls in sequence.
        # We stub the new helpers via patch since they live in trading_db
        # and are imported into dashboard_publish.
        from unittest.mock import patch

        from v2.dashboard_publish import gather_dashboard_data
        with patch("v2.dashboard_publish.get_closed_losers", return_value=[
                    {"id": 1, "ticker": "TSLA", "outcome_30d": -12.0}]), \
             patch("v2.dashboard_publish.get_retired_rules", return_value=[
                    {"id": 1, "rule_text": "X"}]), \
             patch("v2.dashboard_publish.get_signal_attribution", return_value=[
                    {"category": "earnings", "sample_size": 20,
                     "avg_outcome_30d": 1.2}]), \
             patch("v2.dashboard_publish.fetch_spy_benchmark", return_value=[]):

            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = None

            data = gather_dashboard_data(date(2026, 5, 4))

        assert "mistakes" in data
        assert data["mistakes"]["closed_losers"][0]["ticker"] == "TSLA"
        assert data["mistakes"]["retired_rules"][0]["rule_text"] == "X"
        assert "attribution" in data
        assert data["attribution"][0]["category"] == "earnings"


class TestAssembleDeployDirNewPages:
    def _minimal_data(self):
        return {
            "summary": {"portfolio_value": Decimal("100"), "daily_pnl": Decimal("0"),
                        "daily_pnl_pct": Decimal("0"), "total_return_pct": Decimal("0"),
                        "vs_spy_pct": Decimal("0"), "day_number": 1,
                        "last_updated": "2026-05-04"},
            "snapshots": [], "positions": [], "decisions": [], "theses": [],
            "benchmark": [],
            "mistakes": {"closed_losers": [], "retired_rules": []},
            "memos": [], "attribution": [],
            "performance": {"max_drawdown_pct": 0, "win_rate_pct": 0,
                            "avg_days_held": 0, "best_day_pct": 0,
                            "worst_day_pct": 0},
        }

    def _assets(self, tmp_path):
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "styles.css").write_text("/* */")
        (assets / "app.js").write_text("// ")
        return assets

    def test_homepage_emitted(self, tmp_path):
        from v2.dashboard_publish import assemble_deploy_dir

        assets = self._assets(tmp_path)
        deploy = tmp_path / "deploy"
        assemble_deploy_dir(self._minimal_data(), str(deploy), str(assets),
                            base_url="https://example.com")

        index = deploy / "index.html"
        assert index.exists()
        html = index.read_text()
        assert 'data-page="home"' in html

    def test_new_pages_emitted(self, tmp_path):
        from v2.dashboard_publish import assemble_deploy_dir

        assets = self._assets(tmp_path)
        deploy = tmp_path / "deploy"
        data = self._minimal_data()
        data["changelog"] = [{
            "date": "2026-05-15",
            "title": "Repository updates",
            "summary": "1 commit published from git history.",
            "items": [{
                "sha": "abc1234",
                "short_sha": "abc1234",
                "subject": "Add changelog",
            }],
        }]
        assemble_deploy_dir(data, str(deploy), str(assets),
                            base_url="https://example.com")

        for path in ("strategy/index.html", "performance/index.html", "activity/index.html",
                     "changelog/index.html", "learning/index.html",
                     "how-it-works/index.html"):
            assert (deploy / path).exists(), f"missing: {path}"
        assert "Add changelog" in (deploy / "changelog" / "index.html").read_text()

    def test_memo_detail_pages_emitted(self, tmp_path):
        from v2.dashboard_publish import assemble_deploy_dir

        assets = self._assets(tmp_path)
        deploy = tmp_path / "deploy"
        data = self._minimal_data()
        data["memos"] = [{
            "id": 22,
            "session_date": "2026-05-04",
            "memo_type": "reflection",
            "content": "Full memo content",
        }]
        assemble_deploy_dir(data, str(deploy), str(assets),
                            base_url="https://example.com")

        memo_page = deploy / "memo" / "22" / "index.html"
        assert memo_page.exists()
        assert "Full memo content" in memo_page.read_text()

    def test_how_it_works_marks_unready_children(self, tmp_path):
        from v2.dashboard_publish import assemble_deploy_dir

        assets = self._assets(tmp_path)
        deploy = tmp_path / "deploy"
        assemble_deploy_dir(self._minimal_data(), str(deploy), str(assets),
                            base_url="https://example.com")

        html = (deploy / "how-it-works" / "index.html").read_text()
        # None of /about/, /internals/, /trace/ exist in this fixture deploy.
        assert html.count('class="card disabled"') == 3


class TestGenerateChangelogEntries:
    @patch("v2.dashboard_publish.fetch_commit_files", return_value=["v2/dashboard_publish.py"])
    @patch("v2.dashboard_publish.subprocess.run")
    def test_groups_recent_git_commits_by_date(self, mock_run, mock_files):
        mock_run.return_value.stdout = (
            "2026-05-15T12:00:00+00:00\x1fabc1234full\x1fabc1234\x1fAdd changelog\x1f\x1e"
            "2026-05-15T11:00:00+00:00\x1fdef5678full\x1fdef5678\x1fFix dashboard nav\x1f\x1e"
            "2026-05-14T10:00:00+00:00\x1f999aaaafull\x1f999aaaa\x1fValidate executor response schema\x1f"
        )

        entries = generate_changelog_entries(repo_path="/repo", bootstrap_limit=3)

        assert entries == [
            {
                "date": "2026-05-15",
                "title": "Repository updates",
                "summary": "2 commits published from git history.",
                "items": [
                    {"sha": "abc1234full", "short_sha": "abc1234",
                     "subject": "Add changelog"},
                    {"sha": "def5678full", "short_sha": "def5678",
                     "subject": "Fix dashboard nav"},
                ],
            },
            {
                "date": "2026-05-14",
                "title": "Repository updates",
                "summary": "1 commit published from git history.",
                "items": [
                    {"sha": "999aaaafull", "short_sha": "999aaaa",
                     "subject": "Validate executor response schema"},
                ],
            },
        ]
        mock_run.assert_called_once()
        args = mock_run.call_args.args[0]
        assert args[:4] == ["git", "-C", "/repo", "log"]
        assert "--no-merges" in args
        assert "-n3" in args
        assert "HEAD" in args

    @patch("v2.dashboard_publish.fetch_commit_files", return_value=[])
    @patch("v2.dashboard_publish.subprocess.run")
    def test_uses_sha_range_when_pointer_exists(self, mock_run, mock_files):
        mock_run.return_value.stdout = (
            "2026-05-15T12:00:00+00:00\x1fabc1234full\x1fabc1234\x1fAdd changelog\x1f"
        )

        entries = generate_changelog_entries(
            repo_path="/repo",
            from_sha="oldsha",
            to_sha="newsha",
        )

        assert entries[0]["items"] == [
            {"sha": "abc1234full", "short_sha": "abc1234", "subject": "Add changelog"}
        ]
        args = mock_run.call_args.args[0]
        assert "oldsha..newsha" in args
        assert not any(arg.startswith("-n") for arg in args)

    @patch("v2.dashboard_publish.subprocess.run")
    def test_returns_empty_list_when_pointer_matches_target(self, mock_run):
        assert generate_changelog_entries(from_sha="same", to_sha="same") == []
        mock_run.assert_not_called()

    @patch("v2.dashboard_publish.subprocess.run")
    def test_returns_empty_list_when_git_log_fails(self, mock_run):
        mock_run.side_effect = RuntimeError("git unavailable")

        assert generate_changelog_entries(repo_path="/repo") == []


class TestChangelogPointer:
    @patch("v2.dashboard_publish.subprocess.run")
    def test_get_current_git_sha(self, mock_run):
        mock_run.return_value.stdout = "abc123\n"

        assert get_current_git_sha(repo_path="/repo") == "abc123"
        mock_run.assert_called_once_with(
            ["git", "-C", "/repo", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

    @patch("v2.dashboard_publish.subprocess.run")
    def test_get_current_git_sha_returns_none_on_failure(self, mock_run):
        mock_run.side_effect = RuntimeError("git unavailable")

        assert get_current_git_sha(repo_path="/repo") is None

    def test_get_changelog_pointer_reads_state_row(self):
        cur = MagicMock()
        cur.fetchone.return_value = {"value": "abc123"}

        assert get_changelog_pointer(cur) == "abc123"
        cur.execute.assert_called_once_with(
            "SELECT value FROM dashboard_publish_state WHERE key = %s",
            ("changelog_last_published_sha",),
        )

    def test_get_changelog_pointer_returns_none_when_missing(self):
        cur = MagicMock()
        cur.fetchone.return_value = None

        assert get_changelog_pointer(cur) is None

    def test_update_changelog_pointer_upserts_state_row(self):
        cur = MagicMock()

        update_changelog_pointer(cur, "abc123")

        sql, params = cur.execute.call_args.args
        assert "INSERT INTO dashboard_publish_state" in sql
        assert "ON CONFLICT" in sql
        assert params == ("changelog_last_published_sha", "abc123")

    def test_store_changelog_commits_inserts_rows_idempotently(self):
        cur = MagicMock()

        store_changelog_commits(cur, [{
            "sha": "abc123full",
            "short_sha": "abc123",
            "committed_at": "2026-05-15T12:00:00+00:00",
            "subject": "Add changelog",
            "body": "Body",
            "files": ["v2/dashboard_publish.py"],
        }])

        sql, params = cur.execute.call_args.args
        assert "INSERT INTO dashboard_changelog_commits" in sql
        assert "ON CONFLICT (sha) DO NOTHING" in sql
        assert params == (
            "abc123full",
            "abc123",
            "2026-05-15T12:00:00+00:00",
            "Add changelog",
            "Body",
            '["v2/dashboard_publish.py"]',
        )

    def test_store_changelog_entries_inserts_llm_rows(self):
        cur = MagicMock()

        store_changelog_entries(
            cur,
            [{"title": "Safer execution", "summary": "Validation improved.",
              "bullets": ["Reject malformed instructions."],
              "commit_shas": ["abc123full"]}],
            from_sha="old",
            to_sha="new",
            model="model-x",
        )

        sql, params = cur.execute.call_args.args
        assert "INSERT INTO dashboard_changelog_entries" in sql
        assert params == (
            "old",
            "new",
            "Safer execution",
            "Validation improved.",
            '["Reject malformed instructions."]',
            '["abc123full"]',
            "model-x",
            "public_changelog_v1",
        )

    def test_get_recent_changelog_entries_reads_stored_rows(self):
        cur = MagicMock()
        cur.fetchall.side_effect = [[
            {"id": 1, "created_at": datetime(2026, 5, 16, 12, 0),
             "title": "Safer execution", "summary": "Validation improved.",
             "bullets": ["Reject malformed instructions."],
             "commit_shas": ["abc123full"]},
        ]]

        entries = get_recent_changelog_entries(cur, limit=10)

        assert entries[0]["date"] == "2026-05-16"
        assert entries[0]["title"] == "Safer execution"
        assert entries[0]["commit_shas"] == ["abc123full"]
        cur.execute.assert_called_once()
        assert cur.execute.call_args.args[1] == (10,)

    def test_get_recent_changelog_entries_falls_back_to_raw_rows(self):
        cur = MagicMock()
        cur.fetchall.side_effect = [[], [
            {"sha": "abc123full", "short_sha": "abc123",
             "committed_at": datetime(2026, 5, 15, 12, 0),
             "subject": "Add changelog"},
        ]]

        entries = get_recent_changelog_entries(cur, limit=10)

        assert entries[0]["date"] == "2026-05-15"
        assert entries[0]["items"][0]["short_sha"] == "abc123"
        assert cur.execute.call_count == 2
        assert cur.execute.call_args.args[1] == (10,)

    def test_group_changelog_commits_accepts_iso_strings(self):
        entries = group_changelog_commits([{
            "sha": "abc123full",
            "short_sha": "abc123",
            "committed_at": "2026-05-15T12:00:00+00:00",
            "subject": "Add changelog",
        }])

        assert entries[0]["date"] == "2026-05-15"

    def test_validate_changelog_entries_rejects_unknown_shas(self):
        entries = validate_changelog_entries(
            [{"title": "Good", "summary": "Useful", "bullets": ["A"],
              "commit_shas": ["known", "unknown"]}],
            commits=[{"sha": "known"}],
        )

        assert entries == [{
            "title": "Good",
            "summary": "Useful",
            "bullets": ["A"],
            "commit_shas": ["known"],
        }]

    def test_validate_changelog_entries_drops_entries_without_known_shas(self):
        entries = validate_changelog_entries(
            [{"title": "Bad", "summary": "No backing", "bullets": [],
              "commit_shas": ["unknown"]}],
            commits=[{"sha": "known"}],
        )

        assert entries == []

    @patch("v2.dashboard_publish.get_claude_client")
    @patch("v2.dashboard_publish._call_with_retry")
    def test_summarize_changelog_commits_validates_json(self, mock_call, mock_client):
        block = MagicMock()
        block.text = json.dumps({
            "entries": [{
                "title": "Safer execution",
                "summary": "Validation improved.",
                "bullets": ["Reject malformed instructions."],
                "commit_shas": ["abc123full"],
            }]
        })
        mock_call.return_value.content = [block]

        entries = summarize_changelog_commits([{
            "sha": "abc123full",
            "short_sha": "abc123",
            "committed_at": "2026-05-15T12:00:00+00:00",
            "subject": "Validate executor response schema",
            "body": "",
            "files": ["v2/executor.py"],
        }])

        assert entries[0]["title"] == "Safer execution"
        assert entries[0]["commit_shas"] == ["abc123full"]
        mock_call.assert_called_once()

    @patch("v2.dashboard_publish._call_with_retry", side_effect=RuntimeError("down"))
    def test_summarize_changelog_commits_falls_back_empty(self, mock_call):
        assert summarize_changelog_commits([{"sha": "abc123full"}]) == []

    @patch("v2.dashboard_publish.get_cursor")
    @patch("v2.dashboard_publish.get_changelog_pointer", return_value="abc123")
    def test_read_changelog_pointer_uses_cursor(self, mock_get_pointer, mock_cursor):
        cur = MagicMock()
        mock_cursor.return_value.__enter__.return_value = cur

        assert read_changelog_pointer() == "abc123"
        mock_get_pointer.assert_called_once_with(cur)

    @patch("v2.dashboard_publish.get_cursor")
    @patch("v2.dashboard_publish.update_changelog_pointer")
    def test_persist_changelog_pointer_uses_cursor(self, mock_update, mock_cursor):
        cur = MagicMock()
        mock_cursor.return_value.__enter__.return_value = cur

        persist_changelog_pointer("abc123")

        mock_update.assert_called_once_with(cur, "abc123")


class TestEmitStaticPages:
    def test_writes_mistakes_and_attribution_files(self, tmp_path):
        from decimal import Decimal

        from v2.dashboard_publish import emit_static_pages

        data = {
            "mistakes": {
                "closed_losers": [
                    {"id": 1, "date": "2026-04-30", "ticker": "TSLA",
                     "action": "buy", "quantity": 5, "price": 200,
                     "reasoning": "EV", "outcome_7d": Decimal("-3.0"),
                     "outcome_30d": Decimal("-12.0")},
                ],
                "retired_rules": [],
            },
            "attribution": [
                {"category": "earnings", "sample_size": 30, "sample_size_30d": 24,
                 "avg_outcome_7d": Decimal("1.2"), "avg_outcome_30d": Decimal("3.4"),
                 "win_rate_7d": Decimal("0.6"), "win_rate_30d": Decimal("0.5")},
            ],
        }

        emit_static_pages(data, str(tmp_path), base_url="https://example.com")

        mistakes_html = (tmp_path / "mistakes" / "index.html").read_text()
        assert "TSLA" in mistakes_html
        attribution_html = (tmp_path / "attribution" / "index.html").read_text()
        assert "earnings" in attribution_html

        mistakes_png = (tmp_path / "og" / "mistakes.png").read_bytes()
        assert mistakes_png[:8] == b"\x89PNG\r\n\x1a\n"
        attribution_png = (tmp_path / "og" / "attribution.png").read_bytes()
        assert attribution_png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_no_op_when_base_url_missing(self, tmp_path):
        from v2.dashboard_publish import emit_static_pages

        emit_static_pages({"mistakes": {"closed_losers": [], "retired_rules": []},
                           "attribution": []}, str(tmp_path), base_url="")

        # No files should have been written
        assert not (tmp_path / "mistakes").exists()
        assert not (tmp_path / "attribution").exists()


class TestGatherMemos:
    def test_gather_dashboard_data_includes_memos(self):
        from v2.dashboard_publish import gather_dashboard_data

        fake_memos = [
            {"id": 9, "session_date": date(2026, 5, 4),
             "memo_type": "session", "content": "Holding the AI book."},
            {"id": 8, "session_date": date(2026, 5, 3),
             "memo_type": "session", "content": "Macro chop unresolved."},
        ]
        with patch("v2.dashboard_publish.get_recent_strategy_memos",
                   return_value=fake_memos), \
             patch("v2.dashboard_publish.get_cursor"), \
             patch("v2.dashboard_publish.get_signal_attribution", return_value=[]), \
             patch("v2.dashboard_publish.get_closed_losers", return_value=[]), \
             patch("v2.dashboard_publish.get_retired_rules", return_value=[]), \
             patch("v2.dashboard_publish.get_net_deposits",
                   return_value=Decimal("0")), \
             patch("v2.dashboard_publish.fetch_spy_benchmark", return_value=[]):
            data = gather_dashboard_data(date(2026, 5, 4))

        assert "memos" in data
        assert len(data["memos"]) == 2
        assert data["memos"][0]["content"] == "Holding the AI book."

    def test_write_json_files_emits_memos(self, tmp_path):
        from v2.dashboard_publish import write_json_files

        data = {
            "memos": [{"id": 1, "session_date": "2026-05-04",
                       "content": "test"}],
            "summary": {}, "snapshots": [], "positions": [], "decisions": [],
            "theses": [], "benchmark": [], "mistakes": {},
            "attribution": [],
        }
        write_json_files(data, str(tmp_path))
        memos_file = tmp_path / "data" / "memos.json"
        assert memos_file.exists()
        with memos_file.open() as f:
            payload = json.load(f)
        assert payload[0]["content"] == "test"


class TestComputePerformanceStats:
    def test_empty_inputs_produce_zero_struct(self):
        from v2.dashboard_publish import compute_performance_stats

        stats = compute_performance_stats(snapshots=[], decisions=[])
        assert stats == {
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_days_held": 0.0,
            "best_day_pct": 0.0,
            "worst_day_pct": 0.0,
        }

    def test_max_drawdown_basic(self):
        from v2.dashboard_publish import compute_performance_stats

        snapshots = [
            {"snapshot_date": date(2026, 1, 1), "value": Decimal("100")},
            {"snapshot_date": date(2026, 1, 2), "value": Decimal("110")},
            {"snapshot_date": date(2026, 1, 3), "value": Decimal("90")},
            {"snapshot_date": date(2026, 1, 4), "value": Decimal("95")},
        ]
        stats = compute_performance_stats(snapshots=snapshots, decisions=[])
        assert abs(stats["max_drawdown_pct"] - (-18.181818)) < 0.01

    def test_best_and_worst_day(self):
        from v2.dashboard_publish import compute_performance_stats

        snapshots = [
            {"snapshot_date": date(2026, 1, 1), "value": Decimal("100")},
            {"snapshot_date": date(2026, 1, 2), "value": Decimal("105")},
            {"snapshot_date": date(2026, 1, 3), "value": Decimal("95")},
        ]
        stats = compute_performance_stats(snapshots=snapshots, decisions=[])
        assert abs(stats["best_day_pct"] - 5.0) < 0.01
        assert abs(stats["worst_day_pct"] - (-9.523809)) < 0.01

    def test_win_rate_from_closed_decisions(self):
        from v2.dashboard_publish import compute_performance_stats

        decisions = [
            {"outcome_30d_pct": Decimal("3.0")},
            {"outcome_30d_pct": Decimal("-2.0")},
            {"outcome_30d_pct": Decimal("1.0")},
            {"outcome_30d_pct": None},
        ]
        stats = compute_performance_stats(snapshots=[], decisions=decisions)
        assert abs(stats["win_rate_pct"] - 66.666666) < 0.01

    def test_write_json_files_emits_performance(self, tmp_path):
        from v2.dashboard_publish import write_json_files

        data = {
            "performance": {"max_drawdown_pct": -5.0, "win_rate_pct": 60.0,
                            "avg_days_held": 4.0, "best_day_pct": 2.0,
                            "worst_day_pct": -3.0},
            "summary": {}, "snapshots": [], "positions": [], "decisions": [],
            "theses": [], "benchmark": [], "mistakes": {}, "memos": [],
            "attribution": [],
        }
        write_json_files(data, str(tmp_path))
        f = tmp_path / "data" / "performance.json"
        assert f.exists()
        with f.open() as fh:
            payload = json.load(fh)
        assert payload["win_rate_pct"] == 60.0
