"""Tests for dashboard data gathering module."""

import json
import os
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.dashboard_publish import (
    DashboardStageResult,
    _DecimalEncoder,
    gather_dashboard_data,
    push_to_github,
    run_dashboard_stage,
    write_json_files,
)


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
        """All 5 JSON files are written with correct content."""
        data = self._sample_data()
        result = write_json_files(data, str(tmp_path))

        assert len(result) == 5
        for key in ("summary", "snapshots", "positions", "decisions", "theses"):
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


class TestPushToGithub:
    @patch("v2.dashboard_publish.subprocess.run")
    def test_commits_and_pushes(self, mock_run):
        """Verifies 3 git calls: add, commit, push."""
        mock_run.return_value = MagicMock(returncode=0)

        result = push_to_github("/fake/repo")

        assert result is True
        assert mock_run.call_count == 3

        # Verify the three calls
        calls = mock_run.call_args_list
        assert calls[0][0][0] == ["git", "add", "data/"]
        assert calls[0][1]["cwd"] == "/fake/repo"

        assert calls[1][0][0][0:2] == ["git", "commit"]
        assert calls[1][1]["cwd"] == "/fake/repo"

        assert calls[2][0][0] == ["git", "push"]
        assert calls[2][1]["cwd"] == "/fake/repo"

    @patch("v2.dashboard_publish.subprocess.run")
    def test_skips_push_if_nothing_to_commit(self, mock_run):
        """When commit returns non-zero, push is NOT called."""
        add_result = MagicMock(returncode=0)
        commit_result = MagicMock(returncode=1, stdout="nothing to commit")
        mock_run.side_effect = [add_result, commit_result]

        result = push_to_github("/fake/repo")

        assert result is False
        assert mock_run.call_count == 2  # add + commit, no push

    @patch("v2.dashboard_publish.subprocess.run")
    def test_raises_on_push_failure(self, mock_run):
        """RuntimeError raised when push fails."""
        add_result = MagicMock(returncode=0)
        commit_result = MagicMock(returncode=0)
        push_result = MagicMock(returncode=1, stderr="fatal: remote error")
        mock_run.side_effect = [add_result, commit_result, push_result]

        with pytest.raises(RuntimeError, match="fatal: remote error"):
            push_to_github("/fake/repo")


class TestRunDashboardStage:
    @patch("v2.dashboard_publish.push_to_github", return_value=True)
    @patch("v2.dashboard_publish.write_json_files", return_value=["/fake/data/summary.json"])
    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    def test_happy_path(self, mock_gather, mock_write, mock_push):
        """Full pipeline runs and returns published=True."""
        with patch.dict(os.environ, {"DASHBOARD_REPO_PATH": "/fake/repo"}):
            result = run_dashboard_stage(session_date=date(2025, 6, 15))

        assert result.published is True
        assert result.skipped is False
        assert result.errors == []
        mock_gather.assert_called_once_with(date(2025, 6, 15))
        mock_write.assert_called_once_with({"summary": {}}, "/fake/repo")
        mock_push.assert_called_once_with("/fake/repo")

    def test_skipped_when_no_repo_path(self):
        """Returns skipped=True when DASHBOARD_REPO_PATH not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure DASHBOARD_REPO_PATH is not set
            os.environ.pop("DASHBOARD_REPO_PATH", None)
            result = run_dashboard_stage()

        assert result.skipped is True
        assert result.published is False
        assert result.errors == []

    @patch("v2.dashboard_publish.gather_dashboard_data", side_effect=Exception("DB down"))
    def test_handles_gather_error(self, mock_gather):
        """Error in gather step is captured."""
        with patch.dict(os.environ, {"DASHBOARD_REPO_PATH": "/fake/repo"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Data gathering failed" in result.errors[0]

    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.write_json_files", side_effect=Exception("Disk full"))
    def test_handles_write_error(self, mock_write, mock_gather):
        """Error in write step is captured."""
        with patch.dict(os.environ, {"DASHBOARD_REPO_PATH": "/fake/repo"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "JSON writing failed" in result.errors[0]

    @patch("v2.dashboard_publish.gather_dashboard_data", return_value={"summary": {}})
    @patch("v2.dashboard_publish.write_json_files", return_value=[])
    @patch("v2.dashboard_publish.push_to_github", side_effect=RuntimeError("git push failed"))
    def test_handles_push_error(self, mock_push, mock_write, mock_gather):
        """Error in push step is captured."""
        with patch.dict(os.environ, {"DASHBOARD_REPO_PATH": "/fake/repo"}):
            result = run_dashboard_stage()

        assert result.published is False
        assert len(result.errors) == 1
        assert "Git push failed" in result.errors[0]
