"""Tests for v2/formation.py - cold-start detection and formation context."""

from unittest.mock import patch

import pytest

from v2.formation import is_formation_mode, FORMATION_TRADE_THRESHOLD


class TestIsFormationMode:

    @patch("v2.formation.get_recent_decisions")
    def test_formation_when_no_decisions(self, mock_decisions):
        mock_decisions.return_value = []
        assert is_formation_mode() is True

    @patch("v2.formation.get_recent_decisions")
    def test_formation_when_few_completed_cycles(self, mock_decisions):
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "buy"},
            {"outcome_7d": -0.8, "action": "buy"},
            {"outcome_7d": 2.0, "action": "sell"},
        ]
        assert is_formation_mode() is True

    @patch("v2.formation.get_recent_decisions")
    def test_not_formation_when_enough_cycles(self, mock_decisions):
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "buy"},
            {"outcome_7d": -0.8, "action": "buy"},
            {"outcome_7d": 2.0, "action": "sell"},
            {"outcome_7d": 0.3, "action": "buy"},
            {"outcome_7d": -1.2, "action": "sell"},
        ]
        assert is_formation_mode() is False

    @patch("v2.formation.get_recent_decisions")
    def test_hold_decisions_not_counted(self, mock_decisions):
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "hold"},
            {"outcome_7d": -0.8, "action": "hold"},
            {"outcome_7d": 2.0, "action": "buy"},
        ]
        assert is_formation_mode() is True

    @patch("v2.formation.get_recent_decisions")
    def test_null_outcomes_not_counted(self, mock_decisions):
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "buy"},
            {"outcome_7d": None, "action": "buy"},
            {"outcome_7d": None, "action": "sell"},
            {"outcome_7d": 2.0, "action": "sell"},
        ]
        assert is_formation_mode() is True


from v2.formation import get_orphan_positions


class TestGetOrphanPositions:

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_all_positions_orphaned(self, mock_positions, mock_theses):
        mock_positions.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
            {"ticker": "NVDA", "shares": 10, "avg_cost": 800.0},
        ]
        mock_theses.return_value = []
        orphans = get_orphan_positions()
        assert len(orphans) == 2
        assert orphans[0]["ticker"] == "AAPL"

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_no_orphans_when_all_covered(self, mock_positions, mock_theses):
        mock_positions.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
        ]
        mock_theses.return_value = [
            {"ticker": "AAPL", "direction": "long"},
        ]
        orphans = get_orphan_positions()
        assert len(orphans) == 0

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_mixed_orphans(self, mock_positions, mock_theses):
        mock_positions.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
            {"ticker": "COIN", "shares": 10, "avg_cost": 200.0},
            {"ticker": "NVDA", "shares": 10, "avg_cost": 800.0},
        ]
        mock_theses.return_value = [
            {"ticker": "COIN", "direction": "long"},
        ]
        orphans = get_orphan_positions()
        assert len(orphans) == 2
        tickers = [o["ticker"] for o in orphans]
        assert "AAPL" in tickers
        assert "NVDA" in tickers
        assert "COIN" not in tickers

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_no_positions(self, mock_positions, mock_theses):
        mock_positions.return_value = []
        mock_theses.return_value = []
        orphans = get_orphan_positions()
        assert len(orphans) == 0
