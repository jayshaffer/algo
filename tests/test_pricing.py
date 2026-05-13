"""Unit tests for v2/pricing.py — DB-backed cost lookup helper."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from v2.pricing import UnknownModelError, stage_cost_usd


@contextmanager
def _mock_cursor(rows_by_query):
    """Patch v2.database.trading_db.get_cursor; rows_by_query maps
    fetchone() return values keyed by call order."""
    cursor = MagicMock()
    cursor.fetchone.side_effect = rows_by_query
    @contextmanager
    def _gc():
        yield cursor
    with patch("v2.pricing.get_cursor", _gc):
        yield cursor


def test_haiku_cost_matches_hand_computation():
    rates = {
        "input_per_mtok": 1.00,
        "output_per_mtok": 5.00,
        "cache_creation_per_mtok": 1.25,
        "cache_read_per_mtok": 0.10,
    }
    with _mock_cursor([rates]):
        cost = stage_cost_usd(
            model="claude-haiku-4-5-20251001",
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_creation_tokens=200_000,
            cache_read_tokens=800_000,
        )
    # 1.00 + 2.50 + 0.25 + 0.08 = 3.83
    assert cost == pytest.approx(3.83)


def test_opus_cost_matches_hand_computation():
    rates = {
        "input_per_mtok": 15.00,
        "output_per_mtok": 75.00,
        "cache_creation_per_mtok": 18.75,
        "cache_read_per_mtok": 1.50,
    }
    with _mock_cursor([rates]):
        cost = stage_cost_usd(
            model="claude-opus-4-7",
            input_tokens=10,
            output_tokens=5,
            cache_creation_tokens=100,
            cache_read_tokens=400,
        )
    # 0.00015 + 0.000375 + 0.001875 + 0.00060 = 0.0030
    assert cost == pytest.approx(0.003)


def test_zero_tokens_returns_zero():
    rates = {
        "input_per_mtok": 1.00,
        "output_per_mtok": 5.00,
        "cache_creation_per_mtok": 1.25,
        "cache_read_per_mtok": 0.10,
    }
    with _mock_cursor([rates]):
        cost = stage_cost_usd("claude-haiku-4-5", 0, 0, 0, 0)
    assert cost == 0.0


def test_unknown_model_raises():
    with _mock_cursor([None]), pytest.raises(UnknownModelError):
        stage_cost_usd("claude-future-99", 1000, 1000, 0, 0)
