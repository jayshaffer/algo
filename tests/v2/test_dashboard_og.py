"""Tests for v2/dashboard_og.py — OG image generation."""

from datetime import date
from decimal import Decimal
from io import BytesIO

import pytest

from v2.dashboard_og import render_trade_og


def _decoded(png_bytes):
    """Open the PNG bytes via Pillow to validate the format."""
    from PIL import Image
    return Image.open(BytesIO(png_bytes))


class TestRenderTradeOg:
    def test_returns_valid_png_bytes(self):
        decision = {
            "id": 42, "date": date(2026, 5, 3), "ticker": "NVDA",
            "action": "buy", "quantity": 12, "price": Decimal("450.25"),
        }
        png = render_trade_og(decision)
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature

    def test_correct_dimensions(self):
        decision = {
            "id": 42, "date": date(2026, 5, 3), "ticker": "NVDA",
            "action": "buy", "quantity": 12, "price": Decimal("450.25"),
        }
        img = _decoded(render_trade_og(decision))
        assert img.size == (1200, 630)

    def test_handles_missing_optional_fields(self):
        # Partial decision — must not crash.
        decision = {
            "id": 99, "date": date(2026, 5, 3), "ticker": "AAPL",
            "action": "sell",
        }
        png = render_trade_og(decision)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
