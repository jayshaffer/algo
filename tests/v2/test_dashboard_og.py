"""Tests for v2/dashboard_og.py — OG image generation."""

from datetime import date
from decimal import Decimal
from io import BytesIO

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

    def test_renders_non_trivial_content(self):
        """Catch the regression where the canvas is created but draw calls silently no-op."""
        decision = {
            "id": 42, "date": date(2026, 5, 3), "ticker": "NVDA",
            "action": "buy", "quantity": 12, "price": Decimal("450.25"),
        }
        img = _decoded(render_trade_og(decision)).convert("RGB")
        pixels = list(img.getdata())
        distinct = set(pixels)
        # Should have at least 4 distinct colors: background, accent bar, ticker FG,
        # text muted color (and font anti-aliasing produces many more).
        assert len(distinct) >= 4, f"Only {len(distinct)} distinct colors — text didn't render?"


from v2.dashboard_og import render_home_og, render_thesis_og


class TestRenderHomeOg:
    def test_returns_valid_png(self):
        summary = {"portfolio_value": Decimal("12345.67"),
                   "daily_pnl": Decimal("123.45"),
                   "daily_pnl_pct": Decimal("1.01")}
        png = render_home_og(summary)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_correct_dimensions(self):
        summary = {"portfolio_value": 0, "daily_pnl": 0, "daily_pnl_pct": 0}
        img = _decoded(render_home_og(summary))
        assert img.size == (1200, 630)

    def test_handles_negative_pnl(self):
        summary = {"portfolio_value": Decimal("12345.67"),
                   "daily_pnl": Decimal("-456.78"),
                   "daily_pnl_pct": Decimal("-3.50")}
        png = render_home_og(summary)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_non_trivial_content(self):
        summary = {"portfolio_value": Decimal("12345.67"),
                   "daily_pnl": Decimal("123.45"),
                   "daily_pnl_pct": Decimal("1.01")}
        img = _decoded(render_home_og(summary)).convert("RGB")
        distinct = set(img.getdata())
        assert len(distinct) >= 4


class TestRenderThesisOg:
    def _thesis(self, **overrides):
        t = {
            "id": 7, "ticker": "NVDA", "direction": "long",
            "confidence": "high",
            "thesis": "AI capex acceleration is real and unpriced.",
        }
        t.update(overrides)
        return t

    def test_returns_valid_png_bytes(self):
        png = render_thesis_og(self._thesis())
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_correct_dimensions(self):
        img = _decoded(render_thesis_og(self._thesis()))
        assert img.size == (1200, 630)

    def test_handles_long_thesis_text(self):
        png = render_thesis_og(self._thesis(thesis="x" * 5000))  # Should not crash
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_non_trivial_content(self):
        img = _decoded(render_thesis_og(self._thesis())).convert("RGB")
        distinct = set(img.getdata())
        assert len(distinct) >= 4

    def test_handles_missing_optional_fields(self):
        # Only the absolutely required fields — must not crash.
        png = render_thesis_og({"id": 99, "ticker": "AAPL"})
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


class TestRenderMistakesOg:
    def test_returns_valid_png_with_top_loser(self):
        from v2.dashboard_og import render_mistakes_og

        png = render_mistakes_og(
            top_loser={"ticker": "TSLA", "outcome_30d": Decimal("-12.5")},
        )
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_correct_dimensions(self):
        from v2.dashboard_og import render_mistakes_og
        from io import BytesIO
        from PIL import Image

        png = render_mistakes_og(
            top_loser={"ticker": "TSLA", "outcome_30d": Decimal("-12.5")},
        )
        img = Image.open(BytesIO(png))
        assert img.size == (1200, 630)

    def test_handles_no_losers(self):
        from v2.dashboard_og import render_mistakes_og

        png = render_mistakes_og(top_loser=None)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
