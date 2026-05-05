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


class TestFormatTradeHeadline:
    """Past-tense action + dollar notional, rounded — readable at thumbnail
    size where '1.10452487 shares' is unreadable noise."""

    def test_buy_uses_past_tense_with_dollar_notional(self):
        from v2.dashboard_og import _format_trade_headline
        assert _format_trade_headline("buy", Decimal("1.10452487"), Decimal("271.60")) == "Bought $300"

    def test_sell_uses_past_tense(self):
        from v2.dashboard_og import _format_trade_headline
        assert _format_trade_headline("sell", Decimal("2"), Decimal("450.00")) == "Sold $900"

    def test_large_notional_uses_thousands_separator(self):
        from v2.dashboard_og import _format_trade_headline
        assert _format_trade_headline("buy", Decimal("100"), Decimal("250")) == "Bought $25,000"

    def test_uppercases_unknown_actions(self):
        from v2.dashboard_og import _format_trade_headline
        # Trade-posts stage filters out 'hold', but keep the renderer robust.
        result = _format_trade_headline("trim", Decimal("5"), Decimal("100"))
        assert "$500" in result
        assert "Trim" in result

    def test_missing_quantity_or_price_omits_dollars(self):
        from v2.dashboard_og import _format_trade_headline
        # Quantity/price can be None when render is fed a partial row.
        assert _format_trade_headline("buy", None, Decimal("100")) == "Bought"
        assert _format_trade_headline("buy", Decimal("1"), None) == "Bought"


class TestRenderTradeOgNewLayout:
    """The card now leads with the action+notional, then ticker+price, then
    the trade reasoning. Verifies fields land somewhere on the canvas, not
    that we crash on edge cases — see TestRenderTradeOg above for those."""

    def test_renders_reasoning_when_present(self):
        """Adding a reasoning string changes the rendered pixels — proves
        the reasoning is being drawn, not silently dropped."""
        decision_no_reasoning = {
            "id": 1, "ticker": "AAPL", "action": "buy",
            "quantity": Decimal("1"), "price": Decimal("100"),
        }
        decision_with_reasoning = {
            **decision_no_reasoning,
            "reasoning": "Q1 fundamentals are solid; AWS hitting $150B run rate.",
        }
        from v2.dashboard_og import render_trade_og
        without = render_trade_og(decision_no_reasoning)
        with_ = render_trade_og(decision_with_reasoning)
        assert without != with_, "reasoning text should change the rendered PNG"

    def test_handles_long_reasoning_without_overflow(self):
        from v2.dashboard_og import render_trade_og
        long_text = ("This is a very long reasoning string. " * 40).strip()
        decision = {
            "id": 1, "ticker": "AAPL", "action": "buy",
            "quantity": Decimal("1"), "price": Decimal("100"),
            "reasoning": long_text,
        }
        png = render_trade_og(decision)
        img = _decoded(png)
        assert img.size == (1200, 630)


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


class TestRenderAttributionOg:
    def _attr(self):
        return [
            {"category": "earnings", "avg_outcome_30d": Decimal("3.40"), "sample_size": 30},
            {"category": "fed",      "avg_outcome_30d": Decimal("-1.20"), "sample_size": 12},
            {"category": "macro",    "avg_outcome_30d": Decimal("0.80"),  "sample_size": 9},
        ]

    def test_returns_valid_png(self):
        from v2.dashboard_og import render_attribution_og

        png = render_attribution_og(self._attr())
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_handles_empty_attribution(self):
        from v2.dashboard_og import render_attribution_og

        png = render_attribution_og([])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_renders_bars_at_expected_positions(self):
        """Smoke check: pixels at known bar-center positions are not the
        background color (i.e. a bar was actually drawn)."""
        from io import BytesIO
        from PIL import Image
        from v2.dashboard_og import OG_HEIGHT, render_attribution_og

        png = render_attribution_og(self._attr())
        img = Image.open(BytesIO(png)).convert("RGB")
        # First positive bar is positioned at x=200 (per the layout below).
        # Sample inside the bar; the column above the baseline (y=400)
        # should NOT match the background.
        bg = (8, 24, 32)
        non_bg_count = 0
        for x in range(195, 235):
            for y in range(330, 395):
                if img.getpixel((x, y)) != bg:
                    non_bg_count += 1
        assert non_bg_count > 0, "First bar did not render any non-background pixels"
