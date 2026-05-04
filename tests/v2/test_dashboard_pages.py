"""Tests for v2/dashboard_pages.py — HTML page rendering."""

from datetime import date
from decimal import Decimal

from v2.dashboard_pages import render_homepage_meta, render_trade_page


class TestRenderHomepageMeta:
    def test_includes_required_og_tags(self):
        summary = {
            "portfolio_value": Decimal("12345.67"),
            "daily_pnl": Decimal("123.45"),
            "daily_pnl_pct": Decimal("1.01"),
            "last_updated": "2026-05-03",
        }
        html = render_homepage_meta(summary, base_url="https://example.com")

        assert '<meta property="og:title"' in html
        assert '<meta property="og:description"' in html
        assert '<meta property="og:image"' in html
        assert '<meta name="twitter:card" content="summary_large_image"' in html
        assert "https://example.com/og/home.png" in html

    def test_description_includes_portfolio_value(self):
        summary = {
            "portfolio_value": Decimal("12345.67"),
            "daily_pnl": Decimal("123.45"),
            "daily_pnl_pct": Decimal("1.01"),
            "last_updated": "2026-05-03",
        }
        html = render_homepage_meta(summary, base_url="https://example.com")
        assert "$12,345.67" in html

    def test_handles_missing_summary_fields(self):
        # Empty-DB path emits zeros; must not raise.
        summary = {
            "portfolio_value": 0,
            "daily_pnl": 0,
            "daily_pnl_pct": 0,
            "last_updated": "2026-05-03",
        }
        html = render_homepage_meta(summary, base_url="https://example.com")
        assert '<meta property="og:title"' in html


class TestRenderTradePage:
    def _decision(self, **overrides):
        d = {
            "id": 42,
            "date": date(2026, 5, 3),
            "ticker": "NVDA",
            "action": "buy",
            "quantity": 12,
            "price": Decimal("450.25"),
            "reasoning": "Strong AI capex cycle; thesis #7 active.",
            "outcome_7d": None,
            "outcome_30d": None,
            "thesis_id": 7,
        }
        d.update(overrides)
        return d

    def test_includes_ticker_and_action(self):
        html = render_trade_page(
            decision=self._decision(),
            thesis=None,
            position=None,
            base_url="https://example.com",
        )
        assert "NVDA" in html
        assert "BUY" in html or "Buy" in html or "buy" in html
        assert "12" in html
        assert "450.25" in html

    def test_includes_og_tags_with_correct_image_url(self):
        html = render_trade_page(
            decision=self._decision(),
            thesis=None,
            position=None,
            base_url="https://example.com",
        )
        assert "https://example.com/og/trade/42.png" in html
        assert '<meta property="og:image"' in html
        assert '<meta name="twitter:card" content="summary_large_image"' in html

    def test_includes_thesis_link_when_present(self):
        thesis = {
            "id": 7,
            "ticker": "NVDA",
            "direction": "long",
            "thesis": "AI capex acceleration",
            "confidence": "high",
        }
        html = render_trade_page(
            decision=self._decision(),
            thesis=thesis,
            position=None,
            base_url="https://example.com",
        )
        assert "/thesis/7/" in html
        assert "AI capex acceleration" in html

    def test_no_thesis_section_when_absent(self):
        html = render_trade_page(
            decision=self._decision(thesis_id=None),
            thesis=None,
            position=None,
            base_url="https://example.com",
        )
        assert "/thesis/" not in html

    def test_escapes_html_in_reasoning(self):
        html = render_trade_page(
            decision=self._decision(reasoning="<script>alert('x')</script>"),
            thesis=None,
            position=None,
            base_url="https://example.com",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
