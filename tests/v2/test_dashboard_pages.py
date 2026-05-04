"""Tests for v2/dashboard_pages.py — HTML page rendering."""

from datetime import date
from decimal import Decimal

from v2.dashboard_pages import render_homepage_meta, render_trade_page, render_thesis_page


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
        assert ">BUY NVDA</h2>" in html
        assert "shares at $450.25" in html
        assert "12 shares" in html

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
        assert 'href="/thesis/7/">AI capex acceleration</a>' in html

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

    def test_escapes_ticker_and_action(self):
        """Ticker with special chars is single-escaped; double-escaping must not occur."""
        html = render_trade_page(
            decision=self._decision(ticker="AT&T", action="buy"),
            thesis=None,
            position=None,
            base_url="https://example.com",
        )
        # Single-escaped form must appear in og:description and h2
        assert "AT&amp;T" in html
        # Double-escaped form must NOT appear anywhere
        assert "AT&amp;amp;T" not in html

    def test_empty_thesis_text_falls_back_to_label(self):
        """Empty thesis text should produce a visible link label, not an invisible link."""
        thesis = {
            "id": 99,
            "ticker": "NVDA",
            "direction": "long",
            "thesis": "",
            "confidence": "medium",
        }
        html = render_trade_page(
            decision=self._decision(),
            thesis=thesis,
            position=None,
            base_url="https://example.com",
        )
        assert "Thesis #99" in html
        assert 'href="/thesis/99/"' in html


class TestRenderThesisPage:
    def _thesis(self, **overrides):
        t = {
            "id": 7,
            "ticker": "NVDA",
            "direction": "long",
            "confidence": "high",
            "thesis": "AI capex acceleration is real and unpriced.",
            "entry_trigger": "Pullback below $440",
            "exit_trigger": "Hit $520 or stop at $410",
            "invalidation": "Rev growth slows two quarters",
            "status": "active",
        }
        t.update(overrides)
        return t

    def test_includes_thesis_text_and_triggers(self):
        html = render_thesis_page(
            thesis=self._thesis(),
            decisions=[],
            position=None,
            base_url="https://example.com",
        )
        assert "AI capex acceleration" in html
        assert "Pullback below $440" in html
        assert "Hit $520" in html

    def test_og_image_url(self):
        html = render_thesis_page(
            thesis=self._thesis(),
            decisions=[],
            position=None,
            base_url="https://example.com",
        )
        assert "https://example.com/og/thesis/7.png" in html
        assert '<meta property="og:image"' in html

    def test_lists_related_decisions(self):
        decisions = [
            {"id": 42, "date": date(2026, 5, 3), "ticker": "NVDA",
             "action": "buy", "quantity": 12, "price": Decimal("450.25")},
            {"id": 43, "date": date(2026, 5, 5), "ticker": "NVDA",
             "action": "sell", "quantity": 4, "price": Decimal("470.10")},
        ]
        html = render_thesis_page(
            thesis=self._thesis(),
            decisions=decisions,
            position=None,
            base_url="https://example.com",
        )
        assert "/trade/42/" in html
        assert "/trade/43/" in html

    def test_escapes_user_text(self):
        html = render_thesis_page(
            thesis=self._thesis(thesis="<img src=x onerror=alert(1)>"),
            decisions=[],
            position=None,
            base_url="https://example.com",
        )
        assert "<img src=x" not in html
        assert "&lt;img" in html

    def test_escapes_thesis_meta_fields(self):
        """Direction, confidence, status all flow into HTML; verify each is escaped."""
        html = render_thesis_page(
            thesis=self._thesis(direction="long&short", confidence="<high>", status='ac"tive'),
            decisions=[],
            position=None,
            base_url="https://example.com",
        )
        assert "long&amp;short" in html
        assert "&lt;high&gt;" in html
        assert "ac&quot;tive" in html  # html.escape with quote=True (default)
        # Negative — confirm not double-escaped:
        assert "long&amp;amp;short" not in html

    def test_decisions_section_renders_formatted_price(self):
        """Related-decisions list should use $N,NNN.NN formatting (not raw repr)."""
        decisions = [{
            "id": 42, "date": date(2026, 5, 3), "ticker": "NVDA",
            "action": "buy", "quantity": 12, "price": Decimal("1450.25"),
        }]
        html = render_thesis_page(
            thesis=self._thesis(),
            decisions=decisions,
            position=None,
            base_url="https://example.com",
        )
        assert "$1,450.25" in html


from datetime import date as _date


class TestRenderMistakesPage:
    def _losers(self, n=2):
        return [
            {"id": 100 + i, "date": _date(2026, 4, 28 - i), "ticker": f"TKR{i}",
             "action": "buy", "quantity": 10, "price": Decimal("50.00"),
             "reasoning": "Reason text", "outcome_7d": Decimal("-5.0"),
             "outcome_30d": Decimal("-12.0") - Decimal(i)}
            for i in range(n)
        ]

    def _retired(self, n=1):
        return [
            {"id": 200 + i, "rule_text": f"Rule {i}: cap macro at $500/day",
             "category": "macro_signal:fed", "direction": "constraint",
             "confidence": Decimal("0.7"), "retired_at": _date(2026, 4, 20),
             "retirement_reason": "stale data"}
            for i in range(n)
        ]

    def test_renders_section_for_each_loser(self):
        from v2.dashboard_pages import render_mistakes_page

        html = render_mistakes_page(
            closed_losers=self._losers(2),
            retired_rules=self._retired(1),
            base_url="https://example.com",
        )
        assert "TKR0" in html
        assert "TKR1" in html
        assert "Rule 0" in html

    def test_includes_og_meta_block(self):
        from v2.dashboard_pages import render_mistakes_page

        html = render_mistakes_page(
            closed_losers=self._losers(1),
            retired_rules=[],
            base_url="https://example.com",
        )
        assert "https://example.com/og/mistakes.png" in html
        assert '<meta property="og:title"' in html
        assert "https://example.com/mistakes/" in html

    def test_empty_state_when_no_data(self):
        from v2.dashboard_pages import render_mistakes_page

        html = render_mistakes_page(
            closed_losers=[],
            retired_rules=[],
            base_url="https://example.com",
        )
        # Empty state per spec.
        assert "No closed losers" in html or "no losers" in html.lower()

    def test_escapes_user_text(self):
        from v2.dashboard_pages import render_mistakes_page

        html = render_mistakes_page(
            closed_losers=[],
            retired_rules=[{"id": 1, "rule_text": "<script>x</script>",
                            "category": "x", "direction": "x",
                            "confidence": Decimal("0.5"),
                            "retired_at": _date(2026, 4, 1),
                            "retirement_reason": "<b>bad</b>"}],
            base_url="https://example.com",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
