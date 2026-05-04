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


class TestRenderAttributionPage:
    def _attribution(self):
        return [
            {"category": "earnings", "sample_size": 30, "sample_size_30d": 24,
             "avg_outcome_7d": Decimal("1.20"), "avg_outcome_30d": Decimal("3.40"),
             "win_rate_7d": Decimal("0.6"), "win_rate_30d": Decimal("0.55")},
            {"category": "fed", "sample_size": 12, "sample_size_30d": 10,
             "avg_outcome_7d": Decimal("-0.50"), "avg_outcome_30d": Decimal("-1.20"),
             "win_rate_7d": Decimal("0.4"), "win_rate_30d": Decimal("0.42")},
        ]

    def test_renders_table_with_each_row(self):
        from v2.dashboard_pages import render_attribution_page

        html = render_attribution_page(
            attribution=self._attribution(),
            base_url="https://example.com",
        )
        assert "earnings" in html
        assert "fed" in html
        assert "<table" in html

    def test_includes_og_meta_block(self):
        from v2.dashboard_pages import render_attribution_page

        html = render_attribution_page(
            attribution=self._attribution(),
            base_url="https://example.com",
        )
        assert "https://example.com/og/attribution.png" in html
        assert "https://example.com/attribution/" in html

    def test_empty_state_when_no_attribution(self):
        from v2.dashboard_pages import render_attribution_page

        html = render_attribution_page(
            attribution=[],
            base_url="https://example.com",
        )
        assert "Not enough samples" in html or "no attribution" in html.lower()


from v2.dashboard_pages import _render_page_shell


class TestRenderPageShell:
    def _shell(self, **overrides):
        defaults = dict(
            title="Test Title",
            description="Test description",
            active_nav="home",
            content="<p>body</p>",
            og_image="https://example.com/og/test.png",
            page_url="https://example.com/test/",
        )
        defaults.update(overrides)
        return _render_page_shell(**defaults)

    def test_emits_doctype_and_title(self):
        html = self._shell()
        assert html.startswith("<!DOCTYPE html>")
        assert "<title>Test Title — Bikini Bottom Capital</title>" in html

    def test_includes_og_meta(self):
        html = self._shell()
        assert '<meta property="og:title" content="Test Title"' in html
        assert "https://example.com/og/test.png" in html
        assert '<meta name="twitter:card" content="summary_large_image"' in html

    def test_links_stylesheet(self):
        html = self._shell()
        assert '<link rel="stylesheet" href="/styles.css"' in html

    def test_renders_full_nav(self):
        html = self._shell()
        for label, href in [
            ("Home", "/"),
            ("Performance", "/performance/"),
            ("Activity", "/activity/"),
            ("Learning", "/learning/"),
            ("How it works", "/how-it-works/"),
        ]:
            assert f'href="{href}"' in html
            assert f">{label}<" in html

    def test_marks_active_nav_item(self):
        html = self._shell(active_nav="performance")
        assert 'class="active" href="/performance/"' in html
        assert 'class="active" href="/"' not in html

    def test_unknown_active_nav_marks_nothing(self):
        html = self._shell(active_nav="unknown")
        assert 'class="active"' not in html

    def test_includes_footer(self):
        html = self._shell()
        assert "Is mayonnaise a financial instrument?" in html
        assert "alpaca.markets" in html

    def test_content_rendered_in_main(self):
        html = self._shell(content="<p>my body</p>")
        assert "<p>my body</p>" in html

    def test_attaches_data_page_attribute(self):
        html = self._shell(active_nav="activity")
        assert 'data-page="activity"' in html

    def test_data_page_overrides_active_nav(self):
        html = self._shell(active_nav="learning", data_page="mistakes")
        assert 'data-page="mistakes"' in html
        assert 'class="active" href="/learning/"' in html

    def test_loads_app_js(self):
        html = self._shell()
        assert '<script src="/app.js"></script>' in html


from v2.dashboard_pages import render_homepage


class TestRenderHomepage:
    def _data(self, **overrides):
        defaults = dict(
            summary={
                "portfolio_value": Decimal("104231.00"),
                "daily_pnl": Decimal("642.00"),
                "daily_pnl_pct": Decimal("0.62"),
                "total_return_pct": Decimal("4.2"),
                "vs_spy_pct": Decimal("2.1"),
                "day_number": 142,
                "last_updated": "2026-05-04T16:30:00",
            },
            theses=[
                {"id": 7, "ticker": "NVDA", "thesis": "AI infra demand"},
                {"id": 8, "ticker": "AMD", "thesis": "data center share"},
                {"id": 9, "ticker": "XOM", "thesis": "macro hedge"},
            ],
            sparkline_svg='<svg class="sparkline"></svg>',
            today_move={
                "id": 42, "ticker": "NVDA", "action": "buy",
                "notional": Decimal("2400"), "pct_of_portfolio": Decimal("2.3"),
                "reasoning": "Earnings beat + guidance raise. Active thesis on AI infra.",
            },
            attribution_top={
                "category": "earnings_beat", "sample_size": 18,
                "avg_outcome_30d": Decimal("3.2"),
            },
            worst_loser={
                "ticker": "PLTR", "outcome_30d_pct": Decimal("-8.4"),
            },
            memo={
                "session_date": date(2026, 5, 4),
                "content": "Macro chop is unresolved. Holding the AI book but tightening sizing on new entries.",
            },
            how_it_works_state={
                "about": True, "internals": True, "trace": False,
            },
            base_url="https://example.com",
        )
        defaults.update(overrides)
        return defaults

    def test_uses_page_shell_with_home_active(self):
        html = render_homepage(**self._data())
        assert 'data-page="home"' in html
        assert 'class="active" href="/"' in html

    def test_renders_hero_with_stats_and_chips(self):
        html = render_homepage(**self._data())
        assert "Day 142" in html
        assert "$104,231.00" in html
        assert "NVDA" in html
        assert "AMD" in html
        assert "XOM" in html
        assert 'href="/thesis/7/"' in html

    def test_omits_chips_when_no_active_theses(self):
        html = render_homepage(**self._data(theses=[]))
        assert "Currently betting on" not in html

    def test_caps_chips_at_three(self):
        many = [
            {"id": i, "ticker": f"T{i}", "thesis": "x"} for i in range(10)
        ]
        html = render_homepage(**self._data(theses=many))
        assert html.count('class="chip"') == 3

    def test_renders_today_move_card(self):
        html = render_homepage(**self._data())
        assert 'href="/trade/42/"' in html
        assert "Earnings beat + guidance raise" in html
        assert "$2,400" in html or "$2,400.00" in html

    def test_today_move_empty_falls_back_to_link(self):
        html = render_homepage(**self._data(today_move=None))
        assert "No new positions in the last 5 sessions" in html
        assert 'href="/activity/"' in html

    def test_recent_learnings_renders_both_cards(self):
        html = render_homepage(**self._data())
        assert "earnings_beat" in html
        assert "PLTR" in html
        assert 'href="/attribution/"' in html
        assert 'href="/mistakes/"' in html

    def test_recent_learnings_hidden_when_both_empty(self):
        html = render_homepage(
            **self._data(attribution_top=None, worst_loser=None)
        )
        assert "Recent learnings" not in html

    def test_memo_block_present(self):
        html = render_homepage(**self._data())
        assert "Macro chop is unresolved" in html
        assert "memo-block" in html

    def test_memo_block_hidden_when_no_memo(self):
        html = render_homepage(**self._data(memo=None))
        assert "memo-block" not in html

    def test_methodology_strip_links_to_existing_pages(self):
        html = render_homepage(**self._data())
        assert 'href="/about/"' in html
        assert 'href="/internals/"' in html
        assert 'href="/trace/"' not in html
        assert 'href="/how-it-works/"' in html

    def test_sparkline_embedded(self):
        html = render_homepage(**self._data())
        assert '<svg class="sparkline"></svg>' in html

    def test_truncates_long_reasoning(self):
        long_reasoning = "x" * 500
        html = render_homepage(
            **self._data(today_move={
                "id": 1, "ticker": "Z", "action": "buy",
                "notional": Decimal("100"), "pct_of_portfolio": Decimal("0.1"),
                "reasoning": long_reasoning,
            })
        )
        assert ("x" * 150 + "…") in html
        assert ("x" * 151) not in html


from v2.dashboard_pages import render_performance_page


class TestRenderPerformancePage:
    def _data(self, **overrides):
        defaults = dict(
            summary={
                "portfolio_value": Decimal("104231.00"),
                "daily_pnl_pct": Decimal("0.62"),
                "total_return_pct": Decimal("4.2"),
                "vs_spy_pct": Decimal("2.1"),
            },
            performance={
                "max_drawdown_pct": -5.2,
                "win_rate_pct": 60.0,
                "avg_days_held": 4.0,
                "best_day_pct": 2.0,
                "worst_day_pct": -3.0,
            },
            base_url="https://example.com",
        )
        defaults.update(overrides)
        return defaults

    def test_uses_page_shell_with_performance_active(self):
        html = render_performance_page(**self._data())
        assert 'data-page="performance"' in html
        assert 'class="active" href="/performance/"' in html

    def test_renders_stat_strip(self):
        html = render_performance_page(**self._data())
        assert "$104,231.00" in html
        assert "+0.62%" in html
        assert "+2.1" in html

    def test_renders_chart_canvases(self):
        html = render_performance_page(**self._data())
        assert 'id="equity-chart"' in html
        assert 'id="benchmark-chart"' in html

    def test_renders_stats_panel(self):
        html = render_performance_page(**self._data())
        assert "Max drawdown" in html
        assert "Win rate" in html
        assert "60.0" in html

    def test_loads_chart_js(self):
        html = render_performance_page(**self._data())
        assert "chart.js" in html.lower() or "Chart.js" in html


from v2.dashboard_pages import render_activity_page


class TestRenderActivityPage:
    def _data(self, **overrides):
        defaults = dict(
            base_url="https://example.com",
            memos=[
                {"id": 1, "session_date": date(2026, 5, 4),
                 "content": "Holding the AI book."},
                {"id": 2, "session_date": date(2026, 5, 3),
                 "content": "Macro chop unresolved."},
            ],
        )
        defaults.update(overrides)
        return defaults

    def test_uses_page_shell_with_activity_active(self):
        html = render_activity_page(**self._data())
        assert 'data-page="activity"' in html
        assert 'class="active" href="/activity/"' in html

    def test_has_all_anchored_sections(self):
        html = render_activity_page(**self._data())
        assert 'id="holdings"' in html
        assert 'id="theses"' in html
        assert 'id="decisions"' in html
        assert 'id="memos"' in html

    def test_holdings_table_skeleton(self):
        html = render_activity_page(**self._data())
        assert 'id="positions-table"' in html
        assert "Ticker" in html and "Shares" in html

    def test_theses_container(self):
        html = render_activity_page(**self._data())
        assert 'id="theses-list"' in html

    def test_decisions_table_skeleton(self):
        html = render_activity_page(**self._data())
        assert 'id="decisions-table"' in html

    def test_memos_rendered_inline(self):
        html = render_activity_page(**self._data())
        assert "Holding the AI book" in html
        assert "Macro chop unresolved" in html

    def test_memos_empty_state(self):
        html = render_activity_page(**self._data(memos=[]))
        assert "No memos yet" in html
