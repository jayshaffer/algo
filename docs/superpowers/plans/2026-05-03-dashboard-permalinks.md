# Dashboard Permalinks & OG Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Stage 6 of the daily session to emit per-trade and per-thesis HTML pages plus dynamic OG images, so social posts have permalinks with rich Twitter/Bluesky preview cards.

**Architecture:** Two new modules (`v2/dashboard_pages.py` for HTML templating with `string.Template`, `v2/dashboard_og.py` for Pillow-based PNG generation). Extend `v2/dashboard_publish.py` to gather per-page data for **all** decisions and theses (not just the 30-day window — Cloudflare Pages full-bundle replacement makes link permanence load-bearing) and write the new files into the deploy directory. Homepage stays SPA; new pages are static HTML siblings.

**Tech Stack:** Python 3.10+, Pillow ≥10.0, psycopg2, pytest, Cloudflare Pages via wrangler. No JS framework introduced. No new DB tables in this plan (Specs #2–4 add their own).

**Spec:** [`docs/superpowers/specs/2026-05-03-dashboard-permalinks-design.md`](../specs/2026-05-03-dashboard-permalinks-design.md)

**Out of scope:** Mistakes/attribution panels (Spec #3), about/internals/trace pages (Spec #4), social-pipeline changes (Spec #2). Sitemap/robots may be folded in if trivial; otherwise deferred.

---

## File Structure

**New:**
- `v2/dashboard_pages.py` — `render_homepage_meta`, `render_trade_page`, `render_thesis_page`. Pure functions, `string.Template` templating, no DB access.
- `v2/dashboard_og.py` — `render_trade_og`, `render_thesis_og`. Pillow-based PNG generation, no DB access. Uses `ImageFont.load_default()` for v1 (typography polish deferred).
- `tests/v2/test_dashboard_pages.py`
- `tests/v2/test_dashboard_og.py`

**Modified:**
- `v2/dashboard_publish.py` — add `gather_trade_detail`, `gather_thesis_detail`, `gather_all_pages_data`, extend `assemble_deploy_dir`.
- `v2/requirements.txt` — add `pillow>=10.0`.
- `public_dashboard/index.html` — add `<!-- OG_META -->` placeholder and static `og:site_name` / `og:type` / `twitter:card` tags.
- `tests/v2/test_dashboard_publish.py` — add tests for new gather functions and assembly behavior.

---

## Task 1: Add Pillow dependency

**Files:**
- Modify: `v2/requirements.txt`

- [ ] **Step 1: Add Pillow to requirements**

In `v2/requirements.txt`, add a single line at the end:

```
pillow>=10.0
```

- [ ] **Step 2: Install in the dev environment**

Run: `docker compose exec trading pip install -r v2/requirements.txt`
Expected: `Successfully installed pillow-...`

If running tests on host without Docker, also run: `pip install pillow>=10.0`

- [ ] **Step 3: Verify import works**

Run: `python -c "from PIL import Image, ImageDraw, ImageFont; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add v2/requirements.txt
git commit -m "chore(v2): add pillow dependency for OG image generation"
```

---

## Task 2: `render_homepage_meta` — homepage OG meta tags

**Files:**
- Create: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/v2/test_dashboard_pages.py`:

```python
"""Tests for v2/dashboard_pages.py — HTML page rendering."""

from decimal import Decimal

from v2.dashboard_pages import render_homepage_meta


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_pages.py -v`
Expected: `ModuleNotFoundError: No module named 'v2.dashboard_pages'`

- [ ] **Step 3: Write minimal implementation**

Create `v2/dashboard_pages.py`:

```python
"""HTML page rendering for the public dashboard.

Pure-ish functions that turn data dicts into HTML strings. No DB access; the
caller (v2/dashboard_publish.py) gathers data and passes it in.
"""

from decimal import Decimal
from string import Template

_HOMEPAGE_META_TEMPLATE = Template(
    '<meta property="og:title" content="$title" />\n'
    '<meta property="og:description" content="$description" />\n'
    '<meta property="og:image" content="$image_url" />\n'
    '<meta property="og:url" content="$page_url" />\n'
    '<meta name="twitter:card" content="summary_large_image" />\n'
    '<meta name="twitter:title" content="$title" />\n'
    '<meta name="twitter:description" content="$description" />\n'
    '<meta name="twitter:image" content="$image_url" />\n'
)


def _fmt_money(value) -> str:
    if value is None:
        return "$0.00"
    return f"${Decimal(value):,.2f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "0.00%"
    return f"{Decimal(value):+.2f}%"


def render_homepage_meta(summary: dict, base_url: str) -> str:
    """Return the OG/Twitter card meta block for the homepage."""
    title = "Bikini Bottom Capital"
    daily_pnl = summary.get("daily_pnl") or 0
    daily_pct = summary.get("daily_pnl_pct") or 0
    portfolio = summary.get("portfolio_value") or 0
    description = (
        f"Portfolio: {_fmt_money(portfolio)} · "
        f"Today: {_fmt_money(daily_pnl)} ({_fmt_pct(daily_pct)})"
    )
    return _HOMEPAGE_META_TEMPLATE.substitute(
        title=title,
        description=description,
        image_url=f"{base_url.rstrip('/')}/og/home.png",
        page_url=base_url.rstrip("/") + "/",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_pages.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_homepage_meta for OG/Twitter card injection"
```

---

## Task 3: `render_trade_page` — per-trade HTML

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_pages.py`:

```python
from datetime import date
from v2.dashboard_pages import render_trade_page


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_pages.py::TestRenderTradePage -v`
Expected: `ImportError: cannot import name 'render_trade_page'`

- [ ] **Step 3: Write minimal implementation**

Append to `v2/dashboard_pages.py`:

```python
from html import escape as _esc

_TRADE_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>$title — Bikini Bottom Capital</title>
<meta property="og:title" content="$title" />
<meta property="og:description" content="$description" />
<meta property="og:image" content="$og_image" />
<meta property="og:url" content="$page_url" />
<meta property="og:type" content="article" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="$title" />
<meta name="twitter:description" content="$description" />
<meta name="twitter:image" content="$og_image" />
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>$action_caps $ticker</h2>
<p class="trade-summary">$qty shares at \$$price on $trade_date</p>
<h3>Reasoning</h3>
<p>$reasoning</p>
$thesis_section
$outcome_section
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")

_THESIS_LINK_TEMPLATE = Template(
    '<h3>Thesis</h3>'
    '<p><a href="/thesis/$tid/">$thesis_text</a> '
    '<span class="thesis-meta">($direction, $confidence confidence)</span></p>'
)

_OUTCOME_TEMPLATE = Template(
    '<h3>Outcome</h3><p>7-day: $o7 · 30-day: $o30</p>'
)


def _fmt_outcome(value) -> str:
    if value is None:
        return "pending"
    return f"{Decimal(value):+.2f}%"


def render_trade_page(decision: dict, thesis: dict | None,
                      position: dict | None, base_url: str) -> str:
    """Return the full HTML page for one trade."""
    base = base_url.rstrip("/")
    ticker = _esc(str(decision["ticker"]))
    action = str(decision["action"]).lower()
    title = f"{action.upper()} {ticker}"
    qty = decision.get("quantity") or 0
    price = decision.get("price") or 0
    description = f"{action.upper()} {qty} {ticker} @ ${price}"

    if thesis:
        thesis_section = _THESIS_LINK_TEMPLATE.substitute(
            tid=thesis["id"],
            thesis_text=_esc(str(thesis.get("thesis", ""))),
            direction=_esc(str(thesis.get("direction", ""))),
            confidence=_esc(str(thesis.get("confidence", ""))),
        )
    else:
        thesis_section = ""

    if decision.get("outcome_7d") is not None or decision.get("outcome_30d") is not None:
        outcome_section = _OUTCOME_TEMPLATE.substitute(
            o7=_fmt_outcome(decision.get("outcome_7d")),
            o30=_fmt_outcome(decision.get("outcome_30d")),
        )
    else:
        outcome_section = ""

    return _TRADE_PAGE_TEMPLATE.substitute(
        title=title,
        action_caps=action.upper(),
        ticker=ticker,
        qty=qty,
        price=price,
        trade_date=decision["date"].isoformat() if hasattr(decision["date"], "isoformat") else str(decision["date"]),
        reasoning=_esc(str(decision.get("reasoning") or "")),
        thesis_section=thesis_section,
        outcome_section=outcome_section,
        description=_esc(description),
        og_image=f"{base}/og/trade/{decision['id']}.png",
        page_url=f"{base}/trade/{decision['id']}/",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_pages.py::TestRenderTradePage -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): render_trade_page emits per-trade HTML with OG meta"
```

---

## Task 4: `render_thesis_page` — per-thesis HTML

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_pages.py`:

```python
from v2.dashboard_pages import render_thesis_page


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_pages.py::TestRenderThesisPage -v`
Expected: `ImportError: cannot import name 'render_thesis_page'`

- [ ] **Step 3: Write minimal implementation**

Append to `v2/dashboard_pages.py`:

```python
_THESIS_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>$title — Bikini Bottom Capital</title>
<meta property="og:title" content="$title" />
<meta property="og:description" content="$description" />
<meta property="og:image" content="$og_image" />
<meta property="og:url" content="$page_url" />
<meta property="og:type" content="article" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="$title" />
<meta name="twitter:description" content="$description" />
<meta name="twitter:image" content="$og_image" />
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>$ticker — $direction thesis</h2>
<p class="thesis-meta">Confidence: $confidence · Status: $status</p>
<h3>Thesis</h3>
<p>$thesis_text</p>
$triggers_section
$decisions_section
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")


def _render_triggers_section(thesis: dict) -> str:
    parts = []
    if thesis.get("entry_trigger"):
        parts.append(f"<p><strong>Entry:</strong> {_esc(str(thesis['entry_trigger']))}</p>")
    if thesis.get("exit_trigger"):
        parts.append(f"<p><strong>Exit:</strong> {_esc(str(thesis['exit_trigger']))}</p>")
    if thesis.get("invalidation"):
        parts.append(f"<p><strong>Invalidation:</strong> {_esc(str(thesis['invalidation']))}</p>")
    if not parts:
        return ""
    return "<h3>Triggers</h3>" + "".join(parts)


def _render_decisions_section(decisions: list[dict]) -> str:
    if not decisions:
        return ""
    rows = []
    for d in decisions:
        rows.append(
            f'<li><a href="/trade/{d["id"]}/">{d["date"]} '
            f'{_esc(str(d["action"])).upper()} {d.get("quantity") or 0} @ ${d.get("price") or 0}</a></li>'
        )
    return "<h3>Related decisions</h3><ul>" + "".join(rows) + "</ul>"


def render_thesis_page(thesis: dict, decisions: list[dict],
                       position: dict | None, base_url: str) -> str:
    """Return the full HTML page for one thesis."""
    base = base_url.rstrip("/")
    ticker = _esc(str(thesis["ticker"]))
    direction = _esc(str(thesis.get("direction", "")))
    title = f"{ticker} — {direction} thesis"
    description = (str(thesis.get("thesis", ""))[:160]).replace("\n", " ")
    return _THESIS_PAGE_TEMPLATE.substitute(
        title=_esc(title),
        ticker=ticker,
        direction=direction,
        confidence=_esc(str(thesis.get("confidence", ""))),
        status=_esc(str(thesis.get("status", ""))),
        thesis_text=_esc(str(thesis.get("thesis", ""))),
        triggers_section=_render_triggers_section(thesis),
        decisions_section=_render_decisions_section(decisions),
        description=_esc(description),
        og_image=f"{base}/og/thesis/{thesis['id']}.png",
        page_url=f"{base}/thesis/{thesis['id']}/",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_pages.py::TestRenderThesisPage -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): render_thesis_page emits per-thesis HTML with OG meta"
```

---

## Task 5: Add OG_META placeholder + static OG tags to homepage

**Files:**
- Modify: `public_dashboard/index.html`

- [ ] **Step 1: Read the current head**

Run: `head -20 public_dashboard/index.html`

- [ ] **Step 2: Insert placeholder + static tags**

Edit `public_dashboard/index.html`. Replace the existing `<head>` block (lines 3-10) so it ends like this:

```html
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bikini Bottom Capital</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍍</text></svg>">
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <meta property="og:site_name" content="Bikini Bottom Capital" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
    <!-- OG_META -->
</head>
```

The `<!-- OG_META -->` placeholder is what the publish step will replace with the dynamic block.

- [ ] **Step 3: Verify file integrity**

Run: `python -c "import html.parser; html.parser.HTMLParser().feed(open('public_dashboard/index.html').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add public_dashboard/index.html
git commit -m "feat(public-dashboard): add OG meta placeholder and static og tags"
```

---

## Task 6: `render_trade_og` — per-trade OG PNG

**Files:**
- Create: `v2/dashboard_og.py`
- Test: `tests/v2/test_dashboard_og.py`

- [ ] **Step 1: Write the failing test**

Create `tests/v2/test_dashboard_og.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_og.py -v`
Expected: `ModuleNotFoundError: No module named 'v2.dashboard_og'`

- [ ] **Step 3: Write minimal implementation**

Create `v2/dashboard_og.py`:

```python
"""OG image generation for per-trade and per-thesis link previews.

Pure Pillow — no headless browser, no external service. Output is 1200x630
PNG bytes ready to write to the deploy directory.
"""

from decimal import Decimal
from io import BytesIO

OG_WIDTH = 1200
OG_HEIGHT = 630
_BG = (8, 24, 32)        # dark teal
_FG = (220, 240, 230)    # warm off-white
_ACCENT = (0, 212, 170)  # bikini-bottom green
_MUTED = (140, 160, 150)


def _canvas():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (OG_WIDTH, OG_HEIGHT), _BG)
    draw = ImageDraw.Draw(img)
    # Accent bar across the top
    draw.rectangle([(0, 0), (OG_WIDTH, 8)], fill=_ACCENT)
    # Footer line
    draw.text((48, OG_HEIGHT - 56), "BIKINI BOTTOM CAPITAL", fill=_MUTED)
    return img, draw


def _to_png_bytes(img) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_trade_og(decision: dict) -> bytes:
    """Return PNG bytes (1200x630) for the OG card of one trade."""
    img, draw = _canvas()
    ticker = str(decision.get("ticker", "?"))
    action = str(decision.get("action", "")).upper()
    qty = decision.get("quantity") or 0
    price = decision.get("price")
    price_str = f"${Decimal(price):,.2f}" if price is not None else ""

    draw.text((48, 80), action, fill=_ACCENT)
    draw.text((48, 130), ticker, fill=_FG)
    if qty:
        draw.text((48, 360), f"{qty} shares", fill=_FG)
    if price_str:
        draw.text((48, 410), price_str, fill=_MUTED)

    return _to_png_bytes(img)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_og.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_og.py tests/v2/test_dashboard_og.py
git commit -m "feat(v2): add render_trade_og for per-trade OG cards"
```

---

## Task 7: `render_thesis_og` — per-thesis OG PNG

**Files:**
- Modify: `v2/dashboard_og.py`
- Test: `tests/v2/test_dashboard_og.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_og.py`:

```python
from v2.dashboard_og import render_thesis_og


class TestRenderThesisOg:
    def test_returns_valid_png_bytes(self):
        thesis = {
            "id": 7, "ticker": "NVDA", "direction": "long",
            "confidence": "high",
            "thesis": "AI capex acceleration is real and unpriced.",
        }
        png = render_thesis_og(thesis)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_correct_dimensions(self):
        thesis = {
            "id": 7, "ticker": "NVDA", "direction": "long",
            "confidence": "high",
            "thesis": "AI capex acceleration is real and unpriced.",
        }
        img = _decoded(render_thesis_og(thesis))
        assert img.size == (1200, 630)

    def test_handles_long_thesis_text(self):
        thesis = {
            "id": 7, "ticker": "NVDA", "direction": "long",
            "confidence": "high",
            "thesis": "x" * 5000,  # Should not crash on huge text
        }
        png = render_thesis_og(thesis)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_og.py::TestRenderThesisOg -v`
Expected: `ImportError: cannot import name 'render_thesis_og'`

- [ ] **Step 3: Write minimal implementation**

Append to `v2/dashboard_og.py`:

```python
def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def render_thesis_og(thesis: dict) -> bytes:
    """Return PNG bytes (1200x630) for the OG card of one thesis."""
    img, draw = _canvas()
    ticker = str(thesis.get("ticker", "?"))
    direction = str(thesis.get("direction", "")).upper()
    confidence = str(thesis.get("confidence", "")).lower()
    thesis_text = _truncate(str(thesis.get("thesis", "")), 120)

    draw.text((48, 80), f"{direction} THESIS", fill=_ACCENT)
    draw.text((48, 130), ticker, fill=_FG)
    if confidence:
        draw.text((48, 220), f"confidence: {confidence}", fill=_MUTED)
    if thesis_text:
        # Wrap to ~60 chars per line manually
        wrapped = []
        current = ""
        for word in thesis_text.split(" "):
            if len(current) + len(word) + 1 > 60:
                wrapped.append(current)
                current = word
            else:
                current = (current + " " + word).strip()
        if current:
            wrapped.append(current)
        for i, line in enumerate(wrapped[:5]):
            draw.text((48, 320 + i * 36), line, fill=_FG)

    return _to_png_bytes(img)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_og.py::TestRenderThesisOg -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_og.py tests/v2/test_dashboard_og.py
git commit -m "feat(v2): add render_thesis_og for per-thesis OG cards"
```

---

## Task 8: `gather_trade_detail` — DB join for one trade

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
from v2.dashboard_publish import gather_trade_detail


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestGatherTradeDetail -v`
Expected: `ImportError: cannot import name 'gather_trade_detail'`

- [ ] **Step 3: Write minimal implementation**

Add to `v2/dashboard_publish.py` (after `gather_dashboard_data`):

```python
def gather_trade_detail(cur, decision_id: int) -> dict | None:
    """Return full detail for one decision page: decision + thesis + position.

    Caller passes a cursor so this can run in any open transaction.
    Returns None if the decision_id doesn't exist.
    """
    cur.execute(
        """
        SELECT id, date, ticker, action, quantity, price, reasoning,
               outcome_7d, outcome_30d, thesis_id, order_id
        FROM decisions WHERE id = %s
        """,
        (decision_id,),
    )
    decision = cur.fetchone()
    if decision is None:
        return None
    decision = dict(decision)
    decision["order_id"] = _redact_order_id(decision.get("order_id"))

    thesis = None
    if decision.get("thesis_id"):
        cur.execute(
            """
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, status
            FROM theses WHERE id = %s
            """,
            (decision["thesis_id"],),
        )
        row = cur.fetchone()
        thesis = dict(row) if row else None

    cur.execute(
        "SELECT ticker, shares, avg_cost FROM positions WHERE ticker = %s",
        (decision["ticker"],),
    )
    pos_row = cur.fetchone()
    position = dict(pos_row) if pos_row else None

    return {"decision": decision, "thesis": thesis, "position": position}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestGatherTradeDetail -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): gather_trade_detail joins decision/thesis/position"
```

---

## Task 9: `gather_thesis_detail` — DB join for one thesis

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
from v2.dashboard_publish import gather_thesis_detail


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
        ]

        result = gather_thesis_detail(mock_db, thesis_id=7)
        assert result["thesis"]["id"] == 7
        assert len(result["decisions"]) == 1
        assert result["position"]["ticker"] == "NVDA"

    def test_returns_none_when_missing(self, mock_db):
        mock_db.fetchone.side_effect = [None]
        result = gather_thesis_detail(mock_db, thesis_id=999)
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestGatherThesisDetail -v`
Expected: `ImportError: cannot import name 'gather_thesis_detail'`

- [ ] **Step 3: Write minimal implementation**

Add to `v2/dashboard_publish.py` (after `gather_trade_detail`):

```python
def gather_thesis_detail(cur, thesis_id: int) -> dict | None:
    """Return full detail for one thesis page: thesis + decisions + position."""
    cur.execute(
        """
        SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
               invalidation, confidence, status
        FROM theses WHERE id = %s
        """,
        (thesis_id,),
    )
    thesis = cur.fetchone()
    if thesis is None:
        return None
    thesis = dict(thesis)

    cur.execute(
        """
        SELECT id, date, ticker, action, quantity, price,
               outcome_7d, outcome_30d
        FROM decisions
        WHERE thesis_id = %s
        ORDER BY date DESC, id DESC
        """,
        (thesis_id,),
    )
    decisions = [dict(r) for r in cur.fetchall()]

    cur.execute(
        "SELECT ticker, shares, avg_cost FROM positions WHERE ticker = %s",
        (thesis["ticker"],),
    )
    pos_row = cur.fetchone()
    position = dict(pos_row) if pos_row else None

    return {"thesis": thesis, "decisions": decisions, "position": position}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestGatherThesisDetail -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): gather_thesis_detail joins thesis/decisions/position"
```

---

## Task 10: `gather_all_pages_data` — IDs for full-history page emission

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
from v2.dashboard_publish import gather_all_pages_data


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
        assert result == {"decision_ids": [], "thesis_ids": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestGatherAllPagesData -v`
Expected: `ImportError: cannot import name 'gather_all_pages_data'`

- [ ] **Step 3: Write minimal implementation**

Add to `v2/dashboard_publish.py`:

```python
def gather_all_pages_data(cur) -> dict:
    """Return ID lists for every decision and thesis we need to emit pages for.

    No date filter — Cloudflare Pages does full-bundle replacement on each
    deploy, so any URL not in this deploy will 404. Link permanence is a hard
    requirement of the audience-growth strategy.
    """
    cur.execute("SELECT id FROM decisions ORDER BY id")
    decision_ids = [r["id"] for r in cur.fetchall()]
    cur.execute("SELECT id FROM theses ORDER BY id")
    thesis_ids = [r["id"] for r in cur.fetchall()]
    return {"decision_ids": decision_ids, "thesis_ids": thesis_ids}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestGatherAllPagesData -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): gather_all_pages_data returns IDs for full-history emit"
```

---

## Task 11: Inject homepage OG meta during deploy assembly

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
import os
import tempfile


class TestInjectHomepageOgMeta:
    def test_replaces_placeholder_with_meta_block(self, tmp_path):
        from v2.dashboard_publish import inject_homepage_og_meta

        # Create a fake index.html with the placeholder
        index = tmp_path / "index.html"
        index.write_text(
            "<html><head>\n"
            "<title>x</title>\n"
            "<!-- OG_META -->\n"
            "</head><body></body></html>\n"
        )

        summary = {"portfolio_value": 12345, "daily_pnl": 100,
                   "daily_pnl_pct": 1, "last_updated": "2026-05-03"}
        inject_homepage_og_meta(str(tmp_path), summary, base_url="https://example.com")

        rendered = index.read_text()
        assert "<!-- OG_META -->" not in rendered
        assert '<meta property="og:title"' in rendered
        assert "https://example.com/og/home.png" in rendered

    def test_no_op_when_placeholder_missing(self, tmp_path):
        from v2.dashboard_publish import inject_homepage_og_meta

        index = tmp_path / "index.html"
        index.write_text("<html><head></head><body></body></html>")

        # Must not raise; just leaves the file alone.
        summary = {"portfolio_value": 0, "daily_pnl": 0,
                   "daily_pnl_pct": 0, "last_updated": "2026-05-03"}
        inject_homepage_og_meta(str(tmp_path), summary, base_url="https://example.com")

        assert "OG_META" not in index.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestInjectHomepageOgMeta -v`
Expected: `ImportError: cannot import name 'inject_homepage_og_meta'`

- [ ] **Step 3: Write minimal implementation**

Add to `v2/dashboard_publish.py`:

```python
from v2.dashboard_pages import (
    render_homepage_meta,
    render_thesis_page,
    render_trade_page,
)
from v2.dashboard_og import render_thesis_og, render_trade_og


def inject_homepage_og_meta(deploy_dir: str, summary: dict, base_url: str) -> None:
    """Replace the <!-- OG_META --> placeholder in deploy_dir/index.html."""
    index_path = os.path.join(deploy_dir, "index.html")
    with open(index_path) as f:
        html = f.read()
    if "<!-- OG_META -->" not in html:
        return
    block = render_homepage_meta(summary, base_url=base_url)
    html = html.replace("<!-- OG_META -->", block)
    with open(index_path, "w") as f:
        f.write(html)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestInjectHomepageOgMeta -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): inject_homepage_og_meta replaces placeholder at publish"
```

---

## Task 12: Emit per-trade and per-thesis HTML pages during assembly

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
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
                    side_effect=lambda cur, tid: self._stub_thesis_detail(tid)):
            stats = emit_detail_pages(
                mock_db,
                decision_ids=[1, 2],
                thesis_ids=[7],
                deploy_dir=str(tmp_path),
                base_url="https://example.com",
            )

        assert (tmp_path / "trade" / "1" / "index.html").is_file()
        assert (tmp_path / "trade" / "2" / "index.html").is_file()
        assert (tmp_path / "thesis" / "7" / "index.html").is_file()
        assert stats["trades_written"] == 2
        assert stats["theses_written"] == 1
        assert stats["failed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestEmitDetailPages -v`
Expected: `ImportError: cannot import name 'emit_detail_pages'`

- [ ] **Step 3: Write minimal implementation**

Add to `v2/dashboard_publish.py`:

```python
def emit_detail_pages(cur, decision_ids: list[int], thesis_ids: list[int],
                      deploy_dir: str, base_url: str) -> dict:
    """Render per-trade and per-thesis HTML pages into deploy_dir.

    Returns a stats dict: {trades_written, theses_written, failed}.
    Per-page failures are isolated: one bad render doesn't abort the run.
    """
    stats = {"trades_written": 0, "theses_written": 0, "failed": 0}

    for did in decision_ids:
        try:
            detail = gather_trade_detail(cur, did)
            if detail is None:
                continue
            html = render_trade_page(
                decision=detail["decision"],
                thesis=detail["thesis"],
                position=detail["position"],
                base_url=base_url,
            )
            page_dir = os.path.join(deploy_dir, "trade", str(did))
            os.makedirs(page_dir, exist_ok=True)
            with open(os.path.join(page_dir, "index.html"), "w") as f:
                f.write(html)
            stats["trades_written"] += 1
        except Exception:
            logger.warning("Failed to render trade page %s", did, exc_info=True)
            stats["failed"] += 1

    for tid in thesis_ids:
        try:
            detail = gather_thesis_detail(cur, tid)
            if detail is None:
                continue
            html = render_thesis_page(
                thesis=detail["thesis"],
                decisions=detail["decisions"],
                position=detail["position"],
                base_url=base_url,
            )
            page_dir = os.path.join(deploy_dir, "thesis", str(tid))
            os.makedirs(page_dir, exist_ok=True)
            with open(os.path.join(page_dir, "index.html"), "w") as f:
                f.write(html)
            stats["theses_written"] += 1
        except Exception:
            logger.warning("Failed to render thesis page %s", tid, exc_info=True)
            stats["failed"] += 1

    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestEmitDetailPages -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): emit_detail_pages writes per-trade and per-thesis HTML"
```

---

## Task 13: Emit OG PNG files alongside detail pages

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestEmitOgImages -v`
Expected: `ImportError: cannot import name 'emit_og_images'`

- [ ] **Step 3: Write minimal implementation**

Add to `v2/dashboard_publish.py`:

```python
def emit_og_images(cur, decision_ids: list[int], thesis_ids: list[int],
                   deploy_dir: str) -> dict:
    """Render OG PNGs for each decision and thesis into deploy_dir/og/."""
    stats = {"trades_written": 0, "theses_written": 0, "failed": 0}

    trade_dir = os.path.join(deploy_dir, "og", "trade")
    thesis_dir = os.path.join(deploy_dir, "og", "thesis")
    os.makedirs(trade_dir, exist_ok=True)
    os.makedirs(thesis_dir, exist_ok=True)

    for did in decision_ids:
        try:
            detail = gather_trade_detail(cur, did)
            if detail is None:
                continue
            png = render_trade_og(detail["decision"])
            with open(os.path.join(trade_dir, f"{did}.png"), "wb") as f:
                f.write(png)
            stats["trades_written"] += 1
        except Exception:
            logger.warning("Failed to render trade OG %s", did, exc_info=True)
            stats["failed"] += 1

    for tid in thesis_ids:
        try:
            detail = gather_thesis_detail(cur, tid)
            if detail is None:
                continue
            png = render_thesis_og(detail["thesis"])
            with open(os.path.join(thesis_dir, f"{tid}.png"), "wb") as f:
                f.write(png)
            stats["theses_written"] += 1
        except Exception:
            logger.warning("Failed to render thesis OG %s", tid, exc_info=True)
            stats["failed"] += 1

    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestEmitOgImages -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): emit_og_images writes per-trade/per-thesis PNGs"
```

---

## Task 14: Wire all emission steps into `assemble_deploy_dir` + `run_dashboard_stage`

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
class TestAssembleDeployDirEndToEnd:
    def test_emits_static_assets_json_pages_and_og(self, mock_db, tmp_path):
        from unittest.mock import patch as _patch
        from v2.dashboard_publish import assemble_deploy_dir

        # Set up a fake assets dir with the static files assemble_deploy_dir copies.
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "index.html").write_text("<html><head><!-- OG_META --></head></html>")
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

        def _stub_trade(cur, did):
            return {"decision": {
                "id": did, "ticker": "NVDA", "action": "buy",
                "quantity": Decimal("1"), "price": Decimal("100"),
                "date": date(2026, 5, 3), "reasoning": "x",
                "outcome_7d": None, "outcome_30d": None, "thesis_id": None,
                "order_id": None,
            }, "thesis": None, "position": None}

        def _stub_thesis(cur, tid):
            return {"thesis": {
                "id": tid, "ticker": "NVDA", "direction": "long",
                "confidence": "high", "thesis": "x",
                "entry_trigger": None, "exit_trigger": None, "invalidation": None,
                "status": "active",
            }, "decisions": [], "position": None}

        with _patch("v2.dashboard_publish.gather_trade_detail", side_effect=_stub_trade), \
             _patch("v2.dashboard_publish.gather_thesis_detail", side_effect=_stub_thesis):
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
        # Homepage OG meta injected
        assert "<!-- OG_META -->" not in (deploy / "index.html").read_text()
        assert '<meta property="og:title"' in (deploy / "index.html").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/v2/test_dashboard_publish.py::TestAssembleDeployDirEndToEnd -v`
Expected: `TypeError` or `KeyError` — `assemble_deploy_dir` doesn't accept `base_url` yet, doesn't read `_pages`, doesn't call the new emit functions.

- [ ] **Step 3: Modify `assemble_deploy_dir` and add a cursor-providing wrapper**

Replace the existing `assemble_deploy_dir` in `v2/dashboard_publish.py` with:

```python
def assemble_deploy_dir(data: dict, deploy_dir: str, assets_dir: str,
                        base_url: str = "") -> str:
    """Assemble the full deploy directory: static assets, JSON, detail pages, OG images.

    `data` must include a `_pages` key with `decision_ids` and `thesis_ids`
    (added by the extended `gather_dashboard_data` flow). When `_pages` is
    missing, only the static + JSON path runs (legacy behavior).
    """
    os.makedirs(deploy_dir, exist_ok=True)

    # Static assets
    for filename in _STATIC_ASSETS:
        src = os.path.join(assets_dir, filename)
        dst = os.path.join(deploy_dir, filename)
        shutil.copy2(src, dst)

    write_json_files(data, deploy_dir)

    # Inject homepage OG meta (no-op if placeholder absent)
    try:
        inject_homepage_og_meta(deploy_dir, data.get("summary", {}), base_url=base_url)
    except Exception:
        logger.warning("Failed to inject homepage OG meta", exc_info=True)

    # Per-trade / per-thesis pages + OG images
    pages = data.get("_pages")
    if pages and base_url:
        with get_cursor() as cur:
            page_stats = emit_detail_pages(
                cur,
                decision_ids=pages.get("decision_ids", []),
                thesis_ids=pages.get("thesis_ids", []),
                deploy_dir=deploy_dir,
                base_url=base_url,
            )
            og_stats = emit_og_images(
                cur,
                decision_ids=pages.get("decision_ids", []),
                thesis_ids=pages.get("thesis_ids", []),
                deploy_dir=deploy_dir,
            )
        logger.info("Detail pages: %s; OG images: %s", page_stats, og_stats)

    return deploy_dir
```

Also update `gather_dashboard_data` to add the `_pages` key. Find the existing `return` block at the end of `gather_dashboard_data` and modify it:

```python
    # ... existing snapshot/positions/decisions/theses gathering ...

    # NEW: gather full ID lists for page emission (link permanence)
    with get_cursor() as cur:
        pages = gather_all_pages_data(cur)

    return {
        "summary": summary,
        "snapshots": snapshot_dicts,
        "positions": [dict(r) for r in positions],
        "decisions": [
            {**dict(r), "order_id": _redact_order_id(dict(r).get("order_id"))}
            for r in decisions
        ],
        "theses": [dict(r) for r in theses],
        "benchmark": benchmark,
        "_pages": pages,
    }
```

(Note: `_pages` starts with `_` so `write_json_files` skips it — that loop iterates a fixed key tuple.)

Finally, update `run_dashboard_stage` to pass `base_url`:

```python
def run_dashboard_stage(session_date: date | None = None) -> DashboardStageResult:
    # ... existing setup ...

    base_url = os.environ.get("DASHBOARD_URL", "").rstrip("/")

    # ... existing gather call ...

    deploy_dir = tempfile.mkdtemp(prefix="dashboard_deploy_")
    try:
        assemble_deploy_dir(data, deploy_dir, _ASSETS_DIR, base_url=base_url)
    except Exception as e:
        # ... existing error handling ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/v2/test_dashboard_publish.py -v`
Expected: All tests pass, including legacy `TestGatherDashboardData` tests (with the `_pages` key now in the result — update existing tests if they assert `set(result.keys()) == {...}`).

If legacy tests fail on the `_pages` key, update the existing assertion in `TestGatherDashboardData::test_returns_all_sections`:

```python
        assert set(result.keys()) == {
            "summary", "snapshots", "positions", "decisions",
            "theses", "benchmark", "_pages",
        }
```

And update `test_query_count` to add 2 more execute calls (for the new `gather_all_pages_data`):

```python
        assert mock_db.execute.call_count == 9   # was 7
```

Also extend the `fetchall.side_effect` lists in legacy tests to include the two new calls:

```python
        mock_db.fetchall.side_effect = [
            # ... existing 4 lists ...
            [],  # decisions ID list
            [],  # theses ID list
        ]
```

Apply this fix to every test in `TestGatherDashboardData` that sets `fetchall.side_effect`.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): wire detail pages + OG images into assemble_deploy_dir"
```

---

## Task 15: Final integration check + ruff/coverage

**Files:**
- None modified; verification only.

- [ ] **Step 1: Run the full v2 test suite**

Run: `python -m pytest tests/v2/ -v`
Expected: All tests pass.

- [ ] **Step 2: Run linter (if configured)**

Run: `ruff check v2/dashboard_pages.py v2/dashboard_og.py v2/dashboard_publish.py`
Expected: No errors. Fix any reported issues.

- [ ] **Step 3: Smoke-test the publish flow end-to-end (paper account, dry-run)**

The simplest end-to-end test that covers everything is a real publish to paper. Skip the wrangler step by setting `CLOUDFLARE_PAGES_PROJECT=""` and inspecting the generated tempdir:

```bash
docker compose exec trading python -c "
from v2.dashboard_publish import gather_dashboard_data, assemble_deploy_dir, _ASSETS_DIR
from datetime import date
import tempfile, os
data = gather_dashboard_data(date.today())
deploy = tempfile.mkdtemp(prefix='og_smoke_')
assemble_deploy_dir(data, deploy, _ASSETS_DIR, base_url='https://example.com')
print('Deploy dir:', deploy)
print('Files:')
for root, _, files in os.walk(deploy):
    for f in files:
        print(' ', os.path.join(root, f).replace(deploy, ''))
"
```

Expected output: A tree containing `index.html`, `styles.css`, `app.js`, `data/*.json`, `trade/*/index.html`, `thesis/*/index.html`, `og/trade/*.png`, `og/thesis/*.png`. Verify the homepage `index.html` has OG meta tags substituted (no `<!-- OG_META -->` marker).

- [ ] **Step 4: Verify a generated OG image opens**

Pick any trade PNG from the smoke run and confirm it's a valid 1200×630 image:

```bash
python -c "
from PIL import Image
import sys, glob, os
deploy = sys.argv[1]
pngs = glob.glob(os.path.join(deploy, 'og', 'trade', '*.png'))
if pngs:
    img = Image.open(pngs[0])
    print(pngs[0], img.size)
" /tmp/og_smoke_<dir>
```

Expected: `(1200, 630)` printed.

- [ ] **Step 5: Final commit (only if any fixes were applied above)**

```bash
git status
# If anything is modified by ruff/lint fixes:
git add -A
git commit -m "chore(v2): lint fixes for dashboard pages + OG modules"
```

If there are no changes, skip this commit.

---

## Self-Review Notes

**Spec coverage check (Spec #1 → tasks):**

| Spec requirement | Implementing task |
|---|---|
| Per-trade pages at `/trade/<id>/index.html` | Task 3 (rendering), Task 12 (emission) |
| Per-thesis pages at `/thesis/<id>/index.html` | Task 4, Task 12 |
| Dynamic OG PNGs at `/og/trade/<id>.png`, `/og/thesis/<id>.png` | Task 6, Task 7, Task 13 |
| OG meta tags on every page (homepage + per-page) | Task 2, Task 3, Task 4, Task 5, Task 11 |
| New `v2/dashboard_pages.py` module | Tasks 2-4 |
| New `v2/dashboard_og.py` module | Tasks 6-7 |
| `gather_trade_detail`, `gather_thesis_detail` | Tasks 8-9 |
| `assemble_deploy_dir` extension | Task 14 |
| Pillow dep | Task 1 |
| `<!-- OG_META -->` placeholder in `public_dashboard/index.html` | Task 5 |
| Per-page render error isolation | Task 12 (try/except per page) |
| Emit pages for ALL decisions/theses (not 30-day) | Task 10 |

**Deferred to follow-up (called out in spec open questions):**
- Typography polish (TTF font bundling)
- `/positions/<ticker>/` pages
- `sitemap.xml` and `robots.txt`
- 20,000-file Cloudflare ceiling sanity check (do this once before merge by querying actual decision count: `SELECT COUNT(*) FROM decisions; SELECT COUNT(*) FROM theses;` — if combined ≥ 4,000, redesign before deploying)
- >50% page failure threshold (Task 12 isolates per-page failures; the >50% guard from the spec is deferred since each render call is a try/except already, and the existing `DashboardStageResult.errors` channel surfaces any wholesale failure of the assembly step)

**Type consistency check:**
- `gather_trade_detail` returns `{"decision", "thesis", "position"}` — used identically in Tasks 12, 13. ✓
- `gather_thesis_detail` returns `{"thesis", "decisions", "position"}` — used identically in Tasks 12, 13. ✓
- `emit_detail_pages` and `emit_og_images` both return `{"trades_written", "theses_written", "failed"}`. ✓
- `assemble_deploy_dir` signature change: adds `base_url` kwarg (default `""`) — backwards compatible with the existing `run_dashboard_stage` caller, which Task 14 also updates.

**Placeholder scan:** No "TBD", "TODO", or "implement later" entries. All tasks contain executable code.
