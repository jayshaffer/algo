# Public Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the single-scroll public dashboard into a 5-page static site (Home / Performance / Activity / Learning / How-it-works) with shared nav, the P1 terminal palette, and dedicated drilldown pages — without changing the publish pipeline or DB schema.

**Architecture:** Render every page through a shared `_render_page_shell(...)` helper in `v2/dashboard_pages.py`. The homepage (currently hand-authored `public_dashboard/index.html`) becomes a Python renderer like every other page. New pages (`/performance/`, `/activity/`, `/learning/`, `/how-it-works/`) reuse the same shell. Existing permalinks (`/mistakes/`, `/attribution/`, `/trade/<id>/`, `/thesis/<id>/`) get re-skinned to use the shell. Sparkline becomes server-rendered SVG to avoid Chart.js on the homepage.

**Tech Stack:** Python 3 (`string.Template` + raw HTML — same pattern as existing renderers), CSS variables for the new palette, vanilla JS in `public_dashboard/app.js` with a `data-page` attribute for per-page initialization, pytest with the existing dashboard test patterns.

**Spec:** `docs/superpowers/specs/2026-05-04-public-dashboard-redesign-design.md`

---

## File Map

**Modify:**
- `v2/dashboard_pages.py` — add `_render_page_shell`, `render_homepage`, `render_performance_page`, `render_activity_page`, `render_learning_hub`, `render_how_it_works_hub`. Update existing `render_trade_page`, `render_thesis_page`, `render_mistakes_page`, `render_attribution_page` to use the shell.
- `v2/dashboard_publish.py` — add `render_sparkline_svg`, write `memos.json` + `performance.json` from `gather_dashboard_data`, render new pages in `assemble_deploy_dir`, drop `index.html` from `_STATIC_ASSETS`, drop `inject_homepage_og_meta` (the homepage now renders its own meta).
- `public_dashboard/styles.css` — replace palette, add new shared component classes, drop wave-divider / caustics CSS.
- `public_dashboard/app.js` — split into per-page init driven by a `data-page` attribute on `<body>`.
- `tests/v2/test_dashboard_pages.py` — add tests for every new renderer + shell helper + re-skinned existing renderers.
- `tests/v2/test_dashboard_publish.py` — add tests for `render_sparkline_svg`, memos.json/performance.json emission, new-page emission in `assemble_deploy_dir`.

**Delete:**
- `public_dashboard/index.html` — replaced by `render_homepage()`.

**No changes:** `v2/dashboard_og.py` (OG image renderers stay as-is), database schema, JSON file contents (only adding two new files), publish flow / wrangler invocation.

---

## Task 1: P1 palette + shared CSS components

Lay down the visual foundation first — every renderer in later tasks references these classes.

**Files:**
- Modify: `public_dashboard/styles.css` (full rewrite)

- [ ] **Step 1: Replace `styles.css` with P1 palette + shared classes**

```css
/* === Reset & Base === */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-deep: #0d1117;
  --bg-card: #161b22;
  --bg-card-alt: #1c2129;
  --border: #30363d;
  --text: #c9d1d9;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --gain: #3fb950;
  --loss: #f85149;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: "SF Mono", "Cascadia Code", "Fira Code", monospace;
}

body {
  background: var(--bg-deep);
  color: var(--text);
  font-family: var(--font-body);
  line-height: 1.6;
  min-height: 100vh;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.container { max-width: 1100px; margin: 0 auto; padding: 0 1.5rem; }

/* === Site nav === */
.site-nav {
  position: sticky; top: 0; z-index: 50;
  background: var(--bg-deep); border-bottom: 1px solid var(--border);
}
.site-nav .container {
  display: flex; align-items: center; justify-content: space-between;
  padding-top: 0.9rem; padding-bottom: 0.9rem;
}
.site-nav .logo {
  color: var(--accent); font-weight: 700; font-size: 1rem;
  letter-spacing: -0.01em;
}
.site-nav .links a {
  color: var(--text-dim); margin-left: 1.2rem; font-size: 0.85rem;
  padding-bottom: 0.25rem;
}
.site-nav .links a.active {
  color: var(--text); border-bottom: 2px solid var(--accent);
}
.site-nav .hamburger {
  display: none; background: none; border: none; color: var(--text);
  font-size: 1.4rem; cursor: pointer;
}

@media (max-width: 640px) {
  .site-nav .links { display: none; flex-direction: column; }
  .site-nav .links.open { display: flex; }
  .site-nav .hamburger { display: block; }
  .site-nav .container { flex-wrap: wrap; }
  .site-nav .links {
    width: 100%; padding-top: 0.6rem; gap: 0.4rem;
  }
  .site-nav .links a { margin-left: 0; }
}

/* === Hero === */
.hero { padding: 2rem 0 1.5rem; }
.hero .tag {
  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-dim); margin-bottom: 0.4rem;
}
.hero h1 {
  font-family: var(--font-mono); font-size: 1.75rem; font-weight: 700;
  letter-spacing: -0.01em;
}
.hero h1 .strip {
  font-size: 0.95rem; font-weight: 600; margin-left: 0.6rem;
}
.hero .label {
  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-dim); margin: 1rem 0 0.4rem;
}
.hero .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }

/* === Chip === */
.chip {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 4px; padding: 0.4rem 0.6rem; font-size: 0.8rem;
}
.chip:hover { background: var(--bg-card-alt); }
.chip .ticker { font-family: var(--font-mono); color: var(--accent); font-weight: 700; }

/* === Sparkline === */
.sparkline {
  display: block; width: 100%; height: 60px;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px;
  margin-top: 0.8rem;
}
.sparkline polyline { fill: none; stroke: var(--accent); stroke-width: 1.5; }

/* === Stat strip === */
.stat-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.6rem; margin: 1rem 0;
}
.stat {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.7rem 0.9rem;
}
.stat .lbl {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--text-dim);
}
.stat .val {
  font-family: var(--font-mono); font-size: 1.1rem; font-weight: 700;
  margin-top: 0.2rem;
}

/* === Section === */
.section { padding: 1.25rem 0; border-top: 1px solid var(--border); }
.section .head {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 0.7rem;
}
.section h2 { font-size: 1rem; font-weight: 600; }
.section .more { font-size: 0.8rem; }

/* === Move card (single-decision) === */
.move-card {
  display: block; background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.85rem 1rem; color: var(--text);
}
.move-card:hover { background: var(--bg-card-alt); text-decoration: none; }
.move-card .head { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
.move-card .ticker { font-family: var(--font-mono); font-weight: 700; color: var(--accent); }
.move-card .reasoning { color: var(--text-dim); font-size: 0.85rem; margin-top: 0.4rem; }

/* === Card grid (hub pages, homepage 2-up) === */
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.8rem; }
.card {
  display: block; background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.95rem; color: var(--text);
}
.card:hover { background: var(--bg-card-alt); text-decoration: none; }
.card.disabled { opacity: 0.5; cursor: not-allowed; }
.card .lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); }
.card h3 { font-size: 0.95rem; margin: 0.4rem 0 0.3rem; }
.card p { font-size: 0.85rem; color: var(--text-dim); }

/* === Memo block === */
.memo-block {
  background: var(--bg-card); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
  padding: 0.85rem 1rem; font-style: italic; color: var(--text-dim);
  font-size: 0.9rem;
}
.memo-block .meta {
  font-style: normal; font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-dim); margin-bottom: 0.4rem;
}

/* === Tables === */
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
thead th {
  text-align: left; padding: 0.7rem 1rem; color: var(--text-dim);
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
  font-weight: 500; border-bottom: 1px solid var(--border);
  background: var(--bg-card); position: sticky; top: 0;
}
thead th.num { text-align: right; }
tbody td {
  padding: 0.7rem 1rem; border-bottom: 1px solid var(--border);
  vertical-align: top;
}
tbody td.num { text-align: right; font-family: var(--font-mono); }
tbody tr:hover { background: var(--bg-card-alt); }

/* === Action badges === */
.badge {
  display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
  font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
}
.badge-buy { background: rgba(63, 185, 80, 0.15); color: var(--gain); }
.badge-sell { background: rgba(248, 81, 73, 0.15); color: var(--loss); }
.badge-hold { background: rgba(139, 148, 158, 0.15); color: var(--text-dim); }

/* === P&L coloring === */
.gain { color: var(--gain); }
.loss { color: var(--loss); }

/* === Empty state === */
.empty-state { padding: 2rem 1rem; text-align: center; color: var(--text-dim); font-size: 0.875rem; }

/* === Footer === */
footer { border-top: 1px solid var(--border); padding: 2rem 0; margin-top: 2rem; text-align: center; }
footer p { color: var(--text-dim); font-size: 0.85rem; }
footer .attribution { font-size: 0.75rem; margin-top: 0.25rem; }

/* === Methodology footer strip === */
.methodology-strip {
  margin: 1.5rem 0 0; padding: 0.85rem 1rem;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px;
  text-align: center; font-size: 0.85rem; color: var(--text-dim);
}
.methodology-strip a { margin: 0 0.3rem; }

/* === Chart wrap === */
.chart-wrap { padding: 1rem; }

/* === Active theses cards === */
.thesis-card { padding: 1rem; border-bottom: 1px solid var(--border); }
.thesis-card:last-child { border-bottom: none; }
.thesis-card .head { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.4rem; }
.thesis-card .ticker { font-family: var(--font-mono); font-weight: 700; color: var(--accent); }
.thesis-card .direction { font-size: 0.7rem; text-transform: uppercase; font-weight: 600; }
.thesis-card .direction.long { color: var(--gain); }
.thesis-card .direction.short { color: var(--loss); }
.thesis-card .body { color: var(--text-dim); font-size: 0.875rem; margin-bottom: 0.4rem; }
.thesis-card .triggers { font-size: 0.75rem; color: var(--text-dim); font-family: var(--font-mono); }

/* === Reasoning cell === */
.reasoning-cell { max-width: 400px; font-size: 0.8rem; line-height: 1.5; }

/* === Order ID === */
.order-id { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-dim); }
```

- [ ] **Step 2: Visually verify locally**

Run: `cd /home/jay/dev/algo/public_dashboard && python3 -m http.server 8080`

Open `http://localhost:8080`. The page will be visibly broken (the old hand-authored `index.html` is still in place, expecting old class names). That's expected — confirm the palette has switched (slate background, blue accent), then stop the server. You'll fix the markup in later tasks.

- [ ] **Step 3: Commit**

```bash
git add public_dashboard/styles.css
git commit -m "feat(dashboard): replace palette with P1 terminal + shared component classes"
```

---

## Task 2: `_render_page_shell` helper

Foundation for every renderer. Every later task depends on this.

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_pages.py`:

```python
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
        # Active item carries class="active"
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
        # Permalinks like /mistakes/ want active_nav="learning" (for the
        # nav underline) but data-page="mistakes" (for app.js dispatch).
        html = self._shell(active_nav="learning", data_page="mistakes")
        assert 'data-page="mistakes"' in html
        assert 'class="active" href="/learning/"' in html

    def test_loads_app_js(self):
        html = self._shell()
        assert '<script src="/app.js"></script>' in html
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderPageShell -v`
Expected: ImportError or AttributeError on `_render_page_shell`.

- [ ] **Step 3: Implement `_render_page_shell`**

Add to `v2/dashboard_pages.py` (place after `_render_meta_block`):

```python
_NAV_ITEMS = (
    ("home", "/", "Home"),
    ("performance", "/performance/", "Performance"),
    ("activity", "/activity/", "Activity"),
    ("learning", "/learning/", "Learning"),
    ("how-it-works", "/how-it-works/", "How it works"),
)


def _render_nav(active_nav: str) -> str:
    parts = ['<nav class="site-nav"><div class="container">']
    parts.append('<span class="logo">⌬ Bikini Bottom Capital</span>')
    parts.append('<button class="hamburger" aria-label="Menu">☰</button>')
    parts.append('<div class="links">')
    for key, href, label in _NAV_ITEMS:
        cls = ' class="active"' if key == active_nav else ''
        parts.append(f'<a{cls} href="{href}">{label}</a>')
    parts.append('</div></div></nav>')
    return "".join(parts)


_FOOTER_HTML = (
    '<footer><div class="container">'
    '<p>Is mayonnaise a financial instrument?</p>'
    '<p class="attribution">Data from '
    '<a href="https://alpaca.markets" target="_blank" rel="noopener">Alpaca</a></p>'
    '</div></footer>'
)


_PAGE_SHELL_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>$title — Bikini Bottom Capital</title>
$meta_block
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍍</text></svg>" />
<link rel="stylesheet" href="/styles.css" />
$head_extra
</head>
<body data-page="$active_nav">
$nav
<main class="container">
$content
</main>
$footer
<script src="/app.js"></script>
</body>
</html>
""")


def _render_page_shell(*, title: str, description: str, active_nav: str,
                       content: str, og_image: str, page_url: str,
                       og_type: str = "website",
                       head_extra: str = "",
                       data_page: str | None = None) -> str:
    """Wrap page content in the shared <html> + nav + footer scaffolding.

    `data_page` overrides what's emitted as `<body data-page="…">`. Defaults
    to `active_nav`. Use it on permalink pages where the nav highlight (e.g.
    "learning") differs from the app.js dispatch key (e.g. "mistakes").
    """
    meta_block = _render_meta_block(
        title=_esc(title),
        description=_esc(description),
        og_image=og_image,
        page_url=page_url,
        og_type=og_type,
    )
    return _PAGE_SHELL_TEMPLATE.substitute(
        title=_esc(title),
        meta_block=meta_block,
        head_extra=head_extra,
        active_nav=_esc(data_page or active_nav),
        nav=_render_nav(active_nav),
        content=content,
        footer=_FOOTER_HTML,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderPageShell -v`
Expected: 10 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add _render_page_shell — shared scaffolding for all dashboard pages"
```

---

## Task 3: Sparkline SVG helper

Server-rendered inline SVG for the homepage hero. Replaces what would otherwise need Chart.js.

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_publish.py`:

```python
from v2.dashboard_publish import render_sparkline_svg


class TestRenderSparklineSvg:
    def test_returns_empty_string_when_too_few_points(self):
        # Need at least 7 points (one week) per spec.
        snapshots = [{"value": 100} for _ in range(6)]
        assert render_sparkline_svg(snapshots) == ""

    def test_emits_svg_with_polyline(self):
        snapshots = [{"value": 100 + i} for i in range(30)]
        svg = render_sparkline_svg(snapshots)
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "<polyline" in svg
        assert 'class="sparkline"' in svg
        assert 'viewBox="0 0 400 60"' in svg

    def test_polyline_has_one_point_per_snapshot(self):
        snapshots = [{"value": 100 + i} for i in range(30)]
        svg = render_sparkline_svg(snapshots)
        # Each point is "x,y " — count spaces in the points attr.
        import re
        m = re.search(r'points="([^"]+)"', svg)
        assert m, "no points attr"
        points = m.group(1).strip().split()
        assert len(points) == 30

    def test_caps_at_90_points_when_more_supplied(self):
        snapshots = [{"value": 100 + i} for i in range(150)]
        svg = render_sparkline_svg(snapshots)
        import re
        m = re.search(r'points="([^"]+)"', svg)
        points = m.group(1).strip().split()
        assert len(points) == 90

    def test_y_values_within_viewbox_bounds(self):
        snapshots = [{"value": v} for v in [100, 105, 95, 110, 90, 120, 80]]
        svg = render_sparkline_svg(snapshots)
        import re
        m = re.search(r'points="([^"]+)"', svg)
        for pair in m.group(1).strip().split():
            x, y = pair.split(",")
            assert 0 <= float(x) <= 400
            assert 0 <= float(y) <= 60

    def test_handles_flat_series(self):
        # All identical values — must not div-by-zero on normalization.
        snapshots = [{"value": 100} for _ in range(10)]
        svg = render_sparkline_svg(snapshots)
        assert "<polyline" in svg
        # All Y's should equal mid-line (30) when min == max.
        import re
        m = re.search(r'points="([^"]+)"', svg)
        for pair in m.group(1).strip().split():
            _, y = pair.split(",")
            assert float(y) == 30.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestRenderSparklineSvg -v`
Expected: ImportError on `render_sparkline_svg`.

- [ ] **Step 3: Implement `render_sparkline_svg`**

Add to `v2/dashboard_publish.py` (place near other rendering helpers, before `gather_dashboard_data`):

```python
def render_sparkline_svg(snapshots: list[dict]) -> str:
    """Render the last 90 days of equity as an inline SVG polyline.

    Returns "" when fewer than 7 snapshots are supplied — the homepage
    template hides the sparkline in that case.
    """
    if not snapshots or len(snapshots) < 7:
        return ""

    series = [float(s["value"]) for s in snapshots[-90:]]
    n = len(series)
    lo, hi = min(series), max(series)
    span = hi - lo if hi > lo else 1.0  # avoid div-by-zero on flat series
    flat = (hi == lo)

    width, height, pad_y = 400.0, 60.0, 5.0
    plot_h = height - 2 * pad_y

    points = []
    for i, v in enumerate(series):
        x = (i / (n - 1)) * width if n > 1 else width / 2
        if flat:
            y = height / 2
        else:
            # Higher value -> lower Y (SVG y-axis grows downward).
            y = pad_y + (1.0 - (v - lo) / span) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    return (
        f'<svg class="sparkline" viewBox="0 0 400 60" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{" ".join(points)}" />'
        f'</svg>'
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestRenderSparklineSvg -v`
Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): add render_sparkline_svg — server-side equity sparkline"
```

---

## Task 4: Memos JSON emission

Activity page needs a `memos.json` file with the last 10 strategy memos. Backed by the existing `get_recent_strategy_memos` query.

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_publish.py`:

```python
from unittest.mock import patch


class TestGatherMemos:
    def test_gather_dashboard_data_includes_memos(self):
        from v2.dashboard_publish import gather_dashboard_data

        fake_memos = [
            {"id": 9, "session_date": date(2026, 5, 4),
             "memo_type": "session", "content": "Holding the AI book."},
            {"id": 8, "session_date": date(2026, 5, 3),
             "memo_type": "session", "content": "Macro chop unresolved."},
        ]
        # Patch every DB call gather_dashboard_data makes; only memos matter
        # for this test, but the others must not fail.
        with patch("v2.dashboard_publish.get_recent_strategy_memos",
                   return_value=fake_memos), \
             patch("v2.dashboard_publish.get_cursor"), \
             patch("v2.dashboard_publish.get_signal_attribution", return_value=[]), \
             patch("v2.dashboard_publish.get_closed_losers", return_value=[]), \
             patch("v2.dashboard_publish.get_retired_rules", return_value=[]), \
             patch("v2.dashboard_publish.get_net_deposits",
                   return_value=Decimal("0")), \
             patch("v2.dashboard_publish.get_deposit_history", return_value=[]), \
             patch("v2.dashboard_publish.fetch_benchmark_data", return_value=[]):
            data = gather_dashboard_data(date(2026, 5, 4))

        assert "memos" in data
        assert len(data["memos"]) == 2
        assert data["memos"][0]["content"] == "Holding the AI book."

    def test_write_json_files_emits_memos(self, tmp_path):
        from v2.dashboard_publish import write_json_files

        data = {
            "memos": [{"id": 1, "session_date": date(2026, 5, 4),
                       "content": "test"}],
            # Other expected keys with empty values:
            "summary": {}, "snapshots": [], "positions": [], "decisions": [],
            "theses": [], "benchmark": [], "mistakes": {},
            "attribution": [], "performance": {},
        }
        write_json_files(data, str(tmp_path))
        memos_file = tmp_path / "memos.json"
        assert memos_file.exists()
        import json
        with memos_file.open() as f:
            payload = json.load(f)
        assert payload[0]["content"] == "test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestGatherMemos -v`
Expected: KeyError or assertion failure (no `memos` key emitted).

- [ ] **Step 3: Wire memos into `gather_dashboard_data` and `write_json_files`**

In `v2/dashboard_publish.py`:

Add import near the top:
```python
from .database.trading_db import (
    get_closed_losers,
    get_recent_strategy_memos,  # NEW
    get_retired_rules,
    get_signal_attribution,
)
```

Inside `gather_dashboard_data`, after the existing `get_signal_attribution()` call (or any sensible spot before the `return`), add:
```python
    memos = get_recent_strategy_memos(n=10)
    data["memos"] = [
        {
            "id": m["id"],
            "session_date": m["session_date"],
            "memo_type": m.get("memo_type"),
            "content": m["content"],
        }
        for m in memos
    ]
```

In `write_json_files` (find it in the same file), add a new file emission alongside the existing ones:
```python
    with open(os.path.join(deploy_dir, "memos.json"), "w") as f:
        json.dump(data.get("memos", []), f, cls=_DecimalEncoder)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestGatherMemos -v`
Expected: 2 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): emit memos.json from gather_dashboard_data"
```

---

## Task 5: Performance stats JSON

Performance page needs derived stats (max drawdown, win rate, avg days held, best/worst day).

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_publish.py`:

```python
class TestComputePerformanceStats:
    def test_empty_inputs_produce_zero_struct(self):
        from v2.dashboard_publish import compute_performance_stats
        stats = compute_performance_stats(snapshots=[], decisions=[])
        assert stats == {
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_days_held": 0.0,
            "best_day_pct": 0.0,
            "worst_day_pct": 0.0,
        }

    def test_max_drawdown_basic(self):
        from v2.dashboard_publish import compute_performance_stats
        snapshots = [
            {"snapshot_date": date(2026, 1, 1), "value": Decimal("100")},
            {"snapshot_date": date(2026, 1, 2), "value": Decimal("110")},
            {"snapshot_date": date(2026, 1, 3), "value": Decimal("90")},
            {"snapshot_date": date(2026, 1, 4), "value": Decimal("95")},
        ]
        stats = compute_performance_stats(snapshots=snapshots, decisions=[])
        # Peak 110 -> trough 90 = -18.18%
        assert abs(stats["max_drawdown_pct"] - (-18.181818)) < 0.01

    def test_best_and_worst_day(self):
        from v2.dashboard_publish import compute_performance_stats
        snapshots = [
            {"snapshot_date": date(2026, 1, 1), "value": Decimal("100")},
            {"snapshot_date": date(2026, 1, 2), "value": Decimal("105")},  # +5%
            {"snapshot_date": date(2026, 1, 3), "value": Decimal("95")},   # -9.52%
        ]
        stats = compute_performance_stats(snapshots=snapshots, decisions=[])
        assert abs(stats["best_day_pct"] - 5.0) < 0.01
        assert abs(stats["worst_day_pct"] - (-9.523809)) < 0.01

    def test_win_rate_from_closed_decisions(self):
        from v2.dashboard_publish import compute_performance_stats
        decisions = [
            {"outcome_30d_pct": Decimal("3.0")},   # win
            {"outcome_30d_pct": Decimal("-2.0")},  # loss
            {"outcome_30d_pct": Decimal("1.0")},   # win
            {"outcome_30d_pct": None},             # ignored — open
        ]
        stats = compute_performance_stats(snapshots=[], decisions=decisions)
        assert abs(stats["win_rate_pct"] - 66.666666) < 0.01

    def test_write_json_files_emits_performance(self, tmp_path):
        from v2.dashboard_publish import write_json_files
        data = {
            "performance": {"max_drawdown_pct": -5.0, "win_rate_pct": 60.0,
                            "avg_days_held": 4.0, "best_day_pct": 2.0,
                            "worst_day_pct": -3.0},
            "summary": {}, "snapshots": [], "positions": [], "decisions": [],
            "theses": [], "benchmark": [], "mistakes": {}, "memos": [],
            "attribution": [],
        }
        write_json_files(data, str(tmp_path))
        f = tmp_path / "performance.json"
        assert f.exists()
        import json
        with f.open() as fh:
            payload = json.load(fh)
        assert payload["win_rate_pct"] == 60.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestComputePerformanceStats -v`
Expected: ImportError on `compute_performance_stats`.

- [ ] **Step 3: Implement `compute_performance_stats` and wire it in**

Add to `v2/dashboard_publish.py`:

```python
def compute_performance_stats(*, snapshots: list[dict],
                              decisions: list[dict]) -> dict:
    """Derive Performance-page stats from raw snapshots and closed decisions."""
    if not snapshots:
        return {
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_days_held": 0.0,
            "best_day_pct": 0.0,
            "worst_day_pct": 0.0,
        }

    values = [float(s["value"]) for s in snapshots]

    # Max drawdown: largest peak-to-trough drop.
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100.0 if peak else 0.0
        if dd < max_dd:
            max_dd = dd

    # Best / worst day-over-day return.
    best, worst = 0.0, 0.0
    for prev, curr in zip(values, values[1:]):
        if prev <= 0:
            continue
        pct = (curr - prev) / prev * 100.0
        if pct > best:
            best = pct
        if pct < worst:
            worst = pct

    # Win rate: closed decisions with non-null outcome_30d_pct, fraction > 0.
    closed = [d for d in decisions if d.get("outcome_30d_pct") is not None]
    if closed:
        wins = sum(1 for d in closed if float(d["outcome_30d_pct"]) > 0)
        win_rate = wins / len(closed) * 100.0
    else:
        win_rate = 0.0

    # Avg days held: only meaningful when we have entry+exit data.
    # For now, surface 0 — this is wired in once the decisions table exposes it.
    avg_days_held = 0.0

    return {
        "max_drawdown_pct": round(max_dd, 4),
        "win_rate_pct": round(win_rate, 4),
        "avg_days_held": round(avg_days_held, 4),
        "best_day_pct": round(best, 4),
        "worst_day_pct": round(worst, 4),
    }
```

In `gather_dashboard_data`, add after the snapshots and decisions are gathered:
```python
    data["performance"] = compute_performance_stats(
        snapshots=data["snapshots"],
        decisions=data["decisions"],
    )
```

In `write_json_files`, add:
```python
    with open(os.path.join(deploy_dir, "performance.json"), "w") as f:
        json.dump(data.get("performance", {}), f, cls=_DecimalEncoder)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestComputePerformanceStats -v`
Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): emit performance.json with derived stats (drawdown, win rate, best/worst day)"
```

---

## Task 6: Homepage renderer

Replace the hand-authored homepage with `render_homepage()`. This is the largest renderer; subsequent pages reuse most of its building blocks.

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_pages.py`:

```python
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
        # Count chip ticker spans
        assert html.count('class="ticker"') == 3

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
        # Trace not yet shipped — link should fall back to hub.
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
        # 150 chars + ellipsis
        assert ("x" * 150 + "…") in html
        assert ("x" * 151) not in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderHomepage -v`
Expected: ImportError on `render_homepage`.

- [ ] **Step 3: Implement `render_homepage`**

Add to `v2/dashboard_pages.py`:

```python
def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def _fmt_signed_pct(value) -> str:
    if value is None:
        return "—"
    v = float(value)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _hero_chip(t: dict) -> str:
    ticker = _esc(t["ticker"])
    blurb = _esc(t.get("thesis") or "")
    tid = int(t["id"])
    return (
        f'<a class="chip" href="/thesis/{tid}/">'
        f'<span class="ticker">{ticker}</span> {blurb}'
        f'</a>'
    )


def _render_homepage_hero(summary: dict, theses: list[dict],
                          sparkline_svg: str) -> str:
    portfolio = _fmt_money(summary.get("portfolio_value"))
    daily = _fmt_signed_pct(summary.get("daily_pnl_pct"))
    total = _fmt_signed_pct(summary.get("total_return_pct"))
    vs_spy = _fmt_signed_pct(summary.get("vs_spy_pct"))
    daily_class = "gain" if (summary.get("daily_pnl_pct") or 0) >= 0 else "loss"

    day_n = summary.get("day_number") or 0
    last_updated = _esc(str(summary.get("last_updated") or ""))

    chips_html = ""
    if theses:
        chip_items = "".join(_hero_chip(t) for t in theses[:3])
        chips_html = (
            f'<div class="label">Currently betting on</div>'
            f'<div class="chips">{chip_items}</div>'
        )

    return (
        f'<section class="hero">'
        f'<p class="tag">Day {day_n} · Updated {last_updated}</p>'
        f'<h1>{portfolio}'
        f'<span class="strip {daily_class}">'
        f' {daily} today · {total} all time · {vs_spy} vs S&amp;P</span></h1>'
        f'{chips_html}'
        f'{sparkline_svg}'
        f'</section>'
    )


def _render_today_move(today_move: dict | None) -> str:
    if not today_move:
        return (
            '<section class="section"><div class="head">'
            '<h2>Today\'s move</h2></div>'
            '<p class="empty-state">'
            'No new positions in the last 5 sessions — '
            '<a href="/activity/">see the full log →</a>'
            '</p></section>'
        )
    did = int(today_move["id"])
    action = (today_move.get("action") or "").lower()
    badge_cls = f"badge badge-{action}" if action in ("buy", "sell", "hold") else "badge"
    ticker = _esc(today_move.get("ticker") or "")
    notional = _fmt_money(today_move.get("notional"))
    pct = float(today_move.get("pct_of_portfolio") or 0)
    reasoning = _esc(_truncate(today_move.get("reasoning") or "", 150))
    return (
        f'<section class="section"><div class="head">'
        f'<h2>Today\'s move</h2>'
        f'<a class="more" href="/activity/#decisions">All decisions →</a>'
        f'</div>'
        f'<a class="move-card" href="/trade/{did}/">'
        f'<div class="head">'
        f'<span class="{badge_cls}">{action.upper()}</span> '
        f'<span class="ticker">{ticker}</span> · {notional} · {pct:.1f}% of portfolio'
        f'</div>'
        f'<p class="reasoning">{reasoning}</p>'
        f'</a></section>'
    )


def _render_recent_learnings(attribution_top: dict | None,
                             worst_loser: dict | None) -> str:
    if not attribution_top and not worst_loser:
        return ""
    if attribution_top:
        cat = _esc(attribution_top.get("category") or "")
        n = attribution_top.get("sample_size") or 0
        avg = _fmt_signed_pct(attribution_top.get("avg_outcome_30d"))
        working = (
            f'<a class="card" href="/attribution/">'
            f'<div class="lbl">What\'s working</div>'
            f'<h3 class="gain">{cat}</h3>'
            f'<p>{n} trades · {avg} avg</p></a>'
        )
    else:
        working = (
            '<div class="card disabled"><div class="lbl">What\'s working</div>'
            '<p>Not enough samples yet.</p></div>'
        )
    if worst_loser:
        ticker = _esc(worst_loser.get("ticker") or "")
        pct = _fmt_signed_pct(worst_loser.get("outcome_30d_pct"))
        didnt = (
            f'<a class="card" href="/mistakes/">'
            f'<div class="lbl">What didn\'t</div>'
            f'<h3 class="loss"><span class="ticker">{ticker}</span> {pct}</h3>'
            f'<p>Worst recent closed loser.</p></a>'
        )
    else:
        didnt = (
            '<div class="card disabled"><div class="lbl">What didn\'t</div>'
            '<p>No closed losers in window.</p></div>'
        )
    return (
        '<section class="section"><div class="head">'
        '<h2>Recent learnings</h2>'
        '<a class="more" href="/learning/">Learning →</a>'
        '</div>'
        f'<div class="card-grid">{working}{didnt}</div>'
        '</section>'
    )


def _render_memo_block(memo: dict | None) -> str:
    if not memo:
        return ""
    body = _esc(_truncate(memo.get("content") or "", 280))
    session_date = _esc(str(memo.get("session_date") or ""))
    return (
        '<section class="section"><div class="head">'
        '<h2>From today\'s session memo</h2>'
        '<a class="more" href="/activity/#memos">All memos →</a>'
        '</div>'
        f'<blockquote class="memo-block">'
        f'<div class="meta">{session_date}</div>'
        f'{body}</blockquote>'
        '</section>'
    )


def _methodology_link(label: str, child_path: str, ready: bool) -> str:
    href = child_path if ready else "/how-it-works/"
    return f'<a href="{href}">{_esc(label)}</a>'


def _render_methodology_strip(state: dict) -> str:
    state = state or {}
    return (
        '<div class="methodology-strip">'
        'Built by an AI agent (Claude Haiku for execution, Sonnet for strategy). '
        + _methodology_link("How it works", "/about/", state.get("about", False))
        + ' · '
        + _methodology_link("Sample tool-call trace", "/trace/", state.get("trace", False))
        + ' · '
        + _methodology_link("Model & cost", "/internals/", state.get("internals", False))
        + '</div>'
    )


def render_homepage(*, summary: dict, theses: list[dict],
                    sparkline_svg: str, today_move: dict | None,
                    attribution_top: dict | None, worst_loser: dict | None,
                    memo: dict | None, how_it_works_state: dict,
                    base_url: str) -> str:
    """Render the curated landing homepage."""
    base = base_url.rstrip("/")
    daily_pnl = summary.get("daily_pnl") or 0
    portfolio = summary.get("portfolio_value") or 0
    description = (
        f"Portfolio: {_fmt_money(portfolio)} · "
        f"Today: {_fmt_money(daily_pnl)} ({_fmt_signed_pct(summary.get('daily_pnl_pct'))})"
    )

    content = (
        _render_homepage_hero(summary, theses, sparkline_svg)
        + _render_today_move(today_move)
        + _render_recent_learnings(attribution_top, worst_loser)
        + _render_memo_block(memo)
        + _render_methodology_strip(how_it_works_state)
    )

    return _render_page_shell(
        title="Bikini Bottom Capital",
        description=description,
        active_nav="home",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderHomepage -v`
Expected: 13 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_homepage — curated landing with hero, today's move, learnings, memo"
```

---

## Task 7: Performance page renderer

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_pages.py`:

```python
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
        assert "+2.1" in html  # vs S&P

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderPerformancePage -v`
Expected: ImportError.

- [ ] **Step 3: Implement `render_performance_page`**

Add to `v2/dashboard_pages.py`:

```python
_CHART_JS_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/'
    'chart.js@4.4.7/dist/chart.umd.min.js"></script>'
)


def _stat(lbl: str, val: str, cls: str = "") -> str:
    return (
        f'<div class="stat"><div class="lbl">{_esc(lbl)}</div>'
        f'<div class="val {cls}">{val}</div></div>'
    )


def render_performance_page(*, summary: dict, performance: dict,
                            base_url: str) -> str:
    base = base_url.rstrip("/")
    portfolio = _fmt_money(summary.get("portfolio_value"))
    daily = _fmt_signed_pct(summary.get("daily_pnl_pct"))
    total = _fmt_signed_pct(summary.get("total_return_pct"))
    vs_spy = _fmt_signed_pct(summary.get("vs_spy_pct"))

    stat_strip = (
        '<div class="stat-row">'
        + _stat("Portfolio", portfolio)
        + _stat("Today", daily)
        + _stat("All time", total)
        + _stat("vs S&P", vs_spy)
        + '</div>'
    )

    p = performance or {}
    stats_panel = (
        '<section class="section"><div class="head"><h2>Stats</h2></div>'
        '<div class="stat-row">'
        + _stat("Max drawdown", f"{p.get('max_drawdown_pct', 0):+.2f}%")
        + _stat("Win rate", f"{p.get('win_rate_pct', 0):.1f}%")
        + _stat("Avg days held", f"{p.get('avg_days_held', 0):.1f}")
        + _stat("Best day", f"{p.get('best_day_pct', 0):+.2f}%")
        + _stat("Worst day", f"{p.get('worst_day_pct', 0):+.2f}%")
        + '</div></section>'
    )

    charts = (
        '<section class="section"><div class="head"><h2>Equity curve</h2></div>'
        '<div class="chart-wrap"><canvas id="equity-chart"></canvas></div>'
        '<p class="empty-state" id="chart-empty" style="display:none;">No snapshot data yet</p>'
        '</section>'
        '<section class="section"><div class="head"><h2>Performance vs S&amp;P 500</h2></div>'
        '<div class="chart-wrap"><canvas id="benchmark-chart"></canvas></div>'
        '<p class="empty-state" id="benchmark-empty" style="display:none;">No benchmark data yet</p>'
        '</section>'
    )

    content = stat_strip + charts + stats_panel

    return _render_page_shell(
        title="Performance",
        description=f"Equity curve and benchmark comparison. Portfolio: {portfolio}.",
        active_nav="performance",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/performance/",
        head_extra=_CHART_JS_CDN,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderPerformancePage -v`
Expected: 5 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_performance_page — equity + benchmark + derived stats"
```

---

## Task 8: Activity page renderer

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_pages.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderActivityPage -v`
Expected: ImportError.

- [ ] **Step 3: Implement `render_activity_page`**

Add to `v2/dashboard_pages.py`:

```python
def _render_memos_section(memos: list[dict]) -> str:
    if not memos:
        return (
            '<section class="section" id="memos">'
            '<div class="head"><h2>Recent memos</h2></div>'
            '<p class="empty-state">No memos yet.</p></section>'
        )
    items = []
    for m in memos[:10]:
        body = _esc(m.get("content") or "")
        d = _esc(str(m.get("session_date") or ""))
        items.append(
            f'<blockquote class="memo-block">'
            f'<div class="meta">{d}</div>{body}</blockquote>'
        )
    return (
        '<section class="section" id="memos">'
        '<div class="head"><h2>Recent memos</h2></div>'
        + "".join(items)
        + '</section>'
    )


def render_activity_page(*, base_url: str, memos: list[dict]) -> str:
    base = base_url.rstrip("/")

    holdings = (
        '<section class="section" id="holdings">'
        '<div class="head"><h2>Current holdings</h2></div>'
        '<div class="table-wrap"><table id="positions-table">'
        '<thead><tr><th>Ticker</th><th class="num">Shares</th>'
        '<th class="num">Avg Cost</th></tr></thead><tbody></tbody></table></div>'
        '<p class="empty-state" id="positions-empty" style="display:none;">'
        'No open positions</p></section>'
    )

    theses = (
        '<section class="section" id="theses">'
        '<div class="head"><h2>Active theses</h2></div>'
        '<div id="theses-list"></div>'
        '<p class="empty-state" id="theses-empty" style="display:none;">'
        'No active theses</p></section>'
    )

    decisions = (
        '<section class="section" id="decisions">'
        '<div class="head"><h2>Decisions log</h2></div>'
        '<div class="table-wrap"><table id="decisions-table">'
        '<thead><tr><th>Date</th><th>Ticker</th><th>Action</th>'
        '<th class="num">Qty</th><th>Reasoning</th>'
        '<th class="num">Order ID</th></tr></thead><tbody></tbody></table></div>'
        '<p class="empty-state" id="decisions-empty" style="display:none;">'
        'No decisions yet</p></section>'
    )

    content = holdings + theses + decisions + _render_memos_section(memos)

    return _render_page_shell(
        title="Activity",
        description="Holdings, active theses, decisions log, and recent memos.",
        active_nav="activity",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/activity/",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderActivityPage -v`
Expected: 7 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_activity_page — holdings/theses/decisions/memos with anchors"
```

---

## Task 9: Learning hub renderer

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_pages.py`:

```python
from v2.dashboard_pages import render_learning_hub


class TestRenderLearningHub:
    def _data(self, **overrides):
        defaults = dict(
            attribution_top3=[
                {"category": "earnings_beat", "sample_size": 18,
                 "avg_outcome_30d": Decimal("3.2")},
                {"category": "macro_pivot", "sample_size": 12,
                 "avg_outcome_30d": Decimal("1.8")},
                {"category": "sector_rotation", "sample_size": 9,
                 "avg_outcome_30d": Decimal("1.1")},
            ],
            losers_top3=[
                {"ticker": "PLTR", "outcome_30d_pct": Decimal("-8.4")},
                {"ticker": "TSLA", "outcome_30d_pct": Decimal("-5.1")},
                {"ticker": "F", "outcome_30d_pct": Decimal("-3.2")},
            ],
            retired_rules_count=4,
            base_url="https://example.com",
        )
        defaults.update(overrides)
        return defaults

    def test_uses_page_shell_with_learning_active(self):
        html = render_learning_hub(**self._data())
        assert 'data-page="learning"' in html
        assert 'class="active" href="/learning/"' in html

    def test_renders_two_cards(self):
        html = render_learning_hub(**self._data())
        assert "What's working" in html
        assert "What didn't" in html
        assert 'href="/attribution/"' in html
        assert 'href="/mistakes/"' in html

    def test_attribution_top3_listed(self):
        html = render_learning_hub(**self._data())
        assert "earnings_beat" in html
        assert "macro_pivot" in html
        assert "sector_rotation" in html

    def test_losers_top3_listed(self):
        html = render_learning_hub(**self._data())
        assert "PLTR" in html
        assert "TSLA" in html

    def test_retired_rules_count_shown(self):
        html = render_learning_hub(**self._data())
        assert "4" in html
        assert "retired" in html.lower()

    def test_attribution_empty_shows_placeholder(self):
        html = render_learning_hub(**self._data(attribution_top3=[]))
        assert "Not enough samples yet" in html

    def test_losers_empty_shows_placeholder(self):
        html = render_learning_hub(**self._data(losers_top3=[]))
        assert "No closed losers in window" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderLearningHub -v`
Expected: ImportError.

- [ ] **Step 3: Implement `render_learning_hub`**

Add to `v2/dashboard_pages.py`:

```python
def render_learning_hub(*, attribution_top3: list[dict],
                        losers_top3: list[dict], retired_rules_count: int,
                        base_url: str) -> str:
    base = base_url.rstrip("/")

    if attribution_top3:
        rows = "".join(
            f'<li><strong>{_esc(a.get("category") or "")}</strong> · '
            f'{a.get("sample_size") or 0} trades · '
            f'{_fmt_signed_pct(a.get("avg_outcome_30d"))} avg</li>'
            for a in attribution_top3[:3]
        )
        working_body = f'<ul>{rows}</ul>'
    else:
        working_body = '<p>Not enough samples yet.</p>'

    if losers_top3:
        rows = "".join(
            f'<li><span class="ticker">{_esc(l.get("ticker") or "")}</span> '
            f'<span class="loss">{_fmt_signed_pct(l.get("outcome_30d_pct"))}</span></li>'
            for l in losers_top3[:3]
        )
        didnt_body = (
            f'<ul>{rows}</ul>'
            f'<p>{retired_rules_count} retired rule(s) recently.</p>'
        )
    else:
        didnt_body = (
            '<p>No closed losers in window.</p>'
            f'<p>{retired_rules_count} retired rule(s) recently.</p>'
        )

    content = (
        '<section class="hero"><h1>What this thing has learned</h1></section>'
        '<section class="section"><div class="card-grid">'
        f'<a class="card" href="/attribution/">'
        f'<div class="lbl">What\'s working</div>'
        f'<h3>Top signals</h3>{working_body}'
        f'<p class="more">See all →</p></a>'
        f'<a class="card" href="/mistakes/">'
        f'<div class="lbl">What didn\'t</div>'
        f'<h3>Recent losers</h3>{didnt_body}'
        f'<p class="more">See all →</p></a>'
        '</div></section>'
    )

    return _render_page_shell(
        title="Learning",
        description="What this AI agent has learned: signals that work, mistakes it's made.",
        active_nav="learning",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/learning/",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderLearningHub -v`
Expected: 7 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_learning_hub — index of /attribution/ and /mistakes/"
```

---

## Task 10: How-it-works hub renderer

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_pages.py`:

```python
from v2.dashboard_pages import render_how_it_works_hub


class TestRenderHowItWorksHub:
    def _data(self, **overrides):
        defaults = dict(
            child_state={"about": True, "internals": True, "trace": False},
            base_url="https://example.com",
        )
        defaults.update(overrides)
        return defaults

    def test_uses_page_shell_with_how_it_works_active(self):
        html = render_how_it_works_hub(**self._data())
        assert 'data-page="how-it-works"' in html
        assert 'class="active" href="/how-it-works/"' in html

    def test_renders_three_cards(self):
        html = render_how_it_works_hub(**self._data())
        assert "Methodology" in html
        assert "Model & cost" in html
        assert "Tool-call trace" in html

    def test_ready_children_link_to_pages(self):
        html = render_how_it_works_hub(**self._data())
        assert 'href="/about/"' in html
        assert 'href="/internals/"' in html

    def test_unready_child_renders_disabled(self):
        html = render_how_it_works_hub(**self._data())
        # trace is False — should be disabled, no link
        assert 'href="/trace/"' not in html
        assert 'class="card disabled"' in html
        assert "Coming soon" in html

    def test_all_unready(self):
        html = render_how_it_works_hub(
            **self._data(child_state={"about": False, "internals": False, "trace": False})
        )
        assert html.count('class="card disabled"') == 3
        assert html.count("Coming soon") == 3

    def test_all_ready(self):
        html = render_how_it_works_hub(
            **self._data(child_state={"about": True, "internals": True, "trace": True})
        )
        assert 'href="/about/"' in html
        assert 'href="/internals/"' in html
        assert 'href="/trace/"' in html
        assert 'class="card disabled"' not in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderHowItWorksHub -v`
Expected: ImportError.

- [ ] **Step 3: Implement `render_how_it_works_hub`**

Add to `v2/dashboard_pages.py`:

```python
_HOW_IT_WORKS_CHILDREN = (
    ("about", "/about/", "Methodology",
     "How decisions get made — the agentic loop, the prompts, the data."),
    ("internals", "/internals/", "Model & cost",
     "Which Claude model runs each stage, how often, and what it costs."),
    ("trace", "/trace/", "Tool-call trace",
     "A real strategist session — every tool call, redacted but unedited."),
)


def render_how_it_works_hub(*, child_state: dict, base_url: str) -> str:
    base = base_url.rstrip("/")
    cards = []
    for key, href, title, blurb in _HOW_IT_WORKS_CHILDREN:
        ready = bool(child_state.get(key))
        if ready:
            cards.append(
                f'<a class="card" href="{href}">'
                f'<h3>{_esc(title)}</h3>'
                f'<p>{_esc(blurb)}</p>'
                f'<p class="more">Read →</p></a>'
            )
        else:
            cards.append(
                f'<div class="card disabled">'
                f'<h3>{_esc(title)}</h3>'
                f'<p>{_esc(blurb)}</p>'
                f'<p class="more">Coming soon</p></div>'
            )

    content = (
        '<section class="hero"><h1>How this thing works</h1></section>'
        f'<section class="section"><div class="card-grid">{"".join(cards)}</div></section>'
    )

    return _render_page_shell(
        title="How it works",
        description="Methodology, model & cost transparency, and a sample tool-call trace.",
        active_nav="how-it-works",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/how-it-works/",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py::TestRenderHowItWorksHub -v`
Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_how_it_works_hub with Coming-soon detection per child"
```

---

## Task 11: Re-skin existing renderers to use the page shell

`render_trade_page`, `render_thesis_page`, `render_mistakes_page`, `render_attribution_page` all currently emit their own `<html>` wrapper with the old header. Convert each to call `_render_page_shell`.

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Inspect existing tests so the migration doesn't break them**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py -v`
Expected: All existing tests pass (sanity check before changing things).

Read the assertions in:
- `TestRenderTradePage`
- `TestRenderThesisPage`
- `TestRenderMistakesPage`
- `TestRenderAttributionPage`

These hit specific strings in the old templates (e.g., `Bikini Bottom Capital` in the header). Each migrated renderer must continue to satisfy them.

- [ ] **Step 2: Add nav-active assertions to each existing test class**

Add a single new test method to each existing test class in `tests/v2/test_dashboard_pages.py`:

```python
# In TestRenderTradePage
def test_uses_page_shell_with_activity_active(self):
    # Trade detail pages live under Activity in the IA.
    html = render_trade_page(...)  # reuse existing args
    assert 'data-page="activity"' in html
    assert 'class="active" href="/activity/"' in html

# In TestRenderThesisPage
def test_uses_page_shell_with_activity_active(self):
    html = render_thesis_page(...)
    assert 'data-page="activity"' in html
    assert 'class="active" href="/activity/"' in html

# In TestRenderMistakesPage
def test_uses_page_shell_with_learning_active(self):
    html = render_mistakes_page(...)
    assert 'data-page="learning"' in html
    assert 'class="active" href="/learning/"' in html

# In TestRenderAttributionPage
def test_uses_page_shell_with_learning_active(self):
    html = render_attribution_page(...)
    assert 'data-page="learning"' in html
    assert 'class="active" href="/learning/"' in html
```

(Use whatever fixture / helper exists in each class to construct valid args — the point is to assert the new shell wrapping is in place.)

- [ ] **Step 3: Run new assertions to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py -k "uses_page_shell" -v`
Expected: 4 failures — the existing renderers don't call `_render_page_shell` yet.

- [ ] **Step 4: Migrate `render_trade_page` to use the shell**

Replace the body of `render_trade_page` in `v2/dashboard_pages.py`. The function currently builds full `<html>` via `_TRADE_PAGE_TEMPLATE`. Preserve the existing key schema (`decision["date"]`, `decision["quantity"]`, `decision["outcome_7d"]`, `decision["outcome_30d"]`) and the existing OG image path shape (`{base}/og/trade/{id}.png`):

```python
def render_trade_page(decision: dict, thesis: dict | None,
                      position: dict | None, base_url: str) -> str:
    """Return the full HTML page for one trade."""
    base = base_url.rstrip("/")
    decision_id = int(decision["id"])

    raw_ticker = str(decision["ticker"])
    raw_qty = decision.get("quantity") or 0
    raw_price = decision.get("price") or 0
    action_upper = str(decision.get("action", "")).lower().upper()

    ticker_esc = _esc(raw_ticker)
    action_caps = _esc(action_upper)
    qty_display = _esc(str(raw_qty))
    price_display = _fmt_money(raw_price)

    trade_date = (
        decision["date"].isoformat()
        if hasattr(decision["date"], "isoformat")
        else _esc(str(decision["date"]))
    )

    thesis_section = ""
    if thesis:
        tid = int(thesis["id"])
        raw_thesis_text = str(thesis.get("thesis", ""))
        thesis_text = _esc(raw_thesis_text) if raw_thesis_text else f"Thesis #{tid}"
        thesis_section = _THESIS_LINK_TEMPLATE.substitute(
            tid=tid,
            thesis_text=thesis_text,
            direction=_esc(str(thesis.get("direction", ""))),
            confidence=_esc(str(thesis.get("confidence", ""))),
        )

    outcome_section = ""
    if decision.get("outcome_7d") is not None or decision.get("outcome_30d") is not None:
        outcome_section = _OUTCOME_TEMPLATE.substitute(
            o7=_fmt_outcome(decision.get("outcome_7d")),
            o30=_fmt_outcome(decision.get("outcome_30d")),
        )

    title = f"{action_caps} {ticker_esc}"
    description_raw = f"{action_upper} {raw_qty} {raw_ticker} @ {_fmt_money(raw_price)}"

    content = (
        f'<section class="section">'
        f'<h2>{action_caps} {ticker_esc}</h2>'
        f'<p class="trade-summary">{qty_display} shares at {price_display} on {trade_date}</p>'
        f'<h3>Reasoning</h3>'
        f'<p>{_esc(str(decision.get("reasoning") or ""))}</p>'
        f'{thesis_section}{outcome_section}'
        f'</section>'
    )

    return _render_page_shell(
        title=title,
        description=description_raw,
        active_nav="activity",
        content=content,
        og_image=f"{base}/og/trade/{decision_id}.png",
        page_url=f"{base}/trade/{decision_id}/",
        og_type="article",
    )
```

- [ ] **Step 5: Migrate `render_thesis_page` to use the shell**

Replace `render_thesis_page` similarly. Keep the helpers it currently calls (`_render_triggers_section`, `_render_decisions_section`), drop the standalone `_THESIS_PAGE_TEMPLATE`, and feed the inner content to `_render_page_shell`. Match the existing OG image path shape (`{base}/og/thesis/{id}.png`).

```python
def render_thesis_page(thesis: dict, decisions: list[dict],
                       position: dict | None, base_url: str) -> str:
    base = base_url.rstrip("/")
    tid = int(thesis["id"])
    ticker = _esc(str(thesis.get("ticker") or ""))
    direction = _esc(str(thesis.get("direction") or ""))
    confidence = _esc(str(thesis.get("confidence") or ""))
    status = _esc(str(thesis.get("status") or ""))
    thesis_text = _esc(str(thesis.get("thesis") or ""))

    title = f"{ticker} — {direction} thesis"
    description = (
        f"{ticker} {direction} thesis (confidence: {confidence}). "
        f"{thesis_text[:160]}"
    )

    content = (
        f'<section class="section">'
        f'<h2>{ticker} — {direction} thesis</h2>'
        f'<p class="thesis-meta">Confidence: {confidence} · Status: {status}</p>'
        f'<h3>Thesis</h3><p>{thesis_text}</p>'
        f'{_render_triggers_section(thesis)}'
        f'{_render_decisions_section(decisions)}'
        f'</section>'
    )

    return _render_page_shell(
        title=title,
        description=description,
        active_nav="activity",
        content=content,
        og_image=f"{base}/og/thesis/{tid}.png",
        page_url=f"{base}/thesis/{tid}/",
        og_type="article",
    )
```

- [ ] **Step 6: Migrate `render_mistakes_page` to use the shell**

Replace the entire current `render_mistakes_page` in `v2/dashboard_pages.py` (and delete the now-unused `_MISTAKES_PAGE_TEMPLATE` constant defined just above it):

```python
def render_mistakes_page(closed_losers: list[dict], retired_rules: list[dict],
                         base_url: str) -> str:
    """Return the full HTML for /mistakes/index.html."""
    base = base_url.rstrip("/")

    if closed_losers:
        rows = "".join(_render_loser_row(d) for d in closed_losers)
        losers_section = (
            '<section class="section"><div class="head">'
            '<h2>Closed losers</h2></div>'
            f'<ul class="loser-list">{rows}</ul></section>'
        )
    else:
        losers_section = (
            '<section class="section"><div class="head">'
            '<h2>Closed losers</h2></div>'
            '<p class="empty-state">No closed losers in window. '
            'Either we got lucky or we didn\'t trade enough.</p></section>'
        )

    if retired_rules:
        rows = "".join(_render_rule_row(r) for r in retired_rules)
        rules_section = (
            '<section class="section"><div class="head">'
            '<h2>Retired rules</h2></div>'
            f'<ul class="rule-list">{rows}</ul></section>'
        )
    else:
        rules_section = ""

    return _render_page_shell(
        title="What didn't work",
        description="Closed losers and retired rules. The receipts most accounts hide.",
        active_nav="learning",
        data_page="mistakes",
        content=losers_section + rules_section,
        og_image=f"{base}/og/mistakes.png",
        page_url=f"{base}/mistakes/",
        og_type="article",
    )
```

- [ ] **Step 7: Migrate `render_attribution_page` to use the shell**

Replace the entire current `render_attribution_page` (and delete the now-unused `_ATTRIBUTION_PAGE_TEMPLATE` constant defined just above it). Keep `_render_attribution_table` — it's still called.

```python
def render_attribution_page(attribution: list[dict], base_url: str) -> str:
    """Return the full HTML for /attribution/index.html."""
    base = base_url.rstrip("/")

    if attribution:
        body = _render_attribution_table(attribution)
    else:
        body = (
            '<p class="empty-state">'
            "Not enough samples yet. Attribution scores require at least "
            "5 closed decisions per signal type.</p>"
        )

    content = (
        '<section class="section">'
        '<div class="head"><h2>What\'s actually working</h2></div>'
        '<p class="subtitle">'
        'Signal-attribution scores from the last 90 days of decisions.</p>'
        + body + '</section>'
    )

    return _render_page_shell(
        title="What's actually working",
        description="Signal-attribution scores. Which inputs predicted, which were noise.",
        active_nav="learning",
        data_page="attribution",
        content=content,
        og_image=f"{base}/og/attribution.png",
        page_url=f"{base}/attribution/",
        og_type="article",
    )
```

- [ ] **Step 8: Run the full dashboard_pages test suite**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py -v`
Expected: All passing (existing assertions + the 4 new `uses_page_shell` assertions).

If any pre-existing assertion now fails, it was hitting old-template-specific HTML (e.g., the `<a href="/">Back to dashboard</a>` from `_TRADE_PAGE_TEMPLATE`'s footer). Either delete that assertion (it was testing the scaffolding, not the renderer's content) or relax it to test the equivalent shell-supplied behavior.

- [ ] **Step 9: Delete dead template constants**

Now that the migrated renderers no longer use them, delete `_TRADE_PAGE_TEMPLATE` (and any thesis/mistakes/attribution standalone templates that are no longer referenced) from `v2/dashboard_pages.py`.

Verify no references remain:
```bash
grep -n "_TRADE_PAGE_TEMPLATE\|_THESIS_PAGE_TEMPLATE\|_MISTAKES_PAGE_TEMPLATE\|_ATTRIBUTION_PAGE_TEMPLATE" v2/dashboard_pages.py
```
Expected: only the (deleted) definitions show up — no callsites. If a template is still used by another helper, leave it.

- [ ] **Step 10: Run the full test suite to catch regressions**

Run: `python3 -m pytest tests/v2/ -v`
Expected: all passing.

- [ ] **Step 11: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "refactor(v2): migrate trade/thesis/mistakes/attribution pages to _render_page_shell"
```

---

## Task 12: Wire new pages into publish flow

Render every new page during `assemble_deploy_dir`, and stop copying the (now-deleted) hand-authored `index.html`.

**Files:**
- Modify: `v2/dashboard_publish.py`
- Delete: `public_dashboard/index.html`
- Test: `tests/v2/test_dashboard_publish.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/v2/test_dashboard_publish.py`:

```python
class TestAssembleDeployDirNewPages:
    def _minimal_data(self):
        return {
            "summary": {"portfolio_value": Decimal("100"), "daily_pnl": Decimal("0"),
                        "daily_pnl_pct": Decimal("0"), "total_return_pct": Decimal("0"),
                        "vs_spy_pct": Decimal("0"), "day_number": 1, "last_updated": "2026-05-04"},
            "snapshots": [{"snapshot_date": date(2026, 5, 4), "value": Decimal("100")}],
            "positions": [], "decisions": [], "theses": [], "benchmark": [],
            "mistakes": {"closed_losers": [], "retired_rules": []},
            "memos": [], "attribution": [],
            "performance": {"max_drawdown_pct": 0, "win_rate_pct": 0,
                            "avg_days_held": 0, "best_day_pct": 0, "worst_day_pct": 0},
            "_pages": {"decision_ids": [], "thesis_ids": []},
        }

    def test_homepage_emitted(self, tmp_path):
        from v2.dashboard_publish import assemble_deploy_dir
        # Need an assets dir with styles.css and app.js (no index.html anymore).
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "styles.css").write_text("/* */")
        (assets / "app.js").write_text("// ")

        deploy = tmp_path / "deploy"
        assemble_deploy_dir(self._minimal_data(), str(deploy), str(assets),
                            base_url="https://example.com")

        index = deploy / "index.html"
        assert index.exists()
        html = index.read_text()
        assert 'data-page="home"' in html

    def test_new_pages_emitted(self, tmp_path):
        from v2.dashboard_publish import assemble_deploy_dir
        assets = tmp_path / "assets"; assets.mkdir()
        (assets / "styles.css").write_text("")
        (assets / "app.js").write_text("")
        deploy = tmp_path / "deploy"
        assemble_deploy_dir(self._minimal_data(), str(deploy), str(assets),
                            base_url="https://example.com")

        for path in ("performance/index.html", "activity/index.html",
                     "learning/index.html", "how-it-works/index.html"):
            assert (deploy / path).exists(), f"missing: {path}"

    def test_how_it_works_marks_unready_children(self, tmp_path):
        from v2.dashboard_publish import assemble_deploy_dir
        assets = tmp_path / "assets"; assets.mkdir()
        (assets / "styles.css").write_text("")
        (assets / "app.js").write_text("")
        deploy = tmp_path / "deploy"
        assemble_deploy_dir(self._minimal_data(), str(deploy), str(assets),
                            base_url="https://example.com")

        html = (deploy / "how-it-works" / "index.html").read_text()
        # None of /about/, /internals/, /trace/ exist in this fixture deploy.
        assert html.count('class="card disabled"') == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestAssembleDeployDirNewPages -v`
Expected: failures because `assemble_deploy_dir` doesn't emit the new pages and may still expect `index.html` in `_STATIC_ASSETS`.

- [ ] **Step 3: Update `_STATIC_ASSETS` and add new-page emission**

In `v2/dashboard_publish.py`:

Change:
```python
_STATIC_ASSETS = ("index.html", "styles.css", "app.js")
```
to:
```python
_STATIC_ASSETS = ("styles.css", "app.js")
```

Add an import for the new renderers:
```python
from .dashboard_pages import (
    render_attribution_page,
    render_homepage_meta,
    render_mistakes_page,
    render_thesis_page,
    render_trade_page,
    # NEW:
    render_homepage,
    render_performance_page,
    render_activity_page,
    render_learning_hub,
    render_how_it_works_hub,
)
```

Add a helper `emit_new_static_pages` (or extend `emit_static_pages`) — for clarity, add a new helper:

```python
def _detect_how_it_works_state(deploy_dir: str) -> dict:
    """Return readiness flags for /about/, /internals/, /trace/ at publish time."""
    return {
        "about": os.path.exists(os.path.join(deploy_dir, "about", "index.html")),
        "internals": os.path.exists(os.path.join(deploy_dir, "internals", "index.html")),
        "trace": os.path.exists(os.path.join(deploy_dir, "trace", "index.html")),
    }


def _select_today_move(decisions: list[dict],
                       portfolio_value: float | Decimal | None) -> dict | None:
    """Most-recent significant non-hold decision (notional ≥ $100).

    `decisions` is assumed newest-first (matches gather_dashboard_data's
    ordering). Returns the homepage's expected dict shape with `notional`
    and `pct_of_portfolio` derived from quantity * price.
    """
    if not decisions:
        return None
    pv = float(portfolio_value or 0)
    for d in decisions:
        action = (d.get("action") or "").lower()
        if action == "hold":
            continue
        qty = float(d.get("quantity") or 0)
        price = float(d.get("price") or 0)
        notional = qty * price
        if notional < 100:
            continue
        return {
            "id": d["id"],
            "ticker": d.get("ticker"),
            "action": action,
            "notional": notional,
            "pct_of_portfolio": (notional / pv * 100.0) if pv > 0 else 0.0,
            "reasoning": d.get("reasoning"),
        }
    return None


def _select_attribution_top(attribution: list[dict], min_samples: int = 5) -> dict | None:
    if not attribution:
        return None
    eligible = [a for a in attribution
                if (a.get("sample_size") or 0) >= min_samples]
    return eligible[0] if eligible else None


def _select_worst_loser(mistakes: dict) -> dict | None:
    losers = (mistakes or {}).get("closed_losers") or []
    return losers[0] if losers else None


def _select_latest_memo(memos: list[dict]) -> dict | None:
    return memos[0] if memos else None


def emit_homepage(data: dict, deploy_dir: str, base_url: str) -> None:
    """Render and write index.html — replaces the old static index.html copy."""
    summary = data.get("summary") or {}
    snapshots = data.get("snapshots") or []
    sparkline = render_sparkline_svg(snapshots)
    today_move = _select_today_move(
        data.get("decisions") or [],
        portfolio_value=summary.get("portfolio_value"),
    )
    attribution_top = _select_attribution_top(data.get("attribution") or [])
    worst_loser = _select_worst_loser(data.get("mistakes") or {})
    memo = _select_latest_memo(data.get("memos") or [])
    how_it_works = _detect_how_it_works_state(deploy_dir)
    # Active theses only
    theses = [t for t in (data.get("theses") or []) if (t.get("status") or "active") == "active"]

    html = render_homepage(
        summary=summary,
        theses=theses,
        sparkline_svg=sparkline,
        today_move=today_move,
        attribution_top=attribution_top,
        worst_loser=worst_loser,
        memo=memo,
        how_it_works_state=how_it_works,
        base_url=base_url,
    )
    with open(os.path.join(deploy_dir, "index.html"), "w") as f:
        f.write(html)


def emit_new_pages(data: dict, deploy_dir: str, base_url: str) -> None:
    """Render the four new page types: performance, activity, learning, how-it-works."""
    summary = data.get("summary") or {}
    performance = data.get("performance") or {}
    memos = data.get("memos") or []
    attribution = data.get("attribution") or []
    mistakes = data.get("mistakes") or {}

    pages = (
        ("performance", render_performance_page(
            summary=summary, performance=performance, base_url=base_url)),
        ("activity", render_activity_page(
            base_url=base_url, memos=memos)),
        ("learning", render_learning_hub(
            attribution_top3=attribution[:3],
            losers_top3=(mistakes.get("closed_losers") or [])[:3],
            retired_rules_count=len(mistakes.get("retired_rules") or []),
            base_url=base_url)),
        ("how-it-works", render_how_it_works_hub(
            child_state=_detect_how_it_works_state(deploy_dir),
            base_url=base_url)),
    )
    for slug, html in pages:
        page_dir = os.path.join(deploy_dir, slug)
        os.makedirs(page_dir, exist_ok=True)
        with open(os.path.join(page_dir, "index.html"), "w") as f:
            f.write(html)
```

In `assemble_deploy_dir`, replace the homepage-meta injection and reorder so new-page emission happens after detail-page emission (so How-it-works picks up `/about/`, `/internals/`, `/trace/` if they exist):

Change:
```python
    # Inject homepage OG meta (no-op if placeholder absent)
    try:
        inject_homepage_og_meta(deploy_dir, data.get("summary", {}), base_url=base_url)
    except Exception:
        logger.warning("Failed to inject homepage OG meta", exc_info=True)

    emit_home_og_image(data.get("summary", {}), deploy_dir)
    emit_static_pages(data, deploy_dir, base_url=base_url)
```
to:
```python
    emit_home_og_image(data.get("summary", {}), deploy_dir)
    emit_static_pages(data, deploy_dir, base_url=base_url)
```

After the `emit_detail_pages` / `emit_og_images` block (i.e. just before `return deploy_dir`), add:
```python
    # Render homepage + the 4 new pages last so they can detect which
    # /about/, /internals/, /trace/ children have already been written.
    if base_url:
        emit_homepage(data, deploy_dir, base_url=base_url)
        emit_new_pages(data, deploy_dir, base_url=base_url)
```

- [ ] **Step 4: Delete the hand-authored homepage**

```bash
rm public_dashboard/index.html
```

- [ ] **Step 5: Drop the now-dead `inject_homepage_og_meta`**

In `v2/dashboard_publish.py`, delete the `inject_homepage_og_meta` function entirely. Also remove its callsite in `assemble_deploy_dir` (already done in step 3) and any test that asserts on it.

Verify:
```bash
grep -n "inject_homepage_og_meta" v2/dashboard_publish.py tests/v2/
```
Expected: no matches (delete any leftover test class).

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestAssembleDeployDirNewPages -v`
Expected: 3 passing.

Then run the full suite:
```bash
python3 -m pytest tests/v2/ -v
```
Expected: all passing. Fix any test that asserted on the deleted `inject_homepage_og_meta` or old `index.html` copying.

- [ ] **Step 7: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git rm public_dashboard/index.html
git commit -m "feat(v2): render homepage + 4 new pages in assemble_deploy_dir"
```

---

## Task 13: Refactor `app.js` for per-page initialization

Old `app.js` assumed a single homepage. Now it must detect which page it's running on (via the `data-page` body attribute) and initialize accordingly.

**Files:**
- Modify: `public_dashboard/app.js`

- [ ] **Step 1: Read `app.js` and inventory what each block does**

Run: `wc -l public_dashboard/app.js && head -50 public_dashboard/app.js`

Identify the existing helpers (`escapeHtml`, `formatCurrency`, etc.) and the section-renderers (positions table, decisions table, theses list, equity chart, benchmark chart, mistakes table, attribution chart). These need to be split:
- **Performance page** uses: equity chart, benchmark chart.
- **Activity page** uses: positions table, theses list, decisions table.
- **Mistakes page** uses: mistakes/losers table.
- **Attribution page** uses: attribution chart.
- **Homepage** uses: nothing client-side anymore (everything is server-rendered).

- [ ] **Step 2: Rewrite `app.js` with a per-page dispatcher**

Replace `public_dashboard/app.js` with this skeleton (preserve the existing helper bodies and section initializers verbatim — only the dispatch wrapper changes):

```javascript
"use strict";

// === Helpers (unchanged) ===
// ... keep existing escapeHtml, formatCurrency, formatPct, pnlClass,
//     truncate, shortOrderId, computeTWR ...

// === Hamburger toggle (all pages) ===
function setupHamburger() {
  var btn = document.querySelector(".site-nav .hamburger");
  var links = document.querySelector(".site-nav .links");
  if (!btn || !links) return;
  btn.addEventListener("click", function () {
    links.classList.toggle("open");
  });
}

// === Per-page initializers ===
// Each function fetches the JSON it needs and populates the DOM.
// Move existing render functions here, scoped under these initializers.

function initPerformancePage() {
  // ... existing equity-chart and benchmark-chart code ...
}

function initActivityPage() {
  // ... existing positions, theses, decisions code ...
}

function initMistakesPage() {
  // ... existing mistakes-losers + retired-rules code ...
}

function initAttributionPage() {
  // ... existing attribution-chart code ...
}

// === Dispatcher ===
document.addEventListener("DOMContentLoaded", function () {
  setupHamburger();
  var page = document.body.dataset.page;
  switch (page) {
    case "performance":
      initPerformancePage();
      break;
    case "activity":
      initActivityPage();
      break;
    case "learning":
      // Hub — no client-side data.
      break;
    case "how-it-works":
      // Hub — no client-side data.
      break;
    case "home":
      // Server-rendered.
      break;
    default:
      // Mistakes / attribution permalinks set their own data-page values.
      if (page === "mistakes") initMistakesPage();
      else if (page === "attribution") initAttributionPage();
  }
});
```

The existing per-section helpers (e.g., the function that fetches `positions.json` and fills the table) move inside their respective `init*` function bodies — copy them, don't rewrite the logic. Goal is **the same fetch/render code, just gated by the dispatcher.**

- [ ] **Step 3: Add per-permalink `data-page` assertions to existing tests**

`_render_page_shell` already supports a `data_page` override (Task 2), and Task 11 wired `render_mistakes_page` to pass `data_page="mistakes"` and `render_attribution_page` to pass `data_page="attribution"`. Add assertions to lock that in.

In `tests/v2/test_dashboard_pages.py`:

```python
# In TestRenderMistakesPage — add:
def test_data_page_is_mistakes(self):
    # Reuse this class's existing fixture builder for valid args.
    html = render_mistakes_page(closed_losers=[], retired_rules=[],
                                base_url="https://example.com")
    assert 'data-page="mistakes"' in html

# In TestRenderAttributionPage — add:
def test_data_page_is_attribution(self):
    html = render_attribution_page(attribution=[],
                                   base_url="https://example.com")
    assert 'data-page="attribution"' in html
```

- [ ] **Step 4: Run dashboard_pages tests**

Run: `python3 -m pytest tests/v2/test_dashboard_pages.py -v`
Expected: all passing including the new `data-page` assertions.

- [ ] **Step 5: Local browser smoke check**

Build a deploy dir locally and serve it:
```bash
mkdir -p /tmp/dash-preview
python3 -c "
import os, sys
sys.path.insert(0, '/home/jay/dev/algo')
from datetime import date
from decimal import Decimal
from v2.dashboard_publish import assemble_deploy_dir

data = {
    'summary': {'portfolio_value': Decimal('104231'), 'daily_pnl': Decimal('642'),
                'daily_pnl_pct': Decimal('0.62'), 'total_return_pct': Decimal('4.2'),
                'vs_spy_pct': Decimal('2.1'), 'day_number': 142,
                'last_updated': '2026-05-04T16:30:00'},
    'snapshots': [{'snapshot_date': date(2026, 1, i % 28 + 1), 'value': Decimal(100000 + i*30)} for i in range(60)],
    'positions': [], 'decisions': [], 'theses': [], 'benchmark': [],
    'mistakes': {'closed_losers': [], 'retired_rules': []},
    'memos': [], 'attribution': [],
    'performance': {'max_drawdown_pct': -3.2, 'win_rate_pct': 58.0,
                    'avg_days_held': 4.0, 'best_day_pct': 2.1, 'worst_day_pct': -1.5},
    '_pages': {'decision_ids': [], 'thesis_ids': []},
}
assemble_deploy_dir(data, '/tmp/dash-preview',
                    '/home/jay/dev/algo/public_dashboard',
                    base_url='http://localhost:8080')
print('Built /tmp/dash-preview')
"
cd /tmp/dash-preview && python3 -m http.server 8080
```

Open `http://localhost:8080` and click through all five top-nav items. The hub pages should render. The Performance and Activity pages may show empty states (no real data) but should not throw JS errors. Stop the server when satisfied.

- [ ] **Step 6: Commit**

```bash
git add public_dashboard/app.js v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "refactor(dashboard): per-page app.js dispatcher driven by data-page attribute"
```

---

## Task 14: Full-suite verification + final commit

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: all passing. Triage any failure — likely a test that asserted on old homepage HTML or `inject_homepage_og_meta`.

- [ ] **Step 2: Verify no orphaned references to removed code**

Run:
```bash
grep -rn "inject_homepage_og_meta\|<!-- OG_META -->" v2/ tests/v2/ public_dashboard/
grep -rn "wave-divider\|caustics" v2/ public_dashboard/
```
Expected: no matches. If anything is left, delete it.

- [ ] **Step 3: Verify pineapple favicon kept, inline emoji removed**

Run:
```bash
grep -n "🍍\|pineapple\|&#9875;" v2/dashboard_pages.py public_dashboard/
```
Expected: matches **only** in the favicon `<link rel="icon">` data URI inside `_PAGE_SHELL_TEMPLATE`. The old inline `<h1>&#9875; Bikini Bottom Capital</h1>` should not appear anywhere.

- [ ] **Step 4: Re-run the local preview**

Repeat Task 13 Step 5. Click through every nav item, verify the hamburger toggles on a narrow viewport (resize the browser below 640px), confirm the sparkline renders on the homepage, confirm "Coming soon" cards appear on `/how-it-works/` since `/about/`, `/internals/`, `/trace/` aren't built in this preview.

- [ ] **Step 5: Update README**

Open `public_dashboard/README.md` and update the "Data Files" table to add `memos.json` and `performance.json`. Replace the "static assets (`index.html`, `styles.css`, `app.js`)" line with "static assets (`styles.css`, `app.js`)" since `index.html` is now generated.

- [ ] **Step 6: Final commit**

```bash
git add public_dashboard/README.md
git commit -m "docs(dashboard): update README — homepage now generated, memos/performance JSON added"
```

- [ ] **Step 7: Cross-check the spec**

Open `docs/superpowers/specs/2026-05-04-public-dashboard-redesign-design.md` and skim each section. For every requirement in the spec, confirm a task implemented it. If anything is missing, file a follow-up task.

---

## Done criteria

- All 14 tasks committed.
- `python3 -m pytest tests/v2/` is green.
- Local preview shows the 5-page IA with sticky nav, P1 palette, and working hamburger.
- `public_dashboard/index.html` is gone; the homepage is rendered by `render_homepage`.
- `/how-it-works/` correctly degrades to "Coming soon" cards when child pages are absent.
- No grep hits for `inject_homepage_og_meta`, `wave-divider`, or `caustics`.
