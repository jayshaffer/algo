# Mistakes Log & Attribution Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two weekly content slots — a Mistakes log ("what didn't work") and an Attribution panel ("what's actually working") — surfaced as dashboard sections + permalinks + Friday-afternoon social posts that link to them.

**Architecture:** Extends Spec #1's per-page rendering (`v2/dashboard_pages.py`, `v2/dashboard_og.py`, `v2/dashboard_publish.py`) with two new static permalinks (`/mistakes/`, `/attribution/`) and OG images. Adds a new standalone module `v2/social_weekly.py` with two cron-triggered post entrypoints — same shape as `v2/premarket.py` (Spec #2). No new DB tables; reads `decisions`, `strategy_rules`, `signal_attribution` directly.

**Tech Stack:** Python 3.10+, Pillow ≥10.0, psycopg2, pytest, Anthropic SDK, tweepy, atproto. Chart.js (already loaded by the homepage) for the in-browser attribution chart. No new dependencies.

**Spec:** [`docs/superpowers/specs/2026-05-03-mistakes-attribution-design.md`](../specs/2026-05-03-mistakes-attribution-design.md)

**Out of scope:** AI-audience methodology pages (Spec #4); deletion of legacy `twitter.py` / `bluesky.py` recap path (separate cleanup plan after Spec #2 burn-in); pixel-diff regression tests for OG images (use `Image.getpixel` smoke checks instead, mirroring `tests/v2/test_dashboard_og.py`).

---

## File Structure

**New:**
- `v2/social_weekly.py` — `gather_mistakes_context`, `gather_attribution_context`, `generate_mistakes_post`, `generate_attribution_post`, `run_mistakes_post`, `run_attribution_post`, plus an `argparse`-based `__main__` entrypoint. Mirrors `v2/premarket.py`.
- `tests/v2/test_social_weekly.py` — covers context gathering, post generation, and stage runners.

**Modified:**
- `v2/database/trading_db.py` — add `get_closed_losers(reference_date, limit)` and `get_retired_rules(reference_date, limit)`. Existing `get_signal_attribution()` is reused as-is.
- `v2/dashboard_publish.py` — extend `gather_dashboard_data` to include `mistakes` and `attribution` JSON; extend `write_json_files` allowlist; add `emit_static_pages` for `/mistakes/index.html` and `/attribution/index.html`; emit OG PNGs for both; wire into `assemble_deploy_dir`.
- `v2/dashboard_pages.py` — add `render_mistakes_page` and `render_attribution_page` (pure functions, no DB).
- `v2/dashboard_og.py` — add `render_mistakes_og` and `render_attribution_og`.
- `public_dashboard/index.html` — add two `<section class="panel">` blocks (`#mistakes`, `#attribution`) with empty bodies + "see all" links.
- `public_dashboard/app.js` — fetch `mistakes.json` and `attribution.json` and populate the new sections, including a Chart.js bar chart for attribution.
- `tests/v2/conftest.py` — add `patch("v2.social_weekly.get_cursor", ...)` to the `mock_db` fixture (mirrors `v2.premarket`).
- `tests/v2/test_dashboard_pages.py` — add tests for the two new render functions.
- `tests/v2/test_dashboard_og.py` — add tests for the two new OG renders.
- `tests/v2/test_dashboard_publish.py` — extend gather-data coverage to assert `mistakes` and `attribution` keys.
- `crontab` — add two Friday entries.
- `Taskfile.yml` — add `weekly:mistakes` and `weekly:attribution` targets.
- `CLAUDE.md` — document the weekly stages.
- `README.md` — usage/runbook section for the weekly pipeline.

---

## Task 1: DB helpers — closed losers and retired rules

**Files:**
- Modify: `v2/database/trading_db.py`
- Test: `tests/v2/test_trading_db.py`

The `decisions` and `strategy_rules` schemas already support these queries; we add typed helpers so callers don't sprinkle SQL. Helpers take a `reference_date` (default: `date.today()`) so tests are deterministic.

- [ ] **Step 1: Write the failing tests**

Add to `tests/v2/test_trading_db.py` (create the file if it doesn't exist; it does — append a new class):

```python
from datetime import date


class TestGetClosedLosers:
    def test_returns_only_negative_outcomes_within_window(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_closed_losers

        mock_cursor.fetchall.return_value = [
            {"id": 11, "date": date(2026, 4, 30), "ticker": "TSLA", "action": "buy",
             "quantity": 5, "price": 200, "reasoning": "EV cycle",
             "outcome_7d": -3.2, "outcome_30d": -8.7},
        ]
        rows = get_closed_losers(reference_date=date(2026, 5, 4), limit=15)
        assert rows == mock_cursor.fetchall.return_value
        sql = mock_cursor.execute.call_args[0][0]
        assert "outcome_30d IS NOT NULL" in sql
        assert "outcome_30d < 0" in sql
        assert "ORDER BY outcome_30d ASC" in sql
        # window arg is (reference_date,) followed by (limit,)
        params = mock_cursor.execute.call_args[0][1]
        assert params == (date(2026, 5, 4), 15)


class TestGetRetiredRules:
    def test_returns_retired_only_within_window(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_retired_rules

        mock_cursor.fetchall.return_value = [
            {"id": 27, "rule_text": "Cap macro positions at $500/day",
             "retired_at": "2026-04-22", "retirement_reason": "stale"},
        ]
        rows = get_retired_rules(reference_date=date(2026, 5, 4), limit=10)
        assert rows == mock_cursor.fetchall.return_value
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'retired'" in sql
        assert "retired_at" in sql
        assert "ORDER BY retired_at DESC" in sql
        params = mock_cursor.execute.call_args[0][1]
        assert params == (date(2026, 5, 4), 10)
```

- [ ] **Step 2: Run the tests — verify they fail with ImportError**

Run: `docker compose exec trading python -m pytest tests/v2/test_trading_db.py::TestGetClosedLosers tests/v2/test_trading_db.py::TestGetRetiredRules -v`
Expected: `ImportError: cannot import name 'get_closed_losers'` (or similar).

- [ ] **Step 3: Implement the helpers**

Append to `v2/database/trading_db.py` after `select_postable_decisions_for_date` (~line 243):

```python
def get_closed_losers(reference_date, limit: int = 15) -> list[dict]:
    """Decisions in the last 30 days with resolved 30-day outcomes < 0.

    Ordered worst first. Used by the dashboard /mistakes/ page and the
    weekly mistakes social post. `reference_date` is treated as 'today';
    the window is `reference_date - 30 days .. reference_date`.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   outcome_7d, outcome_30d
            FROM decisions
            WHERE date > %s::date - INTERVAL '30 days'
              AND outcome_30d IS NOT NULL
              AND outcome_30d < 0
            ORDER BY outcome_30d ASC
            LIMIT %s
        """, (reference_date, limit))
        return cur.fetchall()


def get_retired_rules(reference_date, limit: int = 10) -> list[dict]:
    """Rules retired in the last 90 days, most recent first.

    `reference_date` is treated as 'today'; the window is
    `reference_date - 90 days .. reference_date`.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, rule_text, category, direction, confidence,
                   retired_at, retirement_reason
            FROM strategy_rules
            WHERE status = 'retired'
              AND retired_at IS NOT NULL
              AND retired_at > %s::date - INTERVAL '90 days'
            ORDER BY retired_at DESC
            LIMIT %s
        """, (reference_date, limit))
        return cur.fetchall()
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_trading_db.py::TestGetClosedLosers tests/v2/test_trading_db.py::TestGetRetiredRules -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_trading_db.py
git commit -m "feat(v2): add get_closed_losers and get_retired_rules helpers"
```

---

## Task 2: `render_mistakes_page` — HTML for `/mistakes/index.html`

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

Pure function — gets data passed in, returns full HTML string. Reuses the existing `_render_meta_block` helper for OG/Twitter cards.

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_dashboard_pages.py`:

```python
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
```

- [ ] **Step 2: Run the tests — verify they fail with ImportError**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_pages.py::TestRenderMistakesPage -v`
Expected: ImportError on `render_mistakes_page`.

- [ ] **Step 3: Implement the renderer**

Append to `v2/dashboard_pages.py`:

```python
_MISTAKES_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>What didn't work — Bikini Bottom Capital</title>
$meta_block
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>What didn't work</h2>
<p class="subtitle">Closed losers (last 30 days) and retired rules (last 90 days). No spin.</p>
$losers_section
$rules_section
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")


def _render_loser_row(d: dict) -> str:
    did = int(d["id"])
    ticker = _esc(str(d["ticker"]))
    action_caps = _esc(str(d.get("action", "")).upper())
    qty = _esc(str(d.get("quantity") or 0))
    price = _fmt_money(d.get("price") or 0)
    o30 = _fmt_outcome(d.get("outcome_30d"))
    trade_date = (
        d["date"].isoformat()
        if hasattr(d["date"], "isoformat")
        else _esc(str(d["date"]))
    )
    reasoning = _esc(str(d.get("reasoning") or ""))
    return (
        f'<li class="loser-row">'
        f'<a href="/trade/{did}/"><strong>{action_caps} {ticker}</strong></a>'
        f' — {trade_date} · {qty} @ {price} · '
        f'<span class="loser-outcome">{o30}</span>'
        f'<p class="loser-reason">{reasoning}</p>'
        f'</li>'
    )


def _render_rule_row(r: dict) -> str:
    text = _esc(str(r.get("rule_text") or ""))
    reason = _esc(str(r.get("retirement_reason") or ""))
    retired_at = r.get("retired_at")
    if hasattr(retired_at, "isoformat"):
        retired_at = retired_at.isoformat()
    retired_at_esc = _esc(str(retired_at or ""))
    return (
        f'<li class="rule-row">'
        f'<p>{text}</p>'
        f'<p class="rule-meta">retired {retired_at_esc} — {reason}</p>'
        f'</li>'
    )


def render_mistakes_page(closed_losers: list[dict], retired_rules: list[dict],
                         base_url: str) -> str:
    """Return the full HTML for /mistakes/index.html."""
    base = base_url.rstrip("/")

    if closed_losers:
        rows = "".join(_render_loser_row(d) for d in closed_losers)
        losers_section = (
            "<h3>Closed losers</h3>"
            f'<ul class="loser-list">{rows}</ul>'
        )
    else:
        losers_section = (
            '<h3>Closed losers</h3>'
            '<p class="empty-state">No closed losers in window. '
            'Either we got lucky or we didn\'t trade enough.</p>'
        )

    if retired_rules:
        rows = "".join(_render_rule_row(r) for r in retired_rules)
        rules_section = (
            "<h3>Retired rules</h3>"
            f'<ul class="rule-list">{rows}</ul>'
        )
    else:
        rules_section = ""

    meta_block = _render_meta_block(
        title="What didn't work — Bikini Bottom Capital",
        description="Closed losers and retired rules. The receipts most accounts hide.",
        og_image=f"{base}/og/mistakes.png",
        page_url=f"{base}/mistakes/",
        og_type="article",
    )

    return _MISTAKES_PAGE_TEMPLATE.substitute(
        meta_block=meta_block,
        losers_section=losers_section,
        rules_section=rules_section,
    )
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_pages.py::TestRenderMistakesPage -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_mistakes_page for /mistakes/ permalink"
```

---

## Task 3: `render_attribution_page` — HTML for `/attribution/index.html`

**Files:**
- Modify: `v2/dashboard_pages.py`
- Test: `tests/v2/test_dashboard_pages.py`

The page renders a static table (signal type, sample size, 7d outcome, 30d outcome). The Chart.js bar chart is added by `app.js` only on the homepage; the permalink is server-render-only HTML so OG previewers see real content.

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_dashboard_pages.py`:

```python
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
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_pages.py::TestRenderAttributionPage -v`
Expected: ImportError.

- [ ] **Step 3: Implement the renderer**

Append to `v2/dashboard_pages.py`:

```python
_ATTRIBUTION_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>What's actually working — Bikini Bottom Capital</title>
$meta_block
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>What's actually working</h2>
<p class="subtitle">Signal-attribution scores from the last 90 days of decisions.</p>
$body
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")


def _render_attribution_table(attribution: list[dict]) -> str:
    rows: list[str] = []
    for r in attribution:
        category = _esc(str(r.get("category") or ""))
        sample_7d = _esc(str(r.get("sample_size") or 0))
        sample_30d = _esc(str(r.get("sample_size_30d") or 0))
        out_7d = _fmt_outcome(r.get("avg_outcome_7d"))
        out_30d = _fmt_outcome(r.get("avg_outcome_30d"))
        rows.append(
            "<tr>"
            f"<td>{category}</td>"
            f'<td class="num">{sample_7d}</td>'
            f'<td class="num">{sample_30d}</td>'
            f'<td class="num">{out_7d}</td>'
            f'<td class="num">{out_30d}</td>'
            "</tr>"
        )
    body = "".join(rows)
    return (
        '<table class="attribution-table">'
        "<thead><tr>"
        "<th>Signal type</th>"
        '<th class="num">N (7d)</th>'
        '<th class="num">N (30d)</th>'
        '<th class="num">Avg 7d</th>'
        '<th class="num">Avg 30d</th>'
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_attribution_page(attribution: list[dict], base_url: str) -> str:
    """Return the full HTML for /attribution/index.html."""
    base = base_url.rstrip("/")

    if attribution:
        body = _render_attribution_table(attribution)
    else:
        body = (
            '<p class="empty-state">'
            "Not enough samples yet. Attribution scores require at least "
            "5 closed decisions per signal type."
            "</p>"
        )

    meta_block = _render_meta_block(
        title="What's actually working — Bikini Bottom Capital",
        description="Signal-attribution scores. Which inputs predicted, which were noise.",
        og_image=f"{base}/og/attribution.png",
        page_url=f"{base}/attribution/",
        og_type="article",
    )

    return _ATTRIBUTION_PAGE_TEMPLATE.substitute(
        meta_block=meta_block,
        body=body,
    )
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_pages.py::TestRenderAttributionPage -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(v2): add render_attribution_page for /attribution/ permalink"
```

---

## Task 4: `render_mistakes_og` — OG image for `/mistakes/`

**Files:**
- Modify: `v2/dashboard_og.py`
- Test: `tests/v2/test_dashboard_og.py`

Lead with the worst recent loser (ticker, 30d outcome, "lessons learned"-free tagline). Falls back to a generic card when there are no losers.

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_dashboard_og.py`:

```python
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
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_og.py::TestRenderMistakesOg -v`
Expected: ImportError.

- [ ] **Step 3: Implement the OG renderer**

Append to `v2/dashboard_og.py`:

```python
def render_mistakes_og(top_loser: dict | None) -> bytes:
    """Return PNG bytes (1200x630) for the /mistakes/ OG card."""
    img, draw = _canvas()

    draw.text((48, 90), "WHAT DIDN'T WORK", fill=_ACCENT, font=_load_font(56))

    if top_loser:
        ticker = str(top_loser.get("ticker", "?"))
        outcome = top_loser.get("outcome_30d")
        if outcome is not None:
            try:
                outcome_str = f"{Decimal(str(outcome)):+.2f}% (30d)"
            except Exception:
                outcome_str = ""
        else:
            outcome_str = ""
        draw.text((48, 200), ticker, fill=_FG, font=_load_font(220))
        if outcome_str:
            draw.text((48, 460), outcome_str, fill=_MUTED, font=_load_font(48))
    else:
        draw.text(
            (48, 240),
            "No closed losers in window.",
            fill=_FG,
            font=_load_font(56),
        )

    return _to_png_bytes(img)
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_og.py::TestRenderMistakesOg -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_og.py tests/v2/test_dashboard_og.py
git commit -m "feat(v2): add render_mistakes_og OG image renderer"
```

---

## Task 5: `render_attribution_og` — OG image with bar chart for `/attribution/`

**Files:**
- Modify: `v2/dashboard_og.py`
- Test: `tests/v2/test_dashboard_og.py`

Pure Pillow-drawn bars; top 5 signal types only. No external chart library. Verifies bar coordinates with `Image.getpixel` at known positions per the spec's testing note.

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_dashboard_og.py`:

```python
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
        # Walk a small box of pixels inside the first bar.
        non_bg_count = 0
        for x in range(195, 235):
            for y in range(330, 395):
                if img.getpixel((x, y)) != bg:
                    non_bg_count += 1
        assert non_bg_count > 0, "First bar did not render any non-background pixels"
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_og.py::TestRenderAttributionOg -v`
Expected: ImportError.

- [ ] **Step 3: Implement the OG renderer**

Append to `v2/dashboard_og.py`:

```python
def render_attribution_og(attribution: list[dict]) -> bytes:
    """Return PNG bytes (1200x630) showing the top-5 signal-attribution bars.

    Layout: title at top, baseline at y=400. Bars are 80px wide, 50px gap,
    starting at x=180. Positive bars (avg_outcome_30d > 0) draw upward in
    accent color; negative bars draw downward in muted color. Categories
    are labelled below the baseline.
    """
    img, draw = _canvas()

    draw.text((48, 50), "WHAT'S ACTUALLY WORKING", fill=_ACCENT, font=_load_font(48))
    draw.text((48, 110), "signal attribution (avg 30d outcome)", fill=_MUTED, font=_load_font(28))

    if not attribution:
        draw.text((48, 280), "Not enough samples yet.", fill=_FG, font=_load_font(56))
        return _to_png_bytes(img)

    BASELINE = 400
    BAR_W = 80
    GAP = 50
    X0 = 180
    MAX_BAR_PX = 200

    # Top-5 by sample_size (largest sample first), or fall back to whatever's
    # there if fewer rows. Mirror the spec's "top 5 signal types only".
    top = sorted(
        attribution,
        key=lambda r: int(r.get("sample_size") or 0),
        reverse=True,
    )[:5]

    # Find scale across the displayed slice
    scores = [
        float(r.get("avg_outcome_30d") or 0)
        for r in top
    ]
    max_abs = max((abs(s) for s in scores), default=1.0) or 1.0

    # Baseline line
    draw.line([(48, BASELINE), (1200 - 48, BASELINE)], fill=_MUTED, width=2)

    label_font = _load_font(24)
    for i, row in enumerate(top):
        x = X0 + i * (BAR_W + GAP)
        score = float(row.get("avg_outcome_30d") or 0)
        height_px = int(round((abs(score) / max_abs) * MAX_BAR_PX))
        if score >= 0:
            top_y = BASELINE - height_px
            color = _ACCENT
            draw.rectangle([(x, top_y), (x + BAR_W, BASELINE)], fill=color)
        else:
            color = _MUTED
            draw.rectangle([(x, BASELINE), (x + BAR_W, BASELINE + height_px)], fill=color)

        # Category label
        category = str(row.get("category") or "")
        # Truncate long category names so they fit under the bar
        if len(category) > 10:
            category = category[:9] + "…"
        draw.text((x, BASELINE + MAX_BAR_PX + 20), category, fill=_FG, font=label_font)

    return _to_png_bytes(img)
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_og.py::TestRenderAttributionOg -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_og.py tests/v2/test_dashboard_og.py
git commit -m "feat(v2): add render_attribution_og bar-chart OG renderer"
```

---

## Task 6: Extend `gather_dashboard_data` with mistakes/attribution data

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

Add two new keys to the gathered-data dict:
- `mistakes`: `{"closed_losers": [...], "retired_rules": [...]}`
- `attribution`: list of attribution rows (output of `get_signal_attribution`)

Both written to JSON by the existing `write_json_files` once we extend its allowlist.

- [ ] **Step 1: Write the failing test**

Append to `tests/v2/test_dashboard_publish.py` (a new class — append after the last existing class):

```python
class TestGatherDashboardDataMistakesAttribution:
    def test_includes_mistakes_and_attribution_keys(self, mock_db, mock_cursor):
        from datetime import date
        from v2.dashboard_publish import gather_dashboard_data

        # gather_dashboard_data executes many cursor calls in sequence.
        # We stub the new helpers via patch since they live in trading_db
        # and are imported into dashboard_publish.
        from unittest.mock import patch
        with patch("v2.dashboard_publish.get_closed_losers", return_value=[
                    {"id": 1, "ticker": "TSLA", "outcome_30d": -12.0}]), \
             patch("v2.dashboard_publish.get_retired_rules", return_value=[
                    {"id": 1, "rule_text": "X"}]), \
             patch("v2.dashboard_publish.get_signal_attribution", return_value=[
                    {"category": "earnings", "sample_size": 20,
                     "avg_outcome_30d": 1.2}]), \
             patch("v2.dashboard_publish.fetch_spy_benchmark", return_value=[]):

            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = None

            data = gather_dashboard_data(date(2026, 5, 4))

        assert "mistakes" in data
        assert data["mistakes"]["closed_losers"][0]["ticker"] == "TSLA"
        assert data["mistakes"]["retired_rules"][0]["rule_text"] == "X"
        assert "attribution" in data
        assert data["attribution"][0]["category"] == "earnings"
```

- [ ] **Step 2: Run the test — verify it fails**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_publish.py::TestGatherDashboardDataMistakesAttribution -v`
Expected: KeyError on `mistakes` (or AttributeError patching `get_closed_losers`).

- [ ] **Step 3: Wire the helpers into `gather_dashboard_data`**

In `v2/dashboard_publish.py`, update the import block at the top of the file (search for `from .database.trading_db import` — there's a SHIM block — but more straightforwardly, scan around `from .dashboard_pages import`). Add a new import group after the existing imports:

```python
from .database.trading_db import (
    get_closed_losers,
    get_retired_rules,
    get_signal_attribution,
)
```

If those imports already exist for other reasons, just add the missing names to the existing import statement.

Then in `gather_dashboard_data`, just before the `return {...}` block (around line 295), add:

```python
    # Mistakes log (closed losers + recently retired rules)
    try:
        closed_losers = get_closed_losers(reference_date=session_date, limit=15)
    except Exception:
        logger.warning("Failed to gather closed losers", exc_info=True)
        closed_losers = []
    try:
        retired_rules = get_retired_rules(reference_date=session_date, limit=10)
    except Exception:
        logger.warning("Failed to gather retired rules", exc_info=True)
        retired_rules = []
    mistakes = {
        "closed_losers": [dict(r) for r in closed_losers],
        "retired_rules": [dict(r) for r in retired_rules],
    }

    # Signal attribution snapshot
    try:
        attribution_rows = get_signal_attribution()
    except Exception:
        logger.warning("Failed to gather signal attribution", exc_info=True)
        attribution_rows = []
    attribution = [dict(r) for r in attribution_rows]
```

Then extend the `return {...}` to include the two new keys:

```python
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
        "mistakes": mistakes,
        "attribution": attribution,
        "_pages": pages,
    }
```

Also extend the JSON allowlist in `write_json_files` (around line 635):

```python
    for key in (
        "summary", "snapshots", "positions", "decisions",
        "theses", "benchmark", "mistakes", "attribution",
    ):
```

- [ ] **Step 4: Run the test — verify it passes**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_publish.py::TestGatherDashboardDataMistakesAttribution -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): gather_dashboard_data emits mistakes + attribution JSON"
```

---

## Task 7: `emit_static_pages` — write `/mistakes/` and `/attribution/` HTML + OG

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

A small helper that takes the data dict and the deploy directory and writes:
- `<deploy>/mistakes/index.html`
- `<deploy>/attribution/index.html`
- `<deploy>/og/mistakes.png`
- `<deploy>/og/attribution.png`

Then wire it into `assemble_deploy_dir`.

- [ ] **Step 1: Write the failing test**

Append to `tests/v2/test_dashboard_publish.py`:

```python
class TestEmitStaticPages:
    def test_writes_mistakes_and_attribution_files(self, tmp_path):
        from decimal import Decimal
        from v2.dashboard_publish import emit_static_pages

        data = {
            "mistakes": {
                "closed_losers": [
                    {"id": 1, "date": "2026-04-30", "ticker": "TSLA",
                     "action": "buy", "quantity": 5, "price": 200,
                     "reasoning": "EV", "outcome_7d": Decimal("-3.0"),
                     "outcome_30d": Decimal("-12.0")},
                ],
                "retired_rules": [],
            },
            "attribution": [
                {"category": "earnings", "sample_size": 30, "sample_size_30d": 24,
                 "avg_outcome_7d": Decimal("1.2"), "avg_outcome_30d": Decimal("3.4"),
                 "win_rate_7d": Decimal("0.6"), "win_rate_30d": Decimal("0.5")},
            ],
        }

        emit_static_pages(data, str(tmp_path), base_url="https://example.com")

        mistakes_html = (tmp_path / "mistakes" / "index.html").read_text()
        assert "TSLA" in mistakes_html
        attribution_html = (tmp_path / "attribution" / "index.html").read_text()
        assert "earnings" in attribution_html

        mistakes_png = (tmp_path / "og" / "mistakes.png").read_bytes()
        assert mistakes_png[:8] == b"\x89PNG\r\n\x1a\n"
        attribution_png = (tmp_path / "og" / "attribution.png").read_bytes()
        assert attribution_png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_no_op_when_base_url_missing(self, tmp_path):
        from v2.dashboard_publish import emit_static_pages

        emit_static_pages({"mistakes": {"closed_losers": [], "retired_rules": []},
                           "attribution": []}, str(tmp_path), base_url="")

        # No files should have been written
        assert not (tmp_path / "mistakes").exists()
        assert not (tmp_path / "attribution").exists()
```

- [ ] **Step 2: Run the test — verify it fails**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_publish.py::TestEmitStaticPages -v`
Expected: ImportError.

- [ ] **Step 3: Implement `emit_static_pages` and wire it into `assemble_deploy_dir`**

Add the imports near the existing dashboard_pages/dashboard_og imports in `v2/dashboard_publish.py`:

```python
from .dashboard_pages import (
    render_homepage_meta,
    render_trade_page,
    render_thesis_page,
    render_mistakes_page,
    render_attribution_page,
)
from .dashboard_og import (
    render_home_og,
    render_trade_og,
    render_thesis_og,
    render_mistakes_og,
    render_attribution_og,
)
```

Add the new function (place near `emit_home_og_image`, ~line 423):

```python
def emit_static_pages(data: dict, deploy_dir: str, base_url: str) -> None:
    """Write /mistakes/index.html, /attribution/index.html, and the
    matching OG PNGs into deploy_dir.

    No-op when base_url is empty (local-only build path).
    """
    if not base_url:
        return

    mistakes = data.get("mistakes") or {"closed_losers": [], "retired_rules": []}
    attribution = data.get("attribution") or []

    # /mistakes/index.html
    try:
        html = render_mistakes_page(
            closed_losers=mistakes.get("closed_losers", []),
            retired_rules=mistakes.get("retired_rules", []),
            base_url=base_url,
        )
        page_dir = os.path.join(deploy_dir, "mistakes")
        os.makedirs(page_dir, exist_ok=True)
        with open(os.path.join(page_dir, "index.html"), "w") as f:
            f.write(html)
    except Exception:
        logger.warning("Failed to render /mistakes/", exc_info=True)

    # /attribution/index.html
    try:
        html = render_attribution_page(
            attribution=attribution,
            base_url=base_url,
        )
        page_dir = os.path.join(deploy_dir, "attribution")
        os.makedirs(page_dir, exist_ok=True)
        with open(os.path.join(page_dir, "index.html"), "w") as f:
            f.write(html)
    except Exception:
        logger.warning("Failed to render /attribution/", exc_info=True)

    # OG images
    og_dir = os.path.join(deploy_dir, "og")
    os.makedirs(og_dir, exist_ok=True)
    losers = mistakes.get("closed_losers", [])
    top_loser = losers[0] if losers else None
    try:
        png = render_mistakes_og(top_loser=top_loser)
        with open(os.path.join(og_dir, "mistakes.png"), "wb") as f:
            f.write(png)
    except Exception:
        logger.warning("Failed to render mistakes OG", exc_info=True)
    try:
        png = render_attribution_og(attribution=attribution)
        with open(os.path.join(og_dir, "attribution.png"), "wb") as f:
            f.write(png)
    except Exception:
        logger.warning("Failed to render attribution OG", exc_info=True)
```

Then call it from `assemble_deploy_dir`. After the existing `emit_home_og_image(...)` call (around line 675):

```python
    emit_home_og_image(data.get("summary", {}), deploy_dir)
    emit_static_pages(data, deploy_dir, base_url=base_url)
```

- [ ] **Step 4: Run the test — verify it passes**

Run: `docker compose exec trading python -m pytest tests/v2/test_dashboard_publish.py::TestEmitStaticPages -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat(v2): emit /mistakes/ and /attribution/ permalinks + OG"
```

---

## Task 8: Homepage teaser sections — `#mistakes` and `#attribution`

**Files:**
- Modify: `public_dashboard/index.html`
- Modify: `public_dashboard/app.js`

Add two empty `<section class="panel">` blocks at the bottom of `<main>` and have `app.js` populate them by fetching `mistakes.json` and `attribution.json`. The attribution section uses Chart.js (already loaded by index.html) for a small bar chart.

- [ ] **Step 1: Add the section markup**

In `public_dashboard/index.html`, just before `</main>` (search for `<!-- Active Theses -->` and add after that section's closing `</section>`):

```html
        <!-- Mistakes Log -->
        <section class="panel" id="mistakes">
            <h2>What didn't work</h2>
            <p class="subtitle">Closed losers + retired rules. <a href="/mistakes/">See all</a></p>
            <h3>Recent losers</h3>
            <div class="table-wrap">
                <table id="mistakes-losers-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Action</th>
                            <th class="num">30d</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
            <p class="empty-state" id="mistakes-empty" style="display:none;">No closed losers in window.</p>
            <h3>Retired rules</h3>
            <ul id="mistakes-rules-list"></ul>
        </section>

        <!-- Attribution -->
        <section class="panel" id="attribution">
            <h2>What's actually working</h2>
            <p class="subtitle">Top signal types by sample size. <a href="/attribution/">See all</a></p>
            <div class="chart-wrap">
                <canvas id="attribution-chart"></canvas>
            </div>
            <p class="empty-state" id="attribution-empty" style="display:none;">Not enough samples yet.</p>
        </section>
```

- [ ] **Step 2: Add the rendering JS**

Open `public_dashboard/app.js`. Find the existing `Promise.all` or fetch-orchestration block (search for `fetch('data/`). Add two new fetches and new render functions:

If app.js follows a `loadData()` pattern that calls `fetch('data/X.json')` for each key, add:

```javascript
fetch('data/mistakes.json').then(r => r.ok ? r.json() : null).then(renderMistakes).catch(() => renderMistakes(null));
fetch('data/attribution.json').then(r => r.ok ? r.json() : null).then(renderAttribution).catch(() => renderAttribution(null));
```

Append the render functions (anywhere after the existing render helpers):

```javascript
function renderMistakes(data) {
    const tbody = document.querySelector('#mistakes-losers-table tbody');
    const empty = document.getElementById('mistakes-empty');
    const rulesList = document.getElementById('mistakes-rules-list');
    if (!data || (!data.closed_losers?.length && !data.retired_rules?.length)) {
        if (empty) empty.style.display = '';
        return;
    }
    const losers = (data.closed_losers || []).slice(0, 3);
    if (tbody) {
        tbody.innerHTML = losers.map(d => {
            const o30 = d.outcome_30d != null ? Number(d.outcome_30d).toFixed(2) + '%' : '—';
            return `<tr>
              <td><a href="/trade/${d.id}/">${escapeHTML(d.ticker || '')}</a></td>
              <td>${escapeHTML((d.action || '').toUpperCase())}</td>
              <td class="num">${o30}</td>
            </tr>`;
        }).join('');
    }
    const rules = (data.retired_rules || []).slice(0, 2);
    if (rulesList) {
        rulesList.innerHTML = rules.map(r =>
            `<li>${escapeHTML(r.rule_text || '')}<br>
              <span class="rule-meta">retired ${escapeHTML(String(r.retired_at || ''))}
              — ${escapeHTML(r.retirement_reason || '')}</span></li>`
        ).join('');
    }
}

function renderAttribution(data) {
    const empty = document.getElementById('attribution-empty');
    const canvas = document.getElementById('attribution-chart');
    if (!data || !data.length) {
        if (empty) empty.style.display = '';
        if (canvas) canvas.style.display = 'none';
        return;
    }
    const top = [...data]
        .sort((a, b) => (b.sample_size || 0) - (a.sample_size || 0))
        .slice(0, 5);
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: top.map(r => r.category),
            datasets: [{
                label: 'avg 30d outcome',
                data: top.map(r => Number(r.avg_outcome_30d || 0)),
                backgroundColor: top.map(r =>
                    Number(r.avg_outcome_30d || 0) >= 0
                        ? 'rgba(0,212,170,0.7)'
                        : 'rgba(220,80,80,0.7)'
                ),
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } },
        }
    });
}

function escapeHTML(s) {
    return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
```

If `escapeHTML` already exists in `app.js`, do not redefine it — drop that helper from the snippet.

- [ ] **Step 3: Manual smoke test**

Run: `docker compose exec trading python -c "
from datetime import date
from v2.dashboard_publish import gather_dashboard_data, assemble_deploy_dir
import os, tempfile
tmp = tempfile.mkdtemp(prefix='smoke_')
os.environ.setdefault('DASHBOARD_URL', 'https://example.com')
data = gather_dashboard_data(date.today())
print('keys:', list(data.keys()))
assemble_deploy_dir(data, tmp, '/app/public_dashboard', base_url='https://example.com')
print('emitted:', sorted(os.listdir(tmp)))
"`

Expected output should include `'mistakes'` and `'attribution'` in the keys list and `'mistakes'` + `'attribution'` directories in the emitted list. (Failing because `'/app/public_dashboard'` is wrong is fine — adjust the assets path to wherever the container mounts the dashboard assets, or skip this step if the path isn't readily known.)

- [ ] **Step 4: Commit**

```bash
git add public_dashboard/index.html public_dashboard/app.js
git commit -m "feat(dashboard): add #mistakes and #attribution homepage teasers"
```

---

## Task 9: `v2/social_weekly.py` — Mistakes context + post generation

**Files:**
- Create: `v2/social_weekly.py`
- Test: `tests/v2/test_social_weekly.py`

Mirrors `v2/premarket.py`. Pure helpers in this task; stage runner in Task 11.

- [ ] **Step 1: Wire `social_weekly` into the `mock_db` fixture**

Edit `tests/v2/conftest.py`. Find the `mock_db` fixture (~line 86) and add to its `with` chain (alongside the `v2.premarket.get_cursor` patch):

```python
         patch("v2.social_weekly.get_cursor", _get_cursor), \
```

- [ ] **Step 2: Write the failing tests**

Create `tests/v2/test_social_weekly.py`:

```python
"""Tests for v2/social_weekly.py — weekly mistakes + attribution social posts."""

import json
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch


def _make_claude_response(json_data: dict):
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(json_data))]
    return response


class TestGatherMistakesContext:
    def test_returns_top_loser_and_retired_rule(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_mistakes_context

        with patch("v2.social_weekly.get_closed_losers", return_value=[
                    {"id": 1, "ticker": "TSLA", "action": "buy",
                     "quantity": 5, "price": Decimal("200"),
                     "outcome_30d": Decimal("-12.5"),
                     "reasoning": "EV cycle"}]), \
             patch("v2.social_weekly.get_retired_rules", return_value=[
                    {"rule_text": "Cap macro at $500/day",
                     "retirement_reason": "stale"}]):
            ctx = gather_mistakes_context(today=date(2026, 5, 8))

        assert "TSLA" in ctx
        assert "-12.5" in ctx
        assert "Cap macro" in ctx

    def test_handles_empty_data(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_mistakes_context
        with patch("v2.social_weekly.get_closed_losers", return_value=[]), \
             patch("v2.social_weekly.get_retired_rules", return_value=[]):
            ctx = gather_mistakes_context(today=date(2026, 5, 8))
        assert ctx == ""


class TestGenerateMistakesPost:
    @patch("v2.social_weekly._call_with_retry")
    @patch("v2.social_weekly.get_claude_client")
    def test_generates_text(self, mock_get_client, mock_retry):
        from v2.social_weekly import generate_mistakes_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Worst trade this week: $TSLA -12.5%. Reason was thin."}
        )

        post = generate_mistakes_post("ctx", dashboard_base_url="https://example.com")

        assert post is not None
        assert "TSLA" in post["text"]
        assert "https://example.com/mistakes/" in post["text"]
        assert post["type"] == "weekly_mistakes"

    @patch("v2.social_weekly._call_with_retry", side_effect=Exception("API down"))
    @patch("v2.social_weekly.get_claude_client")
    def test_llm_failure_returns_none(self, mock_get_client, mock_retry):
        from v2.social_weekly import generate_mistakes_post

        mock_get_client.return_value = MagicMock()
        assert generate_mistakes_post("ctx", dashboard_base_url="") is None
```

- [ ] **Step 3: Run the tests — verify they fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestGatherMistakesContext tests/v2/test_social_weekly.py::TestGenerateMistakesPost -v`
Expected: ImportError on `v2.social_weekly`.

- [ ] **Step 4: Create `v2/social_weekly.py` with mistakes pieces**

```python
"""Weekly social posts -- Bikini Bottom Capital (v2).

Two scheduled-post functions:
- run_mistakes_post: "what didn't work" — links to /mistakes/
- run_attribution_post: signal-attribution roundup — links to /attribution/

Both run from cron Friday afternoon, after the daily session has had time
to publish Stage 6. Skipped on weekends / NYSE holidays via is_trading_day.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

from .claude_client import _call_with_retry, get_claude_client
from .database.connection import get_cursor  # noqa: F401  used via mock_db patch
from .database.trading_db import (
    get_closed_losers,
    get_retired_rules,
    get_signal_attribution,
    insert_tweet,
    posted_tweet_exists,
)
from .market_calendar import is_trading_day

logger = logging.getLogger("social_weekly")


# ---------------------------------------------------------------------------
# Mistakes — context, prompt, generator
# ---------------------------------------------------------------------------

MISTAKES_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital.
You post weekly about what the bot got wrong.

Your voice:
- Honest. Specific. No self-flagellation, no "valuable lesson learned".
- Treat losses as data, not embarrassment.
- Dry, not bitter.

Most trading accounts hide losses. You don't. That's the point.

Generate ONE post about this week's worst trade or retired rule.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL appended after).
- One specific thing — the worst trade, or the retired rule, not a list.
- Reference the actual ticker / rule, not "a position" or "a strategy".
- No "we'll do better next time" / no "lessons learned" cliché."""


def gather_mistakes_context(today: date | None = None) -> str:
    """Plain-text summary of recent losers + retired rules."""
    if today is None:
        today = date.today()

    losers = get_closed_losers(reference_date=today, limit=5)
    rules = get_retired_rules(reference_date=today, limit=5)

    parts: list[str] = []
    if losers:
        parts.append("RECENT LOSERS:")
        for d in losers:
            try:
                outcome = f"{float(d.get('outcome_30d') or 0):+.2f}%"
            except Exception:
                outcome = ""
            parts.append(
                f"  {d.get('ticker','?')} {str(d.get('action','')).upper()}"
                f" {d.get('quantity','?')} @ ${d.get('price','?')}"
                f" — 30d: {outcome}"
                f"  ({d.get('reasoning','')})"
            )
    if rules:
        parts.append("\nRETIRED RULES:")
        for r in rules:
            parts.append(
                f"  {r.get('rule_text','')} "
                f"(reason: {r.get('retirement_reason','')})"
            )

    return "\n".join(parts) if parts else ""


def _generate_post(
    *,
    system_prompt: str,
    context: str,
    type_label: str,
    permalink: str,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Shared LLM call + URL append for both weekly post types."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": context}],
        )
        raw = response.content[0].text.strip()
        logger.info("AI response (%s):\n%s", type_label, raw)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except Exception as e:
        logger.error("Failed to generate %s post: %s", type_label, e)
        return None

    body = result.get("text")
    if not body or not isinstance(body, str):
        logger.warning("LLM returned no text or malformed response: %s", result)
        return None

    suffix = ""
    if dashboard_base_url:
        suffix = "\n" + dashboard_base_url.rstrip("/") + permalink
    return {"text": body + suffix, "type": type_label}


def generate_mistakes_post(
    context: str,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one mistakes-post body."""
    return _generate_post(
        system_prompt=MISTAKES_SYSTEM_PROMPT,
        context=context,
        type_label="weekly_mistakes",
        permalink="/mistakes/",
        dashboard_base_url=dashboard_base_url,
        model=model,
    )
```

- [ ] **Step 5: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestGatherMistakesContext tests/v2/test_social_weekly.py::TestGenerateMistakesPost -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add v2/social_weekly.py tests/v2/test_social_weekly.py tests/v2/conftest.py
git commit -m "feat(v2): add weekly mistakes context + post generator"
```

---

## Task 10: Attribution context + post generation

**Files:**
- Modify: `v2/social_weekly.py`
- Modify: `tests/v2/test_social_weekly.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_social_weekly.py`:

```python
class TestGatherAttributionContext:
    def test_summarizes_top_and_bottom(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_attribution_context

        with patch("v2.social_weekly.get_signal_attribution", return_value=[
                    {"category": "earnings", "sample_size": 30,
                     "avg_outcome_30d": Decimal("3.4")},
                    {"category": "fed", "sample_size": 12,
                     "avg_outcome_30d": Decimal("-1.2")},
                    {"category": "macro", "sample_size": 9,
                     "avg_outcome_30d": Decimal("0.8")},
                ]):
            ctx = gather_attribution_context()

        assert "earnings" in ctx
        assert "fed" in ctx

    def test_handles_no_attribution(self, mock_db, mock_cursor):
        from v2.social_weekly import gather_attribution_context

        with patch("v2.social_weekly.get_signal_attribution", return_value=[]):
            ctx = gather_attribution_context()
        assert ctx == ""


class TestGenerateAttributionPost:
    @patch("v2.social_weekly._call_with_retry")
    @patch("v2.social_weekly.get_claude_client")
    def test_generates_text(self, mock_get_client, mock_retry):
        from v2.social_weekly import generate_attribution_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Earnings signals predicted (+3.4%, n=30); fed news was noise."}
        )
        post = generate_attribution_post("ctx", dashboard_base_url="https://example.com")
        assert post is not None
        assert "Earnings" in post["text"]
        assert "https://example.com/attribution/" in post["text"]
        assert post["type"] == "weekly_attribution"
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestGatherAttributionContext tests/v2/test_social_weekly.py::TestGenerateAttributionPost -v`
Expected: ImportError.

- [ ] **Step 3: Implement the helpers**

Append to `v2/social_weekly.py`:

```python
ATTRIBUTION_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital.
You post weekly about which signal types are actually predictive.

Your voice:
- Curious about the data.
- Comfortable saying "this one didn't work" without spinning it.
- A little nerdy. Slightly overshare-y about methodology.

Generate ONE post about this week's signal attribution scores.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL appended after).
- Name 1–2 signal types and their scores. Not all of them.
- One non-obvious observation, if there is one. Otherwise just the data.
- Don't claim "alpha". Use "predictive" / "useful" / "noise"."""


def gather_attribution_context() -> str:
    """Plain-text summary of best + worst signal types by avg_outcome_30d."""
    rows = get_signal_attribution()
    if not rows:
        return ""

    sortable = [r for r in rows if r.get("avg_outcome_30d") is not None]
    if not sortable:
        return ""
    sortable.sort(key=lambda r: float(r.get("avg_outcome_30d") or 0), reverse=True)

    top = sortable[:3]
    bottom = sortable[-3:][::-1]

    def _fmt(rs):
        out = []
        for r in rs:
            try:
                pct = f"{float(r.get('avg_outcome_30d') or 0):+.2f}%"
            except Exception:
                pct = ""
            out.append(
                f"  {r.get('category','?')}: {pct} (n={r.get('sample_size', 0)})"
            )
        return "\n".join(out)

    parts = ["BEST PREDICTORS:", _fmt(top)]
    if bottom and bottom[-1] is not top[-1]:
        parts.extend(["", "WORST PREDICTORS:", _fmt(bottom)])
    return "\n".join(parts)


def generate_attribution_post(
    context: str,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one attribution-post body."""
    return _generate_post(
        system_prompt=ATTRIBUTION_SYSTEM_PROMPT,
        context=context,
        type_label="weekly_attribution",
        permalink="/attribution/",
        dashboard_base_url=dashboard_base_url,
        model=model,
    )
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestGatherAttributionContext tests/v2/test_social_weekly.py::TestGenerateAttributionPost -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/social_weekly.py tests/v2/test_social_weekly.py
git commit -m "feat(v2): add weekly attribution context + post generator"
```

---

## Task 11: `run_mistakes_post` and `run_attribution_post` stage runners

**Files:**
- Modify: `v2/social_weekly.py`
- Modify: `tests/v2/test_social_weekly.py`

Both runners share a single per-platform helper (avoid copy-paste). Skip on non-trading days. Idempotent via `posted_tweet_exists(today, type_label, platform)`. Honor `ALGO_TRADE_POST_DRY_RUN=1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_social_weekly.py`:

```python
class TestRunMistakesPost:
    @patch("v2.social_weekly.is_trading_day", return_value=False)
    def test_skipped_on_weekend(self, mock_is_td):
        from v2.social_weekly import run_mistakes_post

        result = run_mistakes_post(today=date(2026, 5, 9))  # Saturday
        assert result.skipped is True

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.gather_mistakes_context", return_value="")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_skipped_when_no_data(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_is_td,
    ):
        from v2.social_weekly import run_mistakes_post

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()

        result = run_mistakes_post(today=date(2026, 5, 8))
        assert result.skipped is True
        assert "no data" in (result.skip_reason or "").lower()

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.posted_tweet_exists", return_value=True)
    @patch("v2.social_weekly.gather_mistakes_context", return_value="ctx")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_skipped_when_already_posted(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_dedup, mock_is_td,
    ):
        from v2.social_weekly import run_mistakes_post

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        result = run_mistakes_post(today=date(2026, 5, 8))
        assert result.skipped is True

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.insert_tweet", return_value=1)
    @patch("v2.social_weekly.posted_tweet_exists", return_value=False)
    @patch("v2.social_weekly.post_to_bluesky")
    @patch("v2.social_weekly.post_tweet")
    @patch("v2.social_weekly.generate_mistakes_post")
    @patch("v2.social_weekly.gather_mistakes_context", return_value="ctx")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_posts_to_both_platforms(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_gen,
        mock_post_tw, mock_post_bs, mock_dedup, mock_insert, mock_is_td,
    ):
        from v2.social_weekly import run_mistakes_post

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        mock_gen.return_value = {"text": "x", "type": "weekly_mistakes"}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "weekly_mistakes",
                                     "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "weekly_mistakes",
                                     "error": None}

        result = run_mistakes_post(today=date(2026, 5, 8))
        assert result.skipped is False
        assert result.twitter_posted is True
        assert result.bluesky_posted is True
        assert mock_insert.call_count == 2


class TestRunAttributionPost:
    @patch("v2.social_weekly.is_trading_day", return_value=False)
    def test_skipped_on_weekend(self, mock_is_td):
        from v2.social_weekly import run_attribution_post

        result = run_attribution_post(today=date(2026, 5, 9))
        assert result.skipped is True

    @patch("v2.social_weekly.is_trading_day", return_value=True)
    @patch("v2.social_weekly.gather_attribution_context", return_value="")
    @patch("v2.social_weekly.get_twitter_client")
    @patch("v2.social_weekly.get_bluesky_client")
    def test_skipped_when_no_data(
        self, mock_bs, mock_tw, mock_ctx, mock_is_td,
    ):
        from v2.social_weekly import run_attribution_post

        mock_tw.return_value = object()
        mock_bs.return_value = object()
        result = run_attribution_post(today=date(2026, 5, 8))
        assert result.skipped is True
```

- [ ] **Step 2: Run the tests — verify they fail**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestRunMistakesPost tests/v2/test_social_weekly.py::TestRunAttributionPost -v`
Expected: ImportError on the run_* names.

- [ ] **Step 3: Implement the runners**

Append to `v2/social_weekly.py` (after the helpers from Tasks 9–10):

```python
# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

# Imported late so the pure helpers above stay testable without dragging in
# the full social-platform stack.
from .twitter import get_twitter_client, post_tweet           # noqa: E402
from .bluesky import get_bluesky_client, post_to_bluesky      # noqa: E402


@dataclass
class WeeklyPostResult:
    skipped: bool = False
    skip_reason: str | None = None
    twitter_posted: bool = False
    bluesky_posted: bool = False
    errors: list[str] = field(default_factory=list)


def _is_dry_run() -> bool:
    return os.environ.get("ALGO_TRADE_POST_DRY_RUN") == "1"


def _post_one(
    *,
    platform: str,
    client,
    poster,
    post_body: dict,
    today: date,
    type_label: str,
    result: WeeklyPostResult,
) -> None:
    if _is_dry_run():
        logger.info("[DRY-RUN] %s %s post:\n%s",
                    type_label, platform, post_body["text"])
        if platform == "twitter":
            result.twitter_posted = True
        else:
            result.bluesky_posted = True
        return

    try:
        post_result = poster(post_body, client=client)
        insert_tweet(
            session_date=today,
            tweet_type=type_label,
            tweet_text=post_result["text"],
            tweet_id=post_result.get("tweet_id") or post_result.get("post_id"),
            posted=post_result["posted"],
            error=post_result.get("error"),
            platform=platform,
        )
        if post_result["posted"]:
            if platform == "twitter":
                result.twitter_posted = True
            else:
                result.bluesky_posted = True
    except Exception as e:
        result.errors.append(f"{platform} {type_label} post/log failed: {e}")
        logger.error("%s %s post/log failed: %s", platform, type_label, e)


def _run_weekly(
    *,
    today: date | None,
    type_label: str,
    gather: callable,
    generate: callable,
) -> WeeklyPostResult:
    if today is None:
        today = date.today()

    result = WeeklyPostResult()

    if not is_trading_day(today):
        result.skipped = True
        result.skip_reason = f"{today} is not a trading day"
        logger.info("Weekly %s skipped — %s", type_label, result.skip_reason)
        return result

    twitter_client = get_twitter_client()
    bluesky_client = get_bluesky_client()
    if twitter_client is None and bluesky_client is None:
        result.skipped = True
        result.skip_reason = "no platform credentials"
        logger.info("Weekly %s skipped — no platform credentials", type_label)
        return result

    try:
        context = gather(today=today) if type_label == "weekly_mistakes" else gather()
    except Exception as e:
        result.errors.append(f"Context gather failed: {e}")
        logger.error("%s context gather failed: %s", type_label, e)
        return result

    if not context:
        result.skipped = True
        result.skip_reason = "no data this window"
        logger.info("Weekly %s skipped — no data", type_label)
        return result

    # Idempotency check — both platforms
    tw_already = False
    bs_already = False
    try:
        if twitter_client is not None:
            tw_already = posted_tweet_exists(today, type_label, "twitter")
        if bluesky_client is not None:
            bs_already = posted_tweet_exists(today, type_label, "bluesky")
    except Exception as e:
        logger.warning("Weekly %s dedup check failed: %s; proceeding",
                       type_label, e)

    if (twitter_client is None or tw_already) and (bluesky_client is None or bs_already):
        result.skipped = True
        result.skip_reason = "already posted on all configured platforms"
        logger.info("Weekly %s skipped — already posted today", type_label)
        return result

    dashboard_base_url = os.environ.get("DASHBOARD_URL", "")
    post_body = generate(context, dashboard_base_url=dashboard_base_url)
    if post_body is None:
        result.errors.append("LLM generation returned None")
        return result

    if twitter_client is not None and not tw_already:
        _post_one(
            platform="twitter", client=twitter_client, poster=post_tweet,
            post_body=post_body, today=today, type_label=type_label, result=result,
        )
    if bluesky_client is not None and not bs_already:
        _post_one(
            platform="bluesky", client=bluesky_client, poster=post_to_bluesky,
            post_body=post_body, today=today, type_label=type_label, result=result,
        )

    logger.info("Weekly %s complete: twitter=%s, bluesky=%s",
                type_label, result.twitter_posted, result.bluesky_posted)
    return result


def run_mistakes_post(today: date | None = None) -> WeeklyPostResult:
    return _run_weekly(
        today=today,
        type_label="weekly_mistakes",
        gather=gather_mistakes_context,
        generate=generate_mistakes_post,
    )


def run_attribution_post(today: date | None = None) -> WeeklyPostResult:
    return _run_weekly(
        today=today,
        type_label="weekly_attribution",
        gather=gather_attribution_context,
        generate=generate_attribution_post,
    )
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestRunMistakesPost tests/v2/test_social_weekly.py::TestRunAttributionPost -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/social_weekly.py tests/v2/test_social_weekly.py
git commit -m "feat(v2): add run_mistakes_post and run_attribution_post stage runners"
```

---

## Task 12: CLI entrypoint — `python -m v2.social_weekly mistakes|attribution`

**Files:**
- Modify: `v2/social_weekly.py`
- Modify: `tests/v2/test_social_weekly.py`

Argparse subcommand. Exit 1 on errors so cron sees the failure.

- [ ] **Step 1: Write the failing test**

Append to `tests/v2/test_social_weekly.py`:

```python
class TestCLI:
    @patch("v2.social_weekly.run_mistakes_post")
    def test_mistakes_subcommand_calls_runner(self, mock_run):
        from v2.social_weekly import _main

        mock_run.return_value = MagicMock(errors=[])
        rc = _main(["mistakes"])
        assert rc == 0
        mock_run.assert_called_once()

    @patch("v2.social_weekly.run_attribution_post")
    def test_attribution_subcommand_calls_runner(self, mock_run):
        from v2.social_weekly import _main

        mock_run.return_value = MagicMock(errors=[])
        rc = _main(["attribution"])
        assert rc == 0
        mock_run.assert_called_once()

    @patch("v2.social_weekly.run_mistakes_post")
    def test_nonzero_exit_on_errors(self, mock_run):
        from v2.social_weekly import _main

        mock_run.return_value = MagicMock(errors=["boom"])
        rc = _main(["mistakes"])
        assert rc == 1
```

- [ ] **Step 2: Run the test — verify it fails**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestCLI -v`
Expected: ImportError on `_main`.

- [ ] **Step 3: Implement `_main` and `__main__`**

Append to `v2/social_weekly.py`:

```python
def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v2.social_weekly",
        description="Weekly social posts (mistakes / attribution).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mistakes", help="Run the weekly mistakes post.")
    sub.add_parser("attribution", help="Run the weekly attribution post.")
    args = parser.parse_args(argv)

    if args.cmd == "mistakes":
        result = run_mistakes_post()
    else:
        result = run_attribution_post()
    return 1 if getattr(result, "errors", []) else 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    sys.exit(_main())
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py::TestCLI -v`
Expected: 3 passed.

- [ ] **Step 5: Run full new-test suite to confirm nothing regressed**

Run: `docker compose exec trading python -m pytest tests/v2/test_social_weekly.py tests/v2/test_dashboard_pages.py tests/v2/test_dashboard_og.py tests/v2/test_dashboard_publish.py tests/v2/test_trading_db.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add v2/social_weekly.py tests/v2/test_social_weekly.py
git commit -m "feat(v2): add v2.social_weekly CLI entrypoint"
```

---

## Task 13: Taskfile + crontab entries

**Files:**
- Modify: `Taskfile.yml`
- Modify: `crontab`

Friday post times: 14:00 MST (mistakes), 14:15 MST (attribution). 1pm MST daily session normally finishes well before 14:00, so the linked pages are fresh.

- [ ] **Step 1: Add Taskfile targets**

Open `Taskfile.yml`. Find the existing `premarket:` task (added by Spec #2) and add right after it:

```yaml
  weekly:mistakes:
    desc: Run the weekly mistakes social post (Friday only; self-skips otherwise)
    deps: [docker:up]
    cmds:
      - docker compose exec trading python -m v2.social_weekly mistakes {{.CLI_ARGS}}

  weekly:attribution:
    desc: Run the weekly attribution social post (Friday only; self-skips otherwise)
    deps: [docker:up]
    cmds:
      - docker compose exec trading python -m v2.social_weekly attribution {{.CLI_ARGS}}
```

- [ ] **Step 2: Add crontab entries**

Open `crontab`. Insert these two new entries between the daily-session block and the weekly-learning block (preserve the `# Weekly deep learning analysis` comment header for what follows):

```cron
# Weekly mistakes post (2:00 PM MST / 4:00 PM ET, Friday)
# Posts "what I got wrong" with a link to /mistakes/.
# Self-skips on NYSE holidays. Idempotent on retries.
0 14 * * 5 /home/jay/dev/algo/run-docker.sh trading python -m v2.social_weekly mistakes

# Weekly attribution post (2:15 PM MST / 4:15 PM ET, Friday)
# Posts the attribution roundup with a link to /attribution/.
15 14 * * 5 /home/jay/dev/algo/run-docker.sh trading python -m v2.social_weekly attribution
```

- [ ] **Step 3: Verify the Taskfile target dispatches**

Run: `task --list 2>&1 | grep weekly`
Expected: lists both `weekly:mistakes` and `weekly:attribution` targets.

- [ ] **Step 4: Verify the crontab parses**

Run: `crontab -T crontab 2>&1` — if the local cron supports `-T` it dry-runs validation. Otherwise just visually confirm with: `cat crontab` and ensure the new lines are present.

- [ ] **Step 5: Commit**

```bash
git add Taskfile.yml crontab
git commit -m "chore: add Friday weekly mistakes/attribution cron + task targets"
```

> NOTE: Installing the new crontab onto the host (`crontab /home/jay/dev/algo/crontab`) is a separate operator step. Mention it in the PR description; do not run it in CI.

---

## Task 14: Documentation — CLAUDE.md and README

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

In `CLAUDE.md`, find the `### Live-trade pipeline feature flag` section (introduced in Spec #2). Add a new sibling section right after it:

```markdown
### Weekly social posts

Two cron-triggered posts every Friday afternoon, separate from the daily
session:

- `python -m v2.social_weekly mistakes` — posts "what didn't work" with a
  link to `/mistakes/` on the public dashboard. Surfaces the worst recent
  closed loser or a recently retired rule.
- `python -m v2.social_weekly attribution` — posts the signal-attribution
  roundup with a link to `/attribution/`. Names 1–2 best/worst signal
  types.

Both:
- Self-skip on weekends and NYSE holidays.
- Idempotent on retries via `posted_tweet_exists(today, type_label, platform)`.
- Skip with a non-error log when the underlying data is empty (no losers
  this week / not enough attribution samples yet).
- Honor `ALGO_TRADE_POST_DRY_RUN=1`.

The dashboard publishes `/mistakes/` and `/attribution/` permalinks on every
Stage 6 run, so the linked pages are always fresh against the previous daily
session's data.
```

- [ ] **Step 2: Update README.md**

In `README.md`, find the `Pre-Market Post` subsection (added by Spec #2). Add a new sibling subsection right after it:

```markdown
### Weekly Mistakes & Attribution Posts

Two posts that run every Friday afternoon, independent of the daily session:

```bash
# Friday 14:00 MST — "what didn't work" + link to /mistakes/
task weekly:mistakes
# or directly:
docker compose exec trading python -m v2.social_weekly mistakes

# Friday 14:15 MST — attribution roundup + link to /attribution/
task weekly:attribution
docker compose exec trading python -m v2.social_weekly attribution
```

Both self-skip on weekends and NYSE holidays. Both honor
`ALGO_TRADE_POST_DRY_RUN=1` for log-only runs that don't post or write to the
DB. Both skip with a non-error log when the underlying data is empty.

The cron entries live in the repo `crontab` file. After editing the file in
the repo, install it on the host with:

```bash
crontab /home/jay/dev/algo/crontab
```
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document v2.social_weekly mistakes + attribution pipeline"
```

---

## Task 15: Final verification

**Files:** none

- [ ] **Step 1: Run full test suite for changed modules**

Run:
```
docker compose exec trading python -m pytest \
  tests/v2/test_social_weekly.py \
  tests/v2/test_dashboard_pages.py \
  tests/v2/test_dashboard_og.py \
  tests/v2/test_dashboard_publish.py \
  tests/v2/test_trading_db.py \
  tests/v2/test_premarket.py \
  tests/v2/test_social_trades.py -v
```
Expected: all green. (Don't run the whole v2 suite in CI scope — known-flaky `test_session.py` / `test_entertainment.py` cases are unrelated.)

- [ ] **Step 2: Dry-run the mistakes post in the container**

Run:
```
docker compose exec -e ALGO_TRADE_POST_DRY_RUN=1 trading python -m v2.social_weekly mistakes
```
Expected: log lines showing context + generated text + `[DRY-RUN]` per platform; exit 0. If today is a weekend / holiday, expect "skipped" and exit 0 — that's correct.

- [ ] **Step 3: Dry-run the attribution post**

Run:
```
docker compose exec -e ALGO_TRADE_POST_DRY_RUN=1 trading python -m v2.social_weekly attribution
```
Expected: same shape as Step 2.

- [ ] **Step 4: Verify the dashboard publish emits the new pages**

Trigger a one-off publish (or read the latest deploy directory). The simplest check:

```bash
docker compose exec trading python -c "
from datetime import date
from v2.dashboard_publish import gather_dashboard_data
data = gather_dashboard_data(date.today())
print('mistakes keys:', list((data.get('mistakes') or {}).keys()))
print('attribution rows:', len(data.get('attribution') or []))
"
```
Expected: prints `mistakes keys: ['closed_losers', 'retired_rules']` and an integer count for attribution rows. (Either count may be 0 in the early days — that's fine.)

- [ ] **Step 5: PR title + description**

Suggested PR title: `feat(v2): mistakes log + attribution panel + Friday weekly posts (Spec #3)`

PR body should include:
- One-line summary
- "Closes Spec #3" link to `docs/superpowers/specs/2026-05-03-mistakes-attribution-design.md`
- Operator note: install the new crontab on the host (`crontab /home/jay/dev/algo/crontab`) after merge.
- Note: feature is auto-on (no env flag) — first Friday after merge, posts will fire if there is data.

---

## Self-Review

| Spec section                                  | Tasks    | Notes |
|-----------------------------------------------|----------|-------|
| Mistakes dashboard section + permalink        | 2, 7, 8  | `/mistakes/` page + homepage teaser |
| Attribution dashboard section + permalink     | 3, 7, 8  | `/attribution/` page + homepage chart |
| Mistakes OG image                             | 4, 7     | Pillow-rendered, top-loser lead |
| Attribution OG image (bar chart)              | 5, 7     | Pillow-drawn bars; `Image.getpixel` smoke check |
| `gather_dashboard_data` extension             | 6        | New `mistakes`/`attribution` keys |
| `v2/social_weekly.py`                         | 9–12     | Pure helpers + runners + CLI |
| Friday schedule                               | 13       | Crontab + Taskfile |
| System prompts                                | 9, 10    | Pulled from spec verbatim |
| Empty-data skip behavior                      | 11       | "no data this window" path |
| Idempotency via `posted_tweet_exists`         | 11       | Per-platform |
| Weekend/holiday skip                          | 11       | `is_trading_day` |
| Dry-run support                               | 11       | `ALGO_TRADE_POST_DRY_RUN=1` |
| Documentation                                 | 14       | CLAUDE.md + README |

Sizing decisions from the spec are encoded directly: 15 losers + 10 retired rules on `/mistakes/` (Task 1 helper limits), top 3 + top 2 on the homepage teaser (Task 8 JS slices), top 5 attribution by sample size on `/attribution/` chart and homepage (Tasks 5, 8).
