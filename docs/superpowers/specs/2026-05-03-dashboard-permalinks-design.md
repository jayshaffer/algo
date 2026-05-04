---
name: Dashboard permalinks & OG infrastructure
date: 2026-05-03
status: draft
parent: 2026-05-03-audience-growth-overview.md
---

# Dashboard permalinks & OG infrastructure

Foundation spec for the audience-growth strategy. Adds per-trade and per-thesis permalinks with OG / Twitter card meta tags, plus dynamic OG images. Every other social spec depends on the link targets and preview cards this delivers.

## Goal

Extend Stage 6 (`v2/dashboard_publish.py`) so the deployed Cloudflare Pages site contains:

1. **Per-trade pages** at `/trade/<id>/index.html` — full reasoning, signal sources, fill price, current P&L on the position the trade opened or sized.
2. **Per-thesis pages** at `/thesis/<id>/index.html` — full thesis text, entry/exit triggers, related decisions, status, current P&L on the underlying position.
3. **Dynamic OG images** at `/og/trade/<id>.png` and `/og/thesis/<id>.png` — auto-rendered PNGs used by Twitter / Bluesky / LinkedIn link previews.
4. **OG meta tags** on every emitted page (homepage included), so previews use the right title, description, and image.

The homepage SPA stays as-is. New pages are static HTML siblings.

## Non-goals

- No JS framework (Astro / Eleventy / Next). All templating in Python.
- No backend / Cloudflare Workers — output is static.
- No SEO work beyond OG tags.
- No content (mistakes log, attribution panel) — that's Spec #3.
- No methodology / about pages — that's Spec #4.
- No changes to existing JSON data files; new pages read from the same DB during publish.

## Architecture

### Directory structure of the deploy dir (after this change)

```
deploy/
├── index.html              # existing SPA
├── styles.css              # existing
├── app.js                  # existing
├── data/                   # existing JSON
├── trade/
│   └── <id>/
│       └── index.html      # NEW per-trade page
├── thesis/
│   └── <id>/
│       └── index.html      # NEW per-thesis page
└── og/
    ├── trade/<id>.png      # NEW dynamic OG image
    └── thesis/<id>.png     # NEW dynamic OG image
```

### New module: `v2/dashboard_pages.py`

Splits the page-rendering concerns out of `dashboard_publish.py` (which already approaches the size where it's doing too much). Pure-ish functions, one file, no classes.

**`render_trade_page(decision: dict, position: dict | None, signals: list[dict]) -> str`**
- Returns full HTML for a single trade page.
- Uses Python `string.Template` (stdlib, no Jinja dependency) with a small per-page template embedded in the module.
- Includes OG meta tags: `og:title`, `og:description`, `og:image`, `twitter:card=summary_large_image`.

**`render_thesis_page(thesis: dict, decisions: list[dict], position: dict | None) -> str`**
- Same shape as `render_trade_page`, for a thesis.

**`render_homepage_meta(summary: dict) -> str`**
- Returns the `<meta>` block to inject into `index.html` so the homepage gets fresh OG tags each publish (with current portfolio value, daily P&L, etc.).

### New module: `v2/dashboard_og.py`

OG image generation via Pillow. Self-contained, no external services, no headless browser.

**`render_trade_og(decision: dict) -> bytes`**
- Returns PNG bytes for a 1200×630 image.
- Background: solid color with the wave SVG motif from the existing dashboard, rendered once into a base PNG asset checked into `public_dashboard/og_base.png`.
- Overlay text: ticker (large), action (BUY/SELL), quantity + price, "Bikini Bottom Capital" footer. Rendered with a single embedded TTF (Inter / system-default fallback).

**`render_thesis_og(thesis: dict) -> bytes`**
- Same shape: ticker, direction (long/short), confidence, first 80 chars of thesis text.

Pillow is already a transitive dependency of several scientific Python packages but not declared. Add `pillow>=10.0` to `v2/requirements.txt`.

### Changes to `v2/dashboard_publish.py`

- Add `gather_trade_detail(cur, decision_id) -> dict` — joins `decisions`, `decision_signals`, `news_signals`/`macro_signals`, `positions`, `theses`.
- Add `gather_thesis_detail(cur, thesis_id) -> dict` — joins `theses`, `decisions` (matching ticker, direction), `positions`.
- In `gather_dashboard_data`, fetch the list of `(decision_id, thesis_id)` pairs that need pages emitted. **Emit pages for ALL decisions and ALL theses, not just the 30-day window or active theses.** Cloudflare Pages does full-bundle replacement on every deploy — any URL not in today's deploy 404s — so a tweet linking to a decision from 6 months ago would break otherwise. Link permanence is load-bearing for audience growth.
- Use a separate query path keyed by `id` (no date filter) for the page-emission gather; the homepage JSON files stay bounded by the existing 30-day window.
- Extend `assemble_deploy_dir` to:
  1. For each decision in the window, call `gather_trade_detail`, `render_trade_page`, write `deploy_dir/trade/<id>/index.html`. Same for theses.
  2. For each, call `render_trade_og` / `render_thesis_og`, write `deploy_dir/og/trade/<id>.png`. Same for theses.
  3. Inject `render_homepage_meta(...)` into `index.html` at copy time (replace a `<!-- OG_META -->` placeholder added to `public_dashboard/index.html`).

### Changes to `public_dashboard/index.html`

- Add a `<!-- OG_META -->` placeholder in `<head>` for the publish step to replace.
- Add the static portion of OG meta (`og:site_name`, `og:type=website`, `twitter:site` if we have a handle) inline.

### URL contract

- Per-trade: `https://<dashboard-host>/trade/<decision_id>/`
- Per-thesis: `https://<dashboard-host>/thesis/<thesis_id>/`

Both rely on Cloudflare Pages serving `index.html` for directory paths (default behavior). Decision IDs and thesis IDs are already integers from `serial` columns; they're not sensitive to expose, but we should never expose Alpaca order UUIDs on these pages — `_redact_order_id` already exists; reuse it.

## Data flow

```
session.py Stage 6
  └─ run_dashboard_stage
      ├─ gather_dashboard_data (existing, extended)
      │   ├─ summary / snapshots / positions / decisions / theses (existing)
      │   ├─ trade details for last 30 days (NEW)
      │   └─ thesis details for active theses (NEW)
      ├─ assemble_deploy_dir (existing, extended)
      │   ├─ copy static assets (existing)
      │   ├─ write JSON data files (existing)
      │   ├─ inject homepage OG meta into index.html (NEW)
      │   ├─ render + write per-trade HTML pages (NEW)
      │   ├─ render + write per-thesis HTML pages (NEW)
      │   └─ render + write OG PNGs (NEW)
      └─ deploy_to_cloudflare (existing, unchanged)
```

No schema changes. No new tables.

## Error handling

- A single trade or thesis failing to render must not break the publish stage. Wrap each per-page render in a try/except, log + count, continue. If >50% of pages fail, surface a stage error so it shows up in `session_stages` rather than silently shipping a half-broken site.
- OG image generation failures fall back to a single static fallback PNG. Always better to have a generic preview than no preview.
- Pillow / font loading errors at module import would block the entire stage; defer all Pillow imports inside the render functions.
- Existing graceful-degradation behavior of `gather_dashboard_data` is preserved.

## Testing

- New tests in `tests/test_dashboard_pages.py`:
  - `render_trade_page` returns HTML with the expected OG tags and content for a fixture decision.
  - `render_thesis_page` same.
  - `render_homepage_meta` injects expected fields.
- New tests in `tests/test_dashboard_og.py`:
  - `render_trade_og` returns valid PNG bytes (signature check); image dimensions are 1200×630; doesn't crash on missing optional fields.
- Existing `test_dashboard_publish.py`:
  - Extend `assemble_deploy_dir` test to verify per-trade and per-thesis files are emitted to the right paths.
  - Verify per-page render failures don't abort the run (use a fixture that raises in one render call).
- All Alpaca / Anthropic dependencies stay mocked. Pillow runs for real (pure Python where it matters).

## Open questions left for the implementation plan

- Exact OG image layout (color, font size, layout grid). Defer to plan; not load-bearing for the design.
- Whether to also pre-render `/positions/<ticker>/` pages. Probably yes in a follow-up; not in this spec.
- Whether to add `sitemap.xml` and `robots.txt`. Yes, but trivial; folded into the implementation plan.
- Cloudflare Pages has a 20,000-file-per-deployment limit. With 2 HTML pages + 2 PNGs per decision + per thesis, that supports ~4,000 decisions / theses combined. Several years of headroom at current pace, but worth a sanity check against Alpaca's full decision count before implementation. If we hit it, fold OG images into a single sprite or move them to R2.
