---
name: Public dashboard redesign — multi-page IA + new palette
date: 2026-05-04
status: draft
---

# Public dashboard redesign

The public dashboard at `bikini-bottom-capital.pages.dev` is a single long-scroll page of stacked panels. Visitors arriving from a tweet have to scroll through every section to find anything, and the existing sub-pages (`/mistakes/`, `/attribution/`, per-trade and per-thesis permalinks, plus the in-flight `/about/`, `/internals/`, `/trace/` from specs 4a/4b/4c) are only reachable via inline "See all" links — they are second-class citizens of the site.

This spec turns the dashboard into a 5-page static site with a curated landing, dedicated drilldown pages, and a new palette. Same publish pipeline (`v2/dashboard_publish.py` → `wrangler pages deploy`); same JSON data files; no schema changes.

## Audience

Mix of three: drive-by curious clicking in from a social post, returning audience following the trades, and AI / agentic-systems builders looking at methodology. Operator self-use is not a goal of the public dashboard — that's `dashboard/` (port 3000).

## Information architecture

Five pages, all sharing the same nav, palette, and footer:

| Page | URL | Role |
|---|---|---|
| Home | `/` | Curated landing — hero, today's move, recent learnings, session memo, methodology strip |
| Performance | `/performance/` | Equity curve + vs S&P + benchmark stats. Drilldown for hero stats. |
| Activity | `/activity/` | Holdings, active theses, full decisions log, recent memos. Drilldown for hero chips and "today's move." |
| Learning | `/learning/` | Hub indexing two existing children: `/mistakes/` and `/attribution/`. |
| How it works | `/how-it-works/` | Hub indexing three planned children: `/about/`, `/internals/`, `/trace/`. |

Existing permalinks (`/trade/<id>/`, `/thesis/<id>/`, `/mistakes/`, `/attribution/`, plus 4a/4b/4c outputs once they ship) keep their URLs and content. They get the new nav, footer, and palette via the shared page-shell helper but are otherwise unchanged.

### Hub-page rationale

`/learning/` and `/how-it-works/` exist only so the nav has something to point at. Each is a 2-or-3-card index; the real content lives on the children. This is cheaper than nav dropdowns (which are awkward in a static site) and matches the existing per-page rendering pattern.

### Decoupling from in-flight specs

Specs 4a/4b/4c (about / internals / trace) are mid-flight. The redesign must not block on them. `/how-it-works/` renders all three cards regardless; if a child page hasn't shipped, that card renders in a "Coming soon" state. Detection at publish time: `os.path.exists` against the deploy-dir paths during `assemble_deploy_dir`.

## Homepage

Top to bottom on `/`:

### 1. Sticky top nav

Logo (left) + 5 nav items (right). Active page underlined with the blue accent. On viewports below 640px collapses to a hamburger that toggles a vertical list. No dropdowns, no submenus — hub pages exist precisely to avoid them.

### 2. Hero

- Tag line: `Day {N} · Updated {time} ET`. `{N}` = `(today − inception_date).days`; inception is the first row in `account_snapshots`.
- Mono headline: `${portfolio_value}` followed by an inline strip `+{daily_pct} today · {sign}{total_pct} all time · {sign}{vs_spy_pct} vs S&P`. Gain / loss color from value sign. Numbers from `summary.json` (already published).
- Label `Currently betting on`.
- Up to **3 thesis chips** — ticker (mono, blue) + 3-5 word thesis blurb. Sourced from `theses` where `status='active'`. Order: descending conviction if `conviction` is populated, else most recent. Each chip is an `<a>` to `/thesis/<id>/`.
- **Sparkline** — 90-day equity, server-rendered inline SVG, single `<polyline>`, no axes. Click → `/performance/`. Generated in `dashboard_publish.py` at publish time; not Chart.js.

**Empty cases:**
- No active theses → hide the "Currently betting on" label and chip row.
- Fewer than 7 snapshots → hide the sparkline.
- No snapshots at all → headline shows `—` instead of `${value}`.

### 3. Today's move

Single card showing the most recent **significant non-hold decision** (notional ≥ $100, matching the live-trade pipeline threshold).

- Action badge (BUY / SELL — colored gain / loss) + ticker (mono) + notional + `{x}% of portfolio`.
- Reasoning excerpt — first 150 characters from `decisions.reasoning`, suffixed with `…` if truncated.
- Card itself is an `<a>` to `/trade/<id>/`. Right-aligned `All decisions →` link to `/activity/#decisions`.

**Empty case:** if no significant decision in the last 5 trading days, render the section as a single line `No new positions in the last 5 sessions — see the full log →` linking to `/activity/`. Don't hide entirely; the section header anchors the page.

### 4. Recent learnings (framed)

Section header `Recent learnings` + right-aligned `Learning →` link to `/learning/`. Two cards beneath:

- **What's working** — top signal type by sample size from `signal_attribution`. Format: `{signal_type} · {N} trades · +{avg_return}% avg`. Card → `/attribution/`.
- **What didn't** — worst recent closed loser from the mistakes feed. Format: ticker (mono) + `{pct}%` over the close window. Card → `/mistakes/`.

**Empty cases:**
- Attribution has fewer than `N=5` samples → "What's working" card shows `Not enough samples yet`.
- No closed losers in window → "What didn't" card shows `No closed losers in window.`
- Both empty → hide the entire section (header + cards).

### 5. From today's session memo

Italicized blockquote — first 280 characters of the most recent `strategy_memos` entry. Border-left in the blue accent. Right-aligned `All memos →` link to `/activity/#memos` (no dedicated `/memos/` page; memos live as a section on Activity).

**Empty case:** if no memo from the last session, hide the entire section.

### 6. Methodology footer strip

One-line sentence + 3 inline links:

> Built by an AI agent (Claude Haiku for execution, Sonnet for strategy). [How it works] · [Sample tool-call trace] · [Model & cost]

Each link points to the relevant `/how-it-works/` child if it exists, else to `/how-it-works/` itself.

## Sub-pages

### `/performance/`

- Compact stat strip at top — same numbers as the hero, no chips, no sparkline.
- **Equity curve** — full Chart.js chart, moved verbatim from the current homepage.
- **Performance vs S&P** — full Chart.js chart, moved verbatim from the current homepage.
- Stats panel: max drawdown, win rate %, average days held, best day, worst day. Computed in `dashboard_publish.py` from `account_snapshots` + `decisions`; written to `performance.json`.
- Empty: each chart shows the existing `empty-state` placeholder.

### `/activity/`

Anchored sections so internal links land at the right place:

- `#holdings` — Current holdings table. Move verbatim from the current homepage.
- `#theses` — Active theses cards. Move verbatim from the current homepage.
- `#decisions` — Full decisions log, all rows in one table with sticky header. Each row links to `/trade/<id>/`. Initial cap: no limit (~hundreds of rows after a year is still well under a 1MB page). Revisit if the rendered HTML exceeds 500KB.
- `#memos` — Last 10 strategy memos. Each as a blockquote with date + `Session N` tag. Sourced from a new `memos.json` written in `gather_dashboard_data`.

### `/learning/` (hub)

Section title `What this thing has learned`. Two cards side-by-side:

- **What's working** — top 3 signal types from attribution + `See all →` linking to `/attribution/`.
- **What didn't** — top 3 recent losers + retired-rule count + `See all →` linking to `/mistakes/`.

~25 lines of HTML; cheapest page in the set.

### `/how-it-works/` (hub)

Section title `How this thing works`. Three cards:

- **Methodology** → `/about/` (or "Coming soon")
- **Model & cost** → `/internals/` (or "Coming soon")
- **Tool-call trace** → `/trace/` (or "Coming soon")

Each card has a 1-2 sentence description so visitors who don't click still understand what's there. "Coming soon" cards have reduced opacity, no link, and the description text only.

### Existing pages — re-skin only

`/mistakes/`, `/attribution/`, `/trade/<id>/`, `/thesis/<id>/`: keep current content, data, URLs. They inherit the new nav, footer, and palette via the shared `_render_page_shell` helper. No new logic.

## Visual system

### Palette (P1 Terminal)

```
--bg-deep      #0d1117   page background
--bg-card      #161b22   card / panel background
--bg-card-alt  #1c2129   hover / nested card
--border       #30363d   borders, dividers
--text         #c9d1d9   primary text
--text-dim     #8b949e   secondary text
--accent       #58a6ff   logo, links, active nav, thesis chip ticker, memo border-left
--gain         #3fb950   positive numbers / BUY badge
--loss         #f85149   negative numbers / SELL badge
--font-body    -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif
--font-mono    "SF Mono", "Cascadia Code", "Fira Code", monospace
```

The wave-divider SVG, caustics animation, and inline pineapple emoji all come out — they were anchoring the old vibe. **Pineapple favicon stays** (one charming touchpoint in the browser tab); the inline emoji in the H1 is removed. Logo glyph in the nav becomes `⌬ Bikini Bottom Capital`. The "Is mayonnaise a financial instrument?" footer line stays.

### Shared CSS components

| Class | Used by |
|---|---|
| `.site-nav` | All pages — sticky top nav, mobile hamburger |
| `.hero` | Homepage only |
| `.stat-row` / `.stat` | Hero, Performance top strip |
| `.section` | Generic content section with optional right-aligned `more →` link |
| `.card-grid` / `.card` | Hub pages, homepage 2-up |
| `.move-card` | Today's move card |
| `.memo-block` | Italic blockquote, blue border-left |
| `.chip` | Thesis chip |
| `.empty-state` | All pages — restyled for new palette |

## Implementation

### Templates → shared scaffolding

`v2/dashboard_pages.py` already has a `_render_meta_block` helper. Add:

```python
def _render_page_shell(*, title: str, active_nav: str, content: str,
                       og_meta: str, base_url: str) -> str:
    """Wrap page content in the shared <html> + nav + footer."""
```

`active_nav` is one of `home | performance | activity | learning | how-it-works`. Drives which nav item gets the underline. Every renderer (existing `render_trade_page`, `render_thesis_page`, `render_mistakes_page`, `render_attribution_page`, plus new `render_homepage`, `render_performance_page`, `render_activity_page`, `render_learning_hub`, `render_how_it_works_hub`) calls it.

### Homepage moves to a Python renderer

`public_dashboard/index.html` is currently hand-authored and checked into git, with `<!-- OG_META -->` substitution applied at publish time. Promote it to `render_homepage()` in `dashboard_pages.py` to remove the special case. The checked-in `public_dashboard/index.html` is **deleted**; the homepage is now produced entirely by the renderer and written to `deploy_dir/index.html` at publish time. Static assets (`styles.css`, `app.js`) stay checked in under `public_dashboard/` and are copied into the deploy dir verbatim as today.

`app.js` keeps the JSON-fetch pattern for hero numbers, holdings, decisions, theses on the relevant pages. The sparkline becomes a server-rendered SVG embedded directly in the homepage HTML — no JS needed for it.

### Sparkline — server-side SVG

In `dashboard_publish.py`, new helper `render_sparkline_svg(snapshots: list[dict]) -> str` that:
- Takes the last 90 days of equity values from `snapshots.json`.
- Normalizes Y values to `[0, 50]` over a fixed `400×60` viewBox.
- Returns a single `<svg>` with one `<polyline>` and no axes.

Embedded directly into the homepage HTML at render time.

### New JSON files

- `memos.json` — last 10 entries from `strategy_memos`, fields: `id`, `date`, `session_number`, `body`. Written by `gather_dashboard_data`.
- `performance.json` — derived stats (max drawdown, win rate, avg days held, best/worst day) computed from existing tables. Written by `gather_dashboard_data`.

Everything else uses existing JSON.

### Hub-page "Coming soon" detection

`assemble_deploy_dir` checks for the existence of `/about/index.html`, `/internals/index.html`, `/trace/index.html` after all rendering completes. Booleans flow into `render_how_it_works_hub` so disabled cards render correctly. No import-time coupling to specs 4a/4b/4c.

### Publish flow

Unchanged. The deploy dir gains a few new HTML files (`performance/index.html`, `activity/index.html`, `learning/index.html`, `how-it-works/index.html`) and JSON files (`memos.json`, `performance.json`). `wrangler pages deploy` is unchanged.

### Cloudflare Pages file-count ceiling

The 20,000-file ceiling is shared with `/trade/<id>/` and `/thesis/<id>/` permalinks. This redesign adds 4 fixed pages (Performance, Activity, Learning hub, How-it-works hub). Negligible against the existing per-trade growth rate. No new headroom check needed.

## Testing

- **Unit tests in `tests/test_dashboard_pages.py`** — one test per renderer:
  - `_render_page_shell` — golden HTML snapshot of the wrapper.
  - `render_homepage` — hero numbers, chip count, today's move card, recent learnings empty-case branches, memo present / absent.
  - `render_performance_page` — stat strip, both chart canvases present, stats panel.
  - `render_activity_page` — anchors `#holdings`, `#theses`, `#decisions`, `#memos` all present.
  - `render_learning_hub` — both cards link to correct children.
  - `render_how_it_works_hub` — three states for "Coming soon" detection (none ready, some ready, all ready).
- **Sparkline test in `tests/test_dashboard_publish.py`** — fixture of 90 snapshots, asserts polyline points are within viewBox bounds and count matches input length.
- **Empty-case coverage** — every renderer has at least one empty-data test.
- No new integration tests; the publish flow is untouched.

## Out of scope

- Real-time / live updates. Dashboard remains static, regenerated per session.
- Mobile-app-style features (PWA, offline, install prompts).
- Search or filtering on the decisions log. If the table grows unwieldy (>500KB rendered), revisit with a separate spec.
- Memo dedicated page (`/memos/`). Memos live as a section on `/activity/`.
- Theme toggle / light mode. Single P1 palette only.
- Refactoring the operator-facing `dashboard/` (Flask, port 3000). This spec covers `public_dashboard/` only.

## Open items

- **Decisions-log size strategy.** Initial implementation renders all rows in one table. Revisit if rendered HTML exceeds 500KB or scanning becomes painful in practice.
- **Conviction ordering for thesis chips.** The `theses` table has no formal `conviction` field today. Initial implementation falls back to `created_at DESC`. If a conviction signal becomes available later, swap the ordering.
