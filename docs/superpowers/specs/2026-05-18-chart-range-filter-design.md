# Chart range filter (1W / 1M / 1Y / All)

**Date:** 2026-05-18
**Status:** Design
**Surface:** `public_dashboard/` (Home and Performance pages)

## Goal

Let visitors narrow the three performance charts (Equity, Cumulative P&L, vs S&P 500) to a recent window. Adds a shared `1W | 1M | 1Y | All` button group above the chart grid; default selection is `1W`.

## Scope

In:
- Home page (`/`) and Performance page (`/performance/`) — both already render the same three-chart grid via `initPerformancePage` in `public_dashboard/app.js`.
- Client-side filter of the already-fetched JSON; no new data files or server endpoints.

Out:
- Per-chart range overrides — one shared control governs all three.
- URL state (`?range=7`). Refresh resets to default.
- Range buttons on per-ticker pages — those are server-rendered with no client charts today.
- Drag-to-zoom on the chart canvas itself. Treated as a follow-up; see "Future" below.

## User-facing behavior

- A button group renders above the three-chart grid on Home and Performance: `1W`, `1M`, `1Y`, `All`.
- `1W` is selected on page load. The corresponding chart window paints with that filter immediately — no flash of "all data" first.
- Clicking a button re-renders all three charts to the new window. The active button gets a pressed visual state.
- The window anchor is the most recent snapshot date in the dataset (`latestDate`). The filter keeps rows where `date >= latestDate - N days`. `All` returns the full dataset.
- If the history is shorter than the requested window, the charts show whatever data exists — no disabled buttons, no "not enough data" state. Rationale: the dataset is small today (~11 snapshots) and will grow; disabling buttons would either lie about the data or require maintaining a hardcoded "min days" threshold per button.

## Implementation

### Server (Python — `v2/dashboard_pages.py`)

Two render functions emit the chart grid: `_render_homepage_charts()` (line ~857) and the corresponding block in `render_performance_page()` (line ~970). Both need the same toolbar inserted just above `chart-grid`.

To avoid drift, factor the toolbar into a helper:

```python
def _render_range_control() -> str:
    return (
        '<div class="range-control" role="group" aria-label="Time range">'
        '<button type="button" data-range="7"   class="range-btn">1W</button>'
        '<button type="button" data-range="30"  class="range-btn">1M</button>'
        '<button type="button" data-range="365" class="range-btn">1Y</button>'
        '<button type="button" data-range="all" class="range-btn">All</button>'
        '</div>'
    )
```

The default-active button is marked server-side so the first paint matches the default-filtered chart:

- `1W` button gets `class="range-btn is-active"` and `aria-pressed="true"`.
- Others get `aria-pressed="false"`.

### Client (`public_dashboard/app.js`)

State, scoped to the Performance/Home page init:

```js
var perfData = { snapshots: null, benchmark: null, decisions: null };
var chartInstances = {};  // canvasId -> Chart
var DEFAULT_RANGE_DAYS = 7;
```

Helpers:

- `latestSnapshotDate(snapshots)` — last element's `date` (snapshots are date-ordered ascending in the JSON).
- `filterByRange(rows, dateKey, days, anchor)` — if `days == null`, return `rows` unchanged; else return `rows.filter(r => r[dateKey] >= cutoff)` where `cutoff = anchor - days` (calendar-day arithmetic on `YYYY-MM-DD` strings — parse to `Date`, subtract, format back).
- `destroyCharts()` — iterates `chartInstances`, calls `.destroy()` on each, clears the map. Chart.js requires destruction before re-instantiation on the same canvas.

Renderers (`renderEquityCurve`, `renderPnlChart`, `renderBenchmark`) record their created chart into `chartInstances` keyed by canvas id, so `applyRange` can clean up before the next render.

`applyRange(days)`:

1. `destroyCharts()`.
2. Compute `anchor = latestSnapshotDate(perfData.snapshots)`.
3. Build filtered slices: snapshots, benchmark, decisions all filtered by their `date` key against the same anchor and days.
4. Call the three render functions with the filtered slices.
5. Update `.range-btn` active state and `aria-pressed`.

Wiring:

- `initPerformancePage` stores the fetched data into `perfData`, attaches one delegated `click` listener on `.range-control`, then calls `applyRange(DEFAULT_RANGE_DAYS)`. The first render uses the filtered data — no double render.

### CSS (`public_dashboard/styles.css`)

A small button-group block matching the existing terminal palette:

- `.range-control` — flex row, gap, right-aligned within the chart section header (or above the grid on its own row on narrow screens).
- `.range-btn` — terminal-foreground text on transparent background, subtle border, hover state, monospace label.
- `.range-btn.is-active` — inverted (background-foreground swap), or accent color matching the existing `--accent` if defined.

Match the visual vocabulary of `.badge` and the nav links rather than introducing a new style.

## Tests

`tests/v2/test_dashboard_pages.py`:

- Assert the four range buttons (`data-range="7|30|365|all"`) render on both home and performance pages.
- Assert exactly one button — the `1W` one — has `is-active` and `aria-pressed="true"`.

No JS unit tests exist today for `app.js`; the filter logic ships untested unless we add a harness. Accepted risk — the filter is small and the visual feedback (chart re-render) is immediate during manual QA.

Manual verification:

1. Load `/` and `/performance/` — `1W` button is active, charts show only the last ~5 trading days.
2. Click each button — all three charts re-render in sync.
3. Click `All` — charts show full history; benchmark line rebases correctly (SPY anchor = first in-range snapshot's SPY close, which the existing renderer already handles).
4. With ~11 snapshots in the dataset, `1Y` and `All` produce identical charts. Expected.

## Future / out of scope

- **Drag-to-zoom on canvas:** standard implementation is `chartjs-plugin-zoom` with `mode: 'x'` plus a "Reset zoom" affordance. Distinct interaction from preset buttons — drag is a single-chart visual zoom, presets re-filter the data window across all three charts. Track separately.
- **URL state:** `?range=30` survives refresh and is shareable. Add when there's a use case (e.g., a memo links to "the last month of performance").
- **Per-chart override:** if one chart deserves a different window than the others, revisit. Today the user requested shared control.
