# Drag-to-select date range (brush strip)

**Date:** 2026-05-19
**Status:** Design
**Surface:** `public_dashboard/` (Home and Performance pages)
**Follow-up to:** `2026-05-18-chart-range-filter-design.md` (the preset 1W/1M/1Y/All buttons)

## Goal

Let desktop visitors pick an arbitrary date window for the three performance charts (Equity, Cumulative P&L, vs S&P 500) by dragging a brush below the chart grid. The existing 1W/1M/1Y/All preset buttons stay; the brush is the source of truth and the presets become shortcuts that reposition the brush.

The presets answer "last N days from today." The brush answers any question that doesn't fit that mold — e.g. "what happened the week of March 1?" or "compare two specific weeks."

## Scope

In:
- Home (`/`) and Performance (`/performance/`) — both render the same three-chart grid via `initPerformancePage` in `public_dashboard/app.js`.
- A brush strip below the chart grid: thin overview of the full snapshot timeline with two draggable handles and a draggable selection bar.
- Desktop only — brush hidden below 768px viewport width. Preset buttons remain sole control on mobile/tablet.
- Live re-render of all three charts as the brush moves.
- Bidirectional sync between brush and presets: clicking a preset repositions the brush; dragging the brush clears the preset's active state.

Out:
- URL state (`?from=…&to=…`). Refresh resets to the default 1W window.
- Touch/pointer support for the brush. Mobile/tablet falls back to preset buttons (which work fine with touch).
- Keyboard navigation on the brush handles (arrow keys to nudge endpoints). Presets remain keyboard-reachable for the same intent; full a11y on the brush is a follow-up.
- Drag-to-zoom on the chart canvases themselves. The brush replaces the need; if a future use case wants per-chart zoom, file separately.
- Per-ticker pages and other server-rendered pages — no charts there today.

## User-facing behavior

- Below the three-chart grid, a ~40px-tall strip renders an SVG overview of the full snapshot history (a single polyline of `portfolio_value` normalized to the strip height). Two handles flank a translucent selection bar.
- On page load the brush positions itself over the last 7 days (matches the existing default); the active preset button remains `1W`.
- **Dragging a handle** moves one endpoint; the other stays put. Charts re-render live, throttled to one update per `requestAnimationFrame`.
- **Dragging the bar between handles** slides the whole window without changing its width.
- **Clicking inside the strip outside the bar** snaps the nearest endpoint to the cursor.
- While the user is dragging, no preset button is active (all `aria-pressed="false"`, no `is-active` class).
- **Clicking a preset** computes its window from the latest snapshot date, writes it to the brush (handles animate to the new positions), and marks the preset active.
- If the brush selection happens to coincide exactly with a preset window after a drag (snap on release? no — see below), the matching preset does **not** re-light. Active state is "preset was the last input," not "current window equals a preset."
- No snapping. The brush moves continuously in pixel space; the resulting date window is whatever `pixel → date` mapping yields. Filtering uses `date >= startDate && date <= endDate`, which already tolerates non-snapshot dates gracefully.

## Implementation

### Server (Python — `v2/dashboard_pages.py`)

Add one helper alongside `_render_range_control()`:

```python
def _render_range_brush() -> str:
    """Drag-to-select date range strip rendered below the chart grid.
    Hydrated by app.js after snapshots.json loads — server emits only
    the container; the SVG contents are drawn client-side."""
    return (
        '<div class="range-brush" data-role="range-brush" aria-hidden="true">'
        '<svg class="range-brush-svg" preserveAspectRatio="none"></svg>'
        '</div>'
    )
```

Insert the brush container immediately after `chart-grid` in both `_render_homepage_charts()` (line ~1098) and `render_performance_page()` (line ~1242). `aria-hidden="true"` because the brush is purely visual; screen-reader users get the preset buttons.

### Client (`public_dashboard/app.js`)

State extends the existing `perfData` block:

```js
var DEFAULT_RANGE_DAYS = 7;
var perfData = {
  snapshots: null,
  benchmark: null,
  decisions: null,
  rangeStart: null,  // YYYY-MM-DD
  rangeEnd: null,    // YYYY-MM-DD
};
```

`applyRange` is reshaped from `applyRange(days)` to `applyRange({ start, end, preset })`:

```js
function applyRange(spec) {
  // spec = { start: "YYYY-MM-DD", end: "YYYY-MM-DD", preset: "7"|"30"|...|"all"|null }
  perfData.rangeStart = spec.start;
  perfData.rangeEnd = spec.end;
  destroyCharts();
  var filteredSnapshots = filterByDateWindow(perfData.snapshots, "date", spec.start, spec.end);
  var filteredBenchmark = filterByDateWindow(perfData.benchmark, "date", spec.start, spec.end);
  var filteredDecisions = filterByDateWindow(perfData.decisions, "date", spec.start, spec.end);
  renderEquityCurve(filteredSnapshots, filteredDecisions);
  renderPnlChart(filteredSnapshots, filteredDecisions);
  renderBenchmark(filteredSnapshots, filteredBenchmark, filteredDecisions);
  updatePresetActiveState(spec.preset);
  updateBrushPosition(spec.start, spec.end);
}
```

New helper:

```js
function filterByDateWindow(rows, dateKey, start, end) {
  if (!rows || !start || !end) return rows;
  return rows.filter(function (r) {
    var d = normalizeDate(r[dateKey]);
    return d >= start && d <= end;
  });
}
```

Existing `cutoffDate` and `latestSnapshotDate` are kept and re-used by `presetToWindow` to convert a preset id into a `{ start, end }` pair. The old `filterByRange` (anchor + days) is replaced by `filterByDateWindow` (start + end) — the only caller was `applyRange`, so this is a clean swap:

```js
function presetToWindow(preset) {
  // preset: "7" | "30" | "365" | "all"
  var anchor = latestSnapshotDate(perfData.snapshots);
  if (preset === "all") {
    var first = normalizeDate(perfData.snapshots[0].date);
    return { start: first, end: anchor, preset: "all" };
  }
  var days = parseInt(preset, 10);
  return { start: cutoffDate(anchor, days), end: anchor, preset: preset };
}
```

#### Brush module

A new self-contained module in the same file (kept inline to avoid introducing a build step):

```js
var brush = {
  el: null,             // .range-brush element
  svg: null,            // child <svg>
  width: 0,             // measured on layout
  domain: null,         // { first, last } as YYYY-MM-DD
  dragging: null,       // null | "left" | "right" | "bar"
  dragOffset: 0,        // px offset within bar at drag start
  pendingFrame: null,   // requestAnimationFrame handle
};
```

Lifecycle:

- `initBrush()` — called from `initPerformancePage` once `perfData.snapshots` is loaded.
  - Sets `brush.domain = { first: snapshots[0].date, last: snapshots.at(-1).date }`.
  - Measures the strip width via `getBoundingClientRect()`.
  - Renders the static elements: the overview polyline, two `<rect>` handles, and one `<rect>` selection bar inside `brush.svg`.
  - Attaches `pointerdown` on the SVG, `pointermove`/`pointerup` on `window` (so the gesture survives leaving the strip).
  - Hooks `window.resize` to re-measure and re-position. Throttled.

- `pointerdown` handler — hit-tests the click point against handle bounds and the bar. Stores `brush.dragging` and the in-bar `dragOffset` if dragging the bar.

- `pointermove` handler — converts cursor X to a date via the domain mapping, updates `brush.startDate`/`brush.endDate`, and schedules a re-render via `requestAnimationFrame`. Within the rAF callback, calls `applyRange({ start, end, preset: null })`.

- `pointerup` handler — clears `brush.dragging`. Does *not* snap to snapshot dates; the filter tolerates any date.

Pixel↔date mapping: linear over calendar days between `domain.first` and `domain.last`. The polyline is plotted at snapshot indices, but the brush handles can land between them — the filter handles that.

#### Wiring

`initPerformancePage` becomes:

```js
function initPerformancePage() {
  Promise.all([fetchJSON("snapshots.json"), fetchJSON("benchmark.json"), fetchJSON("decisions.json")])
    .then(function (parts) {
      perfData.snapshots = parts[0];
      perfData.benchmark = parts[1];
      perfData.decisions = parts[2];
      setupRangeControl();
      initBrush();
      applyRange(presetToWindow(String(DEFAULT_RANGE_DAYS)));
    })
    .catch(function (err) { console.error("Failed to load performance data:", err); });
}
```

`setupRangeControl` changes one line: the click handler calls `applyRange(presetToWindow(raw))` instead of `applyRange(days)`.

### CSS (`public_dashboard/styles.css`)

```css
.range-brush {
  display: none;             /* mobile fallback — see media query */
  margin: 8px 0 0;
  height: 40px;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 4px;
  position: relative;
  cursor: default;
  user-select: none;
}

@media (min-width: 768px) {
  .range-brush { display: block; }
}

.range-brush-svg { width: 100%; height: 100%; display: block; }
.range-brush-overview { stroke: #30363d; stroke-width: 1; fill: none; }
.range-brush-bar { fill: rgba(88, 166, 255, 0.18); cursor: grab; }
.range-brush-bar:active { cursor: grabbing; }
.range-brush-handle { fill: #58a6ff; cursor: ew-resize; }
.range-brush-handle:hover { fill: #79b8ff; }
```

Color palette matches the existing chart accents (`#58a6ff` blue, `#30363d` border). Hit targets: handles are 6px wide with an invisible 14px pointer-event surface either side so they're easy to grab.

## Tests

`tests/v2/test_dashboard_pages.py`:

- Assert `range-brush` container renders on Home and Performance.
- Assert it does **not** render on other pages (activity, strategy, learning, how-it-works, mistakes, attribution).
- Assert it carries `data-role="range-brush"` so JS lookups are stable.

No new JS unit tests — same posture as the prior range-filter spec. The brush has internal state and date math worth testing in principle, but no JS test harness exists in this repo today and standing one up is out of scope for this change. Manual QA covers the interaction.

Manual verification checklist:

1. Load `/` and `/performance/` on a desktop viewport — brush strip is visible below the chart grid, positioned to cover the last 7 days, `1W` preset highlighted.
2. Drag the right handle leftward — charts re-render live; `1W` highlight clears.
3. Drag the left handle rightward — endpoint moves; `1W` highlight clears.
4. Drag the bar between handles — window slides; widths preserved.
5. Click outside the bar — nearest endpoint jumps to cursor.
6. Click `1M` — brush handles animate to the 30-day window; `1M` lights up.
7. Click `All` — brush spans the full domain.
8. Resize the window — brush re-measures and stays aligned with the chart grid.
9. Narrow the viewport below 768px — brush hides; preset buttons remain functional.

## Future / out of scope

- **URL state** — `?from=…&to=…` for shareable links. Add when there's a use case (memo deep-linking to a specific window).
- **Touch support** — pointer events technically fire on touch, but the brush is hidden below 768px today so it never runs. If the project later wants mobile drag, the strip needs larger hit targets and a tap-and-hold-to-drag pattern.
- **Keyboard accessibility** — focus ring on handles, arrow keys to nudge by 1 day (Shift+arrow by 7). The preset buttons cover the common case today; revisit once the brush is in use.
- **Snap to snapshot dates** — currently rejected because the filter is `>= start && <= end`, which is robust to mid-day positions. If a future visualization needs exact-snapshot alignment, add it.
- **Visual indicators of selected window** in the brush — e.g. labels "2026-04-22 → 2026-05-19" floating above the handles. Easy to add if the bare numbers turn out to matter; left out of v1 to keep the strip quiet.
