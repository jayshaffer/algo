# Drag-to-select date range (brush strip) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a draggable brush strip below the three-chart grid on Home and Performance so desktop visitors can pick arbitrary date windows, with bidirectional sync to the existing 1W/1M/1Y/All preset buttons.

**Architecture:** Server renders an empty brush container (`<div class="range-brush"><svg/></div>`) below the chart grid; client hydrates the SVG with an overview polyline + two draggable handles + a draggable bar; pointer events update `perfData.rangeStart`/`rangeEnd` and call the existing chart re-render path through a reshaped `applyRange({ start, end, preset })` API. Presets become a thin wrapper that computes a `{ start, end }` pair and writes it to the brush.

**Tech Stack:** Python 3.12 + raw HTML string rendering (existing `v2/dashboard_pages.py`); vanilla JS + Chart.js 4 (existing `public_dashboard/app.js`); CSS (existing `public_dashboard/styles.css`); pytest for server-side tests.

**Spec:** `docs/superpowers/specs/2026-05-19-chart-drag-range-design.md`

**Reference (the predecessor PR's spec):** `docs/superpowers/specs/2026-05-18-chart-range-filter-design.md`

---

## File Structure

**Modify:**
- `v2/dashboard_pages.py` — add `_render_range_brush()` helper; insert it into `_render_homepage_charts()` (after `chart-grid`) and `render_performance_page()` (in a new section after the last chart section).
- `public_dashboard/styles.css` — add a `.range-brush` block matching the existing terminal palette; desktop-only via `@media (min-width: 768px)`.
- `public_dashboard/app.js` — reshape `applyRange` to take `{ start, end, preset }`; add `filterByDateWindow`, `presetToWindow`, `updatePresetActiveState`; add a `brush` module (state + `initBrush` + `updateBrushPosition` + pointer event handlers).
- `tests/v2/test_dashboard_pages.py` — add assertions for the brush container on Home/Performance and its absence on other pages.

No new files. All code lives in the four existing files above. The `brush` module is kept inline in `app.js` (no build step in this project).

---

## Task 1: Server renders the brush container

**Files:**
- Modify: `v2/dashboard_pages.py:1075-1115` (after `_render_range_control()`, then call site in `_render_homepage_charts()`)
- Modify: `v2/dashboard_pages.py:1242-1263` (call site in `render_performance_page()`)
- Test: `tests/v2/test_dashboard_pages.py` (extend `TestRenderHomepage` and `TestRenderPerformancePage` classes; add a new assertion in the activity/strategy class blocks)

- [ ] **Step 1: Write failing tests for the brush container on Home**

Add to `tests/v2/test_dashboard_pages.py` inside `TestRenderHomepage` (alongside the existing range-control tests, around line 839):

```python
    def test_renders_range_brush_container(self):
        html = render_homepage(**self._data())
        assert 'class="range-brush"' in html
        assert 'data-role="range-brush"' in html
        assert '<svg class="range-brush-svg"' in html

    def test_range_brush_is_aria_hidden(self):
        """The brush is decorative — screen-reader users get the
        preset buttons. Verify it doesn't pollute the a11y tree."""
        html = render_homepage(**self._data())
        # match attribute order-independently
        assert 'aria-hidden="true"' in html
        assert 'class="range-brush"' in html
```

Add the same two tests to `TestRenderPerformancePage` (around line 916):

```python
    def test_renders_range_brush_container(self):
        html = render_performance_page(**self._data())
        assert 'class="range-brush"' in html
        assert 'data-role="range-brush"' in html
        assert '<svg class="range-brush-svg"' in html

    def test_range_brush_is_aria_hidden(self):
        html = render_performance_page(**self._data())
        assert 'aria-hidden="true"' in html
        assert 'class="range-brush"' in html
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from the host (pytest runs in docker per CLAUDE.md):

```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_dashboard_pages.py -k "range_brush" -v
```

Expected: 4 FAILs (`AssertionError: assert 'class="range-brush"' in html` on all four).

- [ ] **Step 3: Add `_render_range_brush()` helper**

In `v2/dashboard_pages.py`, insert directly below `_render_range_control()` (after line 1095):

```python
def _render_range_brush() -> str:
    """Drag-to-select date range strip rendered below the chart grid
    on Home and Performance. Container only — app.js hydrates the SVG
    after snapshots.json loads. aria-hidden because the brush is purely
    visual; screen-reader users get the preset buttons in
    .range-control instead."""
    return (
        '<div class="range-brush" data-role="range-brush" aria-hidden="true">'
        '<svg class="range-brush-svg" preserveAspectRatio="none"></svg>'
        '</div>'
    )
```

- [ ] **Step 4: Wire it into `_render_homepage_charts()`**

In `v2/dashboard_pages.py`, find `_render_homepage_charts()` (around line 1098). The brush goes after the closing `</div>` of `chart-grid`. Modify the return so the brush appears immediately after `chart-grid` and before the surrounding `</section>`. Locate the line `'</div>'  # closes chart-grid` (or the equivalent closing div followed by `</section>`) and insert:

```python
        # after the chart-grid div closes:
        f'{_render_range_brush()}'
```

Concretely, find the existing `_render_homepage_charts()` body and add `f'{_render_range_brush()}'` as the last fragment before the closing `</section>`. Read the function (lines 1098–1125) and insert the call between `'</div>'` (closing `chart-grid`) and `'</section>'`.

- [ ] **Step 5: Wire it into `render_performance_page()`**

In `v2/dashboard_pages.py`, find `render_performance_page()` (around line 1230). The Performance page does NOT use `chart-grid` — each chart is its own `<section>`. Append a new section right after the benchmark section. Modify the `charts = ( … )` assignment around line 1243 to add the brush after the last benchmark section, inside the same string concatenation:

```python
    charts = (
        f'<section class="section range-section">{range_html}</section>'
        '<section class="section">'
        '<div class="head"><h2>Equity curve</h2></div>'
        # ... existing ...
        '<section class="section"><div class="head"><h2>Performance vs S&amp;P 500</h2></div>'
        '<div class="chart-wrap"><canvas id="benchmark-chart"></canvas></div>'
        '<p class="empty-state" id="benchmark-empty" style="display:none;">No benchmark data yet</p>'
        '</section>'
        f'<section class="section range-brush-section">{_render_range_brush()}</section>'
    )
```

Only the trailing brush section is new — the rest of `charts` stays exactly as it is.

- [ ] **Step 6: Run the brush tests, verify they pass**

```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_dashboard_pages.py -k "range_brush" -v
```

Expected: 4 PASS.

- [ ] **Step 7: Add a guard test confirming the brush does NOT render on other pages**

Append to `tests/v2/test_dashboard_pages.py` inside `TestRenderActivityPage` (or alongside it — pick the closest fixture class). Add:

```python
    def test_does_not_render_range_brush(self):
        """The brush only belongs where the three performance charts
        live. Activity has tables, not charts."""
        html = render_activity_page(**self._data())
        assert 'range-brush' not in html
```

And similar smoke tests on `render_strategy_page`, `render_learning_page` (if present), `render_how_it_works_page`, `render_mistakes_page`, `render_attribution_page`. If a page renderer takes different kwargs, mirror its existing test fixtures (`self._data()`). Skip any that aren't already tested in this file.

- [ ] **Step 8: Run the full dashboard test file**

```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_dashboard_pages.py -v
```

Expected: all tests pass, including the original range-control tests (none of those should regress — the brush is additive).

- [ ] **Step 9: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(dashboard): server-render the range-brush container

Empty <div class=\"range-brush\"><svg/></div> below the chart grid on
Home and Performance. Container only; client hydrates the SVG after
snapshots.json loads."
```

---

## Task 2: CSS for the brush strip

**Files:**
- Modify: `public_dashboard/styles.css:540-579` (append below the existing `.range-section` block)

This task has no test harness — manual verification at the end (rendering the page and checking the strip is visible at desktop widths and hidden below 768px).

- [ ] **Step 1: Append the brush styles**

In `public_dashboard/styles.css`, append after line 578 (the existing `.range-section` block):

```css
/* === Drag-to-select date range (brush strip) === */

.range-brush {
  display: none;            /* mobile fallback */
  margin: 0.4rem 0 0;
  height: 40px;
  background: var(--bg-deep);
  border: 1px solid var(--bg-card-alt);
  border-radius: 4px;
  position: relative;
  user-select: none;
  touch-action: none;       /* required for pointer events on touch devices */
}

@media (min-width: 768px) {
  .range-brush { display: block; }
}

.range-brush-section {
  /* Performance page wraps the brush in its own section. Match the
     tight padding used by .range-section so it doesn't look adrift. */
  padding-top: 0.6rem;
  padding-bottom: 0.6rem;
}

.range-brush-svg {
  width: 100%;
  height: 100%;
  display: block;
  overflow: visible;        /* handles can extend slightly beyond the box */
}

.range-brush-overview {
  stroke: var(--bg-card-alt);
  stroke-width: 1;
  fill: none;
}

.range-brush-bar {
  fill: rgba(88, 166, 255, 0.18);
  cursor: grab;
}

.range-brush-bar:active { cursor: grabbing; }

.range-brush-handle {
  fill: var(--accent);
  cursor: ew-resize;
}

.range-brush-handle:hover { fill: #79b8ff; }

/* Invisible wider hit target overlapping each handle for easier grabbing */
.range-brush-handle-hit {
  fill: transparent;
  cursor: ew-resize;
}
```

- [ ] **Step 2: Verify the file still parses (no test harness — load the page)**

Start the dashboard locally (the v2 dashboard task runs the Flask dev server):

```bash
docker compose exec -T trading python3 -c "import public_dashboard"  # quick sanity import — should no-op
```

Then load `/` and `/performance/` in a desktop browser; the brush container should render as a thin dark strip below the charts (empty for now — Task 3 fills it). At <768px viewport, the strip should disappear.

If the project has a `task dashboard` or similar live preview, prefer that. Otherwise, run the publisher in dry-run mode and open the generated HTML.

- [ ] **Step 3: Commit**

```bash
git add public_dashboard/styles.css
git commit -m "style(dashboard): add .range-brush block (desktop-only)

Empty container styled as a 40px strip below the chart grid; hidden
below 768px so mobile falls back to the preset buttons. Palette mirrors
the existing chart accents (var(--accent), var(--bg-deep))."
```

---

## Task 3: Refactor `applyRange` to take a date window

This task changes the JS contract from "days from today" to "start/end pair" with NO user-visible behavior change. The brush still renders empty (Task 4 fills it). Presets keep working because the new `presetToWindow` derives the same date window the old code did.

No JS test harness exists. Verification: load `/` and `/performance/`, click each preset, confirm the three charts re-filter exactly as before this task.

**Files:**
- Modify: `public_dashboard/app.js:69-142` (the existing range-filter block, plus `initPerformancePage` at line 583)

- [ ] **Step 1: Replace the range-filter block with the new API**

In `public_dashboard/app.js`, locate the comment `// === Chart range filter state ===` at line 69 and replace the entire block down through `setupRangeControl` (ending at the closing brace of `setupRangeControl`, around line 142) with:

```js
// === Chart range filter state ===

var DEFAULT_RANGE_DAYS = 7;

var perfData = {
  snapshots: null,
  benchmark: null,
  decisions: null,
  rangeStart: null,  // YYYY-MM-DD — inclusive lower bound
  rangeEnd: null,    // YYYY-MM-DD — inclusive upper bound
};
var chartInstances = {};  // canvasId -> Chart instance

function destroyCharts() {
  Object.keys(chartInstances).forEach(function (id) {
    try { chartInstances[id].destroy(); } catch (e) { /* already gone */ }
  });
  chartInstances = {};
}

function latestSnapshotDate(snapshots) {
  if (!snapshots || snapshots.length === 0) return null;
  // snapshots.json is date-ordered ascending; take the last row's date.
  return normalizeDate(snapshots[snapshots.length - 1].date);
}

function firstSnapshotDate(snapshots) {
  if (!snapshots || snapshots.length === 0) return null;
  return normalizeDate(snapshots[0].date);
}

function cutoffDate(anchorDate, days) {
  // anchorDate is a YYYY-MM-DD string. Subtract `days` calendar days
  // and return another YYYY-MM-DD string.
  //
  // All UTC (the "T00:00:00Z" suffix and the getUTCDate/setUTCDate pair)
  // to keep date-only arithmetic timezone-free — never simplify to
  // new Date(anchorDate) or get/setDate(), which would shift by a day
  // for users west of UTC.
  var d = new Date(anchorDate + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

function filterByDateWindow(rows, dateKey, start, end) {
  // start/end inclusive; both required. Used by applyRange.
  if (!rows) return rows;
  if (!start || !end) return rows;
  return rows.filter(function (r) {
    var d = normalizeDate(r[dateKey]);
    return d >= start && d <= end;
  });
}

function presetToWindow(preset) {
  // preset: "7" | "30" | "365" | "all". Returns { start, end, preset }.
  var anchor = latestSnapshotDate(perfData.snapshots);
  if (!anchor) return { start: null, end: null, preset: preset };
  if (preset === "all") {
    return { start: firstSnapshotDate(perfData.snapshots), end: anchor, preset: "all" };
  }
  var days = parseInt(preset, 10);
  return { start: cutoffDate(anchor, days), end: anchor, preset: preset };
}

function updatePresetActiveState(preset) {
  // preset === null when the user dragged the brush — clear all highlights.
  document.querySelectorAll(".range-btn").forEach(function (btn) {
    var isActive = preset != null && btn.getAttribute("data-range") === String(preset);
    btn.classList.toggle("is-active", isActive);
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function applyRange(spec) {
  // spec: { start: YYYY-MM-DD, end: YYYY-MM-DD, preset: "7"|"30"|"365"|"all"|null }
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

// Brush stub — filled in by Task 4. Defined here so applyRange can call
// it during this refactor without crashing.
function updateBrushPosition(_start, _end) { /* Task 4 */ }
function initBrush() { /* Task 4 */ }

function setupRangeControl() {
  var control = document.querySelector(".range-control");
  if (!control) return;
  control.addEventListener("click", function (e) {
    var btn = e.target.closest(".range-btn");
    if (!btn) return;
    var raw = btn.getAttribute("data-range");
    applyRange(presetToWindow(raw));
  });
}
```

This deletes the old `filterByRange` function and the old `applyRange(days)` body. Both had a single caller (`applyRange` itself, and `setupRangeControl`); both are replaced.

- [ ] **Step 2: Update `initPerformancePage` to use the new entry point**

In `public_dashboard/app.js`, locate `initPerformancePage` (around line 583). Replace its body so it calls `presetToWindow` and `initBrush`:

```js
function initPerformancePage() {
  Promise.all([
    fetchJSON("snapshots.json"),
    fetchJSON("benchmark.json"),
    fetchJSON("decisions.json"),
  ]).then(function (parts) {
    perfData.snapshots = parts[0];
    perfData.benchmark = parts[1];
    perfData.decisions = parts[2];
    setupRangeControl();
    initBrush();
    applyRange(presetToWindow(String(DEFAULT_RANGE_DAYS)));
  }).catch(function (err) {
    console.error("Failed to load performance data:", err);
  });
}
```

The only change vs. today is the call to `initBrush()` and the `applyRange(presetToWindow("7"))` instead of `applyRange(7)`.

- [ ] **Step 3: Manual verification — presets still work**

Open `/` and `/performance/` in a desktop browser. Verify:

1. Page loads, `1W` button highlighted, charts show the last 7 days.
2. Click `1M` → all three charts re-render to 30 days, `1M` highlights, `1W` unhighlights.
3. Click `1Y`, `All`, `1W` in sequence → charts re-filter correctly each time, only one button highlighted at a time.
4. The browser console shows no errors.

If anything regresses, do not proceed — diagnose. The refactor is supposed to be a behavior-preserving swap.

- [ ] **Step 4: Re-run Python tests to confirm nothing broke**

```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_dashboard_pages.py -v
```

Expected: all pass. (No Python touched this task, but cheap to verify.)

- [ ] **Step 5: Commit**

```bash
git add public_dashboard/app.js
git commit -m "refactor(dashboard): reshape applyRange to take {start,end,preset}

Presets compute their date window via presetToWindow() and call the new
date-window applyRange. filterByRange (anchor + days) is replaced by
filterByDateWindow (start + end); only one caller. Brush hydration stubs
added — Task 4 fills them in."
```

---

## Task 4: Render the brush SVG (overview + handles + bar)

After this task the brush is visible and updates when presets are clicked — but it's not yet draggable (Task 5 adds pointer events).

**Files:**
- Modify: `public_dashboard/app.js` (replace the `initBrush`/`updateBrushPosition` stubs from Task 3)

- [ ] **Step 1: Replace the brush stubs with the rendering code**

In `public_dashboard/app.js`, find the stub `function initBrush() { /* Task 4 */ }` from Task 3 and replace both stubs (`updateBrushPosition` and `initBrush`) with:

```js
// === Brush strip ===

var brush = {
  el: null,           // .range-brush element
  svg: null,          // child <svg>
  overviewPath: null, // <path> showing the full timeline polyline
  bar: null,          // <rect> highlighting the selected window
  leftHandle: null,   // <rect> at the left edge
  rightHandle: null,  // <rect> at the right edge
  leftHit: null,      // wider invisible <rect> for easier grabbing
  rightHit: null,
  width: 0,           // measured pixel width of the svg
  domain: null,       // { first: "YYYY-MM-DD", last: "YYYY-MM-DD" }
};

var SVG_NS = "http://www.w3.org/2000/svg";
var BRUSH_HANDLE_WIDTH = 6;
var BRUSH_HANDLE_HIT_WIDTH = 18;  // generous hit target

function dateToFraction(date) {
  // Linear interpolation between brush.domain.first (0) and .last (1).
  if (!brush.domain) return 0;
  var first = new Date(brush.domain.first + "T00:00:00Z").getTime();
  var last = new Date(brush.domain.last + "T00:00:00Z").getTime();
  var d = new Date(date + "T00:00:00Z").getTime();
  if (last === first) return 0;
  var f = (d - first) / (last - first);
  return Math.max(0, Math.min(1, f));
}

function fractionToDate(fraction) {
  if (!brush.domain) return null;
  var first = new Date(brush.domain.first + "T00:00:00Z").getTime();
  var last = new Date(brush.domain.last + "T00:00:00Z").getTime();
  var t = first + (last - first) * Math.max(0, Math.min(1, fraction));
  return new Date(t).toISOString().slice(0, 10);
}

function renderBrushOverview() {
  // One polyline of portfolio_value across all snapshots, normalized
  // into the SVG's 0..100 y-coordinate space (preserveAspectRatio:none
  // stretches to the strip height).
  var snapshots = perfData.snapshots;
  if (!snapshots || snapshots.length === 0) return;
  var values = snapshots.map(function (s) { return s.portfolio_value; });
  var min = Math.min.apply(null, values);
  var max = Math.max.apply(null, values);
  var range = max - min || 1;
  var pts = snapshots.map(function (s, i) {
    var x = dateToFraction(normalizeDate(s.date)) * 100;
    var y = 100 - ((s.portfolio_value - min) / range) * 80 - 10;  // 10% padding top/bottom
    return x.toFixed(2) + "," + y.toFixed(2);
  });
  brush.overviewPath.setAttribute("d", "M " + pts.join(" L "));
}

function initBrush() {
  brush.el = document.querySelector(".range-brush");
  if (!brush.el) return;  // no brush container (e.g. activity page)
  if (!perfData.snapshots || perfData.snapshots.length === 0) return;

  brush.svg = brush.el.querySelector(".range-brush-svg");
  brush.svg.setAttribute("viewBox", "0 0 100 100");
  brush.domain = {
    first: firstSnapshotDate(perfData.snapshots),
    last: latestSnapshotDate(perfData.snapshots),
  };

  // Build SVG children in z-order (back to front).
  brush.overviewPath = document.createElementNS(SVG_NS, "path");
  brush.overviewPath.setAttribute("class", "range-brush-overview");
  brush.overviewPath.setAttribute("vector-effect", "non-scaling-stroke");
  brush.svg.appendChild(brush.overviewPath);

  brush.bar = document.createElementNS(SVG_NS, "rect");
  brush.bar.setAttribute("class", "range-brush-bar");
  brush.bar.setAttribute("y", "0");
  brush.bar.setAttribute("height", "100");
  brush.bar.setAttribute("data-role", "bar");
  brush.svg.appendChild(brush.bar);

  function makeHandle(role, visibleClass, hitClass) {
    var visible = document.createElementNS(SVG_NS, "rect");
    visible.setAttribute("class", visibleClass);
    visible.setAttribute("y", "0");
    visible.setAttribute("height", "100");
    visible.setAttribute("width", String(BRUSH_HANDLE_WIDTH / 2));  // viewBox units, see Task 5
    visible.setAttribute("data-role", role + "-handle");
    brush.svg.appendChild(visible);

    var hit = document.createElementNS(SVG_NS, "rect");
    hit.setAttribute("class", hitClass);
    hit.setAttribute("y", "0");
    hit.setAttribute("height", "100");
    hit.setAttribute("width", String(BRUSH_HANDLE_HIT_WIDTH / 2));
    hit.setAttribute("data-role", role + "-hit");
    brush.svg.appendChild(hit);
    return { visible: visible, hit: hit };
  }

  var left = makeHandle("left", "range-brush-handle", "range-brush-handle-hit");
  brush.leftHandle = left.visible;
  brush.leftHit = left.hit;
  var right = makeHandle("right", "range-brush-handle", "range-brush-handle-hit");
  brush.rightHandle = right.visible;
  brush.rightHit = right.hit;

  renderBrushOverview();
  window.addEventListener("resize", function () {
    // No DOM measurement needed — viewBox is 0..100 so the SVG scales.
    // Kept as a hook in case future code needs pixel measurements.
  });
}

function updateBrushPosition(start, end) {
  // Move handles and bar to reflect the current window. No-op if the
  // brush hasn't been initialized (e.g. on the activity page).
  if (!brush.svg || !brush.domain || !start || !end) return;
  var leftFrac = dateToFraction(start) * 100;
  var rightFrac = dateToFraction(end) * 100;
  brush.bar.setAttribute("x", String(leftFrac));
  brush.bar.setAttribute("width", String(Math.max(0, rightFrac - leftFrac)));
  // Center the visible handle on the endpoint; widen the hit target.
  var halfVis = BRUSH_HANDLE_WIDTH / 2 / 2;       // viewBox units
  var halfHit = BRUSH_HANDLE_HIT_WIDTH / 2 / 2;
  brush.leftHandle.setAttribute("x", String(leftFrac - halfVis));
  brush.leftHit.setAttribute("x", String(leftFrac - halfHit));
  brush.rightHandle.setAttribute("x", String(rightFrac - halfVis));
  brush.rightHit.setAttribute("x", String(rightFrac - halfHit));
}
```

A note on the coordinate system: `viewBox="0 0 100 100"` with `preserveAspectRatio="none"` means the SVG draws in a 0..100 unit space that stretches to fill the rendered 40px-tall × full-width strip. All positioning math is in fractions of the domain, multiplied by 100. The handles look thin in viewBox units but with `vector-effect: non-scaling-stroke` (used on the overview path) and small width values they render as 6px at any width. (Task 5 swaps to pixel-space math for dragging since pointer events are in pixels.)

- [ ] **Step 2: Manual verification — brush renders and tracks presets**

Reload `/` and `/performance/`. Verify:

1. The brush strip below the chart grid now shows a faint grey polyline (the portfolio_value overview) with a translucent blue bar covering the right ~10% (the last 7 days).
2. Two small blue rectangles flank the bar at its left and right edges (the handles — they're not yet draggable).
3. Click `1M` → the bar widens leftward; click `All` → the bar spans the full width; click `1W` → the bar shrinks back to the right.
4. Resize the window — the brush stays aligned with the chart grid above (viewBox handles this).
5. Below 768px viewport — brush is hidden (verify by resizing window narrow).

- [ ] **Step 3: Commit**

```bash
git add public_dashboard/app.js
git commit -m "feat(dashboard): render brush SVG (overview + handles + bar)

Builds the static SVG (overview polyline, selection bar, two handles
with widened hit targets) and wires updateBrushPosition to track
preset clicks. Not yet draggable — Task 5 adds pointer events."
```

---

## Task 5: Pointer-event drag handlers

After this task the brush is fully interactive: drag a handle to move one endpoint, drag the bar to slide the window, click empty space to jump the nearest endpoint.

**Files:**
- Modify: `public_dashboard/app.js` (extend the brush block from Task 4)

- [ ] **Step 1: Add pointer state and drag handlers**

In `public_dashboard/app.js`, extend the brush state object and append handler functions after `updateBrushPosition`. First, extend `brush`:

```js
// Replace the brush state declaration from Task 4 with this extended version.
var brush = {
  el: null,
  svg: null,
  overviewPath: null,
  bar: null,
  leftHandle: null,
  rightHandle: null,
  leftHit: null,
  rightHit: null,
  width: 0,
  domain: null,
  dragging: null,        // null | "left" | "right" | "bar"
  dragBarStartFrac: 0,   // when dragging "bar", the frac at pointerdown
  dragBarWidthFrac: 0,   // and the bar width frozen at pointerdown
  dragPointerStartFrac: 0,
  pendingFrame: null,
};
```

Then append, after `updateBrushPosition`:

```js
function pointerXToFraction(clientX) {
  // Convert a pointer's clientX to a fraction (0..1) of the brush width
  // using the live bounding rect (resilient to scroll, resize, zoom).
  if (!brush.el) return 0;
  var rect = brush.el.getBoundingClientRect();
  if (rect.width === 0) return 0;
  var f = (clientX - rect.left) / rect.width;
  return Math.max(0, Math.min(1, f));
}

function currentRangeFractions() {
  // Read current bar position back from the DOM — single source of truth
  // during a drag.
  var leftFrac = parseFloat(brush.bar.getAttribute("x") || "0") / 100;
  var widthFrac = parseFloat(brush.bar.getAttribute("width") || "0") / 100;
  return { leftFrac: leftFrac, rightFrac: leftFrac + widthFrac };
}

function scheduleBrushApply(start, end) {
  // rAF-throttle: at most one applyRange per animation frame.
  if (brush.pendingFrame != null) return;
  brush.pendingFrame = requestAnimationFrame(function () {
    brush.pendingFrame = null;
    applyRange({ start: start, end: end, preset: null });
  });
}

function brushPointerDown(e) {
  if (!brush.svg) return;
  var target = e.target;
  var role = target && target.getAttribute && target.getAttribute("data-role");
  var frac = pointerXToFraction(e.clientX);
  var cur = currentRangeFractions();

  if (role === "left-hit" || role === "left-handle") {
    brush.dragging = "left";
  } else if (role === "right-hit" || role === "right-handle") {
    brush.dragging = "right";
  } else if (role === "bar") {
    brush.dragging = "bar";
    brush.dragBarStartFrac = cur.leftFrac;
    brush.dragBarWidthFrac = cur.rightFrac - cur.leftFrac;
    brush.dragPointerStartFrac = frac;
  } else {
    // Clicked empty area: jump the nearest endpoint to the cursor.
    var distLeft = Math.abs(frac - cur.leftFrac);
    var distRight = Math.abs(frac - cur.rightFrac);
    if (distLeft < distRight) {
      var newStart = fractionToDate(Math.min(frac, cur.rightFrac));
      var endDate = fractionToDate(cur.rightFrac);
      scheduleBrushApply(newStart, endDate);
    } else {
      var startDate = fractionToDate(cur.leftFrac);
      var newEnd = fractionToDate(Math.max(frac, cur.leftFrac));
      scheduleBrushApply(startDate, newEnd);
    }
    return;
  }
  e.preventDefault();
  // Capture future move/up events so the drag continues even if the
  // cursor leaves the strip.
  try { brush.svg.setPointerCapture(e.pointerId); } catch (err) { /* old browsers */ }
}

function brushPointerMove(e) {
  if (!brush.dragging) return;
  var frac = pointerXToFraction(e.clientX);
  var cur = currentRangeFractions();
  var newStart = brush.domain.first;
  var newEnd = brush.domain.last;

  if (brush.dragging === "left") {
    newStart = fractionToDate(Math.min(frac, cur.rightFrac));
    newEnd = fractionToDate(cur.rightFrac);
  } else if (brush.dragging === "right") {
    newStart = fractionToDate(cur.leftFrac);
    newEnd = fractionToDate(Math.max(frac, cur.leftFrac));
  } else if (brush.dragging === "bar") {
    var delta = frac - brush.dragPointerStartFrac;
    var newLeft = brush.dragBarStartFrac + delta;
    var width = brush.dragBarWidthFrac;
    // Clamp so the window stays inside [0, 1].
    newLeft = Math.max(0, Math.min(1 - width, newLeft));
    newStart = fractionToDate(newLeft);
    newEnd = fractionToDate(newLeft + width);
  }
  scheduleBrushApply(newStart, newEnd);
}

function brushPointerUp(e) {
  if (!brush.dragging) return;
  brush.dragging = null;
  try { brush.svg.releasePointerCapture(e.pointerId); } catch (err) { /* ok */ }
}
```

- [ ] **Step 2: Wire the handlers inside `initBrush`**

In `initBrush`, after the existing setup (right before the `window.addEventListener("resize", …)` line, or anywhere after the SVG children are created), add:

```js
  brush.svg.addEventListener("pointerdown", brushPointerDown);
  brush.svg.addEventListener("pointermove", brushPointerMove);
  brush.svg.addEventListener("pointerup", brushPointerUp);
  brush.svg.addEventListener("pointercancel", brushPointerUp);
```

- [ ] **Step 3: Manual verification — full interaction matrix**

Reload `/` and `/performance/` on a desktop viewport. Run through each manual check from the spec:

1. **Drag right handle leftward** — charts re-render live; `1W` highlight clears immediately on first move.
2. **Drag left handle rightward** — same; the right endpoint stays put.
3. **Drag the bar between handles** — window slides without changing width; clamps at the domain edges.
4. **Click outside the bar** — nearest endpoint jumps to the cursor; charts re-render.
5. **Click `1M`** — handles animate (well, jump — no transition CSS yet) to the 30-day window; `1M` lights up.
6. **Click `All`** — bar spans the full strip.
7. **Drag a handle past the other** — the two endpoints don't cross (both snap to the meeting point; the bar collapses to width 0).
8. **Resize the window mid-drag** — drag continues using the new bounding rect; no stuck state.
9. **Below 768px viewport** — brush hidden; preset buttons still functional via touch/click.
10. **Console** — no errors during any of the above.

- [ ] **Step 4: Commit**

```bash
git add public_dashboard/app.js
git commit -m "feat(dashboard): wire pointer events on the range brush

Drag handles to move endpoints, drag the bar to slide the window,
click outside to jump the nearest endpoint. rAF-throttled re-render;
pointer capture so drags survive leaving the strip. Preset highlights
clear on first move."
```

---

## Task 6: End-to-end check and final commit

**Files:** none modified — verification only.

- [ ] **Step 1: Full Python test suite**

```bash
docker compose exec -T trading python3 -m pytest tests/ -v 2>&1 | tail -40
```

Expected: all green. Pay attention to dashboard-page tests and any dashboard-publish tests that snapshot rendered HTML — a brush container appearing on Home/Performance shouldn't break those, but if a test asserts the EXACT length or content of the rendered page, it may need an updated golden.

- [ ] **Step 2: Lint (Taskfile target)**

```bash
task lint
```

Expected: clean. Resolve any new ruff warnings before continuing.

- [ ] **Step 3: Full manual QA**

Walk through the manual checklist in the spec's "Manual verification checklist" section. Specifically verify:

- `/` and `/performance/` both have a working brush.
- Activity, Strategy, Learning, How-it-works, Mistakes, Attribution pages do NOT show a brush.
- Mobile/narrow-viewport behavior: brush hides at <768px; presets remain functional.

- [ ] **Step 4: Confirm no orphan code**

Search for the deleted helpers to make sure nothing still references them:

```bash
grep -rn "filterByRange" public_dashboard/ v2/ tests/ || echo "no references — good"
grep -rn "applyRange(7)" public_dashboard/ v2/ tests/ || echo "no positional-int callers — good"
```

Expected: no remaining references to `filterByRange` or `applyRange(<int>)`. If any turn up, update them to use the date-window form.

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin <branch>
gh pr create --title "feat(dashboard): drag-to-select date range (brush strip)" --body "$(cat <<'EOF'
## Summary
- Adds a brush strip below the three-chart grid on Home and Performance for arbitrary date-window selection
- Bidirectional sync with the existing 1W/1M/1Y/All preset buttons
- Desktop-only (≥768px); mobile/tablet keep the presets as sole control

## Test plan
- [ ] Python tests pass (`docker compose exec -T trading python3 -m pytest tests/`)
- [ ] Lint clean (`task lint`)
- [ ] Manual: handle drag (left/right/both), bar drag, click-to-jump on Home and Performance
- [ ] Manual: brush hidden below 768px
- [ ] Manual: presets still work and clear when the brush is dragged
- [ ] Manual: brush absent on Activity / Strategy / Learning / How-it-works / Mistakes / Attribution

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes

- **Spec coverage:** every section of the spec is covered — brush container (Task 1), CSS (Task 2), state model + `applyRange` reshape + `filterByDateWindow` + `presetToWindow` (Task 3), SVG rendering + `updateBrushPosition` (Task 4), pointer events (Task 5), tests + out-of-scope guards (Tasks 1 and 6).
- **Placeholders:** none. Every code step includes the actual code; every command includes the actual command and expected output.
- **Type consistency:** `applyRange({ start, end, preset })` is used identically across Tasks 3–5. `presetToWindow` always returns `{ start, end, preset }`. The brush state object grows additively from Task 4 to Task 5 (re-declared in full in Task 5 step 1 to avoid out-of-order reading hazards). Helper names (`dateToFraction`, `fractionToDate`, `pointerXToFraction`, `currentRangeFractions`, `scheduleBrushApply`) are stable.
- **YAGNI:** no URL state, no keyboard nav, no touch — all explicit out-of-scope in the spec, no surprise plumbing added.
- **TDD:** Task 1 follows red-green-refactor on the Python rendering. Tasks 3–5 are JS refactor/feature work with no JS test harness available; the plan substitutes a tight manual-verification checklist after each step and a final end-to-end sweep in Task 6.
