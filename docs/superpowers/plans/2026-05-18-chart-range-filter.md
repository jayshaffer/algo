# Chart Range Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared `1W | 1M | 1Y | All` button group above the three performance charts on the Home and Performance pages, filtering all three charts client-side from the already-fetched JSON.

**Architecture:** Server emits a button-group toolbar above the chart grid on both pages (default `1W` marked active). Client-side `app.js` caches the fetched data, destroys and re-creates the three Chart.js instances on each button click, slicing the snapshots/benchmark/decisions arrays by a date cutoff derived from the latest snapshot. No new data files or endpoints.

**Tech Stack:** Python (Jinja-free string-concat HTML in `v2/dashboard_pages.py`), vanilla JS + Chart.js 4.x in `public_dashboard/app.js`, hand-written CSS in `public_dashboard/styles.css`. Tests use pytest with HTML-substring assertions (the established pattern in `tests/v2/test_dashboard_pages.py`).

**Spec:** `docs/superpowers/specs/2026-05-18-chart-range-filter-design.md`

---

## File Structure

**Files modified:**
- `v2/dashboard_pages.py` — add `_render_range_control()` helper, call it from `_render_homepage_charts()` and `render_performance_page()`.
- `public_dashboard/app.js` — add filter helpers, chart-instance tracking, `applyRange()`, default-range wiring inside `initPerformancePage`. Modify the three `renderEquityCurve/renderPnlChart/renderBenchmark` functions to register their created Chart instance.
- `public_dashboard/styles.css` — `.range-control` and `.range-btn` rules.
- `tests/v2/test_dashboard_pages.py` — extend `TestRenderHomepage` and `TestRenderPerformancePage` with range-control assertions.

**Files created:** none.

---

### Task 1: Server — add range control to Home and Performance pages

**Files:**
- Modify: `v2/dashboard_pages.py` (add helper after `_render_homepage_charts` definition area; call from both renderers)
- Test: `tests/v2/test_dashboard_pages.py` (`TestRenderHomepage` ~line 632, `TestRenderPerformancePage` ~line 803)

- [ ] **Step 1.1: Write failing tests for the homepage range control**

Add to `TestRenderHomepage` in `tests/v2/test_dashboard_pages.py`:

```python
def test_renders_range_control_with_four_buttons(self):
    html = render_homepage(**self._data())
    assert 'class="range-control"' in html
    assert 'role="group"' in html
    assert 'aria-label="Time range"' in html
    assert 'data-range="7"' in html
    assert 'data-range="30"' in html
    assert 'data-range="365"' in html
    assert 'data-range="all"' in html
    assert ">1W</button>" in html
    assert ">1M</button>" in html
    assert ">1Y</button>" in html
    assert ">All</button>" in html

def test_range_control_defaults_to_1w_active(self):
    html = render_homepage(**self._data())
    # The 1W button is the active one. It should carry both the
    # is-active class and aria-pressed=true; the other three should
    # carry aria-pressed=false.
    assert 'data-range="7" class="range-btn is-active" aria-pressed="true"' in html
    assert 'data-range="30" class="range-btn" aria-pressed="false"' in html
    assert 'data-range="365" class="range-btn" aria-pressed="false"' in html
    assert 'data-range="all" class="range-btn" aria-pressed="false"' in html
```

- [ ] **Step 1.2: Write failing tests for the performance page range control**

Add to `TestRenderPerformancePage` in the same file:

```python
def test_renders_range_control_with_four_buttons(self):
    html = render_performance_page(**self._data())
    assert 'class="range-control"' in html
    assert 'data-range="7"' in html
    assert 'data-range="30"' in html
    assert 'data-range="365"' in html
    assert 'data-range="all"' in html

def test_range_control_defaults_to_1w_active(self):
    html = render_performance_page(**self._data())
    assert 'data-range="7" class="range-btn is-active" aria-pressed="true"' in html
    assert 'data-range="all" class="range-btn" aria-pressed="false"' in html
```

- [ ] **Step 1.3: Run the new tests and confirm they fail**

Run:
```bash
python3 -m pytest tests/v2/test_dashboard_pages.py -k "range_control" -v
```
Expected: 4 failures — `'class="range-control"' in html` is False (the markup doesn't exist yet).

- [ ] **Step 1.4: Implement the `_render_range_control` helper**

Add to `v2/dashboard_pages.py` immediately above `_render_homepage_charts` (around line 857):

```python
def _render_range_control() -> str:
    """Toolbar above the three-chart grid. Client JS reads data-range
    and re-renders the charts with a filtered slice of the snapshots
    JSON. Default active button = 1W, kept in sync with
    DEFAULT_RANGE_DAYS in public_dashboard/app.js."""
    buttons = [
        ("7", "1W", True),
        ("30", "1M", False),
        ("365", "1Y", False),
        ("all", "All", False),
    ]
    parts = ['<div class="range-control" role="group" aria-label="Time range">']
    for data_range, label, active in buttons:
        cls = "range-btn is-active" if active else "range-btn"
        pressed = "true" if active else "false"
        parts.append(
            f'<button type="button" data-range="{data_range}" '
            f'class="{cls}" aria-pressed="{pressed}">{label}</button>'
        )
    parts.append('</div>')
    return "".join(parts)
```

- [ ] **Step 1.5: Insert the helper into `_render_homepage_charts`**

Locate `_render_homepage_charts` (around line 857). Insert the range control between `<div class="head">…</div>` and `<div class="chart-grid">`. The function becomes:

```python
def _render_homepage_charts() -> str:
    return (
        '<section class="section front-charts">'
        '<div class="head"><h2>Performance</h2>'
        '<a class="more" href="/performance/">Full view →</a></div>'
        + _render_range_control() +
        '<div class="chart-grid">'
        '<div class="chart-panel primary">'
        '<div class="chart-title">Equity curve</div>'
        '<div class="chart-wrap"><canvas id="equity-chart"></canvas></div>'
        '<p class="empty-state" id="chart-empty" style="display:none;">No snapshot data yet</p>'
        '</div>'
        '<div class="chart-panel">'
        '<div class="chart-title">Cumulative P&amp;L</div>'
        '<div class="chart-wrap"><canvas id="pnl-chart"></canvas></div>'
        '<p class="empty-state" id="pnl-empty" style="display:none;">No snapshot data yet</p>'
        '</div>'
        '<div class="chart-panel">'
        '<div class="chart-title">vs S&amp;P 500</div>'
        '<div class="chart-wrap"><canvas id="benchmark-chart"></canvas></div>'
        '<p class="empty-state" id="benchmark-empty" style="display:none;">No benchmark data yet</p>'
        '</div>'
        '</div></section>'
    )
```

- [ ] **Step 1.6: Insert the helper into `render_performance_page`**

In `render_performance_page` (around line 970), find the `charts = (...)` block. Prefix it with the range control as its own `<section class="section">` so it sits above all three chart sections:

```python
charts = (
    '<section class="section range-section">'
    + _render_range_control() +
    '</section>'
    '<section class="section">'
    '<div class="head"><h2>Equity curve</h2></div>'
    '<p class="section-subtitle">Account value over time. '
    'SPY line shows what the same deposits would be worth in an index fund.</p>'
    '<div class="chart-wrap"><canvas id="equity-chart"></canvas></div>'
    '<p class="empty-state" id="chart-empty" style="display:none;">No snapshot data yet</p>'
    '</section>'
    '<section class="section">'
    '<div class="head"><h2>Cumulative P&amp;L</h2></div>'
    '<p class="section-subtitle">Trading gain/loss in dollars, '
    'with cash deposits subtracted out.</p>'
    '<div class="chart-wrap"><canvas id="pnl-chart"></canvas></div>'
    '<p class="empty-state" id="pnl-empty" style="display:none;">No snapshot data yet</p>'
    '</section>'
    '<section class="section"><div class="head"><h2>Performance vs S&amp;P 500</h2></div>'
    '<div class="chart-wrap"><canvas id="benchmark-chart"></canvas></div>'
    '<p class="empty-state" id="benchmark-empty" style="display:none;">No benchmark data yet</p>'
    '</section>'
)
```

- [ ] **Step 1.7: Run the new tests and confirm they pass**

Run:
```bash
python3 -m pytest tests/v2/test_dashboard_pages.py -k "range_control" -v
```
Expected: 4 passes.

- [ ] **Step 1.8: Run the full dashboard test files to confirm no regressions**

Run:
```bash
python3 -m pytest tests/v2/test_dashboard_pages.py tests/v2/test_dashboard_publish.py -v
```
Expected: all pass. If any pre-existing test fails because it counts `<section>` tags or asserts exact substring boundaries that the new section disturbs, that test is checking too tightly — fix it by either making the assertion looser or adjusting it to the new structure.

- [ ] **Step 1.9: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py
git commit -m "feat(dashboard): add range-control toolbar to Home and Performance pages

Server-side toolbar with 1W/1M/1Y/All buttons above the three performance
charts on both pages. Default-active button is 1W. Client-side filter
wiring lands in the next commit."
```

---

### Task 2: Client — filter logic, chart-instance tracking, applyRange wiring

**Files:**
- Modify: `public_dashboard/app.js` (whole-file changes — add module state, helpers; modify three render functions; rewrite `initPerformancePage`)

No automated test harness exists for `app.js`; this task is verified manually in Task 4. The client-side filter logic is small and the visual feedback is immediate.

- [ ] **Step 2.1: Add module-scope state and helpers**

In `public_dashboard/app.js`, after the existing helpers section (after `fetchJSON` around line 67) and before `setupHamburger`, add:

```js
// === Chart range filter state ===

var DEFAULT_RANGE_DAYS = 7;

var perfData = { snapshots: null, benchmark: null, decisions: null };
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

function cutoffDate(anchorDate, days) {
  // anchorDate is a YYYY-MM-DD string. Subtract `days` calendar days
  // and return another YYYY-MM-DD string. Filtering uses `>= cutoff`.
  var d = new Date(anchorDate + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

function filterByRange(rows, dateKey, days, anchor) {
  if (!rows) return rows;
  if (days == null) return rows;  // "All"
  if (!anchor) return rows;
  var cutoff = cutoffDate(anchor, days);
  return rows.filter(function (r) { return normalizeDate(r[dateKey]) >= cutoff; });
}
```

Note: `normalizeDate` already exists in the file (line ~94) — reuse it, don't redefine.

- [ ] **Step 2.2: Modify the three render functions to register chart instances**

In `renderEquityCurve`, replace the line `new Chart(canvas, {` (around line 196) with:

```js
chartInstances["equity-chart"] = new Chart(canvas, {
```

In `renderPnlChart`, replace the line `new Chart(canvas, {` (around line 268) with:

```js
chartInstances["pnl-chart"] = new Chart(canvas, {
```

In `renderBenchmark`, replace the line `new Chart(canvas, {` (around line 378) with:

```js
chartInstances["benchmark-chart"] = new Chart(canvas, {
```

Also, each render function currently early-returns when its dataset is empty by setting `canvas.style.display = "none"` and showing the empty-state message. Add a counterpart: when data IS present and the chart will render, set `canvas.style.display = ""` and hide the empty-state element. This lets re-renders recover from a previously-empty state.

In `renderEquityCurve`, just before the `var labels = …` line (around line 158), insert:

```js
canvas.style.display = "";
if (emptyMsg) emptyMsg.style.display = "none";
```

Do the same in `renderPnlChart` just before its `var labels = …` line (around line 247) and in `renderBenchmark` just before its `var labels = …` line (around line 329).

- [ ] **Step 2.3: Add `applyRange` and the click-listener wiring**

Add this function alongside the helpers from Step 2.1:

```js
function applyRange(days) {
  destroyCharts();
  var snapshots = perfData.snapshots;
  var benchmark = perfData.benchmark;
  var decisions = perfData.decisions;

  var anchor = latestSnapshotDate(snapshots);
  var filteredSnapshots = filterByRange(snapshots, "date", days, anchor);
  var filteredBenchmark = filterByRange(benchmark, "date", days, anchor);
  var filteredDecisions = filterByRange(decisions, "date", days, anchor);

  renderEquityCurve(filteredSnapshots, filteredDecisions);
  renderPnlChart(filteredSnapshots, filteredDecisions);
  renderBenchmark(filteredSnapshots, filteredBenchmark, filteredDecisions);

  document.querySelectorAll(".range-btn").forEach(function (btn) {
    var isActive = btn.getAttribute("data-range") === String(days == null ? "all" : days);
    btn.classList.toggle("is-active", isActive);
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function setupRangeControl() {
  var control = document.querySelector(".range-control");
  if (!control) return;
  control.addEventListener("click", function (e) {
    var btn = e.target.closest(".range-btn");
    if (!btn) return;
    var raw = btn.getAttribute("data-range");
    var days = raw === "all" ? null : parseInt(raw, 10);
    applyRange(days);
  });
}
```

- [ ] **Step 2.4: Rewrite `initPerformancePage` to cache data, wire the control, render with default range**

Replace the existing `initPerformancePage` (around line 498) with:

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
    applyRange(DEFAULT_RANGE_DAYS);
  }).catch(function (err) {
    console.error("Failed to load performance data:", err);
  });
}
```

- [ ] **Step 2.5: Confirm the file still parses by serving the dashboard data dir**

Run the project's existing dashboard tests (they exercise the publish path, not the JS, but they confirm `app.js` is still being shipped):

```bash
python3 -m pytest tests/v2/test_dashboard_publish.py -v
```
Expected: all pass.

For JS-syntax sanity, also run:

```bash
node --check public_dashboard/app.js
```
Expected: no output (success). If `node` is not available, skip this step — it's a sanity check, not a gate.

- [ ] **Step 2.6: Commit**

```bash
git add public_dashboard/app.js
git commit -m "feat(dashboard): wire client-side range filter for the three charts

initPerformancePage caches the fetched snapshots/benchmark/decisions,
renders with the default range (1W), and re-renders all three charts on
range-button click. Chart instances are tracked so they can be destroyed
before re-render."
```

---

### Task 3: CSS — `.range-control` and `.range-btn` styling

**Files:**
- Modify: `public_dashboard/styles.css` (append to the end, or place near the existing `.badge` / nav-link blocks if you prefer co-location)

- [ ] **Step 3.1: Add the range-control styles**

Append to `public_dashboard/styles.css`:

```css
/* === Chart range filter === */

.range-control {
  display: flex;
  gap: 0.4rem;
  margin: 0 0 0.9rem 0;
  flex-wrap: wrap;
}

.range-btn {
  background: transparent;
  color: var(--text-dim);
  border: 1px solid var(--bg-card-alt);
  padding: 0.3rem 0.7rem;
  font-family: inherit;
  font-size: 0.8rem;
  cursor: pointer;
  border-radius: 3px;
  transition: color 0.1s, border-color 0.1s, background 0.1s;
}

.range-btn:hover {
  color: var(--text);
  border-color: var(--accent);
}

.range-btn.is-active {
  color: var(--bg-deep);
  background: var(--accent);
  border-color: var(--accent);
}

/* Performance page wraps the range control in its own section; tighten
   the section's padding so the toolbar doesn't look adrift. */
.range-section {
  padding-top: 0.6rem;
  padding-bottom: 0.6rem;
}
```

If `--text-dim`, `--text`, `--bg-deep`, `--bg-card-alt`, or `--accent` are not defined in `:root`, substitute the closest existing variable — check the top of `styles.css` for the current palette. Based on the existing rules, these five are all defined.

- [ ] **Step 3.2: Commit**

```bash
git add public_dashboard/styles.css
git commit -m "style(dashboard): style the chart range-control button group"
```

---

### Task 4: Manual verification

This task has no code changes — it verifies the feature end-to-end before declaring done. Use the existing `public_dashboard/data/` snapshot to render locally.

- [ ] **Step 4.1: Regenerate the dashboard output and open it locally**

Two ways to view the dashboard:

(a) If you have a published-output directory from a recent `dashboard_publish` run, serve it:

```bash
cd public_dashboard && python3 -m http.server 8765
```
Then open `http://localhost:8765/` in a browser. (`public_dashboard/` contains the static `app.js`, `styles.css`, and `data/`; the HTML pages live wherever the publisher writes them — usually a sibling `dist/` or `_site/` dir. If the HTML isn't co-located with `app.js`, see option (b).)

(b) Render the homepage and performance page directly with Python:

```bash
python3 -c "
from v2.dashboard_pages import render_homepage, render_performance_page
from datetime import date
from decimal import Decimal

base_data = dict(base_url='http://localhost:8765')
home = render_homepage(
    summary={'portfolio_value': Decimal('100000'), 'daily_pnl': Decimal('0'),
             'daily_pnl_pct': Decimal('0'), 'total_return_pct': Decimal('0'),
             'vs_spy_pct': Decimal('0'), 'day_number': 1, 'last_updated': '2026-05-18T16:30:00'},
    theses=[], sparkline_svg='<svg></svg>',
    today_move=None, decisions=[],
    attribution_top=None, worst_loser=None, memo=None,
    how_it_works_state={'about': True, 'internals': True, 'trace': True},
    **base_data,
)
perf = render_performance_page(
    summary={'portfolio_value': Decimal('100000'), 'daily_pnl_pct': Decimal('0'),
             'total_return_pct': Decimal('0'), 'vs_spy_pct': Decimal('0')},
    performance={'max_drawdown_pct': 0, 'win_rate_pct': 0, 'avg_days_held': 0,
                 'best_day_pct': 0, 'worst_day_pct': 0},
    **base_data,
)
open('public_dashboard/index.html', 'w').write(home)
open('public_dashboard/performance.html', 'w').write(perf)
print('Wrote public_dashboard/index.html and performance.html')
"
cd public_dashboard && python3 -m http.server 8765
```

Then visit `http://localhost:8765/index.html` and `http://localhost:8765/performance.html`.

- [ ] **Step 4.2: Verify default state**

On both pages:
- The `1W | 1M | 1Y | All` button group renders above the charts.
- The `1W` button is visually distinct (inverted color) and the others are not.
- The charts render showing only the last week of data (with the bundled ~11-day dataset, this is the last 5–7 snapshots).
- No Chart.js errors in the browser console.

- [ ] **Step 4.3: Verify each preset**

Click each button in turn (`1M`, `1Y`, `All`, then back to `1W`). After each click:
- The active-button highlight moves to the clicked button; the previously-active one returns to inactive styling.
- All three charts re-render to the new window.
- With ~11 days of bundled data, `1M`, `1Y`, and `All` all show the full ~11 days. Expected — see spec's "sparse data" rule.
- Buy/sell markers on the three charts also filter to the window.
- The benchmark chart's SPY rebases to the first in-range date (the SPY line starts at 0% from the chart's left edge).

- [ ] **Step 4.4: Verify no chart-instance leaks**

Click between buttons rapidly several times. Expected:
- Each click produces exactly one re-render per chart (no overlapping ghost lines, which would indicate a leaked Chart.js instance).
- Browser DevTools memory does not climb unboundedly (informal — open Performance tab, take a heap snapshot if curious).

- [ ] **Step 4.5: Verify empty-state recovery**

(Optional, only if you want to exercise the empty-state path.) Temporarily replace `public_dashboard/data/snapshots.json` with `[]`, reload the page. Expected: all three charts show their empty-state message instead of rendering. Restore the file after.

- [ ] **Step 4.6: Clean up the local-render artifacts**

If you used option (b) in Step 4.1:

```bash
rm public_dashboard/index.html public_dashboard/performance.html
```

These are not tracked by git, but remove them to keep the working tree clean.

- [ ] **Step 4.7: Run the full test suite as a final regression check**

```bash
python3 -m pytest tests/v2/ -v
```
Expected: all pass.

---

## Done criteria

- All four tasks complete with their checkboxes ticked.
- `pytest tests/v2/` is green.
- The four range buttons render on Home and Performance, with `1W` active by default.
- Clicking a button re-renders all three charts to the chosen window with no console errors and no ghost chart instances.
- Spec is satisfied: shared control across all three charts, default `1W`, sparse-data permissiveness, no URL state, no per-chart override, no drag-to-zoom (deferred).
