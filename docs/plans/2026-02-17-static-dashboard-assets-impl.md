# Static Dashboard Assets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the static HTML/CSS/JS files for the public GitHub Pages dashboard that renders JSON data produced by the existing `v2/dashboard_publish.py` pipeline.

**Architecture:** Three static files (`index.html`, `styles.css`, `app.js`) in a `public_dashboard/` directory. The page fetches `data/*.json` files on load and renders everything client-side. Chart.js 4.x from CDN for the equity curve. No build step, no framework.

**Tech Stack:** Vanilla HTML/CSS/JS, Chart.js 4.x (CDN).

---

### Task 1: Create index.html

The page structure with all section containers, Chart.js CDN link, and references to styles.css and app.js.

**Files:**
- Create: `public_dashboard/index.html`

**Step 1: Create the HTML file**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bikini Bottom Capital</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>
    <header>
        <div class="container">
            <h1>&#9875; Bikini Bottom Capital</h1>
            <p class="subtitle" id="last-updated">Loading...</p>
        </div>
    </header>

    <main class="container">
        <!-- Summary Cards -->
        <section id="summary" class="card-grid">
            <div class="card">
                <span class="card-label">Portfolio Value</span>
                <span class="card-value" id="portfolio-value">—</span>
            </div>
            <div class="card">
                <span class="card-label">Daily P&L</span>
                <span class="card-value" id="daily-pnl">—</span>
            </div>
            <div class="card">
                <span class="card-label">Total Return</span>
                <span class="card-value" id="total-return">—</span>
            </div>
            <div class="card">
                <span class="card-label">Positions</span>
                <span class="card-value" id="positions-count">—</span>
            </div>
            <div class="card">
                <span class="card-label">Cash</span>
                <span class="card-value" id="cash-value">—</span>
            </div>
        </section>

        <!-- Equity Curve -->
        <section class="panel">
            <h2>Equity Curve (90 days)</h2>
            <div class="chart-wrap">
                <canvas id="equity-chart"></canvas>
            </div>
            <p class="empty-state" id="chart-empty" style="display:none;">No snapshot data yet</p>
        </section>

        <!-- Current Holdings -->
        <section class="panel">
            <h2>Current Holdings</h2>
            <div class="table-wrap">
                <table id="positions-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th class="num">Shares</th>
                            <th class="num">Avg Cost</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
            <p class="empty-state" id="positions-empty" style="display:none;">No open positions</p>
        </section>

        <!-- Recent Decisions -->
        <section class="panel">
            <h2>Recent Decisions</h2>
            <div class="table-wrap">
                <table id="decisions-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Ticker</th>
                            <th>Action</th>
                            <th class="num">Qty</th>
                            <th>Reasoning</th>
                            <th class="num">Order ID</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
            <p class="empty-state" id="decisions-empty" style="display:none;">No recent decisions</p>
        </section>

        <!-- Active Theses -->
        <section class="panel">
            <h2>Active Theses</h2>
            <div id="theses-list"></div>
            <p class="empty-state" id="theses-empty" style="display:none;">No active theses</p>
        </section>
    </main>

    <footer>
        <div class="container">
            <p>Trading from the depths &#127754;</p>
            <p class="attribution">Data from <a href="https://alpaca.markets" target="_blank" rel="noopener">Alpaca</a></p>
        </div>
    </footer>

    <script src="app.js"></script>
</body>
</html>
```

**Step 2: Commit**

```bash
git add public_dashboard/index.html
git commit -m "feat: add index.html for public dashboard"
```

---

### Task 2: Create styles.css

Dark ocean-depth theme with teal accents. Responsive layout. Green/red P&L coloring.

**Files:**
- Create: `public_dashboard/styles.css`

**Step 1: Create the CSS file**

```css
/* === Reset & Base === */
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

:root {
  --bg-deep: #0a1628;
  --bg-card: #132240;
  --bg-card-hover: #1a2d52;
  --accent: #00d4aa;
  --accent-dim: rgba(0, 212, 170, 0.15);
  --text: #e2e8f0;
  --text-dim: #8892a4;
  --gain: #00d4aa;
  --loss: #ff6b6b;
  --border: #1e3a5f;
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

.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

/* === Header === */
header {
  border-bottom: 2px solid var(--border);
  padding: 2rem 0 1.5rem;
  background: linear-gradient(180deg, #0d1f38 0%, var(--bg-deep) 100%);
}

header h1 {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.02em;
}

header .subtitle {
  color: var(--text-dim);
  font-size: 0.875rem;
  margin-top: 0.25rem;
}

/* === Summary Cards === */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.card-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
}

.card-value {
  font-size: 1.5rem;
  font-weight: 700;
  font-family: var(--font-mono);
}

/* === Panels === */
.panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 1.5rem;
  overflow: hidden;
}

.panel h2 {
  font-size: 1rem;
  font-weight: 600;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}

.chart-wrap {
  padding: 1.25rem;
}

/* === Tables === */
.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

thead th {
  text-align: left;
  padding: 0.75rem 1.25rem;
  color: var(--text-dim);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 500;
  border-bottom: 1px solid var(--border);
}

thead th.num {
  text-align: right;
}

tbody td {
  padding: 0.75rem 1.25rem;
  border-bottom: 1px solid rgba(30, 58, 95, 0.5);
  vertical-align: top;
}

tbody td.num {
  text-align: right;
  font-family: var(--font-mono);
}

tbody tr:hover {
  background: var(--bg-card-hover);
}

/* === Action Badges === */
.badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.badge-buy {
  background: rgba(0, 212, 170, 0.15);
  color: var(--gain);
}

.badge-sell {
  background: rgba(255, 107, 107, 0.15);
  color: var(--loss);
}

.badge-hold {
  background: rgba(136, 146, 164, 0.15);
  color: var(--text-dim);
}

/* === Thesis Cards === */
.thesis-card {
  padding: 1.25rem;
  border-bottom: 1px solid rgba(30, 58, 95, 0.5);
}

.thesis-card:last-child {
  border-bottom: none;
}

.thesis-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}

.thesis-ticker {
  font-weight: 700;
  font-size: 1rem;
  font-family: var(--font-mono);
  color: var(--accent);
}

.thesis-direction {
  font-size: 0.75rem;
  text-transform: uppercase;
  font-weight: 600;
}

.thesis-direction.long {
  color: var(--gain);
}

.thesis-direction.short {
  color: var(--loss);
}

.thesis-confidence {
  font-size: 0.7rem;
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  background: var(--accent-dim);
  color: var(--accent);
  text-transform: uppercase;
}

.thesis-body {
  color: var(--text-dim);
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.thesis-triggers {
  font-size: 0.75rem;
  color: var(--text-dim);
  font-family: var(--font-mono);
}

/* === P&L Coloring === */
.gain {
  color: var(--gain);
}

.loss {
  color: var(--loss);
}

/* === Empty States === */
.empty-state {
  padding: 2rem 1.25rem;
  text-align: center;
  color: var(--text-dim);
  font-size: 0.875rem;
}

/* === Footer === */
footer {
  border-top: 2px solid var(--border);
  padding: 2rem 0;
  margin-top: 2rem;
  text-align: center;
}

footer p {
  color: var(--text-dim);
  font-size: 0.875rem;
}

footer .attribution {
  margin-top: 0.25rem;
  font-size: 0.75rem;
}

footer a {
  color: var(--accent);
  text-decoration: none;
}

footer a:hover {
  text-decoration: underline;
}

/* === Reasoning tooltip === */
.reasoning-cell {
  max-width: 250px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: help;
}

/* === Order ID === */
.order-id {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-dim);
}

/* === Responsive === */
@media (max-width: 640px) {
  .card-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .card-value {
    font-size: 1.2rem;
  }

  header h1 {
    font-size: 1.35rem;
  }
}
```

**Step 2: Commit**

```bash
git add public_dashboard/styles.css
git commit -m "feat: add styles.css for public dashboard"
```

---

### Task 3: Create app.js

Client-side JS that fetches all JSON data files and renders every section. Includes Chart.js equity curve config and helper formatters.

**Files:**
- Create: `public_dashboard/app.js`

**Step 1: Create the JS file**

```javascript
"use strict";

// === Helpers ===

function formatCurrency(n) {
  if (n == null) return "—";
  return "$" + Number(n).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatPct(n) {
  if (n == null) return "—";
  var val = Number(n);
  var sign = val >= 0 ? "+" : "";
  return sign + val.toFixed(2) + "%";
}

function pnlClass(n) {
  if (n == null || Number(n) === 0) return "";
  return Number(n) >= 0 ? "gain" : "loss";
}

function truncate(s, max) {
  if (!s) return "—";
  return s.length > max ? s.slice(0, max) + "..." : s;
}

function shortOrderId(id) {
  if (!id) return "—";
  return id.length > 12 ? id.slice(0, 8) + "..." : id;
}

// === Data Fetching ===

async function fetchJSON(file) {
  var resp = await fetch("data/" + file);
  if (!resp.ok) return null;
  return resp.json();
}

async function fetchAllData() {
  var [summary, snapshots, positions, decisions, theses] = await Promise.all([
    fetchJSON("summary.json"),
    fetchJSON("snapshots.json"),
    fetchJSON("positions.json"),
    fetchJSON("decisions.json"),
    fetchJSON("theses.json"),
  ]);
  return { summary: summary, snapshots: snapshots, positions: positions, decisions: decisions, theses: theses };
}

// === Renderers ===

function renderSummary(s) {
  if (!s) return;

  document.getElementById("last-updated").textContent =
    s.last_updated ? "Last updated " + s.last_updated : "";

  document.getElementById("portfolio-value").textContent = formatCurrency(s.portfolio_value);

  var dailyEl = document.getElementById("daily-pnl");
  if (s.daily_pnl != null) {
    dailyEl.textContent = formatCurrency(s.daily_pnl) + " (" + formatPct(s.daily_pnl_pct) + ")";
    dailyEl.className = "card-value " + pnlClass(s.daily_pnl);
  }

  var totalEl = document.getElementById("total-return");
  if (s.total_pnl != null) {
    totalEl.textContent = formatCurrency(s.total_pnl) + " (" + formatPct(s.total_pnl_pct) + ")";
    totalEl.className = "card-value " + pnlClass(s.total_pnl);
  }

  document.getElementById("positions-count").textContent =
    s.positions_count != null ? s.positions_count : "—";

  document.getElementById("cash-value").textContent = formatCurrency(s.cash);
}

function renderEquityCurve(snapshots) {
  var canvas = document.getElementById("equity-chart");
  var emptyMsg = document.getElementById("chart-empty");

  if (!snapshots || snapshots.length === 0) {
    canvas.style.display = "none";
    emptyMsg.style.display = "block";
    return;
  }

  var labels = snapshots.map(function (s) { return s.date; });
  var values = snapshots.map(function (s) { return s.portfolio_value; });

  new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        label: "Portfolio Value",
        data: values,
        borderColor: "#00d4aa",
        backgroundColor: "rgba(0, 212, 170, 0.08)",
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHitRadius: 8,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              return formatCurrency(ctx.parsed.y);
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8892a4", maxTicksLimit: 8 },
          grid: { color: "rgba(30, 58, 95, 0.4)" },
        },
        y: {
          beginAtZero: false,
          ticks: {
            color: "#8892a4",
            callback: function (v) { return "$" + v.toLocaleString(); },
          },
          grid: { color: "rgba(30, 58, 95, 0.4)" },
        },
      },
    },
  });
}

function renderPositions(positions) {
  var tbody = document.querySelector("#positions-table tbody");
  var emptyMsg = document.getElementById("positions-empty");

  if (!positions || positions.length === 0) {
    document.getElementById("positions-table").style.display = "none";
    emptyMsg.style.display = "block";
    return;
  }

  positions.forEach(function (p) {
    var tr = document.createElement("tr");
    tr.innerHTML =
      "<td><strong>" + p.ticker + "</strong></td>" +
      '<td class="num">' + p.shares + "</td>" +
      '<td class="num">' + formatCurrency(p.avg_cost) + "</td>";
    tbody.appendChild(tr);
  });
}

function renderDecisions(decisions) {
  var tbody = document.querySelector("#decisions-table tbody");
  var emptyMsg = document.getElementById("decisions-empty");

  if (!decisions || decisions.length === 0) {
    document.getElementById("decisions-table").style.display = "none";
    emptyMsg.style.display = "block";
    return;
  }

  decisions.forEach(function (d) {
    var badgeClass = "badge badge-" + (d.action || "hold");
    var tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + (d.date || "—") + "</td>" +
      "<td><strong>" + (d.ticker || "—") + "</strong></td>" +
      '<td><span class="' + badgeClass + '">' + (d.action || "—") + "</span></td>" +
      '<td class="num">' + (d.quantity || "—") + "</td>" +
      '<td class="reasoning-cell" title="' + (d.reasoning || "").replace(/"/g, "&quot;") + '">' + truncate(d.reasoning, 60) + "</td>" +
      '<td class="num"><span class="order-id">' + shortOrderId(d.order_id) + "</span></td>";
    tbody.appendChild(tr);
  });
}

function renderTheses(theses) {
  var container = document.getElementById("theses-list");
  var emptyMsg = document.getElementById("theses-empty");

  if (!theses || theses.length === 0) {
    emptyMsg.style.display = "block";
    return;
  }

  theses.forEach(function (t) {
    var card = document.createElement("div");
    card.className = "thesis-card";
    card.innerHTML =
      '<div class="thesis-header">' +
        '<span class="thesis-ticker">' + t.ticker + "</span>" +
        '<span class="thesis-direction ' + (t.direction || "") + '">' + (t.direction || "") + "</span>" +
        '<span class="thesis-confidence">' + (t.confidence || "") + "</span>" +
      "</div>" +
      '<p class="thesis-body">' + (t.thesis || "") + "</p>" +
      '<div class="thesis-triggers">' +
        "Entry: " + (t.entry_trigger || "—") + " &nbsp;|&nbsp; Exit: " + (t.exit_trigger || "—") +
      "</div>";
    container.appendChild(card);
  });
}

// === Init ===

document.addEventListener("DOMContentLoaded", function () {
  fetchAllData().then(function (data) {
    renderSummary(data.summary);
    renderEquityCurve(data.snapshots);
    renderPositions(data.positions);
    renderDecisions(data.decisions);
    renderTheses(data.theses);
  }).catch(function (err) {
    console.error("Failed to load dashboard data:", err);
    document.getElementById("last-updated").textContent = "Failed to load data";
  });
});
```

**Step 2: Commit**

```bash
git add public_dashboard/app.js
git commit -m "feat: add app.js for public dashboard"
```

---

### Task 4: Add sample data for local testing

Create sample JSON files so the dashboard can be tested locally without a database.

**Files:**
- Create: `public_dashboard/data/summary.json`
- Create: `public_dashboard/data/snapshots.json`
- Create: `public_dashboard/data/positions.json`
- Create: `public_dashboard/data/decisions.json`
- Create: `public_dashboard/data/theses.json`

**Step 1: Create sample data files**

`summary.json`:
```json
{
  "portfolio_value": 104523.47,
  "cash": 42180.22,
  "invested": 62343.25,
  "positions_count": 4,
  "last_updated": "2026-02-17",
  "daily_pnl": 1247.83,
  "daily_pnl_pct": 1.21,
  "total_pnl": 4523.47,
  "total_pnl_pct": 4.52,
  "inception_date": "2026-01-15"
}
```

`snapshots.json` — 10 sample data points covering a date range.

`positions.json`:
```json
[
  {"ticker": "AAPL", "shares": 25, "avg_cost": 189.50, "updated_at": "2026-02-17T10:30:00"},
  {"ticker": "GOOGL", "shares": 10, "avg_cost": 175.20, "updated_at": "2026-02-16T14:00:00"},
  {"ticker": "MSFT", "shares": 15, "avg_cost": 420.75, "updated_at": "2026-02-15T11:00:00"},
  {"ticker": "NVDA", "shares": 8, "avg_cost": 890.30, "updated_at": "2026-02-14T09:30:00"}
]
```

`decisions.json`:
```json
[
  {"id": 5, "date": "2026-02-17", "ticker": "AAPL", "action": "buy", "quantity": 5, "price": 192.50, "reasoning": "Strong Q1 earnings beat with iPhone revenue up 12% YoY. Services segment continues to grow.", "outcome_7d": null, "outcome_30d": null, "order_id": "abc-123-def-456"},
  {"id": 4, "date": "2026-02-14", "ticker": "TSLA", "action": "sell", "quantity": 10, "price": 245.00, "reasoning": "Taking profits after 15% run-up. Thesis target reached.", "outcome_7d": 2.3, "outcome_30d": null, "order_id": "ghi-789-jkl-012"}
]
```

`theses.json`:
```json
[
  {"id": 1, "ticker": "AAPL", "direction": "long", "confidence": "high", "thesis": "iPhone 17 cycle and services growth should drive earnings above consensus. AI integration in iOS could expand TAM.", "entry_trigger": "Buy below $195", "exit_trigger": "Sell above $220 or if services growth decelerates", "created_at": "2026-02-10T09:00:00"},
  {"id": 2, "ticker": "NVDA", "direction": "long", "confidence": "medium", "thesis": "Data center demand remains strong but valuation stretched. Position sizing reflects risk.", "entry_trigger": "Add below $850", "exit_trigger": "Trim above $950", "created_at": "2026-02-08T14:00:00"}
]
```

**Step 2: Verify locally**

Run: `cd public_dashboard && python3 -m http.server 8080`
Open: `http://localhost:8080` in browser.
Expected: All sections render with sample data. Equity curve chart visible. Tables populated. Theses displayed as cards.

**Step 3: Commit**

```bash
git add public_dashboard/data/
git commit -m "feat: add sample data for local dashboard testing"
```

---

### Task 5: Add .gitignore for GitHub Pages repo data directory

The `public_dashboard/data/` directory contains sample data for testing but should not be confused with production data. Add a note and ensure the production GitHub Pages repo will have its data/ managed by the pipeline.

**Files:**
- Create: `public_dashboard/README.md`

**Step 1: Create a short README**

```markdown
# Bikini Bottom Capital - Public Dashboard

Static site for GitHub Pages. Copy these files to your `*.github.io` repo.

The `data/` directory here contains **sample data** for local testing.
In production, `data/*.json` files are regenerated by the session pipeline
(`v2/dashboard_publish.py`) on each run.

## Local Testing

```bash
python3 -m http.server 8080
# Open http://localhost:8080
```
```

**Step 2: Commit**

```bash
git add public_dashboard/README.md
git commit -m "docs: add README for public dashboard assets"
```
