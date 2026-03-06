# S&P 500 Benchmark Comparison — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an S&P 500 (SPY) benchmark overlay to the public dashboard equity curve, normalized to % return.

**Architecture:** Fetch SPY daily bars from Alpaca at publish time, save to `benchmark.json`, render as a second dataset on the Chart.js equity curve. Both lines normalized to % return from inception.

**Tech Stack:** Python (alpaca-py SDK), Chart.js 4.4.7, vanilla JS

---

### Task 1: Backend — fetch_spy_benchmark()

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

**Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_publish.py`:

```python
from v2.dashboard_publish import fetch_spy_benchmark


class TestFetchSpyBenchmark:
    @patch("v2.dashboard_publish.StockHistoricalDataClient")
    def test_returns_spy_bars_for_date_range(self, mock_client_cls):
        """Fetches SPY daily bars and returns [{date, close}, ...]."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Simulate Alpaca bar objects
        bar1 = MagicMock()
        bar1.timestamp = datetime(2025, 6, 14, 4, 0)
        bar1.close = 540.50
        bar2 = MagicMock()
        bar2.timestamp = datetime(2025, 6, 15, 4, 0)
        bar2.close = 542.00

        mock_bars = MagicMock()
        mock_bars.__getitem__ = MagicMock(return_value=[bar1, bar2])
        mock_client.get_stock_bars.return_value = mock_bars

        result = fetch_spy_benchmark(date(2025, 6, 14), date(2025, 6, 15))

        assert len(result) == 2
        assert result[0] == {"date": "2025-06-14", "close": 540.50}
        assert result[1] == {"date": "2025-06-15", "close": 542.00}

    @patch("v2.dashboard_publish.StockHistoricalDataClient")
    def test_returns_empty_list_on_api_error(self, mock_client_cls):
        """Returns [] if Alpaca API call fails."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_stock_bars.side_effect = Exception("API down")

        result = fetch_spy_benchmark(date(2025, 6, 14), date(2025, 6, 15))

        assert result == []

    @patch("v2.dashboard_publish.StockHistoricalDataClient")
    def test_returns_empty_list_when_no_bars(self, mock_client_cls):
        """Returns [] if no bars returned."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_bars = MagicMock()
        mock_bars.__getitem__ = MagicMock(return_value=[])
        mock_client.get_stock_bars.return_value = mock_bars

        result = fetch_spy_benchmark(date(2025, 6, 14), date(2025, 6, 15))

        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestFetchSpyBenchmark -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_spy_benchmark'`

**Step 3: Write minimal implementation**

Add to `v2/dashboard_publish.py`, at the top with the other imports:

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
```

Add function after `_build_summary`:

```python
def fetch_spy_benchmark(start_date: date, end_date: date) -> list[dict]:
    """Fetch SPY daily bars from Alpaca for benchmark comparison.

    Returns list of {date, close} dicts, or [] on error.
    """
    try:
        api_key = os.environ.get("APCA_API_KEY_ID")
        secret_key = os.environ.get("APCA_API_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret_key)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.max.time()),
        )
        bars = client.get_stock_bars(request)
        spy_bars = list(bars["SPY"])

        if not spy_bars:
            return []

        return [
            {"date": bar.timestamp.strftime("%Y-%m-%d"), "close": bar.close}
            for bar in spy_bars
        ]
    except Exception:
        logger.warning("Failed to fetch SPY benchmark data", exc_info=True)
        return []
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestFetchSpyBenchmark -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat: add fetch_spy_benchmark() for S&P 500 comparison data"
```

---

### Task 2: Backend — wire benchmark into gather + write pipeline

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

**Step 1: Write failing tests**

Add to `TestGatherDashboardData`:

```python
    @patch("v2.dashboard_publish.fetch_spy_benchmark")
    def test_includes_benchmark_key(self, mock_benchmark, mock_db):
        """gather_dashboard_data includes 'benchmark' key from fetch_spy_benchmark."""
        mock_benchmark.return_value = [{"date": "2025-06-15", "close": 540.0}]
        session_date = date(2025, 6, 15)

        mock_db.fetchall.side_effect = [
            [{"date": date(2025, 6, 14), "portfolio_value": Decimal("99000"),
              "cash": Decimal("49000"), "buying_power": Decimal("49000")},
             {"date": date(2025, 6, 15), "portfolio_value": Decimal("100000"),
              "cash": Decimal("50000"), "buying_power": Decimal("50000")}],
            [], [], [],
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"),
             "long_market_value": Decimal("50000")},
            {"portfolio_value": Decimal("90000"), "date": date(2025, 1, 1)},
            {"portfolio_value": Decimal("99000")},
        ]

        result = gather_dashboard_data(session_date)

        assert "benchmark" in result
        assert result["benchmark"] == [{"date": "2025-06-15", "close": 540.0}]
        mock_benchmark.assert_called_once()
```

Add to `TestWriteJsonFiles`:

```python
    def test_writes_benchmark_file(self, tmp_path):
        """benchmark.json is written when benchmark key present."""
        data = self._sample_data()
        data["benchmark"] = [{"date": "2025-06-15", "close": 540.0}]
        result = write_json_files(data, str(tmp_path))

        benchmark_path = tmp_path / "data" / "benchmark.json"
        assert benchmark_path.exists()
        with open(benchmark_path) as f:
            content = json.load(f)
        assert content == [{"date": "2025-06-15", "close": 540.0}]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestGatherDashboardData::test_includes_benchmark_key tests/v2/test_dashboard_publish.py::TestWriteJsonFiles::test_writes_benchmark_file -v`
Expected: FAIL

**Step 3: Implement**

In `gather_dashboard_data`, after the DB queries block, before the return, add:

```python
    # Fetch SPY benchmark for same date range as snapshots
    benchmark = []
    if snapshots:
        first_date = snapshots[0]["date"]
        benchmark = fetch_spy_benchmark(first_date, session_date)

    return {
        "summary": summary,
        "snapshots": [dict(r) for r in snapshots],
        "positions": [dict(r) for r in positions],
        "decisions": [dict(r) for r in decisions],
        "theses": [dict(r) for r in theses],
        "benchmark": benchmark,
    }
```

In `write_json_files`, update the keys loop:

```python
    for key in ("summary", "snapshots", "positions", "decisions", "theses", "benchmark"):
        if key not in data:
            continue
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py -v`
Expected: ALL PASS

**Step 5: Update existing tests that check keys or file counts**

The `test_returns_all_sections` test checks `set(result.keys())` — update it to include `"benchmark"`. The `test_writes_all_files` test checks `len(result) == 5` — update to 6 and add `"benchmark"` to the loop. The `test_query_count` test needs `fetch_spy_benchmark` patched out since it won't have snapshot data to trigger it.

**Step 6: Run full test suite**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat: wire SPY benchmark into gather/write pipeline"
```

---

### Task 3: Frontend — update equity curve to % return with SPY overlay

**Files:**
- Modify: `public_dashboard/app.js`
- Modify: `public_dashboard/index.html`

**Step 1: Update data fetching in app.js**

In `fetchAllData()`, add `benchmark.json`:

```javascript
async function fetchAllData() {
  var [summary, snapshots, positions, decisions, theses, benchmark] = await Promise.all([
    fetchJSON("summary.json"),
    fetchJSON("snapshots.json"),
    fetchJSON("positions.json"),
    fetchJSON("decisions.json"),
    fetchJSON("theses.json"),
    fetchJSON("benchmark.json"),
  ]);
  return { summary: summary, snapshots: snapshots, positions: positions,
           decisions: decisions, theses: theses, benchmark: benchmark };
}
```

**Step 2: Rewrite renderEquityCurve to normalize % return and add SPY**

Replace the `renderEquityCurve` function:

```javascript
function renderEquityCurve(snapshots, benchmark) {
  var canvas = document.getElementById("equity-chart");
  var emptyMsg = document.getElementById("chart-empty");

  if (!snapshots || snapshots.length === 0) {
    canvas.style.display = "none";
    emptyMsg.style.display = "block";
    return;
  }

  var labels = snapshots.map(function (s) { return s.date; });

  // Normalize portfolio to % return
  var baseValue = snapshots[0].portfolio_value;
  var portfolioReturns = snapshots.map(function (s) {
    return ((s.portfolio_value - baseValue) / baseValue) * 100;
  });

  var datasets = [{
    label: "Portfolio",
    data: portfolioReturns,
    borderColor: "#00d4aa",
    backgroundColor: "rgba(0, 212, 170, 0.08)",
    fill: true,
    tension: 0.3,
    pointRadius: 0,
    pointHitRadius: 8,
    borderWidth: 2,
  }];

  // Add SPY benchmark if available
  if (benchmark && benchmark.length > 0) {
    // Build a date->close map for SPY
    var spyMap = {};
    benchmark.forEach(function (b) { spyMap[b.date] = b.close; });

    // Find SPY base value matching the first snapshot date
    var spyBase = spyMap[labels[0]];

    if (spyBase) {
      var spyReturns = labels.map(function (date) {
        var close = spyMap[date];
        if (close == null) return null;
        return ((close - spyBase) / spyBase) * 100;
      });

      datasets.push({
        label: "S&P 500",
        data: spyReturns,
        borderColor: "#5a6a7a",
        borderDash: [6, 3],
        backgroundColor: "transparent",
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        pointHitRadius: 8,
        borderWidth: 2,
      });
    }
  }

  new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets,
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: datasets.length > 1, labels: { color: "#8892a4" } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              return ctx.dataset.label + ": " + formatPct(ctx.parsed.y);
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
          ticks: {
            color: "#8892a4",
            callback: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(1) + "%"; },
          },
          grid: { color: "rgba(30, 58, 95, 0.4)" },
        },
      },
    },
  });
}
```

**Step 3: Update the renderEquityCurve call in init**

```javascript
renderEquityCurve(data.snapshots, data.benchmark);
```

**Step 4: Update index.html**

Change the equity curve section title:

```html
<h2>Performance vs S&P 500</h2>
```

**Step 5: Manually verify with sample data**

Open `public_dashboard/index.html` in a browser. The chart should show % return on Y-axis. If `benchmark.json` doesn't exist yet, it gracefully renders just the portfolio line.

**Step 6: Commit**

```bash
git add public_dashboard/app.js public_dashboard/index.html
git commit -m "feat: equity curve shows % return with S&P 500 overlay"
```

---

### Task 4: Add "vs S&P" summary card

**Files:**
- Modify: `public_dashboard/index.html`
- Modify: `public_dashboard/app.js`

**Step 1: Add the card to index.html**

After the Cash card in the `card-grid` section, add:

```html
<div class="card">
    <span class="card-label"><span class="card-icon">&#128200;</span> vs S&P</span>
    <span class="card-value" id="vs-sp">—</span>
</div>
```

**Step 2: Compute and render alpha in app.js**

Add to the `renderSummary` function, at the end. Pass `benchmark` and `snapshots` as additional params:

Update `renderSummary` signature to `renderSummary(s, snapshots, benchmark)`.

Add at the end of `renderSummary`:

```javascript
  // vs S&P: portfolio return minus SPY return
  var vsSp = document.getElementById("vs-sp");
  if (snapshots && snapshots.length > 1 && benchmark && benchmark.length > 0) {
    var portfolioBase = snapshots[0].portfolio_value;
    var portfolioNow = snapshots[snapshots.length - 1].portfolio_value;
    var portfolioReturn = ((portfolioNow - portfolioBase) / portfolioBase) * 100;

    var spyMap = {};
    benchmark.forEach(function (b) { spyMap[b.date] = b.close; });
    var spyStart = spyMap[snapshots[0].date];
    var spyEnd = spyMap[snapshots[snapshots.length - 1].date];

    if (spyStart && spyEnd) {
      var spyReturn = ((spyEnd - spyStart) / spyStart) * 100;
      var alpha = portfolioReturn - spyReturn;
      vsSp.textContent = formatPct(alpha);
      vsSp.className = "card-value " + pnlClass(alpha);
    }
  }
```

**Step 3: Update the renderSummary call in init**

```javascript
renderSummary(data.summary, data.snapshots, data.benchmark);
```

**Step 4: Commit**

```bash
git add public_dashboard/index.html public_dashboard/app.js
git commit -m "feat: add vs S&P summary card showing alpha"
```

---

### Task 5: Add sample benchmark.json

**Files:**
- Create: `public_dashboard/data/benchmark.json`

**Step 1: Create sample data**

Create `public_dashboard/data/benchmark.json` with SPY data matching the sample snapshot dates:

```json
[
  {"date": "2026-02-03", "close": 601.25},
  {"date": "2026-02-04", "close": 603.10},
  {"date": "2026-02-05", "close": 598.40},
  {"date": "2026-02-06", "close": 600.85},
  {"date": "2026-02-07", "close": 605.20},
  {"date": "2026-02-10", "close": 603.50},
  {"date": "2026-02-11", "close": 607.15},
  {"date": "2026-02-12", "close": 605.80},
  {"date": "2026-02-13", "close": 609.25},
  {"date": "2026-02-14", "close": 610.40},
  {"date": "2026-02-17", "close": 612.75}
]
```

**Step 2: Commit**

```bash
git add public_dashboard/data/benchmark.json
git commit -m "feat: add sample benchmark.json for local dev"
```

---

### Task 6: Run full test suite and verify

**Step 1: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run dashboard tests specifically**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py -v`
Expected: ALL PASS

**Step 3: Final commit if any fixups needed**
