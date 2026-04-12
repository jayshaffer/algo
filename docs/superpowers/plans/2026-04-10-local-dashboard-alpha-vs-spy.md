# Local Dashboard Alpha vs SPY — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SPY benchmark comparison to the local Flask dashboard: a "vs S&P" alpha card on the portfolio page, an SPY overlay on the equity chart, and a dedicated % return comparison chart on the performance page.

**Architecture:** New `dashboard/benchmark.py` module handles Alpaca SPY data fetching (with in-memory TTL cache) and alpha computation. Routes in `dashboard/app.py` wire these into the existing portfolio and performance pages. Templates updated with Chart.js datasets. All Alpaca I/O is isolated behind `get_spy_benchmark()` for clean mocking.

**Tech Stack:** Python 3.12, Flask, Chart.js (already in base.html), alpaca-py (StockHistoricalDataClient), pytest

---

### Task 1: Add alpaca-py dependency

**Files:**
- Modify: `dashboard/requirements.txt`

- [ ] **Step 1: Add alpaca-py to requirements**

In `dashboard/requirements.txt`, add:

```
alpaca-py>=0.30.0
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/requirements.txt
git commit -m "build: add alpaca-py to dashboard requirements"
```

---

### Task 2: Write compute_alpha with tests (TDD)

**Files:**
- Create: `dashboard/benchmark.py`
- Create: `tests/test_dashboard_benchmark.py`

- [ ] **Step 1: Write failing tests for compute_alpha**

Create `tests/test_dashboard_benchmark.py`:

```python
"""Tests for dashboard/benchmark.py — SPY benchmark and alpha computation."""

from datetime import date
from decimal import Decimal

import pytest

from dashboard.benchmark import compute_alpha


class TestComputeAlpha:
    """Tests for compute_alpha()."""

    def test_happy_path_aligned_dates(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("101000")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("102000")},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-03", "close": 505.0},
            {"date": "2026-01-06", "close": 510.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        assert result is not None
        # portfolio: (102000 - 100000) / 100000 * 100 = 2.0%
        assert result["portfolio_return"] == pytest.approx(2.0)
        # spy: (510 - 500) / 500 * 100 = 2.0%
        assert result["spy_return"] == pytest.approx(2.0)
        assert result["alpha"] == pytest.approx(0.0)

    def test_positive_alpha(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("105000")},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 505.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        # portfolio: 5%, spy: 1%
        assert result["alpha"] == pytest.approx(4.0)

    def test_negative_alpha(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("99000")},
        ]
        benchmark = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 510.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        # portfolio: -1%, spy: 2%
        assert result["alpha"] == pytest.approx(-3.0)

    def test_misaligned_dates_weekend(self):
        """Snapshot starts on Monday, SPY has no weekend data."""
        snapshots = [
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("100000")},  # Sat
            {"date": date(2026, 1, 5), "portfolio_value": Decimal("100000")},  # Mon
            {"date": date(2026, 1, 6), "portfolio_value": Decimal("102000")},  # Tue
        ]
        benchmark = [
            {"date": "2026-01-05", "close": 500.0},
            {"date": "2026-01-06", "close": 510.0},
        ]
        result = compute_alpha(snapshots, benchmark)
        assert result is not None
        # Uses first overlapping date (Jan 5): portfolio 100000->102000 = 2%
        # SPY 500->510 = 2%
        assert result["alpha"] == pytest.approx(0.0)

    def test_empty_benchmark_returns_none(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("101000")},
        ]
        assert compute_alpha(snapshots, []) is None

    def test_single_snapshot_returns_none(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
        ]
        benchmark = [{"date": "2026-01-02", "close": 500.0}]
        assert compute_alpha(snapshots, benchmark) is None

    def test_no_overlapping_dates_returns_none(self):
        snapshots = [
            {"date": date(2026, 1, 2), "portfolio_value": Decimal("100000")},
            {"date": date(2026, 1, 3), "portfolio_value": Decimal("101000")},
        ]
        benchmark = [
            {"date": "2026-01-10", "close": 500.0},
            {"date": "2026-01-11", "close": 505.0},
        ]
        assert compute_alpha(snapshots, benchmark) is None

    def test_none_inputs(self):
        assert compute_alpha(None, [{"date": "2026-01-02", "close": 500.0}]) is None
        assert compute_alpha([], None) is None
        assert compute_alpha(None, None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.benchmark'`

- [ ] **Step 3: Write minimal compute_alpha implementation**

Create `dashboard/benchmark.py`:

```python
"""SPY benchmark data fetching and alpha computation for the local dashboard."""

import logging
import os
import time
from datetime import date, datetime

logger = logging.getLogger(__name__)


def compute_alpha(snapshots, benchmark):
    """Compute portfolio alpha vs SPY benchmark.

    Returns {"portfolio_return", "spy_return", "alpha"} or None.
    """
    if not snapshots or len(snapshots) < 2 or not benchmark:
        return None

    spy_map = {b["date"]: b["close"] for b in benchmark}

    spy_start = None
    port_start = None
    for snap in snapshots:
        date_str = str(snap["date"])
        if date_str in spy_map:
            spy_start = spy_map[date_str]
            port_start = float(snap["portfolio_value"])
            break

    spy_end = None
    port_end = None
    for snap in reversed(snapshots):
        date_str = str(snap["date"])
        if date_str in spy_map:
            spy_end = spy_map[date_str]
            port_end = float(snap["portfolio_value"])
            break

    if spy_start is None or spy_end is None or spy_start == spy_end:
        return None
    if port_start is None or port_start == 0:
        return None

    portfolio_return = ((port_end - port_start) / port_start) * 100
    spy_return = ((spy_end - spy_start) / spy_start) * 100
    alpha = portfolio_return - spy_return

    return {
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
        "alpha": alpha,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard_benchmark.py::TestComputeAlpha -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/benchmark.py tests/test_dashboard_benchmark.py
git commit -m "feat: add compute_alpha for SPY benchmark comparison"
```

---

### Task 3: Write get_spy_benchmark with tests (TDD)

**Files:**
- Modify: `dashboard/benchmark.py`
- Modify: `tests/test_dashboard_benchmark.py`

- [ ] **Step 1: Write failing tests for get_spy_benchmark**

Append to `tests/test_dashboard_benchmark.py`:

```python
from unittest.mock import patch, MagicMock

from dashboard.benchmark import get_spy_benchmark, _clear_cache


class TestGetSpyBenchmark:
    """Tests for get_spy_benchmark() with in-memory TTL cache."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        _clear_cache()
        yield
        _clear_cache()

    def _make_mock_bar(self, dt_str, close):
        bar = MagicMock()
        bar.timestamp = datetime.strptime(dt_str, "%Y-%m-%d")
        bar.close = close
        return bar

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_fetches_from_alpaca(self, mock_client_cls):
        bars = [
            self._make_mock_bar("2026-01-02", 500.0),
            self._make_mock_bar("2026-01-03", 505.0),
        ]
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": bars}
        mock_client_cls.return_value = mock_client

        result = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        assert len(result) == 2
        assert result[0] == {"date": "2026-01-02", "close": 500.0}
        assert result[1] == {"date": "2026-01-03", "close": 505.0}
        mock_client.get_stock_bars.assert_called_once()

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_cache_hit_skips_alpaca(self, mock_client_cls):
        bars = [self._make_mock_bar("2026-01-02", 500.0)]
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": bars}
        mock_client_cls.return_value = mock_client

        result1 = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))
        result2 = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        assert result1 == result2
        assert mock_client.get_stock_bars.call_count == 1

    @patch("dashboard.benchmark.time")
    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_cache_expires_after_ttl(self, mock_client_cls, mock_time):
        bars = [self._make_mock_bar("2026-01-02", 500.0)]
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": bars}
        mock_client_cls.return_value = mock_client

        mock_time.time.return_value = 1000.0
        get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        mock_time.time.return_value = 1000.0 + 901  # Past 900s TTL
        get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))

        assert mock_client.get_stock_bars.call_count == 2

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_alpaca_error_returns_empty(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("connection refused")
        result = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))
        assert result == []

    @patch("dashboard.benchmark.StockHistoricalDataClient")
    def test_empty_bars_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = {"SPY": []}
        mock_client_cls.return_value = mock_client
        result = get_spy_benchmark(date(2026, 1, 2), date(2026, 1, 3))
        assert result == []
```

Note: also needs `from datetime import datetime` added to the imports at the top of the test file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dashboard_benchmark.py::TestGetSpyBenchmark -v`
Expected: FAIL — `ImportError: cannot import name 'get_spy_benchmark'`

- [ ] **Step 3: Implement get_spy_benchmark with cache**

Add to `dashboard/benchmark.py` (after the existing imports and before `compute_alpha`):

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

_TTL_SECONDS = 900  # 15 minutes
_cache: dict[tuple[date, date], tuple[float, list[dict]]] = {}


def _clear_cache():
    _cache.clear()


def get_spy_benchmark(start: date, end: date) -> list[dict]:
    """Fetch SPY daily bars from Alpaca with in-memory TTL cache.

    Returns list of {"date": "YYYY-MM-DD", "close": float}, or [] on error.
    """
    key = (start, end)
    now = time.time()

    cached = _cache.get(key)
    if cached and cached[0] > now:
        return cached[1]

    try:
        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret_key)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.max.time()),
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request)
        spy_bars = list(bars["SPY"])

        if not spy_bars:
            return []

        result = [
            {"date": bar.timestamp.strftime("%Y-%m-%d"), "close": bar.close}
            for bar in spy_bars
        ]
        _cache[key] = (now + _TTL_SECONDS, result)
        return result
    except Exception:
        logger.warning("Failed to fetch SPY benchmark data", exc_info=True)
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dashboard_benchmark.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/benchmark.py tests/test_dashboard_benchmark.py
git commit -m "feat: add get_spy_benchmark with in-memory TTL cache"
```

---

### Task 4: Wire benchmark into portfolio route and template

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/templates/portfolio.html`

- [ ] **Step 1: Update portfolio route to fetch benchmark and compute alpha**

In `dashboard/app.py`, add the import near the top (after the queries import block):

```python
from benchmark import get_spy_benchmark, compute_alpha
```

Replace the `portfolio()` function:

```python
@app.route("/")
def portfolio():
    """Portfolio overview page."""
    positions = get_positions()
    snapshot = get_latest_snapshot()
    playbook = get_today_playbook()
    open_orders = get_open_orders()

    equity_curve = get_equity_curve(days=90)
    benchmark_data = []
    alpha_stats = None
    if equity_curve:
        dates = [row["date"] for row in equity_curve]
        benchmark_data = get_spy_benchmark(dates[0], dates[-1])
        alpha_stats = compute_alpha(equity_curve, benchmark_data)

    return render_template(
        "portfolio.html",
        positions=positions,
        snapshot=snapshot,
        playbook=playbook,
        open_orders=open_orders,
        alpha_stats=alpha_stats,
    )
```

- [ ] **Step 2: Add vs S&P card to portfolio template**

In `dashboard/templates/portfolio.html`, change the grid from `md:grid-cols-4` to `md:grid-cols-5` and add the new card after the Long Market Value card (before the closing `</div>` of the grid):

```html
    <div class="bg-white rounded-lg shadow p-4">
        <div class="text-gray-500 text-sm">vs S&P 500</div>
        {% if alpha_stats %}
        <div class="text-2xl font-bold {% if alpha_stats.alpha > 0 %}text-green-600{% elif alpha_stats.alpha < 0 %}text-red-600{% endif %}">
            {{ "%+.2f"|format(alpha_stats.alpha) }}%
        </div>
        <div class="text-xs text-gray-400">Portfolio {{ "%.1f"|format(alpha_stats.portfolio_return) }}% vs SPY {{ "%.1f"|format(alpha_stats.spy_return) }}%</div>
        {% else %}
        <div class="text-2xl font-bold text-gray-400">&mdash;</div>
        {% endif %}
    </div>
```

- [ ] **Step 3: Run the test suite to verify nothing broke**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: All existing tests PASS. The portfolio route tests still work because `mock_queries.get_equity_curve` returns `[]` by default, so `alpha_stats` will be `None`.

Note: The route now imports from `benchmark`, which must be importable. Since the dashboard app runs from `/dashboard` directory where `from benchmark import ...` resolves, and the test file uses `sys.modules["queries"]` injection, we need to also patch `benchmark` for the dashboard route tests. Add to the `_reset_query_mocks` fixture in `tests/test_dashboard.py`:

At the top of `tests/test_dashboard.py`, add before the `from dashboard.app import app` line:

```python
mock_benchmark = MagicMock()
mock_benchmark.get_spy_benchmark.return_value = []
mock_benchmark.compute_alpha.return_value = None
sys.modules["benchmark"] = mock_benchmark
```

And in the `_reset_query_mocks` fixture, add after the existing resets:

```python
    mock_benchmark.get_spy_benchmark.return_value = []
    mock_benchmark.compute_alpha.return_value = None
```

- [ ] **Step 4: Run tests again**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py dashboard/templates/portfolio.html tests/test_dashboard.py
git commit -m "feat: add vs S&P alpha card to portfolio page"
```

---

### Task 5: Wire benchmark into performance route and equity chart overlay

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/templates/performance.html`

- [ ] **Step 1: Update performance route to pass benchmark data**

In `dashboard/app.py`, replace the `performance()` function:

```python
@app.route("/performance")
def performance():
    """Performance charts page."""
    equity_curve = get_equity_curve(days=90)
    metrics = get_performance_metrics(days=30)

    equity_data = [
        {
            "date": str(row["date"]),
            "portfolio_value": float(row["portfolio_value"]),
            "cash": float(row["cash"]),
            "buying_power": float(row["buying_power"]),
        }
        for row in equity_curve
    ] if equity_curve else []

    benchmark_data = []
    alpha_stats = None
    if equity_curve:
        dates = [row["date"] for row in equity_curve]
        benchmark_data = get_spy_benchmark(dates[0], dates[-1])
        alpha_stats = compute_alpha(equity_curve, benchmark_data)

    return render_template(
        "performance.html",
        equity_data=equity_data,
        metrics=metrics,
        benchmark_data=benchmark_data,
        alpha_stats=alpha_stats,
    )
```

- [ ] **Step 2: Add SPY overlay to equity chart in performance template**

In `dashboard/templates/performance.html`, replace the Performance Metrics grid (lines 10-33) to add a 5th card for alpha:

After the Return card (the last card in the grid), before the closing `</div>` of the grid, add:

```html
    <div class="bg-white rounded-lg shadow p-4">
        <div class="text-gray-500 text-sm">vs S&P 500</div>
        {% if alpha_stats %}
        <div class="text-xl font-bold {% if alpha_stats.alpha > 0 %}text-green-600{% elif alpha_stats.alpha < 0 %}text-red-600{% endif %}">
            {{ "%+.2f"|format(alpha_stats.alpha) }}%
        </div>
        {% else %}
        <div class="text-xl font-bold text-gray-400">&mdash;</div>
        {% endif %}
    </div>
```

Change the grid class from `grid-cols-2 md:grid-cols-4` to `grid-cols-2 md:grid-cols-5`.

- [ ] **Step 3: Add SPY dataset to existing equity chart and new benchmark chart**

In `dashboard/templates/performance.html`, add the benchmark chart canvas. After the "Cash vs Invested" section (before `{% endblock %}`), add:

```html
<!-- Benchmark: Portfolio vs S&P 500 (% Return) -->
<div class="bg-white rounded-lg shadow mt-6">
    <div class="p-4 border-b">
        <h2 class="text-lg font-semibold">Portfolio vs S&P 500 (% Return)</h2>
    </div>
    <div class="p-4">
        {% if equity_data and benchmark_data %}
        <canvas id="benchmarkChart" height="100"></canvas>
        {% else %}
        <p class="text-gray-500">Not enough data for benchmark comparison</p>
        {% endif %}
    </div>
</div>
```

Now replace the entire `{% block scripts %}` section with:

```html
{% block scripts %}
{% if equity_data %}
<script>
    const equityData = {{ equity_data|tojson }};
    const benchmarkData = {{ benchmark_data|tojson }};

    // Build SPY date->close lookup
    const spyMap = {};
    benchmarkData.forEach(b => { spyMap[b.date] = b.close; });

    const labels = equityData.map(d => d.date);

    // --- Equity Curve with SPY overlay ---
    const equityDatasets = [{
        label: 'Portfolio Value',
        data: equityData.map(d => d.portfolio_value),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.1
    }];

    // Normalize SPY to portfolio's starting value
    const spyBase = spyMap[labels[0]];
    if (spyBase && benchmarkData.length > 0) {
        const baseValue = equityData[0].portfolio_value;
        const spyNormalized = labels.map(date => {
            const close = spyMap[date];
            if (close == null) return null;
            return (close / spyBase) * baseValue;
        });
        equityDatasets.push({
            label: 'S&P 500',
            data: spyNormalized,
            borderColor: '#5a6a7a',
            borderDash: [6, 3],
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.1,
            pointRadius: 0
        });
    }

    const equityCtx = document.getElementById('equityChart').getContext('2d');
    new Chart(equityCtx, {
        type: 'line',
        data: { labels: labels, datasets: equityDatasets },
        options: {
            responsive: true,
            plugins: { legend: { display: equityDatasets.length > 1 } },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) { return '$' + value.toLocaleString(); }
                    }
                }
            }
        }
    });

    // --- Allocation Chart ---
    const allocCtx = document.getElementById('allocationChart').getContext('2d');
    new Chart(allocCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Cash',
                data: equityData.map(d => d.cash),
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                fill: true,
                tension: 0.1
            }, {
                label: 'Invested',
                data: equityData.map(d => d.portfolio_value - d.cash),
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    stacked: true,
                    ticks: {
                        callback: function(value) { return '$' + value.toLocaleString(); }
                    }
                }
            }
        }
    });

    // --- Benchmark % Return Chart ---
    if (benchmarkData.length > 0 && document.getElementById('benchmarkChart')) {
        const portStart = equityData[0].portfolio_value;
        const portfolioReturns = equityData.map(d =>
            ((d.portfolio_value - portStart) / portStart) * 100
        );

        const spyReturns = labels.map(date => {
            const close = spyMap[date];
            if (close == null || !spyBase) return null;
            return ((close - spyBase) / spyBase) * 100;
        });

        const benchCtx = document.getElementById('benchmarkChart').getContext('2d');
        new Chart(benchCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Portfolio',
                    data: portfolioReturns,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.08)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }, {
                    label: 'S&P 500',
                    data: spyReturns,
                    borderColor: '#5a6a7a',
                    borderDash: [6, 3],
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: true } },
                scales: {
                    y: {
                        ticks: {
                            callback: function(v) { return (v >= 0 ? '+' : '') + v.toFixed(1) + '%'; }
                        }
                    }
                }
            }
        });
    }
</script>
{% endif %}
{% endblock %}
```

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest tests/test_dashboard.py tests/test_dashboard_benchmark.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py dashboard/templates/performance.html
git commit -m "feat: add SPY overlay and benchmark chart to performance page"
```

---

### Task 6: Add route-level tests for benchmark integration

**Files:**
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add portfolio alpha route tests**

Add to `tests/test_dashboard.py`, inside the `TestPortfolioPage` class:

```python
    def test_portfolio_computes_alpha_when_equity_data_available(self, client):
        mock_queries.get_equity_curve.return_value = [
            make_snapshot_row(date=date(2026, 1, 2), portfolio_value=Decimal("100000")),
            make_snapshot_row(date=date(2026, 1, 6), portfolio_value=Decimal("105000")),
        ]
        mock_benchmark.get_spy_benchmark.return_value = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 505.0},
        ]
        mock_benchmark.compute_alpha.return_value = {
            "portfolio_return": 5.0,
            "spy_return": 1.0,
            "alpha": 4.0,
        }

        resp = client.get("/")
        assert resp.status_code == 200
        assert b"+4.00%" in resp.data

    def test_portfolio_handles_empty_benchmark(self, client):
        mock_queries.get_equity_curve.return_value = []
        mock_benchmark.get_spy_benchmark.return_value = []
        mock_benchmark.compute_alpha.return_value = None

        resp = client.get("/")
        assert resp.status_code == 200
        assert b"mdash" in resp.data
```

- [ ] **Step 2: Add performance benchmark route tests**

Add a new class to `tests/test_dashboard.py`:

```python
class TestPerformancePageBenchmark:
    """Tests for SPY benchmark on /performance."""

    def test_performance_passes_benchmark_data(self, client):
        eq = [
            make_snapshot_row(date=date(2026, 1, 2), portfolio_value=Decimal("100000")),
            make_snapshot_row(date=date(2026, 1, 6), portfolio_value=Decimal("105000")),
        ]
        mock_queries.get_equity_curve.return_value = eq
        mock_queries.get_performance_metrics.return_value = {
            "start_value": 100000,
            "end_value": 105000,
            "pnl": 5000,
            "pnl_pct": 5.0,
            "start_date": date(2026, 1, 2),
            "end_date": date(2026, 1, 6),
        }
        mock_benchmark.get_spy_benchmark.return_value = [
            {"date": "2026-01-02", "close": 500.0},
            {"date": "2026-01-06", "close": 505.0},
        ]
        mock_benchmark.compute_alpha.return_value = {
            "portfolio_return": 5.0,
            "spy_return": 1.0,
            "alpha": 4.0,
        }

        resp = client.get("/performance")
        assert resp.status_code == 200
        assert b"+4.00%" in resp.data
        assert b"S&amp;P 500" in resp.data or b"S&P 500" in resp.data

    def test_performance_no_benchmark_degrades(self, client):
        mock_queries.get_equity_curve.return_value = []
        mock_queries.get_performance_metrics.return_value = None
        mock_benchmark.get_spy_benchmark.return_value = []
        mock_benchmark.compute_alpha.return_value = None

        resp = client.get("/performance")
        assert resp.status_code == 200
```

- [ ] **Step 3: Run all tests**

Run: `python3 -m pytest tests/test_dashboard.py tests/test_dashboard_benchmark.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_dashboard.py
git commit -m "test: add route-level tests for SPY benchmark integration"
```

---

### Task 7: Smoke test in browser

**Files:** None (verification only)

- [ ] **Step 1: Rebuild and restart dashboard container**

```bash
docker compose build dashboard
docker compose up -d dashboard
```

- [ ] **Step 2: Open in browser and verify**

Open `http://localhost:3000/` and check:
- Portfolio page shows the "vs S&P 500" card with a percentage or "—"
- Navigate to `/performance` and verify:
  - Equity chart has a dashed SPY overlay line (if benchmark data available)
  - A new "Portfolio vs S&P 500 (% Return)" chart appears below Cash vs Invested
  - The alpha card appears in the metrics row

- [ ] **Step 3: Run the full test suite one final time**

```bash
python3 -m pytest tests/ -v
```
Expected: All tests PASS, no regressions
