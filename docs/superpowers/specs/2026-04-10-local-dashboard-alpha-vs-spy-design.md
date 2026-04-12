---
name: Local dashboard alpha vs SPY
date: 2026-04-10
status: approved
---

# Local dashboard alpha vs SPY

## Goal

Port the three SPY benchmark features from the public dashboard (`public_dashboard/`) to the local Flask dashboard at `/dashboard` (port 3000):

1. A "vs S&P" summary card showing portfolio alpha over SPY
2. An SPY overlay on the existing equity curve chart
3. A dedicated benchmark chart showing portfolio vs SPY as % returns

## Non-goals

- No DB schema changes.
- No session stage wiring or backfill.
- No changes to `v2/dashboard/` (empty template dir, not currently deployed).
- No refactor of the existing `fetch_spy_benchmark` in `v2/dashboard_publish.py`. The local dashboard gets its own helper to keep the local and publish paths loosely coupled. If we later want to DRY this up, it can move to a shared module.

## Architecture

### New module: `dashboard/benchmark.py`

Two pure-ish functions and a module-level cache.

**`get_spy_benchmark(start: date, end: date) -> list[dict]`**
- Fetches SPY daily bars from Alpaca using `StockHistoricalDataClient` with `DataFeed.IEX`, mirroring `v2/dashboard_publish.py:fetch_spy_benchmark`.
- Returns `list[{"date": "YYYY-MM-DD", "close": float}]`.
- Wrapped in a module-level dict cache keyed by `(start, end)` with a 15-minute TTL (uses `time.time()` so it's mockable).
- On any exception, logs a warning and returns `[]`. The dashboard degrades gracefully: no card, no overlay, no benchmark chart.

**`compute_alpha(snapshots: list[dict], benchmark: list[dict]) -> dict | None`**
- Mirrors the JS logic in `public_dashboard/app.js:115-137`.
- Finds the first and last snapshot dates that have a matching SPY close (weekends/holidays may cause gaps).
- `portfolio_return`: uses TWR over `net_deposits` if available on snapshots; otherwise falls back to `(last_value - first_value) / first_value * 100` (same fallback as the JS at lines 282-287 and the publish code).
- `spy_return`: `(spy_end - spy_start) / spy_start * 100`.
- `alpha`: `portfolio_return - spy_return`.
- Returns `{"portfolio_return", "spy_return", "alpha"}` or `None` when insufficient data (fewer than 2 snapshots, empty benchmark, or no overlapping dates).

### Routes wired in `dashboard/app.py`

**`/` (portfolio)**
- Pull recent snapshots over a reasonable window (reuse `get_equity_curve(days=90)` or equivalent) + SPY over the same range.
- Call `compute_alpha` and pass the result to `portfolio.html` as `alpha_stats`.

**`/performance`**
- Pull 90-day equity curve (already done) + SPY over the same range.
- Pass `benchmark_data` (list of `{date, close}`) and `alpha_stats` to `performance.html`.

### Templates

**`dashboard/templates/portfolio.html`**
- Add a new "vs S&P" card alongside the existing P&L cards. Shows `alpha_stats.alpha` as a signed percentage, colored green/red via the existing P&L class helper. Hidden / shows "—" when `alpha_stats` is `None`.

**`dashboard/templates/performance.html`**
- Add a second dataset to the existing Chart.js equity curve: SPY normalized to the portfolio's starting value, rendered as a dashed line. Matches `renderEquityCurve` in `public_dashboard/app.js:141-199`.
- Add a new canvas below the equity chart: portfolio vs SPY as % returns from day 0. Matches `renderBenchmark` in `public_dashboard/app.js:234+`.
- Both additions are guarded: if `benchmark_data` is empty, neither the overlay nor the second chart renders (show an empty-state message for the second chart, matching the public dashboard).

## Data flow

```
Flask route
  └─ get_equity_curve(90)        → snapshots
  └─ benchmark.get_spy_benchmark → SPY bars (cached 15 min)
  └─ benchmark.compute_alpha     → {portfolio_return, spy_return, alpha}
  └─ render_template(..., alpha_stats=..., benchmark_data=...)
```

## Error handling and graceful degradation

- Alpaca failures → `get_spy_benchmark` returns `[]`, logs a warning.
- Empty benchmark → `compute_alpha` returns `None`.
- Missing/None alpha in templates → card shows "—", equity chart renders without overlay, second benchmark chart shows empty state.
- No try/except in the Flask routes; the helper absorbs failures at the boundary.

## Caching

- In-process dict: `_CACHE: dict[tuple[date, date], tuple[float, list[dict]]]` where the tuple is `(expires_at, data)`.
- TTL: 15 minutes (`_TTL_SECONDS = 900`).
- No thread safety needed — Flask dev server is single-worker locally, and stale entries are harmless (worst case: a second request fetches in parallel).
- Cache is invalidated only by expiry; no manual bust needed.

## Testing

New file: `tests/test_dashboard_benchmark.py`

**`compute_alpha` tests (no mocks):**
- Happy path: aligned snapshot and SPY dates, returns expected alpha.
- Misaligned dates: snapshot start on a weekend, SPY start on next trading day — uses first overlapping date.
- Empty benchmark → returns `None`.
- Single snapshot → returns `None`.
- `net_deposits` path vs. fallback `first_value` path.

**`get_spy_benchmark` tests (mock `StockHistoricalDataClient` + `time.time`):**
- Cache miss: calls Alpaca once, returns bars.
- Cache hit within TTL: no second Alpaca call.
- Cache expiry past TTL: re-fetches.
- Alpaca raises → returns `[]`, logs warning.

**Route tests (patch `get_spy_benchmark` in the dashboard app module):**
- `/` passes `alpha_stats` to the template when benchmark is available.
- `/performance` passes `benchmark_data` and `alpha_stats`.
- Empty benchmark → routes still return 200, template context has `alpha_stats=None` and `benchmark_data=[]`.

## Open risks

- **Equity curve field availability.** The JS fallback at `app.js:282-287` assumes snapshots may not have `net_deposits`. Need to verify what `dashboard/queries.py:get_equity_curve` actually returns. If `net_deposits` is absent, `compute_alpha` uses the first/last portfolio value fallback. Confirmed during implementation.
- **Cache TTL tuning.** 15 min is a guess. If it feels stale for a user hitting refresh, drop to 5 min. If the Alpaca round-trip is noticeable, raise to 60 min. Not worth a config knob yet.
- **First-hit latency.** A cold cache hit blocks the route for ~200ms on the Alpaca call. Acceptable for a local dashboard; revisit if it becomes annoying.
