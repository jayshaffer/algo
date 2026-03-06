# S&P 500 Benchmark Comparison — Design

## Goal

Add an S&P 500 (SPY) benchmark overlay to the public dashboard equity curve so viewers can see portfolio performance vs the market.

## Data Source

Fetch SPY daily bars from the Alpaca data API at dashboard publish time. The project already uses `StockHistoricalDataClient` in `v2/market_data.py`. Match the same date range as the portfolio snapshots (90 days). Save to `data/benchmark.json`.

## Chart Changes

Convert the equity curve from raw dollar values to **% return from inception**. Overlay a second line for SPY % return over the same period. Both start at 0% on day 1.

- Portfolio line: turquoise (`#00d4aa`), filled area
- SPY line: muted gray (`#5a6a7a`), dashed, no fill
- Y-axis: `%` labels instead of `$`
- Legend: enabled (two datasets)
- Tooltip: both values with `%` formatting

## Backend Changes (`v2/dashboard_publish.py`)

1. Add `fetch_spy_benchmark(snapshot_dates, start_date, end_date)`:
   - Uses `StockHistoricalDataClient` + `StockBarsRequest` for SPY daily bars
   - Returns `[{date, close}, ...]` for dates matching the snapshot window
2. Add `"benchmark"` key to `gather_dashboard_data()` return dict
3. Add `"benchmark.json"` to `write_json_files()` loop

## Frontend Changes (`public_dashboard/app.js`)

1. Fetch `benchmark.json` in `fetchAllData()`
2. In `renderEquityCurve()`:
   - Normalize portfolio values to % return (relative to first value)
   - Normalize SPY closes to % return
   - Add SPY as second dataset (dashed gray line)
   - Switch Y-axis from `$` to `%` formatting
   - Enable legend

## HTML Changes (`public_dashboard/index.html`)

1. Update section title from "Equity Curve (90 days)" to "Performance vs S&P 500"
2. Add "vs S&P" summary card showing alpha (portfolio return minus SPY return)

## Files Changed

- `v2/dashboard_publish.py` — benchmark data fetching + JSON export
- `public_dashboard/app.js` — chart rendering with dual datasets
- `public_dashboard/index.html` — section title + new summary card
- `public_dashboard/data/benchmark.json` — sample data
- `tests/test_dashboard_publish.py` — tests for benchmark fetching
