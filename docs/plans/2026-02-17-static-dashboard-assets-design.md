# Static Dashboard Assets Design

**Date:** 2026-02-17
**Status:** Approved

## Overview

Create the static site assets (HTML/CSS/JS) for the public GitHub Pages dashboard. These files live in `public_dashboard/` in this repo and get copied to the GitHub Pages repo to initialize it. The session pipeline only regenerates `data/*.json` — these assets rarely change.

## Visual Style

- **Dark theme** with ocean-depth palette: deep navy background (`#0a1628`), lighter card surfaces (`#132240`)
- **Accent color:** Teal/aqua (`#00d4aa`) — subtle underwater feel without being cartoonish
- **P&L colors:** Green (`#00d4aa`) for gains, coral-red (`#ff6b6b`) for losses
- **Typography:** System font stack, monospace for numbers
- **Bikini Bottom nods:** Header says "Bikini Bottom Capital" with wave/anchor unicode. Section dividers use faint wave CSS pattern. Footer: "Trading from the depths" tagline. Clean first, themed second.

## Sections (top to bottom)

1. **Header** — "Bikini Bottom Capital" + last updated timestamp
2. **Summary cards** — Portfolio value, daily P&L, total return %, positions count, cash
3. **Equity curve** — Chart.js line chart (90 days), teal line on dark background
4. **Current Holdings** — Table: ticker, shares, avg cost
5. **Recent Decisions** — Table: date, ticker, action, quantity, reasoning (truncated), order_id
6. **Active Theses** — Cards: ticker, direction, confidence, thesis text, entry/exit triggers
7. **Footer** — "Data from Alpaca" attribution, "Trading from the depths"

## Technical Decisions

- **Chart.js 4.x from CDN** — same library as internal dashboard, no build step
- **Vanilla JS** — no framework, keeps it simple and fast
- **Responsive** — mobile-friendly with horizontal scroll on tables
- **Data fetched from relative `data/*.json` paths** — works both locally and on GitHub Pages

## Files

```
public_dashboard/
  index.html       <- page structure, Chart.js CDN link
  styles.css       <- dark theme, responsive layout, wave accents
  app.js           <- fetch JSON, render all sections, Chart.js config
```

## JSON Data Shapes (consumed from `data/`)

- `summary.json` — portfolio_value, cash, invested, daily_pnl, daily_pnl_pct, total_pnl, total_pnl_pct, positions_count, last_updated, inception_date
- `snapshots.json` — array of {date, portfolio_value, cash, buying_power}
- `positions.json` — array of {ticker, shares, avg_cost, updated_at}
- `decisions.json` — array of {id, date, ticker, action, quantity, price, reasoning, outcome_7d, outcome_30d, order_id}
- `theses.json` — array of {id, ticker, direction, confidence, thesis, entry_trigger, exit_trigger, created_at}
