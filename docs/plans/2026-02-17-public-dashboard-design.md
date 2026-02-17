# Public Dashboard Design

**Date:** 2026-02-17
**Status:** Approved

## Overview

Add a public-facing dashboard hosted on GitHub Pages so Twitter/Bluesky followers can view Bikini Bottom Capital's holdings, performance, and trading activity. The session pipeline generates JSON data files and pushes them to a GitHub Pages repo. Social posts link to the dashboard for detail.

## Approach

Static site with client-side rendering. The HTML/CSS/JS are static assets that rarely change. Each session run regenerates only the JSON data files, commits, and pushes to GitHub Pages. Charts rendered client-side with Chart.js (same library used in the internal dashboard).

## Pipeline Integration

New Stage 6 in `v2/session.py`, runs after Twitter (5) and Bluesky (5b):

1. Query DB for all public data
2. Write JSON files to local clone of GitHub Pages repo
3. Git commit and push

- Independent stage with error isolation (same pattern as Twitter/Bluesky)
- `--skip-dashboard` CLI flag
- Logged to session output like other stages

## New Module: `v2/dashboard_publish.py`

- `gather_dashboard_data(session_date)` — Queries DB for snapshots, positions, decisions, theses, account summary. Returns dict of data ready for JSON serialization.
- `write_json_files(data, repo_path)` — Writes JSON files to `data/` directory in the local GitHub Pages repo clone.
- `push_to_github(repo_path)` — Git add, commit, push the data directory.
- `run_dashboard_stage(session_date)` — Orchestrator: gather data -> write JSON -> push. Returns stage result.

## GitHub Pages Repo Structure

```
index.html              <- static, rarely changes
styles.css              <- static
app.js                  <- static (Chart.js rendering logic)
data/
  snapshots.json        <- regenerated each run
  positions.json        <- regenerated each run
  decisions.json        <- regenerated each run
  theses.json           <- regenerated each run
  summary.json          <- regenerated each run (headline stats)
```

## Dashboard Sections

Single-page layout with sections:

**Summary Header**
- Portfolio value, cash, daily P&L, total return %
- Last updated timestamp

**Equity Curve**
- Chart.js line chart from snapshots.json (90 days of account_snapshots)

**Current Holdings**
- Table: ticker, shares, avg cost, current value, gain/loss $ and %

**Recent Decisions**
- Last 30 days of trades: date, ticker, action, reasoning (truncated), outcome

**Active Theses**
- Ticker, direction, confidence, core thesis, entry/exit triggers

**Branding:** "Bikini Bottom Capital" in the header, clean professional layout with subtle nautical accent colors. No character images or heavy theming.

## Data Exposure

Full transparency — actual dollar amounts, share counts, cost basis, exact P&L. Real numbers.

## Social Post Updates

Append the GitHub Pages URL to the end of each session post (Twitter and Bluesky). No change to post generation logic, persona, or character limits — just tack on the link.

## Environment Variables

- `DASHBOARD_REPO_PATH` — local path to the cloned GitHub Pages repo
- `DASHBOARD_URL` — the public URL to append to social posts

## Dependencies

- No new Python dependencies (json stdlib + git CLI)
- Frontend: Chart.js from CDN
- GitHub Pages repo (e.g. `bikini-bottom-capital.github.io`)
