# Local Dashboard Linking + Detail Pages

**Status:** spec
**Scope:** `dashboard/` (the live local operator dashboard, Flask, port 3000). Not v2/dashboard, not the public GitHub Pages dashboard.

## Problem

The local dashboard exposes nine pages of rich, relational data — decisions, signals, theses, attribution, playbook, strategy, sessions, costs, events, tweets — but almost nothing is cross-linked. To answer a basic question like "which signals fed this decision, and is that signal category actually predictive?", an operator has to open three pages, mentally join by ticker or ID, and re-derive the relationship every time.

The schema already has the foreign keys (`decision_signals`, `playbook_actions.thesis_id`, `decisions.playbook_action_id`, `tweets.decision_id`, `agent_events.session_id`). The templates just don't surface them as links.

## Goal

Make every meaningful relationship in the schema clickable from the place an operator would naturally look. Four entity types become first-class landing pages (`ticker`, `thesis`, `decision`, `session`); every existing template gets inline cross-links so navigation is "click the thing you want to know about" instead of "remember an ID and type it into the URL bar".

## New routes

### `/ticker/<sym>` — Ticker overview

Aggregates everything filtered to one ticker.

Sections (all collapse cleanly to empty states):
- **Position** — current shares, avg cost, total value (one row from `positions`)
- **Theses** — all theses for this ticker, any status, newest first; ticker omitted from the row (header conveys it)
- **Recent decisions (90 days)** — same row shape as `/decisions` but ticker omitted; each row links to `/decision/<id>` and each signal-ref badge links to its source
- **Recent ticker signals (30 days)** — headline, category, sentiment, time
- **Open orders** — if any for this ticker
- **Attribution by source** — signal-category mini-table for signals that fed decisions on this ticker (computed: join `decision_signals` → `news_signals.category` → `decisions.outcome_*` for this ticker only)

URL is the natural drill-down from any ticker cell on any existing page.

### `/thesis/<id>` — Thesis detail

- Full thesis card (same fields as `theses.html`)
- **Originating session** link if `theses` has a `session_id` or matching `created_at::date` → `sessions` row (lookup by date)
- **Decisions citing this thesis** — rows from `decisions` joined via `decision_signals` where `signal_type='thesis' AND signal_id=<id>`
- **Playbook actions referencing this thesis** — rows from `playbook_actions WHERE thesis_id=<id>`, grouped by playbook date
- Inline invalidate/expire buttons reused from `theses.html` when status='active'

### `/decision/<id>` — Decision detail

- Full reasoning text, action, ticker (→ `/ticker/<sym>`), qty, price, account equity, outcome 7d/30d
- **Source signals** — full denormalized rows from `decision_signals`:
  - `news_signal` → headline, summary, category (→ category drill), sentiment, confidence, published_at, source URL if present
  - `macro_signal` → headline, category, affected sectors, sentiment, published_at
  - `thesis` → linked thesis card (→ `/thesis/<id>`)
- **Parent playbook action** if `playbook_action_id` set → action row + linked thesis
- **Tweets posted about this decision** — rows from `tweets WHERE decision_id=<id>`
- **Session context** — link to the `/session/<id>` whose date matches `decisions.date` (lookup by date)

### `/session/<id>` — Session detail

Single unified view of one session.

- Header: session_date, type, status, started/completed, total cost USD
- **Stages** — `session_stages` rows: stage_name, status, started/completed, cost (reuse `/costs/<id>` logic and inline)
- **Agent events** — last 200 events for this session_id (reuse events.html row shape; collapsed by default with "expand")
- **Decisions** — decisions on this `session_date`
- **Theses created** — theses with `created_at::date = session_date`
- **Memo** — `strategy_memos` row for this session_date
- **Tweets** — tweets with `session_date = <date>`, both platforms

## Inline link changes by template

| Template | Change |
|---|---|
| `portfolio.html` | wrap position-row ticker and open-order ticker in `<a href="/ticker/{{sym}}">`; "Watchlist:" tickers in playbook callout become links |
| `playbook.html` | ticker → `/ticker/{{sym}}`; thesis badge currently `/theses?status=all` → `/thesis/{{action.thesis_id}}`; watchlist pills → `/ticker/{{sym}}` |
| `signals.html` | every ticker → `/ticker/{{sym}}`; category badge → `/attribution#cat-{{category}}` (anchor scroll on attribution page); add `?category=<cat>` filter form (new query param) |
| `theses.html` | ticker → `/ticker/{{sym}}`; whole card has a "Details →" link in footer → `/thesis/{{id}}` |
| `decisions.html` | ticker → `/ticker/{{sym}}`; "Reasoning" sub-row clickable → `/decision/{{id}}` (entire row link); Playbook/Off-Playbook badge → `/decision/{{id}}`; signal-ref badge labels become links:<br>• `news_signal:<id>` → `/decision/{{id}}#signal-news-{{sid}}` (jumps into decision detail signal section — avoids a per-signal page)<br>• `macro_signal:<id>` → `/decision/{{id}}#signal-macro-{{sid}}`<br>• `thesis:<id>` → `/thesis/{{sid}}` |
| `attribution.html` | category cell → `/attribution?category=<cat>` (filtered version of the same page showing only that category PLUS its component signals in last 30d AND decisions that cited them); add `id="cat-<category>"` anchor on each row for cross-page links |
| `strategy.html` | memo's `session_date` → `/session/{{session_id}}` (need `lookup_session_id_by_date(date, type='daily')`); rules: no link target |
| `events.html` | already has session→filter link; add `/session/{{e.session_id}}` link as a small "→" icon next to existing one |
| `tweets.html` | session_date → `/session/{{session_id}}` (lookup); if `tweet.decision_id` set, add small "(decision #N)" link → `/decision/{{decision_id}}` |
| `costs.html` | session_date → `/session/{{session_id}}` (in addition to existing "stages →") |
| `audit.html` | findings list — current detail link works; add ticker link in finding payload when present |

## New queries (`dashboard/queries.py`)

All single-purpose, mockable, follow existing `get_cursor()` pattern.

```python
def get_ticker_position(sym: str) -> dict | None
def get_ticker_theses(sym: str) -> list[dict]
def get_ticker_decisions(sym: str, days: int = 90, limit: int = 50) -> list[dict]
def get_ticker_signals(sym: str, days: int = 30, limit: int = 50) -> list[dict]
def get_ticker_open_orders(sym: str) -> list[dict]
def get_ticker_attribution(sym: str, days: int = 90) -> list[dict]
  # group decision_signals→news/macro by category for decisions on this ticker only

def get_thesis(thesis_id: int) -> dict | None
def get_thesis_decisions(thesis_id: int) -> list[dict]
def get_thesis_playbook_actions(thesis_id: int) -> list[dict]

def get_decision(decision_id: int) -> dict | None
def get_decision_signals_full(decision_id: int) -> list[dict]
  # denormalized: signal_type, signal record fields
def get_decision_tweets(decision_id: int) -> list[dict]
def get_playbook_action(action_id: int) -> dict | None

def get_session(session_id: int) -> dict | None
def get_session_stages(session_id: int) -> list[dict]   # reuse get_session_stage_costs if shape matches
def get_session_decisions(session_id: int) -> list[dict]
  # filter decisions by session.session_date
def get_session_theses_created(session_id: int) -> list[dict]
def get_session_memo(session_id: int) -> dict | None
def get_session_tweets(session_id: int) -> list[dict]
def get_session_events(session_id: int, limit: int = 200) -> list[dict]
  # thin wrapper over existing get_recent_agent_events with session filter

def lookup_session_id_by_date(d: date, session_type: str = 'daily') -> int | None
  # used by strategy.html memo links, tweets.html, decision context

def get_attribution_category_drill(category: str, days: int = 30) -> dict
  # { signals: [...], decisions: [...], stats: {...} }
```

Extend two existing queries to accept an optional `category` filter:
- `get_recent_ticker_signals(days, limit, category=None)`
- `get_recent_macro_signals(days, limit, category=None)`

Extend one for `?category=` filter on attribution route:
- `get_signal_attribution(category=None)`

## New route handlers (`dashboard/app.py`)

```python
@app.route("/ticker/<sym>")             # ticker.html
@app.route("/thesis/<int:thesis_id>")    # thesis_detail.html
@app.route("/decision/<int:decision_id>") # decision_detail.html
@app.route("/session/<int:session_id>")   # session_detail.html
```

Existing `/attribution` and `/signals` accept a new optional `?category=` query param.

## New templates

Four new files under `dashboard/templates/`:
- `ticker.html`
- `thesis_detail.html`
- `decision_detail.html`
- `session_detail.html`

Each reuses Tailwind styles from `base.html`. No new CSS. No JS beyond what theses already use (close-modal pattern reused only on `thesis_detail.html`).

## Empty-state behavior

Every section on every new page renders a single grey-text "No <thing> for this <entity>" line when its query returns empty. No section is hidden — operators need to see "this ticker has no theses" as a positive fact, not a missing section.

## Tests

For each new route, mirroring the existing `tests/test_dashboard.py` `sys.modules["queries"]` injection pattern:

- `test_ticker_overview_renders` — happy path with mocked position/theses/decisions/signals
- `test_ticker_overview_empty` — no position, no decisions
- `test_ticker_overview_404` — currently N/A; ticker page always renders (a never-traded ticker shows empty state). Document this in route docstring.
- `test_thesis_detail_renders` / `_not_found` (404 when `get_thesis` returns None)
- `test_decision_detail_renders` / `_not_found` / `_with_signals` (asserts signal-ref denormalization is rendered)
- `test_session_detail_renders` / `_not_found`
- For each modified existing template: one regression test asserting the new link's `href` is present (e.g., `assert b'href="/ticker/AAPL"' in resp.data`)

Total: ~15 new tests, follow the conftest factories pattern.

## Implementation order

1. New queries in `dashboard/queries.py` + their tests
2. `/decision/<id>` route, template, tests — most-referenced new page
3. `/thesis/<id>` route, template, tests
4. `/ticker/<sym>` route, template, tests
5. `/session/<id>` route, template, tests
6. Inline link changes to existing templates + regression tests
7. `?category=` filter on `/signals` and `/attribution`

Each step is independently shippable; no big-bang merge.

## Out of scope

- v2/dashboard (stub awaiting full template migration; that's a separate cutover project)
- Public dashboard (`dashboard_publish.py`) — operator-only here
- Per-signal detail pages (`/signal/news/<id>`) — handled via anchors on `/decision/<id>` to avoid route bloat
- Strategy-rule → decision text-matching — that's text mining, not linking; FK does not exist
- Search box / global ticker autocomplete — separate UX project
- Auth/permission changes — `X-Requested-With: dashboard` POST guard pattern unchanged
- Performance work — current queries are fast enough at this data scale; revisit if `/ticker/<sym>` aggregate gets slow

## Risks

- **N+1 queries on `/ticker/<sym>`**: page makes ~6 queries. Acceptable for a single-operator dashboard but should be measured against `test_dashboard_benchmark.py` baselines after implementation.
- **Date-based session lookup ambiguity**: `lookup_session_id_by_date(date, type='daily')` could return multiple rows if more than one daily session ran in a day (e.g. retries). Pick the most-recent `started_at`; document in query docstring.
- **Tweet → session linkage is by date, not FK**: relies on `tweets.session_date = sessions.session_date`. If tweets are posted off-schedule from premarket runs, the lookup may resolve to the wrong session. Mitigate by also using `tweets.tweet_type` heuristic ('premarket' → premarket session) when ambiguous.
- **Backwards compatibility of `/signals?category=...`**: pre-existing bookmarks / scripts unaware of the param continue to work (defaults to no filter).
