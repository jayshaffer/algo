---
name: Mistakes log & attribution panel
date: 2026-05-03
status: draft
parent: 2026-05-03-audience-growth-overview.md
depends_on: 2026-05-03-dashboard-permalinks-design.md
---

# Mistakes log & attribution panel

Adds the two weekly content slots from the rotation: "what I got wrong" and the attribution roundup. Each gets a dashboard section that doubles as the link target for the post. Depends on Spec #1 (the per-page rendering pipeline this layers onto).

## Goal

1. Surface a **Mistakes log** on the public dashboard — closed losers, retired rules, honest framing.
2. Surface an **Attribution panel** — chart of which signal types actually predicted, sourced from the existing `signal_attribution` table.
3. Add two weekly scheduled posts: "What I got wrong" and the attribution roundup, each linking to the relevant dashboard section.
4. Generate weekly OG images so the link previews carry the actual chart / mistake text.

## Non-goals

- No new ML / scoring work — `signal_attribution` is already computed by Stage 0.
- No editorializing of mistakes by hand. The post is generated from data + memos, not curated.
- No "what worked" weekly post in this round (it's structurally weaker than "what didn't").
- No backfill of historical mistakes into a different format. Use what's in the DB.

## Architecture

### Dashboard sections (additions to `public_dashboard/index.html`)

Two new `<section class="panel">` blocks on the homepage, plus their permalink subpages:

**`#mistakes` section — "What didn't work"**
- Closed losing decisions (`outcome_30d < 0`, ordered by magnitude descending) over the last 30 days.
- Retired rules (`strategy_rules.status = 'retired'`) with the retirement memo.
- Empty state: "No closed losers in window. Either we got lucky or we didn't trade enough."
- Permalink: `/mistakes/index.html` so the weekly post has a stable URL even when the homepage scroll position changes.

**`#attribution` section — "What's actually working"**
- Bar chart from `signal_attribution`: signal type on x-axis, predictiveness score on y-axis, color-coded above/below 0.
- Below the chart, a 3-column table: signal type, sample size, score.
- Permalink: `/attribution/index.html`.

### New rendering: `v2/dashboard_pages.py` (extending Spec #1)

Spec #1 introduces `dashboard_pages.py`. This spec adds:

**`render_mistakes_page(closed_losers: list[dict], retired_rules: list[dict]) -> str`**
**`render_attribution_page(attribution: list[dict]) -> str`**

Both pages reuse the same shell as the homepage (header, footer, styles.css) and embed the same chart-data JSON the SPA reads. They render server-side enough HTML for OG previews to look correct even before JS executes.

### OG image extensions: `v2/dashboard_og.py`

**`render_mistakes_og(top_loser: dict) -> bytes`**
- Lead with the worst recent loser: ticker, P&L, "lessons learned" tagline.

**`render_attribution_og(attribution: list[dict]) -> bytes`**
- Renders the bar chart directly into the OG PNG using Pillow's drawing primitives (no headless browser). Top 5 signal types only; readable at preview size.

### New data gathering in `dashboard_publish.py`

Extend `gather_dashboard_data` with:

```python
# Closed losers (last 30 days, outcome_30d resolved)
SELECT id, date, ticker, action, quantity, price, reasoning, outcome_30d, ...
FROM decisions
WHERE date > %s - INTERVAL '30 days'
  AND outcome_30d IS NOT NULL
  AND outcome_30d < 0
ORDER BY outcome_30d ASC LIMIT 20;

# Retired rules (last 90 days)
SELECT id, rule_text, retired_at, retired_reason
FROM strategy_rules
WHERE status = 'retired' AND retired_at > NOW() - INTERVAL '90 days'
ORDER BY retired_at DESC LIMIT 20;

# Attribution (latest computed scores)
SELECT signal_type, score, sample_size, computed_at
FROM signal_attribution
ORDER BY computed_at DESC, score DESC;
```

These get added to the JSON files written by `write_json_files`, so the SPA can render them client-side too.

### New module: `v2/social_weekly.py`

Two scheduled-post functions, mirrors the shape of Spec #2's modules:

**`run_mistakes_post(today: date) -> WeeklyPostResult`**
- Pulls the worst N losers from the last 7 days.
- Pulls retired-rule memos.
- Generates a 200-char post via Claude with a counter-positioning system prompt (see below).
- Posts to Twitter + Bluesky, links to `/mistakes/`.
- Idempotent via `posted_tweet_exists(today, "weekly_mistakes", platform)`.

**`run_attribution_post(today: date) -> WeeklyPostResult`**
- Pulls top-3 best and worst signal types from `signal_attribution`.
- Generates a 200-char post: "Signals that worked / didn't this week" + cashtag-free framing.
- Posts to Twitter + Bluesky, links to `/attribution/`.
- Idempotent.

### Schedule

Weekly cron triggers, separate from the daily session:
- **Sunday 17:30 ET**: daily session runs (Stage 6 publishes `/mistakes/` and `/attribution/` with fresh data)
- **Sunday 18:00 ET**: `python -m v2.session --stage weekly_mistakes`
- **Sunday 18:30 ET**: `python -m v2.session --stage weekly_attribution`

Order is load-bearing: the weekly posts must run *after* Sunday's Stage 6 publish so the linked pages reflect the data the post text was generated from. If Stage 6 fails on Sunday, the weekly posts skip with an error rather than linking to stale content (guard checks `dashboard_publishes.completed_at` from the latest session).

## System prompts

### Weekly mistakes

```
You run an algorithmic trading operation called Bikini Bottom Capital.
You post weekly about what the bot got wrong.

Your voice:
- Honest. Specific. No self-flagellation, no "valuable lesson learned".
- Treat losses as data, not embarrassment.
- Dry, not bitter.

Most trading accounts hide losses. You don't. That's the point.

Generate ONE post about this week's worst trade or retired rule.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL appended after).
- One specific thing — the worst trade, or the retired rule, not a list.
- Reference the actual ticker / rule, not "a position" or "a strategy".
- No "we'll do better next time" / no "lessons learned" cliché.
```

### Weekly attribution

```
You run an algorithmic trading operation called Bikini Bottom Capital.
You post weekly about which signal types are actually predictive.

Your voice:
- Curious about the data.
- Comfortable saying "this one didn't work" without spinning it.
- A little nerdy. Slightly overshare-y about methodology.

Generate ONE post about this week's signal attribution scores.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL appended after).
- Name 1–2 signal types and their scores. Not all of them.
- One non-obvious observation, if there is one. Otherwise just the data.
- Don't claim "alpha". Use "predictive" / "useful" / "noise".
```

## Data flow

```
cron Sunday 18:00 ET
  └─ python -m v2.session --stage weekly_mistakes
      └─ run_mistakes_post(today)
          ├─ gather (last 7 days losers + retired rules)
          ├─ generate_mistakes_post (Haiku)
          ├─ post to Twitter + Bluesky (links to /mistakes/)
          └─ insert_tweet(type="weekly_mistakes", ...)

cron Sunday 18:30 ET
  └─ python -m v2.session --stage weekly_attribution
      └─ run_attribution_post(today)
          ├─ gather (latest signal_attribution rows)
          ├─ generate_attribution_post (Haiku)
          ├─ post to Twitter + Bluesky (links to /attribution/)
          └─ insert_tweet(type="weekly_attribution", ...)
```

Stage 6 keeps emitting `/mistakes/index.html` and `/attribution/index.html` daily; weekly post points at whatever's currently published.

## Error handling

- If `signal_attribution` is empty (e.g., not enough samples yet), `run_attribution_post` skips with a non-error log line. The post is content, not infra; missing it is fine.
- Same for `run_mistakes_post` — if there are zero closed losers in the window, skip with a log line. ("No mistakes this week" is not the post we want to make.)
- Page rendering inherits Spec #1's per-page error isolation.

## Testing

- `tests/test_social_weekly.py`:
  - Mistakes post: skips on empty data; generates expected post; correct rerun guard.
  - Attribution post: same.
- `tests/test_dashboard_pages.py` extensions:
  - `render_mistakes_page` produces HTML with the expected closed-loser data.
  - `render_attribution_page` produces HTML with the expected scores table.
- `tests/test_dashboard_og.py` extensions:
  - `render_attribution_og` renders chart bars at expected pixel coordinates (use `Image.getpixel` checks at known positions).

## Open questions left for the implementation plan

- Exactly how many losers and retired rules to surface on `/mistakes/`. 20 is a starting cap; might be too many for the front page.
- Whether the homepage `#mistakes` section should show the same data as `/mistakes/` or a teaser. Probably teaser + "see all" link.
- Whether to add a "regression test" for the chart — e.g., pixel-diff the OG PNG against a checked-in golden. Likely overkill.
