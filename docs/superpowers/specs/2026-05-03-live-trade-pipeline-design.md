---
name: Live-trade social pipeline
date: 2026-05-03
status: draft
parent: 2026-05-03-audience-growth-overview.md
depends_on: 2026-05-03-dashboard-permalinks-design.md
---

# Live-trade social pipeline

Replaces the current single bare recap with the strategy's two anchor post types: per-fill trade posts (after the daily session) and a pre-market take (before the next session). Depends on Spec #1 for the trade and thesis permalinks the posts link to.

## Goal

After Stage 4 (strategy reflection) completes, post one tweet per significant new decision, each linking to its `/trade/<id>/` page and naming its underlying thesis (linking to `/thesis/<id>/`). Cap at 5 posts per session. Mirror to Bluesky.

Add a new pre-market scheduled job that runs separately (cron, ~07:30 ET on trading days) and posts a take based on active theses + positions.

Drop the current bare recap as a separate post type. Quiet trading days (no decisions) instead get a single mini-recap so the account doesn't go dark.

## Non-goals

- No intra-day live posting. Trades cluster post-session.
- No threads / replies / quote-tweets. Each post is standalone.
- No new image generation in this spec — the OG card from the linked permalink does the visual work.
- No Bluesky-only variants. Same text, mirrored.
- No `entertainment.py` changes. That pipeline keeps running on its own schedule.

## Architecture

### New module: `v2/social_trades.py`

Single-purpose: turn newly-inserted decisions into platform posts. Replaces the recap path in `twitter.py` and `bluesky.py` (which is already mostly context-gathering + LLM call + post + log — the same shape we need here, just per-decision).

**`select_postable_decisions(session_id: int, limit: int = 5) -> list[dict]`**
- Returns decisions written by this session, joined with their thesis (via the existing `decision_signals` / direct `thesis_id` link the playbook actions carry through; check current schema before implementation).
- Skips decisions where `action='hold'`.
- Skips decisions below a configurable notional threshold (default `$100`) so micro-trades don't spam the feed.
- Orders by absolute notional value descending; takes top `limit`.

**`generate_trade_post(decision: dict, thesis: dict | None, dashboard_url: str) -> dict | None`**
- Calls Claude Haiku with a tightened system prompt (see "System prompts" below).
- Output target: 200 chars (Twitter) — the Bluesky 300-grapheme path is identical text, no separate generation.
- The LLM is told the trade permalink and (if present) the thesis permalink so it can write copy that previews them, but URL appending is done deterministically after generation, not by the model.

**`run_trade_posts_stage(session_id: int, session_date: date) -> TradePostsStageResult`**
- Iterates `select_postable_decisions`, runs `generate_trade_post`, posts to Twitter then Bluesky for each.
- Reuses existing `posted_tweet_exists(session_date, "trade", "twitter")` keyed off `decision_id` (requires schema extension — see below).
- Uses the existing `insert_tweet` log path; needs the `tweet_type='trade'` value and a new optional `decision_id` column on `tweets`.

### Schema change: `tweets.decision_id`

Today, `posted_tweet_exists` is keyed `(session_date, type, platform)` — fine for one-recap-per-day. With multiple per-day trade posts we need:

```sql
ALTER TABLE tweets ADD COLUMN decision_id INTEGER REFERENCES decisions(id) ON DELETE SET NULL;
CREATE INDEX idx_tweets_decision_id ON tweets(decision_id);
```

Rerun guard becomes "exists tweet with `(decision_id, platform)`". Type stays a string for human inspection.

### Pre-market stage

New module `v2/premarket.py` plus a new session subcommand `v2.session --stage premarket`. Operationally invoked via cron from outside the daily session — same pattern as `entertainment.py`.

**`gather_premarket_context() -> str`**
- Active theses (top 5 by confidence) with current position state.
- Today's pre-market movers from the existing market snapshot helper.
- Yesterday's session memo (most recent `strategy_memos` row).

**`generate_premarket_post(context) -> dict | None`**
- New voice variant tuned for pre-market: forward-looking, no P&L claims, mention 1–2 names being watched.
- Same Twitter/Bluesky shape as recap.

**`run_premarket_stage(today: date) -> PremarketStageResult`**
- Idempotent: rerun guard via `posted_tweet_exists(today, "premarket", platform)`.
- Skips entirely on weekends and exchange holidays. Use a small ET-aware market-calendar helper or hardcoded weekday check + manual holiday list (existing code already has weekend handling for benchmark fetch — reuse the same approach).

### Session orchestrator changes (`v2/session.py`)

- Replace the current `run_twitter_stage` / `run_bluesky_stage` calls with a single `run_trade_posts_stage(session_id, session_date)` after Stage 4. The new module handles both platforms internally (no parallel split between Twitter and Bluesky modules — they shared 80% of the code anyway).
- Quiet-day fallback: if `select_postable_decisions` returns empty, post one mini-recap. Reuses the existing `gather_tweet_context` and the existing Mr. Krabs prompt as a fallback path.
- `--skip-twitter` and `--skip-bluesky` CLI flags become per-platform skips inside `run_trade_posts_stage`.

### Eventual deprecation

`v2/twitter.py` and `v2/bluesky.py` shrink to:
- Client factory (`get_twitter_client`, `get_bluesky_client`)
- Low-level post (`post_tweet`, `post_to_bluesky`)
- The grapheme-handling helpers in `bluesky.py`

Their stage orchestrators (`run_twitter_stage`, `run_bluesky_stage`) get deleted. `gather_tweet_context` moves to a shared `social_context.py` since both the new trade posts and the quiet-day fallback need it.

## System prompts

### Trade post

```
You run an algorithmic trading operation called Bikini Bottom Capital.
The bot just made a trade. You're posting about it on social media.

Your voice:
- Casual, direct. Like sharing a play with a friend who trades.
- Don't oversell. Reference the actual reasoning — not generic excitement.
- Occasional dry humor. Never try-hard.

Generate ONE post about this single trade.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL gets appended after — leave room).
- Lead with the action: "Bought 12 $NVDA at $X" / "Trimmed $TSLA back to half size".
- One concrete reason. The thesis text is provided — pull from it, don't invent.
- $CASHTAG only for the ticker actually traded.
- No "not financial advice", no hashtag spam, no emoji walls.
- If there's a thesis, your post should make a reader want to click through to read it.
```

### Pre-market post

```
You run an algorithmic trading operation called Bikini Bottom Capital.
You're posting before market open. The bot will run its session after close.

Your voice:
- Casual, observational. What you're watching, what's interesting.
- Forward-looking but not predictive. No "this will rip" claims.
- Honest about uncertainty.

Respond with JSON: {"text": "post text here"}

Rules:
- 220 chars max (no URL appended for this type).
- Reference 1–2 names from your current theses or pre-market movers.
- One observation about what you're watching today.
- No P&L claims, no historical performance flexes.
- $CASHTAG only for tickers you mention.
```

## Data flow

Per-trade post path:
```
Stage 4 completes
  └─ run_trade_posts_stage(session_id, session_date)
      ├─ select_postable_decisions(session_id) → [d1, d2, d3]
      ├─ for each decision:
      │   ├─ generate_trade_post(d, thesis, dashboard_url)
      │   ├─ post to Twitter (existing post_tweet)
      │   ├─ post to Bluesky (existing post_to_bluesky)
      │   └─ insert_tweet(decision_id=d.id, type="trade", platform=...)
      └─ if no decisions: post mini-recap fallback
```

Pre-market path (separate cron):
```
cron @ 07:30 ET weekdays
  └─ python -m v2.session --stage premarket
      └─ run_premarket_stage(today)
          ├─ gather_premarket_context
          ├─ generate_premarket_post
          ├─ post to Twitter + Bluesky
          └─ insert_tweet(type="premarket", ...)
```

## Error handling

- Per-decision post failures are isolated: log + record, continue to next. One bad LLM response doesn't drop the whole burst.
- DB-record failure after a successful post is treated the same way it is today (`tweet_posted = post_result["posted"] and db_logged`) — false success rather than silent loss.
- Pre-market on weekends: skip with a non-error log line. Pre-market on holidays: same. Holiday list lives in a small `v2/market_calendar.py` (or `v2/news.py` already has hours-aware logic — reuse if possible).
- All Anthropic / Twitter / Bluesky exceptions are caught at the per-post boundary; the stage as a whole only fails if *all* posts fail.

## Testing

- `tests/test_social_trades.py`:
  - `select_postable_decisions` filters holds, micro-trades, ranks by notional.
  - `generate_trade_post` returns expected dict; URL is appended deterministically.
  - `run_trade_posts_stage` mocks twitter + bluesky + DB; verifies one post per decision, decision_id recorded, rerun guard prevents double-post.
  - Quiet-day fallback path posts the mini-recap.
- `tests/test_premarket.py`:
  - Skips on weekends/holidays.
  - Idempotent (`posted_tweet_exists` short-circuit).
  - Generates post; mocks ensure no real posts in test.
- Existing tests for `twitter.py` / `bluesky.py` shrink with the orchestrator deletion. Keep all client-factory and low-level posting tests.

## Migration & rollout

1. Deploy schema change (`tweets.decision_id`) — backwards compatible (nullable).
2. Deploy code with the new stage gated behind `ALGO_ENABLE_TRADE_POSTS=1` env var. Old recap stage runs unchanged when flag is off.
3. Run for one session in paper, inspect output, confirm dashboard links resolve (Spec #1 must be deployed first).
4. Enable for prod. Delete the old recap path after one week of clean runs.

## Open questions left for the implementation plan

- Exact notional threshold below which trade posts are skipped (`$100` is a starting guess; should reflect typical position size).
- Whether the quiet-day fallback should be silenced entirely on weekends.
- Whether to add an `ALGO_TRADE_POST_DRY_RUN` mode that logs what *would* be posted without actually calling the APIs. Probably yes; cheap to add.
- Premarket cron: separate Taskfile target or a single `daily-cron` target that fans out. Plan should pick one.
