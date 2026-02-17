# Bluesky Integration Design

**Date:** 2026-02-17
**Status:** Approved

## Overview

Add Bluesky as a second social media platform alongside Twitter. Both the session pipeline (Stage 5) and the entertainment pipeline post to both platforms with independently generated content.

## Approach

Parallel posting functions — a new `v2/bluesky.py` module mirrors `v2/twitter.py` structure. No abstract social media layer (YAGNI for 2 platforms). Existing Twitter code stays untouched.

## New Module: `v2/bluesky.py`

- `get_bluesky_client()` — Reads `BLUESKY_HANDLE` and `BLUESKY_APP_PASSWORD` from env. Returns logged-in `atproto.Client` or `None`.
- `generate_bluesky_post(context, model)` — Separate LLM call with Bluesky-specific system prompt (same Mr. Krabs persona, 300 char limit). Returns `{"text": "...", "type": "recap"}`.
- `post_to_bluesky(post, client)` — Posts via `client.send_post()`. Returns `{"text", "type", "posted", "post_id", "error"}`.
- `run_bluesky_stage(session_date)` — Orchestrator: reuses `gather_tweet_context()` from twitter.py → generate → post → log to DB. Returns `BlueskyStageResult`.

## Database Changes

- Migration `009_tweets_platform.sql`: Add `platform TEXT NOT NULL DEFAULT 'twitter'` column to `tweets` table.
- `insert_tweet()` gains a `platform='twitter'` parameter. Bluesky rows pass `platform='bluesky'`.

## Session Orchestration (`v2/session.py`)

- Rename Stage 5 logs from "Twitter posting" to "Social posting".
- Add `--skip-bluesky` flag.
- Run Bluesky stage after Twitter stage (independent — failures don't block each other).
- Add `bluesky_result` and `bluesky_error` to `SessionResult`.

## Entertainment Pipeline (`v2/entertainment.py`)

- After posting to Twitter, generate a separate Bluesky post and post it.
- Add `bluesky_posted`, `bluesky_post_id`, `bluesky_error` to `EntertainmentResult`.
- Both platforms independent.

## Dependencies

- `atproto>=0.0.55` in `v2/requirements.txt`.
- Env vars: `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`.

## Testing

- New `tests/v2/test_bluesky.py` mirroring `test_twitter.py` structure.
- Mock `atproto.Client`, `get_cursor`, `get_claude_client`.
- Update `test_session.py` and `test_entertainment.py` for Bluesky fields.

## Content Strategy

- Each platform gets a separately generated post (independent LLM calls).
- Same Mr. Krabs persona on both.
- Twitter: 280 char limit. Bluesky: 300 char limit.
