# Twitter Stage Migration to v2 Pipeline

**Date:** 2026-02-16
**Status:** Approved

## Summary

Move the Twitter posting stage (Bikini Bottom Capital / Mr. Krabs tweets) from `trading/twitter.py` into the v2 pipeline as a new `v2/twitter.py` module, wired as Stage 5 of `v2/session.py`.

## Decisions

- **LLM backend:** Keep Ollama (`trading.ollama.chat_json`) for tweet generation — free, proven Mr. Krabs voice
- **DB access:** Use `v2.database.connection.get_cursor` and add tweet CRUD to `v2/database/trading_db.py`
- **Approach:** Self-contained copy in v2 (no shared module with original pipeline)

## Architecture

### New file: `v2/twitter.py`

Mirrors the structure of `trading/twitter.py` with these changes:

1. **DB imports:** `from .database.connection import get_cursor` instead of `from . import db`
2. **Context gathering:** Rewritten `gather_tweet_context()` queries adjusted for v2 schema:
   - `strategy_memos` query uses `session_date` column and `memo_type` filter
   - All other queries (decisions, positions, theses, snapshots) use the same table/column names
3. **Tweet generation:** Imports `chat_json` from `trading.ollama` (cross-package import — Ollama client is shared infrastructure)
4. **Tweepy client + posting:** Unchanged logic
5. **Orchestrator:** `run_twitter_stage()` unchanged

### DB additions: `v2/database/trading_db.py`

Add two functions:
- `insert_tweet(session_date, tweet_type, tweet_text, tweet_id, posted, error) -> int`
- `get_tweets_for_date(session_date) -> list`

These use the existing `tweets` table (shared across pipelines).

### Session integration: `v2/session.py`

- Add Stage 5: Twitter posting after strategy reflection
- Add `skip_twitter` parameter and `--skip-twitter` CLI flag
- Add `twitter_result` / `twitter_error` / `skipped_twitter` to `SessionResult`

### Tests: `tests/v2/test_twitter.py`

Port tests from `tests/test_twitter.py`, adjusting:
- Import paths to `v2.twitter`
- Mock paths (`v2.twitter.chat_json`, `v2.database.connection.get_cursor`)
- Context-gathering tests updated for v2 schema queries

## What stays unchanged

- `trading/twitter.py` — left in place (original pipeline still works)
- `trading/session.py` — still has its own Twitter stage
- `tests/test_twitter.py` and `tests/test_twitter_integration.py` — still test original
