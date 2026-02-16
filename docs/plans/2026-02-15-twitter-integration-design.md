# Twitter Integration — Bikini Bottom Capital

## Overview

Add a Stage 5 to the daily session orchestrator that generates and posts tweets about trading activity using the local LLM (Ollama qwen2.5:14b) in the voice of Mr. Krabs running "Bikini Bottom Capital."

## Persona

The LLM is prompted as Mr. Krabs from SpongeBob SquarePants operating an algo trading desk. The system prompt provides his personality traits (money-obsessed, nautical language, dramatic, paranoid about competitors) and the session data. The model finds the voice naturally — no canned lines.

Full transparency: dollar amounts, position sizes, and tickers are all fair game.

## Architecture

```
Session Stages 0-4 complete
        │
        ▼
  gather_tweet_context()     ← queries DB: decisions, snapshot, theses, strategy memo
        │
        ▼
  Ollama qwen2.5:14b         ← Mr. Krabs system prompt + session context
        │
        ▼
  List[dict]                  ← 1-3 tweets with content + type
        │
        ▼
  tweepy.Client.create_tweet()
        │
        ▼
  Log to tweets table
```

## Components

### `trading/twitter.py`

Single module with four functions:

- **`gather_tweet_context(session_result)`** — Builds a plain-text summary of today's session from DB data: decisions made (ticker, action, quantity, price, reasoning), account snapshot (portfolio value, cash, buying power), active theses, today's strategy memo, and any notable signal attribution stats.

- **`generate_tweets(context)`** — Sends context to Ollama with the Mr. Krabs system prompt. Returns a list of 1-3 tweet dicts with `text` and `tweet_type` (recap/trade/thesis/commentary). Uses `chat_json()` for structured output.

- **`post_tweets(tweets)`** — Posts each tweet via tweepy. Returns list of results (tweet_id or error per tweet).

- **`run_twitter_stage(session_result)`** — Orchestrates the above. Returns `TwitterStageResult`. Catches all exceptions so tweet failures never affect other stages.

### `TwitterStageResult` dataclass

```python
@dataclass
class TwitterStageResult:
    tweets_generated: int
    tweets_posted: int
    tweets_failed: int
    errors: list[str]
```

### DB migration: `tweets` table

```sql
CREATE TABLE tweets (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    session_date DATE NOT NULL,
    tweet_type TEXT NOT NULL,        -- recap, trade, thesis, commentary
    tweet_text TEXT NOT NULL,
    tweet_id TEXT,                   -- from X API, null if posting failed
    posted BOOLEAN DEFAULT FALSE,
    error TEXT
);
```

### Session orchestrator changes (`v2/session.py`)

- Import `TwitterStageResult`, `run_twitter_stage`
- Add `twitter_result` and `twitter_error` fields to `SessionResult`
- Add `skip_twitter` parameter and `--skip-twitter` CLI flag
- Add Stage 5 block after Stage 4, same pattern as other stages
- If `TWITTER_API_KEY` env var is missing, silently skip (no error)

## Environment Variables

```
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_TOKEN_SECRET=...
```

Uses the X free tier (write-only, ~500 tweets/month). If credentials are absent, the stage is skipped with an info log.

## Dependencies

- `tweepy` — added to requirements

## Testing

- Mock tweepy client and Ollama calls (same patterns as existing tests)
- Test `gather_tweet_context` with various session results
- Test `generate_tweets` returns valid structure with text under 280 chars
- Test `post_tweets` handles success and failure from tweepy
- Test graceful skip when credentials are missing
- Test session orchestrator wiring (stage runs after strategy reflection)
