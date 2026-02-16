# Twitter Integration (Bikini Bottom Capital) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Stage 5 to the daily session orchestrator that generates tweets in Mr. Krabs' voice using Ollama and posts them to X via tweepy.

**Architecture:** New `trading/twitter.py` module with DB queries for session data, Ollama prompt for tweet generation, and tweepy for posting. Wired into `v2/session.py` as Stage 5. New `tweets` DB table logs all generated/posted tweets. Gracefully skips if Twitter credentials are missing.

**Tech Stack:** tweepy (X API client), trading.ollama.chat_json (local LLM), psycopg2 (DB)

---

### Task 1: DB migration — tweets table

**Files:**
- Create: `db/init/008_tweets.sql`

**Step 1: Write the migration**

```sql
-- 008_tweets.sql: Tweet log for Bikini Bottom Capital

CREATE TABLE IF NOT EXISTS tweets (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    session_date DATE NOT NULL,
    tweet_type TEXT NOT NULL,
    tweet_text TEXT NOT NULL,
    tweet_id TEXT,
    posted BOOLEAN DEFAULT FALSE,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_tweets_session_date ON tweets(session_date DESC);
```

**Step 2: Commit**

```bash
git add db/init/008_tweets.sql
git commit -m "feat: add tweets table migration"
```

---

### Task 2: Add tweepy dependency

**Files:**
- Modify: `v2/requirements.txt`

**Step 1: Add tweepy to requirements**

Add `tweepy>=4.14.0` to `v2/requirements.txt` after the existing dependencies.

**Step 2: Commit**

```bash
git add v2/requirements.txt
git commit -m "feat: add tweepy dependency"
```

---

### Task 3: DB functions for tweets

**Files:**
- Modify: `v2/database/trading_db.py`
- Test: `tests/test_db.py` (or `tests/test_twitter.py` — new file)

**Step 1: Write the failing tests**

Create `tests/test_twitter.py`. Test the two DB functions: `insert_tweet` and `get_tweets_for_date`.

```python
"""Tests for trading/twitter.py — Twitter integration (Bikini Bottom Capital)."""

from datetime import date
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest


# ---------------------------------------------------------------------------
# DB function tests
# ---------------------------------------------------------------------------

class TestInsertTweet:

    def test_inserts_tweet_row(self, mock_db):
        from trading.twitter import insert_tweet
        mock_db.fetchone.return_value = {"id": 42}

        result = insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="recap",
            tweet_text="Arr, me portfolio be lookin' fine!",
            tweet_id="12345",
            posted=True,
        )

        assert result == 42
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO tweets" in sql

    def test_inserts_tweet_with_error(self, mock_db):
        from trading.twitter import insert_tweet
        mock_db.fetchone.return_value = {"id": 43}

        result = insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="recap",
            tweet_text="Arr!",
            posted=False,
            error="API rate limited",
        )

        assert result == 43
        params = mock_db.execute.call_args[0][1]
        assert params[4] is False
        assert params[5] == "API rate limited"


class TestGetTweetsForDate:

    def test_returns_tweets(self, mock_db):
        from trading.twitter import get_tweets_for_date
        mock_db.fetchall.return_value = [
            {"id": 1, "tweet_text": "Arr!", "posted": True},
        ]

        result = get_tweets_for_date(date(2026, 2, 15))

        assert len(result) == 1
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "session_date" in sql

    def test_returns_empty_list(self, mock_db):
        from trading.twitter import get_tweets_for_date
        mock_db.fetchall.return_value = []

        result = get_tweets_for_date(date(2026, 2, 15))
        assert result == []
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_twitter.py -v`
Expected: FAIL — `trading.twitter` does not exist yet

**Step 3: Write minimal implementation**

Create `trading/twitter.py` with the DB functions:

```python
"""Twitter integration — Bikini Bottom Capital.

Generates and posts tweets about trading activity using Ollama
in the voice of Mr. Krabs.
"""

import logging
import os
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

from .db import get_cursor
from .ollama import chat_json

logger = logging.getLogger("twitter")


# ---------------------------------------------------------------------------
# DB functions
# ---------------------------------------------------------------------------

def insert_tweet(
    session_date: date,
    tweet_type: str,
    tweet_text: str,
    tweet_id: Optional[str] = None,
    posted: bool = False,
    error: Optional[str] = None,
) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (session_date, tweet_type, tweet_text, tweet_id, posted, error)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id, posted, error))
        return cur.fetchone()["id"]


def get_tweets_for_date(session_date: date) -> list:
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM tweets WHERE session_date = %s ORDER BY created_at",
            (session_date,),
        )
        return cur.fetchall()
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_twitter.py::TestInsertTweet tests/test_twitter.py::TestGetTweetsForDate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading/twitter.py tests/test_twitter.py
git commit -m "feat: add tweet DB functions with tests"
```

---

### Task 4: gather_tweet_context

**Files:**
- Modify: `trading/twitter.py`
- Test: `tests/test_twitter.py`

**Step 1: Write the failing tests**

Add to `tests/test_twitter.py`:

```python
class TestGatherTweetContext:

    def test_builds_context_string(self, mock_db):
        from trading.twitter import gather_tweet_context

        # Mock the DB calls in sequence
        mock_db.fetchall.side_effect = [
            # get_todays_decisions
            [
                {"ticker": "AAPL", "action": "buy", "quantity": 10,
                 "price": 185.50, "reasoning": "Earnings beat"},
                {"ticker": "MSFT", "action": "hold", "quantity": None,
                 "price": 420.00, "reasoning": "Waiting for entry"},
            ],
            # get_positions
            [
                {"ticker": "AAPL", "shares": 10, "avg_cost": 185.50},
                {"ticker": "NVDA", "shares": 5, "avg_cost": 890.00},
            ],
            # get_active_theses
            [
                {"ticker": "AAPL", "direction": "long", "thesis": "Strong fundamentals", "confidence": "high"},
            ],
        ]
        mock_db.fetchone.side_effect = [
            # get_latest_snapshot
            {"portfolio_value": 150000, "cash": 100000, "buying_power": 200000},
            # get_strategy_memo (most recent)
            {"content": "Good session, signals are working"},
        ]

        context = gather_tweet_context()

        assert "AAPL" in context
        assert "buy" in context.lower()
        assert "150000" in context or "150,000" in context
        assert isinstance(context, str)
        assert len(context) > 0

    def test_handles_no_decisions(self, mock_db):
        from trading.twitter import gather_tweet_context

        mock_db.fetchall.side_effect = [[], [], []]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": 150000, "cash": 100000, "buying_power": 200000},
            None,
        ]

        context = gather_tweet_context()

        assert "no" in context.lower() or "quiet" in context.lower() or len(context) > 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_twitter.py::TestGatherTweetContext -v`
Expected: FAIL — `gather_tweet_context` does not exist

**Step 3: Write implementation**

Add to `trading/twitter.py`:

```python
def gather_tweet_context() -> str:
    """Build a text summary of today's session for tweet generation."""
    lines = []

    with get_cursor() as cur:
        # Today's decisions
        cur.execute("""
            SELECT ticker, action, quantity, price, reasoning
            FROM decisions WHERE date = CURRENT_DATE
            ORDER BY id
        """)
        decisions = cur.fetchall()

        # Current positions
        cur.execute("SELECT ticker, shares, avg_cost FROM positions ORDER BY ticker")
        positions = cur.fetchall()

        # Active theses
        cur.execute("""
            SELECT ticker, direction, thesis, confidence
            FROM theses WHERE status = 'active' ORDER BY created_at DESC
        """)
        theses = cur.fetchall()

        # Latest snapshot
        cur.execute("""
            SELECT portfolio_value, cash, buying_power
            FROM account_snapshots ORDER BY date DESC LIMIT 1
        """)
        snapshot = cur.fetchone()

        # Latest strategy memo
        cur.execute("""
            SELECT content FROM strategy_memos
            ORDER BY created_at DESC LIMIT 1
        """)
        memo = cur.fetchone()

    if snapshot:
        lines.append(f"Portfolio value: ${snapshot['portfolio_value']:,}")
        lines.append(f"Cash: ${snapshot['cash']:,}")
        lines.append(f"Buying power: ${snapshot['buying_power']:,}")
        lines.append("")

    if decisions:
        lines.append(f"Today's decisions ({len(decisions)}):")
        for d in decisions:
            qty_str = f" {d['quantity']} shares" if d['quantity'] else ""
            price_str = f" @ ${d['price']}" if d['price'] else ""
            lines.append(f"  {d['action'].upper()} {d['ticker']}{qty_str}{price_str}: {d['reasoning']}")
        lines.append("")
    else:
        lines.append("No trades today.")
        lines.append("")

    if positions:
        lines.append(f"Current positions ({len(positions)}):")
        for p in positions:
            lines.append(f"  {p['ticker']}: {p['shares']} shares @ ${p['avg_cost']} avg")
        lines.append("")

    if theses:
        lines.append(f"Active theses ({len(theses)}):")
        for t in theses:
            lines.append(f"  {t['ticker']} ({t['direction']}, {t['confidence']}): {t['thesis'][:80]}")
        lines.append("")

    if memo:
        lines.append(f"Strategy memo: {memo['content'][:200]}")

    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_twitter.py::TestGatherTweetContext -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading/twitter.py tests/test_twitter.py
git commit -m "feat: add gather_tweet_context for tweet generation"
```

---

### Task 5: generate_tweets (Ollama + Mr. Krabs prompt)

**Files:**
- Modify: `trading/twitter.py`
- Test: `tests/test_twitter.py`

**Step 1: Write the failing tests**

Add to `tests/test_twitter.py`:

```python
class TestGenerateTweets:

    @patch("trading.twitter.chat_json")
    def test_generates_tweets_from_context(self, mock_chat):
        from trading.twitter import generate_tweets

        mock_chat.return_value = {
            "tweets": [
                {"text": "Arr, bought some AAPL today!", "type": "trade"},
                {"text": "Me portfolio be looking fine", "type": "recap"},
            ]
        }

        tweets = generate_tweets("Some session context here")

        assert len(tweets) == 2
        assert tweets[0]["text"] == "Arr, bought some AAPL today!"
        assert tweets[0]["type"] == "trade"
        mock_chat.assert_called_once()

        # Verify system prompt mentions Mr. Krabs
        call_kwargs = mock_chat.call_args
        system = call_kwargs[1].get("system") or call_kwargs.kwargs.get("system")
        assert "Mr. Krabs" in system or "Krabs" in system

    @patch("trading.twitter.chat_json")
    def test_enforces_280_char_limit(self, mock_chat):
        from trading.twitter import generate_tweets

        mock_chat.return_value = {
            "tweets": [
                {"text": "x" * 300, "type": "recap"},
                {"text": "Short tweet", "type": "trade"},
            ]
        }

        tweets = generate_tweets("context")

        # Over-280 tweets should be truncated
        for t in tweets:
            assert len(t["text"]) <= 280

    @patch("trading.twitter.chat_json")
    def test_handles_empty_response(self, mock_chat):
        from trading.twitter import generate_tweets

        mock_chat.return_value = {"tweets": []}

        tweets = generate_tweets("context")
        assert tweets == []

    @patch("trading.twitter.chat_json")
    def test_handles_malformed_response(self, mock_chat):
        from trading.twitter import generate_tweets

        mock_chat.return_value = {"unexpected": "format"}

        tweets = generate_tweets("context")
        assert tweets == []
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_twitter.py::TestGenerateTweets -v`
Expected: FAIL — `generate_tweets` does not exist

**Step 3: Write implementation**

Add to `trading/twitter.py`:

```python
KRABS_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital. Write tweets about today's trading activity in character.

You are obsessed with money, dramatic about profits and losses, use nautical language, and are deeply paranoid about anyone stealing your secret formula (your trading algorithm). Keep tweets under 280 characters.

Respond with JSON:
{
  "tweets": [
    {"text": "tweet content here", "type": "recap|trade|thesis|commentary"}
  ]
}

Write 1-3 tweets based on what's interesting in the data. If it was a quiet day, one tweet is fine. If there were notable trades or big moves, write more."""


def generate_tweets(context: str, model: str = "qwen2.5:14b") -> list[dict]:
    """Generate tweets using Ollama in Mr. Krabs' voice."""
    try:
        result = chat_json(
            prompt=f"Here is today's trading session data:\n\n{context}",
            model=model,
            system=KRABS_SYSTEM_PROMPT,
        )
    except (ValueError, Exception) as e:
        logger.error("Tweet generation failed: %s", e)
        return []

    tweets = result.get("tweets", [])
    if not isinstance(tweets, list):
        return []

    # Enforce 280 char limit
    for tweet in tweets:
        if len(tweet.get("text", "")) > 280:
            tweet["text"] = tweet["text"][:277] + "..."

    return tweets
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_twitter.py::TestGenerateTweets -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading/twitter.py tests/test_twitter.py
git commit -m "feat: add Mr. Krabs tweet generation via Ollama"
```

---

### Task 6: post_tweets (tweepy integration)

**Files:**
- Modify: `trading/twitter.py`
- Test: `tests/test_twitter.py`

**Step 1: Write the failing tests**

Add to `tests/test_twitter.py`:

```python
class TestPostTweets:

    @patch("trading.twitter.get_twitter_client")
    def test_posts_tweets_successfully(self, mock_get_client):
        from trading.twitter import post_tweets

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = {"id": "99887766"}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client

        tweets = [
            {"text": "Arr, bought AAPL!", "type": "trade"},
            {"text": "Me portfolio be green!", "type": "recap"},
        ]

        results = post_tweets(tweets)

        assert len(results) == 2
        assert results[0]["posted"] is True
        assert results[0]["tweet_id"] == "99887766"
        assert results[1]["posted"] is True
        assert mock_client.create_tweet.call_count == 2

    @patch("trading.twitter.get_twitter_client")
    def test_handles_api_error(self, mock_get_client):
        from trading.twitter import post_tweets

        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = Exception("Rate limited")
        mock_get_client.return_value = mock_client

        tweets = [{"text": "Arr!", "type": "recap"}]

        results = post_tweets(tweets)

        assert len(results) == 1
        assert results[0]["posted"] is False
        assert "Rate limited" in results[0]["error"]

    @patch("trading.twitter.get_twitter_client")
    def test_continues_after_single_failure(self, mock_get_client):
        from trading.twitter import post_tweets

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = {"id": "111"}
        mock_client.create_tweet.side_effect = [
            Exception("fail"),
            mock_response,
        ]
        mock_get_client.return_value = mock_client

        tweets = [
            {"text": "Tweet 1", "type": "trade"},
            {"text": "Tweet 2", "type": "recap"},
        ]

        results = post_tweets(tweets)

        assert results[0]["posted"] is False
        assert results[1]["posted"] is True


class TestGetTwitterClient:

    def test_returns_none_when_no_credentials(self, monkeypatch):
        from trading.twitter import get_twitter_client

        monkeypatch.delenv("TWITTER_API_KEY", raising=False)
        monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN_SECRET", raising=False)

        assert get_twitter_client() is None

    @patch("trading.twitter.tweepy")
    def test_returns_client_with_credentials(self, mock_tweepy, monkeypatch):
        from trading.twitter import get_twitter_client

        monkeypatch.setenv("TWITTER_API_KEY", "key")
        monkeypatch.setenv("TWITTER_API_SECRET", "secret")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "token")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN_SECRET", "token_secret")

        client = get_twitter_client()

        assert client is not None
        mock_tweepy.Client.assert_called_once_with(
            consumer_key="key",
            consumer_secret="secret",
            access_token="token",
            access_token_secret="token_secret",
        )
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_twitter.py::TestPostTweets tests/test_twitter.py::TestGetTwitterClient -v`
Expected: FAIL — functions don't exist

**Step 3: Write implementation**

Add to `trading/twitter.py`:

```python
import tweepy


def get_twitter_client():
    """Create tweepy client from env vars. Returns None if credentials missing."""
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

    if not all([api_key, api_secret, access_token, access_token_secret]):
        return None

    return tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )


def post_tweets(tweets: list[dict]) -> list[dict]:
    """Post tweets to X via tweepy. Returns list of results per tweet."""
    client = get_twitter_client()
    results = []

    for tweet in tweets:
        try:
            response = client.create_tweet(text=tweet["text"])
            tweet_id = response.data["id"]
            results.append({
                "text": tweet["text"],
                "type": tweet["type"],
                "tweet_id": tweet_id,
                "posted": True,
                "error": None,
            })
            logger.info("Posted tweet %s: %s...", tweet_id, tweet["text"][:50])
        except Exception as e:
            results.append({
                "text": tweet["text"],
                "type": tweet["type"],
                "tweet_id": None,
                "posted": False,
                "error": str(e),
            })
            logger.error("Failed to post tweet: %s", e)

    return results
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_twitter.py::TestPostTweets tests/test_twitter.py::TestGetTwitterClient -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading/twitter.py tests/test_twitter.py
git commit -m "feat: add tweepy posting and client factory"
```

---

### Task 7: run_twitter_stage orchestrator

**Files:**
- Modify: `trading/twitter.py`
- Test: `tests/test_twitter.py`

**Step 1: Write the failing tests**

Add to `tests/test_twitter.py`:

```python
class TestRunTwitterStage:

    @patch("trading.twitter.insert_tweet")
    @patch("trading.twitter.post_tweets")
    @patch("trading.twitter.generate_tweets")
    @patch("trading.twitter.gather_tweet_context")
    def test_happy_path(self, mock_context, mock_generate, mock_post, mock_insert):
        from trading.twitter import run_twitter_stage, TwitterStageResult

        mock_context.return_value = "session context"
        mock_generate.return_value = [
            {"text": "Arr!", "type": "recap"},
        ]
        mock_post.return_value = [
            {"text": "Arr!", "type": "recap", "tweet_id": "123", "posted": True, "error": None},
        ]
        mock_insert.return_value = 1

        result = run_twitter_stage()

        assert isinstance(result, TwitterStageResult)
        assert result.tweets_generated == 1
        assert result.tweets_posted == 1
        assert result.tweets_failed == 0
        assert result.errors == []

    @patch("trading.twitter.insert_tweet")
    @patch("trading.twitter.post_tweets")
    @patch("trading.twitter.generate_tweets")
    @patch("trading.twitter.gather_tweet_context")
    def test_post_failure_counted(self, mock_context, mock_generate, mock_post, mock_insert):
        from trading.twitter import run_twitter_stage

        mock_context.return_value = "context"
        mock_generate.return_value = [{"text": "Arr!", "type": "recap"}]
        mock_post.return_value = [
            {"text": "Arr!", "type": "recap", "tweet_id": None, "posted": False, "error": "API down"},
        ]
        mock_insert.return_value = 1

        result = run_twitter_stage()

        assert result.tweets_generated == 1
        assert result.tweets_posted == 0
        assert result.tweets_failed == 1

    @patch("trading.twitter.generate_tweets")
    @patch("trading.twitter.gather_tweet_context")
    def test_no_tweets_generated(self, mock_context, mock_generate):
        from trading.twitter import run_twitter_stage

        mock_context.return_value = "context"
        mock_generate.return_value = []

        result = run_twitter_stage()

        assert result.tweets_generated == 0
        assert result.tweets_posted == 0

    @patch("trading.twitter.gather_tweet_context")
    def test_context_error_handled(self, mock_context):
        from trading.twitter import run_twitter_stage

        mock_context.side_effect = Exception("DB down")

        result = run_twitter_stage()

        assert result.tweets_generated == 0
        assert len(result.errors) == 1
        assert "DB down" in result.errors[0]

    @patch("trading.twitter.get_twitter_client")
    def test_skips_when_no_credentials(self, mock_get_client):
        from trading.twitter import run_twitter_stage

        mock_get_client.return_value = None

        result = run_twitter_stage()

        assert result.tweets_generated == 0
        assert result.tweets_posted == 0
        assert result.skipped is True
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_twitter.py::TestRunTwitterStage -v`
Expected: FAIL — `run_twitter_stage` and `TwitterStageResult` don't exist

**Step 3: Write implementation**

Add to `trading/twitter.py`:

```python
@dataclass
class TwitterStageResult:
    tweets_generated: int = 0
    tweets_posted: int = 0
    tweets_failed: int = 0
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


def run_twitter_stage() -> TwitterStageResult:
    """Run the full tweet generation and posting pipeline."""
    result = TwitterStageResult()

    # Check credentials first
    if get_twitter_client() is None:
        logger.info("Twitter credentials not set — skipping")
        result.skipped = True
        return result

    # Gather context
    try:
        context = gather_tweet_context()
    except Exception as e:
        result.errors.append(f"Context gathering failed: {e}")
        logger.error("Tweet context failed: %s", e)
        return result

    # Generate tweets
    tweets = generate_tweets(context)
    result.tweets_generated = len(tweets)

    if not tweets:
        logger.info("No tweets generated")
        return result

    # Post tweets
    post_results = post_tweets(tweets)

    # Log to DB and tally results
    today = date.today()
    for pr in post_results:
        if pr["posted"]:
            result.tweets_posted += 1
        else:
            result.tweets_failed += 1

        try:
            insert_tweet(
                session_date=today,
                tweet_type=pr["type"],
                tweet_text=pr["text"],
                tweet_id=pr.get("tweet_id"),
                posted=pr["posted"],
                error=pr.get("error"),
            )
        except Exception as e:
            result.errors.append(f"Failed to log tweet: {e}")

    return result
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_twitter.py::TestRunTwitterStage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading/twitter.py tests/test_twitter.py
git commit -m "feat: add run_twitter_stage orchestrator"
```

---

### Task 8: Wire into session orchestrator

**Files:**
- Modify: `trading/session.py` (note: wire into `trading/session.py` which is the active orchestrator; `v2/session.py` is the v2 copy)
- Test: `tests/test_session.py`

The active session orchestrator is `trading/session.py`. Check which one is actually imported/used at runtime. Both `trading/session.py` and `v2/session.py` exist — modify whichever is the primary entry point. Based on the codebase, `trading/session.py` is the one imported in tests.

**Step 1: Write the failing tests**

Add to `tests/test_session.py`:

```python
# Add to imports:
# from trading.twitter import TwitterStageResult

class TestRunSessionTwitterStage:

    def test_twitter_runs_after_trading(self, mock_session_deps):
        """Twitter stage should run after trading."""
        # mock_session_deps needs to include mock for run_twitter_stage
        result = run_session()
        mock_session_deps["run_twitter_stage"].assert_called_once()

    def test_skip_twitter(self, mock_session_deps):
        result = run_session(skip_twitter=True)
        mock_session_deps["run_twitter_stage"].assert_not_called()
        assert result.skipped_twitter is True

    def test_twitter_failure_nonfatal(self, mock_session_deps):
        mock_session_deps["run_twitter_stage"].side_effect = Exception("tweepy error")
        result = run_session()
        assert result.twitter_error == "tweepy error"
        # Should not affect has_errors for trading
        assert result.trading_result is not None
```

Update the `mock_session_deps` fixture to include `run_twitter_stage`:

```python
@pytest.fixture
def mock_session_deps():
    with patch("trading.session.check_dependencies") as mock_deps, \
         patch("trading.session.run_pipeline") as mock_pipeline, \
         patch("trading.session.run_strategist_loop") as mock_strategist, \
         patch("trading.session.run_trading_session") as mock_trading, \
         patch("trading.session.run_twitter_stage") as mock_twitter:

        mock_deps.return_value = True
        mock_pipeline.return_value = _make_pipeline_stats()
        mock_strategist.return_value = _make_ideation_result()
        mock_trading.return_value = _make_trading_result()
        mock_twitter.return_value = TwitterStageResult()

        yield {
            "check_dependencies": mock_deps,
            "run_pipeline": mock_pipeline,
            "run_strategist_loop": mock_strategist,
            "run_trading_session": mock_trading,
            "run_twitter_stage": mock_twitter,
        }
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_session.py::TestRunSessionTwitterStage -v`
Expected: FAIL — `run_twitter_stage` not imported in session.py, `skip_twitter` param doesn't exist

**Step 3: Modify `trading/session.py`**

Add import:
```python
from .twitter import TwitterStageResult, run_twitter_stage
```

Add to `SessionResult`:
```python
twitter_result: Optional[TwitterStageResult] = None
twitter_error: Optional[str] = None
skipped_twitter: bool = False
```

Add `skip_twitter` parameter to `run_session()`.

Add Stage 4 block (after current Stage 3, renumber if needed — or add as final stage):
```python
# Stage 4: Twitter (Bikini Bottom Capital)
if skip_twitter:
    logger.info("[Stage 4] Twitter — SKIPPED")
    result.skipped_twitter = True
else:
    logger.info("[Stage 4] Posting to Twitter")
    try:
        result.twitter_result = run_twitter_stage()
    except Exception as e:
        result.twitter_error = str(e)
        logger.error("Twitter stage failed: %s", e)
```

Add `--skip-twitter` to argparse in `main()`.

**Important:** `twitter_error` should NOT be included in `has_errors` — tweet failure should not cause `sys.exit(1)`. Keep `has_errors` checking only pipeline, strategist, and trading errors.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_session.py -v`
Expected: PASS (all existing + new tests)

**Step 5: Commit**

```bash
git add trading/session.py tests/test_session.py
git commit -m "feat: wire Twitter stage into session orchestrator"
```

---

### Task 9: Run full test suite

**Step 1: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass (existing + new twitter tests)

**Step 2: Run with coverage**

Run: `python3 -m pytest tests/ --cov=trading --cov=dashboard`
Expected: Coverage should remain at or above 89%

---

### Task 10: Final commit and summary

**Step 1: Verify git status is clean**

Run: `git status`
Expected: Clean working tree

**Step 2: Review the full diff**

Run: `git log --oneline -10`
Expected: Series of commits for each task above
