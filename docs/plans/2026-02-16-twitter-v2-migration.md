# Twitter v2 Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the Twitter posting stage (Bikini Bottom Capital / Mr. Krabs tweets) into the v2 pipeline as Stage 5.

**Architecture:** New `v2/twitter.py` module using `v2.database.connection.get_cursor` for DB, `trading.ollama.chat_json` for Ollama LLM calls, and tweepy for posting. Wired into `v2/session.py` after strategy reflection.

**Tech Stack:** Python, psycopg2, tweepy, Ollama (qwen2.5:14b), pytest

**Design doc:** `docs/plans/2026-02-16-twitter-v2-migration-design.md`

---

### Task 1: Add tweet DB functions to v2/database/trading_db.py

**Files:**
- Test: `tests/v2/test_twitter.py` (create)
- Modify: `v2/database/trading_db.py`

**Step 1: Write failing tests for insert_tweet and get_tweets_for_date**

```python
"""Tests for the Twitter integration (Bikini Bottom Capital) on v2 pipeline."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from v2.database.trading_db import insert_tweet, get_tweets_for_date


class TestInsertTweet:
    """Verify insert_tweet issues correct SQL."""

    def test_insert_tweet_basic(self, mock_db):
        result = insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="recap",
            tweet_text="Ahoy! Great day for me treasure!",
        )
        assert result == 1
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO tweets" in sql
        assert "RETURNING id" in sql
        params = mock_db.execute.call_args[0][1]
        assert params == (date(2026, 2, 15), "recap", "Ahoy! Great day for me treasure!", None, False, None)

    def test_insert_tweet_with_all_fields(self, mock_db):
        result = insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="trade",
            tweet_text="Bought more $AAPL!",
            tweet_id="123456789",
            posted=True,
            error=None,
        )
        assert result == 1
        params = mock_db.execute.call_args[0][1]
        assert params[3] == "123456789"
        assert params[4] is True

    def test_insert_tweet_with_error(self, mock_db):
        insert_tweet(
            session_date=date(2026, 2, 15),
            tweet_type="recap",
            tweet_text="Test tweet",
            error="Rate limit exceeded",
        )
        params = mock_db.execute.call_args[0][1]
        assert params[5] == "Rate limit exceeded"


class TestGetTweetsForDate:
    """Verify get_tweets_for_date issues correct SQL."""

    def test_returns_tweets(self, mock_db):
        mock_db.fetchall.return_value = [
            {"id": 1, "tweet_text": "Ahoy!", "tweet_type": "recap"},
            {"id": 2, "tweet_text": "Bought $AAPL!", "tweet_type": "trade"},
        ]
        result = get_tweets_for_date(date(2026, 2, 15))
        assert len(result) == 2
        assert result[0]["tweet_text"] == "Ahoy!"
        sql = mock_db.execute.call_args[0][0]
        assert "session_date = %s" in sql
        assert "ORDER BY created_at" in sql

    def test_returns_empty_list(self, mock_db):
        mock_db.fetchall.return_value = []
        result = get_tweets_for_date(date(2026, 2, 15))
        assert result == []
```

Create file: `tests/v2/test_twitter.py`

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestInsertTweet tests/v2/test_twitter.py::TestGetTweetsForDate -v`
Expected: FAIL with `ImportError: cannot import name 'insert_tweet'`

**Step 3: Implement insert_tweet and get_tweets_for_date**

Add to the end of `v2/database/trading_db.py`:

```python
# --- Tweets ---

def insert_tweet(session_date, tweet_type, tweet_text, tweet_id=None, posted=False, error=None) -> int:
    """Insert a tweet record and return its id."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (session_date, tweet_type, tweet_text, tweet_id, posted, error)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id, posted, error))
        return cur.fetchone()["id"]


def get_tweets_for_date(session_date) -> list:
    """Get all tweets for a given session date."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM tweets WHERE session_date = %s ORDER BY created_at",
            (session_date,),
        )
        return cur.fetchall()
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestInsertTweet tests/v2/test_twitter.py::TestGetTweetsForDate -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add tests/v2/test_twitter.py v2/database/trading_db.py
git commit -m "feat(v2): add tweet DB functions to trading_db"
```

---

### Task 2: Create v2/twitter.py — context gathering

**Files:**
- Create: `v2/twitter.py`
- Test: `tests/v2/test_twitter.py` (append)

**Step 1: Write failing tests for gather_tweet_context**

Append to `tests/v2/test_twitter.py`:

```python
from decimal import Decimal

from v2.twitter import gather_tweet_context


class TestGatherTweetContext:
    """Verify gather_tweet_context builds context string from DB data."""

    def test_full_context(self, mock_db):
        """All five queries return data."""
        mock_db.fetchall.side_effect = [
            # decisions
            [{"ticker": "AAPL", "action": "buy", "quantity": Decimal("10"), "price": Decimal("185.50"), "reasoning": "Earnings beat"}],
            # positions
            [{"ticker": "AAPL", "shares": Decimal("50"), "avg_cost": Decimal("150.00")}],
            # theses
            [{"ticker": "NVDA", "direction": "long", "thesis": "AI demand", "confidence": "high"}],
        ]
        mock_db.fetchone.side_effect = [
            # snapshot
            {"portfolio_value": Decimal("150000"), "cash": Decimal("100000"), "buying_power": Decimal("200000")},
            # strategy memo
            {"content": "Stay bullish on tech"},
        ]

        context = gather_tweet_context(date(2026, 2, 15))

        assert "BUY 10 AAPL @ $185.50" in context
        assert "Earnings beat" in context
        assert "AAPL: 50 shares @ $150.00" in context
        assert "NVDA (long, high): AI demand" in context
        assert "portfolio=$150000" in context
        assert "Stay bullish on tech" in context

    def test_decision_without_price(self, mock_db):
        """Decision with no price omits the @ clause."""
        mock_db.fetchall.side_effect = [
            [{"ticker": "AAPL", "action": "sell", "quantity": Decimal("5"), "price": None, "reasoning": "Taking profits"}],
            [],  # positions
            [],  # theses
        ]
        mock_db.fetchone.side_effect = [None, None]

        context = gather_tweet_context(date(2026, 2, 15))

        assert "SELL 5 AAPL: Taking profits" in context
        assert "@" not in context

    def test_empty_data(self, mock_db):
        """All queries return nothing."""
        mock_db.fetchall.side_effect = [[], [], []]
        mock_db.fetchone.side_effect = [None, None]

        context = gather_tweet_context(date(2026, 2, 15))
        assert context == "No trading activity today."

    def test_partial_data(self, mock_db):
        """Only some queries return data."""
        mock_db.fetchall.side_effect = [
            [],  # decisions — empty
            [{"ticker": "MSFT", "shares": Decimal("20"), "avg_cost": Decimal("400.00")}],
            [],  # theses — empty
        ]
        mock_db.fetchone.side_effect = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("100000")},
            None,  # memo — none
        ]

        context = gather_tweet_context(date(2026, 2, 15))
        assert "MSFT: 20 shares" in context
        assert "portfolio=$100000" in context
        assert "DECISIONS" not in context
        assert "THESES" not in context
        assert "STRATEGY MEMO" not in context

    def test_defaults_to_today(self, mock_db):
        """When no date passed, uses date.today()."""
        mock_db.fetchall.side_effect = [[], [], []]
        mock_db.fetchone.side_effect = [None, None]

        gather_tweet_context()

        first_call_params = mock_db.execute.call_args_list[0][0][1]
        assert first_call_params == (date.today(),)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestGatherTweetContext -v`
Expected: FAIL with `ImportError: cannot import name 'gather_tweet_context' from 'v2.twitter'`

**Step 3: Create v2/twitter.py with context gathering**

```python
"""Twitter integration -- Bikini Bottom Capital (v2 pipeline).

Generates and posts tweets about trading activity using Ollama
in the voice of Mr. Krabs.
"""

import logging
import os
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

from .database.connection import get_cursor
from .database.trading_db import insert_tweet

logger = logging.getLogger("twitter")


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

def gather_tweet_context(session_date: Optional[date] = None) -> str:
    """Build a plain-text summary of today's trading session for tweet generation.

    Queries decisions, positions, active theses, latest snapshot, and latest
    strategy memo for the given date (defaults to today).
    """
    if session_date is None:
        session_date = date.today()

    sections = []

    with get_cursor() as cur:
        # Decisions for today
        cur.execute(
            "SELECT ticker, action, quantity, price, reasoning FROM decisions WHERE date = %s ORDER BY id",
            (session_date,),
        )
        decisions = cur.fetchall()
        if decisions:
            lines = ["TODAY'S DECISIONS:"]
            for d in decisions:
                qty = d['quantity']
                if d.get('price'):
                    lines.append(f"  {d['action'].upper()} {qty} {d['ticker']} @ ${d['price']}: {d['reasoning']}")
                else:
                    lines.append(f"  {d['action'].upper()} {qty} {d['ticker']}: {d['reasoning']}")
            sections.append("\n".join(lines))

        # Current positions
        cur.execute("SELECT ticker, shares, avg_cost FROM positions ORDER BY ticker")
        positions = cur.fetchall()
        if positions:
            lines = ["CURRENT POSITIONS:"]
            for p in positions:
                lines.append(f"  {p['ticker']}: {p['shares']} shares @ ${p['avg_cost']}")
            sections.append("\n".join(lines))

        # Active theses
        cur.execute(
            "SELECT ticker, direction, thesis, confidence FROM theses WHERE status = 'active' ORDER BY created_at DESC LIMIT 10"
        )
        theses = cur.fetchall()
        if theses:
            lines = ["ACTIVE THESES:"]
            for t in theses:
                lines.append(f"  {t['ticker']} ({t['direction']}, {t['confidence']}): {t['thesis']}")
            sections.append("\n".join(lines))

        # Latest snapshot
        cur.execute(
            "SELECT portfolio_value, cash, buying_power FROM account_snapshots ORDER BY date DESC LIMIT 1"
        )
        snapshot = cur.fetchone()
        if snapshot:
            sections.append(
                f"ACCOUNT: portfolio=${snapshot['portfolio_value']}, "
                f"cash=${snapshot['cash']}, buying_power=${snapshot['buying_power']}"
            )

        # Latest strategy memo (v2 schema: has session_date, memo_type columns)
        cur.execute(
            "SELECT content FROM strategy_memos ORDER BY created_at DESC LIMIT 1"
        )
        memo = cur.fetchone()
        if memo:
            sections.append(f"STRATEGY MEMO:\n  {memo['content']}")

    if not sections:
        return "No trading activity today."

    return "\n\n".join(sections)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestGatherTweetContext -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add v2/twitter.py tests/v2/test_twitter.py
git commit -m "feat(v2): add twitter context gathering"
```

---

### Task 3: Add tweet generation and posting to v2/twitter.py

**Files:**
- Modify: `v2/twitter.py`
- Test: `tests/v2/test_twitter.py` (append)

**Step 1: Write failing tests for generate_tweets**

Append to `tests/v2/test_twitter.py`:

```python
from unittest.mock import patch

from v2.twitter import (
    generate_tweets,
    get_twitter_client,
    post_tweets,
    MR_KRABS_SYSTEM_PROMPT,
)


class TestGenerateTweets:
    """Verify generate_tweets calls Ollama and processes response."""

    @patch("v2.twitter.chat_json")
    def test_generates_tweets(self, mock_chat):
        mock_chat.return_value = {
            "tweets": [
                {"text": "Ahoy! Great day for me treasure! $AAPL up big!", "type": "recap"},
                {"text": "Bought more $NVDA. AI is the future, me boy!", "type": "trade"},
            ]
        }

        result = generate_tweets("test context")
        assert len(result) == 2
        assert result[0]["text"] == "Ahoy! Great day for me treasure! $AAPL up big!"
        assert result[0]["type"] == "recap"
        assert result[1]["type"] == "trade"

        call_kwargs = mock_chat.call_args
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", call_kwargs[1].get("system", ""))
        assert call_kwargs.kwargs.get("model") == "qwen2.5:14b"

    @patch("v2.twitter.chat_json")
    def test_custom_model(self, mock_chat):
        mock_chat.return_value = {"tweets": [{"text": "Test", "type": "recap"}]}
        generate_tweets("context", model="llama3:8b")
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs.get("model") == "llama3:8b"

    @patch("v2.twitter.chat_json")
    def test_truncates_long_tweets(self, mock_chat):
        long_text = "A" * 300
        mock_chat.return_value = {"tweets": [{"text": long_text, "type": "recap"}]}
        result = generate_tweets("context")
        assert len(result) == 1
        assert len(result[0]["text"]) == 280
        assert result[0]["text"].endswith("...")

    @patch("v2.twitter.chat_json")
    def test_handles_empty_response(self, mock_chat):
        mock_chat.return_value = {}
        result = generate_tweets("context")
        assert result == []

    @patch("v2.twitter.chat_json")
    def test_handles_malformed_tweets(self, mock_chat):
        mock_chat.return_value = {
            "tweets": [
                {"text": "Good tweet", "type": "recap"},
                {"no_text_key": "bad"},
                "not a dict",
            ]
        }
        result = generate_tweets("context")
        assert len(result) == 1
        assert result[0]["text"] == "Good tweet"

    @patch("v2.twitter.chat_json")
    def test_handles_chat_json_exception(self, mock_chat):
        mock_chat.side_effect = ValueError("Bad JSON")
        result = generate_tweets("context")
        assert result == []

    @patch("v2.twitter.chat_json")
    def test_default_type_is_commentary(self, mock_chat):
        mock_chat.return_value = {"tweets": [{"text": "Just a thought about the market"}]}
        result = generate_tweets("context")
        assert result[0]["type"] == "commentary"

    @patch("v2.twitter.chat_json")
    def test_tweets_not_a_list(self, mock_chat):
        mock_chat.return_value = {"tweets": "not a list"}
        result = generate_tweets("context")
        assert result == []


class TestGetTwitterClient:
    """Verify get_twitter_client credential handling."""

    def test_returns_client_with_creds(self, monkeypatch):
        monkeypatch.setenv("TWITTER_API_KEY", "key")
        monkeypatch.setenv("TWITTER_API_SECRET", "secret")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "token")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN_SECRET", "token_secret")

        mock_tweepy = MagicMock()
        mock_client = MagicMock()
        mock_tweepy.Client.return_value = mock_client

        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = get_twitter_client()

        assert client is mock_client
        mock_tweepy.Client.assert_called_once_with(
            consumer_key="key",
            consumer_secret="secret",
            access_token="token",
            access_token_secret="token_secret",
        )

    def test_returns_none_without_creds(self, monkeypatch):
        monkeypatch.delenv("TWITTER_API_KEY", raising=False)
        monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN_SECRET", raising=False)
        client = get_twitter_client()
        assert client is None

    def test_returns_none_with_partial_creds(self, monkeypatch):
        monkeypatch.setenv("TWITTER_API_KEY", "key")
        monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN_SECRET", raising=False)
        client = get_twitter_client()
        assert client is None


class TestPostTweets:
    """Verify post_tweets calls tweepy and handles errors."""

    @patch("v2.twitter.get_twitter_client")
    def test_posts_successfully(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = {"id": "12345"}
        mock_client.create_tweet.return_value = mock_response
        mock_get_client.return_value = mock_client

        tweets = [{"text": "Hello from Bikini Bottom!", "type": "recap"}]
        results = post_tweets(tweets)

        assert len(results) == 1
        assert results[0]["posted"] is True
        assert results[0]["tweet_id"] == "12345"
        assert results[0]["error"] is None
        assert results[0]["type"] == "recap"

    @patch("v2.twitter.get_twitter_client")
    def test_handles_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.create_tweet.side_effect = Exception("Rate limit")
        mock_get_client.return_value = mock_client

        tweets = [{"text": "Tweet 1", "type": "recap"}]
        results = post_tweets(tweets)

        assert len(results) == 1
        assert results[0]["posted"] is False
        assert "Rate limit" in results[0]["error"]

    @patch("v2.twitter.get_twitter_client")
    def test_continues_after_failure(self, mock_get_client):
        mock_client = MagicMock()
        error_response = Exception("Rate limit")
        success_response = MagicMock()
        success_response.data = {"id": "99999"}
        mock_client.create_tweet.side_effect = [error_response, success_response]
        mock_get_client.return_value = mock_client

        tweets = [
            {"text": "Tweet 1", "type": "recap"},
            {"text": "Tweet 2", "type": "trade"},
        ]
        results = post_tweets(tweets)

        assert len(results) == 2
        assert results[0]["posted"] is False
        assert results[1]["posted"] is True

    @patch("v2.twitter.get_twitter_client")
    def test_no_credentials(self, mock_get_client):
        mock_get_client.return_value = None

        tweets = [{"text": "Tweet", "type": "recap"}]
        results = post_tweets(tweets)

        assert len(results) == 1
        assert results[0]["posted"] is False
        assert "No Twitter credentials" in results[0]["error"]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestGenerateTweets tests/v2/test_twitter.py::TestGetTwitterClient tests/v2/test_twitter.py::TestPostTweets -v`
Expected: FAIL with `ImportError`

**Step 3: Add generation, client, and posting to v2/twitter.py**

Add after the `gather_tweet_context` function in `v2/twitter.py`:

```python
from trading.ollama import chat_json


# ---------------------------------------------------------------------------
# Tweet generation (Ollama)
# ---------------------------------------------------------------------------

MR_KRABS_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about P&L — ecstatic about gains, devastated about losses
- Paranoid that competitors are trying to steal your secret trading formula

Generate tweets based on the trading session context provided. Each tweet must be a standalone post suitable for Twitter/X.

Respond with JSON in this exact format:
{"tweets": [{"text": "tweet text here", "type": "recap|trade|thesis|commentary"}]}

Rules:
- Keep each tweet under 280 characters
- Make them entertaining but grounded in the actual trading data
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning tickers
- Vary the tweet types: session recaps, individual trade callouts, thesis commentary, market color
- Write 1-3 tweets based on what's interesting in the data. If it was a quiet day, one tweet is fine. If there were notable trades or big moves, write more."""


def generate_tweets(context: str, model: str = "qwen2.5:14b") -> list[dict]:
    """Generate tweets from session context using Ollama."""
    try:
        result = chat_json(
            prompt=context,
            system=MR_KRABS_SYSTEM_PROMPT,
            model=model,
        )
    except Exception as e:
        logger.error("Failed to generate tweets: %s", e)
        return []

    tweets = result.get("tweets")
    if not tweets or not isinstance(tweets, list):
        logger.warning("LLM returned no tweets or malformed response: %s", result)
        return []

    cleaned = []
    for t in tweets:
        if not isinstance(t, dict) or "text" not in t:
            continue
        text = t["text"]
        if len(text) > 280:
            text = text[:277] + "..."
        tweet_type = t.get("type", "commentary")
        cleaned.append({"text": text, "type": tweet_type})

    return cleaned


# ---------------------------------------------------------------------------
# Twitter client + posting
# ---------------------------------------------------------------------------

def get_twitter_client():
    """Create a tweepy Client from environment variables."""
    try:
        import tweepy
    except ImportError:
        logger.warning("tweepy not installed — skipping Twitter")
        return None

    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

    if not all([api_key, api_secret, access_token, access_token_secret]):
        logger.warning("Twitter credentials not configured — skipping")
        return None

    return tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )


def post_tweets(tweets: list[dict], client=None) -> list[dict]:
    """Post a list of tweets via the Twitter API."""
    if client is None:
        client = get_twitter_client()
    if client is None:
        return [
            {"text": t["text"], "type": t.get("type", "commentary"), "posted": False, "tweet_id": None, "error": "No Twitter credentials"}
            for t in tweets
        ]

    results = []
    for tweet in tweets:
        tweet_type = tweet.get("type", "commentary")
        try:
            response = client.create_tweet(text=tweet["text"])
            tweet_id = str(response.data["id"])
            results.append({
                "text": tweet["text"],
                "type": tweet_type,
                "posted": True,
                "tweet_id": tweet_id,
                "error": None,
            })
            logger.info("Posted tweet %s: %s...", tweet_id, tweet["text"][:50])
        except Exception as e:
            logger.error("Failed to post tweet: %s", e)
            results.append({
                "text": tweet["text"],
                "type": tweet_type,
                "posted": False,
                "tweet_id": None,
                "error": str(e),
            })

    return results
```

Note: the `from trading.ollama import chat_json` goes at the top of the file with the other imports.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestGenerateTweets tests/v2/test_twitter.py::TestGetTwitterClient tests/v2/test_twitter.py::TestPostTweets -v`
Expected: PASS (16 tests)

**Step 5: Commit**

```bash
git add v2/twitter.py tests/v2/test_twitter.py
git commit -m "feat(v2): add tweet generation and posting"
```

---

### Task 4: Add run_twitter_stage orchestrator to v2/twitter.py

**Files:**
- Modify: `v2/twitter.py`
- Test: `tests/v2/test_twitter.py` (append)

**Step 1: Write failing tests for run_twitter_stage and TwitterStageResult**

Append to `tests/v2/test_twitter.py`:

```python
from v2.twitter import run_twitter_stage, TwitterStageResult


class TestRunTwitterStage:
    """Verify run_twitter_stage orchestration."""

    @patch("v2.twitter.insert_tweet")
    @patch("v2.twitter.post_tweets")
    @patch("v2.twitter.generate_tweets")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_happy_path(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "Today we bought AAPL"
        mock_generate.return_value = [
            {"text": "Ahoy! Bought $AAPL!", "type": "trade"},
            {"text": "Portfolio is looking good!", "type": "recap"},
        ]
        mock_post.return_value = [
            {"text": "Ahoy! Bought $AAPL!", "type": "trade", "posted": True, "tweet_id": "111", "error": None},
            {"text": "Portfolio is looking good!", "type": "recap", "posted": True, "tweet_id": "222", "error": None},
        ]

        result = run_twitter_stage(date(2026, 2, 15))

        assert result.tweets_generated == 2
        assert result.tweets_posted == 2
        assert result.tweets_failed == 0
        assert result.skipped is False
        assert result.errors == []
        assert mock_insert.call_count == 2

    @patch("v2.twitter.get_twitter_client")
    def test_skips_without_credentials(self, mock_client):
        mock_client.return_value = None
        result = run_twitter_stage(date(2026, 2, 15))
        assert result.skipped is True
        assert result.tweets_generated == 0

    @patch("v2.twitter.insert_tweet")
    @patch("v2.twitter.post_tweets")
    @patch("v2.twitter.generate_tweets")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_post_failures_counted(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = [
            {"text": "Tweet 1", "type": "recap"},
            {"text": "Tweet 2", "type": "trade"},
        ]
        mock_post.return_value = [
            {"text": "Tweet 1", "type": "recap", "posted": True, "tweet_id": "111", "error": None},
            {"text": "Tweet 2", "type": "trade", "posted": False, "tweet_id": None, "error": "Rate limit"},
        ]

        result = run_twitter_stage(date(2026, 2, 15))

        assert result.tweets_generated == 2
        assert result.tweets_posted == 1
        assert result.tweets_failed == 1

    @patch("v2.twitter.generate_tweets")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_no_tweets_generated(self, mock_client, mock_context, mock_generate):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "No trading activity today."
        mock_generate.return_value = []
        result = run_twitter_stage(date(2026, 2, 15))
        assert result.tweets_generated == 0
        assert result.tweets_posted == 0

    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_context_error_handled(self, mock_client, mock_context):
        mock_client.return_value = MagicMock()
        mock_context.side_effect = Exception("DB connection failed")
        result = run_twitter_stage(date(2026, 2, 15))
        assert len(result.errors) == 1
        assert "Context gathering failed" in result.errors[0]

    @patch("v2.twitter.insert_tweet")
    @patch("v2.twitter.post_tweets")
    @patch("v2.twitter.generate_tweets")
    @patch("v2.twitter.gather_tweet_context")
    @patch("v2.twitter.get_twitter_client")
    def test_db_log_error_does_not_crash(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = [{"text": "Tweet", "type": "recap"}]
        mock_post.return_value = [
            {"text": "Tweet", "type": "recap", "posted": True, "tweet_id": "111", "error": None},
        ]
        mock_insert.side_effect = Exception("DB write failed")

        result = run_twitter_stage(date(2026, 2, 15))

        assert result.tweets_posted == 1
        assert len(result.errors) == 1
        assert "Failed to log tweet" in result.errors[0]


class TestTwitterStageResult:
    """Verify dataclass defaults."""

    def test_defaults(self):
        r = TwitterStageResult()
        assert r.tweets_generated == 0
        assert r.tweets_posted == 0
        assert r.tweets_failed == 0
        assert r.skipped is False
        assert r.errors == []

    def test_mutable_default(self):
        """Ensure errors list is not shared between instances."""
        r1 = TwitterStageResult()
        r2 = TwitterStageResult()
        r1.errors.append("test")
        assert r2.errors == []
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestRunTwitterStage tests/v2/test_twitter.py::TestTwitterStageResult -v`
Expected: FAIL with `ImportError`

**Step 3: Add orchestrator to v2/twitter.py**

Add to end of `v2/twitter.py`:

```python
# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class TwitterStageResult:
    """Result of the Twitter posting stage."""
    tweets_generated: int = 0
    tweets_posted: int = 0
    tweets_failed: int = 0
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


def run_twitter_stage(session_date: Optional[date] = None) -> TwitterStageResult:
    """Run the full tweet pipeline: context -> generate -> post -> log."""
    if session_date is None:
        session_date = date.today()

    result = TwitterStageResult()

    # Check credentials early
    client = get_twitter_client()
    if client is None:
        result.skipped = True
        logger.info("Twitter stage skipped — no credentials")
        return result

    # Gather context
    try:
        context = gather_tweet_context(session_date)
    except Exception as e:
        result.errors.append(f"Context gathering failed: {e}")
        logger.error("Failed to gather tweet context: %s", e)
        return result

    # Generate tweets
    try:
        tweets = generate_tweets(context)
    except Exception as e:
        result.errors.append(f"Tweet generation failed: {e}")
        logger.error("Failed to generate tweets: %s", e)
        return result

    result.tweets_generated = len(tweets)

    if not tweets:
        logger.info("No tweets generated")
        return result

    # Post tweets
    try:
        post_results = post_tweets(tweets, client=client)
    except Exception as e:
        result.errors.append(f"Tweet posting failed: {e}")
        logger.error("Failed to post tweets: %s", e)
        return result

    # Log results to DB
    for pr in post_results:
        try:
            insert_tweet(
                session_date=session_date,
                tweet_type=pr.get("type", "commentary"),
                tweet_text=pr["text"],
                tweet_id=pr.get("tweet_id"),
                posted=pr["posted"],
                error=pr.get("error"),
            )
        except Exception as e:
            result.errors.append(f"Failed to log tweet: {e}")
            logger.error("Failed to log tweet to DB: %s", e)

        if pr["posted"]:
            result.tweets_posted += 1
        else:
            result.tweets_failed += 1

    logger.info(
        "Twitter stage complete: generated=%d, posted=%d, failed=%d",
        result.tweets_generated, result.tweets_posted, result.tweets_failed,
    )

    return result
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_twitter.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add v2/twitter.py tests/v2/test_twitter.py
git commit -m "feat(v2): add twitter stage orchestrator"
```

---

### Task 5: Wire Twitter into v2/session.py as Stage 5

**Files:**
- Modify: `v2/session.py`
- Test: `tests/v2/test_session.py` (append)

**Step 1: Write failing tests for Stage 5**

Append to `tests/v2/test_session.py`:

```python
from v2.twitter import TwitterStageResult


class TestStage5Twitter:
    def test_stage_5_runs_after_strategy(self):
        """Twitter should run after strategy reflection."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection") as mock_reflect, \
             patch("v2.session.run_twitter_stage") as mock_twitter:

            mock_reflect.side_effect = lambda **kw: call_order.append("reflection")
            mock_twitter.side_effect = lambda: call_order.append("twitter")

            run_session(dry_run=True)

        assert call_order.index("reflection") < call_order.index("twitter")

    def test_stage_5_result_captured(self):
        """Twitter result should be in SessionResult."""
        mock_twitter_result = TwitterStageResult(tweets_generated=2, tweets_posted=2)

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage", return_value=mock_twitter_result):

            result = run_session(dry_run=True)

        assert result.twitter_result is not None
        assert result.twitter_result.tweets_posted == 2

    def test_stage_5_failure_does_not_block(self):
        """Twitter failure should be captured but not crash."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage", side_effect=Exception("Tweepy down")):

            result = run_session(dry_run=True)

        assert result.twitter_error == "Tweepy down"
        assert result.twitter_result is None

    def test_stage_5_error_in_has_errors(self):
        """Twitter error should be included in has_errors check."""
        result = SessionResult(twitter_error="test")
        assert result.has_errors is True

    def test_skip_twitter_flag(self):
        """Twitter should be skippable."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage") as mock_twitter:

            result = run_session(dry_run=True, skip_twitter=True)

        mock_twitter.assert_not_called()
        assert result.skipped_twitter is True
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_session.py::TestStage5Twitter -v`
Expected: FAIL (missing `twitter_result` field, `skip_twitter` parameter, etc.)

**Step 3: Wire Twitter stage into v2/session.py**

Changes to `v2/session.py`:

1. Add import at top:
```python
from .twitter import TwitterStageResult, run_twitter_stage
```

2. Add fields to `SessionResult`:
```python
    twitter_result: Optional[TwitterStageResult] = None
    twitter_error: Optional[str] = None
    skipped_twitter: bool = False
```

3. Add `twitter_error` to the `has_errors` property check.

4. Add `skip_twitter: bool = False` parameter to `run_session()`.

5. Add `skipped_twitter=skip_twitter` to the `SessionResult(...)` constructor call.

6. Add Stage 5 block after Stage 4:
```python
    # Stage 5: Twitter posting
    if skip_twitter:
        logger.info("[Stage 5] Twitter posting — SKIPPED")
        result.skipped_twitter = True
    else:
        logger.info("[Stage 5] Running Twitter posting")
        try:
            result.twitter_result = run_twitter_stage()
        except Exception as e:
            result.twitter_error = str(e)
            logger.error("Twitter stage failed: %s", e)
```

7. Add `twitter_error` to the error summary logging.

8. Add `--skip-twitter` to argparse and pass to `run_session()`.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: PASS (all tests including new Stage 5 tests)

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass (existing + new)

**Step 6: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "feat(v2): wire twitter as session stage 5"
```

---

### Task 6: Update v2/conftest.py mock_db fixture

**Files:**
- Modify: `tests/v2/conftest.py`

The `mock_db` fixture patches `v2.database.connection.get_cursor` and `v2.database.trading_db.get_cursor`. Since `v2/twitter.py` imports `get_cursor` from `v2.database.connection`, it's already covered. But we should also patch `v2.twitter.get_cursor` to be safe if the import gets cached differently.

**Step 1: Check if tests already pass with current fixture**

Run: `python3 -m pytest tests/v2/test_twitter.py -v`

If all pass, this task is done — no change needed. If `gather_tweet_context` tests fail because `get_cursor` isn't patched for `v2.twitter`, add the patch:

In `tests/v2/conftest.py`, add `patch("v2.twitter.get_cursor", _get_cursor)` to the mock_db fixture's `with` block.

**Step 2: Commit if changed**

```bash
git add tests/v2/conftest.py
git commit -m "fix(v2): patch twitter get_cursor in mock_db fixture"
```

---

### Task 7: Final verification

**Step 1: Run full test suite with coverage**

Run: `python3 -m pytest tests/ --cov=v2 --cov=trading -v`
Expected: All tests pass, v2/twitter.py has coverage from new tests

**Step 2: Verify no regressions in original pipeline**

Run: `python3 -m pytest tests/test_twitter.py tests/test_session.py -v`
Expected: All original tests still pass unchanged
