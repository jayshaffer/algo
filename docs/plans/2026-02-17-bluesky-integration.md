# Bluesky Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Bluesky as a second social media platform alongside Twitter, with independently generated content for each platform.

**Architecture:** New `v2/bluesky.py` module mirrors `v2/twitter.py`. Both the session pipeline and entertainment pipeline call Twitter then Bluesky independently. Posts are logged to the same `tweets` table with a new `platform` column.

**Tech Stack:** `atproto` (Bluesky Python SDK), Claude Haiku (LLM generation), psycopg2 (DB)

---

### Task 1: Database migration — add platform column

**Files:**
- Create: `db/init/009_tweets_platform.sql`
- Modify: `v2/database/trading_db.py:510-518` (insert_tweet function)

**Step 1: Write the migration SQL**

Create `db/init/009_tweets_platform.sql`:
```sql
-- 009_tweets_platform.sql: Add platform column to tweets table
ALTER TABLE tweets ADD COLUMN IF NOT EXISTS platform TEXT NOT NULL DEFAULT 'twitter';
CREATE INDEX IF NOT EXISTS idx_tweets_platform ON tweets(platform);
```

**Step 2: Write failing test for insert_tweet with platform param**

Add to `tests/v2/test_twitter.py` in `TestInsertTweet`:
```python
def test_insert_tweet_with_platform(self, mock_db):
    mock_db.fetchone.return_value = {"id": 1}
    result = insert_tweet(
        session_date=date(2026, 2, 15),
        tweet_type="recap",
        tweet_text="Ahoy!",
        platform="bluesky",
    )
    assert result == 1
    params = mock_db.execute.call_args[0][1]
    assert "bluesky" in params
```

**Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestInsertTweet::test_insert_tweet_with_platform -v`
Expected: FAIL — `insert_tweet() got an unexpected keyword argument 'platform'`

**Step 4: Update insert_tweet to accept platform**

In `v2/database/trading_db.py`, update `insert_tweet`:
```python
def insert_tweet(session_date, tweet_type, tweet_text, tweet_id=None, posted=False, error=None, platform="twitter") -> int:
    """Insert a tweet record and return its id."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (session_date, tweet_type, tweet_text, tweet_id, posted, error, platform)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id, posted, error, platform))
        return cur.fetchone()["id"]
```

**Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_twitter.py::TestInsertTweet -v`
Expected: All pass. Existing tests still pass because `platform` defaults to `"twitter"`.

**Step 6: Verify existing insert_tweet tests still pass with new param tuple size**

The existing tests assert `params == (date(2026, 2, 15), "recap", "Ahoy! Great day for me treasure!", None, False, None)`. These will break because the tuple now has 7 elements. Update the three existing assertions:

In `test_insert_tweet_basic`: change expected params to include `"twitter"` at the end:
```python
assert params == (date(2026, 2, 15), "recap", "Ahoy! Great day for me treasure!", None, False, None, "twitter")
```

In `test_insert_tweet_with_all_fields`: the assertion checks `params[3]` and `params[4]` by index — these are fine, no change needed.

In `test_insert_tweet_with_error`: the assertion checks `params[5]` — this is fine, no change needed.

Run: `python3 -m pytest tests/v2/test_twitter.py::TestInsertTweet -v`
Expected: All 4 tests pass.

**Step 7: Commit**

```bash
git add db/init/009_tweets_platform.sql v2/database/trading_db.py tests/v2/test_twitter.py
git commit -m "feat: add platform column to tweets table"
```

---

### Task 2: Add atproto dependency

**Files:**
- Modify: `v2/requirements.txt`

**Step 1: Add atproto to requirements**

Add `atproto>=0.0.55` to `v2/requirements.txt` after the `tweepy` line.

**Step 2: Install the dependency**

Run: `pip install atproto>=0.0.55`

**Step 3: Commit**

```bash
git add v2/requirements.txt
git commit -m "deps: add atproto for Bluesky integration"
```

---

### Task 3: Update conftest for Bluesky mocking

**Files:**
- Modify: `tests/v2/conftest.py:30-37` (mock_db fixture)

**Step 1: Add Bluesky get_cursor mock to conftest**

In `tests/v2/conftest.py`, update the `mock_db` fixture to also patch `v2.bluesky.get_cursor`:
```python
@pytest.fixture
def mock_db(mock_cursor):
    """Patch get_cursor to yield a mock cursor."""
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("v2.database.connection.get_cursor", _get_cursor), \
         patch("v2.database.trading_db.get_cursor", _get_cursor), \
         patch("v2.database.dashboard_db.get_cursor", _get_cursor), \
         patch("v2.twitter.get_cursor", _get_cursor), \
         patch("v2.entertainment.get_cursor", _get_cursor), \
         patch("v2.bluesky.get_cursor", _get_cursor), \
         patch("v2.database.connection.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor
```

Note: This patch target won't exist yet — defer this step until after `v2/bluesky.py` is created in Task 4. Do the conftest update as the first sub-step of Task 4.

**Step 2: Commit** (combined with Task 4)

---

### Task 4: Create v2/bluesky.py — client and posting

**Files:**
- Modify: `tests/v2/conftest.py:30-37`
- Create: `v2/bluesky.py`
- Create: `tests/v2/test_bluesky.py`

**Step 1: Create the bluesky module skeleton and update conftest**

First, create `v2/bluesky.py` with imports and client function:
```python
"""Bluesky integration -- Bikini Bottom Capital (v2 pipeline).

Generates and posts to Bluesky about trading activity using Claude
in the voice of Mr. Krabs.
"""

import json
import logging
import os
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

from .claude_client import get_claude_client, _call_with_retry
from .database.connection import get_cursor
from .database.trading_db import insert_tweet

logger = logging.getLogger("bluesky")


def get_bluesky_client():
    """Create an atproto Client logged into Bluesky from env vars."""
    try:
        from atproto import Client
    except ImportError:
        logger.warning("atproto not installed — skipping Bluesky")
        return None

    handle = os.environ.get("BLUESKY_HANDLE")
    app_password = os.environ.get("BLUESKY_APP_PASSWORD")

    if not handle or not app_password:
        logger.warning("Bluesky credentials not configured — skipping")
        return None

    try:
        client = Client()
        client.login(handle, app_password)
        return client
    except Exception as e:
        logger.error("Failed to login to Bluesky: %s", e)
        return None
```

Then update `tests/v2/conftest.py` mock_db to include `v2.bluesky.get_cursor` (as described in Task 3 Step 1).

**Step 2: Write tests for get_bluesky_client**

Create `tests/v2/test_bluesky.py`:
```python
"""Tests for the Bluesky integration (Bikini Bottom Capital) on v2 pipeline."""

from datetime import date
from unittest.mock import MagicMock, patch, call

import pytest

from v2.bluesky import get_bluesky_client


class TestGetBlueskyClient:
    """Verify get_bluesky_client credential handling."""

    def test_returns_client_with_creds(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "test.bsky.social")
        monkeypatch.setenv("BLUESKY_APP_PASSWORD", "app-password-123")
        mock_atproto = MagicMock()
        mock_client = MagicMock()
        mock_atproto.Client.return_value = mock_client
        with patch.dict("sys.modules", {"atproto": mock_atproto}):
            client = get_bluesky_client()
        assert client is mock_client
        mock_client.login.assert_called_once_with("test.bsky.social", "app-password-123")

    def test_returns_none_without_creds(self, monkeypatch):
        monkeypatch.delenv("BLUESKY_HANDLE", raising=False)
        monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
        client = get_bluesky_client()
        assert client is None

    def test_returns_none_with_partial_creds(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "test.bsky.social")
        monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
        client = get_bluesky_client()
        assert client is None

    def test_returns_none_on_login_failure(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "test.bsky.social")
        monkeypatch.setenv("BLUESKY_APP_PASSWORD", "bad-password")
        mock_atproto = MagicMock()
        mock_client = MagicMock()
        mock_client.login.side_effect = Exception("Invalid credentials")
        mock_atproto.Client.return_value = mock_client
        with patch.dict("sys.modules", {"atproto": mock_atproto}):
            client = get_bluesky_client()
        assert client is None
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/v2/test_bluesky.py::TestGetBlueskyClient -v`
Expected: All 4 pass.

**Step 4: Write post_to_bluesky function**

Add to `v2/bluesky.py`:
```python
def post_to_bluesky(post: dict, client=None) -> dict:
    """Post to Bluesky via the AT Protocol."""
    if client is None:
        client = get_bluesky_client()
    if client is None:
        return {"text": post["text"], "type": post.get("type", "commentary"), "posted": False, "post_id": None, "error": "No Bluesky credentials"}

    post_type = post.get("type", "commentary")
    try:
        response = client.send_post(text=post["text"])
        post_id = response.uri
        logger.info("Posted to Bluesky %s: %s...", post_id, post["text"][:50])
        return {"text": post["text"], "type": post_type, "posted": True, "post_id": post_id, "error": None}
    except Exception as e:
        logger.error("Failed to post to Bluesky: %s", e)
        return {"text": post["text"], "type": post_type, "posted": False, "post_id": None, "error": str(e)}
```

**Step 5: Write tests for post_to_bluesky**

Add to `tests/v2/test_bluesky.py`:
```python
from v2.bluesky import post_to_bluesky


class TestPostToBluesky:
    """Verify post_to_bluesky calls atproto and handles errors."""

    @patch("v2.bluesky.get_bluesky_client")
    def test_posts_successfully(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.uri = "at://did:plc:abc/app.bsky.feed.post/123"
        mock_client.send_post.return_value = mock_response
        mock_get_client.return_value = mock_client
        post = {"text": "Hello from Bikini Bottom!", "type": "recap"}
        result = post_to_bluesky(post)
        assert result["posted"] is True
        assert result["post_id"] == "at://did:plc:abc/app.bsky.feed.post/123"
        assert result["error"] is None
        assert result["type"] == "recap"

    @patch("v2.bluesky.get_bluesky_client")
    def test_handles_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.send_post.side_effect = Exception("Rate limit")
        mock_get_client.return_value = mock_client
        post = {"text": "Post text", "type": "recap"}
        result = post_to_bluesky(post)
        assert result["posted"] is False
        assert "Rate limit" in result["error"]

    @patch("v2.bluesky.get_bluesky_client")
    def test_no_credentials(self, mock_get_client):
        mock_get_client.return_value = None
        post = {"text": "Post text", "type": "recap"}
        result = post_to_bluesky(post)
        assert result["posted"] is False
        assert "No Bluesky credentials" in result["error"]
```

**Step 6: Run tests**

Run: `python3 -m pytest tests/v2/test_bluesky.py -v`
Expected: All 7 pass (4 client + 3 posting).

**Step 7: Commit**

```bash
git add v2/bluesky.py tests/v2/test_bluesky.py tests/v2/conftest.py
git commit -m "feat: add Bluesky client and posting functions"
```

---

### Task 5: Bluesky post generation (separate LLM call)

**Files:**
- Modify: `v2/bluesky.py`
- Modify: `tests/v2/test_bluesky.py`

**Step 1: Write failing tests for generate_bluesky_post**

Add to `tests/v2/test_bluesky.py`:
```python
from v2.bluesky import generate_bluesky_post, BLUESKY_SYSTEM_PROMPT


def _make_claude_response(json_data):
    """Helper: build a mock Claude API response containing JSON text."""
    import json as _json
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=_json.dumps(json_data))]
    return mock_resp


class TestGenerateBlueskyPost:
    """Verify generate_bluesky_post calls Claude and processes response."""

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_generates_post(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({
            "text": "Ahoy! Great day for me treasure! $AAPL up big!",
        })
        result = generate_bluesky_post("test context")
        assert result is not None
        assert result["text"] == "Ahoy! Great day for me treasure! $AAPL up big!"
        assert result["type"] == "recap"
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        assert "300" in call_kwargs.kwargs.get("system", "")

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_custom_model(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Test"})
        generate_bluesky_post("context", model="claude-sonnet-4-5-20250929")
        call_kwargs = mock_retry.call_args
        assert call_kwargs.kwargs.get("model") == "claude-sonnet-4-5-20250929"

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_bluesky_post("context")
        assert result is None

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_non_string_text(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": 123})
        result = generate_bluesky_post("context")
        assert result is None

    @patch("v2.bluesky.get_claude_client")
    def test_handles_api_exception(self, mock_get_client):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_bluesky_post("context")
        assert result is None

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_markdown_fenced_json(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        fenced = '```json\n{"text": "Ahoy!"}\n```'
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_retry.return_value = mock_resp
        result = generate_bluesky_post("context")
        assert result is not None
        assert result["text"] == "Ahoy!"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_bluesky.py::TestGenerateBlueskyPost -v`
Expected: FAIL — `cannot import name 'generate_bluesky_post'`

**Step 3: Implement generate_bluesky_post**

Add to `v2/bluesky.py`:
```python
BLUESKY_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about P&L — ecstatic about gains, devastated about losses.  Avoid talking about total portfolio gain, as it doesn't reflect the cash position correctly.
- Paranoid that competitors are trying to steal your secret trading formula

Generate ONE post that summarizes today's trading session. Condense all trades, P&L, and portfolio status into a single punchy recap.

Respond with JSON in this exact format:
{"text": "post text here"}

Rules:
- Make it entertaining but grounded in the actual trading data
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning tickers
- Summarize the full session: trades made, P&L result, and overall portfolio vibe
- If it was a quiet day with no trades, comment on holding steady
- Maximum 300 characters (Bluesky's limit)"""


def generate_bluesky_post(context: str, model: str = "claude-haiku-4-5-20251001") -> dict | None:
    """Generate a single Bluesky post from session context using Claude."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=BLUESKY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text.strip()
        logger.info("AI response:\n%s", text)
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        result = json.loads(text)
    except Exception as e:
        logger.error("Failed to generate Bluesky post: %s", e)
        return None

    post_text = result.get("text")
    if not post_text or not isinstance(post_text, str):
        logger.warning("LLM returned no post or malformed response: %s", result)
        return None

    return {"text": post_text, "type": "recap"}
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_bluesky.py -v`
Expected: All 13 pass.

**Step 5: Commit**

```bash
git add v2/bluesky.py tests/v2/test_bluesky.py
git commit -m "feat: add Bluesky post generation with Claude"
```

---

### Task 6: Bluesky stage orchestrator

**Files:**
- Modify: `v2/bluesky.py`
- Modify: `tests/v2/test_bluesky.py`

**Step 1: Write tests for BlueskyStageResult and run_bluesky_stage**

Add to `tests/v2/test_bluesky.py`:
```python
from v2.bluesky import run_bluesky_stage, BlueskyStageResult


class TestBlueskyStageResult:
    """Verify dataclass defaults."""

    def test_defaults(self):
        r = BlueskyStageResult()
        assert r.post_posted is False
        assert r.skipped is False
        assert r.errors == []

    def test_mutable_default(self):
        r1 = BlueskyStageResult()
        r2 = BlueskyStageResult()
        r1.errors.append("test")
        assert r2.errors == []


class TestRunBlueskyStage:
    """Verify run_bluesky_stage orchestration."""

    @patch("v2.bluesky.insert_tweet")
    @patch("v2.bluesky.post_to_bluesky")
    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_happy_path(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "Today we bought AAPL"
        mock_generate.return_value = {"text": "Ahoy! Bought $AAPL!", "type": "recap"}
        mock_post.return_value = {
            "text": "Ahoy! Bought $AAPL!", "type": "recap", "posted": True,
            "post_id": "at://did:plc:abc/app.bsky.feed.post/123", "error": None,
        }
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is True
        assert result.skipped is False
        assert result.errors == []
        mock_insert.assert_called_once()
        # Verify platform='bluesky' is passed to insert_tweet
        call_kwargs = mock_insert.call_args
        assert call_kwargs.kwargs.get("platform") == "bluesky" or \
               (call_kwargs[1].get("platform") == "bluesky" if len(call_kwargs) > 1 else False)

    @patch("v2.bluesky.get_bluesky_client")
    def test_skips_without_credentials(self, mock_client):
        mock_client.return_value = None
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.skipped is True
        assert result.post_posted is False

    @patch("v2.bluesky.insert_tweet")
    @patch("v2.bluesky.post_to_bluesky")
    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_post_failure(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Post text", "type": "recap"}
        mock_post.return_value = {
            "text": "Post text", "type": "recap", "posted": False, "post_id": None, "error": "Rate limit",
        }
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is False

    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_no_post_generated(self, mock_client, mock_context, mock_generate):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "No trading activity today."
        mock_generate.return_value = None
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is False

    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_context_error_handled(self, mock_client, mock_context):
        mock_client.return_value = MagicMock()
        mock_context.side_effect = Exception("DB connection failed")
        result = run_bluesky_stage(date(2026, 2, 15))
        assert len(result.errors) == 1
        assert "Context gathering failed" in result.errors[0]

    @patch("v2.bluesky.insert_tweet")
    @patch("v2.bluesky.post_to_bluesky")
    @patch("v2.bluesky.generate_bluesky_post")
    @patch("v2.bluesky.gather_tweet_context")
    @patch("v2.bluesky.get_bluesky_client")
    def test_db_log_error_does_not_crash(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = {"text": "Post", "type": "recap"}
        mock_post.return_value = {
            "text": "Post", "type": "recap", "posted": True,
            "post_id": "at://did:plc:abc/app.bsky.feed.post/123", "error": None,
        }
        mock_insert.side_effect = Exception("DB write failed")
        result = run_bluesky_stage(date(2026, 2, 15))
        assert result.post_posted is True
        assert len(result.errors) == 1
        assert "Failed to log" in result.errors[0]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_bluesky.py::TestRunBlueskyStage -v`
Expected: FAIL — `cannot import name 'run_bluesky_stage'`

**Step 3: Implement BlueskyStageResult and run_bluesky_stage**

Add to `v2/bluesky.py`:
```python
from .twitter import gather_tweet_context


@dataclass
class BlueskyStageResult:
    """Result of the Bluesky posting stage."""
    post_posted: bool = False
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


def run_bluesky_stage(session_date: Optional[date] = None) -> BlueskyStageResult:
    """Run the full Bluesky pipeline: context -> generate -> post -> log."""
    if session_date is None:
        session_date = date.today()

    result = BlueskyStageResult()

    # Check credentials early
    client = get_bluesky_client()
    if client is None:
        result.skipped = True
        logger.info("Bluesky stage skipped — no credentials")
        return result

    # Gather context (reuse from twitter module)
    try:
        context = gather_tweet_context(session_date)
    except Exception as e:
        result.errors.append(f"Context gathering failed: {e}")
        logger.error("Failed to gather context: %s", e)
        return result

    # Generate post
    try:
        post = generate_bluesky_post(context)
    except Exception as e:
        result.errors.append(f"Post generation failed: {e}")
        logger.error("Failed to generate Bluesky post: %s", e)
        return result

    if not post:
        logger.info("No Bluesky post generated")
        return result

    # Post to Bluesky
    try:
        post_result = post_to_bluesky(post, client=client)
    except Exception as e:
        result.errors.append(f"Bluesky posting failed: {e}")
        logger.error("Failed to post to Bluesky: %s", e)
        return result

    # Log result to DB
    try:
        insert_tweet(
            session_date=session_date,
            tweet_type=post_result.get("type", "recap"),
            tweet_text=post_result["text"],
            tweet_id=post_result.get("post_id"),
            posted=post_result["posted"],
            error=post_result.get("error"),
            platform="bluesky",
        )
    except Exception as e:
        result.errors.append(f"Failed to log post: {e}")
        logger.error("Failed to log Bluesky post to DB: %s", e)

    result.post_posted = post_result["posted"]

    logger.info("Bluesky stage complete: posted=%s", result.post_posted)

    return result
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_bluesky.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add v2/bluesky.py tests/v2/test_bluesky.py
git commit -m "feat: add Bluesky stage orchestrator"
```

---

### Task 7: Wire Bluesky into session pipeline

**Files:**
- Modify: `v2/session.py`
- Modify: `tests/v2/test_session.py`

**Step 1: Write failing tests for Bluesky in session**

Add to `tests/v2/test_session.py`:
```python
from v2.bluesky import BlueskyStageResult


class TestStage5Bluesky:
    def test_bluesky_runs_after_twitter(self):
        """Bluesky should run after Twitter posting."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage") as mock_twitter, \
             patch("v2.session.run_bluesky_stage") as mock_bluesky:

            mock_twitter.side_effect = lambda: call_order.append("twitter")
            mock_bluesky.side_effect = lambda: call_order.append("bluesky")

            run_session(dry_run=True)

        assert call_order.index("twitter") < call_order.index("bluesky")

    def test_bluesky_result_captured(self):
        """Bluesky result should be in SessionResult."""
        mock_bluesky_result = BlueskyStageResult(post_posted=True)

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage", return_value=mock_bluesky_result):

            result = run_session(dry_run=True)

        assert result.bluesky_result is not None
        assert result.bluesky_result.post_posted is True

    def test_bluesky_failure_does_not_block(self):
        """Bluesky failure should be captured but not crash."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage", side_effect=Exception("atproto down")):

            result = run_session(dry_run=True)

        assert result.bluesky_error == "atproto down"
        assert result.bluesky_result is None

    def test_bluesky_error_in_has_errors(self):
        """Bluesky error should be included in has_errors check."""
        result = SessionResult(bluesky_error="test")
        assert result.has_errors is True

    def test_skip_bluesky_flag(self):
        """Bluesky should be skippable."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage") as mock_bluesky:

            result = run_session(dry_run=True, skip_bluesky=True)

        mock_bluesky.assert_not_called()
        assert result.skipped_bluesky is True
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_session.py::TestStage5Bluesky -v`
Expected: FAIL — `SessionResult` has no `bluesky_result` field

**Step 3: Update session.py**

Update `v2/session.py` to add Bluesky stage:

1. Add import: `from .bluesky import BlueskyStageResult, run_bluesky_stage`
2. Add fields to `SessionResult`:
   - `bluesky_result: Optional[BlueskyStageResult] = None`
   - `bluesky_error: Optional[str] = None`
   - `skipped_bluesky: bool = False`
3. Update `has_errors` to include `self.bluesky_error`
4. Add `skip_bluesky: bool = False` parameter to `run_session()`
5. Add Bluesky stage after Twitter stage:
```python
    # Stage 5b: Bluesky posting
    if skip_bluesky:
        logger.info("[Stage 5b] Bluesky posting — SKIPPED")
        result.skipped_bluesky = True
    else:
        logger.info("[Stage 5b] Running Bluesky posting")
        try:
            result.bluesky_result = run_bluesky_stage()
        except Exception as e:
            result.bluesky_error = str(e)
            logger.error("Bluesky stage failed: %s", e)
```
6. Add `bluesky_error` to the error summary loop
7. Add `--skip-bluesky` to argparse in `main()`
8. Pass `skip_bluesky=args.skip_bluesky` to `run_session()`

**Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: All pass (existing + new).

**Step 5: Also update TestSessionResult.test_default_values to include new fields**

Add to the existing assertions in `TestSessionResult.test_default_values`:
```python
assert result.bluesky_result is None
assert result.bluesky_error is None
assert result.skipped_bluesky is False
```

**Step 6: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: All pass.

**Step 7: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "feat: wire Bluesky into session pipeline as Stage 5b"
```

---

### Task 8: Wire Bluesky into entertainment pipeline

**Files:**
- Modify: `v2/bluesky.py`
- Modify: `v2/entertainment.py`
- Modify: `tests/v2/test_bluesky.py`
- Modify: `tests/v2/test_entertainment.py`

**Step 1: Add Bluesky entertainment post generation**

Add to `v2/bluesky.py`:
```python
BLUESKY_ENTERTAINMENT_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about market moves — ecstatic about green days, devastated about red
- Paranoid that competitors (especially Plankton) are trying to steal your secret trading formula
- Reference SpongeBob universe characters naturally: SpongeBob (your naive but loyal employee), Squidward (the pessimist), Patrick (the lovable idiot investor), Sandy (the quant), Plankton (your rival)

Generate ONE entertaining post based on the market news and data provided. This is NOT a session recap — it is standalone entertaining commentary meant to engage and grow your audience.

Respond with JSON in this exact format:
{"text": "post text here"}

Rules:
- Be genuinely funny and entertaining, not forced or cringe
- Ground the post in the actual market data provided — reference real tickers, real moves, real news
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning specific stocks
- Mix formats: hot takes, character interactions, market analogies, self-deprecating humor about being a crab
- Aim for a post that people want to repost or quote
- Keep it positive/constructive — smug and fun, not bitter or mean
- Pick the single most interesting thing in the data and craft the best post you can
- Maximum 300 characters (Bluesky's limit)"""


def generate_bluesky_entertainment_post(context: str, model: str = "claude-haiku-4-5-20251001") -> dict | None:
    """Generate a single entertainment post for Bluesky from market context."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=BLUESKY_ENTERTAINMENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text.strip()
        logger.info("AI response:\n%s", text)
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        result = json.loads(text)
    except Exception as e:
        logger.error("Failed to generate Bluesky entertainment post: %s", e)
        return None

    if not isinstance(result, dict) or "text" not in result:
        logger.warning("LLM returned malformed response: %s", result)
        return None

    return {"text": result["text"], "type": "entertainment"}
```

**Step 2: Write tests for generate_bluesky_entertainment_post**

Add to `tests/v2/test_bluesky.py`:
```python
from v2.bluesky import generate_bluesky_entertainment_post, BLUESKY_ENTERTAINMENT_SYSTEM_PROMPT


class TestGenerateBlueskyEntertainmentPost:
    """Verify entertainment post generation via Claude for Bluesky."""

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_generates_post(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({
            "text": "Squidward says the market is overvalued.",
        })
        result = generate_bluesky_entertainment_post("some market context")
        assert result is not None
        assert result["type"] == "entertainment"
        assert "Squidward" in result["text"]
        call_kwargs = mock_retry.call_args
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        assert "300" in call_kwargs.kwargs.get("system", "")

    @patch("v2.bluesky._call_with_retry")
    @patch("v2.bluesky.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_bluesky_entertainment_post("context")
        assert result is None

    @patch("v2.bluesky.get_claude_client")
    def test_handles_api_exception(self, mock_get_client):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_bluesky_entertainment_post("context")
        assert result is None
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/v2/test_bluesky.py -v`
Expected: All pass.

**Step 4: Update entertainment.py to call Bluesky**

Update `v2/entertainment.py`:

1. Add imports:
```python
from .bluesky import get_bluesky_client, post_to_bluesky, generate_bluesky_entertainment_post
```

2. Add fields to `EntertainmentResult`:
```python
@dataclass
class EntertainmentResult:
    """Result of the entertainment tweet pipeline."""
    posted: bool = False
    skipped: bool = False
    tweet_id: str | None = None
    error: str | None = None
    bluesky_posted: bool = False
    bluesky_post_id: str | None = None
    bluesky_error: str | None = None
```

3. Add Bluesky posting at end of `run_entertainment_pipeline()`, after Twitter DB logging (around line 183):
```python
    # --- Bluesky ---
    bluesky_client = get_bluesky_client()
    if bluesky_client is not None:
        try:
            bs_post = generate_bluesky_entertainment_post(context, model=model)
        except Exception as e:
            result.bluesky_error = f"Bluesky generation failed: {e}"
            logger.error("Failed to generate Bluesky entertainment post: %s", e)
            bs_post = None

        if bs_post is not None:
            try:
                bs_result = post_to_bluesky(bs_post, client=bluesky_client)
            except Exception as e:
                result.bluesky_error = f"Bluesky posting failed: {e}"
                logger.error("Failed to post to Bluesky: %s", e)
                bs_result = None

            if bs_result:
                result.bluesky_posted = bs_result["posted"]
                result.bluesky_post_id = bs_result.get("post_id")
                if bs_result.get("error"):
                    result.bluesky_error = bs_result["error"]

                try:
                    insert_tweet(
                        session_date=today,
                        tweet_type="entertainment",
                        tweet_text=bs_result["text"],
                        tweet_id=bs_result.get("post_id"),
                        posted=bs_result["posted"],
                        error=bs_result.get("error"),
                        platform="bluesky",
                    )
                except Exception as e:
                    logger.error("Failed to log Bluesky post to DB: %s", e)
```

**Step 5: Write tests for entertainment + Bluesky**

Add to `tests/v2/test_entertainment.py`:
```python
class TestEntertainmentBluesky:
    """Verify Bluesky integration in entertainment pipeline."""

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_to_bluesky")
    @patch("v2.entertainment.generate_bluesky_entertainment_post")
    @patch("v2.entertainment.get_bluesky_client")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_posts_to_both_platforms(self, mock_tw_client, mock_context, mock_tw_gen,
                                     mock_tw_post, mock_bs_client, mock_bs_gen,
                                     mock_bs_post, mock_insert):
        mock_tw_client.return_value = MagicMock()
        mock_bs_client.return_value = MagicMock()
        mock_context.return_value = "NEWS: NVDA up 5%"
        mock_tw_gen.return_value = {"text": "Twitter tweet!", "type": "entertainment"}
        mock_tw_post.return_value = {
            "text": "Twitter tweet!", "type": "entertainment",
            "posted": True, "tweet_id": "tw-111", "error": None,
        }
        mock_bs_gen.return_value = {"text": "Bluesky post!", "type": "entertainment"}
        mock_bs_post.return_value = {
            "text": "Bluesky post!", "type": "entertainment",
            "posted": True, "post_id": "at://abc/123", "error": None,
        }
        result = run_entertainment_pipeline()
        assert result.posted is True
        assert result.bluesky_posted is True
        assert result.bluesky_post_id == "at://abc/123"
        assert mock_insert.call_count == 2  # one for Twitter, one for Bluesky

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.get_bluesky_client")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_bluesky_skipped_without_credentials(self, mock_tw_client, mock_context,
                                                   mock_tw_gen, mock_tw_post,
                                                   mock_bs_client, mock_insert):
        mock_tw_client.return_value = MagicMock()
        mock_bs_client.return_value = None
        mock_context.return_value = "NEWS"
        mock_tw_gen.return_value = {"text": "Tweet", "type": "entertainment"}
        mock_tw_post.return_value = {
            "text": "Tweet", "type": "entertainment",
            "posted": True, "tweet_id": "111", "error": None,
        }
        result = run_entertainment_pipeline()
        assert result.posted is True
        assert result.bluesky_posted is False

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_to_bluesky")
    @patch("v2.entertainment.generate_bluesky_entertainment_post")
    @patch("v2.entertainment.get_bluesky_client")
    @patch("v2.entertainment.post_tweet")
    @patch("v2.entertainment.generate_entertainment_tweet")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_bluesky_failure_does_not_block_twitter(self, mock_tw_client, mock_context,
                                                      mock_tw_gen, mock_tw_post,
                                                      mock_bs_client, mock_bs_gen,
                                                      mock_bs_post, mock_insert):
        mock_tw_client.return_value = MagicMock()
        mock_bs_client.return_value = MagicMock()
        mock_context.return_value = "NEWS"
        mock_tw_gen.return_value = {"text": "Tweet", "type": "entertainment"}
        mock_tw_post.return_value = {
            "text": "Tweet", "type": "entertainment",
            "posted": True, "tweet_id": "111", "error": None,
        }
        mock_bs_gen.side_effect = Exception("Claude down")
        result = run_entertainment_pipeline()
        assert result.posted is True
        assert result.bluesky_posted is False
        assert "Bluesky generation failed" in result.bluesky_error
```

Also update `TestEntertainmentResult.test_defaults`:
```python
def test_defaults(self):
    r = EntertainmentResult()
    assert r.posted is False
    assert r.skipped is False
    assert r.tweet_id is None
    assert r.error is None
    assert r.bluesky_posted is False
    assert r.bluesky_post_id is None
    assert r.bluesky_error is None
```

**Step 6: Run tests**

Run: `python3 -m pytest tests/v2/test_entertainment.py -v`
Expected: All pass.

**Step 7: Commit**

```bash
git add v2/bluesky.py v2/entertainment.py tests/v2/test_bluesky.py tests/v2/test_entertainment.py
git commit -m "feat: wire Bluesky into entertainment pipeline"
```

---

### Task 9: Full regression + env setup

**Step 1: Run the full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass, no regressions.

**Step 2: Update .env.example with Bluesky vars**

Add to `.env.example` (if it exists) or note in commit:
```
BLUESKY_HANDLE=yourhandle.bsky.social
BLUESKY_APP_PASSWORD=your-app-password
```

**Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: add Bluesky env vars to .env.example"
```
