# Entertainment Tweet Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a standalone pipeline that generates and posts entertaining Mr. Krabs tweets based on live market news and trends, independent of the trading session.

**Architecture:** New module `v2/entertainment.py` that fetches market context (news headlines + market movers) via existing `v2/news.py` and `v2/market_data.py`, generates entertainment-focused tweets via Claude with a dedicated system prompt, then posts and logs them using the existing `v2/twitter.py` infrastructure.

**Tech Stack:** Claude Opus (via `v2/claude_client.py`), Alpaca News + Data APIs (via existing wrappers), tweepy (via existing `post_tweets`), psycopg2 (via existing `insert_tweet`)

---

### Task 1: Add `v2.entertainment` to `mock_db` fixture

The `mock_db` fixture in `conftest.py` must patch `get_cursor` for the new module, otherwise DB calls in tests will hit real Postgres.

**Files:**
- Modify: `tests/v2/conftest.py:30-36` (add `v2.entertainment.get_cursor` patch)

**Step 1: Add the patch to `mock_db`**

In `tests/v2/conftest.py`, add `patch("v2.entertainment.get_cursor", _get_cursor)` to the existing `with` block inside `mock_db`.

**Step 2: Verify existing tests still pass**

Run: `python3 -m pytest tests/v2/test_twitter.py -v --tb=short`
Expected: All 33 tests pass, no regressions.

---

### Task 2: Test and implement `gather_market_context()`

This function fetches live news headlines and market snapshot data, then formats them into a plain-text string for the LLM.

**Files:**
- Create: `tests/v2/test_entertainment.py`
- Create: `v2/entertainment.py`

**Step 1: Write failing tests for `gather_market_context`**

```python
"""Tests for the entertainment tweet pipeline."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from v2.entertainment import gather_market_context


class TestGatherMarketContext:
    """Verify gather_market_context assembles news + market data."""

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_full_context(self, mock_news, mock_snapshot, mock_format):
        mock_news.return_value = [
            MagicMock(headline="NVDA surges on AI demand", symbols=["NVDA"]),
            MagicMock(headline="Fed holds rates steady", symbols=[]),
        ]
        mock_snapshot.return_value = MagicMock()
        mock_format.return_value = "Market Snapshot (2026-02-16 10:00):\n  SPY: +0.50%"
        context = gather_market_context()
        assert "NVDA surges on AI demand" in context
        assert "Fed holds rates steady" in context
        assert "SPY: +0.50%" in context

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_news_only(self, mock_news, mock_snapshot, mock_format):
        mock_news.return_value = [
            MagicMock(headline="Apple announces new product", symbols=["AAPL"]),
        ]
        mock_snapshot.side_effect = Exception("API down")
        context = gather_market_context()
        assert "Apple announces new product" in context
        assert "MARKET DATA" not in context

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_market_data_only(self, mock_news, mock_snapshot, mock_format):
        mock_news.side_effect = Exception("API down")
        mock_snapshot.return_value = MagicMock()
        mock_format.return_value = "Market Snapshot:\n  SPY: -1.20%"
        context = gather_market_context()
        assert "SPY: -1.20%" in context
        assert "NEWS HEADLINES" not in context

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_both_fail_returns_fallback(self, mock_news, mock_snapshot, mock_format):
        mock_news.side_effect = Exception("API down")
        mock_snapshot.side_effect = Exception("API down")
        context = gather_market_context()
        assert context == "No market data available."

    @patch("v2.entertainment.format_market_snapshot")
    @patch("v2.entertainment.get_market_snapshot")
    @patch("v2.entertainment.fetch_broad_news")
    def test_custom_news_limit(self, mock_news, mock_snapshot, mock_format):
        mock_news.return_value = []
        mock_snapshot.side_effect = Exception("skip")
        gather_market_context(news_limit=10)
        mock_news.assert_called_once_with(hours=24, limit=10)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_entertainment.py::TestGatherMarketContext -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v2.entertainment'`

**Step 3: Implement `gather_market_context`**

Create `v2/entertainment.py`:

```python
"""Entertainment tweet pipeline -- Bikini Bottom Capital.

Generates and posts entertaining Mr. Krabs tweets based on live market
news and trends, independent of the daily trading session.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from .claude_client import get_claude_client, _call_with_retry
from .news import fetch_broad_news
from .market_data import get_market_snapshot, format_market_snapshot
from .twitter import get_twitter_client, post_tweets
from .database.trading_db import insert_tweet

logger = logging.getLogger("entertainment")


def gather_market_context(news_hours: int = 24, news_limit: int = 20) -> str:
    """Fetch live news headlines and market snapshot for tweet context."""
    sections = []

    # News headlines
    try:
        news_items = fetch_broad_news(hours=news_hours, limit=news_limit)
        if news_items:
            lines = ["NEWS HEADLINES:"]
            for item in news_items:
                tickers = ", ".join(item.symbols) if item.symbols else ""
                prefix = f"  [{tickers}] " if tickers else "  "
                lines.append(f"{prefix}{item.headline}")
            sections.append("\n".join(lines))
    except Exception as e:
        logger.warning("Failed to fetch news: %s", e)

    # Market data
    try:
        snapshot = get_market_snapshot()
        formatted = format_market_snapshot(snapshot)
        sections.append(f"MARKET DATA:\n{formatted}")
    except Exception as e:
        logger.warning("Failed to fetch market data: %s", e)

    if not sections:
        return "No market data available."

    return "\n\n".join(sections)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_entertainment.py::TestGatherMarketContext -v`
Expected: All 5 pass.

**Step 5: Commit**

```bash
git add v2/entertainment.py tests/v2/test_entertainment.py tests/v2/conftest.py
git commit -m "feat(v2): add entertainment pipeline context gathering"
```

---

### Task 3: Test and implement `generate_entertainment_tweets()`

Calls Claude with an entertainment-tuned Mr. Krabs system prompt. Same JSON response format as the session pipeline but with `tweet_type="entertainment"`.

**Files:**
- Modify: `tests/v2/test_entertainment.py` (add `TestGenerateEntertainmentTweets`)
- Modify: `v2/entertainment.py` (add system prompt + `generate_entertainment_tweets`)

**Step 1: Write failing tests**

Add to `tests/v2/test_entertainment.py`:

```python
from v2.entertainment import (
    generate_entertainment_tweets,
    ENTERTAINMENT_SYSTEM_PROMPT,
)


def _make_claude_response(json_data):
    """Helper: build a mock Claude API response."""
    import json as _json
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=_json.dumps(json_data))]
    return mock_resp


class TestGenerateEntertainmentTweets:
    """Verify entertainment tweet generation via Claude."""

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_generates_tweets(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({
            "tweets": [
                {"text": "Squidward says the market is overvalued. Squidward also eats his cereal dry.", "type": "entertainment"},
            ]
        })
        result = generate_entertainment_tweets("some market context")
        assert len(result) == 1
        assert result[0]["type"] == "entertainment"
        call_kwargs = mock_retry.call_args
        assert "Mr. Krabs" in call_kwargs.kwargs.get("system", "")
        assert "entertainment" in call_kwargs.kwargs.get("system", "").lower()

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_empty_response(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({})
        result = generate_entertainment_tweets("context")
        assert result == []

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_api_exception(self, mock_get_client):
        mock_get_client.side_effect = ValueError("No API key")
        result = generate_entertainment_tweets("context")
        assert result == []

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_handles_markdown_fenced_json(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        fenced = '```json\n{"tweets": [{"text": "Ahoy!", "type": "entertainment"}]}\n```'
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_retry.return_value = mock_resp
        result = generate_entertainment_tweets("context")
        assert len(result) == 1

    @patch("v2.entertainment._call_with_retry")
    @patch("v2.entertainment.get_claude_client")
    def test_default_type_is_entertainment(self, mock_get_client, mock_retry):
        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"tweets": [{"text": "Just vibes"}]})
        result = generate_entertainment_tweets("context")
        assert result[0]["type"] == "entertainment"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_entertainment.py::TestGenerateEntertainmentTweets -v`
Expected: FAIL — `ImportError: cannot import name 'generate_entertainment_tweets'`

**Step 3: Implement the system prompt and generation function**

Add to `v2/entertainment.py`:

```python
ENTERTAINMENT_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about market moves — ecstatic about green days, devastated about red
- Paranoid that competitors (especially Plankton) are trying to steal your secret trading formula
- Reference SpongeBob universe characters naturally: SpongeBob (your naive but loyal employee), Squidward (the pessimist), Patrick (the lovable idiot investor), Sandy (the quant), Plankton (your rival)

Generate entertaining tweets based on the market news and data provided. These are NOT session recaps — they are standalone entertaining commentary meant to engage and grow your audience.

Respond with JSON in this exact format:
{"tweets": [{"text": "tweet text here", "type": "entertainment"}]}

Rules:
- Be genuinely funny and entertaining, not forced or cringe
- Ground tweets in the actual market data provided — reference real tickers, real moves, real news
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning specific stocks
- Mix formats: hot takes, character interactions, market analogies, self-deprecating humor about being a crab
- Aim for tweets that people want to bookmark or quote-tweet
- Keep it positive/constructive — smug and fun, not bitter or mean
- Write 1-3 tweets based on what's interesting in the data"""


def generate_entertainment_tweets(context: str, model: str = "claude-opus-4-6") -> list[dict]:
    """Generate entertainment tweets from market context using Claude."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=1024,
            system=ENTERTAINMENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text.strip()
        logger.info("AI response:\n%s", text)
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        result = json.loads(text)
    except Exception as e:
        logger.error("Failed to generate entertainment tweets: %s", e)
        return []

    tweets = result.get("tweets")
    if not tweets or not isinstance(tweets, list):
        logger.warning("LLM returned no tweets or malformed response: %s", result)
        return []

    cleaned = []
    for t in tweets:
        if not isinstance(t, dict) or "text" not in t:
            continue
        tweet_type = t.get("type", "entertainment")
        cleaned.append({"text": t["text"], "type": tweet_type})

    return cleaned
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_entertainment.py::TestGenerateEntertainmentTweets -v`
Expected: All 5 pass.

**Step 5: Commit**

```bash
git add v2/entertainment.py tests/v2/test_entertainment.py
git commit -m "feat(v2): add entertainment tweet generation with Mr. Krabs prompt"
```

---

### Task 4: Test and implement `run_entertainment_pipeline()` orchestrator

Wires context gathering -> tweet generation -> posting -> DB logging into a single callable, with an `EntertainmentResult` dataclass.

**Files:**
- Modify: `tests/v2/test_entertainment.py` (add `TestEntertainmentResult`, `TestRunEntertainmentPipeline`)
- Modify: `v2/entertainment.py` (add `EntertainmentResult`, `run_entertainment_pipeline`)

**Step 1: Write failing tests**

Add to `tests/v2/test_entertainment.py`:

```python
from v2.entertainment import (
    EntertainmentResult,
    run_entertainment_pipeline,
)


class TestEntertainmentResult:
    """Verify dataclass defaults."""

    def test_defaults(self):
        r = EntertainmentResult()
        assert r.tweets_generated == 0
        assert r.tweets_posted == 0
        assert r.tweets_failed == 0
        assert r.skipped is False
        assert r.errors == []

    def test_mutable_default(self):
        r1 = EntertainmentResult()
        r2 = EntertainmentResult()
        r1.errors.append("test")
        assert r2.errors == []


class TestRunEntertainmentPipeline:
    """Verify end-to-end orchestration."""

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweets")
    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_happy_path(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "NEWS: NVDA up 5%"
        mock_generate.return_value = [
            {"text": "Arg! $NVDA making me money!", "type": "entertainment"},
        ]
        mock_post.return_value = [
            {"text": "Arg! $NVDA making me money!", "type": "entertainment", "posted": True, "tweet_id": "111", "error": None},
        ]
        result = run_entertainment_pipeline()
        assert result.tweets_generated == 1
        assert result.tweets_posted == 1
        assert result.tweets_failed == 0
        assert result.errors == []
        mock_insert.assert_called_once()

    @patch("v2.entertainment.get_twitter_client")
    def test_skips_without_credentials(self, mock_client):
        mock_client.return_value = None
        result = run_entertainment_pipeline()
        assert result.skipped is True
        assert result.tweets_generated == 0

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweets")
    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_post_failures_counted(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = [
            {"text": "Tweet 1", "type": "entertainment"},
            {"text": "Tweet 2", "type": "entertainment"},
        ]
        mock_post.return_value = [
            {"text": "Tweet 1", "type": "entertainment", "posted": True, "tweet_id": "111", "error": None},
            {"text": "Tweet 2", "type": "entertainment", "posted": False, "tweet_id": None, "error": "Rate limit"},
        ]
        result = run_entertainment_pipeline()
        assert result.tweets_posted == 1
        assert result.tweets_failed == 1

    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_no_tweets_generated(self, mock_client, mock_context, mock_generate):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "No market data available."
        mock_generate.return_value = []
        result = run_entertainment_pipeline()
        assert result.tweets_generated == 0
        assert result.tweets_posted == 0

    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_context_error_handled(self, mock_client, mock_context):
        mock_client.return_value = MagicMock()
        mock_context.side_effect = Exception("Total failure")
        result = run_entertainment_pipeline()
        assert len(result.errors) == 1
        assert "Context gathering failed" in result.errors[0]

    @patch("v2.entertainment.insert_tweet")
    @patch("v2.entertainment.post_tweets")
    @patch("v2.entertainment.generate_entertainment_tweets")
    @patch("v2.entertainment.gather_market_context")
    @patch("v2.entertainment.get_twitter_client")
    def test_db_log_error_does_not_crash(self, mock_client, mock_context, mock_generate, mock_post, mock_insert):
        mock_client.return_value = MagicMock()
        mock_context.return_value = "context"
        mock_generate.return_value = [{"text": "Tweet", "type": "entertainment"}]
        mock_post.return_value = [
            {"text": "Tweet", "type": "entertainment", "posted": True, "tweet_id": "111", "error": None},
        ]
        mock_insert.side_effect = Exception("DB write failed")
        result = run_entertainment_pipeline()
        assert result.tweets_posted == 1
        assert len(result.errors) == 1
        assert "Failed to log tweet" in result.errors[0]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_entertainment.py::TestRunEntertainmentPipeline -v`
Expected: FAIL — `ImportError`

**Step 3: Implement the dataclass and orchestrator**

Add to `v2/entertainment.py`:

```python
@dataclass
class EntertainmentResult:
    """Result of the entertainment tweet pipeline."""
    tweets_generated: int = 0
    tweets_posted: int = 0
    tweets_failed: int = 0
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


def run_entertainment_pipeline(
    news_hours: int = 24,
    news_limit: int = 20,
    model: str = "claude-opus-4-6",
) -> EntertainmentResult:
    """Run the full entertainment tweet pipeline: context -> generate -> post -> log."""
    result = EntertainmentResult()
    today = date.today()

    # Check credentials early
    client = get_twitter_client()
    if client is None:
        result.skipped = True
        logger.info("Entertainment pipeline skipped — no credentials")
        return result

    # Gather market context
    try:
        context = gather_market_context(news_hours=news_hours, news_limit=news_limit)
    except Exception as e:
        result.errors.append(f"Context gathering failed: {e}")
        logger.error("Failed to gather market context: %s", e)
        return result

    # Generate tweets
    try:
        tweets = generate_entertainment_tweets(context, model=model)
    except Exception as e:
        result.errors.append(f"Tweet generation failed: {e}")
        logger.error("Failed to generate entertainment tweets: %s", e)
        return result

    result.tweets_generated = len(tweets)

    if not tweets:
        logger.info("No entertainment tweets generated")
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
                session_date=today,
                tweet_type=pr.get("type", "entertainment"),
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
        "Entertainment pipeline complete: generated=%d, posted=%d, failed=%d",
        result.tweets_generated, result.tweets_posted, result.tweets_failed,
    )

    return result
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_entertainment.py -v`
Expected: All tests pass (context, generation, orchestrator, dataclass).

**Step 5: Commit**

```bash
git add v2/entertainment.py tests/v2/test_entertainment.py
git commit -m "feat(v2): add entertainment pipeline orchestrator"
```

---

### Task 5: Add CLI entry point (`__main__`)

Make the pipeline runnable with `python -m v2.entertainment`.

**Files:**
- Modify: `v2/entertainment.py` (add `main()` function)

**Step 1: Add `main()` and `__main__` block**

Add to the bottom of `v2/entertainment.py`:

```python
def main():
    """CLI entry point for entertainment tweet pipeline."""
    from .log_config import setup_logging
    setup_logging()

    import argparse
    parser = argparse.ArgumentParser(description="Generate and post entertainment tweets")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--news-hours", type=int, default=24)
    parser.add_argument("--news-limit", type=int, default=20)
    args = parser.parse_args()

    result = run_entertainment_pipeline(
        news_hours=args.news_hours,
        news_limit=args.news_limit,
        model=args.model,
    )

    if result.skipped:
        print("Skipped — no Twitter credentials configured")
    elif result.errors:
        print(f"Completed with errors: {result.errors}")
    else:
        print(f"Done: {result.tweets_posted} posted, {result.tweets_failed} failed")


if __name__ == "__main__":
    main()
```

**Step 2: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/v2/ -v --tb=short`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add v2/entertainment.py
git commit -m "feat(v2): add entertainment pipeline CLI entry point"
```
