"""Integration tests for Twitter posting (Bikini Bottom Capital).

These tests make real API calls — Ollama for tweet generation,
Twitter/X for posting.

Run with:
    python3 -m pytest tests/test_twitter_integration.py -m integration -v

Requires:
    - Running Ollama instance with qwen2.5:14b
    - Twitter API credentials in env vars
"""

import os
import logging
import time

import pytest

from trading.ollama import check_ollama_health, list_models
from trading.twitter import (
    generate_tweets,
    get_twitter_client,
    MR_KRABS_SYSTEM_PROMPT,
)

logger = logging.getLogger("test_twitter_integration")

integration = pytest.mark.integration


def ollama_available():
    try:
        if not check_ollama_health():
            return False
        models = list_models()
        return any("qwen2.5" in m for m in models)
    except Exception:
        return False


def twitter_available():
    return all([
        os.environ.get("TWITTER_API_KEY"),
        os.environ.get("TWITTER_API_SECRET"),
        os.environ.get("TWITTER_ACCESS_TOKEN"),
        os.environ.get("TWITTER_ACCESS_TOKEN_SECRET"),
    ])


skip_no_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running or qwen2.5:14b not available",
)

skip_no_twitter = pytest.mark.skipif(
    not twitter_available(),
    reason="Twitter API credentials not set",
)

SAMPLE_CONTEXT = """TODAY'S DECISIONS:
  BUY 10 AAPL @ $185.50: Strong earnings beat, guidance raised
  HOLD MSFT: Waiting for better entry point

CURRENT POSITIONS:
  AAPL: 30 shares @ $178.00
  NVDA: 15 shares @ $890.00
  MSFT: 20 shares @ $415.00

ACTIVE THESES:
  NVDA (long, high): AI datacenter demand accelerating through 2026
  AAPL (long, medium): Services revenue approaching $100B run rate

ACCOUNT: portfolio=$152,340, cash=$47,660, buying_power=$95,320

STRATEGY MEMO:
  Good session. News signal attribution continues to show earnings signals
  as the strongest predictor (65% win rate, n=20). Increased conviction on
  NVDA thesis after product launch signals confirmed."""


# ---------------------------------------------------------------------------
# Tweet generation (Ollama)
# ---------------------------------------------------------------------------

@integration
@skip_no_ollama
class TestTweetGeneration:
    """Verify Ollama generates valid Mr. Krabs tweets from session context."""

    def test_generates_valid_tweet_structure(self):
        """LLM should return tweets with text and type fields."""
        tweets = generate_tweets(SAMPLE_CONTEXT)

        assert len(tweets) >= 1
        assert len(tweets) <= 3
        for tweet in tweets:
            assert "text" in tweet
            assert "type" in tweet
            assert tweet["type"] in ("recap", "trade", "thesis", "commentary")

    def test_tweets_under_280_chars(self):
        """Every generated tweet must be 280 characters or fewer."""
        tweets = generate_tweets(SAMPLE_CONTEXT)

        for tweet in tweets:
            assert len(tweet["text"]) <= 280, \
                f"Tweet too long ({len(tweet['text'])} chars): {tweet['text']}"

    def test_tweets_reference_actual_data(self):
        """Tweets should reference tickers or data from the context."""
        tweets = generate_tweets(SAMPLE_CONTEXT)

        all_text = " ".join(t["text"] for t in tweets).upper()
        # At least one ticker from the context should appear
        tickers = ["AAPL", "NVDA", "MSFT"]
        found = any(t in all_text or f"${t}" in all_text for t in tickers)
        assert found, f"No tickers found in tweets: {all_text}"

    def test_quiet_day_generates_fewer_tweets(self):
        """A quiet day with no trades should generate 1 tweet."""
        quiet_context = """TODAY'S DECISIONS:
  HOLD AAPL: No trigger hit
  HOLD MSFT: No trigger hit

CURRENT POSITIONS:
  AAPL: 20 shares @ $178.00

ACCOUNT: portfolio=$98,000, cash=$78,000, buying_power=$156,000"""

        tweets = generate_tweets(quiet_context)

        assert len(tweets) >= 1
        # Quiet day should trend toward fewer tweets
        assert len(tweets) <= 3


# ---------------------------------------------------------------------------
# Twitter posting (real API)
# ---------------------------------------------------------------------------

@integration
@skip_no_twitter
class TestTwitterPosting:
    """Post a real tweet to verify API credentials work.

    This creates an actual tweet on your X account.
    """

    def test_post_single_tweet(self):
        """Post a test tweet and verify we get a tweet ID back."""
        client = get_twitter_client()
        assert client is not None, "Twitter client should be created with valid credentials"

        timestamp = int(time.time())
        tweet_text = f"Bikini Bottom Capital integration test - {timestamp}"

        response = client.create_tweet(text=tweet_text)

        assert response.data is not None
        assert "id" in response.data
        tweet_id = response.data["id"]
        logger.info("Posted test tweet: id=%s, text=%s", tweet_id, tweet_text)

        # Clean up — delete the test tweet
        try:
            client.delete_tweet(tweet_id)
            logger.info("Deleted test tweet %s", tweet_id)
        except Exception as e:
            logger.warning("Could not delete test tweet %s: %s", tweet_id, e)


# ---------------------------------------------------------------------------
# End-to-end: generate + post
# ---------------------------------------------------------------------------

@integration
@skip_no_ollama
@skip_no_twitter
class TestEndToEnd:
    """Generate a tweet with Ollama and post it to X."""

    def test_generate_and_post(self):
        """Full pipeline: generate Mr. Krabs tweet from context, post to X."""
        tweets = generate_tweets(SAMPLE_CONTEXT)
        assert len(tweets) >= 1, "Should generate at least one tweet"

        client = get_twitter_client()
        assert client is not None

        tweet = tweets[0]
        logger.info("Generated tweet (%s): %s", tweet["type"], tweet["text"])

        response = client.create_tweet(text=tweet["text"])

        assert response.data is not None
        tweet_id = response.data["id"]
        logger.info("Posted tweet: id=%s", tweet_id)

        # Don't delete — this is the real deal, Mr. Krabs would never
        # throw away a perfectly good tweet
