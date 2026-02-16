"""Twitter integration -- Bikini Bottom Capital.

Generates and posts tweets about trading activity using Ollama
in the voice of Mr. Krabs.
"""

import logging
import os
from datetime import date
from dataclasses import dataclass, field
from typing import Optional

import tweepy

from . import db
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
    """Insert a tweet record and return its id."""
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (session_date, tweet_type, tweet_text, tweet_id, posted, error)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id, posted, error))
        return cur.fetchone()["id"]


def get_tweets_for_date(session_date: date) -> list:
    """Get all tweets for a given session date."""
    with db.get_cursor() as cur:
        cur.execute(
            "SELECT * FROM tweets WHERE session_date = %s ORDER BY created_at",
            (session_date,),
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------

def gather_tweet_context(session_date: Optional[date] = None) -> str:
    """Build a plain-text summary of today's trading session for tweet generation.

    Queries decisions, positions, active theses, latest snapshot, and latest
    strategy memo for the given date (defaults to today).

    Returns:
        Multi-line text summary suitable for passing to the LLM.
    """
    if session_date is None:
        session_date = date.today()

    sections = []

    with db.get_cursor() as cur:
        # Decisions for today
        cur.execute(
            "SELECT ticker, action, quantity, reasoning FROM decisions WHERE date = %s ORDER BY id",
            (session_date,),
        )
        decisions = cur.fetchall()
        if decisions:
            lines = ["TODAY'S DECISIONS:"]
            for d in decisions:
                lines.append(f"  {d['action'].upper()} {d['quantity']} {d['ticker']} - {d['reasoning']}")
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

        # Latest strategy memo
        cur.execute(
            "SELECT content FROM strategy_memos ORDER BY created_at DESC LIMIT 1"
        )
        memo = cur.fetchone()
        if memo:
            sections.append(f"STRATEGY MEMO:\n  {memo['content']}")

    if not sections:
        return "No trading activity today."

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Tweet generation (Ollama)
# ---------------------------------------------------------------------------

MR_KRABS_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running a hedge fund called Bikini Bottom Capital.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about P&L — ecstatic about gains, devastated about losses
- Paranoid that competitors (especially Plankton) are trying to steal your secret trading formula
- Refer to your portfolio as "me treasure" and losses as "money overboard"
- Occasionally mention your crew (SpongeBob, Squidward) in trading context

Generate tweets based on the trading session context provided. Each tweet must be a standalone post suitable for Twitter/X.

Respond with JSON in this exact format:
{"tweets": [{"text": "tweet text here", "type": "recap|trade|thesis|commentary"}]}

Rules:
- Keep each tweet under 280 characters
- Make them entertaining but grounded in the actual trading data
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning tickers
- Vary the tweet types: session recaps, individual trade callouts, thesis commentary, market color
- Generate 2-4 tweets per session"""


def generate_tweets(context: str) -> list[dict]:
    """Generate tweets from session context using Ollama.

    Args:
        context: Plain-text session summary from gather_tweet_context().

    Returns:
        List of dicts with 'text' and 'type' keys. Empty list on error.
    """
    try:
        result = chat_json(
            prompt=context,
            system=MR_KRABS_SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error("Failed to generate tweets: %s", e)
        return []

    tweets = result.get("tweets")
    if not tweets or not isinstance(tweets, list):
        logger.warning("LLM returned no tweets or malformed response: %s", result)
        return []

    # Enforce 280-char limit and validate structure
    cleaned = []
    for t in tweets:
        if not isinstance(t, dict) or "text" not in t:
            continue
        text = t["text"][:280]
        tweet_type = t.get("type", "commentary")
        cleaned.append({"text": text, "type": tweet_type})

    return cleaned


# ---------------------------------------------------------------------------
# Twitter client + posting
# ---------------------------------------------------------------------------

def get_twitter_client() -> Optional[tweepy.Client]:
    """Create a tweepy Client from environment variables.

    Required env vars:
        TWITTER_API_KEY, TWITTER_API_SECRET,
        TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET

    Returns:
        tweepy.Client if all credentials are present, None otherwise.
    """
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


def post_tweets(tweets: list[dict]) -> list[dict]:
    """Post a list of tweets via the Twitter API.

    Args:
        tweets: List of dicts with at least a 'text' key.

    Returns:
        List of result dicts with 'text', 'posted', 'tweet_id', 'error' fields.
    """
    client = get_twitter_client()
    if client is None:
        return [
            {"text": t["text"], "posted": False, "tweet_id": None, "error": "No Twitter credentials"}
            for t in tweets
        ]

    results = []
    for tweet in tweets:
        try:
            response = client.create_tweet(text=tweet["text"])
            tweet_id = str(response.data["id"])
            results.append({
                "text": tweet["text"],
                "posted": True,
                "tweet_id": tweet_id,
                "error": None,
            })
            logger.info("Posted tweet %s: %s...", tweet_id, tweet["text"][:50])
        except Exception as e:
            logger.error("Failed to post tweet: %s", e)
            results.append({
                "text": tweet["text"],
                "posted": False,
                "tweet_id": None,
                "error": str(e),
            })

    return results


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
    """Run the full tweet pipeline: context -> generate -> post -> log.

    Args:
        session_date: Date for context gathering (defaults to today).

    Returns:
        TwitterStageResult with counts and error info.
    """
    if session_date is None:
        session_date = date.today()

    result = TwitterStageResult()

    # Check credentials early
    if get_twitter_client() is None:
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
        post_results = post_tweets(tweets)
    except Exception as e:
        result.errors.append(f"Tweet posting failed: {e}")
        logger.error("Failed to post tweets: %s", e)
        return result

    # Log results to DB
    for pr in post_results:
        tweet_type = "commentary"
        # Find the matching tweet type from generated tweets
        for t in tweets:
            if t["text"] == pr["text"]:
                tweet_type = t.get("type", "commentary")
                break

        try:
            insert_tweet(
                session_date=session_date,
                tweet_type=tweet_type,
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
