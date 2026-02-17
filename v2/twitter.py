"""Twitter integration -- Bikini Bottom Capital (v2 pipeline).

Generates and posts tweets about trading activity using Claude (Opus)
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

        # Book status: current snapshot + P&L from history
        cur.execute(
            "SELECT date, portfolio_value, cash, buying_power, long_market_value "
            "FROM account_snapshots ORDER BY date DESC LIMIT 2"
        )
        snapshots = cur.fetchall()
        if snapshots:
            today = snapshots[0]
            portfolio = today['portfolio_value']
            cash = today['cash']
            invested = today['long_market_value'] or (portfolio - cash)
            lines = [
                "BOOK STATUS:",
                f"  Portfolio value: ${portfolio:,.2f}",
                f"  Invested: ${invested:,.2f}",
                f"  Cash: ${cash:,.2f}",
                f"  Positions: {len(positions) if positions else 0}",
            ]
            if len(snapshots) == 2:
                prev = snapshots[1]['portfolio_value']
                day_pnl = portfolio - prev
                day_pct = (day_pnl / prev * 100) if prev else 0
                sign = "+" if day_pnl >= 0 else ""
                lines.append(f"  Today's P&L: {sign}${day_pnl:,.2f} ({sign}{day_pct:.2f}%)")
            # Overall return from first snapshot
            cur.execute(
                "SELECT portfolio_value, date FROM account_snapshots ORDER BY date ASC LIMIT 1"
            )
            first = cur.fetchone()
            if first and first['portfolio_value'] and first['date'] != today['date']:
                total_pnl = portfolio - first['portfolio_value']
                total_pct = (total_pnl / first['portfolio_value'] * 100)
                sign = "+" if total_pnl >= 0 else ""
                lines.append(f"  Total return: {sign}${total_pnl:,.2f} ({sign}{total_pct:.2f}%) since {first['date']}")
            sections.append("\n".join(lines))

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

MR_KRABS_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about P&L — ecstatic about gains, devastated about losses.  Avoid talking about total portfolio gain, as it doesn't reflect the cash position correctly.
- Paranoid that competitors are trying to steal your secret trading formula

Generate tweets based on the trading session context provided. Each tweet must be a standalone post suitable for Twitter/X.

Respond with JSON in this exact format:
{"tweets": [{"text": "tweet text here", "type": "recap|trade|thesis|commentary"}]}

Rules:
- Make them entertaining but grounded in the actual trading data
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning tickers
- Vary the tweet types: session recaps, individual trade callouts, thesis commentary, market color
- Write 1-3 tweets based on what's interesting in the data. If it was a quiet day, one tweet is fine. If there were notable trades or big moves, write more."""


def generate_tweets(context: str, model: str = "claude-haiku-4-5-20251001") -> list[dict]:
    """Generate tweets from session context using Claude."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=1024,
            system=MR_KRABS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        text = response.content[0].text.strip()
        logger.info("AI response:\n%s", text)
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        result = json.loads(text)
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
        tweet_type = t.get("type", "commentary")
        cleaned.append({"text": t["text"], "type": tweet_type})

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


def post_tweet(tweet: dict, client=None) -> dict:
    """Post a single tweet via the Twitter API."""
    if client is None:
        client = get_twitter_client()
    if client is None:
        return {"text": tweet["text"], "type": tweet.get("type", "commentary"), "posted": False, "tweet_id": None, "error": "No Twitter credentials"}

    tweet_type = tweet.get("type", "commentary")
    try:
        response = client.create_tweet(text=tweet["text"])
        tweet_id = str(response.data["id"])
        logger.info("Posted tweet %s: %s...", tweet_id, tweet["text"][:50])
        return {"text": tweet["text"], "type": tweet_type, "posted": True, "tweet_id": tweet_id, "error": None}
    except Exception as e:
        logger.error("Failed to post tweet: %s", e)
        return {"text": tweet["text"], "type": tweet_type, "posted": False, "tweet_id": None, "error": str(e)}


def post_tweets(tweets: list[dict], client=None) -> list[dict]:
    """Post a list of tweets via the Twitter API."""
    if client is None:
        client = get_twitter_client()
    return [post_tweet(t, client=client) for t in tweets]


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
