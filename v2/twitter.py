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

Generate ONE tweet that summarizes today's trading session. Condense all trades, P&L, and portfolio status into a single punchy recap.

Respond with JSON in this exact format:
{"text": "tweet text here"}

Rules:
- Make it entertaining but grounded in the actual trading data
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning tickers
- Summarize the full session: trades made, P&L result, and overall portfolio vibe
- If it was a quiet day with no trades, comment on holding steady"""


def generate_tweet(context: str, model: str = "claude-haiku-4-5-20251001") -> dict | None:
    """Generate a single summary tweet from session context using Claude."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
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
        logger.error("Failed to generate tweet: %s", e)
        return None

    tweet_text = result.get("text")
    if not tweet_text or not isinstance(tweet_text, str):
        logger.warning("LLM returned no tweet or malformed response: %s", result)
        return None

    return {"text": tweet_text, "type": "recap"}


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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class TwitterStageResult:
    """Result of the Twitter posting stage."""
    tweet_posted: bool = False
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

    # Generate tweet
    try:
        tweet = generate_tweet(context)
    except Exception as e:
        result.errors.append(f"Tweet generation failed: {e}")
        logger.error("Failed to generate tweet: %s", e)
        return result

    if not tweet:
        logger.info("No tweet generated")
        return result

    # Post tweet
    try:
        post_result = post_tweet(tweet, client=client)
    except Exception as e:
        result.errors.append(f"Tweet posting failed: {e}")
        logger.error("Failed to post tweet: %s", e)
        return result

    # Log result to DB
    try:
        insert_tweet(
            session_date=session_date,
            tweet_type=post_result.get("type", "recap"),
            tweet_text=post_result["text"],
            tweet_id=post_result.get("tweet_id"),
            posted=post_result["posted"],
            error=post_result.get("error"),
        )
    except Exception as e:
        result.errors.append(f"Failed to log tweet: {e}")
        logger.error("Failed to log tweet to DB: %s", e)

    result.tweet_posted = post_result["posted"]

    logger.info(
        "Twitter stage complete: posted=%s",
        result.tweet_posted,
    )

    return result
