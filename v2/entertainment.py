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
from .database.connection import get_cursor
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


# ---------------------------------------------------------------------------
# Entertainment tweet generation (Claude)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entertainment pipeline orchestrator
# ---------------------------------------------------------------------------

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
