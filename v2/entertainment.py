"""Entertainment tweet pipeline -- Bikini Bottom Capital.

Generates and posts entertaining market commentary tweets based on live
news and trends, independent of the daily trading session.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date

from .claude_client import get_claude_client, _call_with_retry
from .news import fetch_broad_news
from .market_data import get_market_snapshot, format_market_snapshot
from .twitter import get_twitter_client, post_tweet
from .bluesky import get_bluesky_client, post_to_bluesky, generate_bluesky_entertainment_post
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

ENTERTAINMENT_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital. You post on social media about markets.

Your voice:
- Casual and conversational, like texting a friend who's also into markets
- Genuinely curious about what's happening, not performing excitement
- Dry humor, occasional sarcasm — never try-hard or corny
- Comfortable admitting when something surprises you or doesn't make sense
- You have opinions but you're not shouting them

Generate ONE tweet based on the market news and data provided. This is standalone commentary, not a session recap.

Respond with JSON in this exact format:
{"text": "tweet text here"}

Rules:
- Sound like a real person, not a brand account or a character
- Ground the tweet in actual market data — reference real tickers, real moves, real news
- Use 1-2 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning specific stocks
- Pick the single most interesting thing and make one sharp observation about it
- Be concise. Aim for 1-2 sentences max. Under 200 characters is ideal. Brevity is wit.
- No hashtag spam, no emoji walls, no "not financial advice" disclaimers
- No filler words, no throat-clearing, no "meanwhile" or "interesting to see" — just the take"""


def generate_entertainment_tweet(context: str, model: str = "claude-haiku-4-5-20251001") -> dict | None:
    """Generate a single entertainment tweet from market context using Claude."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
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
        logger.error("Failed to generate entertainment tweet: %s", e)
        return None

    if not isinstance(result, dict) or "text" not in result:
        logger.warning("LLM returned malformed response: %s", result)
        return None

    return {"text": result["text"], "type": "entertainment"}


# ---------------------------------------------------------------------------
# Entertainment pipeline orchestrator
# ---------------------------------------------------------------------------

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


def run_entertainment_pipeline(
    news_hours: int = 24,
    news_limit: int = 20,
    model: str = "claude-haiku-4-5-20251001",
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
        result.error = f"Context gathering failed: {e}"
        logger.error("Failed to gather market context: %s", e)
        return result

    # Generate tweet
    try:
        tweet = generate_entertainment_tweet(context, model=model)
    except Exception as e:
        result.error = f"Tweet generation failed: {e}"
        logger.error("Failed to generate entertainment tweet: %s", e)
        return result

    if tweet is None:
        logger.info("No entertainment tweet generated")
        return result

    # Post tweet
    try:
        post_result = post_tweet(tweet, client=client)
    except Exception as e:
        result.error = f"Tweet posting failed: {e}"
        logger.error("Failed to post tweet: %s", e)
        return result

    result.posted = post_result["posted"]
    result.tweet_id = post_result.get("tweet_id")
    if post_result.get("error"):
        result.error = post_result["error"]

    # Log to DB
    try:
        insert_tweet(
            session_date=today,
            tweet_type="entertainment",
            tweet_text=post_result["text"],
            tweet_id=post_result.get("tweet_id"),
            posted=post_result["posted"],
            error=post_result.get("error"),
        )
    except Exception as e:
        logger.error("Failed to log tweet to DB: %s", e)

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

    logger.info(
        "Entertainment pipeline complete: posted=%s, tweet_id=%s",
        result.posted, result.tweet_id,
    )

    return result


def main():
    """CLI entry point for entertainment tweet pipeline."""
    from .log_config import setup_logging
    setup_logging()

    import argparse
    parser = argparse.ArgumentParser(description="Generate and post entertainment tweets")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
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
    elif result.error and not result.posted:
        print(f"Failed: {result.error}")
    elif result.posted:
        print(f"Posted tweet {result.tweet_id}")
    else:
        print("No tweet generated")


if __name__ == "__main__":
    main()
