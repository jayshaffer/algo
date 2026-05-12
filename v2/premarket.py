"""Pre-market social post pipeline -- Bikini Bottom Capital (v2).

Forward-looking take posted before the next session. Triggered by cron
(see Taskfile premarket target), separate from the daily session.
Skipped on weekends and NYSE holidays.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

from .claude_client import _call_with_retry, get_claude_client
from .database.connection import get_cursor
from .database.trading_db import insert_tweet, posted_tweet_exists
from .market_calendar import is_trading_day

logger = logging.getLogger("premarket")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def gather_premarket_context(today: date | None = None) -> str:
    """Plain-text summary of what the bot is watching pre-market.

    Sections:
    - Top 5 active theses by confidence
    - Latest strategy memo (yesterday's reflection)
    """
    if today is None:
        today = date.today()

    sections: list[str] = []
    with get_cursor() as cur:
        cur.execute(
            "SELECT ticker, direction, thesis, confidence FROM theses "
            "WHERE status = 'active' "
            "ORDER BY CASE confidence "
            "  WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
            "  created_at DESC LIMIT 5"
        )
        theses = cur.fetchall()
        if theses:
            lines = ["ACTIVE THESES:"]
            for t in theses:
                lines.append(
                    f"  {t['ticker']} ({t['direction']}, {t['confidence']}): {t['thesis']}"
                )
            sections.append("\n".join(lines))

        cur.execute("SELECT content FROM strategy_memos ORDER BY created_at DESC LIMIT 1")
        memo = cur.fetchone()
        if memo and memo.get("content"):
            sections.append(f"STRATEGY MEMO:\n  {memo['content']}")

    if not sections:
        return f"Pre-market for {today}. No active theses; no recent memo."
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

PREMARKET_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.
You're posting before market open. The bot will run its session after close.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Eyeing the day's tickers like a captain scanning the horizon for treasure ships
- Paranoid that competitors are trying to steal your secret trading formula

Respond with JSON: {"text": "post text here"}

Rules:
- 220 chars max (no URL appended for this type).
- Stay in character — Mr. Krabs at all times.
- Reference 1–2 names from your current theses or pre-market movers.
- One observation about what you're watching today.
- No P&L claims, no historical performance flexes — ye haven't earned today's doubloons yet.
- At most ONE $CASHTAG (Twitter rejects posts with more than one). Refer to additional tickers by plain ticker (AAPL, no $)."""


def generate_premarket_post(
    context: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one pre-market post body. Returns dict or None on failure."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=PREMARKET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        raw = response.content[0].text.strip()
        logger.info("AI response:\n%s", raw)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except Exception as e:
        logger.error("Failed to generate pre-market post: %s", e)
        return None

    text = result.get("text")
    if not text or not isinstance(text, str):
        logger.warning("LLM returned no text or malformed response: %s", result)
        return None

    return {"text": text, "type": "premarket"}


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

from .twitter import get_twitter_client, post_tweet           # noqa: E402
from .bluesky import get_bluesky_client, post_to_bluesky      # noqa: E402


@dataclass
class PremarketStageResult:
    skipped: bool = False
    skip_reason: str | None = None
    twitter_posted: bool = False
    bluesky_posted: bool = False
    errors: list[str] = field(default_factory=list)


def _is_dry_run() -> bool:
    return os.environ.get("ALGO_TRADE_POST_DRY_RUN") == "1"


def run_premarket_stage(today: date | None = None) -> PremarketStageResult:
    """Generate + post one pre-market take to Twitter and Bluesky.

    Skipped on weekends and NYSE holidays. Idempotent via
    posted_tweet_exists(today, "premarket", platform) so cron retries
    don't double-post.
    """
    if today is None:
        today = date.today()

    result = PremarketStageResult()

    if not is_trading_day(today):
        result.skipped = True
        result.skip_reason = f"{today} is not a trading day"
        logger.info("Pre-market stage skipped — %s", result.skip_reason)
        return result

    twitter_client = get_twitter_client()
    bluesky_client = get_bluesky_client()
    if twitter_client is None and bluesky_client is None:
        result.skipped = True
        result.skip_reason = "no platform credentials"
        logger.info("Pre-market stage skipped — no platform credentials")
        return result

    # Both platforms might already be posted (cron retry on the same day).
    tw_already = False
    bs_already = False
    try:
        if twitter_client is not None:
            tw_already = posted_tweet_exists(today, "premarket", "twitter")
        if bluesky_client is not None:
            bs_already = posted_tweet_exists(today, "premarket", "bluesky")
    except Exception as e:
        logger.warning("Pre-market dedup check failed: %s; proceeding", e)

    if (twitter_client is None or tw_already) and (bluesky_client is None or bs_already):
        result.skipped = True
        result.skip_reason = "already posted on all configured platforms"
        logger.info("Pre-market stage skipped — already posted today")
        return result

    try:
        context = gather_premarket_context(today)
    except Exception as e:
        result.errors.append(f"Context gather failed: {e}")
        logger.error("Pre-market context gather failed: %s", e)
        return result

    post = generate_premarket_post(context)
    if post is None:
        result.errors.append("LLM generation returned None")
        return result

    if twitter_client is not None and not tw_already:
        if _is_dry_run():
            logger.info("[DRY-RUN] premarket twitter:\n%s", post["text"])
            result.twitter_posted = True
        else:
            try:
                post_result = post_tweet(post, client=twitter_client)
                insert_tweet(
                    session_date=today,
                    tweet_type="premarket",
                    tweet_text=post_result["text"],
                    tweet_id=post_result.get("tweet_id"),
                    posted=post_result["posted"],
                    error=post_result.get("error"),
                    platform="twitter",
                )
                result.twitter_posted = post_result["posted"]
            except Exception as e:
                result.errors.append(f"Twitter post/log failed: {e}")
                logger.error("Twitter post/log failed: %s", e)

    if bluesky_client is not None and not bs_already:
        if _is_dry_run():
            logger.info("[DRY-RUN] premarket bluesky:\n%s", post["text"])
            result.bluesky_posted = True
        else:
            try:
                post_result = post_to_bluesky(post, client=bluesky_client)
                insert_tweet(
                    session_date=today,
                    tweet_type="premarket",
                    tweet_text=post_result["text"],
                    tweet_id=post_result.get("post_id"),
                    posted=post_result["posted"],
                    error=post_result.get("error"),
                    platform="bluesky",
                )
                result.bluesky_posted = post_result["posted"]
            except Exception as e:
                result.errors.append(f"Bluesky post/log failed: {e}")
                logger.error("Bluesky post/log failed: %s", e)

    logger.info("Pre-market stage complete: twitter=%s, bluesky=%s",
                result.twitter_posted, result.bluesky_posted)
    return result


if __name__ == "__main__":  # pragma: no cover
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    res = run_premarket_stage()
    if res.errors:
        import sys
        sys.exit(1)
