"""Live-trade social pipeline -- Bikini Bottom Capital (v2).

Per-fill posts after the daily session: one tweet per significant new
decision, each linking to its /trade/<id>/ page on the public dashboard.
Replaces the bare daily recap when ALGO_ENABLE_TRADE_POSTS=1.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

import requests

from .claude_client import _call_with_retry, get_claude_client
from .database.trading_db import (
    insert_tweet,
    posted_tweet_exists,
    posted_tweet_for_decision_exists,
    select_postable_decisions_for_date,
)

logger = logging.getLogger("social_trades")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

TRADE_POST_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.
The bot just made a trade. You're posting about it on social media.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about each trade — ecstatic about a steal, devastated about coughing up doubloons
- Paranoid that competitors are trying to steal your secret trading formula

Generate ONE post about this single trade.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL gets appended after — leave room).
- Stay in character — Mr. Krabs at all times.
- Lead with the action: "Bought 12 $NVDA at $X" / "Trimmed $TSLA back to half size".
- One concrete reason. The thesis text is provided — pull from it, don't invent.
- $CASHTAG only for the ticker actually traded.
- No "not financial advice", no hashtag spam, no emoji walls.
- If there's a thesis, your post should make a reader want to click through to read it."""


def _build_trade_context(decision: dict) -> str:
    """Plain-text summary of one decision + its thesis, fed to the LLM."""
    parts = [
        f"Trade: {decision['action'].upper()} {decision['quantity']} "
        f"{decision['ticker']} @ ${decision['price']}",
        f"Reasoning: {decision.get('reasoning', '')}",
    ]
    if decision.get("thesis_text"):
        parts.append(
            f"Thesis ({decision.get('thesis_direction', 'long')}): "
            f"{decision['thesis_text']}"
        )
    if decision.get("is_off_playbook"):
        parts.append("Note: this is an off-playbook trade.")
    return "\n".join(parts)


def _fetch_og_image(decision_id: int, dashboard_base_url: str) -> bytes | None:
    """Fetch the pre-rendered OG card image for a trade. Returns the PNG
    bytes, or None on any failure (no base URL configured, HTTP error,
    network exception). Failures degrade to a no-image post — never raise."""
    if not dashboard_base_url:
        return None
    url = f"{dashboard_base_url.rstrip('/')}/og/trade/{decision_id}.png"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.warning("OG image fetch failed for decision %s: %s", decision_id, e)
        return None


def _build_bluesky_card(decision: dict, dashboard_base_url: str) -> dict | None:
    """Card metadata for the Bluesky external embed. Title/description are
    derived from the decision (matching the public dashboard's OG tags),
    thumbnail is the pre-rendered /og/trade/<id>.png — omitted from the
    card if the fetch failed, so the post still gets a (text-only) card."""
    if not dashboard_base_url:
        return None
    base = dashboard_base_url.rstrip("/")
    action = decision["action"].upper()
    ticker = decision["ticker"]
    card = {
        "uri": f"{base}/trade/{decision['id']}/",
        "title": f"{action} {ticker}",
        "description": f"{action} {decision['quantity']} {ticker} @ ${decision['price']:.2f}",
    }
    png = _fetch_og_image(decision["id"], dashboard_base_url)
    if png:
        card["thumb_png"] = png
    return card


def _build_url_suffix(decision: dict, dashboard_base_url: str) -> str:
    """Deterministic trade + (optional) thesis URL append. Empty if no
    DASHBOARD_URL configured — bare text post."""
    if not dashboard_base_url:
        return ""
    base = dashboard_base_url.rstrip("/")
    parts = [f"{base}/trade/{decision['id']}/"]
    if decision.get("thesis_id"):
        parts.append(f"{base}/thesis/{decision['thesis_id']}/")
    return "\n" + "\n".join(parts)


def generate_trade_post(
    decision: dict,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one social-post body for a single decision.

    Returns dict {text, type='trade', decision_id} or None if generation
    fails (LLM error / malformed JSON / no text).
    """
    context = _build_trade_context(decision)
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=TRADE_POST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        raw = response.content[0].text.strip()
        logger.info("AI response for decision %s:\n%s", decision["id"], raw)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except Exception as e:
        logger.error("Failed to generate trade post for decision %s: %s",
                     decision.get("id"), e)
        return None

    body = result.get("text")
    if not body or not isinstance(body, str):
        logger.warning("LLM returned no text or malformed response: %s", result)
        return None

    text = body + _build_url_suffix(decision, dashboard_base_url)
    return {"text": text, "type": "trade", "decision_id": decision["id"]}


# ---------------------------------------------------------------------------
# Imports for the orchestrator (kept here so the pure helpers above stay
# importable in tests without dragging in the whole post stack).
# ---------------------------------------------------------------------------

from .market_calendar import is_trading_day                        # noqa: E402
from .twitter import get_twitter_client, post_tweet, gather_tweet_context, generate_tweet  # noqa: E402
from .bluesky import get_bluesky_client, post_to_bluesky, generate_bluesky_post  # noqa: E402


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

@dataclass
class TradePostsStageResult:
    """Aggregate outcome of run_trade_posts_stage.

    posts_attempted: how many decisions we tried to post about
    posts_skipped_dedup: per-(decision, platform) skips where rerun guard fired
    posts_succeeded_*: count of successful posts per platform
    posts_failed: per-decision generate-or-post failures (excludes dedup skips)
    quiet_day_recap_posted: True if the no-decisions fallback fired
    skipped: stage was a full no-op (no creds for either platform OR dry-run)
    errors: aggregated per-decision/per-platform error strings
    """
    posts_attempted: int = 0
    posts_skipped_dedup: int = 0
    posts_succeeded_twitter: int = 0
    posts_succeeded_bluesky: int = 0
    posts_failed: int = 0
    quiet_day_recap_posted: bool = False
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_NOTIONAL = 100.0
DEFAULT_MAX_POSTS_PER_SESSION = 5


def _is_dry_run() -> bool:
    return os.environ.get("ALGO_TRADE_POST_DRY_RUN") == "1"


def _platform_post_one(
    platform: str,
    post_body: dict,
    decision: dict,
    client,
    poster,
    session_date: date,
    result: TradePostsStageResult,
) -> None:
    """Post one body to one platform; record the outcome."""
    decision_id = decision["id"]

    try:
        if posted_tweet_for_decision_exists(decision_id, platform):
            result.posts_skipped_dedup += 1
            logger.info("Skip %s post for decision %s — already posted on %s",
                        platform, decision_id, platform)
            return
    except Exception as e:
        logger.warning("Dedup check failed for decision %s on %s: %s; proceeding",
                       decision_id, platform, e)

    if _is_dry_run():
        logger.info("[DRY-RUN] %s post for decision %s:\n%s",
                    platform, decision_id, post_body["text"])
        return

    try:
        post_result = poster(post_body, client=client)
    except Exception as e:
        result.posts_failed += 1
        result.errors.append(f"{platform} post failed for decision {decision_id}: {e}")
        logger.error("%s post failed for decision %s: %s", platform, decision_id, e)
        return

    db_logged = True
    try:
        insert_tweet(
            session_date=session_date,
            tweet_type="trade",
            tweet_text=post_result["text"],
            tweet_id=post_result.get("tweet_id") or post_result.get("post_id"),
            posted=post_result["posted"],
            error=post_result.get("error"),
            platform=platform,
            decision_id=decision_id,
        )
    except Exception as e:
        db_logged = False
        result.errors.append(f"DB log failed for decision {decision_id} on {platform}: {e}")
        logger.error("DB log failed for decision %s on %s: %s",
                     decision_id, platform, e)

    if post_result["posted"] and db_logged:
        if platform == "twitter":
            result.posts_succeeded_twitter += 1
        else:
            result.posts_succeeded_bluesky += 1
    else:
        result.posts_failed += 1


def _post_quiet_day_recap(
    session_date: date,
    twitter_client,
    bluesky_client,
    result: TradePostsStageResult,
) -> None:
    """Quiet-day fallback: post the existing recap-style summary on trading
    days when there are no postable decisions. Skipped entirely on weekends
    and NYSE holidays so the account doesn't spam non-content."""
    if not is_trading_day(session_date):
        logger.info("Quiet-day recap skipped — %s is not a trading day", session_date)
        return

    try:
        context = gather_tweet_context(session_date)
    except Exception as e:
        result.errors.append(f"Quiet-day context failed: {e}")
        logger.error("Quiet-day context gather failed: %s", e)
        return

    posted_any = False

    if twitter_client is not None:
        try:
            already = posted_tweet_exists(session_date, "recap", "twitter")
        except Exception as e:
            logger.warning("Quiet-day twitter dedup check failed: %s", e)
            already = False
        if not already:
            tweet = generate_tweet(context)
            if tweet:
                if _is_dry_run():
                    logger.info("[DRY-RUN] quiet-day twitter recap:\n%s", tweet["text"])
                    posted_any = True
                else:
                    try:
                        post_result = post_tweet(tweet, client=twitter_client)
                        insert_tweet(
                            session_date=session_date,
                            tweet_type=post_result.get("type", "recap"),
                            tweet_text=post_result["text"],
                            tweet_id=post_result.get("tweet_id"),
                            posted=post_result["posted"],
                            error=post_result.get("error"),
                            platform="twitter",
                        )
                        if post_result["posted"]:
                            posted_any = True
                    except Exception as e:
                        result.errors.append(f"Quiet-day twitter recap failed: {e}")
                        logger.error("Quiet-day twitter recap failed: %s", e)

    if bluesky_client is not None:
        try:
            already = posted_tweet_exists(session_date, "recap", "bluesky")
        except Exception as e:
            logger.warning("Quiet-day bluesky dedup check failed: %s", e)
            already = False
        if not already:
            post = generate_bluesky_post(context)
            if post:
                if _is_dry_run():
                    logger.info("[DRY-RUN] quiet-day bluesky recap:\n%s", post["text"])
                    posted_any = True
                else:
                    try:
                        post_result = post_to_bluesky(post, client=bluesky_client)
                        insert_tweet(
                            session_date=session_date,
                            tweet_type=post_result.get("type", "recap"),
                            tweet_text=post_result["text"],
                            tweet_id=post_result.get("post_id"),
                            posted=post_result["posted"],
                            error=post_result.get("error"),
                            platform="bluesky",
                        )
                        if post_result["posted"]:
                            posted_any = True
                    except Exception as e:
                        result.errors.append(f"Quiet-day bluesky recap failed: {e}")
                        logger.error("Quiet-day bluesky recap failed: %s", e)

    result.quiet_day_recap_posted = posted_any


def run_trade_posts_stage(
    session_date: date | None = None,
    min_notional: float = DEFAULT_MIN_NOTIONAL,
    max_posts: int = DEFAULT_MAX_POSTS_PER_SESSION,
) -> TradePostsStageResult:
    """Post one tweet per significant non-hold decision today."""
    if session_date is None:
        session_date = date.today()

    result = TradePostsStageResult()

    twitter_client = get_twitter_client()
    bluesky_client = get_bluesky_client()
    if twitter_client is None and bluesky_client is None:
        result.skipped = True
        logger.info("Trade-posts stage skipped — no platform credentials")
        return result

    try:
        decisions = select_postable_decisions_for_date(
            session_date=session_date,
            min_notional=min_notional,
            limit=max_posts,
        )
    except Exception as e:
        result.errors.append(f"Failed to select postable decisions: {e}")
        logger.error("select_postable_decisions failed: %s", e)
        return result

    if not decisions:
        _post_quiet_day_recap(session_date, twitter_client, bluesky_client, result)
        return result

    dashboard_base_url = os.environ.get("DASHBOARD_URL", "")

    for decision in decisions:
        result.posts_attempted += 1

        post_body = generate_trade_post(
            decision=decision,
            dashboard_base_url=dashboard_base_url,
        )
        if post_body is None:
            result.posts_failed += 1
            result.errors.append(
                f"Generation failed for decision {decision['id']}"
            )
            continue

        if twitter_client is not None:
            _platform_post_one(
                "twitter", post_body, decision, twitter_client,
                post_tweet, session_date, result,
            )
        if bluesky_client is not None:
            bs_post_body = dict(post_body)
            card = _build_bluesky_card(decision, dashboard_base_url)
            if card:
                bs_post_body["external_card"] = card
            _platform_post_one(
                "bluesky", bs_post_body, decision, bluesky_client,
                post_to_bluesky, session_date, result,
            )

    logger.info(
        "Trade-posts stage complete: attempted=%d, twitter_ok=%d, bluesky_ok=%d, "
        "failed=%d, dedup_skipped=%d",
        result.posts_attempted, result.posts_succeeded_twitter,
        result.posts_succeeded_bluesky, result.posts_failed,
        result.posts_skipped_dedup,
    )
    return result
