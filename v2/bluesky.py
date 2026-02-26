"""Bluesky integration -- Bikini Bottom Capital (v2 pipeline).

Generates and posts to Bluesky about trading activity using Claude.
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


def post_to_bluesky(post: dict, client=None) -> dict:
    """Post to Bluesky via the AT Protocol."""
    if client is None:
        client = get_bluesky_client()
    if client is None:
        return {"text": post["text"], "type": post.get("type", "commentary"), "posted": False, "post_id": None, "error": "No Bluesky credentials"}

    post_type = post.get("type", "commentary")
    try:
        facets = None
        dashboard_url = post.get("dashboard_url")
        if dashboard_url:
            facets = _build_link_facets(post["text"], _DASHBOARD_LINK_TEXT, dashboard_url)
        response = client.send_post(text=post["text"], facets=facets)
        post_id = response.uri
        logger.info("Posted to Bluesky %s: %s...", post_id, post["text"][:50])
        return {"text": post["text"], "type": post_type, "posted": True, "post_id": post_id, "error": None}
    except Exception as e:
        logger.error("Failed to post to Bluesky: %s", e)
        return {"text": post["text"], "type": post_type, "posted": False, "post_id": None, "error": str(e)}


_BLUESKY_SYSTEM_PROMPT_TEMPLATE = """You run an algorithmic trading operation called Bikini Bottom Capital. You post daily recaps on social media.

Your voice:
- Casual and straightforward, like you're catching up a friend on how the day went
- Honest about what happened — don't sugarcoat bad days or oversell good ones
- Dry humor when it fits, but the recap comes first
- Avoid talking about total portfolio gain, as it doesn't reflect the cash position correctly

Generate ONE post that summarizes today's trading session.

Respond with JSON in this exact format:
{{"text": "post text here"}}

Rules:
- Sound like a real person, not a brand account or a character
- Ground it in the actual trading data — what you bought/sold, how P&L looked
- Use 1-3 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning tickers
- Summarize the full session: trades made, P&L result, and how things stand
- If it was a quiet day with no trades, just say so
- Maximum {limit} characters"""

BLUESKY_GRAPHEME_LIMIT = 300
# LLMs can't count graphemes reliably; tell them a lower number as buffer
_PROMPT_BUFFER = 30
_SHORTEN_MAX_RETRIES = 2

# Keep a static reference for tests/backward compat
BLUESKY_SYSTEM_PROMPT = _BLUESKY_SYSTEM_PROMPT_TEMPLATE.format(limit=BLUESKY_GRAPHEME_LIMIT - _PROMPT_BUFFER)


def _grapheme_length(text: str) -> int:
    """Return the number of grapheme clusters in *text*."""
    import grapheme
    return grapheme.length(text)


def _grapheme_truncate(text: str, limit: int) -> str:
    """Truncate *text* to at most *limit* grapheme clusters."""
    import grapheme
    if grapheme.length(text) <= limit:
        return text
    return grapheme.slice(text, 0, limit)


def _condense_post(client, text: str, limit: int, model: str) -> str | None:
    """Ask the LLM to shorten a post that exceeded the grapheme limit.

    Returns the shortened text, or None if condensing fails.
    """
    try:
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=(
                "You are a copy editor. Shorten the given Bluesky post to fit within "
                f"{limit} characters. Keep the same voice, tone, and key information. "
                "Cut filler, not substance. Respond with JSON: {\"text\": \"shortened post\"}"
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"This post is too long ({_grapheme_length(text)} characters). "
                    f"Shorten it to {limit} characters max:\n\n{text}"
                ),
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        shortened = result.get("text")
        if shortened and isinstance(shortened, str):
            return shortened
    except Exception as e:
        logger.warning("Failed to condense post: %s", e)
    return None


def _enforce_limit(client, text: str, limit: int, model: str) -> str:
    """Try to get the text under *limit* graphemes via LLM condensing, then truncate."""
    for attempt in range(_SHORTEN_MAX_RETRIES):
        if _grapheme_length(text) <= limit:
            return text
        logger.warning(
            "Post exceeds %d graphemes (%d), asking LLM to shorten (attempt %d/%d)",
            limit, _grapheme_length(text), attempt + 1, _SHORTEN_MAX_RETRIES,
        )
        shortened = _condense_post(client, text, limit, model)
        if shortened:
            text = shortened
        else:
            break

    # Last resort: hard truncate
    if _grapheme_length(text) > limit:
        logger.warning("Post still over limit after retries, truncating")
        text = _grapheme_truncate(text, limit)
    return text


_DASHBOARD_LINK_TEXT = "Dashboard"


def _dashboard_url_suffix() -> str:
    """Return the suffix that will be appended for the dashboard link, or empty."""
    dashboard_url = os.environ.get("DASHBOARD_URL")
    if dashboard_url:
        return f"\n{_DASHBOARD_LINK_TEXT}"
    return ""


def _build_link_facets(text: str, link_text: str, url: str) -> list | None:
    """Build AT Protocol facets to make *link_text* a clickable link in *text*.

    Returns a list of facet objects for ``client.send_post(facets=...)``,
    or *None* if the link text isn't found or atproto isn't installed.
    """
    try:
        from atproto import models
    except ImportError:
        return None

    text_bytes = text.encode("utf-8")
    link_bytes = link_text.encode("utf-8")
    byte_start = text_bytes.rfind(link_bytes)
    if byte_start == -1:
        return None
    byte_end = byte_start + len(link_bytes)

    return [
        models.AppBskyRichtextFacet.Main(
            index=models.AppBskyRichtextFacet.ByteSlice(
                byte_start=byte_start,
                byte_end=byte_end,
            ),
            features=[models.AppBskyRichtextFacet.Link(uri=url)],
        )
    ]


def generate_bluesky_post(context: str, model: str = "claude-haiku-4-5-20251001") -> dict | None:
    """Generate a single Bluesky post from session context using Claude."""
    # Reduce the LLM's stated budget: leave room for dashboard URL + prompt buffer
    suffix = _dashboard_url_suffix()
    text_limit = BLUESKY_GRAPHEME_LIMIT - _grapheme_length(suffix) - _PROMPT_BUFFER
    system_prompt = _BLUESKY_SYSTEM_PROMPT_TEMPLATE.format(limit=text_limit)

    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=system_prompt,
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

    # Enforce limit on the post body *before* appending the dashboard URL
    body_limit = BLUESKY_GRAPHEME_LIMIT - _grapheme_length(suffix)
    post_text = _enforce_limit(client, post_text, body_limit, model)

    # Append dashboard link text if configured
    dashboard_url = os.environ.get("DASHBOARD_URL")
    if dashboard_url:
        post_text = f"{post_text}\n{_DASHBOARD_LINK_TEXT}"
        return {"text": post_text, "type": "recap", "dashboard_url": dashboard_url}

    return {"text": post_text, "type": "recap"}


BLUESKY_ENTERTAINMENT_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital. You post on social media about markets.

Your voice:
- Casual and conversational, like texting a friend who's also into markets
- Genuinely curious about what's happening, not performing excitement
- Dry humor, occasional sarcasm — never try-hard or corny
- Comfortable admitting when something surprises you or doesn't make sense
- You have opinions but you're not shouting them

Generate ONE post based on the market news and data provided. This is standalone commentary, not a session recap.

Respond with JSON in this exact format:
{"text": "post text here"}

Rules:
- Sound like a real person, not a brand account or a character
- Ground the post in actual market data — reference real tickers, real moves, real news
- Use 1-2 relevant cashtags ($AAPL, $NVDA, etc.) when mentioning specific stocks
- Pick the single most interesting thing and make one sharp observation about it
- Be concise. 1-2 sentences max. Brevity is wit.
- No filler words, no throat-clearing — just the take
- Maximum 270 characters"""


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

    post_text = result["text"]

    # Enforce Bluesky's grapheme limit (condense via LLM, then truncate as last resort)
    post_text = _enforce_limit(client, post_text, BLUESKY_GRAPHEME_LIMIT, model)

    return {"text": post_text, "type": "entertainment"}


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
