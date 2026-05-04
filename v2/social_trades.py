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

TRADE_POST_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital.
The bot just made a trade. You're posting about it on social media.

Your voice:
- Casual, direct. Like sharing a play with a friend who trades.
- Don't oversell. Reference the actual reasoning — not generic excitement.
- Occasional dry humor. Never try-hard.

Generate ONE post about this single trade.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL gets appended after — leave room).
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
