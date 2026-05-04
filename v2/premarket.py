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

PREMARKET_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital.
You're posting before market open. The bot will run its session after close.

Your voice:
- Casual, observational. What you're watching, what's interesting.
- Forward-looking but not predictive. No "this will rip" claims.
- Honest about uncertainty.

Respond with JSON: {"text": "post text here"}

Rules:
- 220 chars max (no URL appended for this type).
- Reference 1–2 names from your current theses or pre-market movers.
- One observation about what you're watching today.
- No P&L claims, no historical performance flexes.
- $CASHTAG only for tickers you mention."""


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
