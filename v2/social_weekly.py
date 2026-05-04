"""Weekly social posts -- Bikini Bottom Capital (v2).

Two scheduled-post functions:
- run_mistakes_post: "what didn't work" — links to /mistakes/
- run_attribution_post: signal-attribution roundup — links to /attribution/

Both run from cron Friday afternoon, after the daily session has had time
to publish Stage 6. Skipped on weekends / NYSE holidays via is_trading_day.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

from .claude_client import _call_with_retry, get_claude_client
from .database.connection import get_cursor  # noqa: F401  used via mock_db patch
from .database.trading_db import (
    get_closed_losers,
    get_retired_rules,
    get_signal_attribution,
    insert_tweet,
    posted_tweet_exists,
)
from .market_calendar import is_trading_day

logger = logging.getLogger("social_weekly")


# ---------------------------------------------------------------------------
# Mistakes — context, prompt, generator
# ---------------------------------------------------------------------------

MISTAKES_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital.
You post weekly about what the bot got wrong.

Your voice:
- Honest. Specific. No self-flagellation, no "valuable lesson learned".
- Treat losses as data, not embarrassment.
- Dry, not bitter.

Most trading accounts hide losses. You don't. That's the point.

Generate ONE post about this week's worst trade or retired rule.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL appended after).
- One specific thing — the worst trade, or the retired rule, not a list.
- Reference the actual ticker / rule, not "a position" or "a strategy".
- No "we'll do better next time" / no "lessons learned" cliché."""


def gather_mistakes_context(today: date | None = None) -> str:
    """Plain-text summary of recent losers + retired rules."""
    if today is None:
        today = date.today()

    losers = get_closed_losers(reference_date=today, limit=5)
    rules = get_retired_rules(reference_date=today, limit=5)

    parts: list[str] = []
    if losers:
        parts.append("RECENT LOSERS:")
        for d in losers:
            try:
                outcome = f"{float(d.get('outcome_30d') or 0):+.2f}%"
            except Exception:
                outcome = ""
            parts.append(
                f"  {d.get('ticker','?')} {str(d.get('action','')).upper()}"
                f" {d.get('quantity','?')} @ ${d.get('price','?')}"
                f" — 30d: {outcome}"
                f"  ({d.get('reasoning','')})"
            )
    if rules:
        parts.append("\nRETIRED RULES:")
        for r in rules:
            parts.append(
                f"  {r.get('rule_text','')} "
                f"(reason: {r.get('retirement_reason','')})"
            )

    return "\n".join(parts) if parts else ""


def _generate_post(
    *,
    system_prompt: str,
    context: str,
    type_label: str,
    permalink: str,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Shared LLM call + URL append for both weekly post types."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": context}],
        )
        raw = response.content[0].text.strip()
        logger.info("AI response (%s):\n%s", type_label, raw)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except Exception as e:
        logger.error("Failed to generate %s post: %s", type_label, e)
        return None

    body = result.get("text")
    if not body or not isinstance(body, str):
        logger.warning("LLM returned no text or malformed response: %s", result)
        return None

    suffix = ""
    if dashboard_base_url:
        suffix = "\n" + dashboard_base_url.rstrip("/") + permalink
    return {"text": body + suffix, "type": type_label}


def generate_mistakes_post(
    context: str,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one mistakes-post body."""
    return _generate_post(
        system_prompt=MISTAKES_SYSTEM_PROMPT,
        context=context,
        type_label="weekly_mistakes",
        permalink="/mistakes/",
        dashboard_base_url=dashboard_base_url,
        model=model,
    )
