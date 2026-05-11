"""Haiku-based relevance filter for the news firehose.

Sits between the local news_signals cache and the strategist (Opus).
Called by tool_get_curated_news. Pure function: takes signals + a
regime context blob, returns the IDs of the top-N most relevant
signals for today.

Degrades gracefully — any error (API failure, malformed JSON,
all-hallucinated IDs) returns all input IDs so the caller serves the
firehose unfiltered. Telemetry is the caller's job.
"""
import json
import logging

import anthropic

from .claude_client import _call_with_retry, get_claude_client

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """You rank financial news signals by relevance for a trading strategist.

You receive:
  - A short regime_context describing today's market backdrop.
  - A list of news signals, each with: id, ticker, category, sentiment, headline, summary.

You return a JSON object: {"top_ids": [<int>, <int>, ...]} listing the IDs of the
top-N most relevant signals for the strategist to consider today, in order of
relevance. Optimize for: news-worthiness (real catalysts vs. noise), regime-fit
(consistent with today's backdrop), and de-duplication (one slot per distinct
story, not multiple variants of the same event).

Return ONLY the JSON object. No prose. No code fences. No commentary."""


def _call_haiku(*, messages: list[dict]):
    """Indirection point so tests can patch this without monkeypatching the SDK.

    Creates the client internally so patching this single function is enough
    to isolate tests from the API entirely.

    Returns the anthropic Message object.
    """
    client = get_claude_client()
    return _call_with_retry(
        client,
        model=HAIKU_MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )


def _build_user_message(signals: list[dict], target_n: int, regime_context: str) -> str:
    lines = [
        f"regime_context: {regime_context}",
        f"target_n: {target_n}",
        "",
        "signals:",
    ]
    for s in signals:
        lines.append(
            f"[#{s['id']}] {s.get('ticker','?')} {s.get('category','?')}/{s.get('sentiment','?')}: "
            f"{s.get('headline','')}\n  summary: {s.get('summary','')}"
        )
    return "\n".join(lines)


def curate_signals(
    signals: list[dict],
    target_n: int,
    regime_context: str,
) -> list[int]:
    """Return IDs of the top-N most relevant signals.

    Falls back to all input IDs on any error. Empty input returns
    empty list without calling the API.
    """
    if not signals:
        return []

    input_ids = {s["id"] for s in signals}
    fallback = [s["id"] for s in signals]

    user_message = _build_user_message(signals, target_n, regime_context)

    try:
        response = _call_haiku(
            messages=[{"role": "user", "content": user_message}],
        )
    except (anthropic.APIError, anthropic.APIConnectionError, RuntimeError, ValueError) as e:
        logger.warning("Haiku news filter call failed (%s); falling back to firehose", e)
        return fallback

    try:
        text = response.content[0].text.strip()
        # Strip optional ```json fences just in case Haiku ignores instructions.
        if text.startswith("```"):
            parts = text.split("\n", 1)
            text = parts[1] if len(parts) == 2 else parts[0].lstrip("`")
            text = text.rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        raw_ids = parsed.get("top_ids", [])
        if not isinstance(raw_ids, list):
            raise ValueError(f"top_ids is not a list: {raw_ids!r}")
    except (ValueError, KeyError, AttributeError, IndexError) as e:
        logger.warning("Haiku response parse failed (%s); falling back to firehose", e)
        return fallback

    valid_ids = [int(i) for i in raw_ids if isinstance(i, int) and i in input_ids]
    if not valid_ids:
        logger.warning("Haiku returned no valid IDs (input=%d, returned=%d); falling back",
                       len(input_ids), len(raw_ids))
        return fallback

    return valid_ids
