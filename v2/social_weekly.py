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

MISTAKES_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.
You post weekly about what the bot got wrong.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Dramatically emotional about losses — devastated, mourning every doubloon that walked off the ship
- Paranoid that competitors are trying to steal your secret trading formula

Most trading accounts hide losses. Ye don't. That's the point.

Generate ONE post about this week's worst trade or retired rule.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL appended after).
- Stay in character — Mr. Krabs at all times.
- One specific thing — the worst trade, or the retired rule, not a list.
- Reference the actual ticker / rule, not "a position" or "a strategy".
- No "we'll do better next time" / no "lessons learned" cliché — Krabs grieves, he doesn't lecture."""


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


# ---------------------------------------------------------------------------
# Attribution — context, prompt, generator
# ---------------------------------------------------------------------------

ATTRIBUTION_SYSTEM_PROMPT = """You are Mr. Krabs from SpongeBob SquarePants, running an algorithmic trading operation called Bikini Bottom Capital.
You post weekly about which signal types are actually predictive — which ones earn ye doubloons and which ones be worthless.

Your personality:
- Obsessed with money and profits above all else
- Use nautical language and sea metaphors naturally
- Treat predictive signals like prized treasure, useless ones like rotting bait
- Paranoid that competitors are trying to steal your secret trading formula

Generate ONE post about this week's signal attribution scores.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL appended after).
- Stay in character — Mr. Krabs at all times.
- Name 1–2 signal types and their scores. Not all of them.
- One non-obvious observation, if there is one. Otherwise just the data.
- Don't claim "alpha". Use "predictive" / "useful" / "noise" (ye can frame these in Krabs's words too)."""


def gather_attribution_context() -> str:
    """Plain-text summary of best + worst signal types by avg_outcome_30d."""
    rows = get_signal_attribution()
    if not rows:
        return ""

    sortable = [r for r in rows if r.get("avg_outcome_30d") is not None]
    if not sortable:
        return ""
    sortable.sort(key=lambda r: float(r.get("avg_outcome_30d") or 0), reverse=True)

    top = sortable[:3]
    bottom = sortable[-3:][::-1]

    def _fmt(rs):
        out = []
        for r in rs:
            try:
                pct = f"{float(r.get('avg_outcome_30d') or 0):+.2f}%"
            except Exception:
                pct = ""
            out.append(
                f"  {r.get('category','?')}: {pct} (n={r.get('sample_size', 0)})"
            )
        return "\n".join(out)

    parts = ["BEST PREDICTORS:", _fmt(top)]
    if bottom and bottom[-1] is not top[-1]:
        parts.extend(["", "WORST PREDICTORS:", _fmt(bottom)])
    return "\n".join(parts)


def generate_attribution_post(
    context: str,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one attribution-post body."""
    return _generate_post(
        system_prompt=ATTRIBUTION_SYSTEM_PROMPT,
        context=context,
        type_label="weekly_attribution",
        permalink="/attribution/",
        dashboard_base_url=dashboard_base_url,
        model=model,
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

# Imported late so the pure helpers above stay testable without dragging in
# the full social-platform stack.
from .bluesky import get_bluesky_client, post_to_bluesky  # noqa: E402
from .twitter import get_twitter_client, post_tweet  # noqa: E402


@dataclass
class WeeklyPostResult:
    skipped: bool = False
    skip_reason: str | None = None
    twitter_posted: bool = False
    bluesky_posted: bool = False
    errors: list[str] = field(default_factory=list)


def _is_dry_run() -> bool:
    return os.environ.get("ALGO_TRADE_POST_DRY_RUN") == "1"


def _post_one(
    *,
    platform: str,
    client,
    poster,
    post_body: dict,
    today: date,
    type_label: str,
    result: WeeklyPostResult,
) -> None:
    if _is_dry_run():
        logger.info("[DRY-RUN] %s %s post:\n%s",
                    type_label, platform, post_body["text"])
        if platform == "twitter":
            result.twitter_posted = True
        else:
            result.bluesky_posted = True
        return

    try:
        post_result = poster(post_body, client=client)
        insert_tweet(
            session_date=today,
            tweet_type=type_label,
            tweet_text=post_result["text"],
            tweet_id=post_result.get("tweet_id") or post_result.get("post_id"),
            posted=post_result["posted"],
            error=post_result.get("error"),
            platform=platform,
        )
        if post_result["posted"]:
            if platform == "twitter":
                result.twitter_posted = True
            else:
                result.bluesky_posted = True
    except Exception as e:
        result.errors.append(f"{platform} {type_label} post/log failed: {e}")
        logger.error("%s %s post/log failed: %s", platform, type_label, e)


def _run_weekly(
    *,
    today: date | None,
    type_label: str,
    gather,
    generate,
) -> WeeklyPostResult:
    if today is None:
        today = date.today()

    result = WeeklyPostResult()

    if not is_trading_day(today):
        result.skipped = True
        result.skip_reason = f"{today} is not a trading day"
        logger.info("Weekly %s skipped — %s", type_label, result.skip_reason)
        return result

    twitter_client = get_twitter_client()
    bluesky_client = get_bluesky_client()
    if twitter_client is None and bluesky_client is None:
        result.skipped = True
        result.skip_reason = "no platform credentials"
        logger.info("Weekly %s skipped — no platform credentials", type_label)
        return result

    try:
        context = gather(today=today) if type_label == "weekly_mistakes" else gather()
    except Exception as e:
        result.errors.append(f"Context gather failed: {e}")
        logger.error("%s context gather failed: %s", type_label, e)
        return result

    if not context:
        result.skipped = True
        result.skip_reason = "no data this window"
        logger.info("Weekly %s skipped — no data", type_label)
        return result

    # Idempotency check — both platforms
    tw_already = False
    bs_already = False
    try:
        if twitter_client is not None:
            tw_already = posted_tweet_exists(today, type_label, "twitter")
        if bluesky_client is not None:
            bs_already = posted_tweet_exists(today, type_label, "bluesky")
    except Exception as e:
        logger.warning("Weekly %s dedup check failed: %s; proceeding",
                       type_label, e)

    if (twitter_client is None or tw_already) and (bluesky_client is None or bs_already):
        result.skipped = True
        result.skip_reason = "already posted on all configured platforms"
        logger.info("Weekly %s skipped — already posted today", type_label)
        return result

    dashboard_base_url = os.environ.get("DASHBOARD_URL", "")
    post_body = generate(context, dashboard_base_url=dashboard_base_url)
    if post_body is None:
        result.errors.append("LLM generation returned None")
        return result

    if twitter_client is not None and not tw_already:
        _post_one(
            platform="twitter", client=twitter_client, poster=post_tweet,
            post_body=post_body, today=today, type_label=type_label, result=result,
        )
    if bluesky_client is not None and not bs_already:
        _post_one(
            platform="bluesky", client=bluesky_client, poster=post_to_bluesky,
            post_body=post_body, today=today, type_label=type_label, result=result,
        )

    logger.info("Weekly %s complete: twitter=%s, bluesky=%s",
                type_label, result.twitter_posted, result.bluesky_posted)
    return result


def run_mistakes_post(today: date | None = None) -> WeeklyPostResult:
    return _run_weekly(
        today=today,
        type_label="weekly_mistakes",
        gather=gather_mistakes_context,
        generate=generate_mistakes_post,
    )


def run_attribution_post(today: date | None = None) -> WeeklyPostResult:
    return _run_weekly(
        today=today,
        type_label="weekly_attribution",
        gather=gather_attribution_context,
        generate=generate_attribution_post,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v2.social_weekly",
        description="Weekly social posts (mistakes / attribution).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mistakes", help="Run the weekly mistakes post.")
    sub.add_parser("attribution", help="Run the weekly attribution post.")
    args = parser.parse_args(argv)

    if args.cmd == "mistakes":
        result = run_mistakes_post()
    else:
        result = run_attribution_post()
    return 1 if getattr(result, "errors", []) else 0


if __name__ == "__main__":  # pragma: no cover
    import logging as _logging
    import sys
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    sys.exit(_main())
