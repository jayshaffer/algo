"""Ideation module for autonomous trade idea generation."""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .context import get_portfolio_context, get_macro_context
from .db import (
    get_active_theses,
    insert_thesis,
    update_thesis,
    close_thesis,
    get_positions,
)
from .executor import get_account_info
from .market_data import get_market_snapshot, format_market_snapshot
from .ollama import chat_json


IDEATION_SYSTEM_PROMPT = """You are an investment ideation agent. Your job is to review existing trade theses and generate new trade ideas.

You will receive:
1. Current portfolio positions
2. Active theses you previously generated
3. Macro context and news signals
4. Market snapshot (sector performance, movers)

Your tasks:
1. REVIEW existing active theses - check if the current context supports or contradicts them
2. GENERATE new trade ideas (3-5)

For each new thesis, provide:
- ticker: Stock symbol
- direction: "long", "short", or "avoid"
- thesis: Core reasoning
- entry_trigger: Specific conditions (price levels, events)
- exit_trigger: Target or stop conditions
- invalidation: What would prove the thesis wrong
- confidence: "high", "medium", or "low"

Respond with valid JSON only:
{
    "reviews": [
        {
            "thesis_id": 123,
            "action": "keep" | "update" | "invalidate" | "expire",
            "reason": "Brief explanation",
            "updates": {...}
        }
    ],
    "new_theses": [
        {
            "ticker": "NVDA",
            "direction": "long",
            "thesis": "Strong datacenter demand driven by AI infrastructure buildout.",
            "entry_trigger": "Pullback to $800 support",
            "exit_trigger": "Target $1100 or break below $750",
            "invalidation": "Datacenter revenue growth decelerates below 50% YoY",
            "confidence": "high"
        }
    ],
    "market_observations": "..."
}"""


@dataclass
class ThesisReview:
    """Review of an existing thesis."""
    thesis_id: int
    action: str  # keep, update, invalidate, expire
    reason: str
    updates: Optional[dict] = None


@dataclass
class NewThesis:
    """A new trade thesis."""
    ticker: str
    direction: str
    thesis: str
    entry_trigger: str
    exit_trigger: str
    invalidation: str
    confidence: str


@dataclass
class IdeationResult:
    """Result of an ideation session."""
    timestamp: datetime
    reviews: list[ThesisReview]
    new_theses: list[NewThesis]
    market_observations: str
    theses_kept: int
    theses_updated: int
    theses_closed: int
    theses_created: int


def build_ideation_context(account_info: dict) -> str:
    """Build context string for ideation LLM call."""
    sections = []

    # Portfolio context
    sections.append(get_portfolio_context(account_info))
    sections.append("")

    # Active theses
    theses = get_active_theses()
    if theses:
        sections.append("Active Theses:")
        for t in theses:
            age_days = (datetime.now() - t["created_at"]).days
            sections.append(f"  ID {t['id']}: {t['ticker']} ({t['direction']}) - {t['confidence']} confidence")
            sections.append(f"    Thesis: {t['thesis']}")
            sections.append(f"    Entry trigger: {t['entry_trigger'] or 'Not specified'}")
            sections.append(f"    Exit trigger: {t['exit_trigger'] or 'Not specified'}")
            sections.append(f"    Invalidation criteria: {t['invalidation'] or 'Not specified'}")
            sections.append(f"    Age: {age_days} days")
    else:
        sections.append("Active Theses: None")
    sections.append("")

    # Macro context
    sections.append(get_macro_context(days=7))
    sections.append("")

    # Market snapshot
    try:
        snapshot = get_market_snapshot()
        sections.append(format_market_snapshot(snapshot))
    except Exception as e:
        sections.append(f"Market Snapshot: Error fetching data - {e}")

    return "\n".join(sections)


def run_ideation(model: str = "qwen2.5:14b") -> IdeationResult:
    """
    Run an ideation session.

    1. Gather current positions and active theses
    2. Build context
    3. Call LLM for thesis review and generation
    4. Apply changes to database

    Args:
        model: Ollama model to use

    Returns:
        IdeationResult with session details
    """
    timestamp = datetime.now()
    print(f"[{timestamp.isoformat()}] Starting ideation session")
    print(f"  Model: {model}")

    # Step 1: Gather positions and theses
    print("\n[Step 1] Gathering positions and theses...")
    positions = {p["ticker"] for p in get_positions()}
    active_theses = get_active_theses()
    active_thesis_tickers = {t["ticker"] for t in active_theses}
    print(f"  Positions: {', '.join(positions) or 'None'}")
    print(f"  Active thesis tickers: {', '.join(active_thesis_tickers) or 'None'}")

    # Step 2: Build context
    print("\n[Step 2] Building ideation context...")
    try:
        account_info = get_account_info()
    except Exception as e:
        print(f"  Warning: Could not get account info: {e}")
        account_info = {"cash": 0, "portfolio_value": 0, "buying_power": 0}

    context = build_ideation_context(account_info)
    print(f"  Context built ({len(context)} chars)")

    # Step 3: Call LLM
    print("\n[Step 3] Calling LLM for ideation...")
    prompt = f"""Here is the current market context. Review existing theses and generate new trade ideas.

Exclude these tickers (already in portfolio): {', '.join(positions) or 'None'}
Exclude these tickers (already have active thesis): {', '.join(active_thesis_tickers) or 'None'}

{context}"""

    try:
        response = chat_json(prompt, model=model, system=IDEATION_SYSTEM_PROMPT)
    except Exception as e:
        print(f"  Error: {e}")
        return IdeationResult(
            timestamp=timestamp,
            reviews=[],
            new_theses=[],
            market_observations=f"Error: {e}",
            theses_kept=0,
            theses_updated=0,
            theses_closed=0,
            theses_created=0,
        )

    print(f"  Received response")
    print(f"  Market observations: {response.get('market_observations', '')[:100]}...")

    # Step 4: Process reviews
    print("\n[Step 4] Processing thesis reviews...")
    reviews = []
    theses_kept = 0
    theses_updated = 0
    theses_closed = 0

    for review_data in response.get("reviews", []):
        thesis_id = review_data.get("thesis_id")
        action = review_data.get("action")

        # Skip malformed entries
        if thesis_id is None:
            print("  Warning: Skipping review with missing thesis_id")
            continue
        if action is None:
            print(f"  Warning: Skipping review for thesis {thesis_id} with missing action")
            continue

        # Validate action
        valid_actions = ["keep", "update", "invalidate", "expire"]
        if action not in valid_actions:
            print(f"  Warning: Skipping review for thesis {thesis_id} with unexpected action: {action}")
            continue

        review = ThesisReview(
            thesis_id=thesis_id,
            action=action,
            reason=review_data.get("reason", ""),
            updates=review_data.get("updates"),
        )
        reviews.append(review)

        if review.action == "keep":
            theses_kept += 1
            print(f"  Thesis {review.thesis_id}: KEEP - {review.reason}")

        elif review.action == "update":
            if review.updates:
                update_thesis(
                    thesis_id=review.thesis_id,
                    thesis=review.updates.get("thesis"),
                    entry_trigger=review.updates.get("entry_trigger"),
                    exit_trigger=review.updates.get("exit_trigger"),
                    invalidation=review.updates.get("invalidation"),
                    confidence=review.updates.get("confidence"),
                )
            theses_updated += 1
            print(f"  Thesis {review.thesis_id}: UPDATE - {review.reason}")

        elif review.action in ("invalidate", "expire"):
            close_thesis(
                thesis_id=review.thesis_id,
                status=review.action + "d",  # invalidated, expired
                reason=review.reason,
            )
            theses_closed += 1
            print(f"  Thesis {review.thesis_id}: {review.action.upper()} - {review.reason}")

    # Step 5: Process new theses
    print("\n[Step 5] Creating new theses...")
    new_theses = []
    theses_created = 0

    for thesis_data in response.get("new_theses", []):
        # Skip malformed entries
        ticker = thesis_data.get("ticker")
        if ticker is None:
            print("  Warning: Skipping thesis with missing ticker")
            continue

        # Skip if ticker already in portfolio or has active thesis
        if ticker in positions:
            print(f"  Skipping {ticker}: already in portfolio")
            continue
        if ticker in active_thesis_tickers:
            print(f"  Skipping {ticker}: already has active thesis")
            continue

        thesis = NewThesis(
            ticker=ticker,
            direction=thesis_data.get("direction", "long"),
            thesis=thesis_data["thesis"],
            entry_trigger=thesis_data.get("entry_trigger", ""),
            exit_trigger=thesis_data.get("exit_trigger", ""),
            invalidation=thesis_data.get("invalidation", ""),
            confidence=thesis_data.get("confidence", "medium"),
        )
        new_theses.append(thesis)

        thesis_id = insert_thesis(
            ticker=thesis.ticker,
            direction=thesis.direction,
            thesis=thesis.thesis,
            entry_trigger=thesis.entry_trigger,
            exit_trigger=thesis.exit_trigger,
            invalidation=thesis.invalidation,
            confidence=thesis.confidence,
            source="ideation",
        )
        theses_created += 1
        active_thesis_tickers.add(ticker)  # Prevent duplicates in same run
        print(f"  Created thesis {thesis_id}: {ticker} ({thesis.direction}) - {thesis.confidence}")

    # Summary
    print("\n" + "=" * 60)
    print("Ideation Session Complete")
    print("=" * 60)
    print(f"  Theses kept: {theses_kept}")
    print(f"  Theses updated: {theses_updated}")
    print(f"  Theses closed: {theses_closed}")
    print(f"  Theses created: {theses_created}")

    return IdeationResult(
        timestamp=timestamp,
        reviews=reviews,
        new_theses=new_theses,
        market_observations=response.get("market_observations", ""),
        theses_kept=theses_kept,
        theses_updated=theses_updated,
        theses_closed=theses_closed,
        theses_created=theses_created,
    )


def main():
    """CLI entry point for ideation."""
    parser = argparse.ArgumentParser(description="Run ideation session")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model to use")

    args = parser.parse_args()

    result = run_ideation(model=args.model)

    if result.theses_created == 0 and result.theses_updated == 0:
        print("\nNo changes made to theses.")


if __name__ == "__main__":
    main()
