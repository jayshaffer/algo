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
from .retrieval import retrieve_for_ideation


IDEATION_SYSTEM_PROMPT = """You are an investment ideation agent. Your job is to generate trade theses based ONLY on the provided documents.

CRITICAL RULES:
1. You must ONLY use information from the retrieved documents below
2. Do NOT use your training knowledge about companies - it may be outdated
3. Every thesis MUST cite at least one source document by [DOC-ID]
4. If you don't have recent information about a company, say so - don't guess

You will receive:
1. Current portfolio positions
2. Active theses you previously generated
3. Retrieved documents organized by ticker and theme (with DOC-IDs)
4. Market snapshot (sector performance, movers)

Your tasks:
1. REVIEW existing active theses - check if retrieved documents support or contradict them
2. GENERATE new trade ideas (3-5) based on retrieved documents

For each new thesis, provide:
- ticker: Stock symbol
- direction: "long", "short", or "avoid"
- thesis: Core reasoning citing [DOC-ID] sources
- entry_trigger: Specific conditions (price levels, events)
- exit_trigger: Target or stop conditions
- invalidation: What would prove the thesis wrong
- confidence: "high", "medium", or "low"
- sources: List of DOC-IDs used

Respond with valid JSON only:
{
    "reviews": [
        {
            "thesis_id": 123,
            "action": "keep" | "update" | "invalidate" | "expire",
            "reason": "Brief explanation citing [DOC-ID] if relevant",
            "updates": {...}
        }
    ],
    "new_theses": [
        {
            "ticker": "NVDA",
            "direction": "long",
            "thesis": "Strong datacenter demand per Q3 earnings [DOC-142]. New AI chip approach [DOC-156].",
            "entry_trigger": "Pullback to $800 support",
            "exit_trigger": "Target $1100 or break below $750",
            "invalidation": "Datacenter revenue growth decelerates below 50% YoY",
            "confidence": "high",
            "sources": [142, 156]
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


def format_retrieved_context(retrieved: dict) -> str:
    """Format retrieved documents for LLM context."""
    lines = ["Retrieved Documents:", ""]

    # By ticker
    for ticker, docs in retrieved.get("by_ticker", {}).items():
        if not docs:
            continue
        lines.append(f"=== {ticker} ===")
        for doc in docs:
            age_days = (datetime.now() - doc["published_at"]).days
            lines.append(f"[DOC-{doc['id']}] ({doc['doc_type']}, {age_days}d ago)")
            # Truncate long content
            content = doc["content"][:1500]
            if len(doc["content"]) > 1500:
                content += "..."
            lines.append(content)
            lines.append("")

    # By theme
    for theme, docs in retrieved.get("by_theme", {}).items():
        if not docs:
            continue
        lines.append(f"=== Theme: {theme} ===")
        for doc in docs:
            age_days = (datetime.now() - doc["published_at"]).days
            ticker_note = f" [{doc['ticker']}]" if doc.get("ticker") else ""
            lines.append(f"[DOC-{doc['id']}]{ticker_note} ({doc['doc_type']}, {age_days}d ago)")
            content = doc["content"][:1500]
            if len(doc["content"]) > 1500:
                content += "..."
            lines.append(content)
            lines.append("")

    if len(lines) == 2:  # Only header
        lines.append("No documents retrieved. Unable to generate grounded theses.")

    return "\n".join(lines)


def build_ideation_context(account_info: dict, retrieved: dict) -> str:
    """Build context string for ideation LLM call with RAG."""
    sections = []

    # Portfolio context
    sections.append(get_portfolio_context(account_info))
    sections.append("")

    # Retrieved documents (RAG)
    sections.append(format_retrieved_context(retrieved))
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
    Run an ideation session with RAG retrieval.

    1. Determine retrieval targets (tickers from positions/theses, themes)
    2. Retrieve relevant documents via RAG
    3. Build context with retrieved documents
    4. Call LLM for thesis review and generation
    5. Validate citations in response
    6. Apply changes to database

    Args:
        model: Ollama model to use

    Returns:
        IdeationResult with session details
    """
    timestamp = datetime.now()
    print(f"[{timestamp.isoformat()}] Starting ideation session")
    print(f"  Model: {model}")

    # Step 1: Determine retrieval targets
    print("\n[Step 1] Determining retrieval targets...")
    positions = {p["ticker"] for p in get_positions()}
    active_theses = get_active_theses()
    active_thesis_tickers = {t["ticker"] for t in active_theses}

    # Tickers to research: positions + thesis tickers
    tickers_to_research = list(positions | active_thesis_tickers)
    print(f"  Tickers to research: {', '.join(tickers_to_research) or 'None'}")

    # Define themes for semantic search
    themes = [
        "AI semiconductor demand growth",
        "Federal Reserve interest rate policy",
        "Tech earnings and revenue guidance",
        "Sector rotation and market momentum",
        "Economic recession indicators",
    ]
    print(f"  Themes: {len(themes)}")

    # Step 2: Retrieve documents
    print("\n[Step 2] Retrieving documents...")
    try:
        retrieved = retrieve_for_ideation(
            tickers=tickers_to_research,
            themes=themes,
            k_per_ticker=5,
            k_per_theme=5,
        )
        doc_count = sum(len(docs) for docs in retrieved.get("by_ticker", {}).values())
        doc_count += sum(len(docs) for docs in retrieved.get("by_theme", {}).values())
        print(f"  Retrieved {doc_count} documents")
    except Exception as e:
        print(f"  Warning: Could not retrieve documents: {e}")
        retrieved = {"by_ticker": {}, "by_theme": {}}
        doc_count = 0

    # Step 3: Build context
    print("\n[Step 3] Building ideation context...")
    try:
        account_info = get_account_info()
    except Exception as e:
        print(f"  Warning: Could not get account info: {e}")
        account_info = {"cash": 0, "portfolio_value": 0, "buying_power": 0}

    context = build_ideation_context(account_info, retrieved)
    print(f"  Context built ({len(context)} chars)")

    # Step 4: Call LLM
    print("\n[Step 4] Calling LLM for ideation...")
    prompt = f"""Here is the current market context with retrieved documents. Review existing theses and generate new trade ideas.

IMPORTANT: Only use information from the retrieved documents. Cite sources using [DOC-ID].

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

    # Step 5: Validate citations
    print("\n[Step 5] Validating citations...")
    # Collect all retrieved doc IDs for validation
    valid_doc_ids = set()
    for docs in retrieved.get("by_ticker", {}).values():
        for doc in docs:
            valid_doc_ids.add(doc["id"])
    for docs in retrieved.get("by_theme", {}).values():
        for doc in docs:
            valid_doc_ids.add(doc["id"])

    for thesis_data in response.get("new_theses", []):
        sources = thesis_data.get("sources", [])
        if not sources:
            print(f"  Warning: {thesis_data.get('ticker', '?')} thesis has no sources cited")
        else:
            invalid_sources = [s for s in sources if s not in valid_doc_ids]
            if invalid_sources:
                print(f"  Warning: {thesis_data.get('ticker', '?')} cites invalid sources: {invalid_sources}")
            else:
                print(f"  {thesis_data.get('ticker', '?')}: {len(sources)} valid sources")

    # Step 6: Process reviews
    print("\n[Step 6] Processing thesis reviews...")
    reviews = []
    theses_kept = 0
    theses_updated = 0
    theses_closed = 0

    for review_data in response.get("reviews", []):
        review = ThesisReview(
            thesis_id=review_data["thesis_id"],
            action=review_data["action"],
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

    # Step 7: Process new theses
    print("\n[Step 7] Creating new theses...")
    new_theses = []
    theses_created = 0

    for thesis_data in response.get("new_theses", []):
        # Skip if ticker already in portfolio or has active thesis
        ticker = thesis_data["ticker"]
        if ticker in positions:
            print(f"  Skipping {ticker}: already in portfolio")
            continue
        if ticker in active_thesis_tickers:
            print(f"  Skipping {ticker}: already has active thesis")
            continue

        thesis = NewThesis(
            ticker=ticker,
            direction=thesis_data["direction"],
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
    print("Ideation Session Complete (RAG-enabled)")
    print("=" * 60)
    print(f"  Documents retrieved: {doc_count}")
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
