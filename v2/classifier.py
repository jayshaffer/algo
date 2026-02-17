"""News classification using Claude Haiku with batch support."""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .claude_client import get_claude_client


def _sanitize_headline(headline: str) -> str:
    """Sanitize a headline before inserting into an LLM prompt.

    Strips control characters, collapses whitespace, and truncates to 300 chars.
    """
    headline = re.sub(r'[\x00-\x1f\x7f]', ' ', headline)
    headline = re.sub(r'\s+', ' ', headline).strip()
    headline = headline[:300]
    return headline


CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class TickerSignal:
    """Classified ticker-specific news signal."""
    ticker: str
    headline: str
    category: str      # earnings, guidance, analyst, product, legal, noise
    sentiment: str     # bullish, bearish, neutral
    confidence: str    # high, medium, low
    published_at: datetime


@dataclass
class MacroSignal:
    """Classified macro/political news signal."""
    headline: str
    category: str           # fed, trade, regulation, geopolitical, fiscal, election
    affected_sectors: list[str]  # tech, finance, energy, healthcare, defense, all
    sentiment: str          # bullish, bearish, neutral
    published_at: datetime


@dataclass
class ClassificationResult:
    """Result of news classification."""
    news_type: str  # ticker_specific, macro_political, sector, noise
    ticker_signals: list[TickerSignal]
    macro_signal: Optional[MacroSignal]


CLASSIFICATION_SYSTEM = """Classify news headlines for stock trading relevance. Respond with JSON only.

JSON schema:
{
    "type": "ticker_specific" | "macro_political" | "sector" | "noise",
    "tickers": ["AAPL", "MSFT"],
    "category": "earnings" | "guidance" | "analyst" | "product" | "legal" | "fed" | "trade" | "regulation" | "geopolitical" | "fiscal" | "election" | "noise",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low",
    "affected_sectors": ["tech", "finance"]
}

Rules:
- ticker_specific: News about specific companies (earnings, guidance, analyst ratings, products, legal)
- macro_political: Fed policy, trade/tariffs, regulations, geopolitical events, fiscal policy, elections
- sector: General sector news without specific tickers
- noise: Irrelevant to trading (celebrity news, sports, etc.)
- tickers: Stock ticker symbols mentioned (empty list if not ticker_specific)
- affected_sectors can include: tech, finance, energy, healthcare, defense, consumer, industrial, all"""

_CACHED_CLASSIFICATION_SYSTEM = [
    {"type": "text", "text": CLASSIFICATION_SYSTEM, "cache_control": {"type": "ephemeral"}}
]


TICKER_CLASSIFICATION_SYSTEM = """Classify news for a specific stock ticker. Respond with JSON only.

JSON schema:
{
    "category": "earnings" | "guidance" | "analyst" | "product" | "legal" | "noise",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low"
}

Category meanings:
- earnings: Quarterly/annual results, revenue, profit
- guidance: Forward-looking statements, forecasts
- analyst: Ratings, price targets, recommendations
- product: Product launches, features, partnerships
- legal: Lawsuits, regulatory actions, investigations
- noise: Not directly relevant to stock price"""

_CACHED_TICKER_CLASSIFICATION_SYSTEM = [
    {"type": "text", "text": TICKER_CLASSIFICATION_SYSTEM, "cache_control": {"type": "ephemeral"}}
]


BATCH_CLASSIFICATION_SYSTEM = """Classify each news headline for stock trading relevance.

For each headline, respond with a JSON array containing one object per headline (in the same order):
[
  {
    "type": "ticker_specific" | "macro_political" | "sector" | "noise",
    "tickers": ["AAPL"],
    "category": "earnings" | "guidance" | "analyst" | "product" | "legal" | "fed" | "trade" | "regulation" | "geopolitical" | "fiscal" | "election" | "noise",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low",
    "affected_sectors": ["tech"]
  }
]

Rules:
- ticker_specific: News about specific companies (earnings, guidance, analyst ratings, products, legal)
- macro_political: Fed policy, trade/tariffs, regulations, geopolitical events, fiscal policy, elections
- sector: General sector news without specific tickers
- noise: Irrelevant to trading
- tickers: Stock ticker symbols mentioned (empty list if not ticker_specific)
- affected_sectors: tech, finance, energy, healthcare, defense, consumer, industrial, all
- Return exactly one object per headline, in the same order"""

_CACHED_BATCH_CLASSIFICATION_SYSTEM = [
    {"type": "text", "text": BATCH_CLASSIFICATION_SYSTEM, "cache_control": {"type": "ephemeral"}}
]


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _build_classification_result(
    entry: dict, headline: str, published_at: datetime
) -> ClassificationResult:
    """Build a ClassificationResult from a parsed JSON entry."""
    news_type = entry.get("type", "noise")
    ticker_signals = []
    macro_signal = None

    if news_type == "ticker_specific":
        tickers = entry.get("tickers", [])
        for ticker in tickers:
            ticker_signals.append(TickerSignal(
                ticker=ticker.upper(),
                headline=headline,
                category=entry.get("category", "noise"),
                sentiment=entry.get("sentiment", "neutral"),
                confidence=entry.get("confidence", "low"),
                published_at=published_at
            ))
    elif news_type == "macro_political":
        macro_signal = MacroSignal(
            headline=headline,
            category=entry.get("category", "geopolitical"),
            affected_sectors=entry.get("affected_sectors", ["all"]),
            sentiment=entry.get("sentiment", "neutral"),
            published_at=published_at
        )
    elif news_type == "sector":
        macro_signal = MacroSignal(
            headline=headline,
            category="sector",
            affected_sectors=entry.get("affected_sectors", ["all"]),
            sentiment=entry.get("sentiment", "neutral"),
            published_at=published_at
        )

    return ClassificationResult(
        news_type=news_type,
        ticker_signals=ticker_signals,
        macro_signal=macro_signal
    )


def classify_news(
    headline: str, published_at: datetime, client=None
) -> ClassificationResult:
    """
    Classify a single news headline using Claude Haiku.

    Args:
        headline: News headline text
        published_at: When the news was published
        client: Optional Anthropic client (reused across calls)

    Returns:
        ClassificationResult with type and extracted signals
    """
    try:
        if client is None:
            client = get_claude_client()
        response = client.messages.create(
            model=CLASSIFICATION_MODEL,
            max_tokens=256,
            system=_CACHED_CLASSIFICATION_SYSTEM,
            messages=[{"role": "user", "content": _sanitize_headline(headline)}]
        )
        text = _strip_code_fences(response.content[0].text)
        result = json.loads(text)
    except Exception:
        return ClassificationResult(
            news_type="noise",
            ticker_signals=[],
            macro_signal=None
        )

    return _build_classification_result(result, headline, published_at)


def classify_news_batch(
    headlines: list[str],
    published_ats: list[datetime],
    batch_size: int = 50
) -> list[ClassificationResult]:
    """
    Classify multiple headlines in batched Claude Haiku calls.

    Sends batch_size headlines per call to reduce round-trips.
    Falls back to individual classification if a batch fails.

    Args:
        headlines: List of headline texts
        published_ats: Corresponding publication times
        batch_size: Headlines per LLM call (default 50)

    Returns:
        List of ClassificationResult, one per headline
    """
    results = []

    for start in range(0, len(headlines), batch_size):
        batch_headlines = headlines[start:start + batch_size]
        batch_dates = published_ats[start:start + batch_size]

        try:
            batch_results = _classify_batch(batch_headlines, batch_dates)
            results.extend(batch_results)
        except Exception as e:
            print(f"  Batch classification failed ({e}), falling back to individual")
            for headline, pub_date in zip(batch_headlines, batch_dates):
                try:
                    results.append(classify_news(headline, pub_date))
                except Exception:
                    results.append(ClassificationResult(
                        news_type="noise", ticker_signals=[], macro_signal=None
                    ))

    return results


def _classify_batch(
    headlines: list[str], published_ats: list[datetime]
) -> list[ClassificationResult]:
    """Classify a single batch of headlines in one Claude Haiku call."""
    headlines_block = "\n".join(
        f'{i + 1}. "{_sanitize_headline(h)}"' for i, h in enumerate(headlines)
    )

    client = get_claude_client()
    response = client.messages.create(
        model=CLASSIFICATION_MODEL,
        max_tokens=4096,
        system=_CACHED_BATCH_CLASSIFICATION_SYSTEM,
        messages=[{"role": "user", "content": headlines_block}]
    )

    # Parse JSON array from response
    text = _strip_code_fences(response.content[0].text)

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array")

    results = []
    for i, entry in enumerate(parsed):
        if i >= len(headlines):
            break
        results.append(
            _build_classification_result(entry, headlines[i], published_ats[i])
        )

    # Fill missing results with noise
    while len(results) < len(headlines):
        results.append(ClassificationResult(
            news_type="noise", ticker_signals=[], macro_signal=None
        ))

    return results


def classify_ticker_news(
    ticker: str, headline: str, published_at: datetime, client=None
) -> TickerSignal:
    """
    Classify news for a specific ticker (when ticker is already known).

    Args:
        ticker: Stock ticker symbol
        headline: News headline
        published_at: Publication time
        client: Optional Anthropic client (reused across calls)

    Returns:
        TickerSignal with classification
    """
    try:
        if client is None:
            client = get_claude_client()
        response = client.messages.create(
            model=CLASSIFICATION_MODEL,
            max_tokens=256,
            system=_CACHED_TICKER_CLASSIFICATION_SYSTEM,
            messages=[{"role": "user", "content": f"Ticker: {ticker}\nHeadline: {_sanitize_headline(headline)}"}]
        )
        text = _strip_code_fences(response.content[0].text)
        result = json.loads(text)
    except Exception:
        result = {"category": "noise", "sentiment": "neutral", "confidence": "low"}

    return TickerSignal(
        ticker=ticker.upper(),
        headline=headline,
        category=result.get("category", "noise"),
        sentiment=result.get("sentiment", "neutral"),
        confidence=result.get("confidence", "low"),
        published_at=published_at
    )
