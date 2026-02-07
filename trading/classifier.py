"""News classification using qwen3:14b with batch support."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .ollama import chat, chat_json
from .filter import FilteredNewsItem


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


CLASSIFICATION_PROMPT = """Classify this news headline for stock trading relevance.

Headline: "{headline}"

Determine the type and extract relevant information.

Respond with JSON only:
{{
    "type": "ticker_specific" | "macro_political" | "sector" | "noise",
    "tickers": ["AAPL", "MSFT"],  // if ticker_specific, list mentioned tickers
    "category": "earnings" | "guidance" | "analyst" | "product" | "legal" | "fed" | "trade" | "regulation" | "geopolitical" | "fiscal" | "election" | "noise",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low",
    "affected_sectors": ["tech", "finance"]  // if macro_political, list affected sectors
}}

Rules:
- ticker_specific: News about specific companies (earnings, guidance, analyst ratings, products, legal)
- macro_political: Fed policy, trade/tariffs, regulations, geopolitical events, fiscal policy, elections
- sector: General sector news without specific tickers
- noise: Irrelevant to trading (celebrity news, sports, etc.)

For macro news, affected_sectors can include: tech, finance, energy, healthcare, defense, consumer, industrial, all"""


TICKER_CLASSIFICATION_PROMPT = """Classify this news for the stock ticker {ticker}.

Headline: "{headline}"

Respond with JSON only:
{{
    "category": "earnings" | "guidance" | "analyst" | "product" | "legal" | "noise",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low"
}}

Category meanings:
- earnings: Quarterly/annual results, revenue, profit
- guidance: Forward-looking statements, forecasts
- analyst: Ratings, price targets, recommendations
- product: Product launches, features, partnerships
- legal: Lawsuits, regulatory actions, investigations
- noise: Not directly relevant to stock price"""


BATCH_CLASSIFICATION_PROMPT = """Classify each news headline for stock trading relevance.

{headlines_block}

For each headline, respond with a JSON array containing one object per headline (in the same order):
[
  {{
    "type": "ticker_specific" | "macro_political" | "sector" | "noise",
    "tickers": ["AAPL"],
    "category": "earnings" | "guidance" | "analyst" | "product" | "legal" | "fed" | "trade" | "regulation" | "geopolitical" | "fiscal" | "election" | "noise",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low",
    "affected_sectors": ["tech"]
  }}
]

Rules:
- ticker_specific: News about specific companies (earnings, guidance, analyst ratings, products, legal)
- macro_political: Fed policy, trade/tariffs, regulations, geopolitical events, fiscal policy, elections
- sector: General sector news without specific tickers
- noise: Irrelevant to trading
- tickers: Stock ticker symbols mentioned (empty list if not ticker_specific)
- affected_sectors: tech, finance, energy, healthcare, defense, consumer, industrial, all
- You MUST return exactly {count} objects in the array, one per headline, in the same order"""


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


def classify_news(headline: str, published_at: datetime) -> ClassificationResult:
    """
    Classify a single news headline using qwen3:14b.

    Args:
        headline: News headline text
        published_at: When the news was published

    Returns:
        ClassificationResult with type and extracted signals
    """
    prompt = CLASSIFICATION_PROMPT.format(headline=headline)

    try:
        result = chat_json(prompt, model="qwen3:14b")
    except ValueError:
        return ClassificationResult(
            news_type="noise",
            ticker_signals=[],
            macro_signal=None
        )

    return _build_classification_result(result, headline, published_at)


def classify_news_batch(
    headlines: list[str],
    published_ats: list[datetime],
    batch_size: int = 10
) -> list[ClassificationResult]:
    """
    Classify multiple headlines in batched LLM calls.

    Sends batch_size headlines per call to reduce round-trips.
    Falls back to individual classification if a batch fails.

    Args:
        headlines: List of headline texts
        published_ats: Corresponding publication times
        batch_size: Headlines per LLM call (default 10)

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
    """Classify a single batch of headlines in one LLM call."""
    headlines_block = "\n".join(
        f'{i + 1}. "{h}"' for i, h in enumerate(headlines)
    )
    prompt = BATCH_CLASSIFICATION_PROMPT.format(
        headlines_block=headlines_block,
        count=len(headlines)
    )

    response = chat(prompt, model="qwen3:14b", temperature=0.0)

    # Parse JSON array from response
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

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


def classify_ticker_news(ticker: str, headline: str, published_at: datetime) -> TickerSignal:
    """
    Classify news for a specific ticker (when ticker is already known).

    Args:
        ticker: Stock ticker symbol
        headline: News headline
        published_at: Publication time

    Returns:
        TickerSignal with classification
    """
    prompt = TICKER_CLASSIFICATION_PROMPT.format(ticker=ticker, headline=headline)

    try:
        result = chat_json(prompt, model="qwen3:14b")
    except ValueError:
        result = {"category": "noise", "sentiment": "neutral", "confidence": "low"}

    return TickerSignal(
        ticker=ticker.upper(),
        headline=headline,
        category=result.get("category", "noise"),
        sentiment=result.get("sentiment", "neutral"),
        confidence=result.get("confidence", "low"),
        published_at=published_at
    )


def classify_filtered_news(filtered_items: list[FilteredNewsItem]) -> list[ClassificationResult]:
    """
    Classify a batch of filtered news items.

    Args:
        filtered_items: News items that passed relevance filter

    Returns:
        List of classification results
    """
    headlines = [f.item.headline for f in filtered_items]
    published_ats = [f.item.published_at for f in filtered_items]
    return classify_news_batch(headlines, published_ats)
