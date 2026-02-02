"""Phi-3 based news classification."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .ollama import chat_json
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


def classify_news(headline: str, published_at: datetime) -> ClassificationResult:
    """
    Classify a news headline using Phi-3.

    Args:
        headline: News headline text
        published_at: When the news was published

    Returns:
        ClassificationResult with type and extracted signals
    """
    prompt = CLASSIFICATION_PROMPT.format(headline=headline)

    try:
        result = chat_json(prompt, model="phi3:mini")
    except ValueError:
        # If JSON parsing fails, treat as noise
        return ClassificationResult(
            news_type="noise",
            ticker_signals=[],
            macro_signal=None
        )

    news_type = result.get("type", "noise")
    ticker_signals = []
    macro_signal = None

    if news_type == "ticker_specific":
        tickers = result.get("tickers", [])
        for ticker in tickers:
            ticker_signals.append(TickerSignal(
                ticker=ticker.upper(),
                headline=headline,
                category=result.get("category", "noise"),
                sentiment=result.get("sentiment", "neutral"),
                confidence=result.get("confidence", "low"),
                published_at=published_at
            ))

    elif news_type == "macro_political":
        macro_signal = MacroSignal(
            headline=headline,
            category=result.get("category", "geopolitical"),
            affected_sectors=result.get("affected_sectors", ["all"]),
            sentiment=result.get("sentiment", "neutral"),
            published_at=published_at
        )

    elif news_type == "sector":
        # Treat sector news as macro affecting specific sectors
        macro_signal = MacroSignal(
            headline=headline,
            category="sector",
            affected_sectors=result.get("affected_sectors", ["all"]),
            sentiment=result.get("sentiment", "neutral"),
            published_at=published_at
        )

    return ClassificationResult(
        news_type=news_type,
        ticker_signals=ticker_signals,
        macro_signal=macro_signal
    )


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
        result = chat_json(prompt, model="phi3:mini")
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
    results = []
    total = len(filtered_items)

    for i, filtered in enumerate(filtered_items):
        item = filtered.item
        result = classify_news(item.headline, item.published_at)
        results.append(result)

        if (i + 1) % 5 == 0:
            print(f"  Classified {i + 1}/{total} items")

    return results
