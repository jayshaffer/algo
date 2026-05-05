"""News classification using Claude Haiku with batch support."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from .claude_client import _call_with_retry, get_claude_client

logger = logging.getLogger(__name__)

# Allowed category enums. Out-of-vocab values from the LLM (typos like
# "earnigns", hallucinations like "frobnozzle") are coerced to a default
# rather than persisted verbatim — otherwise they create orphan attribution
# buckets like the news_signal:unknown artifact patched 2026-05-02.
VALID_TICKER_CATEGORIES = frozenset({
    "earnings", "guidance", "analyst", "product", "legal", "noise",
})
VALID_MACRO_CATEGORIES = frozenset({
    "fed", "trade", "regulation", "geopolitical", "fiscal", "election",
    "sector",  # set explicitly by the sector-type branch
})


def _coerce_category(value: str, allowed: frozenset, default: str, context: str) -> str:
    if value in allowed:
        return value
    logger.warning(
        "classifier: invalid %s category %r — coercing to %r",
        context, value, default,
    )
    return default


# P3.40: ticker validation. Haiku occasionally emits hallucinated tickers
# (group acronyms like FAANG/MAANG/MAGS, generic acronyms like ETF/CEO/AI,
# or `$TSLA`-prefixed strings). Without filtering, these become rows in
# news_signals and pollute attribution.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
_TICKER_BLOCKLIST = frozenset({
    # group/category acronyms — never tradable tickers
    "FAANG", "MAANG", "MAGS", "MAG", "FAAMG", "BATX", "BAT",
    # generic non-equity acronyms commonly mentioned in news.
    # NOTE: ADP (Automatic Data Processing) and ETN (Eaton Corp) are real
    # S&P 500 tickers that share names with macro indicators / instrument
    # classes — leaving them out matches the BE/ON/GO tradeoff above.
    "ETF", "IPO", "SPAC", "REIT", "MLP",
    "CEO", "CFO", "CTO", "COO", "CIO", "CMO", "CISO",
    "SEC", "FDA", "FTC", "DOJ", "FBI", "CIA", "NSA", "EPA", "IRS",
    "GDP", "CPI", "PPI", "PCE", "PMI", "ISM",
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY",
    # platform/sector buzzwords (lowercase tradable equivalents may exist
    # — block only when LLM emits them as if they were tickers)
    "ESG", "DEI",
})


def _validate_ticker(raw: str) -> str | None:
    """Return a cleaned ticker string, or None if `raw` is not a plausible
    equity ticker. Catches the common hallucinations: $-prefixed strings,
    group acronyms (FAANG), and overly-long alphanumeric soup. Real tickers
    that happen to look like English stop-words (BE, ON, SO, GO) are NOT
    blocked — without an Alpaca-asset allowlist we can't disambiguate, and
    blocking them would cost real signal."""
    if not isinstance(raw, str):
        return None
    # Strip trailing sentence punctuation copied from prose (e.g. "AAPL.", "NVDA,").
    # Class-share tickers like "BRK.A" are unaffected because their dot is followed
    # by a letter, not by end-of-string.
    cleaned = raw.strip().lstrip("$").rstrip(".,;:!?").upper()
    if not cleaned:
        return None
    if not _TICKER_RE.match(cleaned):
        logger.warning("classifier: rejecting non-ticker-shaped string %r", raw)
        return None
    if cleaned in _TICKER_BLOCKLIST:
        logger.warning("classifier: rejecting hallucinated/acronym ticker %r", raw)
        return None
    return cleaned


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
    alpaca_id: str | None = None  # P2.16: source news id for canonical dedup


@dataclass
class MacroSignal:
    """Classified macro/political news signal."""
    headline: str
    category: str           # fed, trade, regulation, geopolitical, fiscal, election
    affected_sectors: list[str]  # tech, finance, energy, healthcare, defense, all
    sentiment: str          # bullish, bearish, neutral
    published_at: datetime
    alpaca_id: str | None = None  # P2.16: source news id for canonical dedup


@dataclass
class ClassificationResult:
    """Result of news classification."""
    news_type: str  # ticker_specific, macro_political, sector, noise
    ticker_signals: list[TickerSignal]
    macro_signal: MacroSignal | None


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

For each headline, respond with a JSON array of objects. Each object MUST include
an "index" field matching the 1-based number from the user's headline list:
[
  {
    "index": 1,
    "type": "ticker_specific" | "macro_political" | "sector" | "noise",
    "tickers": ["AAPL"],
    "category": "earnings" | "guidance" | "analyst" | "product" | "legal" | "fed" | "trade" | "regulation" | "geopolitical" | "fiscal" | "election" | "noise",
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": "high" | "medium" | "low",
    "affected_sectors": ["tech"]
  }
]

Rules:
- index: REQUIRED. The 1-based headline number from the user message. Used to match your output back to the right headline; do NOT rely on array order.
- ticker_specific: News about specific companies (earnings, guidance, analyst ratings, products, legal)
- macro_political: Fed policy, trade/tariffs, regulations, geopolitical events, fiscal policy, elections
- sector: General sector news without specific tickers
- noise: Irrelevant to trading
- tickers: Stock ticker symbols mentioned (empty list if not ticker_specific)
- affected_sectors: tech, finance, energy, healthcare, defense, consumer, industrial, all
- Return exactly one object per headline. If you cannot classify a headline, return {"index": N, "type": "noise"} for it."""

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
    entry: dict, headline: str, published_at: datetime, alpaca_id: str | None = None
) -> ClassificationResult:
    """Build a ClassificationResult from a parsed JSON entry."""
    news_type = entry.get("type", "noise")
    ticker_signals = []
    macro_signal = None

    if news_type == "ticker_specific":
        tickers = entry.get("tickers", [])
        category = _coerce_category(
            entry.get("category", "noise"), VALID_TICKER_CATEGORIES, "noise", "ticker",
        )
        for ticker in tickers:
            cleaned = _validate_ticker(ticker)
            if cleaned is None:
                continue
            ticker_signals.append(TickerSignal(
                ticker=cleaned,
                headline=headline,
                category=category,
                sentiment=entry.get("sentiment", "neutral"),
                confidence=entry.get("confidence", "low"),
                published_at=published_at,
                alpaca_id=alpaca_id,
            ))
    elif news_type == "macro_political":
        macro_signal = MacroSignal(
            headline=headline,
            category=_coerce_category(
                entry.get("category", "geopolitical"),
                VALID_MACRO_CATEGORIES, "geopolitical", "macro",
            ),
            affected_sectors=entry.get("affected_sectors", ["all"]),
            sentiment=entry.get("sentiment", "neutral"),
            published_at=published_at,
            alpaca_id=alpaca_id,
        )
    elif news_type == "sector":
        macro_signal = MacroSignal(
            headline=headline,
            category="sector",
            affected_sectors=entry.get("affected_sectors", ["all"]),
            sentiment=entry.get("sentiment", "neutral"),
            published_at=published_at,
            alpaca_id=alpaca_id,
        )

    return ClassificationResult(
        news_type=news_type,
        ticker_signals=ticker_signals,
        macro_signal=macro_signal
    )


def classify_news(
    headline: str, published_at: datetime, client=None, alpaca_id: str | None = None
) -> ClassificationResult:
    """
    Classify a single news headline using Claude Haiku.

    Args:
        headline: News headline text
        published_at: When the news was published
        client: Optional Anthropic client (reused across calls)
        alpaca_id: Source Alpaca news id for canonical dedup (P2.16)

    Returns:
        ClassificationResult with type and extracted signals
    """
    try:
        if client is None:
            client = get_claude_client()
        # P2.20: retry on rate limits / transient 5xx instead of immediately
        # falling through to "noise" on the first 429.
        response = _call_with_retry(
            client,
            model=CLASSIFICATION_MODEL,
            max_tokens=256,
            system=_CACHED_CLASSIFICATION_SYSTEM,
            messages=[{"role": "user", "content": _sanitize_headline(headline)}],
        )
        text = _strip_code_fences(response.content[0].text)
        result = json.loads(text)
    except Exception:
        return ClassificationResult(
            news_type="noise",
            ticker_signals=[],
            macro_signal=None
        )

    return _build_classification_result(result, headline, published_at, alpaca_id)


def classify_news_batch(
    headlines: list[str],
    published_ats: list[datetime],
    batch_size: int = 50,
    alpaca_ids: list[str | None] | None = None,
) -> list[ClassificationResult]:
    """
    Classify multiple headlines in batched Claude Haiku calls.

    Sends batch_size headlines per call to reduce round-trips.
    Falls back to individual classification if a batch fails.

    Args:
        headlines: List of headline texts
        published_ats: Corresponding publication times
        batch_size: Headlines per LLM call (default 50)
        alpaca_ids: Source Alpaca news ids, used for canonical dedup (P2.16)

    Returns:
        List of ClassificationResult, one per headline
    """
    if alpaca_ids is None:
        alpaca_ids = [None] * len(headlines)

    results = []

    import anthropic as _anthropic

    for start in range(0, len(headlines), batch_size):
        batch_headlines = headlines[start:start + batch_size]
        batch_dates = published_ats[start:start + batch_size]
        batch_ids = alpaca_ids[start:start + batch_size]

        try:
            batch_results = _classify_batch(batch_headlines, batch_dates, batch_ids)
            results.extend(batch_results)
        except _anthropic.RateLimitError as e:
            # P2.20: don't fan out N per-call retries on rate limit — the batch
            # already retried 3 times. Fanning out would hammer the same minute.
            logger.error("Batch classification rate-limited beyond retries (%s) — marking batch as noise", e)
            for _ in batch_headlines:
                results.append(ClassificationResult(
                    news_type="noise", ticker_signals=[], macro_signal=None
                ))
        except Exception as e:
            logger.warning("Batch classification failed (%s), falling back to individual", e)
            for headline, pub_date, aid in zip(batch_headlines, batch_dates, batch_ids, strict=True):
                try:
                    results.append(classify_news(headline, pub_date, alpaca_id=aid))
                except Exception:
                    results.append(ClassificationResult(
                        news_type="noise", ticker_signals=[], macro_signal=None
                    ))

    return results


def _classify_batch(
    headlines: list[str], published_ats: list[datetime],
    alpaca_ids: list[str | None] | None = None,
) -> list[ClassificationResult]:
    """Classify a single batch of headlines in one Claude Haiku call."""
    headlines_block = "\n".join(
        f'{i + 1}. "{_sanitize_headline(h)}"' for i, h in enumerate(headlines)
    )

    client = get_claude_client()
    # P2.20: retry on rate limits / transient 5xx. Without this, a single 429
    # collapses the batch and triggers the per-headline fallback — a 50× API
    # fan-out within the same minute that almost certainly hits the same limit.
    response = _call_with_retry(
        client,
        model=CLASSIFICATION_MODEL,
        max_tokens=4096,
        system=_CACHED_BATCH_CLASSIFICATION_SYSTEM,
        messages=[{"role": "user", "content": headlines_block}],
    )

    # Parse JSON array from response
    text = _strip_code_fences(response.content[0].text)

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array")

    if alpaca_ids is None:
        alpaca_ids = [None] * len(headlines)

    # P2.15: index-based mapping instead of positional zip. Haiku may reorder
    # or truncate the array; previously parsed[i] paired with headlines[i]
    # silently, planting wrong tickers/sentiments on wrong headlines.
    by_idx: dict[int, dict] = {}
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if isinstance(idx, int) and 1 <= idx <= len(headlines):
            by_idx[idx - 1] = entry

    results = []
    for i in range(len(headlines)):
        entry = by_idx.get(i)
        if entry is None:
            results.append(ClassificationResult(
                news_type="noise", ticker_signals=[], macro_signal=None
            ))
        else:
            results.append(
                _build_classification_result(
                    entry, headlines[i], published_ats[i], alpaca_ids[i]
                )
            )

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
        response = _call_with_retry(
            client,
            model=CLASSIFICATION_MODEL,
            max_tokens=256,
            system=_CACHED_TICKER_CLASSIFICATION_SYSTEM,
            messages=[{"role": "user", "content": f"Ticker: {ticker}\nHeadline: {_sanitize_headline(headline)}"}],
        )
        text = _strip_code_fences(response.content[0].text)
        result = json.loads(text)
    except Exception:
        result = {"category": "noise", "sentiment": "neutral", "confidence": "low"}

    return TickerSignal(
        ticker=ticker.upper(),
        headline=headline,
        category=_coerce_category(
            result.get("category", "noise"), VALID_TICKER_CATEGORIES, "noise", "ticker",
        ),
        sentiment=result.get("sentiment", "neutral"),
        confidence=result.get("confidence", "low"),
        published_at=published_at
    )
