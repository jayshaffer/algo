"""Integration tests that send real prompts to Ollama and validate responses.

These tests require a running Ollama instance with qwen3:14b pulled.
They verify the model produces correctly structured JSON for every
prompt template used in the trading system.

Run with:
    python3 -m pytest tests/test_model_integration.py -m integration

Skip in normal test runs (no Ollama needed):
    python3 -m pytest tests/ -m "not integration"
"""

import json
import logging
import time

import pytest

from trading.ollama import chat, chat_json, check_ollama_health, list_models
from trading.agent import TRADING_SYSTEM_PROMPT
from trading.classifier import (
    CLASSIFICATION_PROMPT,
    TICKER_CLASSIFICATION_PROMPT,
    BATCH_CLASSIFICATION_PROMPT,
)
from trading.ideation import IDEATION_SYSTEM_PROMPT

logger = logging.getLogger("test_model_integration")

MODEL = "qwen3:14b"

VALID_NEWS_TYPES = {"ticker_specific", "macro_political", "sector", "noise"}
VALID_CATEGORIES = {
    "earnings", "guidance", "analyst", "product", "legal",
    "fed", "trade", "regulation", "geopolitical", "fiscal", "election",
    "noise", "sector",
}
VALID_SENTIMENTS = {"bullish", "bearish", "neutral"}
VALID_CONFIDENCES = {"high", "medium", "low"}
VALID_ACTIONS = {"buy", "sell", "hold"}
VALID_DIRECTIONS = {"long", "short", "avoid"}
VALID_THESIS_ACTIONS = {"keep", "update", "invalidate", "expire"}


def ollama_available():
    """Check if Ollama is running and has the required model."""
    try:
        if not check_ollama_health():
            return False
        models = list_models()
        return any("qwen3" in m for m in models)
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running or qwen3:14b not available",
)
integration = pytest.mark.integration


def _timed_chat_json(prompt, *, model=MODEL, system=None):
    """Call chat_json with timing and logging."""
    logger.debug("Prompt (%d chars): %.500s", len(prompt), prompt)
    start = time.time()
    result = chat_json(prompt, model=model, system=system)
    elapsed = time.time() - start
    logger.info("chat_json returned in %.1fs", elapsed)
    logger.debug("Parsed result: %s", json.dumps(result, indent=2, default=str)[:2000])
    return result


def _timed_chat(prompt, *, model=MODEL, temperature=0.0):
    """Call chat with timing and logging."""
    logger.debug("Prompt (%d chars): %.500s", len(prompt), prompt)
    start = time.time()
    response = chat(prompt, model=model, temperature=temperature)
    elapsed = time.time() - start
    logger.info("chat returned in %.1fs (%d chars)", elapsed, len(response))
    logger.debug("Raw response: %.2000s", response)
    return response


# ---------------------------------------------------------------------------
# News classification - single headline
# ---------------------------------------------------------------------------

@integration
@skip_no_ollama
class TestClassificationPrompt:
    """Verify qwen3:14b responds correctly to the single-headline classification prompt."""

    def test_ticker_specific_headline(self):
        """Earnings headline should be classified as ticker_specific."""
        prompt = CLASSIFICATION_PROMPT.format(
            headline="Apple reports Q3 revenue of $81.8B, beating estimates by $2B"
        )
        result = _timed_chat_json(prompt)

        assert result["type"] == "ticker_specific"
        assert isinstance(result["tickers"], list)
        assert "AAPL" in result["tickers"]
        assert result["category"] in VALID_CATEGORIES
        assert result["sentiment"] in VALID_SENTIMENTS
        assert result["confidence"] in VALID_CONFIDENCES

    def test_macro_headline(self):
        """Fed rate decision should be classified as macro_political."""
        prompt = CLASSIFICATION_PROMPT.format(
            headline="Federal Reserve holds interest rates steady at 5.25-5.50%"
        )
        result = _timed_chat_json(prompt)

        assert result["type"] == "macro_political"
        assert result["category"] in VALID_CATEGORIES
        assert result["sentiment"] in VALID_SENTIMENTS
        assert isinstance(result.get("affected_sectors", []), list)

    def test_noise_headline(self):
        """Celebrity news should be classified as noise."""
        prompt = CLASSIFICATION_PROMPT.format(
            headline="Taylor Swift announces new world tour dates for 2026"
        )
        result = _timed_chat_json(prompt)

        assert result["type"] == "noise"

    def test_response_has_required_fields(self):
        """Every classification response must have type, category, sentiment, confidence."""
        prompt = CLASSIFICATION_PROMPT.format(
            headline="NVIDIA launches new H200 GPU for AI training workloads"
        )
        result = _timed_chat_json(prompt)

        assert "type" in result
        assert result["type"] in VALID_NEWS_TYPES
        assert "category" in result
        assert "sentiment" in result
        assert result["sentiment"] in VALID_SENTIMENTS
        assert "confidence" in result
        assert result["confidence"] in VALID_CONFIDENCES


# ---------------------------------------------------------------------------
# News classification - ticker-specific
# ---------------------------------------------------------------------------

@integration
@skip_no_ollama
class TestTickerClassificationPrompt:
    """Verify qwen3:14b responds correctly to the ticker-specific classification prompt."""

    def test_earnings_headline(self):
        """Earnings headline for known ticker should classify correctly."""
        prompt = TICKER_CLASSIFICATION_PROMPT.format(
            ticker="MSFT",
            headline="Microsoft cloud revenue surges 29% in Q2, beating Wall Street expectations"
        )
        result = _timed_chat_json(prompt)

        assert result["category"] in VALID_CATEGORIES
        assert result["sentiment"] in VALID_SENTIMENTS
        assert result["confidence"] in VALID_CONFIDENCES

    def test_legal_headline(self):
        """Lawsuit headline should classify as legal."""
        prompt = TICKER_CLASSIFICATION_PROMPT.format(
            ticker="GOOGL",
            headline="Google faces $2.3B antitrust fine from European Commission"
        )
        result = _timed_chat_json(prompt)

        assert result["category"] == "legal"
        assert result["sentiment"] == "bearish"

    def test_irrelevant_headline(self):
        """Unrelated headline should classify as noise."""
        prompt = TICKER_CLASSIFICATION_PROMPT.format(
            ticker="AAPL",
            headline="Local bakery wins award for best sourdough in Portland"
        )
        result = _timed_chat_json(prompt)

        assert result["category"] == "noise"


# ---------------------------------------------------------------------------
# News classification - batch
# ---------------------------------------------------------------------------

@integration
@skip_no_ollama
class TestBatchClassificationPrompt:
    """Verify qwen3:14b responds correctly to the batch classification prompt."""

    def test_batch_returns_correct_count(self):
        """Batch prompt with 3 headlines should return exactly 3 results."""
        headlines = [
            '1. "Tesla reports record deliveries of 500K vehicles in Q4"',
            '2. "Federal Reserve signals potential rate cut in September"',
            '3. "Local dog show breaks attendance record in Milwaukee"',
        ]
        prompt = BATCH_CLASSIFICATION_PROMPT.format(
            headlines_block="\n".join(headlines),
            count=3,
        )
        response = _timed_chat(prompt)

        # Parse JSON array
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_batch_elements_have_required_fields(self):
        """Each element in batch response must have type, category, sentiment."""
        headlines = [
            '1. "Amazon AWS revenue grows 17% year over year"',
            '2. "Oil prices surge after OPEC announces production cuts"',
        ]
        prompt = BATCH_CLASSIFICATION_PROMPT.format(
            headlines_block="\n".join(headlines),
            count=2,
        )
        response = _timed_chat(prompt)

        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        parsed = json.loads(text.strip())
        for entry in parsed:
            assert "type" in entry
            assert entry["type"] in VALID_NEWS_TYPES
            assert "sentiment" in entry
            assert entry["sentiment"] in VALID_SENTIMENTS


# ---------------------------------------------------------------------------
# Trading decisions
# ---------------------------------------------------------------------------

@integration
@skip_no_ollama
class TestTradingDecisionPrompt:
    """Verify qwen3:14b produces valid trading decision JSON."""

    SAMPLE_CONTEXT = """Today's Playbook:
  Market outlook: Cautious — Fed meeting tomorrow, expect volatility
  Priority actions:
    1. BUY NVDA — entry trigger hit at $850, thesis #1, max 5 shares, confidence 0.8
  Watch list: AAPL, MSFT, GOOGL
  Risk notes: Reduce exposure if VIX > 25

Portfolio:
  Cash: $50,000
  Buying power: $100,000
  Positions:
    AAPL: 20 shares @ $175 avg (current: $182)
    MSFT: 15 shares @ $410 avg (current: $415)

Active Theses:
  ID 1: NVDA (long) — high confidence
    Thesis: AI datacenter demand acceleration
    Entry trigger: Price below $860
    Exit trigger: $1100 or below $750
    Invalidation: Datacenter revenue growth < 50% YoY

Macro Context (last 7 days):
  [macro_signal id=5] Fed holds rates steady — neutral for finance, tech

Overnight Ticker Signals:
  [news_signal id=10] NVDA: Product launch — bullish, high confidence
  [news_signal id=11] AAPL: Analyst upgrade — bullish, medium confidence

Signal Attribution:
  news:earnings — Predictive (65% win rate, n=15)
  news:product — Moderate (52% win rate, n=10)

Recent Decisions (last 5):
  AAPL BUY 10 shares — outcome_7d: +2.1%
  MSFT HOLD — outcome_7d: +0.5%"""

    def test_returns_valid_json_structure(self):
        """Trading decision response has all required top-level fields."""
        result = _timed_chat_json(
            f"Here is the current market context. Analyze and provide trading decisions.\n\n{self.SAMPLE_CONTEXT}",
            system=TRADING_SYSTEM_PROMPT,
        )

        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        assert "thesis_invalidations" in result
        assert isinstance(result["thesis_invalidations"], list)
        assert "market_summary" in result
        assert isinstance(result["market_summary"], str)
        assert "risk_assessment" in result
        assert isinstance(result["risk_assessment"], str)

    def test_decisions_have_required_fields(self):
        """Each decision must have action, ticker, reasoning, confidence."""
        result = _timed_chat_json(
            f"Here is the current market context. Analyze and provide trading decisions.\n\n{self.SAMPLE_CONTEXT}",
            system=TRADING_SYSTEM_PROMPT,
        )

        for decision in result["decisions"]:
            assert "action" in decision
            assert decision["action"] in VALID_ACTIONS
            assert "ticker" in decision
            assert isinstance(decision["ticker"], str)
            assert "reasoning" in decision
            assert isinstance(decision["reasoning"], str)
            assert "confidence" in decision
            assert decision["confidence"] in VALID_CONFIDENCES

    def test_buy_decisions_have_quantity(self):
        """Buy/sell decisions must include a positive quantity."""
        result = _timed_chat_json(
            f"Here is the current market context. Analyze and provide trading decisions.\n\n{self.SAMPLE_CONTEXT}",
            system=TRADING_SYSTEM_PROMPT,
        )

        for decision in result["decisions"]:
            if decision["action"] in ("buy", "sell"):
                assert "quantity" in decision
                assert isinstance(decision["quantity"], (int, float))
                assert decision["quantity"] > 0

    def test_decisions_include_signal_refs(self):
        """Decisions should cite signal_refs for attribution."""
        result = _timed_chat_json(
            f"Here is the current market context. Analyze and provide trading decisions.\n\n{self.SAMPLE_CONTEXT}",
            system=TRADING_SYSTEM_PROMPT,
        )

        for decision in result["decisions"]:
            if decision["action"] != "hold":
                assert "signal_refs" in decision
                assert isinstance(decision["signal_refs"], list)

    def test_no_playbook_returns_conservative(self):
        """Without a playbook, model should return holds or empty decisions."""
        conservative_context = """Today's Playbook:
  No playbook available — operate in conservative mode.

Portfolio:
  Cash: $50,000
  Buying power: $100,000
  Positions:
    AAPL: 20 shares @ $175 avg

Macro Context: No signals
Overnight Ticker Signals: None
Signal Attribution: No data
Recent Decisions: None"""

        result = _timed_chat_json(
            f"Here is the current market context. Analyze and provide trading decisions.\n\n{conservative_context}",
            system=TRADING_SYSTEM_PROMPT,
        )

        assert "decisions" in result
        # Should not propose new buys without a playbook
        buy_decisions = [d for d in result["decisions"] if d["action"] == "buy"]
        assert len(buy_decisions) == 0


# ---------------------------------------------------------------------------
# Ideation
# ---------------------------------------------------------------------------

@integration
@skip_no_ollama
class TestIdeationPrompt:
    """Verify qwen3:14b produces valid ideation JSON."""

    SAMPLE_CONTEXT = """Portfolio:
  Cash: $50,000
  Portfolio value: $150,000
  Positions: AAPL (20 shares), MSFT (15 shares)

Active Theses:
  ID 1: NVDA (long) — high confidence
    Thesis: AI datacenter demand acceleration
    Entry trigger: Price below $860
    Exit trigger: $1100 or below $750
    Invalidation: Datacenter revenue growth < 50% YoY
    Age: 5 days

Macro Context (last 7 days):
  Fed holds rates steady
  China trade tensions escalating

Market Snapshot:
  Tech sector: +1.2%
  Energy sector: -0.8%
  Top movers: NVDA +4.5%, TSLA -2.1%"""

    def test_returns_valid_structure(self):
        """Ideation response has reviews, new_theses, market_observations."""
        prompt = f"""Here is the current market context. Review existing theses and generate new trade ideas.

Exclude these tickers (already in portfolio): AAPL, MSFT
Exclude these tickers (already have active thesis): NVDA

{self.SAMPLE_CONTEXT}"""

        result = _timed_chat_json(prompt, system=IDEATION_SYSTEM_PROMPT)

        assert "reviews" in result
        assert isinstance(result["reviews"], list)
        assert "new_theses" in result
        assert isinstance(result["new_theses"], list)
        assert "market_observations" in result
        assert isinstance(result["market_observations"], str)

    def test_reviews_have_required_fields(self):
        """Each thesis review must have thesis_id, action, reason."""
        prompt = f"""Here is the current market context. Review existing theses and generate new trade ideas.

Exclude these tickers (already in portfolio): AAPL, MSFT
Exclude these tickers (already have active thesis): NVDA

{self.SAMPLE_CONTEXT}"""

        result = _timed_chat_json(prompt, system=IDEATION_SYSTEM_PROMPT)

        for review in result["reviews"]:
            assert "thesis_id" in review
            assert "action" in review
            assert review["action"] in VALID_THESIS_ACTIONS
            assert "reason" in review
            assert isinstance(review["reason"], str)

    def test_new_theses_have_required_fields(self):
        """Each new thesis must have ticker, direction, thesis, triggers."""
        prompt = f"""Here is the current market context. Review existing theses and generate new trade ideas.

Exclude these tickers (already in portfolio): AAPL, MSFT
Exclude these tickers (already have active thesis): NVDA

{self.SAMPLE_CONTEXT}"""

        result = _timed_chat_json(prompt, system=IDEATION_SYSTEM_PROMPT)

        for thesis in result["new_theses"]:
            assert "ticker" in thesis
            assert isinstance(thesis["ticker"], str)
            assert "direction" in thesis
            assert thesis["direction"] in VALID_DIRECTIONS
            assert "thesis" in thesis
            assert isinstance(thesis["thesis"], str)
            assert "entry_trigger" in thesis
            assert "exit_trigger" in thesis
            assert "invalidation" in thesis
            assert "confidence" in thesis
            assert thesis["confidence"] in VALID_CONFIDENCES

    def test_excludes_portfolio_tickers(self):
        """New theses should not include tickers already in portfolio."""
        prompt = f"""Here is the current market context. Review existing theses and generate new trade ideas.

Exclude these tickers (already in portfolio): AAPL, MSFT
Exclude these tickers (already have active thesis): NVDA

{self.SAMPLE_CONTEXT}"""

        result = _timed_chat_json(prompt, system=IDEATION_SYSTEM_PROMPT)

        excluded = {"AAPL", "MSFT", "NVDA"}
        for thesis in result["new_theses"]:
            assert thesis["ticker"] not in excluded, \
                f"Model suggested excluded ticker {thesis['ticker']}"
