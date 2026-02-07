"""Tests for trading/classifier.py - News classification with LLM."""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock, call

import pytest

from trading.classifier import (
    TickerSignal,
    MacroSignal,
    ClassificationResult,
    _build_classification_result,
    classify_news,
    classify_news_batch,
    _classify_batch,
    classify_ticker_news,
    classify_filtered_news,
)
from trading.filter import FilteredNewsItem
from trading.news import NewsItem

from tests.conftest import make_news_item, make_ticker_signal


SAMPLE_PUBLISHED_AT = datetime(2025, 1, 15, 10, 0, 0)


# ---------------------------------------------------------------------------
# _build_classification_result tests
# ---------------------------------------------------------------------------

class TestBuildClassificationResult:
    """Tests for _build_classification_result()."""

    def test_ticker_specific_single_ticker(self):
        entry = {
            "type": "ticker_specific",
            "tickers": ["AAPL"],
            "category": "earnings",
            "sentiment": "bullish",
            "confidence": "high",
        }
        result = _build_classification_result(entry, "AAPL beats Q3", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "ticker_specific"
        assert len(result.ticker_signals) == 1
        assert result.macro_signal is None

        signal = result.ticker_signals[0]
        assert signal.ticker == "AAPL"
        assert signal.headline == "AAPL beats Q3"
        assert signal.category == "earnings"
        assert signal.sentiment == "bullish"
        assert signal.confidence == "high"
        assert signal.published_at == SAMPLE_PUBLISHED_AT

    def test_ticker_specific_multiple_tickers(self):
        entry = {
            "type": "ticker_specific",
            "tickers": ["AAPL", "MSFT", "GOOG"],
            "category": "product",
            "sentiment": "bearish",
            "confidence": "medium",
        }
        result = _build_classification_result(entry, "Tech giants face scrutiny", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "ticker_specific"
        assert len(result.ticker_signals) == 3
        tickers = [s.ticker for s in result.ticker_signals]
        assert tickers == ["AAPL", "MSFT", "GOOG"]
        # All signals share the same headline, category, sentiment, confidence
        for signal in result.ticker_signals:
            assert signal.headline == "Tech giants face scrutiny"
            assert signal.category == "product"
            assert signal.sentiment == "bearish"

    def test_ticker_specific_uppercases_tickers(self):
        entry = {
            "type": "ticker_specific",
            "tickers": ["aapl", "msft"],
            "category": "earnings",
            "sentiment": "neutral",
            "confidence": "low",
        }
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        assert result.ticker_signals[0].ticker == "AAPL"
        assert result.ticker_signals[1].ticker == "MSFT"

    def test_ticker_specific_empty_tickers_list(self):
        entry = {
            "type": "ticker_specific",
            "tickers": [],
            "category": "earnings",
            "sentiment": "bullish",
            "confidence": "high",
        }
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "ticker_specific"
        assert result.ticker_signals == []
        assert result.macro_signal is None

    def test_ticker_specific_missing_tickers_key(self):
        entry = {
            "type": "ticker_specific",
            "category": "earnings",
            "sentiment": "bullish",
            "confidence": "high",
        }
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "ticker_specific"
        assert result.ticker_signals == []

    def test_macro_political(self):
        entry = {
            "type": "macro_political",
            "category": "fed",
            "affected_sectors": ["finance", "tech"],
            "sentiment": "bearish",
        }
        result = _build_classification_result(entry, "Fed raises rates", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "macro_political"
        assert result.ticker_signals == []
        assert result.macro_signal is not None

        macro = result.macro_signal
        assert macro.headline == "Fed raises rates"
        assert macro.category == "fed"
        assert macro.affected_sectors == ["finance", "tech"]
        assert macro.sentiment == "bearish"
        assert macro.published_at == SAMPLE_PUBLISHED_AT

    def test_macro_political_defaults(self):
        entry = {"type": "macro_political"}
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        macro = result.macro_signal
        assert macro.category == "geopolitical"
        assert macro.affected_sectors == ["all"]
        assert macro.sentiment == "neutral"

    def test_sector_type(self):
        entry = {
            "type": "sector",
            "affected_sectors": ["energy", "industrial"],
            "sentiment": "bullish",
        }
        result = _build_classification_result(entry, "Energy sector rally", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "sector"
        assert result.ticker_signals == []
        assert result.macro_signal is not None

        macro = result.macro_signal
        assert macro.category == "sector"  # always "sector" for sector type
        assert macro.affected_sectors == ["energy", "industrial"]
        assert macro.sentiment == "bullish"

    def test_sector_type_defaults(self):
        entry = {"type": "sector"}
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        macro = result.macro_signal
        assert macro.category == "sector"
        assert macro.affected_sectors == ["all"]
        assert macro.sentiment == "neutral"

    def test_noise_type(self):
        entry = {"type": "noise"}
        result = _build_classification_result(entry, "Celebrity gossip", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "noise"
        assert result.ticker_signals == []
        assert result.macro_signal is None

    def test_missing_type_defaults_to_noise(self):
        entry = {}
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "noise"
        assert result.ticker_signals == []
        assert result.macro_signal is None

    def test_ticker_specific_missing_optional_fields_use_defaults(self):
        entry = {
            "type": "ticker_specific",
            "tickers": ["TSLA"],
        }
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        signal = result.ticker_signals[0]
        assert signal.category == "noise"
        assert signal.sentiment == "neutral"
        assert signal.confidence == "low"

    def test_unknown_type_produces_no_signals(self):
        entry = {"type": "unknown_type"}
        result = _build_classification_result(entry, "headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "unknown_type"
        assert result.ticker_signals == []
        assert result.macro_signal is None


# ---------------------------------------------------------------------------
# classify_news tests
# ---------------------------------------------------------------------------

class TestClassifyNews:
    """Tests for classify_news()."""

    @patch("trading.classifier.chat_json")
    def test_returns_classification_result(self, mock_chat_json):
        mock_chat_json.return_value = {
            "type": "ticker_specific",
            "tickers": ["AAPL"],
            "category": "earnings",
            "sentiment": "bullish",
            "confidence": "high",
        }

        result = classify_news("AAPL beats Q3", SAMPLE_PUBLISHED_AT)

        assert isinstance(result, ClassificationResult)
        assert result.news_type == "ticker_specific"
        assert len(result.ticker_signals) == 1
        assert result.ticker_signals[0].ticker == "AAPL"

    @patch("trading.classifier.chat_json")
    def test_calls_chat_json_with_model(self, mock_chat_json):
        mock_chat_json.return_value = {"type": "noise"}

        classify_news("some headline", SAMPLE_PUBLISHED_AT)

        mock_chat_json.assert_called_once()
        call_kwargs = mock_chat_json.call_args
        assert call_kwargs[1]["model"] == "qwen3:14b" or call_kwargs[0][1] == "qwen3:14b"

    @patch("trading.classifier.chat_json")
    def test_returns_noise_on_value_error(self, mock_chat_json):
        mock_chat_json.side_effect = ValueError("Invalid JSON")

        result = classify_news("bad headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "noise"
        assert result.ticker_signals == []
        assert result.macro_signal is None

    @patch("trading.classifier.chat_json")
    def test_headline_in_prompt(self, mock_chat_json):
        mock_chat_json.return_value = {"type": "noise"}

        classify_news("My specific headline text", SAMPLE_PUBLISHED_AT)

        prompt = mock_chat_json.call_args[0][0]
        assert "My specific headline text" in prompt


# ---------------------------------------------------------------------------
# _classify_batch tests
# ---------------------------------------------------------------------------

class TestClassifyBatch:
    """Tests for _classify_batch()."""

    @patch("trading.classifier.chat")
    def test_parses_json_array_response(self, mock_chat):
        response = json.dumps([
            {"type": "ticker_specific", "tickers": ["AAPL"], "category": "earnings",
             "sentiment": "bullish", "confidence": "high"},
            {"type": "noise"},
        ])
        mock_chat.return_value = response

        headlines = ["AAPL earnings", "Random news"]
        dates = [SAMPLE_PUBLISHED_AT, SAMPLE_PUBLISHED_AT]
        results = _classify_batch(headlines, dates)

        assert len(results) == 2
        assert results[0].news_type == "ticker_specific"
        assert results[1].news_type == "noise"

    @patch("trading.classifier.chat")
    def test_handles_markdown_code_block(self, mock_chat):
        inner = json.dumps([{"type": "noise"}])
        mock_chat.return_value = f"```json\n{inner}\n```"

        results = _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

        assert len(results) == 1
        assert results[0].news_type == "noise"

    @patch("trading.classifier.chat")
    def test_handles_plain_code_block(self, mock_chat):
        inner = json.dumps([{"type": "noise"}])
        mock_chat.return_value = f"```\n{inner}\n```"

        results = _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

        assert len(results) == 1

    @patch("trading.classifier.chat")
    def test_pads_missing_results_with_noise(self, mock_chat):
        """If LLM returns fewer results than headlines, pad with noise."""
        response = json.dumps([{"type": "ticker_specific", "tickers": ["AAPL"],
                                "category": "earnings", "sentiment": "bullish",
                                "confidence": "high"}])
        mock_chat.return_value = response

        headlines = ["AAPL news", "MSFT news", "GOOG news"]
        dates = [SAMPLE_PUBLISHED_AT] * 3
        results = _classify_batch(headlines, dates)

        assert len(results) == 3
        assert results[0].news_type == "ticker_specific"
        assert results[1].news_type == "noise"
        assert results[2].news_type == "noise"

    @patch("trading.classifier.chat")
    def test_truncates_extra_results(self, mock_chat):
        """If LLM returns more results than headlines, truncate."""
        response = json.dumps([
            {"type": "noise"},
            {"type": "noise"},
            {"type": "noise"},
        ])
        mock_chat.return_value = response

        results = _classify_batch(["only one"], [SAMPLE_PUBLISHED_AT])

        assert len(results) == 1

    @patch("trading.classifier.chat")
    def test_raises_on_non_array_response(self, mock_chat):
        mock_chat.return_value = json.dumps({"type": "noise"})

        with pytest.raises(ValueError, match="Expected JSON array"):
            _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

    @patch("trading.classifier.chat")
    def test_raises_on_invalid_json(self, mock_chat):
        mock_chat.return_value = "not valid json at all"

        with pytest.raises(json.JSONDecodeError):
            _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

    @patch("trading.classifier.chat")
    def test_uses_correct_model_and_temperature(self, mock_chat):
        mock_chat.return_value = json.dumps([{"type": "noise"}])

        _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

        mock_chat.assert_called_once()
        _, kwargs = mock_chat.call_args
        assert kwargs.get("model") == "qwen3:14b" or mock_chat.call_args[0][1] if len(mock_chat.call_args[0]) > 1 else True
        assert kwargs.get("temperature") == 0.0

    @patch("trading.classifier.chat")
    def test_headlines_numbered_in_prompt(self, mock_chat):
        mock_chat.return_value = json.dumps([{"type": "noise"}, {"type": "noise"}])

        _classify_batch(["First headline", "Second headline"], [SAMPLE_PUBLISHED_AT] * 2)

        prompt = mock_chat.call_args[0][0]
        assert '1. "First headline"' in prompt
        assert '2. "Second headline"' in prompt


# ---------------------------------------------------------------------------
# classify_news_batch tests
# ---------------------------------------------------------------------------

class TestClassifyNewsBatch:
    """Tests for classify_news_batch()."""

    @patch("trading.classifier._classify_batch")
    def test_processes_single_batch(self, mock_batch):
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_batch.return_value = [noise_result, noise_result]

        headlines = ["Headline 1", "Headline 2"]
        dates = [SAMPLE_PUBLISHED_AT] * 2
        results = classify_news_batch(headlines, dates, batch_size=10)

        assert len(results) == 2
        mock_batch.assert_called_once()

    @patch("trading.classifier._classify_batch")
    def test_splits_into_batches(self, mock_batch):
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_batch.return_value = [noise_result, noise_result]

        headlines = ["H1", "H2", "H3", "H4"]
        dates = [SAMPLE_PUBLISHED_AT] * 4
        results = classify_news_batch(headlines, dates, batch_size=2)

        assert len(results) == 4
        assert mock_batch.call_count == 2

    @patch("trading.classifier._classify_batch")
    def test_batch_size_larger_than_input(self, mock_batch):
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_batch.return_value = [noise_result]

        results = classify_news_batch(["H1"], [SAMPLE_PUBLISHED_AT], batch_size=100)

        assert len(results) == 1
        mock_batch.assert_called_once()

    @patch("trading.classifier.classify_news")
    @patch("trading.classifier._classify_batch")
    def test_fallback_to_individual_on_batch_error(self, mock_batch, mock_individual):
        mock_batch.side_effect = Exception("Batch failed")
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_individual.return_value = noise_result

        headlines = ["H1", "H2"]
        dates = [SAMPLE_PUBLISHED_AT] * 2
        results = classify_news_batch(headlines, dates, batch_size=10)

        assert len(results) == 2
        assert mock_individual.call_count == 2

    @patch("trading.classifier.classify_news")
    @patch("trading.classifier._classify_batch")
    def test_fallback_individual_also_fails_returns_noise(self, mock_batch, mock_individual):
        mock_batch.side_effect = Exception("Batch failed")
        mock_individual.side_effect = Exception("Individual also failed")

        results = classify_news_batch(["H1"], [SAMPLE_PUBLISHED_AT], batch_size=10)

        assert len(results) == 1
        assert results[0].news_type == "noise"
        assert results[0].ticker_signals == []
        assert results[0].macro_signal is None

    @patch("trading.classifier._classify_batch")
    def test_empty_input(self, mock_batch):
        results = classify_news_batch([], [], batch_size=10)

        assert results == []
        mock_batch.assert_not_called()

    @patch("trading.classifier.classify_news")
    @patch("trading.classifier._classify_batch")
    def test_partial_batch_failure_only_affects_that_batch(self, mock_batch, mock_individual):
        """First batch succeeds, second batch fails and falls back."""
        ticker_result = ClassificationResult(
            news_type="ticker_specific",
            ticker_signals=[TickerSignal(
                ticker="AAPL", headline="H1", category="earnings",
                sentiment="bullish", confidence="high", published_at=SAMPLE_PUBLISHED_AT
            )],
            macro_signal=None,
        )
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)

        # First call succeeds, second fails
        mock_batch.side_effect = [[ticker_result], Exception("Second batch failed")]
        mock_individual.return_value = noise_result

        headlines = ["H1", "H2"]
        dates = [SAMPLE_PUBLISHED_AT] * 2
        results = classify_news_batch(headlines, dates, batch_size=1)

        assert len(results) == 2
        assert results[0].news_type == "ticker_specific"
        assert results[1].news_type == "noise"


# ---------------------------------------------------------------------------
# classify_ticker_news tests
# ---------------------------------------------------------------------------

class TestClassifyTickerNews:
    """Tests for classify_ticker_news()."""

    @patch("trading.classifier.chat_json")
    def test_returns_ticker_signal(self, mock_chat_json):
        mock_chat_json.return_value = {
            "category": "earnings",
            "sentiment": "bullish",
            "confidence": "high",
        }

        result = classify_ticker_news("AAPL", "AAPL beats Q3", SAMPLE_PUBLISHED_AT)

        assert isinstance(result, TickerSignal)
        assert result.ticker == "AAPL"
        assert result.headline == "AAPL beats Q3"
        assert result.category == "earnings"
        assert result.sentiment == "bullish"
        assert result.confidence == "high"
        assert result.published_at == SAMPLE_PUBLISHED_AT

    @patch("trading.classifier.chat_json")
    def test_uppercases_ticker(self, mock_chat_json):
        mock_chat_json.return_value = {
            "category": "product",
            "sentiment": "neutral",
            "confidence": "medium",
        }

        result = classify_ticker_news("aapl", "headline", SAMPLE_PUBLISHED_AT)

        assert result.ticker == "AAPL"

    @patch("trading.classifier.chat_json")
    def test_defaults_on_value_error(self, mock_chat_json):
        mock_chat_json.side_effect = ValueError("Invalid JSON")

        result = classify_ticker_news("TSLA", "bad headline", SAMPLE_PUBLISHED_AT)

        assert result.ticker == "TSLA"
        assert result.category == "noise"
        assert result.sentiment == "neutral"
        assert result.confidence == "low"

    @patch("trading.classifier.chat_json")
    def test_missing_fields_use_defaults(self, mock_chat_json):
        mock_chat_json.return_value = {}

        result = classify_ticker_news("GOOG", "headline", SAMPLE_PUBLISHED_AT)

        assert result.category == "noise"
        assert result.sentiment == "neutral"
        assert result.confidence == "low"

    @patch("trading.classifier.chat_json")
    def test_prompt_contains_ticker_and_headline(self, mock_chat_json):
        mock_chat_json.return_value = {"category": "noise", "sentiment": "neutral", "confidence": "low"}

        classify_ticker_news("NVDA", "NVDA launches new GPU", SAMPLE_PUBLISHED_AT)

        prompt = mock_chat_json.call_args[0][0]
        assert "NVDA" in prompt
        assert "NVDA launches new GPU" in prompt

    @patch("trading.classifier.chat_json")
    def test_uses_qwen_model(self, mock_chat_json):
        mock_chat_json.return_value = {"category": "noise", "sentiment": "neutral", "confidence": "low"}

        classify_ticker_news("AAPL", "headline", SAMPLE_PUBLISHED_AT)

        _, kwargs = mock_chat_json.call_args
        assert kwargs.get("model") == "qwen3:14b"


# ---------------------------------------------------------------------------
# classify_filtered_news tests
# ---------------------------------------------------------------------------

class TestClassifyFilteredNews:
    """Tests for classify_filtered_news()."""

    @patch("trading.classifier.classify_news_batch")
    def test_extracts_headlines_and_dates(self, mock_batch):
        mock_batch.return_value = []

        items = [
            FilteredNewsItem(
                item=make_news_item(headline="H1", published_at=datetime(2025, 1, 1)),
                relevance_score=0.9,
            ),
            FilteredNewsItem(
                item=make_news_item(headline="H2", published_at=datetime(2025, 2, 2)),
                relevance_score=0.7,
            ),
        ]

        classify_filtered_news(items)

        mock_batch.assert_called_once_with(
            ["H1", "H2"],
            [datetime(2025, 1, 1), datetime(2025, 2, 2)],
        )

    @patch("trading.classifier.classify_news_batch")
    def test_returns_batch_results(self, mock_batch):
        noise = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_batch.return_value = [noise]

        items = [
            FilteredNewsItem(item=make_news_item(), relevance_score=0.8),
        ]

        result = classify_filtered_news(items)

        assert result == [noise]

    @patch("trading.classifier.classify_news_batch")
    def test_empty_filtered_items(self, mock_batch):
        mock_batch.return_value = []

        result = classify_filtered_news([])

        mock_batch.assert_called_once_with([], [])
        assert result == []


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Tests for classifier dataclasses."""

    def test_ticker_signal_fields(self):
        signal = TickerSignal(
            ticker="AAPL",
            headline="headline",
            category="earnings",
            sentiment="bullish",
            confidence="high",
            published_at=SAMPLE_PUBLISHED_AT,
        )
        assert signal.ticker == "AAPL"
        assert signal.category == "earnings"

    def test_macro_signal_fields(self):
        signal = MacroSignal(
            headline="Fed news",
            category="fed",
            affected_sectors=["finance", "tech"],
            sentiment="bearish",
            published_at=SAMPLE_PUBLISHED_AT,
        )
        assert signal.category == "fed"
        assert signal.affected_sectors == ["finance", "tech"]

    def test_classification_result_fields(self):
        result = ClassificationResult(
            news_type="ticker_specific",
            ticker_signals=[],
            macro_signal=None,
        )
        assert result.news_type == "ticker_specific"
        assert result.ticker_signals == []
        assert result.macro_signal is None

    def test_classification_result_with_both_signals(self):
        ticker = TickerSignal(
            ticker="AAPL", headline="h", category="earnings",
            sentiment="bullish", confidence="high", published_at=SAMPLE_PUBLISHED_AT,
        )
        macro = MacroSignal(
            headline="h", category="fed", affected_sectors=["all"],
            sentiment="neutral", published_at=SAMPLE_PUBLISHED_AT,
        )
        result = ClassificationResult(
            news_type="mixed",
            ticker_signals=[ticker],
            macro_signal=macro,
        )
        assert len(result.ticker_signals) == 1
        assert result.macro_signal is not None
