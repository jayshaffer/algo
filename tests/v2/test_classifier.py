"""Tests for v2/classifier.py - News classification with Claude Haiku."""

import json
import inspect
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from v2.classifier import (
    TickerSignal,
    MacroSignal,
    ClassificationResult,
    CLASSIFICATION_MODEL,
    _build_classification_result,
    _strip_code_fences,
    classify_news,
    classify_news_batch,
    _classify_batch,
    classify_ticker_news,
)


SAMPLE_PUBLISHED_AT = datetime(2025, 1, 15, 10, 0, 0)


def _make_mock_response(text: str) -> MagicMock:
    """Create a mock Claude API response with the given text content."""
    mock_response = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.text = text
    mock_response.content = [mock_content_block]
    return mock_response


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
        assert macro.category == "sector"
        assert macro.affected_sectors == ["energy", "industrial"]
        assert macro.sentiment == "bullish"

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

    @patch("v2.classifier.get_claude_client")
    def test_returns_classification_result(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(json.dumps({
            "type": "ticker_specific",
            "tickers": ["AAPL"],
            "category": "earnings",
            "sentiment": "bullish",
            "confidence": "high",
        }))

        result = classify_news("AAPL beats Q3", SAMPLE_PUBLISHED_AT)

        assert isinstance(result, ClassificationResult)
        assert result.news_type == "ticker_specific"
        assert len(result.ticker_signals) == 1
        assert result.ticker_signals[0].ticker == "AAPL"

    @patch("v2.classifier.get_claude_client")
    def test_calls_claude_with_correct_model(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps({"type": "noise"})
        )

        classify_news("some headline", SAMPLE_PUBLISHED_AT)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == CLASSIFICATION_MODEL
        assert call_kwargs["max_tokens"] == 256

    @patch("v2.classifier.get_claude_client")
    def test_returns_noise_on_exception(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        result = classify_news("bad headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "noise"
        assert result.ticker_signals == []
        assert result.macro_signal is None

    @patch("v2.classifier.get_claude_client")
    def test_returns_noise_on_invalid_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response("not valid json")

        result = classify_news("bad headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "noise"
        assert result.ticker_signals == []
        assert result.macro_signal is None

    @patch("v2.classifier.get_claude_client")
    def test_headline_in_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps({"type": "noise"})
        )

        classify_news("My specific headline text", SAMPLE_PUBLISHED_AT)

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert "My specific headline text" in messages[0]["content"]

    @patch("v2.classifier.get_claude_client")
    def test_strips_code_fences_from_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            '```json\n{"type": "noise"}\n```'
        )

        result = classify_news("headline", SAMPLE_PUBLISHED_AT)

        assert result.news_type == "noise"


# ---------------------------------------------------------------------------
# classify_news_batch tests
# ---------------------------------------------------------------------------

class TestClassifyNewsBatch:
    """Tests for classify_news_batch()."""

    @patch("v2.classifier._classify_batch")
    def test_processes_single_batch(self, mock_batch):
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_batch.return_value = [noise_result, noise_result]

        headlines = ["Headline 1", "Headline 2"]
        dates = [SAMPLE_PUBLISHED_AT] * 2
        results = classify_news_batch(headlines, dates, batch_size=50)

        assert len(results) == 2
        mock_batch.assert_called_once()

    @patch("v2.classifier._classify_batch")
    def test_splits_into_batches(self, mock_batch):
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_batch.return_value = [noise_result, noise_result]

        headlines = ["H1", "H2", "H3", "H4"]
        dates = [SAMPLE_PUBLISHED_AT] * 4
        results = classify_news_batch(headlines, dates, batch_size=2)

        assert len(results) == 4
        assert mock_batch.call_count == 2

    @patch("v2.classifier._classify_batch")
    def test_batch_size_larger_than_input(self, mock_batch):
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_batch.return_value = [noise_result]

        results = classify_news_batch(["H1"], [SAMPLE_PUBLISHED_AT], batch_size=100)

        assert len(results) == 1
        mock_batch.assert_called_once()

    @patch("v2.classifier.classify_news")
    @patch("v2.classifier._classify_batch")
    def test_fallback_to_individual_on_batch_error(self, mock_batch, mock_individual):
        mock_batch.side_effect = Exception("Batch failed")
        noise_result = ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)
        mock_individual.return_value = noise_result

        headlines = ["H1", "H2"]
        dates = [SAMPLE_PUBLISHED_AT] * 2
        results = classify_news_batch(headlines, dates, batch_size=50)

        assert len(results) == 2
        assert mock_individual.call_count == 2

    @patch("v2.classifier.classify_news")
    @patch("v2.classifier._classify_batch")
    def test_fallback_individual_also_fails_returns_noise(self, mock_batch, mock_individual):
        mock_batch.side_effect = Exception("Batch failed")
        mock_individual.side_effect = Exception("Individual also failed")

        results = classify_news_batch(["H1"], [SAMPLE_PUBLISHED_AT], batch_size=50)

        assert len(results) == 1
        assert results[0].news_type == "noise"
        assert results[0].ticker_signals == []
        assert results[0].macro_signal is None

    @patch("v2.classifier._classify_batch")
    def test_empty_input(self, mock_batch):
        results = classify_news_batch([], [], batch_size=50)

        assert results == []
        mock_batch.assert_not_called()

    @patch("v2.classifier.classify_news")
    @patch("v2.classifier._classify_batch")
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

    def test_default_batch_size_is_50(self):
        """Verify default batch_size parameter is 50."""
        sig = inspect.signature(classify_news_batch)
        assert sig.parameters["batch_size"].default == 50

    @patch("v2.classifier.get_claude_client")
    def test_batch_calls_haiku_model(self, mock_get_client):
        """Verify batch classification uses Claude Haiku."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps([{"type": "noise"}])
        )

        classify_news_batch(["H1"], [SAMPLE_PUBLISHED_AT], batch_size=50)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == CLASSIFICATION_MODEL


# ---------------------------------------------------------------------------
# _classify_batch tests
# ---------------------------------------------------------------------------

class TestClassifyBatch:
    """Tests for _classify_batch()."""

    @patch("v2.classifier.get_claude_client")
    def test_parses_json_array_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        response_text = json.dumps([
            {"type": "ticker_specific", "tickers": ["AAPL"], "category": "earnings",
             "sentiment": "bullish", "confidence": "high"},
            {"type": "noise"},
        ])
        mock_client.messages.create.return_value = _make_mock_response(response_text)

        headlines = ["AAPL earnings", "Random news"]
        dates = [SAMPLE_PUBLISHED_AT, SAMPLE_PUBLISHED_AT]
        results = _classify_batch(headlines, dates)

        assert len(results) == 2
        assert results[0].news_type == "ticker_specific"
        assert results[1].news_type == "noise"

    @patch("v2.classifier.get_claude_client")
    def test_handles_markdown_code_block(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        inner = json.dumps([{"type": "noise"}])
        mock_client.messages.create.return_value = _make_mock_response(
            f"```json\n{inner}\n```"
        )

        results = _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

        assert len(results) == 1
        assert results[0].news_type == "noise"

    @patch("v2.classifier.get_claude_client")
    def test_handles_plain_code_block(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        inner = json.dumps([{"type": "noise"}])
        mock_client.messages.create.return_value = _make_mock_response(
            f"```\n{inner}\n```"
        )

        results = _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

        assert len(results) == 1

    @patch("v2.classifier.get_claude_client")
    def test_pads_missing_results_with_noise(self, mock_get_client):
        """If LLM returns fewer results than headlines, pad with noise."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        response_text = json.dumps([{"type": "ticker_specific", "tickers": ["AAPL"],
                                     "category": "earnings", "sentiment": "bullish",
                                     "confidence": "high"}])
        mock_client.messages.create.return_value = _make_mock_response(response_text)

        headlines = ["AAPL news", "MSFT news", "GOOG news"]
        dates = [SAMPLE_PUBLISHED_AT] * 3
        results = _classify_batch(headlines, dates)

        assert len(results) == 3
        assert results[0].news_type == "ticker_specific"
        assert results[1].news_type == "noise"
        assert results[2].news_type == "noise"

    @patch("v2.classifier.get_claude_client")
    def test_truncates_extra_results(self, mock_get_client):
        """If LLM returns more results than headlines, truncate."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        response_text = json.dumps([
            {"type": "noise"},
            {"type": "noise"},
            {"type": "noise"},
        ])
        mock_client.messages.create.return_value = _make_mock_response(response_text)

        results = _classify_batch(["only one"], [SAMPLE_PUBLISHED_AT])

        assert len(results) == 1

    @patch("v2.classifier.get_claude_client")
    def test_raises_on_non_array_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps({"type": "noise"})
        )

        with pytest.raises(ValueError, match="Expected JSON array"):
            _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

    @patch("v2.classifier.get_claude_client")
    def test_raises_on_invalid_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            "not valid json at all"
        )

        with pytest.raises(json.JSONDecodeError):
            _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

    @patch("v2.classifier.get_claude_client")
    def test_uses_correct_model_and_max_tokens(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps([{"type": "noise"}])
        )

        _classify_batch(["headline"], [SAMPLE_PUBLISHED_AT])

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == CLASSIFICATION_MODEL
        assert call_kwargs["max_tokens"] == 4096

    @patch("v2.classifier.get_claude_client")
    def test_headlines_numbered_in_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps([{"type": "noise"}, {"type": "noise"}])
        )

        _classify_batch(["First headline", "Second headline"], [SAMPLE_PUBLISHED_AT] * 2)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert '1. "First headline"' in prompt
        assert '2. "Second headline"' in prompt


# ---------------------------------------------------------------------------
# classify_ticker_news tests
# ---------------------------------------------------------------------------

class TestClassifyTickerNews:
    """Tests for classify_ticker_news()."""

    @patch("v2.classifier.get_claude_client")
    def test_returns_ticker_signal(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(json.dumps({
            "category": "earnings",
            "sentiment": "bullish",
            "confidence": "high",
        }))

        result = classify_ticker_news("AAPL", "Apple beats Q3", SAMPLE_PUBLISHED_AT)

        assert isinstance(result, TickerSignal)
        assert result.ticker == "AAPL"
        assert result.headline == "Apple beats Q3"
        assert result.category == "earnings"
        assert result.sentiment == "bullish"
        assert result.confidence == "high"
        assert result.published_at == SAMPLE_PUBLISHED_AT

    @patch("v2.classifier.get_claude_client")
    def test_calls_claude_with_correct_model(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps({"category": "noise", "sentiment": "neutral", "confidence": "low"})
        )

        classify_ticker_news("AAPL", "headline", SAMPLE_PUBLISHED_AT)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == CLASSIFICATION_MODEL
        assert call_kwargs["max_tokens"] == 256

    @patch("v2.classifier.get_claude_client")
    def test_uppercases_ticker(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps({"category": "earnings", "sentiment": "bullish", "confidence": "high"})
        )

        result = classify_ticker_news("aapl", "headline", SAMPLE_PUBLISHED_AT)

        assert result.ticker == "AAPL"

    @patch("v2.classifier.get_claude_client")
    def test_returns_defaults_on_exception(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        result = classify_ticker_news("AAPL", "headline", SAMPLE_PUBLISHED_AT)

        assert result.ticker == "AAPL"
        assert result.category == "noise"
        assert result.sentiment == "neutral"
        assert result.confidence == "low"

    @patch("v2.classifier.get_claude_client")
    def test_returns_defaults_on_invalid_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response("not json")

        result = classify_ticker_news("AAPL", "headline", SAMPLE_PUBLISHED_AT)

        assert result.category == "noise"
        assert result.sentiment == "neutral"
        assert result.confidence == "low"

    @patch("v2.classifier.get_claude_client")
    def test_ticker_in_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps({"category": "noise", "sentiment": "neutral", "confidence": "low"})
        )

        classify_ticker_news("TSLA", "Tesla earnings", SAMPLE_PUBLISHED_AT)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "TSLA" in prompt
        assert "Tesla earnings" in prompt

    @patch("v2.classifier.get_claude_client")
    def test_strips_code_fences_from_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            '```json\n{"category": "earnings", "sentiment": "bullish", "confidence": "high"}\n```'
        )

        result = classify_ticker_news("AAPL", "headline", SAMPLE_PUBLISHED_AT)

        assert result.category == "earnings"
        assert result.sentiment == "bullish"
        assert result.confidence == "high"


# ---------------------------------------------------------------------------
# _strip_code_fences tests
# ---------------------------------------------------------------------------

class TestStripCodeFences:
    """Tests for _strip_code_fences()."""

    def test_strips_json_code_fence(self):
        text = '```json\n{"type": "noise"}\n```'
        assert _strip_code_fences(text) == '{"type": "noise"}'

    def test_strips_plain_code_fence(self):
        text = '```\n{"type": "noise"}\n```'
        assert _strip_code_fences(text) == '{"type": "noise"}'

    def test_no_code_fence(self):
        text = '{"type": "noise"}'
        assert _strip_code_fences(text) == '{"type": "noise"}'

    def test_strips_surrounding_whitespace(self):
        text = '  \n{"type": "noise"}\n  '
        assert _strip_code_fences(text) == '{"type": "noise"}'


# ---------------------------------------------------------------------------
# Constants and dataclass tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module-level constants."""

    def test_classification_model(self):
        assert CLASSIFICATION_MODEL == "claude-haiku-4-5-20251001"

    def test_no_classify_filtered_news(self):
        """Verify classify_filtered_news was removed in v2."""
        import v2.classifier as mod
        assert not hasattr(mod, "classify_filtered_news")

    def test_no_filtered_news_item_import(self):
        """Verify FilteredNewsItem is not referenced in v2."""
        import v2.classifier as mod
        assert not hasattr(mod, "FilteredNewsItem")


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
