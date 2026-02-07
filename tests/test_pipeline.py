"""Tests for trading/pipeline.py - News pipeline orchestrator."""

import logging
from datetime import datetime
from unittest.mock import patch, MagicMock, call

import pytest

from trading.pipeline import PipelineStats, run_pipeline, check_dependencies
from trading.classifier import ClassificationResult, TickerSignal, MacroSignal
from trading.filter import FilteredNewsItem, DEFAULT_STRATEGY_CONTEXT
from trading.news import NewsItem

from tests.conftest import make_news_item


SAMPLE_PUBLISHED_AT = datetime(2025, 1, 15, 10, 0, 0)


def _noise_result():
    return ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None)


def _ticker_result(ticker="AAPL", category="earnings", sentiment="bullish", confidence="high"):
    return ClassificationResult(
        news_type="ticker_specific",
        ticker_signals=[TickerSignal(
            ticker=ticker,
            headline=f"{ticker} news",
            category=category,
            sentiment=sentiment,
            confidence=confidence,
            published_at=SAMPLE_PUBLISHED_AT,
        )],
        macro_signal=None,
    )


def _macro_result(category="fed", sentiment="bearish", sectors=None):
    return ClassificationResult(
        news_type="macro_political",
        ticker_signals=[],
        macro_signal=MacroSignal(
            headline="Macro news",
            category=category,
            affected_sectors=sectors or ["finance", "tech"],
            sentiment=sentiment,
            published_at=SAMPLE_PUBLISHED_AT,
        ),
    )


# ---------------------------------------------------------------------------
# PipelineStats dataclass tests
# ---------------------------------------------------------------------------

class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_default_values(self):
        stats = PipelineStats(
            news_fetched=0,
            news_filtered=0,
            ticker_signals_stored=0,
            macro_signals_stored=0,
            noise_dropped=0,
            errors=0,
        )
        assert stats.news_fetched == 0
        assert stats.errors == 0

    def test_populated_values(self):
        stats = PipelineStats(
            news_fetched=100,
            news_filtered=50,
            ticker_signals_stored=20,
            macro_signals_stored=10,
            noise_dropped=20,
            errors=1,
        )
        assert stats.news_fetched == 100
        assert stats.news_filtered == 50
        assert stats.ticker_signals_stored == 20
        assert stats.macro_signals_stored == 10
        assert stats.noise_dropped == 20
        assert stats.errors == 1


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """Tests for run_pipeline()."""

    @patch("trading.pipeline.insert_macro_signals_batch", return_value=1)
    @patch("trading.pipeline.insert_news_signals_batch", return_value=2)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_full_pipeline_flow(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        """End-to-end pipeline with ticker and macro signals."""
        news_items = [make_news_item(id="1"), make_news_item(id="2")]
        mock_fetch.return_value = news_items

        filtered = [
            FilteredNewsItem(item=news_items[0], relevance_score=0.9),
            FilteredNewsItem(item=news_items[1], relevance_score=0.7),
        ]
        mock_filter.return_value = filtered

        mock_classify.return_value = [
            _ticker_result("AAPL"),
            _macro_result("fed"),
        ]

        stats = run_pipeline(hours=24, limit=100, relevance_threshold=0.3)

        assert stats.news_fetched == 2
        assert stats.news_filtered == 2
        assert stats.ticker_signals_stored == 2
        assert stats.macro_signals_stored == 1
        assert stats.noise_dropped == 0
        assert stats.errors == 0

        mock_fetch.assert_called_once_with(hours=24, limit=100)
        mock_filter.assert_called_once()
        mock_classify.assert_called_once()
        mock_insert_news.assert_called_once()
        mock_insert_macro.assert_called_once()

    @patch("trading.pipeline.insert_macro_signals_batch")
    @patch("trading.pipeline.insert_news_signals_batch")
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_dry_run_skips_storage(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        news_items = [make_news_item()]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [
            FilteredNewsItem(item=news_items[0], relevance_score=0.9),
        ]
        mock_classify.return_value = [_ticker_result("AAPL")]

        stats = run_pipeline(dry_run=True)

        mock_insert_news.assert_not_called()
        mock_insert_macro.assert_not_called()
        assert stats.ticker_signals_stored == 0
        assert stats.macro_signals_stored == 0

    @patch("trading.pipeline.fetch_broad_news")
    def test_empty_news_returns_early(self, mock_fetch):
        mock_fetch.return_value = []

        stats = run_pipeline()

        assert stats.news_fetched == 0
        assert stats.news_filtered == 0
        assert stats.errors == 0

    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_empty_filter_returns_early(self, mock_fetch, mock_filter):
        mock_fetch.return_value = [make_news_item()]
        mock_filter.return_value = []

        stats = run_pipeline()

        assert stats.news_fetched == 1
        assert stats.news_filtered == 0
        assert stats.errors == 0

    @patch("trading.pipeline.insert_macro_signals_batch", return_value=0)
    @patch("trading.pipeline.insert_news_signals_batch", return_value=0)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_all_noise_classification(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        news_items = [make_news_item(id="1"), make_news_item(id="2")]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [
            FilteredNewsItem(item=news_items[0], relevance_score=0.5),
            FilteredNewsItem(item=news_items[1], relevance_score=0.4),
        ]
        mock_classify.return_value = [_noise_result(), _noise_result()]

        stats = run_pipeline()

        assert stats.noise_dropped == 2
        assert stats.ticker_signals_stored == 0
        assert stats.macro_signals_stored == 0

    @patch("trading.pipeline.fetch_broad_news")
    def test_fetch_error_increments_errors(self, mock_fetch):
        mock_fetch.side_effect = Exception("API down")

        stats = run_pipeline()

        assert stats.errors == 1
        assert stats.news_fetched == 0

    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_filter_error_increments_errors(self, mock_fetch, mock_filter):
        mock_fetch.return_value = [make_news_item()]
        mock_filter.side_effect = Exception("Ollama down")

        stats = run_pipeline()

        assert stats.errors == 1
        assert stats.news_fetched == 1
        assert stats.news_filtered == 0

    @patch("trading.pipeline.insert_news_signals_batch")
    @patch("trading.pipeline.insert_macro_signals_batch", return_value=0)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_insert_news_error_increments_errors(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_macro, mock_insert_news,
    ):
        news_items = [make_news_item()]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [FilteredNewsItem(item=news_items[0], relevance_score=0.9)]
        mock_classify.return_value = [_ticker_result("AAPL")]
        mock_insert_news.side_effect = Exception("DB error")

        stats = run_pipeline()

        assert stats.errors == 1

    @patch("trading.pipeline.insert_macro_signals_batch")
    @patch("trading.pipeline.insert_news_signals_batch", return_value=0)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_insert_macro_error_increments_errors(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        news_items = [make_news_item()]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [FilteredNewsItem(item=news_items[0], relevance_score=0.9)]
        mock_classify.return_value = [_macro_result()]
        mock_insert_macro.side_effect = Exception("DB error")

        stats = run_pipeline()

        assert stats.errors == 1

    @patch("trading.pipeline.insert_macro_signals_batch", return_value=0)
    @patch("trading.pipeline.insert_news_signals_batch", return_value=3)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_multiple_ticker_signals_from_one_result(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        """A single ticker_specific result with multiple tickers creates multiple signals."""
        news_items = [make_news_item()]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [FilteredNewsItem(item=news_items[0], relevance_score=0.9)]

        multi_ticker = ClassificationResult(
            news_type="ticker_specific",
            ticker_signals=[
                TickerSignal(ticker="AAPL", headline="h", category="earnings",
                             sentiment="bullish", confidence="high",
                             published_at=SAMPLE_PUBLISHED_AT),
                TickerSignal(ticker="MSFT", headline="h", category="earnings",
                             sentiment="bullish", confidence="high",
                             published_at=SAMPLE_PUBLISHED_AT),
                TickerSignal(ticker="GOOG", headline="h", category="earnings",
                             sentiment="bullish", confidence="high",
                             published_at=SAMPLE_PUBLISHED_AT),
            ],
            macro_signal=None,
        )
        mock_classify.return_value = [multi_ticker]

        stats = run_pipeline()

        # Should have 3 tuples in the batch insert
        insert_call = mock_insert_news.call_args[0][0]
        assert len(insert_call) == 3
        assert stats.ticker_signals_stored == 3

    @patch("trading.pipeline.insert_macro_signals_batch", return_value=1)
    @patch("trading.pipeline.insert_news_signals_batch", return_value=1)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_mixed_signal_types(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        """Pipeline processes a mix of ticker, macro, and noise results."""
        news_items = [make_news_item(id=str(i)) for i in range(3)]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [
            FilteredNewsItem(item=n, relevance_score=0.8) for n in news_items
        ]
        mock_classify.return_value = [
            _ticker_result("AAPL"),
            _macro_result("fed"),
            _noise_result(),
        ]

        stats = run_pipeline()

        assert stats.noise_dropped == 1
        assert stats.ticker_signals_stored == 1
        assert stats.macro_signals_stored == 1

    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_custom_strategy_context_passed_to_filter(self, mock_fetch, mock_filter):
        mock_fetch.return_value = [make_news_item()]
        mock_filter.return_value = []

        custom_ctx = "My custom strategy"
        run_pipeline(strategy_context=custom_ctx)

        _, kwargs = mock_filter.call_args
        assert kwargs["strategy_context"] == custom_ctx

    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_custom_threshold_passed_to_filter(self, mock_fetch, mock_filter):
        mock_fetch.return_value = [make_news_item()]
        mock_filter.return_value = []

        run_pipeline(relevance_threshold=0.7)

        _, kwargs = mock_filter.call_args
        assert kwargs["threshold"] == 0.7

    @patch("trading.pipeline.fetch_broad_news")
    def test_default_parameters(self, mock_fetch):
        mock_fetch.return_value = []

        run_pipeline()

        mock_fetch.assert_called_once_with(hours=24, limit=300)

    @patch("trading.pipeline.fetch_broad_news")
    def test_custom_hours_and_limit(self, mock_fetch):
        mock_fetch.return_value = []

        run_pipeline(hours=48, limit=500)

        mock_fetch.assert_called_once_with(hours=48, limit=500)

    @patch("trading.pipeline.insert_macro_signals_batch")
    @patch("trading.pipeline.insert_news_signals_batch")
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_dry_run_logs_signals(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro, caplog,
    ):
        news_items = [make_news_item(), make_news_item(id="2")]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [
            FilteredNewsItem(item=news_items[0], relevance_score=0.9),
            FilteredNewsItem(item=news_items[1], relevance_score=0.8),
        ]
        mock_classify.return_value = [
            _ticker_result("AAPL"),
            _macro_result("fed"),
        ]

        with caplog.at_level(logging.INFO, logger="pipeline"):
            run_pipeline(dry_run=True)

        assert "[DRY RUN]" in caplog.text
        assert "Ticker signal" in caplog.text or "Macro signal" in caplog.text

    @patch("trading.pipeline.insert_macro_signals_batch")
    @patch("trading.pipeline.insert_news_signals_batch")
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_both_insert_errors_counted_separately(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        news_items = [make_news_item(), make_news_item(id="2")]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [
            FilteredNewsItem(item=n, relevance_score=0.9) for n in news_items
        ]
        mock_classify.return_value = [
            _ticker_result("AAPL"),
            _macro_result("fed"),
        ]
        mock_insert_news.side_effect = Exception("news DB error")
        mock_insert_macro.side_effect = Exception("macro DB error")

        stats = run_pipeline()

        assert stats.errors == 2

    @patch("trading.pipeline.insert_macro_signals_batch", return_value=0)
    @patch("trading.pipeline.insert_news_signals_batch", return_value=0)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_ticker_signal_tuple_format(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        """Verify the tuple structure passed to insert_news_signals_batch."""
        news_items = [make_news_item()]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [FilteredNewsItem(item=news_items[0], relevance_score=0.9)]
        mock_classify.return_value = [_ticker_result("AAPL", "earnings", "bullish", "high")]

        run_pipeline()

        batch = mock_insert_news.call_args[0][0]
        assert len(batch) == 1
        ticker, headline, category, sentiment, confidence, pub_at = batch[0]
        assert ticker == "AAPL"
        assert category == "earnings"
        assert sentiment == "bullish"
        assert confidence == "high"
        assert pub_at == SAMPLE_PUBLISHED_AT

    @patch("trading.pipeline.insert_macro_signals_batch", return_value=0)
    @patch("trading.pipeline.insert_news_signals_batch", return_value=0)
    @patch("trading.pipeline.classify_news_batch")
    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_macro_signal_tuple_format(
        self, mock_fetch, mock_filter, mock_classify,
        mock_insert_news, mock_insert_macro,
    ):
        """Verify the tuple structure passed to insert_macro_signals_batch."""
        news_items = [make_news_item()]
        mock_fetch.return_value = news_items
        mock_filter.return_value = [FilteredNewsItem(item=news_items[0], relevance_score=0.9)]
        mock_classify.return_value = [_macro_result("fed", "bearish", ["finance", "tech"])]

        run_pipeline()

        batch = mock_insert_macro.call_args[0][0]
        assert len(batch) == 1
        headline, category, sectors, sentiment, pub_at = batch[0]
        assert headline == "Macro news"
        assert category == "fed"
        assert sectors == ["finance", "tech"]
        assert sentiment == "bearish"
        assert pub_at == SAMPLE_PUBLISHED_AT

    @patch("trading.pipeline.fetch_broad_news")
    def test_fetch_error_returns_stats_immediately(self, mock_fetch):
        """Pipeline should return stats object even on early errors."""
        mock_fetch.side_effect = RuntimeError("Connection refused")

        stats = run_pipeline()

        assert isinstance(stats, PipelineStats)
        assert stats.errors == 1
        assert stats.news_fetched == 0
        assert stats.news_filtered == 0

    @patch("trading.pipeline.filter_by_relevance")
    @patch("trading.pipeline.fetch_broad_news")
    def test_filter_error_returns_stats_immediately(self, mock_fetch, mock_filter):
        mock_fetch.return_value = [make_news_item()]
        mock_filter.side_effect = RuntimeError("Embedding service down")

        stats = run_pipeline()

        assert isinstance(stats, PipelineStats)
        assert stats.errors == 1
        assert stats.news_fetched == 1


# ---------------------------------------------------------------------------
# check_dependencies tests
# ---------------------------------------------------------------------------

class TestCheckDependencies:
    """Tests for check_dependencies()."""

    @patch("trading.pipeline.list_models")
    @patch("trading.pipeline.check_ollama_health")
    def test_returns_true_when_healthy(self, mock_health, mock_models):
        mock_health.return_value = True
        mock_models.return_value = ["qwen2.5:14b", "nomic-embed-text"]

        result = check_dependencies()

        assert result is True
        mock_health.assert_called_once()
        mock_models.assert_called_once()

    @patch("trading.pipeline.check_ollama_health")
    def test_returns_false_when_ollama_not_available(self, mock_health):
        mock_health.return_value = False

        result = check_dependencies()

        assert result is False

    @patch("trading.pipeline.list_models")
    @patch("trading.pipeline.check_ollama_health")
    def test_warns_about_missing_models(self, mock_health, mock_models, caplog):
        mock_health.return_value = True
        mock_models.return_value = ["some-other-model:latest"]

        with caplog.at_level(logging.WARNING, logger="pipeline"):
            result = check_dependencies()

        assert result is True  # still returns True, just warns
        assert "qwen2.5:14b" in caplog.text
        assert "nomic-embed-text" in caplog.text

    @patch("trading.pipeline.list_models")
    @patch("trading.pipeline.check_ollama_health")
    def test_no_warning_when_all_models_present(self, mock_health, mock_models, caplog):
        mock_health.return_value = True
        mock_models.return_value = ["qwen2.5:14b", "nomic-embed-text"]

        with caplog.at_level(logging.WARNING, logger="pipeline"):
            check_dependencies()

        assert "not found" not in caplog.text

    @patch("trading.pipeline.list_models")
    @patch("trading.pipeline.check_ollama_health")
    def test_partial_model_match(self, mock_health, mock_models, caplog):
        """Only one required model is available."""
        mock_health.return_value = True
        mock_models.return_value = ["qwen2.5:14b"]

        with caplog.at_level(logging.WARNING, logger="pipeline"):
            check_dependencies()

        assert "nomic-embed-text" in caplog.text

    @patch("trading.pipeline.list_models")
    @patch("trading.pipeline.check_ollama_health")
    def test_empty_models_list(self, mock_health, mock_models, caplog):
        mock_health.return_value = True
        mock_models.return_value = []

        with caplog.at_level(logging.INFO, logger="pipeline"):
            result = check_dependencies()

        assert result is True
        assert "0 models" in caplog.text

    @patch("trading.pipeline.list_models")
    @patch("trading.pipeline.check_ollama_health")
    def test_model_substring_matching(self, mock_health, mock_models):
        """Model names are matched with 'in' operator, so substrings work."""
        mock_health.return_value = True
        # The check uses: any(model in m for m in models)
        mock_models.return_value = ["qwen2.5:14b-q4_0", "nomic-embed-text:latest"]

        result = check_dependencies()

        assert result is True
