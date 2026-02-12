"""Tests for v2/pipeline.py — news pipeline orchestrator."""

from unittest.mock import patch, MagicMock
from datetime import datetime

from v2.pipeline import run_pipeline, PipelineStats
from v2.classifier import ClassificationResult, TickerSignal, MacroSignal


SAMPLE_PUBLISHED_AT = datetime(2025, 6, 15, 10, 0, 0)


def _make_news_item(headline="Test headline"):
    """Create a mock news item with required attributes."""
    item = MagicMock()
    item.headline = headline
    item.published_at = SAMPLE_PUBLISHED_AT
    return item


class TestRunPipeline:
    """Tests for run_pipeline()."""

    @patch("v2.pipeline.insert_macro_signals_batch", return_value=1)
    @patch("v2.pipeline.insert_news_signals_batch", return_value=2)
    @patch("v2.pipeline.classify_news_batch")
    @patch("v2.pipeline.fetch_broad_news")
    def test_pipeline_fetch_classify_store(self, mock_fetch, mock_classify,
                                           mock_insert_ticker, mock_insert_macro):
        """Full pipeline: fetch -> classify -> store, verify stats."""
        mock_fetch.return_value = [_make_news_item("AAPL beats"), _make_news_item("Fed holds")]

        ticker_signal = TickerSignal(
            ticker="AAPL", headline="AAPL beats", category="earnings",
            sentiment="bullish", confidence="high", published_at=SAMPLE_PUBLISHED_AT
        )
        macro_signal = MacroSignal(
            headline="Fed holds", category="fed",
            affected_sectors=["finance"], sentiment="neutral",
            published_at=SAMPLE_PUBLISHED_AT
        )
        mock_classify.return_value = [
            ClassificationResult(news_type="ticker_specific",
                                 ticker_signals=[ticker_signal], macro_signal=None),
            ClassificationResult(news_type="macro_political",
                                 ticker_signals=[], macro_signal=macro_signal),
        ]

        stats = run_pipeline(hours=24, limit=50)

        assert stats.news_fetched == 2
        assert stats.ticker_signals_stored == 2
        assert stats.macro_signals_stored == 1
        assert stats.noise_dropped == 0
        assert stats.errors == 0
        mock_fetch.assert_called_once_with(hours=24, limit=50)
        mock_classify.assert_called_once()

    @patch("v2.pipeline.fetch_broad_news")
    def test_empty_news(self, mock_fetch):
        """When fetch returns empty list, stats.news_fetched == 0."""
        mock_fetch.return_value = []

        stats = run_pipeline()

        assert stats.news_fetched == 0
        assert stats.ticker_signals_stored == 0
        assert stats.macro_signals_stored == 0
        assert stats.noise_dropped == 0
        assert stats.errors == 0

    @patch("v2.pipeline.insert_macro_signals_batch", return_value=0)
    @patch("v2.pipeline.insert_news_signals_batch", return_value=1)
    @patch("v2.pipeline.classify_news_batch")
    @patch("v2.pipeline.fetch_broad_news")
    def test_stores_ticker_signals(self, mock_fetch, mock_classify,
                                    mock_insert_ticker, mock_insert_macro):
        """When classification returns ticker_specific, insert_news_signals_batch is called."""
        mock_fetch.return_value = [_make_news_item("AAPL earnings beat")]

        ticker_signal = TickerSignal(
            ticker="AAPL", headline="AAPL earnings beat", category="earnings",
            sentiment="bullish", confidence="high", published_at=SAMPLE_PUBLISHED_AT
        )
        mock_classify.return_value = [
            ClassificationResult(news_type="ticker_specific",
                                 ticker_signals=[ticker_signal], macro_signal=None),
        ]

        stats = run_pipeline()

        mock_insert_ticker.assert_called_once()
        args = mock_insert_ticker.call_args[0][0]
        assert len(args) == 1
        assert args[0][0] == "AAPL"
        assert stats.ticker_signals_stored == 1

    @patch("v2.pipeline.insert_macro_signals_batch", return_value=1)
    @patch("v2.pipeline.insert_news_signals_batch", return_value=0)
    @patch("v2.pipeline.classify_news_batch")
    @patch("v2.pipeline.fetch_broad_news")
    def test_stores_macro_signals(self, mock_fetch, mock_classify,
                                   mock_insert_ticker, mock_insert_macro):
        """When classification returns macro_political, insert_macro_signals_batch is called."""
        mock_fetch.return_value = [_make_news_item("Fed raises rates")]

        macro_signal = MacroSignal(
            headline="Fed raises rates", category="fed",
            affected_sectors=["finance", "tech"], sentiment="bearish",
            published_at=SAMPLE_PUBLISHED_AT
        )
        mock_classify.return_value = [
            ClassificationResult(news_type="macro_political",
                                 ticker_signals=[], macro_signal=macro_signal),
        ]

        stats = run_pipeline()

        mock_insert_macro.assert_called_once()
        args = mock_insert_macro.call_args[0][0]
        assert len(args) == 1
        assert args[0][0] == "Fed raises rates"
        assert args[0][1] == "fed"
        assert stats.macro_signals_stored == 1

    @patch("v2.pipeline.insert_macro_signals_batch")
    @patch("v2.pipeline.insert_news_signals_batch")
    @patch("v2.pipeline.classify_news_batch")
    @patch("v2.pipeline.fetch_broad_news")
    def test_dry_run_skips_storage(self, mock_fetch, mock_classify,
                                    mock_insert_ticker, mock_insert_macro):
        """When dry_run=True, insert functions are not called."""
        mock_fetch.return_value = [_make_news_item("AAPL earnings")]

        ticker_signal = TickerSignal(
            ticker="AAPL", headline="AAPL earnings", category="earnings",
            sentiment="bullish", confidence="high", published_at=SAMPLE_PUBLISHED_AT
        )
        mock_classify.return_value = [
            ClassificationResult(news_type="ticker_specific",
                                 ticker_signals=[ticker_signal], macro_signal=None),
        ]

        stats = run_pipeline(dry_run=True)

        mock_insert_ticker.assert_not_called()
        mock_insert_macro.assert_not_called()
        assert stats.ticker_signals_stored == 0
        assert stats.macro_signals_stored == 0

    @patch("v2.pipeline.classify_news_batch")
    @patch("v2.pipeline.fetch_broad_news")
    def test_counts_noise(self, mock_fetch, mock_classify):
        """Noise results increment noise_dropped."""
        mock_fetch.return_value = [
            _make_news_item("Celebrity gossip"),
            _make_news_item("Sports scores"),
            _make_news_item("AAPL earnings"),
        ]

        ticker_signal = TickerSignal(
            ticker="AAPL", headline="AAPL earnings", category="earnings",
            sentiment="bullish", confidence="high", published_at=SAMPLE_PUBLISHED_AT
        )
        mock_classify.return_value = [
            ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None),
            ClassificationResult(news_type="noise", ticker_signals=[], macro_signal=None),
            ClassificationResult(news_type="ticker_specific",
                                 ticker_signals=[ticker_signal], macro_signal=None),
        ]

        with patch("v2.pipeline.insert_news_signals_batch", return_value=1), \
             patch("v2.pipeline.insert_macro_signals_batch", return_value=0):
            stats = run_pipeline()

        assert stats.noise_dropped == 2
        assert stats.news_fetched == 3

    @patch("v2.pipeline.insert_macro_signals_batch", return_value=0)
    @patch("v2.pipeline.insert_news_signals_batch", side_effect=Exception("DB error"))
    @patch("v2.pipeline.classify_news_batch")
    @patch("v2.pipeline.fetch_broad_news")
    def test_error_handling(self, mock_fetch, mock_classify,
                            mock_insert_ticker, mock_insert_macro):
        """When insert_news_signals_batch raises, errors is incremented."""
        mock_fetch.return_value = [_make_news_item("AAPL earnings")]

        ticker_signal = TickerSignal(
            ticker="AAPL", headline="AAPL earnings", category="earnings",
            sentiment="bullish", confidence="high", published_at=SAMPLE_PUBLISHED_AT
        )
        mock_classify.return_value = [
            ClassificationResult(news_type="ticker_specific",
                                 ticker_signals=[ticker_signal], macro_signal=None),
        ]

        stats = run_pipeline()

        assert stats.errors == 1
        assert stats.ticker_signals_stored == 0


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_no_news_filtered_attribute(self):
        """PipelineStats should not have a news_filtered field (V3 removed filter step)."""
        stats = PipelineStats(
            news_fetched=10,
            ticker_signals_stored=5,
            macro_signals_stored=2,
            noise_dropped=3,
            errors=0
        )
        assert not hasattr(stats, "news_filtered")
