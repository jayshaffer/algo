"""Tests for trading/filter.py - Embedding-based relevance filtering."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from trading.filter import (
    FilteredNewsItem,
    filter_by_relevance,
    filter_news_batch,
    DEFAULT_STRATEGY_CONTEXT,
)
from trading.news import NewsItem

from tests.conftest import make_news_item


# ---------------------------------------------------------------------------
# FilteredNewsItem dataclass tests
# ---------------------------------------------------------------------------

class TestFilteredNewsItem:
    """Tests for the FilteredNewsItem dataclass."""

    def test_create_filtered_news_item(self):
        item = make_news_item()
        filtered = FilteredNewsItem(item=item, relevance_score=0.85)
        assert filtered.item is item
        assert filtered.relevance_score == 0.85

    def test_relevance_score_can_be_zero(self):
        filtered = FilteredNewsItem(item=make_news_item(), relevance_score=0.0)
        assert filtered.relevance_score == 0.0

    def test_relevance_score_can_be_one(self):
        filtered = FilteredNewsItem(item=make_news_item(), relevance_score=1.0)
        assert filtered.relevance_score == 1.0


# ---------------------------------------------------------------------------
# filter_by_relevance tests
# ---------------------------------------------------------------------------

class TestFilterByRelevance:
    """Tests for filter_by_relevance()."""

    def test_empty_list_returns_empty(self):
        """No embedding calls should be made for empty input."""
        result = filter_by_relevance([], strategy_context="anything", threshold=0.3)
        assert result == []

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_all_items_above_threshold_are_kept(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [
            make_news_item(id="1", headline="AAPL earnings beat"),
            make_news_item(id="2", headline="Fed raises rates"),
        ]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.2] * 768, [0.3] * 768]
        mock_cosine.return_value = [0.8, 0.6]

        result = filter_by_relevance(items, threshold=0.3)

        assert len(result) == 2
        assert all(isinstance(r, FilteredNewsItem) for r in result)

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_items_below_threshold_are_filtered_out(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [
            make_news_item(id="1", headline="Relevant news"),
            make_news_item(id="2", headline="Irrelevant noise"),
            make_news_item(id="3", headline="Another relevant"),
        ]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768 for _ in items]
        mock_cosine.return_value = [0.5, 0.1, 0.4]

        result = filter_by_relevance(items, threshold=0.3)

        assert len(result) == 2
        # Items with scores 0.1 should be dropped
        ids = [r.item.id for r in result]
        assert "2" not in ids
        assert "1" in ids
        assert "3" in ids

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_results_sorted_by_relevance_descending(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [
            make_news_item(id="low", headline="Low relevance"),
            make_news_item(id="high", headline="High relevance"),
            make_news_item(id="mid", headline="Mid relevance"),
        ]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768 for _ in items]
        mock_cosine.return_value = [0.4, 0.9, 0.6]

        result = filter_by_relevance(items, threshold=0.3)

        assert len(result) == 3
        assert result[0].relevance_score == 0.9
        assert result[1].relevance_score == 0.6
        assert result[2].relevance_score == 0.4
        assert result[0].item.id == "high"
        assert result[1].item.id == "mid"
        assert result[2].item.id == "low"

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_threshold_boundary_equal_is_included(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        """Items with score exactly equal to threshold should be included."""
        items = [make_news_item(id="exact", headline="Exact threshold")]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768]
        mock_cosine.return_value = [0.3]

        result = filter_by_relevance(items, threshold=0.3)

        assert len(result) == 1
        assert result[0].relevance_score == 0.3

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_threshold_boundary_just_below_is_excluded(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [make_news_item(id="below", headline="Just below")]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768]
        mock_cosine.return_value = [0.29999]

        result = filter_by_relevance(items, threshold=0.3)

        assert len(result) == 0

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_all_items_below_threshold_returns_empty(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [
            make_news_item(id="1", headline="Noise 1"),
            make_news_item(id="2", headline="Noise 2"),
        ]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768 for _ in items]
        mock_cosine.return_value = [0.05, 0.1]

        result = filter_by_relevance(items, threshold=0.3)

        assert result == []

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_zero_threshold_keeps_everything(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [
            make_news_item(id="1", headline="Item 1"),
            make_news_item(id="2", headline="Item 2"),
        ]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768 for _ in items]
        mock_cosine.return_value = [0.01, 0.99]

        result = filter_by_relevance(items, threshold=0.0)

        assert len(result) == 2

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_high_threshold_filters_aggressively(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [
            make_news_item(id="1", headline="Good"),
            make_news_item(id="2", headline="Better"),
            make_news_item(id="3", headline="Best"),
        ]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768 for _ in items]
        mock_cosine.return_value = [0.7, 0.85, 0.95]

        result = filter_by_relevance(items, threshold=0.9)

        assert len(result) == 1
        assert result[0].item.id == "3"

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_embed_called_with_strategy_context(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [make_news_item()]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768]
        mock_cosine.return_value = [0.5]

        custom_context = "My custom trading strategy context"
        filter_by_relevance(items, strategy_context=custom_context, threshold=0.3)

        mock_embed.assert_called_once_with(custom_context)

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_embed_batch_called_with_headlines(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [
            make_news_item(headline="Headline A"),
            make_news_item(headline="Headline B"),
        ]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768, [0.1] * 768]
        mock_cosine.return_value = [0.5, 0.5]

        filter_by_relevance(items, threshold=0.3)

        mock_embed_batch.assert_called_once_with(["Headline A", "Headline B"])

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_cosine_similarity_called_with_context_and_headline_embeddings(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [make_news_item()]
        ctx_vec = [0.5] * 768
        headline_vecs = [[0.2] * 768]
        mock_embed.return_value = ctx_vec
        mock_embed_batch.return_value = headline_vecs
        mock_cosine.return_value = [0.6]

        filter_by_relevance(items, threshold=0.3)

        mock_cosine.assert_called_once_with(ctx_vec, headline_vecs)

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_default_strategy_context_used_when_none_provided(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [make_news_item()]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768]
        mock_cosine.return_value = [0.5]

        filter_by_relevance(items)

        mock_embed.assert_called_once_with(DEFAULT_STRATEGY_CONTEXT)

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_single_item_above_threshold(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [make_news_item(id="only", headline="Solo headline")]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768]
        mock_cosine.return_value = [0.75]

        result = filter_by_relevance(items, threshold=0.5)

        assert len(result) == 1
        assert result[0].item.id == "only"
        assert result[0].relevance_score == 0.75

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_preserves_news_item_data(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        """The original NewsItem should be accessible unmodified."""
        original = make_news_item(
            id="preserve-test",
            headline="Important News",
            summary="Details here",
            symbols=["GOOG", "META"],
        )
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768]
        mock_cosine.return_value = [0.8]

        result = filter_by_relevance([original], threshold=0.3)

        assert result[0].item.id == "preserve-test"
        assert result[0].item.headline == "Important News"
        assert result[0].item.summary == "Details here"
        assert result[0].item.symbols == ["GOOG", "META"]

    @patch("trading.filter.cosine_similarity_batch")
    @patch("trading.filter.embed_batch")
    @patch("trading.filter.embed")
    def test_negative_scores_filtered_below_positive_threshold(
        self, mock_embed, mock_embed_batch, mock_cosine
    ):
        items = [make_news_item(id="neg")]
        mock_embed.return_value = [0.1] * 768
        mock_embed_batch.return_value = [[0.1] * 768]
        mock_cosine.return_value = [-0.2]

        result = filter_by_relevance(items, threshold=0.3)

        assert result == []


# ---------------------------------------------------------------------------
# filter_news_batch tests
# ---------------------------------------------------------------------------

class TestFilterNewsBatch:
    """Tests for filter_news_batch()."""

    @patch("trading.filter.filter_by_relevance")
    def test_delegates_to_filter_by_relevance(self, mock_filter):
        items = [make_news_item(), make_news_item(id="2")]
        expected = [FilteredNewsItem(item=items[0], relevance_score=0.9)]
        mock_filter.return_value = expected

        result = filter_news_batch(
            items,
            strategy_context="test context",
            threshold=0.5,
            batch_size=10,
        )

        mock_filter.assert_called_once_with(items, "test context", 0.5)
        assert result is expected

    @patch("trading.filter.filter_by_relevance")
    def test_empty_input(self, mock_filter):
        mock_filter.return_value = []

        result = filter_news_batch([], strategy_context="ctx", threshold=0.3)

        mock_filter.assert_called_once_with([], "ctx", 0.3)
        assert result == []

    @patch("trading.filter.filter_by_relevance")
    def test_prints_progress(self, mock_filter, capsys):
        items = [make_news_item() for _ in range(5)]
        mock_filter.return_value = [
            FilteredNewsItem(item=items[0], relevance_score=0.8),
            FilteredNewsItem(item=items[1], relevance_score=0.6),
        ]

        filter_news_batch(items, threshold=0.3)

        captured = capsys.readouterr()
        assert "Batch filtering 5 items" in captured.out
        assert "Filtered 5 items, 2 passed threshold" in captured.out

    @patch("trading.filter.filter_by_relevance")
    def test_default_strategy_context(self, mock_filter):
        mock_filter.return_value = []

        filter_news_batch([make_news_item()])

        call_args = mock_filter.call_args
        assert call_args[0][1] == DEFAULT_STRATEGY_CONTEXT

    @patch("trading.filter.filter_by_relevance")
    def test_default_threshold(self, mock_filter):
        mock_filter.return_value = []

        filter_news_batch([make_news_item()])

        call_args = mock_filter.call_args
        assert call_args[0][2] == 0.3
