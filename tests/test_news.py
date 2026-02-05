"""Tests for trading/news.py - Alpaca news fetching."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from trading.news import NewsItem, get_news_client, fetch_news, fetch_broad_news, fetch_ticker_news


# ---------------------------------------------------------------------------
# NewsItem dataclass tests
# ---------------------------------------------------------------------------

class TestNewsItem:
    """Tests for the NewsItem dataclass."""

    def test_create_news_item_with_all_fields(self):
        item = NewsItem(
            id="abc-123",
            headline="AAPL beats earnings",
            summary="Apple reported better results",
            author="John Doe",
            source="benzinga",
            symbols=["AAPL"],
            published_at=datetime(2025, 1, 15, 10, 0, 0),
            url="https://example.com/news/1",
        )
        assert item.id == "abc-123"
        assert item.headline == "AAPL beats earnings"
        assert item.summary == "Apple reported better results"
        assert item.author == "John Doe"
        assert item.source == "benzinga"
        assert item.symbols == ["AAPL"]
        assert item.published_at == datetime(2025, 1, 15, 10, 0, 0)
        assert item.url == "https://example.com/news/1"

    def test_create_news_item_with_multiple_symbols(self):
        item = NewsItem(
            id="multi-1",
            headline="AAPL and MSFT merger talks",
            summary="",
            author="",
            source="reuters",
            symbols=["AAPL", "MSFT"],
            published_at=datetime(2025, 6, 1),
            url="",
        )
        assert item.symbols == ["AAPL", "MSFT"]
        assert len(item.symbols) == 2

    def test_create_news_item_with_empty_symbols(self):
        item = NewsItem(
            id="no-sym",
            headline="Fed raises rates",
            summary="",
            author="",
            source="reuters",
            symbols=[],
            published_at=datetime(2025, 6, 1),
            url="",
        )
        assert item.symbols == []

    def test_news_item_equality(self):
        kwargs = dict(
            id="x", headline="h", summary="s", author="a",
            source="src", symbols=[], published_at=datetime(2025, 1, 1), url="u"
        )
        assert NewsItem(**kwargs) == NewsItem(**kwargs)

    def test_news_item_inequality(self):
        base = dict(
            id="x", headline="h", summary="s", author="a",
            source="src", symbols=[], published_at=datetime(2025, 1, 1), url="u"
        )
        item1 = NewsItem(**base)
        item2 = NewsItem(**{**base, "id": "y"})
        assert item1 != item2


# ---------------------------------------------------------------------------
# get_news_client tests
# ---------------------------------------------------------------------------

class TestGetNewsClient:
    """Tests for get_news_client()."""

    def test_raises_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            get_news_client()

    def test_raises_when_only_api_key_set(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test-key")
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            get_news_client()

    def test_raises_when_only_secret_key_set(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            get_news_client()

    def test_raises_when_keys_are_empty_strings(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "")
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            get_news_client()

    @patch("trading.news.NewsClient")
    def test_creates_client_with_valid_keys(self, mock_client_class, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "my-key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "my-secret")
        client = get_news_client()
        mock_client_class.assert_called_once_with("my-key", "my-secret")
        assert client == mock_client_class.return_value


# ---------------------------------------------------------------------------
# Helpers for mocking Alpaca news response
# ---------------------------------------------------------------------------

def _make_mock_news_item(
    news_id="uuid-1",
    headline="Test headline",
    summary="Test summary",
    author="Author",
    source="benzinga",
    symbols=None,
    created_at=None,
    url="https://example.com/1",
):
    """Create a mock Alpaca news object (not our NewsItem dataclass)."""
    mock = MagicMock()
    mock.id = news_id
    mock.headline = headline
    mock.summary = summary
    mock.author = author
    mock.source = source
    mock.symbols = symbols or ["AAPL"]
    mock.created_at = created_at or datetime(2025, 1, 15, 10, 0, 0)
    mock.url = url
    return mock


def _mock_news_client_with_items(items):
    """Create a mock NewsClient whose get_news returns the given items."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = {"news": items}
    mock_client.get_news.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# fetch_news tests
# ---------------------------------------------------------------------------

class TestFetchNews:
    """Tests for fetch_news()."""

    @patch("trading.news.get_news_client")
    def test_returns_empty_list_when_no_news(self, mock_get_client):
        mock_get_client.return_value = _mock_news_client_with_items([])
        result = fetch_news(hours=24)
        assert result == []

    @patch("trading.news.get_news_client")
    def test_converts_alpaca_news_to_news_items(self, mock_get_client):
        alpaca_item = _make_mock_news_item(
            news_id="abc",
            headline="AAPL beats Q3",
            summary="Apple reported strong results",
            author="Jane",
            source="reuters",
            symbols=["AAPL"],
            created_at=datetime(2025, 3, 15, 14, 0, 0),
            url="https://example.com/aapl",
        )
        mock_get_client.return_value = _mock_news_client_with_items([alpaca_item])

        result = fetch_news(hours=24)

        assert len(result) == 1
        item = result[0]
        assert isinstance(item, NewsItem)
        assert item.id == "abc"
        assert item.headline == "AAPL beats Q3"
        assert item.summary == "Apple reported strong results"
        assert item.author == "Jane"
        assert item.source == "reuters"
        assert item.symbols == ["AAPL"]
        assert item.published_at == datetime(2025, 3, 15, 14, 0, 0)
        assert item.url == "https://example.com/aapl"

    @patch("trading.news.get_news_client")
    def test_handles_none_optional_fields(self, mock_get_client):
        """summary, author, source, symbols, and url can be None from Alpaca."""
        alpaca_item = _make_mock_news_item()
        alpaca_item.summary = None
        alpaca_item.author = None
        alpaca_item.source = None
        alpaca_item.symbols = None
        alpaca_item.url = None
        mock_get_client.return_value = _mock_news_client_with_items([alpaca_item])

        result = fetch_news(hours=24)
        item = result[0]
        assert item.summary == ""
        assert item.author == ""
        assert item.source == ""
        assert item.symbols == []
        assert item.url == ""

    @patch("trading.news.get_news_client")
    def test_multiple_news_items(self, mock_get_client):
        items = [
            _make_mock_news_item(news_id=f"id-{i}", headline=f"Headline {i}")
            for i in range(5)
        ]
        mock_get_client.return_value = _mock_news_client_with_items(items)

        result = fetch_news(hours=48, limit=100)
        assert len(result) == 5
        assert result[0].id == "id-0"
        assert result[4].id == "id-4"

    @patch("trading.news.get_news_client")
    @patch("trading.news.NewsRequest")
    def test_passes_symbols_to_request_when_provided(self, mock_request, mock_get_client):
        mock_get_client.return_value = _mock_news_client_with_items([])

        fetch_news(hours=24, symbols=["AAPL", "MSFT"], limit=10)

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["symbols"] == ["AAPL", "MSFT"]
        assert call_kwargs["limit"] == 10
        assert "start" in call_kwargs

    @patch("trading.news.get_news_client")
    @patch("trading.news.NewsRequest")
    def test_omits_symbols_from_request_when_none(self, mock_request, mock_get_client):
        mock_get_client.return_value = _mock_news_client_with_items([])

        fetch_news(hours=24, symbols=None, limit=50)

        call_kwargs = mock_request.call_args[1]
        assert "symbols" not in call_kwargs

    @patch("trading.news.get_news_client")
    @patch("trading.news.NewsRequest")
    def test_sort_is_desc(self, mock_request, mock_get_client):
        mock_get_client.return_value = _mock_news_client_with_items([])

        fetch_news(hours=12)

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["sort"] == "desc"

    @patch("trading.news.get_news_client")
    def test_id_is_converted_to_string(self, mock_get_client):
        alpaca_item = _make_mock_news_item()
        alpaca_item.id = 12345  # integer id
        mock_get_client.return_value = _mock_news_client_with_items([alpaca_item])

        result = fetch_news(hours=24)
        assert result[0].id == "12345"
        assert isinstance(result[0].id, str)


# ---------------------------------------------------------------------------
# fetch_broad_news tests
# ---------------------------------------------------------------------------

class TestFetchBroadNews:
    """Tests for fetch_broad_news()."""

    @patch("trading.news.fetch_news")
    def test_calls_fetch_news_without_symbols(self, mock_fetch):
        mock_fetch.return_value = []
        fetch_broad_news(hours=48, limit=100)
        mock_fetch.assert_called_once_with(hours=48, symbols=None, limit=100)

    @patch("trading.news.fetch_news")
    def test_default_parameters(self, mock_fetch):
        mock_fetch.return_value = []
        fetch_broad_news()
        mock_fetch.assert_called_once_with(hours=24, symbols=None, limit=50)

    @patch("trading.news.fetch_news")
    def test_returns_result_from_fetch_news(self, mock_fetch):
        expected = [MagicMock()]
        mock_fetch.return_value = expected
        result = fetch_broad_news()
        assert result is expected


# ---------------------------------------------------------------------------
# fetch_ticker_news tests
# ---------------------------------------------------------------------------

class TestFetchTickerNews:
    """Tests for fetch_ticker_news()."""

    @patch("trading.news.fetch_news")
    def test_calls_fetch_news_with_ticker_as_list(self, mock_fetch):
        mock_fetch.return_value = []
        fetch_ticker_news("AAPL", hours=12, limit=5)
        mock_fetch.assert_called_once_with(hours=12, symbols=["AAPL"], limit=5)

    @patch("trading.news.fetch_news")
    def test_default_parameters(self, mock_fetch):
        mock_fetch.return_value = []
        fetch_ticker_news("TSLA")
        mock_fetch.assert_called_once_with(hours=24, symbols=["TSLA"], limit=20)

    @patch("trading.news.fetch_news")
    def test_returns_result_from_fetch_news(self, mock_fetch):
        expected = [MagicMock(), MagicMock()]
        mock_fetch.return_value = expected
        result = fetch_ticker_news("MSFT")
        assert result is expected

    @patch("trading.news.fetch_news")
    def test_preserves_ticker_casing(self, mock_fetch):
        """fetch_ticker_news should pass the ticker as-is."""
        mock_fetch.return_value = []
        fetch_ticker_news("aapl")
        mock_fetch.assert_called_once_with(hours=24, symbols=["aapl"], limit=20)
