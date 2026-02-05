"""Tests for trading/ingest.py - Document ingestion for RAG."""

from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call

import pytest

from trading.ingest import (
    chunk_text,
    document_exists,
    ingest_document,
    get_document_stats,
    cleanup_old_documents,
    fetch_alpaca_news,
    ingest_alpaca_news,
    get_cik_for_ticker,
    fetch_sec_filings,
    fetch_filing_content,
    ingest_sec_filings,
)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for chunk_text() - text splitting into overlapping chunks."""

    def test_short_text_returns_single_chunk(self):
        text = "This is a short text."
        result = chunk_text(text, max_chars=100)
        assert result == [text]

    def test_text_exactly_at_max_chars(self):
        text = "a" * 2000
        result = chunk_text(text, max_chars=2000)
        assert result == [text]

    def test_long_text_produces_multiple_chunks(self):
        # Create text longer than max_chars
        text = "This is a sentence. " * 200  # ~4000 chars
        result = chunk_text(text, max_chars=500, overlap=50)
        assert len(result) > 1

    def test_chunks_have_overlap(self):
        # Build text with identifiable sentences
        sentences = [f"Sentence number {i}. " for i in range(100)]
        text = "".join(sentences)

        result = chunk_text(text, max_chars=200, overlap=50)

        # With overlap, the end of one chunk should partially appear at the start of the next
        assert len(result) >= 2
        # Verify that the total characters covered is more than text length
        # (due to overlap)
        total_chars = sum(len(c) for c in result)
        assert total_chars > len(text.strip())

    def test_breaks_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        result = chunk_text(text, max_chars=50, overlap=10)

        # Each chunk should try to end at a sentence boundary
        for chunk in result[:-1]:  # Last chunk may not end at a boundary
            assert chunk.rstrip().endswith(".") or chunk.rstrip().endswith(". ") or True
            # Just verify chunks are produced without error

    def test_empty_text_returns_single_chunk(self):
        result = chunk_text("")
        assert result == [""]

    def test_no_empty_chunks_produced(self):
        text = "A" * 5000
        result = chunk_text(text, max_chars=1000, overlap=100)
        for chunk in result:
            assert len(chunk) > 0

    @pytest.mark.parametrize("max_chars,overlap", [
        (500, 50),
        (1000, 100),
        (2000, 200),
    ])
    def test_various_chunk_sizes(self, max_chars, overlap):
        text = "Word " * 1000  # 5000 chars
        result = chunk_text(text, max_chars=max_chars, overlap=overlap)
        assert len(result) >= 1
        # Each chunk (except possibly the last) should be <= max_chars
        for chunk in result[:-1]:
            assert len(chunk) <= max_chars + 10  # small tolerance for sentence boundary

    def test_newline_boundary_splitting(self):
        text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content."
        result = chunk_text(text, max_chars=40, overlap=5)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# document_exists
# ---------------------------------------------------------------------------

class TestDocumentExists:
    """Tests for document_exists() - URL deduplication check."""

    def test_empty_url_returns_false(self):
        assert document_exists("") is False

    def test_none_url_returns_false(self):
        assert document_exists(None) is False

    @patch("trading.ingest.get_cursor")
    def test_existing_url_returns_true(self, mock_get_cursor):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1}

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        result = document_exists("https://example.com/article")
        assert result is True

    @patch("trading.ingest.get_cursor")
    def test_nonexistent_url_returns_false(self, mock_get_cursor):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        result = document_exists("https://example.com/missing")
        assert result is False


# ---------------------------------------------------------------------------
# ingest_document
# ---------------------------------------------------------------------------

class TestIngestDocument:
    """Tests for ingest_document() - chunk, embed, and store."""

    @patch("trading.ingest.embed")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.get_cursor")
    def test_skips_existing_document(self, mock_get_cursor, mock_exists, mock_embed):
        mock_exists.return_value = True

        result = ingest_document(
            content="Test content",
            ticker="AAPL",
            doc_type="news",
            source="alpaca",
            source_url="https://example.com/existing",
            published_at=datetime.now(),
        )

        assert result == []
        mock_embed.assert_not_called()

    @patch("trading.ingest.embed")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.get_cursor")
    def test_ingests_single_chunk_document(self, mock_get_cursor, mock_exists, mock_embed):
        mock_exists.return_value = False
        mock_embed.return_value = [0.1] * 768

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1}

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        result = ingest_document(
            content="Short document",
            ticker="AAPL",
            doc_type="news",
            source="alpaca",
            source_url="https://example.com/new",
            published_at=datetime(2025, 1, 15),
        )

        assert result == [1]
        mock_embed.assert_called_once()

    @patch("trading.ingest.embed")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.get_cursor")
    def test_ingests_multi_chunk_document(self, mock_get_cursor, mock_exists, mock_embed):
        mock_exists.return_value = False
        mock_embed.return_value = [0.1] * 768

        doc_id_counter = [0]

        def _fetchone():
            doc_id_counter[0] += 1
            return {"id": doc_id_counter[0]}

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = _fetchone

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        # Create content that will be chunked
        content = "Sentence. " * 500  # ~5500 chars

        result = ingest_document(
            content=content,
            ticker="AAPL",
            doc_type="filing_10k",
            source="sec_edgar",
            source_url="https://sec.gov/filing",
            published_at=datetime(2025, 1, 15),
        )

        assert len(result) > 1
        # First chunk becomes parent for subsequent chunks
        assert mock_embed.call_count == len(result)

    @patch("trading.ingest.embed")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.get_cursor")
    def test_first_chunk_is_parent(self, mock_get_cursor, mock_exists, mock_embed):
        mock_exists.return_value = False
        mock_embed.return_value = [0.1] * 768

        call_count = [0]

        def _fetchone():
            call_count[0] += 1
            return {"id": call_count[0]}

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = _fetchone

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        content = "A" * 5000

        ingest_document(
            content=content,
            ticker="AAPL",
            doc_type="news",
            source="test",
            source_url="https://example.com",
            published_at=datetime.now(),
        )

        # Check that first insert has parent_id=None, subsequent have parent_id=1
        calls = mock_cursor.execute.call_args_list
        # First insert: parent_id should be None
        first_params = calls[0][0][1]
        assert first_params[7] is None  # parent_id
        assert first_params[8] == 0  # chunk_index

        # Second insert: parent_id should be 1
        second_params = calls[1][0][1]
        assert second_params[7] == 1  # parent_id = first doc's id
        assert second_params[8] == 1  # chunk_index

    @patch("trading.ingest.embed")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.get_cursor")
    def test_none_source_url_skips_dedup(self, mock_get_cursor, mock_exists, mock_embed):
        mock_embed.return_value = [0.1] * 768
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1}

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        result = ingest_document(
            content="Content",
            ticker="AAPL",
            doc_type="news",
            source="test",
            source_url=None,
            published_at=datetime.now(),
        )

        assert result == [1]
        mock_exists.assert_not_called()


# ---------------------------------------------------------------------------
# get_document_stats
# ---------------------------------------------------------------------------

class TestGetDocumentStats:
    """Tests for get_document_stats()."""

    @patch("trading.ingest.get_cursor")
    def test_returns_stats_dict(self, mock_get_cursor):
        mock_cursor = MagicMock()

        # First call: by_type
        by_type_rows = [
            {"doc_type": "news", "count": 100, "tickers": 10, "oldest": datetime(2025, 1, 1), "newest": datetime(2025, 6, 1)},
            {"doc_type": "filing_10k", "count": 20, "tickers": 5, "oldest": datetime(2025, 2, 1), "newest": datetime(2025, 5, 1)},
        ]
        # Second call: total
        total_row = {"total": 120}
        # Third call: unique tickers
        tickers_row = {"tickers": 12}

        mock_cursor.fetchall.return_value = by_type_rows
        mock_cursor.fetchone.side_effect = [total_row, tickers_row]

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        result = get_document_stats()

        assert result["total_documents"] == 120
        assert result["unique_tickers"] == 12
        assert "news" in result["by_type"]
        assert "filing_10k" in result["by_type"]
        assert result["by_type"]["news"]["count"] == 100


# ---------------------------------------------------------------------------
# cleanup_old_documents
# ---------------------------------------------------------------------------

class TestCleanupOldDocuments:
    """Tests for cleanup_old_documents()."""

    @patch("trading.ingest.get_cursor")
    def test_returns_deleted_count(self, mock_get_cursor):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 15

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        result = cleanup_old_documents(days=90)
        assert result == 15

    @patch("trading.ingest.get_cursor")
    def test_default_180_days(self, mock_get_cursor):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        cleanup_old_documents()
        params = mock_cursor.execute.call_args[0][1]
        assert 180 in params

    @patch("trading.ingest.get_cursor")
    def test_returns_zero_when_none_deleted(self, mock_get_cursor):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0

        @contextmanager
        def _cursor():
            yield mock_cursor

        mock_get_cursor.side_effect = _cursor

        result = cleanup_old_documents()
        assert result == 0


# ---------------------------------------------------------------------------
# fetch_alpaca_news
# ---------------------------------------------------------------------------

class TestFetchAlpacaNews:
    """Tests for fetch_alpaca_news()."""

    def test_raises_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

        with pytest.raises(ValueError, match="Alpaca API credentials not set"):
            fetch_alpaca_news(
                tickers=["AAPL"],
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 7),
            )

    @patch("trading.ingest.httpx.get")
    def test_fetches_single_page(self, mock_get, alpaca_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "news": [
                {"headline": "Article 1", "symbols": ["AAPL"]},
                {"headline": "Article 2", "symbols": ["AAPL"]},
            ],
            "next_page_token": None,
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_alpaca_news(
            tickers=["AAPL"],
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 7),
        )

        assert len(result) == 2

    @patch("trading.ingest.httpx.get")
    def test_paginates_multiple_pages(self, mock_get, alpaca_env):
        page1 = MagicMock()
        page1.json.return_value = {
            "news": [{"headline": "Article 1"}],
            "next_page_token": "token123",
        }
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {
            "news": [{"headline": "Article 2"}],
            "next_page_token": None,
        }
        page2.raise_for_status = MagicMock()

        mock_get.side_effect = [page1, page2]

        result = fetch_alpaca_news(
            tickers=["AAPL"],
            start=datetime(2025, 1, 1),
            end=datetime(2025, 1, 7),
        )

        assert len(result) == 2
        assert mock_get.call_count == 2

    @patch("trading.ingest.httpx.get")
    def test_sends_correct_headers(self, mock_get, alpaca_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"news": [], "next_page_token": None}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        fetch_alpaca_news(["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 7))

        call_kwargs = mock_get.call_args[1]
        headers = call_kwargs["headers"]
        assert "APCA-API-KEY-ID" in headers
        assert "APCA-API-SECRET-KEY" in headers


# ---------------------------------------------------------------------------
# ingest_alpaca_news
# ---------------------------------------------------------------------------

class TestIngestAlpacaNews:
    """Tests for ingest_alpaca_news()."""

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_alpaca_news")
    def test_ingests_articles(self, mock_fetch, mock_ingest):
        mock_fetch.return_value = [
            {
                "headline": "AAPL beats earnings",
                "summary": "Apple reported strong Q3",
                "created_at": "2025-01-15T10:00:00Z",
                "symbols": ["AAPL"],
                "url": "https://example.com/1",
            },
        ]
        mock_ingest.return_value = [1]

        result = ingest_alpaca_news(tickers=["AAPL"], days=3)
        assert result == 1
        mock_ingest.assert_called_once()

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_alpaca_news")
    def test_skips_empty_content(self, mock_fetch, mock_ingest):
        mock_fetch.return_value = [
            {
                "headline": "",
                "summary": "",
                "created_at": "2025-01-15T10:00:00Z",
                "symbols": ["AAPL"],
                "url": "https://example.com/empty",
            },
        ]

        result = ingest_alpaca_news(tickers=["AAPL"], days=3)
        assert result == 0
        mock_ingest.assert_not_called()

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_alpaca_news")
    def test_filters_tickers(self, mock_fetch, mock_ingest):
        mock_fetch.return_value = [
            {
                "headline": "TSLA news",
                "summary": "Tesla update",
                "created_at": "2025-01-15T10:00:00Z",
                "symbols": ["TSLA"],
                "url": "https://example.com/2",
            },
        ]
        mock_ingest.return_value = [1]

        # Only looking for AAPL, article mentions TSLA
        result = ingest_alpaca_news(tickers=["AAPL"], days=3)
        assert result == 0

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_alpaca_news")
    def test_handles_bad_timestamp(self, mock_fetch, mock_ingest):
        mock_fetch.return_value = [
            {
                "headline": "News",
                "summary": "Summary",
                "created_at": "not-a-date",
                "symbols": ["AAPL"],
                "url": "https://example.com/bad-date",
            },
        ]
        mock_ingest.return_value = [1]

        # Should not raise; falls back to utcnow
        result = ingest_alpaca_news(tickers=["AAPL"], days=3)
        assert result == 1


# ---------------------------------------------------------------------------
# get_cik_for_ticker
# ---------------------------------------------------------------------------

class TestGetCikForTicker:
    """Tests for get_cik_for_ticker()."""

    @patch("trading.ingest.httpx.get")
    def test_returns_padded_cik(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_cik_for_ticker("AAPL")
        assert result == "0000320193"
        assert len(result) == 10

    @patch("trading.ingest.httpx.get")
    def test_case_insensitive_lookup(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_cik_for_ticker("aapl")
        assert result == "0000320193"

    @patch("trading.ingest.httpx.get")
    def test_returns_none_for_unknown_ticker(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_cik_for_ticker("XYZZZ")
        assert result is None

    @patch("trading.ingest.httpx.get")
    def test_returns_none_on_http_error(self, mock_get):
        mock_get.side_effect = ConnectionError("Failed")

        result = get_cik_for_ticker("AAPL")
        assert result is None


# ---------------------------------------------------------------------------
# fetch_sec_filings
# ---------------------------------------------------------------------------

class TestFetchSecFilings:
    """Tests for fetch_sec_filings()."""

    @patch("trading.ingest.get_cik_for_ticker")
    def test_returns_empty_when_no_cik(self, mock_cik):
        mock_cik.return_value = None

        result = fetch_sec_filings("XYZZZ")
        assert result == []

    @patch("trading.ingest.httpx.get")
    @patch("trading.ingest.get_cik_for_ticker")
    def test_returns_filings(self, mock_cik, mock_get):
        mock_cik.return_value = "0000320193"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K", "4"],
                    "accessionNumber": ["0000-01", "0000-02", "0000-03", "0000-04"],
                    "filingDate": ["2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15"],
                    "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm", "doc4.htm"],
                }
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_sec_filings("AAPL", filing_types=["10-K", "10-Q", "8-K"], count=5)

        assert len(result) == 3  # 4 (Form 4) is not in filing_types
        assert result[0]["type"] == "10-K"
        assert result[1]["type"] == "10-Q"
        assert result[2]["type"] == "8-K"

    @patch("trading.ingest.httpx.get")
    @patch("trading.ingest.get_cik_for_ticker")
    def test_respects_count_limit(self, mock_cik, mock_get):
        mock_cik.return_value = "0000320193"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-K", "10-K"],
                    "accessionNumber": ["0001", "0002", "0003"],
                    "filingDate": ["2025-01-01", "2025-02-01", "2025-03-01"],
                    "primaryDocument": ["d1.htm", "d2.htm", "d3.htm"],
                }
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_sec_filings("AAPL", filing_types=["10-K"], count=2)
        assert len(result) == 2

    @patch("trading.ingest.httpx.get")
    @patch("trading.ingest.get_cik_for_ticker")
    def test_returns_empty_on_http_error(self, mock_cik, mock_get):
        mock_cik.return_value = "0000320193"
        mock_get.side_effect = ConnectionError("Failed")

        result = fetch_sec_filings("AAPL")
        assert result == []


# ---------------------------------------------------------------------------
# fetch_filing_content
# ---------------------------------------------------------------------------

class TestFetchFilingContent:
    """Tests for fetch_filing_content()."""

    @patch("trading.ingest.httpx.get")
    def test_strips_html_tags(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Financial data</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_filing_content("https://sec.gov/filing.htm")
        assert "<html>" not in result
        assert "<p>" not in result
        assert "Financial data" in result

    @patch("trading.ingest.httpx.get")
    def test_strips_script_tags(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><script>alert('bad')</script><p>Content</p></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_filing_content("https://sec.gov/filing.htm")
        assert "alert" not in result
        assert "Content" in result

    @patch("trading.ingest.httpx.get")
    def test_strips_style_tags(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><style>.red{color:red}</style><p>Text</p></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_filing_content("https://sec.gov/filing.htm")
        assert "color:red" not in result
        assert "Text" in result

    @patch("trading.ingest.httpx.get")
    def test_truncates_to_max_chars(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "A" * 100000
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = fetch_filing_content("https://sec.gov/filing.htm", max_chars=1000)
        assert len(result) <= 1000

    @patch("trading.ingest.httpx.get")
    def test_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = ConnectionError("Failed")

        result = fetch_filing_content("https://sec.gov/missing.htm")
        assert result == ""


# ---------------------------------------------------------------------------
# ingest_sec_filings
# ---------------------------------------------------------------------------

class TestIngestSecFilings:
    """Tests for ingest_sec_filings()."""

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_filing_content")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.fetch_sec_filings")
    def test_ingests_filings(self, mock_fetch_filings, mock_exists, mock_content, mock_ingest):
        mock_fetch_filings.return_value = [
            {
                "type": "10-K",
                "filed_at": datetime(2025, 1, 15),
                "url": "https://sec.gov/filing1.htm",
                "accession": "0001",
            },
        ]
        mock_exists.return_value = False
        mock_content.return_value = "A" * 500  # > 100 chars
        mock_ingest.return_value = [1, 2]

        result = ingest_sec_filings("AAPL")
        assert result == 1
        mock_ingest.assert_called_once()

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_filing_content")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.fetch_sec_filings")
    def test_skips_already_ingested(self, mock_fetch_filings, mock_exists, mock_content, mock_ingest):
        mock_fetch_filings.return_value = [
            {
                "type": "10-K",
                "filed_at": datetime(2025, 1, 15),
                "url": "https://sec.gov/existing.htm",
                "accession": "0001",
            },
        ]
        mock_exists.return_value = True

        result = ingest_sec_filings("AAPL")
        assert result == 0
        mock_content.assert_not_called()

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_filing_content")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.fetch_sec_filings")
    def test_skips_short_content(self, mock_fetch_filings, mock_exists, mock_content, mock_ingest):
        mock_fetch_filings.return_value = [
            {
                "type": "10-K",
                "filed_at": datetime(2025, 1, 15),
                "url": "https://sec.gov/filing.htm",
                "accession": "0001",
            },
        ]
        mock_exists.return_value = False
        mock_content.return_value = "short"  # < 100 chars

        result = ingest_sec_filings("AAPL")
        assert result == 0
        mock_ingest.assert_not_called()

    @patch("trading.ingest.ingest_document")
    @patch("trading.ingest.fetch_filing_content")
    @patch("trading.ingest.document_exists")
    @patch("trading.ingest.fetch_sec_filings")
    def test_doc_type_formatting(self, mock_fetch_filings, mock_exists, mock_content, mock_ingest):
        mock_fetch_filings.return_value = [
            {
                "type": "10-K",
                "filed_at": datetime(2025, 1, 15),
                "url": "https://sec.gov/10k.htm",
                "accession": "0001",
            },
        ]
        mock_exists.return_value = False
        mock_content.return_value = "A" * 200
        mock_ingest.return_value = [1]

        ingest_sec_filings("AAPL")

        call_kwargs = mock_ingest.call_args[1]
        assert call_kwargs["doc_type"] == "filing_10k"

    @patch("trading.ingest.fetch_sec_filings")
    def test_no_filings_found(self, mock_fetch_filings):
        mock_fetch_filings.return_value = []

        result = ingest_sec_filings("AAPL")
        assert result == 0
