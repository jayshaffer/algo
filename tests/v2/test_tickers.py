"""Tests for v2.tickers.resolve_ticker — the shared ticker resolution layer
used by both the classifier (news ingestion) and tools (strategist/executor
tool calls)."""

from v2.tickers import resolve_ticker


class TestResolveTicker:
    def test_passthrough_canonical(self):
        assert resolve_ticker("AAPL") == "AAPL"
        assert resolve_ticker("MSFT") == "MSFT"

    def test_uppercases(self):
        assert resolve_ticker("aapl") == "AAPL"
        assert resolve_ticker("AaPl") == "AAPL"

    def test_strips_whitespace(self):
        assert resolve_ticker(" AAPL ") == "AAPL"
        assert resolve_ticker("\tAAPL\n") == "AAPL"

    def test_strips_dollar_prefix(self):
        assert resolve_ticker("$TSLA") == "TSLA"

    def test_strips_trailing_sentence_punctuation(self):
        for raw, expected in [
            ("AAPL.", "AAPL"),
            ("NVDA,", "NVDA"),
            ("MSFT;", "MSFT"),
            ("TSLA:", "TSLA"),
            ("AMD!", "AMD"),
            ("GOOG?", "GOOG"),
        ]:
            assert resolve_ticker(raw) == expected

    def test_preserves_class_share_dot(self):
        assert resolve_ticker("BRK.B") == "BRK.B"
        assert resolve_ticker("BRK.A") == "BRK.A"
        # Trailing punctuation still stripped without damaging the class dot
        assert resolve_ticker("BRK.A.") == "BRK.A"
        assert resolve_ticker("BRK.B,") == "BRK.B"

    def test_remaps_company_name_to_ticker(self):
        """CARLSMED (Carlsmed Inc, 2025 IPO) is a company name the LLM
        emits because its training data predates the IPO. Map to CARL."""
        assert resolve_ticker("CARLSMED") == "CARL"
        assert resolve_ticker("carlsmed") == "CARL"
        assert resolve_ticker(" $CARLSMED ") == "CARL"

    def test_drops_group_acronyms(self):
        """FAANG/MAANG/MAGS-style group names are never tradable."""
        for raw in ["FAANG", "MAANG", "MAG", "FAAMG", "BATX", "BAT"]:
            assert resolve_ticker(raw) is None, f"{raw!r} should drop"

    def test_drops_generic_acronyms(self):
        """CEO/GDP/ETF prose acronyms aren't intended as tickers."""
        for raw in ["CEO", "CFO", "GDP", "CPI", "ETF", "SPAC", "USD", "EUR"]:
            assert resolve_ticker(raw) is None, f"{raw!r} should drop"

    def test_does_not_drop_dei_or_esg(self):
        """DEI (Douglas Emmett) and ESG (FlexShares ESG ETF) are real
        Alpaca-tradable tickers. Prior blocklist over-blocked them — confirmed
        2 legit Douglas Emmett analyst notes in prod news_signals."""
        assert resolve_ticker("DEI") == "DEI"
        assert resolve_ticker("ESG") == "ESG"

    def test_drops_overlong_alphanumeric(self):
        """Anything past the 5-char limit is almost certainly a hallucination."""
        assert resolve_ticker("ABCDEFG") is None
        assert resolve_ticker("TEST_TICKER") is None
        assert resolve_ticker("$$$$") is None

    def test_remap_target_passes_shape_check(self):
        """The alias remap fires before the shape check, so an 8-char key
        like CARLSMED is allowed as long as the value (CARL) is well-shaped."""
        assert resolve_ticker("CARLSMED") == "CARL"

    def test_none_passthrough(self):
        """Optional-filter callers pass None to mean 'no filter' — preserve."""
        assert resolve_ticker(None) is None

    def test_non_string_returns_none(self):
        assert resolve_ticker(123) is None
        assert resolve_ticker([]) is None
        assert resolve_ticker({}) is None

    def test_empty_string_returns_none(self):
        assert resolve_ticker("") is None
        assert resolve_ticker("   ") is None
        assert resolve_ticker("$") is None
