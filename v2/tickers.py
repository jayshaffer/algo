"""Ticker resolution: validate, remap, and normalize LLM-emitted ticker strings.

Single source of truth used by both the classifier (news ingestion) and tools
(strategist/executor tool calls). Behavior:

  1. Strip `$`-prefix, sentence punctuation, whitespace; uppercase.
  2. Apply alias remap (e.g. CARLSMED -> CARL when the LLM emits a recent-IPO
     company name instead of its post-IPO ticker).
  3. Validate shape against ^[A-Z]{1,5}(\\.[A-Z])?$ — allows class shares
     like BRK.B, rejects long alphanumeric soup.
  4. Drop if the cleaned symbol is on the drop list (group acronyms like
     FAANG, economic indicators like GDP). Returns None.

Config lives at v2/config/ticker_aliases.json so the alias/drop sets can grow
without touching code.
"""
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
_CONFIG_PATH = Path(__file__).parent / "config" / "ticker_aliases.json"


def _load_config() -> tuple[dict[str, str], frozenset[str]]:
    with _CONFIG_PATH.open() as f:
        data = json.load(f)
    aliases = {k.upper(): v.upper() for k, v in data.get("aliases", {}).items()}
    drop = frozenset(s.upper() for s in data.get("drop", []))
    return aliases, drop


_ALIASES, _DROP = _load_config()


def resolve_ticker(raw: str | None) -> str | None:
    """Normalize, remap, and validate an LLM-emitted ticker string.

    Returns the canonical ticker, or None if the input is not a plausible
    equity ticker or is on the drop list. None passes through as None so
    optional-filter callers (e.g. tool_get_active_theses(ticker=None)) keep
    their "no filter" semantics.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().lstrip("$").rstrip(".,;:!?").upper()
    if not cleaned:
        return None
    # Alias remap before shape check — alias keys may be longer than 5 chars
    # (e.g. CARLSMED -> CARL when the LLM emits the company name).
    if cleaned in _ALIASES:
        remapped = _ALIASES[cleaned]
        logger.info("tickers: remapped %r -> %r", raw, remapped)
        cleaned = remapped
    if not _TICKER_RE.match(cleaned):
        logger.warning("tickers: rejecting non-ticker-shaped string %r", raw)
        return None
    if cleaned in _DROP:
        logger.warning("tickers: rejecting hallucinated/acronym ticker %r", raw)
        return None
    return cleaned
