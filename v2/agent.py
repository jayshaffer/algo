"""Executor LLM integration -- gets structured trading decisions from Claude Haiku."""

import json
import logging
import os
from dataclasses import asdict, dataclass
from decimal import Decimal

from .claude_client import AgentPurpose, _call_with_retry, get_claude_client
from .telemetry import record_event

EXECUTOR_KNOWN_TOP_KEYS = {
    "decisions", "thesis_invalidations", "market_summary", "risk_assessment",
}
EXECUTOR_KNOWN_DECISION_KEYS = {
    "playbook_action_id", "ticker", "action", "intent_type",
    "intent_magnitude", "reasoning", "confidence",
    "is_off_playbook", "signal_refs", "thesis_id",
    # Legacy "quantity" field still appears in some test fixtures and
    # historical responses; tolerated by the parser, listed here so the
    # schema-drift canary doesn't flag it as new.
    "quantity",
}

EXECUTOR_RAW_TEXT_CAP = 4096

logger = logging.getLogger(__name__)

DEFAULT_EXECUTOR_MODEL = os.environ.get("ALGO_EXECUTOR_MODEL", "claude-haiku-4-5-20251001")
EXECUTOR_MAX_TOKENS = int(os.environ.get("ALGO_EXECUTOR_MAX_TOKENS", "8192"))


def _safe_int(value) -> int | None:
    """Coerce a value to int, returning None if not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PlaybookAction:
    """A structured action from the playbook.

    intent_type + intent_magnitude are the LLM-authored sizing signal.
    The trader resolves them to exact shares at execution time via v2.intents.

    signal_refs is the list of news/macro/thesis IDs the strategist cited as
    evidence on the underlying thesis. The executor copies it verbatim into
    its decision so attribution can trace the trade back to evidence.
    """
    id: int
    ticker: str
    action: str
    thesis_id: int | None
    reasoning: str
    confidence: str
    intent_type: str | None
    intent_magnitude: Decimal | None
    priority: int
    signal_refs: list[dict] | None = None

    def __post_init__(self):
        if self.signal_refs is None:
            self.signal_refs = []


@dataclass
class ExecutorInput:
    """Structured input for the executor."""
    playbook_actions: list[PlaybookAction]
    positions: list[dict]
    account: dict
    attribution_summary: dict
    recent_outcomes: list[dict]
    market_outlook: str
    risk_notes: str
    current_prices: dict[str, Decimal] = None
    strategy_identity: str = ""
    strategy_rules: str = ""
    equity_summary: str = ""
    todays_decisions: list[dict] = None
    recent_ticker_decisions: list[dict] = None

    def __post_init__(self):
        if self.current_prices is None:
            self.current_prices = {}
        if self.todays_decisions is None:
            self.todays_decisions = []
        if self.recent_ticker_decisions is None:
            self.recent_ticker_decisions = []


@dataclass
class ExecutorDecision:
    """A trading decision from the executor.

    intent_type + intent_magnitude are LLM-authored. quantity is set by the
    trader after resolving the intent against live portfolio state — it is
    NOT populated by the LLM.
    """
    playbook_action_id: int | None
    ticker: str
    action: str
    intent_type: str | None
    intent_magnitude: Decimal | None
    reasoning: str
    confidence: str
    is_off_playbook: bool
    signal_refs: list = None
    thesis_id: int | None = None
    quantity: Decimal | None = None  # filled in by trader.resolve

    def __post_init__(self):
        if self.signal_refs is None:
            self.signal_refs = []


@dataclass
class ThesisInvalidation:
    """A thesis flagged for invalidation."""
    thesis_id: int
    reason: str


@dataclass
class AgentResponse:
    """Full response from trading executor."""
    decisions: list[ExecutorDecision]
    thesis_invalidations: list[ThesisInvalidation]
    market_summary: str
    risk_assessment: str


TRADING_SYSTEM_PROMPT = """You are a trading executor. You receive structured JSON input containing a playbook and market context, and output trading decisions as JSON.

OUTPUT FORMAT: You must respond with a single JSON object and nothing else. No commentary, no markdown, no explanation outside the JSON.

INPUTS (as JSON object):
1. playbook_actions — priority-ordered actions with ticker, action, thesis_id, reasoning, confidence, intent_type, intent_magnitude, priority
2. positions — current portfolio holdings (authoritative share counts)
3. account — cash, buying_power, equity
4. attribution_summary — signal performance stats for the learning loop
5. recent_outcomes — recent decision outcomes for calibration
6. market_outlook — current market conditions summary
7. risk_notes — risk warnings and constraints
8. current_prices — latest ask prices for relevant tickers
9. strategy_identity — the system's evolving trading identity (respect this)
10. strategy_rules — active constraints from past performance (MUST follow)
11. equity_summary — recent account performance
12. todays_decisions — decisions already executed THIS session (don't duplicate)
13. recent_ticker_decisions — for every ticker in today's playbook, the most recent ≤5 buy/sell decisions on that ticker in the past 7 days (with full reasoning). USE THIS to detect when today's playbook reverses a recent decision.

CRITICAL RULE — YOU NEVER AUTHOR SHARE QUANTITIES:
Share counts are computed by the system from live portfolio state. You author an INTENT and a MAGNITUDE; the system divides/multiplies against the real position, current price, buying_power, and portfolio_value. This is how the system stays consistent with actual holdings even if your reasoning drifts.

SELL INTENTS:
- exit_full (magnitude: null) — sell entire position. The common case.
- exit_partial_pct (magnitude: 0-100) — sell this percentage of held shares.
- exit_dollar (magnitude: dollars) — sell roughly this dollar amount; clamped to held.
- trim_to_portfolio_pct (magnitude: 0-100) — sell until position = magnitude% of portfolio; 0 if already below.

BUY INTENTS:
- invest_dollar (magnitude: dollars) — spend this many dollars; clamped by buying_power and MAX_POSITION_PCT.
- invest_portfolio_pct (magnitude: 0-100) — spend this percentage of portfolio_value.
- invest_buying_power_pct (magnitude: 0-100) — spend this percentage of buying_power.
- add_to_target_pct (magnitude: 0-100) — add to existing position until it reaches magnitude% of portfolio; 0 if already at/above.

RULES:
- For each playbook action: execute (same intent), adjust (different intent/magnitude), or skip (hold, with reason)
- Set playbook_action_id to the action's id when executing a playbook action
- Set is_off_playbook to true for trades not in the playbook
- If no playbook available: hold everything, no new positions
- If uncertain: HOLD
- REVERSAL JUSTIFICATION: if your decision on a ticker is the opposite action of any entry in `recent_ticker_decisions` for that ticker, your `reasoning` field MUST explicitly identify (a) the prior decision (date + action), and (b) the new evidence — fundamentals shift, catalyst resolution, price level reached — that justifies reversing. "Re-narrating the same fundamentals" is not new evidence. If you can't articulate (b), HOLD instead.

SIGNAL_REFS — COPY VERBATIM, DO NOT INVENT:
- For decisions ON a playbook action: copy `signal_refs` VERBATIM from the playbook action's `signal_refs` field. Do not edit, drop, or invent IDs. The strategist already cited the evidence; you are passing it through to attribution.
- For off-playbook trades: use `signal_refs: []` (empty). You do not have the structured IDs needed to cite, and inventing them would be silently stripped. Off-playbook trades therefore have no attribution — keep them rare.
- For HOLD decisions: `signal_refs: []` is fine.

JSON SCHEMA:
{"decisions": [{"playbook_action_id": null, "ticker": "SYMBOL", "action": "buy|sell|hold", "intent_type": "exit_full|exit_partial_pct|exit_dollar|trim_to_portfolio_pct|invest_dollar|invest_portfolio_pct|invest_buying_power_pct|add_to_target_pct|null", "intent_magnitude": 500.0, "reasoning": "...", "confidence": "high|medium|low", "is_off_playbook": false, "signal_refs": [{"type": "news_signal|macro_signal|thesis", "id": 1234}], "thesis_id": null}], "thesis_invalidations": [{"thesis_id": null, "reason": "..."}], "market_summary": "...", "risk_assessment": "..."}

For hold decisions, set intent_type and intent_magnitude to null.
For exit_full, set intent_magnitude to null.
If no trades: empty decisions array, explain in market_summary."""


def get_trading_decisions(
    executor_input: ExecutorInput,
    model: str = DEFAULT_EXECUTOR_MODEL,
    session_id: int | None = None,
) -> AgentResponse:
    """
    Get trading decisions from Claude Haiku.

    Args:
        executor_input: Structured executor input with playbook, positions, account, etc.
        model: Claude model to use (default: claude-haiku-4-5-20251001)

    Returns:
        AgentResponse with decisions and analysis
    """
    client = get_claude_client()

    # Serialize ExecutorInput as JSON
    input_data = {
        "playbook_actions": [asdict(a) for a in executor_input.playbook_actions],
        "positions": executor_input.positions,
        "account": executor_input.account,
        "attribution_summary": executor_input.attribution_summary,
        "recent_outcomes": executor_input.recent_outcomes,
        "market_outlook": executor_input.market_outlook,
        "risk_notes": executor_input.risk_notes,
        "current_prices": {k: str(v) for k, v in executor_input.current_prices.items()},
        "strategy_identity": executor_input.strategy_identity,
        "strategy_rules": executor_input.strategy_rules,
        "equity_summary": executor_input.equity_summary,
        "todays_decisions": executor_input.todays_decisions,
        "recent_ticker_decisions": executor_input.recent_ticker_decisions,
    }
    input_json = json.dumps(input_data, default=str)

    response = _call_with_retry(
        client,
        model=model,
        max_tokens=EXECUTOR_MAX_TOKENS,
        system=TRADING_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": input_json,
            }
        ],
        session_id=session_id,
        stage_name="trading",
        purpose=AgentPurpose.EXECUTOR,
    )

    # Extract text response
    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text

    logger.info(
        "Haiku tokens -- input: %d, output: %d, stop_reason: %s",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.stop_reason,
    )

    if response.stop_reason == "max_tokens":
        record_event(
            session_id=session_id,
            stage_name="trading",
            event_type="executor_response",
            payload={
                "parse_succeeded": False,
                "stop_reason": response.stop_reason,
                "decision_count": 0,
                "thesis_invalidation_count": 0,
                "unknown_top_level_keys": [],
                "unknown_decision_keys": [],
                "raw_response_text_truncated": response_text[:EXECUTOR_RAW_TEXT_CAP],
                "error": "max_tokens_truncation",
            },
        )
        raise ValueError(
            f"Executor response truncated at {response.usage.output_tokens} tokens. "
            "Context may be too large -- consider reducing input size."
        )

    # Parse JSON response
    try:
        # Handle potential markdown code blocks
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
    except json.JSONDecodeError as e:
        record_event(
            session_id=session_id,
            stage_name="trading",
            event_type="executor_response",
            payload={
                "parse_succeeded": False,
                "stop_reason": response.stop_reason,
                "decision_count": 0,
                "thesis_invalidation_count": 0,
                "unknown_top_level_keys": [],
                "unknown_decision_keys": [],
                "raw_response_text_truncated": response_text[:EXECUTOR_RAW_TEXT_CAP],
                "error": f"JSONDecodeError: {e}",
            },
        )
        raise ValueError(f"Failed to parse LLM response as JSON: {response_text}") from e

    # Build response object
    decisions = []
    for d in data.get("decisions", []):
        raw_mag = d.get("intent_magnitude")
        magnitude = Decimal(str(raw_mag)) if raw_mag is not None else None
        # T1.3: normalize at the parse boundary so every downstream consumer
        # (sector lookup, position dict keys, sell precheck) sees the same
        # canonical TICKER. LLM has been observed emitting "aapl" or " AAPL ".
        ticker = (d.get("ticker") or "").strip().upper()
        decisions.append(ExecutorDecision(
            playbook_action_id=d.get("playbook_action_id"),
            ticker=ticker,
            action=d.get("action", "hold"),
            intent_type=d.get("intent_type"),
            intent_magnitude=magnitude,
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", "low"),
            is_off_playbook=d.get("is_off_playbook", False),
            signal_refs=d.get("signal_refs", []),
            thesis_id=_safe_int(d.get("thesis_id")),
        ))

    thesis_invalidations = []
    for inv in data.get("thesis_invalidations", []):
        raw_id = inv.get("thesis_id")
        try:
            tid = int(raw_id)
        except (TypeError, ValueError):
            logger.warning("Skipping thesis invalidation with non-integer thesis_id: %s", raw_id)
            continue
        thesis_invalidations.append(ThesisInvalidation(
            thesis_id=tid,
            reason=inv.get("reason", ""),
        ))

    unknown_top = sorted(set(data.keys()) - EXECUTOR_KNOWN_TOP_KEYS)
    unknown_dec = sorted({
        k for d in data.get("decisions", []) if isinstance(d, dict)
        for k in d.keys()
    } - EXECUTOR_KNOWN_DECISION_KEYS)
    record_event(
        session_id=session_id,
        stage_name="trading",
        event_type="executor_response",
        payload={
            "parse_succeeded": True,
            "stop_reason": response.stop_reason,
            "decision_count": len(decisions),
            "thesis_invalidation_count": len(thesis_invalidations),
            "unknown_top_level_keys": unknown_top,
            "unknown_decision_keys": unknown_dec,
            "raw_response_text_truncated": response_text[:EXECUTOR_RAW_TEXT_CAP],
            "error": None,
        },
    )

    return AgentResponse(
        decisions=decisions,
        thesis_invalidations=thesis_invalidations,
        market_summary=data.get("market_summary", ""),
        risk_assessment=data.get("risk_assessment", ""),
    )


def format_decisions_for_logging(response: AgentResponse) -> dict:
    """
    Format agent response for database logging.

    Returns dict suitable for signals_used JSONB column.
    """
    return {
        "market_summary": response.market_summary,
        "risk_assessment": response.risk_assessment,
        "decision_count": len(response.decisions),
        "thesis_invalidation_count": len(response.thesis_invalidations),
    }


# Valid signal types and their corresponding DB tables
_SIGNAL_TYPE_TABLES = {
    "news_signal": "news_signals",
    "macro_signal": "macro_signals",
    "thesis": "theses",
}


def validate_signal_refs(signal_refs: list[dict]) -> list[dict]:
    """Validate signal refs against the database, stripping invalid ones.

    Batches queries by signal type (one query per type) to avoid N+1 behaviour.

    Args:
        signal_refs: List of {"type": str, "id": int} dicts from LLM output

    Returns:
        Filtered list containing only refs that exist in the database.
    """
    if not signal_refs:
        return []

    from .database.connection import get_cursor

    # Group refs by signal type, dropping any with unknown types up-front
    by_type: dict[str, list[dict]] = {}
    for ref in signal_refs:
        sig_type = ref.get("type", "")
        if not _SIGNAL_TYPE_TABLES.get(sig_type):
            logger.warning("Stripping signal ref with unknown type: %s", sig_type)
            continue
        by_type.setdefault(sig_type, []).append(ref)

    valid = []
    for sig_type, refs in by_type.items():
        table = _SIGNAL_TYPE_TABLES[sig_type]
        # Defense-in-depth: f-string interpolation is safe because `table` is
        # sourced exclusively from the hardcoded `_SIGNAL_TYPE_TABLES` allowlist
        # (the loop above drops any sig_type not present). Assert the invariant
        # here so a future refactor that changes how `table` is populated trips
        # immediately rather than silently introducing an injection vector.
        assert table in _SIGNAL_TYPE_TABLES.values(), f"unexpected table: {table!r}"
        ids = [r["id"] for r in refs if r.get("id") is not None]
        if not ids:
            continue
        with get_cursor() as cur:
            cur.execute(f"SELECT id FROM {table} WHERE id = ANY(%s)", (ids,))  # noqa: S608
            found_ids = {row["id"] for row in cur.fetchall()}
        for ref in refs:
            if ref.get("id") in found_ids:
                valid.append(ref)
            else:
                logger.warning(
                    "Stripping signal ref %s:%s — not found in DB", sig_type, ref.get("id")
                )

    return valid
