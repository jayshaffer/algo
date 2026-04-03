"""Executor LLM integration -- gets structured trading decisions from Claude Haiku."""

import json
import logging
from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import Optional

from .claude_client import get_claude_client, _call_with_retry

logger = logging.getLogger(__name__)

DEFAULT_EXECUTOR_MODEL = "claude-haiku-4-5-20251001"


def _safe_int(value) -> Optional[int]:
    """Coerce a value to int, returning None if not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PlaybookAction:
    """A structured action from the playbook."""
    id: int
    ticker: str
    action: str
    thesis_id: int | None
    reasoning: str
    confidence: str
    max_quantity: Decimal | None
    priority: int


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

    def __post_init__(self):
        if self.current_prices is None:
            self.current_prices = {}
        if self.todays_decisions is None:
            self.todays_decisions = []


@dataclass
class ExecutorDecision:
    """A trading decision from the executor."""
    playbook_action_id: int | None
    ticker: str
    action: str
    quantity: float | None
    reasoning: str
    confidence: str
    is_off_playbook: bool
    signal_refs: list = None
    thesis_id: int | None = None

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
1. playbook_actions — priority-ordered actions with ticker, action, thesis_id, reasoning, confidence, max_quantity, priority
2. positions — current portfolio holdings
3. account — cash, buying_power, equity
4. attribution_summary — signal performance stats for the learning loop
5. recent_outcomes — recent decision outcomes for calibration
6. market_outlook — current market conditions summary
7. risk_notes — risk warnings and constraints
8. current_prices — latest ask prices for relevant tickers (use these for dollar-based sizing)
9. strategy_identity — the system's evolving trading identity and style (respect this)
10. strategy_rules — active constraints and preferences from past performance (MUST follow these)
11. equity_summary — recent account performance for position sizing context
12. todays_decisions — decisions already executed THIS session (avoid duplicating trades or over-deploying capital to the same ticker)

RULES:
- For each playbook action: execute, adjust, or skip (with reason)
- Set playbook_action_id to the action's id when executing a playbook action
- Set is_off_playbook to true for trades not in the playbook
- You may add trades if signals warrant, but playbook actions come first
- Use current_prices to calculate position sizes by dollar amount (e.g., to invest $500 in a $200 stock, set quantity to 2.5)
- Conservative sizing: 1-5% of buying power per trade
- Never exceed available buying power
- Fractional shares are supported -- use them to size positions precisely by dollar amount
- Example: to invest $500 in a $200 stock, use quantity 2.5
- Prefer dollar-based sizing over round share counts
- If no playbook is available: hold everything, no new positions
- If uncertain: HOLD
- Every decision MUST cite signal_refs for the learning loop

JSON SCHEMA:
{"decisions": [{"playbook_action_id": null, "ticker": "SYMBOL", "action": "buy|sell|hold", "quantity": 2.5, "reasoning": "...", "confidence": "high|medium|low", "is_off_playbook": false, "signal_refs": [{"type": "news_signal|thesis", "id": 0}], "thesis_id": null}], "thesis_invalidations": [{"thesis_id": 0, "reason": "..."}], "market_summary": "...", "risk_assessment": "..."}

If no trades: empty decisions array, explain in market_summary.
If no invalidations: empty thesis_invalidations array."""


def get_trading_decisions(
    executor_input: ExecutorInput,
    model: str = DEFAULT_EXECUTOR_MODEL,
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
    }
    input_json = json.dumps(input_data, default=str)

    cached_system = [
        {"type": "text", "text": TRADING_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]

    response = _call_with_retry(
        client,
        model=model,
        max_tokens=4096,
        system=cached_system,
        messages=[
            {
                "role": "user",
                "content": input_json,
            }
        ],
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
        raise ValueError(f"Failed to parse LLM response as JSON: {response_text}") from e

    # Build response object
    decisions = []
    for d in data.get("decisions", []):
        decisions.append(ExecutorDecision(
            playbook_action_id=d.get("playbook_action_id"),
            ticker=d.get("ticker", ""),
            action=d.get("action", "hold"),
            quantity=d.get("quantity"),
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


MAX_POSITION_PCT = Decimal("0.10")  # Max 10% of portfolio per trade


def validate_decision(
    decision: ExecutorDecision,
    buying_power: Decimal,
    current_price: Decimal,
    positions: dict[str, Decimal],
    portfolio_value: Decimal = None,
    open_sell_orders: dict[str, Decimal] = None,
) -> tuple[bool, str]:
    """
    Validate a trading decision before execution.

    Args:
        decision: The decision to validate
        buying_power: Available buying power
        current_price: Current stock price
        positions: Dict of ticker -> shares held
        portfolio_value: Total portfolio value (for position-size cap)
        open_sell_orders: Dict of ticker -> shares committed to pending sell orders

    Returns:
        Tuple of (is_valid, reason)
    """
    if decision.action == "hold":
        return True, "Hold requires no validation"

    if decision.action == "buy":
        if decision.quantity is None or decision.quantity <= 0:
            return False, "Buy requires positive quantity"

        cost = current_price * Decimal(str(decision.quantity))
        if cost > buying_power:
            return False, f"Insufficient buying power: need ${cost:.2f}, have ${buying_power:.2f}"

        if portfolio_value and portfolio_value > 0:
            existing_shares = positions.get(decision.ticker, Decimal(0))
            existing_value = existing_shares * current_price
            total_exposure = existing_value + cost
            pct = total_exposure / portfolio_value
            if pct > MAX_POSITION_PCT:
                return False, (
                    f"Total exposure ${total_exposure:.2f} ({pct:.1%} of portfolio) "
                    f"exceeds max {MAX_POSITION_PCT:.0%} "
                    f"(existing: ${existing_value:.2f} + new: ${cost:.2f})"
                )

        return True, "Buy order validated"

    if decision.action == "sell":
        if decision.quantity is None or decision.quantity <= 0:
            return False, "Sell requires positive quantity"

        held = positions.get(decision.ticker, Decimal(0))
        pending_sell = (open_sell_orders or {}).get(decision.ticker, Decimal(0))
        available = held - pending_sell
        if Decimal(str(decision.quantity)) > available:
            return False, (
                f"Insufficient available shares: want to sell {decision.quantity}, "
                f"hold {held}, pending sell {pending_sell}, available {available}"
            )

        return True, "Sell order validated"

    return False, f"Unknown action: {decision.action}"


# Valid signal types and their corresponding DB tables
_SIGNAL_TYPE_TABLES = {
    "news_signal": "news_signals",
    "macro_signal": "macro_signals",
    "thesis": "theses",
}


def validate_signal_refs(signal_refs: list[dict]) -> list[dict]:
    """Validate signal refs against the database, stripping invalid ones.

    Args:
        signal_refs: List of {"type": str, "id": int} dicts from LLM output

    Returns:
        Filtered list containing only refs that exist in the database.
    """
    if not signal_refs:
        return []

    from .database.connection import get_cursor

    valid = []
    for ref in signal_refs:
        sig_type = ref.get("type", "")
        sig_id = ref.get("id")

        table = _SIGNAL_TYPE_TABLES.get(sig_type)
        if not table:
            logger.warning("Stripping signal ref with unknown type: %s", sig_type)
            continue

        with get_cursor() as cur:
            cur.execute(f"SELECT id FROM {table} WHERE id = %s", (sig_id,))
            if cur.fetchone():
                valid.append(ref)
            else:
                logger.warning("Stripping signal ref %s:%s — not found in DB", sig_type, sig_id)

    return valid
