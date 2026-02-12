"""LLM integration for trading decisions via Claude Haiku."""

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .claude_client import get_claude_client, _call_with_retry

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """A trading decision from the LLM."""
    action: str         # buy, sell, hold
    ticker: str
    quantity: Optional[float]
    reasoning: str
    confidence: str     # high, medium, low
    thesis_id: Optional[int] = None  # If acting on a thesis
    signal_refs: list = None  # [{"type": "news_signal", "id": 15}, ...]

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
    """Full response from trading agent."""
    decisions: list[TradingDecision]
    thesis_invalidations: list[ThesisInvalidation]
    market_summary: str
    risk_assessment: str


TRADING_SYSTEM_PROMPT = """You are a trading executor. You receive a playbook and market context, and output trading decisions as JSON.

OUTPUT FORMAT: You must respond with a single JSON object and nothing else. No commentary, no markdown, no explanation outside the JSON.

INPUTS:
1. Today's Playbook — priority actions, watch list, risk notes
2. Portfolio state (positions, cash, buying power)
3. Active theses with entry/exit triggers
4. Macro context, overnight signals
5. Signal attribution scores, recent decision outcomes

RULES:
- For each playbook action: execute, adjust, or skip (with reason)
- You may add trades if overnight signals warrant, but playbook actions come first
- Conservative sizing: 1-5% of buying power per trade
- Never exceed available buying power
- Fractional shares are supported — use them to size positions precisely by dollar amount
- Example: to invest $500 in a $200 stock, use quantity 2.5
- Prefer dollar-based sizing over round share counts
- If no playbook is available: hold everything, no new positions
- If uncertain: HOLD
- Every decision MUST cite signal_refs for the learning loop

JSON SCHEMA:
{"decisions": [{"action": "buy|sell|hold", "ticker": "SYMBOL", "quantity": 2.5, "confidence": "high|medium|low", "thesis_id": null, "signal_refs": [{"type": "news_signal|thesis", "id": 0}]}], "thesis_invalidations": [{"thesis_id": 0, "reason": "..."}], "market_summary": "...", "risk_assessment": "..."}

If no trades: empty decisions array, explain in market_summary.
If no invalidations: empty thesis_invalidations array."""


DEFAULT_EXECUTOR_MODEL = "claude-haiku-4-5-20251001"


def get_trading_decisions(
    context: str,
    model: str = DEFAULT_EXECUTOR_MODEL,
) -> AgentResponse:
    """
    Get trading decisions from Claude Haiku.

    Args:
        context: Compressed trading context string
        model: Claude model to use (default: claude-haiku-4-5-20251001)

    Returns:
        AgentResponse with decisions and analysis
    """
    client = get_claude_client()

    response = _call_with_retry(
        client,
        model=model,
        max_tokens=4096,
        system=TRADING_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Here is the current market context. Analyze and provide trading decisions.\n\n{context}",
            }
        ],
    )

    # Extract text response
    response_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            response_text += block.text

    logger.info(
        "Haiku tokens — input: %d, output: %d, stop_reason: %s",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.stop_reason,
    )

    if response.stop_reason == "max_tokens":
        raise ValueError(
            f"Executor response truncated at {response.usage.output_tokens} tokens. "
            "Context may be too large — consider reducing input size."
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
        decisions.append(TradingDecision(
            action=d.get("action", "hold"),
            ticker=d.get("ticker", ""),
            quantity=d.get("quantity"),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", "low"),
            thesis_id=d.get("thesis_id"),
            signal_refs=d.get("signal_refs", []),
        ))

    thesis_invalidations = []
    for inv in data.get("thesis_invalidations", []):
        thesis_invalidations.append(ThesisInvalidation(
            thesis_id=inv["thesis_id"],
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


def validate_decision(
    decision: TradingDecision,
    buying_power: Decimal,
    current_price: Decimal,
    positions: dict[str, Decimal]
) -> tuple[bool, str]:
    """
    Validate a trading decision before execution.

    Args:
        decision: The decision to validate
        buying_power: Available buying power
        current_price: Current stock price
        positions: Dict of ticker -> shares held

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

        return True, "Buy order validated"

    if decision.action == "sell":
        if decision.quantity is None or decision.quantity <= 0:
            return False, "Sell requires positive quantity"

        held = positions.get(decision.ticker, Decimal(0))
        if Decimal(str(decision.quantity)) > held:
            return False, f"Insufficient shares: want to sell {decision.quantity}, hold {held}"

        return True, "Sell order validated"

    return False, f"Unknown action: {decision.action}"
