"""Local LLM integration for trading decisions via Ollama."""

import os
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import ollama


@dataclass
class TradingDecision:
    """A trading decision from the LLM."""
    action: str         # buy, sell, hold
    ticker: str
    quantity: Optional[int]
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


TRADING_SYSTEM_PROMPT = """You are executing a trading playbook prepared by a senior strategist. Your job is to translate the playbook into concrete trading decisions.

You will receive:
1. Today's Playbook — priority actions, watch list, risk notes from the strategist
2. Current portfolio state (positions, cash, buying power)
3. Active theses with entry/exit triggers
4. Macro economic context
5. Overnight ticker signals
6. Signal attribution scores (which signal types have been historically predictive)
7. Recent decision outcomes

For each priority action in the playbook, decide: execute as-is, adjust quantity/timing, or skip (with reason).

You may propose additional trades if overnight signals warrant it, but playbook actions come first.

Rules:
- Be conservative with position sizing (suggest 1-5% of buying power per trade)
- Provide clear reasoning for each decision
- Weight playbook actions heavily — they represent pre-analyzed opportunities
- Consider signal attribution scores — give more weight to historically predictive signal types
- If no playbook is available, operate in conservative mode: hold everything, no new positions
- Never suggest using more than available buying power
- If uncertain, recommend HOLD

CRITICAL — Signal Citation:
For EVERY decision, you MUST cite which signal IDs and/or thesis IDs informed it in the signal_refs field. This is required for the learning loop. Use the format:
  [{"type": "news_signal", "id": <id>}, {"type": "thesis", "id": <id>}, ...]

Respond with valid JSON only:
{
    "decisions": [
        {
            "action": "buy" | "sell" | "hold",
            "ticker": "NVDA",
            "quantity": 5,
            "reasoning": "Playbook priority action — entry trigger hit...",
            "confidence": "high" | "medium" | "low",
            "thesis_id": 42,
            "signal_refs": [{"type": "thesis", "id": 42}, {"type": "news_signal", "id": 15}]
        }
    ],
    "thesis_invalidations": [
        {
            "thesis_id": 123,
            "reason": "Observed condition matching invalidation criteria..."
        }
    ],
    "market_summary": "Brief summary...",
    "risk_assessment": "Current risk level..."
}

If no action is warranted, return an empty decisions array with explanation in market_summary.
If no theses need invalidation, return an empty thesis_invalidations array."""


def get_ollama_client() -> ollama.Client:
    """Create Ollama client from environment."""
    host = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    return ollama.Client(host=host)


def get_trading_decisions(
    context: str,
    model: str = "qwen2.5:14b",
) -> AgentResponse:
    """
    Get trading decisions from local LLM via Ollama.

    Args:
        context: Compressed trading context string
        model: Ollama model to use (default: qwen2.5:14b)

    Returns:
        AgentResponse with decisions and analysis
    """
    client = get_ollama_client()

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": TRADING_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Here is the current market context. Analyze and provide trading decisions.\n\n{context}"
            }
        ],
        options={
            "temperature": 0.3,  # Lower temperature for more consistent JSON
            "num_predict": 2000,
        },
    )

    # Extract text response
    response_text = response["message"]["content"]

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

        cost = current_price * decision.quantity
        if cost > buying_power:
            return False, f"Insufficient buying power: need ${cost:.2f}, have ${buying_power:.2f}"

        return True, "Buy order validated"

    if decision.action == "sell":
        if decision.quantity is None or decision.quantity <= 0:
            return False, "Sell requires positive quantity"

        held = positions.get(decision.ticker, Decimal(0))
        if decision.quantity > held:
            return False, f"Insufficient shares: want to sell {decision.quantity}, hold {held}"

        return True, "Sell order validated"

    return False, f"Unknown action: {decision.action}"
