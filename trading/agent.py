"""Claude integration for trading decisions."""

import os
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import anthropic


@dataclass
class TradingDecision:
    """A trading decision from Claude."""
    action: str         # buy, sell, hold
    ticker: str
    quantity: Optional[int]
    reasoning: str
    confidence: str     # high, medium, low


@dataclass
class AgentResponse:
    """Full response from trading agent."""
    decisions: list[TradingDecision]
    market_summary: str
    risk_assessment: str


TRADING_SYSTEM_PROMPT = """You are a trading agent for an automated trading system. Your job is to analyze market signals and make trading decisions.

You will receive:
1. Current portfolio state (positions, cash, buying power)
2. Macro economic context (Fed policy, trade news, etc.)
3. Ticker-specific signals (earnings, analyst ratings, etc.)
4. 7-day signal trends
5. Recent decision outcomes (to learn from)
6. Current strategy guidelines

Based on this context, decide whether to BUY, SELL, or HOLD for each relevant ticker.

Rules:
- Be conservative with position sizing (suggest 1-5% of buying power per trade)
- Provide clear reasoning for each decision
- Consider both signals AND macro context
- Learn from recent decision outcomes
- Stay within the current strategy's risk tolerance
- If uncertain, recommend HOLD
- Never suggest using more than available buying power

Respond with valid JSON only in this format:
{
    "decisions": [
        {
            "action": "buy" | "sell" | "hold",
            "ticker": "AAPL",
            "quantity": 10,
            "reasoning": "Strong earnings beat with positive analyst sentiment...",
            "confidence": "high" | "medium" | "low"
        }
    ],
    "market_summary": "Brief summary of current market conditions...",
    "risk_assessment": "Current risk level and concerns..."
}

If no action is warranted, return an empty decisions array with explanation in market_summary."""


def get_anthropic_client() -> anthropic.Anthropic:
    """Create Anthropic client from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set")
    return anthropic.Anthropic(api_key=api_key)


def get_trading_decisions(
    context: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2000
) -> AgentResponse:
    """
    Get trading decisions from Claude based on market context.

    Args:
        context: Compressed trading context string
        model: Claude model to use
        max_tokens: Max response tokens

    Returns:
        AgentResponse with decisions and analysis
    """
    client = get_anthropic_client()

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=TRADING_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Here is the current market context. Analyze and provide trading decisions.\n\n{context}"
            }
        ]
    )

    # Extract text response
    response_text = message.content[0].text

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
        raise ValueError(f"Failed to parse Claude response as JSON: {response_text}") from e

    # Build response object
    decisions = []
    for d in data.get("decisions", []):
        decisions.append(TradingDecision(
            action=d.get("action", "hold"),
            ticker=d.get("ticker", ""),
            quantity=d.get("quantity"),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", "low"),
        ))

    return AgentResponse(
        decisions=decisions,
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
