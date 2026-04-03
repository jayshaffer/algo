# V2 Risk Management & Learning Loop Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 15 identified flaws in the v2 trading system covering order lifecycle safety, position risk controls, attribution accuracy, and structural gaps.

**Architecture:** Changes are organized into 4 independent workstreams: (A) order lifecycle safety, (B) position risk controls, (C) attribution/learning loop accuracy, (D) structural/session integrity. Each workstream can be implemented and tested independently. All changes follow the existing patterns: raw SQL via `get_cursor()`, `Decimal` for money, mocked DB in tests.

**Tech Stack:** Python 3.12, psycopg2 (raw SQL), Alpaca SDK, pytest with unittest.mock

**Codebase conventions:**
- All DB access via `with get_cursor() as cur:` context manager
- Test fixtures in `tests/v2/conftest.py` — factory functions like `make_position_row()`, `make_decision_row()`
- Mock DB with `mock_db` fixture (patches `get_cursor` across all modules)
- `Decimal` for all monetary values, never `float`
- Run tests: `python3 -m pytest tests/v2/ -v`
- Run specific test: `python3 -m pytest tests/v2/test_file.py::TestClass::test_name -v`

---

## Workstream A: Order Lifecycle Safety

### Task 1: Cancel orders on fill timeout

The `wait_for_fill()` function times out after 30s but leaves the order live on Alpaca. It could fill later at a bad price with no tracking.

**Files:**
- Modify: `v2/executor.py:255-297` (add cancellation on timeout)
- Test: `tests/v2/test_executor.py`

- [ ] **Step 1: Write failing test for timeout cancellation**

In `tests/v2/test_executor.py`, add:

```python
class TestWaitForFillCancellation:
    """Verify that timed-out orders get cancelled."""

    @patch("v2.executor.get_trading_client")
    @patch("v2.executor.time")
    def test_timeout_cancels_order(self, mock_time, mock_get_client):
        """On timeout, wait_for_fill should attempt to cancel the order."""
        from v2.executor import wait_for_fill

        client = MagicMock()
        mock_get_client.return_value = client

        # Simulate time passing beyond timeout
        mock_time.monotonic.side_effect = [0, 0, 100]  # start, first check, expired
        mock_time.sleep = MagicMock()

        order = MagicMock()
        order.status.value = "accepted"
        client.get_order_by_id.return_value = order

        result = wait_for_fill("order-123", timeout_seconds=30)

        assert not result.success
        assert "Timeout" in result.error
        client.cancel_order_by_id.assert_called_once_with("order-123")

    @patch("v2.executor.get_trading_client")
    @patch("v2.executor.time")
    def test_timeout_cancel_failure_still_returns_timeout(self, mock_time, mock_get_client):
        """If cancellation fails, still return timeout error."""
        from v2.executor import wait_for_fill

        client = MagicMock()
        mock_get_client.return_value = client

        mock_time.monotonic.side_effect = [0, 0, 100]
        mock_time.sleep = MagicMock()

        order = MagicMock()
        order.status.value = "accepted"
        client.get_order_by_id.return_value = order
        client.cancel_order_by_id.side_effect = Exception("Already filled")

        result = wait_for_fill("order-123", timeout_seconds=30)

        assert not result.success
        assert "Timeout" in result.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_executor.py::TestWaitForFillCancellation -v`
Expected: FAIL — `cancel_order_by_id` never called

- [ ] **Step 3: Implement cancellation on timeout**

In `v2/executor.py`, replace the timeout return block at the end of `wait_for_fill()`:

```python
    # Timeout — attempt to cancel the orphaned order
    try:
        client.cancel_order_by_id(order_id)
        logger.warning("Order %s timed out after %.0fs — cancelled", order_id, timeout_seconds)
    except Exception as cancel_err:
        logger.error("Order %s timed out and cancel failed: %s — order may still be live", order_id, cancel_err)

    return OrderResult(
        success=False,
        order_id=order_id,
        filled_qty=None,
        filled_avg_price=None,
        error=f"Timeout waiting for fill after {timeout_seconds}s (cancel attempted)",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_executor.py::TestWaitForFillCancellation -v`
Expected: PASS

- [ ] **Step 5: Run full executor test suite for regressions**

Run: `python3 -m pytest tests/v2/test_executor.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add v2/executor.py tests/v2/test_executor.py
git commit -m "fix(v2): cancel orders on fill timeout to prevent ghost fills"
```

---

### Task 2: Reuse API clients instead of creating per call

`get_latest_price()` creates a new `StockHistoricalDataClient` on every call, adding latency and rate-limit risk.

**Files:**
- Modify: `v2/executor.py:300-347` (accept optional client param)
- Test: `tests/v2/test_executor.py`

- [ ] **Step 1: Write failing test for client reuse**

```python
class TestGetLatestPriceClientReuse:
    """Verify get_latest_price can accept an external client."""

    @patch("v2.executor.StockHistoricalDataClient")
    def test_uses_provided_client(self, mock_client_cls):
        """When a client is passed, don't create a new one."""
        from v2.executor import get_latest_price

        external_client = MagicMock()
        quote = MagicMock()
        quote.ask_price = 150.0
        quote.bid_price = 149.5
        quote.timestamp = None
        external_client.get_stock_latest_quote.return_value = {"AAPL": quote}

        result = get_latest_price("AAPL", client=external_client)

        assert result == Decimal("150.0")
        mock_client_cls.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_executor.py::TestGetLatestPriceClientReuse -v`
Expected: FAIL — `get_latest_price` doesn't accept `client` param

- [ ] **Step 3: Add optional client parameter**

In `v2/executor.py`, modify `get_latest_price` signature and body:

```python
def get_latest_price(
    ticker: str,
    max_age_seconds: int = 60,
    max_spread_pct: Decimal = Decimal("0.05"),
    client: StockHistoricalDataClient = None,
) -> Optional[Decimal]:
```

Replace the client creation block:

```python
    if client is None:
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret_key)

    request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
```

- [ ] **Step 4: Update trader.py to reuse a single data client**

In `v2/trader.py`, create the client once before the execution loop and pass it through. At the top of `run_trading_session`, after the market hours check:

```python
    # Create a shared data client for price lookups
    from alpaca.data.historical import StockHistoricalDataClient
    import os
    data_client = StockHistoricalDataClient(
        os.environ.get("ALPACA_API_KEY"),
        os.environ.get("ALPACA_SECRET_KEY"),
    )
```

Then update all `get_latest_price(ticker)` calls in the function to `get_latest_price(ticker, client=data_client)`.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/v2/test_executor.py tests/v2/test_trader.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add v2/executor.py v2/trader.py tests/v2/test_executor.py
git commit -m "perf(v2): reuse data client for price lookups instead of creating per call"
```

---

## Workstream B: Position Risk Controls

### Task 3: Validate total exposure per ticker, not just new buy size

The 10% position cap only checks the new buy cost, ignoring existing holdings. A 5% buy on top of an 8% existing position passes validation.

**Files:**
- Modify: `v2/agent.py:252-303` (`validate_decision`)
- Test: `tests/v2/test_agent.py`

- [ ] **Step 1: Write failing tests for total exposure check**

```python
class TestValidateDecisionTotalExposure:
    """Position size cap should include existing holdings."""

    def test_buy_rejected_when_total_exposure_exceeds_cap(self):
        from v2.agent import validate_decision
        decision = make_trading_decision(
            ticker="AAPL", action="buy", quantity=3.0
        )
        # Already hold 8% worth of AAPL. Buying 3 shares @ $150 = $450 = ~4.5%.
        # Total would be ~12.5%, exceeding 10% cap.
        positions = {"AAPL": Decimal("5.33")}  # 5.33 * 150 = ~$800 = 8%
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("5000"),
            current_price=Decimal("150"),
            positions=positions,
            portfolio_value=Decimal("10000"),
        )
        assert not is_valid
        assert "total exposure" in reason.lower() or "position" in reason.lower()

    def test_buy_allowed_when_total_exposure_under_cap(self):
        from v2.agent import validate_decision
        decision = make_trading_decision(
            ticker="AAPL", action="buy", quantity=1.0
        )
        # Hold 5% worth. Buying 1 share @ $150 = $150 = 1.5%. Total = 6.5% < 10%.
        positions = {"AAPL": Decimal("3.33")}  # 3.33 * 150 = ~$500 = 5%
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("5000"),
            current_price=Decimal("150"),
            positions=positions,
            portfolio_value=Decimal("10000"),
        )
        assert is_valid

    def test_buy_new_ticker_still_uses_new_cost_only(self):
        from v2.agent import validate_decision
        decision = make_trading_decision(
            ticker="MSFT", action="buy", quantity=2.0
        )
        positions = {"AAPL": Decimal("10")}  # No MSFT held
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("5000"),
            current_price=Decimal("200"),
            positions=positions,
            portfolio_value=Decimal("10000"),
        )
        # $400 / $10000 = 4% < 10%
        assert is_valid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_agent.py::TestValidateDecisionTotalExposure -v`
Expected: First test FAILS (buy approved when it shouldn't be)

- [ ] **Step 3: Implement total exposure check**

In `v2/agent.py`, in the `validate_decision` function, replace the position size check block inside the `if decision.action == "buy":` branch:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_agent.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v2/agent.py tests/v2/test_agent.py
git commit -m "fix(v2): validate total position exposure including existing holdings"
```

---

### Task 4: Sell validation accounts for pending sell orders

`validate_decision` checks held shares but doesn't subtract shares already committed to open sell orders, allowing double-sells.

**Files:**
- Modify: `v2/agent.py:293-301` (sell validation)
- Modify: `v2/trader.py:190` (pass open orders to validation)
- Test: `tests/v2/test_agent.py`

- [ ] **Step 1: Write failing tests**

```python
class TestValidateDecisionPendingSells:
    """Sell validation should subtract shares in pending sell orders."""

    def test_sell_rejected_when_pending_orders_consume_shares(self):
        from v2.agent import validate_decision
        decision = make_trading_decision(
            ticker="AAPL", action="sell", quantity=8.0
        )
        # Hold 10 shares, but 5 are already in a pending sell order
        positions = {"AAPL": Decimal("10")}
        open_sell_orders = {"AAPL": Decimal("5")}
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("5000"),
            current_price=Decimal("150"),
            positions=positions,
            portfolio_value=Decimal("10000"),
            open_sell_orders=open_sell_orders,
        )
        assert not is_valid
        assert "pending" in reason.lower() or "available" in reason.lower()

    def test_sell_allowed_after_accounting_for_pending(self):
        from v2.agent import validate_decision
        decision = make_trading_decision(
            ticker="AAPL", action="sell", quantity=4.0
        )
        positions = {"AAPL": Decimal("10")}
        open_sell_orders = {"AAPL": Decimal("5")}
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("5000"),
            current_price=Decimal("150"),
            positions=positions,
            portfolio_value=Decimal("10000"),
            open_sell_orders=open_sell_orders,
        )
        assert is_valid

    def test_sell_works_with_no_pending_orders(self):
        """Backwards compatible — no open_sell_orders param defaults to empty."""
        from v2.agent import validate_decision
        decision = make_trading_decision(
            ticker="AAPL", action="sell", quantity=5.0
        )
        positions = {"AAPL": Decimal("10")}
        is_valid, reason = validate_decision(
            decision,
            buying_power=Decimal("5000"),
            current_price=Decimal("150"),
            positions=positions,
            portfolio_value=Decimal("10000"),
        )
        assert is_valid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_agent.py::TestValidateDecisionPendingSells -v`
Expected: FAIL — `validate_decision` doesn't accept `open_sell_orders`

- [ ] **Step 3: Add open_sell_orders parameter to validate_decision**

In `v2/agent.py`, update the `validate_decision` signature:

```python
def validate_decision(
    decision: ExecutorDecision,
    buying_power: Decimal,
    current_price: Decimal,
    positions: dict[str, Decimal],
    portfolio_value: Decimal = None,
    open_sell_orders: dict[str, Decimal] = None,
) -> tuple[bool, str]:
```

Update the sell validation block:

```python
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
```

- [ ] **Step 4: Update trader.py to build and pass open_sell_orders**

In `v2/trader.py`, after `positions = {p["ticker"]: p["shares"] for p in get_positions()}` (line 190), add:

```python
    # Build pending sell orders map for validation
    from .database.trading_db import get_open_orders
    open_orders = get_open_orders()
    open_sell_orders = {}
    for order in open_orders:
        if order["side"] == "sell" and order["status"] in ("new", "accepted", "partially_filled"):
            ticker = order["ticker"]
            remaining = order["qty"] - (order.get("filled_qty") or Decimal(0))
            open_sell_orders[ticker] = open_sell_orders.get(ticker, Decimal(0)) + remaining
```

Then update the `validate_decision` call to include `open_sell_orders=open_sell_orders`.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/v2/test_agent.py tests/v2/test_trader.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add v2/agent.py v2/trader.py tests/v2/test_agent.py
git commit -m "fix(v2): sell validation subtracts shares in pending sell orders"
```

---

### Task 5: Add portfolio-level sector concentration limit

The system can build highly correlated positions (e.g., 5 tech stocks at 8% each = 40% tech). Add a sector concentration check.

**Files:**
- Create: `v2/risk.py`
- Modify: `v2/trader.py` (call risk check before execution loop)
- Create: `tests/v2/test_risk.py`

- [ ] **Step 1: Write failing tests**

Create `tests/v2/test_risk.py`:

```python
"""Tests for portfolio risk checks."""

from decimal import Decimal
import pytest
from v2.risk import check_sector_concentration, SECTOR_MAP, MAX_SECTOR_PCT


class TestSectorConcentration:
    def test_flags_sector_over_limit(self):
        """Concentrated tech portfolio should be flagged."""
        positions = {
            "AAPL": Decimal("2000"),
            "MSFT": Decimal("2000"),
            "GOOGL": Decimal("2000"),
        }
        warnings = check_sector_concentration(positions, portfolio_value=Decimal("10000"))
        assert any("tech" in w.lower() for w in warnings)

    def test_passes_diversified_portfolio(self):
        positions = {
            "AAPL": Decimal("500"),
            "JPM": Decimal("500"),
            "XOM": Decimal("500"),
        }
        warnings = check_sector_concentration(positions, portfolio_value=Decimal("10000"))
        assert len(warnings) == 0

    def test_empty_portfolio_no_warnings(self):
        warnings = check_sector_concentration({}, portfolio_value=Decimal("10000"))
        assert len(warnings) == 0

    def test_unknown_ticker_classified_as_other(self):
        positions = {"ZZZZZ": Decimal("5000")}
        warnings = check_sector_concentration(positions, portfolio_value=Decimal("10000"))
        # Unknown tickers go to "other" — shouldn't crash
        assert isinstance(warnings, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_risk.py -v`
Expected: FAIL — `v2.risk` module doesn't exist

- [ ] **Step 3: Implement risk.py**

Create `v2/risk.py`:

```python
"""Portfolio-level risk checks."""

from decimal import Decimal

# Approximate sector mapping for common tickers.
# This is intentionally simple — a full mapping would come from a data provider.
SECTOR_MAP = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "GOOG": "tech",
    "AMZN": "tech", "META": "tech", "NVDA": "tech", "TSM": "tech",
    "AVGO": "tech", "ORCL": "tech", "CRM": "tech", "AMD": "tech",
    "INTC": "tech", "ADBE": "tech", "CSCO": "tech", "QCOM": "tech",
    # Finance
    "JPM": "finance", "BAC": "finance", "WFC": "finance", "GS": "finance",
    "MS": "finance", "C": "finance", "BLK": "finance", "SCHW": "finance",
    "V": "finance", "MA": "finance", "AXP": "finance",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "EOG": "energy", "OXY": "energy",
    # Healthcare
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "LLY": "healthcare",
    "TMO": "healthcare",
    # Consumer
    "WMT": "consumer", "PG": "consumer", "KO": "consumer", "PEP": "consumer",
    "COST": "consumer", "NKE": "consumer", "SBUX": "consumer", "MCD": "consumer",
    # Industrial
    "CAT": "industrial", "DE": "industrial", "HON": "industrial",
    "UPS": "industrial", "BA": "industrial", "GE": "industrial",
    # Defense
    "LMT": "defense", "RTX": "defense", "NOC": "defense", "GD": "defense",
}

MAX_SECTOR_PCT = Decimal("0.40")  # 40% max per sector


def check_sector_concentration(
    position_values: dict[str, Decimal],
    portfolio_value: Decimal,
) -> list[str]:
    """Check for sector concentration risk.

    Args:
        position_values: Dict of ticker -> market value (shares * price)
        portfolio_value: Total portfolio value

    Returns:
        List of warning strings (empty if no issues).
    """
    if portfolio_value <= 0:
        return []

    sector_totals: dict[str, Decimal] = {}
    for ticker, value in position_values.items():
        sector = SECTOR_MAP.get(ticker, "other")
        sector_totals[sector] = sector_totals.get(sector, Decimal(0)) + value

    warnings = []
    for sector, total in sector_totals.items():
        pct = total / portfolio_value
        if pct > MAX_SECTOR_PCT:
            warnings.append(
                f"Sector '{sector}' concentration {pct:.0%} exceeds {MAX_SECTOR_PCT:.0%} limit "
                f"(${total:,.0f} of ${portfolio_value:,.0f})"
            )

    return warnings
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_risk.py -v`
Expected: All PASS

- [ ] **Step 5: Wire into trader.py as risk_notes**

In `v2/trader.py`, after the position sync and account snapshot steps, before building executor input, add a risk check that injects warnings into the executor context. After `executor_input = build_executor_input(account_info)` (line 149):

```python
        # Check sector concentration and inject warnings
        from .risk import check_sector_concentration
        position_values = {}
        for p in get_positions():
            price = get_latest_price(p["ticker"], client=data_client)
            if price:
                position_values[p["ticker"]] = p["shares"] * price
        sector_warnings = check_sector_concentration(position_values, portfolio_value)
        if sector_warnings:
            logger.warning("Sector concentration warnings: %s", sector_warnings)
            executor_input.risk_notes += "\n" + "\n".join(sector_warnings)
```

- [ ] **Step 6: Run all tests**

Run: `python3 -m pytest tests/v2/test_risk.py tests/v2/test_trader.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add v2/risk.py tests/v2/test_risk.py v2/trader.py
git commit -m "feat(v2): add sector concentration check as risk warning"
```

---

## Workstream C: Attribution & Learning Loop Accuracy

### Task 6: Add rolling time window to attribution

Attribution currently aggregates all historical decisions with equal weight. Old data from different market regimes drowns out recent signal quality.

**Files:**
- Modify: `v2/attribution.py:9-61` (add WHERE date filter)
- Test: `tests/v2/test_attribution.py`

- [ ] **Step 1: Write failing test**

```python
class TestAttributionTimeWindow:
    """Attribution should only consider recent decisions."""

    def test_compute_attribution_filters_by_days(self, mock_db, mock_cursor):
        """SQL should include a date filter when days param is provided."""
        from v2.attribution import compute_signal_attribution

        mock_cursor.fetchall.return_value = []
        compute_signal_attribution(days=60)

        sql = mock_cursor.execute.call_args[0][0]
        assert "d.date" in sql or "decision_date" in sql
        # Should have a date parameter
        params = mock_cursor.execute.call_args[0][1] if len(mock_cursor.execute.call_args[0]) > 1 else None
        assert params is not None

    def test_compute_attribution_defaults_to_90_days(self, mock_db, mock_cursor):
        """Default should be 90 days, not unlimited."""
        from v2.attribution import compute_signal_attribution

        mock_cursor.fetchall.return_value = []
        compute_signal_attribution()

        params = mock_cursor.execute.call_args[0][1] if len(mock_cursor.execute.call_args[0]) > 1 else None
        assert params is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_attribution.py::TestAttributionTimeWindow -v`
Expected: FAIL — `compute_signal_attribution` doesn't accept `days` and SQL has no date filter

- [ ] **Step 3: Add days parameter and date filter**

In `v2/attribution.py`, update `compute_signal_attribution`:

```python
def compute_signal_attribution(days: int = 90) -> list[dict]:
    """
    Compute signal attribution scores from decision_signals joined with decisions.

    Args:
        days: Only include decisions from the last N days (default 90).
    """
    from datetime import date, timedelta
    cutoff_date = date.today() - timedelta(days=days)

    with get_cursor() as cur:
        cur.execute("""
            WITH categorized AS (
                SELECT
                    ds.decision_id,
                    CASE
                        WHEN ds.signal_type = 'news_signal' THEN
                            'news_signal:' || COALESCE(ns.category, 'unknown')
                        WHEN ds.signal_type = 'macro_signal' THEN
                            'macro_signal:' || COALESCE(ms.category, 'unknown')
                        ELSE ds.signal_type
                    END AS category,
                    d.outcome_7d,
                    d.outcome_30d
                FROM decision_signals ds
                JOIN decisions d ON d.id = ds.decision_id
                LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
                LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
                WHERE d.action IN ('buy', 'sell')
                  AND d.date >= %s
            )
            SELECT
                category,
                COUNT(DISTINCT decision_id) AS sample_size,
                AVG(outcome_7d) AS avg_outcome_7d,
                AVG(outcome_30d) AS avg_outcome_30d,
                AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_7d,
                AVG(CASE WHEN outcome_30d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_30d
            FROM categorized
            WHERE outcome_7d IS NOT NULL
            GROUP BY category
            ORDER BY sample_size DESC
        """, (cutoff_date,))
        results = [dict(row) for row in cur.fetchall()]
```

Also update `build_attribution_constraints` label from "last 60 days" to reflect the actual window:

```python
    lines = [f"SIGNAL PERFORMANCE (last {days} days):"]
```

Wait — `build_attribution_constraints` doesn't call `compute_signal_attribution`, it calls `get_signal_attribution()` which reads from the table. The window is baked in at compute time. This is fine — just update the label in `build_attribution_constraints` to be generic:

```python
    lines = ["SIGNAL PERFORMANCE (rolling window):"]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_attribution.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v2/attribution.py tests/v2/test_attribution.py
git commit -m "fix(v2): add 90-day rolling window to signal attribution"
```

---

### Task 7: Split attribution by action direction (buy vs sell)

A signal category that's good for timing buys but bad for timing sells gets a blended average that's meaningless.

**Files:**
- Modify: `v2/attribution.py:9-61` (split by action)
- Modify: `v2/attribution.py:64-132` (update summary and constraints formatters)
- Modify: `v2/database/trading_db.py` (update upsert to handle directional categories)
- Test: `tests/v2/test_attribution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestAttributionByDirection:
    """Attribution should produce separate buy/sell categories."""

    def test_categories_include_action_direction(self, mock_db, mock_cursor):
        from v2.attribution import compute_signal_attribution

        mock_cursor.fetchall.return_value = [
            {"category": "news_signal:earnings:buy", "sample_size": 10,
             "avg_outcome_7d": Decimal("1.5"), "avg_outcome_30d": Decimal("3.0"),
             "win_rate_7d": Decimal("0.6"), "win_rate_30d": Decimal("0.55")},
            {"category": "news_signal:earnings:sell", "sample_size": 8,
             "avg_outcome_7d": Decimal("-0.5"), "avg_outcome_30d": Decimal("-1.0"),
             "win_rate_7d": Decimal("0.4"), "win_rate_30d": Decimal("0.35")},
        ]

        results = compute_signal_attribution()

        categories = [r["category"] for r in results]
        # Should have directional categories
        assert any(":buy" in c for c in categories)
        assert any(":sell" in c for c in categories)

    def test_attribution_sql_groups_by_action(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []
        from v2.attribution import compute_signal_attribution
        compute_signal_attribution()

        sql = mock_cursor.execute.call_args[0][0]
        assert "d.action" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_attribution.py::TestAttributionByDirection -v`
Expected: FAIL

- [ ] **Step 3: Update SQL to group by action**

In `v2/attribution.py`, in the `compute_signal_attribution` SQL, update the `categorized` CTE to append `':' || d.action` to the category:

```python
                    CASE
                        WHEN ds.signal_type = 'news_signal' THEN
                            'news_signal:' || COALESCE(ns.category, 'unknown') || ':' || d.action
                        WHEN ds.signal_type = 'macro_signal' THEN
                            'macro_signal:' || COALESCE(ms.category, 'unknown') || ':' || d.action
                        ELSE ds.signal_type || ':' || d.action
                    END AS category,
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_attribution.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v2/attribution.py tests/v2/test_attribution.py
git commit -m "feat(v2): split signal attribution by buy/sell direction"
```

---

### Task 8: Use trading days instead of calendar days for backfill

Backfill uses `timedelta(days=7)` which lands on weekends/holidays inconsistently, adding noise.

**Files:**
- Modify: `v2/backfill.py:122-130` (use trading day offset)
- Test: `tests/v2/test_backfill.py`

- [ ] **Step 1: Write failing tests**

```python
class TestTradingDayOffset:
    """Backfill should count trading days, not calendar days."""

    def test_friday_decision_exits_next_friday(self):
        """7 trading days from Friday should land on the following Friday, not the next Friday via calendar."""
        from v2.backfill import trading_day_offset
        from datetime import date

        # Friday March 13, 2026
        friday = date(2026, 3, 13)
        result = trading_day_offset(friday, 7)
        # 7 trading days later = Friday March 20 (skipping weekends)
        # Mon 16, Tue 17, Wed 18, Thu 19, Fri 20 = 5 trading days
        # Need 2 more: Mon 23, Tue 24
        # Wait — 7 trading days: Mon=1, Tue=2, Wed=3, Thu=4, Fri=5, Mon=6, Tue=7
        assert result == date(2026, 3, 24)

    def test_monday_decision_exits_wednesday(self):
        from v2.backfill import trading_day_offset
        from datetime import date

        monday = date(2026, 3, 16)
        result = trading_day_offset(monday, 7)
        # 7 trading days: Tue=1, Wed=2, Thu=3, Fri=4, Mon=5, Tue=6, Wed=7
        assert result == date(2026, 3, 25)

    def test_zero_offset_returns_next_trading_day(self):
        from v2.backfill import trading_day_offset
        from datetime import date

        saturday = date(2026, 3, 14)
        result = trading_day_offset(saturday, 0)
        # Should land on Monday
        assert result == date(2026, 3, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_backfill.py::TestTradingDayOffset -v`
Expected: FAIL — `trading_day_offset` doesn't exist

- [ ] **Step 3: Implement trading_day_offset and wire in**

In `v2/backfill.py`, add:

```python
def trading_day_offset(start: date, trading_days: int) -> date:
    """Advance start date by N trading days (skipping weekends).

    Does not account for market holidays — a minor inaccuracy
    that's acceptable for 7/30 day outcome windows.
    """
    current = start
    days_counted = 0
    while days_counted < trading_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            days_counted += 1
    # If we land on a weekend (only if trading_days=0), advance to Monday
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current
```

Then in `backfill_outcomes`, replace:
```python
        exit_date = decision_date + timedelta(days=days)
```
with:
```python
        exit_date = trading_day_offset(decision_date, days)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_backfill.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v2/backfill.py tests/v2/test_backfill.py
git commit -m "fix(v2): use trading days instead of calendar days for outcome backfill"
```

---

## Workstream D: Structural & Session Integrity

### Task 9: Track per-stage completion in sessions

Session idempotency is all-or-nothing. A failed Twitter post marks the whole session as failed, allowing re-runs that duplicate trades.

**Files:**
- Create: `db/init/012_session_stages.sql`
- Modify: `v2/database/trading_db.py` (add stage CRUD)
- Modify: `v2/session.py` (record stage completion, check per-stage on re-run)
- Test: `tests/v2/test_db.py`, `tests/v2/test_session.py`

- [ ] **Step 1: Write migration**

Create `db/init/012_session_stages.sql`:

```sql
CREATE TABLE IF NOT EXISTS session_stages (
    id SERIAL PRIMARY KEY,
    session_id INT REFERENCES sessions(id),
    stage_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'running',  -- running, completed, failed, skipped
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT,
    UNIQUE (session_id, stage_name)
);

CREATE INDEX IF NOT EXISTS idx_session_stages_session ON session_stages(session_id);
```

- [ ] **Step 2: Write failing DB tests**

In `tests/v2/test_db.py`, add:

```python
class TestSessionStages:
    def test_insert_session_stage(self, mock_db, mock_cursor):
        from v2.database.trading_db import insert_session_stage
        mock_cursor.fetchone.return_value = {"id": 1}
        result = insert_session_stage(session_id=1, stage_name="pipeline")
        assert result == 1
        assert "session_stages" in mock_cursor.execute.call_args[0][0]

    def test_complete_session_stage(self, mock_db, mock_cursor):
        from v2.database.trading_db import complete_session_stage
        complete_session_stage(session_id=1, stage_name="pipeline")
        sql = mock_cursor.execute.call_args[0][0]
        assert "completed" in sql

    def test_get_completed_stages(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_completed_stages
        mock_cursor.fetchall.return_value = [
            {"stage_name": "pipeline"},
            {"stage_name": "strategist"},
        ]
        result = get_completed_stages(session_id=1)
        assert result == {"pipeline", "strategist"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_db.py::TestSessionStages -v`
Expected: FAIL — functions don't exist

- [ ] **Step 4: Implement DB functions**

In `v2/database/trading_db.py`, add:

```python
def insert_session_stage(session_id: int, stage_name: str) -> int:
    """Record start of a session stage."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO session_stages (session_id, stage_name, status)
            VALUES (%s, %s, 'running')
            ON CONFLICT (session_id, stage_name) DO UPDATE SET
                status = 'running', started_at = NOW(), completed_at = NULL, error = NULL
            RETURNING id
        """, (session_id, stage_name))
        return cur.fetchone()["id"]


def complete_session_stage(session_id: int, stage_name: str):
    """Mark a session stage as completed."""
    with get_cursor() as cur:
        cur.execute("""
            UPDATE session_stages
            SET status = 'completed', completed_at = NOW()
            WHERE session_id = %s AND stage_name = %s
        """, (session_id, stage_name))


def fail_session_stage(session_id: int, stage_name: str, error: str):
    """Mark a session stage as failed."""
    with get_cursor() as cur:
        cur.execute("""
            UPDATE session_stages
            SET status = 'failed', completed_at = NOW(), error = %s
            WHERE session_id = %s AND stage_name = %s
        """, (error, session_id, stage_name))


def get_completed_stages(session_id: int) -> set[str]:
    """Get set of completed stage names for a session."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT stage_name FROM session_stages
            WHERE session_id = %s AND status = 'completed'
        """, (session_id,))
        return {row["stage_name"] for row in cur.fetchall()}
```

- [ ] **Step 5: Run DB tests**

Run: `python3 -m pytest tests/v2/test_db.py::TestSessionStages -v`
Expected: All PASS

- [ ] **Step 6: Write failing session test for stage-aware re-runs**

In `tests/v2/test_session.py`, add:

```python
class TestSessionStageTracking:
    """Re-runs should skip already-completed stages."""

    @patch("v2.session.get_session_for_date")
    @patch("v2.session.get_completed_stages")
    @patch("v2.session.insert_session_record")
    @patch("v2.session.run_pipeline")
    @patch("v2.session.run_strategist_loop")
    @patch("v2.session.run_trading_session")
    @patch("v2.session.run_strategy_reflection")
    @patch("v2.session.run_twitter_stage")
    @patch("v2.session.run_bluesky_stage")
    @patch("v2.session.run_dashboard_stage")
    @patch("v2.session.run_backfill")
    @patch("v2.session.compute_signal_attribution")
    @patch("v2.session.build_attribution_constraints")
    @patch("v2.session.complete_session")
    @patch("v2.session.insert_session_stage")
    @patch("v2.session.complete_session_stage")
    def test_rerun_skips_completed_stages(
        self, mock_complete_stage, mock_insert_stage,
        mock_complete, mock_constraints, mock_compute, mock_backfill,
        mock_dashboard, mock_bluesky, mock_twitter, mock_strategy,
        mock_trading, mock_strategist, mock_pipeline,
        mock_insert_session, mock_get_completed, mock_get_session,
    ):
        from v2.session import run_session

        # Simulate a previous failed session with pipeline + strategist completed
        mock_get_session.return_value = {"id": 1, "status": "failed"}
        mock_get_completed.return_value = {"pipeline", "strategist"}
        mock_insert_session.return_value = 2
        mock_constraints.return_value = ""

        run_session(force=True)

        # Pipeline and strategist should have been skipped
        # (they were completed in the previous session)
        mock_pipeline.assert_not_called()
        mock_strategist.assert_not_called()
        # Trading should have run (it was not completed)
        mock_trading.assert_called_once()
```

- [ ] **Step 7: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_session.py::TestSessionStageTracking -v`
Expected: FAIL

- [ ] **Step 8: Wire stage tracking into session.py**

In `v2/session.py`:

1. Add imports at the top:
```python
from .database.trading_db import (
    get_session_for_date, insert_session_record, complete_session, fail_session,
    insert_session_stage, complete_session_stage, fail_session_stage, get_completed_stages,
)
```

2. After the idempotency check block, add stage-awareness. Replace the simple `if existing and existing["status"] == "completed"` check with:

```python
    completed_stages = set()
    if not force:
        try:
            existing = get_session_for_date(today)
            if existing and existing["status"] == "completed":
                logger.warning("Session already completed for %s. Use --force to override.", today)
                result.learning_error = f"Session already completed for {today}"
                result.duration_seconds = time.monotonic() - start
                return result
            if existing:
                completed_stages = get_completed_stages(existing["id"])
                if completed_stages:
                    logger.info("Resuming session — already completed: %s", completed_stages)
        except Exception as e:
            logger.warning("Could not check session status: %s — proceeding", e)
```

3. For each stage, wrap the execution in a stage-skip check. Example for pipeline:

```python
    if skip_pipeline or "pipeline" in completed_stages:
        logger.info("[Stage 1] News pipeline — SKIPPED%s",
                     " (completed in prior run)" if "pipeline" in completed_stages else "")
    else:
        logger.info("[Stage 1] Running news pipeline")
        if session_id:
            insert_session_stage(session_id, "pipeline")
        try:
            result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit)
            if session_id:
                complete_session_stage(session_id, "pipeline")
        except Exception as e:
            result.pipeline_error = str(e)
            if session_id:
                fail_session_stage(session_id, "pipeline", str(e))
            logger.error("Pipeline failed: %s — continuing with existing signals", e)
```

Apply the same pattern for all stages: strategist, executor, strategy, twitter, bluesky, dashboard.

- [ ] **Step 9: Run all session tests**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add db/init/012_session_stages.sql v2/database/trading_db.py v2/session.py tests/v2/test_db.py tests/v2/test_session.py
git commit -m "feat(v2): per-stage session tracking to prevent duplicate work on re-runs"
```

---

### Task 10: Mark playbook actions as attempted/executed

If the executor partially executes a playbook, re-runs see the same playbook with all actions. There's no way to know which actions were already attempted.

**Files:**
- Create: `db/init/013_playbook_action_status.sql`
- Modify: `v2/database/trading_db.py` (add status update function)
- Modify: `v2/trader.py` (mark actions after execution)
- Modify: `v2/context.py` (filter out executed actions)
- Test: `tests/v2/test_db.py`, `tests/v2/test_trader.py`

- [ ] **Step 1: Write migration**

Create `db/init/013_playbook_action_status.sql`:

```sql
ALTER TABLE playbook_actions ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending';
-- Values: pending, executed, skipped, failed

CREATE INDEX IF NOT EXISTS idx_playbook_actions_status ON playbook_actions(status);
```

- [ ] **Step 2: Write failing DB test**

In `tests/v2/test_db.py`, add:

```python
class TestPlaybookActionStatus:
    def test_update_playbook_action_status(self, mock_db, mock_cursor):
        from v2.database.trading_db import update_playbook_action_status
        update_playbook_action_status(action_id=1, status="executed")
        sql = mock_cursor.execute.call_args[0][0]
        assert "playbook_actions" in sql
        assert "status" in sql

    def test_get_pending_playbook_actions(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_pending_playbook_actions
        mock_cursor.fetchall.return_value = [make_playbook_action_row()]
        result = get_pending_playbook_actions(playbook_id=1)
        sql = mock_cursor.execute.call_args[0][0]
        assert "pending" in sql
        assert len(result) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_db.py::TestPlaybookActionStatus -v`
Expected: FAIL

- [ ] **Step 4: Implement DB functions**

In `v2/database/trading_db.py`, add:

```python
def update_playbook_action_status(action_id: int, status: str):
    """Update the status of a playbook action."""
    with get_cursor() as cur:
        cur.execute("""
            UPDATE playbook_actions SET status = %s WHERE id = %s
        """, (status, action_id))


def get_pending_playbook_actions(playbook_id: int) -> list[dict]:
    """Get only pending (unexecuted) playbook actions."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM playbook_actions
            WHERE playbook_id = %s AND status = 'pending'
            ORDER BY priority ASC
        """, (playbook_id,))
        return [dict(row) for row in cur.fetchall()]
```

- [ ] **Step 5: Run DB tests**

Run: `python3 -m pytest tests/v2/test_db.py::TestPlaybookActionStatus -v`
Expected: PASS

- [ ] **Step 6: Update context.py to use pending actions**

In `v2/context.py`, in `build_executor_input`, replace:

```python
    playbook_actions_rows = get_playbook_actions(playbook["id"]) if playbook else []
```

with:

```python
    from .database.trading_db import get_pending_playbook_actions
    playbook_actions_rows = get_pending_playbook_actions(playbook["id"]) if playbook else []
```

- [ ] **Step 7: Update trader.py to mark actions after execution**

In `v2/trader.py`, after a successful trade execution (after `trades_executed += 1` at line 252), add:

```python
            # Mark playbook action as executed
            if decision.playbook_action_id:
                try:
                    from .database.trading_db import update_playbook_action_status
                    update_playbook_action_status(decision.playbook_action_id, "executed")
                except Exception as e:
                    logger.warning("Could not mark playbook action %d as executed: %s",
                                   decision.playbook_action_id, e)
```

And after a failed trade (in the `trades_failed` block), mark it as "failed":

```python
            if decision.playbook_action_id:
                try:
                    from .database.trading_db import update_playbook_action_status
                    update_playbook_action_status(decision.playbook_action_id, "failed")
                except Exception:
                    pass
```

- [ ] **Step 8: Run all tests**

Run: `python3 -m pytest tests/v2/test_db.py tests/v2/test_trader.py tests/v2/test_context.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add db/init/013_playbook_action_status.sql v2/database/trading_db.py v2/context.py v2/trader.py tests/v2/test_db.py
git commit -m "feat(v2): track playbook action status to prevent duplicate execution on re-runs"
```

---

### Task 11: Pass current prices to executor LLM

The executor has no price data, so it can't calculate dollar-based position sizes. It guesses quantities that often get rejected by validation.

**Files:**
- Modify: `v2/agent.py:40-48` (add prices field to ExecutorInput)
- Modify: `v2/context.py:373-413` (fetch prices during context build)
- Modify: `v2/agent.py:85-116` (update system prompt to reference prices)
- Test: `tests/v2/test_context.py`

- [ ] **Step 1: Write failing test**

```python
class TestExecutorInputIncludesPrices:
    """Executor input should include current prices for sizing."""

    @patch("v2.context.get_playbook")
    @patch("v2.context.get_pending_playbook_actions")
    @patch("v2.context.get_positions")
    @patch("v2.context.get_recent_decisions")
    @patch("v2.context.get_signal_attribution")
    @patch("v2.context.get_latest_price")
    def test_executor_input_has_prices(
        self, mock_price, mock_attr, mock_decisions, mock_positions,
        mock_actions, mock_playbook,
    ):
        from v2.context import build_executor_input

        mock_playbook.return_value = {"id": 1, "market_outlook": "test", "risk_notes": ""}
        mock_actions.return_value = [make_playbook_action_row(ticker="AAPL")]
        mock_positions.return_value = [make_position_row(ticker="AAPL")]
        mock_decisions.return_value = []
        mock_attr.return_value = []
        mock_price.return_value = Decimal("175.50")

        result = build_executor_input({"buying_power": Decimal("5000"), "portfolio_value": Decimal("10000")})

        assert hasattr(result, "current_prices")
        assert "AAPL" in result.current_prices
        assert result.current_prices["AAPL"] == Decimal("175.50")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_context.py::TestExecutorInputIncludesPrices -v`
Expected: FAIL

- [ ] **Step 3: Add current_prices to ExecutorInput**

In `v2/agent.py`, add `current_prices` field to `ExecutorInput`:

```python
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

    def __post_init__(self):
        if self.current_prices is None:
            self.current_prices = {}
```

- [ ] **Step 4: Fetch prices in context builder**

In `v2/context.py`, in `build_executor_input`, after building the actions list, add price fetching:

```python
    # Fetch current prices for all tickers in playbook and positions
    from .executor import get_latest_price
    tickers_needed = set()
    for a in actions:
        tickers_needed.add(a.ticker)
    for p in positions:
        tickers_needed.add(p["ticker"])

    current_prices = {}
    for ticker in tickers_needed:
        price = get_latest_price(ticker)
        if price:
            current_prices[ticker] = price
```

Then pass it to the `ExecutorInput` constructor:

```python
    return ExecutorInput(
        playbook_actions=actions,
        positions=[dict(p) for p in positions],
        account=account_info,
        attribution_summary=attribution_summary,
        recent_outcomes=recent_outcomes,
        market_outlook=playbook.get("market_outlook", "") if playbook else "No playbook available",
        risk_notes=playbook.get("risk_notes", "") if playbook else "",
        current_prices=current_prices,
    )
```

- [ ] **Step 5: Update system prompt to reference prices**

In `v2/agent.py`, in `TRADING_SYSTEM_PROMPT`, add to the INPUTS section:

```
8. current_prices — latest ask prices for relevant tickers (use these for dollar-based sizing)
```

And add to the RULES section:

```
- Use current_prices to calculate position sizes by dollar amount (e.g., to invest $500 in a $200 stock, set quantity to 2.5)
```

- [ ] **Step 6: Include prices in the serialized input**

In `v2/agent.py`, in `get_trading_decisions`, add to `input_data`:

```python
        "current_prices": {k: str(v) for k, v in executor_input.current_prices.items()},
```

- [ ] **Step 7: Run tests**

Run: `python3 -m pytest tests/v2/test_context.py tests/v2/test_agent.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add v2/agent.py v2/context.py tests/v2/test_context.py
git commit -m "feat(v2): pass current prices to executor for accurate dollar-based sizing"
```

---

### Task 12: Add news signal deduplication

The news pipeline can insert duplicate signals when headlines are re-fetched across overlapping time windows.

**Files:**
- Modify: `v2/database/trading_db.py` (add unique constraint handling)
- Create: `db/init/014_news_signal_dedup.sql`
- Test: `tests/v2/test_db.py`

- [ ] **Step 1: Write migration**

Create `db/init/014_news_signal_dedup.sql`:

```sql
-- Add unique constraint on (ticker, headline, published_at) to prevent duplicates.
-- Use a hash index on headline to handle long text efficiently.
CREATE UNIQUE INDEX IF NOT EXISTS idx_news_signals_dedup
    ON news_signals (ticker, md5(headline), published_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_macro_signals_dedup
    ON macro_signals (md5(headline), published_at);
```

- [ ] **Step 2: Write failing test**

```python
class TestNewsSignalDedup:
    def test_insert_news_signals_batch_uses_on_conflict(self, mock_db, mock_cursor):
        """Batch insert should use ON CONFLICT DO NOTHING for dedup."""
        from v2.database.trading_db import insert_news_signals_batch
        signals = [("AAPL", "Headline", "earnings", "bullish", "high", datetime.now())]
        insert_news_signals_batch(signals)
        # Check that the SQL includes ON CONFLICT
        call_args = str(mock_cursor.execute.call_args) if mock_cursor.execute.called else ""
        # For execute_values, check the template
        from unittest.mock import call
        all_calls = [str(c) for c in mock_cursor.mock_calls]
        sql_used = " ".join(all_calls)
        assert "ON CONFLICT" in sql_used or "on conflict" in sql_used.lower()
```

Note: This test may need adjustment based on how `insert_news_signals_batch` is currently implemented (it likely uses `execute_values`). Read the current implementation to confirm.

- [ ] **Step 3: Update batch insert to use ON CONFLICT**

In `v2/database/trading_db.py`, update `insert_news_signals_batch` to add `ON CONFLICT DO NOTHING` to the `execute_values` template:

```python
def insert_news_signals_batch(signals: list[tuple]) -> int:
    """Batch insert news signals, skipping duplicates."""
    if not signals:
        return 0
    with get_cursor() as cur:
        from psycopg2.extras import execute_values
        execute_values(
            cur,
            """INSERT INTO news_signals (ticker, headline, category, sentiment, confidence, published_at)
               VALUES %s
               ON CONFLICT DO NOTHING""",
            signals,
        )
        return cur.rowcount
```

Apply the same pattern to `insert_macro_signals_batch`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_db.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add db/init/014_news_signal_dedup.sql v2/database/trading_db.py tests/v2/test_db.py
git commit -m "fix(v2): deduplicate news signals on batch insert"
```

---

### Task 13: Handle survivorship bias in attribution

Decisions where backfill can't find a price (delisted, ticker change) are excluded from attribution, biasing toward survivors.

**Files:**
- Modify: `v2/backfill.py:130-135` (record "no data" as explicit outcome)
- Test: `tests/v2/test_backfill.py`

- [ ] **Step 1: Write failing test**

```python
class TestBackfillNoPrice:
    """When no exit price is found, record a sentinel outcome instead of skipping."""

    @patch("v2.backfill.get_data_client")
    @patch("v2.backfill.get_decisions_needing_backfill")
    @patch("v2.backfill.get_price_on_date")
    @patch("v2.backfill.update_outcome")
    def test_no_price_records_sentinel(self, mock_update, mock_get_price, mock_get_decisions, mock_client):
        from v2.backfill import backfill_outcomes
        from datetime import date
        from decimal import Decimal

        mock_client.return_value = MagicMock()
        mock_get_decisions.return_value = [
            {"id": 1, "date": date(2026, 1, 1), "ticker": "DELIST",
             "action": "buy", "price": Decimal("50.00")},
        ]
        mock_get_price.return_value = None  # No price found

        stats = backfill_outcomes(days=7)

        # Should have attempted to record a -100% outcome for missing data
        mock_update.assert_called_once()
        args = mock_update.call_args[0]
        assert args[0] == 1  # decision_id
        assert args[1] == 7  # days
        assert args[2] == Decimal("-100")  # sentinel for "total loss assumed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_backfill.py::TestBackfillNoPrice -v`
Expected: FAIL — currently skips with `continue`

- [ ] **Step 3: Implement sentinel outcome for missing prices**

In `v2/backfill.py`, in `backfill_outcomes`, replace the "no price" skip block:

```python
        if exit_price is None:
            # Assume total loss for missing data (delisted, ticker change, etc.)
            # This prevents survivorship bias in attribution.
            outcome = Decimal("-100")
            if dry_run:
                print(f"  [{decision_id}] {ticker}: No price data for {exit_date} — assuming -100% [DRY RUN]")
            else:
                try:
                    update_outcome(decision_id, days, outcome)
                    print(f"  [{decision_id}] {ticker}: No price data for {exit_date} — recorded -100%")
                    stats["outcomes_filled"] += 1
                except Exception as e:
                    print(f"  [{decision_id}] Error updating: {e}")
                    stats["errors"] += 1
            continue
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_backfill.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v2/backfill.py tests/v2/test_backfill.py
git commit -m "fix(v2): record -100% sentinel for missing backfill prices to reduce survivorship bias"
```

---

### Task 14: Add dry-run price handling to prevent NULL decision prices

Dry runs return `filled_avg_price=None` and the logging fallback calls `get_latest_price()` live. If that fails, `price=NULL` decisions are logged and excluded from backfill.

**Files:**
- Modify: `v2/executor.py:164-171` (dry-run returns quote price)
- Modify: `v2/trader.py:322-323` (never log NULL price)
- Test: `tests/v2/test_executor.py`, `tests/v2/test_trader.py`

- [ ] **Step 1: Write failing test**

```python
class TestDryRunPrice:
    def test_dry_run_order_uses_provided_price(self):
        """Dry run should accept and return a simulated fill price."""
        from v2.executor import execute_market_order

        result = execute_market_order("AAPL", "buy", Decimal("5"), dry_run=True, simulated_price=Decimal("150.00"))
        assert result.filled_avg_price == Decimal("150.00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_executor.py::TestDryRunPrice -v`
Expected: FAIL — `execute_market_order` doesn't accept `simulated_price`

- [ ] **Step 3: Add simulated_price parameter**

In `v2/executor.py`, update `execute_market_order`:

```python
def execute_market_order(
    ticker: str,
    side: str,
    qty: Decimal,
    dry_run: bool = False,
    simulated_price: Decimal = None,
) -> OrderResult:
    """Execute a market order."""
    if dry_run:
        return OrderResult(
            success=True,
            order_id="DRY_RUN",
            filled_qty=qty,
            filled_avg_price=simulated_price,
            error=None,
        )
```

- [ ] **Step 4: Update trader.py to pass price to dry-run orders**

In `v2/trader.py`, update the `execute_market_order` call (around line 233):

```python
        result = execute_market_order(
            ticker=decision.ticker,
            side=decision.action,
            qty=Decimal(decision.quantity),
            dry_run=dry_run,
            simulated_price=price,  # Pass validated quote price for dry-run logging
        )
```

- [ ] **Step 5: Add guard against NULL price in decision logging**

In `v2/trader.py`, in the decision logging loop (around line 323), add a guard:

```python
            price = result.filled_avg_price if result and result.filled_avg_price else get_latest_price(decision.ticker, client=data_client)
            if price is None:
                errors.append(f"No price available for {decision.ticker} — skipping decision log")
                logger.error("Cannot log decision for %s: no price available", decision.ticker)
                continue
```

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/v2/test_executor.py tests/v2/test_trader.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add v2/executor.py v2/trader.py tests/v2/test_executor.py tests/v2/test_trader.py
git commit -m "fix(v2): dry-run orders carry simulated price to prevent NULL decision prices"
```

---

### Task 15: Full regression test run and cleanup

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/v2/ -v --tb=short`
Expected: All PASS, no regressions

- [ ] **Step 2: Run with coverage**

Run: `python3 -m pytest tests/v2/ --cov=v2 --cov-report=term-missing`
Expected: Coverage >= 89% (baseline)

- [ ] **Step 3: Fix any regressions**

If any existing tests broke due to signature changes (e.g., `validate_decision` now takes `open_sell_orders`, `get_latest_price` now takes `client`, `execute_market_order` now takes `simulated_price`), update those tests to pass the new optional params or use defaults.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test(v2): fix regressions from risk and learning loop improvements"
```

---

## Dependency Graph

```
Workstream A (Order Safety):
  Task 1 (cancel on timeout) — standalone
  Task 2 (client reuse) — standalone

Workstream B (Risk Controls):
  Task 3 (total exposure) — standalone
  Task 4 (pending sells) — standalone
  Task 5 (sector concentration) — depends on Task 2 (uses shared data_client)

Workstream C (Attribution):
  Task 6 (time window) — standalone
  Task 7 (buy/sell split) — after Task 6
  Task 8 (trading days) — standalone
  Task 13 (survivorship bias) — standalone

Workstream D (Structural):
  Task 9 (stage tracking) — standalone
  Task 10 (playbook action status) — standalone
  Task 11 (prices to executor) — depends on Task 2 (uses get_latest_price)
  Task 12 (news dedup) — standalone
  Task 14 (dry-run prices) — depends on Task 2 (uses client param)

Task 15 (regression sweep) — after all others
```

Tasks within a workstream can be parallelized across agents if they have no dependencies. Cross-workstream, the only hard dependency is Task 2 (client reuse) before Tasks 5, 11, and 14.
