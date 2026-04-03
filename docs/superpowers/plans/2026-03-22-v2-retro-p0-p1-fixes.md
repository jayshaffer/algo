# V2 Pipeline Retro P0/P1 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 7 P0/P1 issues from `v2/RETRO.md` — market hours gating, stale quote protection, fill confirmation, signal ref validation, expected-value attribution, session idempotency, and buying power tracking.

**Architecture:** Seven independent fixes across the v2 execution, attribution, and session layers. Each fix is self-contained: add a function or guard, wire it in, test it. No new dependencies — everything uses existing Alpaca SDK (`alpaca-py`), psycopg2, and the Claude client.

**Tech Stack:** Python 3, Alpaca SDK (`alpaca-py`), psycopg2, pytest

**Important notes:**
- The canonical source package is `v2/` (not `trading/`). All test imports use `from v2.*`.
- DB access is via `v2.database.connection` (`get_cursor()`) and `v2.database.trading_db` (CRUD functions).
- The decision dataclass is `ExecutorDecision` (not `TradingDecision`), defined in `v2/agent.py`.
- The order result dataclass is `OrderResult`, defined in `v2/executor.py`.
- Tests live in `tests/v2/` and use class-based grouping (`class TestXxx:`).
- Tests use `mock_db` and `mock_cursor` fixtures from `tests/v2/conftest.py`. The `mock_db` fixture patches `get_cursor` across all v2 modules.
- Factory functions in `tests/v2/conftest.py`: `make_trading_decision()` (returns `ExecutorDecision`), `make_agent_response()`, `make_news_signal_row()`, etc.
- v2 tests use inline `with patch("v2.module.name")` blocks — there is no `_patch_all()` helper.
- `v2/trader.py` already has an `order_results` dict (line 181) and partially prefers fill price at logging time (line 274). The fill confirmation task builds on this.
- Run all tests with: `python3 -m pytest tests/v2/ -v`

---

## File Map

| File | Change | Purpose |
|------|--------|---------|
| `v2/executor.py` | Add `is_market_open()`, `wait_for_fill()`, add staleness/spread checks to `get_latest_price()` | P0: market hours, fill confirmation, stale quotes |
| `v2/trader.py` | Wire in market hours gate, use fill price for buying power + logging, refresh buying power from Alpaca after fills | P0+P1: orchestration changes |
| `v2/agent.py` | Add `validate_signal_refs()` | P1: signal ref validation |
| `v2/attribution.py` | Replace win-rate thresholds with expected value in `build_attribution_constraints()` and `get_attribution_summary()` | P1: better signal filtering |
| `v2/session.py` | Add session-level idempotency check | P1: no duplicate runs |
| `v2/database/trading_db.py` | Add `get_session_for_date()`, `insert_session_record()`, `complete_session()` | P1: session tracking |
| `db/init/011_sessions.sql` | `sessions` table | P1: session dedup schema |
| `tests/v2/test_executor.py` | New file — tests for `is_market_open()`, `wait_for_fill()`, stale quote checks | All P0 tests |
| `tests/v2/test_trader.py` | Tests for market hours gate, fill-based buying power, fill-based logging | P0+P1 wiring tests |
| `tests/v2/test_agent.py` | Tests for `validate_signal_refs()` | P1 tests |
| `tests/v2/test_attribution.py` | Tests for expected value metric | P1 tests |
| `tests/v2/test_session.py` | Tests for idempotency gate | P1 tests |
| `tests/v2/test_db.py` | Tests for session DB functions | P1 tests |

---

### Task 1: Market Hours Check (P0-3)

**Files:**
- Modify: `v2/executor.py` (add `is_market_open()` after line 63)
- Modify: `v2/trader.py:57-100` (gate on market open between Step 1 sync and Step 2 snapshot)
- Create: `tests/v2/test_executor.py`
- Modify: `tests/v2/test_trader.py`

- [ ] **Step 1: Write the failing test for `is_market_open()`**

Create `tests/v2/test_executor.py`:

```python
"""Tests for v2 executor functions."""

from decimal import Decimal
from unittest.mock import patch, MagicMock

from v2.executor import OrderResult


class TestIsMarketOpen:
    @patch("v2.executor.get_trading_client")
    def test_returns_true_when_open(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_client.return_value.get_clock.return_value = mock_clock

        from v2.executor import is_market_open
        assert is_market_open() is True

    @patch("v2.executor.get_trading_client")
    def test_returns_false_when_closed(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_client.return_value.get_clock.return_value = mock_clock

        from v2.executor import is_market_open
        assert is_market_open() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_executor.py::TestIsMarketOpen -v`
Expected: FAIL — `ImportError: cannot import name 'is_market_open'`

- [ ] **Step 3: Implement `is_market_open()`**

Add to `v2/executor.py` after `get_account_info()` (after line 63):

```python
def is_market_open() -> bool:
    """Check if the market is currently open via Alpaca clock API."""
    client = get_trading_client()
    clock = client.get_clock()
    return clock.is_open
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_executor.py::TestIsMarketOpen -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for market hours gate in trader**

Add to `tests/v2/test_trader.py`:

```python
class TestMarketHoursGate:
    def test_skips_trading_when_market_closed(self, mock_db, mock_cursor):
        """When market is closed and not dry_run, should return early after sync."""
        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot") as mock_snap:

            result = run_trading_session(dry_run=False)

        # Should not proceed to account snapshot when market is closed
        mock_acct.assert_not_called()
        assert result.trades_executed == 0
        assert any("market" in e.lower() for e in result.errors)

    def test_allows_trading_when_market_open(self, mock_db, mock_cursor):
        """When market is open, should proceed normally."""
        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions:

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[],
                market_summary="No trades", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        mock_acct.assert_called_once()

    def test_dry_run_bypasses_market_hours_check(self, mock_db, mock_cursor):
        """Dry run should work even when market is closed."""
        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions:

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[],
                market_summary="No trades", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=True)

        # Should proceed past the gate
        mock_acct.assert_called_once()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_trader.py::TestMarketHoursGate -v`
Expected: FAIL — `is_market_open` not imported / not called

- [ ] **Step 7: Wire market hours gate into `trader.py`**

In `v2/trader.py`, add `is_market_open` to the import from `.executor` (line 10-18):

```python
from .executor import (
    get_account_info,
    take_account_snapshot,
    sync_positions_from_alpaca,
    sync_orders_from_alpaca,
    execute_market_order,
    get_latest_price,
    calculate_position_size,
    is_market_open,
)
```

Then insert the market hours gate after Step 1 (after line 99, before Step 2 comment):

```python
    # Market hours gate — skip trading if market is closed (dry_run bypasses)
    if not dry_run and not is_market_open():
        logger.warning("Market is closed. Skipping trading session (use --dry-run to bypass)")
        errors.append("Market is closed — skipped trading")
        return TradingSessionResult(
            timestamp=timestamp,
            account_snapshot_id=0,
            positions_synced=positions_synced,
            orders_synced=orders_synced,
            decisions_made=0,
            trades_executed=0,
            trades_failed=0,
            total_buy_value=Decimal(0),
            total_sell_value=Decimal(0),
            errors=errors,
        )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_trader.py tests/v2/test_executor.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add v2/executor.py v2/trader.py tests/v2/test_executor.py tests/v2/test_trader.py
git commit -m "feat(v2): add market hours gate (P0-3)

Block live trading when market is closed. Dry run bypasses the check.
Gate placed after position sync so we still get up-to-date data."
```

---

### Task 2: Stale Quote Protection (P0-2)

**Files:**
- Modify: `v2/executor.py:243-262` (enhance `get_latest_price()`)
- Modify: `tests/v2/test_executor.py` (add stale quote tests)

- [ ] **Step 1: Write the failing tests for stale quote checks**

Add to `tests/v2/test_executor.py`:

```python
from datetime import datetime, timezone, timedelta


class TestGetLatestPrice:
    @patch("v2.executor.StockHistoricalDataClient")  # patching module-level import
    def test_returns_price_for_fresh_quote(self, mock_data_client_cls):
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.25
        mock_quote.bid_price = 150.00
        mock_quote.timestamp = datetime.now(timezone.utc)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price == Decimal("150.25")

    @patch("v2.executor.StockHistoricalDataClient")  # patching module-level import
    def test_returns_none_for_stale_quote(self, mock_data_client_cls):
        """Quote older than max_age_seconds should return None."""
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.25
        mock_quote.bid_price = 150.00
        mock_quote.timestamp = datetime.now(timezone.utc) - timedelta(seconds=120)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL", max_age_seconds=60)

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")  # patching module-level import
    def test_returns_none_for_wide_spread(self, mock_data_client_cls):
        """Quote with bid-ask spread > max_spread_pct should return None."""
        mock_quote = MagicMock()
        mock_quote.ask_price = 160.0  # 10% above bid
        mock_quote.bid_price = 145.0
        mock_quote.timestamp = datetime.now(timezone.utc)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL", max_spread_pct=Decimal("0.05"))

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")  # patching module-level import
    def test_returns_none_for_zero_price(self, mock_data_client_cls):
        mock_quote = MagicMock()
        mock_quote.ask_price = 0
        mock_quote.bid_price = 0
        mock_quote.timestamp = datetime.now(timezone.utc)
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")  # patching module-level import
    def test_returns_none_on_api_error(self, mock_data_client_cls):
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.side_effect = Exception("API error")
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price is None

    @patch("v2.executor.StockHistoricalDataClient")  # patching module-level import
    def test_staleness_check_skipped_when_no_timestamp(self, mock_data_client_cls):
        """If quote has no timestamp attr, skip staleness check (backwards compat)."""
        mock_quote = MagicMock(spec=["ask_price", "bid_price"])
        mock_quote.ask_price = 150.25
        mock_quote.bid_price = 150.00
        mock_client = MagicMock()
        mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        mock_data_client_cls.return_value = mock_client

        with patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}):
            from v2.executor import get_latest_price
            price = get_latest_price("AAPL")

        assert price == Decimal("150.25")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_executor.py::TestGetLatestPrice -v`
Expected: FAIL — signature mismatch or missing staleness logic

- [ ] **Step 3: Rewrite `get_latest_price()` with staleness and spread checks**

First, move the local imports in `get_latest_price()` to module level. Add these imports at the top of `v2/executor.py` (after the existing alpaca imports around line 10-11):

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
```

Then replace `get_latest_price()` in `v2/executor.py` (lines 243-262):

```python
def get_latest_price(
    ticker: str,
    max_age_seconds: int = 60,
    max_spread_pct: Decimal = Decimal("0.05"),
) -> Optional[Decimal]:
    """Get latest quote price for a ticker with staleness and spread validation.

    Args:
        ticker: Stock ticker symbol
        max_age_seconds: Reject quotes older than this (seconds). 0 to disable.
        max_spread_pct: Reject quotes with bid-ask spread wider than this fraction. 0 to disable.

    Returns:
        Ask price as Decimal, or None if quote is stale, wide-spread, zero, or unavailable.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockLatestQuoteRequest(symbol_or_symbols=ticker)

    try:
        quotes = client.get_stock_latest_quote(request)
        quote = quotes[ticker]

        ask = Decimal(str(quote.ask_price))
        if ask == 0:
            return None

        # Staleness check
        if max_age_seconds > 0 and hasattr(quote, "timestamp") and quote.timestamp:
            from datetime import timezone
            age = (datetime.now(timezone.utc) - quote.timestamp).total_seconds()
            if age > max_age_seconds:
                logger.warning("%s: quote is %.0fs old (max %ds) — rejecting", ticker, age, max_age_seconds)
                return None

        # Spread check
        bid = Decimal(str(quote.bid_price)) if hasattr(quote, "bid_price") else Decimal(0)
        if max_spread_pct > 0 and bid > 0:
            spread_pct = (ask - bid) / bid
            if spread_pct > max_spread_pct:
                logger.warning("%s: spread %.2f%% exceeds max %.2f%% — rejecting", ticker, float(spread_pct) * 100, float(max_spread_pct) * 100)
                return None

        return ask
    except Exception:
        return None
```

Also add `import logging` and `logger = logging.getLogger(__name__)` at the top of `v2/executor.py` if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_executor.py::TestGetLatestPrice -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `python3 -m pytest tests/v2/ -v`
Expected: ALL PASS (existing callers pass no kwargs, get default behavior)

- [ ] **Step 6: Commit**

```bash
git add v2/executor.py tests/v2/test_executor.py
git commit -m "feat(v2): add staleness and spread checks to get_latest_price (P0-2)

Reject quotes older than 60s or with bid-ask spread > 5%.
Both thresholds are configurable via kwargs."
```

---

### Task 3: Fill Confirmation (P0-1)

**Files:**
- Modify: `v2/executor.py` (add `wait_for_fill()`)
- Modify: `v2/trader.py:213-230` (call `wait_for_fill()` after `execute_market_order()`, use fill price)
- Modify: `tests/v2/test_executor.py` (add fill confirmation tests)
- Modify: `tests/v2/test_trader.py` (add fill-based price tests)

- [ ] **Step 1: Write the failing tests for `wait_for_fill()`**

Add to `tests/v2/test_executor.py`:

```python
import time


class TestWaitForFill:
    @patch("v2.executor.get_trading_client")
    def test_returns_filled_order(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "filled"
        mock_order.filled_qty = "2.5"
        mock_order.filled_avg_price = "150.25"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is True
        assert result.filled_qty == Decimal("2.5")
        assert result.filled_avg_price == Decimal("150.25")

    @patch("v2.executor.get_trading_client")
    def test_returns_error_on_timeout(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "accepted"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=0.05, poll_interval=0.01)

        assert result.success is False
        assert "timeout" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_returns_error_on_cancelled(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "canceled"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is False
        assert "canceled" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_returns_error_on_rejected(self, mock_client):
        mock_order = MagicMock()
        mock_order.status.value = "rejected"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is False
        assert "rejected" in result.error.lower()

    @patch("v2.executor.get_trading_client")
    def test_polls_until_filled(self, mock_client):
        """Should poll multiple times until order status becomes filled."""
        pending = MagicMock()
        pending.status.value = "accepted"

        filled = MagicMock()
        filled.status.value = "filled"
        filled.filled_qty = "5"
        filled.filled_avg_price = "100.00"

        mock_client.return_value.get_order_by_id.side_effect = [pending, pending, filled]

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is True
        assert mock_client.return_value.get_order_by_id.call_count == 3

    @patch("v2.executor.get_trading_client")
    def test_partially_filled_waits(self, mock_client):
        """Partially filled should keep polling."""
        partial = MagicMock()
        partial.status.value = "partially_filled"

        filled = MagicMock()
        filled.status.value = "filled"
        filled.filled_qty = "10"
        filled.filled_avg_price = "200.00"

        mock_client.return_value.get_order_by_id.side_effect = [partial, filled]

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.01)

        assert result.success is True
        assert result.filled_qty == Decimal("10")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_executor.py::TestWaitForFill -v`
Expected: FAIL — `ImportError: cannot import name 'wait_for_fill'`

- [ ] **Step 3: Implement `wait_for_fill()`**

Add to `v2/executor.py` after `execute_limit_order()`:

```python
def wait_for_fill(
    order_id: str,
    timeout_seconds: float = 30,
    poll_interval: float = 0.5,
) -> OrderResult:
    """Poll Alpaca until order is filled, cancelled, or timeout.

    Args:
        order_id: Alpaca order ID to check
        timeout_seconds: Max time to wait for fill
        poll_interval: Seconds between status checks

    Returns:
        OrderResult with fill details, or error if timeout/cancelled/rejected.
    """
    import time

    client = get_trading_client()
    terminal_failures = {"canceled", "cancelled", "expired", "rejected", "suspended"}
    start = time.monotonic()

    while time.monotonic() - start < timeout_seconds:
        order = client.get_order_by_id(order_id)
        status = order.status.value if hasattr(order.status, "value") else str(order.status)

        if status == "filled":
            return OrderResult(
                success=True,
                order_id=order_id,
                filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else None,
                filled_avg_price=Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
                error=None,
            )

        if status in terminal_failures:
            return OrderResult(
                success=False,
                order_id=order_id,
                filled_qty=None,
                filled_avg_price=None,
                error=f"Order {status}",
            )

        time.sleep(poll_interval)

    return OrderResult(
        success=False,
        order_id=order_id,
        filled_qty=None,
        filled_avg_price=None,
        error=f"Timeout waiting for fill after {timeout_seconds}s",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_executor.py::TestWaitForFill -v`
Expected: ALL PASS

- [ ] **Step 5: Write the failing test for fill-based price in trader**

Add to `tests/v2/test_trader.py`:

```python
class TestFillConfirmation:
    def test_uses_fill_price_for_logging(self, mock_db, mock_cursor):
        """After fill confirmation, logged price should be fill price, not quote."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=2.5, reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        submit_result = MagicMock(success=True, order_id="order-123", filled_avg_price=None, error=None)
        fill_result = MagicMock(success=True, order_id="order-123", filled_qty=Decimal("2.5"), filled_avg_price=Decimal("151.50"), error=None)

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price", return_value=Decimal("150.00")), \
             patch("v2.trader.execute_market_order", return_value=submit_result), \
             patch("v2.trader.wait_for_fill", return_value=fill_result), \
             patch("v2.trader.insert_decision", return_value=1) as mock_insert, \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        # Logged price should be fill price (151.50), not quote price (150.00)
        mock_insert.assert_called_once()
        call_kwargs = mock_insert.call_args
        logged_price = call_kwargs.kwargs.get("price") or call_kwargs[1].get("price")
        assert logged_price == Decimal("151.50")

    def test_dry_run_skips_wait_for_fill(self, mock_db, mock_cursor):
        """Dry run should NOT call wait_for_fill."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=2.5, reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price", return_value=Decimal("150.00")), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.wait_for_fill") as mock_wait, \
             patch("v2.trader.insert_decision", return_value=1), \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )
            mock_exec.return_value = MagicMock(success=True, order_id="DRY_RUN", filled_qty=Decimal("2.5"), filled_avg_price=None, error=None)

            result = run_trading_session(dry_run=True)

        mock_wait.assert_not_called()

    def test_fill_timeout_marks_trade_failed(self, mock_db, mock_cursor):
        """If fill confirmation times out, trade should be marked failed."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=2.5, reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        submit_result = MagicMock(success=True, order_id="order-123", filled_avg_price=None, error=None)
        fill_result = MagicMock(success=False, order_id="order-123", filled_qty=None, filled_avg_price=None, error="Timeout")

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price", return_value=Decimal("150.00")), \
             patch("v2.trader.execute_market_order", return_value=submit_result), \
             patch("v2.trader.wait_for_fill", return_value=fill_result), \
             patch("v2.trader.insert_decision", return_value=1), \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")}
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        assert result.trades_failed == 1
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_trader.py::TestFillConfirmation -v`
Expected: FAIL — `wait_for_fill` not imported or not called

- [ ] **Step 7: Wire fill confirmation into `trader.py`**

Add `wait_for_fill` to the executor import in `v2/trader.py`:

```python
from .executor import (
    get_account_info,
    take_account_snapshot,
    sync_positions_from_alpaca,
    sync_orders_from_alpaca,
    execute_market_order,
    get_latest_price,
    calculate_position_size,
    is_market_open,
    wait_for_fill,
)
```

Replace the post-execution block in `v2/trader.py` (lines 220-249). After `execute_market_order()` returns (line 218), replace the success handling:

```python
        if result.success:
            # Wait for fill confirmation (skip for dry run — fills are instant)
            if not dry_run and result.order_id != "DRY_RUN":
                fill = wait_for_fill(result.order_id)
                if not fill.success:
                    trades_failed += 1
                    errors.append(f"{decision.ticker} fill failed: {fill.error}")
                    logger.error("  %s: fill failed: %s", decision.ticker, fill.error)
                    continue
                # Update result with fill data
                result = fill

            trades_executed += 1
            order_ids[i] = result.order_id
            order_results[i] = result

            # Use fill price if available, fall back to quote price
            fill_price = result.filled_avg_price if result.filled_avg_price else price
            trade_value = fill_price * Decimal(str(decision.quantity))

            if decision.action == "buy":
                total_buy_value += trade_value
                buying_power -= trade_value
            else:
                total_sell_value += trade_value

            status = "[DRY RUN]" if dry_run else f"Order {result.order_id} filled @ ${fill_price}"
            logger.info("  %s - Success", status)

            # Mark thesis as executed if trade was based on one
            if decision.thesis_id and not dry_run:
                try:
                    close_thesis(
                        thesis_id=decision.thesis_id,
                        status="executed",
                        reason=f"Trade executed: {decision.action} {decision.quantity} shares @ ${fill_price}"
                    )
                    logger.info("  Thesis %d marked as executed", decision.thesis_id)
                except Exception as e:
                    errors.append(f"Failed to update thesis {decision.thesis_id}: {e}")
        else:
            trades_failed += 1
            errors.append(f"{decision.ticker} execution failed: {result.error}")
            logger.error("  %s: execution failed: %s", decision.ticker, result.error)
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_trader.py tests/v2/test_executor.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add v2/executor.py v2/trader.py tests/v2/test_executor.py tests/v2/test_trader.py
git commit -m "feat(v2): add fill confirmation via wait_for_fill (P0-1)

Poll Alpaca for fill status after order submission. Use actual fill
price for buying power updates, decision logging, and thesis closure.
Trades that timeout or get rejected are marked failed."
```

---

### Task 4: Signal Ref Validation (P1-1)

**Files:**
- Modify: `v2/agent.py` (add `validate_signal_refs()`)
- Modify: `v2/trader.py:294-303` (call `validate_signal_refs()` before insert)
- Modify: `tests/v2/test_agent.py` (add signal ref validation tests)
- Modify: `tests/v2/test_trader.py` (add wiring test)

- [ ] **Step 1: Write the failing tests for `validate_signal_refs()`**

Add to `tests/v2/test_agent.py`:

```python
class TestValidateSignalRefs:
    def test_valid_news_signal_passes(self, mock_db, mock_cursor):
        """Existing news_signal ID should pass validation."""
        mock_cursor.fetchone.return_value = {"id": 5}

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "news_signal", "id": 5}])
        assert valid == [{"type": "news_signal", "id": 5}]

    def test_invalid_signal_id_stripped(self, mock_db, mock_cursor):
        """Non-existent signal ID should be stripped."""
        mock_cursor.fetchone.return_value = None

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "news_signal", "id": 99999}])
        assert valid == []

    def test_invalid_signal_type_stripped(self, mock_db, mock_cursor):
        """Unknown signal type should be stripped."""
        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "invalid_type", "id": 1}])
        assert valid == []

    def test_mixed_valid_and_invalid(self, mock_db, mock_cursor):
        """Should keep valid refs and strip invalid ones."""
        def fetchone_side_effect():
            # First call returns a row (valid), second returns None (invalid)
            results = iter([{"id": 1}, None])
            return lambda: next(results)

        call_count = [0]
        def mock_fetchone():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"id": 1}
            return None

        mock_cursor.fetchone.side_effect = mock_fetchone

        from v2.agent import validate_signal_refs
        refs = [
            {"type": "news_signal", "id": 1},
            {"type": "news_signal", "id": 99999},
        ]
        valid = validate_signal_refs(refs)
        assert len(valid) == 1
        assert valid[0]["id"] == 1

    def test_empty_refs_returns_empty(self, mock_db, mock_cursor):
        from v2.agent import validate_signal_refs
        assert validate_signal_refs([]) == []

    def test_thesis_type_validated(self, mock_db, mock_cursor):
        """thesis signal type should also be validated."""
        mock_cursor.fetchone.return_value = {"id": 3}

        from v2.agent import validate_signal_refs
        valid = validate_signal_refs([{"type": "thesis", "id": 3}])
        assert valid == [{"type": "thesis", "id": 3}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_agent.py::TestValidateSignalRefs -v`
Expected: FAIL — `ImportError: cannot import name 'validate_signal_refs'`

- [ ] **Step 3: Implement `validate_signal_refs()`**

Add to `v2/agent.py` after `validate_decision()`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_agent.py::TestValidateSignalRefs -v`
Expected: ALL PASS

- [ ] **Step 5: Wire validation into trader.py**

Add `validate_signal_refs` to the import from `.agent` in `v2/trader.py`:

```python
from .agent import (
    get_trading_decisions,
    validate_decision,
    validate_signal_refs,
    format_decisions_for_logging,
    AgentResponse,
    ExecutorDecision,
    DEFAULT_EXECUTOR_MODEL,
)
```

Replace the signal-logging block in `v2/trader.py` (lines 294-303):

```python
        # Log signal-decision links for attribution (validate first)
        if decision.signal_refs:
            try:
                validated_refs = validate_signal_refs(decision.signal_refs)
                if len(validated_refs) < len(decision.signal_refs):
                    logger.warning("%s: stripped %d invalid signal refs",
                                   decision.ticker,
                                   len(decision.signal_refs) - len(validated_refs))
                if validated_refs:
                    signal_links = [
                        (decision_id, ref["type"], ref["id"])
                        for ref in validated_refs
                    ]
                    insert_decision_signals_batch(signal_links)
            except Exception as e:
                errors.append(f"Failed to log signal links for {decision.ticker}: {e}")
```

- [ ] **Step 6: Run all tests**

Run: `python3 -m pytest tests/v2/test_agent.py tests/v2/test_trader.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add v2/agent.py v2/trader.py tests/v2/test_agent.py tests/v2/test_trader.py
git commit -m "feat(v2): validate signal refs against DB before insertion (P1-1)

Strip hallucinated signal IDs from LLM output. Only news_signal,
macro_signal, and thesis types are valid. Invalid refs are logged
as warnings for prompt tuning."
```

---

### Task 5: Expected Value Attribution (P1-2)

**Files:**
- Modify: `v2/attribution.py:91-132` (replace win-rate thresholds with EV)
- Modify: `v2/attribution.py:64-88` (update summary to show EV)
- Modify: `tests/v2/test_attribution.py` (update threshold tests)

- [ ] **Step 1: Write the failing tests for EV-based constraints**

Add to `tests/v2/test_attribution.py`:

```python
class TestExpectedValueConstraints:
    def test_profitable_low_winrate_is_strong(self):
        """40% win rate but +8% wins / -2% losses = EV +2.0% → STRONG."""
        mock_rows = [
            {
                "category": "news_signal:contrarian",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("2.0"),  # Positive avg return = profitable
                "avg_outcome_30d": Decimal("3.0"),
                "win_rate_7d": Decimal("0.40"),
                "win_rate_30d": Decimal("0.45"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        # Should be STRONG based on positive avg return, despite <45% win rate
        assert "STRONG" in result
        assert "news_signal:contrarian" in result

    def test_unprofitable_high_winrate_is_weak(self):
        """60% win rate but tiny wins, big losses = negative avg return → WEAK."""
        mock_rows = [
            {
                "category": "news_signal:momentum",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("-0.5"),  # Negative avg return
                "avg_outcome_30d": Decimal("-1.0"),
                "win_rate_7d": Decimal("0.60"),
                "win_rate_30d": Decimal("0.55"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        # Should be WEAK based on negative avg return, despite >55% win rate
        assert "WEAK" in result
        assert "news_signal:momentum" in result

    def test_neutral_ev_not_categorized(self):
        """Near-zero avg return (between -0.5% and +0.5%) should not be STRONG or WEAK."""
        mock_rows = [
            {
                "category": "news_signal:flat",
                "sample_size": 20,
                "avg_outcome_7d": Decimal("0.05"),
                "avg_outcome_30d": Decimal("0.1"),
                "win_rate_7d": Decimal("0.50"),
                "win_rate_30d": Decimal("0.50"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        # Should appear in neither STRONG nor WEAK — neutral zone
        assert "STRONG" not in result
        assert "WEAK" not in result

    def test_insufficient_data_unchanged(self):
        """Below min_samples should still be INSUFFICIENT regardless of EV."""
        mock_rows = [
            {
                "category": "news_signal:rare",
                "sample_size": 2,
                "avg_outcome_7d": Decimal("10.0"),
                "avg_outcome_30d": Decimal("15.0"),
                "win_rate_7d": Decimal("1.0"),
                "win_rate_30d": Decimal("1.0"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        assert "INSUFFICIENT DATA" in result
        assert "news_signal:rare" in result

    def test_constraint_text_references_expected_value(self):
        """Constraint text should mention expected value / avg return."""
        mock_rows = [
            {
                "category": "news_signal:test",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("-2.0"),
                "avg_outcome_30d": Decimal("-1.5"),
                "win_rate_7d": Decimal("0.35"),
                "win_rate_30d": Decimal("0.40"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints(min_samples=5)

        assert "avg return" in result.lower() or "expected value" in result.lower() or "avg 7d" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_attribution.py::TestExpectedValueConstraints -v`
Expected: FAIL — old win-rate thresholds classify incorrectly

- [ ] **Step 3: Update `build_attribution_constraints()` to use expected value**

Replace `build_attribution_constraints()` in `v2/attribution.py` (lines 91-132):

```python
def build_attribution_constraints(min_samples: int = 5) -> str:
    """Format signal attribution into constraint block for strategist system prompt.

    Uses expected value (avg_outcome_7d) as the primary metric instead of win rate.
    A signal that wins rarely but wins big is more valuable than one that wins
    often but loses more when it loses.

    Categories:
      STRONG: avg 7d return > +0.5%, >= min_samples
      WEAK: avg 7d return < -0.5%, >= min_samples
      INSUFFICIENT DATA: < min_samples
    """
    rows = get_signal_attribution()
    if not rows:
        return ""

    strong, weak, insufficient = [], [], []

    for r in rows:
        cat = r["category"]
        n = r["sample_size"]
        avg = float(r["avg_outcome_7d"]) if r.get("avg_outcome_7d") else 0
        wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0

        if n < min_samples:
            insufficient.append(f"{cat} (n={n})")
        elif avg > 0.5:
            strong.append(f"{cat} ({avg:+.2f}% avg 7d return, {wr:.0f}% win rate, n={n})")
        elif avg < -0.5:
            weak.append(f"{cat} ({avg:+.2f}% avg 7d return, {wr:.0f}% win rate, n={n})")

    lines = ["SIGNAL PERFORMANCE (last 60 days):"]
    if strong:
        lines.append(f"  STRONG (positive avg return): {', '.join(strong)}")
    if weak:
        lines.append(f"  WEAK (negative avg return): {', '.join(weak)}")
    if insufficient:
        lines.append(f"  INSUFFICIENT DATA (<{min_samples} samples): {', '.join(insufficient)}")

    lines.append("")
    lines.append("CONSTRAINT: Do not create theses primarily based on WEAK signal categories")
    lines.append("(negative avg 7d return) unless you have a specific reason to override (explain in thesis text).")

    return "\n".join(lines)
```

- [ ] **Step 4: Update `get_attribution_summary()` to emphasize EV**

Replace `get_attribution_summary()` in `v2/attribution.py` (lines 64-88):

```python
def get_attribution_summary() -> str:
    """Format attribution scores as advisory text for LLM context."""
    rows = get_signal_attribution()
    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    lines = ["Signal Attribution:"]
    profitable = [r for r in rows if r.get("avg_outcome_7d") and float(r["avg_outcome_7d"]) > 0]
    unprofitable = [r for r in rows if r.get("avg_outcome_7d") and float(r["avg_outcome_7d"]) <= 0]

    if profitable:
        lines.append("Profitable signal types (positive avg 7d return):")
        for r in profitable:
            avg = float(r.get("avg_outcome_7d") or 0)
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0
            lines.append(f"  - {r['category']}: {avg:+.2f}% avg 7d return, {wr:.0f}% win rate (n={r['sample_size']})")

    if unprofitable:
        lines.append("Unprofitable signal types (negative avg 7d return):")
        for r in unprofitable:
            avg = float(r.get("avg_outcome_7d") or 0)
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0
            lines.append(f"  - {r['category']}: {avg:+.2f}% avg 7d return, {wr:.0f}% win rate (n={r['sample_size']})")

    return "\n".join(lines)
```

- [ ] **Step 5: Update existing tests that assert on old format**

The existing tests in `TestBuildAttributionConstraints` will need their assertions updated:

- `test_formats_strong_and_weak_categories`: Change `"STRONG (>55% win rate)"` to `"STRONG (positive avg return)"` and `"WEAK (<45% win rate)"` to `"WEAK (negative avg return)"`. Remove the `"70%"` assertion (it's now shown differently). Update to check `"avg 7d return"` in result.
- `test_threshold_boundaries`: Update to test based on avg_outcome_7d thresholds (+0.5% / -0.5%) instead of win rate 55%/45%.
- `TestGetAttributionSummary.test_formats_predictive_and_weak`: Change `"Predictive signal types"` to `"Profitable signal types"` and `"Weak/non-predictive"` to `"Unprofitable signal types"`.

- [ ] **Step 6: Run all attribution tests**

Run: `python3 -m pytest tests/v2/test_attribution.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add v2/attribution.py tests/v2/test_attribution.py
git commit -m "feat(v2): use expected value instead of win rate for attribution (P1-2)

STRONG/WEAK classification now based on avg 7d return (>+0.5% / <-0.5%)
instead of directional win rate. A signal that wins 40% of the time
but averages +2% per trade is now correctly flagged STRONG."
```

---

### Task 6: Session Idempotency (P1-3)

**Files:**
- Create: `db/init/011_sessions.sql` (sessions table)
- Modify: `v2/database/trading_db.py` (add session CRUD functions)
- Modify: `v2/session.py` (add idempotency gate)
- Modify: `tests/v2/test_db.py` (add session DB tests)
- Modify: `tests/v2/test_session.py` (add idempotency tests)

- [ ] **Step 1: Create the sessions table migration**

Create `db/init/011_sessions.sql`:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL,
    session_type VARCHAR(50) NOT NULL DEFAULT 'daily',
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT,
    UNIQUE (session_date, session_type)
);
```

- [ ] **Step 2: Write the failing tests for session DB functions**

Add to `tests/v2/test_db.py`:

```python
class TestSessionTracking:
    def test_insert_session_record(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_session_record
        result = insert_session_record(date(2026, 3, 22), "daily")
        assert result == 1
        assert "INSERT INTO sessions" in mock_cursor.execute.call_args[0][0]

    def test_get_session_for_date(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1, "status": "completed"}
        from v2.database.trading_db import get_session_for_date
        result = get_session_for_date(date(2026, 3, 22), "daily")
        assert result["status"] == "completed"

    def test_get_session_for_date_returns_none(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.trading_db import get_session_for_date
        result = get_session_for_date(date(2026, 3, 22), "daily")
        assert result is None

    def test_complete_session(self, mock_db, mock_cursor):
        from v2.database.trading_db import complete_session
        complete_session(1)
        sql = mock_cursor.execute.call_args[0][0]
        assert "UPDATE sessions" in sql
        assert "completed" in sql

    def test_fail_session(self, mock_db, mock_cursor):
        from v2.database.trading_db import fail_session
        fail_session(1, "Something broke")
        sql = mock_cursor.execute.call_args[0][0]
        assert "UPDATE sessions" in sql
        assert "failed" in sql
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_db.py::TestSessionTracking -v`
Expected: FAIL — `ImportError: cannot import name 'insert_session_record'`

- [ ] **Step 4: Implement session DB functions**

Add to `v2/database/trading_db.py` after the signal attribution section:

```python
# --- Session Tracking ---

def insert_session_record(session_date, session_type="daily") -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO sessions (session_date, session_type, status)
            VALUES (%s, %s, 'running')
            RETURNING id
        """, (session_date, session_type))
        return cur.fetchone()["id"]


def get_session_for_date(session_date, session_type="daily"):
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM sessions
            WHERE session_date = %s AND session_type = %s
            ORDER BY id DESC LIMIT 1
        """, (session_date, session_type))
        return cur.fetchone()


def complete_session(session_id):
    with get_cursor() as cur:
        cur.execute("""
            UPDATE sessions SET status = 'completed', completed_at = NOW()
            WHERE id = %s
        """, (session_id,))


def fail_session(session_id, error_text):
    with get_cursor() as cur:
        cur.execute("""
            UPDATE sessions SET status = 'failed', completed_at = NOW(), error = %s
            WHERE id = %s
        """, (error_text, session_id))
```

- [ ] **Step 5: Run DB tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_db.py::TestSessionTracking -v`
Expected: ALL PASS

- [ ] **Step 6: Write the failing tests for session idempotency gate**

Add to `tests/v2/test_session.py`:

```python
class TestSessionIdempotency:
    def test_blocks_duplicate_session(self):
        """Should refuse to run if a completed session exists for today."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline:

            mock_get.return_value = {"id": 1, "status": "completed"}

            result = run_session(dry_run=True)

        # Should not run any stages
        mock_pipeline.assert_not_called()
        assert result.has_errors or result.skipped_executor

    def test_allows_session_when_none_exists(self):
        """Should proceed normally when no session exists for today."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=1), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

        mock_pipeline.assert_called_once()

    def test_allows_session_when_previous_failed(self):
        """Should allow re-run if previous session failed."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.insert_session_record", return_value=2), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "failed"}

            result = run_session(dry_run=True)

        mock_pipeline.assert_called_once()

    def test_force_flag_overrides_idempotency(self):
        """--force should allow running even if completed session exists."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.insert_session_record", return_value=2), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            mock_get.return_value = {"id": 1, "status": "completed"}

            result = run_session(dry_run=True, force=True)

        mock_pipeline.assert_called_once()
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_session.py::TestSessionIdempotency -v`
Expected: FAIL — no `get_session_for_date` import, no `force` parameter

- [ ] **Step 8: Wire idempotency into `session.py`**

Add imports to `v2/session.py`:

```python
from .database.trading_db import get_session_for_date, insert_session_record, complete_session, fail_session
```

Add `force` parameter to `run_session()` signature:

```python
def run_session(
    dry_run: bool = False,
    model: str = "claude-opus-4-6",
    executor_model: str = DEFAULT_EXECUTOR_MODEL,
    max_turns: int = 25,
    skip_pipeline: bool = False,
    skip_ideation: bool = False,
    skip_executor: bool = False,
    skip_strategy: bool = False,
    skip_twitter: bool = False,
    skip_bluesky: bool = False,
    skip_dashboard: bool = False,
    pipeline_hours: int = 24,
    pipeline_limit: int = 300,
    force: bool = False,
) -> SessionResult:
```

Add idempotency check at the top of `run_session()`, right after `result = SessionResult(...)`:

```python
    # Idempotency check — prevent duplicate sessions
    from datetime import date
    today = date.today()
    session_id = None

    if not force:
        try:
            existing = get_session_for_date(today)
            if existing and existing["status"] == "completed":
                logger.warning("Session already completed for %s. Use --force to override.", today)
                result.learning_error = f"Session already completed for {today}"
                result.duration_seconds = time.monotonic() - start
                return result
        except Exception as e:
            logger.warning("Could not check session status: %s — proceeding", e)

    try:
        session_id = insert_session_record(today)
        logger.info("Session ID: %d", session_id)
    except Exception as e:
        logger.warning("Could not create session record: %s — proceeding without tracking", e)
```

At the end of `run_session()`, before the summary logging, mark session complete or failed:

```python
    # Mark session status
    if session_id:
        try:
            if result.has_errors:
                error_summary = "; ".join(
                    str(getattr(result, f))
                    for f in ["learning_error", "pipeline_error", "strategist_error", "trading_error", "strategy_error", "twitter_error", "bluesky_error", "dashboard_error"]
                    if getattr(result, f)
                )
                fail_session(session_id, error_summary)
            else:
                complete_session(session_id)
        except Exception as e:
            logger.warning("Could not update session status: %s", e)
```

Add `--force` to the CLI argparser:

```python
    parser.add_argument("--force", action="store_true", help="Override session idempotency check")
```

And pass it through:

```python
    result = run_session(
        ...,
        force=args.force,
    )
```

- [ ] **Step 9: Run all session and DB tests**

Run: `python3 -m pytest tests/v2/test_session.py tests/v2/test_db.py -v`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add db/init/011_sessions.sql v2/database/trading_db.py v2/session.py tests/v2/test_db.py tests/v2/test_session.py
git commit -m "feat(v2): add session idempotency check (P1-3)

Track sessions in DB with unique (date, type) constraint. Block
duplicate runs unless previous session failed or --force is used.
Mark sessions as completed or failed at the end."
```

---

### Task 7: Buying Power Refresh (P1-4)

**Files:**
- Modify: `v2/trader.py:220-230` (refresh buying power from Alpaca after each fill)
- Modify: `tests/v2/test_trader.py` (add buying power refresh tests)

- [ ] **Step 1: Write the failing tests for buying power refresh**

Add to `tests/v2/test_trader.py`:

```python
class TestBuyingPowerRefresh:
    def test_refreshes_buying_power_after_fill(self, mock_db, mock_cursor):
        """After a fill, buying power should be re-fetched from Alpaca."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=2.5, reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        submit_result = MagicMock(success=True, order_id="order-123", filled_avg_price=None, error=None)
        fill_result = MagicMock(success=True, order_id="order-123", filled_qty=Decimal("2.5"), filled_avg_price=Decimal("150.00"), error=None)

        # First call returns initial account info, second returns refreshed
        account_calls = [
            {"portfolio_value": Decimal("100000"), "cash": Decimal("50000"), "buying_power": Decimal("50000")},
            {"portfolio_value": Decimal("99625"), "cash": Decimal("49625"), "buying_power": Decimal("49625")},
        ]
        call_count = [0]
        def mock_get_account():
            idx = min(call_count[0], len(account_calls) - 1)
            call_count[0] += 1
            return account_calls[idx]

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info", side_effect=mock_get_account), \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price", return_value=Decimal("150.00")), \
             patch("v2.trader.execute_market_order", return_value=submit_result), \
             patch("v2.trader.wait_for_fill", return_value=fill_result), \
             patch("v2.trader.validate_signal_refs", return_value=[]), \
             patch("v2.trader.insert_decision", return_value=1), \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[decision], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=False)

        # get_account_info should be called twice: once at snapshot, once after fill
        assert call_count[0] >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_trader.py::TestBuyingPowerRefresh -v`
Expected: FAIL — `get_account_info` called only once

- [ ] **Step 3: Add buying power refresh after fill in `trader.py`**

In `v2/trader.py`, after the fill confirmation succeeds and `result = fill` is set, add a buying power refresh. Inside the `if not dry_run and result.order_id != "DRY_RUN":` block, after the fill confirmation succeeds, replace the local buying power update:

```python
            trades_executed += 1
            order_ids[i] = result.order_id
            order_results[i] = result

            # Use fill price if available, fall back to quote price
            fill_price = result.filled_avg_price if result.filled_avg_price else price
            trade_value = fill_price * Decimal(str(decision.quantity))

            if decision.action == "buy":
                total_buy_value += trade_value
            else:
                total_sell_value += trade_value

            # Refresh buying power from Alpaca after real trades
            if not dry_run:
                try:
                    refreshed = get_account_info()
                    buying_power = refreshed["buying_power"]
                    portfolio_value = refreshed["portfolio_value"]
                except Exception as e:
                    # Fall back to local estimate if refresh fails
                    logger.warning("Could not refresh buying power: %s — using local estimate", e)
                    if decision.action == "buy":
                        buying_power -= trade_value
            else:
                # Dry run: use local estimate
                if decision.action == "buy":
                    buying_power -= trade_value
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_trader.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/v2/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add v2/trader.py tests/v2/test_trader.py
git commit -m "feat(v2): refresh buying power from Alpaca after each fill (P1-4)

After fill confirmation, re-query Alpaca for real buying power instead
of maintaining a local estimate. Falls back to local estimate if the
refresh fails. Eliminates drift across multi-trade sessions."
```

---

## Final Verification

After all 7 tasks are complete:

- [ ] Run the full v2 test suite: `python3 -m pytest tests/v2/ -v`
- [ ] Run with coverage: `python3 -m pytest tests/v2/ --cov=v2 --cov-report=term-missing`
- [ ] Run the entire test suite: `python3 -m pytest tests/ -v`
- [ ] Verify no regressions in non-v2 tests
