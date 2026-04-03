# V2 Pipeline Retro P0/P1 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all P0 and P1 issues from `v2/RETRO.md` — fill confirmation, stale quote protection, market hours gating, signal ref validation, expected-value attribution, session idempotency, and buying power tracking.

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

---

## File Map

| File | Change | Purpose |
|------|--------|---------|
| `v2/executor.py` | Add `wait_for_fill()`, `is_market_open()`, add staleness/spread checks to `get_latest_price()` | P0: fill confirmation, market hours, stale quotes |
| `v2/trader.py` | Wire in market hours gate, fill-based price logging, buying power from fills | P0+P1: orchestration changes |
| `v2/agent.py` | Add `validate_signal_refs()` | P1: signal ref validation |
| `v2/attribution.py` | Replace win-rate with expected value in `build_attribution_constraints()` and `get_attribution_summary()` | P1: better signal filtering |
| `v2/session.py` | Add session-level idempotency check | P1: no duplicate runs |
| `v2/database/trading_db.py` | Add `get_session_for_date()`, `insert_session_record()`, `complete_session()` | P1: session tracking |
| `db/init/011_sessions.sql` | `sessions` table | P1: session dedup schema |
| `tests/v2/test_executor.py` | New file — tests for `wait_for_fill()`, `is_market_open()`, stale quote checks | All P0 tests |
| `tests/v2/test_trader.py` | Tests for wiring changes | P0+P1 tests |
| `tests/v2/test_agent.py` | Tests for signal ref validation | P1 tests |
| `tests/v2/test_attribution.py` | Tests for expected value metric | P1 tests |
| `tests/v2/test_session.py` | Tests for idempotency | P1 tests |
| `tests/v2/test_db.py` | Tests for session DB functions | P1 tests |

---

### Task 1: Market Hours Check (P0)

**Files:**
- Modify: `v2/executor.py` (add `is_market_open()`)
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

- [ ] **Step 5: Write the failing tests for market hours gate in trader**

Add to `tests/v2/test_trader.py`:

```python
class TestMarketHoursGate:
    def test_aborts_when_market_closed_and_not_dry_run(self, mock_db, mock_cursor):
        with patch("v2.trader.sync_positions_from_alpaca", return_value=2), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=False):

            result = run_trading_session(dry_run=False)

        assert result.trades_executed == 0
        assert any("market" in e.lower() for e in result.errors)

    def test_allows_dry_run_when_market_closed(self, mock_db, mock_cursor):
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
                market_outlook="Neutral", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[], thesis_invalidations=[],
                market_summary="No trades", risk_assessment="Low",
            )

            result = run_trading_session(dry_run=True)

        # Dry run should proceed even when market is closed
        assert not any("market" in e.lower() for e in result.errors)
```

Also add the missing imports at the top of the file:

```python
from v2.agent import ExecutorInput, ExecutorDecision, AgentResponse, PlaybookAction
from v2.executor import OrderResult
```

Note: `ExecutorInput` and `AgentResponse` are already imported in the existing test file (line 7). Add `OrderResult` import.

- [ ] **Step 6: Wire market hours check into `run_trading_session()`**

In `v2/trader.py`, add import at line 16:

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

After Step 1 (sync, line 99) and before Step 2 (snapshot, line 101), add:

```python
    # Market hours gate (live trading only)
    if not dry_run:
        try:
            if not is_market_open():
                errors.append("Market is closed — aborting live trading session")
                logger.error("Market is closed. Aborting session.")
                return TradingSessionResult(
                    timestamp=timestamp,
                    account_snapshot_id=0,
                    positions_synced=positions_synced,
                    orders_synced=orders_synced,
                    decisions_made=0, trades_executed=0, trades_failed=0,
                    total_buy_value=Decimal(0), total_sell_value=Decimal(0),
                    errors=errors,
                )
        except Exception as e:
            errors.append(f"Market hours check failed: {e}")
            logger.warning("Could not check market hours: %s — proceeding cautiously", e)
```

- [ ] **Step 7: Run tests**

Run: `python3 -m pytest tests/v2/test_executor.py::TestIsMarketOpen tests/v2/test_trader.py::TestMarketHoursGate -v`
Expected: PASS

- [ ] **Step 8: Run full test suite for regressions**

Run: `python3 -m pytest tests/v2/test_executor.py tests/v2/test_trader.py -v`
Expected: All existing tests still PASS. Existing trader tests use `dry_run=True` so the market hours gate is not triggered.

- [ ] **Step 9: Commit**

```bash
git add v2/executor.py v2/trader.py tests/v2/test_executor.py tests/v2/test_trader.py
git commit -m "feat: gate live trading on market hours check (P0)"
```

---

### Task 2: Stale Quote Protection (P0)

**Files:**
- Modify: `v2/executor.py:243-262` (enhance `get_latest_price()`)
- Modify: `tests/v2/test_executor.py`

**Design note:** This changes `get_latest_price()` from returning `ask_price` to returning bid-ask midpoint. Midpoint is a better estimate of fair value. Since midpoint < ask, buy validations become slightly more permissive (acceptable — the fill price may be anywhere in the spread). The function currently imports `StockHistoricalDataClient` inside the function body (line 245); we move it to module level so it's patchable.

**Staleness approach:** The retro flags both stale quotes and wide spreads. Rather than checking quote timestamps (which the Alpaca SDK doesn't reliably expose on `StockLatestQuote`), we use spread width as a proxy: a wide bid-ask spread during market hours indicates low liquidity or a stale/halted quote. This catches the practical failure mode (validating against a price nobody would trade at) without requiring timestamp parsing. A zero bid or ask also signals no real quote.

- [ ] **Step 1: Write the failing tests**

Add to `tests/v2/test_executor.py`:

```python
class TestGetLatestPriceProtection:
    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_for_zero_ask(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        mock_quote = MagicMock()
        mock_quote.ask_price = 0
        mock_quote.bid_price = 0
        mock_client_cls.return_value.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        from v2.executor import get_latest_price
        assert get_latest_price("AAPL") is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_none_for_wide_spread(self, mock_client_cls, monkeypatch):
        """Spread > 2% of midpoint should return None (low liquidity)."""
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        mock_quote = MagicMock()
        mock_quote.ask_price = 110.0
        mock_quote.bid_price = 90.0  # 20% spread
        mock_client_cls.return_value.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        from v2.executor import get_latest_price
        assert get_latest_price("AAPL") is None

    @patch("v2.executor.StockHistoricalDataClient")
    def test_returns_midpoint_for_tight_spread(self, mock_client_cls, monkeypatch):
        """Should return midpoint of bid/ask, not just ask."""
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.10
        mock_quote.bid_price = 149.90
        mock_client_cls.return_value.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        from v2.executor import get_latest_price
        price = get_latest_price("AAPL")
        assert price == Decimal("150.00")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_executor.py::TestGetLatestPriceProtection -v`
Expected: FAIL — zero-ask test returns `None` by accident (current code checks `price == 0`), but midpoint test fails since current code returns ask_price not midpoint.

- [ ] **Step 3: Rewrite `get_latest_price()` with spread/staleness checks**

In `v2/executor.py`, move the imports from inside the function to the top of the file (after line 7):

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
```

Add the spread constant after the existing imports (around line 22):

```python
MAX_SPREAD_PCT = Decimal("0.02")  # 2% max bid-ask spread
```

Replace `get_latest_price()` (lines 243-262):

```python
def get_latest_price(ticker: str) -> Optional[Decimal]:
    """Get latest quote midpoint for a ticker.

    Returns None if:
    - Quote unavailable
    - Ask or bid is zero (no real quote)
    - Bid-ask spread exceeds MAX_SPREAD_PCT (low liquidity / stale)
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockLatestQuoteRequest(symbol_or_symbols=ticker)

    try:
        quotes = client.get_stock_latest_quote(request)
        quote = quotes[ticker]

        ask = Decimal(str(quote.ask_price))
        bid = Decimal(str(quote.bid_price))

        if ask <= 0 or bid <= 0:
            return None

        midpoint = (ask + bid) / 2
        spread_pct = (ask - bid) / midpoint

        if spread_pct > MAX_SPREAD_PCT:
            return None

        return midpoint.quantize(Decimal("0.01"))
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_executor.py::TestGetLatestPriceProtection -v`
Expected: PASS

- [ ] **Step 5: Update existing trader tests that patch `get_latest_price`**

Existing tests in `tests/v2/test_trader.py` patch `v2.trader.get_latest_price` and pass a single Decimal value. Since the function now returns midpoint, the mock return values are still valid — the tests mock the function entirely and don't exercise the spread logic. No changes needed to existing tests.

Verify: `python3 -m pytest tests/v2/test_trader.py -v`
Expected: PASS

- [ ] **Step 6: Run full executor tests for regressions**

Run: `python3 -m pytest tests/v2/test_executor.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add v2/executor.py tests/v2/test_executor.py
git commit -m "feat: add spread/staleness checks to get_latest_price, use midpoint (P0)"
```

---

### Task 3: Fill Confirmation & Price Reconciliation (P0)

**Files:**
- Modify: `v2/executor.py` (add `wait_for_fill()`)
- Modify: `v2/trader.py:220-248` (use fill price for trade value and buying power)
- Modify: `tests/v2/test_executor.py`
- Modify: `tests/v2/test_trader.py`

**Context:** `v2/trader.py` already has an `order_results` dict (line 181) and already prefers fill price at logging time (line 274: `result.filled_avg_price if result and result.filled_avg_price else get_latest_price(...)`). However, `execute_market_order()` returns immediately — `filled_avg_price` is `None` for market orders at submission time. The `wait_for_fill()` function polls Alpaca until the order reaches a terminal state, then provides the actual fill data.

**Timeout behavior:** If `wait_for_fill` times out or the fill fails, the trade is still counted as `trades_executed += 1` using the quote price as fallback. This is a deliberate choice: the order was accepted by Alpaca and will likely fill — we just couldn't confirm it in time. The next session's position sync (Step 1) will reconcile actual state. A future improvement could add a "pending_confirmation" status, but that's out of scope for this fix.

- [ ] **Step 1: Write the failing tests for `wait_for_fill()`**

Add to `tests/v2/test_executor.py`:

```python
class TestWaitForFill:
    @patch("v2.executor.get_trading_client")
    @patch("v2.executor.time")
    def test_returns_filled_order(self, mock_time, mock_client):
        mock_time.monotonic.side_effect = [0, 1]  # start, first check

        mock_order = MagicMock()
        mock_order.status.value = "filled"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.25"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.1)

        assert result.success is True
        assert result.filled_qty == Decimal("10")
        assert result.filled_avg_price == Decimal("150.25")

    @patch("v2.executor.get_trading_client")
    @patch("v2.executor.time")
    def test_returns_failure_on_cancel(self, mock_time, mock_client):
        mock_time.monotonic.side_effect = [0, 1]

        mock_order = MagicMock()
        mock_order.status.value = "canceled"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.1)

        assert result.success is False

    @patch("v2.executor.get_trading_client")
    @patch("v2.executor.time")
    def test_times_out(self, mock_time, mock_client):
        # monotonic returns: start=0, check=1, check=6 (past deadline of 5)
        mock_time.monotonic.side_effect = [0, 1, 6]
        mock_time.sleep = MagicMock()

        mock_order = MagicMock()
        mock_order.status.value = "new"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from v2.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=1.0)

        assert result.success is False
        assert "timeout" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_executor.py::TestWaitForFill -v`
Expected: FAIL — `ImportError: cannot import name 'wait_for_fill'`

- [ ] **Step 3: Implement `wait_for_fill()`**

Add to `v2/executor.py` after `execute_limit_order()` (after line 240), and add `import time` at the top of the file:

```python
import time

TERMINAL_STATUSES = {"filled", "canceled", "expired", "replaced", "suspended"}


def wait_for_fill(
    order_id: str,
    timeout_seconds: float = 30.0,
    poll_interval: float = 1.0,
) -> OrderResult:
    """Poll Alpaca for order fill status.

    Returns OrderResult with actual fill price and quantity once terminal.
    """
    client = get_trading_client()
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            order = client.get_order_by_id(order_id)
        except Exception as e:
            return OrderResult(
                success=False, order_id=order_id,
                filled_qty=None, filled_avg_price=None,
                error=f"Failed to fetch order: {e}",
            )

        status = order.status.value
        if status in TERMINAL_STATUSES:
            filled_qty = Decimal(str(order.filled_qty)) if order.filled_qty else Decimal(0)
            filled_price = Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None

            if status == "filled" and filled_qty > 0:
                return OrderResult(
                    success=True, order_id=order_id,
                    filled_qty=filled_qty, filled_avg_price=filled_price,
                    error=None,
                )
            else:
                return OrderResult(
                    success=False, order_id=order_id,
                    filled_qty=filled_qty, filled_avg_price=filled_price,
                    error=f"Order {status}",
                )

        time.sleep(poll_interval)

    return OrderResult(
        success=False, order_id=order_id,
        filled_qty=None, filled_avg_price=None,
        error=f"Timeout waiting for fill after {timeout_seconds}s",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_executor.py::TestWaitForFill -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for fill-based trade value in trader**

Add to `tests/v2/test_trader.py`:

```python
class TestFillConfirmation:
    def test_uses_fill_price_for_trade_value(self, mock_db, mock_cursor):
        """Trade value and buying power should use actual fill price, not quote."""
        decision = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=10, reasoning="Entry hit", confidence="high",
            is_off_playbook=False, signal_refs=[{"type": "news_signal", "id": 1}],
            thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price", return_value=Decimal("150")), \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.wait_for_fill") as mock_fill, \
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
            mock_exec.return_value = OrderResult(
                success=True, order_id="ord-1",
                filled_qty=None, filled_avg_price=None, error=None,
            )
            # Fill comes back at $151, not the $150 quote
            mock_fill.return_value = OrderResult(
                success=True, order_id="ord-1",
                filled_qty=Decimal("10"), filled_avg_price=Decimal("151.00"),
                error=None,
            )

            result = run_trading_session(dry_run=False)

        assert result.trades_executed == 1
        # Buy value should reflect fill price: 10 * $151 = $1510
        assert result.total_buy_value == Decimal("1510.00")
```

- [ ] **Step 6: Wire fill confirmation into `v2/trader.py`**

Add import at the top (extend the existing executor import block):

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

Replace the post-execution success block (lines 220-233) with fill confirmation:

```python
        if result.success:
            trades_executed += 1
            order_ids[i] = result.order_id
            order_results[i] = result

            fill_price = price  # default to quote
            fill_qty = Decimal(str(decision.quantity))

            # Wait for actual fill (live trades only)
            if not dry_run and result.order_id and result.order_id != "DRY_RUN":
                fill_result = wait_for_fill(result.order_id)
                if fill_result.success:
                    fill_price = fill_result.filled_avg_price or price
                    fill_qty = fill_result.filled_qty or fill_qty
                    # Update order_results with fill data for logging step
                    order_results[i] = fill_result
                    logger.info("  Fill confirmed: %.4g @ $%.2f", fill_qty, fill_price)
                else:
                    logger.warning("  Fill not confirmed: %s — using quote price", fill_result.error)

            trade_value = fill_price * fill_qty

            if decision.action == "buy":
                total_buy_value += trade_value
                buying_power -= trade_value
            else:
                total_sell_value += trade_value

            status = "[DRY RUN]" if dry_run else f"Order {result.order_id}"
            logger.info("  %s - Success", status)
```

- [ ] **Step 7: Run all trader tests**

Run: `python3 -m pytest tests/v2/test_trader.py -v`
Expected: PASS. Existing tests use `dry_run=True` which skips `wait_for_fill`. Tests that use `dry_run=False` will need the `wait_for_fill` patch added — check if any fail and add the patch.

- [ ] **Step 8: Commit**

```bash
git add v2/executor.py v2/trader.py tests/v2/test_executor.py tests/v2/test_trader.py
git commit -m "feat: add fill confirmation and price reconciliation (P0)"
```

---

### Task 4: Signal Ref Validation (P1)

**Files:**
- Modify: `v2/agent.py` (add `validate_signal_refs()`)
- Modify: `v2/trader.py:294-303` (validate before inserting)
- Modify: `tests/v2/conftest.py` (add `v2.agent.get_cursor` to `mock_db` fixture)
- Modify: `tests/v2/test_agent.py`
- Modify: `tests/v2/test_trader.py`

**Architectural note:** This adds a DB dependency (`get_cursor`) to `v2/agent.py`, which was previously DB-free (pure Claude API integration). This is acceptable because signal ref validation is tightly coupled to the executor's decision output — it validates refs the LLM produced. Placing it in `agent.py` keeps validation close to the data it validates. The conftest `mock_db` fixture must be updated to patch `v2.agent.get_cursor` for tests.

- [ ] **Step 1: Update conftest `mock_db` fixture**

In `tests/v2/conftest.py`, add `v2.agent.get_cursor` to the `mock_db` fixture's patch list (line 30-38):

```python
@pytest.fixture
def mock_db(mock_cursor):
    """Patch get_cursor to yield a mock cursor."""
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("v2.database.connection.get_cursor", _get_cursor), \
         patch("v2.database.trading_db.get_cursor", _get_cursor), \
         patch("v2.database.dashboard_db.get_cursor", _get_cursor), \
         patch("v2.twitter.get_cursor", _get_cursor), \
         patch("v2.entertainment.get_cursor", _get_cursor), \
         patch("v2.bluesky.get_cursor", _get_cursor), \
         patch("v2.dashboard_publish.get_cursor", _get_cursor), \
         patch("v2.agent.get_cursor", _get_cursor), \
         patch("v2.database.connection.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/v2/test_agent.py`:

```python
from contextlib import contextmanager


class TestValidateSignalRefs:
    def test_filters_invalid_ids(self, mock_db, mock_cursor):
        """Invalid signal IDs should be removed from the refs list."""
        # First query: news_signals WHERE id = ANY([1, 99999]) → only id=1 exists
        # Second query: theses WHERE id = ANY([5]) → id=5 exists
        mock_cursor.fetchall.side_effect = [
            [{"id": 1}],
            [{"id": 5}],
        ]

        from v2.agent import validate_signal_refs
        refs = [
            {"type": "news_signal", "id": 1},
            {"type": "news_signal", "id": 99999},
            {"type": "thesis", "id": 5},
        ]
        valid = validate_signal_refs(refs)

        assert len(valid) == 2
        assert {"type": "news_signal", "id": 1} in valid
        assert {"type": "thesis", "id": 5} in valid

    def test_handles_empty(self):
        from v2.agent import validate_signal_refs
        assert validate_signal_refs([]) == []

    def test_drops_unknown_type(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = []

        from v2.agent import validate_signal_refs
        refs = [{"type": "magic_signal", "id": 1}]
        valid = validate_signal_refs(refs)

        assert valid == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_agent.py::TestValidateSignalRefs -v`
Expected: FAIL — `ImportError: cannot import name 'validate_signal_refs'`

- [ ] **Step 3: Implement `validate_signal_refs()`**

Add to `v2/agent.py` after the `validate_decision()` function (after line 303):

```python
from .database.connection import get_cursor

# Table names are compile-time constants — safe for f-string interpolation
VALID_SIGNAL_TABLES = {
    "news_signal": "news_signals",
    "macro_signal": "macro_signals",
    "thesis": "theses",
}


def validate_signal_refs(refs: list[dict]) -> list[dict]:
    """Filter signal_refs to only those whose IDs exist in the database."""
    if not refs:
        return []

    # Group IDs by type
    by_type: dict[str, list[int]] = {}
    for ref in refs:
        signal_type = ref.get("type", "")
        if signal_type in VALID_SIGNAL_TABLES:
            by_type.setdefault(signal_type, []).append(ref["id"])

    if not by_type:
        return []

    # Query each table for existing IDs
    existing: set[tuple[str, int]] = set()
    with get_cursor() as cur:
        for signal_type, ids in by_type.items():
            table = VALID_SIGNAL_TABLES[signal_type]
            cur.execute(
                f"SELECT id FROM {table} WHERE id = ANY(%s)",
                (ids,),
            )
            for row in cur.fetchall():
                existing.add((signal_type, row["id"]))

    return [
        ref for ref in refs
        if (ref.get("type"), ref.get("id")) in existing
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_agent.py::TestValidateSignalRefs -v`
Expected: PASS

- [ ] **Step 5: Wire into `v2/trader.py`**

Add import at the top of `v2/trader.py`:

```python
from .agent import (
    get_trading_decisions,
    validate_decision,
    format_decisions_for_logging,
    validate_signal_refs,
    AgentResponse,
    ExecutorDecision,
    DEFAULT_EXECUTOR_MODEL,
)
```

Replace the signal_refs block in Step 6 (lines 294-303):

```python
        # Log signal-decision links for attribution
        if decision.signal_refs:
            valid_refs = validate_signal_refs(decision.signal_refs)
            if len(valid_refs) < len(decision.signal_refs):
                logger.warning(
                    "%s: dropped %d invalid signal refs",
                    decision.ticker,
                    len(decision.signal_refs) - len(valid_refs),
                )
            if valid_refs:
                try:
                    signal_links = [
                        (decision_id, ref["type"], ref["id"])
                        for ref in valid_refs
                    ]
                    insert_decision_signals_batch(signal_links)
                except Exception as e:
                    errors.append(f"Failed to log signal links for {decision.ticker}: {e}")
```

- [ ] **Step 6: Update existing trader tests to patch `validate_signal_refs`**

Any existing tests that exercise the signal_refs logging path need the patch added. Check `test_logs_playbook_action_id` — it has `signal_refs=[{"type": "news_signal", "id": 5}]` and calls `insert_decision_signals_batch`. Add the patch:

```python
             patch("v2.trader.validate_signal_refs", side_effect=lambda refs: refs), \
```

- [ ] **Step 7: Run all tests**

Run: `python3 -m pytest tests/v2/test_agent.py tests/v2/test_trader.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add v2/agent.py v2/trader.py tests/v2/test_agent.py tests/v2/test_trader.py
git commit -m "feat: validate signal refs before inserting into attribution loop (P1)"
```

---

### Task 5: Expected Value Attribution (P1)

**Files:**
- Modify: `v2/attribution.py:64-132` (replace win-rate classification with expected value in both `get_attribution_summary()` and `build_attribution_constraints()`)
- Modify: `tests/v2/test_attribution.py` (update existing tests, add new EV tests)

**Context:** Both functions already exist in `v2/attribution.py`. The change is to use `avg_outcome_7d` (expected value) instead of `win_rate_7d` for the STRONG/WEAK classification. The retro calls out that "a signal type that wins 40% of the time but averages +5% per win and -1% per loss is highly profitable, but gets flagged WEAK."

- [ ] **Step 1: Write the failing test for EV-based classification**

Add to `tests/v2/test_attribution.py`:

```python
class TestExpectedValueAttribution:
    def test_classifies_by_expected_value_not_win_rate(self):
        """A signal with low win rate but positive avg return should be STRONG."""
        mock_rows = [
            {
                "category": "news_signal:earnings",
                "sample_size": 20,
                "win_rate_7d": Decimal("0.40"),     # 40% win rate — WEAK under old logic
                "win_rate_30d": Decimal("0.50"),
                "avg_outcome_7d": Decimal("3.5"),    # +3.5% avg — clearly profitable
                "avg_outcome_30d": Decimal("5.0"),
            },
            {
                "category": "news_signal:rumors",
                "sample_size": 20,
                "win_rate_7d": Decimal("0.60"),     # 60% win rate — looks good
                "win_rate_30d": Decimal("0.55"),
                "avg_outcome_7d": Decimal("-0.5"),   # negative avg — actually unprofitable
                "avg_outcome_30d": Decimal("-0.3"),
            },
        ]

        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import build_attribution_constraints
            result = build_attribution_constraints()

        lines = result.split("\n")
        strong_lines = [l for l in lines if "STRONG" in l]
        weak_lines = [l for l in lines if "WEAK" in l]

        assert len(strong_lines) == 1
        assert len(weak_lines) == 1
        assert "earnings" in strong_lines[0]
        assert "rumors" in weak_lines[0]

    def test_summary_uses_ev_not_win_rate(self):
        """get_attribution_summary should classify by expected value."""
        mock_rows = [
            {
                "category": "news_signal:earnings",
                "sample_size": 20,
                "win_rate_7d": Decimal("0.40"),
                "win_rate_30d": Decimal("0.50"),
                "avg_outcome_7d": Decimal("3.5"),
                "avg_outcome_30d": Decimal("5.0"),
            },
        ]
        with patch("v2.attribution.get_signal_attribution", return_value=mock_rows):
            from v2.attribution import get_attribution_summary
            result = get_attribution_summary()

        # earnings has positive EV so should be "Predictive"
        assert "Predictive" in result
        assert "earnings" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_attribution.py::TestExpectedValueAttribution -v`
Expected: FAIL — earnings has 40% win rate which is <45%, so old code puts it in WEAK, but test expects STRONG.

- [ ] **Step 3: Update `build_attribution_constraints()` with expected-value logic**

Replace `build_attribution_constraints()` in `v2/attribution.py` (lines 91-132):

```python
def build_attribution_constraints(min_samples: int = 5) -> str:
    """Format signal attribution into constraint block for strategist system prompt.

    Categories based on expected value (avg outcome), not just win rate:
      STRONG: avg_outcome_7d > 0 AND sample_size >= min_samples
      WEAK: avg_outcome_7d <= 0 AND sample_size >= min_samples
      INSUFFICIENT DATA: < min_samples
    """
    rows = get_signal_attribution()
    if not rows:
        return ""

    strong, weak, insufficient = [], [], []

    for r in rows:
        cat = r["category"]
        n = r["sample_size"]
        avg_7d = float(r["avg_outcome_7d"]) if r.get("avg_outcome_7d") else 0
        wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0

        if n < min_samples:
            insufficient.append(f"{cat} (n={n})")
        elif avg_7d > 0:
            strong.append(f"{cat} (EV={avg_7d:+.2f}%, WR={wr:.0f}%, n={n})")
        else:
            weak.append(f"{cat} (EV={avg_7d:+.2f}%, WR={wr:.0f}%, n={n})")

    lines = ["SIGNAL PERFORMANCE (last 60 days):"]
    if strong:
        lines.append(f"  STRONG (positive expected value): {', '.join(strong)}")
    if weak:
        lines.append(f"  WEAK (negative expected value): {', '.join(weak)}")
    if insufficient:
        lines.append(f"  INSUFFICIENT DATA (<{min_samples} samples): {', '.join(insufficient)}")

    lines.append("")
    lines.append("CONSTRAINT: Do not create theses primarily based on WEAK signal categories")
    lines.append("unless you have a specific reason to override (explain in thesis text).")

    return "\n".join(lines)
```

- [ ] **Step 4: Update `get_attribution_summary()` with expected-value classification**

Replace `get_attribution_summary()` in `v2/attribution.py` (lines 64-88):

```python
def get_attribution_summary() -> str:
    """Format attribution scores as advisory text for LLM context."""
    rows = get_signal_attribution()
    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    lines = ["Signal Attribution:"]
    predictive = [r for r in rows if r.get("avg_outcome_7d") and float(r["avg_outcome_7d"]) > 0]
    weak = [r for r in rows if not r.get("avg_outcome_7d") or float(r["avg_outcome_7d"]) <= 0]

    if predictive:
        lines.append("Predictive signal types (positive expected value):")
        for r in predictive:
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: EV={avg:+.2f}%, WR={wr:.0f}% (n={r['sample_size']})")

    if weak:
        lines.append("Weak/non-predictive signal types:")
        for r in weak:
            wr = float(r["win_rate_7d"]) * 100 if r.get("win_rate_7d") else 0
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: EV={avg:+.2f}%, WR={wr:.0f}% (n={r['sample_size']})")

    return "\n".join(lines)
```

- [ ] **Step 5: Update existing tests in `TestBuildAttributionConstraints`**

The existing `test_formats_strong_and_weak_categories` asserts on `"STRONG (>55% win rate)"` and `"WEAK (<45% win rate)"`. Update these assertions:

Replace in `tests/v2/test_attribution.py` `test_formats_strong_and_weak_categories`:
- `assert "STRONG (>55% win rate)" in result` → `assert "STRONG (positive expected value)" in result`
- `assert "70%" in result` → `assert "EV=+2.50%" in result`
- `assert "WEAK (<45% win rate)" in result` → `assert "WEAK (negative expected value)" in result`
- `assert "30%" in result` → `assert "EV=-1.00%" in result`
- Keep `assert "news_signal:earnings" in result` and `assert "macro_signal:fed" in result` (still valid)
- Keep `assert "INSUFFICIENT DATA" in result` and `assert "news_signal:rumor (n=3)" in result` (still valid)
- Keep `assert "CONSTRAINT:" in result` (still valid)

Update `test_threshold_boundaries`:
- The test data has `avg_outcome_7d` values. With EV logic:
  - `neutral_high` has `avg_outcome_7d=1.0` → positive → STRONG (was neutral zone before)
  - `neutral_low` has `avg_outcome_7d=-0.5` → negative → WEAK (was neutral zone before)
  - `just_above_strong` has `avg_outcome_7d=1.5` → positive → STRONG
  - `just_below_weak` has `avg_outcome_7d=-1.0` → negative → WEAK
- Update assertions to match EV-based classification:
  - `neutral_high` IS now in STRONG (positive EV)
  - `neutral_low` IS now in WEAK (negative EV)
  - Remove assertions that said these were not in the output

Update `test_formats_predictive_and_weak` in `TestGetAttributionSummary`:
- `assert "Predictive signal types:" in result` → `assert "Predictive signal types (positive expected value):" in result`
- `assert "70% win rate" in result` → `assert "EV=+2.50%" in result`
- `assert "+2.50% avg 7d return" in result` → remove (format changed)
- `assert "30% win rate" in result` → `assert "EV=-1.00%" in result`
- `assert "-1.00% avg 7d return" in result` → remove (format changed)

- [ ] **Step 6: Run all attribution tests**

Run: `python3 -m pytest tests/v2/test_attribution.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add v2/attribution.py tests/v2/test_attribution.py
git commit -m "feat: use expected value instead of win rate for signal attribution (P1)"
```

---

### Task 6: Session Idempotency (P1)

**Files:**
- Create: `db/init/011_sessions.sql`
- Modify: `v2/database/trading_db.py` (add session tracking functions)
- Modify: `v2/session.py:89-204` (add idempotency check and session registration)
- Modify: `tests/v2/test_db.py`
- Modify: `tests/v2/test_session.py`

**Design note:** Dry runs and live runs are treated as separate sessions (a dry run won't block a subsequent live run). A duplicate insert (race condition) raises a constraint violation, which is caught and degrades gracefully. The idempotency check uses a `try/except` so the system works even if the sessions table doesn't exist yet.

- [ ] **Step 1: Create the sessions table migration**

Create `db/init/011_sessions.sql`:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (session_date, dry_run)
);
```

- [ ] **Step 2: Write the failing tests for session DB functions**

Add to `tests/v2/test_db.py`:

```python
class TestSessionTracking:
    def test_get_session_for_date_returns_existing(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1, "status": "completed", "dry_run": False}
        from v2.database.trading_db import get_session_for_date
        result = get_session_for_date(date.today(), dry_run=False)

        assert result is not None
        assert result["status"] == "completed"
        assert "session_date" in mock_cursor.execute.call_args[0][0]

    def test_get_session_for_date_returns_none_when_empty(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.trading_db import get_session_for_date
        result = get_session_for_date(date.today(), dry_run=False)

        assert result is None

    def test_insert_session_record(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 42}
        from v2.database.trading_db import insert_session_record
        result = insert_session_record(date.today(), dry_run=False)

        assert result == 42
        assert "INSERT INTO sessions" in mock_cursor.execute.call_args[0][0]

    def test_complete_session(self, mock_db, mock_cursor):
        from v2.database.trading_db import complete_session
        complete_session(42, "completed")

        sql = mock_cursor.execute.call_args[0][0]
        assert "UPDATE sessions" in sql
        assert "completed_at" in sql
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_db.py::TestSessionTracking -v`
Expected: FAIL — `ImportError: cannot import name 'get_session_for_date'`

- [ ] **Step 4: Implement session tracking in `v2/database/trading_db.py`**

Add to the end of `v2/database/trading_db.py`:

```python
# --- Sessions ---

def get_session_for_date(session_date, dry_run: bool = False) -> dict | None:
    """Check if a session already ran for this date and mode."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, status, dry_run FROM sessions WHERE session_date = %s AND dry_run = %s",
            (session_date, dry_run),
        )
        return cur.fetchone()


def insert_session_record(session_date, dry_run: bool = False) -> int:
    """Insert a new session record. Returns session ID."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO sessions (session_date, dry_run)
            VALUES (%s, %s)
            RETURNING id
        """, (session_date, dry_run))
        return cur.fetchone()["id"]


def complete_session(session_id: int, status: str = "completed"):
    """Mark a session as completed or failed."""
    with get_cursor() as cur:
        cur.execute("""
            UPDATE sessions SET completed_at = NOW(), status = %s
            WHERE id = %s
        """, (status, session_id))
```

- [ ] **Step 5: Run DB function tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_db.py::TestSessionTracking -v`
Expected: PASS

- [ ] **Step 6: Write the failing test for session idempotency**

Add to `tests/v2/test_session.py`:

```python
class TestSessionIdempotency:
    def test_skips_if_already_completed(self):
        """Session should skip all stages if a completed session exists for today."""
        with patch("v2.session.get_session_for_date") as mock_get, \
             patch("v2.session.insert_session_record"), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill") as mock_backfill, \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session") as mock_trade:

            mock_get.return_value = {"id": 1, "status": "completed", "dry_run": False}

            result = run_session(dry_run=False)

        # No stages should have run
        mock_backfill.assert_not_called()
        mock_pipeline.assert_not_called()
        mock_strat.assert_not_called()
        mock_trade.assert_not_called()
        assert result.trading_result is None
        assert result.pipeline_result is None

    def test_proceeds_if_no_prior_session(self):
        """Session should run normally if no prior session exists."""
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

            result = run_session(dry_run=False)

        mock_pipeline.assert_called_once()

    def test_dry_run_does_not_block_live_run(self):
        """A completed dry run should not block a live run."""
        with patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=1), \
             patch("v2.session.complete_session"), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=False)

        # get_session_for_date is called with dry_run=False
        mock_trade.assert_called_once()

    def test_degrades_gracefully_if_sessions_table_missing(self):
        """Session should proceed if the sessions table doesn't exist yet."""
        with patch("v2.session.get_session_for_date", side_effect=Exception("relation does not exist")), \
             patch("v2.session.insert_session_record", side_effect=Exception("relation does not exist")), \
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
```

- [ ] **Step 7: Wire idempotency check into `v2/session.py`**

Add imports at the top of `v2/session.py`:

```python
from datetime import date
from .database.trading_db import get_session_for_date, insert_session_record, complete_session
```

At the top of `run_session()`, after `result = SessionResult(...)` (line 91), add:

```python
    # Idempotency check
    existing = None
    try:
        existing = get_session_for_date(date.today(), dry_run=dry_run)
    except Exception:
        pass  # Table may not exist yet; proceed without guard

    if existing and existing["status"] == "completed":
        logger.info("Session already completed for today (id=%d, dry_run=%s) — skipping",
                     existing["id"], dry_run)
        result.duration_seconds = time.monotonic() - start
        return result

    # Register this session
    session_id = None
    try:
        session_id = insert_session_record(date.today(), dry_run=dry_run)
    except Exception as e:
        logger.warning("Could not register session: %s — continuing without idempotency guard", e)
```

At the end of `run_session()`, before `return result` (line 204), add:

```python
    if session_id:
        try:
            complete_session(session_id, "failed" if result.has_errors else "completed")
        except Exception as e:
            logger.warning("Could not update session status: %s", e)
```

- [ ] **Step 8: Update existing session tests to patch new imports**

Existing tests that call `run_session()` will need patches for the new DB functions. Add to each existing test's `with patch(...)` block:

```python
             patch("v2.session.get_session_for_date", return_value=None), \
             patch("v2.session.insert_session_record", return_value=1), \
             patch("v2.session.complete_session"), \
```

This applies to all tests in `TestRunSession`, `TestStage4StrategyReflection`, `TestStage5Twitter`, `TestStage5Bluesky`, and `TestStage6Dashboard`.

- [ ] **Step 9: Run all session tests**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add db/init/011_sessions.sql v2/database/trading_db.py v2/session.py tests/v2/test_db.py tests/v2/test_session.py
git commit -m "feat: add session-level idempotency to prevent duplicate daily runs (P1)"
```

---

### Task 7: Buying Power Tracking from Fills (P1)

This is largely addressed by Task 3 (fill confirmation). The remaining piece is verifying that `buying_power` in the execution loop subtracts the *fill-based* trade value instead of the quote-based one across multiple trades, so the second trade's validation uses the correct remaining buying power.

**Files:**
- Modify: `tests/v2/test_trader.py` (add multi-trade buying power test)

- [ ] **Step 1: Write the test for buying power tracking across multiple trades**

Add to `tests/v2/test_trader.py`:

```python
class TestBuyingPowerTracking:
    def test_buying_power_decrements_by_fill_price_across_trades(self, mock_db, mock_cursor):
        """After a fill at $151, buying power should reflect fill price for subsequent validation."""
        buy1 = ExecutorDecision(
            playbook_action_id=1, ticker="AAPL", action="buy",
            quantity=10, reasoning="Entry", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )
        buy2 = ExecutorDecision(
            playbook_action_id=2, ticker="MSFT", action="buy",
            quantity=5, reasoning="Entry", confidence="high",
            is_off_playbook=False, signal_refs=[], thesis_id=None,
        )

        with patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input") as mock_build, \
             patch("v2.trader.get_trading_decisions") as mock_decisions, \
             patch("v2.trader.get_latest_price", return_value=Decimal("150")), \
             patch("v2.trader.validate_decision") as mock_val, \
             patch("v2.trader.execute_market_order") as mock_exec, \
             patch("v2.trader.wait_for_fill") as mock_fill, \
             patch("v2.trader.validate_signal_refs", side_effect=lambda refs: refs), \
             patch("v2.trader.insert_decision", return_value=1), \
             patch("v2.trader.insert_decision_signals_batch"), \
             patch("v2.trader.get_positions", return_value=[]):

            mock_acct.return_value = {
                "portfolio_value": Decimal("100000"),
                "cash": Decimal("50000"),
                "buying_power": Decimal("5000"),
            }
            mock_build.return_value = ExecutorInput(
                playbook_actions=[], positions=[], account={},
                attribution_summary={}, recent_outcomes=[],
                market_outlook="", risk_notes="",
            )
            mock_decisions.return_value = AgentResponse(
                decisions=[buy1, buy2], thesis_invalidations=[],
                market_summary="Active", risk_assessment="Low",
            )
            mock_val.return_value = (True, "OK")
            mock_exec.return_value = OrderResult(
                success=True, order_id="ord-1",
                filled_qty=None, filled_avg_price=None, error=None,
            )
            # AAPL fills at $151 (not $150 quote)
            mock_fill.return_value = OrderResult(
                success=True, order_id="ord-1",
                filled_qty=Decimal("10"), filled_avg_price=Decimal("151.00"),
                error=None,
            )

            result = run_trading_session(dry_run=False)

        # validate_decision should be called twice
        assert mock_val.call_count == 2

        # Second call's buying_power arg should be 5000 - (151 * 10) = 3490
        second_call_args = mock_val.call_args_list[1]
        buying_power_arg = second_call_args[0][1]  # second positional arg
        assert buying_power_arg == Decimal("3490.00")
```

- [ ] **Step 2: Run test**

Run: `python3 -m pytest tests/v2/test_trader.py::TestBuyingPowerTracking -v`

If it passes, Task 3's wiring already handles this correctly. If it fails, adjust the buying power subtraction in the execution loop to use `fill_price * fill_qty` (should already be done in Task 3's wiring).

- [ ] **Step 3: Commit**

```bash
git add tests/v2/test_trader.py
git commit -m "test: verify buying power tracks from actual fill prices (P1)"
```

---

## Post-Implementation

After all tasks are complete:

1. Run full v2 test suite: `python3 -m pytest tests/v2/ -v`
2. Run with coverage: `python3 -m pytest tests/v2/ --cov=v2`
3. Verify no regressions in existing tests
4. Run the full project test suite: `python3 -m pytest tests/ -v` (to ensure v1 `trading/` tests still pass)
