# Retro P0/P1 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all P0 and P1 issues from the v2 pipeline retrospective — fill confirmation, stale quote protection, market hours gating, signal ref validation, expected-value attribution, session idempotency, and buying power tracking.

**Architecture:** Seven independent fixes across the execution, attribution, and session layers. Each fix is self-contained: add a function or guard, wire it in, test it. No new dependencies — everything uses existing Alpaca SDK and psycopg2.

**Tech Stack:** Python 3, Alpaca SDK (`alpaca-py`), psycopg2, pytest

**Important notes:**
- The canonical source package is `trading/` (not `v2/`). All test imports use `from trading.*`.
- DB access is via `trading.db` (not `trading.database.trading_db`).
- The decision dataclass is `TradingDecision` (not `ExecutorDecision`).
- Tests use class-based grouping (`class TestXxx:`).
- The `_patch_all()` pattern in `tests/test_trader.py` provides all trader mocks — new tests should extend this dict.
- The `mock_db` fixture in `tests/test_attribution.py` wraps `get_cursor()` as a contextmanager — follow this pattern for any test needing cursor mocks.

---

## File Map

| File | Change | Purpose |
|------|--------|---------|
| `trading/executor.py` | Add `wait_for_fill()`, `is_market_open()`, add staleness/spread checks to `get_latest_price()` | P0: fill confirmation, market hours, stale quotes |
| `trading/trader.py` | Wire in market hours gate, fill-based price logging, buying power from fills | P0+P1: orchestration changes |
| `trading/agent.py` | Add `validate_signal_refs()` | P1: signal ref validation |
| `trading/attribution.py` | Replace win-rate with expected value metric | P1: better signal filtering |
| `trading/session.py` | Add session-level idempotency check | P1: no duplicate runs |
| `trading/db.py` | Add `get_session_for_date()`, `insert_session_record()`, `complete_session()` | P1: session tracking |
| `db/init/011_sessions.sql` | `sessions` table | P1: session dedup schema |
| `tests/test_executor.py` | Tests for new executor functions | All P0 tests |
| `tests/test_trader.py` | Tests for wiring changes | P0+P1 tests |
| `tests/test_agent.py` | Tests for signal ref validation | P1 tests |
| `tests/test_attribution.py` | Tests for expected value metric | P1 tests |
| `tests/test_session.py` | Tests for idempotency | P1 tests |

---

### Task 1: Market Hours Check (P0)

**Files:**
- Modify: `trading/executor.py` (add `is_market_open()`)
- Modify: `trading/trader.py:57-81` (gate on market open)
- Test: `tests/test_executor.py`, `tests/test_trader.py`

- [ ] **Step 1: Write the failing test for `is_market_open()`**

Add to `tests/test_executor.py`:

```python
class TestIsMarketOpen:
    @patch("trading.executor.get_trading_client")
    def test_returns_true_when_open(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_client.return_value.get_clock.return_value = mock_clock

        from trading.executor import is_market_open
        assert is_market_open() is True

    @patch("trading.executor.get_trading_client")
    def test_returns_false_when_closed(self, mock_client):
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_client.return_value.get_clock.return_value = mock_clock

        from trading.executor import is_market_open
        assert is_market_open() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_executor.py::TestIsMarketOpen -v`
Expected: FAIL — `ImportError: cannot import name 'is_market_open'`

- [ ] **Step 3: Implement `is_market_open()`**

Add to `trading/executor.py`:

```python
def is_market_open() -> bool:
    """Check if the market is currently open via Alpaca clock API."""
    client = get_trading_client()
    clock = client.get_clock()
    return clock.is_open
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_executor.py::TestIsMarketOpen -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for market hours gate in trader**

Add `"is_market_open"` to `_patch_all()` in `tests/test_trader.py`:

```python
# In _patch_all(), add:
"is_market_open": patch("trading.trader.is_market_open", return_value=True),
```

Then add the test:

```python
class TestMarketHoursGate:
    def test_aborts_when_market_closed_and_not_dry_run(self):
        patches = _patch_all()
        patches["is_market_open"] = patch("trading.trader.is_market_open", return_value=False)

        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"], \
             patches["is_market_open"]:

            result = run_trading_session(dry_run=False)

        assert result.trades_executed == 0
        assert any("market" in e.lower() for e in result.errors)

    def test_allows_dry_run_when_market_closed(self):
        patches = _patch_all()
        patches["is_market_open"] = patch("trading.trader.is_market_open", return_value=False)

        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"], \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"], \
             patches["is_market_open"]:

            m_dec.return_value = make_agent_response(decisions=[])

            result = run_trading_session(dry_run=True)

        # Dry run should proceed even when market is closed
        assert not any("market" in e.lower() for e in result.errors)
```

- [ ] **Step 6: Wire market hours check into `run_trading_session()`**

In `trading/trader.py`, add import and gate between Step 1 (sync) and Step 2 (snapshot):

```python
from .executor import is_market_open

# After syncing, before taking snapshot:
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

Run: `python3 -m pytest tests/test_executor.py::TestIsMarketOpen tests/test_trader.py::TestMarketHoursGate -v`
Expected: PASS

- [ ] **Step 8: Run full test suite to check for regressions**

Run: `python3 -m pytest tests/test_executor.py tests/test_trader.py -v`
Expected: all existing tests still PASS (existing tests use `_patch_all()` which now needs the `is_market_open` patch — update `_patch_all()` and add it to existing `with` blocks)

- [ ] **Step 9: Commit**

```bash
git add trading/executor.py trading/trader.py tests/test_executor.py tests/test_trader.py
git commit -m "feat: gate live trading on market hours check (P0)"
```

---

### Task 2: Stale Quote Protection (P0)

**Files:**
- Modify: `trading/executor.py:278-295` (enhance `get_latest_price()`)
- Test: `tests/test_executor.py`

**Note:** This changes `get_latest_price()` from returning ask price to returning bid-ask midpoint. This is intentional — midpoint is a better estimate of fair value. Callers (trader.py validation and logging) benefit from a more accurate price. Since midpoint < ask, buy validations become slightly more permissive (acceptable — the fill price may be anywhere in the spread).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_executor.py`:

```python
class TestGetLatestPriceProtection:
    @patch("trading.executor.StockHistoricalDataClient")
    def test_returns_none_for_zero_ask(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        mock_quote = MagicMock()
        mock_quote.ask_price = 0
        mock_quote.bid_price = 0
        mock_client_cls.return_value.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        assert get_latest_price("AAPL") is None

    @patch("trading.executor.StockHistoricalDataClient")
    def test_returns_none_for_wide_spread(self, mock_client_cls, monkeypatch):
        """Spread > 2% of midpoint should return None (low liquidity)."""
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        mock_quote = MagicMock()
        mock_quote.ask_price = 110.0
        mock_quote.bid_price = 90.0  # 20% spread
        mock_client_cls.return_value.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        assert get_latest_price("AAPL") is None

    @patch("trading.executor.StockHistoricalDataClient")
    def test_returns_midpoint_for_tight_spread(self, mock_client_cls, monkeypatch):
        """Should return midpoint of bid/ask, not just ask."""
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.10
        mock_quote.bid_price = 149.90
        mock_client_cls.return_value.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        price = get_latest_price("AAPL")
        assert price == Decimal("150.00")
```

Note: We need to move the `from alpaca.data.historical import StockHistoricalDataClient` to the top of `trading/executor.py` (out of the function) so it can be patched. Alternatively, patch the full import path inside the function. The test above patches `trading.executor.StockHistoricalDataClient` which requires the import to be at module level.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_executor.py::TestGetLatestPriceProtection -v`
Expected: FAIL

- [ ] **Step 3: Rewrite `get_latest_price()` with spread/staleness checks**

In `trading/executor.py`, move the imports to the top of the file:

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
```

Replace `get_latest_price()`:

```python
MAX_SPREAD_PCT = Decimal("0.02")  # 2% max bid-ask spread


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

Run: `python3 -m pytest tests/test_executor.py::TestGetLatestPriceProtection -v`
Expected: PASS

- [ ] **Step 5: Fix existing `TestGetLatestPrice` tests for midpoint change**

The existing tests at `tests/test_executor.py:454-479` need two updates:
1. Change patch target from `"alpaca.data.historical.StockHistoricalDataClient"` to `"trading.executor.StockHistoricalDataClient"` (since the import is now at module level)
2. Add `mock_quote.bid_price = 155.25` to `test_returns_decimal_price` (midpoint of bid=ask gives same value)
3. The `test_returns_none_on_exception` test should work as-is (exception path unchanged)

- [ ] **Step 6: Run full executor tests for regressions**

Run: `python3 -m pytest tests/test_executor.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add trading/executor.py tests/test_executor.py
git commit -m "feat: add spread/staleness checks to get_latest_price, use midpoint (P0)"
```

---

### Task 3: Fill Confirmation & Price Reconciliation (P0)

**Files:**
- Modify: `trading/executor.py` (add `wait_for_fill()`)
- Modify: `trading/trader.py:198-235` (use fill price for logging and buying power)
- Test: `tests/test_executor.py`, `tests/test_trader.py`

- [ ] **Step 1: Write the failing tests for `wait_for_fill()`**

Add to `tests/test_executor.py`:

```python
class TestWaitForFill:
    @patch("trading.executor.get_trading_client")
    @patch("trading.executor.time")
    def test_returns_filled_order(self, mock_time, mock_client):
        mock_time.monotonic.side_effect = [0, 1]  # start, first check

        mock_order = MagicMock()
        mock_order.status.value = "filled"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.25"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from trading.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.1)

        assert result.success is True
        assert result.filled_qty == Decimal("10")
        assert result.filled_avg_price == Decimal("150.25")

    @patch("trading.executor.get_trading_client")
    @patch("trading.executor.time")
    def test_returns_failure_on_cancel(self, mock_time, mock_client):
        mock_time.monotonic.side_effect = [0, 1]

        mock_order = MagicMock()
        mock_order.status.value = "canceled"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from trading.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=0.1)

        assert result.success is False

    @patch("trading.executor.get_trading_client")
    @patch("trading.executor.time")
    def test_times_out(self, mock_time, mock_client):
        # monotonic returns: start=0, check=1, check=6 (past deadline of 5)
        mock_time.monotonic.side_effect = [0, 1, 6]
        mock_time.sleep = MagicMock()

        mock_order = MagicMock()
        mock_order.status.value = "new"
        mock_client.return_value.get_order_by_id.return_value = mock_order

        from trading.executor import wait_for_fill
        result = wait_for_fill("order-123", timeout_seconds=5, poll_interval=1.0)

        assert result.success is False
        assert "timeout" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_executor.py::TestWaitForFill -v`
Expected: FAIL — `ImportError: cannot import name 'wait_for_fill'`

- [ ] **Step 3: Implement `wait_for_fill()`**

Add to `trading/executor.py`:

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

Run: `python3 -m pytest tests/test_executor.py::TestWaitForFill -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for fill-based logging in trader**

Add `"wait_for_fill"` to `_patch_all()` in `tests/test_trader.py`:

```python
# In _patch_all(), add:
"wait_for_fill": patch("trading.trader.wait_for_fill"),
```

Add test:

```python
class TestFillConfirmation:
    def test_logs_fill_price_not_quote(self):
        """Decisions should be logged with actual fill price, not the quote price."""
        patches = _patch_all()

        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"], \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"], \
             patches["execute"] as m_exec, \
             patches["latest_price"], \
             patches["get_positions"], \
             patches["insert_decision"] as m_insert, \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"], \
             patches["is_market_open"], \
             patches["wait_for_fill"] as m_fill:

            buy = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            m_dec.return_value = make_agent_response(decisions=[buy])

            # Quote is $150, but fill is $151
            m_fill.return_value = OrderResult(
                success=True, order_id="ord-1",
                filled_qty=Decimal("10"), filled_avg_price=Decimal("151.00"),
                error=None,
            )

            result = run_trading_session(dry_run=False)

            assert result.trades_executed == 1
            # Check insert_decision was called with fill price $151, not quote $150
            call_kwargs = m_insert.call_args
            assert call_kwargs[1]["price"] == Decimal("151.00") or call_kwargs[0][4] == Decimal("151.00")
```

- [ ] **Step 6: Wire fill confirmation into trader.py**

In `trading/trader.py`, add import:

```python
from .executor import is_market_open, wait_for_fill
```

Replace the post-execution block (after `execute_market_order()` succeeds, lines 208-219) with:

```python
        if result.success:
            fill_price = price  # default to quote
            fill_qty = Decimal(str(decision.quantity))

            if not dry_run and result.order_id:
                fill_result = wait_for_fill(result.order_id)
                if fill_result.success:
                    fill_price = fill_result.filled_avg_price or price
                    fill_qty = fill_result.filled_qty or fill_qty
                    logger.info("  Fill confirmed: %.4g @ $%.2f", fill_qty, fill_price)
                else:
                    logger.warning("  Fill not confirmed: %s — using quote price", fill_result.error)

            trades_executed += 1
            trade_value = fill_price * fill_qty

            if decision.action == "buy":
                total_buy_value += trade_value
                buying_power -= trade_value
            else:
                total_sell_value += trade_value

            status = "[DRY RUN]" if dry_run else f"Order {result.order_id}"
            logger.info("  %s - Success", status)
```

Also add a `fill_prices` dict to store fill prices for the logging step. Before the execution loop:

```python
    fill_prices = {}
```

Inside the success block, after setting fill_price:

```python
            fill_prices[decision.ticker] = fill_price
```

In Step 6 (logging), replace `price = get_latest_price(decision.ticker)` (line 258) with:

```python
            price = fill_prices.get(decision.ticker) or get_latest_price(decision.ticker)
```

- [ ] **Step 7: Run all trader tests**

Run: `python3 -m pytest tests/test_trader.py -v`
Expected: PASS (update existing tests to include `wait_for_fill` in their `with` blocks)

- [ ] **Step 8: Commit**

```bash
git add trading/executor.py trading/trader.py tests/test_executor.py tests/test_trader.py
git commit -m "feat: add fill confirmation and price reconciliation (P0)"
```

---

### Task 4: Signal Ref Validation (P1)

**Files:**
- Modify: `trading/agent.py` (add `validate_signal_refs()`)
- Modify: `trading/trader.py:275-284` (validate before inserting)
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
from contextlib import contextmanager


class TestValidateSignalRefs:
    def test_filters_invalid_ids(self):
        """Invalid signal IDs should be removed from the refs list."""
        mock_cursor = MagicMock()
        # First call: news_signals WHERE id = ANY([1, 99999]) → only id=1 exists
        # Second call: theses WHERE id = ANY([5]) → id=5 exists
        mock_cursor.fetchall.side_effect = [
            [{"id": 1}],
            [{"id": 5}],
        ]

        @contextmanager
        def _mock_cursor():
            yield mock_cursor

        with patch("trading.agent.get_cursor", _mock_cursor):
            from trading.agent import validate_signal_refs
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
        from trading.agent import validate_signal_refs
        assert validate_signal_refs([]) == []

    def test_drops_unknown_type(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        @contextmanager
        def _mock_cursor():
            yield mock_cursor

        with patch("trading.agent.get_cursor", _mock_cursor):
            from trading.agent import validate_signal_refs
            refs = [{"type": "magic_signal", "id": 1}]
            valid = validate_signal_refs(refs)

        assert valid == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent.py::TestValidateSignalRefs -v`
Expected: FAIL — `ImportError: cannot import name 'validate_signal_refs'`

- [ ] **Step 3: Implement `validate_signal_refs()`**

Add to `trading/agent.py`:

```python
from .db import get_cursor

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

Run: `python3 -m pytest tests/test_agent.py::TestValidateSignalRefs -v`
Expected: PASS

- [ ] **Step 5: Wire into trader.py**

In `trading/trader.py`, add import:

```python
from .agent import validate_signal_refs
```

Replace the signal_refs logging block (lines 275-284):

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

- [ ] **Step 6: Add `validate_signal_refs` to `_patch_all()` in test_trader.py**

```python
"validate_refs": patch("trading.trader.validate_signal_refs", side_effect=lambda refs: refs),
```

- [ ] **Step 7: Run all tests**

Run: `python3 -m pytest tests/test_agent.py tests/test_trader.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add trading/agent.py trading/trader.py tests/test_agent.py tests/test_trader.py
git commit -m "feat: validate signal refs before inserting into attribution loop (P1)"
```

---

### Task 5: Expected Value Attribution (P1)

**Files:**
- Modify: `trading/attribution.py:91-132` (replace win-rate with expected value)
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_attribution.py`:

```python
from trading.attribution import build_attribution_constraints


class TestExpectedValueAttribution:
    def test_classifies_by_expected_value_not_win_rate(self, mock_db):
        """A signal with low win rate but positive avg return should be STRONG."""
        mock_db._mock_get_attr.return_value = [
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

        result = build_attribution_constraints()
        lines = result.split("\n")

        strong_lines = [l for l in lines if "STRONG" in l]
        weak_lines = [l for l in lines if "WEAK" in l]

        assert len(strong_lines) == 1
        assert len(weak_lines) == 1
        assert "earnings" in strong_lines[0]
        assert "rumors" in weak_lines[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_attribution.py::TestExpectedValueAttribution -v`
Expected: FAIL — `ImportError: cannot import name 'build_attribution_constraints'` (function doesn't exist yet in `trading/attribution.py`)

- [ ] **Step 3: Add `build_attribution_constraints()` with expected-value logic**

Add to `trading/attribution.py` (this is a new function — `v2/attribution.py` has a win-rate version, but `trading/attribution.py` only has `compute_signal_attribution()` and `get_attribution_summary()`):

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

Also update `get_attribution_summary()` to use EV-based classification:

```python
def get_attribution_summary() -> str:
    """Format attribution scores as advisory text for LLM context."""
    rows = get_signal_attribution()
    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    lines = ["Signal Attribution:"]
    predictive = [r for r in rows if r.get("avg_outcome_7d") and r["avg_outcome_7d"] > 0]
    weak = [r for r in rows if not r.get("avg_outcome_7d") or r["avg_outcome_7d"] <= 0]

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_attribution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trading/attribution.py tests/test_attribution.py
git commit -m "feat: use expected value instead of win rate for signal attribution (P1)"
```

---

### Task 6: Session Idempotency (P1)

**Files:**
- Create: `db/init/011_sessions.sql`
- Modify: `trading/db.py` (add session tracking functions)
- Modify: `trading/session.py:75-92` (check for existing session)
- Test: `tests/test_session.py`

**Design note:** Dry runs and live runs are treated as separate sessions (a dry run won't block a subsequent live run). A duplicate insert (race condition) raises a constraint violation, which is caught and degrades gracefully.

- [ ] **Step 1: Create the sessions table migration**

```sql
-- db/init/011_sessions.sql
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL UNIQUE,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    dry_run BOOLEAN NOT NULL DEFAULT FALSE
);
```

Note: The UNIQUE on session_date means only one session per day. Since we want dry_run and live to be separate, change to:

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

- [ ] **Step 2: Write the failing tests for session tracking functions**

Add to `tests/test_session.py`:

```python
from contextlib import contextmanager


class TestSessionIdempotency:
    def test_get_session_for_date_returns_existing(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1, "status": "completed", "dry_run": False}

        @contextmanager
        def _mock_cursor():
            yield mock_cursor

        with patch("trading.db.get_cursor", _mock_cursor):
            from trading.db import get_session_for_date
            result = get_session_for_date(date.today(), dry_run=False)

        assert result is not None
        assert result["status"] == "completed"

    def test_get_session_for_date_returns_none_when_empty(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        @contextmanager
        def _mock_cursor():
            yield mock_cursor

        with patch("trading.db.get_cursor", _mock_cursor):
            from trading.db import get_session_for_date
            result = get_session_for_date(date.today(), dry_run=False)

        assert result is None

    def test_run_session_skips_if_already_completed(self):
        """Session should skip all stages if a completed session exists for today."""
        # Patch the new DB functions + existing session deps
        # trading/session.py imports: check_dependencies, run_pipeline,
        # run_strategist_loop, run_trading_session, run_twitter_stage
        with patch("trading.session.get_session_for_date") as mock_get, \
             patch("trading.session.insert_session_record"), \
             patch("trading.session.complete_session"), \
             patch("trading.session.check_dependencies", return_value=True), \
             patch("trading.session.run_pipeline"), \
             patch("trading.session.run_strategist_loop"), \
             patch("trading.session.run_trading_session"), \
             patch("trading.session.run_twitter_stage"):

            mock_get.return_value = {"id": 1, "status": "completed", "dry_run": False}

            result = run_session(dry_run=False)

        # No stages should have run — early return from idempotency check
        assert result.trading_result is None
        assert result.pipeline_result is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_session.py::TestSessionIdempotency -v`
Expected: FAIL — `ImportError: cannot import name 'get_session_for_date'`

- [ ] **Step 4: Implement session tracking in trading/db.py**

Add to `trading/db.py`:

```python
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

Run: `python3 -m pytest tests/test_session.py::TestSessionIdempotency::test_get_session_for_date_returns_existing tests/test_session.py::TestSessionIdempotency::test_get_session_for_date_returns_none_when_empty -v`
Expected: PASS

- [ ] **Step 6: Wire idempotency check into `run_session()`**

In `trading/session.py`, add imports:

```python
from datetime import date
from .db import get_session_for_date, insert_session_record, complete_session
```

At the top of `run_session()`, after `start = time.monotonic()` and `result = SessionResult(...)`:

```python
    # Idempotency check
    existing = None
    try:
        existing = get_session_for_date(date.today(), dry_run=dry_run)
    except Exception:
        pass  # Table may not exist yet; proceed without guard

    if existing and existing["status"] == "completed":
        logger.info("Session already completed for today (id=%d, dry_run=%s) — skipping", existing["id"], dry_run)
        result.duration_seconds = time.monotonic() - start
        return result

    # Register this session
    session_id = None
    try:
        session_id = insert_session_record(date.today(), dry_run=dry_run)
    except Exception as e:
        logger.warning("Could not register session: %s — continuing without idempotency guard", e)
```

At the end of `run_session()`, before `return result`:

```python
    if session_id:
        try:
            complete_session(session_id, "failed" if result.has_errors else "completed")
        except Exception as e:
            logger.warning("Could not update session status: %s", e)
```

- [ ] **Step 7: Run session idempotency test**

Run: `python3 -m pytest tests/test_session.py::TestSessionIdempotency::test_run_session_skips_if_already_completed -v`
Expected: PASS

- [ ] **Step 8: Run full session tests for regressions**

Run: `python3 -m pytest tests/test_session.py -v`
Expected: PASS (existing tests will need patches for the new imports: `get_session_for_date`, `insert_session_record`, `complete_session`)

- [ ] **Step 9: Commit**

```bash
git add db/init/011_sessions.sql trading/db.py trading/session.py tests/test_session.py
git commit -m "feat: add session-level idempotency to prevent duplicate daily runs (P1)"
```

---

### Task 7: Buying Power Tracking from Fills (P1)

This is largely addressed by Task 3 (fill confirmation). The remaining piece is that `buying_power` in the execution loop now subtracts the *fill-based* trade value instead of the quote-based one. This task verifies it works correctly across multiple trades in a single session.

**Files:**
- Test: `tests/test_trader.py`

- [ ] **Step 1: Write the test for buying power tracking across multiple trades**

```python
class TestBuyingPowerTracking:
    def test_buying_power_decrements_by_fill_price_across_trades(self):
        """After a fill at $151, buying power should reflect fill price for subsequent validation."""
        patches = _patch_all()

        with patches["sync_positions"], \
             patches["sync_orders"], \
             patches["get_account"] as m_acct, \
             patches["snapshot"], \
             patches["build_ctx"], \
             patches["get_decisions"] as m_dec, \
             patches["validate"] as m_val, \
             patches["execute"] as m_exec, \
             patches["latest_price"] as m_price, \
             patches["get_positions"], \
             patches["insert_decision"], \
             patches["insert_signals_batch"], \
             patches["close_thesis"], \
             patches["format_log"], \
             patches["calc_size"], \
             patches["is_market_open"], \
             patches["wait_for_fill"] as m_fill, \
             patches["validate_refs"]:

            # Two buy decisions
            buy1 = make_trading_decision(action="buy", ticker="AAPL", quantity=10)
            buy2 = make_trading_decision(action="buy", ticker="MSFT", quantity=5)
            m_dec.return_value = make_agent_response(decisions=[buy1, buy2])

            # Starting buying power: $5000
            account = dict(ACCOUNT_INFO)
            account["buying_power"] = Decimal("5000")
            m_acct.return_value = account

            # AAPL fills at $151 (not $150 quote)
            m_fill.return_value = OrderResult(
                success=True, order_id="ord-1",
                filled_qty=Decimal("10"), filled_avg_price=Decimal("151.00"),
                error=None,
            )

            result = run_trading_session(dry_run=False)

            # validate_decision should be called twice
            assert m_val.call_count == 2

            # Second call should have buying_power = 5000 - (151 * 10) = 3490
            second_call = m_val.call_args_list[1]
            buying_power_arg = second_call[0][1]  # second positional arg
            assert buying_power_arg == Decimal("3490")
```

- [ ] **Step 2: Run test**

Run: `python3 -m pytest tests/test_trader.py::TestBuyingPowerTracking -v`

If it passes, Task 3's wiring already handles this. If it fails, adjust the buying power subtraction in the execution loop to use `fill_price * fill_qty`.

- [ ] **Step 3: Commit if any changes needed**

```bash
git add trading/trader.py tests/test_trader.py
git commit -m "test: verify buying power tracks from actual fill prices (P1)"
```

---

## Post-Implementation

After all tasks are complete:

1. Run full test suite: `python3 -m pytest tests/ -v`
2. Run with coverage: `python3 -m pytest tests/ --cov=trading --cov=dashboard`
3. Verify no regressions in existing tests
4. Sync changes to `v2/` directory if it's meant to stay in sync with `trading/`
