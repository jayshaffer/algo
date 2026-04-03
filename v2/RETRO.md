# V2 Pipeline Retrospective

Audit of P0 and P1 issues in the v2 trading pipeline, organized by the specific execution path where each bug lives.

---

## P0: Money-Risk Issues

These bugs can cause real financial loss on every session run.

---

### P0-1: Orders fire without fill confirmation

**Where:** `executor.py:145-190` → `trader.py:213-218` → `trader.py:220-230`

`execute_market_order()` calls `submit_order()` and returns immediately. At submission time, `OrderResult.filled_qty` and `filled_avg_price` are `None` for market orders — the order is accepted, not filled. The pipeline treats this as a completed trade:

- `trader.py:224` computes `trade_value = price * qty` using the quote price, not the fill price
- `trader.py:228` deducts this estimated value from `buying_power`
- `trader.py:274` tries to use `result.filled_avg_price` for logging but falls back to `get_latest_price()` since the fill hasn't happened yet

**Consequence:** Every logged decision records a price that may not match reality. The attribution system backfills outcomes against this phantom entry price, so the entire learning loop is training on inaccurate data. If the fill price is significantly worse (fast-moving stock, low liquidity), the system doesn't know it overpaid.

**Fix scope:** Add a `wait_for_fill()` function in `executor.py` that polls the order status until filled or timed out. Wire it into `trader.py` after `execute_market_order()` so buying power updates and decision logging use the actual fill price.

---

### P0-2: Stale quotes used for validation and logging

**Where:** `executor.py:243-262` → `trader.py:192` → `agent.py:279`

`get_latest_price()` fetches `quote.ask_price` with zero staleness checks:

```python
# executor.py:254-260
quotes = client.get_stock_latest_quote(request)
quote = quotes[ticker]
price = Decimal(str(quote.ask_price))
if price == 0:
    return None
return price
```

No timestamp validation. No spread check. No market-hours awareness. If the market is closed, halted, or the feed is delayed, this returns whatever the last quote was — could be hours or days old.

This stale price flows into two critical paths:

1. **Validation** (`trader.py:200-202`): `validate_decision()` checks `cost <= buying_power` and position size against portfolio value using this price. A stale price that's 20% below current could let through a trade that blows the position limit.
2. **Logging** (`trader.py:274`): When `filled_avg_price` is unavailable (which is always, per P0-1), the stale quote is the recorded entry price.

**Fix scope:** Add timestamp and spread validation to `get_latest_price()`. Reject quotes older than N seconds. Reject quotes with bid-ask spread wider than a threshold. Return a richer object (price, timestamp, spread) so callers can make informed decisions.

---

### P0-3: No market hours gate

**Where:** `trader.py:57-328` (entire `run_trading_session()`), `session.py:132-140` (Stage 3)

Nothing in the pipeline checks whether the market is open before executing trades. `execute_market_order()` submits with `TimeInForce.DAY`, so after-hours submissions queue for next open. But:

- The system logs the trade as executed now, at the current (stale) quote price
- Buying power is decremented immediately
- The thesis is marked as "executed" (`trader.py:236-243`)
- Next morning when the order actually fills, the price could be materially different
- If the session runs again the next day (before the queued order fills), it may try to buy the same thing again

The session orchestrator (`session.py:132-140`) has no market-hours guard either — it hands off to the executor unconditionally.

**Fix scope:** Add `is_market_open()` in `executor.py` using the Alpaca clock API. Gate `run_trading_session()` on market open (with override for `--dry-run`). The gate should be in `trader.py` between Step 1 (sync) and Step 2 (snapshot), so we still sync positions even if we can't trade.

---

## P1: Learning Loop & Data Integrity Issues

These bugs degrade the attribution system, corrupt the learning feedback loop, or create data quality problems that compound over time.

---

### P1-1: Signal refs accepted without validation

**Where:** `agent.py:210` → `trader.py:295-303` → `database/trading_db.py:397-405`

The executor LLM returns `signal_refs` as part of each decision:

```python
# agent.py:210
signal_refs=d.get("signal_refs", [])
```

These refs (e.g., `{"type": "news_signal", "id": 99999}`) are never validated. The flow:

1. LLM produces `signal_refs` — no schema enforcement beyond the system prompt asking nicely (`agent.py:110`)
2. `trader.py:297-300` builds `(decision_id, ref["type"], ref["id"])` tuples directly from the LLM output
3. `insert_decision_signals_batch()` inserts them with `ON CONFLICT DO NOTHING`

There's no check that `signal_type` is a valid enum. No check that `signal_id` exists in the referenced table. No FK constraint in the database. The LLM can hallucinate signal IDs (it often does — it has no way to verify IDs exist), and these phantom references flow into `compute_signal_attribution()` where they JOIN against `news_signals`/`macro_signals`. Phantom refs produce NULL categories, which silently drop from the attribution results — but they still occupy rows in `decision_signals`, inflating the apparent sample count.

**Fix scope:** Add `validate_signal_refs()` in `agent.py` that queries the DB to confirm each referenced signal exists. Strip invalid refs before insertion. Log warnings for hallucinated refs so we can tune the system prompt.

---

### P1-2: Attribution measures win rate, not expected value

**Where:** `attribution.py:42-43` → `attribution.py:91-132` → `session.py:99`

The attribution query computes win rate:

```sql
-- attribution.py:42-43
AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_7d
```

It also computes `AVG(outcome_7d)` (average return), but `build_attribution_constraints()` only uses the win rate to categorize signals:

```python
# attribution.py:115-118
if wr > 55:
    strong.append(...)
elif wr < 45:
    weak.append(...)
```

This is a fundamental flaw. A signal type with 40% win rate but +8% average wins and -2% average losses is highly profitable (EV = 0.4 * 8 - 0.6 * 2 = +2.0% per trade). Under the current system, it gets flagged WEAK and the strategist is told to avoid it. Conversely, a signal type with 60% win rate but +0.5% average wins and -3% average losses is a money loser (EV = 0.6 * 0.5 - 0.4 * 3 = -0.9%) but gets flagged STRONG.

The `get_attribution_summary()` function (`attribution.py:64-88`) does include `avg_outcome_7d` in the text it sends to the LLM, but the STRONG/WEAK categorization in constraints is purely directional.

**Fix scope:** Replace win-rate thresholds with expected value. The query already computes `AVG(outcome_7d)` — use it directly as the primary metric. Optionally compute a risk-adjusted metric (avg_return / stddev or a simplified Sharpe). Update both `build_attribution_constraints()` and `get_attribution_summary()`.

---

### P1-3: No session idempotency

**Where:** `session.py:75-204` → `trader.py:275-288` → `database/trading_db.py:124-139`

Running the session twice in a day produces duplicate everything:

- `insert_decision()` always creates a new row — no deduplication on (date, ticker, action)
- `insert_decision_signals_batch()` uses `ON CONFLICT DO NOTHING` on the unique constraint, but since the decision_id is new, the "same" signal link gets a fresh row
- The news pipeline (Stage 1) re-fetches and re-classifies the same headlines
- Social media posts go out twice
- Attribution data doubles, skewing sample sizes

There's no `sessions` table, no session ID, no "already ran today" check.

**Fix scope:** Add a `sessions` table with a unique constraint on `(session_date, session_type)`. Add `get_session_for_date()` and `insert_session_record()` to `trading_db.py`. Gate `run_session()` on "no completed session exists for today" with a `--force` override flag.

---

### P1-4: Buying power drifts from reality across trades

**Where:** `trader.py:171` → `trader.py:224-228`

Buying power is snapshotted once at session start:

```python
# trader.py:171
buying_power = account_info["buying_power"]
```

Then locally decremented per trade:

```python
# trader.py:224-228
trade_value = price * Decimal(str(decision.quantity))
if decision.action == "buy":
    total_buy_value += trade_value
    buying_power -= trade_value
```

Problems:

1. `trade_value` uses the quote price, not the fill price (see P0-1). If fill is worse, the real buying power consumed is higher.
2. Alpaca's buying power accounts for margin, pending settlements, and fees — the local calculation doesn't.
3. Sell proceeds aren't added back to buying power (line 230 only tracks `total_sell_value`, doesn't credit `buying_power`).
4. After 3-4 trades, the local `buying_power` can be significantly wrong in either direction.

A later trade in the same session could be rejected (buying power exhausted locally but available on Alpaca) or approved (buying power available locally but Alpaca would reject due to margin/settlement).

**Fix scope:** After each fill confirmation (see P0-1), re-query `get_account_info()` to get the real buying power from Alpaca. Use that for subsequent validations instead of the local estimate.

---

## Pipeline Data Flow (current state)

Shows where each P0/P1 issue injects error into the pipeline:

```
session.py:run_session()
  │
  ├─ Stage 0: backfill + attribution
  │    └─ attribution.py:compute_signal_attribution()
  │         └─ [P1-2] Win rate only, no expected value
  │         └─ [P1-1] Phantom signal refs inflate sample counts
  │
  ├─ Stage 2: strategist
  │    └─ Receives attribution constraints (advisory only)
  │
  ├─ Stage 3: executor
  │    └─ trader.py:run_trading_session()
  │         ├─ [P0-3] No market hours check
  │         ├─ executor.py:get_latest_price()
  │         │    └─ [P0-2] No staleness/spread validation
  │         ├─ agent.py:validate_decision()
  │         │    └─ Uses stale price + drifted buying power
  │         ├─ executor.py:execute_market_order()
  │         │    └─ [P0-1] Returns before fill
  │         ├─ buying_power -= estimated_cost
  │         │    └─ [P1-4] Drift accumulates per trade
  │         ├─ insert_decision(price=stale_or_phantom)
  │         │    └─ [P1-3] No dedup, no session tracking
  │         └─ insert_decision_signals_batch(unvalidated_refs)
  │              └─ [P1-1] Hallucinated signal IDs stored
  │
  └─ Next day: backfill computes outcomes against phantom entry prices
       └─ Attribution trains on garbage → strategist gets bad constraints
```

---

## Priority Summary

| ID | Issue | Risk | Root Cause |
|----|-------|------|------------|
| P0-1 | No fill confirmation | Phantom entry prices poison attribution | `execute_market_order()` returns at submission, not fill |
| P0-2 | Stale quotes | Bad validation, bad logging | `get_latest_price()` has no timestamp/spread checks |
| P0-3 | No market hours gate | Queued orders at wrong prices | Neither `trader.py` nor `session.py` check market clock |
| P1-1 | Unvalidated signal refs | Corrupt attribution data | LLM output accepted as-is, no DB validation |
| P1-2 | Win rate instead of EV | Profitable signals flagged weak | `build_attribution_constraints()` ignores magnitude |
| P1-3 | No idempotency | Duplicate trades, doubled data | No session tracking, no dedup on decisions |
| P1-4 | Buying power drift | Wrong validation after first trade | Local estimate diverges from Alpaca state |
