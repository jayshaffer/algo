# V2 Pipeline Retrospective

Audit of limitations, gaps, and risks in the v2 trading pipeline.

---

## 1. Execution Model

### Buying power not tracked across trades in a session
`trader.py:171` snapshots `buying_power` once from Alpaca at session start. As trades execute, it subtracts estimated cost (line 226), but this is based on `get_latest_price()` at validation time — not the actual fill price. If Alpaca's reported buying power differs from the local accounting (margin, pending settlements), later trades in the same session can over-commit.

### No fill confirmation
`execute_market_order()` returns immediately after `submit_order()`. The order hasn't filled yet — it's just accepted. The `OrderResult.filled_qty` and `filled_avg_price` will be `None` for market orders at submission time (`executor.py:178-179`). The system logs decisions as if they executed at the quote price, not the fill price. There's no reconciliation step that checks what actually happened.

### Stale quote risk
`get_latest_price()` (`executor.py:243-259`) uses `ask_price` from the latest quote with no staleness check. If the market is closed, halted, or the quote is stale, the system validates and logs trades against a potentially outdated price. No spread check either — a wide bid-ask spread on a low-liquidity name could mean significant slippage.

### Max 10 trades hardcoded
`trader.py:178` — `max_trades_per_session = 10`. Not configurable via CLI or session args. If the strategist writes a playbook with 15 actions, the executor silently drops the last 5.

### Position size cap is validation-only
The 10% position size cap (`agent.py:233`) is enforced in `validate_decision()` but the executor LLM doesn't know about it. The LLM might propose a 15% position, get rejected, and the capital is wasted for that session. The cap should be in the system prompt so the LLM can self-correct.

---

## 2. Learning Loop

### Attribution uses only 7-day win rate
`attribution.py:43-46` — The WHERE clause filters to `outcome_7d IS NOT NULL`. The 30-day data is computed but `build_attribution_constraints()` only uses `win_rate_7d` (lines 111-118). This biases toward short-term signal quality and ignores whether signals that look bad at 7 days recover by 30 days (or vice versa).

### Win rate is a blunt metric
Attribution labels signals as STRONG (>55%) or WEAK (<45%) based purely on directional win rate. No consideration of magnitude — a signal type that wins 40% of the time but averages +5% per win and -1% per loss is highly profitable, but gets flagged WEAK. Expected value or Sharpe-like metric would be more informative.

### Backfill uses calendar days, not trading days
`backfill.py:128` — `exit_date = decision_date + timedelta(days=days)`. A decision made on Friday gets its 7-day outcome measured on the following Friday (including the weekend). If the exit date lands on a weekend/holiday, `get_price_on_date()` searches forward up to 5 calendar days, but this means "7-day" outcomes are really 5-7 trading days depending on when the decision was made. Inconsistent measurement window.

### Missing prices silently skipped
`backfill.py:132-135` — When `get_price_on_date()` returns None (delisted stock, data gap, API error), the decision is silently skipped. These decisions never get backfilled and accumulate as permanent holes in the attribution data. No alerting, no retry queue.

### Attribution constraints are soft
`build_attribution_constraints()` appends a text block to the strategist prompt saying "Do not create theses primarily based on WEAK signal categories unless you have a specific reason to override." The strategist can trivially override by writing "I have a specific reason" with no actual enforcement. The constraint is advisory, not structural.

---

## 3. Strategist

### No budget awareness
The strategist (Claude Opus, 25 turns) has no token budget visibility. Each turn can use up to 4096 output tokens. With 25 turns, that's potentially 100K+ output tokens per session at Opus pricing. There's no early-exit if the strategist has finished its work — it runs until it stops calling tools or hits the turn limit.

### Playbook actions don't require max_quantity
`tools.py:568` — `max_quantity` is not in the `required` array for playbook actions. The strategist can write "buy AAPL" with no quantity cap, and the executor decides sizing independently. This creates a disconnect: the strategist reasons about what to buy but the executor makes the sizing decision without the strategist's full context.

### One thesis per ticker restriction
`tools.py:98-103` — `tool_create_thesis()` rejects if an active thesis already exists for the ticker. This prevents the strategist from holding a long-term bull thesis and a short-term tactical sell thesis on the same name simultaneously. Also prevents creating a thesis for a ticker already in the portfolio (`tools.py:106-107`), which means you can't set exit conditions on existing positions via the thesis system.

### Web search has no grounding verification
The strategist has access to `web_search` (10 uses max per session) but there's no mechanism to verify that search results are accurate or that the strategist correctly interprets them. Search results flow directly into thesis creation with no fact-checking layer.

---

## 4. Executor

### No market hours check
Nothing in the pipeline checks whether the market is open before executing trades. `execute_market_order()` submits with `TimeInForce.DAY`, so orders submitted after hours will queue for next open — but the system logs them as if they executed at the current (stale) quote price.

### Signal refs not validated
`agent.py:200` — `signal_refs` from the LLM response are accepted as-is. The executor can cite `{"type": "news_signal", "id": 99999}` referencing a signal that doesn't exist. These bad references flow into `decision_signals` and corrupt the attribution loop. No FK constraint check before insert.

### Dry run doesn't test validation
Dry run skips `execute_market_order()` but still calls `validate_decision()`. However, the price fetched via `get_latest_price()` is real — so dry runs during market hours validate against live prices, but dry runs after hours validate against stale prices. Inconsistent behavior.

### No order type flexibility
The executor only supports market orders (`trader.py:212-217`). Limit orders exist in `executor.py:193-240` but are never called from the trading session. For any position where entry price matters (which the strategist specifies via `entry_trigger`), the system ignores the trigger and market-orders immediately.

---

## 5. News Pipeline

### Alpaca News API is the only source
V1 had plans for SEC EDGAR integration (10-K, 10-Q, 8-K). V2 dropped this and relies solely on the Alpaca News API. This means:
- No fundamental data (earnings reports, filings)
- No analyst estimate data
- Classification quality is bounded by headline quality
- Weekend/holiday gaps in news flow

### Batch classification failure mode is expensive
`classifier.py` sends 50 headlines per Haiku call. On batch failure, it falls back to classifying each headline individually — 50 separate API calls instead of 1. No circuit breaker pattern: if Haiku is having issues, the fallback will fail 50 times too.

### No deduplication
The pipeline fetches news by time window (`--hours 24`). If the session runs twice in a day (retry after failure, manual re-run), duplicate signals get inserted. No unique constraint on headline+ticker+timestamp.

---

## 6. Strategy Reflection

### Unbounded identity drift
`strategy.py` allows the reflection stage to `update_strategy_identity()` every session. There's no constraint on how much the identity can change per session. Over many sessions, the identity could drift far from any coherent strategy. No mechanism to detect or prevent contradictory rules accumulating.

### Rules never expire
`tool_propose_rule()` adds rules that stay active until explicitly retired. Old rules based on market conditions that no longer apply (e.g., "avoid tech during rate hikes") persist indefinitely. No time-decay or automatic review cycle.

### No reflection on reflection
The strategy reflection stage reviews decisions and proposes rules, but never reviews whether its own rules improved performance. The attribution system measures signal quality but not rule quality.

---

## 7. Social Media

### Twitter: no length validation
`twitter.py:149` says "Maximum 200 characters" in the prompt, but `generate_tweet()` never validates the output length. If Claude generates >280 characters, the post will fail at the Twitter API. Dashboard URL appended afterward (`twitter.py:187`) could push a valid tweet over the limit.

### Bluesky: length enforcement is good, Twitter has none
Bluesky has proper grapheme counting, LLM-based condensing, and hard truncation as fallback (`bluesky.py:96-165`). Twitter has zero enforcement. Inconsistent handling of the same problem.

### Both platforms share context but generate independently
Twitter and Bluesky each make a separate Claude call to generate their post from the same context. They could produce contradictory takes on the same session. No coordination between the two.

### No duplicate post prevention
If the session runs twice, both platforms get posted to twice. No check for "did we already post today?"

---

## 8. Database & Infrastructure

### No connection pooling
`get_cursor()` creates a new connection per call. Under load (batch classification + multiple tool calls), this could exhaust PostgreSQL connection limits. No connection pool (pgbouncer, psycopg2.pool).

### SQL injection surface in backfill
`backfill.py:61` and `backfill.py:95` use f-strings for column names: `f"outcome_{days_threshold}d"`. The `days` parameter comes from code (7 or 30), not user input, so it's safe today — but the pattern is fragile. If `days` ever came from CLI input without validation, it's injectable.

### No database migrations
Schema changes appear to be applied manually. No migration tool (alembic, flyway). Adding a column or table requires coordinating manual SQL across environments.

### Single point of failure on PostgreSQL
If PostgreSQL is down, every stage crashes. No graceful degradation, no local cache, no retry. The session exits with code 1 and the day's pipeline is lost.

---

## 9. Operational Gaps

### No alerting
Session errors are logged but not alerted. If the session fails at 4pm every day for a week, nobody knows unless they check the logs.

### No idempotency
Running the session twice in a day creates duplicate decisions, duplicate signals, duplicate tweets. No session-level deduplication or "already ran today" check.

### No cost tracking
Claude API usage (Opus strategist + Haiku executor + Haiku classifier + Haiku tweets + Opus reflection) is not tracked or budgeted. A session with a chatty strategist (25 turns of Opus) could cost $5-10+ per run with no visibility.

### Env var inconsistency
V2 uses `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (`executor.py:37-38`, `backfill.py:18-19`), while the project's `.env` documentation and v1 use `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`. Two different naming conventions for the same credentials.

### No scheduling
The session is designed to run daily after market close, but there's no cron job, systemd timer, or scheduler defined. It's invoked manually or via an undocumented external trigger.

---

## 10. What's Working Well

For balance — areas that are solid:

- **Structured executor input** — The `ExecutorInput` dataclass is a clean contract between strategist and executor. Much better than v1's prose context.
- **Decision-signal linkage** — The `decision_signals` FK table enables proper attribution. This is the backbone of the learning loop.
- **Stage isolation** — Each session stage is independent with try/except. A Twitter failure doesn't kill the trading session.
- **Playbook system** — The strategist → playbook → executor flow gives clear traceability for why trades were made.
- **Bluesky length enforcement** — The grapheme counting + LLM condensing + hard truncation pattern is well-designed.
- **Test coverage** — Comprehensive test suite with good mocking patterns.

---

## Priority Ranking

| Priority | Issue | Risk |
|----------|-------|------|
| P0 | No fill confirmation / price reconciliation | Money |
| P0 | Stale quotes → bad validation | Money |
| P0 | No market hours check | Money |
| P1 | Signal refs not validated → corrupt attribution | Learning loop integrity |
| P1 | Attribution uses only win rate, not expected value | Bad signal filtering |
| P1 | No idempotency (duplicate runs) | Data integrity |
| P1 | Buying power drift across multi-trade sessions | Money |
| P2 | Max trades hardcoded | Missed opportunities |
| P2 | No connection pooling | Reliability |
| P2 | Twitter length not validated | Failed posts |
| P2 | No alerting | Operational blindness |
| P2 | No cost tracking | Budget |
| P3 | One thesis per ticker | Strategy limitation |
| P3 | Rules never expire | Strategy drift |
| P3 | Env var naming inconsistency | Confusing |
