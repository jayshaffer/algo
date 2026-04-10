# Gap Analysis Tier 1 & 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the system's broken learning loop and add critical safety guardrails so the trading system can actually learn from its decisions and stop losing money.

**Architecture:** Seven tasks in dependency order. Tasks 1-3 fix the learning loop (attribution granularity, signal linkage, rule lifecycle). Tasks 4-7 add safety (circuit breaker, dedup, price constraint, data cleanup). Each task is independently testable and committable.

**Tech Stack:** Python 3, PostgreSQL 16, psycopg2, pytest, Alpaca API

---

## Task 1: Collapse Attribution Categories

The composite key `news_signal:earnings:bullish:buy` creates an explosion of low-N categories (n=1-4). With only 42 signal-linked decisions, splitting across 4 dimensions guarantees noise. Collapse to 2-part keys: `news_signal:earnings`, `macro_signal:geopolitical`, `thesis`.

**Files:**
- Modify: `v2/attribution.py:24-68` (SQL query)
- Modify: `v2/patterns.py` (align `analyze_signal_categories` to match)
- Test: `tests/v2/test_attribution.py`
- Test: `tests/v2/test_patterns.py`

- [ ] **Step 1: Write failing test for collapsed categories**

In `tests/v2/test_attribution.py`, add a test that verifies the SQL groups by 2-part key:

```python
class TestCollapsedCategories:
    def test_categories_exclude_sentiment_and_action(self):
        """Attribution categories should be signal_type:subcategory only,
        not signal_type:subcategory:sentiment:action."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("1.5"),
                "avg_outcome_30d": Decimal("2.0"),
                "win_rate_7d": Decimal("0.60"),
                "win_rate_30d": Decimal("0.55"),
            }
        ]

        @contextmanager
        def _get_cursor():
            yield mock_cursor

        with patch("v2.attribution.get_cursor", _get_cursor), \
             patch("v2.attribution.upsert_signal_attribution"):
            from v2.attribution import compute_signal_attribution
            results = compute_signal_attribution(days=90)

        # Verify the SQL doesn't include sentiment or action in category
        executed_sql = mock_cursor.execute.call_args[0][0]
        # The CASE statement should produce 'news_signal:' || category, NOT
        # 'news_signal:' || category || ':' || sentiment || ':' || action
        assert "|| d.action" not in executed_sql
        assert "|| COALESCE(ns.sentiment" not in executed_sql

    def test_thesis_category_has_no_action_suffix(self):
        """Thesis category should be just 'thesis', not 'thesis:buy'."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "category": "thesis",
                "sample_size": 25,
                "avg_outcome_7d": Decimal("-0.76"),
                "avg_outcome_30d": Decimal("3.94"),
                "win_rate_7d": Decimal("0.44"),
                "win_rate_30d": Decimal("0.44"),
            }
        ]

        @contextmanager
        def _get_cursor():
            yield mock_cursor

        with patch("v2.attribution.get_cursor", _get_cursor), \
             patch("v2.attribution.upsert_signal_attribution"):
            from v2.attribution import compute_signal_attribution
            results = compute_signal_attribution(days=90)

        executed_sql = mock_cursor.execute.call_args[0][0]
        # The ELSE branch should just be ds.signal_type, not ds.signal_type || ':' || d.action
        assert "ELSE ds.signal_type\n" in executed_sql or "ELSE ds.signal_type " in executed_sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_attribution.py::TestCollapsedCategories -v`
Expected: FAIL — current SQL includes sentiment and action in category key

- [ ] **Step 3: Update the attribution SQL**

In `v2/attribution.py`, replace the CASE statement in `compute_signal_attribution()` (lines 28-37):

```python
def compute_signal_attribution(days: int = 90) -> list[dict]:
    """
    Compute signal attribution scores from decision_signals joined with decisions.

    Uses alpha (outcome - benchmark) instead of raw returns so attribution
    reflects signal quality independent of market conditions.

    Groups by 2-part category (signal_type + subcategory) to keep sample sizes
    meaningful. Sentiment and action are NOT part of the category key.
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
                    d.outcome_30d,
                    d.benchmark_7d,
                    d.benchmark_30d,
                    CASE WHEN d.benchmark_7d IS NOT NULL
                         THEN d.outcome_7d - d.benchmark_7d
                         ELSE NULL END AS alpha_7d,
                    CASE WHEN d.benchmark_30d IS NOT NULL
                         THEN d.outcome_30d - d.benchmark_30d
                         ELSE NULL END AS alpha_30d
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
                AVG(alpha_7d) AS avg_outcome_7d,
                AVG(alpha_30d) AS avg_outcome_30d,
                AVG(CASE WHEN alpha_7d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_7d,
                AVG(CASE WHEN alpha_30d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_30d
            FROM categorized
            WHERE outcome_7d IS NOT NULL
              AND alpha_7d IS NOT NULL
            GROUP BY category
            ORDER BY sample_size DESC
        """, (cutoff_date,))
        results = [dict(row) for row in cur.fetchall()]

    for row in results:
        upsert_signal_attribution(
            category=row["category"],
            sample_size=row["sample_size"],
            avg_outcome_7d=row["avg_outcome_7d"] or Decimal(0),
            avg_outcome_30d=row["avg_outcome_30d"] or Decimal(0),
            win_rate_7d=row["win_rate_7d"] or Decimal(0),
            win_rate_30d=row["win_rate_30d"] or Decimal(0),
        )

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_attribution.py -v`
Expected: All pass, including new `TestCollapsedCategories` tests

- [ ] **Step 5: Update patterns.py to match**

The `analyze_signal_categories()` in `v2/patterns.py` already uses 2-part keys (`news_signal:category`, `macro_signal:category`). Verify existing pattern tests still pass:

Run: `python3 -m pytest tests/v2/test_patterns.py -v`
Expected: All pass (patterns.py already uses the simpler grouping)

- [ ] **Step 6: Commit**

```bash
git add v2/attribution.py tests/v2/test_attribution.py
git commit -m "fix(attribution): collapse categories to 2-part keys for meaningful sample sizes

The 4-part key (type:category:sentiment:action) created an explosion of n=1-4
categories. Collapsing to type:category consolidates samples and enables
statistically meaningful attribution."
```

---

## Task 2: Fix Signal Linkage

47.5% of trade decisions (38 of 80 buy/sell) have zero entries in `decision_signals`. The root causes:
1. The LLM sometimes returns empty `signal_refs`
2. `validate_signal_refs()` does N+1 queries (one per ref), making it slow and fragile
3. No enforcement or fallback when signal_refs are missing

Fix: batch-validate signal refs in one query, and log a structured warning when a buy/sell decision has no refs so we can track the gap.

**Files:**
- Modify: `v2/agent.py:347-378` (validate_signal_refs)
- Modify: `v2/trader.py:410-428` (signal logging block)
- Test: `tests/v2/test_agent.py`

- [ ] **Step 1: Write failing test for batch validation**

In `tests/v2/test_agent.py`, add:

```python
class TestValidateSignalRefsBatch:
    def test_batch_validates_in_single_query_per_type(self, mock_db):
        """Should batch-validate all refs of same type in one query, not N+1."""
        refs = [
            {"type": "news_signal", "id": 1},
            {"type": "news_signal", "id": 2},
            {"type": "news_signal", "id": 99},  # doesn't exist
            {"type": "thesis", "id": 5},
        ]
        # news_signals query returns ids 1 and 2 (not 99)
        # theses query returns id 5
        mock_db.fetchall.side_effect = [
            [{"id": 1}, {"id": 2}],  # news_signals batch
            [{"id": 5}],              # theses batch
        ]

        from v2.agent import validate_signal_refs
        result = validate_signal_refs(refs)

        assert len(result) == 3  # 1, 2, and 5 valid; 99 stripped
        # Should have made exactly 2 queries (one per signal type), not 4
        assert mock_db.execute.call_count == 2

    def test_returns_empty_for_empty_input(self, mock_db):
        from v2.agent import validate_signal_refs
        result = validate_signal_refs([])
        assert result == []
        assert mock_db.execute.call_count == 0

    def test_strips_unknown_signal_types(self, mock_db):
        refs = [{"type": "unknown_type", "id": 1}]
        from v2.agent import validate_signal_refs
        result = validate_signal_refs(refs)
        assert result == []
        assert mock_db.execute.call_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_agent.py::TestValidateSignalRefsBatch -v`
Expected: FAIL — current implementation does N+1 queries

- [ ] **Step 3: Rewrite validate_signal_refs to batch**

In `v2/agent.py`, replace `validate_signal_refs()` (lines 347-378):

```python
def validate_signal_refs(signal_refs: list[dict]) -> list[dict]:
    """Validate signal refs against the database, stripping invalid ones.

    Batches queries by signal type (one query per type) instead of N+1.

    Args:
        signal_refs: List of {"type": str, "id": int} dicts from LLM output

    Returns:
        Filtered list containing only refs that exist in the database.
    """
    if not signal_refs:
        return []

    from .database.connection import get_cursor

    # Group refs by signal type
    by_type: dict[str, list[dict]] = {}
    for ref in signal_refs:
        sig_type = ref.get("type", "")
        table = _SIGNAL_TYPE_TABLES.get(sig_type)
        if not table:
            logger.warning("Stripping signal ref with unknown type: %s", sig_type)
            continue
        by_type.setdefault(sig_type, []).append(ref)

    valid = []
    for sig_type, refs in by_type.items():
        table = _SIGNAL_TYPE_TABLES[sig_type]
        ids = [r["id"] for r in refs if r.get("id") is not None]
        if not ids:
            continue

        with get_cursor() as cur:
            cur.execute(
                f"SELECT id FROM {table} WHERE id = ANY(%s)",
                (ids,)
            )
            found_ids = {row["id"] for row in cur.fetchall()}

        for ref in refs:
            if ref.get("id") in found_ids:
                valid.append(ref)
            else:
                logger.warning("Stripping signal ref %s:%s — not found in DB", sig_type, ref.get("id"))

    return valid
```

- [ ] **Step 4: Run all agent tests**

Run: `python3 -m pytest tests/v2/test_agent.py -v`
Expected: All pass including new batch tests and existing `TestValidateSignalRefs` tests

- [ ] **Step 5: Commit**

```bash
git add v2/agent.py tests/v2/test_agent.py
git commit -m "fix(agent): batch-validate signal refs instead of N+1 queries

Grouped by signal type with ANY() array match. Fixes performance and reduces
DB connections from N to max 3 (one per signal type table)."
```

---

## Task 3: Add Rule Lifecycle Guardrails

All 23 strategy rules have been proposed then retired — zero durable knowledge. The system oscillates: propose rule → follow it for 1-2 sessions → retire it → propose similar rule. Fix by adding minimum tenure before retirement and a per-session retirement cap.

**Files:**
- Modify: `v2/strategy.py:142-148` (tool_retire_rule)
- Modify: `v2/database/trading_db.py` (add get_rule_age helper)
- Create: `db/init/016_rule_tenure.sql`
- Test: `tests/v2/test_strategy.py`

- [ ] **Step 1: Write failing test for tenure enforcement**

In `tests/v2/test_strategy.py`, add:

```python
class TestRuleTenureGuard:
    def test_cannot_retire_rule_before_min_tenure(self, mock_db):
        """Rules must be active for at least 5 sessions before retirement."""
        from datetime import datetime, timedelta
        # Rule created 2 days ago
        mock_db.fetchone.return_value = {
            "id": 1, "rule_text": "Test rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=2),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }

        from v2.strategy import tool_retire_rule
        result = tool_retire_rule(rule_id=1, reason="Not working")

        assert "minimum tenure" in result.lower() or "too new" in result.lower()
        # Should NOT have called the retire SQL
        retire_calls = [c for c in mock_db.execute.call_args_list
                        if "retired" in str(c).lower() and "UPDATE" in str(c).upper()]
        assert len(retire_calls) == 0

    def test_can_retire_rule_after_min_tenure(self, mock_db):
        """Rules active for >= 5 days can be retired normally."""
        from datetime import datetime, timedelta
        mock_db.fetchone.return_value = {
            "id": 1, "rule_text": "Test rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=6),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        mock_db.rowcount = 1

        from v2.strategy import tool_retire_rule
        result = tool_retire_rule(rule_id=1, reason="Data no longer supports it")

        assert "retired" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestRuleTenureGuard -v`
Expected: FAIL — current code retires immediately without tenure check

- [ ] **Step 3: Add rule age lookup and enforce tenure**

In `v2/database/trading_db.py`, add a helper (near the existing `retire_strategy_rule`):

```python
def get_strategy_rule(rule_id: int) -> dict | None:
    """Fetch a single strategy rule by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM strategy_rules WHERE id = %s", (rule_id,))
        return cur.fetchone()
```

In `v2/strategy.py`, modify `tool_retire_rule()`:

```python
MIN_RULE_TENURE_DAYS = 5

def tool_retire_rule(rule_id: int, reason: str) -> str:
    """Retire a strategy rule (with tenure guard)."""
    from .database.trading_db import get_strategy_rule
    rule = get_strategy_rule(rule_id)
    if not rule:
        return f"Error: Rule ID {rule_id} not found"
    if rule["status"] != "active":
        return f"Error: Rule ID {rule_id} is already {rule['status']}"

    age_days = (date.today() - rule["created_at"].date()).days
    if age_days < MIN_RULE_TENURE_DAYS:
        remaining = MIN_RULE_TENURE_DAYS - age_days
        return (
            f"Rule ID {rule_id} is too new to retire ({age_days}d old, "
            f"minimum tenure is {MIN_RULE_TENURE_DAYS}d). "
            f"Wait {remaining} more day(s) or write a memo noting your concern."
        )

    logger.info(f"Retiring rule {rule_id}: {reason}")
    success = retire_strategy_rule(rule_id=rule_id, reason=reason)
    if success:
        return f"Retired rule ID {rule_id}. Reason: {reason}"
    return f"Error: Failed to retire rule ID {rule_id}"
```

- [ ] **Step 4: Write test for per-session retirement cap**

```python
class TestRetirementCap:
    def test_max_retirements_per_session(self, mock_db):
        """At most 2 rules can be retired per session to prevent mass purges."""
        from datetime import datetime, timedelta

        old_rule = {
            "id": 1, "rule_text": "Old rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=30),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        mock_db.fetchone.return_value = old_rule
        mock_db.rowcount = 1

        from v2.strategy import tool_retire_rule, reset_session
        reset_session()  # Clear per-session state

        # First 2 should succeed
        r1 = tool_retire_rule(rule_id=1, reason="Data changed")
        assert "retired" in r1.lower()
        r2 = tool_retire_rule(rule_id=2, reason="No longer valid")
        assert "retired" in r2.lower()

        # Third should be blocked
        r3 = tool_retire_rule(rule_id=3, reason="Also bad")
        assert "limit" in r3.lower() or "maximum" in r3.lower()
```

- [ ] **Step 5: Implement per-session retirement cap**

In `v2/strategy.py`, add session-scoped tracking:

```python
MAX_RETIREMENTS_PER_SESSION = 2
_session_retirements: list[int] = []


def reset_session():
    """Reset per-session counters. Called at start of strategy reflection."""
    _session_retirements.clear()
```

Update `tool_retire_rule()` to check the cap after the tenure check:

```python
    if len(_session_retirements) >= MAX_RETIREMENTS_PER_SESSION:
        return (
            f"Retirement limit reached ({MAX_RETIREMENTS_PER_SESSION} per session). "
            f"Write a memo noting that rule {rule_id} should be reviewed next session."
        )
```

And append to `_session_retirements` after a successful retirement:

```python
    if success:
        _session_retirements.append(rule_id)
        return f"Retired rule ID {rule_id}. Reason: {reason}"
```

Also add a `reset_session()` call at the start of `run_strategy_reflection()`.

- [ ] **Step 6: Run all strategy tests**

Run: `python3 -m pytest tests/v2/test_strategy.py -v`
Expected: All pass

- [ ] **Step 7: Update the reflection system prompt**

In `v2/strategy.py`, add to `STRATEGY_REFLECTION_SYSTEM` (in the "Rule Management" section):

```
## Rule Tenure

Rules have a minimum tenure of 5 days. You cannot retire a rule until it has been active for at least 5 sessions. This prevents oscillation (propose → retire → re-propose). If a new rule seems wrong, write a memo instead and revisit next session.

You can retire at most 2 rules per session. If more rules need attention, prioritize the most clearly unsupported ones and note the rest in your memo.
```

- [ ] **Step 8: Commit**

```bash
git add v2/strategy.py v2/database/trading_db.py tests/v2/test_strategy.py
git commit -m "fix(strategy): add rule tenure guard and per-session retirement cap

Rules must be active >= 5 days before retirement (stops oscillation).
Max 2 retirements per session (prevents mass purges).
System has retired all 23 rules — this enforces durability."
```

---

## Task 4: Add Daily Loss Limit and Circuit Breaker

The system has no aggregate safety net. Add a pre-flight check before trading that compares today's equity to the previous snapshot and aborts if the daily loss exceeds a threshold.

**Files:**
- Modify: `v2/risk.py` (add check_daily_loss_limit)
- Modify: `v2/trader.py` (add pre-flight check before Step 4)
- Test: `tests/v2/test_risk.py` (new file)
- Test: `tests/v2/test_trader.py` (add circuit breaker test)

- [ ] **Step 1: Write failing tests for daily loss limit**

Create `tests/v2/test_risk.py`:

```python
"""Tests for v2 risk management."""

import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock
from contextlib import contextmanager


class TestCheckDailyLossLimit:
    def test_blocks_when_loss_exceeds_threshold(self):
        """Should return a warning when daily loss exceeds threshold."""
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("950"),
            previous_value=Decimal("1000"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is not None
        assert "loss" in result.lower()

    def test_allows_when_within_threshold(self):
        """Should return None when loss is within threshold."""
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("985"),
            previous_value=Decimal("1000"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None

    def test_allows_when_portfolio_is_up(self):
        """Should return None when portfolio gained value."""
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("1050"),
            previous_value=Decimal("1000"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None

    def test_handles_zero_previous_value(self):
        """Should return None rather than division error."""
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("1000"),
            previous_value=Decimal("0"),
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None

    def test_handles_none_previous_value(self):
        """Should return None when no previous snapshot exists."""
        from v2.risk import check_daily_loss_limit
        result = check_daily_loss_limit(
            current_value=Decimal("1000"),
            previous_value=None,
            max_loss_pct=Decimal("0.03"),
        )
        assert result is None


class TestCheckSectorConcentration:
    def test_warns_on_concentration(self):
        from v2.risk import check_sector_concentration
        warnings = check_sector_concentration(
            {"AAPL": Decimal("500"), "MSFT": Decimal("500")},
            Decimal("1000"),
        )
        assert len(warnings) == 1
        assert "tech" in warnings[0].lower()

    def test_no_warning_within_limit(self):
        from v2.risk import check_sector_concentration
        warnings = check_sector_concentration(
            {"AAPL": Decimal("300"), "JPM": Decimal("300")},
            Decimal("1000"),
        )
        assert len(warnings) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_risk.py::TestCheckDailyLossLimit -v`
Expected: FAIL — `check_daily_loss_limit` doesn't exist yet

- [ ] **Step 3: Implement check_daily_loss_limit**

In `v2/risk.py`, add:

```python
MAX_DAILY_LOSS_PCT = Decimal("0.03")  # 3% daily loss limit


def check_daily_loss_limit(
    current_value: Decimal,
    previous_value: Decimal | None,
    max_loss_pct: Decimal = MAX_DAILY_LOSS_PCT,
) -> str | None:
    """Check if daily loss exceeds threshold.

    Args:
        current_value: Current portfolio value
        previous_value: Previous day's closing portfolio value
        max_loss_pct: Maximum allowed daily loss as decimal (0.03 = 3%)

    Returns:
        Warning string if limit breached, None if OK.
    """
    if previous_value is None or previous_value <= 0:
        return None

    change_pct = (current_value - previous_value) / previous_value
    if change_pct < -max_loss_pct:
        return (
            f"CIRCUIT BREAKER: Daily loss {change_pct:.2%} exceeds "
            f"{-max_loss_pct:.1%} limit (${current_value:,.2f} vs "
            f"prior ${previous_value:,.2f}). Trading halted."
        )
    return None
```

- [ ] **Step 4: Run risk tests**

Run: `python3 -m pytest tests/v2/test_risk.py -v`
Expected: All pass

- [ ] **Step 5: Add get_previous_snapshot to trading_db.py**

In `v2/database/trading_db.py`, add:

```python
def get_previous_snapshot() -> dict | None:
    """Get the most recent account snapshot before today."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM account_snapshots
            WHERE date < CURRENT_DATE
            ORDER BY date DESC
            LIMIT 1
        """)
        return cur.fetchone()
```

- [ ] **Step 6: Wire circuit breaker into trader.py**

In `v2/trader.py`, add after the market hours gate (after line 121) and before the data client setup:

```python
    # Circuit breaker — halt trading if daily loss exceeds threshold
    from .risk import check_daily_loss_limit
    from .database.trading_db import get_previous_snapshot
    try:
        account_info_preflight = get_account_info()
        prev_snapshot = get_previous_snapshot()
        if prev_snapshot:
            loss_warning = check_daily_loss_limit(
                current_value=account_info_preflight["portfolio_value"],
                previous_value=prev_snapshot["portfolio_value"],
            )
            if loss_warning:
                logger.error(loss_warning)
                errors.append(loss_warning)
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
    except Exception as e:
        logger.warning("Circuit breaker check failed: %s — continuing", e)
```

- [ ] **Step 7: Write trader integration test**

In `tests/v2/test_trader.py`, add:

```python
class TestCircuitBreaker:
    def test_halts_trading_on_daily_loss(self, mock_db, mock_claude_client):
        """Trading should halt if daily loss exceeds 3%."""
        # Mock account showing 5% loss
        with patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.sync_positions_from_alpaca", return_value=3), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_previous_snapshot") as mock_prev:
            mock_acct.return_value = {
                "portfolio_value": Decimal("950"),
                "buying_power": Decimal("500"),
            }
            mock_prev.return_value = {
                "portfolio_value": Decimal("1000"),
                "date": "2026-04-07",
            }

            from v2.trader import run_trading_session
            result = run_trading_session(dry_run=False)

        assert result.trades_executed == 0
        assert any("CIRCUIT BREAKER" in e for e in result.errors)
```

- [ ] **Step 8: Run all trader tests**

Run: `python3 -m pytest tests/v2/test_trader.py tests/v2/test_risk.py -v`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add v2/risk.py v2/trader.py v2/database/trading_db.py tests/v2/test_risk.py tests/v2/test_trader.py
git commit -m "feat(risk): add daily loss limit circuit breaker (3% threshold)

Checks current portfolio value against previous snapshot before trading.
Halts all trading if daily loss exceeds 3%. The system has been losing money
systematically — this prevents further damage on bad days."
```

---

## Task 5: Deduplicate Decisions

Same ticker bought 3x on same day (AMZN, TSM on Feb 7). Add a UNIQUE constraint and application-level guard.

**Files:**
- Create: `db/init/016_decision_dedup.sql`
- Modify: `v2/database/trading_db.py` (dedup check before insert)
- Test: `tests/v2/test_db.py`

- [ ] **Step 1: Create migration**

Create `db/init/016_decision_dedup.sql`:

```sql
-- Prevent duplicate trade decisions (same ticker, action, date)
-- Allow hold decisions to be duplicated (they're not actionable)
CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_dedup
ON decisions (date, ticker, action)
WHERE action IN ('buy', 'sell');
```

- [ ] **Step 2: Write failing test for application-level dedup**

In `tests/v2/test_db.py`, add:

```python
class TestDecisionDedup:
    def test_check_decision_exists(self, mock_db):
        """Should check for existing decision before insert."""
        from datetime import date
        mock_db.fetchone.return_value = {"id": 42}

        from v2.database.trading_db import check_decision_exists
        result = check_decision_exists(date.today(), "AAPL", "buy")
        assert result == 42

    def test_check_decision_not_found(self, mock_db):
        """Should return None when no matching decision exists."""
        from datetime import date
        mock_db.fetchone.return_value = None

        from v2.database.trading_db import check_decision_exists
        result = check_decision_exists(date.today(), "AAPL", "buy")
        assert result is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_db.py::TestDecisionDedup -v`
Expected: FAIL — function doesn't exist

- [ ] **Step 4: Implement check_decision_exists**

In `v2/database/trading_db.py`, add near `insert_decision()`:

```python
def check_decision_exists(decision_date, ticker: str, action: str) -> int | None:
    """Check if a buy/sell decision already exists for this ticker today.

    Returns the existing decision ID if found, None otherwise.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id FROM decisions
            WHERE date = %s AND ticker = %s AND action = %s
              AND action IN ('buy', 'sell')
            LIMIT 1
        """, (decision_date, ticker, action))
        row = cur.fetchone()
        return row["id"] if row else None
```

- [ ] **Step 5: Wire dedup check into trader.py**

In `v2/trader.py`, in the decision logging block (around line 391), add before `insert_decision()`:

```python
        # Skip duplicate decisions (same ticker+action already logged today)
        from .database.trading_db import check_decision_exists
        existing_id = check_decision_exists(date.today(), decision.ticker, decision.action)
        if existing_id:
            logger.warning("%s: duplicate %s decision — already logged as ID %d",
                           decision.ticker, decision.action, existing_id)
            continue
```

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/v2/test_db.py::TestDecisionDedup tests/v2/test_trader.py -v`
Expected: All pass

- [ ] **Step 7: Apply migration to running database**

Run: `docker compose exec -T db psql -U algo -d trading -f /docker-entrypoint-initdb.d/016_decision_dedup.sql`
(Or connect directly: `docker exec algo-db-1 psql -U algo -d trading -c "CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_dedup ON decisions (date, ticker, action) WHERE action IN ('buy', 'sell');"`)

Note: This will fail if existing duplicate data violates the constraint. If so, resolve by deleting the extra rows first:

```sql
DELETE FROM decisions a USING decisions b
WHERE a.id > b.id
  AND a.date = b.date AND a.ticker = b.ticker AND a.action = b.action
  AND a.action IN ('buy', 'sell');
```

- [ ] **Step 8: Commit**

```bash
git add db/init/016_decision_dedup.sql v2/database/trading_db.py v2/trader.py tests/v2/test_db.py
git commit -m "fix(db): add decision dedup constraint and application-level check

Prevents same ticker+action being logged twice in one day. Found 5 sets of
duplicate decisions in prod data (AMZN 3x, TSM 3x, etc on Feb 7)."
```

---

## Task 6: Require Entry Price on Trade Decisions

10 decisions have `price = NULL` and can never be backfilled. Add NOT NULL on price for buy/sell decisions at the application level (CHECK constraint would break hold inserts that legitimately have no price).

**Files:**
- Modify: `v2/trader.py` (skip decision logging if price is None — already partially handled)
- Test: `tests/v2/test_trader.py`

- [ ] **Step 1: Write test**

In `tests/v2/test_trader.py`, add:

```python
class TestDecisionPriceRequired:
    def test_skips_logging_decision_without_price(self, mock_db, mock_claude_client):
        """Decisions without a price should not be logged to the database."""
        from tests.v2.conftest import make_trading_decision, make_agent_response

        decision = make_trading_decision(action="buy")
        response = make_agent_response(decisions=[decision])

        with patch("v2.trader.get_account_info") as mock_acct, \
             patch("v2.trader.sync_positions_from_alpaca", return_value=0), \
             patch("v2.trader.sync_orders_from_alpaca", return_value=0), \
             patch("v2.trader.is_market_open", return_value=True), \
             patch("v2.trader.get_previous_snapshot", return_value=None), \
             patch("v2.trader.take_account_snapshot", return_value=1), \
             patch("v2.trader.build_executor_input"), \
             patch("v2.trader.get_trading_decisions", return_value=response), \
             patch("v2.trader.get_latest_price", return_value=None), \
             patch("v2.trader.get_positions", return_value=[]), \
             patch("v2.trader.get_open_orders", return_value=[]), \
             patch("v2.trader.check_sector_concentration", return_value=[]), \
             patch("v2.trader.insert_decision") as mock_insert:
            mock_acct.return_value = {
                "portfolio_value": Decimal("10000"),
                "buying_power": Decimal("5000"),
            }

            from v2.trader import run_trading_session
            result = run_trading_session(dry_run=True)

        # Decision should NOT have been inserted (no price)
        mock_insert.assert_not_called()
```

- [ ] **Step 2: Verify behavior**

Run: `python3 -m pytest tests/v2/test_trader.py::TestDecisionPriceRequired -v`

The existing code at `trader.py:378-381` already skips logging when price is None. Verify the test passes with existing code. If it does, the application-level guard is already in place — the 10 null-price decisions are from the pre-session era.

- [ ] **Step 3: Add CHECK constraint for future safety**

Create `db/init/017_decision_price_check.sql`:

```sql
-- Ensure buy/sell decisions always have a price.
-- Hold decisions can have NULL price (they're not trades).
ALTER TABLE decisions ADD CONSTRAINT chk_trade_price
CHECK (action NOT IN ('buy', 'sell') OR price IS NOT NULL);
```

Note: This will fail if existing NULL-price buy/sell rows exist. Fix first:

```sql
-- Backfill missing prices from existing data where possible, or mark as 'skip'
UPDATE decisions SET action = 'skip' WHERE action IN ('buy', 'sell') AND price IS NULL;
```

- [ ] **Step 4: Commit**

```bash
git add db/init/017_decision_price_check.sql
git commit -m "fix(db): add CHECK constraint requiring price on buy/sell decisions

10 historical decisions had NULL prices and were permanently unbackfillable.
Reclassified them as 'skip' and added constraint to prevent future occurrences."
```

---

## Task 7: Clean Up Stale Attribution Data

The signal_attribution table has rows from the old 4-part category scheme that will never be updated again (e.g., `news_signal:earnings:bearish:buy`). Clean them out so the strategist isn't reading stale data.

**Files:**
- No new code — this is a data migration

- [ ] **Step 1: Verify stale rows exist**

Run against prod DB:
```sql
SELECT category, sample_size, updated_at
FROM signal_attribution
ORDER BY updated_at;
```

Look for rows with old `updated_at` timestamps that use 3-part or 4-part category keys.

- [ ] **Step 2: Delete stale attribution rows**

```sql
-- Remove rows using old category scheme (3-part and 4-part keys)
-- Keep only rows that match the new 2-part scheme
DELETE FROM signal_attribution
WHERE category LIKE '%:%:%';
```

Verify: `SELECT * FROM signal_attribution;` should show only 2-part categories (or be empty, which is fine — next session will repopulate).

- [ ] **Step 3: Recompute attribution**

Run: `docker compose exec trading python -m v2.attribution`

This will repopulate signal_attribution with the new collapsed categories.

- [ ] **Step 4: Verify new categories**

```sql
SELECT category, sample_size, avg_outcome_7d, win_rate_7d
FROM signal_attribution
ORDER BY sample_size DESC;
```

Expected: Fewer categories, larger sample sizes per category.

- [ ] **Step 5: Commit a note (no code change)**

No code commit needed — this is a one-time data cleanup. The Task 1 code change ensures future attributions use 2-part keys.

---

## Dependency Order

```
Task 1 (attribution categories) ──┐
Task 2 (signal linkage)       ────┤── can be done in parallel
Task 3 (rule lifecycle)       ────┘

Task 4 (circuit breaker)      ──┐
Task 5 (decision dedup)       ──┤── can be done in parallel, independent of 1-3
Task 6 (price constraint)     ──┘

Task 7 (data cleanup)         ──── depends on Task 1 (needs new category scheme first)
```

Tasks 1-3 and 4-6 are independent groups and can be parallelized across two workers.
