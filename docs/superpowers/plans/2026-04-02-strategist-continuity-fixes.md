# Strategist Continuity & Determinism Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close information flow gaps between trading stages and structurally enforce cooldown rules that the LLM has repeatedly violated.

**Architecture:** Three phases of independent fixes that compound in value. Phase 1 fixes what the strategist reads/writes. Phase 2 adds code-level enforcement of cooldown rules. Phase 3 reduces noise in identity and rule state.

**Tech Stack:** Python 3.12, psycopg2, pytest, unittest.mock

**Spec:** `docs/superpowers/specs/2026-04-02-strategist-continuity-design.md`

---

## File Map

**Create:**
- `v2/cooldown.py` — cooldown computation and enforcement logic
- `tests/v2/test_cooldown.py` — cooldown unit tests

**Modify:**
- `v2/tools.py:321-333` — two-tier memo truncation
- `v2/ideation_claude.py:7-14,280-314` — import + pre-seed memos
- `v2/session.py:175-193,195-213` — persist strategist summary, skip executor without playbook
- `v2/agent.py:40-53,90-123,143-153` — ExecutorInput field, prompt, serialization
- `v2/context.py:374-428` — pre-filter cooldown tickers, build cooldown map
- `v2/trader.py:234-259` — post-validation cooldown check
- `v2/strategy.py:42-67,72-93,115-120` — reflection prompt, identity guard, retire handler
- `v2/database/trading_db.py:535-541` — retire_strategy_rule with reason

**Test files:**
- `tests/v2/test_tools.py` — memo truncation tests
- `tests/v2/test_ideation_claude.py` — pre-seed verification
- `tests/v2/test_session.py` — summary persistence, executor skip
- `tests/v2/test_strategy.py` — identity guard, retire reason
- `tests/v2/test_cooldown.py` — cooldown rules (new file)
- `tests/v2/test_context.py` — cooldown filtering
- `tests/v2/test_trader.py` — post-validation cooldown

---

## Phase 1: Information Flow

### Task 1: Two-tier memo retrieval

**Files:**
- Modify: `v2/tools.py:321-333`
- Test: `tests/v2/test_tools.py`

- [ ] **Step 1: Write failing tests for two-tier truncation**

Add to `tests/v2/test_tools.py`:

```python
class TestGetStrategyHistoryTruncation:
    @patch("v2.tools.get_recent_strategy_memos")
    def test_recent_memos_not_truncated(self, mock_get):
        """Last 2 memos should be returned in full."""
        from v2.tools import tool_get_strategy_history
        long_content = "A" * 2000
        mock_get.return_value = [
            make_strategy_memo_row(content=long_content, session_date=date(2026, 4, 2)),
            make_strategy_memo_row(content=long_content, session_date=date(2026, 4, 1)),
        ]
        result = tool_get_strategy_history(n=5)
        # Both memos should contain full 2000-char content, no "..."
        assert long_content in result
        assert result.count("...") == 0

    @patch("v2.tools.get_recent_strategy_memos")
    def test_older_memos_truncated_to_300(self, mock_get):
        """Memos beyond position 2 should be truncated to 300 chars."""
        from v2.tools import tool_get_strategy_history
        long_content = "B" * 2000
        mock_get.return_value = [
            make_strategy_memo_row(content="recent1", session_date=date(2026, 4, 2)),
            make_strategy_memo_row(content="recent2", session_date=date(2026, 4, 1)),
            make_strategy_memo_row(content=long_content, session_date=date(2026, 3, 31)),
        ]
        result = tool_get_strategy_history(n=5)
        lines = result.split("\n")
        # Third memo (index 2) should be truncated
        assert "B" * 300 in lines[2]
        assert "B" * 301 not in lines[2]
        assert lines[2].endswith("...")

    @patch("v2.tools.get_recent_strategy_memos")
    def test_short_older_memos_not_truncated(self, mock_get):
        """Short older memos (<300 chars) should not get '...' appended."""
        from v2.tools import tool_get_strategy_history
        mock_get.return_value = [
            make_strategy_memo_row(content="recent", session_date=date(2026, 4, 2)),
            make_strategy_memo_row(content="recent", session_date=date(2026, 4, 1)),
            make_strategy_memo_row(content="short old memo", session_date=date(2026, 3, 31)),
        ]
        result = tool_get_strategy_history(n=5)
        assert "short old memo" in result
        assert "..." not in result.split("\n")[2]
```

Add missing import at top of file if needed:

```python
from datetime import date
from tests.v2.conftest import make_strategy_memo_row
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_tools.py::TestGetStrategyHistoryTruncation -v`
Expected: FAIL — `test_recent_memos_not_truncated` fails because current code truncates at 200 chars.

- [ ] **Step 3: Implement two-tier truncation**

Edit `v2/tools.py:321-333` — replace `tool_get_strategy_history`:

```python
def tool_get_strategy_history(n: int = 5, full_recent: int = 2) -> str:
    """Get recent strategy memos. Last `full_recent` shown in full, older truncated to 300 chars."""
    logger.info(f"Getting strategy history (last {n})")
    memos = get_recent_strategy_memos(n=n)
    if not memos:
        return "No strategy memos yet. This is the first session."

    lines = []
    for i, m in enumerate(memos):
        if i < full_recent:
            content = m['content']
        else:
            content = m['content'][:300] + "..." if len(m['content']) > 300 else m['content']
        lines.append(f"[{m['session_date']}] {m['memo_type']}: {content}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_tools.py::TestGetStrategyHistoryTruncation tests/v2/test_tools.py::TestGetStrategyHistory -v`
Expected: All PASS (new tests + existing tests still work).

- [ ] **Step 5: Commit**

```bash
git add v2/tools.py tests/v2/test_tools.py
git commit -m "feat(v2): two-tier memo truncation — full text for last 2, 300 chars for older"
```

---

### Task 2: Pre-seed strategy memos in strategist

**Files:**
- Modify: `v2/ideation_claude.py:7-14,280-314`
- Test: `tests/v2/test_ideation_claude.py`

- [ ] **Step 1: Write failing test for memo pre-seeding**

Add to `tests/v2/test_ideation_claude.py`:

```python
class TestStrategistPreSeedMemos:
    def test_strategy_history_in_preseed(self):
        """Strategy memos should appear in the pre-seeded initial message."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100
        mock_result.cache_creation_input_tokens = 0
        mock_result.cache_read_input_tokens = 0

        with patch("v2.ideation_claude.get_claude_client", return_value=MagicMock()), \
             patch("v2.ideation_claude.reset_session"), \
             patch("v2.ideation_claude.run_agentic_loop") as mock_loop, \
             patch("v2.ideation_claude.extract_final_text", return_value="Summary"), \
             patch("v2.ideation_claude.tool_get_portfolio_state", return_value="portfolio"), \
             patch("v2.ideation_claude.tool_get_active_theses", return_value="theses"), \
             patch("v2.ideation_claude.tool_get_decision_history", return_value="decisions"), \
             patch("v2.ideation_claude.tool_get_signal_attribution", return_value="attribution"), \
             patch("v2.ideation_claude.tool_get_strategy_identity", return_value="identity"), \
             patch("v2.ideation_claude.tool_get_strategy_rules", return_value="rules"), \
             patch("v2.ideation_claude.tool_get_strategy_history", return_value="memo content here"):
            mock_loop.return_value = mock_result

            run_strategist_loop(model="claude-opus-4-6", max_turns=1)

        call_kwargs = mock_loop.call_args
        initial_msg = call_kwargs.kwargs.get("initial_message", "")
        assert "Strategy History" in initial_msg or "strategy history" in initial_msg.lower()
        assert "memo content here" in initial_msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py::TestStrategistPreSeedMemos -v`
Expected: FAIL — `tool_get_strategy_history` is not imported or pre-seeded.

- [ ] **Step 3: Add memo pre-seeding to strategist**

Edit `v2/ideation_claude.py:9-14` — add `tool_get_strategy_history` to import:

```python
from .tools import (
    TOOL_DEFINITIONS, TOOL_HANDLERS, reset_session,
    tool_get_portfolio_state, tool_get_active_theses,
    tool_get_decision_history, tool_get_signal_attribution,
    tool_get_strategy_identity, tool_get_strategy_rules,
    tool_get_strategy_history,
)
```

Edit `v2/ideation_claude.py` — add after the Strategy Rules block (after line 305):

```python
    try:
        context_parts.append(f"=== Strategy History ===\n{tool_get_strategy_history()}")
    except Exception:
        context_parts.append("=== Strategy History ===\n(unavailable)")
```

Edit `v2/ideation_claude.py:314` — update the "do NOT re-fetch" instruction:

```python
1. Review the data above — do NOT re-fetch portfolio, theses, decisions, attribution, identity, rules, or strategy history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/ideation_claude.py tests/v2/test_ideation_claude.py
git commit -m "feat(v2): pre-seed strategy memos in strategist initial context"
```

---

### Task 3: Persist strategist summary as a memo

**Files:**
- Modify: `v2/session.py:175-185`
- Test: `tests/v2/test_session.py`

- [ ] **Step 1: Write failing test for memo persistence**

Add to `tests/v2/test_session.py`:

```python
class TestStrategistMemoPersistence:
    def test_strategist_summary_written_as_memo(self):
        """After Stage 2 completes, the summary should be saved as a strategy memo."""
        mock_ideation_result = MagicMock()
        mock_ideation_result.final_summary = "Strategist reasoning about today's decisions"

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", return_value=mock_ideation_result), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_strategy_memo") as mock_memo, \
             patch("v2.session.get_current_strategy_state", return_value={"id": 1}):

            run_session(dry_run=True)

        mock_memo.assert_called_once()
        call_kwargs = mock_memo.call_args
        # Check it was called with the right memo_type and content
        args = call_kwargs.kwargs if call_kwargs.kwargs else {}
        positional = call_kwargs.args if call_kwargs.args else ()
        assert "strategist_notes" in str(call_kwargs)
        assert "Strategist reasoning" in str(call_kwargs)

    def test_strategist_memo_not_written_on_failure(self):
        """If Stage 2 fails, no memo should be written."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", side_effect=Exception("Opus down")), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"), \
             patch("v2.session.insert_strategy_memo") as mock_memo:

            run_session(dry_run=True)

        mock_memo.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_session.py::TestStrategistMemoPersistence -v`
Expected: FAIL — `insert_strategy_memo` is not imported or called in session.py.

- [ ] **Step 3: Add memo persistence after Stage 2**

Edit `v2/session.py` — add imports near the top (after line 36):

```python
from .database.trading_db import (
    get_session_for_date, insert_session_record, complete_session, fail_session,
    insert_session_stage, complete_session_stage, fail_session_stage, get_completed_stages,
    insert_strategy_memo, get_current_strategy_state, get_playbook,
)
```

Edit `v2/session.py` — add after the strategist result is captured (after line 180, inside the `try` block, before the `complete_session_stage` call):

```python
            # Persist strategist reasoning for the reflection agent
            try:
                if result.strategist_result and result.strategist_result.final_summary:
                    state = get_current_strategy_state()
                    insert_strategy_memo(
                        session_date=today,
                        memo_type='strategist_notes',
                        content=result.strategist_result.final_summary,
                        strategy_state_id=state['id'] if state else None,
                    )
                    logger.info("Strategist summary saved as memo")
            except Exception as e:
                logger.warning("Could not save strategist memo: %s", e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: All PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "feat(v2): persist strategist summary as strategy memo for reflection agent"
```

---

## Phase 2: Structural Enforcement

### Task 4: Cooldown enforcement module

**Files:**
- Create: `v2/cooldown.py`
- Create: `tests/v2/test_cooldown.py`

- [ ] **Step 1: Write cooldown tests**

Create `tests/v2/test_cooldown.py`:

```python
"""Tests for v2/cooldown.py — structural cooldown enforcement."""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest


def _make_decision(ticker, action, days_ago, playbook_action_id=1):
    """Helper to create a decision dict at a specific age."""
    return {
        "ticker": ticker,
        "action": action,
        "date": date.today() - timedelta(days=days_ago),
        "playbook_action_id": playbook_action_id,
    }


class TestAddBusinessDays:
    def test_weekday_to_weekday(self):
        from v2.cooldown import _add_business_days
        # Monday + 3 business days = Thursday
        monday = date(2026, 3, 30)  # Monday
        assert _add_business_days(monday, 3) == date(2026, 4, 2)  # Thursday

    def test_friday_spans_weekend(self):
        from v2.cooldown import _add_business_days
        # Friday + 1 business day = Monday
        friday = date(2026, 3, 27)  # Friday
        assert _add_business_days(friday, 1) == date(2026, 3, 30)  # Monday

    def test_friday_plus_3(self):
        from v2.cooldown import _add_business_days
        # Friday + 3 business days = Wednesday
        friday = date(2026, 3, 27)  # Friday
        assert _add_business_days(friday, 3) == date(2026, 4, 1)  # Wednesday

    def test_zero_days(self):
        from v2.cooldown import _add_business_days
        monday = date(2026, 3, 30)
        assert _add_business_days(monday, 0) == monday


class TestGetTickerCooldowns:
    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_hold_blocks_sell_for_3_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=1)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "hold", days_ago=1),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert cooldowns["AAPL"].proposed_action == "sell"
        assert "Rule #20" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_hold_sell_allowed_after_cooldown(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=5)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "hold", days_ago=5),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" not in cooldowns

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_buy_blocks_sell_for_3_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=1)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "buy", days_ago=1),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert cooldowns["AAPL"].proposed_action == "sell"
        assert "Rule #3" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_sell_blocks_buy_for_5_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        mock_dates.return_value = {date.today() - timedelta(days=2)}
        mock_decisions.return_value = [
            _make_decision("AAPL", "sell", days_ago=2),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert cooldowns["AAPL"].proposed_action == "buy"
        assert "Rule #6" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_post_accumulation_sell_blocks_buy_for_10_days(self, mock_dates, mock_decisions):
        from v2.cooldown import get_ticker_cooldowns
        all_dates = {date.today() - timedelta(days=i) for i in range(10)}
        mock_dates.return_value = all_dates
        mock_decisions.return_value = [
            _make_decision("AAPL", "sell", days_ago=2),
            _make_decision("AAPL", "buy", days_ago=3),
            _make_decision("AAPL", "buy", days_ago=4),
            _make_decision("AAPL", "buy", days_ago=5),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" in cooldowns
        assert "Rule #10" in cooldowns["AAPL"].rule

    @patch("v2.cooldown.get_recent_decisions")
    @patch("v2.cooldown.get_playbook_dates")
    def test_fallback_hold_does_not_create_cooldown(self, mock_dates, mock_decisions):
        """HOLDs from sessions without a playbook should not create cooldowns."""
        from v2.cooldown import get_ticker_cooldowns
        decision_date = date.today() - timedelta(days=1)
        mock_dates.return_value = set()  # No playbooks — all decisions are fallback
        mock_decisions.return_value = [
            _make_decision("AAPL", "hold", days_ago=1, playbook_action_id=None),
        ]
        cooldowns = get_ticker_cooldowns()
        assert "AAPL" not in cooldowns


class TestCheckCooldown:
    def test_blocked_action(self):
        from v2.cooldown import check_cooldown, CooldownViolation
        cooldowns = {
            "AAPL": CooldownViolation(
                ticker="AAPL", proposed_action="sell", blocking_action="hold",
                blocking_date=date.today(), cooldown_expires=date.today() + timedelta(days=3),
                rule="test rule",
            )
        }
        blocked, reason = check_cooldown("AAPL", "sell", cooldowns)
        assert blocked is True
        assert "test rule" in reason

    def test_different_action_not_blocked(self):
        from v2.cooldown import check_cooldown, CooldownViolation
        cooldowns = {
            "AAPL": CooldownViolation(
                ticker="AAPL", proposed_action="sell", blocking_action="hold",
                blocking_date=date.today(), cooldown_expires=date.today() + timedelta(days=3),
                rule="test rule",
            )
        }
        blocked, reason = check_cooldown("AAPL", "buy", cooldowns)
        assert blocked is False

    def test_unknown_ticker_not_blocked(self):
        from v2.cooldown import check_cooldown
        blocked, reason = check_cooldown("MSFT", "sell", {})
        assert blocked is False


class TestFormatCooldownMap:
    def test_formats_for_executor(self):
        from v2.cooldown import format_cooldown_map, CooldownViolation
        cooldowns = {
            "AAPL": CooldownViolation(
                ticker="AAPL", proposed_action="sell", blocking_action="hold",
                blocking_date=date(2026, 4, 1), cooldown_expires=date(2026, 4, 4),
                rule="HOLD→SELL 3-day lockout",
            )
        }
        result = format_cooldown_map(cooldowns)
        assert result == {"AAPL": "HOLD→SELL 3-day lockout (expires 2026-04-04)"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_cooldown.py -v`
Expected: FAIL — `v2.cooldown` module does not exist.

- [ ] **Step 3: Implement cooldown module**

Create `v2/cooldown.py`:

```python
"""Cooldown enforcement — structural prevention of rapid action reversals.

Enforces rules that the LLM has repeatedly violated:
  Rule #20: HOLD → SELL: 3 business days
  Rule #3:  BUY  → SELL: 3 business days
  Rule #6:  SELL → BUY:  5 business days
  Rule #10: SELL (after 3+ BUYs) → BUY: 10 business days
"""

import logging
from datetime import date, timedelta
from dataclasses import dataclass

from .database.trading_db import get_recent_decisions

logger = logging.getLogger(__name__)


@dataclass
class CooldownViolation:
    ticker: str
    proposed_action: str
    blocking_action: str
    blocking_date: date
    cooldown_expires: date
    rule: str


def _add_business_days(start: date, days: int) -> date:
    """Add business days (Mon-Fri) to a date."""
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def get_playbook_dates(lookback_days: int = 15) -> set[date]:
    """Get dates that have a playbook (used to identify fallback decisions)."""
    from .database.trading_db import get_cursor
    from .database.connection import get_cursor
    cutoff = date.today() - timedelta(days=lookback_days)
    with get_cursor() as cur:
        cur.execute("SELECT DISTINCT date FROM playbooks WHERE date >= %s", (cutoff,))
        return {row["date"] for row in cur.fetchall()}


def get_ticker_cooldowns(lookback_days: int = 15) -> dict[str, CooldownViolation]:
    """Compute active cooldowns for all tickers based on recent decisions.

    Returns dict of ticker -> CooldownViolation for tickers currently in cooldown.
    Fallback HOLDs (from sessions without a playbook) are excluded.
    """
    decisions = get_recent_decisions(days=lookback_days)
    playbook_dates = get_playbook_dates(lookback_days)
    today = date.today()
    cooldowns: dict[str, CooldownViolation] = {}

    # Group decisions by ticker, most recent first (query returns DESC)
    by_ticker: dict[str, list] = {}
    for d in decisions:
        ticker = d["ticker"]
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(d)

    for ticker, ticker_decisions in by_ticker.items():
        if not ticker_decisions:
            continue

        latest = ticker_decisions[0]
        latest_action = latest["action"]
        latest_date = latest["date"]

        # Skip fallback HOLDs (no playbook existed for that date)
        if latest_action == "hold" and latest_date not in playbook_dates:
            continue

        # HOLD → SELL lockout: 3 business days (Rule #20)
        if latest_action == "hold":
            expires = _add_business_days(latest_date, 3)
            if today < expires:
                cooldowns[ticker] = CooldownViolation(
                    ticker=ticker, proposed_action="sell",
                    blocking_action="hold", blocking_date=latest_date,
                    cooldown_expires=expires,
                    rule="HOLD→SELL 3-day lockout (Rule #20)",
                )

        # BUY → SELL lockout: 3 business days (Rule #3)
        elif latest_action == "buy":
            expires = _add_business_days(latest_date, 3)
            if today < expires:
                cooldowns[ticker] = CooldownViolation(
                    ticker=ticker, proposed_action="sell",
                    blocking_action="buy", blocking_date=latest_date,
                    cooldown_expires=expires,
                    rule="BUY→SELL 3-day lockout (Rule #3)",
                )

        # SELL → BUY lockout: 5 or 10 business days (Rule #6 / #10)
        elif latest_action == "sell":
            # Check for accumulation streak (3+ consecutive BUYs before the SELL)
            buy_streak = 0
            for d in ticker_decisions[1:]:
                if d["action"] == "buy":
                    buy_streak += 1
                else:
                    break

            if buy_streak >= 3:
                expires = _add_business_days(latest_date, 10)
                rule = f"Post-accumulation SELL→BUY 10-day lockout (Rule #10, {buy_streak} prior BUYs)"
            else:
                expires = _add_business_days(latest_date, 5)
                rule = "SELL→BUY/HOLD 5-day lockout (Rule #6)"

            if today < expires:
                cooldowns[ticker] = CooldownViolation(
                    ticker=ticker, proposed_action="buy",
                    blocking_action="sell", blocking_date=latest_date,
                    cooldown_expires=expires, rule=rule,
                )

    return cooldowns


def check_cooldown(
    ticker: str, proposed_action: str, cooldowns: dict[str, CooldownViolation],
) -> tuple[bool, str]:
    """Check if a proposed action violates cooldown. Returns (is_blocked, reason)."""
    if ticker not in cooldowns:
        return False, ""

    violation = cooldowns[ticker]
    if proposed_action == violation.proposed_action:
        return True, (
            f"Cooldown: {violation.rule} — "
            f"{violation.blocking_action.upper()} on {violation.blocking_date}, "
            f"expires {violation.cooldown_expires}"
        )

    return False, ""


def format_cooldown_map(cooldowns: dict[str, CooldownViolation]) -> dict[str, str]:
    """Format cooldowns as human-readable strings for the executor."""
    return {
        ticker: f"{v.rule} (expires {v.cooldown_expires})"
        for ticker, v in cooldowns.items()
    }
```

- [ ] **Step 4: Fix the duplicate import in get_playbook_dates**

The `get_playbook_dates` function has a duplicate import. Fix it:

```python
def get_playbook_dates(lookback_days: int = 15) -> set[date]:
    """Get dates that have a playbook (used to identify fallback decisions)."""
    from .database.connection import get_cursor
    cutoff = date.today() - timedelta(days=lookback_days)
    with get_cursor() as cur:
        cur.execute("SELECT DISTINCT date FROM playbooks WHERE date >= %s", (cutoff,))
        return {row["date"] for row in cur.fetchall()}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_cooldown.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add v2/cooldown.py tests/v2/test_cooldown.py
git commit -m "feat(v2): add cooldown enforcement module with structural rule enforcement"
```

---

### Task 5: Integrate cooldowns into executor pipeline

**Files:**
- Modify: `v2/agent.py:40-53,90-123,143-153`
- Modify: `v2/context.py:374-428`
- Modify: `v2/trader.py:234-259`
- Test: `tests/v2/test_context.py`, `tests/v2/test_trader.py`

- [ ] **Step 1: Add cooldown_tickers field to ExecutorInput**

Edit `v2/agent.py:40-53`:

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
    cooldown_tickers: dict[str, str] = None

    def __post_init__(self):
        if self.current_prices is None:
            self.current_prices = {}
        if self.cooldown_tickers is None:
            self.cooldown_tickers = {}
```

- [ ] **Step 2: Add cooldown to executor system prompt**

Edit `v2/agent.py` — add after line 102 (`8. current_prices...`):

```
9. cooldown_tickers — tickers with active trading restrictions. Do NOT propose actions that violate these cooldowns.
```

- [ ] **Step 3: Add cooldown to serialization**

Edit `v2/agent.py:143-153` — add `cooldown_tickers` to the `input_data` dict:

```python
    input_data = {
        "playbook_actions": [asdict(a) for a in executor_input.playbook_actions],
        "positions": executor_input.positions,
        "account": executor_input.account,
        "attribution_summary": executor_input.attribution_summary,
        "recent_outcomes": executor_input.recent_outcomes,
        "market_outlook": executor_input.market_outlook,
        "risk_notes": executor_input.risk_notes,
        "current_prices": {k: str(v) for k, v in executor_input.current_prices.items()},
        "cooldown_tickers": executor_input.cooldown_tickers,
    }
```

- [ ] **Step 4: Integrate cooldowns into build_executor_input**

Edit `v2/context.py:374` — add import at top of file:

```python
from .cooldown import get_ticker_cooldowns, format_cooldown_map
```

Edit `v2/context.py` — in `build_executor_input`, after the `actions` list is built (after line 390) and before the return (before line 419), add cooldown filtering and map:

```python
    # Compute cooldowns and filter blocked playbook actions
    cooldowns = get_ticker_cooldowns()
    if cooldowns:
        blocked = [a.ticker for a in actions if a.ticker in cooldowns
                   and a.action == cooldowns[a.ticker].proposed_action]
        if blocked:
            logger.info("Cooldown: filtering %d playbook actions: %s", len(blocked), blocked)
        actions = [a for a in actions if not (a.ticker in cooldowns
                   and a.action == cooldowns[a.ticker].proposed_action)]
    cooldown_map = format_cooldown_map(cooldowns)
```

Then update the return statement to include `cooldown_tickers=cooldown_map`:

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
        cooldown_tickers=cooldown_map,
    )
```

- [ ] **Step 5: Add post-validation cooldown check in trader.py**

Edit `v2/trader.py` — add import near top:

```python
from .cooldown import get_ticker_cooldowns, check_cooldown
```

Edit `v2/trader.py` — after `open_sell_orders` is built (after line 224) and before the loop, compute cooldowns:

```python
    cooldowns = get_ticker_cooldowns()
```

Edit `v2/trader.py` — inside the loop, after the hold check (after line 237) and before the trade limit check, add cooldown validation:

```python
        # Cooldown check (safety net for off-playbook decisions)
        is_blocked, cooldown_reason = check_cooldown(decision.ticker, decision.action, cooldowns)
        if is_blocked:
            errors.append(f"{decision.ticker} cooldown: {cooldown_reason}")
            logger.warning("%s: COOLDOWN - %s", decision.ticker, cooldown_reason)
            trades_failed += 1
            continue
```

- [ ] **Step 6: Write tests for cooldown integration**

Add to `tests/v2/test_context.py`:

```python
class TestCooldownFiltering:
    @patch("v2.context.get_ticker_cooldowns")
    @patch("v2.context.format_cooldown_map")
    def test_cooldown_tickers_in_executor_input(self, mock_format, mock_cooldowns, mock_db, mock_cursor):
        """build_executor_input should include cooldown_tickers."""
        from v2.context import build_executor_input
        from v2.cooldown import CooldownViolation
        from datetime import date, timedelta

        mock_cooldowns.return_value = {}
        mock_format.return_value = {}
        mock_cursor.fetchone.return_value = {"id": 1, "date": date.today(),
            "market_outlook": "test", "risk_notes": "test"}
        mock_cursor.fetchall.return_value = []

        with patch("v2.context.get_latest_price", return_value=None):
            result = build_executor_input(
                {"cash": 10000, "portfolio_value": 50000, "buying_power": 10000}
            )

        assert hasattr(result, "cooldown_tickers")
        assert result.cooldown_tickers == {}
```

- [ ] **Step 7: Run all tests**

Run: `python3 -m pytest tests/v2/test_cooldown.py tests/v2/test_context.py tests/v2/test_trader.py tests/v2/test_agent.py -v`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add v2/agent.py v2/context.py v2/trader.py tests/v2/test_context.py tests/v2/test_trader.py
git commit -m "feat(v2): integrate cooldown enforcement into executor pipeline"
```

---

### Task 6: Skip executor without playbook

**Files:**
- Modify: `v2/session.py:195-213`
- Test: `tests/v2/test_session.py`

- [ ] **Step 1: Write failing test**

Add to `tests/v2/test_session.py`:

```python
class TestExecutorPlaybookDependency:
    def test_executor_skipped_when_strategist_fails_and_no_playbook(self):
        """Stage 3 should be skipped if Stage 2 failed and no playbook exists."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", side_effect=Exception("Opus down")), \
             patch("v2.session.get_playbook", return_value=None), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

        mock_trade.assert_not_called()
        assert result.skipped_executor is True
        assert result.strategist_error == "Opus down"

    def test_executor_runs_when_strategist_fails_but_playbook_exists(self):
        """Stage 3 should run if Stage 2 failed but a prior playbook exists."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop", side_effect=Exception("Opus down")), \
             patch("v2.session.get_playbook", return_value={"id": 1}), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

        mock_trade.assert_called_once()

    def test_executor_runs_normally_when_strategist_succeeds(self):
        """Stage 3 should run normally when Stage 2 succeeds."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_twitter_stage"), \
             patch("v2.session.run_bluesky_stage"), \
             patch("v2.session.run_dashboard_stage"):

            result = run_session(dry_run=True)

        mock_trade.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_session.py::TestExecutorPlaybookDependency -v`
Expected: FAIL — executor runs even when strategist fails and no playbook.

- [ ] **Step 3: Add playbook dependency check**

`get_playbook` was already added to the session.py imports in Task 3. If not, add it now.

Edit `v2/session.py:195-200` — replace the Stage 3 skip check:

```python
    # Stage 3: Trading session
    # Skip if strategist failed and no playbook exists (executor depends on playbook)
    if not skip_executor and not ("executor" in completed_stages) and result.strategist_error:
        playbook_exists = get_playbook(today) is not None
        if not playbook_exists:
            logger.warning("Strategist failed and no playbook exists for %s — skipping executor", today)
            skip_executor = True
            result.skipped_executor = True

    if skip_executor or "executor" in completed_stages:
        logger.info("[Stage 3] Trading executor — SKIPPED%s",
                     " (completed in prior run)" if "executor" in completed_stages else "")
    else:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "fix(v2): skip executor when strategist fails and no playbook exists"
```

---

## Phase 3: Identity & Rule Hygiene

### Task 7: Stabilize strategy identity

**Files:**
- Modify: `v2/strategy.py:42-67,72-93`
- Test: `tests/v2/test_strategy.py`

- [ ] **Step 1: Write failing test for identity update guard**

Add to `tests/v2/test_strategy.py`:

```python
class TestIdentityUpdateGuard:
    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_warns_if_recently_updated(self, mock_get, mock_clear, mock_insert):
        """Should return warning if identity was updated within 3 days."""
        from v2.strategy import tool_update_strategy_identity
        from datetime import datetime, timezone
        mock_get.return_value = make_strategy_state_row(
            version=5,
            created_at=datetime.now(timezone.utc),  # Updated today
        )

        result = tool_update_strategy_identity(
            identity_text="New identity",
            risk_posture="aggressive",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )

        # Should warn, not update
        assert "Warning" in result
        mock_clear.assert_not_called()
        mock_insert.assert_not_called()

    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_allows_update_if_not_recent(self, mock_get, mock_clear, mock_insert):
        """Should allow update if identity hasn't been updated in >3 days."""
        from v2.strategy import tool_update_strategy_identity
        from datetime import datetime, timedelta, timezone
        mock_get.return_value = make_strategy_state_row(
            version=5,
            created_at=datetime.now(timezone.utc) - timedelta(days=5),
        )
        mock_insert.return_value = 6

        result = tool_update_strategy_identity(
            identity_text="New identity",
            risk_posture="moderate",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )

        assert "version 6" in result
        mock_clear.assert_called_once()
        mock_insert.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestIdentityUpdateGuard -v`
Expected: FAIL — no guard exists.

- [ ] **Step 3: Add identity update guard**

Edit `v2/strategy.py:72-93` — add guard at the start of `tool_update_strategy_identity`:

```python
def tool_update_strategy_identity(
    identity_text: str,
    risk_posture: str,
    sector_biases: dict,
    preferred_signals: list,
    avoided_signals: list,
) -> str:
    """Update the system's strategy identity (creates new versioned row)."""
    logger.info("Updating strategy identity")
    current = get_current_strategy_state()

    # Soft guard: warn if updated within last 3 days
    if current and (date.today() - current["created_at"].date()).days < 3:
        return (
            f"Warning: Identity was updated within the last 3 days "
            f"(v{current['version']} on {current['created_at'].date()}). "
            f"Consider writing a memo instead unless the system's fundamental "
            f"character has changed. To proceed anyway, call update_strategy_identity again."
        )

    new_version = (current["version"] + 1) if current else 1

    clear_current_strategy_state()
    state_id = insert_strategy_state(
        identity_text=identity_text,
        risk_posture=risk_posture,
        sector_biases=sector_biases,
        preferred_signals=preferred_signals,
        avoided_signals=avoided_signals,
        version=new_version,
    )
    return f"Strategy identity updated to version {new_version} (ID: {state_id})"
```

Ensure `date` is imported at the top of the file (it already is: `from datetime import date`).

- [ ] **Step 4: Add identity vs. memos guidance to reflection prompt**

Edit `v2/strategy.py:42-67` — append to `STRATEGY_REFLECTION_SYSTEM` before the closing `"""`:

```

## Identity vs. Memos

The strategy identity describes WHO this system is as a trader — its style, risk philosophy, signal preferences, and core beliefs. It should be stable across sessions and NOT reference specific session numbers, individual trades, or recent events.

Session-specific observations belong in memos, not the identity.

Only update the identity when the system's fundamental character has genuinely shifted (e.g., from momentum trader to value trader, or from aggressive to conservative). Cosmetic updates ("in its 36th session") are not identity changes.

A good identity reads like a bio. A bad identity reads like a session log.
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_strategy.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add v2/strategy.py tests/v2/test_strategy.py
git commit -m "feat(v2): stabilize identity updates with 3-day guard and prompt guidance"
```

---

### Task 8: Rule management improvements

**Files:**
- Modify: `v2/strategy.py:42-67,115-120`
- Modify: `v2/database/trading_db.py:535-541`
- Test: `tests/v2/test_strategy.py`, `tests/v2/test_db.py`
- Migration: `ALTER TABLE strategy_rules ADD COLUMN retirement_reason TEXT`

- [ ] **Step 1: Add retirement_reason column**

Run the migration against the database:

```bash
docker compose exec -T trading python -c "
from v2.database.connection import get_cursor
with get_cursor() as cur:
    cur.execute('ALTER TABLE strategy_rules ADD COLUMN IF NOT EXISTS retirement_reason TEXT')
print('Migration complete')
"
```

- [ ] **Step 2: Write failing test for reason persistence**

Add to `tests/v2/test_strategy.py`:

```python
class TestRetireRuleWithReason:
    @patch("v2.strategy.retire_strategy_rule")
    def test_passes_reason_to_db(self, mock_retire):
        """retire_rule should pass the reason to the database function."""
        from v2.strategy import tool_retire_rule
        mock_retire.return_value = True
        tool_retire_rule(rule_id=5, reason="Superseded by structural enforcement")
        mock_retire.assert_called_once_with(rule_id=5, reason="Superseded by structural enforcement")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestRetireRuleWithReason -v`
Expected: FAIL — `retire_strategy_rule` does not accept `reason`.

- [ ] **Step 4: Update retire_strategy_rule to persist reason**

Edit `v2/database/trading_db.py:535-541`:

```python
def retire_strategy_rule(rule_id, reason=None) -> bool:
    with get_cursor() as cur:
        cur.execute("""
            UPDATE strategy_rules
            SET status = 'retired', retired_at = NOW(), retirement_reason = %s
            WHERE id = %s AND status = 'active'
        """, (reason, rule_id))
        return cur.rowcount > 0
```

- [ ] **Step 5: Update tool_retire_rule to pass reason through**

Edit `v2/strategy.py:115-120`:

```python
def tool_retire_rule(rule_id: int, reason: str) -> str:
    """Retire a strategy rule."""
    logger.info(f"Retiring rule {rule_id}: {reason}")
    success = retire_strategy_rule(rule_id=rule_id, reason=reason)
    if success:
        return f"Retired rule ID {rule_id}. Reason: {reason}"
    return f"Error: Rule ID {rule_id} not found or already retired"
```

- [ ] **Step 6: Add rule management guidance to reflection prompt**

Edit `v2/strategy.py:42-67` — append to `STRATEGY_REFLECTION_SYSTEM` (after the Identity vs. Memos section added in Task 7):

```

## Rule Management

Before proposing a new rule:
1. Check if an existing active rule covers the same pattern
2. If so, update the existing rule's confidence, scope, or evidence rather than creating a new one
3. Only create a new rule if the pattern is genuinely distinct from all existing rules
```

- [ ] **Step 7: Update conftest factory for new column**

Edit `tests/v2/conftest.py` — update `make_strategy_rule_row` defaults:

```python
def make_strategy_rule_row(**kwargs):
    """Create a strategy_rules dict like what DB returns."""
    defaults = {
        "id": 1,
        "rule_text": "Fade legal news signals — 38% win rate over 12 trades",
        "category": "news_signal:legal",
        "direction": "constraint",
        "confidence": Decimal("0.80"),
        "supporting_evidence": "Historical win rate below 40% across 12 decisions",
        "status": "active",
        "created_at": datetime.now(),
        "retired_at": None,
        "retirement_reason": None,
    }
    defaults.update(kwargs)
    return defaults
```

- [ ] **Step 8: Run all tests**

Run: `python3 -m pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 9: Commit**

```bash
git add v2/strategy.py v2/database/trading_db.py tests/v2/test_strategy.py tests/v2/conftest.py
git commit -m "feat(v2): persist rule retirement reason and add rule management prompt guidance"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
python3 -m pytest tests/ -v --tb=short
```

- [ ] **Verify no import errors in new module**

```bash
python3 -c "from v2.cooldown import get_ticker_cooldowns, check_cooldown, format_cooldown_map; print('OK')"
```
