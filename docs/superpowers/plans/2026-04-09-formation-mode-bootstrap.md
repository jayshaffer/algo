# Formation Mode & Orphan Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the cold-start death spiral where no attribution data leads to no playbook actions leads to no trades leads to no attribution data. The system should detect when it's in "formation" mode and behave more exploratorily until it has enough completed trade cycles to self-sustain.

**Architecture:** Two complementary mechanisms injected into the existing session pipeline:
1. A `formation` module that detects cold-start conditions (no completed trade cycles, orphan positions) and generates context injections for the strategist and reflection prompts.
2. An `adopt_thesis` tool that lets the strategist create theses for positions already in the portfolio (bypassing the current `create_thesis` guard), so orphan positions can be reasoned about.

Formation mode is self-healing: once the system accumulates enough completed trade cycles (configurable threshold, default 5), the formation context drops away and normal behavior resumes.

**Tech Stack:** Python, psycopg2 (raw SQL), existing Claude agentic loop infrastructure

---

## File Structure

| File | Responsibility |
|------|---------------|
| `v2/formation.py` (create) | Cold-start detection, orphan detection, formation context generation |
| `v2/tools.py` (modify) | Add `adopt_thesis` tool handler + definition |
| `v2/ideation_claude.py` (modify) | Inject formation context into strategist prompt |
| `v2/strategy.py` (modify) | Inject formation context into reflection prompt |
| `tests/v2/test_formation.py` (create) | Tests for formation detection and context generation |
| `tests/v2/test_tools.py` (modify) | Tests for the adopt_thesis tool (add to existing) |
| `tests/v2/test_ideation_claude.py` (modify) | Test formation context injection in strategist |
| `tests/v2/test_strategy.py` (modify) | Test formation context injection in reflection |

---

### Task 1: Formation detection module

**Files:**
- Create: `v2/formation.py`
- Create: `tests/v2/test_formation.py`

This module answers two questions: (1) is the system in formation mode? (2) what orphan positions exist?

- [ ] **Step 1: Write failing tests for `is_formation_mode()`**

```python
"""Tests for v2/formation.py - cold-start detection and formation context."""

from unittest.mock import patch

import pytest

from v2.formation import is_formation_mode, FORMATION_TRADE_THRESHOLD


class TestIsFormationMode:

    @patch("v2.formation.get_recent_decisions")
    def test_formation_when_no_decisions(self, mock_decisions):
        mock_decisions.return_value = []
        assert is_formation_mode() is True

    @patch("v2.formation.get_recent_decisions")
    def test_formation_when_few_completed_cycles(self, mock_decisions):
        # 3 decisions with outcomes, below threshold of 5
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "buy"},
            {"outcome_7d": -0.8, "action": "buy"},
            {"outcome_7d": 2.0, "action": "sell"},
        ]
        assert is_formation_mode() is True

    @patch("v2.formation.get_recent_decisions")
    def test_not_formation_when_enough_cycles(self, mock_decisions):
        # 5 decisions with 7d outcomes = enough completed cycles
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "buy"},
            {"outcome_7d": -0.8, "action": "buy"},
            {"outcome_7d": 2.0, "action": "sell"},
            {"outcome_7d": 0.3, "action": "buy"},
            {"outcome_7d": -1.2, "action": "sell"},
        ]
        assert is_formation_mode() is False

    @patch("v2.formation.get_recent_decisions")
    def test_hold_decisions_not_counted(self, mock_decisions):
        # hold decisions don't count as completed cycles
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "hold"},
            {"outcome_7d": -0.8, "action": "hold"},
            {"outcome_7d": 2.0, "action": "buy"},
        ]
        assert is_formation_mode() is True

    @patch("v2.formation.get_recent_decisions")
    def test_null_outcomes_not_counted(self, mock_decisions):
        # decisions without 7d outcome aren't completed cycles
        mock_decisions.return_value = [
            {"outcome_7d": 1.5, "action": "buy"},
            {"outcome_7d": None, "action": "buy"},
            {"outcome_7d": None, "action": "sell"},
            {"outcome_7d": 2.0, "action": "sell"},
        ]
        assert is_formation_mode() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_formation.py::TestIsFormationMode -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement `is_formation_mode()`**

```python
"""Formation mode detection and context generation.

Detects cold-start conditions (no completed trade cycles, orphan positions)
and generates context injections that encourage exploratory behavior until
the system has enough data to self-sustain.
"""

import logging
from datetime import date

from .database.trading_db import (
    get_recent_decisions,
    get_positions,
    get_active_theses,
)

logger = logging.getLogger(__name__)

FORMATION_TRADE_THRESHOLD = 5


def is_formation_mode() -> bool:
    """Check if the system is in formation mode.

    Formation mode is active when fewer than FORMATION_TRADE_THRESHOLD
    buy/sell decisions have completed 7-day outcome cycles.
    """
    decisions = get_recent_decisions(days=90)
    completed_cycles = sum(
        1 for d in decisions
        if d.get("outcome_7d") is not None and d["action"] in ("buy", "sell")
    )
    return completed_cycles < FORMATION_TRADE_THRESHOLD
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_formation.py::TestIsFormationMode -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Write failing tests for `get_orphan_positions()`**

Add to `tests/test_formation.py`:

```python
from v2.formation import get_orphan_positions


class TestGetOrphanPositions:

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_all_positions_orphaned(self, mock_positions, mock_theses):
        mock_positions.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
            {"ticker": "NVDA", "shares": 10, "avg_cost": 800.0},
        ]
        mock_theses.return_value = []
        orphans = get_orphan_positions()
        assert len(orphans) == 2
        assert orphans[0]["ticker"] == "AAPL"

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_no_orphans_when_all_covered(self, mock_positions, mock_theses):
        mock_positions.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
        ]
        mock_theses.return_value = [
            {"ticker": "AAPL", "direction": "long"},
        ]
        orphans = get_orphan_positions()
        assert len(orphans) == 0

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_mixed_orphans(self, mock_positions, mock_theses):
        mock_positions.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
            {"ticker": "COIN", "shares": 10, "avg_cost": 200.0},
            {"ticker": "NVDA", "shares": 10, "avg_cost": 800.0},
        ]
        mock_theses.return_value = [
            {"ticker": "COIN", "direction": "long"},
        ]
        orphans = get_orphan_positions()
        assert len(orphans) == 2
        tickers = [o["ticker"] for o in orphans]
        assert "AAPL" in tickers
        assert "NVDA" in tickers
        assert "COIN" not in tickers

    @patch("v2.formation.get_active_theses")
    @patch("v2.formation.get_positions")
    def test_no_positions(self, mock_positions, mock_theses):
        mock_positions.return_value = []
        mock_theses.return_value = []
        orphans = get_orphan_positions()
        assert len(orphans) == 0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_formation.py::TestGetOrphanPositions -v`
Expected: FAIL (function does not exist)

- [ ] **Step 7: Implement `get_orphan_positions()`**

Add to `v2/formation.py`:

```python
def get_orphan_positions() -> list[dict]:
    """Find positions with no active thesis.

    Returns list of position dicts for tickers that have no
    corresponding active thesis.
    """
    positions = get_positions()
    if not positions:
        return []

    theses = get_active_theses()
    covered_tickers = {t["ticker"] for t in theses}

    return [p for p in positions if p["ticker"] not in covered_tickers]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_formation.py::TestGetOrphanPositions -v`
Expected: PASS (all 4 tests)

- [ ] **Step 9: Commit**

```bash
git add v2/formation.py tests/v2/test_formation.py
git commit -m "feat(formation): add cold-start detection and orphan position finder"
```

---

### Task 2: Formation context generation

**Files:**
- Modify: `v2/formation.py`
- Modify: `tests/v2/test_formation.py`

Build the text blocks that get injected into strategist and reflection prompts when formation mode is active.

- [ ] **Step 1: Write failing tests for `build_formation_context()`**

Add to `tests/test_formation.py`:

```python
from v2.formation import build_formation_context


class TestBuildFormationContext:

    @patch("v2.formation.get_orphan_positions")
    @patch("v2.formation.is_formation_mode")
    def test_returns_empty_when_not_formation(self, mock_formation, mock_orphans):
        mock_formation.return_value = False
        mock_orphans.return_value = []
        ctx = build_formation_context()
        assert ctx == ""

    @patch("v2.formation.get_orphan_positions")
    @patch("v2.formation.is_formation_mode")
    def test_formation_with_orphans(self, mock_formation, mock_orphans):
        mock_formation.return_value = True
        mock_orphans.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
            {"ticker": "NVDA", "shares": 10, "avg_cost": 800.0},
        ]
        ctx = build_formation_context()
        assert "FORMATION MODE" in ctx
        assert "AAPL" in ctx
        assert "NVDA" in ctx
        assert "adopt_thesis" in ctx

    @patch("v2.formation.get_orphan_positions")
    @patch("v2.formation.is_formation_mode")
    def test_formation_without_orphans(self, mock_formation, mock_orphans):
        mock_formation.return_value = True
        mock_orphans.return_value = []
        ctx = build_formation_context()
        assert "FORMATION MODE" in ctx
        assert "adopt_thesis" not in ctx

    @patch("v2.formation.get_orphan_positions")
    @patch("v2.formation.is_formation_mode")
    def test_not_formation_but_orphans_still_flagged(self, mock_formation, mock_orphans):
        """Even outside formation mode, orphan positions should be flagged."""
        mock_formation.return_value = False
        mock_orphans.return_value = [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
        ]
        ctx = build_formation_context()
        assert "AAPL" in ctx
        assert "FORMATION MODE" not in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_formation.py::TestBuildFormationContext -v`
Expected: FAIL (function does not exist)

- [ ] **Step 3: Implement `build_formation_context()`**

Add to `v2/formation.py`:

```python
def build_formation_context() -> str:
    """Build context injection for strategist/reflection prompts.

    Returns a string to append to the system prompt. Empty string
    if neither formation mode nor orphan positions apply.
    """
    in_formation = is_formation_mode()
    orphans = get_orphan_positions()

    if not in_formation and not orphans:
        return ""

    parts = []

    if in_formation:
        parts.append(f"""## FORMATION MODE ACTIVE

This system has fewer than {FORMATION_TRADE_THRESHOLD} completed trade cycles. \
It cannot learn without trades. Your priority is to break the cold-start loop:

1. **Generate actionable playbook actions.** Every session MUST produce at least 2-3 \
concrete buy/sell actions in the playbook. Conservative inaction is the wrong move \
during formation — the system needs live trades to build its evidence base.
2. **Size positions small.** Use small position sizes (1-5 shares or $200-$500) to \
limit downside while generating learning data.
3. **Prefer high-conviction, well-researched plays.** Don't trade randomly — but don't \
let the absence of historical data paralyze you either. Use your market research to \
form theses and act on them.
4. **Formation mode exits automatically** once {FORMATION_TRADE_THRESHOLD} trades have \
completed their 7-day outcome measurement.""")

    if orphans:
        orphan_lines = []
        for p in orphans:
            orphan_lines.append(
                f"- {p['ticker']}: {p['shares']} shares @ ${float(p['avg_cost']):.2f} avg"
            )
        orphan_list = "\n".join(orphan_lines)

        if in_formation:
            parts.append(f"""
## ORPHAN POSITIONS (no thesis)

These positions exist in the portfolio but have no active thesis. The system \
cannot reason about them, set exit triggers, or learn from them.

{orphan_list}

Use the `adopt_thesis` tool to create theses for these positions. For each one, \
decide: is this a hold worth keeping (create a long thesis with exit/invalidation \
triggers), or should it be exited (create a playbook sell action)?""")
        else:
            parts.append(f"""
## ORPHAN POSITIONS (no thesis)

These positions have no active thesis. Use `adopt_thesis` to create theses so the \
system can reason about exit triggers and learn from outcomes.

{orphan_list}""")

    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_formation.py::TestBuildFormationContext -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add v2/formation.py tests/v2/test_formation.py
git commit -m "feat(formation): add context builder for strategist prompt injection"
```

---

### Task 3: Add `adopt_thesis` tool

**Files:**
- Modify: `v2/tools.py` (add handler + definition)
- Modify: `tests/v2/test_tools.py` (add adopt_thesis tests)

The existing `create_thesis` tool rejects tickers already in the portfolio (line 110-111 of `v2/tools.py`). We need a separate `adopt_thesis` tool that does the opposite: it *requires* the ticker to be in the portfolio.

- [ ] **Step 1: Write failing tests for `tool_adopt_thesis()`**

Add to `tests/v2/test_tools.py` — update the import block to include `tool_adopt_thesis`, then add a new test class:

```python
from v2.tools import tool_adopt_thesis  # add to existing imports


class TestToolAdoptThesis:

    @patch("v2.tools.insert_thesis")
    @patch("v2.tools.get_active_theses")
    @patch("v2.tools.get_positions")
    def test_adopt_success(self, mock_positions, mock_theses, mock_insert):
        mock_positions.return_value = [{"ticker": "AAPL"}]
        mock_theses.return_value = []
        mock_insert.return_value = 42

        result = tool_adopt_thesis(
            ticker="AAPL",
            direction="long",
            thesis="Strong ecosystem and services growth",
            exit_trigger="Price drops below $130",
            invalidation="iPhone sales decline 3 consecutive quarters",
            confidence="medium",
        )
        assert "Created thesis ID 42" in result
        assert "adopted" in result.lower() or "AAPL" in result
        mock_insert.assert_called_once()
        # Verify source is "adoption" not "claude_ideation"
        call_kwargs = mock_insert.call_args
        assert call_kwargs[1]["source"] == "adoption" or call_kwargs[0][-1] == "adoption"

    @patch("v2.tools.get_active_theses")
    @patch("v2.tools.get_positions")
    def test_reject_not_in_portfolio(self, mock_positions, mock_theses):
        mock_positions.return_value = [{"ticker": "NVDA"}]
        mock_theses.return_value = []

        result = tool_adopt_thesis(
            ticker="TSLA",
            direction="long",
            thesis="EV growth",
            exit_trigger="...",
            invalidation="...",
            confidence="medium",
        )
        assert "Error" in result
        assert "not in the portfolio" in result

    @patch("v2.tools.get_active_theses")
    @patch("v2.tools.get_positions")
    def test_reject_existing_thesis(self, mock_positions, mock_theses):
        mock_positions.return_value = [{"ticker": "AAPL"}]
        mock_theses.return_value = [{"id": 1, "ticker": "AAPL"}]

        result = tool_adopt_thesis(
            ticker="AAPL",
            direction="long",
            thesis="...",
            exit_trigger="...",
            invalidation="...",
            confidence="medium",
        )
        assert "Error" in result
        assert "already exists" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_tools.py::TestToolAdoptThesis -v`
Expected: FAIL (function does not exist)

- [ ] **Step 3: Implement `tool_adopt_thesis()` and register it**

Add the handler to `v2/tools.py` after `tool_create_thesis`:

```python
def tool_adopt_thesis(
    ticker: str,
    direction: str,
    thesis: str,
    exit_trigger: str,
    invalidation: str,
    confidence: str,
) -> str:
    """Adopt an existing portfolio position by creating a thesis for it.

    Unlike create_thesis, this REQUIRES the ticker to already be in the portfolio.
    Used to bring orphan positions under thesis management.
    """
    logger.info(f"Adopting thesis for existing position {ticker} ({direction})")

    existing = get_active_theses(ticker=ticker)
    if existing:
        return (
            f"Error: Active thesis already exists for {ticker} "
            f"(ID {existing[0]['id']}). Update it instead."
        )

    positions = {p["ticker"] for p in get_positions()}
    if ticker not in positions:
        return f"Error: {ticker} is not in the portfolio. Use create_thesis for new ideas."

    thesis_id = insert_thesis(
        ticker=ticker,
        direction=direction,
        thesis=thesis,
        entry_trigger="Already held — adopted into thesis management",
        exit_trigger=exit_trigger,
        invalidation=invalidation,
        confidence=confidence,
        source="adoption",
    )

    logger.info(f"Adopted position {ticker} as thesis ID {thesis_id}")
    return f"Created thesis ID {thesis_id} for {ticker} (adopted existing position, {direction}, {confidence} confidence)"
```

Add the tool definition to `TOOL_DEFINITIONS` list (after the `create_thesis` entry):

```python
{
    "name": "adopt_thesis",
    "description": "Adopt an existing portfolio position by creating a thesis. Use for orphan positions (held but no thesis). REQUIRES ticker to be in portfolio.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "direction": {"type": "string", "enum": ["long", "short", "avoid"]},
            "thesis": {"type": "string", "description": "Why you believe in this position"},
            "exit_trigger": {"type": "string", "description": "When to exit"},
            "invalidation": {"type": "string", "description": "What proves thesis wrong"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["ticker", "direction", "thesis", "exit_trigger", "invalidation", "confidence"],
    },
},
```

Add to `TOOL_HANDLERS` dict:

```python
"adopt_thesis": tool_adopt_thesis,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_tools.py::TestToolAdoptThesis -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add v2/tools.py tests/v2/test_tools.py
git commit -m "feat(tools): add adopt_thesis tool for orphan position management"
```

---

### Task 4: Inject formation context into strategist

**Files:**
- Modify: `v2/ideation_claude.py`
- Modify: `tests/v2/test_ideation_claude.py`

The strategist's `run_strategist_loop()` builds a system prompt and initial message. When formation mode is active, we append the formation context to the system prompt.

- [ ] **Step 1: Write failing test for formation injection**

Add to `tests/v2/test_ideation_claude.py`. Uses the same `with patch(...)` style as existing tests in this file:

```python
class TestFormationInjection:
    def test_formation_context_appended_to_system_prompt(self):
        """Formation context should appear in strategist system prompt."""
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
             patch("v2.ideation_claude.build_formation_context",
                   return_value="## FORMATION MODE ACTIVE\nTest formation context"):
            mock_loop.return_value = mock_result

            run_strategist_loop(model="claude-opus-4-6", max_turns=1)

        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_prompt is None:
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "strategist" in arg.lower():
                    system_prompt = arg
                    break
        assert "FORMATION MODE ACTIVE" in system_prompt

    def test_no_formation_context_when_graduated(self):
        """Empty formation context should not alter the system prompt."""
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
             patch("v2.ideation_claude.build_formation_context", return_value=""):
            mock_loop.return_value = mock_result

            run_strategist_loop(model="claude-opus-4-6", max_turns=1)

        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_prompt is None:
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "strategist" in arg.lower():
                    system_prompt = arg
                    break
        assert "FORMATION MODE" not in system_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py::TestFormationInjection -v`
Expected: FAIL (import error or assertion failure)

- [ ] **Step 3: Modify `run_strategist_loop()` to inject formation context**

In `v2/ideation_claude.py`, add import at top:

```python
from .formation import build_formation_context
```

In `run_strategist_loop()`, after the line that builds `base_prompt` with attribution constraints (around line 279), add:

```python
    formation_context = build_formation_context()
    if formation_context:
        base_prompt = base_prompt + "\n\n" + formation_context
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py::TestFormationInjection -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add v2/ideation_claude.py tests/v2/test_ideation_claude.py
git commit -m "feat(strategist): inject formation mode context into system prompt"
```

---

### Task 5: Inject formation context into reflection stage

**Files:**
- Modify: `v2/strategy.py`
- Modify: `tests/v2/test_strategy.py`

The reflection agent (Stage 4) should also know about formation mode so it can bootstrap an initial identity and write memos that acknowledge the system's formation state rather than treating zero-activity as normal.

- [ ] **Step 1: Write failing test**

Add to `tests/v2/test_strategy.py`. Add `run_strategy_reflection` to the imports from `v2.strategy`, then add a new test class:

```python
from v2.strategy import run_strategy_reflection  # add to existing imports


class TestReflectionFormationInjection:
    def test_formation_context_in_reflection_system_prompt(self):
        """Formation context should appear in reflection system prompt."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100

        with patch("v2.strategy.get_claude_client", return_value=MagicMock()), \
             patch("v2.strategy.reset_session"), \
             patch("v2.strategy.run_agentic_loop") as mock_loop, \
             patch("v2.strategy.build_formation_context",
                   return_value="## FORMATION MODE ACTIVE\nBe exploratory"):
            mock_loop.return_value = mock_result

            run_strategy_reflection(model="claude-haiku-4-5-20251001", max_turns=3)

        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_prompt is None:
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "reflection" in arg.lower():
                    system_prompt = arg
                    break
        assert "FORMATION MODE ACTIVE" in system_prompt

    def test_no_formation_context_when_graduated(self):
        """Empty formation context should not alter the reflection prompt."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100

        with patch("v2.strategy.get_claude_client", return_value=MagicMock()), \
             patch("v2.strategy.reset_session"), \
             patch("v2.strategy.run_agentic_loop") as mock_loop, \
             patch("v2.strategy.build_formation_context", return_value=""):
            mock_loop.return_value = mock_result

            run_strategy_reflection(model="claude-haiku-4-5-20251001", max_turns=3)

        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_prompt is None:
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "reflection" in arg.lower():
                    system_prompt = arg
                    break
        assert "FORMATION MODE" not in system_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestReflectionFormationInjection -v`
Expected: FAIL (import error — `build_formation_context` not imported in strategy.py)

- [ ] **Step 3: Modify `run_strategy_reflection()` to inject formation context**

In `v2/strategy.py`, add import at top:

```python
from .formation import build_formation_context
```

In `run_strategy_reflection()`, before the `run_agentic_loop()` call (around line 436), add:

```python
    system_prompt = STRATEGY_REFLECTION_SYSTEM
    formation_context = build_formation_context()
    if formation_context:
        system_prompt = system_prompt + "\n\n" + formation_context
```

Then change the `run_agentic_loop()` call to use `system_prompt` instead of `STRATEGY_REFLECTION_SYSTEM`:

```python
    result = run_agentic_loop(
        client=client,
        model=model,
        system=system_prompt,
        ...
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestReflectionFormationInjection -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All existing tests pass, new tests pass

- [ ] **Step 6: Commit**

```bash
git add v2/strategy.py tests/v2/test_strategy.py
git commit -m "feat(reflection): inject formation mode context into strategy reflection"
```

---

### Task 6: Integration verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite with coverage**

Run: `python3 -m pytest tests/ --cov=v2 --tb=short -q`
Expected: All tests pass, `v2/formation.py` has >90% coverage

- [ ] **Step 2: Verify the formation module imports cleanly from session context**

Run: `python3 -c "from v2.formation import is_formation_mode, get_orphan_positions, build_formation_context; print('OK')"`
Expected: prints "OK"

- [ ] **Step 3: Verify the adopt_thesis tool is registered**

Run: `python3 -c "from v2.tools import TOOL_HANDLERS, TOOL_DEFINITIONS; assert 'adopt_thesis' in TOOL_HANDLERS; assert any(t['name'] == 'adopt_thesis' for t in TOOL_DEFINITIONS); print('OK')"`
Expected: prints "OK"

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: formation mode bootstrap — final cleanup"
```
