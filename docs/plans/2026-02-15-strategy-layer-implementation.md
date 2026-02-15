# Persistent Strategy Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a persistent strategy layer to the v2 pipeline that maintains an evolving trading identity, accumulated rules, and session-by-session reasoning memory.

**Architecture:** Three new DB tables (`strategy_state`, `strategy_rules`, `strategy_memos`), CRUD functions in `v2/database/trading_db.py`, a new `v2/strategy.py` module with its own agentic loop (Stage 4), and three read-only tools added to the existing strategist (Stage 2).

**Tech Stack:** PostgreSQL (raw SQL + psycopg2), Claude Opus (via existing `claude_client.py` agentic loop), Python dataclasses

---

### Task 1: Database Migration — Strategy Tables

**Files:**
- Create: `db/init/007_strategy.sql`

**Step 1: Write the migration SQL**

```sql
-- 007_strategy.sql: Persistent strategy layer tables

CREATE TABLE IF NOT EXISTS strategy_state (
    id SERIAL PRIMARY KEY,
    identity_text TEXT NOT NULL,
    risk_posture VARCHAR(20) NOT NULL DEFAULT 'moderate',
    sector_biases JSONB NOT NULL DEFAULT '{}',
    preferred_signals JSONB NOT NULL DEFAULT '[]',
    avoided_signals JSONB NOT NULL DEFAULT '[]',
    version INT NOT NULL DEFAULT 1,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_strategy_state_current ON strategy_state(is_current) WHERE is_current = TRUE;

CREATE TABLE IF NOT EXISTS strategy_rules (
    id SERIAL PRIMARY KEY,
    rule_text TEXT NOT NULL,
    category VARCHAR(64) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence DECIMAL NOT NULL DEFAULT 0.5,
    supporting_evidence TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    retired_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_strategy_rules_status ON strategy_rules(status);

CREATE TABLE IF NOT EXISTS strategy_memos (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL,
    memo_type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    strategy_state_id INT REFERENCES strategy_state(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_strategy_memos_date ON strategy_memos(session_date DESC);
```

**Step 2: Commit**

```bash
git add db/init/007_strategy.sql
git commit -m "feat: add strategy layer DB migration (007_strategy.sql)"
```

---

### Task 2: DB CRUD Functions — Strategy Tables

**Files:**
- Modify: `v2/database/trading_db.py` (append new section)
- Test: `tests/v2/test_db.py` (append new test classes)

**Step 1: Write failing tests for strategy CRUD**

Add to `tests/v2/test_db.py`:

```python
class TestStrategyState:
    def test_insert_strategy_state(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_strategy_state
        result = insert_strategy_state(
            identity_text="Momentum-focused trader",
            risk_posture="moderate",
            sector_biases={"tech": "overweight"},
            preferred_signals=["earnings"],
            avoided_signals=["legal"],
            version=1,
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO strategy_state" in sql

    def test_get_current_strategy_state(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1, "identity_text": "test", "is_current": True}
        from v2.database.trading_db import get_current_strategy_state
        result = get_current_strategy_state()
        assert result is not None
        sql = mock_cursor.execute.call_args[0][0]
        assert "is_current = TRUE" in sql

    def test_get_current_strategy_state_none(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = None
        from v2.database.trading_db import get_current_strategy_state
        result = get_current_strategy_state()
        assert result is None

    def test_clear_current_strategy_state(self, mock_db, mock_cursor):
        from v2.database.trading_db import clear_current_strategy_state
        clear_current_strategy_state()
        sql = mock_cursor.execute.call_args[0][0]
        assert "UPDATE strategy_state" in sql
        assert "is_current = FALSE" in sql


class TestStrategyRules:
    def test_insert_strategy_rule(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_strategy_rule
        result = insert_strategy_rule(
            rule_text="Fade legal news signals",
            category="news_signal:legal",
            direction="constraint",
            confidence=0.8,
            supporting_evidence="38% win rate over 12 trades",
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO strategy_rules" in sql

    def test_get_active_strategy_rules(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "rule_text": "test", "status": "active"}]
        from v2.database.trading_db import get_active_strategy_rules
        result = get_active_strategy_rules()
        assert len(result) == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'active'" in sql

    def test_retire_strategy_rule(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 1
        from v2.database.trading_db import retire_strategy_rule
        result = retire_strategy_rule(rule_id=1)
        assert result is True
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'retired'" in sql
        assert "retired_at" in sql

    def test_retire_strategy_rule_not_found(self, mock_db, mock_cursor):
        mock_cursor.rowcount = 0
        from v2.database.trading_db import retire_strategy_rule
        result = retire_strategy_rule(rule_id=999)
        assert result is False


class TestStrategyMemos:
    def test_insert_strategy_memo(self, mock_db, mock_cursor):
        mock_cursor.fetchone.return_value = {"id": 1}
        from v2.database.trading_db import insert_strategy_memo
        result = insert_strategy_memo(
            session_date="2026-02-15",
            memo_type="reflection",
            content="Today we learned...",
            strategy_state_id=1,
        )
        assert result == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO strategy_memos" in sql

    def test_get_recent_strategy_memos(self, mock_db, mock_cursor):
        mock_cursor.fetchall.return_value = [{"id": 1, "content": "test"}]
        from v2.database.trading_db import get_recent_strategy_memos
        result = get_recent_strategy_memos(n=5)
        assert len(result) == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "LIMIT" in sql
        assert "ORDER BY" in sql
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_db.py::TestStrategyState tests/v2/test_db.py::TestStrategyRules tests/v2/test_db.py::TestStrategyMemos -v`
Expected: FAIL — `ImportError: cannot import name 'insert_strategy_state'`

**Step 3: Write the CRUD functions**

Append to `v2/database/trading_db.py`:

```python
# --- Strategy State ---

def insert_strategy_state(identity_text, risk_posture, sector_biases, preferred_signals, avoided_signals, version) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy_state (identity_text, risk_posture, sector_biases, preferred_signals, avoided_signals, version)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (identity_text, risk_posture, Json(sector_biases), Json(preferred_signals), Json(avoided_signals), version))
        return cur.fetchone()["id"]


def get_current_strategy_state() -> dict | None:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM strategy_state WHERE is_current = TRUE LIMIT 1")
        return cur.fetchone()


def clear_current_strategy_state():
    with get_cursor() as cur:
        cur.execute("UPDATE strategy_state SET is_current = FALSE WHERE is_current = TRUE")


# --- Strategy Rules ---

def insert_strategy_rule(rule_text, category, direction, confidence, supporting_evidence=None) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy_rules (rule_text, category, direction, confidence, supporting_evidence)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (rule_text, category, direction, confidence, supporting_evidence))
        return cur.fetchone()["id"]


def get_active_strategy_rules() -> list:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM strategy_rules WHERE status = 'active' ORDER BY created_at DESC")
        return cur.fetchall()


def retire_strategy_rule(rule_id) -> bool:
    with get_cursor() as cur:
        cur.execute("""
            UPDATE strategy_rules SET status = 'retired', retired_at = NOW()
            WHERE id = %s AND status = 'active'
        """, (rule_id,))
        return cur.rowcount > 0


# --- Strategy Memos ---

def insert_strategy_memo(session_date, memo_type, content, strategy_state_id=None) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy_memos (session_date, memo_type, content, strategy_state_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (session_date, memo_type, content, strategy_state_id))
        return cur.fetchone()["id"]


def get_recent_strategy_memos(n=5) -> list:
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM strategy_memos
            ORDER BY created_at DESC
            LIMIT %s
        """, (n,))
        return cur.fetchall()
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_db.py::TestStrategyState tests/v2/test_db.py::TestStrategyRules tests/v2/test_db.py::TestStrategyMemos -v`
Expected: PASS

**Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "feat: add strategy layer CRUD functions and tests"
```

---

### Task 3: Conftest Factory Functions

**Files:**
- Modify: `tests/v2/conftest.py` (append factories)

**Step 1: Add factory functions**

```python
def make_strategy_state_row(**kwargs):
    """Create a strategy_state dict like what DB returns."""
    defaults = {
        "id": 1,
        "identity_text": "Momentum-focused trader favoring earnings signals",
        "risk_posture": "moderate",
        "sector_biases": {"tech": "overweight"},
        "preferred_signals": ["earnings", "fed"],
        "avoided_signals": ["legal"],
        "version": 1,
        "is_current": True,
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


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
    }
    defaults.update(kwargs)
    return defaults


def make_strategy_memo_row(**kwargs):
    """Create a strategy_memos dict like what DB returns."""
    defaults = {
        "id": 1,
        "session_date": date.today(),
        "memo_type": "reflection",
        "content": "Today's session showed strong performance in tech earnings plays.",
        "strategy_state_id": 1,
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults
```

**Step 2: Commit**

```bash
git add tests/v2/conftest.py
git commit -m "feat: add strategy layer factory functions to v2 conftest"
```

---

### Task 4: Read-Only Strategy Tools for Strategist (Stage 2)

**Files:**
- Modify: `v2/tools.py` (add 3 read-only tool handlers + definitions)
- Test: `tests/v2/test_tools.py` (add test classes)

**Step 1: Write failing tests**

Add to `tests/v2/test_tools.py`:

```python
from tests.v2.conftest import make_strategy_state_row, make_strategy_rule_row, make_strategy_memo_row

class TestGetStrategyIdentity:
    @patch("v2.tools.get_current_strategy_state")
    def test_returns_formatted_identity(self, mock_get):
        from v2.tools import tool_get_strategy_identity
        mock_get.return_value = make_strategy_state_row()
        result = tool_get_strategy_identity()
        assert "Momentum-focused" in result
        assert "moderate" in result

    @patch("v2.tools.get_current_strategy_state")
    def test_returns_null_message_when_empty(self, mock_get):
        from v2.tools import tool_get_strategy_identity
        mock_get.return_value = None
        result = tool_get_strategy_identity()
        assert "No strategy identity" in result or "no identity" in result.lower()


class TestGetStrategyRules:
    @patch("v2.tools.get_active_strategy_rules")
    def test_returns_formatted_rules(self, mock_get):
        from v2.tools import tool_get_strategy_rules
        mock_get.return_value = [make_strategy_rule_row()]
        result = tool_get_strategy_rules()
        assert "Fade legal" in result
        assert "constraint" in result

    @patch("v2.tools.get_active_strategy_rules")
    def test_returns_empty_message(self, mock_get):
        from v2.tools import tool_get_strategy_rules
        mock_get.return_value = []
        result = tool_get_strategy_rules()
        assert "No active" in result or "no rules" in result.lower()


class TestGetStrategyHistory:
    @patch("v2.tools.get_recent_strategy_memos")
    def test_returns_formatted_memos(self, mock_get):
        from v2.tools import tool_get_strategy_history
        mock_get.return_value = [make_strategy_memo_row()]
        result = tool_get_strategy_history(n=5)
        assert "reflection" in result or "session" in result.lower()

    @patch("v2.tools.get_recent_strategy_memos")
    def test_returns_empty_message(self, mock_get):
        from v2.tools import tool_get_strategy_history
        mock_get.return_value = []
        result = tool_get_strategy_history(n=5)
        assert "No strategy" in result or "no memos" in result.lower()


class TestStrategyToolDefinitions:
    def test_strategy_tools_in_definitions(self):
        from v2.tools import TOOL_DEFINITIONS
        names = [d.get("name") for d in TOOL_DEFINITIONS]
        assert "get_strategy_identity" in names
        assert "get_strategy_rules" in names
        assert "get_strategy_history" in names

    def test_strategy_tools_in_handlers(self):
        from v2.tools import TOOL_HANDLERS
        assert "get_strategy_identity" in TOOL_HANDLERS
        assert "get_strategy_rules" in TOOL_HANDLERS
        assert "get_strategy_history" in TOOL_HANDLERS
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_tools.py::TestGetStrategyIdentity tests/v2/test_tools.py::TestGetStrategyRules tests/v2/test_tools.py::TestGetStrategyHistory tests/v2/test_tools.py::TestStrategyToolDefinitions -v`
Expected: FAIL — `ImportError`

**Step 3: Add tool handlers and definitions to `v2/tools.py`**

Add imports at top of `v2/tools.py`:

```python
from .database.trading_db import (
    # ... existing imports ...
    get_current_strategy_state,
    get_active_strategy_rules,
    get_recent_strategy_memos,
)
```

Add tool handlers:

```python
def tool_get_strategy_identity() -> str:
    """Get the system's current strategy identity."""
    logger.info("Getting strategy identity")
    state = get_current_strategy_state()
    if state is None:
        return "No strategy identity established yet. This is the first session."

    lines = [
        f"Strategy Identity (v{state['version']}, updated {state['created_at'].strftime('%Y-%m-%d')}):",
        f"  Identity: {state['identity_text']}",
        f"  Risk Posture: {state['risk_posture']}",
        f"  Sector Biases: {state['sector_biases']}",
        f"  Preferred Signals: {state['preferred_signals']}",
        f"  Avoided Signals: {state['avoided_signals']}",
    ]
    return "\n".join(lines)


def tool_get_strategy_rules() -> str:
    """Get all active strategy rules."""
    logger.info("Getting strategy rules")
    rules = get_active_strategy_rules()
    if not rules:
        return "No active strategy rules yet."

    lines = []
    for r in rules:
        lines.append(
            f"Rule {r['id']} ({r['direction']}, confidence: {r['confidence']}): "
            f"{r['rule_text']}"
        )
        if r.get("supporting_evidence"):
            lines.append(f"  Evidence: {r['supporting_evidence']}")
        lines.append(f"  Category: {r['category']}")
        lines.append("")
    return "\n".join(lines)


def tool_get_strategy_history(n: int = 5) -> str:
    """Get recent strategy memos."""
    logger.info(f"Getting strategy history (last {n})")
    memos = get_recent_strategy_memos(n=n)
    if not memos:
        return "No strategy memos yet. This is the first session."

    lines = []
    for m in memos:
        lines.append(f"[{m['session_date']}] ({m['memo_type']}):")
        lines.append(f"  {m['content']}")
        lines.append("")
    return "\n".join(lines)
```

Add to `TOOL_DEFINITIONS` list (before the closing bracket):

```python
    {
        "name": "get_strategy_identity",
        "description": (
            "Get the system's evolving strategy identity — who it is as a trader, "
            "risk posture, sector biases, and preferred/avoided signal types. "
            "Returns null if no identity has been established yet (first session)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_rules",
        "description": (
            "Get all active strategy rules — accumulated lessons from past performance. "
            "Rules are either constraints (don't do X) or preferences (favor X)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_history",
        "description": (
            "Get recent strategy reflection memos — session-by-session reasoning "
            "about what the system learned and how its strategy evolved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of recent memos to retrieve (default: 5)",
                },
            },
            "required": [],
        },
    },
```

Add to `TOOL_HANDLERS` dict:

```python
    "get_strategy_identity": tool_get_strategy_identity,
    "get_strategy_rules": tool_get_strategy_rules,
    "get_strategy_history": tool_get_strategy_history,
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_tools.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add v2/tools.py tests/v2/test_tools.py
git commit -m "feat: add read-only strategy tools to strategist (Stage 2)"
```

---

### Task 5: Strategist System Prompt Update

**Files:**
- Modify: `v2/ideation_claude.py` (add line to `_STRATEGIST_TEMPLATE`)
- Test: `tests/v2/test_ideation_claude.py` (add assertion)

**Step 1: Write failing test**

Add to `tests/v2/test_ideation_claude.py`:

```python
class TestStrategistPromptIncludesStrategy:
    def test_strategist_prompt_mentions_strategy_tools(self):
        from v2.ideation_claude import CLAUDE_STRATEGIST_SYSTEM
        assert "strategy identity" in CLAUDE_STRATEGIST_SYSTEM.lower() or \
               "strategy" in CLAUDE_STRATEGIST_SYSTEM.lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py::TestStrategistPromptIncludesStrategy -v`
Expected: FAIL (or PASS if "strategy" already appears — check first)

**Step 3: Add strategy line to `_STRATEGIST_TEMPLATE`**

In `v2/ideation_claude.py`, add to the `## Tool Usage` section of `_STRATEGIST_TEMPLATE`:

```
- Use `get_strategy_identity` to understand the system's evolving trading identity
- Use `get_strategy_rules` to see accumulated trading rules from past performance
- Use `get_strategy_history` to review recent strategy reflection memos
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add v2/ideation_claude.py tests/v2/test_ideation_claude.py
git commit -m "feat: add strategy tool guidance to strategist system prompt"
```

---

### Task 6: Strategy Reflection Module (`v2/strategy.py`)

**Files:**
- Create: `v2/strategy.py`
- Create: `tests/v2/test_strategy.py`

**Step 1: Write failing tests for the strategy module**

Create `tests/v2/test_strategy.py`:

```python
"""Tests for v2/strategy.py — strategy reflection stage (Stage 4)."""

import json
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, MagicMock, call

import pytest

from tests.v2.conftest import (
    make_strategy_state_row,
    make_strategy_rule_row,
    make_strategy_memo_row,
    make_decision_row,
)


class TestStrategyReflectionResult:
    def test_dataclass_fields(self):
        from v2.strategy import StrategyReflectionResult
        result = StrategyReflectionResult(
            rules_proposed=2,
            rules_retired=1,
            identity_updated=True,
            memo_written=True,
            input_tokens=1000,
            output_tokens=500,
            turns_used=3,
        )
        assert result.rules_proposed == 2
        assert result.rules_retired == 1
        assert result.identity_updated is True
        assert result.memo_written is True


class TestStrategyToolHandlers:
    """Test the strategy-specific tool handlers (write tools)."""

    @patch("v2.strategy.get_current_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.insert_strategy_state")
    def test_tool_update_strategy_identity(self, mock_insert, mock_clear, mock_get):
        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = make_strategy_state_row(version=2)
        mock_insert.return_value = 3

        result = tool_update_strategy_identity(
            identity_text="Contrarian trader",
            risk_posture="aggressive",
            sector_biases={"energy": "overweight"},
            preferred_signals=["macro"],
            avoided_signals=["legal"],
        )
        mock_clear.assert_called_once()
        mock_insert.assert_called_once()
        assert "3" in result or "version 3" in result.lower()

    @patch("v2.strategy.get_current_strategy_state")
    @patch("v2.strategy.clear_current_strategy_state")
    @patch("v2.strategy.insert_strategy_state")
    def test_tool_update_strategy_identity_bootstrap(self, mock_insert, mock_clear, mock_get):
        """First-ever identity creation (no existing state)."""
        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = None
        mock_insert.return_value = 1

        result = tool_update_strategy_identity(
            identity_text="New trader",
            risk_posture="conservative",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )
        mock_insert.assert_called_once()
        assert "1" in result

    @patch("v2.strategy.insert_strategy_rule")
    def test_tool_propose_rule(self, mock_insert):
        from v2.strategy import tool_propose_rule
        mock_insert.return_value = 5

        result = tool_propose_rule(
            rule_text="Favor earnings signals",
            category="news_signal:earnings",
            direction="preference",
            confidence=0.7,
            supporting_evidence="62% win rate",
        )
        mock_insert.assert_called_once()
        assert "5" in result

    @patch("v2.strategy.retire_strategy_rule")
    def test_tool_retire_rule(self, mock_retire):
        from v2.strategy import tool_retire_rule
        mock_retire.return_value = True
        result = tool_retire_rule(rule_id=1, reason="No longer predictive")
        assert "Retired" in result or "retired" in result

    @patch("v2.strategy.retire_strategy_rule")
    def test_tool_retire_rule_not_found(self, mock_retire):
        from v2.strategy import tool_retire_rule
        mock_retire.return_value = False
        result = tool_retire_rule(rule_id=999, reason="test")
        assert "not found" in result.lower() or "Error" in result

    @patch("v2.strategy.insert_strategy_memo")
    @patch("v2.strategy.get_current_strategy_state")
    def test_tool_write_strategy_memo(self, mock_state, mock_insert):
        from v2.strategy import tool_write_strategy_memo
        mock_state.return_value = make_strategy_state_row()
        mock_insert.return_value = 1

        result = tool_write_strategy_memo(
            memo_type="reflection",
            content="Session showed strong momentum plays.",
        )
        mock_insert.assert_called_once()
        assert "Memo" in result or "memo" in result


class TestStrategyToolDefinitions:
    def test_all_write_tools_defined(self):
        from v2.strategy import STRATEGY_TOOL_DEFINITIONS
        names = [d.get("name") for d in STRATEGY_TOOL_DEFINITIONS]
        assert "update_strategy_identity" in names
        assert "propose_rule" in names
        assert "retire_rule" in names
        assert "write_strategy_memo" in names

    def test_all_read_tools_defined(self):
        from v2.strategy import STRATEGY_TOOL_DEFINITIONS
        names = [d.get("name") for d in STRATEGY_TOOL_DEFINITIONS]
        assert "get_strategy_identity" in names
        assert "get_strategy_rules" in names
        assert "get_strategy_history" in names
        assert "get_session_summary" in names

    def test_handlers_match_definitions(self):
        from v2.strategy import STRATEGY_TOOL_DEFINITIONS, STRATEGY_TOOL_HANDLERS
        client_tool_names = [d["name"] for d in STRATEGY_TOOL_DEFINITIONS if "type" not in d]
        for name in client_tool_names:
            assert name in STRATEGY_TOOL_HANDLERS, f"Missing handler for {name}"


class TestRunStrategyReflection:
    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    def test_returns_reflection_result(self, mock_loop, mock_client):
        from v2.strategy import run_strategy_reflection, StrategyReflectionResult
        from v2.claude_client import AgenticLoopResult

        mock_loop.return_value = AgenticLoopResult(
            messages=[
                {"role": "user", "content": "Begin reflection"},
                {"role": "assistant", "content": [MagicMock(text="Done reflecting", type="text")]},
            ],
            turns_used=2,
            stop_reason="end_turn",
            input_tokens=1000,
            output_tokens=500,
        )

        result = run_strategy_reflection()
        assert isinstance(result, StrategyReflectionResult)
        assert result.turns_used == 2
        assert result.input_tokens == 1000

    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    def test_counts_proposed_rules(self, mock_loop, mock_client):
        from v2.strategy import run_strategy_reflection

        mock_loop.return_value = MagicMock(
            messages=[
                {"role": "user", "content": [
                    {"type": "tool_result", "content": "Created rule ID 1"},
                    {"type": "tool_result", "content": "Created rule ID 2"},
                ]},
                {"role": "assistant", "content": [MagicMock(text="Done", type="text")]},
            ],
            turns_used=3,
            stop_reason="end_turn",
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        result = run_strategy_reflection()
        assert result.rules_proposed == 2

    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    def test_counts_retired_rules(self, mock_loop, mock_client):
        from v2.strategy import run_strategy_reflection

        mock_loop.return_value = MagicMock(
            messages=[
                {"role": "user", "content": [
                    {"type": "tool_result", "content": "Retired rule ID 1"},
                ]},
                {"role": "assistant", "content": [MagicMock(text="Done", type="text")]},
            ],
            turns_used=2,
            stop_reason="end_turn",
            input_tokens=500,
            output_tokens=200,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

        result = run_strategy_reflection()
        assert result.rules_retired == 1
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'v2.strategy'`

**Step 3: Create `v2/strategy.py`**

```python
"""Strategy reflection stage (Stage 4).

Runs after the trading session to reflect on performance,
update the system's evolving trading identity, manage accumulated
rules, and write session memos.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .claude_client import get_claude_client, run_agentic_loop, extract_final_text
from .tools import tool_get_strategy_identity, tool_get_strategy_rules, tool_get_strategy_history
from .attribution import get_attribution_summary
from .database.trading_db import (
    get_current_strategy_state,
    clear_current_strategy_state,
    insert_strategy_state,
    insert_strategy_rule,
    retire_strategy_rule,
    insert_strategy_memo,
    get_recent_strategy_memos,
    get_active_strategy_rules,
    get_recent_decisions,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyReflectionResult:
    rules_proposed: int
    rules_retired: int
    identity_updated: bool
    memo_written: bool
    input_tokens: int
    output_tokens: int
    turns_used: int


STRATEGY_REFLECTION_SYSTEM = """You are the strategy reflection agent for an autonomous trading system. You have just completed a trading session. Your job is to review what happened, update the system's evolving trading identity, and manage its accumulated rules.

You are NOT making trades. You are reflecting on performance and shaping who this system is as a trader.

## Your Process

1. **Review Current State**: Get the current strategy identity, active rules, and recent memos to understand where the system is.

2. **Analyze Session**: Get the session summary to see what happened today — decisions made, outcomes of past decisions, and signal attribution scores.

3. **Update Rules**: Based on attribution data and decision outcomes:
   - Propose new rules when patterns emerge (e.g., a signal type consistently wins or loses)
   - Retire rules that are no longer supported by data
   - Rules should be specific and evidence-based

4. **Update Identity**: If the system's behavior or performance suggests a shift in trading style, update the identity. The identity should reflect what the system IS, not what it aspires to be.

5. **Write Memo**: Always write a reflection memo summarizing what you observed and any changes you made. This is the system's memory.

## Critical Rules

1. **Evidence-based**: Every rule must cite specific data (win rates, sample sizes, outcome averages)
2. **First session**: If no identity exists yet, bootstrap one from attribution data and today's results
3. **Concise memos**: Memos should be 2-4 paragraphs, focused on actionable observations
4. **Always write a memo**: Even if nothing changed, document why
5. **Don't over-rotate**: A single bad session doesn't warrant major strategy changes. Look for patterns across multiple sessions."""


# --- Write Tool Handlers ---

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


def tool_propose_rule(
    rule_text: str,
    category: str,
    direction: str,
    confidence: float,
    supporting_evidence: str,
) -> str:
    """Propose a new strategy rule."""
    logger.info(f"Proposing rule: {rule_text[:50]}...")
    rule_id = insert_strategy_rule(
        rule_text=rule_text,
        category=category,
        direction=direction,
        confidence=confidence,
        supporting_evidence=supporting_evidence,
    )
    return f"Created rule ID {rule_id}: {rule_text}"


def tool_retire_rule(rule_id: int, reason: str) -> str:
    """Retire a strategy rule."""
    logger.info(f"Retiring rule {rule_id}: {reason}")
    success = retire_strategy_rule(rule_id=rule_id)
    if success:
        return f"Retired rule ID {rule_id}. Reason: {reason}"
    return f"Error: Rule ID {rule_id} not found or already retired"


def tool_write_strategy_memo(memo_type: str, content: str) -> str:
    """Write a strategy reflection memo."""
    logger.info(f"Writing strategy memo ({memo_type})")
    current = get_current_strategy_state()
    state_id = current["id"] if current else None
    memo_id = insert_strategy_memo(
        session_date=date.today(),
        memo_type=memo_type,
        content=content,
        strategy_state_id=state_id,
    )
    return f"Memo written (ID: {memo_id})"


def tool_get_session_summary(days: int = 30) -> str:
    """Get today's session summary — recent decisions, outcomes, attribution."""
    logger.info("Getting session summary")
    lines = []

    decisions = get_recent_decisions(days=days)
    if decisions:
        lines.append(f"Recent decisions ({len(decisions)}):")
        for d in decisions[:10]:
            outcome_7d = f"{d['outcome_7d']:+.2f}%" if d.get("outcome_7d") is not None else "pending"
            outcome_30d = f"{d['outcome_30d']:+.2f}%" if d.get("outcome_30d") is not None else "pending"
            lines.append(
                f"  [{d['date']}] {d['action'].upper()} {d['ticker']}: "
                f"7d={outcome_7d}, 30d={outcome_30d}"
            )
    else:
        lines.append("No recent decisions.")

    lines.append("")
    lines.append("Signal Attribution:")
    lines.append(get_attribution_summary())

    return "\n".join(lines)


# --- Tool Definitions ---

STRATEGY_TOOL_DEFINITIONS = [
    {
        "name": "get_strategy_identity",
        "description": "Get the system's current strategy identity.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_rules",
        "description": "Get all active strategy rules.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_history",
        "description": "Get recent strategy reflection memos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of memos (default: 5)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_session_summary",
        "description": (
            "Get a summary of recent trading activity — decisions, outcomes, "
            "and signal attribution scores. Use this to understand what happened."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Look back period (default: 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "update_strategy_identity",
        "description": (
            "Update the system's trading identity. Creates a new versioned identity. "
            "Use this to reflect changes in trading style, risk posture, or signal preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identity_text": {"type": "string", "description": "Who is this system as a trader?"},
                "risk_posture": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive"],
                },
                "sector_biases": {
                    "type": "object",
                    "description": "Sector biases, e.g. {\"tech\": \"overweight\", \"energy\": \"avoid\"}",
                },
                "preferred_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Signal types to favor",
                },
                "avoided_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Signal types to avoid",
                },
            },
            "required": ["identity_text", "risk_posture", "sector_biases", "preferred_signals", "avoided_signals"],
        },
    },
    {
        "name": "propose_rule",
        "description": (
            "Propose a new strategy rule based on observed patterns. "
            "Rules are either constraints (avoid X) or preferences (favor X)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_text": {"type": "string", "description": "Human-readable rule"},
                "category": {"type": "string", "description": "Domain, e.g. news_signal:legal, position_sizing"},
                "direction": {"type": "string", "enum": ["constraint", "preference"]},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "supporting_evidence": {"type": "string", "description": "Data backing this rule"},
            },
            "required": ["rule_text", "category", "direction", "confidence", "supporting_evidence"],
        },
    },
    {
        "name": "retire_rule",
        "description": "Retire a strategy rule that is no longer supported by data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "integer", "description": "ID of rule to retire"},
                "reason": {"type": "string", "description": "Why this rule is being retired"},
            },
            "required": ["rule_id", "reason"],
        },
    },
    {
        "name": "write_strategy_memo",
        "description": (
            "Write a strategy reflection memo. Always write one at the end of reflection. "
            "Types: reflection (general), rule_change (when rules changed), identity_update (when identity changed)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memo_type": {
                    "type": "string",
                    "enum": ["reflection", "rule_change", "identity_update"],
                },
                "content": {"type": "string", "description": "The memo content (2-4 paragraphs)"},
            },
            "required": ["memo_type", "content"],
        },
    },
]

STRATEGY_TOOL_HANDLERS = {
    "get_strategy_identity": tool_get_strategy_identity,
    "get_strategy_rules": tool_get_strategy_rules,
    "get_strategy_history": tool_get_strategy_history,
    "get_session_summary": tool_get_session_summary,
    "update_strategy_identity": tool_update_strategy_identity,
    "propose_rule": tool_propose_rule,
    "retire_rule": tool_retire_rule,
    "write_strategy_memo": tool_write_strategy_memo,
}


def _count_actions(messages: list[dict]) -> tuple[int, int, bool, bool]:
    """Count strategy actions from tool results."""
    proposed = 0
    retired = 0
    identity_updated = False
    memo_written = False

    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        result_text = item.get("content", "")
                        if isinstance(result_text, str):
                            if "Created rule ID" in result_text:
                                proposed += 1
                            elif "Retired rule ID" in result_text:
                                retired += 1
                            elif "identity updated" in result_text.lower():
                                identity_updated = True
                            elif "Memo written" in result_text:
                                memo_written = True

    return proposed, retired, identity_updated, memo_written


def run_strategy_reflection(
    model: str = "claude-opus-4-6",
    max_turns: int = 10,
) -> StrategyReflectionResult:
    """Run the strategy reflection stage (Stage 4)."""
    logger.info("Starting strategy reflection (model=%s, max_turns=%d)", model, max_turns)

    client = get_claude_client()

    result = run_agentic_loop(
        client=client,
        model=model,
        system=STRATEGY_REFLECTION_SYSTEM,
        initial_message=(
            "Begin your strategy reflection. Start by:\n"
            "1. Getting the current strategy identity and rules\n"
            "2. Getting the session summary (recent decisions and attribution)\n"
            "3. Getting recent strategy memos for context\n"
            "4. Analyzing what happened and making any necessary updates\n"
            "5. Writing a reflection memo\n"
        ),
        tools=STRATEGY_TOOL_DEFINITIONS,
        tool_handlers=STRATEGY_TOOL_HANDLERS,
        max_turns=max_turns,
    )

    proposed, retired, identity_updated, memo_written = _count_actions(result.messages)

    logger.info(
        "Strategy reflection complete: %d rules proposed, %d retired, "
        "identity_updated=%s, memo_written=%s",
        proposed, retired, identity_updated, memo_written,
    )

    return StrategyReflectionResult(
        rules_proposed=proposed,
        rules_retired=retired,
        identity_updated=identity_updated,
        memo_written=memo_written,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        turns_used=result.turns_used,
    )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_strategy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add v2/strategy.py tests/v2/test_strategy.py
git commit -m "feat: add strategy reflection module (Stage 4)"
```

---

### Task 7: Session Integration — Wire Stage 4 into `run_session()`

**Files:**
- Modify: `v2/session.py`
- Test: `tests/v2/test_session.py`

**Step 1: Write failing tests**

Add to `tests/v2/test_session.py`:

```python
class TestStage4StrategyReflection:
    def test_stage_4_runs_after_trading(self):
        """Strategy reflection should run after trading session."""
        call_order = []

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session") as mock_trade, \
             patch("v2.session.run_strategy_reflection") as mock_reflect:

            mock_trade.side_effect = lambda **kw: call_order.append("trader")
            mock_reflect.side_effect = lambda **kw: call_order.append("reflection")

            run_session(dry_run=True)

        assert call_order.index("trader") < call_order.index("reflection")

    def test_stage_4_result_captured(self):
        """Strategy reflection result should be in SessionResult."""
        from v2.strategy import StrategyReflectionResult

        mock_reflection_result = StrategyReflectionResult(
            rules_proposed=1, rules_retired=0, identity_updated=True,
            memo_written=True, input_tokens=500, output_tokens=200, turns_used=3,
        )

        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection", return_value=mock_reflection_result):

            result = run_session(dry_run=True)

        assert result.strategy_result is not None
        assert result.strategy_result.rules_proposed == 1

    def test_stage_4_failure_does_not_block(self):
        """Strategy reflection failure should be captured but not affect session."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection", side_effect=Exception("Opus down")):

            result = run_session(dry_run=True)

        assert result.strategy_error == "Opus down"
        assert result.strategy_result is None

    def test_stage_4_error_in_has_errors(self):
        """Strategy error should be included in has_errors check."""
        result = SessionResult(strategy_error="test")
        assert result.has_errors is True

    def test_skip_strategy_flag(self):
        """Strategy reflection should be skippable."""
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection") as mock_reflect:

            result = run_session(dry_run=True, skip_strategy=True)

        mock_reflect.assert_not_called()
        assert result.skipped_strategy is True
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_session.py::TestStage4StrategyReflection -v`
Expected: FAIL

**Step 3: Modify `v2/session.py`**

Add import:

```python
from .strategy import StrategyReflectionResult, run_strategy_reflection
```

Update `SessionResult` dataclass:

```python
@dataclass
class SessionResult:
    pipeline_result: Optional[PipelineStats] = None
    strategist_result: Optional[ClaudeIdeationResult] = None
    trading_result: Optional[TradingSessionResult] = None
    strategy_result: Optional[StrategyReflectionResult] = None  # NEW

    learning_error: Optional[str] = None
    pipeline_error: Optional[str] = None
    strategist_error: Optional[str] = None
    trading_error: Optional[str] = None
    strategy_error: Optional[str] = None  # NEW

    skipped_pipeline: bool = False
    skipped_ideation: bool = False
    skipped_strategy: bool = False  # NEW
    duration_seconds: float = 0.0

    @property
    def has_errors(self) -> bool:
        return any([self.learning_error, self.pipeline_error,
                    self.strategist_error, self.trading_error,
                    self.strategy_error])  # NEW
```

Update `run_session()` signature to add `skip_strategy: bool = False`:

```python
def run_session(
    dry_run: bool = False,
    model: str = "claude-opus-4-6",
    executor_model: str = DEFAULT_EXECUTOR_MODEL,
    max_turns: int = 25,
    skip_pipeline: bool = False,
    skip_ideation: bool = False,
    skip_strategy: bool = False,  # NEW
    pipeline_hours: int = 24,
    pipeline_limit: int = 300,
) -> SessionResult:
```

Add Stage 4 block after Stage 3, before `result.duration_seconds`:

```python
    # Stage 4: Strategy reflection
    if skip_strategy:
        logger.info("[Stage 4] Strategy reflection — SKIPPED")
        result.skipped_strategy = True
    else:
        logger.info("[Stage 4] Running strategy reflection")
        try:
            result.strategy_result = run_strategy_reflection(model=model, max_turns=10)
        except Exception as e:
            result.strategy_error = str(e)
            logger.error("Strategy reflection failed: %s", e)
```

Update the error logging section to include `strategy_error`:

```python
    for field_name in ["learning_error", "pipeline_error", "strategist_error", "trading_error", "strategy_error"]:
```

Update `main()` to add `--skip-strategy` arg:

```python
    parser.add_argument("--skip-strategy", action="store_true")
```

And pass it through:

```python
    result = run_session(
        ...,
        skip_strategy=args.skip_strategy,
    )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: PASS

**Step 5: Run full v2 test suite**

Run: `python3 -m pytest tests/v2/ -v`
Expected: PASS (no regressions)

**Step 6: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "feat: wire strategy reflection (Stage 4) into session orchestrator"
```

---

### Task 8: Full Test Suite Verification

**Files:**
- No changes — verification only

**Step 1: Run complete test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All tests pass, no regressions from v1 tests

**Step 2: Run v2 tests with coverage**

Run: `python3 -m pytest tests/v2/ --cov=v2 --cov-report=term-missing`
Expected: New strategy code covered, overall coverage maintained

**Step 3: Commit (if any fixes needed)**

Only if test fixes were required.
