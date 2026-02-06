# System Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the trading system into a Claude strategist (post-market) + Ollama executor (pre-market) architecture with a signal attribution learning loop.

**Architecture:** Claude runs after market close to backfill outcomes, compute signal attribution, manage theses, and write a daily playbook. Ollama runs before market open to execute that playbook. A new `decision_signals` join table links every decision to the signals/theses that motivated it, enabling quantitative learning.

**Tech Stack:** Python 3, psycopg2, Anthropic SDK, Ollama, Flask, PostgreSQL

---

## Task 1: Database Schema — New Tables

Add three new tables (`playbooks`, `decision_signals`, `signal_attribution`) and a migration to drop `documents` + `strategy`.

**Files:**
- Create: `db/init/005_redesign.sql`
- Test: `tests/test_db_redesign.py`

**Step 1: Write the migration SQL**

Create `db/init/005_redesign.sql`:

```sql
-- Redesign migration: playbooks, decision_signals, signal_attribution

-- New: Daily trading plan from Claude strategist to Ollama executor
CREATE TABLE IF NOT EXISTS playbooks (
    id              SERIAL PRIMARY KEY,
    date            DATE UNIQUE,
    market_outlook  TEXT,
    priority_actions JSONB,
    watch_list      TEXT[],
    risk_notes      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_playbooks_date ON playbooks(date);

-- New: Links decisions to the signals/theses that motivated them
CREATE TABLE IF NOT EXISTS decision_signals (
    decision_id  INT REFERENCES decisions(id),
    signal_type  TEXT NOT NULL,        -- 'news_signal', 'macro_signal', 'thesis'
    signal_id    INT NOT NULL,
    PRIMARY KEY (decision_id, signal_type, signal_id)
);

CREATE INDEX idx_decision_signals_signal ON decision_signals(signal_type, signal_id);

-- New: Precomputed scores showing which signal types are predictive
CREATE TABLE IF NOT EXISTS signal_attribution (
    id              SERIAL PRIMARY KEY,
    category        TEXT UNIQUE NOT NULL,
    sample_size     INT,
    avg_outcome_7d  NUMERIC(8,4),
    avg_outcome_30d NUMERIC(8,4),
    win_rate_7d     NUMERIC(5,4),
    win_rate_30d    NUMERIC(5,4),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Drop tables that are no longer needed
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS strategy CASCADE;
```

**Step 2: Write failing tests for new DB functions**

Create `tests/test_db_redesign.py` with tests for the new DB helper functions we'll add in Step 3:

```python
"""Tests for new database functions: playbooks, decision_signals, signal_attribution."""

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest

from trading.db import (
    upsert_playbook,
    get_playbook,
    insert_decision_signal,
    insert_decision_signals_batch,
    get_decision_signals,
    upsert_signal_attribution,
    get_signal_attribution,
)


@pytest.fixture
def mock_cursor():
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 1}
    cursor.fetchall.return_value = []
    cursor.rowcount = 1
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("trading.db.get_cursor", _get_cursor), \
         patch("trading.db.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor


class TestPlaybooks:
    def test_upsert_playbook(self, mock_db):
        result = upsert_playbook(
            playbook_date=date.today(),
            market_outlook="Bullish tech",
            priority_actions=[{"ticker": "NVDA", "action": "buy"}],
            watch_list=["AAPL", "MSFT"],
            risk_notes="Watch Fed meeting",
        )
        assert result == 1
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO playbooks" in sql
        assert "ON CONFLICT (date)" in sql

    def test_get_playbook_returns_row(self, mock_db):
        mock_db.fetchone.return_value = {
            "id": 1,
            "date": date.today(),
            "market_outlook": "Bullish",
            "priority_actions": [],
            "watch_list": [],
            "risk_notes": "",
        }
        result = get_playbook(date.today())
        assert result["market_outlook"] == "Bullish"

    def test_get_playbook_returns_none(self, mock_db):
        mock_db.fetchone.return_value = None
        result = get_playbook(date.today())
        assert result is None


class TestDecisionSignals:
    def test_insert_decision_signal(self, mock_db):
        insert_decision_signal(decision_id=1, signal_type="news_signal", signal_id=42)
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO decision_signals" in sql

    def test_insert_batch(self, mock_db):
        signals = [
            (1, "news_signal", 42),
            (1, "thesis", 5),
        ]
        insert_decision_signals_batch(signals)
        mock_db.execute.assert_called_once()

    def test_insert_batch_empty(self, mock_db):
        result = insert_decision_signals_batch([])
        assert result == 0
        mock_db.execute.assert_not_called()

    def test_get_decision_signals(self, mock_db):
        mock_db.fetchall.return_value = [
            {"decision_id": 1, "signal_type": "news_signal", "signal_id": 42},
        ]
        result = get_decision_signals(decision_id=1)
        assert len(result) == 1


class TestSignalAttribution:
    def test_upsert_signal_attribution(self, mock_db):
        upsert_signal_attribution(
            category="news:earnings",
            sample_size=20,
            avg_outcome_7d=Decimal("1.5"),
            avg_outcome_30d=Decimal("3.2"),
            win_rate_7d=Decimal("0.62"),
            win_rate_30d=Decimal("0.55"),
        )
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "INSERT INTO signal_attribution" in sql
        assert "ON CONFLICT (category)" in sql

    def test_get_signal_attribution(self, mock_db):
        mock_db.fetchall.return_value = [
            {"category": "news:earnings", "sample_size": 20, "win_rate_7d": Decimal("0.62")},
        ]
        result = get_signal_attribution()
        assert len(result) == 1
```

**Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_db_redesign.py -v`
Expected: ImportError — functions don't exist yet

**Step 4: Implement the DB functions**

Add to `trading/db.py` (append after the `clear_closed_orders` function):

```python
# --- Playbooks ---

def upsert_playbook(
    playbook_date: date,
    market_outlook: str,
    priority_actions: list,
    watch_list: list[str],
    risk_notes: str,
) -> int:
    """Insert or update a daily playbook."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO playbooks (date, market_outlook, priority_actions, watch_list, risk_notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                market_outlook = EXCLUDED.market_outlook,
                priority_actions = EXCLUDED.priority_actions,
                watch_list = EXCLUDED.watch_list,
                risk_notes = EXCLUDED.risk_notes,
                created_at = NOW()
            RETURNING id
        """, (playbook_date, market_outlook, Json(priority_actions), watch_list, risk_notes))
        return cur.fetchone()["id"]


def get_playbook(playbook_date: date) -> dict | None:
    """Get playbook for a specific date."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM playbooks WHERE date = %s
        """, (playbook_date,))
        return cur.fetchone()


# --- Decision Signals ---

def insert_decision_signal(decision_id: int, signal_type: str, signal_id: int):
    """Link a decision to a signal or thesis that motivated it."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO decision_signals (decision_id, signal_type, signal_id)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (decision_id, signal_type, signal_id))


def insert_decision_signals_batch(signals: list[tuple]) -> int:
    """Batch insert decision-signal links.

    Args:
        signals: List of (decision_id, signal_type, signal_id) tuples

    Returns:
        Number of signals inserted
    """
    if not signals:
        return 0
    with get_cursor() as cur:
        execute_values(cur, """
            INSERT INTO decision_signals (decision_id, signal_type, signal_id)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, signals)
        return len(signals)


def get_decision_signals(decision_id: int) -> list:
    """Get all signals linked to a decision."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM decision_signals WHERE decision_id = %s
        """, (decision_id,))
        return cur.fetchall()


# --- Signal Attribution ---

def upsert_signal_attribution(
    category: str,
    sample_size: int,
    avg_outcome_7d: Decimal,
    avg_outcome_30d: Decimal,
    win_rate_7d: Decimal,
    win_rate_30d: Decimal,
):
    """Insert or update signal attribution scores for a category."""
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO signal_attribution (category, sample_size, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (category) DO UPDATE SET
                sample_size = EXCLUDED.sample_size,
                avg_outcome_7d = EXCLUDED.avg_outcome_7d,
                avg_outcome_30d = EXCLUDED.avg_outcome_30d,
                win_rate_7d = EXCLUDED.win_rate_7d,
                win_rate_30d = EXCLUDED.win_rate_30d,
                updated_at = NOW()
        """, (category, sample_size, avg_outcome_7d, avg_outcome_30d, win_rate_7d, win_rate_30d))


def get_signal_attribution() -> list:
    """Get all signal attribution scores."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM signal_attribution
            ORDER BY sample_size DESC
        """)
        return cur.fetchall()
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_db_redesign.py -v`
Expected: All PASS

**Step 6: Add factory functions to conftest.py**

Add to `tests/conftest.py`:

```python
def make_playbook_row(**kwargs):
    """Create a playbook dict like what DB returns."""
    defaults = {
        "id": 1,
        "date": date.today(),
        "market_outlook": "Bullish on tech, cautious on energy",
        "priority_actions": [
            {"ticker": "NVDA", "action": "buy", "thesis_id": 1, "reasoning": "Entry trigger hit", "max_quantity": 5, "confidence": 0.8}
        ],
        "watch_list": ["AAPL", "MSFT", "GOOGL"],
        "risk_notes": "Fed meeting tomorrow, watch for volatility",
        "created_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults


def make_attribution_row(**kwargs):
    """Create a signal attribution dict like what DB returns."""
    defaults = {
        "id": 1,
        "category": "news:earnings",
        "sample_size": 20,
        "avg_outcome_7d": Decimal("1.50"),
        "avg_outcome_30d": Decimal("3.20"),
        "win_rate_7d": Decimal("0.62"),
        "win_rate_30d": Decimal("0.55"),
        "updated_at": datetime.now(),
    }
    defaults.update(kwargs)
    return defaults
```

**Step 7: Run full test suite to ensure no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: All existing tests still pass

**Step 8: Commit**

```bash
git add db/init/005_redesign.sql trading/db.py tests/test_db_redesign.py tests/conftest.py
git commit -m "feat: add playbooks, decision_signals, signal_attribution schema and DB functions"
```

---

## Task 2: Signal Attribution Engine

Compute attribution scores from `decision_signals` joined with `decisions`. This replaces the loosely-joined `patterns.py` approach with an explicit signal-decision link.

**Files:**
- Create: `trading/attribution.py`
- Test: `tests/test_attribution.py`

**Step 1: Write failing tests**

Create `tests/test_attribution.py`:

```python
"""Tests for signal attribution computation."""

from decimal import Decimal
from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager

import pytest

from trading.attribution import compute_signal_attribution, get_attribution_summary


@pytest.fixture
def mock_cursor():
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("trading.db.get_cursor", _get_cursor), \
         patch("trading.db.get_connection") as mock_conn:
        mock_conn.return_value = MagicMock()
        yield mock_cursor


class TestComputeAttribution:
    def test_computes_from_decision_signals(self, mock_db):
        """Attribution query joins decision_signals with decisions."""
        mock_db.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 15,
                "avg_outcome_7d": Decimal("2.10"),
                "avg_outcome_30d": Decimal("4.30"),
                "win_rate_7d": Decimal("0.67"),
                "win_rate_30d": Decimal("0.60"),
            },
            {
                "category": "thesis",
                "sample_size": 8,
                "avg_outcome_7d": Decimal("3.50"),
                "avg_outcome_30d": Decimal("5.10"),
                "win_rate_7d": Decimal("0.75"),
                "win_rate_30d": Decimal("0.63"),
            },
        ]

        results = compute_signal_attribution()
        assert len(results) == 2

        # Verify SQL references decision_signals
        sql = mock_db.execute.call_args_list[0][0][0]
        assert "decision_signals" in sql
        assert "decisions" in sql

    def test_empty_data_returns_empty(self, mock_db):
        mock_db.fetchall.return_value = []
        results = compute_signal_attribution()
        assert results == []

    def test_upserts_results(self, mock_db):
        """After computing, results are upserted to signal_attribution."""
        mock_db.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 10,
                "avg_outcome_7d": Decimal("1.5"),
                "avg_outcome_30d": Decimal("3.0"),
                "win_rate_7d": Decimal("0.60"),
                "win_rate_30d": Decimal("0.55"),
            },
        ]

        compute_signal_attribution()

        # Should have SELECT + at least one upsert
        assert mock_db.execute.call_count >= 2
        upsert_sql = mock_db.execute.call_args_list[-1][0][0]
        assert "signal_attribution" in upsert_sql


class TestGetAttributionSummary:
    def test_formats_summary(self, mock_db):
        mock_db.fetchall.return_value = [
            {
                "category": "news_signal:earnings",
                "sample_size": 15,
                "avg_outcome_7d": Decimal("2.10"),
                "avg_outcome_30d": None,
                "win_rate_7d": Decimal("0.67"),
                "win_rate_30d": None,
                "updated_at": None,
            },
        ]

        summary = get_attribution_summary()
        assert "news_signal:earnings" in summary
        assert "67" in summary  # win rate percentage

    def test_no_data_returns_message(self, mock_db):
        mock_db.fetchall.return_value = []
        summary = get_attribution_summary()
        assert "No attribution data" in summary
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_attribution.py -v`
Expected: ImportError

**Step 3: Implement attribution engine**

Create `trading/attribution.py`:

```python
"""Signal attribution engine — computes which signal types are predictive."""

from decimal import Decimal

from .db import get_cursor, upsert_signal_attribution, get_signal_attribution


def compute_signal_attribution() -> list[dict]:
    """
    Compute signal attribution scores from decision_signals joined with decisions.

    Groups by signal category (signal_type + news category for news signals)
    and computes avg outcomes and win rates.

    Returns:
        List of attribution dicts, also upserts into signal_attribution table.
    """
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
        """)
        results = [dict(row) for row in cur.fetchall()]

    # Upsert each result into signal_attribution table
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


def get_attribution_summary() -> str:
    """
    Format signal attribution scores as a text summary for LLM context.

    Returns:
        Formatted string with predictive and non-predictive signal categories.
    """
    rows = get_signal_attribution()

    if not rows:
        return "Signal Attribution:\n- No attribution data yet"

    lines = ["Signal Attribution:"]

    # Sort into predictive vs non-predictive
    predictive = [r for r in rows if r.get("win_rate_7d") and r["win_rate_7d"] > Decimal("0.5")]
    weak = [r for r in rows if r.get("win_rate_7d") and r["win_rate_7d"] <= Decimal("0.5")]

    if predictive:
        lines.append("Predictive signal types:")
        for r in predictive:
            wr = float(r["win_rate_7d"]) * 100
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: {wr:.0f}% win rate, {avg:+.2f}% avg 7d return (n={r['sample_size']})")

    if weak:
        lines.append("Weak/non-predictive signal types:")
        for r in weak:
            wr = float(r["win_rate_7d"]) * 100
            avg = float(r.get("avg_outcome_7d") or 0)
            lines.append(f"  - {r['category']}: {wr:.0f}% win rate, {avg:+.2f}% avg 7d return (n={r['sample_size']})")

    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_attribution.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add trading/attribution.py tests/test_attribution.py
git commit -m "feat: add signal attribution engine with compute and summary"
```

---

## Task 3: Context Builder — Add Playbook + Attribution Sections

Replace `get_strategy_context()` with `get_playbook_context()` and add `get_attribution_context()`. Update `build_trading_context()` to use playbook + attribution instead of strategy.

**Files:**
- Modify: `trading/context.py`
- Modify: `tests/test_context.py`

**Step 1: Write failing tests for new context functions**

Add tests to `tests/test_context.py` (find the existing test file and add):

```python
class TestPlaybookContext:
    def test_with_playbook(self, mock_db):
        mock_db.fetchone.return_value = make_playbook_row()
        result = get_playbook_context(date.today())
        assert "Playbook" in result
        assert "NVDA" in result

    def test_without_playbook(self, mock_db):
        mock_db.fetchone.return_value = None
        result = get_playbook_context(date.today())
        assert "No playbook" in result

    def test_conservative_fallback_message(self, mock_db):
        mock_db.fetchone.return_value = None
        result = get_playbook_context(date.today())
        assert "conservative" in result.lower()


class TestAttributionContext:
    def test_with_data(self, mock_db):
        mock_db.fetchall.return_value = [
            make_attribution_row(),
        ]
        result = get_attribution_context()
        assert "Attribution" in result

    def test_without_data(self, mock_db):
        mock_db.fetchall.return_value = []
        result = get_attribution_context()
        assert "No attribution" in result
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_context.py::TestPlaybookContext -v`
Expected: ImportError (functions don't exist)

**Step 3: Implement new context functions**

In `trading/context.py`, add imports and new functions:

```python
# Add to imports
from .db import get_playbook, get_signal_attribution
from .attribution import get_attribution_summary

def get_playbook_context(playbook_date: date) -> str:
    """
    Build playbook context section for the executor.

    Args:
        playbook_date: Date to fetch playbook for

    Returns:
        Formatted playbook context string
    """
    playbook = get_playbook(playbook_date)

    if not playbook:
        return (
            "Today's Playbook:\n"
            "- No playbook available. Operate in conservative mode: hold all positions, no new trades."
        )

    lines = ["Today's Playbook:"]

    if playbook.get("market_outlook"):
        lines.append(f"Market Outlook: {playbook['market_outlook']}")

    actions = playbook.get("priority_actions") or []
    if actions:
        lines.append("")
        lines.append("Priority Actions:")
        for action in actions:
            ticker = action.get("ticker", "?")
            act = action.get("action", "?").upper()
            reasoning = action.get("reasoning", "")
            confidence = action.get("confidence", "?")
            max_qty = action.get("max_quantity", "?")
            thesis_id = action.get("thesis_id")
            thesis_note = f" (thesis #{thesis_id})" if thesis_id else ""
            lines.append(f"- {act} {ticker}: {reasoning} [confidence: {confidence}, max qty: {max_qty}]{thesis_note}")

    watch = playbook.get("watch_list") or []
    if watch:
        lines.append(f"\nWatch List: {', '.join(watch)}")

    if playbook.get("risk_notes"):
        lines.append(f"\nRisk Notes: {playbook['risk_notes']}")

    return "\n".join(lines)


def get_attribution_context() -> str:
    """Build attribution context section."""
    return get_attribution_summary()
```

Then update `build_trading_context()` to replace `get_strategy_context()`:

```python
def build_trading_context(account_info: dict, playbook_date: date = None) -> str:
    """
    Build complete compressed context for trading agent.

    Args:
        account_info: Dict with cash, portfolio_value, buying_power from Alpaca
        playbook_date: Date for playbook lookup (defaults to today)

    Returns:
        Complete formatted context string
    """
    if playbook_date is None:
        playbook_date = date.today()

    sections = [
        get_playbook_context(playbook_date),
        "",
        get_portfolio_context(account_info),
        "",
        get_theses_context(),
        "",
        get_macro_context(days=7),
        "",
        get_ticker_signals_context(days=1),
        "",
        get_signal_trend_context(days=7),
        "",
        get_decision_outcomes_context(days=30),
        "",
        get_attribution_context(),
    ]

    return "\n".join(sections)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_context.py -v`
Expected: All PASS (both old and new tests)

**Step 5: Commit**

```bash
git add trading/context.py tests/test_context.py
git commit -m "feat: context builder reads playbook and attribution instead of strategy"
```

---

## Task 4: Ollama Agent — Playbook-Oriented System Prompt

Update the Ollama agent's system prompt to be oriented around executing a playbook. Require signal/thesis citation in every decision.

**Files:**
- Modify: `trading/agent.py`
- Modify: `tests/test_agent.py`

**Step 1: Write failing test for new response format**

Add to `tests/test_agent.py`:

```python
def test_decisions_include_signal_refs():
    """Decisions should include signal_refs list."""
    decision = TradingDecision(
        action="buy",
        ticker="NVDA",
        quantity=5,
        reasoning="Entry trigger hit per playbook",
        confidence="high",
        thesis_id=42,
        signal_refs=[{"type": "news_signal", "id": 15}, {"type": "thesis", "id": 42}],
    )
    assert len(decision.signal_refs) == 2
    assert decision.signal_refs[0]["type"] == "news_signal"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent.py::test_decisions_include_signal_refs -v`
Expected: TypeError (signal_refs not a field)

**Step 3: Update TradingDecision and system prompt**

In `trading/agent.py`:

1. Add `signal_refs` field to `TradingDecision`:

```python
@dataclass
class TradingDecision:
    """A trading decision from the LLM."""
    action: str         # buy, sell, hold
    ticker: str
    quantity: Optional[int]
    reasoning: str
    confidence: str     # high, medium, low
    thesis_id: Optional[int] = None
    signal_refs: list = None  # [{"type": "news_signal", "id": 15}, ...]

    def __post_init__(self):
        if self.signal_refs is None:
            self.signal_refs = []
```

2. Replace `TRADING_SYSTEM_PROMPT` with the new playbook-oriented prompt:

```python
TRADING_SYSTEM_PROMPT = """You are executing a trading playbook prepared by a senior strategist. Your job is to translate the playbook into concrete trading decisions.

You will receive:
1. Today's Playbook — priority actions, watch list, risk notes from the strategist
2. Current portfolio state (positions, cash, buying power)
3. Active theses with entry/exit triggers
4. Macro economic context
5. Overnight ticker signals
6. Signal attribution scores (which signal types have been historically predictive)
7. Recent decision outcomes

For each priority action in the playbook, decide: execute as-is, adjust quantity/timing, or skip (with reason).

You may propose additional trades if overnight signals warrant it, but playbook actions come first.

Rules:
- Be conservative with position sizing (suggest 1-5% of buying power per trade)
- Provide clear reasoning for each decision
- Weight playbook actions heavily — they represent pre-analyzed opportunities
- Consider signal attribution scores — give more weight to historically predictive signal types
- If no playbook is available, operate in conservative mode: hold everything, no new positions
- Never suggest using more than available buying power
- If uncertain, recommend HOLD

CRITICAL — Signal Citation:
For EVERY decision, you MUST cite which signal IDs and/or thesis IDs informed it in the signal_refs field. This is required for the learning loop. Use the format:
  [{"type": "news_signal", "id": <id>}, {"type": "thesis", "id": <id>}, ...]

Respond with valid JSON only:
{
    "decisions": [
        {
            "action": "buy" | "sell" | "hold",
            "ticker": "NVDA",
            "quantity": 5,
            "reasoning": "Playbook priority action — entry trigger hit...",
            "confidence": "high" | "medium" | "low",
            "thesis_id": 42,
            "signal_refs": [{"type": "thesis", "id": 42}, {"type": "news_signal", "id": 15}]
        }
    ],
    "thesis_invalidations": [
        {
            "thesis_id": 123,
            "reason": "Observed condition matching invalidation criteria..."
        }
    ],
    "market_summary": "Brief summary...",
    "risk_assessment": "Current risk level..."
}"""
```

3. Update `get_trading_decisions` to parse `signal_refs`:

In the decision-building loop, add:
```python
signal_refs=d.get("signal_refs", []),
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add trading/agent.py tests/test_agent.py
git commit -m "feat: agent uses playbook-oriented prompt with signal citation"
```

---

## Task 5: Trader — Log Decision-Signal Links

Update `run_trading_session()` to insert `decision_signals` rows after logging each decision. Parse `signal_refs` from agent response.

**Files:**
- Modify: `trading/trader.py`
- Modify: `tests/test_trader.py`

**Step 1: Write failing test**

Add to `tests/test_trader.py`:

```python
def test_logs_decision_signal_links(mock_all):
    """After logging a decision, signal_refs should be inserted into decision_signals."""
    # Set up a decision with signal_refs
    decision = make_trading_decision(
        signal_refs=[{"type": "thesis", "id": 42}, {"type": "news_signal", "id": 15}],
    )
    mock_all["get_trading_decisions"].return_value = make_agent_response(
        decisions=[decision],
    )

    result = run_trading_session(dry_run=True)

    # Should have called insert_decision_signals_batch
    mock_all["insert_decision_signals_batch"].assert_called_once()
    args = mock_all["insert_decision_signals_batch"].call_args[0][0]
    assert len(args) == 2  # thesis + news_signal
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trader.py::test_logs_decision_signal_links -v`
Expected: FAIL

**Step 3: Update trader.py**

In `trading/trader.py`:

1. Add import:
```python
from .db import insert_decision, get_positions, close_thesis, insert_decision_signals_batch
```

2. In Step 6 (logging decisions), after `insert_decision()`, add:
```python
            # Log signal-decision links for attribution
            if decision.signal_refs:
                signal_links = [
                    (decision_id, ref["type"], ref["id"])
                    for ref in decision.signal_refs
                ]
                insert_decision_signals_batch(signal_links)
```

Where `decision_id` is the return value of `insert_decision()` (currently unused — capture it).

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_trader.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add trading/trader.py tests/test_trader.py
git commit -m "feat: trader logs decision-signal links for attribution loop"
```

---

## Task 6: Claude Strategist — New Tools

Add three new tools to `trading/tools.py` for the strategist session: `get_signal_attribution`, `get_decision_history`, `write_playbook`.

**Files:**
- Modify: `trading/tools.py`
- Modify: `tests/test_tools.py`

**Step 1: Write failing tests**

Add to `tests/test_tools.py`:

```python
class TestNewTools:
    def test_get_signal_attribution_tool(self, mock_db):
        mock_db.fetchall.return_value = [make_attribution_row()]
        result = tool_get_signal_attribution()
        assert "earnings" in result

    def test_get_decision_history_tool(self, mock_db):
        mock_db.fetchall.return_value = [make_decision_row()]
        result = tool_get_decision_history(days=30)
        assert "AAPL" in result

    def test_write_playbook_tool(self, mock_db):
        result = tool_write_playbook(
            market_outlook="Bullish tech",
            priority_actions=[{"ticker": "NVDA", "action": "buy"}],
            watch_list=["AAPL"],
            risk_notes="Watch Fed",
        )
        assert "Playbook written" in result
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_tools.py::TestNewTools -v`
Expected: ImportError

**Step 3: Implement new tools**

Add to `trading/tools.py`:

```python
# Add imports
from .attribution import get_attribution_summary
from .db import get_recent_decisions, upsert_playbook
from datetime import date

def tool_get_signal_attribution() -> str:
    """Get signal attribution scores."""
    logger.info("Getting signal attribution")
    return get_attribution_summary()


def tool_get_decision_history(days: int = 30) -> str:
    """Get recent decisions with outcomes."""
    logger.info(f"Getting decision history ({days} days)")
    decisions = get_recent_decisions(days=days)

    if not decisions:
        return f"No decisions in the last {days} days."

    lines = []
    for d in decisions:
        outcome_7d = f"{d['outcome_7d']:+.2f}%" if d.get("outcome_7d") is not None else "pending"
        outcome_30d = f"{d['outcome_30d']:+.2f}%" if d.get("outcome_30d") is not None else "pending"
        lines.append(
            f"- [{d['date']}] {d['action'].upper()} {d['ticker']}: "
            f"7d={outcome_7d}, 30d={outcome_30d} — {d['reasoning'][:80]}"
        )

    return "\n".join(lines)


def tool_write_playbook(
    market_outlook: str,
    priority_actions: list,
    watch_list: list,
    risk_notes: str,
) -> str:
    """Write tomorrow's playbook to the database."""
    logger.info("Writing playbook")
    try:
        # Write for tomorrow (next trading day)
        from datetime import timedelta
        tomorrow = date.today() + timedelta(days=1)
        playbook_id = upsert_playbook(
            playbook_date=tomorrow,
            market_outlook=market_outlook,
            priority_actions=priority_actions,
            watch_list=watch_list,
            risk_notes=risk_notes,
        )
        return f"Playbook written for {tomorrow} (ID: {playbook_id})"
    except Exception as e:
        logger.exception("Failed to write playbook")
        return f"Error writing playbook: {e}"
```

Add to `TOOL_DEFINITIONS`:

```python
    {
        "name": "get_signal_attribution",
        "description": (
            "Get signal attribution scores showing which signal types "
            "(news categories, macro categories, theses) have been "
            "historically predictive based on decision outcomes."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_decision_history",
        "description": (
            "Get recent trading decisions with their outcomes (7d and 30d P&L). "
            "Use this to review what trades were made and how they performed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look back period in days (default: 30)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "write_playbook",
        "description": (
            "Write tomorrow's trading playbook. This is the primary output of "
            "the strategist session — it tells the executor what to do."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "market_outlook": {
                    "type": "string",
                    "description": "Brief market outlook for tomorrow",
                },
                "priority_actions": {
                    "type": "array",
                    "description": "Ordered list of priority trades",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "action": {"type": "string", "enum": ["buy", "sell"]},
                            "thesis_id": {"type": "integer"},
                            "reasoning": {"type": "string"},
                            "max_quantity": {"type": "integer"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["ticker", "action", "reasoning", "confidence"],
                    },
                },
                "watch_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tickers to monitor for signals",
                },
                "risk_notes": {
                    "type": "string",
                    "description": "Risk factors and warnings",
                },
            },
            "required": ["market_outlook", "priority_actions", "watch_list", "risk_notes"],
        },
    },
```

Add to `TOOL_HANDLERS`:

```python
    "get_signal_attribution": tool_get_signal_attribution,
    "get_decision_history": tool_get_decision_history,
    "write_playbook": tool_write_playbook,
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tools.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add trading/tools.py tests/test_tools.py
git commit -m "feat: add strategist tools — attribution, decision history, write playbook"
```

---

## Task 7: Claude Strategist Session

Expand `ideation_claude.py` into the full strategist session: backfill -> attribution -> self-reflection -> thesis management -> playbook generation.

**Files:**
- Modify: `trading/ideation_claude.py`
- Modify: `tests/test_ideation_claude.py`

**Step 1: Write failing tests**

Add to `tests/test_ideation_claude.py`:

```python
class TestStrategistSession:
    def test_runs_backfill_first(self, mock_all):
        """Strategist session should run backfill before starting Claude loop."""
        run_strategist_session()
        mock_all["run_backfill"].assert_called_once()

    def test_computes_attribution(self, mock_all):
        """Strategist session should compute attribution after backfill."""
        run_strategist_session()
        mock_all["compute_signal_attribution"].assert_called_once()

    def test_system_prompt_includes_strategist_role(self, mock_all):
        """System prompt should reference strategist role and playbook writing."""
        run_strategist_session()
        system = mock_all["run_agentic_loop"].call_args[1]["system"]
        assert "strategist" in system.lower() or "playbook" in system.lower()

    def test_returns_strategist_result(self, mock_all):
        result = run_strategist_session()
        assert hasattr(result, "outcomes_backfilled")
        assert hasattr(result, "attribution_computed")
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ideation_claude.py::TestStrategistSession -v`
Expected: ImportError

**Step 3: Implement strategist session**

Rename `run_ideation_claude` to `run_strategist_session` (keep old name as alias for backward compat). Update the function to:

1. Run backfill first
2. Compute signal attribution
3. Include attribution + decision history in the initial message to Claude
4. Update system prompt to include strategist role and playbook writing
5. Return a `StrategistResult` dataclass with all session details

The new system prompt:

```python
CLAUDE_STRATEGIST_SYSTEM = """You are the strategist for an automated trading system. You run after market close to review results, manage theses, and write tomorrow's playbook for the executor.

## Your Daily Process

1. **Review**: Check the portfolio state, recent decision outcomes, and signal attribution scores you've been given. Identify what's working and what isn't.

2. **Thesis Management**: For each active thesis:
   - If still valid, keep it
   - If conditions changed, update it
   - If invalidation criteria met, close it
   - If aged out without action, close as expired

3. **Research & New Theses**: Use web search to investigate opportunities. Create 2-4 new well-reasoned theses with specific entry/exit triggers.

4. **Write Playbook**: This is your primary output. Use the `write_playbook` tool to create tomorrow's trading plan with:
   - Market outlook
   - Priority actions (specific trades the executor should consider)
   - Watch list (tickers to monitor)
   - Risk notes (warnings, upcoming events)

## Tool Usage
- Use `get_signal_attribution` to see which signal types have been historically predictive
- Use `get_decision_history` to review recent trading performance
- Use `web_search` to research current market conditions and companies
- Use `get_market_snapshot` to see sector performance and movers
- Use thesis tools to manage trade ideas
- Use `write_playbook` to write tomorrow's plan (REQUIRED — always write a playbook)

## Critical Rules
1. **Always write a playbook** — the executor depends on it
2. **Quality theses only** — specific entry/exit triggers, not vague ideas
3. **Learn from attribution** — weight signal types that have been predictive
4. **No duplicate theses** — check existing before creating"""
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ideation_claude.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add trading/ideation_claude.py tests/test_ideation_claude.py
git commit -m "feat: expand Claude ideation into full strategist session with backfill + attribution + playbook"
```

---

## Task 8: Remove RAG Infrastructure

Delete files that are no longer needed: `ingest.py`, `ingest_scheduler.py`, `retrieval.py`, `ollama.py`. Remove their tests. Update imports.

**Files:**
- Delete: `trading/ingest.py`
- Delete: `trading/ingest_scheduler.py`
- Delete: `trading/retrieval.py`
- Delete: `trading/ollama.py`
- Delete: `db/init/004_documents.sql`
- Delete: associated test files
- Modify: `tests/conftest.py` (remove Ollama fixtures)

**Step 1: Identify all imports of removed modules**

Search for imports of `ingest`, `ingest_scheduler`, `retrieval`, `ollama` across the codebase to ensure nothing else depends on them.

Run: `grep -rn "from.*trading\.\(ingest\|retrieval\|ollama\)" trading/ --include="*.py"`
Run: `grep -rn "import.*trading\.\(ingest\|retrieval\|ollama\)" trading/ --include="*.py"`

Expected files referencing them:
- `trading/classifier.py` imports `ollama.chat_json` — needs update
- `trading/filter.py` imports `ollama.embed`, `ollama.cosine_similarity_batch` — needs update
- `trading/pipeline.py` may reference classifier which uses ollama
- `trading/learn.py` may import patterns/strategy
- `trading/ideation.py` references `ollama` — this is the old Ollama ideation

**Step 2: Update classifier.py and filter.py (or defer)**

The `classifier.py` and `filter.py` currently depend on Ollama for classification and embedding. Per the design doc, the signal pipeline stays but the underlying LLM calls need updating. Since this is a larger change (swapping Ollama for Claude in classification), we should **keep the Ollama module for now but only the chat functions needed by classifier/filter**, or defer the classifier migration to a separate task.

**Decision:** Keep `ollama.py` for now since `classifier.py` and `filter.py` depend on it for the signal pipeline. Only delete `ingest.py`, `ingest_scheduler.py`, `retrieval.py`, and `004_documents.sql`.

**Step 3: Delete files**

```bash
rm trading/ingest.py trading/ingest_scheduler.py trading/retrieval.py
rm db/init/004_documents.sql
```

**Step 4: Delete associated tests**

```bash
rm tests/test_ingest.py tests/test_ingest_scheduler.py tests/test_retrieval.py
```

(Only delete tests that exist — check first.)

**Step 5: Remove unused fixtures from conftest.py**

The `mock_embed`, `mock_embed_batch` fixtures in conftest.py should be checked for remaining users. If only RAG tests use them, remove them.

**Step 6: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All remaining tests pass (RAG tests are gone, everything else works)

**Step 7: Commit**

```bash
git add -A
git commit -m "chore: remove RAG infrastructure (ingest, retrieval, documents table)"
```

---

## Task 9: Remove Strategy Module

Replace the `strategy` table with `playbooks`. Remove `trading/strategy.py` and update `trading/learn.py`.

**Files:**
- Delete: `trading/strategy.py`
- Modify: `trading/learn.py`
- Modify: `trading/context.py` (remove `get_strategy_context` if still present)
- Delete: associated tests

**Step 1: Remove strategy references from learn.py**

`trading/learn.py` currently imports `evolve_strategy` from `strategy.py`. Update to remove Step 3 (strategy evolution) and replace with attribution computation:

```python
"""Learning loop orchestrator - backfill and compute attribution."""

from dataclasses import dataclass
from datetime import datetime

from .backfill import run_backfill
from .attribution import compute_signal_attribution
from .patterns import generate_pattern_report


@dataclass
class LearningResult:
    """Result of a learning loop run."""
    timestamp: datetime
    outcomes_backfilled: int
    attribution_computed: int
    pattern_report: str
    errors: list[str]
```

Update `run_learning_loop()` to call `compute_signal_attribution()` instead of `evolve_strategy()`.

**Step 2: Delete strategy.py and its tests**

```bash
rm trading/strategy.py tests/test_strategy.py
```

**Step 3: Remove get_strategy_context from context.py**

Delete the `get_strategy_context()` function if it's still in `context.py`. It was replaced by `get_playbook_context()` in Task 3.

**Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove strategy module, replace with playbook + attribution in learn.py"
```

---

## Task 10: Dashboard — Playbook and Attribution Views

Add playbook view and attribution dashboard to the Flask app.

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `dashboard/app.py`
- Create: `dashboard/templates/playbook.html`
- Create: `dashboard/templates/attribution.html`
- Modify: `tests/test_dashboard.py`

**Step 1: Add dashboard query functions**

Add to `dashboard/queries.py`:

```python
def get_today_playbook():
    """Get today's playbook."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT * FROM playbooks
            WHERE date = CURRENT_DATE
            ORDER BY created_at DESC
            LIMIT 1
        """)
        return cur.fetchone()


def get_signal_attribution():
    """Get signal attribution scores."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT category, sample_size, avg_outcome_7d, avg_outcome_30d,
                   win_rate_7d, win_rate_30d, updated_at
            FROM signal_attribution
            ORDER BY sample_size DESC
        """)
        return cur.fetchall()


def get_decision_signal_refs(decision_id: int):
    """Get signal refs for a decision."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT signal_type, signal_id
            FROM decision_signals
            WHERE decision_id = %s
        """, (decision_id,))
        return cur.fetchall()
```

**Step 2: Add dashboard routes**

Add to `dashboard/app.py`:

```python
@app.route("/playbook")
def playbook():
    """Today's playbook view."""
    today = get_today_playbook()
    return render_template("playbook.html", playbook=today)


@app.route("/attribution")
def attribution():
    """Signal attribution dashboard."""
    scores = get_signal_attribution()
    return render_template("attribution.html", scores=scores)
```

**Step 3: Update portfolio route to use playbook instead of strategy**

Replace `get_current_strategy()` with `get_today_playbook()` in the portfolio route.

**Step 4: Create templates**

Create minimal `playbook.html` and `attribution.html` templates.

**Step 5: Write tests**

Add dashboard tests for the new routes and queries.

**Step 6: Run tests**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add dashboard/ tests/test_dashboard.py
git commit -m "feat: add playbook and attribution views to dashboard"
```

---

## Task 11: Docker Compose Update

Update `docker-compose.yml` to remove pgvector image dependency.

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Update db service image**

Change `pgvector/pgvector:pg16` to `postgres:16`:

```yaml
  db:
    image: postgres:16
```

**Step 2: Verify init scripts don't reference pgvector**

Confirm `004_documents.sql` was deleted in Task 8 (it had `CREATE EXTENSION IF NOT EXISTS vector`).

**Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: switch from pgvector to plain postgres (RAG removed)"
```

---

## Task 12: Integration Test — Full Cycle

Write an integration test that exercises the full strategist -> executor cycle with mocked external services.

**Files:**
- Create: `tests/test_full_cycle.py`

**Step 1: Write integration test**

```python
"""Integration test: strategist writes playbook, executor reads and acts on it."""

def test_strategist_writes_playbook_executor_reads_it(mock_all):
    """
    1. Strategist session writes a playbook
    2. Executor session reads the playbook and makes decisions
    3. Decisions include signal_refs from the playbook
    4. Attribution can be computed from decision_signals
    """
    # Step 1: Strategist writes playbook
    from trading.db import upsert_playbook, get_playbook
    from datetime import date

    upsert_playbook(
        playbook_date=date.today(),
        market_outlook="Bullish tech",
        priority_actions=[{"ticker": "NVDA", "action": "buy", "thesis_id": 1, "reasoning": "Entry hit", "confidence": 0.8, "max_quantity": 5}],
        watch_list=["AAPL"],
        risk_notes="Fed meeting",
    )

    # Step 2: Executor reads playbook
    playbook = get_playbook(date.today())
    assert playbook is not None
    assert playbook["market_outlook"] == "Bullish tech"

    # Step 3: Context includes playbook
    from trading.context import get_playbook_context
    ctx = get_playbook_context(date.today())
    assert "NVDA" in ctx
    assert "Entry hit" in ctx
```

**Step 2: Run test**

Run: `python3 -m pytest tests/test_full_cycle.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_full_cycle.py
git commit -m "test: add integration test for strategist -> executor cycle"
```

---

## Task 13: Update conftest.py and Ensure Full Suite Passes

Final cleanup: update factory functions, remove stale fixtures, run full test suite.

**Files:**
- Modify: `tests/conftest.py`
- Modify: `trading/__init__.py` (if needed)

**Step 1: Update make_trading_decision factory**

Add `signal_refs` default:

```python
def make_trading_decision(**kwargs):
    defaults = {
        "action": "buy",
        "ticker": "AAPL",
        "quantity": 10,
        "reasoning": "Strong earnings beat",
        "confidence": "high",
        "thesis_id": None,
        "signal_refs": [],
    }
    defaults.update(kwargs)
    return TradingDecision(**defaults)
```

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 3: Run coverage**

Run: `python3 -m pytest tests/ --cov=trading --cov=dashboard`
Expected: Coverage at or above previous level for modified files

**Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "chore: update test factories and fixtures for redesign"
```

---

## Summary of Changes

### New Files
| File | Purpose |
|------|---------|
| `db/init/005_redesign.sql` | Schema for playbooks, decision_signals, signal_attribution |
| `trading/attribution.py` | Signal attribution computation engine |
| `tests/test_db_redesign.py` | Tests for new DB functions |
| `tests/test_attribution.py` | Tests for attribution engine |
| `tests/test_full_cycle.py` | Integration test |
| `dashboard/templates/playbook.html` | Playbook view template |
| `dashboard/templates/attribution.html` | Attribution dashboard template |

### Modified Files
| File | Change |
|------|--------|
| `trading/db.py` | Add playbook, decision_signal, attribution CRUD functions |
| `trading/context.py` | Replace strategy context with playbook + attribution |
| `trading/agent.py` | New playbook-oriented prompt, signal_refs field |
| `trading/trader.py` | Log decision-signal links |
| `trading/tools.py` | Add 3 strategist tools |
| `trading/ideation_claude.py` | Expand to full strategist session |
| `trading/learn.py` | Replace strategy evolution with attribution |
| `dashboard/queries.py` | Add playbook + attribution queries |
| `dashboard/app.py` | Add playbook + attribution routes |
| `docker-compose.yml` | Switch to plain postgres |
| `tests/conftest.py` | Add new factories, update existing |

### Deleted Files
| File | Reason |
|------|--------|
| `trading/ingest.py` | RAG removed |
| `trading/ingest_scheduler.py` | RAG removed |
| `trading/retrieval.py` | RAG removed |
| `trading/strategy.py` | Replaced by playbooks |
| `db/init/004_documents.sql` | RAG removed |
| `tests/test_ingest.py` | RAG removed |
| `tests/test_ingest_scheduler.py` | RAG removed |
| `tests/test_retrieval.py` | RAG removed |
| `tests/test_strategy.py` | Strategy replaced |
