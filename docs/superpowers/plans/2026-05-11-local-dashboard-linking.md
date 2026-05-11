# Local Dashboard Linking + Detail Pages — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 entity detail pages (`/ticker/<sym>`, `/thesis/<id>`, `/decision/<id>`, `/session/<id>`) and inline cross-links across the local operator dashboard so every meaningful relationship in the schema is one click away.

**Architecture:** Pure additive change to the Flask local dashboard (`dashboard/`). New query functions wrap existing schema joins (`decision_signals`, `playbook_actions.thesis_id`, `tweets.decision_id`, etc.); new route handlers compose those queries into context dicts; new templates extend `base.html`. Existing templates get `<a href="...">` wraps around tickers, IDs, and categories. No schema changes, no v2/dashboard work.

**Tech Stack:** Python 3, Flask, Jinja2, psycopg2 (`get_cursor()` context manager), Tailwind via CDN. Tests use pytest with `sys.modules["queries"]` MagicMock injection pattern from `tests/test_dashboard.py`.

**Spec:** `docs/superpowers/specs/2026-05-11-local-dashboard-linking-design.md`

---

## File Structure

**New files:**
- `dashboard/templates/ticker.html` — ticker overview page
- `dashboard/templates/thesis_detail.html` — single thesis detail
- `dashboard/templates/decision_detail.html` — single decision detail
- `dashboard/templates/session_detail.html` — single session detail

**Modified files:**
- `dashboard/queries.py` — ~15 new functions, 3 extended functions
- `dashboard/app.py` — 4 new routes, 2 modified routes (`?category=` filter)
- `dashboard/templates/portfolio.html` — ticker links
- `dashboard/templates/playbook.html` — ticker + thesis links
- `dashboard/templates/signals.html` — ticker + category links
- `dashboard/templates/theses.html` — ticker + detail links
- `dashboard/templates/decisions.html` — ticker + signal-ref + detail links
- `dashboard/templates/attribution.html` — category links + anchor IDs
- `dashboard/templates/strategy.html` — memo session links
- `dashboard/templates/events.html` — session detail link
- `dashboard/templates/tweets.html` — session + decision links
- `dashboard/templates/costs.html` — session detail link
- `tests/conftest.py` — 3 new factory functions (`make_session_row`, `make_session_stage_cost_row`, `make_agent_event_row`)
- `tests/test_dashboard.py` — ~25 new tests

---

## Phase 1 — New Query Functions

Build the query layer first. Each task adds a focused set of queries with unit-mocked tests against `dashboard/queries.py`. Tests in this phase **do not** hit the dashboard app — they patch `dashboard.queries.get_cursor` directly. Look at `tests/test_dashboard.py` for the existing patterns; this phase uses a different approach (direct `get_cursor` mock) because we are testing the SQL layer, not the route.

### Task 1: Test fixtures for sessions, stage costs, agent events

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add factory functions to conftest.py**

Append the following three factories to `tests/conftest.py`, right after `make_tweet_row` (around line 489):

```python
def make_session_row(**kwargs):
    """Create a sessions dict like what DB returns."""
    defaults = {
        "id": 1,
        "session_date": date.today(),
        "session_type": "daily",
        "status": "completed",
        "started_at": datetime.now() - timedelta(hours=2),
        "completed_at": datetime.now() - timedelta(hours=1),
        "error": None,
    }
    defaults.update(kwargs)
    return defaults


def make_session_stage_cost_row(**kwargs):
    """Create a session_stage_costs dict like what DB returns."""
    defaults = {
        "id": 1,
        "stage_name": "ideation",
        "status": "completed",
        "started_at": datetime.now() - timedelta(hours=2),
        "completed_at": datetime.now() - timedelta(hours=2) + timedelta(minutes=4),
        "model": "claude-sonnet-4-6",
        "input_tokens": 12000,
        "output_tokens": 800,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 10000,
        "cost_usd": Decimal("0.0420"),
    }
    defaults.update(kwargs)
    return defaults


def make_agent_event_row(**kwargs):
    """Create an agent_events dict like what DB returns."""
    defaults = {
        "id": 1,
        "session_id": 1,
        "stage_name": "trading",
        "event_type": "tool_call",
        "payload": {"tool": "get_portfolio_state", "args": {}},
        "occurred_at": datetime.now() - timedelta(hours=1),
    }
    defaults.update(kwargs)
    return defaults
```

- [ ] **Step 2: Verify factories importable**

Run: `python -c "from tests.conftest import make_session_row, make_session_stage_cost_row, make_agent_event_row; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test(conftest): add session, stage-cost, and agent-event factories"
```

---

### Task 2: `lookup_session_id_by_date`

**Files:**
- Modify: `dashboard/queries.py`
- Test: `tests/test_dashboard_queries.py` (new file)

- [ ] **Step 1: Create test file with failing test**

Create `tests/test_dashboard_queries.py`:

```python
"""Direct unit tests for dashboard/queries.py SQL functions.

These tests patch `dashboard.queries.get_cursor` to a context manager that
yields a MagicMock cursor — no Flask app, no DB. We assert the SQL string
and bound parameters where it matters, and the shaped return value always.
"""

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    make_agent_event_row,
    make_decision_row,
    make_macro_signal_row,
    make_news_signal_row,
    make_open_order_row,
    make_playbook_action_row,
    make_position_row,
    make_session_row,
    make_session_stage_cost_row,
    make_strategy_memo_row,
    make_thesis_row,
    make_tweet_row,
)


@pytest.fixture
def cur():
    """Return a MagicMock cursor patched into dashboard.queries.get_cursor."""
    mock_cursor = MagicMock()

    @contextmanager
    def _get_cursor():
        yield mock_cursor

    with patch("dashboard.queries.get_cursor", _get_cursor):
        yield mock_cursor


class TestLookupSessionIdByDate:
    def test_returns_id_when_found(self, cur):
        from dashboard.queries import lookup_session_id_by_date
        cur.fetchone.return_value = {"id": 42}
        result = lookup_session_id_by_date(date(2026, 5, 11))
        assert result == 42

    def test_returns_none_when_no_session(self, cur):
        from dashboard.queries import lookup_session_id_by_date
        cur.fetchone.return_value = None
        result = lookup_session_id_by_date(date(2026, 5, 11))
        assert result is None

    def test_filters_by_session_type(self, cur):
        from dashboard.queries import lookup_session_id_by_date
        cur.fetchone.return_value = {"id": 7}
        lookup_session_id_by_date(date(2026, 5, 11), session_type="premarket")
        # Verify the executed SQL had the type param
        called_sql = cur.execute.call_args[0][0]
        called_params = cur.execute.call_args[0][1]
        assert "session_type" in called_sql
        assert "premarket" in called_params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestLookupSessionIdByDate -v`
Expected: FAIL with `ImportError: cannot import name 'lookup_session_id_by_date'`.

- [ ] **Step 3: Implement `lookup_session_id_by_date`**

Append to `dashboard/queries.py`:

```python
def lookup_session_id_by_date(d, session_type: str = 'daily'):
    """Return the most recent sessions.id for a given date + type, or None.

    Multiple sessions can share a (date, type) only if the UNIQUE constraint
    is bypassed — in practice the ON CONFLICT path keeps one row per pair —
    but we ORDER BY started_at DESC for safety.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT id
            FROM sessions
            WHERE session_date = %s AND session_type = %s
            ORDER BY started_at DESC
            LIMIT 1
        """, (d, session_type))
        row = cur.fetchone()
        return row["id"] if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestLookupSessionIdByDate -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add lookup_session_id_by_date query"
```

---

### Task 3: Thesis detail queries

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `tests/test_dashboard_queries.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_dashboard_queries.py`:

```python
class TestGetThesis:
    def test_returns_thesis_row(self, cur):
        from dashboard.queries import get_thesis
        cur.fetchone.return_value = make_thesis_row(id=5)
        result = get_thesis(5)
        assert result["id"] == 5

    def test_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_thesis
        cur.fetchone.return_value = None
        assert get_thesis(999) is None


class TestGetThesisDecisions:
    def test_returns_decisions_joined_through_decision_signals(self, cur):
        from dashboard.queries import get_thesis_decisions
        cur.fetchall.return_value = [make_decision_row(id=1), make_decision_row(id=2)]
        result = get_thesis_decisions(5)
        assert len(result) == 2
        # Verify SQL joins decision_signals filtered by signal_type='thesis'
        called_sql = cur.execute.call_args[0][0]
        assert "decision_signals" in called_sql
        assert "thesis" in called_sql
        assert 5 in cur.execute.call_args[0][1]

    def test_returns_empty_when_no_decisions(self, cur):
        from dashboard.queries import get_thesis_decisions
        cur.fetchall.return_value = []
        assert get_thesis_decisions(5) == []


class TestGetThesisPlaybookActions:
    def test_returns_actions_for_thesis(self, cur):
        from dashboard.queries import get_thesis_playbook_actions
        cur.fetchall.return_value = [make_playbook_action_row(thesis_id=5)]
        result = get_thesis_playbook_actions(5)
        assert len(result) == 1
        assert 5 in cur.execute.call_args[0][1]
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestGetThesis tests/test_dashboard_queries.py::TestGetThesisDecisions tests/test_dashboard_queries.py::TestGetThesisPlaybookActions -v`
Expected: ImportError on `get_thesis`, `get_thesis_decisions`, `get_thesis_playbook_actions`.

- [ ] **Step 3: Implement queries**

Append to `dashboard/queries.py`:

```python
def get_thesis(thesis_id: int):
    """Return one thesis row, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, source, status,
                   created_at, updated_at, closed_at, close_reason
            FROM theses
            WHERE id = %s
        """, (thesis_id,))
        return cur.fetchone()


def get_thesis_decisions(thesis_id: int):
    """Return decisions that cited this thesis (via decision_signals)."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT d.id, d.date, d.ticker, d.action, d.quantity, d.price,
                   d.reasoning, d.account_equity, d.outcome_7d, d.outcome_30d,
                   d.is_off_playbook, d.playbook_action_id
            FROM decisions d
            JOIN decision_signals ds ON ds.decision_id = d.id
            WHERE ds.signal_type = 'thesis' AND ds.signal_id = %s
            ORDER BY d.date DESC, d.id DESC
        """, (thesis_id,))
        return cur.fetchall()


def get_thesis_playbook_actions(thesis_id: int):
    """Return playbook_actions that reference this thesis, with playbook date."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT pa.id, pa.playbook_id, pa.ticker, pa.action, pa.thesis_id,
                   pa.reasoning, pa.confidence, pa.intent_type,
                   pa.intent_magnitude, pa.priority, pa.created_at,
                   p.date AS playbook_date
            FROM playbook_actions pa
            JOIN playbooks p ON p.id = pa.playbook_id
            WHERE pa.thesis_id = %s
            ORDER BY p.date DESC, pa.priority ASC NULLS LAST
        """, (thesis_id,))
        return cur.fetchall()
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard_queries.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add thesis detail queries"
```

---

### Task 4: Decision detail queries

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `tests/test_dashboard_queries.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_dashboard_queries.py`:

```python
class TestGetDecision:
    def test_returns_decision_row(self, cur):
        from dashboard.queries import get_decision
        cur.fetchone.return_value = make_decision_row(id=7)
        assert get_decision(7)["id"] == 7

    def test_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_decision
        cur.fetchone.return_value = None
        assert get_decision(999) is None


class TestGetDecisionSignalsFull:
    def test_returns_denormalized_signals(self, cur):
        from dashboard.queries import get_decision_signals_full
        cur.fetchall.return_value = [
            {
                "signal_type": "news_signal",
                "signal_id": 100,
                "news_headline": "AAPL beats Q3",
                "news_summary": "Apple beat estimates",
                "news_category": "earnings",
                "news_sentiment": "bullish",
                "news_confidence": "high",
                "news_published_at": datetime(2026, 5, 11, 10, 0),
                "news_ticker": "AAPL",
                "macro_headline": None,
                "macro_category": None,
                "macro_affected_sectors": None,
                "macro_sentiment": None,
                "macro_published_at": None,
                "thesis_text": None,
                "thesis_ticker": None,
                "thesis_direction": None,
                "thesis_status": None,
            },
        ]
        result = get_decision_signals_full(7)
        assert len(result) == 1
        assert result[0]["signal_type"] == "news_signal"
        assert result[0]["news_headline"] == "AAPL beats Q3"

    def test_returns_empty_when_no_signals(self, cur):
        from dashboard.queries import get_decision_signals_full
        cur.fetchall.return_value = []
        assert get_decision_signals_full(7) == []


class TestGetDecisionTweets:
    def test_returns_tweets_for_decision(self, cur):
        from dashboard.queries import get_decision_tweets
        cur.fetchall.return_value = [make_tweet_row(id=1)]
        result = get_decision_tweets(7)
        assert len(result) == 1
        assert 7 in cur.execute.call_args[0][1]


class TestGetPlaybookAction:
    def test_returns_action_with_thesis_join(self, cur):
        from dashboard.queries import get_playbook_action
        cur.fetchone.return_value = make_playbook_action_row(id=3)
        assert get_playbook_action(3)["id"] == 3

    def test_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_playbook_action
        cur.fetchone.return_value = None
        assert get_playbook_action(999) is None
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestGetDecision tests/test_dashboard_queries.py::TestGetDecisionSignalsFull tests/test_dashboard_queries.py::TestGetDecisionTweets tests/test_dashboard_queries.py::TestGetPlaybookAction -v`
Expected: ImportError on each new function.

- [ ] **Step 3: Implement queries**

Append to `dashboard/queries.py`:

```python
def get_decision(decision_id: int):
    """Return one decision row, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   signals_used, account_equity, buying_power,
                   outcome_7d, outcome_30d, is_off_playbook, playbook_action_id
            FROM decisions
            WHERE id = %s
        """, (decision_id,))
        return cur.fetchone()


def get_decision_signals_full(decision_id: int):
    """Return decision_signals rows denormalized with the full signal record.

    Result rows always contain all three signal blocks; only the matching
    one is populated for any given row. Template renders the populated one.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT ds.signal_type, ds.signal_id,
                   ns.headline   AS news_headline,
                   ns.summary    AS news_summary,
                   ns.category   AS news_category,
                   ns.sentiment  AS news_sentiment,
                   ns.confidence AS news_confidence,
                   ns.published_at AS news_published_at,
                   ns.ticker     AS news_ticker,
                   ms.headline   AS macro_headline,
                   ms.category   AS macro_category,
                   ms.affected_sectors AS macro_affected_sectors,
                   ms.sentiment  AS macro_sentiment,
                   ms.published_at AS macro_published_at,
                   t.thesis      AS thesis_text,
                   t.ticker      AS thesis_ticker,
                   t.direction   AS thesis_direction,
                   t.status      AS thesis_status
            FROM decision_signals ds
            LEFT JOIN news_signals  ns ON ds.signal_type = 'news_signal'  AND ds.signal_id = ns.id
            LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ds.signal_id = ms.id
            LEFT JOIN theses        t  ON ds.signal_type = 'thesis'       AND ds.signal_id = t.id
            WHERE ds.decision_id = %s
            ORDER BY ds.signal_type, ds.signal_id
        """, (decision_id,))
        return cur.fetchall()


def get_decision_tweets(decision_id: int):
    """Return tweets posted for a given decision_id."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, session_date, tweet_type, tweet_text, platform,
                   posted, error, created_at
            FROM tweets
            WHERE decision_id = %s
            ORDER BY created_at DESC
        """, (decision_id,))
        return cur.fetchall()


def get_playbook_action(action_id: int):
    """Return one playbook_action joined with its thesis info, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT pa.id, pa.playbook_id, pa.ticker, pa.action, pa.thesis_id,
                   pa.reasoning, pa.confidence, pa.intent_type,
                   pa.intent_magnitude, pa.priority, pa.created_at,
                   t.thesis    AS thesis_text,
                   t.direction AS thesis_direction,
                   t.status    AS thesis_status,
                   p.date      AS playbook_date
            FROM playbook_actions pa
            LEFT JOIN theses    t ON t.id = pa.thesis_id
            LEFT JOIN playbooks p ON p.id = pa.playbook_id
            WHERE pa.id = %s
        """, (action_id,))
        return cur.fetchone()
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard_queries.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add decision detail queries"
```

---

### Task 5: Ticker overview queries

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `tests/test_dashboard_queries.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_dashboard_queries.py`:

```python
class TestTickerQueries:
    def test_get_ticker_position_returns_row_when_held(self, cur):
        from dashboard.queries import get_ticker_position
        cur.fetchone.return_value = make_position_row(ticker="AAPL")
        assert get_ticker_position("AAPL")["ticker"] == "AAPL"

    def test_get_ticker_position_returns_none_when_not_held(self, cur):
        from dashboard.queries import get_ticker_position
        cur.fetchone.return_value = None
        assert get_ticker_position("XYZ") is None

    def test_get_ticker_theses_returns_all_statuses(self, cur):
        from dashboard.queries import get_ticker_theses
        cur.fetchall.return_value = [
            make_thesis_row(id=1, ticker="AAPL", status="active"),
            make_thesis_row(id=2, ticker="AAPL", status="expired"),
        ]
        result = get_ticker_theses("AAPL")
        assert len(result) == 2

    def test_get_ticker_decisions_filters_by_days(self, cur):
        from dashboard.queries import get_ticker_decisions
        cur.fetchall.return_value = [make_decision_row(id=1, ticker="AAPL")]
        result = get_ticker_decisions("AAPL", days=90)
        assert len(result) == 1
        assert "AAPL" in cur.execute.call_args[0][1]

    def test_get_ticker_signals_returns_news(self, cur):
        from dashboard.queries import get_ticker_signals
        cur.fetchall.return_value = [make_news_signal_row(ticker="AAPL")]
        assert get_ticker_signals("AAPL")[0]["ticker"] == "AAPL"

    def test_get_ticker_open_orders(self, cur):
        from dashboard.queries import get_ticker_open_orders
        cur.fetchall.return_value = [make_open_order_row(ticker="AAPL")]
        assert len(get_ticker_open_orders("AAPL")) == 1

    def test_get_ticker_attribution_groups_by_category(self, cur):
        from dashboard.queries import get_ticker_attribution
        cur.fetchall.return_value = [
            {"signal_type": "news_signal", "category": "earnings",
             "sample_size": 3, "avg_outcome_7d": Decimal("1.5"),
             "avg_outcome_30d": Decimal("3.0")},
        ]
        result = get_ticker_attribution("AAPL", days=90)
        assert result[0]["category"] == "earnings"
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestTickerQueries -v`
Expected: ImportError on each ticker function.

- [ ] **Step 3: Implement queries**

Append to `dashboard/queries.py`:

```python
def get_ticker_position(sym: str):
    """Return one position row for this ticker, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT ticker, shares, avg_cost, updated_at
            FROM positions
            WHERE ticker = %s
        """, (sym,))
        return cur.fetchone()


def get_ticker_theses(sym: str):
    """Return all theses (any status) for this ticker, newest first."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, source, status,
                   created_at, updated_at, closed_at, close_reason
            FROM theses
            WHERE ticker = %s
            ORDER BY created_at DESC
        """, (sym,))
        return cur.fetchall()


def get_ticker_decisions(sym: str, days: int = 90, limit: int = 50):
    """Return recent decisions for this ticker."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   account_equity, outcome_7d, outcome_30d,
                   is_off_playbook, playbook_action_id
            FROM decisions
            WHERE ticker = %s
              AND date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC, id DESC
            LIMIT %s
        """, (sym, days, limit))
        return cur.fetchall()


def get_ticker_signals(sym: str, days: int = 30, limit: int = 50):
    """Return recent news signals for this ticker."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, ticker, headline, summary, category, sentiment,
                   confidence, published_at
            FROM news_signals
            WHERE ticker = %s
              AND published_at > NOW() - INTERVAL '%s days'
            ORDER BY published_at DESC
            LIMIT %s
        """, (sym, days, limit))
        return cur.fetchall()


def get_ticker_open_orders(sym: str):
    """Return open orders for this ticker."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT order_id, ticker, side, order_type, qty, filled_qty,
                   limit_price, stop_price, status, submitted_at, updated_at
            FROM open_orders
            WHERE ticker = %s
            ORDER BY submitted_at DESC
        """, (sym,))
        return cur.fetchall()


def get_ticker_attribution(sym: str, days: int = 90):
    """Per-category attribution for signals that fed decisions on this ticker.

    Joins decision_signals through to news/macro categories and aggregates
    decision outcomes. Theses are excluded (not a 'category').
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                CASE ds.signal_type
                    WHEN 'news_signal'  THEN 'news:'  || ns.category
                    WHEN 'macro_signal' THEN 'macro:' || ms.category
                END AS category,
                ds.signal_type,
                COUNT(*) AS sample_size,
                AVG(d.outcome_7d)::numeric(8,4)  AS avg_outcome_7d,
                AVG(d.outcome_30d)::numeric(8,4) AS avg_outcome_30d
            FROM decision_signals ds
            JOIN decisions d ON d.id = ds.decision_id
            LEFT JOIN news_signals  ns ON ds.signal_type = 'news_signal'  AND ds.signal_id = ns.id
            LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ds.signal_id = ms.id
            WHERE d.ticker = %s
              AND d.date > CURRENT_DATE - INTERVAL '%s days'
              AND ds.signal_type IN ('news_signal', 'macro_signal')
              AND ds.signal_id IS NOT NULL
            GROUP BY category, ds.signal_type
            ORDER BY sample_size DESC
        """, (sym, days))
        return cur.fetchall()
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard_queries.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add ticker overview queries"
```

---

### Task 6: Session overview queries

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `tests/test_dashboard_queries.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_dashboard_queries.py`:

```python
class TestSessionQueries:
    def test_get_session_returns_row(self, cur):
        from dashboard.queries import get_session
        cur.fetchone.return_value = make_session_row(id=1)
        assert get_session(1)["id"] == 1

    def test_get_session_returns_none_when_not_found(self, cur):
        from dashboard.queries import get_session
        cur.fetchone.return_value = None
        assert get_session(999) is None

    def test_get_session_decisions(self, cur):
        from dashboard.queries import get_session_decisions
        cur.fetchall.return_value = [make_decision_row(id=1)]
        result = get_session_decisions(1)
        assert len(result) == 1
        # Should filter by joining session_date
        called_sql = cur.execute.call_args[0][0]
        assert "sessions" in called_sql or "session_date" in called_sql

    def test_get_session_theses_created(self, cur):
        from dashboard.queries import get_session_theses_created
        cur.fetchall.return_value = [make_thesis_row(id=1)]
        result = get_session_theses_created(1)
        assert len(result) == 1

    def test_get_session_memo_returns_row(self, cur):
        from dashboard.queries import get_session_memo
        cur.fetchone.return_value = make_strategy_memo_row(id=1)
        assert get_session_memo(1)["id"] == 1

    def test_get_session_memo_returns_none(self, cur):
        from dashboard.queries import get_session_memo
        cur.fetchone.return_value = None
        assert get_session_memo(999) is None

    def test_get_session_tweets(self, cur):
        from dashboard.queries import get_session_tweets
        cur.fetchall.return_value = [make_tweet_row(id=1)]
        assert len(get_session_tweets(1)) == 1

    def test_get_session_events_uses_existing_filter(self, cur):
        from dashboard.queries import get_session_events
        cur.fetchall.return_value = [make_agent_event_row(session_id=1)]
        result = get_session_events(1, limit=50)
        assert len(result) == 1
        assert 1 in cur.execute.call_args[0][1]
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestSessionQueries -v`
Expected: ImportError on each session function.

- [ ] **Step 3: Implement queries**

Append to `dashboard/queries.py`:

```python
def get_session(session_id: int):
    """Return one sessions row, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT id, session_date, session_type, status, started_at,
                   completed_at, error
            FROM sessions
            WHERE id = %s
        """, (session_id,))
        return cur.fetchone()


def get_session_decisions(session_id: int):
    """Return decisions made during this session (by date match)."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT d.id, d.date, d.ticker, d.action, d.quantity, d.price,
                   d.reasoning, d.account_equity, d.outcome_7d, d.outcome_30d,
                   d.is_off_playbook, d.playbook_action_id
            FROM decisions d
            JOIN sessions s ON s.session_date = d.date
            WHERE s.id = %s
            ORDER BY d.id ASC
        """, (session_id,))
        return cur.fetchall()


def get_session_theses_created(session_id: int):
    """Return theses created on this session's date."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT t.id, t.ticker, t.direction, t.thesis, t.entry_trigger,
                   t.exit_trigger, t.invalidation, t.confidence, t.source,
                   t.status, t.created_at, t.updated_at
            FROM theses t
            JOIN sessions s ON s.session_date = t.created_at::date
            WHERE s.id = %s
            ORDER BY t.created_at ASC
        """, (session_id,))
        return cur.fetchall()


def get_session_memo(session_id: int):
    """Return the strategy_memos row for this session's date, or None."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT m.id, m.session_date, m.memo_type, m.content, m.created_at
            FROM strategy_memos m
            JOIN sessions s ON s.session_date = m.session_date
            WHERE s.id = %s
            ORDER BY m.created_at DESC
            LIMIT 1
        """, (session_id,))
        return cur.fetchone()


def get_session_tweets(session_id: int):
    """Return tweets posted on this session's date."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT tw.id, tw.session_date, tw.tweet_type, tw.tweet_text,
                   tw.platform, tw.posted, tw.error, tw.created_at,
                   tw.decision_id
            FROM tweets tw
            JOIN sessions s ON s.session_date = tw.session_date
            WHERE s.id = %s
            ORDER BY tw.created_at DESC
        """, (session_id,))
        return cur.fetchall()


def get_session_events(session_id: int, limit: int = 200):
    """Return agent_events filtered to this session (thin wrapper)."""
    return get_recent_agent_events(limit=limit, session_id=session_id)
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard_queries.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add session overview queries"
```

---

### Task 7: Category filter extensions

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `tests/test_dashboard_queries.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_dashboard_queries.py`:

```python
class TestCategoryFilters:
    def test_recent_ticker_signals_filters_by_category(self, cur):
        from dashboard.queries import get_recent_ticker_signals
        cur.fetchall.return_value = []
        get_recent_ticker_signals(days=7, limit=50, category="earnings")
        params = cur.execute.call_args[0][1]
        assert "earnings" in params

    def test_recent_ticker_signals_no_category(self, cur):
        from dashboard.queries import get_recent_ticker_signals
        cur.fetchall.return_value = []
        get_recent_ticker_signals(days=7, limit=50)
        called_sql = cur.execute.call_args[0][0]
        # No category WHERE clause when not filtered
        assert "category = " not in called_sql

    def test_recent_macro_signals_filters_by_category(self, cur):
        from dashboard.queries import get_recent_macro_signals
        cur.fetchall.return_value = []
        get_recent_macro_signals(days=7, limit=20, category="fed")
        params = cur.execute.call_args[0][1]
        assert "fed" in params

    def test_signal_attribution_filters_by_category(self, cur):
        from dashboard.queries import get_signal_attribution
        cur.fetchall.return_value = []
        get_signal_attribution(category="news:earnings")
        params = cur.execute.call_args[0][1]
        assert "news:earnings" in params
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestCategoryFilters -v`
Expected: TypeError (unexpected kwarg) on `category=`.

- [ ] **Step 3: Extend `get_recent_ticker_signals`**

Locate `get_recent_ticker_signals` (around line 70 of `dashboard/queries.py`) and replace:

```python
def get_recent_ticker_signals(days: int = 7, limit: int = 50, category: str | None = None):
    """Fetch recent ticker-specific news signals, optionally filtered by category."""
    where = ["published_at > NOW() - INTERVAL '%s days'"]
    params: list = [days]
    if category:
        where.append("category = %s")
        params.append(category)
    params.append(limit)
    with get_cursor() as cur:
        cur.execute(f"""
            SELECT id, ticker, headline, summary, category, sentiment, confidence,
                   published_at, processed_at
            FROM news_signals
            WHERE {" AND ".join(where)}
            ORDER BY published_at DESC
            LIMIT %s
        """, params)
        return cur.fetchall()
```

- [ ] **Step 4: Extend `get_recent_macro_signals`**

Replace `get_recent_macro_signals`:

```python
def get_recent_macro_signals(days: int = 7, limit: int = 20, category: str | None = None):
    """Fetch recent macro signals, optionally filtered by category."""
    where = ["published_at > NOW() - INTERVAL '%s days'"]
    params: list = [days]
    if category:
        where.append("category = %s")
        params.append(category)
    params.append(limit)
    with get_cursor() as cur:
        cur.execute(f"""
            SELECT id, headline, category, affected_sectors, sentiment, published_at
            FROM macro_signals
            WHERE {" AND ".join(where)}
            ORDER BY published_at DESC
            LIMIT %s
        """, params)
        return cur.fetchall()
```

- [ ] **Step 5: Extend `get_signal_attribution`**

Replace `get_signal_attribution`:

```python
def get_signal_attribution(category: str | None = None):
    """Get latest attribution scores, optionally filtered to one category."""
    where = ""
    params: list = []
    if category:
        where = "WHERE category = %s"
        params.append(category)
    with get_cursor() as cur:
        cur.execute(f"""
            SELECT category, sample_size, avg_outcome_7d, avg_outcome_30d,
                   win_rate_7d, win_rate_30d, updated_at
            FROM signal_attribution
            {where}
            ORDER BY sample_size DESC
        """, params)
        return cur.fetchall()
```

- [ ] **Step 6: Verify all tests pass (including existing)**

Run: `python3 -m pytest tests/test_dashboard_queries.py tests/test_dashboard.py -v`
Expected: all PASS. (Existing `tests/test_dashboard.py` calls these without `category` — kwarg has default so backward-compatible.)

- [ ] **Step 7: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): add optional category filter to signal/attribution queries"
```

---

## Phase 2 — New Routes + Templates

Each route follows: write test asserting 200 + key strings → fail → add route + template → pass → empty-data test → not-found test → commit.

All tests in this phase go into `tests/test_dashboard.py` and use the existing `mock_queries` MagicMock injection — see lines 26-99 of that file for the established pattern.

### Task 8: `/decision/<id>` route + template

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/templates/base.html` (optional: do NOT add this route to nav; detail pages are entered via links, not the nav bar)
- Create: `dashboard/templates/decision_detail.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# Decision detail
# ---------------------------------------------------------------------------


class TestDecisionDetail:
    def test_renders_200_with_decision(self, client):
        mock_queries.get_decision.return_value = make_decision_row(id=5, ticker="AAPL")
        mock_queries.get_decision_signals_full.return_value = []
        mock_queries.get_decision_tweets.return_value = []
        mock_queries.get_playbook_action.return_value = None
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/decision/5")
        assert resp.status_code == 200
        assert b"AAPL" in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_decision.return_value = None
        resp = client.get("/decision/999")
        assert resp.status_code == 404

    def test_renders_signal_refs(self, client):
        mock_queries.get_decision.return_value = make_decision_row(id=5)
        mock_queries.get_decision_signals_full.return_value = [
            {
                "signal_type": "news_signal", "signal_id": 100,
                "news_headline": "Big earnings beat",
                "news_summary": "Apple report", "news_category": "earnings",
                "news_sentiment": "bullish", "news_confidence": "high",
                "news_published_at": datetime(2026, 5, 11, 10, 0),
                "news_ticker": "AAPL",
                "macro_headline": None, "macro_category": None,
                "macro_affected_sectors": None, "macro_sentiment": None,
                "macro_published_at": None,
                "thesis_text": None, "thesis_ticker": None,
                "thesis_direction": None, "thesis_status": None,
            },
        ]
        mock_queries.get_decision_tweets.return_value = []
        mock_queries.get_playbook_action.return_value = None
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/decision/5")
        assert resp.status_code == 200
        assert b"Big earnings beat" in resp.data
        # Anchor target for inbound links from /decisions
        assert b'id="signal-news-100"' in resp.data

    def test_links_to_session_when_session_exists(self, client):
        mock_queries.get_decision.return_value = make_decision_row(id=5)
        mock_queries.get_decision_signals_full.return_value = []
        mock_queries.get_decision_tweets.return_value = []
        mock_queries.get_playbook_action.return_value = None
        mock_queries.lookup_session_id_by_date.return_value = 42
        resp = client.get("/decision/5")
        assert b'href="/session/42"' in resp.data
```

Update the autouse mock setup near the top of the file. Inside `_reset_query_mocks`, add safe defaults (so other tests don't blow up after this phase):

```python
    mock_queries.get_decision.return_value = None
    mock_queries.get_decision_signals_full.return_value = []
    mock_queries.get_decision_tweets.return_value = []
    mock_queries.get_playbook_action.return_value = None
    mock_queries.lookup_session_id_by_date.return_value = None
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py::TestDecisionDetail -v`
Expected: FAIL (route not found, 404).

- [ ] **Step 3: Add route to `dashboard/app.py`**

In `dashboard/app.py`, locate the imports block (around line 14):

```python
from queries import (
```

Add to the imported names:
```python
    get_decision,
    get_decision_signals_full,
    get_decision_tweets,
    get_playbook_action,
    lookup_session_id_by_date,
```

Append the route (after the existing `/decisions` route, around line 148):

```python
@app.route("/decision/<int:decision_id>")
def decision_detail(decision_id):
    """Single decision deep-dive with linked signals, thesis, and session."""
    decision = get_decision(decision_id)
    if not decision:
        abort(404)
    signals = get_decision_signals_full(decision_id)
    tweets = get_decision_tweets(decision_id)
    parent_action = (
        get_playbook_action(decision["playbook_action_id"])
        if decision.get("playbook_action_id")
        else None
    )
    session_id = lookup_session_id_by_date(decision["date"])
    return render_template(
        "decision_detail.html",
        decision=decision,
        signals=signals,
        tweets=tweets,
        parent_action=parent_action,
        session_id=session_id,
    )
```

Add `abort` to the Flask import at the top of the file:
```python
from flask import Flask, abort, jsonify, render_template, request
```

- [ ] **Step 4: Create `dashboard/templates/decision_detail.html`**

```html
{% extends "base.html" %}

{% block title %}Decision #{{ decision.id }} - Alpaca Trading{% endblock %}

{% block content %}
<div class="mb-4">
    <a href="/decisions" class="text-sm text-blue-600 hover:underline">← All decisions</a>
</div>

<h1 class="text-2xl font-bold mb-2">
    Decision #{{ decision.id }} —
    <a href="/ticker/{{ decision.ticker }}" class="text-blue-600 hover:underline">{{ decision.ticker }}</a>
</h1>
<div class="text-sm text-gray-500 mb-6">
    {{ decision.date.strftime('%Y-%m-%d') if decision.date else '' }}
    {% if session_id %}
    · <a href="/session/{{ session_id }}" class="text-blue-600 hover:underline">Session #{{ session_id }}</a>
    {% endif %}
</div>

<!-- Action / outcome summary -->
<div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
    <div class="bg-white rounded-lg shadow p-4">
        <div class="text-gray-500 text-sm">Action</div>
        <div class="text-2xl font-bold {% if decision.action == 'buy' %}text-green-600{% elif decision.action == 'sell' %}text-red-600{% endif %}">
            {{ decision.action|upper }}
        </div>
    </div>
    <div class="bg-white rounded-lg shadow p-4">
        <div class="text-gray-500 text-sm">Quantity</div>
        <div class="text-2xl font-bold">{{ decision.quantity or '-' }}</div>
    </div>
    <div class="bg-white rounded-lg shadow p-4">
        <div class="text-gray-500 text-sm">Price</div>
        <div class="text-2xl font-bold">{{ "$%.2f"|format(decision.price|float) if decision.price else '-' }}</div>
    </div>
    <div class="bg-white rounded-lg shadow p-4">
        <div class="text-gray-500 text-sm">7d Outcome</div>
        <div class="text-2xl font-bold {% if decision.outcome_7d and decision.outcome_7d > 0 %}text-green-600{% elif decision.outcome_7d and decision.outcome_7d < 0 %}text-red-600{% endif %}">
            {{ "%.1f"|format(decision.outcome_7d|float) if decision.outcome_7d is not none else '-' }}%
        </div>
    </div>
    <div class="bg-white rounded-lg shadow p-4">
        <div class="text-gray-500 text-sm">30d Outcome</div>
        <div class="text-2xl font-bold {% if decision.outcome_30d and decision.outcome_30d > 0 %}text-green-600{% elif decision.outcome_30d and decision.outcome_30d < 0 %}text-red-600{% endif %}">
            {{ "%.1f"|format(decision.outcome_30d|float) if decision.outcome_30d is not none else '-' }}%
        </div>
    </div>
</div>

<!-- Reasoning -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Reasoning</h2></div>
    <div class="p-4 whitespace-pre-line text-gray-700">{{ decision.reasoning or 'No reasoning recorded' }}</div>
</div>

<!-- Parent playbook action -->
{% if parent_action %}
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Parent Playbook Action</h2></div>
    <div class="p-4 text-sm">
        <div class="mb-2">
            <span class="font-medium">Action:</span> {{ parent_action.action }}
            · <span class="font-medium">Confidence:</span> {{ parent_action.confidence or '-' }}
        </div>
        <div class="text-gray-600 mb-2">{{ parent_action.reasoning }}</div>
        {% if parent_action.thesis_id %}
        <a href="/thesis/{{ parent_action.thesis_id }}" class="text-blue-600 hover:underline">
            View thesis #{{ parent_action.thesis_id }} →
        </a>
        {% endif %}
    </div>
</div>
{% endif %}

<!-- Source signals -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Source Signals</h2></div>
    <div class="p-4">
        {% if signals %}
        <div class="space-y-3">
            {% for s in signals %}
            {% if s.signal_type == 'news_signal' %}
            <div id="signal-news-{{ s.signal_id }}" class="border-l-4 border-blue-400 pl-3 py-1">
                <div class="text-xs text-blue-700 font-mono mb-0.5">news_signal #{{ s.signal_id }}</div>
                <div class="font-medium">{{ s.news_headline }}</div>
                {% if s.news_summary %}<div class="text-sm text-gray-600 mt-0.5">{{ s.news_summary }}</div>{% endif %}
                <div class="text-xs text-gray-500 mt-1 flex gap-2 flex-wrap">
                    {% if s.news_ticker %}<a href="/ticker/{{ s.news_ticker }}" class="text-blue-600 hover:underline">{{ s.news_ticker }}</a>{% endif %}
                    <a href="/attribution?category=news:{{ s.news_category }}" class="bg-gray-100 px-2 rounded hover:bg-gray-200">{{ s.news_category }}</a>
                    <span class="sentiment-{{ s.news_sentiment }}">{{ s.news_sentiment }}</span>
                    <span>conf: {{ s.news_confidence }}</span>
                    <span>{{ s.news_published_at.strftime('%Y-%m-%d %H:%M') if s.news_published_at else '' }}</span>
                </div>
            </div>
            {% elif s.signal_type == 'macro_signal' %}
            <div id="signal-macro-{{ s.signal_id }}" class="border-l-4 border-purple-400 pl-3 py-1">
                <div class="text-xs text-purple-700 font-mono mb-0.5">macro_signal #{{ s.signal_id }}</div>
                <div class="font-medium">{{ s.macro_headline }}</div>
                <div class="text-xs text-gray-500 mt-1 flex gap-2 flex-wrap">
                    <a href="/attribution?category=macro:{{ s.macro_category }}" class="bg-gray-100 px-2 rounded hover:bg-gray-200">{{ s.macro_category }}</a>
                    <span class="sentiment-{{ s.macro_sentiment }}">{{ s.macro_sentiment }}</span>
                    {% if s.macro_affected_sectors %}<span>Sectors: {{ s.macro_affected_sectors|join(', ') }}</span>{% endif %}
                    <span>{{ s.macro_published_at.strftime('%Y-%m-%d %H:%M') if s.macro_published_at else '' }}</span>
                </div>
            </div>
            {% elif s.signal_type == 'thesis' %}
            <div class="border-l-4 border-green-400 pl-3 py-1">
                <div class="text-xs text-green-700 font-mono mb-0.5">thesis #{{ s.signal_id }}</div>
                <div class="font-medium">
                    <a href="/thesis/{{ s.signal_id }}" class="text-blue-600 hover:underline">{{ s.thesis_text or '(thesis text missing)' }}</a>
                </div>
                <div class="text-xs text-gray-500 mt-1 flex gap-2">
                    {% if s.thesis_ticker %}<a href="/ticker/{{ s.thesis_ticker }}" class="text-blue-600 hover:underline">{{ s.thesis_ticker }}</a>{% endif %}
                    <span>{{ s.thesis_direction or '' }}</span>
                    <span>{{ s.thesis_status or '' }}</span>
                </div>
            </div>
            {% endif %}
            {% endfor %}
        </div>
        {% else %}
        <p class="text-gray-500">No signals linked to this decision.</p>
        {% endif %}
    </div>
</div>

<!-- Tweets from this decision -->
{% if tweets %}
<div class="bg-white rounded-lg shadow">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Tweets</h2></div>
    <div class="divide-y">
        {% for t in tweets %}
        <div class="p-4">
            <div class="text-xs text-gray-500 mb-1">
                {{ t.platform }} · {{ t.session_date }}
                {% if t.posted %}<span class="text-green-700">posted</span>
                {% elif t.error %}<span class="text-red-700">failed</span>
                {% else %}<span class="text-yellow-700">pending</span>
                {% endif %}
            </div>
            <p class="text-gray-700 whitespace-pre-line">{{ t.tweet_text }}</p>
        </div>
        {% endfor %}
    </div>
</div>
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py::TestDecisionDetail -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/templates/decision_detail.html tests/test_dashboard.py
git commit -m "feat(dashboard): /decision/<id> detail page"
```

---

### Task 9: `/thesis/<id>` route + template

**Files:**
- Modify: `dashboard/app.py`
- Create: `dashboard/templates/thesis_detail.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# Thesis detail
# ---------------------------------------------------------------------------


class TestThesisDetail:
    def test_renders_200(self, client):
        mock_queries.get_thesis.return_value = make_thesis_row(id=3, ticker="NVDA")
        mock_queries.get_thesis_decisions.return_value = []
        mock_queries.get_thesis_playbook_actions.return_value = []
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/thesis/3")
        assert resp.status_code == 200
        assert b"NVDA" in resp.data
        # Ticker link present
        assert b'href="/ticker/NVDA"' in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_thesis.return_value = None
        resp = client.get("/thesis/999")
        assert resp.status_code == 404

    def test_shows_linked_decisions(self, client):
        mock_queries.get_thesis.return_value = make_thesis_row(id=3, ticker="NVDA")
        mock_queries.get_thesis_decisions.return_value = [
            make_decision_row(id=10, ticker="NVDA", action="buy"),
        ]
        mock_queries.get_thesis_playbook_actions.return_value = []
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/thesis/3")
        assert b'href="/decision/10"' in resp.data
```

Add safe defaults in `_reset_query_mocks`:

```python
    mock_queries.get_thesis.return_value = None
    mock_queries.get_thesis_decisions.return_value = []
    mock_queries.get_thesis_playbook_actions.return_value = []
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py::TestThesisDetail -v`
Expected: FAIL (404).

- [ ] **Step 3: Add imports + route to `dashboard/app.py`**

Add to the queries import block:
```python
    get_thesis,
    get_thesis_decisions,
    get_thesis_playbook_actions,
```

Append route:

```python
@app.route("/thesis/<int:thesis_id>")
def thesis_detail(thesis_id):
    """Single thesis with linked decisions and playbook actions."""
    thesis = get_thesis(thesis_id)
    if not thesis:
        abort(404)
    decisions = get_thesis_decisions(thesis_id)
    actions = get_thesis_playbook_actions(thesis_id)
    origin_session_id = lookup_session_id_by_date(thesis["created_at"].date()) if thesis.get("created_at") else None
    return render_template(
        "thesis_detail.html",
        thesis=thesis,
        decisions=decisions,
        actions=actions,
        origin_session_id=origin_session_id,
    )
```

- [ ] **Step 4: Create `dashboard/templates/thesis_detail.html`**

```html
{% extends "base.html" %}

{% block title %}Thesis #{{ thesis.id }} - Alpaca Trading{% endblock %}

{% block content %}
<div class="mb-4">
    <a href="/theses" class="text-sm text-blue-600 hover:underline">← All theses</a>
</div>

<h1 class="text-2xl font-bold mb-2">
    Thesis #{{ thesis.id }} —
    <a href="/ticker/{{ thesis.ticker }}" class="text-blue-600 hover:underline">{{ thesis.ticker }}</a>
    <span class="text-base ml-2 align-middle">
        {% if thesis.direction == 'long' %}<span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">long</span>
        {% elif thesis.direction == 'short' %}<span class="bg-red-100 text-red-800 text-xs px-2 py-1 rounded">short</span>
        {% else %}<span class="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded">avoid</span>{% endif %}
        {% if thesis.status == 'active' %}<span class="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">active</span>
        {% elif thesis.status == 'executed' %}<span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">executed</span>
        {% elif thesis.status == 'invalidated' %}<span class="bg-red-100 text-red-800 text-xs px-2 py-1 rounded">invalidated</span>
        {% else %}<span class="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded">{{ thesis.status }}</span>{% endif %}
    </span>
</h1>
<div class="text-sm text-gray-500 mb-6">
    Created: {{ thesis.created_at.strftime('%Y-%m-%d') if thesis.created_at else '-' }}
    {% if origin_session_id %}
    · <a href="/session/{{ origin_session_id }}" class="text-blue-600 hover:underline">Originating session #{{ origin_session_id }}</a>
    {% endif %}
</div>

<!-- Thesis text + triggers -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Thesis</h2></div>
    <div class="p-4 space-y-3 text-sm">
        <div><div class="font-medium text-gray-700">Reasoning</div><div class="text-gray-600">{{ thesis.thesis }}</div></div>
        {% if thesis.entry_trigger %}<div><div class="font-medium text-gray-700">Entry Trigger</div><div class="text-gray-600">{{ thesis.entry_trigger }}</div></div>{% endif %}
        {% if thesis.exit_trigger %}<div><div class="font-medium text-gray-700">Exit Trigger</div><div class="text-gray-600">{{ thesis.exit_trigger }}</div></div>{% endif %}
        {% if thesis.invalidation %}<div><div class="font-medium text-gray-700">Invalidation</div><div class="text-gray-600">{{ thesis.invalidation }}</div></div>{% endif %}
        {% if thesis.close_reason %}<div><div class="font-medium text-gray-700">Close Reason</div><div class="text-gray-600">{{ thesis.close_reason }}</div></div>{% endif %}
    </div>
</div>

<!-- Decisions citing this thesis -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Decisions Citing This Thesis</h2></div>
    <div class="p-4 overflow-x-auto">
        {% if decisions %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Date</th><th class="pb-2">Action</th><th class="pb-2">Qty</th><th class="pb-2">Price</th><th class="pb-2">7d</th><th class="pb-2">30d</th><th class="pb-2"></th>
            </tr></thead>
            <tbody>
            {% for d in decisions %}
            <tr class="border-t">
                <td class="py-2">{{ d.date.strftime('%Y-%m-%d') if d.date else '' }}</td>
                <td class="py-2">{{ d.action|upper }}</td>
                <td class="py-2">{{ d.quantity or '-' }}</td>
                <td class="py-2">{{ "$%.2f"|format(d.price|float) if d.price else '-' }}</td>
                <td class="py-2 {% if d.outcome_7d and d.outcome_7d > 0 %}text-green-600{% elif d.outcome_7d and d.outcome_7d < 0 %}text-red-600{% endif %}">{{ "%.1f"|format(d.outcome_7d|float) if d.outcome_7d is not none else '-' }}%</td>
                <td class="py-2 {% if d.outcome_30d and d.outcome_30d > 0 %}text-green-600{% elif d.outcome_30d and d.outcome_30d < 0 %}text-red-600{% endif %}">{{ "%.1f"|format(d.outcome_30d|float) if d.outcome_30d is not none else '-' }}%</td>
                <td class="py-2"><a href="/decision/{{ d.id }}" class="text-blue-600 hover:underline">view →</a></td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No decisions have cited this thesis yet.</p>
        {% endif %}
    </div>
</div>

<!-- Playbook actions referencing this thesis -->
<div class="bg-white rounded-lg shadow">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Playbook Actions</h2></div>
    <div class="p-4 overflow-x-auto">
        {% if actions %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Playbook Date</th><th class="pb-2">Action</th><th class="pb-2">Confidence</th><th class="pb-2">Reasoning</th>
            </tr></thead>
            <tbody>
            {% for a in actions %}
            <tr class="border-t">
                <td class="py-2">{{ a.playbook_date }}</td>
                <td class="py-2">{{ a.action|upper }}</td>
                <td class="py-2">{{ a.confidence or '-' }}</td>
                <td class="py-2 text-gray-600">{{ a.reasoning }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No playbook actions reference this thesis.</p>
        {% endif %}
    </div>
</div>
{% endblock %}
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py::TestThesisDetail -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/templates/thesis_detail.html tests/test_dashboard.py
git commit -m "feat(dashboard): /thesis/<id> detail page"
```

---

### Task 10: `/ticker/<sym>` route + template

**Files:**
- Modify: `dashboard/app.py`
- Create: `dashboard/templates/ticker.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# Ticker overview
# ---------------------------------------------------------------------------


class TestTickerOverview:
    def test_renders_with_position(self, client):
        mock_queries.get_ticker_position.return_value = make_position_row(ticker="AAPL")
        mock_queries.get_ticker_theses.return_value = []
        mock_queries.get_ticker_decisions.return_value = []
        mock_queries.get_ticker_signals.return_value = []
        mock_queries.get_ticker_open_orders.return_value = []
        mock_queries.get_ticker_attribution.return_value = []
        resp = client.get("/ticker/AAPL")
        assert resp.status_code == 200
        assert b"AAPL" in resp.data

    def test_renders_when_never_traded(self, client):
        """Ticker page renders even for symbols never traded — shows empty state, not 404."""
        mock_queries.get_ticker_position.return_value = None
        mock_queries.get_ticker_theses.return_value = []
        mock_queries.get_ticker_decisions.return_value = []
        mock_queries.get_ticker_signals.return_value = []
        mock_queries.get_ticker_open_orders.return_value = []
        mock_queries.get_ticker_attribution.return_value = []
        resp = client.get("/ticker/ZZZZ")
        assert resp.status_code == 200
        assert b"ZZZZ" in resp.data

    def test_links_to_decisions_and_theses(self, client):
        mock_queries.get_ticker_position.return_value = None
        mock_queries.get_ticker_theses.return_value = [make_thesis_row(id=7, ticker="AAPL")]
        mock_queries.get_ticker_decisions.return_value = [make_decision_row(id=11, ticker="AAPL")]
        mock_queries.get_ticker_signals.return_value = []
        mock_queries.get_ticker_open_orders.return_value = []
        mock_queries.get_ticker_attribution.return_value = []
        resp = client.get("/ticker/AAPL")
        assert b'href="/thesis/7"' in resp.data
        assert b'href="/decision/11"' in resp.data
```

Add safe defaults in `_reset_query_mocks`:

```python
    mock_queries.get_ticker_position.return_value = None
    mock_queries.get_ticker_theses.return_value = []
    mock_queries.get_ticker_decisions.return_value = []
    mock_queries.get_ticker_signals.return_value = []
    mock_queries.get_ticker_open_orders.return_value = []
    mock_queries.get_ticker_attribution.return_value = []
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py::TestTickerOverview -v`
Expected: FAIL (404).

- [ ] **Step 3: Add imports + route to `dashboard/app.py`**

Add to imports:
```python
    get_ticker_attribution,
    get_ticker_decisions,
    get_ticker_open_orders,
    get_ticker_position,
    get_ticker_signals,
    get_ticker_theses,
```

Append route:

```python
@app.route("/ticker/<sym>")
def ticker_overview(sym):
    """Aggregate view of position, theses, decisions, signals for one ticker."""
    sym = sym.upper()
    position = get_ticker_position(sym)
    theses = get_ticker_theses(sym)
    decisions = get_ticker_decisions(sym)
    signals = get_ticker_signals(sym)
    open_orders = get_ticker_open_orders(sym)
    attribution = get_ticker_attribution(sym)
    return render_template(
        "ticker.html",
        sym=sym,
        position=position,
        theses=theses,
        decisions=decisions,
        signals=signals,
        open_orders=open_orders,
        attribution=attribution,
    )
```

- [ ] **Step 4: Create `dashboard/templates/ticker.html`**

```html
{% extends "base.html" %}

{% block title %}{{ sym }} - Alpaca Trading{% endblock %}

{% block content %}
<h1 class="text-2xl font-bold mb-6">{{ sym }}</h1>

<!-- Position -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Position</h2></div>
    <div class="p-4">
        {% if position %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Shares</th><th class="pb-2">Avg Cost</th><th class="pb-2">Total</th><th class="pb-2">Updated</th>
            </tr></thead>
            <tbody><tr class="border-t">
                <td class="py-2 font-semibold">{{ position.shares }}</td>
                <td class="py-2">${{ "%.2f"|format(position.avg_cost|float) }}</td>
                <td class="py-2">${{ "%.2f"|format((position.shares|float * position.avg_cost|float)) }}</td>
                <td class="py-2 text-gray-500">{{ position.updated_at.strftime('%Y-%m-%d %H:%M') if position.updated_at else '-' }}</td>
            </tr></tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No open position.</p>
        {% endif %}
    </div>
</div>

<!-- Open orders -->
{% if open_orders %}
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Open Orders</h2></div>
    <div class="p-4 overflow-x-auto">
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Side</th><th class="pb-2">Type</th><th class="pb-2">Qty</th><th class="pb-2">Filled</th><th class="pb-2">Limit</th><th class="pb-2">Stop</th><th class="pb-2">Status</th><th class="pb-2">Submitted</th>
            </tr></thead>
            <tbody>
            {% for o in open_orders %}
            <tr class="border-t">
                <td class="py-2"><span class="px-2 py-0.5 rounded {% if o.side == 'buy' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">{{ o.side|upper }}</span></td>
                <td class="py-2">{{ o.order_type }}</td>
                <td class="py-2">{{ o.qty }}</td>
                <td class="py-2">{{ o.filled_qty }}</td>
                <td class="py-2">{{ "$%.2f"|format(o.limit_price|float) if o.limit_price else '-' }}</td>
                <td class="py-2">{{ "$%.2f"|format(o.stop_price|float) if o.stop_price else '-' }}</td>
                <td class="py-2">{{ o.status }}</td>
                <td class="py-2 text-gray-500">{{ o.submitted_at.strftime('%Y-%m-%d %H:%M') if o.submitted_at else '-' }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
</div>
{% endif %}

<!-- Theses -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Theses</h2></div>
    <div class="p-4">
        {% if theses %}
        <div class="space-y-3">
            {% for t in theses %}
            <div class="border rounded p-3">
                <div class="flex justify-between items-center mb-1 text-sm">
                    <div>
                        <a href="/thesis/{{ t.id }}" class="text-blue-600 hover:underline font-semibold">#{{ t.id }}</a>
                        <span class="ml-2 px-2 py-0.5 rounded text-xs
                            {% if t.direction == 'long' %}bg-green-100 text-green-800
                            {% elif t.direction == 'short' %}bg-red-100 text-red-800
                            {% else %}bg-gray-100 text-gray-800{% endif %}">{{ t.direction }}</span>
                        <span class="ml-1 px-2 py-0.5 rounded text-xs
                            {% if t.status == 'active' %}bg-blue-100 text-blue-800
                            {% elif t.status == 'executed' %}bg-green-100 text-green-800
                            {% elif t.status == 'invalidated' %}bg-red-100 text-red-800
                            {% else %}bg-gray-100 text-gray-600{% endif %}">{{ t.status }}</span>
                    </div>
                    <span class="text-gray-500 text-xs">{{ t.created_at.strftime('%Y-%m-%d') if t.created_at else '' }}</span>
                </div>
                <div class="text-sm text-gray-700">{{ t.thesis }}</div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <p class="text-gray-500">No theses for this ticker.</p>
        {% endif %}
    </div>
</div>

<!-- Decisions -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Recent Decisions (90 days)</h2></div>
    <div class="p-4 overflow-x-auto">
        {% if decisions %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Date</th><th class="pb-2">Action</th><th class="pb-2">Qty</th><th class="pb-2">Price</th><th class="pb-2">7d</th><th class="pb-2">30d</th><th class="pb-2"></th>
            </tr></thead>
            <tbody>
            {% for d in decisions %}
            <tr class="border-t">
                <td class="py-2">{{ d.date.strftime('%Y-%m-%d') if d.date else '' }}</td>
                <td class="py-2">{{ d.action|upper }}</td>
                <td class="py-2">{{ d.quantity or '-' }}</td>
                <td class="py-2">{{ "$%.2f"|format(d.price|float) if d.price else '-' }}</td>
                <td class="py-2 {% if d.outcome_7d and d.outcome_7d > 0 %}text-green-600{% elif d.outcome_7d and d.outcome_7d < 0 %}text-red-600{% endif %}">{{ "%.1f"|format(d.outcome_7d|float) if d.outcome_7d is not none else '-' }}%</td>
                <td class="py-2 {% if d.outcome_30d and d.outcome_30d > 0 %}text-green-600{% elif d.outcome_30d and d.outcome_30d < 0 %}text-red-600{% endif %}">{{ "%.1f"|format(d.outcome_30d|float) if d.outcome_30d is not none else '-' }}%</td>
                <td class="py-2"><a href="/decision/{{ d.id }}" class="text-blue-600 hover:underline">view →</a></td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No decisions in last 90 days.</p>
        {% endif %}
    </div>
</div>

<!-- Signals -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Recent Signals (30 days)</h2></div>
    <div class="p-4">
        {% if signals %}
        <div class="space-y-2">
            {% for s in signals %}
            <div class="border-l-4 pl-3 py-1 {% if s.sentiment == 'bullish' %}border-green-500 bg-bullish{% elif s.sentiment == 'bearish' %}border-red-500 bg-bearish{% else %}border-gray-400 bg-neutral{% endif %}">
                <div class="text-sm">{{ s.headline }}</div>
                <div class="text-xs text-gray-500 flex gap-2 mt-0.5">
                    <a href="/attribution?category=news:{{ s.category }}" class="bg-gray-100 px-2 rounded hover:bg-gray-200">{{ s.category }}</a>
                    <span class="sentiment-{{ s.sentiment }}">{{ s.sentiment }}</span>
                    <span>{{ s.published_at.strftime('%m/%d %H:%M') if s.published_at else '' }}</span>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <p class="text-gray-500">No recent signals for this ticker.</p>
        {% endif %}
    </div>
</div>

<!-- Attribution by source -->
<div class="bg-white rounded-lg shadow">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Attribution by Source (90 days)</h2></div>
    <div class="p-4 overflow-x-auto">
        {% if attribution %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Category</th><th class="pb-2">Sample</th><th class="pb-2">Avg 7d</th><th class="pb-2">Avg 30d</th>
            </tr></thead>
            <tbody>
            {% for a in attribution %}
            <tr class="border-t">
                <td class="py-2"><a href="/attribution?category={{ a.category }}" class="text-blue-600 hover:underline">{{ a.category }}</a></td>
                <td class="py-2">{{ a.sample_size }}</td>
                <td class="py-2 {% if a.avg_outcome_7d and a.avg_outcome_7d|float > 0 %}text-green-600{% elif a.avg_outcome_7d and a.avg_outcome_7d|float < 0 %}text-red-600{% endif %}">{{ "%.2f%%"|format(a.avg_outcome_7d|float) if a.avg_outcome_7d is not none else '-' }}</td>
                <td class="py-2 {% if a.avg_outcome_30d and a.avg_outcome_30d|float > 0 %}text-green-600{% elif a.avg_outcome_30d and a.avg_outcome_30d|float < 0 %}text-red-600{% endif %}">{{ "%.2f%%"|format(a.avg_outcome_30d|float) if a.avg_outcome_30d is not none else '-' }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No attribution data for this ticker yet.</p>
        {% endif %}
    </div>
</div>
{% endblock %}
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py::TestTickerOverview -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/templates/ticker.html tests/test_dashboard.py
git commit -m "feat(dashboard): /ticker/<sym> overview page"
```

---

### Task 11: `/session/<id>` route + template

**Files:**
- Modify: `dashboard/app.py`
- Create: `dashboard/templates/session_detail.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_dashboard.py`:

```python
from tests.conftest import make_session_row, make_agent_event_row  # noqa: E402

# ---------------------------------------------------------------------------
# Session detail
# ---------------------------------------------------------------------------


class TestSessionDetail:
    def test_renders_200(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = []
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        resp = client.get("/session/42")
        assert resp.status_code == 200
        assert b"Session #42" in resp.data

    def test_404_when_not_found(self, client):
        mock_queries.get_session.return_value = None
        resp = client.get("/session/999")
        assert resp.status_code == 404

    def test_renders_decisions_with_links(self, client):
        mock_queries.get_session.return_value = make_session_row(id=42)
        mock_queries.get_session_stage_costs.return_value = []
        mock_queries.get_session_decisions.return_value = [make_decision_row(id=11, ticker="AAPL")]
        mock_queries.get_session_theses_created.return_value = []
        mock_queries.get_session_memo.return_value = None
        mock_queries.get_session_tweets.return_value = []
        mock_queries.get_session_events.return_value = []
        resp = client.get("/session/42")
        assert b'href="/decision/11"' in resp.data
        assert b'href="/ticker/AAPL"' in resp.data
```

Add safe defaults in `_reset_query_mocks`:

```python
    mock_queries.get_session.return_value = None
    mock_queries.get_session_stage_costs.return_value = []
    mock_queries.get_session_decisions.return_value = []
    mock_queries.get_session_theses_created.return_value = []
    mock_queries.get_session_memo.return_value = None
    mock_queries.get_session_tweets.return_value = []
    mock_queries.get_session_events.return_value = []
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py::TestSessionDetail -v`
Expected: FAIL (404).

- [ ] **Step 3: Add imports + route**

Add to queries import block in `dashboard/app.py`:
```python
    get_session,
    get_session_decisions,
    get_session_events,
    get_session_memo,
    get_session_stage_costs,
    get_session_theses_created,
    get_session_tweets,
```

(`get_session_stage_costs` already exists in `queries.py` from a prior change; if it's not in the import block, add it.)

Append route:

```python
@app.route("/session/<int:session_id>")
def session_detail(session_id):
    """Unified per-session view: stages, events, decisions, theses, tweets, memo."""
    session = get_session(session_id)
    if not session:
        abort(404)
    stages = get_session_stage_costs(session_id)
    decisions = get_session_decisions(session_id)
    theses_created = get_session_theses_created(session_id)
    memo = get_session_memo(session_id)
    tweets = get_session_tweets(session_id)
    events = get_session_events(session_id, limit=200)
    return render_template(
        "session_detail.html",
        session=session,
        stages=stages,
        decisions=decisions,
        theses_created=theses_created,
        memo=memo,
        tweets=tweets,
        events=events,
    )
```

- [ ] **Step 4: Create `dashboard/templates/session_detail.html`**

```html
{% extends "base.html" %}

{% block title %}Session #{{ session.id }} - Alpaca Trading{% endblock %}

{% block content %}
<div class="mb-4">
    <a href="/costs" class="text-sm text-blue-600 hover:underline">← All sessions</a>
</div>

<h1 class="text-2xl font-bold mb-2">Session #{{ session.id }}</h1>
<div class="text-sm text-gray-500 mb-6">
    {{ session.session_date }} · {{ session.session_type }} · {{ session.status }}
    {% if session.started_at %} · started {{ session.started_at.strftime('%H:%M:%S') }}{% endif %}
    {% if session.completed_at %} · completed {{ session.completed_at.strftime('%H:%M:%S') }}{% endif %}
</div>

{% if session.error %}
<div class="bg-red-50 border-l-4 border-red-400 p-4 mb-6">
    <div class="text-red-800 font-medium mb-1">Session Error</div>
    <pre class="text-red-700 text-sm whitespace-pre-wrap">{{ session.error }}</pre>
</div>
{% endif %}

<!-- Stages -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Stages</h2></div>
    <div class="p-4 overflow-x-auto">
        {% if stages %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Stage</th><th class="pb-2">Status</th><th class="pb-2">Model</th><th class="pb-2 text-right">Input</th><th class="pb-2 text-right">Output</th><th class="pb-2 text-right">Cost USD</th>
            </tr></thead>
            <tbody>
            {% for s in stages %}
            <tr class="border-t">
                <td class="py-2 font-mono text-xs">{{ s.stage_name }}</td>
                <td class="py-2">{{ s.status }}</td>
                <td class="py-2 text-xs">{{ s.model or '-' }}</td>
                <td class="py-2 text-right font-mono">{{ '{:,}'.format(s.input_tokens or 0) }}</td>
                <td class="py-2 text-right font-mono">{{ '{:,}'.format(s.output_tokens or 0) }}</td>
                <td class="py-2 text-right font-mono">{% if s.cost_usd is not none %}${{ '%.4f' % s.cost_usd|float }}{% else %}—{% endif %}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No stage cost rows recorded.</p>
        {% endif %}
    </div>
</div>

<!-- Decisions made -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Decisions</h2></div>
    <div class="p-4 overflow-x-auto">
        {% if decisions %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Ticker</th><th class="pb-2">Action</th><th class="pb-2">Qty</th><th class="pb-2">Price</th><th class="pb-2"></th>
            </tr></thead>
            <tbody>
            {% for d in decisions %}
            <tr class="border-t">
                <td class="py-2"><a href="/ticker/{{ d.ticker }}" class="text-blue-600 hover:underline font-semibold">{{ d.ticker }}</a></td>
                <td class="py-2">{{ d.action|upper }}</td>
                <td class="py-2">{{ d.quantity or '-' }}</td>
                <td class="py-2">{{ "$%.2f"|format(d.price|float) if d.price else '-' }}</td>
                <td class="py-2"><a href="/decision/{{ d.id }}" class="text-blue-600 hover:underline">view →</a></td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No decisions in this session.</p>
        {% endif %}
    </div>
</div>

<!-- Theses created -->
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Theses Created</h2></div>
    <div class="p-4">
        {% if theses_created %}
        <div class="space-y-2">
            {% for t in theses_created %}
            <div class="border rounded p-3 text-sm">
                <div class="flex gap-2 items-center mb-1">
                    <a href="/thesis/{{ t.id }}" class="text-blue-600 hover:underline font-semibold">#{{ t.id }}</a>
                    <a href="/ticker/{{ t.ticker }}" class="text-blue-600 hover:underline">{{ t.ticker }}</a>
                    <span class="px-2 py-0.5 rounded text-xs bg-gray-100">{{ t.direction }}</span>
                </div>
                <div class="text-gray-600">{{ t.thesis }}</div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <p class="text-gray-500">No theses created in this session.</p>
        {% endif %}
    </div>
</div>

<!-- Memo -->
{% if memo %}
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Memo</h2></div>
    <div class="p-4 whitespace-pre-line text-gray-700 text-sm">{{ memo.content }}</div>
</div>
{% endif %}

<!-- Tweets -->
{% if tweets %}
<div class="bg-white rounded-lg shadow mb-6">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Tweets</h2></div>
    <div class="divide-y">
        {% for t in tweets %}
        <div class="p-4">
            <div class="text-xs text-gray-500 mb-1">
                {{ t.platform }} · {{ t.tweet_type }}
                {% if t.decision_id %} · <a href="/decision/{{ t.decision_id }}" class="text-blue-600 hover:underline">decision #{{ t.decision_id }}</a>{% endif %}
            </div>
            <p class="text-gray-700 whitespace-pre-line text-sm">{{ t.tweet_text }}</p>
        </div>
        {% endfor %}
    </div>
</div>
{% endif %}

<!-- Events -->
<div class="bg-white rounded-lg shadow">
    <div class="p-4 border-b"><h2 class="text-lg font-semibold">Events ({{ events|length }})</h2></div>
    <div class="p-4 overflow-x-auto">
        {% if events %}
        <table class="w-full text-sm">
            <thead><tr class="text-left text-gray-500">
                <th class="pb-2">Time</th><th class="pb-2">Stage</th><th class="pb-2">Event</th><th class="pb-2">Payload</th>
            </tr></thead>
            <tbody>
            {% for e in events %}
            <tr class="border-t">
                <td class="py-2 align-top text-xs text-gray-500 whitespace-nowrap">{{ e.occurred_at.strftime('%H:%M:%S') if e.occurred_at else '' }}</td>
                <td class="py-2 align-top text-xs">{{ e.stage_name or '—' }}</td>
                <td class="py-2 align-top"><span class="bg-gray-100 px-2 py-0.5 rounded text-xs font-mono">{{ e.event_type }}</span></td>
                <td class="py-2 align-top"><pre class="text-xs text-gray-700 whitespace-pre-wrap break-all max-w-2xl">{{ e.payload|tojson(indent=2) }}</pre></td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="text-gray-500">No agent events for this session.</p>
        {% endif %}
    </div>
</div>
{% endblock %}
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py::TestSessionDetail -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/templates/session_detail.html tests/test_dashboard.py
git commit -m "feat(dashboard): /session/<id> detail page"
```

---

## Phase 3 — Inline Link Pass on Existing Templates

Each task: edit one template, add one regression test asserting the new `href` appears, commit.

### Task 12: portfolio.html — link tickers

**Files:**
- Modify: `dashboard/templates/portfolio.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing regression test**

In `tests/test_dashboard.py`, locate `class TestPortfolioPage` (or wherever portfolio tests are) and add:

```python
    def test_position_ticker_links_to_ticker_page(self, client):
        mock_queries.get_positions.return_value = [make_position_row(ticker="AAPL")]
        mock_queries.get_open_orders.return_value = []
        resp = client.get("/")
        assert b'href="/ticker/AAPL"' in resp.data

    def test_open_order_ticker_links(self, client):
        mock_queries.get_positions.return_value = []
        mock_queries.get_open_orders.return_value = [make_open_order_row(ticker="TSLA")]
        resp = client.get("/")
        assert b'href="/ticker/TSLA"' in resp.data
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py -k "ticker_links" -v`
Expected: FAIL.

- [ ] **Step 3: Edit `dashboard/templates/portfolio.html`**

Replace the position ticker cell (around line 59):

```html
<td class="py-2 font-semibold"><a href="/ticker/{{ pos.ticker }}" class="text-blue-600 hover:underline">{{ pos.ticker }}</a></td>
```

Replace the open-order ticker cell (around line 98):

```html
<td class="py-2 font-semibold"><a href="/ticker/{{ o.ticker }}" class="text-blue-600 hover:underline">{{ o.ticker }}</a></td>
```

Replace watchlist tickers (around line 138):

```html
<span class="bg-gray-100 px-2 py-1 rounded">Watchlist:
    {% for ticker in playbook.watch_list %}
    <a href="/ticker/{{ ticker }}" class="text-blue-600 hover:underline">{{ ticker }}</a>{% if not loop.last %}, {% endif %}
    {% endfor %}
</span>
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/templates/portfolio.html tests/test_dashboard.py
git commit -m "feat(dashboard): link tickers from portfolio page"
```

---

### Task 13: playbook.html — link tickers + thesis detail

**Files:**
- Modify: `dashboard/templates/playbook.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing test**

Append to the playbook test class in `tests/test_dashboard.py`:

```python
    def test_playbook_links_ticker_and_thesis(self, client):
        mock_queries.get_today_playbook.return_value = make_playbook_row()
        mock_queries.get_playbook_actions.return_value = [
            make_playbook_action_row(ticker="NVDA", thesis_id=7),
        ]
        resp = client.get("/playbook")
        assert b'href="/ticker/NVDA"' in resp.data
        assert b'href="/thesis/7"' in resp.data
```

- [ ] **Step 2: Verify test fails**

Run: `python3 -m pytest tests/test_dashboard.py -k "playbook_links" -v`
Expected: FAIL.

- [ ] **Step 3: Edit `dashboard/templates/playbook.html`**

Replace the ticker cell (around line 54):

```html
<td class="py-2 font-semibold"><a href="/ticker/{{ action.ticker }}" class="text-blue-600 hover:underline">{{ action.ticker }}</a></td>
```

Replace the thesis link (around lines 60-67):

```html
<td class="py-2 text-sm">
    {% if action.thesis_id %}
    <a href="/thesis/{{ action.thesis_id }}" class="text-blue-600 hover:underline">
        {{ action.thesis_direction|upper if action.thesis_direction else '' }} #{{ action.thesis_id }}
    </a>
    {% else %}
    <span class="text-gray-400">-</span>
    {% endif %}
</td>
```

Replace watchlist tickers (around lines 105-109):

```html
{% for ticker in playbook.watch_list %}
<a href="/ticker/{{ ticker }}" class="bg-blue-100 text-blue-800 px-3 py-1 rounded font-medium hover:bg-blue-200">{{ ticker }}</a>
{% endfor %}
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/templates/playbook.html tests/test_dashboard.py
git commit -m "feat(dashboard): link tickers and thesis detail from playbook"
```

---

### Task 14: signals.html — link tickers + category drill + category filter

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/templates/signals.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing tests**

Append to the signals test class in `tests/test_dashboard.py`:

```python
    def test_signals_links_tickers_and_categories(self, client):
        mock_queries.get_recent_ticker_signals.return_value = [
            make_news_signal_row(ticker="AAPL", category="earnings"),
        ]
        mock_queries.get_recent_macro_signals.return_value = []
        mock_queries.get_signal_summary.return_value = [
            {"ticker": "AAPL", "bullish": 2, "bearish": 1, "neutral": 0},
        ]
        resp = client.get("/signals")
        assert b'href="/ticker/AAPL"' in resp.data
        assert b'href="/attribution?category=news%3Aearnings"' in resp.data or \
               b'href="/attribution?category=news:earnings"' in resp.data

    def test_signals_category_filter_param(self, client):
        mock_queries.get_recent_ticker_signals.return_value = []
        mock_queries.get_recent_macro_signals.return_value = []
        mock_queries.get_signal_summary.return_value = []
        resp = client.get("/signals?category=earnings")
        assert resp.status_code == 200
        # The category kwarg should be passed to both queries
        mock_queries.get_recent_ticker_signals.assert_called_with(days=7, limit=50, category="earnings")
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py -k "signals" -v`
Expected: FAIL.

- [ ] **Step 3: Update `/signals` route in `dashboard/app.py`**

Find the existing `signals()` route (around line 99) and replace:

```python
@app.route("/signals")
def signals():
    """Market signals page (optionally filtered by category)."""
    category = request.args.get("category") or None
    ticker_signals = get_recent_ticker_signals(days=7, limit=50, category=category)
    macro_signals = get_recent_macro_signals(days=7, limit=20, category=category)
    signal_summary = get_signal_summary(days=7)
    return render_template(
        "signals.html",
        ticker_signals=ticker_signals,
        macro_signals=macro_signals,
        signal_summary=signal_summary,
        current_category=category,
    )
```

- [ ] **Step 4: Edit `dashboard/templates/signals.html`**

Wrap the summary ticker (around line 18):

```html
<div class="font-bold text-lg"><a href="/ticker/{{ summary.ticker }}" class="text-blue-600 hover:underline">{{ summary.ticker }}</a></div>
```

For macro signal categories (around line 45):

```html
<a href="/attribution?category=macro:{{ signal.category }}" class="bg-gray-200 px-2 py-0.5 rounded hover:bg-gray-300">{{ signal.category }}</a>
```

For ticker signal cells (around lines 82, 89):

```html
<td class="py-2 align-top font-semibold"><a href="/ticker/{{ signal.ticker }}" class="text-blue-600 hover:underline">{{ signal.ticker }}</a></td>
```
```html
<td class="py-2 align-top"><a href="/attribution?category=news:{{ signal.category }}" class="bg-gray-100 px-2 py-0.5 rounded text-sm hover:bg-gray-200">{{ signal.category }}</a></td>
```

Add a current-filter banner near the top of `{% block content %}`:

```html
{% if current_category %}
<div class="bg-blue-50 border-l-4 border-blue-400 p-3 mb-4 text-sm">
    Filtered to category <code class="font-mono">{{ current_category }}</code> ·
    <a href="/signals" class="text-blue-600 hover:underline">clear</a>
</div>
{% endif %}
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/templates/signals.html tests/test_dashboard.py
git commit -m "feat(dashboard): link tickers, drill into attribution, accept ?category filter on /signals"
```

---

### Task 15: theses.html — link tickers + detail page

**Files:**
- Modify: `dashboard/templates/theses.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing test**

Append to the theses test class in `tests/test_dashboard.py`:

```python
    def test_theses_link_ticker_and_detail(self, client):
        mock_queries.get_theses.return_value = [make_thesis_row(id=11, ticker="MSFT")]
        resp = client.get("/theses")
        assert b'href="/ticker/MSFT"' in resp.data
        assert b'href="/thesis/11"' in resp.data
```

- [ ] **Step 2: Verify test fails**

Run: `python3 -m pytest tests/test_dashboard.py -k "theses_link" -v`
Expected: FAIL.

- [ ] **Step 3: Edit `dashboard/templates/theses.html`**

Replace the ticker span (around line 77):

```html
<a href="/ticker/{{ thesis.ticker }}" class="font-bold text-lg text-blue-600 hover:underline">{{ thesis.ticker }}</a>
```

Add a "Details →" link to the footer area (within the card, after the source badge around line 154). Replace lines 143-156 with:

```html
<div class="flex items-center gap-2">
    {% if thesis.status == 'active' %}
    <button onclick="openCloseModal({{ thesis.id }}, '{{ thesis.ticker }}', 'invalidated')"
            class="bg-red-100 hover:bg-red-200 text-red-800 text-xs px-2 py-1 rounded">
        Invalidate
    </button>
    <button onclick="openCloseModal({{ thesis.id }}, '{{ thesis.ticker }}', 'expired')"
            class="bg-gray-200 hover:bg-gray-300 text-gray-700 text-xs px-2 py-1 rounded">
        Expire
    </button>
    {% endif %}
    <span class="bg-gray-200 px-2 py-0.5 rounded">{{ thesis.source }}</span>
    <a href="/thesis/{{ thesis.id }}" class="text-blue-600 hover:underline text-xs">Details →</a>
</div>
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/templates/theses.html tests/test_dashboard.py
git commit -m "feat(dashboard): link ticker and detail page from theses cards"
```

---

### Task 16: decisions.html — link tickers + signal refs + detail

**Files:**
- Modify: `dashboard/templates/decisions.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing test**

Append to the decisions test class in `tests/test_dashboard.py`:

```python
    def test_decisions_link_ticker_signal_refs_and_detail(self, client):
        mock_queries.get_recent_decisions.return_value = [
            make_decision_row(id=20, ticker="GOOG", playbook_action_id=5),
        ]
        mock_queries.get_decision_signal_refs_batch.return_value = {
            20: [
                {"signal_type": "news_signal", "signal_id": 100, "label": "Big news"},
                {"signal_type": "thesis", "signal_id": 7, "label": "Strong thesis"},
            ],
        }
        resp = client.get("/decisions")
        assert b'href="/ticker/GOOG"' in resp.data
        assert b'href="/decision/20"' in resp.data
        # news signal links into decision detail anchor
        assert b'href="/decision/20#signal-news-100"' in resp.data
        # thesis ref links directly to thesis detail
        assert b'href="/thesis/7"' in resp.data
```

- [ ] **Step 2: Verify test fails**

Run: `python3 -m pytest tests/test_dashboard.py -k "decisions_link" -v`
Expected: FAIL.

- [ ] **Step 3: Edit `dashboard/templates/decisions.html`**

Replace the ticker cell (around line 67):

```html
<td class="py-2 font-semibold"><a href="/ticker/{{ d.ticker }}" class="text-blue-600 hover:underline">{{ d.ticker }}</a></td>
```

Replace the Source cell (lines 73-81) to link the badge to decision detail:

```html
<td class="py-2">
    {% if d.is_off_playbook %}
    <a href="/decision/{{ d.id }}" class="px-2 py-0.5 rounded text-sm bg-orange-100 text-orange-800 hover:bg-orange-200">Off-Playbook</a>
    {% elif d.playbook_action_id %}
    <a href="/decision/{{ d.id }}" class="px-2 py-0.5 rounded text-sm bg-blue-100 text-blue-800 hover:bg-blue-200">Playbook</a>
    {% else %}
    <a href="/decision/{{ d.id }}" class="text-gray-400 hover:underline">-</a>
    {% endif %}
</td>
```

Replace the signal-ref badge block (lines 96-106) to link each badge to its source:

```html
{% if signal_refs.get(d.id) %}
<div class="mt-1 flex flex-wrap gap-1">
    {% for ref in signal_refs[d.id] %}
    {% if ref.signal_type == 'thesis' %}
    <a href="/thesis/{{ ref.signal_id }}" class="inline-block px-2 py-0.5 rounded text-xs bg-green-50 text-green-700 hover:bg-green-100">{{ ref.label }}</a>
    {% elif ref.signal_type == 'news_signal' %}
    <a href="/decision/{{ d.id }}#signal-news-{{ ref.signal_id }}" class="inline-block px-2 py-0.5 rounded text-xs bg-blue-50 text-blue-700 hover:bg-blue-100">{{ ref.label }}</a>
    {% elif ref.signal_type == 'macro_signal' %}
    <a href="/decision/{{ d.id }}#signal-macro-{{ ref.signal_id }}" class="inline-block px-2 py-0.5 rounded text-xs bg-purple-50 text-purple-700 hover:bg-purple-100">{{ ref.label }}</a>
    {% endif %}
    {% endfor %}
</div>
{% endif %}
```

Add a "View" link in the reasoning row. Replace the entire reasoning `<tr>` block (lines 92-109). Note: the reasoning `<tr>` spans 9 cols; add a detail-link at the end:

```html
<tr class="border-t bg-gray-50">
    <td colspan="9" class="py-2 px-4 text-sm text-gray-600">
        <div class="flex items-start justify-between gap-3">
            <div class="flex-1">
                <strong>Reasoning:</strong> {{ d.reasoning or 'No reasoning provided' }}
                {% if signal_refs.get(d.id) %}
                <div class="mt-1 flex flex-wrap gap-1">
                    {% for ref in signal_refs[d.id] %}
                    {% if ref.signal_type == 'thesis' %}
                    <a href="/thesis/{{ ref.signal_id }}" class="inline-block px-2 py-0.5 rounded text-xs bg-green-50 text-green-700 hover:bg-green-100">{{ ref.label }}</a>
                    {% elif ref.signal_type == 'news_signal' %}
                    <a href="/decision/{{ d.id }}#signal-news-{{ ref.signal_id }}" class="inline-block px-2 py-0.5 rounded text-xs bg-blue-50 text-blue-700 hover:bg-blue-100">{{ ref.label }}</a>
                    {% elif ref.signal_type == 'macro_signal' %}
                    <a href="/decision/{{ d.id }}#signal-macro-{{ ref.signal_id }}" class="inline-block px-2 py-0.5 rounded text-xs bg-purple-50 text-purple-700 hover:bg-purple-100">{{ ref.label }}</a>
                    {% endif %}
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            <a href="/decision/{{ d.id }}" class="text-blue-600 hover:underline text-xs whitespace-nowrap">Details →</a>
        </div>
    </td>
</tr>
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/templates/decisions.html tests/test_dashboard.py
git commit -m "feat(dashboard): link tickers, signal refs, and decision detail from /decisions"
```

---

### Task 17: attribution.html — clickable categories + anchors + filter

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/templates/attribution.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing test**

Append to the attribution test class:

```python
    def test_attribution_categories_link_and_filter(self, client):
        mock_queries.get_signal_attribution.return_value = [
            make_attribution_row(category="news:earnings"),
        ]
        resp = client.get("/attribution")
        # Category cell should be a link that filters the same page
        assert b'href="/attribution?category=news%3Aearnings"' in resp.data or \
               b'href="/attribution?category=news:earnings"' in resp.data
        # Anchor target for cross-page links
        assert b'id="cat-news:earnings"' in resp.data

    def test_attribution_accepts_category_filter(self, client):
        mock_queries.get_signal_attribution.return_value = []
        resp = client.get("/attribution?category=news:earnings")
        assert resp.status_code == 200
        mock_queries.get_signal_attribution.assert_called_with(category="news:earnings")
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py -k "attribution" -v`
Expected: FAIL.

- [ ] **Step 3: Update `/attribution` route in `dashboard/app.py`**

Replace the route (around line 92):

```python
@app.route("/attribution")
def attribution():
    """Signal attribution dashboard, optionally filtered to one category."""
    category = request.args.get("category") or None
    scores = get_signal_attribution(category=category)
    return render_template(
        "attribution.html",
        scores=scores,
        current_category=category,
    )
```

- [ ] **Step 4: Edit `dashboard/templates/attribution.html`**

Add a filter banner near the top of `{% block content %}`:

```html
{% if current_category %}
<div class="bg-blue-50 border-l-4 border-blue-400 p-3 mb-4 text-sm">
    Filtered to <code class="font-mono">{{ current_category }}</code> ·
    <a href="/attribution" class="text-blue-600 hover:underline">clear</a>
</div>
{% endif %}
```

Replace the category cell (around line 29):

```html
<td id="cat-{{ row.category }}" class="py-2 font-semibold">
    <a href="/attribution?category={{ row.category }}" class="text-blue-600 hover:underline">{{ row.category }}</a>
</td>
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/templates/attribution.html tests/test_dashboard.py
git commit -m "feat(dashboard): clickable category cells + ?category filter on /attribution"
```

---

### Task 18: strategy.html — link memo session

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/templates/strategy.html`
- Modify: `tests/test_dashboard.py`

The strategy route currently calls `get_strategy_memos`. We need to enrich each memo with `session_id` looked up by date. Cleanest path: do the lookup in the route handler, attaching a `session_id` field to each memo dict before passing to the template.

- [ ] **Step 1: Add failing test**

Append a new test class in `tests/test_dashboard.py`:

```python
class TestStrategyPage:
    def test_strategy_memo_links_to_session(self, client):
        memo_date = date(2026, 5, 11)
        mock_queries.get_current_strategy.return_value = make_strategy_state_row()
        mock_queries.get_strategy_rules.return_value = []
        mock_queries.get_strategy_memos.return_value = [
            make_strategy_memo_row(session_date=memo_date),
        ]
        mock_queries.lookup_session_id_by_date.return_value = 42
        resp = client.get("/strategy")
        assert b'href="/session/42"' in resp.data

    def test_strategy_memo_without_session(self, client):
        mock_queries.get_current_strategy.return_value = make_strategy_state_row()
        mock_queries.get_strategy_rules.return_value = []
        mock_queries.get_strategy_memos.return_value = [make_strategy_memo_row()]
        mock_queries.lookup_session_id_by_date.return_value = None
        resp = client.get("/strategy")
        assert resp.status_code == 200
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard.py::TestStrategyPage -v`
Expected: FAIL.

- [ ] **Step 3: Update `/strategy` route in `dashboard/app.py`**

Find the route (around line 184) and replace:

```python
@app.route("/strategy")
def strategy():
    """Strategy identity, rules, and recent memos."""
    state = get_current_strategy()
    rules = get_strategy_rules(status='active')
    memos_raw = get_strategy_memos(days=30)
    memos = []
    for m in memos_raw:
        m = dict(m)
        m["session_id"] = lookup_session_id_by_date(m["session_date"])
        memos.append(m)
    return render_template("strategy.html", state=state, rules=rules, memos=memos)
```

- [ ] **Step 4: Edit `dashboard/templates/strategy.html`**

Replace the memo date span (around line 126):

```html
<span class="text-sm font-medium text-gray-500">
    {% if memo.session_id %}
    <a href="/session/{{ memo.session_id }}" class="text-blue-600 hover:underline">{{ memo.session_date }}</a>
    {% else %}
    {{ memo.session_date }}
    {% endif %}
</span>
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/templates/strategy.html tests/test_dashboard.py
git commit -m "feat(dashboard): link memo dates to /session/<id>"
```

---

### Task 19: events.html — link session detail

**Files:**
- Modify: `dashboard/templates/events.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing test**

Append to the events test class (or add a new class if none exists):

```python
class TestEventsPage:
    def test_event_session_links_to_session_detail(self, client):
        mock_queries.get_recent_agent_events.return_value = [make_agent_event_row(session_id=42)]
        mock_queries.get_agent_event_types.return_value = []
        resp = client.get("/events")
        assert b'href="/session/42"' in resp.data
```

- [ ] **Step 2: Verify test fails**

Run: `python3 -m pytest tests/test_dashboard.py::TestEventsPage -v`
Expected: FAIL.

- [ ] **Step 3: Edit `dashboard/templates/events.html`**

Replace the session cell (around lines 61-65):

```html
<td class="py-2 align-top text-xs">
    {% if e.session_id %}
    <a href="/session/{{ e.session_id }}" class="text-blue-600 hover:underline">{{ e.session_id }}</a>
    <a href="/events?session={{ e.session_id }}{% if current_type %}&type={{ current_type }}{% endif %}" class="text-gray-400 hover:text-gray-600 ml-1" title="Filter events to this session">⋯</a>
    {% else %}—{% endif %}
</td>
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/templates/events.html tests/test_dashboard.py
git commit -m "feat(dashboard): link session_id in events to /session/<id>"
```

---

### Task 20: tweets.html — link session + decision

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `dashboard/app.py`
- Modify: `dashboard/templates/tweets.html`
- Modify: `tests/test_dashboard.py`
- Modify: `tests/test_dashboard_queries.py`

The existing `get_recent_tweets` doesn't surface `decision_id` or a joined `session_id`. Extend it.

- [ ] **Step 1: Add failing tests for query extension**

Append to `tests/test_dashboard_queries.py`:

```python
class TestRecentTweetsExtended:
    def test_returns_decision_id_and_session_id(self, cur):
        from dashboard.queries import get_recent_tweets
        cur.fetchall.return_value = [
            {**make_tweet_row(id=1), "decision_id": 11, "session_id": 42},
        ]
        result = get_recent_tweets()
        assert result[0]["decision_id"] == 11
        assert result[0]["session_id"] == 42
```

Also append to `tests/test_dashboard.py`:

```python
    def test_tweets_link_session_and_decision(self, client):
        mock_queries.get_recent_tweets.return_value = [
            {**make_tweet_row(id=1), "decision_id": 11, "session_id": 42},
        ]
        resp = client.get("/tweets")
        assert b'href="/session/42"' in resp.data
        assert b'href="/decision/11"' in resp.data
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m pytest tests/test_dashboard_queries.py::TestRecentTweetsExtended tests/test_dashboard.py -k tweets_link -v`
Expected: FAIL (missing `session_id` column / no link in template).

- [ ] **Step 3: Extend `get_recent_tweets` in `dashboard/queries.py`**

Replace the existing function:

```python
def get_recent_tweets(days=30, limit=50):
    """Fetch recent tweets joined with their session row."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT tw.id, tw.session_date, tw.tweet_type, tw.tweet_text,
                   tw.platform, tw.posted, tw.error, tw.created_at,
                   tw.decision_id,
                   s.id AS session_id
            FROM tweets tw
            LEFT JOIN sessions s ON s.session_date = tw.session_date
                                 AND s.session_type = 'daily'
            WHERE tw.session_date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY tw.session_date DESC, tw.created_at DESC
            LIMIT %s
        """, (days, limit))
        return cur.fetchall()
```

- [ ] **Step 4: Edit `dashboard/templates/tweets.html`**

Replace the metadata line (around lines 13-18):

```html
<div class="flex items-center justify-between mb-2">
    <div class="flex items-center gap-2 text-sm">
        <span class="font-medium text-gray-500">
            {% if tweet.session_id %}
            <a href="/session/{{ tweet.session_id }}" class="text-blue-600 hover:underline">{{ tweet.session_date }}</a>
            {% else %}
            {{ tweet.session_date }}
            {% endif %}
        </span>
        <span class="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">{{ tweet.tweet_type }}</span>
        <span class="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">{{ tweet.platform }}</span>
        {% if tweet.decision_id %}
        <a href="/decision/{{ tweet.decision_id }}" class="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs hover:bg-blue-100">decision #{{ tweet.decision_id }}</a>
        {% endif %}
    </div>
    <div>
        {% if tweet.posted %}<span class="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs font-medium">Posted</span>
        {% elif tweet.error %}<span class="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs font-medium">Failed</span>
        {% else %}<span class="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded text-xs font-medium">Pending</span>
        {% endif %}
    </div>
</div>
```

- [ ] **Step 5: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py tests/test_dashboard_queries.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/queries.py dashboard/templates/tweets.html tests/test_dashboard.py tests/test_dashboard_queries.py
git commit -m "feat(dashboard): link tweets to session and decision detail"
```

---

### Task 21: costs.html — link session detail

**Files:**
- Modify: `dashboard/templates/costs.html`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Add failing test**

Append to the costs test class (create one if absent):

```python
class TestCostsPage:
    def test_session_date_links_to_session_detail(self, client):
        mock_queries.get_recent_session_costs.return_value = [{
            "session_id": 42,
            "session_date": date(2026, 5, 11),
            "session_type": "daily",
            "status": "completed",
            "total_cost_usd": Decimal("0.50"),
            "total_input_tokens": 1000,
            "total_output_tokens": 200,
            "total_cache_creation_tokens": 0,
            "total_cache_read_tokens": 0,
            "started_at": datetime(2026, 5, 11, 10, 0),
            "completed_at": datetime(2026, 5, 11, 10, 5),
        }]
        resp = client.get("/costs")
        assert b'href="/session/42"' in resp.data
```

- [ ] **Step 2: Verify test fails**

Run: `python3 -m pytest tests/test_dashboard.py::TestCostsPage -v`
Expected: FAIL.

- [ ] **Step 3: Edit `dashboard/templates/costs.html`**

Replace the session_date cell (around line 36):

```html
<td class="py-2"><a href="/session/{{ s.session_id }}" class="text-blue-600 hover:underline">{{ s.session_date }}</a></td>
```

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/templates/costs.html tests/test_dashboard.py
git commit -m "feat(dashboard): link session_date to /session/<id> on costs page"
```

---

## Phase 4 — Final verification

### Task 22: Full test sweep + manual smoke

**Files:**
- None (verification only)

- [ ] **Step 1: Full test suite passes**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS, no new failures vs. baseline. 782+ existing tests should remain green, plus the new ones (~30) from this plan.

- [ ] **Step 2: Coverage check (informational)**

Run: `python3 -m pytest tests/test_dashboard.py tests/test_dashboard_queries.py --cov=dashboard --cov-report=term-missing | tail -40`
Expected: `dashboard/app.py` and `dashboard/queries.py` show meaningful coverage. New routes and queries should be covered.

- [ ] **Step 3: Manual smoke — start the dashboard**

Run: `docker compose up -d dashboard db` (or `task up` / `task paper:up` per project conventions)

Open in browser:
- `http://localhost:3000/` → click a position ticker → confirms `/ticker/<sym>` renders
- From `/ticker/<sym>` → click any thesis → confirms `/thesis/<id>` renders
- From `/thesis/<id>` → click any decision → confirms `/decision/<id>` renders
- From `/decision/<id>` → click signal-ref → scrolls to anchor on same page
- From `/decision/<id>` → click session link → confirms `/session/<id>` renders
- From `/strategy` → click a memo date → `/session/<id>`
- From `/tweets` → click session_date → `/session/<id>`
- From `/costs` → click session_date → `/session/<id>`
- From `/attribution` → click a category → filtered view → "clear" works
- From `/signals` add `?category=earnings` to URL → filtered list

- [ ] **Step 4: No-changes commit (just records verification)**

If everything above passes, no further commit is needed. The plan is complete.

If anything is broken, file an issue or open a follow-up task; do not gold-plate.

---

## Implementation Notes

- **Order matters across tasks within a phase** because mock defaults are added incrementally to `_reset_query_mocks`. If executing out of order, copy the relevant defaults from later tasks first.
- **Per the spec, no schema changes.** All linking is composition over existing FKs.
- **Tweets→session join uses `session_type='daily'`**. Premarket tweets posted by `v2.premarket` will likely link to the premarket session of the same date if that exists, but the current query hardcodes `'daily'`. Future enhancement: try `'daily'` first, fall back to most recent same-date session of any type. Out of scope here — note for follow-up if it becomes a real problem.
- **Ticker normalization**: route uppercases `<sym>` before querying. Existing schema stores tickers uppercase, so `/ticker/aapl` and `/ticker/AAPL` both resolve.
- **`session_stage_costs` is a view, not a table** (per `dashboard/queries.py` comment in `get_session_stage_costs`). Don't accidentally treat it as the underlying `session_stages` table.

## Spec coverage check (self-review)

Checking each section of `docs/superpowers/specs/2026-05-11-local-dashboard-linking-design.md`:

- ✅ `/ticker/<sym>` — Task 10
- ✅ `/thesis/<id>` — Task 9
- ✅ `/decision/<id>` — Task 8 (signal anchors at Step 4)
- ✅ `/session/<id>` — Task 11
- ✅ portfolio.html links — Task 12
- ✅ playbook.html links (ticker + thesis-detail upgrade) — Task 13
- ✅ signals.html links + `?category=` filter — Task 14
- ✅ theses.html links — Task 15
- ✅ decisions.html links — Task 16
- ✅ attribution.html clickable + `?category=` — Task 17
- ✅ strategy.html memo→session — Task 18
- ✅ events.html session link — Task 19
- ✅ tweets.html session + decision links — Task 20
- ✅ costs.html session link — Task 21
- ✅ New queries — Tasks 2-7
- ✅ Test coverage including empty/404 — Tasks 8-11
- ✅ Backward-compatible category filter defaults — Task 7 step 6 explicit
- ⊘ `audit.html` ticker links — spec listed as "add ticker link in finding payload when present"; this is conditional on payload schema variance and represents low-value gold-plating. Skipping in this plan; reopen if the payload reliably contains tickers.

No placeholders. No "TBD". Type names consistent across tasks (`get_session`, `get_thesis`, `get_ticker_*` — confirmed identical between query implementation and route imports).
