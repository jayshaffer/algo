# Public Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a public-facing GitHub Pages dashboard showing holdings, performance, and trading activity, updated daily by the session pipeline.

**Architecture:** New Stage 6 in the session pipeline queries the DB and writes JSON files to a local GitHub Pages repo clone, then git pushes. The static site (HTML/CSS/JS) renders the data client-side with Chart.js. Social posts append the dashboard URL. Each trade includes its Alpaca order_id for verification.

**Tech Stack:** Python (json stdlib, subprocess for git), Chart.js (CDN), GitHub Pages hosting.

---

### Task 1: Add order_id to decisions table

The `decisions` table doesn't store Alpaca order IDs. We need this for trade verification on the public dashboard.

**Files:**
- Create: `db/init/010_decision_order_id.sql`
- Modify: `v2/database/trading_db.py:124-136` (insert_decision function)
- Modify: `v2/trader.py:178-279` (execution + logging loops)
- Test: `tests/v2/test_trader.py`

**Step 1: Write the migration**

```sql
-- db/init/010_decision_order_id.sql
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS order_id VARCHAR(50);
```

**Step 2: Write failing test for insert_decision with order_id**

In the existing test file for trading_db, add:

```python
def test_insert_decision_with_order_id(self, mock_db):
    mock_db.fetchone.return_value = {"id": 1}
    result = insert_decision(
        decision_date=date(2026, 2, 17),
        ticker="AAPL",
        action="buy",
        quantity=Decimal("5"),
        price=Decimal("150"),
        reasoning="Test",
        signals_used={},
        account_equity=Decimal("100000"),
        buying_power=Decimal("50000"),
        order_id="abc-123-def",
    )
    assert result == 1
    sql = mock_db.execute.call_args[0][0]
    assert "order_id" in sql
    params = mock_db.execute.call_args[0][1]
    assert "abc-123-def" in params
```

**Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_trading_db.py -k "order_id" -v`
Expected: FAIL — `insert_decision()` got unexpected keyword argument 'order_id'

**Step 4: Update insert_decision to accept order_id**

In `v2/database/trading_db.py`, modify `insert_decision`:

```python
def insert_decision(decision_date, ticker, action, quantity, price, reasoning, signals_used, account_equity, buying_power, playbook_action_id=None, is_off_playbook=False, order_id=None) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO decisions (date, ticker, action, quantity, price, reasoning, signals_used, account_equity, buying_power, playbook_action_id, is_off_playbook, order_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (decision_date, ticker, action, quantity, price, reasoning,
              Json(signals_used) if signals_used else None,
              account_equity, buying_power, playbook_action_id, is_off_playbook, order_id))
        return cur.fetchone()["id"]
```

**Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_trading_db.py -k "order_id" -v`
Expected: PASS

**Step 6: Update trader.py to thread order_id through**

In `v2/trader.py`, collect order_ids during execution (step 5) and pass them when logging (step 6):

Before the step 5 loop, add:
```python
    order_ids = {}  # decision index -> order_id
```

Inside the execution loop, after `result.success`:
```python
            order_ids[i] = result.order_id
```

Change the step 5 loop to use `enumerate`:
```python
    for i, decision in enumerate(response.decisions):
```

In step 6 logging loop, also enumerate and pass order_id:
```python
    for i, decision in enumerate(response.decisions):
        ...
        decision_id = insert_decision(
            ...,
            order_id=order_ids.get(i),
        )
```

**Step 7: Run full test suite to verify nothing breaks**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass

**Step 8: Commit**

```bash
git add db/init/010_decision_order_id.sql v2/database/trading_db.py v2/trader.py tests/
git commit -m "feat: store Alpaca order_id on decisions for verification"
```

---

### Task 2: Create dashboard_publish module — data gathering

The core module that queries the DB and structures data for JSON export.

**Files:**
- Create: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

**Step 1: Write failing test for gather_dashboard_data**

```python
# tests/v2/test_dashboard_publish.py
import pytest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch, call
from v2.dashboard_publish import gather_dashboard_data


class TestGatherDashboardData:
    def test_returns_all_sections(self, mock_db):
        # Set up mock cursor to return data for each query
        mock_db.fetchall.side_effect = [
            # snapshots (90 days)
            [{"date": date(2026, 2, 17), "portfolio_value": Decimal("100000"),
              "cash": Decimal("50000"), "buying_power": Decimal("50000")}],
            # positions
            [{"ticker": "AAPL", "shares": Decimal("10"), "avg_cost": Decimal("150.00"),
              "updated_at": datetime.now()}],
            # decisions (30 days)
            [{"id": 1, "date": date(2026, 2, 17), "ticker": "AAPL", "action": "buy",
              "quantity": Decimal("5"), "price": Decimal("150"), "reasoning": "Test",
              "outcome_7d": Decimal("2.5"), "outcome_30d": None, "order_id": "abc-123"}],
            # theses (active)
            [{"id": 1, "ticker": "AAPL", "direction": "long", "confidence": "high",
              "thesis": "Strong earnings", "entry_trigger": ">150", "exit_trigger": ">180",
              "created_at": datetime.now()}],
        ]
        mock_db.fetchone.side_effect = [
            # latest snapshot
            {"date": date(2026, 2, 17), "portfolio_value": Decimal("100000"),
             "cash": Decimal("50000"), "long_market_value": Decimal("50000")},
            # first snapshot (for total return)
            {"date": date(2026, 1, 1), "portfolio_value": Decimal("95000")},
            # previous snapshot (for daily P&L)
            {"date": date(2026, 2, 16), "portfolio_value": Decimal("99000")},
        ]

        data = gather_dashboard_data(date(2026, 2, 17))

        assert "summary" in data
        assert "snapshots" in data
        assert "positions" in data
        assert "decisions" in data
        assert "theses" in data
        assert data["summary"]["portfolio_value"] == 100000
        assert data["summary"]["last_updated"] == "2026-02-17"

    def test_empty_database(self, mock_db):
        mock_db.fetchall.return_value = []
        mock_db.fetchone.return_value = None

        data = gather_dashboard_data(date(2026, 2, 17))

        assert data["positions"] == []
        assert data["decisions"] == []
        assert data["theses"] == []
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestGatherDashboardData -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement gather_dashboard_data**

```python
# v2/dashboard_publish.py
"""Public dashboard publisher — generates JSON data for GitHub Pages."""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from .database.connection import get_cursor

logger = logging.getLogger("dashboard_publish")


class _DecimalEncoder(json.JSONEncoder):
    """Encode Decimals as floats and dates/datetimes as ISO strings."""
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def gather_dashboard_data(session_date: date) -> dict:
    """Query DB for all data needed by the public dashboard.

    Returns a dict with keys: summary, snapshots, positions, decisions, theses.
    All Decimal values are kept as-is (serialized by _DecimalEncoder).
    """
    data = {
        "summary": {},
        "snapshots": [],
        "positions": [],
        "decisions": [],
        "theses": [],
    }

    with get_cursor() as cur:
        # Snapshots (90 days for equity curve)
        cur.execute("""
            SELECT date, portfolio_value, cash, buying_power
            FROM account_snapshots
            WHERE date > CURRENT_DATE - INTERVAL '90 days'
            ORDER BY date ASC
        """)
        data["snapshots"] = [dict(r) for r in cur.fetchall()]

        # Current positions
        cur.execute("""
            SELECT ticker, shares, avg_cost, updated_at
            FROM positions ORDER BY ticker
        """)
        data["positions"] = [dict(r) for r in cur.fetchall()]

        # Recent decisions (30 days) with order_id for verification
        cur.execute("""
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   outcome_7d, outcome_30d, order_id
            FROM decisions
            WHERE date > CURRENT_DATE - INTERVAL '30 days'
            ORDER BY date DESC, id DESC
        """)
        data["decisions"] = [dict(r) for r in cur.fetchall()]

        # Active theses
        cur.execute("""
            SELECT id, ticker, direction, confidence, thesis,
                   entry_trigger, exit_trigger, created_at
            FROM theses WHERE status = 'active'
            ORDER BY created_at DESC
        """)
        data["theses"] = [dict(r) for r in cur.fetchall()]

        # Summary: latest snapshot + P&L
        cur.execute("""
            SELECT date, portfolio_value, cash, long_market_value
            FROM account_snapshots ORDER BY date DESC LIMIT 1
        """)
        latest = cur.fetchone()

        cur.execute("""
            SELECT date, portfolio_value
            FROM account_snapshots ORDER BY date ASC LIMIT 1
        """)
        first = cur.fetchone()

        cur.execute("""
            SELECT date, portfolio_value
            FROM account_snapshots ORDER BY date DESC LIMIT 1 OFFSET 1
        """)
        prev = cur.fetchone()

        if latest:
            pv = latest["portfolio_value"]
            summary = {
                "portfolio_value": pv,
                "cash": latest["cash"],
                "invested": latest["long_market_value"] or (pv - latest["cash"]),
                "positions_count": len(data["positions"]),
                "last_updated": session_date.isoformat(),
            }
            if prev and prev["portfolio_value"]:
                day_pnl = pv - prev["portfolio_value"]
                summary["daily_pnl"] = day_pnl
                summary["daily_pnl_pct"] = float(day_pnl / prev["portfolio_value"] * 100)
            if first and first["portfolio_value"] and first["date"] != latest["date"]:
                total_pnl = pv - first["portfolio_value"]
                summary["total_pnl"] = total_pnl
                summary["total_pnl_pct"] = float(total_pnl / first["portfolio_value"] * 100)
                summary["inception_date"] = first["date"]
            data["summary"] = summary
        else:
            data["summary"] = {"last_updated": session_date.isoformat()}

    return data
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestGatherDashboardData -v`
Expected: PASS

**Step 5: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat: add dashboard data gathering for public dashboard"
```

---

### Task 3: Dashboard publish module — JSON writing and git push

**Files:**
- Modify: `v2/dashboard_publish.py`
- Test: `tests/v2/test_dashboard_publish.py`

**Step 1: Write failing test for write_json_files**

```python
class TestWriteJsonFiles:
    def test_writes_all_files(self, tmp_path):
        data = {
            "summary": {"portfolio_value": 100000, "last_updated": "2026-02-17"},
            "snapshots": [{"date": "2026-02-17", "portfolio_value": 100000}],
            "positions": [{"ticker": "AAPL", "shares": 10}],
            "decisions": [{"ticker": "AAPL", "action": "buy"}],
            "theses": [{"ticker": "AAPL", "direction": "long"}],
        }

        from v2.dashboard_publish import write_json_files
        write_json_files(data, str(tmp_path))

        data_dir = tmp_path / "data"
        assert (data_dir / "summary.json").exists()
        assert (data_dir / "snapshots.json").exists()
        assert (data_dir / "positions.json").exists()
        assert (data_dir / "decisions.json").exists()
        assert (data_dir / "theses.json").exists()

        import json
        summary = json.loads((data_dir / "summary.json").read_text())
        assert summary["portfolio_value"] == 100000

    def test_creates_data_dir_if_missing(self, tmp_path):
        data = {"summary": {}, "snapshots": [], "positions": [], "decisions": [], "theses": []}
        from v2.dashboard_publish import write_json_files
        write_json_files(data, str(tmp_path))
        assert (tmp_path / "data").is_dir()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestWriteJsonFiles -v`
Expected: FAIL — ImportError

**Step 3: Implement write_json_files**

Add to `v2/dashboard_publish.py`:

```python
def write_json_files(data: dict, repo_path: str) -> list[str]:
    """Write dashboard data as JSON files to the repo's data/ directory.

    Returns list of file paths written.
    """
    data_dir = os.path.join(repo_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    written = []
    for key in ("summary", "snapshots", "positions", "decisions", "theses"):
        file_path = os.path.join(data_dir, f"{key}.json")
        with open(file_path, "w") as f:
            json.dump(data[key], f, cls=_DecimalEncoder, indent=2)
        written.append(file_path)
        logger.info("Wrote %s", file_path)

    return written
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestWriteJsonFiles -v`
Expected: PASS

**Step 5: Write failing test for push_to_github**

```python
class TestPushToGithub:
    @patch("v2.dashboard_publish.subprocess.run")
    def test_commits_and_pushes(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        from v2.dashboard_publish import push_to_github
        push_to_github(str(tmp_path))

        calls = mock_run.call_args_list
        # Should run: git add, git commit, git push
        assert len(calls) == 3
        assert "add" in calls[0].args[0]
        assert "commit" in calls[1].args[0]
        assert "push" in calls[2].args[0]

    @patch("v2.dashboard_publish.subprocess.run")
    def test_skips_push_if_nothing_to_commit(self, mock_run, tmp_path):
        # git commit returns 1 (nothing to commit)
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=1, stderr="nothing to commit"),  # git commit
        ]

        from v2.dashboard_publish import push_to_github
        result = push_to_github(str(tmp_path))

        assert result is False
        # Should NOT call git push
        assert len(mock_run.call_args_list) == 2
```

**Step 6: Implement push_to_github**

Add to `v2/dashboard_publish.py`:

```python
def push_to_github(repo_path: str) -> bool:
    """Git add, commit, and push the data/ directory.

    Returns True if pushed, False if nothing to commit.
    """
    def _run(cmd):
        return subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

    _run(["git", "add", "data/"])

    result = _run(["git", "commit", "-m", f"Update dashboard data {date.today().isoformat()}"])
    if result.returncode != 0:
        logger.info("Nothing to commit: %s", result.stderr.strip())
        return False

    result = _run(["git", "push"])
    if result.returncode != 0:
        logger.error("Git push failed: %s", result.stderr.strip())
        raise RuntimeError(f"Git push failed: {result.stderr.strip()}")

    logger.info("Pushed dashboard data to GitHub Pages")
    return True
```

**Step 7: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add v2/dashboard_publish.py tests/v2/test_dashboard_publish.py
git commit -m "feat: add JSON writing and git push for dashboard publish"
```

---

### Task 4: Dashboard stage orchestrator and session integration

**Files:**
- Modify: `v2/dashboard_publish.py`
- Modify: `v2/session.py`
- Test: `tests/v2/test_dashboard_publish.py`
- Test: `tests/v2/test_session.py`

**Step 1: Write failing test for run_dashboard_stage**

```python
class TestRunDashboardStage:
    @patch("v2.dashboard_publish.push_to_github")
    @patch("v2.dashboard_publish.write_json_files")
    @patch("v2.dashboard_publish.gather_dashboard_data")
    def test_happy_path(self, mock_gather, mock_write, mock_push, mock_db):
        mock_gather.return_value = {"summary": {}, "snapshots": [], "positions": [], "decisions": [], "theses": []}
        mock_write.return_value = ["data/summary.json"]
        mock_push.return_value = True

        from v2.dashboard_publish import run_dashboard_stage
        with patch.dict(os.environ, {"DASHBOARD_REPO_PATH": "/tmp/repo"}):
            result = run_dashboard_stage(date(2026, 2, 17))

        assert result.published is True
        assert result.skipped is False
        assert result.errors == []
        mock_gather.assert_called_once_with(date(2026, 2, 17))
        mock_push.assert_called_once_with("/tmp/repo")

    def test_skipped_when_no_repo_path(self, mock_db):
        from v2.dashboard_publish import run_dashboard_stage
        with patch.dict(os.environ, {}, clear=True):
            # Remove DASHBOARD_REPO_PATH if present
            os.environ.pop("DASHBOARD_REPO_PATH", None)
            result = run_dashboard_stage(date(2026, 2, 17))

        assert result.skipped is True
        assert result.published is False

    @patch("v2.dashboard_publish.gather_dashboard_data")
    def test_handles_gather_error(self, mock_gather, mock_db):
        mock_gather.side_effect = Exception("DB error")

        from v2.dashboard_publish import run_dashboard_stage
        with patch.dict(os.environ, {"DASHBOARD_REPO_PATH": "/tmp/repo"}):
            result = run_dashboard_stage(date(2026, 2, 17))

        assert result.published is False
        assert len(result.errors) == 1
        assert "DB error" in result.errors[0]
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestRunDashboardStage -v`
Expected: FAIL — ImportError for run_dashboard_stage / DashboardStageResult

**Step 3: Implement DashboardStageResult and run_dashboard_stage**

Add to `v2/dashboard_publish.py`:

```python
@dataclass
class DashboardStageResult:
    """Result of the dashboard publishing stage."""
    published: bool = False
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


def run_dashboard_stage(session_date: Optional[date] = None) -> DashboardStageResult:
    """Run the dashboard publish pipeline: gather data -> write JSON -> git push."""
    if session_date is None:
        session_date = date.today()

    result = DashboardStageResult()

    repo_path = os.environ.get("DASHBOARD_REPO_PATH")
    if not repo_path:
        result.skipped = True
        logger.info("Dashboard stage skipped — DASHBOARD_REPO_PATH not set")
        return result

    # Gather data
    try:
        data = gather_dashboard_data(session_date)
    except Exception as e:
        result.errors.append(f"Data gathering failed: {e}")
        logger.error("Failed to gather dashboard data: %s", e)
        return result

    # Write JSON files
    try:
        write_json_files(data, repo_path)
    except Exception as e:
        result.errors.append(f"JSON writing failed: {e}")
        logger.error("Failed to write JSON files: %s", e)
        return result

    # Push to GitHub
    try:
        pushed = push_to_github(repo_path)
        result.published = pushed
    except Exception as e:
        result.errors.append(f"Git push failed: {e}")
        logger.error("Failed to push to GitHub: %s", e)
        return result

    logger.info("Dashboard stage complete: published=%s", result.published)
    return result
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/v2/test_dashboard_publish.py::TestRunDashboardStage -v`
Expected: PASS

**Step 5: Integrate into session.py**

Add to `v2/session.py`:

Import:
```python
from .dashboard_publish import DashboardStageResult, run_dashboard_stage
```

Add to `SessionResult` dataclass:
```python
    dashboard_result: Optional[DashboardStageResult] = None
    dashboard_error: Optional[str] = None
    skipped_dashboard: bool = False
```

Update `has_errors` property to include `self.dashboard_error`.

Add `skip_dashboard: bool = False` parameter to `run_session()`.

Add Stage 6 block after Stage 5b:
```python
    # Stage 6: Dashboard publish
    if skip_dashboard:
        logger.info("[Stage 6] Dashboard publish — SKIPPED")
        result.skipped_dashboard = True
    else:
        logger.info("[Stage 6] Publishing public dashboard")
        try:
            result.dashboard_result = run_dashboard_stage()
        except Exception as e:
            result.dashboard_error = str(e)
            logger.error("Dashboard publish failed: %s", e)
```

Add `--skip-dashboard` to argparse in `main()`.

Thread `skip_dashboard` through `run_session()` call.

Add `"dashboard_error"` to the error summary loop.

**Step 6: Update test_session.py for dashboard stage**

Add a test verifying Stage 6 runs and can be skipped. Follow the existing pattern from the Twitter/Bluesky stage tests.

**Step 7: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass

**Step 8: Commit**

```bash
git add v2/dashboard_publish.py v2/session.py tests/v2/test_dashboard_publish.py tests/v2/test_session.py
git commit -m "feat: add Stage 6 dashboard publishing to session pipeline"
```

---

### Task 5: Append dashboard URL to social posts

**Files:**
- Modify: `v2/twitter.py`
- Modify: `v2/bluesky.py`
- Test: `tests/v2/test_twitter.py`
- Test: `tests/v2/test_bluesky.py`

**Step 1: Write failing test for URL appending in Twitter**

```python
def test_generate_tweet_appends_dashboard_url(self, mock_db, mock_claude_client):
    # Set up mock response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"text": "Ahoy! Great day!"}')]
    mock_claude_client.messages.create.return_value = mock_response

    with patch.dict(os.environ, {"DASHBOARD_URL": "https://example.github.io"}):
        result = generate_tweet("context")

    assert "https://example.github.io" in result["text"]

def test_generate_tweet_no_url_when_not_set(self, mock_db, mock_claude_client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"text": "Ahoy! Great day!"}')]
    mock_claude_client.messages.create.return_value = mock_response

    # Make sure DASHBOARD_URL is not set
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DASHBOARD_URL", None)
        result = generate_tweet("context")

    assert result["text"] == "Ahoy! Great day!"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_twitter.py -k "dashboard_url" -v`
Expected: FAIL — URL not in result

**Step 3: Implement URL appending**

In both `v2/twitter.py` (`generate_tweet`) and `v2/bluesky.py` (`generate_bluesky_post`), after the `return` line at the end of the function, add URL appending logic:

```python
    post_text = result.get("text")
    if not post_text or not isinstance(post_text, str):
        ...
        return None

    # Append dashboard URL if configured
    dashboard_url = os.environ.get("DASHBOARD_URL")
    if dashboard_url:
        post_text = f"{post_text}\n{dashboard_url}"

    return {"text": post_text, "type": "recap"}
```

**Step 4: Write the same test for Bluesky and implement**

Mirror the Twitter test pattern for `generate_bluesky_post`.

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_twitter.py tests/v2/test_bluesky.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add v2/twitter.py v2/bluesky.py tests/v2/test_twitter.py tests/v2/test_bluesky.py
git commit -m "feat: append dashboard URL to social posts"
```

---

### Task 6: Create static site assets

These files live in the GitHub Pages repo (not this codebase). Create them in a `public_dashboard/` directory that can be copied or used to initialize the GitHub Pages repo.

**Files:**
- Create: `public_dashboard/index.html`
- Create: `public_dashboard/styles.css`
- Create: `public_dashboard/app.js`

**Step 1: Create index.html**

Single-page layout with sections: summary header, equity curve canvas, holdings table, decisions table, theses list. Loads Chart.js from CDN. Fetches from `data/*.json`. "Bikini Bottom Capital" header with clean professional styling.

Key elements:
- `<div id="summary">` — portfolio value, cash, daily P&L, total return
- `<canvas id="equity-chart">` — Chart.js line chart
- `<table id="positions">` — holdings table
- `<table id="decisions">` — recent trades with order_id column
- `<div id="theses">` — active theses cards
- `<footer>` — "Last updated" timestamp, "Data from Alpaca" attribution

**Step 2: Create styles.css**

Clean professional layout:
- System font stack
- Responsive table styling
- Subtle nautical accent color (e.g., teal/navy) for headers and accents
- Green/red for positive/negative P&L values
- Mobile-friendly responsive layout

**Step 3: Create app.js**

- `fetchData()` — fetch all JSON files from `data/`
- `renderSummary(data)` — populate summary header
- `renderEquityCurve(snapshots)` — Chart.js line chart
- `renderPositions(positions)` — build holdings table rows
- `renderDecisions(decisions)` — build decisions table with order_id
- `renderTheses(theses)` — build thesis cards
- `formatCurrency(n)` — helper for $1,234.56 formatting
- `formatPct(n)` — helper for +2.50% formatting
- Call everything on `DOMContentLoaded`

**Step 4: Test locally**

Manually verify by placing sample JSON files in `public_dashboard/data/` and opening `index.html` in a browser. (The files need to be served by a local HTTP server due to CORS — `python3 -m http.server 8080`.)

**Step 5: Commit**

```bash
git add public_dashboard/
git commit -m "feat: add static site assets for public dashboard"
```

---

### Task 7: Update conftest and add mock_db patch for dashboard_publish

**Files:**
- Modify: `tests/v2/conftest.py`

**Step 1: Add dashboard_publish to mock_db fixture**

Add `patch("v2.dashboard_publish.get_cursor", _get_cursor)` to the existing `mock_db` fixture's patch stack, following the same pattern as the Twitter/Bluesky patches.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/v2/conftest.py
git commit -m "test: add dashboard_publish to mock_db fixture"
```

---

### Task 8: Final integration test and cleanup

**Step 1: Run full test suite with coverage**

Run: `python3 -m pytest tests/ --cov=trading --cov=dashboard --cov=v2 -q`
Expected: All tests pass, no regressions

**Step 2: Verify the session can run with --skip-dashboard**

Run: `python3 -c "from v2.session import run_session; print('import OK')"`
Expected: "import OK" — verifies no import errors

**Step 3: Final commit if any fixups needed**

```bash
git add -A && git commit -m "chore: final cleanup for public dashboard"
```
