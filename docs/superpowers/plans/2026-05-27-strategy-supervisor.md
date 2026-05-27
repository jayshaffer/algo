# Strategy Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-27-strategy-supervisor-design.md`

**Goal:** Ship an observer-only strategy critic (`python -m v2.supervisor`) that runs an agentic Claude loop with read-only DB tools, persists one free-form markdown memo per run to `supervisor_memos`, and surfaces those memos on the operator dashboard.

**Architecture:** New module `v2/supervisor.py` orchestrates `claude_client.run_agentic_loop` against a tool registry composed of (a) existing read tools re-exported from `v2/tools.py` and (b) ~11 new read-only helpers added alongside them. Mutator overlap is forbidden and enforced by a unit test. Memos persist to a new `supervisor_memos` table (migration 012) and render via two new routes on the v1 operator dashboard (`dashboard/app.py`) using a markdown library added to `v2/requirements.txt`. No cron, no session wiring, no write access.

**Tech Stack:** Python 3.12, Anthropic SDK, psycopg2, PostgreSQL 16, Flask (v1 dashboard), Markdown library (new dep), pytest, Docker Compose, Task v3.

**Dashboard decision:** Spec says "v2 dashboard" but `v2/dashboard/` has no rendered templates today. Operator-facing routes go on the live v1 dashboard (`dashboard/app.py`); if/when v2 cuts over, the routes migrate with the rest. This was confirmed with the operator before plan was written.

**Naming convention:** Claude-facing tool names in the supervisor registry use the spec's `get_*` form (`get_active_rules`, `get_retired_rules`, etc.). Python function names follow the existing project convention `tool_get_*` in `v2/tools.py`. When the spec name differs from an existing tool name (e.g., spec `get_active_rules` vs existing tool `get_strategy_rules`), the supervisor's `TOOL_DEFS` registers the new spec name and points at the existing handler — no duplicate helpers.

---

## File Map

**Create:**
- `db/migrations/012_supervisor_memos.sql` — additive migration for existing databases.
- `db/init/012_supervisor_memos.sql` — same statements, picked up by fresh DB initialization.
- `v2/supervisor.py` — orchestrator: system prompt, tool registry, run_supervisor(), main().
- `tests/v2/test_supervisor.py` — supervisor module unit + integration tests.
- `dashboard/templates/supervisor.html` — list + most-recent-memo page.
- `dashboard/templates/supervisor_detail.html` — permalink page for one memo.

**Modify:**
- `v2/tools.py` — add 11 new `tool_get_*` read helpers + their `TOOL_DEFS` entries to the existing module's bottom section.
- `v2/requirements.txt` — add `markdown>=3.5`.
- `dashboard/queries.py` — add `get_recent_supervisor_memos(limit)` and `get_supervisor_memo(memo_id)`.
- `dashboard/app.py` — add `/supervisor` and `/supervisor/<int:memo_id>` routes.
- `dashboard/templates/base.html` — add internals-nav link to `/supervisor`.
- `Taskfile.yml` — add `supervise` and `supervise:dry-run` targets.
- `tests/v2/test_tools.py` — add unit tests for the new `tool_get_*` helpers (mocked DB).
- `tests/test_dashboard.py` — add tests for the supervisor dashboard routes/queries.
- `CLAUDE.md` — one-line note under v2 modules pointing to the supervisor.

---

## Phase 1: Migration

### Task 1.1: Add `supervisor_memos` table migration

**Files:**
- Create: `db/migrations/012_supervisor_memos.sql`
- Create: `db/init/012_supervisor_memos.sql`

- [ ] **Step 1: Write the migration SQL**

Identical content for both files. The two-file pattern (migrations + init) is the convention used through migration 011.

```sql
-- 012_supervisor_memos.sql
-- Strategy supervisor memos. Observer-only critic output, one row per run.

CREATE TABLE IF NOT EXISTS supervisor_memos (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    content         TEXT,
    status          TEXT NOT NULL CHECK (status IN ('ok', 'max_turns', 'error')),
    turns_used      INT NOT NULL,
    tool_calls      JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_tokens    INT,
    output_tokens   INT,
    cost_usd        NUMERIC(10, 4),
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS supervisor_memos_created_idx
    ON supervisor_memos (created_at DESC);

COMMENT ON TABLE supervisor_memos IS
    'Strategy supervisor critic output. One row per `python -m v2.supervisor` run.';
COMMENT ON COLUMN supervisor_memos.content IS
    'Markdown body. NULL when status != ok.';
COMMENT ON COLUMN supervisor_memos.tool_calls IS
    'Compact per-tool call counts, e.g. [{"name":"get_active_rules","count":2}, ...]';
```

- [ ] **Step 2: Apply the migration**

The project has no migration runner; existing migrations were applied with `psql -f` from a path the operator could reach. Pipe the file in over stdin so no copy step is needed:

```bash
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  < db/migrations/012_supervisor_memos.sql
```

Expected output: `CREATE TABLE` / `CREATE INDEX` / two `COMMENT` lines. No errors.

If the env vars aren't exported on the host, source `.env` first or substitute the literal `pinchy`/`postgres` values used by the running container (see `docker-compose.yml`).

- [ ] **Step 3: Verify the table exists**

```bash
docker compose exec -T db psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "\d supervisor_memos"
```

Expected output: column list matching the migration, with `id` as PK and the `supervisor_memos_created_idx` btree index.

- [ ] **Step 4: Commit**

```bash
git add db/migrations/012_supervisor_memos.sql db/init/012_supervisor_memos.sql
git commit -m "Add supervisor_memos table (migration 012)"
```

---

## Phase 2: Read-only tool helpers

The supervisor's tool registry mixes existing `tool_get_*` helpers from `v2/tools.py` with new ones added in this phase. Each new helper is a plain function with the same shape as the existing ones: uses `get_cursor()` context manager, returns a JSON-serializable dict or list of dicts, and has a `TOOL_DEFS` entry registered for Claude.

Pre-existing tools the supervisor reuses verbatim (no changes needed; tasks below add only the *new* helpers): `get_strategy_identity`, `get_signal_attribution`. The spec name `get_active_rules` maps to the existing `tool_get_strategy_rules` (the supervisor registry will register both names; see Phase 3 Task 3.2).

### Task 2.1: Add rule-history helpers

**Files:**
- Modify: `v2/tools.py` (append helpers + TOOL_DEFS entries at the bottom of the module, alongside the existing `tool_get_strategy_*` group)
- Test: `tests/v2/test_tools.py` (append test cases)

Add: `tool_get_retired_rules(limit=50)`, `tool_get_rule_bind_history(rule_id, days=30)`.

- [ ] **Step 1: Write failing tests**

Append to `tests/v2/test_tools.py`:

```python
def test_get_retired_rules_returns_recent_retirements(mock_db, mock_cursor):
    from v2.tools import tool_get_retired_rules
    mock_cursor.fetchall.return_value = [
        (7, "Avoid earnings-week entries", "risk", "evidence A", "retired",
         "2026-04-15T00:00:00+00:00", "2026-05-01T00:00:00+00:00", "low signal"),
    ]
    result = tool_get_retired_rules(limit=10)
    assert result == [{
        "rule_id": 7,
        "rule_text": "Avoid earnings-week entries",
        "category": "risk",
        "supporting_evidence": "evidence A",
        "status": "retired",
        "created_at": "2026-04-15T00:00:00+00:00",
        "retired_at": "2026-05-01T00:00:00+00:00",
        "retirement_reason": "low signal",
    }]
    args, _ = mock_cursor.execute.call_args
    assert "status = 'retired'" in args[0]
    assert "LIMIT %s" in args[0]
    assert args[1] == (10,)


def test_get_rule_bind_history_counts_citations(mock_db, mock_cursor):
    from v2.tools import tool_get_rule_bind_history
    mock_cursor.fetchall.return_value = [
        (101, "2026-05-20", "AAPL", "buy", "rule #7 supports caution"),
        (115, "2026-05-22", "MSFT", "sell", "lift_condition for #7 met"),
    ]
    result = tool_get_rule_bind_history(rule_id=7, days=30)
    assert result["rule_id"] == 7
    assert result["bind_count"] == 2
    assert len(result["citations"]) == 2
    assert result["citations"][0]["decision_id"] == 101
    args, _ = mock_cursor.execute.call_args
    assert "decisions" in args[0]
    assert "%s" in args[0]  # rule_id param
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py::test_get_retired_rules_returns_recent_retirements tests/v2/test_tools.py::test_get_rule_bind_history_counts_citations -v
```

Expected: ImportError (`cannot import name 'tool_get_retired_rules' from 'v2.tools'`).

- [ ] **Step 3: Implement the helpers**

Append to `v2/tools.py`, in the `tool_get_strategy_*` section:

```python
def tool_get_retired_rules(limit: int = 50) -> list[dict]:
    """Read-only: most recently retired strategy rules."""
    sql = """
        SELECT id, rule_text, category, supporting_evidence, status,
               created_at, retired_at, retirement_reason
        FROM strategy_rules
        WHERE status = 'retired'
        ORDER BY retired_at DESC NULLS LAST
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "rule_id": r[0],
            "rule_text": r[1],
            "category": r[2],
            "supporting_evidence": r[3],
            "status": r[4],
            "created_at": r[5].isoformat() if hasattr(r[5], "isoformat") else r[5],
            "retired_at": r[6].isoformat() if hasattr(r[6], "isoformat") else r[6],
            "retirement_reason": r[7],
        }
        for r in rows
    ]


def tool_get_rule_bind_history(rule_id: int, days: int = 30) -> dict:
    """Read-only: decisions that cited this rule within `days`."""
    sql = """
        SELECT d.id, d.session_date::text, d.ticker, d.action, d.reasoning
        FROM decisions d
        WHERE d.session_date >= (CURRENT_DATE - %s::int)
          AND (d.reasoning ILIKE '%%rule #' || %s::text || '%%'
               OR d.reasoning ILIKE '%%rule_id=' || %s::text || '%%')
        ORDER BY d.session_date ASC
    """
    with get_cursor() as cur:
        cur.execute(sql, (days, rule_id, rule_id))
        rows = cur.fetchall()
    return {
        "rule_id": rule_id,
        "window_days": days,
        "bind_count": len(rows),
        "citations": [
            {
                "decision_id": r[0],
                "date": r[1],
                "ticker": r[2],
                "action": r[3],
                "reasoning_excerpt": (r[4] or "")[:200],
            }
            for r in rows
        ],
    }
```

Note: the bind-history SQL is text-based citation matching because rules are referenced free-form in `decisions.reasoning`. If a structured `decision_rules` join table is introduced later, swap this implementation; the tool contract stays the same.

- [ ] **Step 4: Add TOOL_DEFS entries**

In the `TOOL_DEFS` list near the existing `get_strategy_rules` entry, append:

```python
    {
        "name": "get_retired_rules",
        "description": "Read-only. List the most recently retired strategy rules.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 50}},
        },
    },
    {
        "name": "get_rule_bind_history",
        "description": "Read-only. Decisions that cited a given rule within the window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "integer"},
                "days": {"type": "integer", "default": 30},
            },
            "required": ["rule_id"],
        },
    },
```

And in the `TOOL_HANDLERS` dict (same module):

```python
    "get_retired_rules": tool_get_retired_rules,
    "get_rule_bind_history": tool_get_rule_bind_history,
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py::test_get_retired_rules_returns_recent_retirements tests/v2/test_tools.py::test_get_rule_bind_history_counts_citations -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add v2/tools.py tests/v2/test_tools.py
git commit -m "Add read-only rule-history helpers for supervisor"
```

### Task 2.2: Add thesis read helpers

**Files:**
- Modify: `v2/tools.py`
- Test: `tests/v2/test_tools.py`

Add: `tool_get_theses(status='all', limit=50)`, `tool_get_thesis_lineage(thesis_id)`.

- [ ] **Step 1: Write failing tests**

Append to `tests/v2/test_tools.py`:

```python
def test_get_theses_filters_by_status(mock_db, mock_cursor):
    from v2.tools import tool_get_theses
    mock_cursor.fetchall.return_value = [
        (3, "AAPL", "Long iPhone cycle", "RSI<40", "RSI>70 or earnings",
         "active", "2026-05-01T00:00:00+00:00", None, None),
    ]
    result = tool_get_theses(status="active", limit=25)
    assert len(result) == 1
    assert result[0]["thesis_id"] == 3
    assert result[0]["status"] == "active"
    args, _ = mock_cursor.execute.call_args
    assert "status = %s" in args[0]
    assert args[1] == ("active", 25)


def test_get_theses_all_status_skips_filter(mock_db, mock_cursor):
    from v2.tools import tool_get_theses
    mock_cursor.fetchall.return_value = []
    tool_get_theses(status="all", limit=50)
    args, _ = mock_cursor.execute.call_args
    assert "status = %s" not in args[0]
    assert args[1] == (50,)


def test_get_thesis_lineage_joins_decisions(mock_db, mock_cursor):
    from v2.tools import tool_get_thesis_lineage
    mock_cursor.fetchone.return_value = (3, "AAPL", "Long cycle", "active")
    mock_cursor.fetchall.return_value = [
        (88, "2026-05-10", "buy", 10.0, "thesis #3 entry", 1.5, 3.2),
    ]
    result = tool_get_thesis_lineage(thesis_id=3)
    assert result["thesis_id"] == 3
    assert result["ticker"] == "AAPL"
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["outcome_7d_pct"] == 1.5
    assert result["decisions"][0]["outcome_30d_pct"] == 3.2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py -k "get_theses or get_thesis_lineage" -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the helpers**

Append to `v2/tools.py`:

```python
def tool_get_theses(status: str = "all", limit: int = 50) -> list[dict]:
    """Read-only: theses by status (all|active|closed)."""
    base = """
        SELECT id, ticker, hypothesis, entry_trigger, exit_trigger,
               status, created_at, closed_at, closure_reason
        FROM theses
    """
    params: tuple
    if status == "all":
        sql = base + " ORDER BY created_at DESC LIMIT %s"
        params = (limit,)
    else:
        sql = base + " WHERE status = %s ORDER BY created_at DESC LIMIT %s"
        params = (status, limit)
    with get_cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "thesis_id": r[0],
            "ticker": r[1],
            "hypothesis": r[2],
            "entry_trigger": r[3],
            "exit_trigger": r[4],
            "status": r[5],
            "created_at": r[6].isoformat() if hasattr(r[6], "isoformat") else r[6],
            "closed_at": r[7].isoformat() if hasattr(r[7], "isoformat") else r[7],
            "closure_reason": r[8],
        }
        for r in rows
    ]


def tool_get_thesis_lineage(thesis_id: int) -> dict:
    """Read-only: decisions tagged with this thesis, with outcomes."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, ticker, hypothesis, status FROM theses WHERE id = %s",
            (thesis_id,),
        )
        head = cur.fetchone()
        if head is None:
            return {"thesis_id": thesis_id, "error": "not found"}
        cur.execute(
            """
            SELECT d.id, d.session_date::text, d.action, d.quantity,
                   d.reasoning, d.outcome_7d_pct, d.outcome_30d_pct
            FROM decisions d
            WHERE d.thesis_id = %s
            ORDER BY d.session_date ASC
            """,
            (thesis_id,),
        )
        rows = cur.fetchall()
    return {
        "thesis_id": head[0],
        "ticker": head[1],
        "hypothesis": head[2],
        "status": head[3],
        "decisions": [
            {
                "decision_id": r[0],
                "date": r[1],
                "action": r[2],
                "quantity": float(r[3]) if r[3] is not None else None,
                "reasoning_excerpt": (r[4] or "")[:200],
                "outcome_7d_pct": float(r[5]) if r[5] is not None else None,
                "outcome_30d_pct": float(r[6]) if r[6] is not None else None,
            }
            for r in rows
        ],
    }
```

- [ ] **Step 4: Register in TOOL_DEFS and TOOL_HANDLERS**

```python
    {
        "name": "get_theses",
        "description": "Read-only. Theses filtered by status (all|active|closed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["all", "active", "closed"], "default": "all"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "get_thesis_lineage",
        "description": "Read-only. Decisions tagged with a thesis, with outcomes.",
        "input_schema": {
            "type": "object",
            "properties": {"thesis_id": {"type": "integer"}},
            "required": ["thesis_id"],
        },
    },
```

```python
    "get_theses": tool_get_theses,
    "get_thesis_lineage": tool_get_thesis_lineage,
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py -k "get_theses or get_thesis_lineage" -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add v2/tools.py tests/v2/test_tools.py
git commit -m "Add read-only thesis helpers for supervisor"
```

### Task 2.3: Add executor-behavior helpers

**Files:**
- Modify: `v2/tools.py`
- Test: `tests/v2/test_tools.py`

Add: `tool_get_recent_decisions(days=14)`, `tool_get_decision_detail(decision_id)`, `tool_get_flip_flop_report(days=30, min_reversals=3)`, `tool_get_executor_behavior_summary(days=14)`.

`tool_get_flip_flop_report` should reuse `v2.patterns.analyze_round_trips` rather than reimplement the SQL.

- [ ] **Step 1: Write failing tests**

Append to `tests/v2/test_tools.py`:

```python
def test_get_recent_decisions_returns_compact_rows(mock_db, mock_cursor):
    from v2.tools import tool_get_recent_decisions
    mock_cursor.fetchall.return_value = [
        (501, "2026-05-20", "AAPL", "buy", 10.0,
         "This is a long reasoning that goes on and on and on " * 20,
         "news,thesis", 1.25),
    ]
    result = tool_get_recent_decisions(days=14)
    assert len(result) == 1
    assert result[0]["decision_id"] == 501
    assert len(result[0]["reasoning_excerpt"]) <= 200
    assert result[0]["signals_referenced"] == "news,thesis"


def test_get_decision_detail_includes_referenced_signals(mock_db, mock_cursor):
    from v2.tools import tool_get_decision_detail
    mock_cursor.fetchone.return_value = (
        501, "2026-05-20", "AAPL", "buy", 10.0, "full reasoning text", 1.25,
    )
    mock_cursor.fetchall.return_value = [
        (12, "news_signal", "earnings beat", "positive"),
        (44, "thesis", "long cycle", None),
    ]
    result = tool_get_decision_detail(decision_id=501)
    assert result["decision_id"] == 501
    assert result["reasoning"] == "full reasoning text"
    assert len(result["signals"]) == 2


def test_get_flip_flop_report_uses_round_trip_analyzer(mock_db, mock_cursor, monkeypatch):
    from v2 import tools
    from dataclasses import dataclass
    @dataclass
    class FakeRoundTrip:
        ticker: str
        pairs: int
        first_date: str
        last_date: str
        reasoning_excerpts: list
    monkeypatch.setattr(
        tools, "analyze_round_trips",
        lambda **kw: [FakeRoundTrip("GOOGL", 11, "2026-05-01", "2026-05-22", ["flip A", "flop B"])],
    )
    result = tools.tool_get_flip_flop_report(days=30, min_reversals=3)
    assert len(result) == 1
    assert result[0]["ticker"] == "GOOGL"
    assert result[0]["reversal_count"] == 11


def test_get_executor_behavior_summary_returns_aggregates(mock_db, mock_cursor):
    from v2.tools import tool_get_executor_behavior_summary
    # First fetchall: size histogram rows. Second: sector counts. Third: round-trip count.
    mock_cursor.fetchall.side_effect = [
        [("small", 12), ("medium", 7), ("large", 1)],
        [("Tech", 14), ("Energy", 3), ("Health", 3)],
    ]
    mock_cursor.fetchone.side_effect = [(20,), (4,)]  # decisions count, round-trip count
    result = tool_get_executor_behavior_summary(days=14)
    assert result["window_days"] == 14
    assert result["size_histogram"] == {"small": 12, "medium": 7, "large": 1}
    assert "Tech" in result["sector_concentration"]
    assert result["round_trip_count"] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py -k "get_recent_decisions or get_decision_detail or get_flip_flop_report or get_executor_behavior_summary" -v
```

Expected: ImportError on all four.

- [ ] **Step 3: Implement the helpers**

Append to `v2/tools.py`. The flip-flop helper imports `analyze_round_trips` from `v2.patterns`:

```python
from v2.patterns import analyze_round_trips  # add at top of v2/tools.py if not already there


def tool_get_recent_decisions(days: int = 14) -> list[dict]:
    """Read-only: compact recent decisions for behavior review."""
    sql = """
        SELECT d.id, d.session_date::text, d.ticker, d.action, d.quantity,
               d.reasoning,
               COALESCE(string_agg(DISTINCT ds.signal_type, ','), '') AS signals_referenced,
               d.realized_pnl
        FROM decisions d
        LEFT JOIN decision_signals ds ON ds.decision_id = d.id
        WHERE d.session_date >= (CURRENT_DATE - %s::int)
        GROUP BY d.id
        ORDER BY d.session_date ASC, d.id ASC
    """
    with get_cursor() as cur:
        cur.execute(sql, (days,))
        rows = cur.fetchall()
    return [
        {
            "decision_id": r[0],
            "date": r[1],
            "ticker": r[2],
            "action": r[3],
            "quantity": float(r[4]) if r[4] is not None else None,
            "reasoning_excerpt": (r[5] or "")[:200],
            "signals_referenced": r[6],
            "realized_pnl": float(r[7]) if r[7] is not None else None,
        }
        for r in rows
    ]


def tool_get_decision_detail(decision_id: int) -> dict:
    """Read-only: full decision detail + all referenced signals."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, session_date::text, ticker, action, quantity,
                   reasoning, realized_pnl
            FROM decisions WHERE id = %s
            """,
            (decision_id,),
        )
        head = cur.fetchone()
        if head is None:
            return {"decision_id": decision_id, "error": "not found"}
        cur.execute(
            """
            SELECT ds.signal_id, ds.signal_type, ds.headline, ds.sentiment
            FROM decision_signals ds
            WHERE ds.decision_id = %s
            ORDER BY ds.signal_id
            """,
            (decision_id,),
        )
        sigs = cur.fetchall()
    return {
        "decision_id": head[0],
        "date": head[1],
        "ticker": head[2],
        "action": head[3],
        "quantity": float(head[4]) if head[4] is not None else None,
        "reasoning": head[5],
        "realized_pnl": float(head[6]) if head[6] is not None else None,
        "signals": [
            {"signal_id": s[0], "signal_type": s[1], "headline": s[2], "sentiment": s[3]}
            for s in sigs
        ],
    }


def tool_get_flip_flop_report(days: int = 30, min_reversals: int = 3) -> list[dict]:
    """Read-only: tickers with N+ round-trip reversals in the window."""
    trips = analyze_round_trips(days=days, gap_days=7, min_pairs=min_reversals)
    return [
        {
            "ticker": t.ticker,
            "reversal_count": t.pairs,
            "first_date": str(t.first_date),
            "last_date": str(t.last_date),
            "reasoning_excerpts": [
                (excerpt or "")[:200] for excerpt in (t.reasoning_excerpts or [])
            ][:5],
        }
        for t in trips
    ]


def tool_get_executor_behavior_summary(days: int = 14) -> dict:
    """Read-only: size distribution, sector concentration, round-trip count, hold rate."""
    with get_cursor() as cur:
        # Size histogram. Buckets are coarse on purpose; refine later if needed.
        cur.execute(
            """
            SELECT
                CASE
                    WHEN ABS(quantity * COALESCE(filled_price, 0)) < 500 THEN 'small'
                    WHEN ABS(quantity * COALESCE(filled_price, 0)) < 2500 THEN 'medium'
                    ELSE 'large'
                END AS bucket,
                COUNT(*) AS n
            FROM decisions
            WHERE session_date >= (CURRENT_DATE - %s::int)
            GROUP BY bucket
            """,
            (days,),
        )
        size_rows = cur.fetchall()

        # Sector concentration. Joined via positions for sector mapping.
        cur.execute(
            """
            SELECT COALESCE(p.sector, 'unknown') AS sector, COUNT(*) AS n
            FROM decisions d
            LEFT JOIN positions p ON p.ticker = d.ticker
            WHERE d.session_date >= (CURRENT_DATE - %s::int)
            GROUP BY sector
            ORDER BY n DESC
            """,
            (days,),
        )
        sector_rows = cur.fetchall()

        cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE session_date >= (CURRENT_DATE - %s::int)",
            (days,),
        )
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM decisions WHERE session_date >= (CURRENT_DATE - %s::int) AND action = 'hold'",
            (days,),
        )
        holds = cur.fetchone()[0]

    round_trips = analyze_round_trips(days=days, gap_days=7, min_pairs=2)
    return {
        "window_days": days,
        "decision_count": total or 0,
        "hold_rate": (holds / total) if total else 0.0,
        "size_histogram": {r[0]: r[1] for r in size_rows},
        "sector_concentration": {r[0]: r[1] for r in sector_rows},
        "round_trip_count": sum(t.pairs for t in round_trips),
    }
```

Schema reality-check before pasting the SQL above. Mocked tests don't catch column-name mismatches; verify the columns exist:

```bash
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d decisions" | grep -E "filled_price|quantity"
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d positions" | grep -E "sector"
```

If `decisions.filled_price` doesn't exist, replace `ABS(quantity * COALESCE(filled_price, 0))` with `ABS(COALESCE(quantity, 0))` and switch the bucket thresholds to share counts (e.g. `< 5`, `< 20`, `else large`). If `positions.sector` doesn't exist, drop the join and group decisions by `ticker` instead — sector concentration becomes ticker concentration. Note the substitution in the commit message; the tool contract is unchanged.

- [ ] **Step 4: Register in TOOL_DEFS and TOOL_HANDLERS**

```python
    {
        "name": "get_recent_decisions",
        "description": "Read-only. Compact recent decisions for behavior review.",
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "default": 14}},
        },
    },
    {
        "name": "get_decision_detail",
        "description": "Read-only. Full decision detail plus all referenced signals.",
        "input_schema": {
            "type": "object",
            "properties": {"decision_id": {"type": "integer"}},
            "required": ["decision_id"],
        },
    },
    {
        "name": "get_flip_flop_report",
        "description": "Read-only. Tickers with N+ reversals in the window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 30},
                "min_reversals": {"type": "integer", "default": 3},
            },
        },
    },
    {
        "name": "get_executor_behavior_summary",
        "description": "Read-only. Sizing, sector mix, round trips, hold rate.",
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "default": 14}},
        },
    },
```

```python
    "get_recent_decisions": tool_get_recent_decisions,
    "get_decision_detail": tool_get_decision_detail,
    "get_flip_flop_report": tool_get_flip_flop_report,
    "get_executor_behavior_summary": tool_get_executor_behavior_summary,
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py -k "get_recent_decisions or get_decision_detail or get_flip_flop_report or get_executor_behavior_summary" -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add v2/tools.py tests/v2/test_tools.py
git commit -m "Add read-only executor-behavior helpers for supervisor"
```

### Task 2.4: Add reflection + session metadata helpers

**Files:**
- Modify: `v2/tools.py`
- Test: `tests/v2/test_tools.py`

Add: `tool_get_session_memos(limit=10)`, `tool_get_reflection_actions(limit=10)`, `tool_get_session_summary_window(days=14)`.

(Renamed from spec's `get_session_summary` to `get_session_summary_window` to avoid colliding with the existing `tool_get_session_summary` in `v2/strategy.py`. The supervisor TOOL_DEFS still registers the Claude-facing name as `get_session_summary` in Phase 3.)

- [ ] **Step 1: Write failing tests**

Append to `tests/v2/test_tools.py`:

```python
def test_get_session_memos_returns_recent(mock_db, mock_cursor):
    from v2.tools import tool_get_session_memos
    mock_cursor.fetchall.return_value = [
        (12, "2026-05-20", "reflection", "Memo body A", "2026-05-20T22:00:00+00:00"),
        (11, "2026-05-19", "reflection", "Memo body B", "2026-05-19T22:00:00+00:00"),
    ]
    result = tool_get_session_memos(limit=5)
    assert len(result) == 2
    assert result[0]["memo_id"] == 12
    args, _ = mock_cursor.execute.call_args
    assert args[1] == (5,)


def test_get_reflection_actions_aggregates_per_session(mock_db, mock_cursor):
    from v2.tools import tool_get_reflection_actions
    mock_cursor.fetchall.return_value = [
        # session_id, session_date, proposed, retired, revalidated, identity_updated, memo_words
        (44, "2026-05-20", 2, 1, 0, True, 312),
        (43, "2026-05-19", 0, 0, 0, False, 95),
    ]
    result = tool_get_reflection_actions(limit=10)
    assert len(result) == 2
    assert result[0]["rules_proposed"] == 2
    assert result[0]["identity_updated"] is True
    assert result[1]["memo_word_count"] == 95


def test_get_session_summary_window_aggregates(mock_db, mock_cursor):
    from v2.tools import tool_get_session_summary_window
    mock_cursor.fetchall.return_value = [
        (44, "2026-05-20", 14, 152.40, 0, 0.42),
        (43, "2026-05-19", 11, -33.10, 1, 0.38),
    ]
    result = tool_get_session_summary_window(days=14)
    assert len(result) == 2
    assert result[0]["decisions_count"] == 14
    assert result[0]["stage_failures"] == 0
    assert result[1]["cost_usd"] == 0.38
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py -k "get_session_memos or get_reflection_actions or get_session_summary_window" -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the helpers**

```python
def tool_get_session_memos(limit: int = 10) -> list[dict]:
    """Read-only: most recent strategy memos."""
    sql = """
        SELECT id, session_date::text, memo_type, content, created_at
        FROM strategy_memos
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "memo_id": r[0],
            "session_date": r[1],
            "memo_type": r[2],
            "content": r[3],
            "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else r[4],
        }
        for r in rows
    ]


def tool_get_reflection_actions(limit: int = 10) -> list[dict]:
    """Read-only: per-session reflection-stage actions taken."""
    sql = """
        SELECT
            s.id AS session_id,
            s.session_date::text,
            COALESCE(SUM(CASE WHEN sr.created_at::date = s.session_date::date THEN 1 ELSE 0 END), 0) AS proposed,
            COALESCE(SUM(CASE WHEN sr.retired_at::date = s.session_date::date THEN 1 ELSE 0 END), 0) AS retired,
            COALESCE(SUM(CASE WHEN sr.revalidated_at::date = s.session_date::date THEN 1 ELSE 0 END), 0) AS revalidated,
            EXISTS(
                SELECT 1 FROM strategy_state ss
                WHERE ss.updated_at::date = s.session_date::date
            ) AS identity_updated,
            COALESCE((
                SELECT array_length(regexp_split_to_array(sm.content, '\\s+'), 1)
                FROM strategy_memos sm
                WHERE sm.session_date = s.session_date
                ORDER BY sm.created_at DESC LIMIT 1
            ), 0) AS memo_words
        FROM sessions s
        LEFT JOIN strategy_rules sr ON TRUE
        GROUP BY s.id, s.session_date
        ORDER BY s.session_date DESC
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "session_id": r[0],
            "session_date": r[1],
            "rules_proposed": r[2],
            "rules_retired": r[3],
            "rules_revalidated": r[4],
            "identity_updated": bool(r[5]),
            "memo_word_count": r[6],
        }
        for r in rows
    ]


def tool_get_session_summary_window(days: int = 14) -> list[dict]:
    """Read-only: per-session decisions/P&L/stage failures/cost within a window."""
    sql = """
        SELECT
            s.id,
            s.session_date::text,
            COALESCE((SELECT COUNT(*) FROM decisions d WHERE d.session_date = s.session_date), 0) AS decisions_count,
            COALESCE((SELECT SUM(realized_pnl) FROM decisions d WHERE d.session_date = s.session_date), 0)::numeric AS pnl,
            COALESCE((SELECT COUNT(*) FROM session_stages st
                      WHERE st.session_id = s.id AND st.status = 'failed'), 0) AS failures,
            COALESCE(s.cost_usd, 0)::numeric AS cost
        FROM sessions s
        WHERE s.session_date >= (CURRENT_DATE - %s::int)
        ORDER BY s.session_date DESC
    """
    with get_cursor() as cur:
        cur.execute(sql, (days,))
        rows = cur.fetchall()
    return [
        {
            "session_id": r[0],
            "session_date": r[1],
            "decisions_count": r[2],
            "pnl_usd": float(r[3]),
            "stage_failures": r[4],
            "cost_usd": float(r[5]),
        }
        for r in rows
    ]
```

Schema reality-check (same caveat as Task 2.3: mocks don't catch column mismatch):

```bash
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d sessions" | grep -E "cost_usd|session_date"
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d session_stages" | grep -E "status"
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d strategy_rules" | grep -E "retired_at|revalidated_at|retirement_reason"
```

If `sessions.cost_usd` is missing, set the SELECT expression to `0::numeric`. If `strategy_rules.revalidated_at` is missing, drop that column from the SELECT and return `rules_revalidated = 0`. Note any substitutions in the commit message.

- [ ] **Step 4: Register in TOOL_DEFS and TOOL_HANDLERS**

```python
    {
        "name": "get_session_memos",
        "description": "Read-only. Most recent strategy memos.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}},
    },
    {
        "name": "get_reflection_actions",
        "description": "Read-only. Per-session reflection-stage actions taken.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}},
    },
    {
        "name": "get_session_summary",
        "description": "Read-only. Per-session decisions, P&L, failures, cost within a window.",
        "input_schema": {"type": "object", "properties": {"days": {"type": "integer", "default": 14}}},
    },
```

```python
    "get_session_memos": tool_get_session_memos,
    "get_reflection_actions": tool_get_reflection_actions,
    "get_session_summary": tool_get_session_summary_window,
```

(The Claude-facing name `get_session_summary` deliberately maps to `tool_get_session_summary_window`. The existing per-session `tool_get_session_summary` in `v2/strategy.py` is *not* re-exported via this entry and remains an internal reflection helper.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_tools.py -k "get_session_memos or get_reflection_actions or get_session_summary_window" -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add v2/tools.py tests/v2/test_tools.py
git commit -m "Add read-only reflection + session metadata helpers for supervisor"
```

---

## Phase 3: Supervisor module + CLI

### Task 3.1: Scaffold `v2/supervisor.py` with system prompt and constants

**Files:**
- Create: `v2/supervisor.py`

- [ ] **Step 1: Create the module skeleton (no tests yet — testing happens in 3.3)**

```python
"""Strategy supervisor: observer-only critic of Pinchy's strategy stack.

Runs a one-shot agentic loop with a read-only tool registry and persists
a single markdown memo to `supervisor_memos`. No writes to strategy state.

CLI: `python -m v2.supervisor [--model MODEL] [--max-turns N] [--dry-run]`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from collections import Counter

from v2 import claude_client
from v2 import tools as v2_tools
from v2.db import get_cursor

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1.0.0"
DEFAULT_SUPERVISOR_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TURNS = 20

STRATEGY_SUPERVISOR_SYSTEM = """\
You are the Strategy Supervisor for Pinchy, an agentic trading system.

Your role is to critique the trading strategy from a senior, skeptical
vantage point. You read state — you do not change it. There are no
write tools available to you.

Your four areas of focus:

1. Rule coherence & quality
   - Do active rules contradict each other?
   - Any rule that oscillates (binds/lifts repeatedly within days)?
   - Any active-but-dormant rule that hasn't bound in 30+ days?
   - Any rule churned out within a week of being added?
   - Is each active rule grounded in evidence or pet theory?

2. Thesis discipline
   - Are theses honored at their stated entry/exit triggers?
   - Any thesis lingering past its entry window with no action?
   - Do any active theses contradict each other?
   - Are closed theses being learned from?

3. Identity + behavior drift
   - Is the strategy identity coherent across recent memos, or whipsawing?
   - Does what the executor actually does (sizing, flip-flops, sector mix,
     round-trip frequency) match the identity?

4. Reflection quality
   - Did the recent reflection stages take action, or coast?
   - Did they ignore obvious problems (flip-flops, dormant rules)?
   - Are memos substantive or vacuous?

Investigate before you opine. Use get_* tools to verify any pattern
you suspect — pull bind histories, decision detail, thesis lineage.
Cite specific rule_ids, thesis_ids, decision_ids, and dates in your
critique. A claim without a citation should not appear in the memo.

Be direct. Don't soften. The point of this role is to surface what
the reflection stage missed. If you find nothing wrong, say so plainly —
do not invent concerns to seem thorough. A short "no major concerns
this week, here's why" memo is more valuable than a padded one.

Output: a single markdown memo with sections matching the four areas
above. Skip a section entirely if you have nothing to say about it.
End with a "Watchlist" section: 1-5 specific things to revisit on
the next supervisor run.
"""


# Mutator tool names — must NEVER appear in the supervisor's registered handlers.
# Source of truth for the mutator-overlap defense test in Task 4.1.
STRATEGY_MUTATOR_NAMES: frozenset[str] = frozenset({
    "propose_rule",
    "retire_rule",
    "revalidate_rule",
    "update_strategy_identity",
    "write_strategy_memo",
})


# Claude-facing tool name → Python handler.
# Names are the spec's `get_*` form. Some entries reuse existing tools.py handlers
# under a renamed key (e.g. spec's "get_active_rules" → existing tool_get_strategy_rules).
SUPERVISOR_TOOL_HANDLERS: dict = {
    # Strategy state
    "get_strategy_identity": v2_tools.tool_get_strategy_identity,
    "get_active_rules": v2_tools.tool_get_strategy_rules,
    "get_retired_rules": v2_tools.tool_get_retired_rules,
    "get_rule_bind_history": v2_tools.tool_get_rule_bind_history,
    # Theses
    "get_theses": v2_tools.tool_get_theses,
    "get_thesis_lineage": v2_tools.tool_get_thesis_lineage,
    # Behavior
    "get_recent_decisions": v2_tools.tool_get_recent_decisions,
    "get_decision_detail": v2_tools.tool_get_decision_detail,
    "get_flip_flop_report": v2_tools.tool_get_flip_flop_report,
    "get_executor_behavior_summary": v2_tools.tool_get_executor_behavior_summary,
    "get_signal_attribution": v2_tools.tool_get_signal_attribution,
    # Reflection / sessions
    "get_session_memos": v2_tools.tool_get_session_memos,
    "get_reflection_actions": v2_tools.tool_get_reflection_actions,
    "get_session_summary": v2_tools.tool_get_session_summary_window,
}


def build_supervisor_tool_defs() -> list[dict]:
    """Filter the registered TOOL_DEFS in v2.tools to just the supervisor's set."""
    wanted = set(SUPERVISOR_TOOL_HANDLERS.keys())
    return [td for td in v2_tools.TOOL_DEFS if td.get("name") in wanted]
```

- [ ] **Step 2: Smoke-test the import path**

```bash
docker compose exec -T trading python -c "import v2.supervisor as s; print(len(s.SUPERVISOR_TOOL_HANDLERS), 'tools')"
```

Expected: `14 tools`.

- [ ] **Step 3: Commit**

```bash
git add v2/supervisor.py
git commit -m "Scaffold v2.supervisor module (constants + tool registry)"
```

### Task 3.2: Implement `run_supervisor()` and `main()`

**Files:**
- Modify: `v2/supervisor.py`

- [ ] **Step 1: Append `run_supervisor` and `main`**

```python
def _summarize_tool_calls(messages: list) -> list[dict]:
    """Walk assistant messages and count tool_use blocks by name."""
    counter: Counter = Counter()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                counter[block.get("name", "?")] += 1
    return [{"name": name, "count": count} for name, count in counter.most_common()]


def _insert_memo(
    *,
    model: str,
    content: str | None,
    status: str,
    turns_used: int,
    tool_calls: list[dict],
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
    error_message: str | None,
) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO supervisor_memos
              (model, prompt_version, content, status, turns_used,
               tool_calls, input_tokens, output_tokens, cost_usd, error_message)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                model,
                PROMPT_VERSION,
                content,
                status,
                turns_used,
                json.dumps(tool_calls),
                input_tokens,
                output_tokens,
                cost_usd,
                error_message,
            ),
        )
        return cur.fetchone()[0]


def run_supervisor(
    model: str = DEFAULT_SUPERVISOR_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    dry_run: bool = False,
) -> int | None:
    """Run one supervisor pass. Returns inserted memo id, or None if dry_run."""
    client = claude_client.get_claude_client()
    tool_defs = build_supervisor_tool_defs()
    initial = "Begin your strategy review. Investigate, cite IDs, then write the memo."

    try:
        with claude_client.capture_usage() as usage:
            result = claude_client.run_agentic_loop(
                client=client,
                model=model,
                system=STRATEGY_SUPERVISOR_SYSTEM,
                initial_message=initial,
                tools=tool_defs,
                tool_handlers=SUPERVISOR_TOOL_HANDLERS,
                max_turns=max_turns,
                stage_name="supervisor",
                purpose="strategy_critique",
            )
    except Exception as exc:  # noqa: BLE001 — top-level guard for the agentic loop
        log.exception("supervisor loop failed")
        if dry_run:
            raise
        return _insert_memo(
            model=model,
            content=None,
            status="error",
            turns_used=0,
            tool_calls=[],
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )

    final_text = claude_client.extract_final_text(result.messages)
    tool_calls = _summarize_tool_calls(result.messages)
    cost = claude_client.compute_cost(
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        cache_read_tokens=usage.cache_read_tokens,
    ) if hasattr(claude_client, "compute_cost") else None

    if final_text is None:
        status = "max_turns"
        content = None
        error_message = "loop did not produce final text within max_turns"
    else:
        status = "ok"
        content = final_text
        error_message = None

    if dry_run:
        print("=== SUPERVISOR MEMO (dry-run, not persisted) ===")
        print(content or f"[no final text — status={status}]")
        return None

    return _insert_memo(
        model=model,
        content=content,
        status=status,
        turns_used=result.turns_used,
        tool_calls=tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        error_message=error_message,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pinchy strategy supervisor (observer-only)")
    parser.add_argument("--model", default=os.environ.get("ALGO_SUPERVISOR_MODEL", DEFAULT_SUPERVISOR_MODEL))
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--dry-run", action="store_true", help="Run the loop, print the memo, skip the INSERT")
    args = parser.parse_args()
    try:
        memo_id = run_supervisor(model=args.model, max_turns=args.max_turns, dry_run=args.dry_run)
    except Exception:
        traceback.print_exc()
        return 2
    if memo_id is not None:
        print(f"supervisor_memo_id={memo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note on `compute_cost`: confirm the helper name before pasting. Run:

```bash
docker compose exec -T trading grep -nE "def compute_cost|cost_usd\s*=|MODEL_PRICING|price_per" v2/claude_client.py v2/session.py v2/strategy.py | head -20
```

Replace the `claude_client.compute_cost(...)` call with whatever helper the project already uses to derive `cost_usd` per LLM run. If no helper exists (only a constant-pricing table), inline the same arithmetic the reflection stage uses — the goal is one `cost_usd` value comparable to the `sessions.cost_usd` rows already in the DB. If costing is genuinely missing project-wide, store `cost_usd=None` and file a follow-up; do not invent a pricing constant here.

- [ ] **Step 2: Verify CLI parses**

```bash
docker compose exec -T trading python -m v2.supervisor --help
```

Expected: argparse help text listing `--model`, `--max-turns`, `--dry-run`.

- [ ] **Step 3: Commit**

```bash
git add v2/supervisor.py
git commit -m "Implement run_supervisor + CLI"
```

### Task 3.3: Unit-test run_supervisor with mocked loop

**Files:**
- Create: `tests/v2/test_supervisor.py`

- [ ] **Step 1: Write the test file**

```python
"""Unit tests for v2.supervisor."""

from unittest.mock import MagicMock, patch
import pytest

from v2 import supervisor as sup


@pytest.fixture
def fake_loop_result():
    """A successful agentic loop result with one tool use and a final text."""
    return MagicMock(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Begin..."}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "get_active_rules", "input": {}, "id": "t1"},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[]"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "## Watchlist\n- Nothing"}]},
        ],
        turns_used=2,
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def test_run_supervisor_inserts_ok_row(fake_loop_result, mock_db, mock_cursor):
    mock_cursor.fetchone.return_value = (42,)
    fake_usage = MagicMock(
        input_tokens=100, output_tokens=50,
        cache_creation_tokens=0, cache_read_tokens=0,
    )
    with patch.object(sup.claude_client, "get_claude_client"), \
         patch.object(sup.claude_client, "run_agentic_loop", return_value=fake_loop_result), \
         patch.object(sup.claude_client, "capture_usage") as cap, \
         patch.object(sup.claude_client, "compute_cost", return_value=0.0123, create=True):
        cap.return_value.__enter__.return_value = fake_usage
        memo_id = sup.run_supervisor(dry_run=False)
    assert memo_id == 42
    insert_call = mock_cursor.execute.call_args
    sql, params = insert_call.args
    assert "INSERT INTO supervisor_memos" in sql
    # status='ok', content non-null
    assert params[3] == "ok"
    assert "Watchlist" in params[2]
    # tool_calls summary contains the single get_active_rules use
    import json as _json
    tool_calls = _json.loads(params[5])
    assert tool_calls == [{"name": "get_active_rules", "count": 1}]


def test_run_supervisor_max_turns_inserts_null_content(mock_db, mock_cursor):
    no_final = MagicMock(
        messages=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "name": "get_theses", "input": {}, "id": "x"},
            ]},
        ],
        turns_used=20,
        stop_reason="max_turns",
        input_tokens=400, output_tokens=200,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )
    mock_cursor.fetchone.return_value = (43,)
    fake_usage = MagicMock(
        input_tokens=400, output_tokens=200,
        cache_creation_tokens=0, cache_read_tokens=0,
    )
    with patch.object(sup.claude_client, "get_claude_client"), \
         patch.object(sup.claude_client, "run_agentic_loop", return_value=no_final), \
         patch.object(sup.claude_client, "capture_usage") as cap, \
         patch.object(sup.claude_client, "compute_cost", return_value=0.05, create=True):
        cap.return_value.__enter__.return_value = fake_usage
        sup.run_supervisor(dry_run=False)
    _, params = mock_cursor.execute.call_args.args
    assert params[3] == "max_turns"
    assert params[2] is None  # content NULL
    assert params[9] == "loop did not produce final text within max_turns"


def test_run_supervisor_dry_run_does_not_insert(mock_db, mock_cursor, fake_loop_result, capsys):
    fake_usage = MagicMock(
        input_tokens=10, output_tokens=10,
        cache_creation_tokens=0, cache_read_tokens=0,
    )
    with patch.object(sup.claude_client, "get_claude_client"), \
         patch.object(sup.claude_client, "run_agentic_loop", return_value=fake_loop_result), \
         patch.object(sup.claude_client, "capture_usage") as cap, \
         patch.object(sup.claude_client, "compute_cost", return_value=0.001, create=True):
        cap.return_value.__enter__.return_value = fake_usage
        memo_id = sup.run_supervisor(dry_run=True)
    assert memo_id is None
    out = capsys.readouterr().out
    assert "Watchlist" in out
    # No INSERT should have happened
    for call in mock_cursor.execute.call_args_list:
        assert "INSERT INTO supervisor_memos" not in call.args[0]


def test_run_supervisor_api_error_inserts_error_row(mock_db, mock_cursor):
    mock_cursor.fetchone.return_value = (44,)
    with patch.object(sup.claude_client, "get_claude_client"), \
         patch.object(sup.claude_client, "run_agentic_loop", side_effect=RuntimeError("api 500")):
        memo_id = sup.run_supervisor(dry_run=False)
    assert memo_id == 44
    _, params = mock_cursor.execute.call_args.args
    assert params[3] == "error"
    assert params[2] is None
    assert "RuntimeError: api 500" in params[9]
```

- [ ] **Step 2: Run tests**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_supervisor.py -v
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/v2/test_supervisor.py
git commit -m "Unit-test supervisor run paths (ok / max_turns / dry-run / error)"
```

---

## Phase 4: Mutator-overlap defense

### Task 4.1: Test that the supervisor cannot call any strategy mutator

**Files:**
- Modify: `tests/v2/test_supervisor.py`

This is the defense-in-depth check called for in the spec under "Cost & safety controls". The mutator name set is defined in `v2/supervisor.py` as `STRATEGY_MUTATOR_NAMES`. If a future contributor wires a mutator into the supervisor handlers, this test fails.

- [ ] **Step 1: Append the test**

```python
def test_supervisor_tools_have_zero_overlap_with_mutators():
    from v2.supervisor import SUPERVISOR_TOOL_HANDLERS, STRATEGY_MUTATOR_NAMES
    overlap = set(SUPERVISOR_TOOL_HANDLERS.keys()) & set(STRATEGY_MUTATOR_NAMES)
    assert overlap == set(), (
        f"Supervisor MUST NOT include strategy mutator tools. Overlap found: {overlap}"
    )


def test_supervisor_tools_are_all_get_prefixed():
    from v2.supervisor import SUPERVISOR_TOOL_HANDLERS
    bad = [name for name in SUPERVISOR_TOOL_HANDLERS if not name.startswith("get_")]
    assert bad == [], f"Non-get_ tools should not be in supervisor registry: {bad}"


def test_mutator_set_matches_strategy_module_exports():
    """Sanity: the mutator name set is in sync with v2/strategy.py's actual mutators."""
    import v2.strategy as strat
    actual_mutators = {
        name.removeprefix("tool_")
        for name in dir(strat)
        if name in {
            "tool_propose_rule", "tool_retire_rule", "tool_revalidate_rule",
            "tool_update_strategy_identity", "tool_write_strategy_memo",
        }
    }
    from v2.supervisor import STRATEGY_MUTATOR_NAMES
    assert actual_mutators == set(STRATEGY_MUTATOR_NAMES), (
        f"STRATEGY_MUTATOR_NAMES drift: {actual_mutators ^ set(STRATEGY_MUTATOR_NAMES)}"
    )
```

- [ ] **Step 2: Run tests**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_supervisor.py -k "overlap or get_prefixed or mutator_set_matches" -v
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/v2/test_supervisor.py
git commit -m "Defend against accidental mutator wiring in supervisor"
```

---

## Phase 5: Integration test against a seeded DB

### Task 5.1: Seed a fixture DB and assert the supervisor cites known IDs

The spec asks for: "One integration test against a fixture DB containing a known oscillating rule, a dormant rule, and a thesis past its trigger window. Assert (loosely) that the memo content references those IDs."

This is intentionally a loose assertion — the supervisor calls a real Claude API. If the project's test suite policy forbids live LLM calls, gate this test behind an env flag matching the existing convention (look for `ALGO_LIVE_LLM` or similar in `tests/v2/conftest.py` autouse blockers; add the flag if absent).

**Files:**
- Modify: `tests/v2/test_supervisor.py`
- May modify: `tests/v2/conftest.py` (to expose a live-LLM gate)

- [ ] **Step 1: Add a live-LLM gate fixture if one doesn't exist**

In `tests/v2/conftest.py`, find the autouse block `_block_social_llm_and_session_db_calls` (or equivalent). Add a marker-based escape hatch:

```python
@pytest.fixture
def live_llm_enabled():
    """Returns True when the live-LLM env flag is set. Tests that need a real
    Anthropic API call must request this fixture and skip if False."""
    return os.environ.get("ALGO_LIVE_LLM") == "1"
```

- [ ] **Step 2: Append the integration test**

```python
@pytest.mark.integration
def test_supervisor_cites_oscillating_rule_and_dormant_rule(live_llm_enabled, real_db_seeded):
    """Live-LLM integration. Skipped unless ALGO_LIVE_LLM=1 is set.

    Seeds: rule_id=900 (oscillating, 11 binds in 22d), rule_id=901 (dormant 60d),
    thesis_id=800 (entry trigger passed 14d ago, no decisions tagged).
    Asserts the produced memo references at least two of {900, 901, 800}.
    """
    if not live_llm_enabled:
        pytest.skip("ALGO_LIVE_LLM not set; skipping live-LLM integration test")
    from v2 import supervisor as sup
    memo_id = sup.run_supervisor(max_turns=20, dry_run=False)
    assert memo_id is not None
    with real_db_seeded.cursor() as cur:
        cur.execute("SELECT content, status FROM supervisor_memos WHERE id=%s", (memo_id,))
        content, status = cur.fetchone()
    assert status == "ok", f"supervisor status={status}"
    citations = sum(1 for needle in ("900", "901", "800") if needle in content)
    assert citations >= 2, f"Expected memo to cite at least 2 of the seeded IDs; content:\n{content}"
```

`real_db_seeded` is a new fixture. Implement it in `tests/v2/conftest.py` to:
1. Use a separate test database (or a transaction-wrapped session against the dev DB if isolation is acceptable).
2. Seed the three records above plus enough `decisions` rows to make rule 900 look oscillating (alternating binds/lifts across 22 days).
3. Yield a psycopg2 connection. Roll back at the end.

If the project does not have an existing integration-DB pattern, mark this whole task as **deferred to a follow-up** and skip it for v1. Note that decision in the commit message rather than papering over it with a brittle fixture.

- [ ] **Step 3: Run, expecting skip in normal CI and pass under the live flag**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_supervisor.py::test_supervisor_cites_oscillating_rule_and_dormant_rule -v
# Expected: SKIPPED (no ALGO_LIVE_LLM)

ALGO_LIVE_LLM=1 docker compose exec -T trading python -m pytest tests/v2/test_supervisor.py::test_supervisor_cites_oscillating_rule_and_dormant_rule -v
# Expected: PASS (with real Anthropic + seeded DB)
```

- [ ] **Step 4: Commit**

```bash
git add tests/v2/test_supervisor.py tests/v2/conftest.py
git commit -m "Add live-LLM integration test for supervisor (gated)"
```

---

## Phase 6: Dashboard surface

### Task 6.1: Add dashboard queries for supervisor memos

**Files:**
- Modify: `dashboard/queries.py`
- Modify: `tests/test_dashboard_queries.py` (or whichever file covers `dashboard/queries.py` — verify via `grep -l "from queries import\|import queries" tests/`)

- [ ] **Step 1: Write failing tests**

Add to the queries test file:

```python
def test_get_recent_supervisor_memos(mock_cursor):
    from queries import get_recent_supervisor_memos
    mock_cursor.fetchall.return_value = [
        (5, "2026-05-27T10:00:00+00:00", "claude-opus-4-7", "v1.0.0", "ok", 12, 0.082),
        (4, "2026-05-20T10:00:00+00:00", "claude-opus-4-7", "v1.0.0", "ok", 9, 0.061),
    ]
    rows = get_recent_supervisor_memos(limit=10)
    assert len(rows) == 2
    assert rows[0]["id"] == 5
    assert rows[0]["status"] == "ok"
    args, _ = mock_cursor.execute.call_args
    assert "supervisor_memos" in args[0]
    assert args[1] == (10,)


def test_get_supervisor_memo_returns_full_row(mock_cursor):
    from queries import get_supervisor_memo
    mock_cursor.fetchone.return_value = (
        5, "2026-05-27T10:00:00+00:00", "claude-opus-4-7", "v1.0.0",
        "## Rules\n- Rule 27 oscillates", "ok", 12,
        [{"name": "get_active_rules", "count": 2}],
        1200, 800, 0.082, None,
    )
    row = get_supervisor_memo(5)
    assert row["id"] == 5
    assert "Rule 27" in row["content"]
    assert row["status"] == "ok"
    assert row["tool_calls"] == [{"name": "get_active_rules", "count": 2}]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T trading python -m pytest tests/test_dashboard_queries.py -k "supervisor" -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the queries**

Append to `dashboard/queries.py`:

```python
def get_recent_supervisor_memos(limit: int = 10) -> list[dict]:
    sql = """
        SELECT id, created_at, model, prompt_version, status, turns_used, cost_usd
        FROM supervisor_memos
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "created_at": r[1].isoformat() if hasattr(r[1], "isoformat") else r[1],
            "model": r[2],
            "prompt_version": r[3],
            "status": r[4],
            "turns_used": r[5],
            "cost_usd": float(r[6]) if r[6] is not None else None,
        }
        for r in rows
    ]


def get_supervisor_memo(memo_id: int) -> dict | None:
    sql = """
        SELECT id, created_at, model, prompt_version, content, status, turns_used,
               tool_calls, input_tokens, output_tokens, cost_usd, error_message
        FROM supervisor_memos
        WHERE id = %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (memo_id,))
        r = cur.fetchone()
    if r is None:
        return None
    return {
        "id": r[0],
        "created_at": r[1].isoformat() if hasattr(r[1], "isoformat") else r[1],
        "model": r[2],
        "prompt_version": r[3],
        "content": r[4],
        "status": r[5],
        "turns_used": r[6],
        "tool_calls": r[7],
        "input_tokens": r[8],
        "output_tokens": r[9],
        "cost_usd": float(r[10]) if r[10] is not None else None,
        "error_message": r[11],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose exec -T trading python -m pytest tests/test_dashboard_queries.py -k "supervisor" -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/queries.py tests/test_dashboard_queries.py
git commit -m "Add dashboard queries for supervisor memos"
```

### Task 6.2: Add `markdown` dependency

**Files:**
- Modify: `v2/requirements.txt`

- [ ] **Step 1: Add the line**

Append to `v2/requirements.txt`:

```
markdown>=3.5
```

- [ ] **Step 2: Rebuild the trading container so the dep is installed**

```bash
docker compose build trading
docker compose up -d trading
```

- [ ] **Step 3: Verify import**

```bash
docker compose exec -T trading python -c "import markdown; print(markdown.__version__)"
```

Expected: a version string >= 3.5.

- [ ] **Step 4: Commit**

```bash
git add v2/requirements.txt
git commit -m "Add markdown dependency for supervisor memo rendering"
```

### Task 6.3: Add dashboard routes for `/supervisor` and `/supervisor/<id>`

**Files:**
- Modify: `dashboard/app.py`
- Create: `dashboard/templates/supervisor.html`
- Create: `dashboard/templates/supervisor_detail.html`
- Modify: `dashboard/templates/base.html` (add internals-nav link)
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing tests**

Per the project's dashboard test convention (see `tests/test_dashboard.py` existing tests + the conftest memory note: "Dashboard tests use sys.modules['queries'] injection — don't reset_mock() on it"), set return values directly on the injected `queries` mock rather than using `unittest.mock.patch`. Whatever fixture name the existing dashboard tests use for the `queries` mock, reuse it here — name varies. Read an existing passing test like `test_attribution_renders` in `tests/test_dashboard.py` first to mirror the pattern, then add:

```python
def test_supervisor_route_renders_latest_and_list(client, queries_mock):
    queries_mock.get_recent_supervisor_memos.return_value = [
        {"id": 5, "created_at": "2026-05-27T10:00:00+00:00", "model": "claude-opus-4-7",
         "prompt_version": "v1.0.0", "status": "ok", "turns_used": 12, "cost_usd": 0.08},
    ]
    queries_mock.get_supervisor_memo.return_value = {
        "id": 5, "created_at": "2026-05-27T10:00:00+00:00", "model": "claude-opus-4-7",
        "prompt_version": "v1.0.0", "content": "## Rules\n- Rule 27 oscillates",
        "status": "ok", "turns_used": 12,
        "tool_calls": [{"name": "get_active_rules", "count": 2}],
        "input_tokens": 1200, "output_tokens": 800, "cost_usd": 0.08, "error_message": None,
    }
    resp = client.get("/supervisor")
    assert resp.status_code == 200
    assert b"Rule 27 oscillates" in resp.data
    assert b"claude-opus-4-7" in resp.data


def test_supervisor_detail_route_renders_specific_memo(client, queries_mock):
    queries_mock.get_supervisor_memo.return_value = {
        "id": 5, "created_at": "2026-05-27T10:00:00+00:00", "model": "claude-opus-4-7",
        "prompt_version": "v1.0.0", "content": "## Theses\n- Thesis 17 stale",
        "status": "ok", "turns_used": 8, "tool_calls": [], "input_tokens": 500,
        "output_tokens": 200, "cost_usd": 0.02, "error_message": None,
    }
    resp = client.get("/supervisor/5")
    assert resp.status_code == 200
    assert b"Thesis 17 stale" in resp.data


def test_supervisor_detail_404_when_missing(client, queries_mock):
    queries_mock.get_supervisor_memo.return_value = None
    resp = client.get("/supervisor/9999")
    assert resp.status_code == 404


def test_supervisor_route_handles_no_memos(client, queries_mock):
    queries_mock.get_recent_supervisor_memos.return_value = []
    queries_mock.get_supervisor_memo.return_value = None
    resp = client.get("/supervisor")
    assert resp.status_code == 200
    assert b"No supervisor memos yet" in resp.data
```

If the actual fixture name is `mock_queries`, `qmock`, etc., substitute accordingly — the pattern (set return values on the injected mock) is what matters.

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T trading python -m pytest tests/test_dashboard.py -k "supervisor" -v
```

Expected: 404 on `/supervisor` (route not defined).

- [ ] **Step 3: Add the routes to `dashboard/app.py`**

Add near the other internals-page routes:

```python
import markdown as _markdown  # at top of file alongside other imports

# ... existing routes ...

@app.route("/supervisor")
def supervisor_index():
    """Latest supervisor memo + list of recent runs."""
    recent = get_recent_supervisor_memos(limit=10)
    latest = get_supervisor_memo(recent[0]["id"]) if recent else None
    latest_html = _markdown.markdown(latest["content"]) if (latest and latest.get("content")) else None
    return render_template(
        "supervisor.html",
        recent=recent,
        latest=latest,
        latest_html=latest_html,
    )


@app.route("/supervisor/<int:memo_id>")
def supervisor_detail(memo_id: int):
    memo = get_supervisor_memo(memo_id)
    if memo is None:
        abort(404)
    body_html = _markdown.markdown(memo["content"]) if memo.get("content") else None
    return render_template("supervisor_detail.html", memo=memo, body_html=body_html)
```

And add to the `from queries import (...)` block at the top:

```python
    get_recent_supervisor_memos,
    get_supervisor_memo,
```

- [ ] **Step 4: Create `dashboard/templates/supervisor.html`**

Mirror the structure of `dashboard/templates/attribution.html` (read it first so the page slots into the existing layout). Minimal content:

```html
{% extends "base.html" %}
{% block title %}Strategy Supervisor{% endblock %}
{% block content %}
<div class="row">
  <div class="col-lg-9">
    <h1>Strategy Supervisor</h1>
    {% if latest %}
      <div class="text-muted small mb-2">
        Run {{ latest.created_at }} · model {{ latest.model }} · prompt {{ latest.prompt_version }}
        · {{ latest.turns_used }} turns
        {% if latest.cost_usd is not none %} · ${{ '%.4f'|format(latest.cost_usd) }}{% endif %}
        · <a href="{{ url_for('supervisor_detail', memo_id=latest.id) }}">permalink</a>
      </div>
      {% if latest.status != 'ok' %}
        <div class="alert alert-warning">
          Last run did not complete cleanly. Status: <strong>{{ latest.status }}</strong>.
          {% if latest.error_message %}<br><code>{{ latest.error_message }}</code>{% endif %}
        </div>
      {% endif %}
      <article class="supervisor-memo">{{ latest_html|safe if latest_html else '' }}</article>
    {% else %}
      <p>No supervisor memos yet. Run <code>task supervise</code> to generate one.</p>
    {% endif %}
  </div>
  <aside class="col-lg-3">
    <h5>Recent runs</h5>
    <ul class="list-unstyled small">
      {% for r in recent %}
        <li>
          <a href="{{ url_for('supervisor_detail', memo_id=r.id) }}">{{ r.created_at[:10] }}</a>
          <span class="text-muted">({{ r.status }})</span>
        </li>
      {% else %}
        <li class="text-muted">none</li>
      {% endfor %}
    </ul>
  </aside>
</div>
{% endblock %}
```

- [ ] **Step 5: Create `dashboard/templates/supervisor_detail.html`**

```html
{% extends "base.html" %}
{% block title %}Supervisor memo #{{ memo.id }}{% endblock %}
{% block content %}
<p><a href="{{ url_for('supervisor_index') }}">&larr; All supervisor memos</a></p>
<h1>Supervisor memo #{{ memo.id }}</h1>
<div class="text-muted small mb-3">
  {{ memo.created_at }} · model {{ memo.model }} · prompt {{ memo.prompt_version }}
  · {{ memo.turns_used }} turns
  {% if memo.cost_usd is not none %} · ${{ '%.4f'|format(memo.cost_usd) }}{% endif %}
</div>
{% if memo.status != 'ok' %}
  <div class="alert alert-warning">
    Status: <strong>{{ memo.status }}</strong>.
    {% if memo.error_message %}<br><code>{{ memo.error_message }}</code>{% endif %}
  </div>
{% endif %}
{% if body_html %}
  <article class="supervisor-memo">{{ body_html|safe }}</article>
{% else %}
  <p class="text-muted">No content recorded for this run.</p>
{% endif %}
<hr>
<h5>Tool calls</h5>
<ul class="small">
  {% for tc in memo.tool_calls %}
    <li><code>{{ tc.name }}</code> &times; {{ tc.count }}</li>
  {% else %}
    <li class="text-muted">none</li>
  {% endfor %}
</ul>
{% endblock %}
```

- [ ] **Step 6: Add nav link to `dashboard/templates/base.html`**

Open `dashboard/templates/base.html`. Find the internals-page nav list (where `/attribution`, `/strategy`, etc. are linked). Add a new `<li>` (keep formatting consistent with adjacent items):

```html
<li class="nav-item">
  <a class="nav-link" href="{{ url_for('supervisor_index') }}">Supervisor</a>
</li>
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
docker compose exec -T trading python -m pytest tests/test_dashboard.py -k "supervisor" -v
```

Expected: 4 passed.

- [ ] **Step 8: Commit**

```bash
git add dashboard/app.py dashboard/templates/supervisor.html dashboard/templates/supervisor_detail.html dashboard/templates/base.html tests/test_dashboard.py
git commit -m "Surface supervisor memos on the operator dashboard"
```

---

## Phase 7: Taskfile + docs

### Task 7.1: Add `supervise` and `supervise:dry-run` task targets

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Add the targets**

In the "Trading workflows" section, after the `learn` target:

```yaml
  supervise:
    desc: Run the strategy supervisor (observer-only critic, persists one memo)
    deps: [docker:up]
    cmds:
      - docker compose exec trading python -m v2.supervisor {{.CLI_ARGS}}

  supervise:dry-run:
    desc: Run the supervisor loop and print the memo without persisting
    deps: [docker:up]
    cmds:
      - docker compose exec trading python -m v2.supervisor --dry-run {{.CLI_ARGS}}
```

- [ ] **Step 2: Verify**

```bash
task -l | grep supervise
```

Expected: both targets listed with their descriptions.

- [ ] **Step 3: Commit**

```bash
git add Taskfile.yml
git commit -m "Add supervise / supervise:dry-run Task targets"
```

### Task 7.2: Add a one-line CLAUDE.md note

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Edit**

Find the "### Key v2 Modules" list. Add (alphabetically, near `strategy.py`):

```markdown
- **`supervisor.py`** — Observer-only strategy critic. Read-only DB tool registry + Opus loop, persists one markdown memo per run to `supervisor_memos`. Run with `task supervise` (or `task supervise:dry-run`). Spec: `docs/superpowers/specs/2026-05-27-strategy-supervisor-design.md`.
```

And in the database schema list, add:

```markdown
- `supervisor_memos` — Free-form markdown critiques from `python -m v2.supervisor`
```

- [ ] **Step 2: Verify the file still parses cleanly**

```bash
head -200 CLAUDE.md | grep -A1 supervisor
```

Expected: the two lines you just added appear.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document supervisor module in CLAUDE.md"
```

---

## Phase 8: End-to-end smoke

### Task 8.1: Run the supervisor against the real (paper) DB and verify a memo lands

This is a one-shot manual verification — not committed code, but a step the operator should not skip.

- [ ] **Step 1: Dry-run first**

```bash
task supervise:dry-run
```

Expected: a markdown memo printed to stdout, no row in `supervisor_memos`. Verify:

```bash
docker compose exec -T db psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "SELECT COUNT(*) FROM supervisor_memos;"
```

Expected: count unchanged.

- [ ] **Step 2: Real run**

```bash
task supervise
```

Expected: `supervisor_memo_id=N` printed.

- [ ] **Step 3: Inspect the row**

```bash
docker compose exec -T db psql -U $POSTGRES_USER -d $POSTGRES_DB -c \
  "SELECT id, created_at, model, status, turns_used, cost_usd FROM supervisor_memos ORDER BY id DESC LIMIT 1;"
```

Expected: a single ok row with non-zero `turns_used` and a small (cents-to-dollars) `cost_usd`.

- [ ] **Step 4: Load the dashboard page**

Open `http://localhost:3000/supervisor` in a browser. Verify the memo renders as Markdown, the recent-runs sidebar lists at least one entry, and the permalink under the memo header works.

- [ ] **Step 5: Run the full test suite one more time**

```bash
docker compose exec -T trading python -m pytest tests/v2/test_supervisor.py tests/v2/test_tools.py tests/test_dashboard.py tests/test_dashboard_queries.py -v
```

Expected: all green (the live-LLM integration test from Task 5.1 skips by default).

---

## Done criteria

All boxes checked, all tests green, the supervisor produced at least one memo against the real DB, and the dashboard page renders it. Follow-up candidates (structured findings, trend charts, weekly cron, dedicated read-only DB role, feeding memos into the strategist) are explicitly out of scope per the spec.
