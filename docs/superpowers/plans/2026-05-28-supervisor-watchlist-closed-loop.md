# Supervisor Watchlist → Closed-Loop Action — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the strategy supervisor emit a structured, stage-owned watchlist that the ideation and reflection stages must each ingest and resolve before completing, with the supervisor running as a session stage (after learning refresh, before ideation).

**Architecture:** A new `supervisor_watchlist_items` table stores items tagged with `owner_stage` (`ideation` or `reflection`). The supervisor (now a session stage at "Stage 0.5") records items via a write tool scoped to that table. Each acting stage loads its open items into context, must call `resolve_watchlist_item` for each (acted/dismissed), and a hard `RuntimeError` gate fires if any owned item is left open. Shared tool/formatter/gate logic lives in a new `v2/watchlist.py`. A separate `amend_rule` tool stops the retire-and-replace rule churn.

**Tech Stack:** Python 3.12, psycopg2 (raw SQL via `get_cursor()`), Anthropic SDK agentic loops (`v2/claude_client.py`), pytest (run in docker), ruff.

**Spec:** `docs/superpowers/specs/2026-05-27-supervisor-watchlist-closed-loop-design.md`

**Reconciliation note vs spec:** the spec data model listed both a `status` and a `resolution` column; this plan folds them into a single `status` column (`open`/`acted`/`dismissed`) to avoid redundancy. The tool's `resolution` argument maps directly onto `status`.

**Test runner:** all pytest commands run in docker (host python is 3.10; codebase needs 3.12+):
```
docker compose exec -T trading python3 -m pytest <path> -v
```
If the `trading` container is not up, start it with `docker compose up -d trading db`.

---

## File Structure

- **Create** `db/init/034_supervisor_watchlist_items.sql` — the new table.
- **Modify** `v2/database/trading_db.py` — 4 new DB helpers (`record_watchlist_item`, `get_open_watchlist_items`, `resolve_watchlist_item`, `amend_strategy_rule`).
- **Create** `v2/watchlist.py` — shared tool def, resolve-handler factory, context formatter, and post-loop gate helper. One responsibility: the watchlist resolution contract shared by both acting stages.
- **Modify** `v2/supervisor.py` — `record_watchlist_item` tool + buffered handler; persist buffered items after the memo INSERT; prompt update.
- **Modify** `v2/session.py` — wire the supervisor as Stage 0.5 (after learning refresh, before pipeline); new `SessionResult` fields.
- **Modify** `v2/strategy.py` — register `resolve_watchlist_item`, ingest reflection-owned items, add the gate, add the `amend_rule` tool + handler + prompt guidance.
- **Modify** `v2/ideation_claude.py` — register `resolve_watchlist_item`, ingest ideation-owned items, add the gate.
- **Tests:** `tests/v2/test_watchlist.py` (new, covers DB helpers + shared module), additions to `tests/v2/test_supervisor.py`, `tests/v2/test_strategy.py`, `tests/v2/test_ideation_claude.py`, `tests/v2/test_session.py`.

---

## Task 1: Create the `supervisor_watchlist_items` migration

**Files:**
- Create: `db/init/034_supervisor_watchlist_items.sql`

- [ ] **Step 1: Write the migration**

```sql
-- 034_supervisor_watchlist_items.sql
-- Structured watchlist emitted by the strategy supervisor. Each item is
-- owned by exactly one acting stage, which must resolve it (acted/dismissed)
-- before that stage may complete. See spec
-- docs/superpowers/specs/2026-05-27-supervisor-watchlist-closed-loop-design.md

CREATE TABLE IF NOT EXISTS supervisor_watchlist_items (
    id                      SERIAL PRIMARY KEY,
    source_memo_id          INT NOT NULL REFERENCES supervisor_memos(id),
    title                   TEXT NOT NULL,
    detail                  TEXT NOT NULL,
    owner_stage             TEXT NOT NULL CHECK (owner_stage IN ('ideation', 'reflection')),
    status                  TEXT NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open', 'acted', 'dismissed')),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at             TIMESTAMPTZ,
    resolution_note         TEXT,
    resolved_by_session_id  INT,
    resolved_by_stage       TEXT
);

CREATE INDEX IF NOT EXISTS idx_watchlist_open_by_stage
    ON supervisor_watchlist_items (owner_stage, status);

COMMENT ON TABLE supervisor_watchlist_items IS
    'Structured supervisor watchlist. One row per item the supervisor flags; resolved by the owning acting stage.';
COMMENT ON COLUMN supervisor_watchlist_items.owner_stage IS
    'Which acting stage must resolve this: ideation (theses/playbook) or reflection (rules/identity).';
COMMENT ON COLUMN supervisor_watchlist_items.status IS
    'open until resolved; acted = a change was made; dismissed = reasoned no-op.';
```

- [ ] **Step 2: Apply and verify against the running DB**

Run:
```bash
U=$(grep -m1 POSTGRES_USER .env | cut -d= -f2); D=$(grep -m1 POSTGRES_DB .env | cut -d= -f2)
docker compose exec -T db psql -U "$U" -d "$D" -f /docker-entrypoint-initdb.d/034_supervisor_watchlist_items.sql 2>&1 || \
  docker compose exec -T db psql -U "$U" -d "$D" < db/init/034_supervisor_watchlist_items.sql
docker compose exec -T db psql -U "$U" -d "$D" -c "\d supervisor_watchlist_items"
```
Expected: table description prints with the columns and CHECK constraints above. (init scripts only auto-run on a fresh volume, so apply it manually here.)

- [ ] **Step 3: Commit**

```bash
git add db/init/034_supervisor_watchlist_items.sql
git commit -m "Add supervisor_watchlist_items table"
```

---

## Task 2: DB helpers in `trading_db.py`

**Files:**
- Modify: `v2/database/trading_db.py` (add 4 functions near the other strategy-rule helpers, ~line 1000+)
- Test: `tests/v2/test_watchlist.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/v2/test_watchlist.py`:

```python
"""Unit tests for the supervisor watchlist DB helpers and shared module."""

from unittest.mock import MagicMock, patch

import pytest

from v2.database import trading_db as db


@pytest.fixture
def fake_cursor():
    cur = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cur)
    cm.__exit__ = MagicMock(return_value=False)
    return cm, cur


def test_record_watchlist_item_inserts_and_returns_id(fake_cursor):
    cm, cur = fake_cursor
    cur.fetchone.return_value = {"id": 7}
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        item_id = db.record_watchlist_item(
            source_memo_id=3, title="Retire Rule 43",
            detail="auto-lift fired", owner_stage="reflection",
        )
    assert item_id == 7
    sql = cur.execute.call_args[0][0]
    assert "INSERT INTO supervisor_watchlist_items" in sql


def test_record_watchlist_item_rejects_bad_owner_stage(fake_cursor):
    cm, _ = fake_cursor
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        with pytest.raises(ValueError):
            db.record_watchlist_item(
                source_memo_id=3, title="x", detail="y", owner_stage="trading",
            )


def test_get_open_watchlist_items_filters_by_stage(fake_cursor):
    cm, cur = fake_cursor
    cur.fetchall.return_value = [{"id": 1, "title": "t", "detail": "d"}]
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        rows = db.get_open_watchlist_items("ideation")
    assert rows == [{"id": 1, "title": "t", "detail": "d"}]
    args = cur.execute.call_args[0]
    assert "status = 'open'" in args[0]
    assert args[1] == ("ideation",)


def test_resolve_watchlist_item_scopes_to_owner_stage(fake_cursor):
    cm, cur = fake_cursor
    cur.rowcount = 1
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        ok = db.resolve_watchlist_item(
            item_id=5, resolution="acted", note="retired Rule 43",
            session_id=42, stage="reflection",
        )
    assert ok is True
    sql, params = cur.execute.call_args[0]
    assert "owner_stage = %s" in sql
    assert "status = 'open'" in sql
    assert params == ("acted", "retired Rule 43", 42, "reflection", 5, "reflection")


def test_resolve_watchlist_item_rejects_bad_resolution(fake_cursor):
    cm, _ = fake_cursor
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        with pytest.raises(ValueError):
            db.resolve_watchlist_item(
                item_id=5, resolution="maybe", note="x",
                session_id=1, stage="reflection",
            )


def test_resolve_watchlist_item_returns_false_when_no_row(fake_cursor):
    cm, cur = fake_cursor
    cur.rowcount = 0
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        ok = db.resolve_watchlist_item(
            item_id=999, resolution="dismissed", note="n",
            session_id=1, stage="ideation",
        )
    assert ok is False


def test_amend_strategy_rule_updates_in_place(fake_cursor):
    cm, cur = fake_cursor
    cur.rowcount = 1
    with patch("v2.database.trading_db.get_cursor", return_value=cm):
        ok = db.amend_strategy_rule(
            rule_id=48, new_rule_text="updated text",
            new_evidence="beat-rate now 61% over 24 samples", reason="evidence refresh",
        )
    assert ok is True
    sql = cur.execute.call_args[0][0]
    assert "UPDATE strategy_rules" in sql
    assert "retired_at" not in sql  # amend must NOT retire
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_watchlist.py -v`
Expected: FAIL with `AttributeError: module 'v2.database.trading_db' has no attribute 'record_watchlist_item'`

- [ ] **Step 3: Implement the helpers**

Append to `v2/database/trading_db.py` (the module already imports `get_cursor`):

```python
_WATCHLIST_OWNER_STAGES = {"ideation", "reflection"}
_WATCHLIST_RESOLUTIONS = {"acted", "dismissed"}


def record_watchlist_item(
    source_memo_id: int, title: str, detail: str, owner_stage: str
) -> int:
    """Insert one supervisor watchlist item. Returns the new id."""
    if owner_stage not in _WATCHLIST_OWNER_STAGES:
        raise ValueError(f"owner_stage must be one of {_WATCHLIST_OWNER_STAGES}")
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO supervisor_watchlist_items
              (source_memo_id, title, detail, owner_stage)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (source_memo_id, title, detail, owner_stage),
        )
        row = cur.fetchone()
    return row["id"] if isinstance(row, dict) else row[0]


def get_open_watchlist_items(owner_stage: str) -> list[dict]:
    """Return all open watchlist items for one owner stage, oldest first."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, source_memo_id, title, detail, owner_stage, created_at
            FROM supervisor_watchlist_items
            WHERE status = 'open' AND owner_stage = %s
            ORDER BY created_at ASC, id ASC
            """,
            (owner_stage,),
        )
        return list(cur.fetchall())


def resolve_watchlist_item(
    item_id: int, resolution: str, note: str, session_id: int | None, stage: str
) -> bool:
    """Mark an open item resolved. Scoped to the resolving stage so one stage
    cannot resolve another stage's items. Returns True if a row was updated."""
    if resolution not in _WATCHLIST_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {_WATCHLIST_RESOLUTIONS}")
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE supervisor_watchlist_items
            SET status = %s,
                resolution_note = %s,
                resolved_at = now(),
                resolved_by_session_id = %s,
                resolved_by_stage = %s
            WHERE id = %s AND status = 'open' AND owner_stage = %s
            """,
            (resolution, note, session_id, stage, item_id, stage),
        )
        return cur.rowcount > 0


def amend_strategy_rule(
    rule_id: int, new_rule_text: str, new_evidence: str, reason: str
) -> bool:
    """Update an active rule's text/evidence in place (no retire-and-replace).
    Returns True if the rule was active and updated."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE strategy_rules
            SET rule_text = %s,
                supporting_evidence = %s
            WHERE id = %s AND status = 'active'
            """,
            (new_rule_text, new_evidence, rule_id),
        )
        return cur.rowcount > 0
```

Note: `reason` is accepted for a consistent call signature and logging; it is not persisted as a column. If you add a logger line, use `logger.info("Amended rule %s: %s", rule_id, reason)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_watchlist.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_watchlist.py
git commit -m "Add watchlist + amend_rule DB helpers"
```

---

## Task 3: Shared watchlist module `v2/watchlist.py`

**Files:**
- Create: `v2/watchlist.py`
- Test: append to `tests/v2/test_watchlist.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_watchlist.py`:

```python
from v2 import watchlist as wl


def test_resolve_tool_def_shape():
    d = wl.RESOLVE_WATCHLIST_TOOL_DEF
    assert d["name"] == "resolve_watchlist_item"
    props = d["input_schema"]["properties"]
    assert set(props) == {"item_id", "resolution", "note"}
    assert props["resolution"]["enum"] == ["acted", "dismissed"]
    assert set(d["input_schema"]["required"]) == {"item_id", "resolution", "note"}


def test_make_resolve_handler_binds_session_and_stage():
    with patch("v2.watchlist.db.resolve_watchlist_item", return_value=True) as m:
        handler = wl.make_resolve_handler(session_id=42, stage="reflection")
        out = handler(item_id=5, resolution="acted", note="retired Rule 43")
    m.assert_called_once_with(
        item_id=5, resolution="acted", note="retired Rule 43",
        session_id=42, stage="reflection",
    )
    assert "5" in out and "acted" in out


def test_make_resolve_handler_reports_miss():
    with patch("v2.watchlist.db.resolve_watchlist_item", return_value=False):
        handler = wl.make_resolve_handler(session_id=1, stage="ideation")
        out = handler(item_id=99, resolution="dismissed", note="n")
    assert "Error" in out or "not found" in out.lower()


def test_format_open_watchlist_items_empty():
    assert "no open" in wl.format_open_watchlist_items([]).lower()


def test_format_open_watchlist_items_lists_each():
    items = [
        {"id": 1, "title": "Retire Rule 43", "detail": "auto-lift fired"},
        {"id": 2, "title": "Close thesis 267", "detail": "Rule 43 grounded"},
    ]
    out = wl.format_open_watchlist_items(items)
    assert "item 1" in out and "Retire Rule 43" in out
    assert "item 2" in out and "Close thesis 267" in out
    assert "resolve_watchlist_item" in out  # instructs the model what to do


def test_assert_watchlist_resolved_raises_when_open_remain():
    with patch("v2.watchlist.db.get_open_watchlist_items",
               return_value=[{"id": 3, "title": "x", "detail": "y"}]):
        with pytest.raises(RuntimeError) as exc:
            wl.assert_watchlist_resolved("reflection")
    assert "3" in str(exc.value)


def test_assert_watchlist_resolved_passes_when_clear():
    with patch("v2.watchlist.db.get_open_watchlist_items", return_value=[]):
        wl.assert_watchlist_resolved("ideation")  # no raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_watchlist.py -v -k "tool_def or resolve_handler or format_open or assert_watchlist"`
Expected: FAIL with `ModuleNotFoundError: No module named 'v2.watchlist'`

- [ ] **Step 3: Implement `v2/watchlist.py`**

```python
"""Shared supervisor-watchlist resolution contract for acting stages.

Both the ideation (Stage 2) and reflection (Stage 4) stages ingest their
open watchlist items, must resolve each via `resolve_watchlist_item`, and
are hard-gated by `assert_watchlist_resolved` so a stage cannot complete
with an unresolved owned item.
"""

from __future__ import annotations

import logging

from v2.database import trading_db as db

logger = logging.getLogger(__name__)

RESOLVE_WATCHLIST_TOOL_DEF: dict = {
    "name": "resolve_watchlist_item",
    "description": (
        "Resolve one supervisor watchlist item you own. Use 'acted' when you "
        "made a change in response (retired/amended a rule, closed/updated a "
        "thesis, re-spec'd the playbook), or 'dismissed' when no change is "
        "warranted. A dismissal MUST justify why in the note. Every open item "
        "must be resolved before you finish."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "item_id": {"type": "integer"},
            "resolution": {"type": "string", "enum": ["acted", "dismissed"]},
            "note": {
                "type": "string",
                "description": "What you did (acted) or why nothing was needed (dismissed).",
            },
        },
        "required": ["item_id", "resolution", "note"],
    },
}


def make_resolve_handler(session_id: int | None, stage: str):
    """Return a handler bound to the resolving session + stage."""
    def _handler(item_id: int, resolution: str, note: str) -> str:
        try:
            ok = db.resolve_watchlist_item(
                item_id=item_id, resolution=resolution, note=note,
                session_id=session_id, stage=stage,
            )
        except ValueError as e:
            return f"Error: {e}"
        if not ok:
            return (
                f"Error: watchlist item {item_id} not found, already resolved, "
                f"or not owned by the {stage} stage."
            )
        logger.info("Resolved watchlist item %s (%s) in %s", item_id, resolution, stage)
        return f"Resolved watchlist item {item_id}: {resolution}."
    return _handler


def format_open_watchlist_items(items: list[dict]) -> str:
    """Render open items as a mandatory-action block for the stage's context."""
    if not items:
        return "SUPERVISOR WATCHLIST: no open items for this stage."
    lines = [
        "SUPERVISOR WATCHLIST (MANDATORY): the supervisor flagged the following "
        "for you this session. You MUST call resolve_watchlist_item for EACH "
        "before finishing — 'acted' (you made a change) or 'dismissed' (reasoned no-op).",
    ]
    for it in items:
        lines.append(f"- item {it['id']}: {it['title']}")
        lines.append(f"    {it['detail']}")
    return "\n".join(lines)


def assert_watchlist_resolved(owner_stage: str) -> None:
    """Raise RuntimeError if any item owned by this stage is still open.

    Called after the agentic loop. New items are only created by the
    supervisor (earlier in the session), so any remaining open row is an
    unresolved item — the forcing function."""
    remaining = db.get_open_watchlist_items(owner_stage)
    if remaining:
        ids = ", ".join(str(it["id"]) for it in remaining)
        raise RuntimeError(
            f"{owner_stage} finished with unresolved watchlist item(s): {ids}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_watchlist.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add v2/watchlist.py tests/v2/test_watchlist.py
git commit -m "Add shared watchlist resolution module"
```

---

## Task 4: Supervisor records structured watchlist items

**Files:**
- Modify: `v2/supervisor.py`
- Test: append to `tests/v2/test_supervisor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_supervisor.py`:

```python
def test_record_watchlist_handler_buffers_and_validates():
    buf = []
    handler = sup._make_record_watchlist_handler(buf)
    out = handler(title="Retire Rule 43", detail="auto-lift fired", owner_stage="reflection")
    assert buf == [{"title": "Retire Rule 43", "detail": "auto-lift fired",
                    "owner_stage": "reflection"}]
    assert "recorded" in out.lower()
    err = handler(title="x", detail="y", owner_stage="trading")
    assert "Error" in err
    assert len(buf) == 1  # bad item not buffered


def test_record_watchlist_tool_in_supervisor_defs():
    names = {d["name"] for d in sup.build_supervisor_tool_defs()}
    assert "record_watchlist_item" in names


def test_run_supervisor_persists_buffered_items(fake_loop_result, fake_usage):
    # The loop "calls" record_watchlist_item by appending to the buffer.
    def fake_loop(*args, **kwargs):
        handler = kwargs["tool_handlers"]["record_watchlist_item"]
        handler(title="Close thesis 267", detail="Rule 43 grounded", owner_stage="ideation")
        return fake_loop_result
    with patch("v2.supervisor.claude_client.run_agentic_loop", side_effect=fake_loop), \
         patch("v2.supervisor.claude_client.capture_usage") as cu, \
         patch("v2.supervisor.claude_client.get_claude_client"), \
         patch("v2.supervisor.claude_client.extract_final_text", return_value="## Watchlist\n- x"), \
         patch("v2.supervisor._compute_cost_safely", return_value=0.5), \
         patch("v2.supervisor._insert_memo", return_value=99) as ins, \
         patch("v2.supervisor.db.record_watchlist_item", return_value=1) as rec:
        cu.return_value.__enter__.return_value = fake_usage
        memo_id = sup.run_supervisor()
    assert memo_id == 99
    ins.assert_called_once()
    rec.assert_called_once_with(
        source_memo_id=99, title="Close thesis 267",
        detail="Rule 43 grounded", owner_stage="ideation",
    )


def test_run_supervisor_dry_run_does_not_persist_items(fake_loop_result, fake_usage):
    def fake_loop(*args, **kwargs):
        kwargs["tool_handlers"]["record_watchlist_item"](
            title="t", detail="d", owner_stage="reflection")
        return fake_loop_result
    with patch("v2.supervisor.claude_client.run_agentic_loop", side_effect=fake_loop), \
         patch("v2.supervisor.claude_client.capture_usage") as cu, \
         patch("v2.supervisor.claude_client.get_claude_client"), \
         patch("v2.supervisor.claude_client.extract_final_text", return_value="memo"), \
         patch("v2.supervisor._compute_cost_safely", return_value=0.0), \
         patch("v2.supervisor.db.record_watchlist_item") as rec:
        cu.return_value.__enter__.return_value = fake_usage
        sup.run_supervisor(dry_run=True)
    rec.assert_not_called()
```

(If `fake_usage` is not already a module fixture, reuse the one defined at the top of `test_supervisor.py`; the head shown in the spec confirms it exists.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_supervisor.py -v -k watchlist or dry_run`
Expected: FAIL (`AttributeError: ... has no attribute '_make_record_watchlist_handler'`)

- [ ] **Step 3: Implement in `v2/supervisor.py`**

3a. Add the import near the top (after the existing `from v2.database.connection import get_cursor`):

```python
from v2.database import trading_db as db
```

3b. Add the tool def constant and handler factory (after `STRATEGY_MUTATOR_NAMES`):

```python
RECORD_WATCHLIST_TOOL_DEF: dict = {
    "name": "record_watchlist_item",
    "description": (
        "Record ONE watchlist item for a specific acting stage to resolve next. "
        "Call this once per item in your Watchlist section. owner_stage MUST be "
        "'reflection' for rule/identity items, or 'ideation' for thesis/playbook "
        "items. This writes only to the supervisor's own watchlist table — it does "
        "NOT change strategy state."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short item label."},
            "detail": {"type": "string", "description": "What to check / what's wrong."},
            "owner_stage": {"type": "string", "enum": ["reflection", "ideation"]},
        },
        "required": ["title", "detail", "owner_stage"],
    },
}


def _make_record_watchlist_handler(buffer: list[dict]):
    """Handler that buffers items in-memory; persisted after the memo INSERT
    (the source_memo_id FK isn't known until the memo row exists)."""
    def _handler(title: str, detail: str, owner_stage: str) -> str:
        if owner_stage not in {"reflection", "ideation"}:
            return "Error: owner_stage must be 'reflection' or 'ideation'"
        buffer.append({"title": title, "detail": detail, "owner_stage": owner_stage})
        return f"recorded watchlist item for {owner_stage}: {title}"
    return _handler
```

3c. Make `build_supervisor_tool_defs` include the new write tool. Replace the function body's return with:

```python
def build_supervisor_tool_defs() -> list[dict]:
    wanted = set(SUPERVISOR_TOOL_HANDLERS.keys())
    read_defs = [td for td in v2_tools.TOOL_DEFINITIONS if td.get("name") in wanted]
    return read_defs + [RECORD_WATCHLIST_TOOL_DEF]
```

3d. In `run_supervisor`, create the buffer, register the handler, and persist after the memo INSERT. Edit the `try` block and the final return:

```python
    client = claude_client.get_claude_client()
    tool_defs = build_supervisor_tool_defs()
    watchlist_buffer: list[dict] = []
    handlers = {
        **SUPERVISOR_TOOL_HANDLERS,
        "record_watchlist_item": _make_record_watchlist_handler(watchlist_buffer),
    }
    initial = "Begin your strategy review. Investigate, cite IDs, then write the memo."

    try:
        with claude_client.capture_usage() as usage:
            result = claude_client.run_agentic_loop(
                client=client,
                model=model,
                system=STRATEGY_SUPERVISOR_SYSTEM,
                initial_message=initial,
                tools=tool_defs,
                tool_handlers=handlers,
                max_turns=max_turns,
                stage_name="supervisor",
                purpose="supervisor_loop",
            )
    except Exception as exc:  # noqa: BLE001
        # (unchanged error path)
        ...
```

Then, just before the final `return _insert_memo(...)` in the success path, capture the id and persist buffered items:

```python
    if dry_run:
        print("=== SUPERVISOR MEMO (dry-run, not persisted) ===")
        print(content or f"[no final text — status={status}]")
        if watchlist_buffer:
            print(f"=== {len(watchlist_buffer)} watchlist item(s) (dry-run, not persisted) ===")
            for it in watchlist_buffer:
                print(f"  [{it['owner_stage']}] {it['title']}")
        return None

    memo_id = _insert_memo(
        model=model,
        content=content,
        status=status,
        turns_used=result.turns_used,
        tool_calls=tool_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd,
        error_message=error_message,
    )
    for it in watchlist_buffer:
        try:
            db.record_watchlist_item(
                source_memo_id=memo_id, title=it["title"],
                detail=it["detail"], owner_stage=it["owner_stage"],
            )
        except Exception:
            log.exception("Failed to persist watchlist item %r", it.get("title"))
    return memo_id
```

3e. Update `STRATEGY_SUPERVISOR_SYSTEM`: change the final paragraph to instruct structured recording. Replace the closing lines (from "End with a \"Watchlist\" section...") with:

```python
End with a "Watchlist" section: 1-5 specific things to revisit. For EACH
watchlist entry, also call record_watchlist_item with an owner_stage:
- owner_stage="reflection" for rule lifecycle and identity items (retire/
  amend/revalidate a rule, identity drift).
- owner_stage="ideation" for thesis and playbook items (stale/contradictory
  theses, playbook conditionals that never clear).
The markdown Watchlist section and the recorded items must match. The owning
stage will be forced to resolve each item this same session.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_supervisor.py -v`
Expected: PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add v2/supervisor.py tests/v2/test_supervisor.py
git commit -m "Supervisor emits structured stage-owned watchlist items"
```

---

## Task 5: Wire the supervisor as session Stage 0.5

**Files:**
- Modify: `v2/session.py`
- Test: append to `tests/v2/test_session.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_session.py` (match the module's existing import/mocking style; the key behaviors):

```python
def test_supervisor_stage_runs_between_learning_and_pipeline(monkeypatch):
    import v2.session as session
    calls = []
    monkeypatch.setattr(session, "_run_learning_refresh",
                        lambda *a, **k: calls.append("learning") or "")
    monkeypatch.setattr(session, "_run_supervisor_stage",
                        lambda *a, **k: calls.append("supervisor"))
    monkeypatch.setattr(session, "_run_pipeline_stage",
                        lambda *a, **k: calls.append("pipeline"))
    monkeypatch.setattr(session, "_run_strategist_stage", lambda *a, **k: None)
    monkeypatch.setattr(session, "_run_executor_stage", lambda *a, **k: None)
    monkeypatch.setattr(session, "_run_strategy_stage", lambda *a, **k: None)
    monkeypatch.setattr(session, "_run_dashboard_stage_wrapper", lambda *a, **k: None)
    monkeypatch.setattr(session, "_init_session", lambda *a, **k: (1, set(), None))
    session.run_session()
    assert calls.index("learning") < calls.index("supervisor") < calls.index("pipeline")


def test_supervisor_stage_failure_does_not_abort_session(monkeypatch):
    import v2.session as session
    result = session.SessionResult()
    monkeypatch.setattr(session, "run_supervisor",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(session, "_start_stage", lambda *a, **k: None)
    monkeypatch.setattr(session, "_fail_stage", lambda *a, **k: None)
    monkeypatch.setattr(session, "_complete_stage", lambda *a, **k: None)
    session._run_supervisor_stage(result, session_id=1, completed_stages=set(), skip=False)
    assert result.supervisor_error == "boom"
```

(Adjust `_init_session`/`run_session` arg shapes to the real signatures in the file; the ordering assertion is the load-bearing part.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_session.py -v -k supervisor`
Expected: FAIL (`AttributeError: module 'v2.session' has no attribute '_run_supervisor_stage'`)

- [ ] **Step 3: Implement in `v2/session.py`**

3a. Add the import:

```python
from .supervisor import run_supervisor
```

3b. Add fields to `SessionResult` (alongside the other stage fields, e.g. near `pipeline_error`):

```python
    supervisor_memo_id: int | None = None
    supervisor_error: str | None = None
    skipped_supervisor: bool = False
```

3c. Add `"supervisor_error"` to the error-aggregation tuple/list (the one currently containing `"learning_error", "pipeline_error", ...` at ~line 61).

3d. Add the stage function (place it before `_run_pipeline_stage`):

```python
def _run_supervisor_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
) -> None:
    """Stage 0.5 — observer-only critic that records the watchlist the acting
    stages must resolve. Runs after the learning refresh (fresh attribution)
    and before ideation. Independent: a failure here does not abort the session;
    acting stages then resolve any items still open from a prior run.

    Cost is recorded in supervisor_memos by run_supervisor itself. We do not
    wrap this in capture_usage: run_supervisor's own capture_usage swaps the
    active accumulator, so an outer one would capture nothing.
    """
    if skip or "supervisor" in completed_stages:
        logger.info("[Stage 0.5] Supervisor — SKIPPED%s",
                    " (completed in prior run)" if "supervisor" in completed_stages else "")
        result.skipped_supervisor = True
        return
    logger.info("[Stage 0.5] Running strategy supervisor")
    _start_stage(session_id, "supervisor")
    try:
        result.supervisor_memo_id = run_supervisor()
        _complete_stage(session_id, "supervisor")
    except Exception as e:
        result.supervisor_error = str(e)
        _fail_stage(session_id, "supervisor", str(e))
        logger.error("Supervisor failed: %s — continuing; acting stages use existing watchlist", e)
```

3e. Call it in `run_session` between learning and pipeline (after line 525):

```python
        attribution_constraints = _run_learning_refresh(result, session_id, completed_stages)
        _run_supervisor_stage(result, session_id, completed_stages, skip_supervisor)
        _run_pipeline_stage(result, session_id, completed_stages, skip_pipeline, pipeline_hours, pipeline_limit)
```

3f. Add a `skip_supervisor` parameter to `run_session` (default `False`) and a `--skip-supervisor` CLI flag in `main()`, mirroring the existing `skip_pipeline` / `--skip-pipeline` wiring. Paper runs may pass `skip_supervisor=True` if desired, but per the spec it runs every session by default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "Run supervisor as session Stage 0.5 (after learning, before ideation)"
```

---

## Task 6: Reflection ingests + gates its watchlist items

**Files:**
- Modify: `v2/strategy.py`
- Test: append to `tests/v2/test_strategy.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_strategy.py`:

```python
from v2 import strategy as strat


def test_reflection_registers_resolve_tool_and_gates(monkeypatch):
    # Loop does nothing; an open reflection item remains -> gate must raise.
    monkeypatch.setattr(strat, "get_rule_gate_revalidation_candidates", lambda days=30: [])
    monkeypatch.setattr(strat, "build_formation_context", lambda: "")
    monkeypatch.setattr(strat, "get_claude_client", lambda: MagicMock())
    fake_result = MagicMock(messages=[], input_tokens=1, output_tokens=1, turns_used=1)
    monkeypatch.setattr(strat, "run_agentic_loop", lambda **k: fake_result)
    monkeypatch.setattr(strat, "_count_actions", lambda msgs: (0, 0, False, True))
    # one open reflection item that the (no-op) loop never resolves
    monkeypatch.setattr("v2.watchlist.db.get_open_watchlist_items",
                        lambda stage: [{"id": 9, "title": "Retire Rule 43", "detail": "d"}]
                        if stage == "reflection" else [])
    with pytest.raises(RuntimeError) as exc:
        strat.run_strategy_reflection(session_id=5)
    assert "9" in str(exc.value)


def test_reflection_passes_when_items_resolved(monkeypatch):
    monkeypatch.setattr(strat, "get_rule_gate_revalidation_candidates", lambda days=30: [])
    monkeypatch.setattr(strat, "build_formation_context", lambda: "")
    monkeypatch.setattr(strat, "get_claude_client", lambda: MagicMock())
    fake_result = MagicMock(messages=[], input_tokens=1, output_tokens=1, turns_used=1)
    monkeypatch.setattr(strat, "run_agentic_loop", lambda **k: fake_result)
    monkeypatch.setattr(strat, "_count_actions", lambda msgs: (0, 0, False, True))
    monkeypatch.setattr("v2.watchlist.db.get_open_watchlist_items", lambda stage: [])
    res = strat.run_strategy_reflection(session_id=5)  # no raise
    assert res.memo_written is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_strategy.py -v -k watchlist or "resolve_tool"`
Expected: FAIL (gate not present; resolve tool not registered)

- [ ] **Step 3: Implement in `v2/strategy.py`**

3a. Add imports near the top:

```python
from v2 import watchlist as wl
from v2.database.trading_db import get_open_watchlist_items
```

3b. Register the resolve tool def. Add `wl.RESOLVE_WATCHLIST_TOOL_DEF` to `STRATEGY_TOOL_DEFINITIONS` (append to the list at ~line 605).

3c. In `run_strategy_reflection`, after `gated_rule_ids = _gated_rule_ids(...)` and before building `initial_parts`, load the items:

```python
    open_watchlist = get_open_watchlist_items("reflection")
```

Add to `initial_parts` (before the "Begin your strategy reflection" block):

```python
    initial_parts.append(wl.format_open_watchlist_items(open_watchlist))
    initial_parts.append("")
```

3d. Register the bound handler. In the `handlers = {**STRATEGY_TOOL_HANDLERS, ...}` dict, add:

```python
        "resolve_watchlist_item": wl.make_resolve_handler(
            session_id=session_id, stage="reflection"
        ),
```

3e. Add the gate after the existing `missing_revalidations` check (after line 787):

```python
    wl.assert_watchlist_resolved("reflection")
```

3f. Add one line to `STRATEGY_REFLECTION_SYSTEM` after the "Rule Revalidation Gate" section:

```
## Supervisor Watchlist Gate

If the initial context lists a SUPERVISOR WATCHLIST, you MUST call
resolve_watchlist_item for every listed item before writing your memo —
'acted' if you made a change in response, or 'dismissed' with a reason if
no change is warranted. You cannot finish with an open item.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_strategy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2/strategy.py tests/v2/test_strategy.py
git commit -m "Reflection ingests + hard-gates its supervisor watchlist items"
```

---

## Task 7: Ideation ingests + gates its watchlist items

**Files:**
- Modify: `v2/ideation_claude.py`
- Test: append to `tests/v2/test_ideation_claude.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_ideation_claude.py`:

```python
from v2 import ideation_claude as ic


def test_strategist_gates_on_open_ideation_items(monkeypatch):
    monkeypatch.setattr(ic, "build_formation_context", lambda orphans=None: "")
    monkeypatch.setattr(ic, "get_orphan_positions", lambda: [])
    monkeypatch.setattr(ic, "_build_pre_seeded_context", lambda: "STATE")
    monkeypatch.setattr(ic, "_build_orphan_block", lambda orphans=None: ("", "", 0))
    monkeypatch.setattr(ic, "get_claude_client", lambda: MagicMock())
    fake = MagicMock(messages=[], turns_used=1, input_tokens=1, output_tokens=1)
    monkeypatch.setattr(ic, "run_agentic_loop", lambda **k: fake)
    monkeypatch.setattr("v2.watchlist.db.get_open_watchlist_items",
                        lambda stage: [{"id": 4, "title": "Close thesis 267", "detail": "d"}]
                        if stage == "ideation" else [])
    with pytest.raises(RuntimeError) as exc:
        ic.run_strategist_loop(session_id=7)
    assert "4" in str(exc.value)


def test_strategist_passes_when_ideation_items_clear(monkeypatch):
    monkeypatch.setattr(ic, "build_formation_context", lambda orphans=None: "")
    monkeypatch.setattr(ic, "get_orphan_positions", lambda: [])
    monkeypatch.setattr(ic, "_build_pre_seeded_context", lambda: "STATE")
    monkeypatch.setattr(ic, "_build_orphan_block", lambda orphans=None: ("", "", 0))
    monkeypatch.setattr(ic, "get_claude_client", lambda: MagicMock())
    fake = MagicMock(messages=[], turns_used=1, input_tokens=1, output_tokens=1)
    monkeypatch.setattr(ic, "run_agentic_loop", lambda **k: fake)
    monkeypatch.setattr("v2.watchlist.db.get_open_watchlist_items", lambda stage: [])
    ic.run_strategist_loop(session_id=7)  # no raise
```

(If `_run_claude_loop` rather than `run_agentic_loop` is the seam that returns the result, patch `ic._run_claude_loop` to return `fake` instead — match the real call path in the file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_ideation_claude.py -v -k watchlist or gates`
Expected: FAIL (gate not present)

- [ ] **Step 3: Implement in `v2/ideation_claude.py`**

3a. Add imports:

```python
from v2 import watchlist as wl
from v2.database.trading_db import get_open_watchlist_items
```

3b. Register the resolve tool def + handler. In `_run_claude_loop` (the shared loop runner at ~line 211 that builds `handlers = {**TOOL_HANDLERS, ...}` and passes `tools=TOOL_DEFINITIONS`), add the resolve tool to the tools list and the bound handler to the handlers dict. Since `_run_claude_loop` receives `session_id`, bind there:

```python
    handlers = {
        **TOOL_HANDLERS,
        "resolve_watchlist_item": wl.make_resolve_handler(
            session_id=session_id, stage="ideation"
        ),
    }
    tools = TOOL_DEFINITIONS + [wl.RESOLVE_WATCHLIST_TOOL_DEF]
    result = run_agentic_loop(
        ...
        tools=tools,
        tool_handlers=handlers,
        ...
    )
```

3c. In `run_strategist_loop`, load open items and inject into `initial_message`. After `pre_seeded = ...` (line 433-436), add:

```python
    open_watchlist = get_open_watchlist_items("ideation")
    watchlist_block = wl.format_open_watchlist_items(open_watchlist)
```

Insert `watchlist_block` into the `initial_message` f-string, e.g. right after the pre-seeded state:

```python
    initial_message = f"""Here is the current state (pre-loaded to save round-trips):

{pre_seeded}

{watchlist_block}

Now proceed with your strategist session:
... (unchanged numbered steps) ...
When you've completed your work, provide a summary of your findings and actions."""
```

3d. Add the gate. `run_strategist_loop` returns `_run_claude_loop(...)`. Capture the result and gate before returning:

```python
    result = _run_claude_loop(
        system=base_prompt,
        initial_message=initial_message,
        model=model,
        max_turns=max_turns,
        label="Strategist Loop",
        session_id=session_id,
        stage_name="ideation",
    )
    wl.assert_watchlist_resolved("ideation")
    return result
```

3e. Add a line to the strategist system prompt (`CLAUDE_SESSION_STRATEGIST_SYSTEM`, wherever it's defined) instructing watchlist resolution:

```
## Supervisor Watchlist Gate
If your initial context lists a SUPERVISOR WATCHLIST, you MUST call
resolve_watchlist_item for every item before finishing — 'acted' (you
closed/updated a thesis or re-spec'd the playbook in response) or
'dismissed' (with a reason). You cannot finish with an open item.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_ideation_claude.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2/ideation_claude.py tests/v2/test_ideation_claude.py
git commit -m "Ideation ingests + hard-gates its supervisor watchlist items"
```

---

## Task 8: `amend_rule` tool (stops retire-and-replace churn)

**Files:**
- Modify: `v2/strategy.py`
- Test: append to `tests/v2/test_strategy.py`

(The DB helper `amend_strategy_rule` was added in Task 2.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_strategy.py`:

```python
def test_amend_rule_updates_in_place(monkeypatch):
    monkeypatch.setattr(strat, "amend_strategy_rule", lambda **k: True)
    out = strat.tool_amend_rule(
        rule_id=48, new_rule_text="updated", new_evidence="beat 61% n=24",
        reason="evidence refresh",
    )
    assert "48" in out and "amend" in out.lower()


def test_amend_rule_reports_inactive(monkeypatch):
    monkeypatch.setattr(strat, "amend_strategy_rule", lambda **k: False)
    out = strat.tool_amend_rule(
        rule_id=999, new_rule_text="x", new_evidence="y", reason="z",
    )
    assert "Error" in out


def test_amend_rule_in_tool_defs():
    names = {d["name"] for d in strat.STRATEGY_TOOL_DEFINITIONS}
    assert "amend_rule" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_strategy.py -v -k amend`
Expected: FAIL (`AttributeError: ... 'tool_amend_rule'`)

- [ ] **Step 3: Implement in `v2/strategy.py`**

3a. Import the DB helper (extend the existing `from .database.trading_db import (...)` block):

```python
    amend_strategy_rule,
```

3b. Add the handler (near `tool_retire_rule`):

```python
def tool_amend_rule(
    rule_id: int, new_rule_text: str, new_evidence: str, reason: str
) -> str:
    """Update an active rule's text/evidence in place — no retire-and-replace.
    Use this when only the embedded evidence changed (e.g. attribution
    refreshed) and the rule's substance is unchanged."""
    new_rule_text = (new_rule_text or "").strip()
    new_evidence = (new_evidence or "").strip()
    if not new_rule_text:
        return "Error: new_rule_text is required"
    if len(new_evidence) < MIN_RULE_EVIDENCE_CHARS:
        return "Error: new_evidence is required"
    ok = amend_strategy_rule(
        rule_id=rule_id, new_rule_text=new_rule_text,
        new_evidence=new_evidence, reason=(reason or "").strip(),
    )
    if not ok:
        return f"Error: Rule ID {rule_id} not found or not active"
    logger.info("Amended rule %s in place: %s", rule_id, reason)
    return f"Amended rule ID {rule_id} in place (no new rule created)."
```

3c. Add the tool def to `STRATEGY_TOOL_DEFINITIONS`:

```python
    {
        "name": "amend_rule",
        "description": (
            "Update an active rule's text/evidence IN PLACE without retiring it. "
            "Prefer this over retire_rule + propose_rule when only the embedded "
            "evidence changed and the rule's substance is the same — it avoids "
            "rule-churn (new ids for the same rule every attribution refresh)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "integer"},
                "new_rule_text": {"type": "string"},
                "new_evidence": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["rule_id", "new_rule_text", "new_evidence", "reason"],
        },
    },
```

3d. Register the handler in `STRATEGY_TOOL_HANDLERS`:

```python
    "amend_rule": tool_amend_rule,
```

3e. Add guidance to `STRATEGY_REFLECTION_SYSTEM` under "Rule Management":

```
When attribution data refreshes but a rule's substance is unchanged, use
amend_rule to update its evidence in place. Do NOT retire-and-replace a
rule just to refresh its evidence string — that churns rule ids for no
behavioral change.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_strategy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2/strategy.py tests/v2/test_strategy.py
git commit -m "Add amend_rule tool to stop retire-and-replace rule churn"
```

---

## Task 9: Full suite + lint, then data-hygiene rehearsal

**Files:** none (verification + documentation)

- [ ] **Step 1: Run the full v2 suite + lint**

Run:
```bash
docker compose exec -T trading python3 -m pytest tests/v2/ -q
docker compose exec -T trading ruff check .
```
Expected: all pass; ruff clean.

- [ ] **Step 2: Dry-run the supervisor against real data to confirm it records items**

Run: `docker compose exec -T trading python3 -m v2.supervisor --dry-run`
Expected: prints the memo AND a "watchlist item(s) (dry-run, not persisted)" block listing items with their `owner_stage`. (No DB writes.)

- [ ] **Step 3: Document the data-hygiene execution decision (Rule 43, theses 233/267, Rule 49)**

The spec leaves the choice between the loop path and the manual path to implementation time. Decide and record in the PR description which path you took. Reference for the manual path (PROD db, run only if you choose manual):
```sql
-- Rule 43: retire (auto-lift round-trip count is < 6)
-- Theses 233, 267: close as ungrounded once Rule 43 is gone
-- Rule 49: retire if still n<5 / beat-rate unrecovered
```
Do not run these blindly — confirm the live numbers first with
`get_flip_flop_report` and the rule bind history. **No prod writes are part of this plan; this step is a decision + documentation gate.**

- [ ] **Step 4: Commit any doc/notes if produced** (otherwise skip).

---

## Phase 5 (separate investigation): Diagnostics

These are investigations, not TDD tasks. Do them after the loop lands, or in parallel — they don't block Tasks 1-9.

- [ ] **Investigate silent stage failures** in sessions 4642 (2026-05-22, 2 failures) and 4644 (2026-05-25, 1 failure):
```bash
U=$(grep -m1 POSTGRES_USER .env | cut -d= -f2); D=$(grep -m1 POSTGRES_DB .env | cut -d= -f2)
docker compose exec -T db psql -U "$U" -d "$D" -c \
  "SELECT session_id, stage_name, status, error, started_at, completed_at
   FROM session_stages WHERE session_id IN (4642, 4644) ORDER BY session_id, started_at;"
```
Classify each failure. If it's a code bug, write a follow-up plan; if transient (API timeout), note and close.

- [ ] **Investigate the duplicate-session / cost artifact on 2026-05-22** (two records, $4.42 and $0.88):
```bash
docker compose exec -T db psql -U "$U" -d "$D" -c \
  "SELECT id, session_date, created_at, status FROM sessions
   WHERE session_date = '2026-05-22' ORDER BY created_at;"
```
Determine whether two `sessions` rows exist for one date (resume/re-run artifact) and whether the cost split is expected. Document findings; fix only if a code bug produced the duplicate.

---

## Self-Review (completed during planning)

- **Spec coverage:** Phase 1 (structured watchlist + session wiring) → Tasks 1, 4, 5. Phase 2 (ingestion + gate) → Tasks 3, 6, 7. Phase 3 (data hygiene, both paths) → Task 9 Step 3. Phase 4 (amend-in-place) → Tasks 2, 8. Phase 5 (diagnostics) → Phase 5 section. Stage ownership table → enforced by `owner_stage` scoping in Task 2 `resolve_watchlist_item` + per-stage handlers in Tasks 6/7. All spec testing bullets map to tests in Tasks 2-8.
- **Placeholder scan:** every code step contains full code; no TBD/TODO.
- **Type consistency:** `record_watchlist_item`, `get_open_watchlist_items`, `resolve_watchlist_item`, `amend_strategy_rule` signatures are identical across Task 2 (def), Task 3 (`make_resolve_handler` call), Task 4 (`db.record_watchlist_item` call), Tasks 6/7 (`get_open_watchlist_items`), Task 8 (`amend_strategy_rule`). Tool name `resolve_watchlist_item` and `amend_rule` consistent across defs, handlers, and tests. The spec's separate `resolution` column was folded into `status` (documented at top).
