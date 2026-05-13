# Per-Run Session ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every agent invocation create its own row in `sessions` (regardless of date or type) and rewire every "session detail" path to use that row's `id` instead of joining by date.

**Architecture:** Drop the `UNIQUE (session_date, session_type)` constraint. Add a nullable `session_id INT REFERENCES sessions(id)` column to `decisions`, `theses`, `strategy_memos`, `tweets`. Backfill historical rows from the current one-to-one date mapping. Thread `session_id` from each entry point (daily session, premarket, weekly mistakes/attribution, entertainment) through to the `insert_*` calls. Rewrite the five date-based JOINs in `dashboard/queries.py` to join on `session_id`. Rework `_check_and_record_session` so the existence check is the only gate (`--force` bypasses it); the unsafe plain INSERT is gone because the unique constraint is gone.

**Tech Stack:** PostgreSQL 16, psycopg2, raw SQL, pytest.

---

## File Structure

**New file:**
- `db/init/029_session_id_propagation.sql` — schema migration: add `session_id` columns, backfill, drop the unique constraint.

**Modified files (DB layer):**
- `v2/database/trading_db.py` — add `session_id` parameter to `insert_decision`, `insert_thesis`, `insert_strategy_memo`, `insert_tweet`; update `insert_session_record` doctring (no behavior change).

**Modified files (daily session plumbing):**
- `v2/session.py` — rework `_check_and_record_session`: keep idempotency check, drop resume/`completed_stages`, drop the post-check plain INSERT (it's now the only INSERT). Stop importing/using `get_completed_stages`.
- `v2/trader.py` — pass `session_id` into `insert_decision` via `_insert_decision_with_retry`.
- `v2/tools.py` — accept `session_id` in `tool_create_thesis` / `tool_add_thesis` handlers and pass to `insert_thesis`.
- `v2/ideation_claude.py` — bind `session_id` into the thesis-creation tool handlers (same pattern as the existing telemetry binding in `strategy.py`).
- `v2/strategy.py` — pass `session_id` into `insert_strategy_memo` from the reflection path and the `tool_write_strategy_memo` handler.
- `v2/twitter.py`, `v2/bluesky.py`, `v2/social_trades.py` — accept `session_id` in their stage entry points and pass to every `insert_tweet`.

**Modified files (cron-driven scripts, each creates its own session row):**
- `v2/premarket.py` — create a `session_type='premarket'` session at start; complete/fail it; pass `session_id` to `insert_tweet`.
- `v2/social_weekly.py` — create a `session_type='weekly_mistakes'` or `'weekly_attribution'` session per subcommand; complete/fail; pass `session_id` to `insert_tweet`.
- `v2/entertainment.py` — create a `session_type='entertainment'` session per run (no idempotency gate); complete/fail; pass `session_id` to `insert_tweet`.

**Modified files (dashboard):**
- `dashboard/queries.py` — five queries (lines 431, 563, 578, 591, 607) switch from date-based JOINs to `session_id`-based JOINs.

**Modified tests:**
- `tests/v2/test_db.py` — `TestSessionTracking` tests; insert_* signature tests.
- `tests/v2/test_session.py` — drop `completed_stages` resume tests; update `--force` tests; assert `session_id` flows to inserts.
- `tests/v2/test_trader.py` — assert `session_id` is passed to `insert_decision`.
- `tests/v2/test_tools.py` — assert `session_id` flows into `insert_thesis`.
- `tests/v2/test_strategy.py` — assert `session_id` flows into `insert_strategy_memo`.
- `tests/v2/test_twitter.py`, `test_bluesky.py`, `test_social_trades.py` — assert `session_id` flows into `insert_tweet`.
- `tests/v2/test_premarket.py`, `test_social_weekly.py`, `test_entertainment.py` — assert session row is created with the right `session_type` and threaded into `insert_tweet`.
- `tests/dashboard/test_queries.py` — update fixtures to set `session_id` on rows; rewrite query expectations.

---

## Task 1: Schema migration

**Files:**
- Create: `db/init/029_session_id_propagation.sql`

- [ ] **Step 1: Write the migration**

Create `db/init/029_session_id_propagation.sql`:

```sql
-- Per-run session IDs: thread sessions.id into every table the dashboard
-- currently joins to sessions by date, then drop the per-date uniqueness
-- constraint so multiple session rows can coexist for one date.

ALTER TABLE decisions
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);
ALTER TABLE theses
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);
ALTER TABLE strategy_memos
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);
ALTER TABLE tweets
    ADD COLUMN IF NOT EXISTS session_id INT REFERENCES sessions(id);

CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_theses_session ON theses(session_id);
CREATE INDEX IF NOT EXISTS idx_strategy_memos_session ON strategy_memos(session_id);
CREATE INDEX IF NOT EXISTS idx_tweets_session ON tweets(session_id);

-- Backfill. UNIQUE (session_date, session_type) guarantees at most one
-- daily session per date pre-migration, so this is deterministic.
UPDATE decisions d
SET session_id = s.id
FROM sessions s
WHERE d.session_id IS NULL
  AND s.session_date = d.date
  AND s.session_type = 'daily';

UPDATE theses t
SET session_id = s.id
FROM sessions s
WHERE t.session_id IS NULL
  AND s.session_date = t.created_at::date
  AND s.session_type = 'daily';

UPDATE strategy_memos m
SET session_id = s.id
FROM sessions s
WHERE m.session_id IS NULL
  AND s.session_date = m.session_date
  AND s.session_type = 'daily';

UPDATE tweets tw
SET session_id = s.id
FROM sessions s
WHERE tw.session_id IS NULL
  AND s.session_date = tw.session_date
  AND s.session_type = 'daily';

ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_session_date_session_type_key;
```

- [ ] **Step 2: Apply migration locally**

Run: `docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/029_session_id_propagation.sql`

(If running off-stack: `docker compose down && docker compose up -d db` — init scripts only run on a fresh volume, so for an existing volume, apply manually via `psql -f`.)

Expected: no errors. `\d decisions` shows new `session_id` column with FK to `sessions(id)`.

- [ ] **Step 3: Verify backfill**

Run:
```bash
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
  SELECT 'decisions' AS t, COUNT(*) FILTER (WHERE session_id IS NULL) AS nulls, COUNT(*) AS total FROM decisions
  UNION ALL SELECT 'theses', COUNT(*) FILTER (WHERE session_id IS NULL), COUNT(*) FROM theses
  UNION ALL SELECT 'strategy_memos', COUNT(*) FILTER (WHERE session_id IS NULL), COUNT(*) FROM strategy_memos
  UNION ALL SELECT 'tweets', COUNT(*) FILTER (WHERE session_id IS NULL), COUNT(*) FROM tweets;
"
```

Expected: any row whose date matches an existing daily session has `session_id` set. Remaining NULLs (if any) are rows from dates with no `sessions` row — leave NULL.

- [ ] **Step 4: Commit**

```bash
git add db/init/029_session_id_propagation.sql
git commit -m "feat(db): add session_id FK columns + drop sessions unique constraint"
```

---

## Task 2: Add session_id to insert_decision

**Files:**
- Modify: `v2/database/trading_db.py:159-174`
- Test: `tests/v2/test_db.py` (existing TestDecisions class)

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_db.py` (find the existing test class for decisions; if no class exists, group under `TestDecisions`):

```python
def test_insert_decision_passes_session_id(self, mock_db, mock_cursor):
    from datetime import date
    mock_cursor.fetchone.return_value = {"id": 1}
    from v2.database.trading_db import insert_decision
    insert_decision(
        decision_date=date(2026, 5, 13), ticker="AAPL", action="buy",
        quantity=10, price=200.0, reasoning="t", signals_used=[],
        account_equity=10000, buying_power=5000, session_id=42,
    )
    sql = mock_cursor.execute.call_args[0][0]
    params = mock_cursor.execute.call_args[0][1]
    assert "session_id" in sql
    assert 42 in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_db.py::TestDecisions::test_insert_decision_passes_session_id -v`
Expected: FAIL (parameter not accepted or SQL missing `session_id`).

- [ ] **Step 3: Update `insert_decision`**

Replace `v2/database/trading_db.py:159-174` with:

```python
def insert_decision(decision_date, ticker, action, quantity, price, reasoning, signals_used, account_equity, buying_power, playbook_action_id=None, is_off_playbook=False, order_id=None, session_id=None) -> int:
    """
    Insert a trading decision.

    V3 additions:
    - playbook_action_id: Links decision to a specific playbook action
    - is_off_playbook: Marks decisions made outside the playbook
    - order_id: Alpaca order ID for trade verification
    - session_id: FK to sessions.id (per-run; nullable for legacy rows)
    """
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO decisions (date, ticker, action, quantity, price, reasoning, signals_used, account_equity, buying_power, playbook_action_id, is_off_playbook, order_id, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (decision_date, ticker, action, quantity, price, reasoning, Json(signals_used), account_equity, buying_power, playbook_action_id, is_off_playbook, order_id, session_id))
        return cur.fetchone()["id"]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_db.py -k "decision" -v`
Expected: new test PASSES; existing decision tests still pass (session_id defaults to None).

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "feat(db): thread session_id through insert_decision"
```

---

## Task 3: Add session_id to insert_thesis

**Files:**
- Modify: `v2/database/trading_db.py:329-336`
- Test: `tests/v2/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
def test_insert_thesis_passes_session_id(self, mock_db, mock_cursor):
    mock_cursor.fetchone.return_value = {"id": 1}
    from v2.database.trading_db import insert_thesis
    insert_thesis(ticker="AAPL", direction="long", thesis="t", session_id=42)
    sql = mock_cursor.execute.call_args[0][0]
    params = mock_cursor.execute.call_args[0][1]
    assert "session_id" in sql
    assert 42 in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_db.py -k "test_insert_thesis_passes_session_id" -v`
Expected: FAIL.

- [ ] **Step 3: Update `insert_thesis`**

Replace `v2/database/trading_db.py:329-336`:

```python
def insert_thesis(ticker, direction, thesis, entry_trigger=None, exit_trigger=None, invalidation=None, confidence="medium", source="ideation", session_id=None) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO theses (ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source, session_id))
        return cur.fetchone()["id"]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_db.py -k "thesis" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "feat(db): thread session_id through insert_thesis"
```

---

## Task 4: Add session_id to insert_strategy_memo

**Files:**
- Modify: `v2/database/trading_db.py:822-829`
- Test: `tests/v2/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
def test_insert_strategy_memo_passes_session_id(self, mock_db, mock_cursor):
    from datetime import date
    mock_cursor.fetchone.return_value = {"id": 1}
    from v2.database.trading_db import insert_strategy_memo
    insert_strategy_memo(
        session_date=date(2026, 5, 13), memo_type="notes",
        content="x", session_id=42,
    )
    sql = mock_cursor.execute.call_args[0][0]
    params = mock_cursor.execute.call_args[0][1]
    assert "session_id" in sql
    assert 42 in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_db.py -k "test_insert_strategy_memo_passes_session_id" -v`
Expected: FAIL.

- [ ] **Step 3: Update `insert_strategy_memo`**

Replace `v2/database/trading_db.py:822-829`:

```python
def insert_strategy_memo(session_date, memo_type, content, strategy_state_id=None, session_id=None) -> int:
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO strategy_memos (session_date, memo_type, content, strategy_state_id, session_id)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, memo_type, content, strategy_state_id, session_id))
        return cur.fetchone()["id"]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_db.py -k "memo" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "feat(db): thread session_id through insert_strategy_memo"
```

---

## Task 5: Add session_id to insert_tweet

**Files:**
- Modify: `v2/database/trading_db.py:844-870`
- Test: `tests/v2/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
def test_insert_tweet_passes_session_id(self, mock_db, mock_cursor):
    from datetime import date
    mock_cursor.fetchone.return_value = {"id": 1}
    from v2.database.trading_db import insert_tweet
    insert_tweet(
        session_date=date(2026, 5, 13), tweet_type="recap",
        tweet_text="t", session_id=42,
    )
    sql = mock_cursor.execute.call_args[0][0]
    params = mock_cursor.execute.call_args[0][1]
    assert "session_id" in sql
    assert 42 in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_db.py -k "test_insert_tweet_passes_session_id" -v`
Expected: FAIL.

- [ ] **Step 3: Update `insert_tweet`**

Replace `v2/database/trading_db.py:844-870`:

```python
def insert_tweet(
    session_date,
    tweet_type: str,
    tweet_text: str,
    tweet_id: str | None = None,
    posted: bool = False,
    error: str | None = None,
    platform: str = "twitter",
    decision_id: int | None = None,
    session_id: int | None = None,
) -> int:
    """Log a tweet/post to the audit table.

    decision_id ties a per-trade post back to its source decision.
    session_id ties the row to the sessions.id of the run that produced it
    (per-run; nullable for legacy rows).
    """
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (
                session_date, tweet_type, tweet_text, tweet_id,
                posted, error, platform, decision_id, session_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id,
              posted, error, platform, decision_id, session_id))
        return cur.fetchone()["id"]
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_db.py -k "tweet" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "feat(db): thread session_id through insert_tweet"
```

---

## Task 6: Rework `_check_and_record_session` in session.py

The current function plays three roles: (a) idempotency gate, (b) resume-from-partial via `get_completed_stages`, (c) plain `INSERT` that's now the only INSERT (no unique constraint to violate). Per the design choice, role (b) goes away — every run is its own session. Role (a) stays. Role (c) becomes a clean, never-failing insert.

**Files:**
- Modify: `v2/session.py:37` (import), `v2/session.py:166-191` (function body), `v2/session.py:569` (caller — `completed_stages` is no longer used).
- Test: `tests/v2/test_session.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_session.py` near the existing force-flag tests:

```python
def test_force_creates_new_session_row_when_one_already_exists(self):
    """Per-run uniqueness: --force inserts a brand-new sessions row
    even when a completed session already exists for today."""
    from v2.session import _check_and_record_session
    from datetime import date
    with patch("v2.session.insert_session_record", return_value=99) as mock_insert, \
         patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "completed"}):
        session_id, completed, err = _check_and_record_session(force=True, session_date=date(2026, 5, 13))
    assert session_id == 99
    assert completed == set()
    assert err is None
    mock_insert.assert_called_once()

def test_no_force_skips_when_completed_session_exists(self):
    from v2.session import _check_and_record_session
    from datetime import date
    with patch("v2.session.insert_session_record") as mock_insert, \
         patch("v2.session.get_session_for_date", return_value={"id": 7, "status": "completed"}):
        session_id, completed, err = _check_and_record_session(force=False, session_date=date(2026, 5, 13))
    assert session_id is None
    assert err is not None
    mock_insert.assert_not_called()

def test_no_force_creates_new_session_when_prior_was_failed(self):
    """A failed prior session does not gate; a fresh run gets its own
    session row (no resume, no stage skipping)."""
    from v2.session import _check_and_record_session
    from datetime import date
    with patch("v2.session.insert_session_record", return_value=12) as mock_insert, \
         patch("v2.session.get_session_for_date", return_value={"id": 5, "status": "failed"}):
        session_id, completed, err = _check_and_record_session(force=False, session_date=date(2026, 5, 13))
    assert session_id == 12
    assert completed == set()
    assert err is None
    mock_insert.assert_called_once()
```

Also locate and **delete** the existing tests that assert resume-from-`completed_stages` behavior — search `tests/v2/test_session.py` for `completed_stages` and `get_completed_stages` and remove those test bodies (or update them to assert the new always-fresh behavior). Common ones to remove or rewrite: any test that mocks `get_completed_stages` returning a non-empty set.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_session.py -k "force_creates_new_session_row_when_one_already_exists or no_force_creates_new_session_when_prior_was_failed" -v`
Expected: FAIL (current code returns `completed_stages` for in-progress sessions, doesn't call insert on partial).

- [ ] **Step 3: Update `_check_and_record_session`**

In `v2/session.py`, remove `get_completed_stages` from the import at line 37 (it's no longer used). Replace lines 166-191 with:

```python
def _check_and_record_session(force: bool, session_date) -> tuple[int | None, set, str | None]:
    """Returns (session_id, completed_stages, early_error).

    Per-run sessions: every invocation creates a new sessions row.
    completed_stages is always the empty set — no resume across runs.

    Idempotency: if force=False and a session of session_type='daily'
    is already 'completed' for this date, we skip with early_error set.
    --force bypasses that gate.
    """
    if not force:
        try:
            existing = get_session_for_date(session_date)
            if existing and existing["status"] == "completed":
                logger.warning("Session already completed for %s. Use --force to override.", session_date)
                return None, set(), f"Session already completed for {session_date}"
        except Exception as e:
            logger.warning("Could not check session status: %s — proceeding", e)
    try:
        session_id = insert_session_record(session_date)
        logger.info("Session ID: %d", session_id)
        return session_id, set(), None
    except Exception as e:
        logger.warning("Could not create session record: %s — proceeding without tracking", e)
        return None, set(), None
```

- [ ] **Step 4: Audit the caller for `completed_stages` usage**

Read `v2/session.py:569` and the lines below. The returned `completed_stages` set was previously checked before each stage to decide whether to skip. Since we now always return `set()`, those skip-checks become dead branches but are still safe (the set is empty, the conditions are false, every stage runs). **Leave the skip-check call sites in place** — they're harmless and removing them is out of scope here. If you spot one that confused you, add a one-line comment, but do not refactor.

(If you find any place that bombs because `completed_stages` is required to be non-empty, that's a bug — fix it locally.)

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/v2/test_session.py -v`
Expected: new tests PASS; any old resume-flavored tests you replaced are gone or rewritten. No tests left red.

- [ ] **Step 6: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "feat(session): per-run sessions; drop resume; --force creates new row"
```

---

## Task 7: Thread session_id from trader into insert_decision

`v2/trader.py` already carries `session_id` through `run_trader → _execute_decisions → _insert_decision_with_retry` (see lines 538, 712, 1060). Only the final hop into `insert_decision` is missing.

**Files:**
- Modify: `v2/trader.py:916-940` (function `_insert_decision_with_retry`)
- Test: `tests/v2/test_trader.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_trader.py`:

```python
def test_insert_decision_receives_session_id(self):
    from v2.trader import _insert_decision_with_retry
    with patch("v2.trader.insert_decision", return_value=1) as mock_ins:
        _insert_decision_with_retry(
            payload={
                "decision_date": date(2026, 5, 13), "ticker": "AAPL",
                "action": "buy", "quantity": 10, "price": 200.0,
                "reasoning": "t", "signals_used": [],
                "account_equity": 10000, "buying_power": 5000,
            },
            session_id=42,
        )
    kwargs = mock_ins.call_args.kwargs
    assert kwargs.get("session_id") == 42
```

(Adjust the keyword names to match the actual `_insert_decision_with_retry` signature — read v2/trader.py:916-940 first.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_trader.py -k "test_insert_decision_receives_session_id" -v`
Expected: FAIL.

- [ ] **Step 3: Update `_insert_decision_with_retry`**

Read v2/trader.py:916-940 and add `session_id: int | None = None` to the signature. In the `insert_decision(**payload)` call body, merge `session_id` into the call: replace `return insert_decision(**payload)` with `return insert_decision(**payload, session_id=session_id)`.

Then update the call site at v2/trader.py:1036 to pass `session_id=session_id` (the variable already exists in `run_trader`'s scope per line 1060).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_trader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/trader.py tests/v2/test_trader.py
git commit -m "feat(trader): pass session_id to insert_decision"
```

---

## Task 8: Thread session_id into thesis-creation tool handlers

`v2/tools.py` has two `insert_thesis` calls (lines 180, 227) inside tool handler functions. These handlers are bound into a registry by `ideation_claude.py`. The existing telemetry pattern in `v2/strategy.py:592` uses `functools.partial`-style binding of `session_id` into a tool handler. Follow that pattern.

**Files:**
- Modify: `v2/tools.py:180, 227` (thesis-creation handlers) — accept `session_id` param.
- Modify: `v2/ideation_claude.py` — bind `session_id` into the handlers in the tool registry (find where these tools are registered; mirror the strategy.py:592 pattern).
- Test: `tests/v2/test_tools.py`

- [ ] **Step 1: Read context**

Open `v2/tools.py` around lines 175-235 to see both thesis handlers. Open `v2/ideation_claude.py` and find where these handlers are registered into the tool dispatch dict (look for "create_thesis" / "add_thesis" strings). Open `v2/strategy.py:585-605` for the binding pattern.

- [ ] **Step 2: Write the failing test**

Add to `tests/v2/test_tools.py`:

```python
def test_thesis_handler_passes_session_id_to_insert(self):
    from v2.tools import tool_create_thesis  # name may differ — match actual symbol
    with patch("v2.tools.insert_thesis", return_value=1) as mock_ins:
        tool_create_thesis(
            ticker="AAPL", direction="long", thesis="t",
            session_id=42,
        )
    assert mock_ins.call_args.kwargs.get("session_id") == 42
```

If the handler is `tool_add_thesis` or named differently, mirror against the actual name. Add a parallel test for the second handler at v2/tools.py:227.

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_tools.py -k "session_id_to_insert" -v`
Expected: FAIL (handler doesn't accept `session_id`).

- [ ] **Step 4: Update handlers**

In each of the two handlers in `v2/tools.py`, add `session_id: int | None = None` as a keyword param. Pass it through to `insert_thesis(..., session_id=session_id)`.

- [ ] **Step 5: Bind session_id at the registration site**

In `v2/ideation_claude.py`, where the thesis tools are registered, wrap each handler with a `functools.partial(handler, session_id=session_id)` (the registration site already has `session_id` in scope — see ideation_claude.py:213, 301, 395). Mirror exactly what `strategy.py:592` does for `tool_get_session_summary_with_telemetry`.

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/v2/test_tools.py tests/v2/test_ideation_claude.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add v2/tools.py v2/ideation_claude.py tests/v2/test_tools.py
git commit -m "feat(strategist): thread session_id into thesis-creation tools"
```

---

## Task 9: Thread session_id into strategy memo writes

`v2/strategy.py` has two `insert_strategy_memo` callers: the reflection path at line 268 (inside `tool_write_strategy_memo`) and `v2/session.py:272` (direct call). Both need `session_id`.

**Files:**
- Modify: `v2/strategy.py:263-274` (handler) — accept `session_id`.
- Modify: `v2/strategy.py` near line 558 — bind `session_id` into the handler registration (same partial pattern).
- Modify: `v2/session.py:272-278` — pass `session_id` directly.
- Test: `tests/v2/test_strategy.py`, `tests/v2/test_session.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/v2/test_strategy.py
def test_write_strategy_memo_tool_passes_session_id(self):
    from v2.strategy import tool_write_strategy_memo
    with patch("v2.strategy.insert_strategy_memo", return_value=1) as mock_ins, \
         patch("v2.strategy.get_current_strategy_state", return_value=None):
        tool_write_strategy_memo(memo_type="notes", content="t", session_id=42)
    assert mock_ins.call_args.kwargs.get("session_id") == 42

# tests/v2/test_session.py — add to the strategist-stage area
def test_strategist_memo_save_passes_session_id(self):
    # Locate the path that calls insert_strategy_memo from session.py:272
    # and assert session_id is forwarded.
    ...  # mirror existing strategist-stage tests
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_strategy.py -k "session_id" tests/v2/test_session.py -k "memo_save_passes_session_id" -v`
Expected: FAIL.

- [ ] **Step 3: Update `tool_write_strategy_memo`**

Replace v2/strategy.py:263-274 with:

```python
def tool_write_strategy_memo(memo_type: str, content: str, session_id: int | None = None) -> str:
    """Write a strategy reflection memo."""
    logger.info(f"Writing strategy memo ({memo_type})")
    current = get_current_strategy_state()
    state_id = current["id"] if current else None
    memo_id = insert_strategy_memo(
        session_date=date.today(),
        memo_type=memo_type,
        content=content,
        strategy_state_id=state_id,
        session_id=session_id,
    )
    return f"Memo written (ID: {memo_id})"
```

- [ ] **Step 4: Bind session_id at registration**

In `v2/strategy.py` (around the tool registry near line 590), wrap `tool_write_strategy_memo` with `functools.partial(tool_write_strategy_memo, session_id=session_id)`. The local `session_id` parameter is already in scope per strategy.py:558, 564, 604.

- [ ] **Step 5: Update direct call in session.py**

Replace v2/session.py:272-277 with:

```python
        insert_strategy_memo(
            session_date=session_date,
            memo_type='strategist_notes',
            content=summary,
            strategy_state_id=state['id'] if state else None,
            session_id=session_id,
        )
```

(`session_id` is already in scope — `run_strategist_stage` receives it per session.py:285.)

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/v2/test_strategy.py tests/v2/test_session.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add v2/strategy.py v2/session.py tests/v2/test_strategy.py tests/v2/test_session.py
git commit -m "feat(strategy): thread session_id into strategy_memos writes"
```

---

## Task 10: Thread session_id into daily-session tweet writes

The daily-session tweet writes are `v2/twitter.py:368` (recap), `v2/bluesky.py:418` (recap), and `v2/social_trades.py:244, 306, 336` (per-trade + quiet-day fallback). Their stage entry points are called from session.py with `session_id` in scope.

**Files:**
- Modify: `v2/twitter.py` — add `session_id` to `run_twitter_stage` (or equivalent entry point) and pass to `insert_tweet`.
- Modify: `v2/bluesky.py` — same for `run_bluesky_stage`.
- Modify: `v2/social_trades.py` — same for `run_trade_posts_stage`.
- Modify: `v2/session.py` stage 5 (around line 374 onward) — pass `session_id` to whichever entry point fires.
- Test: `tests/v2/test_twitter.py`, `tests/v2/test_bluesky.py`, `tests/v2/test_social_trades.py`.

- [ ] **Step 1: Identify each stage entry point**

Open each file and find the function name called from session.py for stage 5. Confirm parameter signature.

```bash
grep -n "def run_twitter_stage\|def run_bluesky_stage\|def run_trade_posts_stage\|def run_twitter\|def run_bluesky" v2/twitter.py v2/bluesky.py v2/social_trades.py
```

- [ ] **Step 2: Write the failing tests**

For each of the three files, add a test asserting `insert_tweet` is called with `session_id=42`. Mirror the test style of existing tests in that file.

Example for twitter.py:

```python
def test_recap_writes_tweet_with_session_id(self):
    from v2.twitter import run_twitter_stage  # adjust to actual name
    with patch("v2.twitter.insert_tweet", return_value=1) as mock_ins, \
         patch("v2.twitter.post_tweet", return_value="tweet_id_xyz"):
        run_twitter_stage(..., session_id=42)  # fill required args
    assert mock_ins.call_args.kwargs.get("session_id") == 42
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_twitter.py tests/v2/test_bluesky.py tests/v2/test_social_trades.py -k "session_id" -v`
Expected: FAIL.

- [ ] **Step 4: Update each stage**

In each entry point, add `session_id: int | None = None` to the signature. At every `insert_tweet(...)` call, pass `session_id=session_id`.

- [ ] **Step 5: Update session.py stage 5 dispatch**

In `v2/session.py` (Stage 5 — around the legacy `run_twitter_stage` / `run_bluesky_stage` call and the `ALGO_ENABLE_TRADE_POSTS=1` branch that calls `run_trade_posts_stage`), pass `session_id=session_id` to each.

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/v2/test_twitter.py tests/v2/test_bluesky.py tests/v2/test_social_trades.py tests/v2/test_session.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add v2/twitter.py v2/bluesky.py v2/social_trades.py v2/session.py \
        tests/v2/test_twitter.py tests/v2/test_bluesky.py tests/v2/test_social_trades.py tests/v2/test_session.py
git commit -m "feat(social): thread session_id into daily-session tweet writes"
```

---

## Task 11: Premarket — create its own session row

**Files:**
- Modify: `v2/premarket.py` — wrap the run in a session record (type `'premarket'`), thread session_id into the two `insert_tweet` calls.
- Test: `tests/v2/test_premarket.py`.

- [ ] **Step 1: Read existing structure**

Open `v2/premarket.py`. Find the main entry function (likely `run_premarket()` or `main()`) and the existing `posted_tweet_exists` idempotency guard. The new session-row idempotency check supersedes the inline guard for "did we already run today" — but the platform-level `posted_tweet_exists` may still be needed as a finer-grained dedup. Decide based on what the existing function does. Default: keep `posted_tweet_exists` for platform dedup; add `_check_and_record_session(force, today)`-style gating at the top.

- [ ] **Step 2: Write the failing test**

Add to `tests/v2/test_premarket.py`:

```python
def test_premarket_creates_session_row(self):
    from v2.premarket import run_premarket  # match actual entry point
    with patch("v2.premarket.insert_session_record", return_value=77) as mock_sess, \
         patch("v2.premarket.complete_session") as mock_complete, \
         patch("v2.premarket.insert_tweet", return_value=1) as mock_tweet, \
         patch("v2.premarket.posted_tweet_exists", return_value=False), \
         patch("v2.premarket.post_tweet", return_value="abc"):
        # ... fill other mocks for active theses, memo, etc. ...
        run_premarket(force=True)
    mock_sess.assert_called_once()
    # session_type='premarket' is the second positional or named arg
    args, kwargs = mock_sess.call_args
    assert kwargs.get("session_type") == "premarket" or "premarket" in args
    # insert_tweet receives session_id
    assert mock_tweet.call_args.kwargs.get("session_id") == 77
    mock_complete.assert_called_once_with(77)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_premarket.py -k "creates_session_row" -v`
Expected: FAIL.

- [ ] **Step 4: Wire session-row management into premarket**

In `v2/premarket.py`:

1. Import: `from .database.trading_db import insert_session_record, get_session_for_date, complete_session, fail_session`.
2. Add `force` parameter to the entry function if not present.
3. At the top of the function (after weekend/holiday skip), add:

```python
today = date.today()
if not force:
    existing = get_session_for_date(today, session_type="premarket")
    if existing and existing["status"] == "completed":
        logger.info("Premarket already completed for %s — skipping. Use --force to override.", today)
        return
session_id = insert_session_record(today, session_type="premarket")
logger.info("Premarket session ID: %d", session_id)
```

4. Wrap the body in try/except — on success call `complete_session(session_id)`, on exception call `fail_session(session_id, str(e))` and re-raise.
5. At each `insert_tweet(...)` call (lines 203, 224), pass `session_id=session_id`.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/v2/test_premarket.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add v2/premarket.py tests/v2/test_premarket.py
git commit -m "feat(premarket): create premarket session row + thread session_id"
```

---

## Task 12: Weekly social — create their own session rows

`v2/social_weekly.py` has two subcommands: `mistakes` and `attribution`. Each gets its own session_type.

**Files:**
- Modify: `v2/social_weekly.py` — both subcommands wrap themselves in a session record.
- Test: `tests/v2/test_social_weekly.py`.

- [ ] **Step 1: Read existing structure**

Open `v2/social_weekly.py`. Find the two subcommand entry points (likely `run_mistakes()` and `run_attribution()` or a dispatch on argv).

- [ ] **Step 2: Write the failing tests**

Add two tests — one per subcommand — asserting the right `session_type` is used and `session_id` is passed to `insert_tweet`:

```python
def test_mistakes_creates_weekly_mistakes_session(self):
    from v2.social_weekly import run_mistakes  # match actual entry point
    with patch("v2.social_weekly.insert_session_record", return_value=88) as mock_sess, \
         patch("v2.social_weekly.complete_session"), \
         patch("v2.social_weekly.insert_tweet", return_value=1) as mock_tweet, \
         patch("v2.social_weekly.posted_tweet_exists", return_value=False), \
         patch("v2.social_weekly.post_tweet", return_value="abc"):
        # plus mocks for data the mistakes flow reads
        run_mistakes(force=True)
    assert mock_sess.call_args.kwargs.get("session_type") == "weekly_mistakes"
    assert mock_tweet.call_args.kwargs.get("session_id") == 88

def test_attribution_creates_weekly_attribution_session(self):
    # parallel — session_type='weekly_attribution'
    ...
```

- [ ] **Step 3: Run tests to verify failure**

Run: `python3 -m pytest tests/v2/test_social_weekly.py -k "session" -v`
Expected: FAIL.

- [ ] **Step 4: Wire session-row management**

Mirror the Task 11 pattern for each subcommand. Use `session_type='weekly_mistakes'` and `'weekly_attribution'` respectively. Wrap each in try/except → complete_session/fail_session.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/v2/test_social_weekly.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add v2/social_weekly.py tests/v2/test_social_weekly.py
git commit -m "feat(social-weekly): weekly_mistakes/weekly_attribution session rows"
```

---

## Task 13: Entertainment — create a session row per invocation

Entertainment posts are on-demand. Per the design decision, every invocation creates a session row of type `'entertainment'` with **no idempotency gate** (you can run it as often as you want).

**Files:**
- Modify: `v2/entertainment.py` — create session row at start, thread session_id, complete/fail at end.
- Test: `tests/v2/test_entertainment.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_entertainment_creates_session_per_run(self):
    from v2.entertainment import run_entertainment  # match actual entry
    with patch("v2.entertainment.insert_session_record", return_value=55) as mock_sess, \
         patch("v2.entertainment.complete_session") as mock_complete, \
         patch("v2.entertainment.insert_tweet", return_value=1) as mock_tweet, \
         patch("v2.entertainment.post_tweet", return_value="abc"):
        run_entertainment()
    assert mock_sess.call_args.kwargs.get("session_type") == "entertainment"
    assert mock_tweet.call_args.kwargs.get("session_id") == 55
    mock_complete.assert_called_once_with(55)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/v2/test_entertainment.py -k "creates_session_per_run" -v`
Expected: FAIL.

- [ ] **Step 3: Wire session-row management**

Mirror Task 11/12 but skip the idempotency gate. Always insert; always complete/fail. Both `insert_tweet` calls (lines 175, 245) get `session_id=session_id`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/v2/test_entertainment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/entertainment.py tests/v2/test_entertainment.py
git commit -m "feat(entertainment): per-run session rows"
```

---

## Task 14: Rewrite dashboard date-joins to session-id joins

`dashboard/queries.py` has five JOINs that use `session_date`. With multiple sessions per date, those joins fan out. Convert each to a `session_id`-based join.

**Files:**
- Modify: `dashboard/queries.py:425-436` (`list_tweets` — left join), `463-490` (`get_session_*` cost query — already FK-based, verify), `555-567` (`get_session_decisions`), `570-582` (`get_session_theses_created`), `585-596` (`get_session_memo`), `599-611` (`get_session_tweets`).
- Test: `tests/dashboard/test_queries.py`.

- [ ] **Step 1: Read each query**

Open `dashboard/queries.py:425-611` and read each query in full. Note exactly which `LEFT JOIN`/`JOIN` lines reference `session_date`.

- [ ] **Step 2: Write failing tests**

For each rewritten query, add a test that:
- Inserts 2 session rows for the **same** date (e.g. 2 daily sessions because user re-ran with `--force`).
- Inserts decisions/theses/memos/tweets tied to **session B's id** only.
- Calls the dashboard query with **session A's id**.
- Asserts it returns 0 rows for that decision/thesis/memo/tweet (proves the join is by id, not date).

The current code would return the same content under both sessions. After fix, content surfaces only under its actual `session_id`.

Place these in `tests/dashboard/test_queries.py`. Use the existing fixtures pattern. Example:

```python
def test_get_session_decisions_is_id_scoped(self):
    # session A and B both on 2026-05-13
    session_a = insert_session_record(date(2026, 5, 13))
    session_b = insert_session_record(date(2026, 5, 13))  # second --force run
    # decision tied to session B
    insert_decision(
        decision_date=date(2026, 5, 13), ticker="AAPL", action="buy",
        quantity=10, price=200.0, reasoning="t", signals_used=[],
        account_equity=10000, buying_power=5000, session_id=session_b,
    )
    from dashboard.queries import get_session_decisions
    assert get_session_decisions(session_a) == []
    assert len(get_session_decisions(session_b)) == 1
```

(If `tests/dashboard/test_queries.py` mocks at module level instead of hitting a real DB, write per-query mock-based tests that assert the SQL string contains `session_id` not `session_date` on the relevant JOIN.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/dashboard/test_queries.py -k "session_id or id_scoped" -v`
Expected: FAIL.

- [ ] **Step 4: Rewrite each query**

Replace the date-JOIN lines with id-JOIN. Examples:

`get_session_decisions` (current lines 558-566 → new):

```python
def get_session_decisions(session_id: int):
    """Return decisions made during this session (by session_id)."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT d.id, d.date, d.ticker, d.action, d.quantity, d.price,
                   d.reasoning, d.account_equity, d.outcome_7d, d.outcome_30d,
                   d.is_off_playbook, d.playbook_action_id
            FROM decisions d
            WHERE d.session_id = %s
            ORDER BY d.id ASC
        """, (session_id,))
        return cur.fetchall()
```

`get_session_theses_created` (lines 573-581 → new):

```python
def get_session_theses_created(session_id: int):
    with get_cursor() as cur:
        cur.execute("""
            SELECT t.id, t.ticker, t.direction, t.thesis, t.entry_trigger,
                   t.exit_trigger, t.invalidation, t.confidence, t.source,
                   t.status, t.created_at, t.updated_at
            FROM theses t
            WHERE t.session_id = %s
            ORDER BY t.created_at ASC
        """, (session_id,))
        return cur.fetchall()
```

`get_session_memo` (lines 587-595 → new):

```python
def get_session_memo(session_id: int):
    with get_cursor() as cur:
        cur.execute("""
            SELECT m.id, m.session_date, m.memo_type, m.content, m.created_at
            FROM strategy_memos m
            WHERE m.session_id = %s
            ORDER BY m.created_at DESC
            LIMIT 1
        """, (session_id,))
        return cur.fetchone()
```

`get_session_tweets` (lines 601-610 → new):

```python
def get_session_tweets(session_id: int):
    with get_cursor() as cur:
        cur.execute("""
            SELECT tw.id, tw.session_date, tw.tweet_type, tw.tweet_text,
                   tw.platform, tw.posted, tw.error, tw.created_at,
                   tw.decision_id
            FROM tweets tw
            WHERE tw.session_id = %s
            ORDER BY tw.created_at DESC
        """, (session_id,))
        return cur.fetchall()
```

`list_tweets` (lines 425-436 — the LEFT JOIN to sessions): the current join is decorative (just exposes `s.id` for the listing UI). Replace `LEFT JOIN sessions s ON s.session_date = tw.session_date AND s.session_type = 'daily'` with `LEFT JOIN sessions s ON s.id = tw.session_id`. Legacy rows with NULL `tweets.session_id` will get NULL `session_id` in the result — that's correct.

`get_session_cost_breakdown` (lines 463-490 if it exists): already keyed by `session_id` per the schema view — verify and skip if no change needed.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/dashboard/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/queries.py tests/dashboard/test_queries.py
git commit -m "feat(dashboard): switch session-detail queries to session_id joins"
```

---

## Task 15: Full-suite regression + manual smoke

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -q`
Expected: All pass. If anything red, fix locally before continuing — do not skip.

- [ ] **Step 2: Run ruff**

Run: `python3 -m ruff check v2/ dashboard/ tests/`
Expected: clean. Fix any lint introduced.

- [ ] **Step 3: Paper-stack smoke test**

```bash
task paper:up
task paper:session:dry-run
task paper:session:dry-run  # second run, same day, no --force
```

Expected:
- First run: creates a session row, runs all stages, sets status='completed'.
- Second run: idempotency check trips, logs "Session already completed for <date>. Use --force to override.", exits cleanly. **No SQL error.**

Then:

```bash
docker compose exec trading python -m v2.session --dry-run --force
```

Expected: creates a **second** session row for the same date, runs stages, completes. Both session rows visible:

```bash
docker compose exec db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "SELECT id, session_date, session_type, status FROM sessions WHERE session_date = CURRENT_DATE;"
```

- [ ] **Step 4: Dashboard smoke**

If you can hit the paper dashboard at :3001, open the two session detail pages for today and verify:
- Each shows the decisions/theses/memos/tweets tied to *its* session_id only.
- No fan-out (a decision doesn't appear under both sessions).

- [ ] **Step 5: Commit any fixups**

```bash
git add -A
git commit -m "chore: regression fixups for per-run session id"
```

(If no fixups needed, skip.)

- [ ] **Step 6: Update CLAUDE.md (optional)**

If the per-run-session model changes the operator's mental model meaningfully, add a 2-3 line note to `CLAUDE.md` under "v2 Daily Session" explaining that `--force` now creates a new session row rather than overwriting today's.

```bash
git add CLAUDE.md
git commit -m "docs: note per-run session row behavior for --force"
```

---

## Self-Review Notes

- Spec coverage: schema migration (Task 1) ✓; insert_* signatures (2-5) ✓; session.py rework (6) ✓; daily-session plumbing (7-10) ✓; cron-driven sessions (11-13) ✓; dashboard rewrite (14) ✓; regression (15) ✓.
- Backfill: Task 1 step 3 verifies. Historical rows tied to one-and-only daily session per date. New cron-typed sessions don't exist historically, so `tweets`/`strategy_memos` rows written by past cron runs stay NULL — acceptable per the original scope decision.
- Out of scope (call out so the executor doesn't expand): adding NOT NULL constraints on session_id, tightening dedup unique indexes, adding session_id to `news_signals`/`macro_signals`/`account_snapshots`, refactoring the date-keyed `posted_tweet_exists` rerun guards. All deferrable; revisit after one week of clean prod data.
