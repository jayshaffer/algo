# Playbook Action Expiry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an idempotent session-start sweep that marks any `pending` playbook_actions belonging to a playbook with `date < today` as `expired`. This closes a data-hygiene hole: 200 stale `pending` rows have accumulated because no existing transition covers actions the executor never picked up.

**Architecture:** New DB helper `expire_stale_playbook_actions(today)` in `v2/database/trading_db.py`, called once at the top of `run_session` in `v2/session.py`, after the market date is resolved and before any stage runs. Pure SQL, idempotent, no behavior changes to executor or strategist. First production run backfills the 200 existing rows automatically.

**Tech Stack:** Python, psycopg2 with `get_cursor()` context manager, pytest with `mock_db` / `mock_cursor` fixtures from `tests/v2/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-05-19-playbook-action-expiry-design.md`

---

### Task 1: Add `expire_stale_playbook_actions` helper

**Files:**
- Modify: `v2/database/trading_db.py` (add new function after `update_playbook_action_status` near line 545)
- Test: `tests/v2/test_db.py` (add new tests inside `TestPlaybookActionStatus` class, line 578)

- [ ] **Step 1: Write the failing tests**

Append the following tests inside `class TestPlaybookActionStatus:` in `tests/v2/test_db.py`:

```python
    def test_expire_stale_playbook_actions_runs_update(self, mock_db, mock_cursor):
        from datetime import date as _date
        from v2.database.trading_db import expire_stale_playbook_actions
        mock_cursor.rowcount = 7
        result = expire_stale_playbook_actions(_date(2026, 5, 19))
        sql = mock_cursor.execute.call_args[0][0]
        params = mock_cursor.execute.call_args[0][1]
        assert "UPDATE playbook_actions" in sql
        assert "status = 'expired'" in sql
        assert "status = 'pending'" in sql
        assert "date < %s" in sql
        assert params == (_date(2026, 5, 19),)
        assert result == 7

    def test_expire_stale_playbook_actions_zero_rows(self, mock_db, mock_cursor):
        from datetime import date as _date
        from v2.database.trading_db import expire_stale_playbook_actions
        mock_cursor.rowcount = 0
        result = expire_stale_playbook_actions(_date(2026, 5, 19))
        assert result == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_db.py::TestPlaybookActionStatus::test_expire_stale_playbook_actions_runs_update tests/v2/test_db.py::TestPlaybookActionStatus::test_expire_stale_playbook_actions_zero_rows -v
```
Expected: FAIL with `ImportError` or `AttributeError` on `expire_stale_playbook_actions`.

- [ ] **Step 3: Implement the helper**

In `v2/database/trading_db.py`, immediately after the `update_playbook_action_status` function (around line 545), add:

```python
def expire_stale_playbook_actions(today) -> int:
    """Mark pending playbook_actions from playbooks dated before `today` as 'expired'.

    Closes a lifecycle hole: prior-day pending actions had no transition path
    (executor only updates status on submit/reject/fail, strategist only wipes
    today's actions). Idempotent — running twice is a no-op.

    Returns the number of rows updated.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE playbook_actions
               SET status = 'expired'
             WHERE status = 'pending'
               AND playbook_id IN (
                   SELECT id FROM playbooks WHERE date < %s
               )
            """,
            (today,),
        )
        return cur.rowcount
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_db.py::TestPlaybookActionStatus -v
```
Expected: All tests in `TestPlaybookActionStatus` pass (the existing 5 + the 2 new ones = 7).

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "$(cat <<'EOF'
feat(playbook): add expire_stale_playbook_actions helper

Marks pending actions from prior-day playbooks as 'expired'. Closes
the lifecycle hole that left 200 rows accumulating in 'pending'.
Idempotent — safe to run every session.

Spec: docs/superpowers/specs/2026-05-19-playbook-action-expiry-design.md
EOF
)"
```

---

### Task 2: Call sweep at session start

**Files:**
- Modify: `v2/session.py` (add import near other `trading_db` imports; add call inside `run_session` between line 497 `today = current_market_date()` and line 499 `session_id, completed_stages, early_error = ...`)
- Test: `tests/v2/test_session.py` (add new test class `TestPlaybookActionExpiry` at end of file)

- [ ] **Step 1: Write the failing test**

Append to `tests/v2/test_session.py`:

```python
class TestPlaybookActionExpiry:
    def test_sweep_runs_before_any_stage(self):
        """Stale pending playbook_actions should be expired before stages dispatch."""
        call_order = []

        with patch("v2.session.expire_stale_playbook_actions") as mock_expire, \
             patch("v2.session.run_backfill") as mock_backfill, \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline") as mock_pipeline, \
             patch("v2.session.run_strategist_loop") as mock_strat, \
             patch("v2.session.run_trading_session") as mock_trade:

            mock_expire.side_effect = lambda *a, **kw: call_order.append("expire") or 0
            mock_backfill.side_effect = lambda **kw: call_order.append("backfill")
            mock_pipeline.side_effect = lambda **kw: call_order.append("pipeline")
            mock_strat.side_effect = lambda **kw: call_order.append("strategist")
            mock_trade.side_effect = lambda **kw: call_order.append("trader")

            run_session(dry_run=False)

        assert call_order, "no stages ran"
        assert call_order[0] == "expire", f"expire must run first, got {call_order}"

    def test_sweep_receives_market_date(self):
        """Sweep should be called with the same canonical date used elsewhere."""
        market_date = date(2026, 5, 19)
        with patch("v2.session.current_market_date", return_value=market_date), \
             patch("v2.session.expire_stale_playbook_actions") as mock_expire, \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"):

            run_session(dry_run=False)

        mock_expire.assert_called_once_with(market_date)

    def test_sweep_failure_does_not_block_session(self):
        """If the sweep itself raises, the session still runs (data-hygiene only)."""
        with patch("v2.session.expire_stale_playbook_actions", side_effect=Exception("DB blip")), \
             patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session") as mock_trade:

            result = run_session(dry_run=False)

        assert mock_trade.called, "trader stage should still run after sweep failure"
        assert result.idempotent_skip is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_session.py::TestPlaybookActionExpiry -v
```
Expected: FAIL — `expire_stale_playbook_actions` is not yet imported into `v2.session`, so `patch("v2.session.expire_stale_playbook_actions")` raises `AttributeError`.

- [ ] **Step 3: Add the import**

In `v2/session.py`, find the existing block of `from v2.database.trading_db import ...` imports near the top of the file. Add `expire_stale_playbook_actions` to that import list. If there is no such import block (the helpers are imported individually), add at the top with the other trading_db imports:

```python
from v2.database.trading_db import expire_stale_playbook_actions
```

Confirm with:
```bash
grep -n "expire_stale_playbook_actions\|from v2.database.trading_db" v2/session.py
```
Expected: at least one line showing the new import.

- [ ] **Step 4: Add the sweep call to `run_session`**

In `v2/session.py`, locate this block (line ~497–505):

```python
    today = current_market_date()

    session_id, completed_stages, early_error = _check_and_record_session(force, today)
    if early_error:
        # T2.1: idempotent skip is not a failure. main() reads this field
        # and exits 0 with a "nothing to do" log line.
        result.idempotent_skip = early_error
        result.duration_seconds = time.monotonic() - start
        return result
```

Replace it with:

```python
    today = current_market_date()

    try:
        expired = expire_stale_playbook_actions(today)
        if expired:
            logger.info("Expired %d stale pending playbook_actions from prior days", expired)
    except Exception as e:
        logger.warning("expire_stale_playbook_actions failed (non-fatal): %s", e)

    session_id, completed_stages, early_error = _check_and_record_session(force, today)
    if early_error:
        # T2.1: idempotent skip is not a failure. main() reads this field
        # and exits 0 with a "nothing to do" log line.
        result.idempotent_skip = early_error
        result.duration_seconds = time.monotonic() - start
        return result
```

The try/except keeps a transient DB blip in the sweep from blocking the trading session — this is data hygiene, not a load-bearing operation.

- [ ] **Step 5: Verify `logger` exists in session.py**

Run:
```bash
grep -n "^logger = \|^logger=\|getLogger" v2/session.py | head -3
```
Expected: a line defining `logger` (typically `logger = logging.getLogger(__name__)`). If for some reason no logger exists, add `import logging` and `logger = logging.getLogger(__name__)` at the top.

- [ ] **Step 6: Run the new tests to verify they pass**

Run:
```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_session.py::TestPlaybookActionExpiry -v
```
Expected: All 3 tests pass.

- [ ] **Step 7: Run the full session test module to confirm no regressions**

Run:
```bash
docker compose exec -T trading python3 -m pytest tests/v2/test_session.py -v
```
Expected: All tests in the file pass.

- [ ] **Step 8: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "$(cat <<'EOF'
feat(session): call expire_stale_playbook_actions at session start

Closes the playbook_action lifecycle hole. Sweep runs before any
stage; failure is logged but does not block the session (hygiene
only). First prod run will backfill the ~200 existing pending rows.

Spec: docs/superpowers/specs/2026-05-19-playbook-action-expiry-design.md
EOF
)"
```

---

### Task 3: Full test suite + production verification plan

**Files:** none (verification step only)

- [ ] **Step 1: Run the full v2 test suite**

Run:
```bash
docker compose exec -T trading python3 -m pytest tests/v2/ -q
```
Expected: All tests pass. If any unrelated test fails, do not paper over it — investigate before claiming completion.

- [ ] **Step 2: Document the post-deploy verification check**

After the next 19:00 UTC session lands (next time `run_session` runs against prod), confirm the backfill took effect by running:

```bash
docker compose exec -T db psql -U algo -d trading -c "SELECT status, COUNT(*) FROM playbook_actions GROUP BY status ORDER BY status;"
```

Expected before deploy:
```
 status  | count
---------+-------
 executed |  17
 pending  | 200
```

Expected after one session post-deploy:
```
 status  | count
---------+-------
 executed |  17
 expired  | ~200
 pending  | ~3-5    (only today's actions, if the strategist generated any)
```

Note: the exact `executed` count will drift; the key signal is that `pending` drops to a small number (today's actions only) and `expired` appears with the bulk of the historical rows.

There is no automated verification step — this is an out-of-band check the operator performs after deploy. If `pending` is still >10 after a session, investigate before declaring the fix successful.

- [ ] **Step 3: No commit needed for this task**

Verification step only. No code changed.

---

## Self-Review Notes

- **Spec coverage:** Spec components 1 (helper), 2 (session.py call), 3 (tests), 4 (backfill) → all covered by Tasks 1, 2, and the verification step in Task 3.
- **Risk surface:** Both `playbook_actions.status` consumers checked in the spec (`dashboard_db.py:145` counts `executed`; `trading_db.py:552` filters `pending OR NULL`) are unaffected by the new `expired` value.
- **Defense-in-depth:** Added a try/except around the sweep call so a transient DB error doesn't take down a session — the original spec didn't specify this, but it's consistent with the spec's "data hygiene" framing.
- **Idempotency:** The SQL is `WHERE status = 'pending'`, so any row already marked `expired` is skipped on subsequent runs. Verified by the `test_expire_stale_playbook_actions_zero_rows` test (rowcount=0 case).
