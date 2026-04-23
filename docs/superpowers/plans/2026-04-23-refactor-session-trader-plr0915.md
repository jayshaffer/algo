# Refactor `v2/session.py` and `v2/trader.py` to satisfy PLR0915

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `run_session` (236 stmts) and `run_trading_session` (257 stmts) under the `max-statements = 50` guardrail by extracting cohesive private helpers, while preserving exact runtime behavior. No external API or database contract changes.

**Architecture:** Extract-method refactors into private (`_`-prefixed) helpers in the same modules. Orchestrators become thin drivers that call per-stage/per-phase helpers. SessionResult/TradingSessionResult are mutated in place where appropriate to minimize signature churn; where a helper returns a scalar, the caller assigns it to the result dataclass.

**Tech Stack:** Python 3.12, ruff 0.15, pytest + `pytest-cov` with `--cov-branch`, docker compose (tests run in the `trading` container).

**Hard prerequisite:** Both `v2/session.py` and `v2/trader.py` MUST reach 100% branch coverage BEFORE any refactor step runs. Baseline at plan start (line/branch approximate): session.py ~72%/lower, trader.py ~66%/lower. Tasks 1–3 close that gap; Tasks 4–7 perform the refactor. Any step in Tasks 4–7 that shows coverage dropping below 100% branch must stop and re-examine — the refactor missed a branch, the test missed a branch, or the coverage config is wrong.

---

## Context an agentic worker must have

1. **Repository layout.** `v2/` is the active codebase; `trading/` is legacy and already excluded from PLR0915 via `ruff.toml`. Tests live in `tests/v2/`. `Taskfile.yml` defines `task test` (in container) and `task lint` (on host).
2. **Test execution.** Tests require Postgres. Run with `docker compose exec -T trading python -m pytest tests/v2/<file> -q`. Full suite is `tests/` but skip `tests/test_classifier.py` (pre-existing ImportError unrelated to this work) and don't fix it in this plan.
3. **Pre-existing test failures unrelated to the PLR0915 work:**
   - `tests/test_classifier.py` — collection ImportError (out of scope).
   - `tests/v2/test_entertainment.py::TestRunEntertainmentPipeline::test_happy_path` — missing `get_bluesky_client` mock, actually hits real APIs (out of scope; do NOT re-run this test casually).
4. **Tests already passing (starting point for coverage work):**
   - `tests/v2/test_trader.py` — 16/16 pass but only ~66% line / lower branch. Task 3 extends this suite to 100% branch; Task 6 uses that coverage as its safety net.
5. **Tests broken by DB leak (blocking):** 20 tests in `tests/v2/test_session.py`. All share the root cause: `run_session` hits `get_session_for_date(today)` against the real `db` container, finds a completed session, and returns early with `"Session already completed"`. Task 0 fixes this.
6. **No worktree.** Work proceeds directly on `main` with uncommitted changes from prior tasks already staged in the working tree. Commit granularity is per-task.
7. **Behavior preservation.** These functions run against a live Alpaca account. Every branch must survive the refactor. Do not change `logger.info` strings or ordering — existing monitoring/dashboards depend on the log shape.
8. **Ruff gate.** `task lint` currently reports 2 PLR0915 violations after prior work. Final state: 0 violations.

---

## File Structure

- **Modify** `v2/session.py` — add `_start_stage`, `_complete_stage`, `_fail_stage` DB-call wrappers; extract one `_run_stage_*` helper per stage; shrink `run_session` to idempotency check + 7 stage calls + summary.
- **Modify** `v2/trader.py` — extract `_sync_from_alpaca`, `_snapshot_account`, `_build_executor_context`, `_execute_decision`, `_log_decision` helpers; shrink `run_trading_session` to a linear driver.
- **Modify** `tests/v2/test_session.py` — (a) fix the DB-leak idempotency issue so all 20 currently-failing tests pass (Task 0); (b) add new tests to close branch-coverage gaps on `v2/session.py` (Task 2).
- **Modify** `tests/v2/test_trader.py` — add new tests to close branch-coverage gaps on `v2/trader.py` (Task 3). Expect to expand this file substantially — exception handlers and early-exit paths are currently uncovered.

---

## Task 0: Restore the session-test safety net

**Why first:** A 236-statement refactor of orchestration that touches live trading requires a trustworthy test suite. Currently 20 of 41 session tests fail because the DB idempotency check short-circuits `run_session` before any stage runs. Fix once, unblock all.

**Files:**
- Modify: `tests/v2/test_session.py`

- [ ] **Step 1: Reproduce the baseline failure count and capture the exact set.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | grep -E "^FAILED" | sort > /tmp/session_fail_before.txt
wc -l /tmp/session_fail_before.txt
```
Expected: `20` lines. Keep the file for diffing in Step 5.

- [ ] **Step 2: Inspect the failure mode on a representative test.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_session.py::TestStrategistMemoPersistence::test_strategist_summary_written_as_memo -q 2>&1 | grep -A2 "Session already completed"
```
Expected: log line "Session already completed for \<today\>. Use --force to override." confirming the idempotency short-circuit. This is the shared root cause.

- [ ] **Step 3: Read the test file around the failing cases to identify the `run_session` invocations.**

Read `tests/v2/test_session.py` and locate every direct or indirect call to `run_session(...)`. Note whether each call passes `force=True` and whether the test patches `v2.session.get_session_for_date` or `v2.session.get_completed_stages`.

- [ ] **Step 4: Add a shared autouse fixture that mocks the idempotency lookup at the session module boundary.**

In `tests/v2/test_session.py`, at the top of the test module (after imports, before any test class), add:

```python
import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def _bypass_session_idempotency():
    """All tests in this module should exercise run_session as if no prior session exists today.

    The production idempotency check reads the live Postgres `sessions` row for
    today's date; leaving that intact makes tests flaky against shared DB state.
    """
    with patch("v2.session.get_session_for_date", return_value=None), \
         patch("v2.session.get_completed_stages", return_value=set()):
        yield
```

If the file already defines an `autouse` fixture, merge the two `patch` context managers into it instead of adding a duplicate. Do NOT modify production code in `v2/session.py` in this task.

- [ ] **Step 5: Verify all 20 failing tests now pass and no passing tests regress.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | grep -E "^FAILED" | sort > /tmp/session_fail_after.txt
comm -23 /tmp/session_fail_before.txt /tmp/session_fail_after.txt  # tests fixed
comm -13 /tmp/session_fail_before.txt /tmp/session_fail_after.txt  # tests newly broken
```
Expected: the first `comm` prints all 20 previously-failing tests (they now pass), the second prints nothing (no regressions). Also confirm the final line reads `41 passed`.

- [ ] **Step 6: Commit.**

```
git add tests/v2/test_session.py
git commit -m "test(v2/session): mock idempotency DB lookup so tests don't leak real session state"
```

---

## Task 1: Branch-coverage gap analysis

**Why:** You can't write "just the missing tests" without first knowing which branches are uncovered. This task captures a machine-readable gap list that Tasks 2 and 3 work against.

**Files:**
- Read-only: `v2/session.py`, `v2/trader.py`, `tests/v2/test_session.py`, `tests/v2/test_trader.py`

- [ ] **Step 1: Confirm `pytest-cov` is available and supports `--cov-branch`.**

Run:
```
docker compose exec -T trading python -m pytest --version
docker compose exec -T trading python -c "import pytest_cov; print(pytest_cov.__version__)"
```
Expected: both commands succeed. If `pytest_cov` is missing, stop and flag — installation is out of scope for this plan but trivial to add.

- [ ] **Step 2: Capture baseline branch coverage for `v2/session.py`.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_session.py --cov=v2.session --cov-branch --cov-report=term-missing -q > /tmp/session_cov_baseline.txt 2>&1
tail -25 /tmp/session_cov_baseline.txt
```
Expected: a "Missing" column listing uncovered line numbers and branch targets (e.g., `42->45` means the branch from line 42 to 45 was not taken). Save the full file for reference during Task 2.

- [ ] **Step 3: Capture baseline branch coverage for `v2/trader.py`.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py --cov=v2.trader --cov-branch --cov-report=term-missing -q > /tmp/trader_cov_baseline.txt 2>&1
tail -25 /tmp/trader_cov_baseline.txt
```
Expected: similar output for `v2/trader.py`. Save for Task 3.

- [ ] **Step 4: Build a structured gap list for `v2/session.py`.**

Open `v2/session.py` and `/tmp/session_cov_baseline.txt` side by side. For every uncovered line or branch, note:
- The line range
- The code construct (e.g., `except Exception as e: logger.warning(...)`)
- The trigger condition needed to exercise it (e.g., "`insert_session_stage` raises")
- The stage it belongs to

Categories you will almost certainly see in `v2/session.py`:
- Idempotency-check exception handlers (line ~124-125)
- `insert_session_record` failure (line ~130-131)
- Stage-tracking DB exception handlers (`insert_session_stage`, `complete_session_stage`, `fail_session_stage` inside each stage — multiple occurrences)
- Per-stage Exception paths (pipeline_error, strategist_error, etc.)
- Memo persistence failure (line ~199)
- Executor short-circuit (strategist_error AND no playbook)
- Final `fail_session` / `complete_session` error handler
- `main()` CLI entry point (lines ~404-433) and the `if __name__ == "__main__"` guard

Write the gap list to `/tmp/session_gaps.md` as a checklist the Task 2 tests will work through.

- [ ] **Step 5: Build a structured gap list for `v2/trader.py`.**

Same method against `/tmp/trader_cov_baseline.txt`. Expected categories:
- Position/order sync exception handlers
- Market-closed early-return branch (not dry_run + market closed)
- Account snapshot exception handler with early return
- Context build exception path (fallback ExecutorInput)
- Sector concentration warnings appended to existing vs empty risk_notes
- LLM decision failure early return
- `get_latest_price` returns None for a decision
- Intent resolution: sell / buy / unsupported action / buy without magnitude
- Resolved qty ≤ epsilon early-continue
- Alpaca precheck: exception / None / zero-available / trimming / update_playbook_action_status failure
- `execute_market_order` failure branch (result.success is False)
- Fill failure path (not dry_run, fill.success is False)
- Playbook-action-status updates after executed / failed (both exception paths)
- Buying-power refresh failure
- Dry-run local estimate for buys
- Thesis lifecycle: partial exit vs full exit, and the exception path
- Thesis invalidations loop (empty and with-invalidations, and the exception inside the loop)
- Decision logging: price-None for buy/sell, duplicate-skip, logged_qty branches, log-row exception, validated_refs shorter than original, no validated_refs, signal-link exception, no signal_refs for buy/sell warning
- `main()` CLI and the `if __name__` guard

Write the gap list to `/tmp/trader_gaps.md`.

- [ ] **Step 6: Commit nothing yet — this task produces analysis, not code.**

Move to Task 2.

---

## Task 2: Close `v2/session.py` branch coverage gaps

**Why:** Creates the safety net for the session refactor. Every missed branch becomes a dedicated test; the bar is 100% branch coverage before any refactor step runs.

**Files:**
- Modify: `tests/v2/test_session.py`

**Working methodology** (apply to each gap in `/tmp/session_gaps.md`):

1. Pick one uncovered branch.
2. Write a test that triggers exactly that branch. Use `unittest.mock.patch` at the `v2.session` module boundary (not at the original definition site).
3. Run the single test — verify it passes.
4. Re-run coverage — verify the targeted line/branch moved from "Missing" to covered.
5. Commit the single new test with a descriptive message.
6. Repeat.

**Representative test templates.** Copy the shape that matches the gap:

```python
# Template A: an insert_session_stage DB-tracking exception handler
@patch("v2.session.run_pipeline")
@patch("v2.session.insert_session_stage", side_effect=RuntimeError("db dead"))
def test_pipeline_stage_start_db_failure_is_swallowed(mock_insert, mock_pipeline):
    mock_pipeline.return_value = MagicMock()
    # run_session should NOT raise even if insert_session_stage throws
    result = run_session(
        skip_pipeline=False, skip_ideation=True, skip_executor=True,
        skip_strategy=True, skip_twitter=True, skip_bluesky=True,
        skip_dashboard=True, force=True,
    )
    mock_pipeline.assert_called_once()  # stage still ran despite tracking failure
    assert result.pipeline_error is None


# Template B: a stage-body exception sets the error field and records fail_session_stage
@patch("v2.session.fail_session_stage")
@patch("v2.session.run_pipeline", side_effect=RuntimeError("kaboom"))
def test_pipeline_failure_captured_and_tracked(mock_pipeline, mock_fail):
    result = run_session(
        skip_ideation=True, skip_executor=True, skip_strategy=True,
        skip_twitter=True, skip_bluesky=True, skip_dashboard=True, force=True,
    )
    assert result.pipeline_error == "kaboom"
    mock_fail.assert_called_once()
    assert mock_fail.call_args.args[1] == "pipeline"


# Template C: the memo-persistence exception handler
@patch("v2.session.get_current_strategy_state", side_effect=RuntimeError("state gone"))
@patch("v2.session.run_strategist_loop")
def test_strategist_memo_persist_exception_does_not_block(mock_strat, mock_state):
    mock_strat.return_value = MagicMock(final_summary="hi")
    result = run_session(
        skip_pipeline=True, skip_executor=True, skip_strategy=True,
        skip_twitter=True, skip_bluesky=True, skip_dashboard=True, force=True,
    )
    # Strategist succeeded; memo persistence failure is swallowed.
    assert result.strategist_error is None


# Template D: the final fail_session exception handler
@patch("v2.session.fail_session", side_effect=RuntimeError("db flaky"))
@patch("v2.session.run_pipeline", side_effect=RuntimeError("kaboom"))
def test_fail_session_exception_is_swallowed(mock_pipeline, mock_fail_session):
    # A stage failed so has_errors is True → run_session calls fail_session.
    # fail_session itself throws. run_session must still return normally.
    result = run_session(
        skip_ideation=True, skip_executor=True, skip_strategy=True,
        skip_twitter=True, skip_bluesky=True, skip_dashboard=True, force=True,
    )
    assert result.has_errors
```

- [ ] **Step 1: For each gap in `/tmp/session_gaps.md` NOT already covered, write one test and commit it individually.**

Iterate the methodology above. Do NOT batch multiple uncovered branches into one test unless they are genuinely the same branch path exercised twice — one-branch-per-test is the rule.

For the `main()` CLI entry point, write one test that calls `main()` with `sys.argv` patched, mocks `run_session`, and asserts it was called with the expected kwargs. Also write a test for the `sys.exit(1)` path when `result.has_errors` is True.

- [ ] **Step 2: Verify 100% branch coverage.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_session.py --cov=v2.session --cov-branch --cov-report=term-missing --cov-fail-under=100 -q 2>&1 | tail -10
```
Expected: `v2/session.py` line 100% and branch 100%, and the `--cov-fail-under=100` gate passes. If it fails, the "Missing" column shows the remaining gaps — go back to Step 1.

- [ ] **Step 3: Final commit.**

All individual test commits should already be in. Confirm `git status` is clean and `git log --oneline -10` shows the new test commits with clear messages.

---

## Task 3: Close `v2/trader.py` branch coverage gaps

**Why:** Same safety-net work for trader.py. This is the larger of the two gaps — trader.py has more exception-handling and early-exit branches.

**Files:**
- Modify: `tests/v2/test_trader.py`

**Working methodology:** Identical to Task 2. One-branch-per-test, commit-per-test.

**Representative test templates specific to trader.py:**

```python
# Template A: market-closed early return
@patch("v2.trader.is_market_open", return_value=False)
@patch("v2.trader.sync_orders_from_alpaca", return_value=0)
@patch("v2.trader.sync_positions_from_alpaca", return_value=0)
def test_market_closed_returns_early_without_snapshot(mock_pos, mock_ord, mock_open):
    result = run_trading_session(dry_run=False)
    assert result.account_snapshot_id == 0
    assert "Market is closed" in result.errors[0]


# Template B: account snapshot failure early return
@patch("v2.trader.take_account_snapshot", side_effect=RuntimeError("alpaca 500"))
@patch("v2.trader.get_account_info", side_effect=RuntimeError("alpaca 500"))
@patch("v2.trader.is_market_open", return_value=True)
@patch("v2.trader.sync_orders_from_alpaca", return_value=0)
@patch("v2.trader.sync_positions_from_alpaca", return_value=0)
def test_snapshot_failure_returns_early(...):
    result = run_trading_session(dry_run=False)
    assert "Account snapshot failed" in result.errors[0]
    assert result.account_snapshot_id == 0


# Template C: intent resolution error tags decision as invalid
def test_buy_without_magnitude_is_rejected(...):
    # Set up a BUY decision with intent_magnitude=None
    # Stub resolve_buy_intent or rely on the IntentError raise path
    # Assert: decision.action == "invalid", trades_failed == 1, decision.reasoning contains "intent error"


# Template D: resolved_qty ≤ epsilon continues without executing
def test_zero_resolved_qty_skips_execution(...):
    # Stub resolve_sell_intent to return Decimal("0.00001")
    # Assert: execute_market_order NOT called for this decision
    # decision.action == "invalid", reasoning contains "resolved to 0 shares"


# Template E: Alpaca precheck zero-available rejection
@patch("v2.trader.get_live_available_qty", return_value=Decimal("0"))
def test_sell_rejected_when_alpaca_shows_zero(...):
    # decision.action == "invalid", reasoning contains "Alpaca reports 0 available shares"
    # trades_failed incremented


# Template F: thesis close on full exit
def test_full_sell_closes_thesis(...):
    # held = 10, sell 10 → remaining = 0 → close_thesis called with status="closed"


# Template G: partial sell keeps thesis active
def test_partial_sell_keeps_thesis_active(...):
    # held = 10, sell 5 → remaining = 5 → close_thesis NOT called, log says "kept active"


# Template H: decision logging duplicate skip
@patch("v2.trader.check_decision_exists", return_value=42)
def test_duplicate_decision_is_skipped(...):
    # insert_decision NOT called for that decision
```

- [ ] **Step 1: For each gap in `/tmp/trader_gaps.md`, write one test and commit it individually.**

Work top-to-bottom through the function. Tests that require more than three levels of mocking should use a pytest fixture `mock_trader_env` that sets up the default "happy path" mocks (positions sync, snapshot, executor input, `get_trading_decisions`, `get_latest_price`, `execute_market_order`, `wait_for_fill`) and each test overrides only the mock relevant to the branch it's exercising. If this fixture doesn't exist in `tests/v2/test_trader.py`, build it as your first step before writing branch tests.

- [ ] **Step 2: Verify 100% branch coverage.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py --cov=v2.trader --cov-branch --cov-report=term-missing --cov-fail-under=100 -q 2>&1 | tail -10
```
Expected: 100% line, 100% branch, `--cov-fail-under=100` passes.

- [ ] **Step 3: Confirm clean git status and commit list.**

---

## Task 4: Extract DB-call helpers in `v2/session.py`

**Why:** Every stage repeats a 5-line `try: insert_session_stage(...); except Exception: pass` pattern (and symmetric `complete`/`fail` variants). Extracting them is a mechanical reduction that is safe in isolation.

**Files:**
- Modify: `v2/session.py`

- [ ] **Step 1: Baseline — tests and lint.**

Run:
```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | tail -3
task lint 2>&1 | tail -5
```
Expected: `41 passed`. Ruff reports `v2/session.py:87:5: PLR0915 Too many statements (236 > 50)` and `v2/trader.py:74:5: ...`.

- [ ] **Step 2: Add three helpers near the top of `v2/session.py` (after the `SessionResult` dataclass, before `run_session`).**

```python
def _start_stage(session_id: int | None, stage: str) -> None:
    if session_id is None:
        return
    try:
        insert_session_stage(session_id, stage)
    except Exception:
        pass


def _complete_stage(session_id: int | None, stage: str) -> None:
    if session_id is None:
        return
    try:
        complete_session_stage(session_id, stage)
    except Exception:
        pass


def _fail_stage(session_id: int | None, stage: str, error: str) -> None:
    if session_id is None:
        return
    try:
        fail_session_stage(session_id, stage, error)
    except Exception:
        pass
```

- [ ] **Step 3: Replace every inline DB-tracking block in `run_session` with calls to the helpers.**

For each stage (pipeline, strategist, executor, strategy, twitter, bluesky, dashboard), replace:

```python
if session_id:
    try:
        insert_session_stage(session_id, "<stage>")
    except Exception:
        pass
```
with
```python
_start_stage(session_id, "<stage>")
```

And similarly for `complete_session_stage` → `_complete_stage`, `fail_session_stage` → `_fail_stage`. Preserve the stage names exactly (`"pipeline"`, `"strategist"`, `"executor"`, `"strategy"`, `"twitter"`, `"bluesky"`, `"dashboard"`).

- [ ] **Step 4: Run tests.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | tail -3
```
Expected: `41 passed`. Any failure means a stage name was typo'd or a helper call was missed — fix before continuing.

- [ ] **Step 5: Run lint — statement count should drop but still exceed 50.**

```
task lint 2>&1 | grep PLR0915
```
Expected: `v2/session.py:87:5: PLR0915 Too many statements (N > 50)` where N is in the 160–190 range (roughly −60 from 236).

- [ ] **Step 6: Verify branch coverage still 100%.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py --cov=v2.session --cov-branch --cov-fail-under=100 -q 2>&1 | tail -5
```
Expected: `--cov-fail-under=100` passes. If coverage dropped, the helper introduced a branch not exercised by the existing tests — either the helper has dead code (remove it) or a new branch needs one more test before proceeding.

- [ ] **Step 7: Commit.**

```
git add v2/session.py
git commit -m "refactor(v2/session): extract _start_stage/_complete_stage/_fail_stage helpers"
```

---

## Task 5: Extract one helper per stage in `v2/session.py`

**Why:** Each stage (pipeline through dashboard) follows a uniform pattern: skip-check, `_start_stage`, run the stage, `_complete_stage` on success / `_fail_stage` on exception, attach result/error to `SessionResult`. Each stage is ~15 statements — lifting them into helpers is mechanical and each is independently verifiable.

**Files:**
- Modify: `v2/session.py`

- [ ] **Step 1: Extract Stage 0 (learning refresh).**

Add above `run_session`:

```python
def _run_learning_refresh(result: SessionResult) -> str:
    """Stage 0 — returns attribution_constraints (possibly empty)."""
    logger.info("[Stage 0] Refreshing learning data")
    try:
        run_backfill()
        compute_signal_attribution()
        constraints = build_attribution_constraints()
        logger.info("Learning refresh complete")
        return constraints
    except Exception as e:
        result.learning_error = str(e)
        logger.warning("Learning refresh failed: %s — continuing with stale data", e)
        return ""
```

In `run_session`, replace the entire Stage 0 block with:
```python
attribution_constraints = _run_learning_refresh(result)
```

- [ ] **Step 2: Run tests after Stage 0 extraction.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | tail -3
```
Expected: `41 passed`.

- [ ] **Step 3: Extract Stage 1 (pipeline).**

Add:
```python
def _run_pipeline_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
    pipeline_hours: int,
    pipeline_limit: int,
) -> None:
    if skip or "pipeline" in completed_stages:
        logger.info("[Stage 1] News pipeline — SKIPPED%s",
                    " (completed in prior run)" if "pipeline" in completed_stages else "")
        return
    logger.info("[Stage 1] Running news pipeline")
    _start_stage(session_id, "pipeline")
    try:
        result.pipeline_result = run_pipeline(hours=pipeline_hours, limit=pipeline_limit)
        _complete_stage(session_id, "pipeline")
    except Exception as e:
        result.pipeline_error = str(e)
        _fail_stage(session_id, "pipeline", str(e))
        logger.error("Pipeline failed: %s — continuing with existing signals", e)
```

In `run_session`, replace the Stage 1 block with:
```python
_run_pipeline_stage(result, session_id, completed_stages, skip_pipeline, pipeline_hours, pipeline_limit)
```

- [ ] **Step 4: Run tests.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | tail -3
```
Expected: `41 passed`.

- [ ] **Step 5: Extract Stage 2 (strategist), including the memo-persistence side effect.**

Add:
```python
def _persist_strategist_memo(result: SessionResult, session_date) -> None:
    try:
        if result.strategist_result and result.strategist_result.final_summary:
            state = get_current_strategy_state()
            insert_strategy_memo(
                session_date=session_date,
                memo_type='strategist_notes',
                content=result.strategist_result.final_summary,
                strategy_state_id=state['id'] if state else None,
            )
            logger.info("Strategist summary saved as memo")
    except Exception as e:
        logger.warning("Could not save strategist memo: %s", e)


def _run_strategist_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
    model: str,
    max_turns: int,
    attribution_constraints: str,
    session_date,
) -> None:
    if skip or "strategist" in completed_stages:
        logger.info("[Stage 2] Strategist — SKIPPED%s",
                    " (completed in prior run)" if "strategist" in completed_stages else "")
        return
    logger.info("[Stage 2] Running Claude strategist")
    _start_stage(session_id, "strategist")
    try:
        result.strategist_result = run_strategist_loop(
            model=model,
            max_turns=max_turns,
            attribution_constraints=attribution_constraints,
        )
        _persist_strategist_memo(result, session_date)
        _complete_stage(session_id, "strategist")
    except Exception as e:
        result.strategist_error = str(e)
        _fail_stage(session_id, "strategist", str(e))
        logger.error("Strategist failed: %s — continuing with existing playbook", e)
```

In `run_session`, replace Stage 2 with:
```python
_run_strategist_stage(
    result, session_id, completed_stages, skip_ideation,
    model, max_turns, attribution_constraints, today,
)
```

- [ ] **Step 6: Run tests after Stage 2.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | tail -3
```
Expected: `41 passed`. If `TestStrategistMemoPersistence` fails, you moved the memo call outside the success branch — inspect and fix.

- [ ] **Step 7: Extract Stage 3 (executor) including the strategist-failure-with-no-playbook short-circuit.**

Add:
```python
def _run_executor_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
    dry_run: bool,
    executor_model: str,
    session_date,
) -> None:
    # Short-circuit: strategist failed AND no playbook → skip executor
    if (
        not skip
        and "executor" not in completed_stages
        and result.strategist_error
        and get_playbook(session_date) is None
    ):
        logger.warning("Strategist failed and no playbook exists for %s — skipping executor", session_date)
        result.skipped_executor = True
        return

    if skip or "executor" in completed_stages:
        logger.info("[Stage 3] Trading executor — SKIPPED%s",
                    " (completed in prior run)" if "executor" in completed_stages else "")
        return

    logger.info("[Stage 3] Running trading session")
    _start_stage(session_id, "executor")
    try:
        result.trading_result = run_trading_session(dry_run=dry_run, model=executor_model)
        _complete_stage(session_id, "executor")
    except Exception as e:
        result.trading_error = str(e)
        _fail_stage(session_id, "executor", str(e))
        logger.error("Trading session failed: %s", e)
```

In `run_session`, replace the Stage 3 block (both the short-circuit and the main stage body — lines ~216–252 in the current file) with:
```python
_run_executor_stage(result, session_id, completed_stages, skip_executor, dry_run, executor_model, today)
```

- [ ] **Step 8: Run tests after Stage 3.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | tail -3
```
Expected: `41 passed`. Both `TestExecutorPlaybookDependency` tests specifically exercise the short-circuit; if either fails, recheck the condition ordering in the helper.

- [ ] **Step 9: Extract Stage 4 (strategy reflection).**

Add:
```python
def _run_strategy_stage(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
) -> None:
    if skip or "strategy" in completed_stages:
        logger.info("[Stage 4] Strategy reflection — SKIPPED%s",
                    " (completed in prior run)" if "strategy" in completed_stages else "")
        result.skipped_strategy = True
        return
    logger.info("[Stage 4] Running strategy reflection")
    _start_stage(session_id, "strategy")
    try:
        result.strategy_result = run_strategy_reflection(
            model=DEFAULT_REFLECTION_MODEL,
            max_turns=10,
            trading_result=result.trading_result,
        )
        _complete_stage(session_id, "strategy")
    except Exception as e:
        result.strategy_error = str(e)
        _fail_stage(session_id, "strategy", str(e))
        logger.error("Strategy reflection failed: %s", e)
```

Replace Stage 4 with:
```python
_run_strategy_stage(result, session_id, completed_stages, skip_strategy)
```

- [ ] **Step 10: Extract Stages 5, 5b, 6 (twitter, bluesky, dashboard) using the same pattern.**

These three are structurally identical. Add:

```python
def _run_twitter_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set, skip: bool,
) -> None:
    if skip or "twitter" in completed_stages:
        logger.info("[Stage 5] Twitter posting — SKIPPED%s",
                    " (completed in prior run)" if "twitter" in completed_stages else "")
        result.skipped_twitter = True
        return
    logger.info("[Stage 5] Running Twitter posting")
    _start_stage(session_id, "twitter")
    try:
        result.twitter_result = run_twitter_stage()
        _complete_stage(session_id, "twitter")
    except Exception as e:
        result.twitter_error = str(e)
        _fail_stage(session_id, "twitter", str(e))
        logger.error("Twitter stage failed: %s", e)


def _run_bluesky_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set, skip: bool,
) -> None:
    if skip or "bluesky" in completed_stages:
        logger.info("[Stage 5b] Bluesky posting — SKIPPED%s",
                    " (completed in prior run)" if "bluesky" in completed_stages else "")
        result.skipped_bluesky = True
        return
    logger.info("[Stage 5b] Running Bluesky posting")
    _start_stage(session_id, "bluesky")
    try:
        result.bluesky_result = run_bluesky_stage()
        _complete_stage(session_id, "bluesky")
    except Exception as e:
        result.bluesky_error = str(e)
        _fail_stage(session_id, "bluesky", str(e))
        logger.error("Bluesky stage failed: %s", e)


def _run_dashboard_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set, skip: bool,
) -> None:
    if skip or "dashboard" in completed_stages:
        logger.info("[Stage 6] Dashboard publish — SKIPPED%s",
                    " (completed in prior run)" if "dashboard" in completed_stages else "")
        result.skipped_dashboard = True
        return
    logger.info("[Stage 6] Publishing public dashboard")
    _start_stage(session_id, "dashboard")
    try:
        result.dashboard_result = run_dashboard_stage()
        _complete_stage(session_id, "dashboard")
    except Exception as e:
        result.dashboard_error = str(e)
        _fail_stage(session_id, "dashboard", str(e))
        logger.error("Dashboard publish failed: %s", e)
```

Note the `_wrapper` suffix: `run_twitter_stage`, `run_bluesky_stage`, `run_dashboard_stage` are imported names from sibling modules — we can't shadow them. Choose `_wrapper` consistently.

Replace Stages 5, 5b, 6 with:
```python
_run_twitter_stage_wrapper(result, session_id, completed_stages, skip_twitter)
_run_bluesky_stage_wrapper(result, session_id, completed_stages, skip_bluesky)
_run_dashboard_stage_wrapper(result, session_id, completed_stages, skip_dashboard)
```

- [ ] **Step 11: Extract the idempotency check and session-status finalization into helpers.**

Add:
```python
def _check_and_record_session(force: bool, session_date) -> tuple[int | None, set, str | None]:
    """Returns (session_id, completed_stages, early_error).

    early_error is non-None when the caller should return immediately
    (e.g., session already completed and force=False).
    """
    session_id: int | None = None
    completed_stages: set = set()
    if not force:
        try:
            existing = get_session_for_date(session_date)
            if existing and existing["status"] == "completed":
                logger.warning("Session already completed for %s. Use --force to override.", session_date)
                return None, set(), f"Session already completed for {session_date}"
            if existing:
                completed_stages = get_completed_stages(existing["id"])
                if completed_stages:
                    logger.info("Resuming session — already completed: %s", completed_stages)
        except Exception as e:
            logger.warning("Could not check session status: %s — proceeding", e)
    try:
        session_id = insert_session_record(session_date)
        logger.info("Session ID: %d", session_id)
    except Exception as e:
        logger.warning("Could not create session record: %s — proceeding without tracking", e)
    return session_id, completed_stages, None


_ERROR_FIELDS = (
    "learning_error", "pipeline_error", "strategist_error", "trading_error",
    "strategy_error", "twitter_error", "bluesky_error", "dashboard_error",
)


def _finalize_session(result: SessionResult, session_id: int | None) -> None:
    if session_id:
        try:
            if result.has_errors:
                error_summary = "; ".join(
                    str(getattr(result, f)) for f in _ERROR_FIELDS if getattr(result, f)
                )
                fail_session(session_id, error_summary)
            else:
                complete_session(session_id)
        except Exception as e:
            logger.warning("Could not update session status: %s", e)

    logger.info("=" * 60)
    logger.info("Session complete in %.1fs", result.duration_seconds)
    if result.has_errors:
        for field_name in _ERROR_FIELDS:
            err = getattr(result, field_name)
            if err:
                logger.error("  %s: %s", field_name, err)
    else:
        logger.info("  All stages completed successfully")
    logger.info("=" * 60)
```

Use `_ERROR_FIELDS` to replace the duplicated list in `SessionResult.has_errors` as well:
```python
@property
def has_errors(self) -> bool:
    return any(getattr(self, f) for f in _ERROR_FIELDS)
```
(Move `_ERROR_FIELDS` above `SessionResult` so it's defined first.)

- [ ] **Step 12: Rewrite `run_session` to be a thin driver.**

Target body (inside `run_session`, replacing lines ~103 to end):
```python
start = time.monotonic()
result = SessionResult(
    skipped_pipeline=skip_pipeline, skipped_ideation=skip_ideation,
    skipped_executor=skip_executor, skipped_strategy=skip_strategy,
    skipped_twitter=skip_twitter, skipped_bluesky=skip_bluesky,
    skipped_dashboard=skip_dashboard,
)

from datetime import date
today = date.today()

session_id, completed_stages, early_error = _check_and_record_session(force, today)
if early_error:
    result.learning_error = early_error
    result.duration_seconds = time.monotonic() - start
    return result

attribution_constraints = _run_learning_refresh(result)
_run_pipeline_stage(result, session_id, completed_stages, skip_pipeline, pipeline_hours, pipeline_limit)
_run_strategist_stage(
    result, session_id, completed_stages, skip_ideation,
    model, max_turns, attribution_constraints, today,
)
_run_executor_stage(result, session_id, completed_stages, skip_executor, dry_run, executor_model, today)
_run_strategy_stage(result, session_id, completed_stages, skip_strategy)
_run_twitter_stage_wrapper(result, session_id, completed_stages, skip_twitter)
_run_bluesky_stage_wrapper(result, session_id, completed_stages, skip_bluesky)
_run_dashboard_stage_wrapper(result, session_id, completed_stages, skip_dashboard)

result.duration_seconds = time.monotonic() - start
_finalize_session(result, session_id)
return result
```

- [ ] **Step 13: Run the full session test file.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py -q 2>&1 | tail -5
```
Expected: `41 passed`.

- [ ] **Step 14: Run ruff — `v2/session.py` should drop out of PLR0915.**

```
task lint 2>&1 | grep PLR0915
```
Expected: only `v2/trader.py:74:5: PLR0915 ...` remains. If `v2/session.py` is still listed, count the remaining statements in `run_session` and extract one more coherent block.

- [ ] **Step 15: Verify branch coverage still 100%.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py --cov=v2.session --cov-branch --cov-fail-under=100 -q 2>&1 | tail -5
```
Expected: `--cov-fail-under=100` passes.

- [ ] **Step 16: Commit.**

```
git add v2/session.py
git commit -m "refactor(v2/session): extract per-stage helpers to satisfy PLR0915"
```

---

## Task 6: Refactor `v2/trader.py::run_trading_session`

**Why:** 257 statements in one function, mixing early-exits, per-decision intent resolution, order execution, and bulk decision logging. The 16 passing `test_trader.py` tests are the safety net. Extract in phases with tests between each phase.

**Files:**
- Modify: `v2/trader.py`

- [ ] **Step 1: Baseline.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
task lint 2>&1 | grep PLR0915
```
Expected: `16 passed` and `v2/trader.py:74:5: PLR0915 Too many statements (257 > 50)`.

- [ ] **Step 2: Lift top-level imports.**

Move these imports from inside `run_trading_session` to the module top alongside the existing imports:

```python
import os

from alpaca.data.historical import StockHistoricalDataClient
from .agent import ExecutorInput
from .database.trading_db import update_playbook_action_status
from .risk import check_sector_concentration
```

Delete the inline `import os`, `from alpaca.data.historical ...`, `from .agent import ExecutorInput`, `from .risk import check_sector_concentration`, and the three inline `from .database.trading_db import update_playbook_action_status` occurrences.

- [ ] **Step 3: Test + commit.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
git add v2/trader.py
git commit -m "refactor(v2/trader): hoist inline imports to module top"
```
Expected: `16 passed`.

- [ ] **Step 4: Extract `_sync_from_alpaca`.**

Add near the top of `v2/trader.py` (above `run_trading_session`):

```python
def _sync_from_alpaca(errors: list[str]) -> tuple[int, int]:
    """Sync positions and open orders; return counts, append failures to errors."""
    try:
        positions_synced = sync_positions_from_alpaca()
        logger.info("Synced %d positions", positions_synced)
    except Exception as e:
        errors.append(f"Position sync failed: {e}")
        logger.error("Position sync failed: %s", e)
        positions_synced = 0

    try:
        orders_synced = sync_orders_from_alpaca()
        logger.info("Synced %d open orders", orders_synced)
    except Exception as e:
        errors.append(f"Order sync failed: {e}")
        logger.error("Order sync failed: %s", e)
        orders_synced = 0

    return positions_synced, orders_synced
```

In `run_trading_session`, replace the Step 1 block with:
```python
logger.info("[Step 1] Syncing positions and orders from Alpaca")
positions_synced, orders_synced = _sync_from_alpaca(errors)
```

- [ ] **Step 5: Test.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
```
Expected: `16 passed`.

- [ ] **Step 6: Extract `_snapshot_account`.**

Add:
```python
def _snapshot_account(errors: list[str]) -> tuple[dict | None, int]:
    """Take account snapshot; return (account_info, snapshot_id) or (None, 0) on failure."""
    try:
        account_info = get_account_info()
        snapshot_id = take_account_snapshot()
        logger.info("Snapshot ID: %d", snapshot_id)
        logger.info(
            "Portfolio value: $%s  Buying power: $%s",
            f"{float(account_info['portfolio_value']):,.2f}",
            f"{float(account_info['buying_power']):,.2f}",
        )
        return account_info, snapshot_id
    except Exception as e:
        errors.append(f"Account snapshot failed: {e}")
        logger.error("Account snapshot failed: %s", e, exc_info=True)
        return None, 0
```

In `run_trading_session`, replace the Step 2 block (including the early-return `TradingSessionResult` on exception) with:
```python
logger.info("[Step 2] Taking account snapshot")
account_info, snapshot_id = _snapshot_account(errors)
if account_info is None:
    return TradingSessionResult(
        timestamp=timestamp,
        account_snapshot_id=0,
        positions_synced=positions_synced,
        orders_synced=orders_synced,
        decisions_made=0,
        trades_executed=0,
        trades_failed=0,
        total_buy_value=Decimal(0),
        total_sell_value=Decimal(0),
        errors=errors,
    )
```

- [ ] **Step 7: Test.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
```
Expected: `16 passed`.

- [ ] **Step 8: Extract `_build_executor_context`.**

Add:
```python
def _build_executor_context(account_info: dict, data_client, errors: list[str]):
    """Build executor input and augment risk_notes with sector concentration warnings."""
    try:
        executor_input = build_executor_input(account_info)
        logger.info("Executor input built")
    except Exception as e:
        errors.append(f"Context build failed: {e}")
        logger.error("Context build failed: %s", e, exc_info=True)
        executor_input = ExecutorInput(
            playbook_actions=[],
            positions=[],
            account=account_info,
            attribution_summary={},
            recent_outcomes=[],
            market_outlook=f"Error building context: {e}",
            risk_notes="",
        )

    position_values = {}
    for p in get_positions():
        price = get_latest_price(p["ticker"], client=data_client)
        if price:
            position_values[p["ticker"]] = p["shares"] * price
    sector_warnings = check_sector_concentration(position_values, account_info["portfolio_value"])
    if sector_warnings:
        logger.warning("Sector concentration warnings: %s", sector_warnings)
        if executor_input.risk_notes:
            executor_input.risk_notes += "\n" + "\n".join(sector_warnings)
        else:
            executor_input.risk_notes = "\n".join(sector_warnings)

    return executor_input
```

In `run_trading_session`, replace Step 3 (context build + risk injection) with:
```python
logger.info("[Step 3] Building executor input")
executor_input = _build_executor_context(account_info, data_client, errors)
```

- [ ] **Step 9: Test.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
```
Expected: `16 passed`.

- [ ] **Step 10: Extract `_resolve_decision_qty`.**

Add:
```python
def _resolve_decision_qty(
    decision, held: Decimal, price: Decimal,
    portfolio_value: Decimal, buying_power: Decimal,
) -> Decimal:
    """Resolve an intent to a concrete share count. Raises IntentError on failure."""
    if decision.action == "sell":
        intent = SellIntent(
            type=decision.intent_type,
            magnitude=(
                Decimal(str(decision.intent_magnitude))
                if decision.intent_magnitude is not None else None
            ),
        )
        return resolve_sell_intent(
            intent, held=held, price=price, portfolio_value=portfolio_value,
        )
    if decision.action == "buy":
        if decision.intent_magnitude is None:
            raise IntentError("buy intents require a magnitude")
        intent = BuyIntent(
            type=decision.intent_type,
            magnitude=Decimal(str(decision.intent_magnitude)),
        )
        return resolve_buy_intent(
            intent, held=held, price=price,
            portfolio_value=portfolio_value, buying_power=buying_power,
        )
    raise IntentError(f"unsupported action: {decision.action}")
```

In `run_trading_session`, replace the `try: / if decision.action == "sell": ... elif decision.action == "buy": ... else: raise` block inside the decision loop with:
```python
try:
    resolved_qty = _resolve_decision_qty(
        decision, held=held, price=price,
        portfolio_value=portfolio_value, buying_power=buying_power,
    )
except IntentError as e:
    errors.append(f"{decision.ticker} intent error: {e}")
    logger.warning("%s: INVALID - intent error: %s", decision.ticker, e)
    trades_failed += 1
    decision.reasoning = f"[REJECTED: intent error: {e}] {decision.reasoning}"
    decision.action = "invalid"
    continue
```

- [ ] **Step 11: Test.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
```
Expected: `16 passed`.

- [ ] **Step 12: Extract `_precheck_sell_against_alpaca`.**

Add:
```python
def _precheck_sell_against_alpaca(
    decision, held: Decimal, errors: list[str],
) -> bool:
    """Return True if the sell should proceed; False if it was fully rejected.

    Contract: returning False always represents a rejection worth counting toward
    trades_failed, so the caller should unconditionally `trades_failed += 1` on False.
    Trim-to-available is not a rejection and returns True after mutating
    decision.quantity.
    """
    try:
        available = get_live_available_qty(decision.ticker)
    except Exception as e:
        logger.warning(
            "%s: live availability check failed (%s) — proceeding",
            decision.ticker, e,
        )
        return True

    if available is None or available >= decision.quantity:
        return True

    if available <= Decimal("0.0001"):
        reason = f"Alpaca reports 0 available shares (DB said {held})"
        errors.append(f"{decision.ticker} pre-submit check failed: {reason}")
        logger.warning("%s: SKIP - %s", decision.ticker, reason)
        if decision.playbook_action_id:
            try:
                update_playbook_action_status(decision.playbook_action_id, "skipped")
            except Exception:
                pass
        decision.reasoning = f"[REJECTED: {reason}] {decision.reasoning}"
        decision.action = "invalid"
        return False

    logger.info(
        "%s: trimming sell from %s to %s (Alpaca available)",
        decision.ticker, decision.quantity, available,
    )
    decision.quantity = available
    return True
```

In `run_trading_session`, replace the existing `if decision.action == "sell" and not dry_run:` pre-submit block with:
```python
if decision.action == "sell" and not dry_run:
    if not _precheck_sell_against_alpaca(decision, held, errors):
        trades_failed += 1
        continue
```

- [ ] **Step 13: Test.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
```
Expected: `16 passed`.

- [ ] **Step 14: Extract `_log_decisions`.**

Add:
```python
def _log_decisions(
    response, order_ids: dict, order_results: dict,
    data_client, account_info: dict, errors: list[str],
) -> int:
    """Insert decision rows and signal-links. Returns count of successfully logged decisions."""
    signals_used = format_decisions_for_logging(response)
    logged_count = 0
    for i, decision in enumerate(response.decisions):
        try:
            result = order_results.get(i)
            price = (
                result.filled_avg_price if result and result.filled_avg_price
                else get_latest_price(decision.ticker, client=data_client)
            )
            if price is None and decision.action in ("buy", "sell"):
                errors.append(f"No price available for {decision.ticker} — skipping decision log")
                logger.error("Cannot log decision for %s: no price available", decision.ticker)
                continue

            existing_id = check_decision_exists(date.today(), decision.ticker, decision.action)
            if existing_id:
                logger.warning(
                    "%s: duplicate %s decision — already logged as ID %d",
                    decision.ticker, decision.action, existing_id,
                )
                continue

            if result and result.filled_qty is not None:
                logged_qty = Decimal(str(result.filled_qty))
            elif decision.quantity:
                logged_qty = decision.quantity
            else:
                logged_qty = None

            decision_id = insert_decision(
                decision_date=date.today(),
                ticker=decision.ticker,
                action=decision.action,
                quantity=logged_qty,
                price=price,
                reasoning=decision.reasoning,
                signals_used=signals_used,
                account_equity=account_info["portfolio_value"],
                buying_power=account_info["buying_power"],
                playbook_action_id=decision.playbook_action_id,
                is_off_playbook=decision.is_off_playbook,
                order_id=order_ids.get(i),
            )
            logged_count += 1
        except Exception as e:
            errors.append(f"Failed to log decision for {decision.ticker}: {e}")
            logger.error("Error logging %s: %s", decision.ticker, e)
            continue

        if decision.signal_refs:
            try:
                validated_refs = validate_signal_refs(decision.signal_refs)
                if len(validated_refs) < len(decision.signal_refs):
                    logger.warning(
                        "%s: stripped %d invalid signal refs",
                        decision.ticker,
                        len(decision.signal_refs) - len(validated_refs),
                    )
                if validated_refs:
                    signal_links = [
                        (decision_id, ref["type"], ref["id"])
                        for ref in validated_refs
                    ]
                    insert_decision_signals_batch(signal_links)
            except Exception as e:
                errors.append(f"Failed to log signal links for {decision.ticker}: {e}")
        elif decision.action in ("buy", "sell"):
            logger.warning(
                "%s: no signal_refs cited — decision will be excluded from attribution",
                decision.ticker,
            )
    return logged_count
```

In `run_trading_session`, replace the entire Step 6 block with:
```python
logger.info("[Step 6] Logging decisions")
logged_count = _log_decisions(
    response, order_ids, order_results, data_client, account_info, errors,
)
logger.info("Logged %d decisions (%d emitted by executor)", logged_count, len(response.decisions))
```

- [ ] **Step 15: Test.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py -q 2>&1 | tail -3
```
Expected: `16 passed`.

- [ ] **Step 16: Run ruff — `v2/trader.py` should drop out of PLR0915.**

```
task lint 2>&1 | grep PLR0915 || echo "CLEAN"
```
Expected: `CLEAN`. If PLR0915 still reports `v2/trader.py`, count remaining statements in `run_trading_session`. The most likely additional extraction is the intra-decision execution block (lines handling `execute_market_order` → `wait_for_fill` → buying-power refresh → thesis-close) — if needed, lift that into `_execute_decision_order(decision, price, data_client, dry_run, positions, buying_power_ref, portfolio_value_ref, total_buy_ref, total_sell_ref, order_ids, order_results, i, errors) -> bool`.

- [ ] **Step 17: Full lint passes.**

```
task lint 2>&1 | tail -3
```
Expected: `All checks passed!`

- [ ] **Step 18: Verify branch coverage still 100%.**

```
docker compose exec -T trading python -m pytest tests/v2/test_trader.py --cov=v2.trader --cov-branch --cov-fail-under=100 -q 2>&1 | tail -5
```
Expected: `--cov-fail-under=100` passes.

- [ ] **Step 19: Commit.**

```
git add v2/trader.py
git commit -m "refactor(v2/trader): extract sync/snapshot/context/intent/log helpers to satisfy PLR0915"
```

---

## Task 7: Final verification

- [ ] **Step 1: Run ruff across the repo.**

```
task lint
```
Expected: `All checks passed!`.

- [ ] **Step 2: Run the v2 test subdirectory (excluding the two known-broken tests).**

```
docker compose exec -T trading python -m pytest tests/v2/ --deselect tests/v2/test_entertainment.py::TestRunEntertainmentPipeline::test_happy_path -q 2>&1 | tail -5
```
Expected: all pass; no new failures introduced by the refactor.

- [ ] **Step 3: Confirm 100% branch coverage on both target files.**

```
docker compose exec -T trading python -m pytest tests/v2/test_session.py tests/v2/test_trader.py \
  --cov=v2.session --cov=v2.trader --cov-branch --cov-fail-under=100 --cov-report=term-missing -q 2>&1 | tail -15
```
Expected: both files at 100% line + 100% branch, `--cov-fail-under=100` gate passes, no `Missing` entries for either file.

- [ ] **Step 4: Confirm pre-commit hook passes.**

```
./.githooks/pre-commit
```
Expected: `ruff check` passes, `pytest` runs — may still fail on the pre-existing `test_classifier.py` collection error and `test_entertainment.py::test_happy_path`. If those two are the only failures, flag them to the user and stop; do not attempt to fix them under this plan.
