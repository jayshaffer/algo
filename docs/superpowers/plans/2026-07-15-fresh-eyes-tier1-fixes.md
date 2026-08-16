# Fresh-Eyes Audit: Same-Day Ops + Tier 1 Money-Path Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the triage head of `docs/audits/2026-07-15-fresh-eyes-audit.md`: the three same-day items (0.1 hiatus visibility + halt switch, 3.1 DB backups, 3.3 apply drifted migrations) and all six Tier 1 money-path fixes (1.1–1.6) that are prerequisites for ever re-enabling trading.

**Architecture:** Ops items touch `Taskfile.yml`, `crontab`, `run-docker.sh`, and new docs; no schema changes. Code fixes are all in `v2/trader.py`, `v2/session.py`, and `v2/database/trading_db.py`, each TDD'd in the existing mocked test suite (`tests/v2/test_trader.py` `_happy_path`/`_make_decision` harness, `tests/v2/test_session.py`).

**Tech Stack:** Python 3.12 (in docker), pytest (mocked — no live API/DB), go-task, psql/pg_dump inside compose containers.

## Global Constraints

- All work happens on branch `fresh-eyes-audit-2026-07` (already contains the audit report commit).
- Tests run **in docker**, never on host python (host is 3.10): `docker compose exec -T trading python3 -m pytest tests/ -q`. Bring the stack up first with `docker compose up -d db trading` (the trading service's command is `sleep infinity` — it never trades on its own).
- All DB access is raw SQL + psycopg2 via the `get_cursor()` context manager; tests mock it (`mock_db`, `mock_cursor` fixtures).
- Do NOT touch the user's installed crontab (`crontab -l`). Only the repo `crontab` file is edited; installing it is the user's call (documented in the runbook).
- Never modify `db/init/` in this plan — no schema changes are made.
- The prod Alpaca account is empty and trading is on deliberate hiatus; the prod `db` container may be brought up freely for migrations/backups.
- Match existing code style: module-level docstring comments reference audit/phase IDs (e.g. "P1.6:", "T1.2:") — new code comments should reference the audit finding (e.g. "A.1:").
- Commit after every task, message style matches repo history (imperative, no prefix noise), ending with the Claude co-author trailer.

---

### Task 1: Apply drifted migrations to prod and paper DBs (audit 3.3 / B.4)

**Files:** none (operational task, no repo changes)

**Interfaces:**
- Produces: prod + paper `schema_migrations` containing `013`, `014`, `015`; `model_pricing` rows for `claude-fable-5` and corrected Opus 4.x pricing.

- [ ] **Step 1: Check current drift on prod**

Run: `docker compose up -d db && docker compose exec -T db psql -U algo -d trading -tAc "SELECT filename FROM schema_migrations ORDER BY filename DESC LIMIT 3"`
Expected: newest row is `012_supervisor_memos.sql` (per audit B.4). If 013–015 already present, skip to Step 4.

- [ ] **Step 2: Apply prod migrations**

Run: `task db:migrate`
Expected: `==> applying 013_...`, `014_model_pricing_fable5_opus4x.sql`, `015_...` lines; no errors.

- [ ] **Step 3: Verify prod pricing rows**

Run: `docker compose exec -T db psql -U algo -d trading -c "SELECT model, input_per_mtok, output_per_mtok FROM model_pricing WHERE model LIKE 'claude-fable%' OR model LIKE 'claude-opus-4%' ORDER BY model"`
Expected: a `claude-fable-5` row exists; Opus 4.x rows show the corrected (post-repricing) rates, not $15/$75.

- [ ] **Step 4: Apply and verify paper migrations**

Run: `task paper:db:migrate` then the same SELECTs via `docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper psql -U algo -d trading ...`
Expected: same end state as prod.

- [ ] **Step 5: No commit (nothing in-repo changed); note results for the final summary**

---

### Task 2: DB backup tasks + nightly cron lines + recovery runbook (audit 3.1 / C.1, part of C.6)

**Files:**
- Modify: `Taskfile.yml` (add `db:backup`, `paper:db:backup` next to the existing `db:migrate` tasks)
- Modify: `crontab` (add two backup lines)
- Modify: `.gitignore` (add `backups/`)
- Create: `docs/runbook-recovery.md`

**Interfaces:**
- Produces: `task db:backup` / `task paper:db:backup` writing `backups/prod-YYYYmmdd-HHMMSS.dump` / `backups/paper-...` (`pg_dump -Fc`), keeping the newest 14, optionally copying to `$ALGO_BACKUP_COPY_DIR`. Task 3's runbook section links here.

- [ ] **Step 1: Add the two Taskfile targets** (mirror the `db:migrate`/`paper:db:migrate` pair's structure and psql_exec style)

```yaml
  db:backup:
    desc: pg_dump the prod db to backups/ (keeps newest 14; set ALGO_BACKUP_COPY_DIR for an off-WSL copy)
    deps: [docker:up]
    cmds:
      - |
        mkdir -p backups
        f="backups/prod-$(date +%Y%m%d-%H%M%S).dump"
        docker compose exec -T db pg_dump -U algo -d trading -Fc > "$f"
        echo "==> wrote $f ($(du -h "$f" | cut -f1))"
        ls -t backups/prod-*.dump | tail -n +15 | xargs -r rm --
        if [ -n "${ALGO_BACKUP_COPY_DIR:-}" ]; then
          mkdir -p "$ALGO_BACKUP_COPY_DIR" && cp "$f" "$ALGO_BACKUP_COPY_DIR"/ && echo "==> copied to $ALGO_BACKUP_COPY_DIR"
        fi

  paper:db:backup:
    desc: pg_dump the paper db to backups/ (keeps newest 14; set ALGO_BACKUP_COPY_DIR for an off-WSL copy)
    deps: [paper:up]
    cmds:
      - |
        mkdir -p backups
        f="backups/paper-$(date +%Y%m%d-%H%M%S).dump"
        docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper pg_dump -U algo -d trading -Fc > "$f"
        echo "==> wrote $f ($(du -h "$f" | cut -f1))"
        ls -t backups/paper-*.dump | tail -n +15 | xargs -r rm --
        if [ -n "${ALGO_BACKUP_COPY_DIR:-}" ]; then
          mkdir -p "$ALGO_BACKUP_COPY_DIR" && cp "$f" "$ALGO_BACKUP_COPY_DIR"/ && echo "==> copied to $ALGO_BACKUP_COPY_DIR"
        fi
```

- [ ] **Step 2: Add `backups/` to `.gitignore`**

- [ ] **Step 3: Add nightly cron lines to the repo `crontab` file** (after the existing jobs; these stay active during the hiatus — backups must run regardless)

```
# Nightly DB backups (8 PM MST, Mon-Fri) — see docs/runbook-recovery.md
0 20 * * 1-5 cd /home/jay/dev/algo/ && task db:backup >> logs/backup.log 2>&1
5 20 * * 1-5 cd /home/jay/dev/algo/ && task paper:db:backup >> logs_paper/backup.log 2>&1
```

- [ ] **Step 4: Create `docs/runbook-recovery.md`** with these sections (write real content, not stubs):
  - **Backups**: what `task db:backup`/`paper:db:backup` do, where dumps land, retention (14), `ALGO_BACKUP_COPY_DIR` for an off-WSL copy (recommend a `/mnt/c/...` path), cron schedule, note that `db:backup` leaves the prod `db` container running.
  - **Restore procedure**:
    ```bash
    docker compose up -d db
    docker compose exec -T db pg_restore -U algo -d trading --clean --if-exists < backups/prod-<stamp>.dump
    task db:migrate   # re-apply anything newer than the dump
    ```
  - **Secrets inventory** (names only, never values): `.env` / `.env.paper` hold `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` (Alpaca dashboard), `ANTHROPIC_API_KEY` (console.anthropic.com), Cloudflare account/token/project (Cloudflare dashboard), `POSTGRES_*`. Point to `.env.example` for the full key list.
  - **Host bootstrap**: install cron with `crontab /home/jay/dev/algo/crontab`; cron only runs because Windows Task Scheduler fires `start-wsl-cron.bat` (`wsl -u root service cron start`) at login — re-create that entry on a new machine; then `task db:migrate` + `task paper:db:migrate`.
  - **Halt / resume**: placeholder section header — Task 3 fills it in.

- [ ] **Step 5: Verify a real backup + restore listing**

Run: `task db:backup` then `docker compose exec -T db pg_restore --list /dev/stdin < backups/prod-*.dump | head -5`
Expected: dump file exists, `pg_restore --list` prints a TOC (proves the archive is readable). Run `task paper:db:backup` too.

- [ ] **Step 6: Commit**

```bash
git add Taskfile.yml crontab .gitignore docs/runbook-recovery.md
git commit -m "Add DB backup tasks, nightly backup cron, and recovery runbook (audit 3.1)"
```

---

### Task 3: Halt switch + hiatus visibility (audit 0.1 + C.5)

**Files:**
- Create: `HALT` (committed sentinel documenting the hiatus)
- Modify: `run-docker.sh` (check sentinel before starting containers)
- Modify: `v2/session.py` (env-var check at top of `run_session`)
- Modify: `crontab` (comment the prod session + weekly learn lines with a HIATUS note)
- Modify: `.env.example`, `CLAUDE.md` (document `ALGO_TRADING_HALTED`)
- Modify: `docs/runbook-recovery.md` (fill in Halt/Resume section)
- Test: `tests/v2/test_session.py`

**Interfaces:**
- Produces: two independent halt mechanisms — host-side `HALT` sentinel (checked by `run-docker.sh`, git-visible) and container-side `ALGO_TRADING_HALTED` env var (checked in `run_session`, reuses `result.idempotent_skip` so `main()` exits 0).

- [ ] **Step 1: Write the failing test** (in `tests/v2/test_session.py`, near other `run_session` tests; use the file's existing import style)

```python
class TestTradingHalted:
    def test_halted_env_skips_session(self, monkeypatch):
        """C.5: ALGO_TRADING_HALTED short-circuits run_session before any
        stage runs or a sessions row is written — deliberate halts must be
        loud in logs but exit 0 (not a failure alert)."""
        monkeypatch.setenv("ALGO_TRADING_HALTED", "1")
        with patch("v2.session.insert_session_record") as mock_insert, \
             patch("v2.session.expire_stale_playbook_actions") as mock_expire:
            result = run_session()
        assert result.idempotent_skip
        assert "ALGO_TRADING_HALTED" in result.idempotent_skip
        mock_insert.assert_not_called()
        mock_expire.assert_not_called()

    def test_halted_env_false_values_do_not_skip(self, monkeypatch):
        monkeypatch.setenv("ALGO_TRADING_HALTED", "0")
        with patch("v2.session._check_and_record_session",
                   return_value=(None, set(), "already ran")):
            result = run_session()
        assert result.idempotent_skip == "already ran"
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_session.py -k TradingHalted -v`
Expected: first test FAILS (no halt check exists); second may pass already.

- [ ] **Step 3: Implement the check at the top of `run_session`** (`v2/session.py`, immediately after `result = SessionResult(...)` / `today = current_market_date()`, BEFORE `expire_stale_playbook_actions`; ensure `import os` is present)

```python
    # C.5: explicit operator kill switch. Deliberate halts (e.g. the 2026-06
    # hiatus) previously had no mechanism besides hand-editing the installed
    # crontab, which is invisible to git and ambiguous to future readers.
    # Reuses idempotent_skip so main() logs loudly and exits 0 — a halt is
    # not a failure. Host-side twin: the HALT sentinel file in run-docker.sh.
    if os.environ.get("ALGO_TRADING_HALTED", "").strip().lower() in ("1", "true", "yes"):
        result.idempotent_skip = (
            "ALGO_TRADING_HALTED is set — session skipped "
            "(halt/resume procedure: docs/runbook-recovery.md)"
        )
        result.duration_seconds = time.monotonic() - start
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_session.py -q`
Expected: PASS (whole file — confirm no regression from the early return).

- [ ] **Step 5: Add the sentinel check to `run-docker.sh`** (after the `$# -eq 0` usage check, BEFORE `trap cleanup EXIT` so exiting doesn't tear down or alert)

```bash
# C.5: operator kill switch. `touch HALT` stops cron sessions cold (exit 0,
# no failure alert); `rm HALT` resumes. See docs/runbook-recovery.md.
if [ -f "$SCRIPT_DIR/HALT" ]; then
    echo "[$(date -Is)] HALT sentinel present — skipping: $CMD_DESC"
    exit 0
fi
```

- [ ] **Step 6: Create the committed `HALT` file** documenting the current hiatus:

```
Trading hiatus — deliberate halt.

Since:   2026-06 (owner decision, confirmed 2026-07-15 during the fresh-eyes audit)
Why:     Prod Alpaca account fully liquidated (positions closed, cash withdrawn).
         Economics under review (audit finding 0.2); Tier 1 money-path fixes
         were prerequisites for any resume.
Effect:  run-docker.sh exits 0 while this file exists, so cron session/learn
         jobs are no-ops. ALGO_TRADING_HALTED in .env is the in-container twin.
Resume:  see docs/runbook-recovery.md "Halt / Resume". Short version:
         review audit Tier 1/0.2 status, `rm HALT`, commit, reinstall crontab
         (`crontab /home/jay/dev/algo/crontab`) with the session line active.
```

- [ ] **Step 7: Sync the repo `crontab`** — comment out the prod session line (matching installed reality) and the weekly learn line, each annotated:

```
# HIATUS since 2026-06 (see HALT file + docs/runbook-recovery.md). The HALT
# sentinel makes these no-ops anyway; kept commented to mirror the installed
# crontab. Uncomment + `rm HALT` + reinstall to resume.
# 0 13 * * 1-5 /home/jay/dev/algo/run-docker.sh trading python -m v2.session
# 0 5 * * 0 /home/jay/dev/algo/run-docker.sh trading python -m v2.learn --days 60
```

- [ ] **Step 8: Document the knob** — add `ALGO_TRADING_HALTED` to `.env.example` (commented, with one-line description) and to CLAUDE.md's "Optional knobs" list (note: read at session start, runtime — unlike the other knobs it does NOT need a container restart). Fill in the runbook's Halt/Resume section: halt = `touch HALT` (and/or set `ALGO_TRADING_HALTED=1` in `.env`), resume = reverse both + reinstall crontab; note the daily-loss breaker and cost ceiling remain the automated backstops.

- [ ] **Step 9: Commit**

```bash
git add HALT run-docker.sh v2/session.py tests/v2/test_session.py crontab .env.example CLAUDE.md docs/runbook-recovery.md
git commit -m "Add HALT sentinel + ALGO_TRADING_HALTED kill switch, make hiatus git-visible (audit 0.1, C.5)"
```

---

### Task 4: Sell precheck rejects when Alpaca reports no position (audit 1.1 / A.1)

**Files:**
- Modify: `v2/trader.py:282-283` (`_precheck_sell_against_alpaca`)
- Test: `tests/v2/test_trader.py` (class `TestAlpacaPrecheck`)

**Interfaces:**
- Consumes: `get_live_available_qty(ticker) -> Decimal | None` (None == position does not exist at Alpaca; API errors raise).

- [ ] **Step 1: Write the failing test** (in `TestAlpacaPrecheck`, mirroring `test_live_availability_check_exception_skips_sell`)

```python
    def test_no_position_at_alpaca_rejects_sell(self, mock_db, mock_cursor):
        """A.1: get_live_available_qty returns None when Alpaca says the
        position does not exist — strictly worse than 0 available. A stale
        DB row after a failed position sync must not reach the broker (on a
        margin account a market sell of a non-held symbol opens a short).
        """
        decision = _make_decision(ticker="AAPL", action="sell",
                                  intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_live_available_qty": MagicMock(return_value=None),
            })
            result = run_trading_session(dry_run=False)
        mocks["execute_market_order"].assert_not_called()
        assert decision.action == "invalid"
        assert "no position" in decision.reasoning
        assert result.trades_failed == 1
```

- [ ] **Step 2: Run it — must FAIL** (`execute_market_order` gets called today)

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_trader.py -k no_position_at_alpaca -v`

- [ ] **Step 3: Fix `_precheck_sell_against_alpaca`** — replace the combined condition:

```python
    if available is None:
        # A.1: None means Alpaca has no position at all — reject like the
        # zero-available branch instead of sailing through on stale DB state.
        return _reject(f"Alpaca reports no position (DB said {held})")

    if available >= decision.quantity:
        return True
```

- [ ] **Step 4: Run the trader suite**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_trader.py -q`
Expected: all pass. If an existing test relied on `None` passing the precheck, read it — it's asserting the buggy behavior; update it to expect rejection and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add v2/trader.py tests/v2/test_trader.py
git commit -m "Reject sells when Alpaca reports no position at all (audit 1.1)"
```

---

### Task 5: Intra-batch duplicate (ticker, action) decisions execute only once (audit 1.2 / A.2)

**Files:**
- Modify: `v2/trader.py` (new helper `_reject_intra_batch_duplicates` + call in `_execute_decisions`)
- Test: `tests/v2/test_trader.py` (new class `TestIntraBatchDedup`)

**Interfaces:**
- Produces: `_reject_intra_batch_duplicates(response, session_id=None) -> int` — stamps losers `action="invalid"`, returns count rejected. Playbook-backed decision wins over off-playbook regardless of order.

- [ ] **Step 1: Write the failing tests**

```python
class TestIntraBatchDedup:
    """A.2: pre-submit DB dedup only sees rows written in Step 6, so two
    same-(ticker, action) decisions in one executor response both reached
    the broker; the second fill got no decision row."""

    def test_duplicate_buys_execute_once_playbook_wins(self, mock_db, mock_cursor):
        off = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                             is_off_playbook=True)
        pb = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[off, pb])
            result = run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 1
        assert off.action == "invalid"
        assert "duplicate" in off.reasoning
        assert pb.action == "buy"
        assert result.trades_executed == 1
        assert result.trades_failed == 1

    def test_duplicate_buys_playbook_first_keeps_playbook(self, mock_db, mock_cursor):
        pb = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        off = _make_decision(ticker="AAPL", action="buy", playbook_action_id=None,
                             is_off_playbook=True)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[pb, off])
            run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 1
        assert off.action == "invalid"
        assert pb.action == "buy"

    def test_buy_and_sell_same_ticker_not_deduped(self, mock_db, mock_cursor):
        buy = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        sell = _make_decision(ticker="AAPL", action="sell", playbook_action_id=8,
                              intent_type="exit_full", intent_magnitude=None)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[buy, sell], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
            })
            run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 2
```

(Adjust `_make_decision` kwargs to its actual signature — read it first; it's at `tests/v2/test_trader.py:23`.)

- [ ] **Step 2: Run — must FAIL** (both orders currently submit)

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_trader.py -k IntraBatchDedup -v`

- [ ] **Step 3: Implement the helper** (place near `_record_decision_rejection` in `v2/trader.py`)

```python
def _reject_intra_batch_duplicates(response, session_id: int | None = None) -> int:
    """A.2: all three dedup layers (DB rows, client_order_id, playbook
    uniqueness) are blind to duplicates *within one executor response* —
    decision rows are only written after the execution loop, and a playbook
    buy + off-playbook buy of the same ticker sign different order ids.
    Stamp losers invalid before the loop. The playbook-backed decision wins
    so signal/thesis linkage survives. Returns the number rejected.
    """
    rejected = 0
    seen: dict[tuple[str, str], object] = {}
    for decision in response.decisions:
        if decision.action not in ("buy", "sell"):
            continue
        key = (decision.ticker, decision.action)
        prior = seen.get(key)
        if prior is None:
            seen[key] = decision
            continue
        if prior.playbook_action_id is None and decision.playbook_action_id is not None:
            loser, seen[key] = prior, decision
        else:
            loser = decision
        reason = f"duplicate {loser.action} for {loser.ticker} within one executor batch"
        logger.warning("%s: SKIP - %s", loser.ticker, reason)
        _record_decision_rejection(
            session_id=session_id, stage_name="trading", decision=loser,
            reason_code="intra_batch_duplicate", reason_text=reason,
        )
        loser.reasoning = f"[REJECTED: {reason}] {loser.reasoning}"
        loser.action = "invalid"
        rejected += 1
    return rejected
```

And in `_execute_decisions`, right after `totals = _ExecutionTotals()`:

```python
    totals.trades_failed += _reject_intra_batch_duplicates(response, session_id=session_id)
```

- [ ] **Step 4: Run the trader suite** — `docker compose exec -T trading python3 -m pytest tests/v2/test_trader.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add v2/trader.py tests/v2/test_trader.py
git commit -m "Reject intra-batch duplicate (ticker, action) decisions before submit (audit 1.2)"
```

---

### Task 6: Broker-rejected / failed-fill orders stamped invalid, not logged as real trades (audit 1.3 / A.3)

**Files:**
- Modify: `v2/trader.py:337-367` (`_execute_decision_order` failure branches)
- Test: `tests/v2/test_trader.py` (new class `TestFailedOrderLogging`)

**Interfaces:**
- Consumes: `_log_decisions` already logs `action="invalid"` rows without dedup and allows NULL price; backfill/attribution filter `action IN ('buy','sell')` — so stamping is sufficient to keep phantoms out of learning data AND to free the (date, ticker, action) dedup key for a `--force` retry.

- [ ] **Step 1: Write the failing tests**

```python
class TestFailedOrderLogging:
    """A.3: submit/fill failures previously left decision.action untouched,
    so _log_decisions inserted a real buy/sell row (phantom trade in the
    learning data) whose dedup key also blocked same-day --force retries."""

    def test_failed_submit_stamped_invalid(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, error="insufficient buying power",
                    duplicate_client_order_id=False)),
            })
            result = run_trading_session(dry_run=False)
        assert decision.action == "invalid"
        assert "[FAILED:" in decision.reasoning
        assert result.trades_failed == 1
        logged_action = mocks["insert_decision"].call_args.kwargs.get("action")
        assert logged_action == "invalid"

    def test_failed_fill_stamped_invalid_and_action_marked_failed(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "wait_for_fill": MagicMock(return_value=MagicMock(
                    success=False, error="timeout after 30s", order_id="ord-1",
                    filled_qty=None, filled_avg_price=None)),
            })
            run_trading_session(dry_run=False)
        assert decision.action == "invalid"
        assert "[FAILED:" in decision.reasoning
        mocks["update_playbook_action_status"].assert_called_with(7, "failed")

    def test_duplicate_client_order_id_not_stamped(self, mock_db, mock_cursor):
        """P1.6 race-loser stays a benign skip: the winner's row is the real
        record; stamping the loser would create a bogus rejection row."""
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "execute_market_order": MagicMock(return_value=MagicMock(
                    success=False, error="duplicate", duplicate_client_order_id=True)),
            })
            run_trading_session(dry_run=False)
        assert decision.action == "buy"
```

(If `insert_decision` is called positionally, adapt the kwargs assertion — check `_insert_decision_with_retry` first.)

- [ ] **Step 2: Run — first two must FAIL.**

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_trader.py -k FailedOrderLogging -v`

- [ ] **Step 3: Implement.** In `_execute_decision_order`, in the `not result.success` branch AFTER the duplicate_client_order_id early-return, add before `return _DecisionOutcome(...)`:

```python
        # A.3: stamp like every pre-submit rejection path so _log_decisions
        # records an 'invalid' audit row instead of a phantom buy/sell that
        # feeds backfill/attribution and blocks same-day --force retries.
        decision.reasoning = f"[FAILED: {result.error}] {decision.reasoning}"
        decision.action = "invalid"
```

In the fill-failure branch (`if not fill.success:`), add the same stamping plus the playbook status update (mirroring the submit-failure branch):

```python
            decision.reasoning = f"[FAILED: {fill.error}] {decision.reasoning}"
            decision.action = "invalid"
            if decision.playbook_action_id:
                try:
                    update_playbook_action_status(decision.playbook_action_id, "failed")
                except Exception:
                    pass
```

- [ ] **Step 4: Run the trader suite** — all pass; fix any test asserting the old phantom-logging behavior (it's asserting the bug).

- [ ] **Step 5: Commit**

```bash
git add v2/trader.py tests/v2/test_trader.py
git commit -m "Stamp broker-rejected and failed-fill orders invalid instead of logging phantom trades (audit 1.3)"
```

---

### Task 7: Validate LLM-authored thesis_id / playbook_action_id before DB writes (audit 1.4 / A.6, D.3)

**Files:**
- Modify: `v2/trader.py` (new `_validate_llm_ids`, called in `run_trading_session` between Step 4 and Step 5; add `get_active_theses` to the trading_db imports)
- Test: `tests/v2/test_trader.py` (new class `TestLlmIdValidation`; add `"get_active_theses": MagicMock(return_value=[])` to `_happy_path` defaults)

**Interfaces:**
- Consumes: `executor_input.playbook_actions: list[PlaybookAction]` (fields `id`, `ticker`, `thesis_id`); `get_active_theses(ticker=...) -> list[dict]` (rows with `"id"`).
- Produces: `_validate_llm_ids(response, executor_input, session_id=None) -> None` — nulls invalid `decision.playbook_action_id` (setting `is_off_playbook=True`) and invalid `decision.thesis_id`; filters `response.thesis_invalidations` to thesis ids visible in today's playbook.

- [ ] **Step 1: Write the failing tests**

```python
class TestLlmIdValidation:
    """A.6/D.3: these were the only LLM-authored pointers written to the DB
    unvalidated — a hallucinated id could mark an arbitrary historical
    playbook action executed or close/invalidate an unrelated active thesis."""

    def _playbook_action(self, id=7, ticker="AAPL", thesis_id=3):
        return PlaybookAction(
            id=id, ticker=ticker, action="buy", thesis_id=thesis_id,
            reasoning="r", confidence="high", intent_type="add_pct_bp",
            intent_magnitude=Decimal("10"), priority=1)

    def _input_with_action(self, **kw):
        return ExecutorInput(
            playbook_actions=[self._playbook_action(**kw)], positions=[],
            account={}, attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="")

    def test_hallucinated_playbook_action_id_nulled(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="buy", playbook_action_id=999)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "build_executor_input": MagicMock(return_value=self._input_with_action()),
            })
            run_trading_session(dry_run=False)
        assert decision.playbook_action_id is None
        assert decision.is_off_playbook is True
        for call in mocks["update_playbook_action_status"].call_args_list:
            assert call.args[0] != 999

    def test_ticker_mismatch_nulls_playbook_action_id(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="MSFT", action="buy", playbook_action_id=7)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "build_executor_input": MagicMock(return_value=self._input_with_action(ticker="AAPL")),
            })
            run_trading_session(dry_run=False)
        assert decision.playbook_action_id is None

    def test_hallucinated_thesis_id_nulled_no_blind_close(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="sell", playbook_action_id=7,
                                  intent_type="exit_full", intent_magnitude=None,
                                  thesis_id=555)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[decision], overrides={
                "build_executor_input": MagicMock(return_value=self._input_with_action(thesis_id=3)),
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_active_theses": MagicMock(return_value=[]),
            })
            run_trading_session(dry_run=False)
        assert decision.thesis_id is None
        mocks["close_thesis"].assert_not_called()

    def test_thesis_id_matching_active_thesis_kept(self, mock_db, mock_cursor):
        decision = _make_decision(ticker="AAPL", action="sell", playbook_action_id=None,
                                  is_off_playbook=True, intent_type="exit_full",
                                  intent_magnitude=None, thesis_id=42)
        with ExitStack() as stack:
            _happy_path(stack, decisions=[decision], overrides={
                "get_positions": MagicMock(return_value=[{"ticker": "AAPL", "shares": Decimal("5")}]),
                "get_active_theses": MagicMock(return_value=[{"id": 42}]),
            })
            run_trading_session(dry_run=False)
        assert decision.thesis_id == 42

    def test_unknown_thesis_invalidation_dropped(self, mock_db, mock_cursor):
        inv = ThesisInvalidation(thesis_id=888, reason="gone")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[], invalidations=[inv], overrides={
                "build_executor_input": MagicMock(return_value=self._input_with_action(thesis_id=3)),
            })
            run_trading_session(dry_run=False)
        mocks["close_thesis"].assert_not_called()

    def test_known_thesis_invalidation_processed(self, mock_db, mock_cursor):
        inv = ThesisInvalidation(thesis_id=3, reason="broken")
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[], invalidations=[inv], overrides={
                "build_executor_input": MagicMock(return_value=self._input_with_action(thesis_id=3)),
            })
            run_trading_session(dry_run=False)
        mocks["close_thesis"].assert_called_once_with(
            thesis_id=3, status="invalidated", reason="broken")
```

(Import `PlaybookAction`, `ExecutorInput`, `ThesisInvalidation` from `v2.agent` at the top of the test file if not already there; check the existing `test_thesis_invalidations_processed` test for the `ThesisInvalidation` constructor shape and reuse it.)

- [ ] **Step 2: Run — must FAIL** (`docker compose exec -T trading python3 -m pytest tests/v2/test_trader.py -k LlmIdValidation -v`)

- [ ] **Step 3: Implement `_validate_llm_ids`** in `v2/trader.py` (near `_handle_thesis_invalidations`); add `get_active_theses` to the existing trading_db import:

```python
def _validate_llm_ids(response, executor_input, session_id: int | None = None) -> None:
    """A.6/D.3: thesis_id and playbook_action_id are LLM-authored and were
    the only such pointers written to the DB unvalidated (signal_refs get DB
    validation, tickers get normalization). A hallucinated/transposed id
    could mark an arbitrary historical action executed or close an unrelated
    active thesis. Invalid ids are nulled/dropped with a logged warning —
    never fatal, mirroring how signal_refs degrade.
    """
    actions_by_id = {a.id: a for a in (executor_input.playbook_actions or [])}
    known_thesis_ids = {
        a.thesis_id for a in (executor_input.playbook_actions or [])
        if a.thesis_id is not None
    }

    for decision in response.decisions:
        action = None
        if decision.playbook_action_id is not None:
            action = actions_by_id.get(decision.playbook_action_id)
            if action is None or action.ticker != decision.ticker:
                logger.warning(
                    "%s: playbook_action_id=%s not in today's playbook for this "
                    "ticker — treating as off-playbook",
                    decision.ticker, decision.playbook_action_id,
                )
                record_event(
                    session_id=session_id, stage_name="trading",
                    event_type="id_validation",
                    payload={"ticker": decision.ticker, "field": "playbook_action_id",
                             "value": decision.playbook_action_id},
                )
                decision.playbook_action_id = None
                decision.is_off_playbook = True
                action = None

        if decision.thesis_id is not None:
            valid = action is not None and action.thesis_id == decision.thesis_id
            if not valid:
                try:
                    active = get_active_theses(ticker=decision.ticker)
                    valid = any(t.get("id") == decision.thesis_id for t in active)
                except Exception as e:
                    logger.warning("Could not verify thesis_id %s for %s: %s",
                                   decision.thesis_id, decision.ticker, e)
                    valid = False
            if not valid:
                logger.warning(
                    "%s: thesis_id=%s does not match the playbook action or any "
                    "active thesis for this ticker — dropping",
                    decision.ticker, decision.thesis_id,
                )
                record_event(
                    session_id=session_id, stage_name="trading",
                    event_type="id_validation",
                    payload={"ticker": decision.ticker, "field": "thesis_id",
                             "value": decision.thesis_id},
                )
                decision.thesis_id = None

    kept = []
    for inv in response.thesis_invalidations:
        if inv.thesis_id in known_thesis_ids:
            kept.append(inv)
            continue
        logger.warning(
            "Dropping thesis invalidation for id=%s — not visible in today's "
            "playbook (executor never saw it)", inv.thesis_id,
        )
        record_event(
            session_id=session_id, stage_name="trading",
            event_type="id_validation",
            payload={"field": "thesis_invalidation", "value": inv.thesis_id},
        )
    response.thesis_invalidations = kept
```

Call it in `run_trading_session` right after the `response is None` check (Step 4→5 boundary):

```python
    _validate_llm_ids(response, executor_input, session_id=session_id)
```

Note: `run_trading_session` builds `executor_input` via `_build_executor_context(...)` — it's already in scope.

- [ ] **Step 4: Run the trader suite.** The existing `test_thesis_invalidations_processed` will now fail unless its invalidation id is playbook-visible — update it to route through `build_executor_input` with a matching `PlaybookAction`, and note in the commit that blind invalidations are now dropped by design.

- [ ] **Step 5: Commit**

```bash
git add v2/trader.py tests/v2/test_trader.py
git commit -m "Validate LLM-authored playbook_action_id/thesis_id before DB writes (audit 1.4)"
```

---

### Task 8: Fail closed when the post-fill account refresh fails (audit 1.5 / C.5 gap b)

**Files:**
- Modify: `v2/trader.py` (`_refresh_buying_power` retry; halt in `_execute_decisions`)
- Test: `tests/v2/test_trader.py` (new class `TestPostFillRefreshFailClosed`)

**Interfaces:**
- Consumes: `_refresh_buying_power(...) -> (buying_power, portfolio_value, refreshed_info | None)`; `_execute_decisions` already breaks the loop on breaker breach using `refreshed_info`.
- Produces: refresh now retries `get_account_info()` once; `refreshed_info is None and not dry_run` halts the loop (no further submits) with a `risk_block` event, `reason_code="account_refresh_failed"`.

- [ ] **Step 1: Write the failing test**

```python
class TestPostFillRefreshFailClosed:
    def test_refresh_failure_after_fill_halts_loop(self, mock_db, mock_cursor):
        """C.5(b): the per-fill daily-loss re-check was silently skipped when
        the post-fill refresh failed — precisely the flaky-API moment a
        breaker matters. Now: retry once, then halt new submits (fail closed).
        """
        d1 = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        d2 = _make_decision(ticker="MSFT", action="buy", playbook_action_id=8)
        # First call feeds Step 2's snapshot; every later call (the post-fill
        # refresh + its retry) fails.
        acct = MagicMock(side_effect=[_DEFAULT_ACCOUNT] + [RuntimeError("alpaca 502")] * 4)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[d1, d2], overrides={
                "get_account_info": acct,
            })
            result = run_trading_session(dry_run=False)
        assert mocks["execute_market_order"].call_count == 1
        assert result.trades_executed == 1
        assert any("account refresh failed" in e for e in result.errors)

    def test_dry_run_refresh_estimate_still_continues(self, mock_db, mock_cursor):
        d1 = _make_decision(ticker="AAPL", action="buy", playbook_action_id=7)
        d2 = _make_decision(ticker="MSFT", action="buy", playbook_action_id=8)
        with ExitStack() as stack:
            mocks = _happy_path(stack, decisions=[d1, d2])
            run_trading_session(dry_run=True)
        assert mocks["execute_market_order"].call_count == 2
```

(Check how `_snapshot_account`/`take_account_snapshot` consume `get_account_info` in `_happy_path` — `take_account_snapshot` is separately mocked, so only `_snapshot_account` and the refresh hit `get_account_info`; adjust the side_effect list length accordingly. Check the `TradingSessionResult` field name for errors before asserting.)

- [ ] **Step 2: Run — first must FAIL** (both orders currently submit).

- [ ] **Step 3: Implement.** In `_refresh_buying_power`, replace the single try with a two-attempt loop:

```python
    if not dry_run:
        # C.5(b): one retry before giving up — a single transient fetch error
        # shouldn't halt the session, but flying blind past it must not
        # silently skip the daily-loss re-check either. The caller treats a
        # (None, not dry_run) result as a mid-session halt.
        for attempt in (1, 2):
            try:
                refreshed = get_account_info()
                return refreshed["buying_power"], refreshed["portfolio_value"], refreshed
            except Exception as e:
                logger.warning(
                    "Could not refresh buying power (attempt %d/2): %s", attempt, e,
                )
        if decision.action == "buy":
            buying_power -= trade_value
        elif decision.action == "sell":
            buying_power += trade_value
        return buying_power, portfolio_value, None
```

In `_execute_decisions`, extend the post-refresh block:

```python
        if refreshed_info is not None:
            loss_breach = check_daily_loss_limit(...)  # unchanged
            ...
        elif not dry_run:
            # C.5(b): fail closed — without live account state the daily-loss
            # breaker cannot run; stop submitting rather than fly blind.
            msg = ("Mid-session halt: post-fill account refresh failed after "
                   "retry — cannot verify daily-loss breaker (fail closed)")
            errors.append(msg)
            logger.error(msg)
            record_event(
                session_id=session_id, stage_name="trading",
                event_type="risk_block",
                payload={"reason_code": "account_refresh_failed",
                         "reason_text": msg,
                         "trades_executed": totals.trades_executed},
            )
            break
```

- [ ] **Step 4: Run the trader suite.** Any existing test that exercises refresh-failure-then-continue is asserting the old fail-open behavior — update it (it likely becomes the halt test).

- [ ] **Step 5: Commit**

```bash
git add v2/trader.py tests/v2/test_trader.py
git commit -m "Fail closed when post-fill account refresh fails (audit 1.5)"
```

---

### Task 9: `--force` replace preserves executed playbook actions (audit 1.6 / A.5)

**Files:**
- Modify: `v2/database/trading_db.py:754-759` (`replace_playbook_actions_atomic`)
- Test: `tests/v2/test_trading_db.py` (or wherever `replace_playbook_actions_atomic` is currently tested — `grep -rn replace_playbook_actions_atomic tests/`)

**Interfaces:**
- Produces: same signature `replace_playbook_actions_atomic(...) -> (playbook_id, action_count)`; only `pending`/NULL-status actions are cleared and unlinked. Executed/failed/skipped/expired rows and their `decisions.playbook_action_id` links survive a strategist re-run.

- [ ] **Step 1: Write the failing test** (follow the file's existing mock_cursor SQL-assertion pattern — read a neighboring test first):

```python
def test_replace_playbook_actions_preserves_non_pending(mock_db, mock_cursor):
    """A.5: a --force retry re-runs the strategist, whose write_playbook
    previously DELETEd *all* actions for the date — including rows already
    executed in the failed run — and nulled their decision links, silently
    destroying playbook_action_history and carry-forward context."""
    mock_cursor.fetchone.return_value = {"id": 11}
    replace_playbook_actions_atomic(
        date(2026, 7, 15), "outlook", [], "watch", "risk",
        [{"ticker": "AAPL", "action": "buy"}],
    )
    executed_sql = [c.args[0] for c in mock_cursor.execute.call_args_list]
    unlink = next(s for s in executed_sql if "SET playbook_action_id = NULL" in s)
    delete = next(s for s in executed_sql if s.strip().startswith("DELETE FROM playbook_actions"))
    for stmt in (unlink, delete):
        assert "status = 'pending'" in stmt and "status IS NULL" in stmt
```

- [ ] **Step 2: Run — must FAIL.**

Run: `docker compose exec -T trading python3 -m pytest tests/ -k replace_playbook_actions -v`

- [ ] **Step 3: Implement** — in `replace_playbook_actions_atomic`, replace the unlink+delete pair:

```python
        # A.5: only clear actions still awaiting execution. A --force retry
        # after a partial session re-runs the strategist; deleting executed/
        # failed/skipped rows here destroyed playbook_action_history and
        # severed decisions.playbook_action_id for trades that really filled.
        cur.execute(
            "UPDATE decisions SET playbook_action_id = NULL "
            "WHERE playbook_action_id IN ("
            "  SELECT id FROM playbook_actions "
            "  WHERE playbook_id = %s AND (status = 'pending' OR status IS NULL))",
            (playbook_id,),
        )
        cur.execute(
            "DELETE FROM playbook_actions "
            "WHERE playbook_id = %s AND (status = 'pending' OR status IS NULL)",
            (playbook_id,),
        )
```

- [ ] **Step 4: Run the DB + tools suites** (tool_write_playbook goes through this function):

Run: `docker compose exec -T trading python3 -m pytest tests/v2/test_trading_db.py tests/v2/test_tools.py -q`

- [ ] **Step 5: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_trading_db.py
git commit -m "Preserve executed playbook actions across --force strategist re-runs (audit 1.6)"
```

---

### Task 10: Full verification + PR

- [ ] **Step 1: Full suite in docker**

Run: `docker compose exec -T trading python3 -m pytest tests/ -q`
Expected: ~1,965+ tests pass, 0 failures.

- [ ] **Step 2: Lint**

Run: `task lint` (host ruff). Fix anything it flags in touched files.

- [ ] **Step 3: Update the audit report status lines** — in `docs/audits/2026-07-15-fresh-eyes-audit.md`, annotate findings 0.1, 1.1–1.6, 3.1, 3.3 with `**Fixed <date> (<commit/PR>)**` the same way 0.1/0.3 already carry resolution notes. Commit as `Mark same-day + Tier 1 audit findings fixed`.

- [ ] **Step 4: Push and open PR** against `main` from `fresh-eyes-audit-2026-07` (the branch also carries the audit report itself). PR body: summary table mapping finding # → commit, note that Tier 2 (learning gradient) and the 0.2 economics decision are deliberately NOT in this PR, end with the Claude Code attribution line.

- [ ] **Step 5: Update memory** — refresh `project_fresh_eyes_audit_2026_07.md`: Tier 1 + same-day items fixed (PR #), migrations applied to both DBs, backups + HALT mechanism now exist; remaining: 0.2 decision, Tier 2, Tier 3.2/3.4/3.5, Tier 4.
