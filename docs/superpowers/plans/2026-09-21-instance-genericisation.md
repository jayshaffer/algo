# Instance Genericisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded prod/paper pipeline duality with N named instances, each defined by one env file under `instances/`, addressed explicitly via compose project names.

**Architecture:** One generic `docker-compose.yml` parameterized by interpolation vars; every invocation runs `docker compose -p pinchy-<name> --env-file instances/<name>.env`. The Taskfile collapses to one instance-scoped target set (`INSTANCE=` required, no default). The only Python change is an `ALGO_DASHBOARD_PUBLISH` opt-in gate in `v2/session.py`. Host scripts (`cron-wrap.sh`, `run-docker.sh`) gain instance awareness including per-instance HALT sentinels.

**Tech Stack:** Docker Compose (project names + env-file interpolation), go-task (Taskfile v3), bash, Python 3.12 / pytest (in-container).

**Spec:** `docs/superpowers/specs/2026-09-21-instance-genericisation-design.md`

## Global Constraints

- No default instance anywhere: every stack-touching command must fail loudly when `INSTANCE` is missing or names a nonexistent file.
- Compose interpolation always uses the required form `${VAR:?}` — never a silent-empty fallback.
- Compose project name is exactly `pinchy-<instance>`; service names stay `db`, `trading`, `dashboard`.
- `ALGO_DASHBOARD_PUBLISH` defaults to **false** (opt-in publish); CLI `--skip-dashboard` and `--dry-run` promotion always win over it.
- `ALPACA_PAPER` / `ALPACA_BASE_URL` cross-check in `v2/executor.py` is untouched.
- HALT semantics: root `HALT` = host trades nothing; `instances/<name>.HALT` = that instance trades nothing; `--ignore-halt` remains for protect-the-data jobs only. Halted jobs still ping the heartbeat and exit 0.
- Backup naming: `backups/<instance>-<timestamp>.dump`, retention newest 14 per instance.
- pytest runs **in the container**, not on the host (host python is 3.10): `docker compose -p pinchy-<x> --env-file instances/<x>.env exec -T trading python -m pytest …`. During implementation (before any instance exists), the old-style stack may still be up; `docker compose exec -T trading python -m pytest tests/ -q` works until Task 3 lands. From Task 3 on, use a scratch instance (Task 9 creates real ones).
- Commit after every task; `ruff check .` runs as a pre-commit hook and must pass.

## File Structure

- `v2/session.py` — add `_dashboard_publish_enabled()`, thread a skip reason into the stage-5 wrapper (Task 1).
- `tests/v2/test_session.py` — new `TestDashboardPublishGate` class; autouse fixture sets the flag so existing tests keep old behavior (Task 1).
- `instances/example.env` — new committed canonical instance-config reference; `.env.example` deleted; `.gitignore` rules (Task 2).
- `docker-compose.yml` — genericised; `docker-compose.paper.yml` deleted (Task 3).
- `Taskfile.yml` — rewritten: one instance-scoped target set, `paper:*` deleted, `db:restore` added (Task 4).
- `run-docker.sh` — instance as first argument, project-scoped teardown (Task 5).
- `cron-wrap.sh` — `--instance` flag + per-instance HALT check (Task 6).
- `crontab` — instance-scoped lines, `<instance>-<job>` labels (Task 7).
- `README.md`, `CLAUDE.md`, `docs/runbook-recovery.md`, `.env.host.example` — docs sweep (Task 8).
- Cutover: operator-run checklist, no repo files beyond `HALT` removal (Task 9).

---

### Task 1: `ALGO_DASHBOARD_PUBLISH` gate in `v2/session.py`

**Files:**
- Modify: `v2/session.py` (helper near `_trading_halted` at ~line 166; `run_session` dry-run promotion block at ~line 586; `_run_dashboard_stage_wrapper` at ~line 485)
- Test: `tests/v2/test_session.py`

**Interfaces:**
- Consumes: existing `run_session(..., skip_dashboard: bool = False)`, `_run_dashboard_stage_wrapper(result, session_id, completed_stages, skip)`.
- Produces: `_dashboard_publish_enabled() -> bool`; `_run_dashboard_stage_wrapper` gains a 5th positional parameter `skip_reason: str | None = None`. `run_session` behavior change: stage 5 is skipped unless `ALGO_DASHBOARD_PUBLISH` is truthy (`1`/`true`/`yes`, case-insensitive, stripped) — read at call time, not import.

- [ ] **Step 1: Update the module autouse fixture so existing tests keep publish-by-default behavior**

In `tests/v2/test_session.py`, the module-level autouse fixture `_bypass_session_idempotency` (top of file, ~line 14) gains `monkeypatch` and sets the flag:

```python
@pytest.fixture(autouse=True)
def _bypass_session_idempotency(monkeypatch):
    """All tests in this module should exercise run_session as if no prior session exists today.

    ... (keep existing docstring text) ...

    Also opts this module into dashboard publishing: ALGO_DASHBOARD_PUBLISH
    defaults to false (spec 2026-09-21 instance genericisation), and the
    pre-existing stage-5 tests were written when publish was the default.
    Tests exercising the gate itself delete/override the var in their bodies.
    """
    monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "true")
    with patch("v2.session.get_session_for_date", return_value=None), \
         patch("v2.session.get_playbook", return_value={"id": 1}):
        yield
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/v2/test_session.py` (module already imports `patch`, `pytest`, `run_session`):

```python
class TestDashboardPublishGate:
    """Stage 5 is opt-in per instance: ALGO_DASHBOARD_PUBLISH must be truthy.

    There is exactly one public Cloudflare Pages site; a freshly configured
    instance must not publish over it by accident. Spec:
    docs/superpowers/specs/2026-09-21-instance-genericisation-design.md §3.
    """

    def _run(self):
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_supervisor"), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage") as mock_dashboard:
            result = run_session(dry_run=False)
        return result, mock_dashboard

    def test_unset_skips_publish(self, monkeypatch):
        monkeypatch.delenv("ALGO_DASHBOARD_PUBLISH", raising=False)
        result, mock_dashboard = self._run()
        mock_dashboard.assert_not_called()
        assert result.skipped_dashboard is True
        assert result.has_errors is False

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "  ", "banana"])
    def test_non_truthy_skips_publish(self, monkeypatch, value):
        monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", value)
        result, mock_dashboard = self._run()
        mock_dashboard.assert_not_called()
        assert result.skipped_dashboard is True

    @pytest.mark.parametrize("value", ["true", "1", "yes", " TRUE "])
    def test_truthy_publishes(self, monkeypatch, value):
        monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", value)
        result, mock_dashboard = self._run()
        mock_dashboard.assert_called_once()
        assert result.skipped_dashboard is False

    def test_cli_skip_wins_over_enabled(self, monkeypatch):
        monkeypatch.setenv("ALGO_DASHBOARD_PUBLISH", "true")
        with patch("v2.session.run_backfill"), \
             patch("v2.session.compute_signal_attribution", return_value=[]), \
             patch("v2.session.build_attribution_constraints", return_value=""), \
             patch("v2.session.run_supervisor"), \
             patch("v2.session.run_pipeline"), \
             patch("v2.session.run_strategist_loop"), \
             patch("v2.session.run_trading_session"), \
             patch("v2.session.run_strategy_reflection"), \
             patch("v2.session.run_dashboard_stage") as mock_dashboard:
            result = run_session(dry_run=False, skip_dashboard=True)
        mock_dashboard.assert_not_called()
        assert result.skipped_dashboard is True
```

Note: the patch-target names above are verified against the module's existing stage tests (`v2.session.run_supervisor` matches `TestSupervisorStage` at ~line 1524; the rest match `TestStage5Dashboard.test_skip_dashboard_flag` at ~line 380). If you add or reorder patches, mirror what the existing tests patch, exactly — un-mocked stages spend real API tokens (2026-05-28 incident, $626).

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `docker compose exec -T trading python -m pytest tests/v2/test_session.py::TestDashboardPublishGate -v`
Expected: the `test_unset_skips_publish` and `test_non_truthy_skips_publish` cases FAIL (dashboard currently publishes whenever not explicitly skipped); truthy cases pass vacuously.

- [ ] **Step 4: Implement the gate**

In `v2/session.py`, add below `_trading_halted()` (~line 178):

```python
def _dashboard_publish_enabled() -> bool:
    """Instance-level opt-in for the public dashboard publish (stage 5).

    There is exactly one public Cloudflare Pages site, but N instances run
    this code — publish must be something an instance's env file asks for,
    not a default it forgets to turn off. Read at call time (not import) so
    a flip takes effect on the next session without a container restart,
    matching ALGO_TRADING_HALTED. Strict-affirmative like _trading_halted,
    but in the opposite safety direction: a typo'd value skipping the
    publish is a better failure than one publishing publicly.
    """
    return os.environ.get("ALGO_DASHBOARD_PUBLISH", "").strip().lower() in (
        "1", "true", "yes",
    )
```

In `run_session()`, replace the dry-run promotion block (~line 586) tail so the env gate resolves after CLI flags, and record why:

```python
    if dry_run:
        skip_ideation = True
        skip_strategy = True
        skip_dashboard = True

    if skip_dashboard:
        skip_dashboard_reason = "--skip-dashboard/--dry-run"
    elif not _dashboard_publish_enabled():
        skip_dashboard = True
        skip_dashboard_reason = "ALGO_DASHBOARD_PUBLISH not enabled for this instance"
    else:
        skip_dashboard_reason = None
```

Change the wrapper call (~line 621) to pass the reason:

```python
        _run_dashboard_stage_wrapper(
            result, session_id, completed_stages, skip_dashboard, skip_dashboard_reason,
        )
```

And the wrapper itself (~line 485):

```python
def _run_dashboard_stage_wrapper(
    result: SessionResult, session_id: int | None, completed_stages: set,
    skip: bool, skip_reason: str | None = None,
) -> None:
    if skip or "dashboard" in completed_stages:
        logger.info(
            "[Stage 5] Dashboard publish — SKIPPED (%s)",
            "completed in prior run" if "dashboard" in completed_stages
            else (skip_reason or "unspecified"),
        )
        result.skipped_dashboard = True
        return
```

(The rest of the wrapper body is unchanged.)

- [ ] **Step 5: Run the module and full suite**

Run: `docker compose exec -T trading python -m pytest tests/v2/test_session.py -q` then `docker compose exec -T trading python -m pytest tests/ -q`
Expected: all pass. If any pre-existing test fails on the flipped default, it is missing the autouse fixture's env (check it isn't overriding `monkeypatch` scope) — fix by setting `ALGO_DASHBOARD_PUBLISH=true` in that test, not by weakening the gate.

- [ ] **Step 6: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "Gate stage-5 dashboard publish behind ALGO_DASHBOARD_PUBLISH (opt-in)"
```

---

### Task 2: `instances/example.env` + gitignore rules

**Files:**
- Create: `instances/example.env`
- Modify: `.gitignore`
- Delete: `.env.example`

**Interfaces:**
- Produces: the canonical instance-config reference. Later tasks rely on these exact var names: `INSTANCE`, `DB_HOST_PORT`, `DASHBOARD_HOST_PORT`, `LOGS_DIR`, `ALGO_DASHBOARD_PUBLISH`, plus everything `.env` held.

- [ ] **Step 1: Create `instances/example.env`**

Content (derived from `.env.example`, plus the instance/deployment block):

```bash
# Instance config — one file per instance, e.g. instances/paper.env.
# Copy this file: cp instances/example.env instances/<name>.env
# Everything here reaches the containers via compose env_file; the four
# deployment vars in the first block are ALSO read by compose interpolation
# (docker compose -p pinchy-<name> --env-file instances/<name>.env).
# The instances/ directory is gitignored (secrets) except this example and
# *.HALT sentinels.

# --- Instance identity & deployment shape (compose interpolation) ---
# INSTANCE must match this file's basename: instances/<INSTANCE>.env.
# The compose file resolves env_file as instances/${INSTANCE:?}.env.
INSTANCE=example
# Host ports (127.0.0.1-bound). Must be unique per instance on this host —
# a collision fails loudly at `task up` with a bind error.
DB_HOST_PORT=5432
DASHBOARD_HOST_PORT=3000
# Host directory bind-mounted at /app/logs. Convention: ./logs/<INSTANCE>
LOGS_DIR=./logs/example

# --- Public dashboard (stage 5) ---
# Opt-in: exactly one instance should own the public Cloudflare Pages site.
# Unset/false = stage 5 skipped. Read at session start (no restart needed).
ALGO_DASHBOARD_PUBLISH=false
# Only the publishing instance needs these:
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_PAGES_PROJECT=your-project-name
DASHBOARD_URL=https://your-project.pages.dev

# --- Alpaca API (which account this instance trades) ---
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
# Required: true for paper, false for live. Cross-checked against
# ALPACA_BASE_URL at module load — mismatched values raise immediately.
ALPACA_PAPER=true

# --- Finnhub API (market data - free tier) ---
FINNHUB_API_KEY=your_finnhub_api_key

# --- Claude API ---
ANTHROPIC_API_KEY=your_secret_key

# --- Database (in-stack Postgres; no need to vary per instance) ---
POSTGRES_USER=algo
POSTGRES_PASSWORD=algo
POSTGRES_DB=trading
DATABASE_URL=postgresql://algo:algo@db:5432/trading

# --- Operational knobs (optional) ---
# Kill switch: set to 1/true/yes to make every session a no-op (logs + exits
# 0). Read at session start. NOTE: on a long-running stack, an env-file edit
# needs a container recreate (`task up INSTANCE=<name>`) to reach the
# container — the no-restart halt path is the sentinel file
# instances/<name>.HALT checked by cron-wrap.sh. See docs/runbook-recovery.md.
# ALGO_TRADING_HALTED=

# Daily-loss circuit breaker, % vs previous close (default 3.0; <= 0 disables).
# ALGO_DAILY_LOSS_LIMIT_PCT=3.0

# Per-agentic-loop cost ceiling in USD (default 30; <= 0 disables).
# ALGO_LOOP_COST_CEILING_USD=30

# Executor model / token cap overrides (read at import — restart to apply).
# ALGO_EXECUTOR_MODEL=claude-haiku-4-5-20251001
# ALGO_EXECUTOR_MAX_TOKENS=8192

# NOTE: ALGO_ALERT_WEBHOOK_URL, ALGO_HEARTBEAT_URL* and ALGO_BACKUP_COPY_DIR
# are read by host-side scripts (cron-wrap.sh, task db:backup) — env_file only
# feeds containers, so setting them here has NO effect. They live in .env.host
# (instance-agnostic; per-job heartbeat overrides key off cron labels).
```

- [ ] **Step 2: Update `.gitignore`**

Replace the environment block at the top:

```gitignore
# Environment
.env
.env.paper
.env.host
```

with:

```gitignore
# Environment
# (.env / .env.paper retired 2026-09 — instance genericisation; see
# docs/superpowers/specs/2026-09-21-instance-genericisation-design.md)
.env
.env.paper
.env.host

# Instance configs hold secrets. The committed example is the reference;
# *.HALT sentinels stay tracked so a halt/resume is visible in git.
instances/*
!instances/example.env
!instances/*.HALT
```

(Keep `.env`/`.env.paper` lines until Task 9 deletes the files, so they never get accidentally staged mid-migration. Keep the existing `logs/` and `logs_paper/` entries — `logs/` already covers the new `logs/<instance>/` convention.)

- [ ] **Step 3: Delete `.env.example`**

```bash
git rm .env.example
```

`instances/example.env` supersedes it; a stale second copy of the same reference is how docs drift.

- [ ] **Step 4: Verify gitignore behavior**

Run: `touch instances/smoke.env instances/smoke.HALT && git status --porcelain instances/ && rm instances/smoke.env instances/smoke.HALT`
Expected: `instances/example.env` (staged) and `instances/smoke.HALT` show; `instances/smoke.env` does NOT.

- [ ] **Step 5: Commit**

```bash
git add .gitignore instances/example.env
git commit -m "Add instances/example.env as canonical instance config; retire .env.example"
```

---

### Task 3: Genericise `docker-compose.yml`, delete the paper overlay

**Files:**
- Modify: `docker-compose.yml`
- Delete: `docker-compose.paper.yml`

**Interfaces:**
- Consumes: `INSTANCE`, `DB_HOST_PORT`, `DASHBOARD_HOST_PORT`, `LOGS_DIR` from Task 2's env-file contract.
- Produces: the single stack file every later task invokes as `docker compose -p pinchy-<name> --env-file instances/<name>.env …`. Volume is named `postgres_data` (project-prefixed by compose to `pinchy-<name>_postgres_data`).

- [ ] **Step 1: Rewrite `docker-compose.yml`**

Full new content (comments from the current file preserved where the mounts they explain survive):

```yaml
# Generic per-instance stack. Never `docker compose up` this file bare —
# always through the Taskfile (or run-docker.sh), which supply
#   -p pinchy-<instance> --env-file instances/<instance>.env
# The --env-file feeds the ${...} interpolations below (it does NOT reach the
# containers; the env_file: entries do that). Every interpolation is the
# required form ${VAR:?} so a bare invocation fails loudly instead of
# binding empty ports or mounting empty paths.
services:
  db:
    image: postgres:16
    env_file:
      - instances/${INSTANCE:?}.env
    ports:
      - "127.0.0.1:${DB_HOST_PORT:?}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U algo -d trading"]
      interval: 5s
      timeout: 5s
      retries: 5

  trading:
    build: .
    depends_on:
      db:
        condition: service_healthy
    env_file:
      - instances/${INSTANCE:?}.env
    volumes:
      - ./trading:/app/trading:ro
      - ./v2:/app/v2:ro
      - ./dashboard:/app/dashboard:ro
      - ./public_dashboard:/app/public_dashboard:ro
      - ${LOGS_DIR:?}:/app/logs
      - ./tests:/app/tests:ro
      - ./pytest.ini:/app/pytest.ini:ro
      # Whole db/ tree, not just init/: tests/test_schema_mirror.py compares
      # db/init against db/migrations, and mounting only one half made the
      # check pass vacuously in the container while failing in CI.
      - ./db:/app/db:ro
    command: ["sleep", "infinity"]

  dashboard:
    build: ./dashboard
    ports:
      - "127.0.0.1:${DASHBOARD_HOST_PORT:?}:3000"
    depends_on:
      db:
        condition: service_healthy
    env_file:
      - instances/${INSTANCE:?}.env
    volumes:
      # queries.py imports v2.market_calendar (single source of truth for the
      # NYSE calendar); the dashboard image builds only ./dashboard, so mount v2.
      - ./v2:/app/v2:ro

volumes:
  postgres_data:
```

(Differences from today, deliberate: `env_file` paths are interpolated; ports and logs mount are interpolated; the tests/pytest.ini/db mounts — present today on prod-`trading` but missing on `trading-paper` — are now uniform, which `task test INSTANCE=<x>` requires.)

- [ ] **Step 2: Delete the overlay**

```bash
git rm docker-compose.paper.yml
```

- [ ] **Step 3: Verify config resolution**

```bash
mkdir -p logs/smoke
cp instances/example.env instances/smoke.env
sed -i 's/^INSTANCE=example/INSTANCE=smoke/; s/^DB_HOST_PORT=5432/DB_HOST_PORT=5599/; s/^DASHBOARD_HOST_PORT=3000/DASHBOARD_HOST_PORT=3599/; s|^LOGS_DIR=./logs/example|LOGS_DIR=./logs/smoke|' instances/smoke.env
docker compose -p pinchy-smoke --env-file instances/smoke.env config >/dev/null && echo OK
docker compose config 2>&1 | head -2   # bare invocation must FAIL on ${INSTANCE:?}
rm instances/smoke.env && rmdir logs/smoke
```

Expected: first `config` prints OK; bare `config` errors mentioning `INSTANCE`.

Note: the old-style stack (`docker compose exec trading …`) is unusable from this commit on. Test runs for Tasks 4–8 use a scratch instance: keep a `instances/dev.env` locally (copy of example with unique ports, real-ish `DATABASE_URL` unchanged — tests mock the DB) and run `task test INSTANCE=dev` once Task 4 lands.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "Genericise compose stack: one file, per-instance interpolation; drop paper overlay"
```

---

### Task 4: Rewrite `Taskfile.yml`

**Files:**
- Modify: `Taskfile.yml`

**Interfaces:**
- Consumes: compose contract from Task 3; env-file contract from Task 2.
- Produces: the target set every later task and the crontab call: `up`, `down`, `build`, `logs`, `stop:session`, `session`, `session:dry-run`, `trade`, `trade:dry-run`, `ideation`, `pipeline`, `learn`, `supervise`, `supervise:dry-run`, `backfill`, `backfill:decision`, `dashboard:publish`, `test`, `test:coverage`, `db:migrate`, `db:backup`, `db:restore` — all requiring `INSTANCE`; `lint`, `install-hooks`, `audit:*` unchanged and instance-agnostic.

- [ ] **Step 1: Rewrite the file**

Full new content. Notes baked in: `COMPOSE` is a global var (CLI-passed `INSTANCE` is global in go-task, so deps like `up` see it without explicit var plumbing); `requires` + `preconditions` are repeated verbatim on every stack-touching task (no YAML anchors — go-task's schema rejects unknown root keys, and explicit beats clever here); the old `docker:up` grep-based `status:` check is dropped (`up -d` is idempotent, and the old check was wrong the moment two projects share a host).

```yaml
version: "3"

# Host-side config (ALGO_BACKUP_COPY_DIR, alerting, heartbeat URLs). Optional —
# Task tolerates the file being absent. cron-wrap.sh sources the same file, so
# a scheduled `task db:backup` and a hand-run one see identical settings; that
# divergence is how the off-host backup copy stayed unconfigured. Container
# config lives in instances/<name>.env via compose env_file.
dotenv: [".env.host"]

# Every stack-touching target requires INSTANCE explicitly — there is no
# default instance. Real money: "forgot the flag" must fail, not silently
# target an account. Usage: task session INSTANCE=paper
vars:
  COMPOSE: docker compose -p pinchy-{{.INSTANCE}} --env-file instances/{{.INSTANCE}}.env

tasks:
  # ---------------------------------------------------------------------------
  # Infrastructure
  # ---------------------------------------------------------------------------
  up:
    desc: Start the instance's services (task up INSTANCE=<name>)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    cmds:
      - "{{.COMPOSE}} up -d"

  down:
    desc: Stop the instance's services
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    cmds:
      - "{{.COMPOSE}} down"

  build:
    desc: Rebuild the instance's images
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    cmds:
      - "{{.COMPOSE}} build"

  logs:
    desc: Follow container logs (pass service name as arg)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    cmds:
      - "{{.COMPOSE}} logs -f {{.CLI_ARGS}}"

  stop:session:
    desc: Stop session services (trading); leave db + dashboard running
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    cmds:
      - "{{.COMPOSE}} stop trading"

  # ---------------------------------------------------------------------------
  # Trading workflows
  # ---------------------------------------------------------------------------
  session:
    desc: Run full consolidated daily session
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      # P1.11: session is idempotent per market date; pass --force via
      # {{.CLI_ARGS}} only for deliberate retries.
      - "{{.COMPOSE}} exec trading python -m v2.session {{.CLI_ARGS}}"

  session:dry-run:
    desc: Dry-run the full daily session
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.session --dry-run {{.CLI_ARGS}}"

  trade:
    desc: Run the trader
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.trader {{.CLI_ARGS}}"

  trade:dry-run:
    desc: Dry-run the trader
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.trader --dry-run {{.CLI_ARGS}}"

  ideation:
    desc: Run Claude strategist ideation
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.ideation_claude {{.CLI_ARGS}}"

  pipeline:
    desc: Run news processing pipeline
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.pipeline {{.CLI_ARGS}}"

  learn:
    desc: Run learning loop
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.learn {{.CLI_ARGS}}"

  supervise:
    desc: Run the strategy supervisor (observer-only critic, persists one memo)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.supervisor {{.CLI_ARGS}}"

  supervise:dry-run:
    desc: Run the supervisor loop and print the memo without persisting
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.supervisor --dry-run {{.CLI_ARGS}}"

  backfill:
    desc: Backfill decision outcomes
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.backfill {{.CLI_ARGS}}"

  backfill:decision:
    desc: Re-run 7d+30d backfill for a single decision (pass --decision-id <ID> via CLI_ARGS). Audit auto-fix path for MISSING_OUTCOMES findings.
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -m v2.backfill {{.CLI_ARGS}}"

  dashboard:publish:
    desc: Publish the public dashboard to Cloudflare Pages (stage 5 only; bypasses the ALGO_DASHBOARD_PUBLISH gate — explicit invocation is the opt-in)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec trading python -c \"from v2.dashboard_publish import run_dashboard_stage; r = run_dashboard_stage(); print(r); raise SystemExit(0 if r.published or r.skipped else 1)\""

  # ---------------------------------------------------------------------------
  # Dev
  # ---------------------------------------------------------------------------
  test:
    desc: Run tests (in the instance's trading container; external deps are mocked)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec -T trading python -m pytest tests/ {{.CLI_ARGS}}"

  test:coverage:
    desc: Run tests with coverage (in the instance's trading container)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec -T trading python -m pytest tests/ --cov=trading --cov=dashboard --cov=v2 {{.CLI_ARGS}}"

  lint:
    desc: Run ruff lint (host, no docker)
    cmds:
      - ruff check {{.CLI_ARGS}} .

  install-hooks:
    desc: Point git at the tracked .githooks directory
    cmds:
      - git config core.hooksPath .githooks
      - echo "Git hooks installed. Bypass with 'git commit --no-verify'."

  # ---------------------------------------------------------------------------
  # Database
  # ---------------------------------------------------------------------------
  db:migrate:
    desc: Apply pending db/migrations/*.sql to the instance db (tracked via schema_migrations)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - |
        psql_exec() { {{.COMPOSE}} exec -T db psql -U algo -d trading "$@"; }
        psql_exec -v ON_ERROR_STOP=1 -c "CREATE TABLE IF NOT EXISTS schema_migrations (filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW());" >/dev/null
        for f in $(ls db/migrations/*.sql | sort); do
          name=$(basename "$f")
          applied=$(psql_exec -tAc "SELECT 1 FROM schema_migrations WHERE filename = '$name'")
          if [ "$applied" = "1" ]; then
            echo "==> skipping $name (already applied)"
            continue
          fi
          echo "==> applying $name"
          { cat "$f"; printf "\nINSERT INTO schema_migrations (filename) VALUES ('%s');\n" "$name"; } \
            | psql_exec -v ON_ERROR_STOP=1 --single-transaction -f -
        done

  db:backup:
    desc: pg_dump the instance db to backups/<instance>-<ts>.dump (keeps newest 14; set ALGO_BACKUP_COPY_DIR for an off-WSL copy)
    requires: { vars: [INSTANCE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
    deps: [up]
    cmds:
      - |
        mkdir -p backups
        f="backups/{{.INSTANCE}}-$(date +%Y%m%d-%H%M%S).dump"
        {{.COMPOSE}} exec -T db pg_dump -U algo -d trading -Fc > "$f"
        echo "==> wrote $f ($(du -h "$f" | cut -f1))"
        ls -t backups/{{.INSTANCE}}-*.dump | tail -n +15 | xargs -r rm --
        if [ -n "${ALGO_BACKUP_COPY_DIR:-}" ]; then
          mkdir -p "$ALGO_BACKUP_COPY_DIR" && cp "$f" "$ALGO_BACKUP_COPY_DIR"/ && echo "==> copied to $ALGO_BACKUP_COPY_DIR"
        fi

  db:restore:
    desc: "pg_restore a dump into the instance db (task db:restore INSTANCE=x FILE=backups/foo.dump). DESTRUCTIVE: --clean drops existing objects first."
    requires: { vars: [INSTANCE, FILE] }
    preconditions:
      - sh: test -f instances/{{.INSTANCE}}.env
        msg: "No such instance: instances/{{.INSTANCE}}.env"
      - sh: test -f {{.FILE}}
        msg: "No such dump file: {{.FILE}}"
    deps: [up]
    cmds:
      - "{{.COMPOSE}} exec -T db pg_restore -U algo -d trading --clean --if-exists --no-owner < {{.FILE}}"
      - '{{.COMPOSE}} exec -T db psql -U algo -d trading -tAc "SELECT COUNT(*) FROM schema_migrations" | xargs echo "==> restored; schema_migrations rows:"'

  # ---------------------------------------------------------------------------
  # Audit (driven by docs/audit-playbook.md + Claude Code /loop)
  # ---------------------------------------------------------------------------
  audit:rehearse:
    desc: Run one Phase A discovery tick with no Jira writes; prints would-be payloads
    cmds:
      - claude "/audit-rehearse"

  audit:loop:start:
    desc: Start the long-lived audit /loop in a detached tmux session named 'audit-loop'
    cmds:
      - |
        if tmux has-session -t audit-loop 2>/dev/null; then
          echo "Session 'audit-loop' already exists. Attach with: task audit:loop:attach"
          exit 1
        fi
        tmux new-session -d -s audit-loop -c "$(pwd)" "claude '/loop 24h /audit-tick'"
        echo "Started. Attach with: task audit:loop:attach"

  audit:loop:attach:
    desc: Attach to the running audit /loop tmux session (detach with Ctrl-b d)
    interactive: true
    cmds:
      - tmux attach -t audit-loop

  audit:loop:stop:
    desc: Kill the audit /loop tmux session
    cmds:
      - tmux kill-session -t audit-loop && echo "Stopped." || echo "No session 'audit-loop' running."
```

- [ ] **Step 2: Smoke-test the guardrails**

```bash
task --list >/dev/null && echo LIST-OK
task session 2>&1 | head -2                 # expect: missing required variable INSTANCE
task session INSTANCE=nope 2>&1 | head -2   # expect: "No such instance: instances/nope.env"
```

- [ ] **Step 3: Smoke-test a real target against a scratch instance**

```bash
mkdir -p logs/dev
cp instances/example.env instances/dev.env
sed -i 's/^INSTANCE=example/INSTANCE=dev/; s/^DB_HOST_PORT=5432/DB_HOST_PORT=5598/; s/^DASHBOARD_HOST_PORT=3000/DASHBOARD_HOST_PORT=3598/; s|^LOGS_DIR=./logs/example|LOGS_DIR=./logs/dev|' instances/dev.env
task up INSTANCE=dev
task test INSTANCE=dev -- -q
docker volume ls | grep pinchy-dev    # expect pinchy-dev_postgres_data
task down INSTANCE=dev
```

Expected: full suite passes inside `pinchy-dev-trading-1`. Keep `instances/dev.env` for the remaining tasks' test runs (it is gitignored).

- [ ] **Step 4: Commit**

```bash
git add Taskfile.yml
git commit -m "Collapse Taskfile to instance-scoped targets; add db:restore; drop paper:* family"
```

---

### Task 5: `run-docker.sh` takes an instance

**Files:**
- Modify: `run-docker.sh`

**Interfaces:**
- Consumes: compose contract from Task 3; per-instance HALT convention (`instances/<name>.HALT`).
- Produces: `./run-docker.sh <instance> <service> <cmd...>` — used by the crontab (Task 7) and runbook (Task 8). Ephemeral semantics preserved: up → exec → project-scoped down on exit; HALT (global or instance) exits 0 before the EXIT trap is installed; failure alerting unchanged (suppressed under `ALGO_CRON_WRAPPED`).

- [ ] **Step 1: Rewrite the argument handling and compose invocations**

Keep the file's existing comments and structure; the changed regions in full:

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <instance> <service> <command> [args...]"
    echo "Example: $0 live trading python -m v2.session"
    exit 1
fi

INSTANCE="$1"
shift
ENV_FILE="$SCRIPT_DIR/instances/$INSTANCE.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "No such instance: instances/$INSTANCE.env" >&2
    exit 2
fi
# Project-scoped compose invocation: `down` below must only ever touch THIS
# instance's stack — a bare `docker compose down` would take whichever
# project matched the directory name, including another instance's.
COMPOSE=(docker compose -p "pinchy-$INSTANCE" --env-file "$ENV_FILE")

CMD_DESC="[$INSTANCE] $*"
FAILURE_LOG="$SCRIPT_DIR/logs/session_failures.log"

# C.5: operator kill switch, two levels. Global `touch HALT` stops every
# instance; `touch instances/<name>.HALT` stops just that one. Checked before
# the EXIT trap is installed, so a halt neither tears down containers nor
# fires the failure alert below — a deliberate halt is not a failure and
# exits 0. In-container twin: ALGO_TRADING_HALTED (v2/session.py).
# See docs/runbook-recovery.md "Halt / Resume".
if [ -f "$SCRIPT_DIR/HALT" ]; then
    echo "[$(date -Is)] HALT sentinel present — skipping: $CMD_DESC"
    exit 0
fi
if [ -f "$SCRIPT_DIR/instances/$INSTANCE.HALT" ]; then
    echo "[$(date -Is)] instances/$INSTANCE.HALT present — skipping: $CMD_DESC"
    exit 0
fi
```

The `notify_failure()` and `cleanup()` functions keep their bodies except every `docker compose` becomes `"${COMPOSE[@]}"`:

```bash
cleanup() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        notify_failure "$status"
    fi
    echo "Stopping containers..."
    "${COMPOSE[@]}" down || echo "[$(date -Is)] docker compose down failed" >&2
    exit "$status"
}

trap cleanup EXIT

echo "Starting containers..."
"${COMPOSE[@]}" up -d

echo "Waiting for services..."
sleep 5

echo "Running: $@"
"${COMPOSE[@]}" exec -T "$@"
```

- [ ] **Step 2: Smoke-test**

```bash
bash -n run-docker.sh && echo SYNTAX-OK
./run-docker.sh 2>&1 | head -2                       # usage, exit 1
./run-docker.sh nope trading true 2>&1 | head -1     # "No such instance", exit 2
touch instances/dev.HALT
./run-docker.sh dev trading true                     # halt line, exit 0, no containers started
rm instances/dev.HALT
./run-docker.sh dev trading python -c 'print("ok")'  # up → ok → down
docker compose -p pinchy-dev --env-file instances/dev.env ps --format '{{.Name}}' | wc -l  # expect 0
```

- [ ] **Step 3: Commit**

```bash
git add run-docker.sh
git commit -m "run-docker.sh: instance-scoped stacks and per-instance HALT"
```

---

### Task 6: `cron-wrap.sh --instance` + per-instance HALT

**Files:**
- Modify: `cron-wrap.sh`

**Interfaces:**
- Consumes: `instances/<name>.HALT` convention.
- Produces: `./cron-wrap.sh [--ignore-halt] [--instance <name>] <label> <command...>` — used by the crontab (Task 7). Halted jobs (either sentinel) still ping the heartbeat and exit 0.

- [ ] **Step 1: Replace the flag parsing and HALT check**

Current parsing (single `--ignore-halt` check, ~line 47) becomes an option loop; update the usage message too:

```bash
# --ignore-halt is for jobs that protect the system rather than trade with it
# (backups, health checks). HALT means "do not trade", not "do not protect the
# data" — a hiatus is precisely when an unnoticed backup gap would hurt most.
# --instance <name> additionally honors instances/<name>.HALT, the
# per-instance halt: "this instance trades nothing" without stopping the rest.
IGNORE_HALT=
INSTANCE=
while [ $# -gt 0 ]; do
    case "$1" in
        --ignore-halt) IGNORE_HALT=1; shift ;;
        --instance)
            INSTANCE="${2:-}"
            if [ -z "$INSTANCE" ]; then
                echo "--instance requires a name" >&2
                exit 2
            fi
            shift 2 ;;
        *) break ;;
    esac
done

if [ $# -lt 2 ]; then
    echo "Usage: $0 [--ignore-halt] [--instance <name>] <label> <command> [args...]" >&2
    echo "Example: $0 --instance paper paper-session task session INSTANCE=paper" >&2
    exit 2
fi
```

And the halt check (~line 111) becomes two-level. The existing comment block above it stays; the code:

```bash
if [ -z "$IGNORE_HALT" ]; then
    if [ -f "$SCRIPT_DIR/HALT" ]; then
        echo "[$(date -Is)] cron-wrap[$LABEL] HALT sentinel present — skipping: $CMD_DESC"
        ping_heartbeat
        exit 0
    fi
    if [ -n "$INSTANCE" ] && [ -f "$SCRIPT_DIR/instances/$INSTANCE.HALT" ]; then
        echo "[$(date -Is)] cron-wrap[$LABEL] instances/$INSTANCE.HALT present — skipping: $CMD_DESC"
        ping_heartbeat
        exit 0
    fi
fi
```

Everything else (heartbeat, `ALGO_CRON_WRAPPED`, failure alerting) is untouched.

- [ ] **Step 2: Smoke-test**

```bash
bash -n cron-wrap.sh && echo SYNTAX-OK
./cron-wrap.sh --instance 2>&1 | head -1            # "--instance requires a name", exit 2
./cron-wrap.sh smoke-test true && echo WRAP-OK      # runs, exit 0
touch instances/dev.HALT
./cron-wrap.sh --instance dev smoke-test false && echo HALTED-OK   # halt line, exit 0 despite `false`
./cron-wrap.sh --ignore-halt --instance dev smoke-test true && echo IGNORE-OK
rm instances/dev.HALT
```

(These append benign lines to `logs/session_failures.log` via `log()` only on failure paths; the halt path echoes to stdout only.)

- [ ] **Step 3: Commit**

```bash
git add cron-wrap.sh
git commit -m "cron-wrap.sh: --instance flag and per-instance HALT sentinel"
```

---

### Task 7: Rewrite `crontab`

**Files:**
- Modify: `crontab`

**Interfaces:**
- Consumes: `task session INSTANCE=…` (Task 4), `run-docker.sh <instance> …` (Task 5), `cron-wrap.sh --instance` (Task 6).
- Produces: label convention `<instance>-<job>`. **Label change:** `prod-backup` → `live-backup` — the healthchecks.io check and any `ALGO_HEARTBEAT_URL_PROD_BACKUP` var in `.env.host` must be renamed at cutover (Task 9).

- [ ] **Step 1: Rewrite the file**

```cron
# Pinchy Crontab (v2, instance-genericised)
# Install with: crontab /home/jay/dev/algo/crontab
# All times are MST (America/Denver, UTC-7)
#
# Every job runs through cron-wrap.sh [flags] <label> <command...>, which adds
# the HALT sentinel checks (global HALT file + per-instance
# instances/<name>.HALT via --instance), the dead-man's-switch ping, failure
# logging, and the alert webhook. Do not add a bare cron line here — a job
# outside the wrapper is a job whose failure nobody hears, which is exactly
# how the system sat dead from 2026-06-15 to 2026-08-13 (audit 0.3).
# Host-side config lives in .env.host. Labels are <instance>-<job>; per-job
# heartbeat overrides key off them (ALGO_HEARTBEAT_URL_<LABEL>).

PATH=/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin

# Paper session (12:30 PM MST / 2:30 PM ET, Mon-Fri).
# The live-account hiatus is expressed as instances/live.HALT, so this line
# needs no --ignore-halt: paper halts only via the global HALT file or its
# own instances/paper.HALT.
30 12 * * 1-5 cd /home/jay/dev/algo/ && ./cron-wrap.sh --instance paper paper-session task session INSTANCE=paper

# HIATUS since 2026-06 — the two live-instance jobs below are deliberately
# commented out AND instances/live.HALT is present; both must be reversed to
# resume. See instances/live.HALT and docs/runbook-recovery.md "Halt / Resume".
# This mirrors the installed crontab rather than drifting from it (audit 0.1).
#
# Live daily session (1 PM MST / 3 PM ET, Mon-Fri) — ephemeral stack:
# run-docker.sh brings the instance up, runs the session, tears it down.
# 0 13 * * 1-5 /home/jay/dev/algo/cron-wrap.sh --instance live live-session /home/jay/dev/algo/run-docker.sh live trading python -m v2.session

# Weekly deep learning analysis (5 AM MST / 7 AM ET, Sunday)
# 0 5 * * 0 /home/jay/dev/algo/cron-wrap.sh --instance live weekly-learn /home/jay/dev/algo/run-docker.sh live trading python -m v2.learn --days 60

# Nightly DB backups (8 PM MST, Mon-Fri) — see docs/runbook-recovery.md.
# --ignore-halt: these stay active through any hiatus. HALT means "do not
# trade", not "do not protect the data" — a hiatus is exactly when an
# unnoticed backup gap would cost the most.
0 20 * * 1-5 cd /home/jay/dev/algo/ && ./cron-wrap.sh --ignore-halt live-backup task db:backup INSTANCE=live >> logs/live/backup.log 2>&1
5 20 * * 1-5 cd /home/jay/dev/algo/ && ./cron-wrap.sh --ignore-halt paper-backup task db:backup INSTANCE=paper >> logs/paper/backup.log 2>&1
```

- [ ] **Step 2: Sanity-check**

Run: `grep -v '^#' crontab | grep -v '^\s*$' | grep -v '^PATH='` and confirm every active line goes through `./cron-wrap.sh`, and that redirect targets (`logs/live/`, `logs/paper/`) match the `LOGS_DIR` convention (dirs are created at cutover).

- [ ] **Step 3: Commit**

```bash
git add crontab
git commit -m "crontab: instance-scoped jobs, <instance>-<job> labels, per-instance halt"
```

(Do NOT run `crontab crontab` here — the installed crontab changes only at cutover, Task 9.)

---

### Task 8: Documentation sweep

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/runbook-recovery.md`, `.env.host.example`

**Interfaces:**
- Consumes: everything above. Historical docs (`docs/audits/`, `docs/superpowers/specs/` others, plan/spec archives) keep their prod/paper language — do not touch them.

- [ ] **Step 1: README.md**

Update the four paper/prod references (lines ~56, ~61, ~75–78): the runtime bullet becomes "Docker Compose stack (trading agent, db, dashboard); N isolated instances on one host, each defined by `instances/<name>.env` and addressed as `task <target> INSTANCE=<name>`". The quickstart's `cp .env.example .env` becomes `cp instances/example.env instances/paper.env` (edit keys, then `task up INSTANCE=paper`). The paper-overlay paragraph is replaced by a short "Instances" paragraph: each instance gets its own compose project (`pinchy-<name>`), Postgres volume, ports, and logs dir; add a second instance by adding a second env file.

- [ ] **Step 2: CLAUDE.md**

- Replace the "Pipelines: Paper vs Prod" section (title → "Instances") and its table with the instance model: one generic compose file; `instances/<name>.env` holds all config incl. `INSTANCE`, `DB_HOST_PORT`, `DASHBOARD_HOST_PORT`, `LOGS_DIR`, `ALGO_DASHBOARD_PUBLISH`; invocation shape `docker compose -p pinchy-<name> --env-file instances/<name>.env`; current instances `live` (5432/3000, publishes dashboard) and `paper` (5433/3001); no default instance — every task takes `INSTANCE=`.
- Update the "Commands" section: `task up INSTANCE=paper`, `task session INSTANCE=paper`, `task db:migrate INSTANCE=live`, `task db:restore INSTANCE=live FILE=backups/live-….dump`, and the raw compose equivalents.
- Update "Environment Variables": `.env`/`.env.paper` → `instances/<name>.env`; document `ALGO_DASHBOARD_PUBLISH` (opt-in, read at session start, exactly one publishing instance); update the `ALGO_TRADING_HALTED` entry's host-side twin wording to the two-level HALT (global `HALT`, per-instance `instances/<name>.HALT`, both checked by `cron-wrap.sh`/`run-docker.sh`); update the `ALGO_EXECUTOR_MODEL` sentence that references `.env.paper`.
- Update the Cron paragraph: label convention `<instance>-<job>`, `--instance` flag, `--ignore-halt` reserved for protect-the-data jobs.

- [ ] **Step 3: docs/runbook-recovery.md**

- "Halt / Resume": document both levels — global `HALT` (host trades nothing) vs `instances/<name>.HALT` (one instance), `touch`/`rm` + commit for visibility; note halted jobs still heartbeat-ping; note `ALGO_TRADING_HALTED` in an instance env file is the in-container twin but needs a container recreate on a long-running stack (sentinel files are the no-restart path). The current "paper exempted via --ignore-halt" wiring description is replaced by the per-instance sentinel model.
- Restore procedure: replace any manual `pg_restore` steps with `task db:restore INSTANCE=<name> FILE=backups/<file>.dump` followed by `task db:migrate INSTANCE=<name>` (expected no-op; non-empty output = investigate), keeping the manual form as a fallback footnote.
- Backup section: naming is now `backups/<instance>-<ts>.dump`.

- [ ] **Step 4: .env.host.example**

Update comments only: heartbeat label examples become `<instance>-<job>` (e.g. `ALGO_HEARTBEAT_URL_PAPER_SESSION`, `ALGO_HEARTBEAT_URL_LIVE_BACKUP`); any `task paper:db:backup` reference becomes `task db:backup INSTANCE=paper`.

- [ ] **Step 5: Sweep for stragglers**

Run: `grep -rn -i 'paper:\|\.env\.paper\|docker-compose\.paper\|logs_paper\|prod-backup\|paper-session task paper' README.md CLAUDE.md docs/runbook-recovery.md .env.host.example crontab Taskfile.yml`
Expected: no hits outside deliberately historical text. Fix any found.

- [ ] **Step 6: Full suite + lint, then commit**

Run: `task test INSTANCE=dev -- -q` and `ruff check .`
Expected: pass (docs changes can still break `test_schema_mirror`-style repo-inspection tests if paths moved — they didn't, but verify).

```bash
git add README.md CLAUDE.md docs/runbook-recovery.md .env.host.example
git commit -m "Docs: paper/prod duality replaced by named instances"
```

---

### Task 9: Cutover (operator-run, on the live host)

**Files:**
- Create (untracked): `instances/live.env`, `instances/paper.env`
- Create (tracked): `instances/live.HALT`
- Delete: `HALT`, `.env`, `.env.paper`

**Interfaces:**
- Consumes: everything above, merged to `main`.
- Timing: run outside 12:20–13:30 MST (paper session) and 19:55–20:20 MST (backups). Prod is halted, so only paper timing matters.

This task is a checklist, not code — execute it interactively with the operator, verifying each step's output before the next.

- [ ] **Step 1: Pre-flight (before touching the stacks)**

```bash
git checkout main && git pull            # branch merged
crontab -l | diff - crontab || true      # note drift; new file installs in step 7
mkdir -p logs/live logs/paper
```

- [ ] **Step 2: Cutover-time dumps of both old DBs (old stacks still up)**

```bash
docker compose up -d db
docker compose exec -T db pg_dump -U algo -d trading -Fc > backups/cutover-live-$(date +%Y%m%d-%H%M%S).dump
docker compose -f docker-compose.yml -f docker-compose.paper.yml up -d db-paper 2>/dev/null \
  || docker start algo-db-paper-1   # overlay file is deleted on main; start the existing container directly
docker exec algo-db-paper-1 pg_dump -U algo -d trading -Fc > backups/cutover-paper-$(date +%Y%m%d-%H%M%S).dump
ls -la backups/cutover-*            # both dumps non-trivially sized
```

(Old compose files are gone from the merged tree, so drive old containers by name — `docker ps -a` shows the exact names, likely `algo-db-1` / `algo-db-paper-1`.)

- [ ] **Step 3: Stop old stacks**

```bash
docker ps --format '{{.Names}}'                       # inventory first
docker stop $(docker ps -q --filter name=algo-)       # old project containers only
```

- [ ] **Step 4: Write the two instance env files**

`instances/live.env`: contents of old `.env`, plus `INSTANCE=live`, `DB_HOST_PORT=5432`, `DASHBOARD_HOST_PORT=3000`, `LOGS_DIR=./logs/live`, `ALGO_DASHBOARD_PUBLISH=true`.
`instances/paper.env`: contents of old `.env.paper`, plus `INSTANCE=paper`, `DB_HOST_PORT=5433`, `DASHBOARD_HOST_PORT=3001`, `LOGS_DIR=./logs/paper`, and no `ALGO_DASHBOARD_PUBLISH` (or `=false`).
Verify each against `instances/example.env` for any var the old files lack (esp. `ALGO_DASHBOARD_PUBLISH`, `INSTANCE`).

- [ ] **Step 5: Bring up new stacks, restore, migrate**

```bash
task up INSTANCE=live  && task db:restore INSTANCE=live  FILE=backups/cutover-live-<ts>.dump
task up INSTANCE=paper && task db:restore INSTANCE=paper FILE=backups/cutover-paper-<ts>.dump
task db:migrate INSTANCE=live    # expected: every file "skipping (already applied)"
task db:migrate INSTANCE=paper   # same. ANY "applying" line = STOP and investigate.
```

- [ ] **Step 6: Verify data parity + dry-run**

```bash
for i in live paper; do
  docker compose -p pinchy-$i --env-file instances/$i.env exec -T db psql -U algo -d trading -tAc \
    "SELECT (SELECT MAX(session_date) FROM sessions), (SELECT COUNT(*) FROM decisions), (SELECT COUNT(*) FROM theses), (SELECT COUNT(*) FROM strategy_memos)"
done
# compare against the same query run via the old containers (docker start algo-db-1 …) — must match
task session:dry-run INSTANCE=paper   # completes; stage 5 SKIPPED (dry-run)
```

- [ ] **Step 7: Swap HALT, install crontab, update monitoring**

```bash
git mv HALT instances/live.HALT    # then edit: update the Effect/scope text to the per-instance model (it now means "the live instance trades nothing"; the paper --ignore-halt history note becomes past tense)
git commit -m "Move trading hiatus to instances/live.HALT (per-instance halt)"
rm .env .env.paper                 # contents now live in instances/
crontab crontab && crontab -l | head -5
# .env.host: rename ALGO_HEARTBEAT_URL_PROD_BACKUP → ALGO_HEARTBEAT_URL_LIVE_BACKUP if present
# healthchecks.io (or equivalent): rename/re-create the prod-backup check as live-backup
```

- [ ] **Step 8: Confirm halt behavior end-to-end**

```bash
./cron-wrap.sh --instance live live-session ./run-docker.sh live trading python -m v2.session
# expect: "instances/live.HALT present — skipping", exit 0, no containers started
./cron-wrap.sh --instance paper smoke task session:dry-run INSTANCE=paper   # runs (idempotency may no-op it — fine)
```

- [ ] **Step 9: Decommission window**

Leave old containers stopped and old volumes (`algo_postgres_data`, `algo_postgres_data_paper` — confirm names via `docker volume ls`) in place until BOTH: one cron-driven paper session has succeeded (check `logs/paper/`, heartbeat dashboard) and one nightly backup pair has landed in `backups/` with `live-*`/`paper-*` names. Then:

```bash
docker rm $(docker ps -aq --filter name=algo-)
docker volume rm algo_postgres_data algo_postgres_data_paper
```

---

## Self-Review Notes

- Spec §1 (instance file, compose, `${VAR:?}`, self-naming `INSTANCE`) → Tasks 2–3. §2 (Taskfile, no default, `db:restore`, backup naming) → Task 4. §3 (publish gate, default false, CLI wins, skip-source logging) → Task 1. §4 (run-docker instance arg, cron labels, two-level HALT) → Tasks 5–7. §5 (live/paper definitions, cutover, volume retention) → Task 9. §6 (tests, CI untouched, docs) → Tasks 1, 8.
- Type consistency: `_dashboard_publish_enabled()` and `skip_reason` names match across Task 1 steps; `COMPOSE`/`INSTANCE`/`pinchy-<name>` shapes match across Tasks 3–7 and 9.
- Known intentional behavior changes, all spec'd: publish default flips to opt-in; paper's `trading` container gains ro mounts of tests/db; `docker:*` task names lose their prefix; `prod-*` labels become `live-*`.
