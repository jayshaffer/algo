# Instance Genericisation — Design Spec

**Date:** 2026-09-21
**Status:** Approved design, pending implementation plan

## Problem

The repo hardcodes exactly two pipelines — prod and paper — as parallel
artifacts: a `docker-compose.paper.yml` overlay that near-duplicates the base
compose file with `-paper` service names, a `paper:*` Taskfile family that
duplicates `db:migrate`/`db:backup` line for line, `.env` vs `.env.paper`, and
`./logs` vs `./logs_paper`. Running a third account (e.g. a second paper
account piloting a different executor model) means copying all of it again.

The driving use case is **multiple instances side by side on one host**, each
pointed at its own Alpaca account. The paper/prod concept disappears; what
remains is N named instances, each defined by one config file.

The Python code in `v2/` is already instance-agnostic — it reads env vars.
`ALPACA_PAPER` is honest Alpaca-client configuration (which kind of account is
this key for), not pipeline identity, and survives unchanged along with its
`ALPACA_BASE_URL` cross-check.

## Design

### 1. Instance definition & compose

An instance is **one env file**: `instances/<name>.env`. The `instances/`
directory is gitignored except a committed, fully commented
`instances/example.env`.

Each file contains everything the current `.env` holds (Alpaca keys,
`ALPACA_PAPER`, Anthropic key, Postgres creds, model/risk knobs) plus four
deployment vars consumed by compose interpolation:

```bash
INSTANCE=live              # must match the filename; see env_file note below
DB_HOST_PORT=5432          # host port for Postgres (127.0.0.1-bound)
DASHBOARD_HOST_PORT=3000   # host port for the local dashboard
LOGS_DIR=./logs/live       # host bind-mount for /app/logs
```

Note on `env_file`: the `--env-file` flag only supplies compose
*interpolation* variables — it does not inject env into containers. The
services therefore keep an `env_file:` entry, pointed at the instance file
via interpolation: `env_file: [instances/${INSTANCE:?}.env]`, with
`INSTANCE` declared inside each instance file (self-naming). Interpolated
vars use the `${VAR:?}` required form throughout, so a manual
`docker compose` invocation missing the env file fails immediately instead
of mounting empty paths or binding empty ports.

`docker-compose.yml` becomes the single generic stack file — same three
services (`db`, `trading`, `dashboard`), with interpolated host ports and logs
mount. `docker-compose.paper.yml` is deleted. Every invocation is:

```
docker compose -p pinchy-<name> --env-file instances/<name>.env ...
```

The project name namespaces containers (`pinchy-paper-db-1`), the network, and
the volume (`pinchy-paper_postgres_data`). Service names stay plain. Port
collisions between instances fail loudly at `up` (bind error); there is no
auto-allocation.

`.env` and `.env.paper` are retired. `.env.host` stays as-is: host-side,
instance-agnostic (per-job heartbeat overrides already distinguish instances
via cron labels).

### 2. Taskfile

One set of targets, all instance-scoped; the `paper:*` family is deleted.
Shared invocation defined once:

```yaml
vars:
  COMPOSE: docker compose -p pinchy-{{.INSTANCE}} --env-file instances/{{.INSTANCE}}.env
```

Every stack-touching target requires the instance explicitly — there is **no
default instance** (real money; "forgot the flag" must fail, not silently
target an account):

```yaml
requires: { vars: [INSTANCE] }
preconditions:
  - sh: test -f instances/{{.INSTANCE}}.env
    msg: "No such instance: instances/{{.INSTANCE}}.env"
```

Usage: `task session INSTANCE=paper`, `task db:migrate INSTANCE=live`.

- `docker:up/down/build/logs/stop:session` → `up/down/build/logs/stop:session`,
  instance-scoped.
- Workflow targets (`session`, `trade`, `ideation`, `pipeline`, `learn`,
  `supervise`, `backfill`, `dashboard:publish`, dry-run variants) keep their
  names, gain instance plumbing, exec into plain `trading`.
- `db:migrate` / `db:backup` collapse to one copy each. Backups are
  `backups/<instance>-<timestamp>.dump`, retention pruned per instance
  (newest 14).
- New target: `task db:restore INSTANCE=x FILE=backups/foo.dump`
  (pg_restore into the instance's db) — needed by the cutover, and gives the
  DR runbook a command where it currently documents only manual steps.
- `test` / `test:coverage` take `INSTANCE` (external deps are mocked; any
  instance's trading container works).
- `lint`, `install-hooks`, `audit:*` stay instance-agnostic.

### 3. Dashboard-publish control

"Paper doesn't publish" is currently the `paper:session` target hardcoding
`--skip-dashboard`. That becomes instance config: new env var
**`ALGO_DASHBOARD_PUBLISH`**, read at session start (same semantics as
`ALGO_TRADING_HALTED` — no container restart needed).

- **Default `false` (opt-in).** There is exactly one public Cloudflare Pages
  site; a new instance accidentally publishing over it is the failure mode to
  prevent. Only the instance owning the public dashboard sets it `true`, and
  only that instance needs Cloudflare credentials in its env file.
- This **flips the current default** (prod publishes unless told otherwise
  today). `instances/live.env` sets `ALGO_DASHBOARD_PUBLISH=true` explicitly.
- CLI `--skip-dashboard` stays and wins over the env var (one-off override).
  The `--dry-run` promotion to skip is untouched.
- Implementation: `run_session()` resolves
  `skip_dashboard = skip_dashboard or not env_flag("ALGO_DASHBOARD_PUBLISH")`,
  and the stage-5 skip log line names which source caused the skip.

### 4. Host scripts, cron, HALT

**`run-docker.sh`** (ephemeral up→exec→down wrapper) gains the instance as
its first argument: `./run-docker.sh <instance> <service> <cmd...>`. It builds
the same `-p pinchy-<name> --env-file instances/<name>.env` prefix as the
Taskfile. Teardown becomes project-scoped — a bare `docker compose down`
would take other instances' stacks down once projects share the host.

**Crontab** lines become instance-scoped invocations, e.g.
`./cron-wrap.sh paper-session task session INSTANCE=paper`. Label convention:
`<instance>-<job>`. Existing per-label heartbeat overrides
(`ALGO_HEARTBEAT_URL_<LABEL>`) work unchanged.

**HALT becomes two-level:**

- **Global `HALT`** (repo root, unchanged): "this host trades nothing."
  Checked by `cron-wrap.sh` and `run-docker.sh`. `--ignore-halt` remains for
  protect-the-data jobs (backups).
- **Per-instance `instances/<name>.HALT`**: `cron-wrap.sh` gains an optional
  `--instance <name>` flag and checks both sentinels. The current state —
  paper trading through a prod hiatus, expressed as global HALT plus
  `--ignore-halt` smuggled onto the paper session line — becomes: no global
  HALT, `instances/live.HALT` present, no `--ignore-halt` on any trading job.
- `ALGO_TRADING_HALTED` stays as the in-container twin, naturally
  per-instance now. Documented caveat: env-file changes need a container
  recreate on a long-running stack; the sentinel files are the no-restart
  path.

`cron-wrap.sh`'s failure log stays global at `logs/session_failures.log` —
one place to look; labels disambiguate.

### 5. Initial instances & cutover

The two pipelines become instances **`live`** and **`paper`**:

| | `instances/live.env` | `instances/paper.env` |
|---|---|---|
| Source | `.env` | `.env.paper` |
| `DB_HOST_PORT` | 5432 | 5433 |
| `DASHBOARD_HOST_PORT` | 3000 | 3001 |
| `LOGS_DIR` | `./logs/live` | `./logs/paper` |
| `ALGO_DASHBOARD_PUBLISH` | `true` | `false` (or omitted) |

Log convention becomes `logs/<instance>/`. Old `logs/` files and `logs_paper/`
stay in place as history; cron redirect targets move to the new paths.

**Cutover sequence** (outside the 12:30 paper session and 20:00 backup
windows; prod is halted so only paper timing matters):

1. Land all changes on a branch; full test suite green;
   `task session:dry-run` exercised against a scratch instance.
2. Take fresh dumps of both DBs (cutover-time ones, not just the nightlies).
3. `docker compose down` both old stacks; merge the branch; write the two
   instance env files.
4. `task up INSTANCE=live` / `INSTANCE=paper` — fresh volumes seed from
   `db/init/`. Then `task db:restore` each dump, then `task db:migrate` each.
   Migrate is expected to be a no-op (`schema_migrations` rides along in the
   dump); a non-empty result is a red flag — stop and investigate.
5. Verify: latest session date and row counts per instance match the old DBs;
   `task session:dry-run INSTANCE=paper` passes.
6. Install the rewritten crontab. Replace root `HALT` with
   `instances/live.HALT` (explanatory text moves with it). Update
   `docs/runbook-recovery.md`.
7. Keep old `algo_postgres_data*` volumes until one cron-driven paper session
   and one nightly backup have succeeded, then delete them. `.env` /
   `.env.paper` are deleted at merge.

### 6. Tests, CI, docs

- **CI untouched.** `tests.yml` sets env vars directly (no compose env
  files); its `paper-api` values are Alpaca-client config and survive.
  `db/check_mirror.sh` is instance-agnostic.
- **Python tests:** `ALPACA_PAPER` cross-check tests stay. New unit tests for
  `ALGO_DASHBOARD_PUBLISH` gating in `run_session`: unset → skip, `false` →
  skip, `true` → publish, CLI `--skip-dashboard` wins over `true`. Existing
  `test_session` tests assuming publish-by-default get the flag set. No new
  session stages, so no `_NETWORK_PATCH_TARGETS` additions.
- **Shell changes** (`cron-wrap.sh --instance`, `run-docker.sh` instance arg,
  Taskfile) have no test harness; verified via the cutover checklist plus
  `bash -n` / `task --list` smoke checks. No shell test framework added.
- **Docs:** CLAUDE.md "Pipelines: Paper vs Prod" becomes "Instances"; README,
  `docs/runbook-recovery.md` (halt/resume + restore), crontab header
  comments, and `.env.host.example` comments updated. `instances/example.env`
  is the canonical instance-config reference. Historical docs/audits keep
  their prod/paper language.

## Out of scope

- Auto-allocation of ports; collision = loud bind error at `up`.
- Any change to the `v2/` trading logic beyond the `ALGO_DASHBOARD_PUBLISH`
  gate in `session.py`.
- Multi-host orchestration, shared services between instances, or per-instance
  compose overrides beyond the env file (YAGNI until a real instance needs
  one).
- Renaming historical backup files or log archives.
