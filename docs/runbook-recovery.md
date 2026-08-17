# Recovery Runbook

Written in response to the 2026-07-15 fresh-eyes audit (findings C.1 "zero DB
backups" and C.6 "machine loss is unrecoverable-by-documentation"). Everything
below assumes the repo is checked out at `/home/jay/dev/algo` on a WSL2 host
with Docker.

The code is the easy part — it's on GitHub. This document covers the parts that
only exist on one machine: the database, the secrets, and the host wiring that
makes cron fire.

---

## Backups

`task db:backup` and `task paper:db:backup` each write a compressed
`pg_dump -Fc` archive to `backups/` (gitignored):

```
backups/prod-20260715-200000.dump
backups/paper-20260715-200500.dump
```

- **Retention:** the newest 14 dumps per pipeline; older ones are deleted by the
  task itself.
- **Off-host copy:** set `ALGO_BACKUP_COPY_DIR` and each backup is also copied
  there. Point it at the Windows filesystem (e.g.
  `/mnt/c/Users/<you>/pinchy-backups`) or anything synced off the machine —
  `backups/` alone lives inside the same WSL2 `ext4.vhdx` as the docker volume
  it is backing up, so it does not protect against the failure mode that
  matters most (VHD corruption, `docker compose down -v`, disk death).
- **Schedule:** the repo `crontab` runs both nightly at 8:00/8:05 PM MST,
  Mon–Fri, logging to `logs/backup.log` and `logs_paper/backup.log`. These lines
  stay active during a trading hiatus — the learning history is worth preserving
  whether or not the system is trading.
- `task db:backup` depends on `docker:up` and **leaves the prod `db` container
  running**. That is harmless (the resting state is prod-down/paper-up; the
  `trading` service is not started), but it means `docker compose ps` will show
  a prod container between sessions.

## Restore

```bash
docker compose up -d db
docker compose exec -T db pg_restore -U algo -d trading --clean --if-exists < backups/prod-<stamp>.dump
task db:migrate   # re-apply any migrations newer than the dump
```

For paper, substitute `db-paper` and the paper compose overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.paper.yml up -d db-paper
docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper \
    pg_restore -U algo -d trading --clean --if-exists < backups/paper-<stamp>.dump
task paper:db:migrate
```

To verify an archive is readable without restoring it:

```bash
docker compose exec -T db pg_restore --list < backups/prod-<stamp>.dump | head
```

Note the bare `--list` with no filename: it must read stdin. Passing
`/dev/stdin` explicitly fails with "did not find magic string in file header",
because a custom-format archive read that way isn't seekable — the archive is
fine, the command isn't. A healthy prod dump lists ~27 `TABLE DATA` entries
including `decisions`, `theses`, `strategy_memos`, and `signal_attribution`.

**Restoring into a fresh volume:** `db/init/` runs only on a brand-new volume,
so a fresh `docker compose up -d db` gives you a schema at whatever revision
`db/init/` describes. Restore on top of it with `--clean --if-exists`, then run
`task db:migrate` — the dump carries its own `schema_migrations` rows, so the
migrate step correctly applies only what the dump predates.

## Secrets inventory

Names only — values live in `.env` / `.env.paper`, which are gitignored and
**not backed up by anything**. Keep an encrypted copy off the host (password
manager or an encrypted archive on the Windows side). `.env.example` lists the
full set of keys.

| Secret | Where to re-issue |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Alpaca dashboard — live keys in `.env`, paper keys in `.env.paper` |
| `ALPACA_BASE_URL`, `ALPACA_PAPER` | Not secret, but must agree with each other or module import raises |
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_PROJECT_NAME` | Cloudflare dashboard (prod `.env` only — paper deliberately has no Cloudflare creds so it can never publish the public dashboard) |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Self-chosen; must match what the existing volume was initialized with, or restore into a fresh volume |

## Host bootstrap (new machine)

Cloning the repo is not enough. In order:

1. **Secrets:** restore `.env` and `.env.paper` from the encrypted off-host copy
   (or re-issue per the table above). Also create `.env.host` from
   `.env.host.example` — host-side scripts read it, and it is where alerting
   and the dead-man's switch are configured.
2. **Stacks:** `docker compose up -d db` / `task paper:up`.
3. **Schema:** `task db:migrate` and `task paper:db:migrate`. Fresh volumes run
   `db/init/` automatically, but long-lived volumes drift — this is the third
   time that drift has bitten (audit 3.3). If restoring data, do the restore
   first, then migrate.
4. **Cron:** `crontab /home/jay/dev/algo/crontab`.
5. **Make cron actually run:** WSL2 does not start cron on boot. It runs only
   because `start-wsl-cron.bat` (`wsl -u root service cron start`) is wired into
   **Windows Task Scheduler** to fire at login. Re-create that entry — without
   it the system goes permanently, silently quiet, and nothing alerts (audit
   C.4). This is the single least-obvious dependency in the whole setup.
6. **Backups:** set `ALGO_BACKUP_COPY_DIR` in `.env.host` and confirm
   `task db:backup` writes both locally and to the off-host directory.
7. **Monitoring:** set `ALGO_ALERT_WEBHOOK_URL` and per-job
   `ALGO_HEARTBEAT_URL_*` in `.env.host`, then confirm a ping lands. Steps 4–6
   are all silent when they fail; this is the step that makes them audible.
   Verify with `./cron-wrap.sh smoke-test true` (success ping) and
   `./cron-wrap.sh --ignore-halt smoke-test false` (alert + failure ping).

## Halt / Resume

Two independent switches, either of which stops trading. Both are deliberately
git-visible — the 2026-06 hiatus was implemented by hand-editing the *installed*
crontab, which left no trace in the repo and was indistinguishable from an
unnoticed failure (audit 0.1, C.2).

**Halt:**

```bash
touch HALT      # cron-wrap.sh / run-docker.sh exit 0 before starting containers
```

and/or set `ALGO_TRADING_HALTED=1` in `.env` (checked at session start inside
the container, so it also covers `task session` / manual `python -m v2.session`
invocations that bypass the host scripts).

Both paths log loudly and **exit 0** — a deliberate halt is not a failure and
must not trip the failure alerting. A halted job *does* still send its liveness
ping: the dead-man's switch answers "is this host up and is cron firing", and
during a hiatus the answer is yes. Without that, going quiet on purpose would
page you exactly as loudly as going quiet by accident, and you would learn to
ignore both.

The nightly backups run under `--ignore-halt` and keep going through a hiatus.

**Resume:**

1. Review the audit's Tier 1 (money-path safety) and 0.2 (economics) status —
   Tier 1 was explicitly a prerequisite for re-enabling the cron.
2. `rm HALT` and remove/clear `ALGO_TRADING_HALTED` in `.env`.
3. Uncomment the session + weekly learn lines in the repo `crontab`.
4. Reinstall: `crontab /home/jay/dev/algo/crontab`.
5. Commit the crontab + HALT removal so the resume is as visible as the halt was.
6. Confirm the prod Alpaca account actually holds the capital you expect — the
   account was fully liquidated during the hiatus, and nothing in the DB records
   deposits or withdrawals (audit B.6/2.6). An equity jump from a re-deposit
   will read as performance to the dashboard and the reflection stage.

**Current state:** halted. See the `HALT` file in the repo root for why and
since when.

The automated breakers (`ALGO_DAILY_LOSS_LIMIT_PCT` daily-loss circuit breaker,
`ALGO_LOOP_COST_CEILING_USD` per-loop cost ceiling) are unaffected by the halt
switches and remain the backstops once trading resumes. They halt *new orders*;
neither cancels open orders.
