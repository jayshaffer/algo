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

`task db:backup INSTANCE=<name>` writes a compressed `pg_dump -Fc` archive to
`backups/` (gitignored), named `backups/<instance>-<ts>.dump`:

```
backups/live-20260715-200000.dump
backups/paper-20260715-200500.dump
```

- **Retention:** the newest 14 dumps per instance; older ones are deleted by the
  task itself.
- **Off-host copy:** set `ALGO_BACKUP_COPY_DIR` and each backup is also copied
  there. Point it at the Windows filesystem (e.g.
  `/mnt/c/Users/<you>/pinchy-backups`) or anything synced off the machine —
  `backups/` alone lives inside the same WSL2 `ext4.vhdx` as the docker volume
  it is backing up, so it does not protect against the failure mode that
  matters most (VHD corruption, `docker compose down -v`, disk death).
- **Schedule:** the repo `crontab` runs both nightly at 8:00/8:05 PM MST,
  Mon–Fri (`live-backup`, `paper-backup`), logging to `logs/live/backup.log`
  and `logs/paper/backup.log`. These lines stay active during a trading
  hiatus — the learning history is worth preserving whether or not the
  system is trading.
- `task db:backup` depends on `up` and **leaves the instance's containers
  running** — the task doesn't tear anything down afterward. For `live`,
  whose normal operating pattern is ephemeral via `run-docker.sh`, that means
  a nightly backup leaves `docker compose -p pinchy-live ps` showing a
  running stack until the next session tears it down. Harmless (no trading
  happens outside an explicit session), but worth knowing if you're using
  `docker compose ps` as a signal for "is `live` running."

## Restore

```bash
task db:restore INSTANCE=live FILE=backups/live-<stamp>.dump
task db:migrate INSTANCE=live   # re-apply any migrations newer than the dump; expect no output
```

`task db:restore` brings the instance's stack up if it isn't already running,
then runs `pg_restore --clean --if-exists --no-owner` against that instance's
`db` container. The `db:migrate` step afterward is expected to be a no-op —
the dump carries its own `schema_migrations` rows — so if it prints anything
under `==> applying`, the dump predates a migration and that's worth
investigating before trusting the restored data. Substitute
`INSTANCE=paper` (or any other instance name) for a different instance.

To verify an archive is readable without restoring it:

```bash
docker compose -p pinchy-live --env-file instances/live.env exec -T db \
    pg_restore --list < backups/live-<stamp>.dump | head
```

Note the bare `--list` with no filename: it must read stdin. Passing
`/dev/stdin` explicitly fails with "did not find magic string in file header",
because a custom-format archive read that way isn't seekable — the archive is
fine, the command isn't. A healthy dump lists ~27 `TABLE DATA` entries
including `decisions`, `theses`, `strategy_memos`, and `signal_attribution`.

**Restoring into a fresh volume:** `db/init/` runs only on a brand-new volume,
so a fresh `task up INSTANCE=<name>` gives you a schema at whatever revision
`db/init/` describes. `task db:restore` applies the dump on top of it with
`--clean --if-exists`, then `task db:migrate` correctly applies only what the
dump predates.

**Manual fallback**, if the Taskfile targets themselves are unavailable:

```bash
docker compose -p pinchy-live --env-file instances/live.env up -d db
docker compose -p pinchy-live --env-file instances/live.env exec -T db \
    pg_restore -U algo -d trading --clean --if-exists --no-owner < backups/live-<stamp>.dump
```

## Secrets inventory

Names only — values live in `instances/<name>.env` (one file per instance),
which are gitignored except `instances/example.env` and **not backed up by
anything**. Keep an encrypted copy off the host (password manager or an
encrypted archive on the Windows side). `instances/example.env` lists the
full set of keys.

| Secret | Where to re-issue |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Alpaca dashboard — one key pair per instance, in that instance's `instances/<name>.env` |
| `ALPACA_BASE_URL`, `ALPACA_PAPER` | Not secret, but must agree with each other or module import raises |
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_PAGES_PROJECT` | Cloudflare dashboard (only needed by the instance with `ALGO_DASHBOARD_PUBLISH=true` — `live` at cutover; `paper` deliberately has no Cloudflare creds so it can never publish the public dashboard) |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Self-chosen; must match what the existing volume was initialized with, or restore into a fresh volume |

## Host bootstrap (new machine)

Cloning the repo is not enough. In order:

1. **Secrets:** restore `instances/live.env` and `instances/paper.env` (or
   whichever instances this host runs) from the encrypted off-host copy (or
   re-issue per the table above). Also create `.env.host` from
   `.env.host.example` — host-side scripts read it, and it is where alerting
   and the dead-man's switch are configured.
2. **Stacks:** `task up INSTANCE=live` / `task up INSTANCE=paper`.
3. **Schema:** `task db:migrate INSTANCE=live` and `task db:migrate
   INSTANCE=paper`. Fresh volumes run `db/init/` automatically, but
   long-lived volumes drift — this is the third time that drift has bitten
   (audit 3.3). If restoring data, do the restore first, then migrate.
4. **Cron:** `crontab /home/jay/dev/algo/crontab`.
5. **Make cron actually run:** WSL2 does not start cron on boot. It runs only
   because `start-wsl-cron.bat` (`wsl -u root service cron start`) is wired into
   **Windows Task Scheduler** to fire at login. Re-create that entry — without
   it the system goes permanently, silently quiet, and nothing alerts (audit
   C.4). This is the single least-obvious dependency in the whole setup.
6. **Backups:** set `ALGO_BACKUP_COPY_DIR` in `.env.host` and confirm
   `task db:backup INSTANCE=<name>` writes both locally and to the off-host
   directory, for each instance this host runs.
7. **Monitoring:** set `ALGO_ALERT_WEBHOOK_URL` and per-job
   `ALGO_HEARTBEAT_URL_*` in `.env.host`, then confirm a ping lands. Steps 4–6
   are all silent when they fail; this is the step that makes them audible.
   Verify with `./cron-wrap.sh smoke-test true` (success ping) and
   `./cron-wrap.sh --ignore-halt smoke-test false` (alert + failure ping).

## Halt / Resume

Two independent sentinel levels, either of which stops trading, plus an
in-container twin. All are deliberately git-visible — the 2026-06 hiatus was
implemented by hand-editing the *installed* crontab, which left no trace in
the repo and was indistinguishable from an unnoticed failure (audit 0.1, C.2).

**Halt:**

```bash
touch HALT                     # global: every instance halts
touch instances/<name>.HALT    # per-instance: just that one halts
```

`cron-wrap.sh` always checks the global `HALT` file (unless the job was
invoked with `--ignore-halt`); when invoked with `--instance <name>` it also
checks `instances/<name>.HALT`. `run-docker.sh` always checks both — it
always operates on a single named instance, so there's no `--ignore-halt`
equivalent for it. This replaced an earlier scheme where a paper exemption
lived as a `--ignore-halt` flag hand-added to one crontab line; a per-instance
sentinel file now expresses "halt just this instance" directly, without
touching any job's `--ignore-halt` status (which stays reserved for jobs that
protect data rather than trade, like the nightly backups).

`touch`/`rm` these files and **commit the change** — a sentinel with no git
trace is indistinguishable from an unnoticed failure, which is the reason
this file-based mechanism exists at all.

Both levels have `ALGO_TRADING_HALTED=1` — set in the affected instance's
`instances/<name>.env` — as the in-container twin, checked at session start,
so it also covers `task session` / manual `python -m v2.session` invocations
that bypass the host scripts. Unlike the sentinel files, though, it lives in
the container's environment: on a long-running stack, editing the env file
alone does not reach a container that's already up, so it needs a recreate
(`task up INSTANCE=<name>`) before it takes effect. The sentinel files are the
no-restart path — they're checked by the host scripts *before* a container is
even started, so a `touch`/`rm` takes effect on the very next scheduled run
with nothing to recreate.

Every halt path logs loudly and **exits 0** — a deliberate halt is not a
failure and must not trip the failure alerting. A halted job *does* still
send its liveness ping: the dead-man's switch answers "is this host up and is
cron firing", and during a hiatus the answer is yes. Without that, going
quiet on purpose would page you exactly as loudly as going quiet by accident,
and you would learn to ignore both.

The nightly backups run under `--ignore-halt` for every instance and keep
going through a hiatus.

**Resume (for one instance):**

1. Review the audit's Tier 1 (money-path safety) and 0.2 (economics) status —
   Tier 1 was explicitly a prerequisite for re-enabling the cron.
2. `rm instances/<name>.HALT` (or `rm HALT` to resume every instance at once),
   and clear `ALGO_TRADING_HALTED` in that instance's `instances/<name>.env`,
   then `task up INSTANCE=<name>` to recreate the container so the change
   takes effect.
3. Uncomment that instance's session + weekly-learn lines in the repo
   `crontab`.
4. Reinstall: `crontab /home/jay/dev/algo/crontab`.
5. Commit the crontab + sentinel removal so the resume is as visible as the
   halt was.
6. Confirm the resuming instance's Alpaca account actually holds the capital
   you expect — an account fully liquidated during a hiatus has no record of
   it in the DB as a deposit or withdrawal (audit B.6/2.6). An equity jump
   from a re-deposit will read as performance to the dashboard and the
   reflection stage.

**Current state:** see the `HALT` file in the repo root for current scope,
why, and since when, and check for any `instances/<name>.HALT` files for
halts scoped to a single instance.

The automated breakers (`ALGO_DAILY_LOSS_LIMIT_PCT` daily-loss circuit breaker,
`ALGO_LOOP_COST_CEILING_USD` per-loop cost ceiling) are unaffected by the halt
switches and remain the backstops once trading resumes. They halt *new orders*;
neither cancels open orders.
