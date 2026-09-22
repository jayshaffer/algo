# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pinchy — an agentic trading system that uses Claude (via Anthropic API) to integrate with the Alpaca trading API, learn from past behavior, and make trading decisions. The public dashboard is the project's surface area; the automated social posting layer was retired 2026-05-15.

**Status:** Active development — `v2/` is the current active codebase.

## Codebase Layout

- **`v2/`** — Current active codebase. All new work goes here.
- **`trading/`** — Legacy v1 module. Mostly sunset; individual pieces are pulled into the v2 pipeline as needed. Do not add new features here.
- **`tests/`** — Test suite covering both v1 and v2.
- **`dashboard/`** — Legacy v1 dashboard (Flask on port 3000). v2 dashboard lives in `v2/dashboard/`.

## Instances

The same v2 code and one generic `docker-compose.yml` run any number of named
instances on one host. An instance is `instances/<name>.env`: it holds all
container config (Alpaca account, Anthropic key, Postgres credentials) plus
the deployment shape — `INSTANCE` (self-naming, must match the filename),
`DB_HOST_PORT`, `DASHBOARD_HOST_PORT`, `LOGS_DIR`, and `ALGO_DASHBOARD_PUBLISH`
(whether this instance owns the public dashboard publish). `instances/` is
gitignored except `instances/example.env` (the template) and `*.HALT`
sentinel files.

Invocation shape: `docker compose -p pinchy-<name> --env-file
instances/<name>.env`. Every Taskfile target that touches a stack requires
`INSTANCE=<name>` explicitly — there is no default instance, so a forgotten
flag fails loudly instead of silently hitting whichever account was last
used. Targets are unprefixed (`up`, `down`, `session`, `db:migrate`,
`db:backup`, `db:restore INSTANCE=x FILE=...`, `test`, etc.); there is no
`paper:*` or `docker:*` target family.

Planned instances at cutover: `live` (`DB_HOST_PORT=5432`,
`DASHBOARD_HOST_PORT=3000`, publishes the public dashboard) and `paper`
(`5433`/`3001`, does not publish). Each instance has its own compose
project, Postgres volume, ports, and logs directory, so data never crosses
between instances.

## Project Goals

- Prove whether agentic trading can find an edge
- Claude (Haiku for execution, Sonnet/Opus for ideation & reflection) makes trading decisions
- Daily automated session after market close
- Learning system that journals behavior, computes signal attribution, and reflects on strategy
- Single Alpaca account with an evolving day-to-day strategy
- Public dashboard published to Cloudflare Pages

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Docker Compose Stack                        │
├───────────────┬───────────────┬───────────────┬──────────────────┤
│  PostgreSQL   │   Claude API  │   Trading     │   Dashboard      │
│  (pgvector)   │  (Anthropic)  │   Agent (v2)  │   (v2/dashboard) │
│  :5432        │               │               │                  │
└───────────────┴───────────────┴───────────────┴──────────────────┘
```

- **LLM:** Claude via Anthropic API (Haiku for execution, larger models for ideation/reflection)
- **Database:** PostgreSQL 16 + pgvector
- **API:** Alpaca Trading API (read/write)
- **Dashboard:** Published to Cloudflare Pages

## v2 Daily Session (`v2/session.py`)

The session orchestrator runs stages sequentially. Each stage is independent — failures don't block subsequent stages.

| Stage | Module | Purpose |
|-------|--------|---------|
| 0 | `backfill.py`, `attribution.py` | Learning refresh: backfill decision outcomes, compute signal attribution |
| 0.5 | `supervisor.py` | Strategy supervisor: observer-only critic, records watchlist items the acting stages must resolve |
| 1 | `pipeline.py` | News pipeline: fetch from Alpaca, classify with Haiku, store signals |
| 2 | `ideation_claude.py` | Strategist: thesis management + playbook generation (agentic loop with tools) |
| 3 | `trader.py` | Executor: decisions from playbook + order execution |
| 4 | `strategy.py` | Reflection: update strategy identity, rules, and write session memo |
| 5 | `dashboard_publish.py` | Public dashboard publish |

A session is idempotent per market date: if ANY session row already exists
for the date (completed, failed, or running), a re-run is a no-op unless
`--force` is passed. Deliberate retries after a failure need `--force`.

The dashboard renders permalinks at `/mistakes/` and `/attribution/`
(surfacing closed losers + retired rules, and best/worst signal types
respectively) so those pages stay fresh against each session's data.

### Retired: social posting pipeline

The Twitter/Bluesky/entertainment/premarket/weekly stack and its
"Mr. Krabs from SpongeBob, running Bikini Bottom Capital" persona were
removed 2026-05-15. The `tweets` table is retained for historical data;
no new rows are written. If you find vestigial references in `docs/`
historical plan/spec docs, leave them — they document the project's
prior identity.

### Key v2 Modules

- **`agent.py`** — Executor LLM integration. Gets structured trading decisions from Claude Haiku.
- **`claude_client.py`** — Claude API client with tool handling and agentic loop support.
- **`context.py`** — Context builder. Aggregates positions, signals, theses, playbook, and attribution into compressed LLM context.
- **`ideation_claude.py`** — Strategist stage. Agentic loop where Claude manages theses and generates playbooks using database tools.
- **`strategy.py`** — Post-session reflection. Claude reviews outcomes, updates trading identity, proposes/retires rules, writes memos.
- **`supervisor.py`** — Observer-only strategy critic. Read-only DB tool registry + agentic loop (pinned to `claude-fable-5`), persists one markdown memo per run to `supervisor_memos`. Run with `task supervise` (or `task supervise:dry-run`). Spec: `docs/superpowers/specs/2026-05-27-strategy-supervisor-design.md`.
- **`attribution.py`** — Computes which signal types are predictive by joining decisions with their source signals.
- **`patterns.py`** — Pattern analysis: signal performance, sentiment performance metrics.
- **`tools.py`** — Tool definitions and handlers for the agentic loops (portfolio state, theses, history, attribution, etc.).
- **`risk.py`** — Risk management and position sizing.
- **`executor.py`** — Alpaca API integration (orders, positions, account info).
- **`learn.py`** — Learning loop orchestrator (backfill + attribution + pattern reports).

### Strategy Persistence (Run-to-Run Memory)

The strategist maintains continuity between sessions via:
- **Strategy identity** — An evolving description of who the system is as a trader, updated by the reflection stage
- **Strategy rules** — Evidence-based rules proposed/retired based on attribution data
- **Strategy memos** — Session-by-session reflection notes (the system's journal)
- **Theses** — Persistent trade ideas with entry/exit triggers, carried forward across sessions
- **Playbook** — Generated actions derived from theses, consumed by the executor
- **Signal attribution** — Historical scores showing which signal types are predictive

### Database Schema
- `news_signals` — Ticker-specific news with category classification
- `macro_signals` — Macro/political news affecting sectors
- `positions` — Current portfolio holdings (synced from Alpaca)
- `decisions` — Trading decisions with reasoning, outcomes, and P&L
- `decision_signals` — FK join table linking decisions to their source signals
- `theses` — Trade ideas with entry/exit triggers and status
- `playbooks` / `playbook_actions` — Structured actions generated by the strategist
- `signal_attribution` — Computed scores for signal type predictiveness
- `strategy_state` — Current trading identity
- `strategy_rules` — Active and retired trading rules
- `strategy_memos` — Session reflection notes
- `account_snapshots` — Daily account value snapshots
- `sessions` / `session_stages` — Session tracking and stage completion
- `supervisor_memos` — Free-form markdown critiques from `python -m v2.supervisor`

**Migration convention:** `db/init/` only runs on a FRESH Postgres volume —
long-lived prod/paper volumes never re-run it. The two directories must stay
mirrored **in both directions**:

- Every new `db/init/NNN_*.sql` needs a `db/migrations/*.sql` mirror, applied
  to every instance with `task db:migrate INSTANCE=<name>` (tracked in
  `schema_migrations`). Skipping it is how prod silently missed the fable-5
  pricing row and the opus repricing (init/036–037) until 2026-06-10.
- Every new `db/migrations/*.sql` needs a `db/init/NNN_*.sql` mirror. Skipping
  it is how `thesis_signals` and `llm_call_contexts` existed only on live
  volumes for three months: CI seeds from `db/init/*` alone, so it was testing
  a schema structurally different from prod, and a DR restore onto a fresh
  volume came up missing tables the strategist writes (audit 2.2).

Each file names its counterpart in a header comment (`-- mirror of
db/init/036`). Two checks enforce this:

- `tests/test_schema_mirror.py` — runs in the normal suite, checks that every
  file declares its counterpart. Opt out with `-- mirror-check: no-init-mirror`
  (nothing for a fresh volume to do) or `-- mirror-check: skip` (cannot be
  replayed at all), always with a reason.
- `db/check_mirror.sh` — runs in CI against a real Postgres. Applies every
  migration on top of an init-seeded scratch database and fails if the schema
  changes. Migrations must therefore be idempotent.

## Commands

```bash
# Start an instance's stack
task up INSTANCE=paper

# Run full daily session (Taskfile target brings the stack up first)
task session INSTANCE=paper

# Skip individual stages (there is no --stage flag; stages are opted OUT of;
# extra args after -- pass through as CLI_ARGS)
task session INSTANCE=paper -- --skip-supervisor
task session INSTANCE=paper -- --skip-pipeline --skip-ideation
task session:dry-run INSTANCE=paper   # also skips ideation/strategy/dashboard

# Re-run a session for a date that already has a session row (any status)
task session INSTANCE=paper -- --force

# Run learning loop standalone
task learn INSTANCE=paper

# Apply pending DB migrations to a long-lived volume (tracked in schema_migrations)
task db:migrate INSTANCE=live
task db:migrate INSTANCE=paper

# Restore a backup into an instance, then re-apply anything newer
task db:restore INSTANCE=live FILE=backups/live-20260921-200000.dump
task db:migrate INSTANCE=live

# Raw compose equivalent (rarely needed — the Taskfile targets above wrap this)
docker compose -p pinchy-paper --env-file instances/paper.env exec trading python -m v2.session

# View public dashboard
# Published via Cloudflare Pages by stage 5, gated on ALGO_DASHBOARD_PUBLISH
```

## Environment Variables

Required in `instances/<name>.env`:
- `ALPACA_API_KEY` — Alpaca API key
- `ALPACA_SECRET_KEY` — Alpaca API secret
- `ALPACA_BASE_URL` — Alpaca REST endpoint (`https://api.alpaca.markets` for live, `https://paper-api.alpaca.markets` for paper)
- `ALPACA_PAPER` — `true` or `false`, must agree with `ALPACA_BASE_URL`. Cross-checked at module load — mismatched values raise immediately to prevent an instance from silently routing orders to the wrong account.
- `ANTHROPIC_API_KEY` — Anthropic API key for Claude
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — Database credentials

Deployment shape, also required (see "Instances" above; read by compose
interpolation as well as the container):
- `INSTANCE` — must match the filename (`instances/live.env` → `INSTANCE=live`)
- `DB_HOST_PORT`, `DASHBOARD_HOST_PORT` — host-bound ports, unique per instance
- `LOGS_DIR` — host directory bind-mounted at `/app/logs`
- `ALGO_DASHBOARD_PUBLISH` — opt-in (`1`/`true`/`yes`; default false) publish
  gate for stage 5. Read at session start, no restart needed. Exactly one
  instance should have this set — it's what makes an instance "the" public
  dashboard. `--skip-dashboard`/`--dry-run` on the CLI still win over it.

Kill switches (see `docs/runbook-recovery.md`, "Halt / Resume"):
- `ALGO_TRADING_HALTED` — set to `1`/`true`/`yes` in an instance's env file and
  every session for that instance becomes a no-op: logs the halt and exits 0
  (a deliberate halt is not a failure). Unlike the knobs below it is read **at
  session start**, but on a long-running stack the env-file edit still needs a
  container recreate (`task up INSTANCE=<name>`) to reach the container — env
  files only apply at container creation. Host-side, no-restart twin: two
  sentinel files, checked by `cron-wrap.sh` and `run-docker.sh` before they
  start containers. A repo-root `HALT` file stops every instance; an
  `instances/<name>.HALT` file stops just that one. **The system is currently
  halted**; the repo `HALT` file explains why, current scope, and how to
  resume.

Optional in-container knobs (read at module import — container restart required after changing):
- `ALGO_EXECUTOR_MODEL` — overrides the executor model. Defaults to `claude-haiku-4-5-20251001`. Set it in one instance's env file to flip that instance's executor independently of the others (e.g. `claude-sonnet-4-6` for the Sonnet pilot on `paper`).
- `ALGO_EXECUTOR_MAX_TOKENS` — overrides the executor `max_tokens` cap. Defaults to `8192` (Haiku 4.5's model max). Raise this knob if executor responses are being truncated.
- `ALGO_DAILY_LOSS_LIMIT_PCT` — daily-loss circuit breaker (default `3.0`). Halts the trading stage when account equity is down more than this % vs the previous close (Alpaca `last_equity`); re-checked after every fill. `<= 0` disables.
- `ALGO_LOOP_COST_CEILING_USD` — per-agentic-loop cost ceiling (default `30`). The strategist/reflection/supervisor loops abort with `stop_reason="cost_ceiling"` once cumulative token cost (priced via `model_pricing`) crosses the cap. `<= 0` disables.

**Host-side** variables in `.env.host` (gitignored; copy `.env.host.example`).
These are read by scripts running on the host, *not* in a container — putting
them in `instances/<name>.env` has no effect, since compose `env_file` only
feeds containers. That exact mismatch is why alerting was documented but never
actually wired up (audit 0.3). They are instance-agnostic — one `.env.host`
covers every instance:
- `ALGO_ALERT_WEBHOOK_URL` — JSON webhook (Slack/Discord/ntfy-style). `cron-wrap.sh` POSTs a short alert when a wrapped job exits nonzero; failures also append to `logs/session_failures.log`.
- `ALGO_HEARTBEAT_URL` — dead-man's switch (healthchecks.io-style). `cron-wrap.sh` pings `<URL>/start` before a job, `<URL>` on success, `<URL>/<exit-status>` on failure. The webhook tells you a job *ran and failed*; only a missing heartbeat catches "the host is off" or "cron stopped firing" — the failure mode that actually kept the system dead for two months. Per-job override: `ALGO_HEARTBEAT_URL_<LABEL>` (label uppercased, `-`→`_`); give each job its own check so one live job can't mask a dead one.
- `ALGO_BACKUP_COPY_DIR` — off-WSL copy target for `task db:backup INSTANCE=<name>`.

**Cron:** every scheduled job goes through `./cron-wrap.sh [--ignore-halt]
[--instance <name>] <label> <command...>`, which owns the HALT check
(both the global `HALT` file and, when `--instance` is passed, that
instance's `instances/<name>.HALT`), heartbeat, failure log, and alert
webhook. Label convention is `<instance>-<job>` (e.g. `paper-session`,
`live-backup`) — per-job heartbeat overrides key off this label. Never add
a bare line to `crontab` — a job outside the wrapper is a job whose failure
nobody hears. Pass `--ignore-halt` for jobs that protect the system rather than
trade with it (the nightly backups): HALT means "do not trade", not "do not
protect the data".

**Audit:** the audit runs as a Claude Code `/loop 24h` session driven by `docs/audit-playbook.md`. It files Jira tickets via the Atlassian MCP (no `JIRA_*` env vars required). See spec `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`.
