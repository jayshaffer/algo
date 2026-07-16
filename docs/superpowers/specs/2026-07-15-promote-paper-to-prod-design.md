# Promote the Paper Stack to Primary — Design

**Date:** 2026-07-15
**Status:** Approved (design review in conversation)
**Context:** Live trading is on deliberate hiatus: the owner commented out the
prod cron line, then fully liquidated the prod Alpaca account (positions
closed, cash withdrawn). The 2026-07-15 fresh-eyes audit
(`docs/audits/2026-07-15-fresh-eyes-audit.md`) found the live-account
economics structurally negative at current scale and the learning loop
stalled. Go-forward decision: the **paper pipeline becomes the primary
pipeline** so the learning system keeps accruing samples with zero capital at
risk.

## Decisions made

| Question | Decision |
|---|---|
| What "move paper to prod" means | Promote the paper stack: its DB (learning history), account, and config become the primary daily pipeline |
| Old prod stack + DB | Archive (`pg_dump`) + leave volume dormant; nothing destroyed |
| Public dashboard | Promoted pipeline takes over Cloudflare publishing, **relabeled as paper trading** |
| Implementation shape | Full swap: paper volume/env/config rebadged into the default (unprefixed) stack slots |
| Executor model | Keep the Sonnet pilot (`ALGO_EXECUTOR_MODEL` carried from `.env.paper`) — default taken on approval |
| Archive destination | Off-VHD copy under `/mnt/c/` — default taken on approval |
| Alert webhook | `ALGO_ALERT_WEBHOOK_URL` stubbed in crontab with a TODO until the owner supplies a URL — default taken on approval |

## End state

| | Primary (default stack) |
|---|---|
| Compose | `docker-compose.yml` only |
| Env | `.env` (paper Alpaca keys, `ALPACA_PAPER=true`, paper `POSTGRES_*`, Cloudflare creds, Sonnet-pilot knobs) |
| DB | `db` on `:5432`, backed by the **current paper volume** (`algo_postgres_data_paper`) — no data copy |
| Dashboard | `:3000` locally; stage 5 publishes to Cloudflare Pages, labeled paper |
| Cron | `0 13 * * 1-5` via `run-docker.sh trading python -m v2.session` + nightly `pg_dump` line |
| Logs | `./logs` |
| Old prod volume | `algo_postgres_data` dormant; final dump archived on-repo (gitignored) + off-VHD |
| `docker-compose.paper.yml`, `.env.paper`, `Taskfile` `paper:*` | Retired (removed from active use; recoverable from git history) |

## Section 1 — Archive the retired live-trading history (prerequisite)

1. With the prod `db` container up: `pg_dump -Fc` →
   `backups/prod-live-final-2026-07-15.dump` (add `backups/` to `.gitignore`).
2. Copy the dump off the WSL VHD to `/mnt/c/` (exact folder chosen at
   implementation time; recorded in the runbook note).
3. `docker compose down` the old prod stack. Volume `algo_postgres_data`
   remains, dormant. Recovery = remount or `pg_restore`.
4. Preserve the old live `.env` as `.env.live-retired` (gitignored) so the
   live Alpaca/Cloudflare credentials are not lost in the swap.

## Section 2 — The swap (compose + env)

1. `docker-compose.yml` `db` service re-points its volume to
   `algo_postgres_data_paper`. No data copy; the paper history becomes the
   primary DB at `:5432`.
2. New `.env` = paper Alpaca keys + paper `ALPACA_BASE_URL`
   (`https://paper-api.alpaca.markets`) + `ALPACA_PAPER=true` + the paper
   volume's `POSTGRES_*` credentials (must match what the volume was
   initialized with) + Cloudflare creds carried from the old `.env` +
   `ANTHROPIC_API_KEY` + `ALGO_EXECUTOR_MODEL` (Sonnet pilot) and any other
   knobs carried from `.env.paper`.
3. Retire `docker-compose.paper.yml` and `.env.paper` (delete from working
   tree; git history keeps them). Remove `paper:*` Taskfile targets;
   unprefixed targets now mean the promoted pipeline.
4. **Migration check is part of the swap:** verify `schema_migrations` on the
   paper volume against `ls db/migrations` and run `task db:migrate`. (The
   audit proved mirrors don't apply themselves; the paper volume has its own
   drift history.)

## Section 3 — Dashboard

1. Stage 5 (`dashboard_publish`) runs again — drop the paper-default
   `--skip-dashboard` behavior for the promoted pipeline.
2. Relabel the public dashboard as **paper trading**: banner/title plus any
   copy implying real money. Scope is relabeling only.
3. Out of scope here: deposit-adjusted return math (audit finding 2.6).

## Section 4 — Cron + ops

1. Rewrite the repo `crontab` to match intended reality: one daily session
   line for the promoted pipeline via `run-docker.sh` (so failures hit
   `session_failures.log` and the webhook path), paper lines removed.
   Install it (`crontab crontab`) and commit — ending installed-vs-repo
   drift.
2. Bundled one-liner audit fixes (same file, same edit):
   - Nightly `pg_dump` cron line for the new primary DB (audit 3.1).
   - `ALGO_ALERT_WEBHOOK_URL=` defined in the crontab so `run-docker.sh`
     alerting can actually fire (audit 3.2) — stubbed with a TODO until a
     webhook URL exists.

## Section 5 — Docs + verification

1. CLAUDE.md: rewrite the Pipelines section (single pipeline on the paper
   account; remove the two-stack table), update env-var docs, add a note
   about the retired live stack and where its archive lives.
2. Verification plan:
   - Stack up → DB connectivity, `schema_migrations` current, row counts
     sanity-match the paper DB's history.
   - A real watched session on the next market day.
   - Deliberately **no `--dry-run` smoke test** — audit A.11: dry-run seeds
     phantom decision rows into the learning DB.
3. Tests: suite is fully mocked; no test changes expected beyond any that
   assert on paper-specific Taskfile/compose plumbing.

## Out of scope

- Tier 1 money-path fixes and Tier 2 learning-gradient redesign from the
  audit (separate efforts; Tier 1 still applies to paper because it exercises
  the same code).
- Dashboard deposit-adjustment (audit 2.6).
- Dead-man's switch / external health-check service (audit 3.2 residual).
- Any future second experimental tier (the retired paper overlay can be
  revived if that ever happens).

## Risks / notes

- `POSTGRES_*` mismatch between new `.env` and the paper volume's initialized
  credentials would break everything at once — verified first in the swap.
- `ALPACA_PAPER=true` in `.env` inverts the historical assumption
  (".env = live account"); CLAUDE.md must be unambiguous. The module-load
  cross-check (`v2/executor.py`) passes by design since URL and flag agree.
- Old prod volume and the new primary volume coexist under compose-project
  volume names; the dormant one must not be referenced by the active compose
  file.
- If live trading ever resumes, that's a new design (new account, new env,
  and the audit's Tier 1 fixes as a precondition).
