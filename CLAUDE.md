# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpaca Learning Platform - an agentic trading system that uses Claude (via Anthropic API) to integrate with the Alpaca trading API, learn from past behavior, and make trading decisions.

**Status:** Active development — `v2/` is the current active codebase.

## Codebase Layout

- **`v2/`** — Current active codebase. All new work goes here.
- **`trading/`** — Legacy v1 module. Mostly sunset; individual pieces are pulled into the v2 pipeline as needed. Do not add new features here.
- **`tests/`** — Test suite covering both v1 and v2.
- **`dashboard/`** — Legacy v1 dashboard (Flask on port 3000). v2 dashboard lives in `v2/dashboard/`.

## Pipelines: Paper vs Prod

The same v2 code runs against two isolated pipelines selected by docker-compose overlay and env file:

| | Prod | Paper |
|---|---|---|
| Compose files | `docker-compose.yml` | `docker-compose.yml` + `docker-compose.paper.yml` |
| Env file | `.env` | `.env.paper` |
| Trading service | `trading` | `trading-paper` |
| Database service | `db` (Postgres `:5432`) | `db-paper` (Postgres `:5433`) |
| Dashboard | `dashboard` (`:3000`) | `dashboard-paper` (`:3001`) |
| Logs | `./logs` | `./logs_paper` |
| Alpaca account | Live account | Paper account |

Paper runs skip social/public-dashboard stages by default (`--skip-twitter --skip-bluesky --skip-dashboard`). Taskfile targets prefixed `paper:*` (e.g. `paper:up`, `paper:session`, `paper:session:dry-run`) exercise the paper pipeline; the unprefixed targets (`session`, `trade`, etc.) run against prod. The two stacks use separate Postgres volumes, so data never crosses between them.

## Project Goals

- Prove whether agentic trading can find an edge
- Claude (Haiku for execution, Sonnet/Opus for ideation & reflection) makes trading decisions
- Daily automated session after market close
- Learning system that journals behavior, computes signal attribution, and reflects on strategy
- Single Alpaca account with an evolving day-to-day strategy
- Public dashboard published to GitHub Pages

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
- **Dashboard:** Published to GitHub Pages

## v2 Daily Session (`v2/session.py`)

The session orchestrator runs stages sequentially. Each stage is independent — failures don't block subsequent stages.

| Stage | Module | Purpose |
|-------|--------|---------|
| 0 | `backfill.py`, `attribution.py` | Learning refresh: backfill decision outcomes, compute signal attribution |
| 1 | `pipeline.py` | News pipeline: fetch from Alpaca, classify with Haiku, store signals |
| 2 | `ideation_claude.py` | Strategist: thesis management + playbook generation (agentic loop with tools) |
| 3 | `trader.py` | Executor: decisions from playbook + order execution |
| 4 | `strategy.py` | Reflection: update strategy identity, rules, and write session memo |
| 5 | `twitter.py` / `bluesky.py` (legacy) or `social_trades.py` (new, gated by `ALGO_ENABLE_TRADE_POSTS=1`) | Social posting |
| 6 | `dashboard_publish.py` | Public dashboard publish |

### Pre-market post stage

Independent of the daily session. Triggered by cron via `task premarket`
(or `python -m v2.premarket` directly). Skipped on weekends and NYSE
holidays. Posts a forward-looking take referencing 1–2 names from
active theses + the latest session memo.

### Live-trade pipeline feature flag

When `ALGO_ENABLE_TRADE_POSTS=1`, Stage 5 runs `run_trade_posts_stage`
instead of the legacy `run_twitter_stage` + `run_bluesky_stage`:

- Iterates today's significant non-hold decisions (notional ≥ `$100`).
- Posts one tweet per decision to Twitter + Bluesky, each linking to
  `/trade/<id>/` and (if present) `/thesis/<id>/` on the public dashboard.
- Caps at 5 posts per session.
- Quiet-day fallback: if no postable decisions, posts a mini-recap on
  trading days only.
- `ALGO_TRADE_POST_DRY_RUN=1` logs generated post bodies and skips both
  platform posts and the DB audit row.

The legacy recap path (twitter.py / bluesky.py orchestrators) stays
intact while the new pipeline is being validated; a follow-up plan will
delete it after one week of clean prod runs.

### Key v2 Modules

- **`agent.py`** — Executor LLM integration. Gets structured trading decisions from Claude Haiku.
- **`claude_client.py`** — Claude API client with tool handling and agentic loop support.
- **`context.py`** — Context builder. Aggregates positions, signals, theses, playbook, and attribution into compressed LLM context.
- **`ideation_claude.py`** — Strategist stage. Agentic loop where Claude manages theses and generates playbooks using database tools.
- **`strategy.py`** — Post-session reflection. Claude reviews outcomes, updates trading identity, proposes/retires rules, writes memos.
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

## Commands

```bash
# Start the stack
docker compose up -d

# Run full daily session
docker compose exec trading python -m v2.session

# Run individual stages
docker compose exec trading python -m v2.session --stage pipeline
docker compose exec trading python -m v2.session --stage ideation
docker compose exec trading python -m v2.session --stage trading --dry-run
docker compose exec trading python -m v2.session --stage strategy

# Run learning loop standalone
docker compose exec trading python -m v2.learn

# View public dashboard
# Published via GitHub Pages by stage 6
```

## Environment Variables

Required in `.env`:
- `ALPACA_API_KEY` — Alpaca API key
- `ALPACA_SECRET_KEY` — Alpaca API secret
- `ALPACA_BASE_URL` — Alpaca REST endpoint (`https://api.alpaca.markets` for live, `https://paper-api.alpaca.markets` for paper)
- `ALPACA_PAPER` — `true` or `false`, must agree with `ALPACA_BASE_URL`. Cross-checked at module load — mismatched values raise immediately to prevent silent paper/prod misrouting.
- `ANTHROPIC_API_KEY` — Anthropic API key for Claude
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — Database credentials
