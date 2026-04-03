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
| 5 | `twitter.py`, `bluesky.py` | Social posting |
| 6 | `dashboard_publish.py` | Public dashboard publish |

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
- `APCA_API_KEY_ID` — Alpaca API key
- `APCA_API_SECRET_KEY` — Alpaca API secret
- `ANTHROPIC_API_KEY` — Anthropic API key for Claude
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` — Database credentials
