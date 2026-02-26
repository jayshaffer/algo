# Taskfile.yml Design

**Date:** 2026-02-20
**Status:** Approved

## Goal

Add a `Taskfile.yml` (go-task) to wrap common project workflows into discoverable, repeatable commands.

## Structure

Single `Taskfile.yml` at repo root. Tasks organized by namespace prefix.

## Tasks

### Infrastructure

| Task | Command | Notes |
|------|---------|-------|
| `docker:up` | `docker compose up -d` | Status check: no-op if already running |
| `docker:down` | `docker compose down` | Stop all services |
| `docker:build` | `docker compose build` | Rebuild images |
| `docker:logs` | `docker compose logs -f {{.CLI_ARGS}}` | Follow logs, optional service filter |
| `setup:ollama` | `scripts/setup-ollama.sh` | Pull models after first start |

### Trading Workflows

All run via `docker compose exec trading python -m trading.<module>`.
All depend on `docker:up`. All pass `{{.CLI_ARGS}}` for extra flags.

| Task | Module | Description |
|------|--------|-------------|
| `session` | `trading.session` | Full consolidated daily session |
| `session:dry-run` | `trading.session --dry-run` | Dry run of full session |
| `trade` | `trading.trader` | Run trader only |
| `trade:dry-run` | `trading.trader --dry-run` | Dry run trader |
| `ideation` | `trading.ideation` | Ollama ideation |
| `ideation:claude` | `trading.ideation_claude` | Claude strategist ideation |
| `pipeline` | `trading.pipeline` | News processing pipeline |
| `learn` | `trading.learn` | Learning loop |
| `backfill` | `trading.backfill` | Backfill decision outcomes |

### Dev

| Task | Command | Notes |
|------|---------|-------|
| `test` | `python3 -m pytest tests/` | Run locally |
| `test:coverage` | `python3 -m pytest tests/ --cov=trading --cov=dashboard` | With coverage |

## Design Decisions

- **Preconditions**: Trading tasks use `deps: [docker:up]` with status check so stack auto-starts but is a no-op if running.
- **CLI_ARGS passthrough**: `task trade -- --model X` works.
- **No default task**: Prevents accidental execution.
- **Tests run locally**: Not in Docker, matching existing workflow.
