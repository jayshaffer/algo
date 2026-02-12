# V2 — Alpaca Learning Platform (Restructured)

Isolated rebuild of the trading platform with a closed learning loop, structured data flow, and Claude-only LLM stack.

## What changed from V1

| Area | V1 (`trading/`) | V2 (`v2/`) |
|------|-----------------|------------|
| LLM | Ollama (qwen2.5:14b) + Claude | Claude only (Haiku + Opus) |
| News classification | Ollama + embedding filter | Haiku batch classifier (batch size 50) |
| Strategist | Claude Opus agentic loop | Same, but with attribution constraints injected |
| Executor input | Prose string context | Structured `ExecutorInput` dataclass |
| Playbook | Text blob | `playbook_actions` table with FK traceability |
| Decision linkage | None | `decision_signals` FK ties decisions to signals |
| Learning loop | Manual backfill + attribution | Stage 0 auto-refresh, constraints fed to strategist |
| Session stages | 3 (pipeline, strategist, executor) | 4 (learning refresh, pipeline, strategist, executor) |
| DB layer | Single `db.py` | Split into `database/connection.py`, `trading_db.py`, `dashboard_db.py` |

## Architecture

```
Session (4 stages)
│
├── Stage 0: Learning Refresh
│   ├── backfill.py         — Fill 7d/30d outcomes for past decisions
│   ├── attribution.py      — Compute signal win rates via decision_signals FK
│   └── build_attribution_constraints() → constraint string
│
├── Stage 1: News Pipeline
│   ├── news.py             — Fetch from Alpaca News API
│   ├── classifier.py       — Haiku batch classification
│   └── pipeline.py         — Orchestrate fetch → classify → store
│
├── Stage 2: Strategist (Claude Opus)
│   ├── ideation_claude.py  — Agentic loop with tool use
│   ├── tools.py            — 11 tools (thesis CRUD, market data, playbook)
│   └── attribution_constraints injected into system prompt
│
└── Stage 3: Executor (Claude Haiku)
    ├── agent.py            — Structured ExecutorInput → ExecutorDecision
    ├── context.py          — Build ExecutorInput from DB state
    ├── trader.py           — Validate, execute, log with playbook_action_id
    └── executor.py         — Alpaca order execution
```

## Module index

| Module | Purpose |
|--------|---------|
| `session.py` | 4-stage session orchestrator, CLI entry point |
| `pipeline.py` | News fetch → classify → store |
| `classifier.py` | Haiku news classifier (single, batch, ticker-specific) |
| `ideation_claude.py` | Opus strategist agentic loop with tool use |
| `tools.py` | 11 tool definitions + handlers for strategist |
| `agent.py` | Executor LLM: `ExecutorInput` → `ExecutorDecision` |
| `context.py` | Prose context builders + `build_executor_input()` |
| `trader.py` | Trading session: validate, execute, log decisions |
| `attribution.py` | Signal attribution + constraint generation |
| `patterns.py` | Pattern analysis (signal categories, sentiment, tickers) |
| `backfill.py` | Outcome backfill (7d/30d P&L) |
| `learn.py` | Learning loop orchestrator (backfill + patterns + attribution) |
| `executor.py` | Alpaca API: orders, positions, account |
| `market_data.py` | Market snapshot, sectors, movers |
| `news.py` | Alpaca News API client |
| `claude_client.py` | Claude API client with retry + agentic loop |
| `log_config.py` | Logging setup |
| `database/connection.py` | `get_cursor()` context manager |
| `database/trading_db.py` | All trading CRUD operations |
| `database/dashboard_db.py` | Read-only dashboard queries |
| `dashboard/app.py` | Flask web dashboard |

## Schema migration

`db/init/006_v3.sql` adds:

- `playbook_actions` — structured actions with ticker, action, thesis_id, priority
- `decisions.playbook_action_id` — FK linking decisions to playbook actions
- `decisions.is_off_playbook` — flag for off-playbook trades
- Index on `decision_signals(decision_id)`

## Commands

```bash
# Run full daily session
python -m v2.session --dry-run

# Run individual stages
python -m v2.pipeline --hours 24 --limit 300
python -m v2.ideation_claude
python -m v2.trader --dry-run
python -m v2.learn --days 60

# Dashboard
python -m flask --app v2.dashboard.app run --port 3000
```

## Tests

```bash
# Run all 254 tests
python3 -m pytest tests/v2/ -v

# With coverage
python3 -m pytest tests/v2/ --cov=v2 --cov-report=term-missing
```

All external dependencies (Alpaca, Claude, PostgreSQL) are mocked. Tests use `mock_db` and `mock_cursor` fixtures from `tests/v2/conftest.py`.

## Environment variables

Required in `.env`:

```
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
ANTHROPIC_API_KEY=...
DATABASE_URL=postgresql://user:pass@localhost:5432/algo
```
