# Alpaca Learning Platform

An autonomous trading system that uses LLMs to research markets, generate trade theses, make trading decisions, and learn from outcomes.

## Overview

This system connects to Alpaca for trade execution and uses two LLMs with distinct roles:

- **Claude Opus** (strategist) - researches markets via web search and tools, manages theses, writes a daily playbook
- **Qwen3 14B** (executor) - makes buy/sell/hold decisions based on the playbook and market context
- **Qwen3 14B** (classifier) - categorizes news as ticker-specific or macro signals
- **Nomic Embed** (filter) - filters irrelevant news before classification

Claude runs via API. Qwen3 and Nomic run locally on your GPU via Ollama.

## Features

- **Consolidated daily session** - Single cron job runs news pipeline, strategist, and executor
- **Claude strategist** - Agentic loop with 11 tools including web search, thesis management, and attribution analysis
- **Playbook system** - Strategist writes a daily plan (market outlook, priority actions, watch list) for the executor
- **Signal attribution** - Tracks which signal types (news categories, macro events, theses) are predictive
- **Ideation system** - Generates trade theses with entry/exit triggers, reviews and invalidates stale ideas
- **Learning system** - Journals every decision with reasoning, backfills 7/30-day outcomes, analyzes patterns
- **News pipeline** - Fetches, filters, and classifies market news into actionable signals
- **Web dashboard** - Portfolio, playbook, signals, theses, attribution, decision history, performance charts
- **Paper trading** - Test with Alpaca paper trading before going live

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                          │
├──────────────┬──────────────┬───────────────┬───────────────┤
│   Ollama     │  PostgreSQL  │    Trading    │   Dashboard   │
│  (GPU LLM)   │   (Data)     │   (Agent)     │   (Flask)     │
│  :11434      │   :5432      │               │   :3000       │
└──────────────┴──────────────┴───────────────┴───────────────┘
```

### Data Flow

```
News API → Filter (embeddings) → Classify (Qwen3) → Store signals
                                                          ↓
Claude strategist (web search + tools) → Manage theses → Write playbook
                                                          ↓
Executor (Qwen3) → Read playbook + context → Make decisions → Execute trades → Log
                                                          ↓
                    Backfill outcomes → Signal attribution → Feed back to strategist
```

## Quick Start

### Prerequisites

- Docker with GPU support (nvidia-container-toolkit)
- Alpaca account (paper trading recommended)
- Anthropic API key (for Claude strategist)
- NVIDIA GPU with 16GB+ VRAM (RTX 5070 Ti or similar)

### Setup

1. **Clone and configure**
   ```bash
   git clone <repo>
   cd algo
   cp .env.example .env
   # Edit .env with your Alpaca and Anthropic credentials
   ```

2. **Start services**
   ```bash
   docker compose up -d
   ```

3. **Pull required models**
   ```bash
   docker compose exec ollama ollama pull qwen3:14b
   docker compose exec ollama ollama pull nomic-embed-text
   ```

4. **Verify setup**
   ```bash
   docker compose exec trading python main.py
   ```

5. **Run first session (dry run)**
   ```bash
   docker compose exec trading python -m trading.session --dry-run
   ```

## Usage

### Consolidated Daily Session

The recommended way to run everything — processes news, runs the Claude strategist, then executes trades:

```bash
# Full session (news → strategist → executor)
docker compose exec trading python -m trading.session

# Skip news pipeline
docker compose exec trading python -m trading.session --skip-pipeline

# Skip strategist (just news + executor)
docker compose exec trading python -m trading.session --skip-ideation
```

### Claude Strategist

Run the Claude-powered research and planning session independently:

```bash
# Strategist session (backfill → attribution → research → playbook)
docker compose exec trading python -m trading.ideation_claude
```

The strategist:
- Uses web search and market data tools to research opportunities
- Reviews and manages active theses (create, update, close)
- Analyzes signal attribution to understand what's been predictive
- Writes a daily playbook with market outlook and priority actions

### Trading Session

Run the Qwen3 executor independently:

```bash
# Dry run (no real trades)
docker compose exec trading python -m trading.trader --dry-run

# Live trading
docker compose exec trading python -m trading.trader
```

The executor:
- Reads today's playbook for priorities and risk guidance
- Receives active theses as part of context
- Acts on thesis entry triggers when conditions are met
- Links decisions to motivating signals for attribution tracking

### News Pipeline

Fetch and classify recent news:

```bash
# Process last 24 hours of news
docker compose exec trading python -m trading.pipeline

# Custom options
docker compose exec trading python -m trading.pipeline --hours 48 --limit 100 --threshold 0.25
```

### Learning Loop

```bash
# Backfill decision outcomes
docker compose exec trading python -m trading.backfill

# Full learning loop (backfill + patterns + attribution)
docker compose exec trading python -m trading.learn

# Patterns only
docker compose exec trading python -m trading.learn --patterns-only
```

### Entertainment Tweets

Fire off entertaining Mr. Krabs tweets based on live market news and trends, independent of the daily session:

```bash
# Generate and post entertainment tweets
docker compose exec trading python -m v2.entertainment

# Custom options
docker compose exec trading python -m v2.entertainment --news-hours 12 --news-limit 10
```

This pulls current market headlines and movers, then generates tweets in the Bikini Bottom Capital voice — referencing real tickers, real moves, and SpongeBob universe characters. Tweets are posted automatically and logged to the DB with `tweet_type="entertainment"`.

### Live-Trade Social Pipeline

Replaces the single bare daily recap with one tweet per significant decision, each linking to its `/trade/<id>/` page on the public dashboard. Gated behind a feature flag — opt in by setting `ALGO_ENABLE_TRADE_POSTS=1` in `.env`. When the flag is off, the original recap path runs unchanged.

**What it does:**

- Iterates today's non-hold decisions with notional value ≥ $100, capped at 5 posts per session
- Generates one Haiku-authored tweet per decision (Bikini Bottom Capital voice, casual + direct)
- Posts to Twitter and Bluesky with deterministic `/trade/<id>/` and `/thesis/<id>/` URL appends — the LLM never builds URLs, eliminating broken-link risk
- Per-decision rerun guard: a cron retry won't double-post the same decision
- Per-decision error isolation: one bad LLM response doesn't drop the rest of the burst
- Quiet-day fallback: on trading days with no postable decisions, posts the existing Mr. Krabs-style recap so the account doesn't go dark. Weekends and NYSE holidays produce no post

**Enable in prod:**

```bash
# Add to .env
ALGO_ENABLE_TRADE_POSTS=1

# Restart trading service so the flag is picked up
docker compose up -d trading

# Next session run uses the new pipeline
docker compose exec trading python -m v2.session
```

**Dry-run mode (no real posts):**

```bash
# Logs generated post bodies and skips both platform posts and the DB audit row
docker compose exec -e ALGO_TRADE_POST_DRY_RUN=1 -e ALGO_ENABLE_TRADE_POSTS=1 \
    trading python -m v2.session --skip-dashboard
```

`ALGO_TRADE_POST_DRY_RUN=1` honors are also respected by `python -m v2.premarket` below.

**Smoke-test in paper first:**

```bash
docker compose exec -e ALGO_ENABLE_TRADE_POSTS=1 -e ALGO_TRADE_POST_DRY_RUN=1 \
    trading-paper python -m v2.session --skip-dashboard
```

### Pre-Market Post

A separate, cron-triggered post that runs **before** the daily session (typically ~07:30 ET on weekdays). Forward-looking voice referencing 1–2 names from active theses + the latest strategy memo. Skipped on weekends and NYSE holidays. Idempotent on cron retries via `posted_tweet_exists(today, "premarket", platform)`.

```bash
# Run once (Taskfile)
task premarket

# Or directly
docker compose exec trading python -m v2.premarket

# Dry run
docker compose exec -e ALGO_TRADE_POST_DRY_RUN=1 trading python -m v2.premarket
```

### Weekly Mistakes & Attribution Posts

Two posts that run every Friday afternoon, independent of the daily session:

```bash
# Friday 14:00 MST — "what didn't work" + link to /mistakes/
task weekly:mistakes
# or directly:
docker compose exec trading python -m v2.social_weekly mistakes

# Friday 14:15 MST — attribution roundup + link to /attribution/
task weekly:attribution
docker compose exec trading python -m v2.social_weekly attribution
```

Both self-skip on weekends and NYSE holidays. Both honor
`ALGO_TRADE_POST_DRY_RUN=1` for log-only runs that don't post or write to the
DB. Both skip with a non-error log when the underlying data is empty.

The cron entries live in the repo `crontab` file. After editing the file in
the repo, install it on the host with:

```bash
crontab /home/jay/dev/algo/crontab
```

### Dashboard

Access at http://localhost:3000

- `/` - Portfolio overview and positions
- `/playbook` - Today's playbook from the Claude strategist
- `/signals` - Recent ticker and macro signals
- `/theses` - Active trade theses with filtering
- `/attribution` - Signal attribution scores
- `/decisions` - Trading decision history with reasoning
- `/performance` - Equity curve and performance metrics

### Public Dashboard

The system can publish a read-only snapshot of portfolio data to a GitHub Pages site, updated daily as part of the session pipeline.

**What gets published:**

| File | Contents |
|------|----------|
| `data/summary.json` | Portfolio value, daily P&L, total return, inception date |
| `data/snapshots.json` | 90-day equity curve history |
| `data/positions.json` | Current holdings with cost basis |
| `data/decisions.json` | Last 30 days of trades with reasoning and outcomes |
| `data/theses.json` | Active trade theses with entry/exit triggers |

**Setup:**

1. Create a GitHub Pages repository (e.g. `your-org.github.io`) with your static frontend (`index.html`, `app.js`, etc.)
2. Clone it locally on the machine running the trading stack
3. Set environment variables in `.env`:
   ```bash
   DASHBOARD_REPO_PATH=/path/to/your-org.github.io
   DASHBOARD_URL=https://your-org.github.io
   ```

**How it works:**

Publishing runs as Stage 6 of the daily session (`v2/session.py`). It gathers data from the database, writes JSON files to the cloned repo, and pushes to GitHub. If `DASHBOARD_REPO_PATH` is not set, the stage is skipped.

The `DASHBOARD_URL` is appended to social media posts (Twitter and Bluesky) when configured, linking followers to the live dashboard.

**Run manually:**

```bash
docker compose exec trading python -c "from v2.dashboard_publish import run_dashboard_stage; run_dashboard_stage()"
```

## Configuration

### Environment Variables

```bash
# Alpaca API
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets

# Anthropic API (for Claude strategist)
ANTHROPIC_API_KEY=your_key

# Database
POSTGRES_USER=algo
POSTGRES_PASSWORD=algo
POSTGRES_DB=trading

# Ollama
OLLAMA_URL=http://ollama:11434

# Twitter/X (optional — for Bikini Bottom Capital tweets)
TWITTER_API_KEY=your_key
TWITTER_API_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_TOKEN_SECRET=your_token_secret

# Public Dashboard (optional — publish to GitHub Pages)
DASHBOARD_REPO_PATH=/path/to/your-org.github.io
DASHBOARD_URL=https://your-org.github.io

# Live-trade social pipeline (optional)
# When set to "1", Stage 5 of the daily session posts one tweet per
# significant decision instead of the legacy single recap. Off by default.
ALGO_ENABLE_TRADE_POSTS=0
# When set to "1", any social-post stage (live-trade, premarket, quiet-day
# recap fallback) logs generated bodies and skips both the platform post
# and the DB audit row. Useful for end-to-end smoke tests.
ALGO_TRADE_POST_DRY_RUN=0
```

### Model Options

| Model | VRAM | Use Case |
|-------|------|----------|
| `qwen3:8b` | ~5GB | Limited VRAM (executor/classifier) |
| `qwen3:14b` | ~10GB | Recommended (all local tasks) |
| `qwen3:32b` | ~20GB | Better reasoning (24GB+ VRAM) |

## Database Schema

| Table | Purpose |
|-------|---------|
| `news_signals` | Ticker-specific signals (earnings, analyst ratings, etc.) |
| `macro_signals` | Economic/political signals (Fed, trade policy, etc.) |
| `theses` | Trade ideas with entry/exit triggers and status |
| `playbooks` | Daily trading plans from Claude strategist |
| `positions` | Current portfolio holdings |
| `account_snapshots` | Daily equity curve |
| `decisions` | Trading journal with reasoning and outcomes |
| `decision_signals` | Links decisions to motivating signals for attribution |
| `signal_attribution` | Precomputed scores for which signal types are predictive |

## Automation

Example crontab (`crontab -e` or `crontab /path/to/algo/crontab`):

The repo ships a working crontab at [`crontab`](./crontab) — install with:

```bash
crontab /home/jay/dev/algo/crontab
```

Times are **MST** (America/Denver, UTC-7). Adjust the hour fields if your server runs in a different timezone. The defaults:

```cron
# Pre-market social post (5:30 AM MST / 7:30 AM ET, Mon-Fri)
30 5 * * 1-5 cd /home/jay/dev/algo && (task premarket ; task docker:stop:session)

# Daily session (1 PM MST / 3 PM ET, Mon-Fri) — runs all 7 stages
0 13 * * 1-5 cd /home/jay/dev/algo && (task session ; task docker:stop:session)

# Weekly deep learning analysis (5 AM MST / 7 AM ET, Sunday)
0 5 * * 0 cd /home/jay/dev/algo && (task learn -- --days 60 ; task docker:stop:session)
```

The pre-market entry self-skips on weekends and NYSE holidays, so a fixed weekday cron is correct. Stage 5 of the daily session honors `ALGO_ENABLE_TRADE_POSTS` from `.env` — flip it to `1` to switch from the legacy recap to the live-trade pipeline.

## Project Structure

```
algo/
├── trading/
│   ├── session.py        # Consolidated daily orchestrator
│   ├── ideation_claude.py # Claude strategist (research + theses + playbook)
│   ├── claude_client.py  # Claude API client with agentic loop
│   ├── tools.py          # Tool definitions for Claude strategist
│   ├── trader.py         # Trading session executor (Qwen3)
│   ├── agent.py          # Qwen3 integration for trade decisions
│   ├── context.py        # Context builder for executor
│   ├── executor.py       # Alpaca trade execution
│   ├── ideation.py       # Ollama-based thesis generation (legacy)
│   ├── pipeline.py       # News pipeline orchestrator
│   ├── news.py           # News fetching from Alpaca
│   ├── filter.py         # Relevance filtering (embeddings)
│   ├── classifier.py     # News classification (Qwen3)
│   ├── attribution.py    # Signal attribution engine
│   ├── backfill.py       # Outcome measurement (7d/30d P&L)
│   ├── patterns.py       # Pattern analysis
│   ├── learn.py          # Learning loop (backfill + patterns + attribution)
│   ├── market_data.py    # Market snapshot for ideation
│   ├── db.py             # Database operations
│   ├── ollama.py         # Ollama utilities
│   └── log_config.py     # Logging configuration
├── dashboard/
│   ├── app.py            # Flask application
│   ├── queries.py        # Dashboard queries
│   └── templates/        # Jinja2 templates
├── db/
│   └── init/
│       ├── 001_schema.sql
│       ├── 002_theses.sql
│       └── 005_redesign.sql
├── v2/
│   ├── session.py          # V2 consolidated session (6 stages)
│   ├── dashboard_publish.py # Public dashboard publisher (GitHub Pages)
│   ├── twitter.py          # Twitter posting
│   ├── bluesky.py          # Bluesky posting
│   └── entertainment.py    # Entertainment tweet generation
├── docker-compose.yml
├── .env.example
└── README.md
```

## How It Learns

1. **Research** - Claude strategist uses web search and market tools to research opportunities
2. **Thesis management** - Creates theses with entry/exit triggers, reviews and closes stale ideas
3. **Playbook** - Strategist writes a daily plan with priorities, watch list, and risk notes
4. **Execution** - Qwen3 executor acts on playbook and theses, linking decisions to motivating signals
5. **Outcome tracking** - 7-day and 30-day P&L backfilled from price data
6. **Signal attribution** - Computes which signal categories (news types, macro events, theses) are predictive
7. **Feedback loop** - Attribution scores are fed back to the strategist for next session

## Testing

```bash
# Run unit tests (no external services needed)
python3 -m pytest tests/

# Run with coverage
python3 -m pytest tests/ --cov=trading --cov=dashboard

# Run model integration tests (requires running Ollama with qwen3:14b)
python3 -m pytest tests/test_model_integration.py -m integration -v
```

Integration tests send real prompts to Ollama and validate that qwen3:14b returns correctly structured JSON for every prompt template (classification, trading decisions, ideation). They are skipped by default in normal test runs.

### Running Integration Tests in Docker

```bash
# Model integration tests (requires Ollama with qwen3:14b)
docker compose exec trading python3 -m pytest tests/test_model_integration.py -m integration -v

# Twitter integration tests (requires Ollama + Twitter API credentials in .env)
docker compose exec trading python3 -m pytest tests/test_twitter_integration.py -m integration -v

# All integration tests
docker compose exec trading python3 -m pytest tests/ -m integration -v
```

## Development

```bash
# View logs
docker compose logs -f trading

# Enter trading container
docker compose exec trading bash

# Reset database
docker compose down -v
docker compose up -d

# Check Ollama models
docker compose exec ollama ollama list
```

## Self-healing audit

A daily auditor (`v2/audit.py`) detects metadata integrity issues, proposes
rule-overfitting / contradictions via a single Haiku call, and surfaces
application-level regressions. Findings appear at
[http://127.0.0.1:3000/audit](http://127.0.0.1:3000/audit). They never feed
the strategist.

### Manual run

```bash
task audit              # propose-only (recommended default)
task audit:apply        # apply Tier-1 auto-fixes (orphan FK delete, backfill re-run)
task paper:audit        # same against paper stack
```

### Daily cron

Runs propose-only. Add to host crontab (`crontab -e`):

```
MAILTO=you@example.com
30 22 * * * cd /home/jay/dev/algo && /usr/local/bin/task audit >> logs/cron-audit.log 2>&1
```

The auditor exits 0 (clean), 1 (>=1 critical finding open — MAILTO fires), or
2 (run itself failed — MAILTO fires). Advisory-lock contention exits 0.

### Enabling auto-fix in cron

Once you've reviewed propose-only runs for a week and trust the auto-fix
behavior, replace `task audit` with `task audit:apply` in the cron entry.

### CLI options

```
python -m v2.audit                    # propose-only
python -m v2.audit --apply            # apply Tier-1 auto-fixes
python -m v2.audit --max-auto-fix N   # override ceiling (default 100)
```

## License

MIT
