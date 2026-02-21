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

```cron
# Backfill outcomes (6 AM ET, Mon-Fri)
0 8 * * 1-5 /path/to/algo/run-docker.sh trading python -m trading.backfill

# Consolidated daily session (3 PM ET, Mon-Fri)
0 19 * * 1-5 /path/to/algo/run-docker.sh trading python -m trading.session

# Weekly learning loop (7 AM ET Sunday)
0 7 * * 0 /path/to/algo/run-docker.sh trading python -m trading.learn --days 60
```

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

## License

MIT
