# Alpaca Learning Platform

An autonomous trading system that uses local LLMs to make trading decisions, learns from outcomes, and evolves its strategy over time.

## Overview

This system connects to Alpaca for trade execution and uses LLMs for:
- **Trading decisions** (Claude) - analyzes market context and decides buy/sell/hold
- **Ideation** (Qwen3 14B) - generates and manages trade theses based on market data
- **News classification** (Qwen3 14B) - categorizes news as ticker-specific or macro signals
- **Relevance filtering** (Nomic Embed) - filters irrelevant news before processing

Local inference runs on your GPU via Ollama. Trading decisions use Claude API.

## Features

- **Autonomous trading** - Daily automation via cron after market close
- **Ideation system** - LLM generates trade theses with entry/exit triggers, reviews and invalidates stale ideas
- **Learning system** - Journals every decision with reasoning, tracks 7/30-day outcomes
- **Strategy evolution** - Analyzes patterns and adjusts watchlist, risk tolerance, focus sectors
- **News pipeline** - Fetches, filters, and classifies market news into actionable signals
- **Web dashboard** - Portfolio view, signals, decision history, performance charts
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
Market Data → Ideation (Qwen) → Generate/review theses → Store
                                                          ↓
Cron trigger → Build context (signals + theses) → Claude decision → Execute trade → Log
                                                          ↓
                              Backfill outcomes → Pattern analysis → Evolve strategy
```

## Quick Start

### Prerequisites

- Docker with GPU support (nvidia-container-toolkit)
- Alpaca account (paper trading recommended)
- Anthropic API key (for Claude)
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

5. **Run first trading session (dry run)**
   ```bash
   docker compose exec trading python -m trading.trader --dry-run
   ```

## Usage

### Ideation Session

Run before market open to generate/review trade theses:

```bash
# Generate and review theses
docker compose exec trading python -m trading.ideation

# Custom model
docker compose exec trading python -m trading.ideation --model qwen3:32b
```

The ideation system:
- Reviews existing active theses (keep, update, invalidate, expire)
- Generates 3-5 new trade ideas based on market data
- Each thesis includes entry/exit triggers and invalidation criteria

### Trading Session

Run daily after market close (4:30 PM ET recommended):

```bash
# Dry run (no real trades)
docker compose exec trading python -m trading.trader --dry-run

# Live trading
docker compose exec trading python -m trading.trader

# Custom model
docker compose exec trading python -m trading.trader --model claude-opus-4-20250514
```

The trading agent:
- Receives active theses as part of context
- May act on thesis if entry trigger conditions are met
- Flags theses for invalidation if conditions observed
- Marks theses as executed when traded

### News Pipeline

Fetch and classify recent news:

```bash
# Process last 24 hours of news
docker compose exec trading python pipeline.py

# Custom options
docker compose exec trading python pipeline.py --hours 48 --limit 100 --threshold 0.25
```

### Learning Loop

```bash
# Backfill decision outcomes
docker compose exec trading python backfill.py

# Analyze patterns and evolve strategy
docker compose exec trading python strategy.py --days 60

# Full learning loop
docker compose exec trading python learn.py
```

### Dashboard

Access at http://localhost:3000

- `/` - Portfolio overview and positions
- `/signals` - Recent ticker and macro signals
- `/decisions` - Trading decision history with reasoning
- `/performance` - Equity curve and performance metrics

## Configuration

### Environment Variables

```bash
# Alpaca API
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets

# Anthropic API (for trading decisions)
ANTHROPIC_API_KEY=your_key

# Database
DATABASE_URL=postgresql://algo:algo@db:5432/trading

# Ollama
OLLAMA_URL=http://ollama:11434
```

### Model Options

| Model | VRAM | Use Case |
|-------|------|----------|
| `qwen3:8b` | ~5GB | Limited VRAM (ideation) |
| `qwen3:14b` | ~10GB | Recommended (all tasks) |
| `qwen3:32b` | ~20GB | Better reasoning (24GB+ VRAM) |

## Database Schema

| Table | Purpose |
|-------|---------|
| `news_signals` | Ticker-specific signals (earnings, analyst ratings, etc.) |
| `macro_signals` | Economic/political signals (Fed, trade policy, etc.) |
| `theses` | Trade ideas with entry/exit triggers and status |
| `positions` | Current portfolio holdings |
| `account_snapshots` | Daily equity curve |
| `decisions` | Trading journal with reasoning and outcomes |
| `strategy` | Current strategy state and evolution history |

## Automation

Example crontab for daily automation:

```cron
# Ideation (before market open)
0 7 * * 1-5 cd /path/to/algo && docker compose exec -T trading python -m trading.ideation

# News pipeline (every 6 hours during market hours)
0 9,12,15 * * 1-5 cd /path/to/algo && docker compose exec -T trading python pipeline.py

# Trading session (30 min after market close)
30 16 * * 1-5 cd /path/to/algo && docker compose exec -T trading python -m trading.trader

# Backfill outcomes (morning before market)
0 6 * * 1-5 cd /path/to/algo && docker compose exec -T trading python backfill.py

# Strategy evolution (weekly)
0 7 * * 0 cd /path/to/algo && docker compose exec -T trading python strategy.py
```

## Project Structure

```
algo/
├── trading/
│   ├── agent.py       # Claude integration for decisions
│   ├── trader.py      # Trading session orchestrator
│   ├── ideation.py    # Thesis generation and review
│   ├── market_data.py # Market snapshot for ideation
│   ├── context.py     # Context builder for LLM
│   ├── executor.py    # Alpaca trade execution
│   ├── pipeline.py    # News pipeline orchestrator
│   ├── news.py        # News fetching
│   ├── filter.py      # Relevance filtering
│   ├── classifier.py  # News classification
│   ├── backfill.py    # Outcome measurement
│   ├── patterns.py    # Pattern analysis
│   ├── strategy.py    # Strategy evolution
│   ├── learn.py       # Learning loop
│   ├── db.py          # Database operations
│   └── ollama.py      # Ollama utilities
├── dashboard/
│   ├── app.py         # Flask application
│   ├── queries.py     # Dashboard queries
│   └── templates/     # Jinja2 templates
├── db/
│   └── init/
│       ├── 001_schema.sql
│       └── 002_theses.sql
├── docker-compose.yml
├── .env.example
└── README.md
```

## How It Learns

1. **Ideation** - LLM generates trade theses with entry/exit triggers based on market data
2. **Journaling** - Every decision stored with full reasoning and signals used
3. **Outcome tracking** - 7-day and 30-day P&L backfilled from price data
4. **Pattern analysis** - Identifies which signals, sentiments, and tickers perform best
5. **Strategy evolution** - Adjusts watchlist, avoid list, risk tolerance, and focus sectors

The trading agent receives active theses and recent decision outcomes as part of its context, enabling it to act on pre-researched ideas and learn from past performance.

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
