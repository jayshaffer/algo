# Alpaca Learning Platform - Project Plan

## Goal

Prove whether agentic trading can find an edge using Claude Code integrated with the Alpaca trading API.

## Core Concepts

- Single Alpaca account with daily adjustable strategy
- Claude makes trading decisions based on processed signals
- System learns from past behavior through journaled data
- Runs daily after market close via cron automation
- Local web dashboard for visibility on strategy and reasoning

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                            │
├─────────────┬─────────────┬─────────────┬──────────────────────┤
│   Ollama    │  Postgres   │   Trading   │     Dashboard        │
│  (phi3,     │  (signals,  │    App      │    (Next.js/         │
│  nomic)     │  decisions) │  (Python)   │     Flask)           │
└─────────────┴─────────────┴─────────────┴──────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      Alpaca API         │
                    │  - Trading (buy/sell)   │
                    │  - Market Data          │
                    │  - News Feed            │
                    └─────────────────────────┘
```

## Docker Compose Stack

```yaml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # Remove deploy block if no GPU

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: algo
      POSTGRES_PASSWORD: algo
      POSTGRES_DB: trading
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  trading:
    build: .
    depends_on:
      - ollama
      - db
    environment:
      OLLAMA_URL: http://ollama:11434
      DATABASE_URL: postgresql://algo:algo@db:5432/trading
      ALPACA_API_KEY: ${ALPACA_API_KEY}
      ALPACA_SECRET_KEY: ${ALPACA_SECRET_KEY}
    volumes:
      - ./logs:/app/logs

  dashboard:
    build: ./dashboard
    ports:
      - "3000:3000"
    depends_on:
      - db
    environment:
      DATABASE_URL: postgresql://algo:algo@db:5432/trading

volumes:
  ollama_data:
  postgres_data:
```

## Local LLM Models

| Model | Purpose | Size |
|-------|---------|------|
| phi3:mini | News classification, relevance filtering | ~2GB |
| nomic-embed-text | Embedding for similarity filtering | ~270MB |

Pull after stack is up:
```bash
docker compose exec ollama ollama pull phi3:mini
docker compose exec ollama ollama pull nomic-embed-text
```

## News Processing Pipeline

```
Alpaca News API (no symbol filter for broad coverage)
        ↓
Embedding Filter (nomic-embed-text)
  - Embed strategy context once
  - Embed headlines, drop below 0.3 cosine similarity
        ↓
Phi-3 Classification
  - Type: [ticker_specific, macro_political, sector, noise]
  - Macro subtags: [fed, trade, regulation, geopolitical, fiscal, election]
  - Ticker signals: sentiment, confidence, category
        ↓
Store in Postgres (news_signals, macro_signals tables)
        ↓
Aggregate into compressed context for trading agent
```

## Database Schema

```sql
-- Ticker-specific news signals
CREATE TABLE news_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    headline TEXT,
    category VARCHAR(20),      -- earnings, guidance, analyst, product, legal, noise
    sentiment VARCHAR(10),     -- bullish, bearish, neutral
    confidence VARCHAR(10),    -- high, medium, low
    published_at TIMESTAMP,
    processed_at TIMESTAMP
);

-- Macro/political news signals
CREATE TABLE macro_signals (
    id SERIAL PRIMARY KEY,
    headline TEXT,
    category VARCHAR(20),      -- fed, trade, regulation, geopolitical, fiscal, election
    affected_sectors TEXT[],   -- tech, finance, energy, healthcare, defense, all
    sentiment VARCHAR(10),     -- bullish, bearish, neutral
    published_at TIMESTAMP,
    processed_at TIMESTAMP
);

-- Portfolio state
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    shares DECIMAL,
    avg_cost DECIMAL,
    updated_at TIMESTAMP
);

-- Trading decisions and outcomes (learning journal)
CREATE TABLE decisions (
    id SERIAL PRIMARY KEY,
    date DATE,
    ticker VARCHAR(10),
    action VARCHAR(10),        -- buy, sell, hold
    quantity DECIMAL,
    price DECIMAL,
    reasoning TEXT,
    signals_used JSONB,
    outcome_7d DECIMAL,        -- P&L after 7 days (backfilled)
    outcome_30d DECIMAL        -- P&L after 30 days (backfilled)
);

-- Strategy state (what Claude is currently doing)
CREATE TABLE strategy (
    id SERIAL PRIMARY KEY,
    date DATE,
    description TEXT,
    watchlist TEXT[],
    risk_tolerance VARCHAR(20), -- conservative, moderate, aggressive
    focus_sectors TEXT[]
);
```

## Trading Agent Context Format

Compressed signals to avoid blowing out context window:

```
Current Portfolio:
- AAPL: 100 shares @ $178 avg
- MSFT: 50 shares @ $412 avg
- Cash: $5,000

Macro Context:
- Fed: rates held steady, dovish tone (bullish)
- Trade: new China tariffs on semiconductors (bearish for NVDA, AMD)

Today's Signals:
- AAPL: 2 bullish (earnings beat, analyst upgrade), 1 neutral
- NVDA: 1 bearish (guidance cut) + sector headwind from tariffs

7-Day Signal Trend:
- AAPL: 5 bullish, 1 bearish
- MSFT: neutral, no significant news

Recent Decision Outcomes:
- 2025-01-20 BUY AAPL: +3.2% (7d)
- 2025-01-15 SELL NVDA: avoided -5.1% drop

Current Strategy:
Momentum-based, moderate risk, focusing on tech sector
```

## Daily Automation Flow

```
Market Close (4 PM ET)
        ↓
Cron triggers trading app
        ↓
1. Fetch news from Alpaca (last 24h)
2. Process through embedding filter
3. Classify with Phi-3
4. Store signals in Postgres
5. Sync portfolio state from Alpaca
6. Build compressed context
7. Claude evaluates and decides
8. Execute trades via Alpaca API
9. Log decision + reasoning to decisions table
        ↓
Separate cron backfills outcomes (7d, 30d) for learning
```

## Crontab

```cron
# Run trading agent 30 min after market close
30 16 * * 1-5 cd /path/to/algo && docker compose exec trading python main.py

# Backfill outcomes daily
0 6 * * * cd /path/to/algo && docker compose exec trading python backfill_outcomes.py
```

## Web Dashboard

Local dashboard for visibility into:

- Current portfolio and positions
- Today's signals (ticker + macro)
- Recent decisions with reasoning
- Historical performance
- Current strategy description
- Signal accuracy over time

Tech options:
- Flask + Jinja (simple)
- Next.js (richer UI)
- Streamlit (fastest to build)

## Environment Variables

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading first
OLLAMA_URL=http://ollama:11434
DATABASE_URL=postgresql://algo:algo@db:5432/trading
```

## Learning System

The system learns by:

1. **Journaling every decision** with reasoning and signals used
2. **Backfilling outcomes** (7d, 30d P&L) to see what worked
3. **Querying patterns** - Claude can ask "which signal types led to profitable trades?"
4. **Strategy adjustment** - Daily strategy can evolve based on what's working

Key constraint: Keep context small. Store raw data in Postgres, only surface aggregated insights to Claude.

## Development Phases

### Phase 1: Infrastructure
- [ ] Docker Compose setup
- [ ] Postgres schema
- [ ] Ollama with models
- [ ] Basic Alpaca API integration

### Phase 2: News Pipeline
- [ ] Alpaca news fetching
- [ ] Embedding filter
- [ ] Phi-3 classification
- [ ] Signal storage

### Phase 3: Trading Agent
- [ ] Context builder
- [ ] Claude integration
- [ ] Trade execution
- [ ] Decision logging

### Phase 4: Dashboard
- [ ] Portfolio view
- [ ] Signals view
- [ ] Decision history
- [ ] Performance charts

### Phase 5: Learning Loop
- [ ] Outcome backfill job
- [ ] Pattern queries
- [ ] Strategy evolution

## Paper Trading First

Start with Alpaca paper trading (`paper-api.alpaca.markets`) to validate the system before using real money.

## Data Sources

| Source | Use |
|--------|-----|
| Alpaca Trading API | Execute trades, portfolio state |
| Alpaca Market Data | Price data, bars, quotes |
| Alpaca News API | News feed (no filter for broad coverage) |
| FRED API (optional) | Fed data directly |
| Economic calendars (optional) | FOMC dates, CPI releases |
