# News Pipeline Architecture

## Overview

Agentic trading system that processes market news through a local LLM pipeline before feeding signals to the main Claude trading agent.

## Stack

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

## Models (Ollama)

```bash
docker compose exec ollama ollama pull phi3:mini        # Classification
docker compose exec ollama ollama pull nomic-embed-text # Embeddings
```

## News Pipeline

```
Alpaca News API (no symbol filter)
        ↓
Embedding Filter (nomic-embed-text)
  - Embed strategy context once
  - Embed headlines, drop below 0.3 similarity
        ↓
Phi-3 Classification
  - Type: [ticker_specific, macro_political, sector, noise]
  - If macro: subtag [fed, trade, regulation, geopolitical, fiscal, election]
  - If ticker: extract tickers, sentiment, confidence
        ↓
Route to appropriate table
        ↓
Aggregate signals for trading agent context
```

## Database Schema

```sql
-- Ticker-specific news signals
CREATE TABLE news_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    headline TEXT,
    category VARCHAR(20),      -- earnings, guidance, analyst, product, legal, macro, noise
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

-- Trading decisions and outcomes (for learning)
CREATE TABLE decisions (
    id SERIAL PRIMARY KEY,
    date DATE,
    ticker VARCHAR(10),
    action VARCHAR(10),        -- buy, sell, hold
    reasoning TEXT,
    signals_used JSONB,
    outcome_7d DECIMAL,        -- filled in later by cron
    outcome_30d DECIMAL
);
```

## Phi-3 Classification Prompts

### Ticker-Specific News
```
News: "{headline}"
Ticker: {ticker}
Classify: [earnings, guidance, analyst, product, legal, macro, noise]
Sentiment: [bullish, bearish, neutral]
Confidence: [high, medium, low]
Respond as JSON only.
```

### Macro Classification
```
Classify this news:
Headline: {headline}

Type: [ticker_specific, macro_political, sector, noise]
If macro_political, subtag: [fed, trade, regulation, geopolitical, fiscal, election]
If ticker_specific, extract tickers mentioned.
Respond as JSON only.
```

## Trading Agent Context Format

Keep context small - aggregated signals, not raw articles:

```
Macro context:
- Fed: rates held steady, dovish tone (bullish)
- Trade: new China tariffs on semiconductors (bearish for NVDA, AMD)

Today's signals:
- AAPL: 2 bullish (earnings beat, analyst upgrade), 1 neutral
- NVDA: 1 bearish (guidance cut) + sector headwind from tariffs

Recent 7-day trend:
- AAPL: 5 bullish, 1 bearish
- MSFT: neutral, no significant news
```

## Data Sources

| Source | Use |
|--------|-----|
| Alpaca News API | Primary news feed (no symbol filter for broad coverage) |
| Alpaca Market Data | Price data, bars, quotes |
| FRED API (optional) | Fed data directly (rates, employment) |
| Economic calendars (optional) | FOMC dates, CPI releases |

## Automation

- Cron job runs daily after market close
- Fetches news, processes through pipeline
- Claude trading agent reviews signals and makes decisions
- Decisions logged to `decisions` table
- Separate cron backfills 7d/30d outcomes for learning

## Environment Variables

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
OLLAMA_URL=http://ollama:11434
DATABASE_URL=postgresql://algo:algo@db:5432/trading
```
