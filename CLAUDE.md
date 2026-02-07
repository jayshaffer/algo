# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpaca Learning Platform - an agentic trading system that uses a local LLM (Ollama) to integrate with the Alpaca trading API, learn from past behavior, and make trading decisions.

**Status:** Active development - core trading infrastructure implemented

## Project Goals

- Prove whether agentic trading can find an edge
- Local LLM (qwen2.5:14b via Ollama) makes trading decisions
- Daily automation after market close
- Learning system that journals behavior and outcomes
- Single Alpaca account with adjustable day-to-day strategy
- Local web dashboard for strategy visibility and reasoning

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
├──────────────┬──────────────┬───────────────┬───────────────┤
│   Ollama     │  PostgreSQL  │   Trading     │   Dashboard   │
│  (qwen2.5)     │   (pgvector) │   Agent       │   (Flask)     │
│  :11434      │   :5432      │               │   :3000       │
└──────────────┴──────────────┴───────────────┴───────────────┘
```

- **LLM:** Ollama with qwen2.5:14b (GPU accelerated)
- **Database:** PostgreSQL 16 + pgvector for RAG
- **API:** Alpaca Trading API (read/write)
- **Dashboard:** Flask web app on port 3000

## Key Concepts

### Trading Session (`trader.py`)
Daily workflow:
1. Sync positions/orders from Alpaca
2. Take account snapshot
3. Build compressed context
4. Get decisions from LLM
5. Validate and execute trades
6. Log decisions to database

### Thesis Engine (`ideation.py`)
Generates and manages trade ideas using RAG:
- Retrieves relevant documents for each ticker
- Reviews existing theses against fresh data
- Generates new theses grounded in retrieved documents
- Every thesis must cite [DOC-ID] sources

### RAG System (`retrieval.py`, `ingest.py`)

The ideation engine uses Retrieval-Augmented Generation to ground theses in fresh documents rather than stale model knowledge.

**Document Sources:**
- Alpaca News API (free with trading account)
- SEC EDGAR (10-K, 10-Q, 8-K filings)

**Retrieval modes:**
- `retrieve_by_ticker(ticker)` - Get recent docs for a specific company
- `retrieve_by_query(query)` - Semantic search across all documents
- `retrieve_for_ideation(tickers, themes)` - Combined retrieval for ideation

**Time decay scoring:**
Documents lose 50% relevance after 30 days: `score = similarity * exp(-age/30)`

**Citation requirement:**
The LLM must cite `[DOC-ID]` for every claim. Theses without citations are flagged.

**Ingestion schedule:**
```bash
# Run daily at 6am ET before market open
docker compose exec trading python -m trading.ingest_scheduler
```

### Database Schema
- `documents` - RAG document store with pgvector embeddings
- `news_signals` - Ticker-specific news
- `macro_signals` - Macro/political news affecting sectors
- `positions` - Current portfolio holdings
- `decisions` - Trading decisions with reasoning and outcomes
- `theses` - Trade ideas with entry/exit triggers

## Commands

```bash
# Start the stack
docker compose up -d

# Run document ingestion
docker compose exec trading python -m trading.ingest_scheduler

# Run ideation with RAG
docker compose exec trading python -m trading.ideation

# Run trading session (dry run)
docker compose exec trading python -m trading.trader --dry-run

# View dashboard
open http://localhost:3000
```

## Environment Variables

Required in `.env`:
- `APCA_API_KEY_ID` - Alpaca API key
- `APCA_API_SECRET_KEY` - Alpaca API secret
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` - Database credentials
- `OLLAMA_URL` - Ollama endpoint (default: http://ollama:11434)
