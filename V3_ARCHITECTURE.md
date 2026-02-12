# Alpaca Learning Platform - Complete Architecture Document (V2 → V3 Reference)

> Generated 2026-02-12. Exhaustive dump of the entire V2 codebase: every module, function, dataclass, database table, query pattern, API integration, test, and infrastructure component.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Data Flow](#3-data-flow)
4. [Module Dependency Graph](#4-module-dependency-graph)
5. [Trading Module - All 22 Files](#5-trading-module)
6. [Dashboard Module](#6-dashboard-module)
7. [Complete Database Schema](#7-complete-database-schema)
8. [All SQL Query Patterns](#8-all-sql-query-patterns)
9. [External API Integrations](#9-external-api-integrations)
10. [Infrastructure & Docker](#10-infrastructure--docker)
11. [Test Suite](#11-test-suite)
12. [Constants & Configuration](#12-constants--configuration)
13. [Known Gaps & Technical Debt](#13-known-gaps--technical-debt)

---

## 1. System Overview

**What it does:** An agentic trading system that uses LLMs (Claude + Ollama/Qwen) to analyze news, generate trade theses, make trading decisions, execute orders via Alpaca, and learn from outcomes.

**Core thesis:** Prove whether agentic trading can find an edge using a feedback loop: signals → decisions → outcomes → attribution → better signals.

**Key components:**
- **News Pipeline:** Fetch → filter (embeddings) → classify (Qwen) → store signals
- **Claude Strategist:** Autonomous research agent (Opus) that manages theses and writes daily playbooks
- **Trading Executor:** Claude Haiku reads context + playbook, makes buy/sell/hold decisions
- **Learning Loop:** Backfill outcomes → compute attribution → pattern analysis → feed back to strategist

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Stack                          │
├───────────────┬──────────────┬────────────────┬──────────────────────┤
│   Ollama      │  PostgreSQL  │   Trading      │   Dashboard          │
│  qwen2.5:14b  │  16+pgvector │   Agent        │   Flask :3000        │
│  nomic-embed  │   :5432      │   (Python)     │   (Tailwind+Chart.js)│
│  :11434       │              │                │                      │
└───────────────┴──────────────┴────────────────┴──────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              Alpaca Trading    Alpaca Data      Anthropic Claude
              (orders/acct)    (bars/quotes)    (Opus+Haiku)
```

---

## 3. Data Flow

### Daily Session (`session.py → run_session()`)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: News Pipeline (pipeline.py)                                │
│  fetch_broad_news(hours=24, limit=300)                             │
│  → filter_by_relevance(threshold=0.3) via nomic-embed-text          │
│  → classify_news_batch(batch_size=10) via qwen2.5:14b              │
│  → insert_news_signals_batch() + insert_macro_signals_batch()       │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 2: Claude Strategist (ideation_claude.py)                     │
│  backfill_outcomes(7d, 30d)                                         │
│  → compute_signal_attribution()                                     │
│  → run_strategist_loop(model=claude-opus-4-6, max_turns=25)        │
│    Tools: get_market_snapshot, get_portfolio_state, get_active_theses│
│           create_thesis, update_thesis, close_thesis, get_news_signals│
│           get_macro_context, get_signal_attribution, get_decision_history│
│           write_playbook, web_search                                │
│  → Outputs: updated theses + today's playbook                       │
├─────────────────────────────────────────────────────────────────────┤
│ Stage 3: Trading Executor (trader.py)                               │
│  sync_positions_from_alpaca()                                       │
│  sync_orders_from_alpaca()                                          │
│  take_account_snapshot()                                            │
│  → build_trading_context() (portfolio, macro, signals, theses,      │
│     playbook, attribution, decision outcomes)                       │
│  → get_trading_decisions(model=claude-haiku-4-5) via agent.py       │
│  → validate_decision() per decision                                 │
│  → execute_market_order() per validated decision                    │
│  → insert_decision() + insert_decision_signals_batch()              │
│  → close/mark theses as executed/invalidated                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Learning Loop (`learn.py → run_learning_loop()`)

```
backfill_outcomes(7d) → backfill_outcomes(30d)
→ generate_pattern_report(days=60)
  → analyze_signal_categories, analyze_sentiment_performance,
    analyze_ticker_performance, analyze_confidence_correlation,
    get_best/worst_performing_signals
→ compute_signal_attribution()
  → JOIN decision_signals + decisions + news/macro_signals
  → GROUP BY category → upsert signal_attribution
```

---

## 4. Module Dependency Graph

```
session.py (orchestration)
  ├─ pipeline.py (news processing)
  │  ├─ news.py (Alpaca News API fetch)
  │  ├─ filter.py (embedding relevance filtering)
  │  │  └─ ollama.py (nomic-embed-text embeddings)
  │  ├─ classifier.py (news classification)
  │  │  └─ ollama.py (qwen2.5:14b chat)
  │  └─ db.py (signal storage)
  │
  ├─ ideation_claude.py (strategist)
  │  ├─ claude_client.py (Claude API + agentic loop)
  │  ├─ tools.py (11 tool definitions + handlers)
  │  │  ├─ market_data.py (sector/index/movers data)
  │  │  ├─ executor.py (account info, positions)
  │  │  ├─ context.py (macro, signals, theses, attribution)
  │  │  └─ db.py (thesis CRUD, playbook write)
  │  ├─ backfill.py (outcome backfill)
  │  └─ attribution.py (signal attribution)
  │
  └─ trader.py (execution)
     ├─ agent.py (Claude Haiku decisions)
     │  └─ claude_client.py
     ├─ context.py (full context builder)
     │  ├─ db.py (all reads)
     │  └─ attribution.py
     ├─ executor.py (Alpaca trade execution)
     └─ db.py (decision logging)

learn.py (learning loop - separate)
  ├─ backfill.py (7d/30d outcome backfill)
  ├─ patterns.py (signal/sentiment/ticker analysis)
  ├─ attribution.py (predictive signal scoring)
  └─ db.py

ideation.py (legacy Ollama-based ideation)
  ├─ ollama.py
  ├─ market_data.py
  └─ db.py
```

---

## 5. Trading Module

22 Python files in `trading/`.

### 5.1 `agent.py` - LLM Trading Decisions

**Purpose:** Gets structured buy/sell/hold decisions from Claude Haiku.

**Dataclasses:**
```python
@dataclass
class TradingDecision:
    action: str          # buy, sell, hold
    ticker: str
    quantity: float
    reasoning: str
    confidence: str      # high, medium, low
    thesis_id: int = None
    signal_refs: list = field(default_factory=list)  # [{type, id}]

@dataclass
class ThesisInvalidation:
    thesis_id: int
    reason: str

@dataclass
class AgentResponse:
    decisions: list[TradingDecision]
    thesis_invalidations: list[ThesisInvalidation]
    market_summary: str
    risk_assessment: str
```

**Functions:**
```python
get_trading_decisions(context: str, model: str = DEFAULT_EXECUTOR_MODEL) -> AgentResponse
# Calls Claude with TRADING_SYSTEM_PROMPT + context, parses JSON response

format_decisions_for_logging(response: AgentResponse) -> dict
# Formats for JSONB storage in decisions.signals_used

validate_decision(decision: TradingDecision, buying_power: Decimal,
                  current_price: Decimal, positions: dict) -> tuple[bool, str]
# Validates: buy checks buying_power, sell checks shares held
```

**Constants:**
- `TRADING_SYSTEM_PROMPT`: System prompt defining trading rules, JSON output format
- `DEFAULT_EXECUTOR_MODEL`: `"claude-haiku-4-5-20251001"`

**Key business logic:**
- JSON parsing with markdown code block handling (````json ... ````)
- Fractional share support
- Position size limits (1-5% of buying power recommended)
- All decisions must cite signal references for the learning loop

---

### 5.2 `attribution.py` - Signal Performance Analysis

**Purpose:** Computes which signal types are predictive by joining decision_signals with outcomes.

**Functions:**
```python
compute_signal_attribution() -> list[dict]
# Complex JOIN: decision_signals + decisions + news_signals + macro_signals
# Groups by composite category (e.g., "news_signal:earnings", "macro_signal:fed")
# Computes: sample_size, avg_outcome_7d/30d, win_rate_7d/30d
# Upserts results to signal_attribution table

get_attribution_summary() -> str
# Formats attribution scores into readable text for LLM context
```

**SQL:** CTE with conditional categorization, LEFT JOINs, GROUP BY, AVG with CASE.

---

### 5.3 `backfill.py` - Outcome Backfill

**Purpose:** Fills in 7d and 30d P&L for past trading decisions using Alpaca historical data.

**Functions:**
```python
get_data_client() -> StockHistoricalDataClient
get_price_on_date(client, ticker: str, target_date: date) -> Optional[Decimal]
# Fetches closing price with weekend/holiday buffer

get_decisions_needing_backfill(days_threshold: int) -> list
# WHERE outcome_{days}d IS NULL AND date <= cutoff AND action IN ('buy','sell')

calculate_outcome(action: str, entry_price: Decimal, exit_price: Decimal) -> Decimal
# BUY: positive if price up; SELL: positive if price down

update_outcome(decision_id: int, days: int, outcome: Decimal)
backfill_outcomes(days: int = 7, dry_run: bool = False) -> dict
run_backfill(dry_run: bool = False) -> dict  # Runs 7d then 30d
main()  # CLI: --dry-run, --days
```

---

### 5.4 `classifier.py` - News Classification (Qwen)

**Purpose:** Classifies news headlines into ticker-specific, macro-political, sector, or noise.

**Dataclasses:**
```python
@dataclass
class TickerSignal:
    ticker: str
    headline: str
    category: str      # earnings, guidance, analyst, product, legal, noise
    sentiment: str     # bullish, bearish, neutral
    confidence: str    # high, medium, low
    published_at: datetime

@dataclass
class MacroSignal:
    headline: str
    category: str      # fed, trade, regulation, geopolitical, fiscal, election
    affected_sectors: list[str]
    sentiment: str
    published_at: datetime

@dataclass
class ClassificationResult:
    news_type: str     # ticker_specific, macro_political, sector, noise
    ticker_signals: list[TickerSignal]
    macro_signal: MacroSignal  # or None
```

**Functions:**
```python
classify_news(headline: str, published_at: datetime) -> ClassificationResult
# Single headline via qwen2.5:14b

classify_news_batch(headlines: list[str], published_ats: list[datetime],
                    batch_size: int = 10) -> list[ClassificationResult]
# Batches headlines (10 per LLM call), falls back to individual on failure

classify_ticker_news(ticker: str, headline: str, published_at: datetime) -> TickerSignal
# When ticker is already known

classify_filtered_news(filtered_items: list[FilteredNewsItem]) -> list[ClassificationResult]
```

**LLM:** Uses `chat_json()` with qwen2.5:14b, temperature=0.0 for classification.

---

### 5.5 `claude_client.py` - Claude API Client

**Purpose:** Claude API communication with retry logic, prompt caching, and agentic loops.

**Dataclasses:**
```python
@dataclass
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool

@dataclass
class AgenticLoopResult:
    messages: list
    turns_used: int
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
```

**Functions:**
```python
get_claude_client() -> anthropic.Anthropic
# From ANTHROPIC_API_KEY env var

_call_with_retry(client, max_retries=3, **create_kwargs) -> Response
# Exponential backoff + jitter; retries: RateLimitError, InternalServerError, APIConnectionError
# Rate limit: 60s delay; others: 2^attempt * 2s + jitter

_messages_with_cache_breakpoint(messages: list) -> list
# Adds ephemeral cache_control to last user message for prompt caching

run_agentic_loop(client, model, system, initial_message, tools,
                 tool_handlers, max_turns=20) -> AgenticLoopResult
# Multi-turn agentic conversation; handles tool_use blocks
# Tracks token usage across all turns

extract_final_text(messages: list) -> Optional[str]
# Gets final text response from conversation history
```

**Constants:**
- `API_MAX_RETRIES`: 3
- `API_RATE_LIMIT_DELAY`: 60s
- `API_RETRY_BASE_DELAY`: 2s
- `RETRYABLE_ERRORS`: `(RateLimitError, InternalServerError, APIConnectionError)`

---

### 5.6 `context.py` - Context Builder

**Purpose:** Aggregates all signals, positions, theses into compressed text context for LLM.

**Functions:**
```python
get_portfolio_context(account_info: dict) -> str
# Positions, open orders, cash, buying power

get_macro_context(days: int = 7) -> str
# Macro signals grouped by category (fed, trade, regulation, etc.)

get_ticker_signals_context(days: int = 1) -> str
# Today's signals grouped by ticker with sentiment distribution

get_signal_trend_context(days: int = 7) -> str
# 7-day signal trend summary

get_decision_outcomes_context(days: int = 30) -> str
# Recent decisions with 7d P&L outcomes

get_theses_context() -> str
# Active theses with triggers and confidence

get_playbook_context(playbook_date: date) -> str
# Today's playbook (market outlook, priority actions, watch list, risk notes)

get_attribution_context() -> str
# Signal attribution summary

build_trading_context(account_info: dict, playbook_date: date = None) -> str
# Assembles ALL sections into single context string
```

---

### 5.7 `db.py` - Database Client (All CRUD)

**Purpose:** All database operations. Raw SQL + psycopg2, no ORM.

**Connection:**
```python
def get_connection():
    # Reads DATABASE_URL env var → psycopg2.connect()

@contextmanager
def get_cursor():
    # Yields RealDictCursor; auto-commits on success, rolls back on error
```

**All functions (grouped by table):**

| Table | Function | Operation |
|-------|----------|-----------|
| **news_signals** | `insert_news_signal(ticker, headline, category, sentiment, confidence, published_at)` | INSERT → id |
| | `insert_news_signals_batch(signals: list[tuple])` | Batch INSERT via execute_values → count |
| | `get_news_signals(ticker: Optional, days: int = 7)` | SELECT with optional ticker filter |
| **macro_signals** | `insert_macro_signal(headline, category, affected_sectors, sentiment, published_at)` | INSERT → id |
| | `insert_macro_signals_batch(signals: list[tuple])` | Batch INSERT → count |
| | `get_macro_signals(category: Optional, days: int = 7)` | SELECT with optional category filter |
| **account_snapshots** | `insert_account_snapshot(date, cash, portfolio_value, buying_power, long_mv, short_mv)` | UPSERT (ON CONFLICT date) → id |
| | `get_account_snapshots(days: int = 30)` | SELECT |
| **decisions** | `insert_decision(date, ticker, action, qty, price, reasoning, signals_used, equity, bp)` | INSERT (signals_used as JSONB) → id |
| | `get_recent_decisions(days: int = 30)` | SELECT ORDER BY date DESC |
| **positions** | `upsert_position(ticker, shares, avg_cost)` | UPSERT (ON CONFLICT ticker) → id |
| | `get_positions()` | SELECT ORDER BY ticker |
| | `delete_position(ticker)` | DELETE → bool |
| **theses** | `insert_thesis(ticker, direction, thesis, entry_trigger, exit_trigger, invalidation, confidence, source)` | INSERT → id |
| | `get_active_theses(ticker: Optional)` | SELECT WHERE status='active' |
| | `update_thesis(thesis_id, thesis?, entry_trigger?, exit_trigger?, invalidation?, confidence?)` | Dynamic UPDATE → bool |
| | `close_thesis(thesis_id, status, reason)` | UPDATE status/closed_at/close_reason → bool |
| **open_orders** | `upsert_open_order(order_id, ticker, side, type, qty, filled_qty, limit_price, stop_price, status, submitted_at)` | UPSERT (ON CONFLICT order_id) → id |
| | `get_open_orders()` | SELECT ORDER BY submitted_at DESC |
| | `delete_open_order(order_id)` | DELETE → bool |
| | `clear_closed_orders()` | DELETE WHERE status NOT IN active statuses → count |
| **playbooks** | `upsert_playbook(date, outlook, priority_actions, watch_list, risk_notes)` | UPSERT (ON CONFLICT date) → id |
| | `get_playbook(playbook_date)` | SELECT → dict or None |
| **decision_signals** | `insert_decision_signal(decision_id, signal_type, signal_id)` | INSERT ON CONFLICT DO NOTHING |
| | `insert_decision_signals_batch(signals: list[tuple])` | Batch INSERT → count |
| | `get_decision_signals(decision_id)` | SELECT |
| **signal_attribution** | `upsert_signal_attribution(category, sample_size, avg_7d, avg_30d, wr_7d, wr_30d)` | UPSERT (ON CONFLICT category) |
| | `get_signal_attribution()` | SELECT ORDER BY sample_size DESC |

---

### 5.8 `executor.py` - Trade Execution

**Purpose:** Executes trades via Alpaca API; manages positions and orders.

**Dataclasses:**
```python
@dataclass
class OrderResult:
    success: bool
    order_id: str
    filled_qty: float
    filled_avg_price: Decimal
    error: str
```

**Functions:**
```python
get_trading_client() -> TradingClient
# From ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

get_account_info() -> dict
# Returns: account_id, status, cash, portfolio_value, buying_power, equity, daytrade_count, pattern_day_trader

take_account_snapshot() -> int
# Calls insert_account_snapshot for today

sync_positions_from_alpaca() -> int
# Fetches all positions, upserts to DB, deletes stale ones

sync_orders_from_alpaca() -> int
# Fetches all open orders, upserts to DB, deletes stale ones

execute_market_order(ticker, side, qty, dry_run=False) -> OrderResult
# Market order (TimeInForce.DAY); simulates if dry_run

execute_limit_order(ticker, side, qty, limit_price, dry_run=False) -> OrderResult

get_latest_price(ticker) -> Optional[Decimal]
# Latest ask price from Alpaca quote

calculate_position_size(buying_power, price, risk_pct=0.05) -> float
# Fractional shares: (buying_power * risk_pct) / price
```

---

### 5.9 `filter.py` - Embedding-Based News Filtering

**Purpose:** Filters news by relevance using embedding cosine similarity.

**Dataclasses:**
```python
@dataclass
class FilteredNewsItem:
    item: NewsItem
    relevance_score: float
```

**Functions:**
```python
filter_by_relevance(news_items: list, strategy_context: str,
                    threshold: float = 0.3) -> list[FilteredNewsItem]
# Embeds strategy context once, batch embeds headlines, vectorized cosine similarity
# Filters ≥ threshold, sorts descending

filter_news_batch(news_items, strategy_context, threshold, batch_size) -> list[FilteredNewsItem]
# Wrapper with progress logging
```

**Constants:**
- `DEFAULT_STRATEGY_CONTEXT`: Text defining what's relevant to the trading strategy

**LLM:** Uses `nomic-embed-text` embeddings via Ollama.

---

### 5.10 `ideation.py` - Ollama-Based Ideation (Legacy)

**Purpose:** Generates/manages trade theses using local Ollama LLM.

**Dataclasses:**
```python
@dataclass
class ThesisReview:
    thesis_id: int
    action: str  # keep, update, invalidate, expire
    reason: str
    updates: dict

@dataclass
class NewThesis:
    ticker: str
    direction: str  # long, short, avoid
    thesis: str
    entry_trigger: str
    exit_trigger: str
    invalidation: str
    confidence: str

@dataclass
class IdeationResult:
    timestamp: datetime
    reviews: list[ThesisReview]
    new_theses: list[NewThesis]
    market_observations: str
    theses_kept: int
    theses_updated: int
    theses_closed: int
    theses_created: int
```

**Functions:**
```python
build_ideation_context(account_info) -> str
# Gathers portfolio, theses, macro, market snapshot

run_ideation(model="qwen2.5:14b") -> IdeationResult
main()  # CLI
```

**Logic:** Skips tickers already in portfolio or with active thesis.

---

### 5.11 `ideation_claude.py` - Claude-Based Strategist

**Purpose:** Autonomous research agent (Claude Opus) + strategist that writes playbooks.

**Dataclasses:**
```python
@dataclass
class StrategistResult:
    timestamp: datetime
    model: str
    turns_used: int
    outcomes_backfilled: dict
    attribution_computed: list
    theses_created: int
    theses_updated: int
    theses_closed: int
    final_summary: str
    input_tokens: int
    output_tokens: int

@dataclass
class ClaudeIdeationResult:
    timestamp: datetime
    model: str
    turns_used: int
    theses_created: int
    theses_updated: int
    theses_closed: int
    final_summary: str
    input_tokens: int
    output_tokens: int
```

**Functions:**
```python
count_actions(messages) -> tuple
# Counts "Created thesis ID", "Updated thesis ID", "Closed thesis ID" in tool results

run_ideation_claude(model="claude-opus-4-6", max_turns=20) -> ClaudeIdeationResult
# Research mode - thesis generation

run_strategist_loop(model="claude-opus-4-6", max_turns=25, system_prompt=None) -> ClaudeIdeationResult
# Strategist only (reads attribution from DB, no backfill/compute)

run_strategist_session(model="claude-opus-4-6", max_turns=25, dry_run=False) -> StrategistResult
# Full session: backfill + attribution + strategist loop

main()  # CLI: --model, --max-turns, --dry-run, --legacy
```

**Cost model:**
- Input: $5/1M tokens (minus cache)
- Cache creation: $6.25/1M tokens
- Cache read: $0.50/1M tokens
- Output: $25/1M tokens

**System prompts:** `CLAUDE_IDEATION_SYSTEM`, `CLAUDE_STRATEGIST_SYSTEM`, `CLAUDE_SESSION_STRATEGIST_SYSTEM`

---

### 5.12 `learn.py` - Learning Loop Orchestrator

**Dataclasses:**
```python
@dataclass
class LearningResult:
    timestamp: datetime
    outcomes_backfilled: dict
    attribution_computed: list
    pattern_report: str
    errors: list[str]
```

**Functions:**
```python
run_learning_loop(analysis_days=60, dry_run=False, skip_backfill=False,
                  skip_attribution=False) -> LearningResult
# Runs: backfill → pattern analysis → attribution (each stage independent)

main()  # CLI: --days, --dry-run, --skip-backfill, --skip-attribution, --patterns-only
```

---

### 5.13 `log_config.py` - Logging

```python
FILE_LOGGERS = ["trader", "pipeline", "session", "trading.ideation_claude", "trading.tools"]

def setup_logging(level=logging.INFO, log_dir="logs"):
    # Root logger + dedicated file handlers per module
```

---

### 5.14 `main.py` - Connectivity Check

```python
check_connectivity()   # Prints account info
get_positions()        # Lists holdings with P&L
get_quote(symbol)      # Shows bid/ask
main()                 # Full connectivity test
```

---

### 5.15 `market_data.py` - Market Data for Ideation

**Dataclasses:**
```python
@dataclass
class SectorPerformance:
    sector: str
    etf: str
    change_1d: float
    change_5d: float

@dataclass
class StockMover:
    ticker: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int

@dataclass
class MarketSnapshot:
    timestamp: datetime
    sectors: list[SectorPerformance]
    indices: dict
    gainers: list[StockMover]
    losers: list[StockMover]
    unusual_volume: list[StockMover]
```

**Constants:**
```python
SECTOR_ETFS = {
    "Technology": "XLK", "Financials": "XLF", "Healthcare": "XLV",
    "Consumer Discretionary": "XLY", "Industrials": "XLI", "Energy": "XLE",
    "Consumer Staples": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
    "Materials": "XLB", "Communication Services": "XLC"
}
INDEX_ETFS = ["SPY", "QQQ", "IWM"]
```

**Functions:**
```python
get_data_client() -> StockHistoricalDataClient
get_bar_change(client, symbol, days) -> Optional[float]  # % change
get_sector_performance(client) -> list[SectorPerformance]
get_index_levels(client) -> dict  # 1-day change for SPY/QQQ/IWM
get_top_movers(client, universe, top_n=10) -> tuple  # (gainers, losers)
get_unusual_volume(client, universe, threshold=2.0, top_n=10) -> list
get_default_universe() -> list[str]  # S&P 100 subset (~60 stocks)
get_market_snapshot(universe=None) -> MarketSnapshot
format_market_snapshot(snapshot) -> str  # Formatted for LLM context
```

---

### 5.16 `news.py` - Alpaca News API

**Dataclasses:**
```python
@dataclass
class NewsItem:
    id: str
    headline: str
    summary: str
    author: str
    source: str
    symbols: list[str]
    published_at: datetime
    url: str
```

**Functions:**
```python
get_news_client() -> NewsClient
fetch_news(hours=24, symbols=None, limit=50) -> list[NewsItem]
fetch_broad_news(hours=24, limit=50) -> list[NewsItem]  # No symbol filter
fetch_ticker_news(ticker, hours=24, limit=20) -> list[NewsItem]
```

---

### 5.17 `ollama.py` - Local LLM & Embeddings

**Functions:**
```python
get_ollama_url() -> str  # OLLAMA_URL env var, default http://localhost:11434

# Embeddings (nomic-embed-text)
embed(text, model="nomic-embed-text") -> list[float]
embed_batch(texts, model="nomic-embed-text") -> list[list[float]]

# Chat (qwen2.5:14b)
chat(prompt, model="qwen2.5:14b", system=None, temperature=0.1) -> str
chat_json(prompt, model="qwen2.5:14b", system=None) -> dict
# Parses JSON with markdown code block handling

# Similarity
cosine_similarity(a, b) -> float  # Numpy-based
cosine_similarity_batch(query, embeddings) -> list[float]  # Vectorized

# Health
check_ollama_health() -> bool
list_models() -> list[str]
```

---

### 5.18 `patterns.py` - Pattern Analysis

**Dataclasses:**
```python
@dataclass
class SignalPerformance:
    category: str
    total_signals: int
    avg_outcome_7d: float
    avg_outcome_30d: float
    win_rate_7d: float
    win_rate_30d: float

@dataclass
class SentimentPerformance:
    sentiment: str
    total_decisions: int
    avg_outcome_7d: float
    avg_outcome_30d: float
    win_rate_7d: float

@dataclass
class TickerPerformance:
    ticker: str
    total_decisions: int
    buys: int
    sells: int
    avg_outcome_7d: float
    avg_outcome_30d: float
    total_pnl_7d: float

@dataclass
class ConfidenceCorrelation:
    confidence: str
    total_decisions: int
    avg_outcome_7d: float
    win_rate_7d: float
```

**Functions:**
```python
analyze_signal_categories(days=90) -> list[SignalPerformance]
# JOINs decisions with news_signals within 7-day window, groups by category

analyze_sentiment_performance(days=90) -> list[SentimentPerformance]
# Groups by sentiment within 3-day window

analyze_ticker_performance(days=90) -> list[TickerPerformance]
# Groups by ticker, counts buys/sells, sums P&L

analyze_confidence_correlation(days=90) -> list[ConfidenceCorrelation]
# Stated confidence vs actual outcomes

get_best_performing_signals(days=90, min_occurrences=3) -> list[dict]
get_worst_performing_signals(days=90, min_occurrences=3) -> list[dict]

generate_pattern_report(days=90) -> str
# Human-readable summary of all analyses
```

---

### 5.19 `pipeline.py` - News Processing Pipeline

**Dataclasses:**
```python
@dataclass
class PipelineStats:
    news_fetched: int
    news_filtered: int
    ticker_signals_stored: int
    macro_signals_stored: int
    noise_dropped: int
    errors: list[str]
```

**Functions:**
```python
run_pipeline(hours=24, limit=300, relevance_threshold=0.3,
             strategy_context=DEFAULT_STRATEGY_CONTEXT, dry_run=False) -> PipelineStats
# Full pipeline: fetch → filter → classify → batch insert

check_dependencies() -> bool  # Checks Ollama health + models
main()  # CLI: --hours, --limit, --threshold, --dry-run, --check
```

---

### 5.20 `session.py` - Consolidated Daily Session

**Dataclasses:**
```python
@dataclass
class SessionResult:
    pipeline_result: PipelineStats  # or None
    strategist_result: StrategistResult  # or None
    trading_result: TradingSessionResult  # or None
    pipeline_error: str
    strategist_error: str
    trading_error: str
    pipeline_skipped: bool
    ideation_skipped: bool
    duration_seconds: float

    @property
    def has_errors(self) -> bool
```

**Functions:**
```python
run_session(dry_run=False, model="claude-opus-4-6",
            executor_model=DEFAULT_EXECUTOR_MODEL, max_turns=25,
            skip_pipeline=False, skip_ideation=False,
            pipeline_hours=24, pipeline_limit=300) -> SessionResult
# Runs 3 stages with error isolation (each wrapped in try/except)

main()  # CLI entry point
```

---

### 5.21 `tools.py` - Claude Agentic Tools

**11 tool handlers:**

| Tool Name | Handler | Description |
|-----------|---------|-------------|
| `get_market_snapshot` | `tool_get_market_snapshot()` | Current sectors, indices, movers, volume |
| `get_portfolio_state` | `tool_get_portfolio_state()` | Positions + account info |
| `get_active_theses` | `tool_get_active_theses(ticker?)` | Active theses |
| `create_thesis` | `tool_create_thesis(ticker, direction, thesis, ...)` | Creates thesis (rejects duplicates) |
| `update_thesis` | `tool_update_thesis(thesis_id, ...)` | Updates existing thesis |
| `close_thesis` | `tool_close_thesis(thesis_id, status, reason)` | Closes thesis |
| `get_news_signals` | `tool_get_news_signals(ticker?, days=7)` | Recent ticker news |
| `get_macro_context` | `tool_get_macro_context(days=7)` | Macro signals |
| `get_signal_attribution` | `tool_get_signal_attribution()` | Attribution summary |
| `get_decision_history` | `tool_get_decision_history(days=30)` | Recent decisions with outcomes |
| `write_playbook` | `tool_write_playbook(outlook, actions, watch_list, risk_notes)` | Writes today's playbook |

Plus server-side `web_search` tool (type: `"web_search_20250305"`).

---

### 5.22 `trader.py` - Trading Session Orchestrator

**Dataclasses:**
```python
@dataclass
class TradeResult:
    decision: TradingDecision
    executed: bool
    order_id: str
    filled_price: Decimal
    error: str

@dataclass
class TradingSessionResult:
    timestamp: datetime
    snapshot_id: int
    positions_synced: int
    orders_synced: int
    decisions_made: int
    trades_executed: int
    trades_failed: int
    total_buy_value: Decimal
    total_sell_value: Decimal
    errors: list[str]
```

**Functions:**
```python
run_trading_session(dry_run=False, model=DEFAULT_EXECUTOR_MODEL) -> TradingSessionResult
main()  # CLI: --dry-run, --model
```

**Execution flow:**
1. `sync_positions_from_alpaca()` + `sync_orders_from_alpaca()`
2. `take_account_snapshot()`
3. `build_trading_context(account_info, playbook_date)`
4. `get_trading_decisions(context, model)` → Claude Haiku
5. For each decision: `validate_decision()` → `execute_market_order()` (hold → skip)
6. Mark theses as "executed"/"invalidated" via `close_thesis()`
7. `insert_decision()` + `insert_decision_signals_batch()` per decision

---

## 6. Dashboard Module

Flask web app on port 3000 with Tailwind CSS + Chart.js.

### 6.1 Routes

| Route | Method | Handler | Data Sources |
|-------|--------|---------|--------------|
| `/` | GET | `portfolio()` | positions, latest_snapshot, today_playbook |
| `/playbook` | GET | `playbook()` | today_playbook |
| `/attribution` | GET | `attribution()` | signal_attribution |
| `/signals` | GET | `signals()` | signal_summary(7d), ticker_signals(7d, 50), macro_signals(7d, 20) |
| `/theses` | GET | `theses()` | thesis_stats, theses(status_filter, sort_by) |
| `/decisions` | GET | `decisions()` | recent_decisions(30d, 50), decision_stats(30d) |
| `/performance` | GET | `performance()` | equity_curve(90d), performance_metrics(30d) |
| `/api/theses/<id>/close` | POST | `api_close_thesis()` | JSON body: {status, reason} |
| `/api/portfolio` | GET | `api_portfolio()` | positions, snapshot (JSON) |
| `/api/signals` | GET | `api_signals()` | ticker_signals, macro_signals (JSON) |
| `/health` | GET | `health()` | {"status": "healthy"} |

### 6.2 `queries.py` - Dashboard DB Queries

Separate from `trading/db.py`. Has its own `get_connection()` / `get_cursor()` from `DATABASE_URL`.

**Functions:**
```python
# Portfolio
get_positions() -> list[dict]
get_latest_snapshot() -> dict

# Signals
get_recent_ticker_signals(days=7, limit=50) -> list[dict]
get_recent_macro_signals(days=7, limit=20) -> list[dict]
get_signal_summary(days=7) -> list[dict]  # Aggregated by ticker with bullish/bearish/neutral counts

# Decisions
get_recent_decisions(days=30, limit=50) -> list[dict]
get_decision_stats(days=30) -> dict  # total, buys, sells, holds, avg_outcome_7d/30d

# Performance
get_equity_curve(days=90) -> list[dict]  # date, cash, portfolio_value, buying_power
get_performance_metrics(days=30) -> dict | None  # start/end values, pnl, pnl_pct

# Playbook
get_today_playbook() -> dict | None

# Attribution
get_signal_attribution() -> list[dict]
get_decision_signal_refs(decision_id) -> list[dict]

# Theses
get_thesis_stats() -> dict  # active/executed/invalidated/expired counts, confidence_dist
get_theses(status_filter='active', sort_by='newest') -> list[dict]
close_thesis(thesis_id, status, reason) -> bool
```

### 6.3 Templates (Jinja2)

| Template | Key Components |
|----------|---------------|
| `base.html` | Nav bar (Portfolio, Playbook, Signals, Theses, Decisions, Attribution, Performance), Tailwind CSS CDN, Chart.js CDN, custom sentiment CSS classes |
| `portfolio.html` | 4 account summary cards, positions table, today's playbook preview |
| `playbook.html` | Market outlook, risk notes (yellow alert), priority actions table (ticker/action/reasoning/qty/confidence), watch list tags |
| `signals.html` | 7-day signal summary cards, macro signals with sentiment-colored borders, ticker signals table |
| `theses.html` | 6 analytics cards, filter/sort dropdowns, thesis cards with direction/confidence/status badges, close modal (JS) |
| `decisions.html` | Decision stats cards, decisions table with reasoning sub-rows, outcome coloring |
| `attribution.html` | Attribution scores table (category, sample_size, outcomes, win rates) |
| `performance.html` | 4 metric cards, Chart.js equity curve (90d), Chart.js cash vs invested (stacked) |

### 6.4 Frontend Stack

- **CSS:** Tailwind CSS v3 (CDN)
- **Charts:** Chart.js (CDN) - line charts for equity curve + allocation
- **JS:** Vanilla JS only (thesis close modal, chart initialization)
- **Data refresh:** Server-side rendering, full page reload on actions

---

## 7. Complete Database Schema

### SQL Migration Files

- `db/init/001_schema.sql` - Core tables
- `db/init/002_theses.sql` - Thesis tracking
- `db/init/005_redesign.sql` - Playbooks, decision_signals, signal_attribution (drops: documents, strategy)
- `db/migrations/001_widen_ticker_columns.sql` - VARCHAR(10) → VARCHAR(128)

### All Tables

#### `news_signals`
```sql
CREATE TABLE news_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(128) NOT NULL,
    headline TEXT NOT NULL,
    category VARCHAR(20),           -- earnings, guidance, analyst, product, legal, noise
    sentiment VARCHAR(10),          -- bullish, bearish, neutral
    confidence VARCHAR(10),         -- high, medium, low
    published_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_news_signals_ticker ON news_signals(ticker);
CREATE INDEX idx_news_signals_published_at ON news_signals(published_at);
```

#### `macro_signals`
```sql
CREATE TABLE macro_signals (
    id SERIAL PRIMARY KEY,
    headline TEXT NOT NULL,
    category VARCHAR(20),           -- fed, trade, regulation, geopolitical, fiscal, election
    affected_sectors TEXT[],        -- tech, finance, energy, healthcare, defense, all
    sentiment VARCHAR(10),          -- bullish, bearish, neutral
    published_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_macro_signals_category ON macro_signals(category);
CREATE INDEX idx_macro_signals_published_at ON macro_signals(published_at);
```

#### `positions`
```sql
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(128) NOT NULL UNIQUE,
    shares DECIMAL NOT NULL,
    avg_cost DECIMAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_positions_ticker ON positions(ticker);
```

#### `open_orders`
```sql
CREATE TABLE open_orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL UNIQUE,
    ticker VARCHAR(128) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    qty DECIMAL NOT NULL,
    filled_qty DECIMAL DEFAULT 0,
    limit_price DECIMAL,
    stop_price DECIMAL,
    status VARCHAR(20) NOT NULL,
    submitted_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_open_orders_ticker ON open_orders(ticker);
CREATE INDEX idx_open_orders_status ON open_orders(status);
```

#### `account_snapshots`
```sql
CREATE TABLE account_snapshots (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    cash DECIMAL NOT NULL,
    portfolio_value DECIMAL NOT NULL,
    buying_power DECIMAL NOT NULL,
    long_market_value DECIMAL,
    short_market_value DECIMAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_account_snapshots_date ON account_snapshots(date);
```

#### `decisions`
```sql
CREATE TABLE decisions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(128) NOT NULL,
    action VARCHAR(10) NOT NULL,        -- buy, sell, hold
    quantity DECIMAL,
    price DECIMAL,
    reasoning TEXT,
    signals_used JSONB,
    account_equity DECIMAL,
    buying_power DECIMAL,
    outcome_7d DECIMAL,                 -- backfilled
    outcome_30d DECIMAL                 -- backfilled
);
CREATE INDEX idx_decisions_date ON decisions(date);
CREATE INDEX idx_decisions_ticker ON decisions(ticker);
```

#### `theses`
```sql
CREATE TABLE theses (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(128) NOT NULL,
    direction VARCHAR(10) NOT NULL,      -- long, short, avoid
    thesis TEXT NOT NULL,
    entry_trigger TEXT,
    exit_trigger TEXT,
    invalidation TEXT,
    confidence VARCHAR(10),              -- high, medium, low
    source VARCHAR(20) DEFAULT 'ideation',
    status VARCHAR(20) DEFAULT 'active', -- active, executed, invalidated, expired
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    close_reason TEXT
);
CREATE INDEX idx_theses_ticker ON theses(ticker);
CREATE INDEX idx_theses_status ON theses(status);
CREATE INDEX idx_theses_created_at ON theses(created_at);
```

#### `playbooks`
```sql
CREATE TABLE playbooks (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE,
    market_outlook TEXT,
    priority_actions JSONB,
    watch_list TEXT[],
    risk_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_playbooks_date ON playbooks(date);
```

#### `decision_signals`
```sql
CREATE TABLE decision_signals (
    decision_id INT REFERENCES decisions(id),
    signal_type TEXT NOT NULL,          -- 'news_signal', 'macro_signal', 'thesis'
    signal_id INT NOT NULL,
    PRIMARY KEY (decision_id, signal_type, signal_id)
);
CREATE INDEX idx_decision_signals_signal ON decision_signals(signal_type, signal_id);
```

#### `signal_attribution`
```sql
CREATE TABLE signal_attribution (
    id SERIAL PRIMARY KEY,
    category TEXT UNIQUE NOT NULL,
    sample_size INT,
    avg_outcome_7d NUMERIC(8,4),
    avg_outcome_30d NUMERIC(8,4),
    win_rate_7d NUMERIC(5,4),
    win_rate_30d NUMERIC(5,4),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Dropped Tables (from 005_redesign)
- `documents` - pgvector document store (RAG removed)
- `strategy` - replaced by playbooks

### Table Relationships
```
decisions ←──FK── decision_signals ──refs──→ news_signals
                                    ──refs──→ macro_signals
                                    ──refs──→ theses

decision_signals ──agg──→ signal_attribution (computed)
decisions ──agg──→ signal_attribution (via patterns/attribution)
```

---

## 8. All SQL Query Patterns

### UPSERT Pattern (used heavily)
```sql
INSERT INTO table (cols) VALUES (vals)
ON CONFLICT (unique_col) DO UPDATE SET col1 = EXCLUDED.col1, ...
RETURNING id
```
Used for: positions, open_orders, account_snapshots, playbooks, signal_attribution, decision_signals.

### Batch Insert (execute_values)
```python
from psycopg2.extras import execute_values
execute_values(cur, "INSERT INTO table (cols) VALUES %s", data_list)
```
Used for: news_signals, macro_signals, decision_signals.

### Time-Window Joins (patterns.py)
```sql
-- Signal-decision correlation with time window
JOIN news_signals ns ON ns.ticker = d.ticker
    AND ns.published_at::date <= d.date
    AND ns.published_at::date >= d.date - INTERVAL '7 days'
```

### Attribution CTE (attribution.py)
```sql
WITH categorized AS (
    SELECT ds.decision_id,
           CASE WHEN ds.signal_type = 'news_signal'
                THEN 'news_signal:' || COALESCE(ns.category, 'unknown')
                WHEN ds.signal_type = 'macro_signal'
                THEN 'macro_signal:' || COALESCE(ms.category, 'unknown')
                ELSE ds.signal_type END AS category,
           d.outcome_7d, d.outcome_30d
    FROM decision_signals ds
    JOIN decisions d ON d.id = ds.decision_id
    LEFT JOIN news_signals ns ON ds.signal_type = 'news_signal' AND ns.id = ds.signal_id
    LEFT JOIN macro_signals ms ON ds.signal_type = 'macro_signal' AND ms.id = ds.signal_id
    WHERE d.action IN ('buy', 'sell')
)
SELECT category, COUNT(DISTINCT decision_id) AS sample_size,
       AVG(outcome_7d), AVG(outcome_30d),
       AVG(CASE WHEN outcome_7d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_7d,
       AVG(CASE WHEN outcome_30d > 0 THEN 1.0 ELSE 0.0 END) AS win_rate_30d
FROM categorized WHERE outcome_7d IS NOT NULL
GROUP BY category ORDER BY sample_size DESC
```

### Dynamic UPDATE (db.py update_thesis)
```python
updates = []
params = []
if thesis is not None:
    updates.append("thesis = %s")
    params.append(thesis)
# ... more optional fields
sql = f"UPDATE theses SET {', '.join(updates)}, updated_at = NOW() WHERE id = %s"
```

### Dashboard Aggregations
```sql
-- Signal summary with sentiment counts
SELECT ticker,
    SUM(CASE WHEN sentiment = 'bullish' THEN 1 ELSE 0 END) as bullish,
    SUM(CASE WHEN sentiment = 'bearish' THEN 1 ELSE 0 END) as bearish,
    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
    COUNT(*) as total
FROM news_signals WHERE published_at > NOW() - INTERVAL '%s days'
GROUP BY ticker ORDER BY total DESC

-- Performance metrics via CTE
WITH data AS (
    SELECT date, portfolio_value,
           FIRST_VALUE(portfolio_value) OVER (ORDER BY date) as start_val,
           FIRST_VALUE(date) OVER (ORDER BY date) as start_date
    FROM account_snapshots WHERE date > CURRENT_DATE - INTERVAL '30 days'
)
SELECT start_val, start_date, portfolio_value as end_val, date as end_date,
       portfolio_value - start_val as pnl,
       ((portfolio_value - start_val) / start_val * 100) as pnl_pct
FROM data ORDER BY date DESC LIMIT 1
```

---

## 9. External API Integrations

### Alpaca (3 clients)

| Client | Package | Purpose | Auth |
|--------|---------|---------|------|
| `TradingClient` | `alpaca.trading` | Account, positions, orders, submit orders | ALPACA_API_KEY + ALPACA_SECRET_KEY |
| `StockHistoricalDataClient` | `alpaca.data.historical` | Bars, quotes, snapshots | ALPACA_API_KEY + ALPACA_SECRET_KEY |
| `NewsClient` | `alpaca.data.news` | News feed | ALPACA_API_KEY + ALPACA_SECRET_KEY |

**Base URL:** `ALPACA_BASE_URL` (paper: `https://paper-api.alpaca.markets`)

### Anthropic Claude (2 models)

| Model | Purpose | Used In |
|-------|---------|---------|
| `claude-opus-4-6` | Strategist/ideation (autonomous research, thesis management, playbook writing) | `ideation_claude.py` |
| `claude-haiku-4-5-20251001` | Trading executor (fast, cheap decisions) | `agent.py` |

**Features:** Prompt caching (ephemeral), tool use, retry with exponential backoff, agentic loop.

### Ollama (2 models)

| Model | Purpose | Used In |
|-------|---------|---------|
| `qwen2.5:14b` | News classification, legacy ideation | `classifier.py`, `ideation.py` |
| `nomic-embed-text` | Embedding for relevance filtering | `filter.py` |

**API:** HTTP at `OLLAMA_URL` (default: `http://localhost:11434`)
- `/api/embeddings` - Single embedding
- `/api/embed` - Batch embedding
- `/api/chat` - Chat completion

---

## 10. Infrastructure & Docker

### docker-compose.yml

```yaml
services:
  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:16
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: pg_isready
      interval: 5s, retries: 5

  trading:
    build: .
    depends_on: [ollama, db]
    environment:
      DATABASE_URL, OLLAMA_URL, ALPACA_*, ANTHROPIC_API_KEY
    volumes: [./logs:/app/logs]

  dashboard:
    build: ./dashboard
    ports: ["3000:3000"]
    depends_on: [db]
    environment: [DATABASE_URL]
```

### Dockerfiles

**trading/Dockerfile:** python:3.12-slim → pip install → copy trading/ → `python -m trading.main`
**dashboard/Dockerfile:** python:3.12-slim → pip install → copy . → expose 3000 → `python app.py`

### Requirements

**trading/requirements.txt:**
```
alpaca-py>=0.21.0
anthropic>=0.40.0
ollama>=0.4.0
psycopg2-binary>=2.9.9
httpx>=0.27.0
requests>=2.31.0
python-dotenv>=1.0.0
pytz>=2025.2
numpy>=1.24.0
pgvector>=0.2.0
pytest>=8.0.0
pytest-cov>=5.0.0
```

**dashboard/requirements.txt:**
```
flask>=3.0.0
psycopg2-binary>=2.9.9
pytz>=2025.2
```

### Environment Variables

```bash
# Alpaca
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Claude
ANTHROPIC_API_KEY=...

# Database
POSTGRES_USER=algo
POSTGRES_PASSWORD=algo
POSTGRES_DB=trading
DATABASE_URL=postgresql://algo:algo@db:5432/trading

# Ollama
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:14b
```

### Shell Scripts

- `run-docker.sh` - Starts compose, waits 5s, runs command, traps EXIT to clean up
- `scripts/setup-ollama.sh` - Pulls qwen3:14b (~10GB) and nomic-embed-text (~270MB)

---

## 11. Test Suite

### Overview

- **782 tests** across **24 test files**
- **89% code coverage**
- All external dependencies fully mocked
- Run: `python3 -m pytest tests/` or `python3 -m pytest tests/ --cov=trading --cov=dashboard`

### Test Files by Module

| Test File | Module Tested | Tests | Key Coverage |
|-----------|--------------|-------|-------------|
| `test_agent.py` | `agent.py` | ~20 | Dataclasses, decision parsing, validation |
| `test_attribution.py` | `attribution.py` | ~8 | Attribution computation, upsert |
| `test_backfill.py` | `backfill.py` | ~25 | Price fetching, outcome calculation, backfill flow |
| `test_classifier.py` | `classifier.py` | ~40 | Single/batch/ticker classification, JSON parsing |
| `test_claude_client.py` | `claude_client.py` | ~15 | Retry logic, agentic loop, token tracking |
| `test_context.py` | `context.py` | ~20 | All context builder functions |
| `test_dashboard.py` | `dashboard/app.py` | ~15 | All routes, JSON APIs |
| `test_db.py` | `db.py` | ~100+ | All CRUD operations |
| `test_db_redesign.py` | `db.py` (new funcs) | ~10 | Playbooks, decision_signals, attribution |
| `test_executor.py` | `executor.py` | ~20 | Account, sync, order execution |
| `test_filter.py` | `filter.py` | ~12 | Embedding relevance filtering |
| `test_full_cycle.py` | Integration | ~10 | Strategist → executor data flow |
| `test_ideation.py` | `ideation.py` | ~15 | Thesis generation, review |
| `test_ideation_claude.py` | `ideation_claude.py` | ~12 | Claude ideation, action counting |
| `test_learn.py` | `learn.py` | ~18 | Learning loop orchestration, error isolation |
| `test_market_data.py` | `market_data.py` | ~20 | Sectors, indices, movers, volume |
| `test_model_integration.py` | All prompts | ~15 | Integration tests (requires Ollama) |
| `test_news.py` | `news.py` | ~24 | News fetching, dataclass conversion |
| `test_ollama.py` | `ollama.py` | ~25 | Embed, chat, similarity, health |
| `test_patterns.py` | `patterns.py` | ~25 | All analysis functions |
| `test_pipeline.py` | `pipeline.py` | ~15 | Full pipeline flow |
| `test_session.py` | `session.py` | ~25 | Session orchestration, error isolation |
| `test_tools.py` | `tools.py` | ~20 | All tool handlers |
| `test_trader.py` | `trader.py` | ~15 | Trading session flow |

### Key Test Patterns

**Database mocking (conftest.py):**
```python
@pytest.fixture
def mock_cursor():
    cursor = MagicMock()
    cursor.fetchone.return_value = {"id": 1}
    cursor.fetchall.return_value = []
    cursor.rowcount = 1
    return cursor

@pytest.fixture
def mock_db(mock_cursor):
    @contextmanager
    def _get_cursor():
        yield mock_cursor
    with patch("trading.db.get_cursor", _get_cursor):
        yield mock_cursor
```

**Dashboard mocking (sys.modules injection):**
```python
mock_queries = MagicMock()
sys.modules["queries"] = mock_queries
from dashboard.app import app
# Important: don't use reset_mock() on mock_queries (breaks bound references)
```

**Factory functions (conftest.py):**
```python
make_news_item()           # → NewsItem dataclass
make_trading_decision()    # → TradingDecision with thesis_id + signal_refs
make_agent_response()      # → AgentResponse with decisions + invalidations
make_ticker_signal()       # → TickerSignal
make_position_row()        # → dict matching DB schema
make_thesis_row()          # → dict
make_decision_row()        # → dict
make_snapshot_row()        # → dict
make_news_signal_row()     # → dict
make_macro_signal_row()    # → dict
make_playbook_row()        # → dict
make_attribution_row()     # → dict
```

### Coverage Gaps
- CLI `main()` argparse wrappers (integration entry points)
- `trading/main.py` connectivity check (requires live services)
- `dashboard/queries.py` bypassed (mocked at module level)

---

## 12. Constants & Configuration

| Constant | Value | Location | Usage |
|----------|-------|----------|-------|
| `DEFAULT_EXECUTOR_MODEL` | `"claude-haiku-4-5-20251001"` | `agent.py` | Trading decisions |
| `API_MAX_RETRIES` | `3` | `claude_client.py` | Claude API retry |
| `API_RATE_LIMIT_DELAY` | `60` (sec) | `claude_client.py` | Rate limit backoff |
| `API_RETRY_BASE_DELAY` | `2` (sec) | `claude_client.py` | Non-rate-limit backoff |
| `SECTOR_ETFS` | 11 mappings | `market_data.py` | Sector performance |
| `INDEX_ETFS` | `[SPY, QQQ, IWM]` | `market_data.py` | Index tracking |
| `FILE_LOGGERS` | 5 modules | `log_config.py` | Dedicated log files |
| `DEFAULT_STRATEGY_CONTEXT` | Strategy text | `filter.py` | News relevance |
| `RETRYABLE_ERRORS` | 3 error types | `claude_client.py` | Retry logic |

### LLM Prompt Constants

| Constant | Location | Purpose |
|----------|----------|---------|
| `TRADING_SYSTEM_PROMPT` | `agent.py` | Trading rules + JSON output format for Haiku |
| `CLASSIFICATION_PROMPT` | `classifier.py` | Single headline classification |
| `TICKER_CLASSIFICATION_PROMPT` | `classifier.py` | Ticker-specific classification |
| `BATCH_CLASSIFICATION_PROMPT` | `classifier.py` | Batch classification |
| `IDEATION_SYSTEM_PROMPT` | `ideation.py` | Ollama ideation agent |
| `CLAUDE_IDEATION_SYSTEM` | `ideation_claude.py` | Claude research agent |
| `CLAUDE_STRATEGIST_SYSTEM` | `ideation_claude.py` | Daily strategist (post-market) |
| `CLAUDE_SESSION_STRATEGIST_SYSTEM` | `ideation_claude.py` | Intra-session strategist |

---

## 13. Known Gaps & Technical Debt

### Architecture
- **No connection pooling:** Each `get_cursor()` creates a new connection
- **Duplicate DB layers:** `trading/db.py` and `dashboard/queries.py` both have `get_cursor()` + overlapping queries
- **No ORM:** All raw SQL means schema changes require manual query updates everywhere
- **No migrations framework:** Manual SQL files in `db/init/`, numbering gaps (001, 002, 005)
- **pgvector installed but unused:** Documents table was dropped in 005_redesign

### Reliability
- **No retry on DB operations:** Only Claude API has retry logic
- **No health checks for trading container:** Only postgres has healthcheck
- **No deadlock handling:** Concurrent upserts could conflict
- **Session error isolation is coarse:** Each stage is all-or-nothing

### Learning Loop
- **Pattern analysis uses time-window JOINs** (not decision_signals FK): `patterns.py` joins decisions ↔ news_signals by ticker + date range, while `attribution.py` uses the proper `decision_signals` FK
- **success_rate in thesis_stats is hardcoded to None** (TODO)
- **No automated scheduling:** Requires manual or cron-based invocation

### Dashboard
- **Server-side rendering only:** No WebSocket/SSE for real-time updates
- **No authentication:** Dashboard is publicly accessible on port 3000
- **Chart.js data embedded in HTML:** Large datasets could bloat page size
- **Thesis close modal uses full page reload** instead of client-side update

### Testing
- **No integration tests with real DB:** All DB is mocked
- **Model integration tests require live Ollama:** Marked with `@integration`
- **Dashboard queries.py untested directly:** Mocked at module level

### Cost & Performance
- **Claude Opus for strategist is expensive:** ~$5/1M input + $25/1M output
- **Ollama classification is sequential per batch:** Could parallelize
- **No caching of market data:** Sector/index performance fetched fresh each time
- **Embedding batches are unbounded:** Large news volumes could OOM

---

## File Tree Summary

```
algo/
├── CLAUDE.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── run-docker.sh
├── scripts/
│   └── setup-ollama.sh
├── db/
│   ├── init/
│   │   ├── 001_schema.sql
│   │   ├── 002_theses.sql
│   │   └── 005_redesign.sql
│   └── migrations/
│       └── 001_widen_ticker_columns.sql
├── trading/
│   ├── __init__.py
│   ├── agent.py              # LLM trading decisions (Claude Haiku)
│   ├── attribution.py        # Signal performance analysis
│   ├── backfill.py           # Outcome backfill (7d/30d P&L)
│   ├── classifier.py         # News classification (Qwen)
│   ├── claude_client.py      # Claude API + agentic loop
│   ├── context.py            # Context builder for LLM
│   ├── db.py                 # All database operations
│   ├── executor.py           # Alpaca trade execution
│   ├── filter.py             # Embedding relevance filtering
│   ├── ideation.py           # Ollama-based ideation (legacy)
│   ├── ideation_claude.py    # Claude strategist + ideation
│   ├── learn.py              # Learning loop orchestrator
│   ├── log_config.py         # Logging setup
│   ├── main.py               # Connectivity check
│   ├── market_data.py        # Market data (sectors, movers)
│   ├── news.py               # Alpaca news fetching
│   ├── ollama.py             # Local LLM + embeddings
│   ├── patterns.py           # Pattern analysis
│   ├── pipeline.py           # News processing pipeline
│   ├── session.py            # Daily session orchestrator
│   ├── tools.py              # Claude tool definitions + handlers
│   └── trader.py             # Trading session executor
├── dashboard/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                # Flask routes
│   ├── queries.py            # Dashboard DB queries
│   └── templates/
│       ├── base.html
│       ├── portfolio.html
│       ├── playbook.html
│       ├── signals.html
│       ├── theses.html
│       ├── decisions.html
│       ├── attribution.html
│       └── performance.html
├── tests/
│   ├── conftest.py           # Fixtures + factory functions
│   ├── test_agent.py
│   ├── test_attribution.py
│   ├── test_backfill.py
│   ├── test_classifier.py
│   ├── test_claude_client.py
│   ├── test_context.py
│   ├── test_dashboard.py
│   ├── test_db.py
│   ├── test_db_redesign.py
│   ├── test_executor.py
│   ├── test_filter.py
│   ├── test_full_cycle.py
│   ├── test_ideation.py
│   ├── test_ideation_claude.py
│   ├── test_learn.py
│   ├── test_market_data.py
│   ├── test_model_integration.py
│   ├── test_news.py
│   ├── test_ollama.py
│   ├── test_patterns.py
│   ├── test_pipeline.py
│   ├── test_session.py
│   ├── test_tools.py
│   └── test_trader.py
└── logs/
```
