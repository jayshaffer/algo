# System Redesign: Strategic/Tactical Split with Learning Loop

**Date:** 2026-02-05
**Status:** Design

## Motivation

The current system has two key weaknesses:

1. **Decision quality** — The Ollama trader only sees pre-classified signals (headline + sentiment score). It never reads the RAG documents that are being ingested. Meanwhile, Claude ideation has web search and produces richer analysis, but its output loosely connects to what Ollama acts on.

2. **No learning loop** — Decision outcomes get backfilled (7d/30d P&L), but the system never asks "which signals were actually predictive?" There's no feedback mechanism that improves future decisions based on past results.

## Design Principles

- Claude is the strategist. Ollama is the executor. They share state through structured database tables, not loose context strings.
- Learning is quantitative. Signal attribution scores and thesis win rates live in queryable tables, not fuzzy journal entries.
- The RAG pipeline (document ingestion, embeddings, pgvector) is removed. Claude uses web search for real-time research. Ollama uses structured signals + theses. The embedding infrastructure wasn't paying off.
- The system degrades gracefully. If Claude's session fails, Ollama falls back to conservative mode.

## Daily Cycle

```
Market Close (4:00 PM ET)
         |
         v
+-------------------------------------+
|         CLAUDE STRATEGIST           |
|         (runs ~4:30 PM ET)          |
|                                     |
|  1. Sync positions & orders         |
|  2. Backfill decision outcomes      |
|  3. Compute signal attribution      |
|  4. Self-reflect on recent results  |
|  5. Review/update/close theses      |
|     (web search for research)       |
|  6. Generate new theses if needed   |
|  7. Write daily playbook:           |
|     - Priority actions for tomorrow |
|     - Tickers to watch              |
|     - Risk notes                    |
|  8. Update learning metrics         |
|                                     |
|  Outputs: theses, playbook,         |
|           signal scores, metrics    |
+-------------------+-----------------+
                    | (stored in DB)
                    v
           +----------------+
           |   PostgreSQL   |
           +--------+-------+
                    |
                    v
+-------------------------------------+
|         OLLAMA EXECUTOR             |
|         (runs ~9:15 AM ET)          |
|                                     |
|  1. Read today's playbook           |
|  2. Read active theses              |
|  3. Read overnight signals          |
|  4. Read signal attribution scores  |
|  5. Build compressed context        |
|  6. Make buy/sell/hold decisions     |
|  7. Validate & execute trades       |
|  8. Log decisions with signal refs  |
|                                     |
|  Inputs: playbook, theses, signals, |
|          attribution scores         |
+-------------------------------------+
```

Claude and Ollama share state through the database. Claude writes structured outputs (playbook, theses, metrics) that Ollama reads as first-class inputs.

## New Database Tables

### `playbooks` — Daily trading plan from Claude to Ollama

```sql
CREATE TABLE playbooks (
    id              SERIAL PRIMARY KEY,
    date            DATE UNIQUE,
    market_outlook  TEXT,
    priority_actions JSONB,
    watch_list      TEXT[],
    risk_notes      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

Example `priority_actions`:

```json
[
    {
        "ticker": "NVDA",
        "action": "buy",
        "thesis_id": 42,
        "reasoning": "Entry trigger hit: pulled back to $118 support. Thesis still valid.",
        "max_quantity": 5,
        "confidence": 0.8
    },
    {
        "ticker": "XLE",
        "action": "sell",
        "thesis_id": 38,
        "reasoning": "Oil thesis invalidated by OPEC production increase. Cut position.",
        "confidence": 0.9
    }
]
```

### `decision_signals` — Links decisions to the signals/theses that motivated them

```sql
CREATE TABLE decision_signals (
    decision_id  INT REFERENCES decisions(id),
    signal_type  TEXT,        -- 'news_signal', 'macro_signal', 'thesis'
    signal_id    INT,
    PRIMARY KEY (decision_id, signal_type, signal_id)
);
```

### `signal_attribution` — Precomputed scores showing which signal types are predictive

```sql
CREATE TABLE signal_attribution (
    id              SERIAL PRIMARY KEY,
    category        TEXT UNIQUE, -- e.g. 'news:earnings', 'macro:fed_policy', 'thesis'
    sample_size     INT,
    avg_outcome_7d  NUMERIC(8,4),
    avg_outcome_30d NUMERIC(8,4),
    win_rate_7d     NUMERIC(5,4),
    win_rate_30d    NUMERIC(5,4),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

This table is small (20-30 rows covering signal categories). Claude recomputes it daily from `decision_signals` joined with `decisions`. Ollama reads it as a concise summary in its context.

### Removed tables

- `documents` — RAG document store (pgvector embeddings)
- `strategy` — replaced by `playbooks`

## Signal Attribution: How Learning Works

1. **At decision time** — Ollama cites specific signal IDs and thesis IDs in every decision. These populate `decision_signals`.

2. **At backfill time** (7d and 30d) — Once outcomes are known, they propagate back through `decision_signals` to the signals and theses that were cited.

3. **Attribution scoring** — SQL aggregation groups by signal category and computes:
   - Average 7d/30d return for decisions citing that category
   - Win rate (% of decisions with positive outcome)
   - Sample size

4. **Feeding it back** — Attribution scores become a section in Ollama's context: "These signal types have been predictive: [...]. These have not: [...]." This is a precomputed summary, not RAG retrieval.

Example insight the system could surface:
- "Thesis-driven buys had 62% win rate vs 41% for non-thesis buys"
- "Macro signals in 'fed_policy' category were cited in 8 decisions, avg outcome -1.3%"
- "Ticker signals with sentiment > 0.7 led to +2.1% avg 7d return across 14 decisions"

## Claude Strategist Session (Post-Market)

The expanded `ideation_claude.py` runs Claude's daily strategic session.

### Steps

1. **Sync & snapshot** — Pull positions/orders from Alpaca, take account snapshot. Same as today.

2. **Backfill outcomes** — Run existing 7d/30d outcome computation. Propagate outcomes to `decision_signals` for attribution.

3. **Compute signal attribution** — SQL aggregation over `decisions` joined with `decision_signals`. Group by signal category, compute avg outcomes and win rates. Upsert into `signal_attribution`.

4. **Self-reflection** — Claude receives:
   - Today's decisions and their reasoning
   - Recent outcomes (7d/30d) on past decisions
   - Current attribution scores
   - Active theses and their status

   Prompt: *"Review today's trading activity and recent outcomes. What patterns do you see? What's working? What isn't? Use this analysis to inform your thesis updates and tomorrow's playbook."*

   The reflection is ephemeral — used within this session to inform the next steps, not stored.

5. **Thesis management** — Using the existing tool loop (web search, get_market_snapshot, etc.), Claude:
   - Reviews each active thesis against current data
   - Closes theses that hit exit/invalidation triggers
   - Updates theses with new information
   - Generates new theses where opportunities exist

6. **Write playbook** — Based on active theses, attribution scores, and reflection, Claude writes tomorrow's playbook: priority actions, watch list, risk notes.

### Tools available

Carried from current system:
- `web_search` — real-time research
- `get_market_snapshot` — sectors, indices, movers
- `get_portfolio_state` — holdings, cash, P&L
- `get_active_theses` / `create_thesis` / `update_thesis` / `close_thesis`
- `get_news_signals` — ticker-specific signals
- `get_macro_context` — macro signals

New tools:
- `get_signal_attribution` — read current attribution scores
- `get_decision_history` — recent decisions with outcomes
- `write_playbook` — write tomorrow's playbook to DB

### Token budget

This session is larger than today's ideation — roughly 50-80k tokens with tool loops. At Claude's pricing that's approximately $0.50-1.00/day.

## Ollama Executor Session (Pre-Market)

The morning trading session gets simpler and more focused.

### Steps

1. **Read playbook** — Fetch today's playbook from DB. If none exists (Claude session failed), fall back to conservative mode: hold everything, no new positions.

2. **Read context** — Compressed context built from:
   - Playbook (market outlook, priority actions, risk notes)
   - Active theses (entry/exit triggers, confidence)
   - Overnight signals (news_signals + macro_signals since last session)
   - Signal attribution summary (top 5 most/least predictive categories)
   - Portfolio state (positions, cash, buying power)

3. **Make decisions** — System prompt:

   > "You are executing a trading playbook prepared by a senior strategist. For each priority action, decide: execute as-is, adjust quantity/timing, or skip (with reason). You may propose additional trades if overnight signals warrant it, but playbook actions come first. For every decision, cite which signal IDs and thesis IDs informed it."

   Ollama must cite signal IDs and thesis IDs in its response. This populates `decision_signals` and closes the attribution loop.

4. **Validate & execute** — Same as today: check buying power, check shares held, submit market orders via Alpaca. Dry-run mode preserved.

5. **Log decisions** — Insert to `decisions` table. Insert rows into `decision_signals` linking each decision to its cited signals and theses.

### Differences from current system

- Ollama's job shrinks from "analyze everything and decide" to "execute a curated plan with judgment"
- Signal/thesis citation is required, not optional — this feeds the learning loop
- Playbook fallback mode means graceful degradation if Claude's session fails
- Context is smaller and higher-signal (no raw macro dump, just what Claude flagged)

## What Gets Removed

### Files removed
- `trading/ingest.py` — document chunking, embedding, ingestion
- `trading/ingest_scheduler.py` — scheduled ingestion job
- `trading/retrieval.py` — vector search with time decay
- `trading/ollama.py` — only used for embeddings; chat already handled by `agent.py`

### Database removed
- `documents` table
- `strategy` table (replaced by `playbooks`)
- pgvector extension

### Docker changes
- Ollama container stays (runs qwen2.5:14b) but drops `nomic-embed-text` model
- PostgreSQL drops pgvector extension setup

### What stays unchanged
- `trading/news.py`, `trading/classifier.py`, `trading/filter.py`, `trading/pipeline.py` — signal pipeline
- `trading/market_data.py` — sector ETFs, snapshots, movers
- `trading/executor.py` — Alpaca order execution
- `trading/db.py` — database layer

### What gets modified
- `trading/context.py` — reads playbook + theses + signals + attribution instead of building everything from scratch
- `trading/agent.py` — new system prompt oriented around playbook execution
- `trading/ideation_claude.py` — expanded to include daily review, playbook generation, and attribution computation
- `trading/trader.py` — reordered to read playbook first, simplified decision flow
- `trading/backfill.py` — extended to propagate outcomes to `decision_signals`

## Architecture Diagram (New)

```
+-------------------------------------------------------------+
|                    Docker Compose Stack                      |
+--------------+--------------+---------------+---------------+
|   Ollama     |  PostgreSQL  |   Trading     |   Dashboard   |
|  (qwen2.5)  |   (no       |   Agent       |   (Flask)     |
|  :11434      |   pgvector) |               |   :3000       |
+--------------+--------------+---------------+---------------+

PostgreSQL Tables:
  positions          - synced from Alpaca
  open_orders        - synced from Alpaca
  account_snapshots  - daily equity curve
  news_signals       - classified ticker signals
  macro_signals      - classified macro signals
  theses             - trade ideas with triggers
  playbooks          - daily plans (Claude -> Ollama)      [NEW]
  decisions          - trading journal with outcomes
  decision_signals   - links decisions to their inputs     [NEW]
  signal_attribution - what signal types are predictive    [NEW]

Cron Schedule:
  4:30 PM ET  - Claude strategist session
  9:15 AM ET  - Ollama executor session
```

## Dashboard Changes

New pages/sections:
- **Playbook view** — today's playbook with priority actions, watch list, risk notes
- **Attribution dashboard** — signal category performance table (win rates, avg returns, sample size)
- **Learning metrics** — trend of attribution scores over time

Existing pages gain:
- **Decisions page** — shows which signals/theses were cited per decision
- **Theses page** — shows win/loss stats per thesis (from attribution data)
