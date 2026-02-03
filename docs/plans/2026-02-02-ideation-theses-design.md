# Ideation Phase & Thesis Table Design

## Overview

Add autonomous idea generation to the trading system. The LLM will propose trade ideas based on market data and macro context, track them as theses, and act on them when entry conditions are met.

## Design Decisions

- **Theses are optional** - Signal-based trades can happen without them; theses are for longer-term conviction ideas
- **Separate scheduled job** - Ideation runs independently from trading session
- **All theses actionable** - No approval step; trading session LLM judges them
- **Rolling refresh** - Each ideation run reviews existing theses, updates or closes stale ones
- **Context: macro + portfolio + market data** - Real market context grounds the ideas

## Database Schema

```sql
-- db/init/002_theses.sql

CREATE TABLE theses (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,      -- long, short, avoid
    thesis TEXT NOT NULL,                 -- core reasoning
    entry_trigger TEXT,                   -- what would trigger entry
    exit_trigger TEXT,                    -- what would trigger exit
    invalidation TEXT,                    -- what would prove thesis wrong
    confidence VARCHAR(10),               -- high, medium, low
    source VARCHAR(20) DEFAULT 'ideation', -- ideation, manual
    status VARCHAR(20) DEFAULT 'active',  -- active, executed, invalidated, expired
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    close_reason TEXT
);

CREATE INDEX idx_theses_ticker ON theses(ticker);
CREATE INDEX idx_theses_status ON theses(status);
CREATE INDEX idx_theses_created_at ON theses(created_at);
```

## New Modules

### trading/market_data.py

Fetches market context for ideation:
- Sector performance (1d/5d) via sector ETF proxies
- Top gainers/losers (10 each)
- Unusual volume stocks (2x+ average)
- Index levels (SPY, QQQ, IWM)

### trading/ideation.py

Runs ideation session:
1. Load macro context + portfolio + active theses + market snapshot
2. Send to LLM with ideation prompt
3. Parse response: thesis reviews + new ideas
4. Update database: close invalidated, update reviewed, insert new

LLM prompt instructs it to:
- Review active theses: still valid, needs update, or invalidate
- Generate 3-5 new ideas not in portfolio or active theses
- Provide: ticker, direction, thesis, entry/exit triggers, invalidation criteria
- Mix of idea types: momentum, value, sector rotation, event-driven

## Modified Modules

### trading/db.py

Add thesis CRUD functions:
- `insert_thesis()`
- `get_active_theses()`
- `update_thesis()`
- `close_thesis()`

### trading/context.py

Add `get_theses_context()` to format active theses for trading session.

### trading/agent.py

Update `TRADING_SYSTEM_PROMPT` to include thesis handling:
- Consider entry if trigger conditions met
- May act on thesis without today's signals
- Flag theses for invalidation if conditions observed

## Entry Points

```bash
# Run ideation (separate from trading)
python -m trading.ideation --model qwen2.5:14b

# Run trading session (picks up active theses)
python -m trading.trader --dry-run
```

## Scheduling

```
# Ideation at 7am ET (before market open)
0 7 * * 1-5 cd /app && python -m trading.ideation

# Trading at 4:30pm ET (after market close)
30 16 * * 1-5 cd /app && python -m trading.trader
```

## Data Flow

```
Market Data (Alpaca)
       ↓
[market_data.py] → sector perf, movers, volume
       ↓
[ideation.py] ← macro signals + portfolio + active theses
       ↓
LLM generates/reviews theses
       ↓
[db.py] → theses table
       ↓
... later ...
       ↓
[trader.py] runs session
       ↓
[context.py] includes active theses
       ↓
LLM decides (may execute on thesis)
       ↓
[db.py] → mark thesis executed if traded
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `db/init/002_theses.sql` | Create |
| `trading/market_data.py` | Create |
| `trading/ideation.py` | Create |
| `trading/db.py` | Modify - add thesis functions |
| `trading/context.py` | Modify - add `get_theses_context()` |
| `trading/agent.py` | Modify - update prompt |
