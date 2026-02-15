# Persistent Strategy Layer Design

**Date:** 2026-02-15
**Branch:** strategy_layer
**Approach:** Session-End Strategy Reflection (Approach 1)

## Problem

The v2 pipeline recomputes attribution constraints fresh each session. There is no persistent identity — the system doesn't remember what kind of trader it's becoming, what strategic reasoning led to past decisions, or what rules it has learned over time. Each session starts from scratch beyond raw attribution statistics.

## Goals

- **Persistent identity:** The system develops a durable trading identity that emerges from performance data
- **Strategy memory:** Session-by-session reasoning is preserved and queryable
- **Learning accumulation:** Learned rules persist and shape future behavior until explicitly retired

## Design

### Database Schema

Three new tables.

#### `strategy_state`

Single active row representing the current strategy identity. Versioned — old identities kept for history.

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| identity_text | TEXT | LLM-written narrative: what kind of trader the system is becoming |
| risk_posture | VARCHAR(20) | `conservative`, `moderate`, `aggressive` |
| sector_biases | JSONB | e.g., `{"tech": "overweight", "energy": "avoid"}` |
| preferred_signals | JSONB | e.g., `["earnings", "fed"]` |
| avoided_signals | JSONB | e.g., `["legal"]` |
| version | INT | Monotonically increasing |
| is_current | BOOLEAN | Only one row is true |
| created_at | TIMESTAMPTZ | |

#### `strategy_rules`

Accumulated rules the LLM extracts from performance data.

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| rule_text | TEXT | Human-readable (e.g., "Fade legal news — 38% win rate over 12 trades") |
| category | VARCHAR(64) | Domain (e.g., `news_signal:legal`, `position_sizing`) |
| direction | VARCHAR(10) | `constraint` or `preference` |
| confidence | DECIMAL | LLM-assigned, 0.0–1.0 |
| supporting_evidence | TEXT | LLM's reasoning for why this rule exists |
| status | VARCHAR(20) | `active`, `retired` |
| created_at | TIMESTAMPTZ | |
| retired_at | TIMESTAMPTZ | |

#### `strategy_memos`

Session-by-session reasoning log.

| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| session_date | DATE | |
| memo_type | VARCHAR(20) | `reflection`, `rule_change`, `identity_update` |
| content | TEXT | LLM-written memo |
| strategy_state_id | INT FK | Which identity version was active |
| created_at | TIMESTAMPTZ | |

### Session Flow

Strategy reflection is a new **Stage 4** at the end of `run_session()`:

```
Stage 0: Learning Refresh (backfill, attribution)
Stage 1: News Pipeline (fetch, classify, store)
Stage 2: Strategist (thesis management, playbook)
Stage 3: Executor (decisions, trades)
Stage 4: Strategy Reflection (NEW)
    ├─ Load current strategy_state + strategy_rules
    ├─ Load today's session results (decisions, outcomes, attribution)
    ├─ Load recent strategy_memos (last 5)
    ├─ Opus reflects via tool calls
    └─ Persist updates to DB
```

The feedback loop across sessions:

```
Stage 4 (yesterday) writes strategy state
        ↓
Stage 2 (today) reads strategy state via tools
        ↓
Stage 4 (today) reflects on what happened, updates state
```

### Strategy Reflection Stage (Stage 4)

New module: `strategy.py`

**System prompt concept:**

> You are the strategy reflection agent for an autonomous trading system. You have just completed a trading session. Your job is to review what happened, update the system's evolving trading identity, and manage its accumulated rules.
>
> You are NOT making trades. You are reflecting on performance and shaping who this system is as a trader.
>
> If this is the first session (no existing identity), bootstrap one from the attribution data and today's results.

**Write tools:**
- `propose_rule(rule_text, category, direction, confidence, supporting_evidence)` — creates a new active rule
- `retire_rule(rule_id, reason)` — marks a rule as retired, sets retired_at
- `update_strategy_identity(identity_text, risk_posture, sector_biases, preferred_signals, avoided_signals)` — creates a new versioned strategy_state row, sets is_current=true on it and false on the previous
- `write_strategy_memo(memo_type, content)` — logs the reflection

**Read tools:**
- `get_strategy_identity()` — current strategy_state row
- `get_strategy_rules()` — all active rules
- `get_strategy_history(n)` — recent memos
- `get_session_summary()` — today's decisions, outcomes, attribution

**Result dataclass:**
```python
@dataclass
class StrategyReflectionResult:
    rules_proposed: int
    rules_retired: int
    identity_updated: bool
    memo_written: bool
    input_tokens: int
    output_tokens: int
    turns_used: int
```

**Max turns:** 10

### Strategist Integration (Stage 2)

Three read-only tools added to the existing strategist tool set in `tools.py`:

- `get_strategy_identity()` — returns current strategy_state row, or null if no identity exists yet
- `get_strategy_rules()` — returns all active rules, or empty list
- `get_strategy_history(n=5)` — returns last N memos

One line added to the strategist system prompt:

> You have access to the system's evolving strategy identity, learned rules, and reflection history. Consult these when managing theses and writing the playbook.

No other prompt changes. The LLM decides when and whether to call them.

### Bootstrap & Edge Cases

**First session (cold start):**
- Read tools return null/empty. Strategist operates without strategy context (same as today).
- Stage 4 bootstraps the first identity from attribution data and the day's results.
- By session 2, the strategist has something to read.

**Dry run:**
- Stage 4 still runs. Reflection on a dry run is useful.

**Stage 4 failure:**
- Wrapped in try/except like other stages.
- SessionResult gets `strategy_result` and `strategy_error` fields.
- No impact on the trading session (already happened).

**Identity drift / rule accumulation:**
- No guardrails initially. The LLM can rewrite identity or accumulate rules freely.
- Memos provide full audit trail.
- Can add constraints later if needed.

### Cost Impact

~$0.10–0.35/day additional on top of ~$0.60–2.00/day baseline (10-20% increase). Uses Opus for the reflection stage.
