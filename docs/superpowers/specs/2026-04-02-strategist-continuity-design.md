# Strategist Run-to-Run Continuity & Determinism

## Problem Statement

The v2 trading system has two related problems:

**1. Information flow gaps.** The strategist operates on truncated memos, loses its own reasoning between sessions, and doesn't feed context back to the reflection agent. The reflection agent writes rich 2000+ char analyses that get truncated to 200 chars before anyone reads them.

**2. Behavioral instability.** The system flip-flops: HOLD→SELL within 1-2 days, BUY→SELL reversals, re-entry cycles. Three confirmed recurrences of the same pattern (AVGO Sessions 3-5, XLE/AMZN Session 35, CRM/ANET Session 36). The reflection agent keeps proposing stricter rules (7 rules addressing cooldowns) but they're enforced only via LLM prompt text — and the LLM keeps violating them.

### Root causes

Five root causes contribute to both problems:

1. **Strategist can't read the journal.** Memos aren't pre-seeded, and when retrieved via tool they're truncated to 200 chars out of 2000-2800. The strategist is making decisions without the reflection agent's analysis.

2. **Strategist reasoning isn't persisted.** The final summary is printed to stdout but never written to the database. The reflection agent reflects on outcomes without knowing intent.

3. **Cooldown rules have no structural enforcement.** Rules #3, #6, #8, #10, #20 define cooldown periods (3-10 trading days), but they exist as prompt text for an LLM that demonstrably doesn't follow them. No code prevents a HOLD→SELL reversal.

4. **Executor runs when the strategist fails.** session.py treats all stages as independent (`session.py:14`). When Stage 2 (strategist) fails, Stage 3 (executor) still runs without a playbook. On 3/31, this produced 6 "No playbook available" HOLD decisions that got sold 1-2 days later, triggering the instability pattern the reflection agent flagged. The executor depends on the strategist — this isn't a truly independent stage.

5. **Identity is rewritten every session as a status report.** 30 identity versions in 36 sessions. Every one starts "This system is a multi-asset trader in its Nth session..." followed by session-specific detail ("Session 36 issued two SELL decisions..."). The identity is doing double duty as a session report AND a trading personality. This makes the strategist reactive to recent events rather than grounded in stable principles. Memos are the right place for session-specific detail; identity should evolve slowly.

---

## Fixes

### Fix 1: Pre-seed strategy memos in the strategist

`ideation_claude.py:281-305` pre-seeds 6 context blocks but not memos. The initial message says "do NOT re-fetch" the pre-seeded items, which may discourage the LLM from calling any tools early — including `get_strategy_history` for memos.

**Change:** Add `tool_get_strategy_history()` as a 7th pre-seeded block in `run_strategist_loop()`. Update the "do NOT re-fetch" list to include strategy history/memos.

**Files:** `v2/ideation_claude.py`  
**Effort:** Trivial — ~6 lines, same try/except pattern as the other 6.

---

### Fix 2: Two-tier memo retrieval

`tools.py:331` truncates all memos to 200 characters. The reflection agent writes 2000-2800 char analyses — multi-session behavioral arcs, rule violation patterns, portfolio phase transitions. The strategist sees only the headline.

**Change:** Modify `tool_get_strategy_history()`:
- Last 2 memos: full text (no truncation)
- Older memos (positions 3-5): truncated to 300 chars

This gives ~5-6K chars of recent reflection context versus ~1K today.

**Files:** `v2/tools.py`  
**Effort:** Low.

**Note:** Compounds with Fix 1. Fix 1 puts memos in the pre-seed; Fix 2 makes them worth reading. Do both or the value of either is halved.

---

### Fix 3: Persist strategist summary as a memo

The strategist's `final_summary` is returned in `ClaudeIdeationResult` but never hits the database. The reflection agent then reflects on outcomes without knowing why the strategist made its choices.

**Change:** In `session.py`, after Stage 2 completes successfully, write the summary to `strategy_memos` with `memo_type = 'strategist_notes'`:

```python
from .database.trading_db import insert_strategy_memo, get_current_strategy_state

state = get_current_strategy_state()
insert_strategy_memo(
    session_date=today,
    memo_type='strategist_notes',
    content=ideation_result.final_summary,
    strategy_state_id=state['id'] if state else None,
)
```

The reflection agent already reads all recent memos via `get_recent_strategy_memos()`, so it automatically sees the strategist's reasoning. No changes needed to Stage 4.

**Files:** `v2/session.py`  
**Effort:** Low — ~10 lines.

**Ordering matters:** This memo is written before the executor (Stage 3) and reflection (Stage 4). The reflection agent reads strategist intent → reviews execution outcomes against it. This is the right flow.

---

### Fix 4: Structural cooldown enforcement

The system has 7 active rules defining cooldown periods (#3, #6, #8, #10, #20). All are LLM prompt text. The LLM has violated them across 3+ confirmed recurrences, each time prompting the reflection agent to propose a stricter version of the same rule. Rule #20 is already labeled a "hard constraint" — if prompt enforcement were going to work, it would have by now.

**Change:** Add a `check_cooldowns()` function that enforces cooldown rules in code. Two enforcement points:

**a) Pre-executor filtering** (`v2/context.py:build_executor_input`): Before the executor sees playbook actions, filter out any action that targets a ticker in cooldown. This prevents the executor from wasting a decision on something that will be blocked.

**b) Post-executor validation** (`v2/trader.py`, in the validation loop at line 234): Safety net for off-playbook decisions. If the executor proposes a trade on a cooldown ticker anyway, block it before execution and log the violation.

Cooldown rules to enforce in code:

| Pattern | Cooldown | Source Rule |
|---------|----------|-------------|
| HOLD → SELL | 3 trading days | #20 |
| BUY → SELL | 3 trading days | #3 |
| SELL → BUY/HOLD | 5 trading days | #6 |
| SELL after accumulation (3+ BUYs) → BUY | 10 trading days | #10 |

The function queries recent decisions for the ticker and computes trading-day gaps. Blocked decisions are logged with reason `"cooldown_violation"` so they appear in attribution data.

**Files:** New `v2/cooldown.py` (logic), `v2/context.py` (pre-filter), `v2/trader.py` (post-validation)  
**Effort:** Moderate.

**Important interaction with Fix 6:** Fallback HOLDs from a missing-playbook session (Fix 6) must NOT create cooldown obligations. A fallback HOLD is not a deliberate strategic commitment — it's the executor's default when it has no instructions. Cooldown should only apply to decisions backed by a playbook or explicit off-playbook reasoning.

---

### Fix 5: Inject cooldown state into executor context

Even with structural enforcement (Fix 4), the executor wastes its single LLM call proposing actions on cooldown tickers. It should know upfront which tickers are blocked.

**Change:** Add a `cooldown_tickers` field to `ExecutorInput`:

```python
cooldown_tickers: dict[str, str]  # {"XLE": "SELL→BUY lockout until 2026-04-04"}
```

Build from recent decisions in `build_executor_input()` using the same logic as Fix 4. Add to `TRADING_SYSTEM_PROMPT`:

```
9. cooldown_tickers — tickers with active trading restrictions. Do NOT propose actions that violate these.
```

**Files:** `v2/agent.py` (dataclass + prompt), `v2/context.py` (build map)  
**Effort:** Low — shares cooldown logic with Fix 4.

---

### Fix 6: Don't run executor without a playbook

`session.py:14` says "Each stage is independent" and runs Stage 3 even when Stage 2 fails. But the executor *depends* on the strategist's playbook. On 3/31, no playbook was generated. The executor fell back to 6 "No playbook available" HOLD decisions. Those HOLDs then got sold within 1-2 days, triggering the HOLD→SELL instability that the reflection agent flagged as the "third recurrence."

The 3/31 incident wasn't a strategist instability problem — it was an orchestration bug producing phantom decisions that looked like instability.

**Change:** In `session.py`, skip Stage 3 if Stage 2 failed and no playbook exists for today:

```python
# Stage 3: Trading session
playbook_exists = get_playbook(today) is not None
if result.strategist_error and not playbook_exists:
    logger.warning("Strategist failed and no playbook exists — skipping executor")
    result.skipped_executor = True
```

If the strategist failed but a playbook exists from a prior run (session resume), still run the executor.

**Files:** `v2/session.py`  
**Effort:** Low — ~5 lines + import.

**Alternative considered:** Tag fallback HOLDs as `is_fallback=True` and exclude them from cooldown. Rejected — this adds schema complexity to compensate for an orchestration problem. Better to just not generate phantom decisions.

---

### Fix 7: Stabilize strategy identity

30 identity rewrites in 36 sessions. Every version is a session report:

> "This system is a multi-asset trader in its 36th session, operating in an observation-and-screening phase following a systematic liquidation (Sessions 29-33). Session 36 (2026-04-02) issued two SELL decisions..."

The identity is doing the job of memos. It should be a slowly-evolving self-description — trading style, risk philosophy, signal preferences — not a play-by-play.

**Changes:**

**a) Reflection prompt guidance** (`v2/strategy.py`): Add to `STRATEGY_REFLECTION_SYSTEM`:

```
## Identity vs. Memos

The strategy identity describes WHO this system is as a trader — its style, risk philosophy, signal preferences, and core beliefs. It should be stable across sessions and NOT reference specific session numbers, individual trades, or recent events.

Session-specific observations belong in memos, not the identity.

Only update the identity when the system's fundamental character has genuinely shifted (e.g., from momentum trader to value trader, or from aggressive to conservative). Cosmetic updates ("in its 36th session") are not identity changes.

A good identity reads like a bio. A bad identity reads like a session log.
```

**b) Update frequency guard** (`v2/strategy.py:tool_update_strategy_identity`): Add a soft guard that warns if the identity was updated in the last 3 sessions:

```python
if current and (date.today() - current['created_at'].date()).days < 3:
    return (
        "Warning: Identity was updated within the last 3 sessions "
        f"(v{current['version']} on {current['created_at'].date()}). "
        "Consider writing a memo instead unless the system's fundamental "
        "character has changed. Proceed with update? Call again to confirm."
    )
```

**Files:** `v2/strategy.py`  
**Effort:** Low.

**Expected outcome:** Identity updates drop from ~1/session to maybe 1 every 5-10 sessions. Session-level detail moves to memos where it belongs. The strategist reads a stable identity ("conservative, evidence-gated, watch-first pipeline") instead of a rolling status report.

---

### Fix 8: Reflection agent — update rules, don't duplicate them

The reflection agent has proposed 20 rules, with 7 addressing cooldowns at different scopes. When a pattern recurs, it proposes a new rule instead of strengthening the existing one. This creates rule bloat that dilutes the signal.

The reflection agent has actually done better than the first draft of this plan gave it credit for — it retired Rules #5 and #19 when #20 superseded them. But it could be more disciplined.

**Changes:**

**a) Prompt guidance** — Add to `STRATEGY_REFLECTION_SYSTEM`:

```
## Rule Management

Before proposing a new rule:
1. Check if an existing active rule covers the same pattern
2. If so, update the existing rule's confidence, scope, or evidence rather than creating a new one
3. Only create a new rule if the pattern is genuinely distinct from all existing rules
```

**b) Store retirement reason** — Add a `retirement_reason` column to `strategy_rules`. The `retire_rule` tool already accepts a `reason` parameter but only logs it — it should be persisted so future sessions can see *why* a rule was retired without digging through memos.

```sql
ALTER TABLE strategy_rules ADD COLUMN retirement_reason TEXT;
```

Update `retire_strategy_rule()` to write the reason:

```python
def retire_strategy_rule(rule_id, reason=None) -> bool:
    with get_cursor() as cur:
        cur.execute("""
            UPDATE strategy_rules
            SET status = 'retired', retired_at = NOW(), retirement_reason = %s
            WHERE id = %s AND status = 'active'
        """, (reason, rule_id))
        return cur.rowcount > 0
```

**Files:** `v2/strategy.py` (prompt + tool handler), `v2/database/trading_db.py` (persist reason), migration  
**Effort:** Low.

**Note on Fix 4 interaction:** When structural cooldown enforcement ships, the reflection agent will organically discover that cooldown rules are now redundant (violations stop appearing in outcomes). It should retire them on its own — no manual cleanup needed. The retirement_reason column ensures the "why" is captured when it does.

---

## Not Recommended

### Pass strategy rules to the executor

With structural cooldown enforcement (Fix 4), the most critical rules are enforced in code. The remaining rules (signal quality, sector exposure) are the strategist's job to encode into the playbook. Adding rules to the executor's context makes Haiku's job more ambiguous and risks conflicts between playbook actions and rules.

**Revisit if:** Non-cooldown rules are consistently violated at execution time.

### Persist conversation history across sessions

The structured state (identity, rules, memos, theses, attribution) is the right abstraction. Raw conversation replay is expensive, grows unboundedly, and breaks when prompt formats change. The fixes above make the structured state more complete.

### Strategist mid-session memo tool

Fix 3 (persisting the final summary) covers this. A mid-session tool burns tool-call turns in a 25-turn-limited loop for marginal benefit. Revisit if the summary consistently misses important mid-session reasoning.

---

## Implementation Order

| Phase | Fix | Effort | Root Cause Addressed |
|-------|-----|--------|---------------------|
| **1: Information flow** | Fix 1: Pre-seed memos | Trivial | #1 |
| | Fix 2: Two-tier memo truncation | Low | #1 |
| | Fix 3: Persist strategist summary | Low | #2 |
| **2: Structural enforcement** | Fix 4: Cooldown enforcement in code | Moderate | #3 |
| | Fix 5: Cooldown context in executor | Low | #3 |
| | Fix 6: Skip executor without playbook | Low | #4 |
| **3: Identity & rule hygiene** | Fix 7: Stabilize identity updates | Low | #5 |
| | Fix 8: Rule dedup guidance | Trivial | #3 (symptom) |

**Phase 1** closes the information loop. Can be shipped independently and immediately improves strategist decision quality.

**Phase 2** makes cooldown rules structurally unbreakable. Eliminates the system's most persistent behavioral failure. Fix 6 prevents the category of bug that caused the 3/31 incident.

**Phase 3** is cleanup. Reduces noise in the identity and rule set so the signal-to-noise ratio of the strategist's context improves over time.

---

## Testing

**Fixes 1-3 (information flow):**
- Fix 1: Verify memos appear in the pre-seeded context string passed to `run_agentic_loop`
- Fix 2: Verify last 2 memos are full text, older ones truncated to 300 chars
- Fix 3: Verify `strategy_memos` row with `memo_type='strategist_notes'` is inserted after Stage 2

**Fixes 4-5 (cooldown):**
- HOLD on day 0 → SELL on day 1: blocked (3-day HOLD lockout)
- HOLD on day 0 → SELL on day 4: allowed
- BUY on day 0 → SELL on day 2: blocked (3-day BUY→SELL)
- SELL on day 0 → BUY on day 3: blocked (5-day SELL→BUY)
- 4 consecutive BUYs → SELL → BUY on day 8: blocked (10-day post-accumulation)
- Fallback HOLD (no playbook) → SELL on day 1: allowed (fallback HOLDs don't create cooldowns)
- Cooldown tickers appear in `ExecutorInput.cooldown_tickers` with human-readable descriptions
- Playbook actions for cooldown tickers are filtered out before the executor sees them
- Off-playbook decisions on cooldown tickers are blocked in post-validation

**Fix 6 (orchestration):**
- Stage 2 fails + no playbook for today → Stage 3 skipped, `result.skipped_executor = True`
- Stage 2 fails + playbook exists from prior run → Stage 3 runs normally
- Stage 2 succeeds → Stage 3 runs normally (no change)

**Fix 7 (identity):**
- Identity updated within last 3 sessions → warning returned, not updated
- Identity update called again after warning → update proceeds
- Verify identity text doesn't reference session numbers (manual review after a few sessions)

**Fix 8 (rules):**
- No automated test — verify through manual review of rule proposals after a few sessions
