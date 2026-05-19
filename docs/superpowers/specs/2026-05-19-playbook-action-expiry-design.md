# Playbook Action Expiry — Design

**Date:** 2026-05-19
**Status:** Approved
**Scope:** Data-hygiene fix for `playbook_actions.status` lifecycle.

## Problem

The `playbook_actions` table accumulates `pending` rows indefinitely. As of 2026-05-19, 200 rows are in `pending` status, all from playbooks dated >3 days ago.

The status lifecycle today:

| Trigger | New status | Where |
|---|---|---|
| Strategist generates today's playbook | `pending` | `replace_playbook_actions_atomic` in `v2/database/trading_db.py` |
| Executor submits buy/sell successfully | `executed` | `_execute_decision_order` in `v2/trader.py` |
| Order submit fails | `failed` | same |
| Sell pre-submit rejects (no shares) | `skipped` | `_validate_sell_availability` in `v2/trader.py` |

There is no transition for prior-day actions that the executor never picked up — either because it chose `hold` instead of the strategist's proposed `buy`/`sell`, or because the action's `intent_type` was inherently non-trading (e.g. `invest_dollar` for "hold these positions"), or because the executor simply didn't include the ticker in its decisions.

`replace_playbook_actions_atomic` only wipes actions for the *same date* (`ON CONFLICT (date) DO UPDATE`), so yesterday's `pending` rows are never cleaned up.

## Goal

Every `pending` action from a playbook with `date < today` becomes `expired`. The fix is idempotent and runs every session as a no-op when there's nothing to clean.

## Non-goals

- Tracking *why* an action went stale (e.g. `overridden` vs. `noop`). One terminal status, `expired`, is sufficient.
- Changes to executor or strategist behavior. This is pure data hygiene.
- A separate data-migration script. The first production run of the sweep handles the existing 200 rows.

## Approach

Add a session-start sweep that runs once per session, before any stage. Pure SQL, idempotent, no dependencies on stage state.

### Component 1 — `v2/database/trading_db.py`

New helper:

```python
def expire_stale_playbook_actions(today: date) -> int:
    """Mark any pending playbook_actions for a playbook with date < today as 'expired'.

    Idempotent. Returns the number of rows updated.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE playbook_actions
               SET status = 'expired'
             WHERE status = 'pending'
               AND playbook_id IN (
                   SELECT id FROM playbooks WHERE date < %s
               )
            """,
            (today,),
        )
        return cur.rowcount
```

### Component 2 — `v2/session.py`

Call the helper once at session start, before any stage runs. Log the count.

```python
expired = expire_stale_playbook_actions(session_date)
if expired:
    logger.info("Expired %d stale pending playbook_actions from prior days", expired)
```

Placement: top of `run_session` (or equivalent entry point), after `session_date` is resolved but before the first stage dispatch.

### Component 3 — Tests

Unit tests in `tests/v2/test_trading_db.py` (or wherever `replace_playbook_actions_atomic` is tested) covering:

1. Prior-day `pending` → `expired`
2. Today's `pending` → untouched
3. Already-`executed` prior-day actions → untouched
4. Empty case → returns 0
5. Idempotency: running twice does not double-count or revert any rows

Integration test in `tests/v2/test_session.py` (or wherever the session orchestrator is tested) asserting the sweep runs before stage dispatch.

### Component 4 — Backfill

Handled implicitly by the first session run after deploy. No standalone migration script.

## Risk Analysis

`playbook_actions.status` is consumed in two places:

1. `v2/database/dashboard_db.py:145` — counts rows where `status = 'executed'`. Unaffected.
2. `v2/database/trading_db.py:552` (`get_pending_playbook_actions`) — filters `status = 'pending' OR status IS NULL`. Adding `expired` to the universe does not affect this filter.

No other consumer reads `status`. Adding a new terminal value is safe.

## Rollout

1. Land the helper + tests.
2. Land the session-start call.
3. On the next 19:00 UTC session, the sweep runs once with the 200-row backfill effect. Verify via:
   ```sql
   SELECT status, COUNT(*) FROM playbook_actions GROUP BY status;
   ```
   Expected: ~200 `expired`, today's actions still in their normal mix.

No feature flag, no staged rollout. Reversibility: if the sweep misbehaves, `UPDATE playbook_actions SET status='pending' WHERE status='expired'` restores prior state.
