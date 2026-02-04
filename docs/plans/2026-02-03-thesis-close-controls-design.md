# Thesis Close Controls Design

**Date:** 2026-02-03
**Status:** Approved

## Overview

Add ability to close active theses from the dashboard with "Invalidate" or "Expire" buttons. Creation and editing remain with the Claude ideation engine.

## Scope

- View existing theses (already implemented)
- Close active theses with optional reason
- Two close statuses: `invalidated`, `expired`
- "Executed" status reserved for automatic trade-triggered closure

## UI Changes (theses.html)

Each **active** thesis card gets two inline buttons in the footer:
- **Invalidate** (red) - Thesis reasoning was wrong
- **Expire** (gray) - Thesis is stale/no longer relevant

Clicking either opens a modal:
- Title: "Invalidate Thesis" or "Expire Thesis"
- Shows ticker for confirmation
- Optional textarea for reason
- Cancel / Confirm buttons

Only active theses display close buttons.

## Backend Changes

### New Route (app.py)

```python
POST /api/theses/<id>/close
Body: { "status": "invalidated"|"expired", "reason": "..." }
Returns: { "success": true } or { "error": "message" }
```

### New Query Function (dashboard/queries.py)

```python
def close_thesis(thesis_id: int, status: str, reason: str = None) -> bool:
    """Close a thesis with the given status and optional reason."""
```

Wraps existing `close_thesis()` from `trading/db.py`.

## Behavior

- After closing, page refreshes to show updated status
- Closed theses display with existing status badges (no close buttons)
- Invalid status values rejected with 400 error

## Files to Modify

1. `dashboard/app.py` - Add POST endpoint
2. `dashboard/queries.py` - Add close_thesis function
3. `dashboard/templates/theses.html` - Add buttons and modal
