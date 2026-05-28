"""Shared supervisor-watchlist resolution contract for acting stages.

Both the ideation (Stage 2) and reflection (Stage 4) stages ingest their
open watchlist items, must resolve each via `resolve_watchlist_item`, and
are hard-gated by `assert_watchlist_resolved` so a stage cannot complete
with an unresolved owned item.
"""

from __future__ import annotations

import logging

from v2.database import trading_db as db

logger = logging.getLogger(__name__)

RESOLVE_WATCHLIST_TOOL_DEF: dict = {
    "name": "resolve_watchlist_item",
    "description": (
        "Resolve one supervisor watchlist item you own. Use 'acted' when you "
        "made a change in response (retired/amended a rule, closed/updated a "
        "thesis, re-spec'd the playbook), or 'dismissed' when no change is "
        "warranted. A dismissal MUST justify why in the note. Every open item "
        "must be resolved before you finish."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "item_id": {"type": "integer"},
            "resolution": {"type": "string", "enum": ["acted", "dismissed"]},
            "note": {
                "type": "string",
                "description": "What you did (acted) or why nothing was needed (dismissed).",
            },
        },
        "required": ["item_id", "resolution", "note"],
    },
}


def make_resolve_handler(session_id: int | None, stage: str):
    """Return a handler bound to the resolving session + stage."""
    def _handler(item_id: int, resolution: str, note: str) -> str:
        try:
            ok = db.resolve_watchlist_item(
                item_id=item_id, resolution=resolution, note=note,
                session_id=session_id, stage=stage,
            )
        except ValueError as e:
            return f"Error: {e}"
        if not ok:
            return (
                f"Error: watchlist item {item_id} not found, already resolved, "
                f"or not owned by the {stage} stage."
            )
        logger.info("Resolved watchlist item %s (%s) in %s", item_id, resolution, stage)
        return f"Resolved watchlist item {item_id}: {resolution}."
    return _handler


def format_open_watchlist_items(items: list[dict]) -> str:
    """Render open items as a mandatory-action block for the stage's context."""
    if not items:
        return "SUPERVISOR WATCHLIST: no open items for this stage."
    lines = [
        "SUPERVISOR WATCHLIST (MANDATORY): the supervisor flagged the following "
        "for you this session. You MUST call resolve_watchlist_item for EACH "
        "before finishing — 'acted' (you made a change) or 'dismissed' (reasoned no-op).",
    ]
    for it in items:
        lines.append(f"- item {it['id']}: {it['title']}")
        lines.append(f"    {it['detail']}")
    return "\n".join(lines)


def assert_watchlist_resolved(owner_stage: str) -> None:
    """Raise RuntimeError if any item owned by this stage is still open.

    Called after the agentic loop. New items are only created by the
    supervisor (earlier in the session), so any remaining open row is an
    unresolved item — the forcing function."""
    remaining = db.get_open_watchlist_items(owner_stage)
    if remaining:
        ids = ", ".join(str(it["id"]) for it in remaining)
        raise RuntimeError(
            f"{owner_stage} finished with unresolved watchlist item(s): {ids}"
        )
