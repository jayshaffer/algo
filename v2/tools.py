"""Tool definitions and implementations for Claude ideation agent."""

import logging
from datetime import date

from .agent import validate_signal_refs
from .attribution import get_attribution_summary
from .context import format_active_theses_by_role, get_macro_context, get_portfolio_context
from .database.connection import get_cursor
from .database.trading_db import (
    close_thesis,
    get_active_strategy_rules,
    get_active_theses,
    get_current_strategy_state,
    get_macro_signals,
    get_news_signals,
    get_positions,
    get_recent_decisions,
    get_recent_playbooks_with_actions,
    get_recent_strategy_memos,
    get_recent_thesis_decision_dates,
    get_thesis_by_id,
    insert_thesis,
    insert_thesis_signals,
    replace_playbook_actions_atomic,
    update_thesis,
)
from .executor import get_account_info
from .market_data import format_market_snapshot, get_market_snapshot
from .news_filter import curate_signals
from .patterns import analyze_round_trips

logger = logging.getLogger(__name__)


_curated_news_cache: dict[tuple, list[int]] = {}


def reset_session():
    """Reset session state. Call at start of each ideation run."""
    _curated_news_cache.clear()
    logger.info("Session state reset")


from .tickers import resolve_ticker as _norm_ticker  # noqa: E402


def _resolve_required_ticker(raw: str | None) -> tuple[str | None, str | None]:
    """Resolve a ticker that the tool requires (create_thesis, adopt_thesis,
    playbook actions). Returns (ticker, error_message). If `raw` is non-empty
    but resolution drops it (unknown shape, blocked acronym), the error message
    explains the rejection so the strategist sees the failure in its loop and
    can reissue with the real ticker."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, "Error: ticker is required."
    resolved = _norm_ticker(raw)
    if resolved is None:
        return None, (
            f"Error: ticker {raw!r} is not a recognized equity symbol "
            f"(group acronym, non-equity abbreviation, or malformed). "
            f"Use the actual Alpaca-tradable symbol."
        )
    return resolved, None


# --- Tool Handlers ---


def tool_get_market_snapshot() -> str:
    """Get current market state."""
    logger.info("Getting market snapshot")
    try:
        snapshot = get_market_snapshot()
        return format_market_snapshot(snapshot)
    except Exception as e:
        logger.exception("Failed to get market snapshot")
        return f"Error getting market snapshot: {e}"


def tool_get_portfolio_state() -> str:
    """Get portfolio positions and account info."""
    logger.info("Getting portfolio state")
    try:
        account_info = get_account_info()
        return get_portfolio_context(account_info)
    except Exception as e:
        logger.exception("Failed to get portfolio state")
        return f"Error getting portfolio state: {e}"


def tool_get_active_theses(ticker: str | None = None) -> str:
    """Get active theses."""
    ticker = _norm_ticker(ticker)
    logger.info(f"Getting active theses (ticker filter: {ticker})")
    theses = get_active_theses(ticker=ticker)

    if not theses:
        return "No active theses."

    positions = get_positions()
    last_decision_dates = get_recent_thesis_decision_dates(days=60)
    return format_active_theses_by_role(theses, positions, last_decision_dates)


def _persist_signal_refs(thesis_id: int, signal_refs: list[dict] | None) -> str:
    """Validate, persist, and return a human-readable note for tool output.

    Returns a status fragment describing how many refs were stripped (if any),
    so the strategist sees its own miscites within the agentic loop and can
    correct on the next turn. Empty string if no refs given or all valid.
    """
    if not signal_refs:
        return ""
    submitted = len(signal_refs)
    valid = validate_signal_refs(signal_refs)
    insert_thesis_signals(thesis_id, valid)
    stripped = submitted - len(valid)
    if stripped:
        return (
            f" Note: {stripped} of {submitted} signal_refs were stripped as invalid "
            f"(IDs not found in DB)."
        )
    return f" Cited {len(valid)} signal(s)."


def tool_create_thesis(
    ticker: str,
    direction: str,
    thesis: str,
    entry_trigger: str,
    exit_trigger: str,
    invalidation: str,
    confidence: str,
    signal_refs: list[dict] | None = None,
    session_id: int | None = None,
) -> str:
    """Create a new thesis.

    signal_refs: list of {"type": "news_signal"|"macro_signal"|"thesis", "id": int}
    citing the evidence that justified this thesis. IDs must come from
    get_news_signals / get_macro_signals / get_active_theses output. Invalid
    IDs are stripped at write time and reported back in the tool result.

    At least one signal_ref is REQUIRED so downstream attribution can score the
    decision. Use tool_adopt_thesis instead for theses created from already-held
    orphan positions.
    """
    ticker, err = _resolve_required_ticker(ticker)
    if err:
        return err
    logger.info(f"Creating thesis for {ticker} ({direction})")

    # Check for duplicates
    existing = get_active_theses(ticker=ticker)
    if existing:
        return (
            f"Error: Active thesis already exists for {ticker} "
            f"(ID {existing[0]['id']}). Update or close it first."
        )

    positions = {p["ticker"] for p in get_positions()}
    if ticker in positions:
        return f"Error: {ticker} is already in the portfolio. Cannot create thesis."

    # Enforce that the strategist cites evidence. Empty/invalid refs were the
    # root cause of audit finding THESES_NO_SIGNAL_REFS (36 of 49 theses) and
    # the downstream attribution coverage gap.
    valid_refs = validate_signal_refs(signal_refs) if signal_refs else []
    if not valid_refs:
        return (
            "Error: create_thesis requires at least one valid signal_ref. "
            "Cite the news_signal / macro_signal / thesis IDs that justified "
            "this thesis (use get_news_signals / get_macro_signals / "
            "get_active_theses to find IDs). Use tool_adopt_thesis instead "
            "for already-held orphan positions."
        )

    # Create the thesis
    thesis_id = insert_thesis(
        ticker=ticker,
        direction=direction,
        thesis=thesis,
        entry_trigger=entry_trigger,
        exit_trigger=exit_trigger,
        invalidation=invalidation,
        confidence=confidence,
        source="claude_ideation",
        session_id=session_id,
    )

    note = _persist_signal_refs(thesis_id, signal_refs)
    logger.info(f"Created thesis ID {thesis_id} for {ticker}")
    return (
        f"Created thesis ID {thesis_id} for {ticker} "
        f"({direction}, {confidence} confidence).{note}"
    )


def tool_adopt_thesis(
    ticker: str,
    direction: str,
    thesis: str,
    exit_trigger: str,
    invalidation: str,
    confidence: str,
    signal_refs: list[dict] | None = None,
    session_id: int | None = None,
) -> str:
    """Adopt an existing portfolio position by creating a thesis for it.

    Unlike create_thesis, this REQUIRES the ticker to already be in the portfolio.
    Used to bring orphan positions under thesis management.
    """
    ticker, err = _resolve_required_ticker(ticker)
    if err:
        return err
    logger.info(f"Adopting thesis for existing position {ticker} ({direction})")

    existing = get_active_theses(ticker=ticker)
    if existing:
        return (
            f"Error: Active thesis already exists for {ticker} "
            f"(ID {existing[0]['id']}). Update it instead."
        )

    positions = {p["ticker"] for p in get_positions()}
    if ticker not in positions:
        return f"Error: {ticker} is not in the portfolio. Use create_thesis for new ideas."

    thesis_id = insert_thesis(
        ticker=ticker,
        direction=direction,
        thesis=thesis,
        entry_trigger="Already held — adopted into thesis management",
        exit_trigger=exit_trigger,
        invalidation=invalidation,
        confidence=confidence,
        source="adoption",
        session_id=session_id,
    )

    note = _persist_signal_refs(thesis_id, signal_refs)
    logger.info(f"Adopted position {ticker} as thesis ID {thesis_id}")
    # T2.12: distinct return prefix so `count_actions` can separate adoption
    # (an existing position pulled under thesis management) from outright
    # creation. Conflating them inflated the "created" counter on sessions
    # whose work was mostly housekeeping for orphans.
    return (
        f"Adopted thesis ID {thesis_id} for {ticker} "
        f"(adopted existing position, {direction}, {confidence} confidence).{note}"
    )


def tool_update_thesis(
    thesis_id: int,
    thesis: str | None = None,
    entry_trigger: str | None = None,
    exit_trigger: str | None = None,
    invalidation: str | None = None,
    confidence: str | None = None,
    add_signal_refs: list[dict] | None = None,
) -> str:
    """Update an existing thesis.

    add_signal_refs appends new signal citations to the thesis (idempotent).
    Use it as evidence accumulates over the life of the thesis.
    """
    logger.info(f"Updating thesis ID {thesis_id}")

    has_field_updates = any(
        v is not None for v in (thesis, entry_trigger, exit_trigger, invalidation, confidence)
    )

    # T1.7: enforce input shape and existence BEFORE any DB write. The
    # add_signal_refs-only path previously skipped existence checks; an invalid
    # thesis_id would crash _persist_signal_refs with a raw FK error instead of
    # surfacing a clean tool_result the strategist can act on.
    if not has_field_updates and not add_signal_refs:
        return "Error: no updates provided (must set at least one field or add_signal_refs)"
    if get_thesis_by_id(thesis_id) is None:
        return f"Error: thesis ID {thesis_id} not found"

    if has_field_updates:
        update_thesis(
            thesis_id=thesis_id,
            thesis=thesis,
            entry_trigger=entry_trigger,
            exit_trigger=exit_trigger,
            invalidation=invalidation,
            confidence=confidence,
        )

    note = _persist_signal_refs(thesis_id, add_signal_refs)

    return f"Updated thesis ID {thesis_id}.{note}"


def tool_close_thesis(thesis_id: int, status: str, reason: str) -> str:
    """Close a thesis."""
    logger.info(f"Closing thesis ID {thesis_id} with status {status}")

    success = close_thesis(thesis_id=thesis_id, status=status, reason=reason)

    if success:
        return f"Closed thesis ID {thesis_id} with status '{status}'"
    else:
        return f"Error: Thesis ID {thesis_id} not found"


def tool_get_news_signals(ticker: str = None, days: int = 7) -> str:
    """Get recent ticker-specific news signals.

    Each line is prefixed with [#<id>] so the strategist can cite the signal
    by ID via signal_refs on create_thesis / update_thesis. IDs that don't
    appear in tool output should not be cited — they will be stripped by
    validation at thesis-creation time.
    """
    ticker = _norm_ticker(ticker)
    logger.info(f"Getting news signals (ticker: {ticker}, days: {days})")
    signals = get_news_signals(ticker=ticker, days=days)

    if not signals:
        if ticker:
            return f"No news signals for {ticker} in the last {days} days."
        return f"No news signals in the last {days} days."

    lines = []
    for s in signals:
        date_str = s["published_at"].strftime("%m-%d %H:%M")
        headline = s["headline"][:60]
        lines.append(
            f"[#{s['id']}] {date_str} {s['ticker']} "
            f"{s['category']}/{s['sentiment']}/{s['confidence']}: {headline}"
        )

    return "\n".join(lines)


def tool_get_curated_news(
    ticker: str = None,
    days: int = 7,
    target_n: int = 30,
) -> str:
    """Curated news signals — Haiku-filtered to the ~target_n most
    relevant for today's market regime. Use this by default for thesis
    research; use get_news_signals if you need the raw firehose.

    Each line is prefixed with [#<id>] (same format as get_news_signals)
    so signal_refs citation works unchanged.
    """
    ticker = _norm_ticker(ticker)
    cache_key = (ticker, days, target_n)
    cached_ids = _curated_news_cache.get(cache_key)

    logger.info("Getting curated news (ticker=%s, days=%d, target_n=%d)", ticker, days, target_n)
    rows = get_news_signals(ticker=ticker, days=days)

    # Drop rows without usable summaries — pre-backfill safety + empty-summary cleanup.
    candidates = [r for r in rows if r.get("summary")]

    if not candidates:
        if ticker:
            return f"No news signals for {ticker} in the last {days} days."
        return f"No news signals in the last {days} days."

    if cached_ids is None:
        if len(candidates) <= target_n:
            # No filtering needed — return everything.
            selected_ids = [r["id"] for r in candidates]
        else:
            regime_context = get_macro_context(days=2)
            selected_ids = curate_signals(
                candidates,
                target_n=target_n,
                regime_context=regime_context,
            )
        _curated_news_cache[cache_key] = selected_ids
    else:
        selected_ids = cached_ids

    selected_set = set(selected_ids)
    selected_rows = [r for r in candidates if r["id"] in selected_set]
    # Preserve curate_signals' ordering (rank order), not DB order:
    order = {sid: i for i, sid in enumerate(selected_ids)}
    selected_rows.sort(key=lambda r: order.get(r["id"], 1_000_000))

    lines = []
    for s in selected_rows:
        date_str = s["published_at"].strftime("%m-%d %H:%M")
        headline = s["headline"][:60]
        lines.append(
            f"[#{s['id']}] {date_str} {s['ticker']} "
            f"{s['category']}/{s['sentiment']}/{s['confidence']}: {headline}"
        )

    return "\n".join(lines)


def tool_get_macro_context(days: int = 7) -> str:
    """Get macro economic context."""
    logger.info(f"Getting macro context (last {days} days)")
    return get_macro_context(days=days)


def tool_get_macro_signals(days: int = 7) -> str:
    """Get recent macro signals as a list, one per line, with IDs.

    Parallels tool_get_news_signals — use this when you need IDs to cite on
    a thesis via signal_refs. tool_get_macro_context is the higher-level
    summary by category; this tool returns the raw list with IDs.
    """
    logger.info(f"Getting macro signals (last {days} days)")
    signals = get_macro_signals(days=days)

    if not signals:
        return f"No macro signals in the last {days} days."

    lines = []
    for s in signals:
        date_str = s["published_at"].strftime("%m-%d %H:%M")
        headline = s["headline"][:80]
        lines.append(
            f"[#{s['id']}] {date_str} {s['category']}/{s['sentiment']}: {headline}"
        )

    return "\n".join(lines)


def tool_get_signal_attribution() -> str:
    """Get signal attribution scores."""
    logger.info("Getting signal attribution")
    return get_attribution_summary()


def tool_get_decision_history(days: int = 30) -> str:
    """Get recent decisions with outcomes."""
    logger.info(f"Getting decision history ({days} days)")
    decisions = get_recent_decisions(days=days)

    if not decisions:
        return f"No decisions in the last {days} days."

    lines = []
    for d in decisions:
        outcome_7d = f"{d['outcome_7d']:+.1f}%" if d.get("outcome_7d") is not None else "-"
        outcome_30d = f"{d['outcome_30d']:+.1f}%" if d.get("outcome_30d") is not None else "-"
        lines.append(
            f"{d['date']} {d['action'].upper()} {d['ticker']} 7d:{outcome_7d} 30d:{outcome_30d} {d['reasoning'][:60]}"
        )

    return "\n".join(lines)


def tool_get_recent_playbooks(n: int = 3) -> str:
    """Return the past N playbooks (planned actions) so the strategist
    can detect reversals against its own recent plans before writing a
    new playbook."""
    logger.info(f"Getting recent playbooks (n={n})")
    playbooks = get_recent_playbooks_with_actions(n=n)
    if not playbooks:
        return "No recent playbooks."

    lines = [f"Recent Playbooks ({len(playbooks)} most recent):", ""]
    for pb in playbooks:
        lines.append(f"{pb['pb_date']} (Playbook #{pb['pb_id']}):")
        if not pb["actions"]:
            lines.append("  (no actions)")
        for a in pb["actions"]:
            mag = a.get("intent_magnitude")
            intent = a.get("intent_type") or ""
            mag_str = f"={mag}" if mag is not None else ""
            intent_part = f" {intent}{mag_str}" if intent else ""
            outcome = a.get("outcome_class") or "unknown"
            status = a.get("status") or "pending"
            decision = a.get("decision_action")
            outcome_part = f" [{outcome}; status={status}"
            if decision:
                outcome_part += f"; executor={decision}"
            outcome_part += "]"
            reasoning = (a.get("reasoning") or "").strip()
            lines.append(
                f"  {a['ticker']}: {a['action'].upper()}{intent_part}"
                f"{outcome_part} — {reasoning}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def tool_write_playbook(
    market_outlook: str,
    priority_actions: list,
    watch_list: list,
    risk_notes: str,
) -> str:
    """Write today's playbook to the database.

    V3 behavior:
    - Validates no conflicting actions (buy + sell same ticker)
    - Writes playbook row via upsert_playbook()
    - Deletes old playbook_actions for this playbook
    - Inserts new playbook_actions rows for each priority action
    - Returns confirmation with playbook ID and action count
    """
    logger.info("Writing playbook")

    # T1.3: normalize ticker on every action so persisted playbook rows are
    # canonical TICKER, not "aapl"/"AAPL "/etc. Mutates in place so the
    # downstream insert sees the cleaned values too. Hallucinated tickers
    # (group acronyms, company-name-instead-of-symbol) are rejected at
    # write-time so the strategist sees the failure inside its tool loop
    # rather than producing a playbook that fails at executor pricing.
    for action in priority_actions:
        raw = action.get("ticker")
        resolved, err = _resolve_required_ticker(raw)
        if err:
            return f"Error in priority_actions: {err.removeprefix('Error: ')}"
        action["ticker"] = resolved

    # Validate no conflicting actions (buy + sell same ticker)
    # AND no duplicate (ticker, action) pairs. T1.6: a playbook with two buys
    # for AAPL was previously accepted; the executor then evaluated both and
    # could double-up size against the same idea. Rejecting at write-time
    # forces the strategist to dedupe in its own loop.
    actions_by_ticker: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for action in priority_actions:
        ticker = action.get("ticker")
        act = action.get("action")
        if ticker in actions_by_ticker and actions_by_ticker[ticker] != act:
            return (
                f"Error: Conflicting actions for {ticker} — "
                f"cannot {actions_by_ticker[ticker]} and {act} the same ticker."
            )
        pair = (ticker, act)
        if pair in seen_pairs:
            return (
                f"Error: Duplicate (ticker, action) pair for {ticker} {act}. "
                f"Each (ticker, action) must appear at most once per playbook."
            )
        seen_pairs.add(pair)
        actions_by_ticker[ticker] = act

    # Playbook buy/sell actions must cite a thesis. The strategist's job
    # is producing thesis-backed actions; allowing NULL thesis_id creates
    # orphans in playbook_actions that break the attribution joins
    # (decision_signals → theses) and re-opens the same shape that was
    # fixed for news_signal:unknown.
    for action in priority_actions:
        act = action.get("action")
        ticker = action.get("ticker")
        if act in ("buy", "sell") and action.get("thesis_id") is None:
            return (
                f"Error: Playbook {act} for {ticker} is missing thesis_id. "
                f"Every buy/sell action in the playbook must cite a thesis. "
                f"Create or adopt a thesis first, then reference its id."
            )

    # ALGO-13: reject zero-magnitude / missing-magnitude intents that require
    # a positive number. The strategist was using `exit_partial_pct=0` as a
    # "hold" proxy, which the executor correctly translated to action='hold'
    # but at the cost of (a) polluting the playbook with no-op rows, (b)
    # misrepresenting strategist intent in decisions.reasoning, and (c)
    # burning tool-call budget on actions that don't change behavior. The
    # correct shape for "hold this position" is to omit it from
    # priority_actions; the correct shape for "fully exit" is exit_full.
    _MAG_REQUIRED_INTENTS = {
        "exit_partial_pct", "exit_dollar", "trim_to_portfolio_pct",
        "invest_dollar", "invest_portfolio_pct", "invest_buying_power_pct",
        "add_to_target_pct",
    }
    for action in priority_actions:
        intent = action.get("intent_type")
        mag = action.get("intent_magnitude")
        if intent in _MAG_REQUIRED_INTENTS and (mag is None or float(mag) <= 0):
            return (
                f"Error: Playbook {action.get('action')} for {action.get('ticker')} "
                f"has intent_type={intent} with intent_magnitude={mag}. This intent "
                f"requires a positive magnitude. To hold/no-op a position, omit it "
                f"from priority_actions (the default for any unlisted position is "
                f"hold). For a full exit, use intent_type=exit_full instead."
            )

    # ALGO-18: reject exit-style intents whose ticker has 0 live shares. The
    # strategist was repeatedly writing exit_full against zeroed positions
    # (e.g. AMD stub cleanup ran 6 sessions before being caught), and every
    # session produced a runtime "invalid" decision with no downstream
    # effect. Catch at write time using live positions.
    _EXIT_INTENTS_REQUIRING_HELD = {
        "exit_full", "exit_partial_pct", "exit_dollar", "trim_to_portfolio_pct",
    }
    _ZERO_HELD_EPS = 0.0001
    if any(a.get("intent_type") in _EXIT_INTENTS_REQUIRING_HELD for a in priority_actions):
        live_positions = {p["ticker"]: float(p["shares"]) for p in get_positions()}
        for action in priority_actions:
            intent = action.get("intent_type")
            ticker = action.get("ticker")
            if intent in _EXIT_INTENTS_REQUIRING_HELD:
                held = live_positions.get(ticker, 0.0)
                if abs(held) <= _ZERO_HELD_EPS:
                    return (
                        f"Error: Playbook {action.get('action')} for {ticker} has "
                        f"intent_type={intent}, but the live position is empty "
                        f"(held={held}). Omit this action — there is nothing to "
                        f"exit. If the underlying thesis is closed, retire it via "
                        f"close_thesis instead of writing a no-op exit."
                    )

    try:
        playbook_date = date.today()
        # P2.22: single transaction for upsert + delete + N inserts. The
        # previous flow had three separate connections, so a mid-loop failure
        # left the playbook row + a subset of actions with no rollback.
        playbook_id, action_count = replace_playbook_actions_atomic(
            playbook_date=playbook_date,
            market_outlook=market_outlook,
            priority_actions=priority_actions,
            watch_list=watch_list,
            risk_notes=risk_notes,
            actions=priority_actions,
        )
        return (
            f"Playbook written for {playbook_date} "
            f"(ID: {playbook_id}, {action_count} actions)"
        )
    except Exception as e:
        logger.exception("Failed to write playbook")
        return f"Error writing playbook: {e}"


def tool_get_theses(status: str = "all", limit: int = 50) -> list[dict]:
    """Read-only: theses by status (all|active|closed)."""
    base = """
        SELECT id, ticker, thesis, entry_trigger, exit_trigger,
               status, created_at, closed_at, close_reason
        FROM theses
    """
    params: tuple
    if status == "all":
        sql = base + " ORDER BY created_at DESC LIMIT %s"
        params = (limit,)
    else:
        sql = base + " WHERE status = %s ORDER BY created_at DESC LIMIT %s"
        params = (status, limit)
    with get_cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "thesis_id": r["id"],
            "ticker": r["ticker"],
            "thesis": r["thesis"],
            "entry_trigger": r["entry_trigger"],
            "exit_trigger": r["exit_trigger"],
            "status": r["status"],
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
            "closed_at": r["closed_at"].isoformat() if hasattr(r["closed_at"], "isoformat") else r["closed_at"],
            "close_reason": r["close_reason"],
        }
        for r in rows
    ]


def tool_get_thesis_lineage(thesis_id: int) -> dict:
    """Read-only: decisions tagged with this thesis, with outcomes.

    Theses link to decisions via playbook_actions
    (decisions.playbook_action_id → playbook_actions.id; playbook_actions.thesis_id → theses.id).
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, ticker, thesis, status FROM theses WHERE id = %s",
            (thesis_id,),
        )
        head = cur.fetchone()
        if head is None:
            return {"thesis_id": thesis_id, "error": "not found"}
        cur.execute(
            """
            SELECT id, date, action, quantity, reasoning, outcome_7d, outcome_30d
            FROM (
                SELECT d.id, d.date::text AS date, d.action, d.quantity, d.reasoning,
                       d.outcome_7d, d.outcome_30d
                FROM decisions d
                JOIN playbook_actions pa ON pa.id = d.playbook_action_id
                WHERE pa.thesis_id = %s

                UNION

                SELECT d.id, d.date::text AS date, d.action, d.quantity, d.reasoning,
                       d.outcome_7d, d.outcome_30d
                FROM decisions d
                JOIN decision_signals ds ON ds.decision_id = d.id
                WHERE ds.signal_type = 'thesis' AND ds.signal_id = %s
            ) thesis_decisions
            ORDER BY date ASC
            """,
            (thesis_id, thesis_id),
        )
        rows = cur.fetchall()
    return {
        "thesis_id": head["id"],
        "ticker": head["ticker"],
        "thesis": head["thesis"],
        "status": head["status"],
        "decisions": [
            {
                "decision_id": r["id"],
                "date": r["date"],
                "action": r["action"],
                "quantity": float(r["quantity"]) if r["quantity"] is not None else None,
                "reasoning_excerpt": (r["reasoning"] or "")[:200],
                "outcome_7d": float(r["outcome_7d"]) if r["outcome_7d"] is not None else None,
                "outcome_30d": float(r["outcome_30d"]) if r["outcome_30d"] is not None else None,
            }
            for r in rows
        ],
    }


def tool_get_strategy_identity() -> str:
    """Get the system's current strategy identity."""
    logger.info("Getting strategy identity")
    state = get_current_strategy_state()
    if state is None:
        return "No strategy identity established yet. This is the first session."

    lines = [
        f"Strategy Identity (v{state['version']}, updated {state['created_at'].strftime('%Y-%m-%d')}):",
        f"  Identity: {state['identity_text']}",
        f"  Risk Posture: {state['risk_posture']}",
        f"  Sector Biases: {state['sector_biases']}",
        f"  Preferred Signals: {state['preferred_signals']}",
        f"  Avoided Signals: {state['avoided_signals']}",
    ]
    return "\n".join(lines)


def tool_get_strategy_rules() -> str:
    """Get all active strategy rules."""
    logger.info("Getting strategy rules")
    rules = get_active_strategy_rules()
    if not rules:
        return "No active strategy rules yet."

    lines = []
    for r in rules:
        evidence = f" | {r['supporting_evidence'][:80]}" if r.get("supporting_evidence") else ""
        lift = f" | lift:{r['lift_condition']}" if r.get("lift_condition") else " | lift:UNSET"
        lines.append(
            f"#{r['id']} {r['direction']}/{r['category']} conf:{r['confidence']}: "
            f"{r['rule_text']}{lift}{evidence}"
        )
    return "\n".join(lines)


def tool_get_retired_rules(limit: int = 50) -> list[dict]:
    """Read-only: most recently retired strategy rules."""
    sql = """
        SELECT id, rule_text, category, supporting_evidence, status,
               created_at, retired_at, retirement_reason
        FROM strategy_rules
        WHERE status = 'retired'
        ORDER BY retired_at DESC NULLS LAST
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "rule_id": r["id"],
            "rule_text": r["rule_text"],
            "category": r["category"],
            "supporting_evidence": r["supporting_evidence"],
            "status": r["status"],
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
            "retired_at": r["retired_at"].isoformat() if hasattr(r["retired_at"], "isoformat") else r["retired_at"],
            "retirement_reason": r["retirement_reason"],
        }
        for r in rows
    ]


def tool_get_rule_bind_history(rule_id: int, days: int = 30) -> dict:
    """Read-only: decisions that cited this rule within `days`."""
    sql = """
        SELECT d.id, d.date::text, d.ticker, d.action, d.reasoning
        FROM decisions d
        JOIN decision_signals ds
          ON ds.decision_id = d.id
         AND ds.signal_type = 'rule_gate'
         AND ds.signal_id = %s
        WHERE d.date >= CURRENT_DATE - INTERVAL '1 day' * %s
        ORDER BY d.date ASC
    """
    with get_cursor() as cur:
        cur.execute(sql, (rule_id, days))
        rows = cur.fetchall()
    return {
        "rule_id": rule_id,
        "window_days": days,
        "bind_count": len(rows),
        "citations": [
            {
                "decision_id": r["id"],
                "date": r["date"],
                "ticker": r["ticker"],
                "action": r["action"],
                "reasoning_excerpt": (r["reasoning"] or "")[:200],
            }
            for r in rows
        ],
    }


def tool_get_strategy_history(n: int = 5, full_recent: int = 2) -> str:
    """Get recent strategy memos. Last `full_recent` shown in full, older truncated to 300 chars."""
    logger.info(f"Getting strategy history (last {n})")
    memos = get_recent_strategy_memos(n=n)
    if not memos:
        return "No strategy memos yet. This is the first session."

    lines = []
    for i, m in enumerate(memos):
        if i < full_recent:
            content = m['content']
        else:
            content = m['content'][:300] + "..." if len(m['content']) > 300 else m['content']
        lines.append(f"[{m['session_date']}] {m['memo_type']}: {content}")
    return "\n".join(lines)


def tool_get_recent_decisions(days: int = 14) -> list[dict]:
    """Read-only: compact recent decisions for behavior review."""
    sql = """
        SELECT d.id, d.date::text AS date, d.ticker, d.action, d.quantity,
               d.reasoning,
               COALESCE(string_agg(DISTINCT ds.signal_type, ','), '') AS signals_referenced,
               d.outcome_7d
        FROM decisions d
        LEFT JOIN decision_signals ds ON ds.decision_id = d.id
        WHERE d.date >= CURRENT_DATE - INTERVAL '1 day' * %s
        GROUP BY d.id, d.date, d.ticker, d.action, d.quantity, d.reasoning, d.outcome_7d
        ORDER BY d.date ASC, d.id ASC
    """
    with get_cursor() as cur:
        cur.execute(sql, (days,))
        rows = cur.fetchall()
    return [
        {
            "decision_id": r["id"],
            "date": r["date"],
            "ticker": r["ticker"],
            "action": r["action"],
            "quantity": float(r["quantity"]) if r["quantity"] is not None else None,
            "reasoning_excerpt": (r["reasoning"] or "")[:200],
            "signals_referenced": r["signals_referenced"],
            "outcome_7d": float(r["outcome_7d"]) if r["outcome_7d"] is not None else None,
        }
        for r in rows
    ]


def tool_get_decision_detail(decision_id: int) -> dict:
    """Read-only: full decision detail + all referenced signals.

    Signal rows are returned as {signal_id, signal_type} only — to fetch the
    underlying signal's text/sentiment, call the appropriate get_* tool with
    the signal_id.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, date::text AS date, ticker, action, quantity,
                   reasoning, outcome_7d, outcome_30d, price
            FROM decisions WHERE id = %s
            """,
            (decision_id,),
        )
        head = cur.fetchone()
        if head is None:
            return {"decision_id": decision_id, "error": "not found"}
        cur.execute(
            """
            SELECT signal_id, signal_type
            FROM decision_signals
            WHERE decision_id = %s
            ORDER BY signal_type, signal_id
            """,
            (decision_id,),
        )
        sigs = cur.fetchall()
    return {
        "decision_id": head["id"],
        "date": head["date"],
        "ticker": head["ticker"],
        "action": head["action"],
        "quantity": float(head["quantity"]) if head["quantity"] is not None else None,
        "price": float(head["price"]) if head["price"] is not None else None,
        "reasoning": head["reasoning"],
        "outcome_7d": float(head["outcome_7d"]) if head["outcome_7d"] is not None else None,
        "outcome_30d": float(head["outcome_30d"]) if head["outcome_30d"] is not None else None,
        "signals": [
            {"signal_id": s["signal_id"], "signal_type": s["signal_type"]}
            for s in sigs
        ],
    }


def tool_get_flip_flop_report(days: int = 30, min_reversals: int = 3) -> list[dict]:
    """Read-only: tickers with N+ same-window opposing-action pairs."""
    trips = analyze_round_trips(days=days, gap_days=7, min_pairs=min_reversals)
    return [
        {
            "ticker": t.ticker,
            "reversal_count": t.pair_count,
            "first_date": str(t.first_date),
            "last_date": str(t.last_date),
        }
        for t in trips
    ]


def tool_get_session_memos(limit: int = 10) -> list[dict]:
    """Read-only: most recent strategy memos."""
    sql = """
        SELECT id, session_date::text AS session_date, memo_type, content, created_at
        FROM strategy_memos
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "memo_id": r["id"],
            "session_date": r["session_date"],
            "memo_type": r["memo_type"],
            "content": r["content"],
            "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
        }
        for r in rows
    ]


def tool_get_reflection_actions(limit: int = 10) -> list[dict]:
    """Read-only: per-session reflection-stage actions taken.

    rules_revalidated is always 0 — the project does not track revalidation
    as a column on strategy_rules. If a revalidation log table is added,
    this can be updated.
    """
    sql = """
        SELECT
            s.id AS session_id,
            s.session_date::text AS session_date,
            (SELECT COUNT(*) FROM strategy_rules sr
                WHERE sr.created_at::date = s.session_date) AS rules_proposed,
            (SELECT COUNT(*) FROM strategy_rules sr
                WHERE sr.retired_at::date = s.session_date) AS rules_retired,
            EXISTS(
                SELECT 1 FROM strategy_state ss
                WHERE ss.created_at::date = s.session_date
            ) AS identity_updated,
            COALESCE((
                SELECT array_length(regexp_split_to_array(sm.content, '\\s+'), 1)
                FROM strategy_memos sm
                WHERE sm.session_date = s.session_date
                ORDER BY sm.created_at DESC LIMIT 1
            ), 0) AS memo_word_count
        FROM sessions s
        ORDER BY s.session_date DESC
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "session_id": r["session_id"],
            "session_date": r["session_date"],
            "rules_proposed": r["rules_proposed"],
            "rules_retired": r["rules_retired"],
            "rules_revalidated": 0,
            "identity_updated": bool(r["identity_updated"]),
            "memo_word_count": r["memo_word_count"],
        }
        for r in rows
    ]


def tool_get_session_summary_window(days: int = 14) -> list[dict]:
    """Read-only: per-session decisions/P&L/stage failures/cost within a window.

    Uses the `session_costs` view (defined in db/init/024_session_stage_token_usage.sql)
    for total_cost_usd. P&L proxy uses sum of outcome_7d for that session's decisions.
    """
    sql = """
        SELECT
            sc.session_id,
            sc.session_date::text AS session_date,
            COALESCE((
                SELECT COUNT(*) FROM decisions d
                WHERE d.session_id = sc.session_id
            ), 0) AS decisions_count,
            COALESCE((
                SELECT SUM(d.outcome_7d) FROM decisions d
                WHERE d.session_id = sc.session_id
            ), 0)::numeric AS pnl_usd,
            COALESCE((
                SELECT COUNT(*) FROM session_stages st
                WHERE st.session_id = sc.session_id AND st.status = 'failed'
            ), 0) AS stage_failures,
            COALESCE(sc.total_cost_usd, 0)::numeric AS cost_usd
        FROM session_costs sc
        WHERE sc.session_date >= CURRENT_DATE - INTERVAL '1 day' * %s
        ORDER BY sc.session_date DESC
    """
    with get_cursor() as cur:
        cur.execute(sql, (days,))
        rows = cur.fetchall()
    return [
        {
            "session_id": r["session_id"],
            "session_date": r["session_date"],
            "decisions_count": r["decisions_count"],
            "pnl_usd": float(r["pnl_usd"]),
            "stage_failures": r["stage_failures"],
            "cost_usd": float(r["cost_usd"]),
        }
        for r in rows
    ]


def tool_get_executor_behavior_summary(days: int = 14) -> dict:
    """Read-only: sizing buckets, ticker concentration, round-trip count, hold rate."""
    with get_cursor() as cur:
        # Size histogram. Bucket by trade notional |quantity * price|.
        cur.execute(
            """
            SELECT
                CASE
                    WHEN ABS(COALESCE(quantity, 0) * COALESCE(price, 0)) < 100 THEN 'small'
                    WHEN ABS(COALESCE(quantity, 0) * COALESCE(price, 0)) < 300 THEN 'medium'
                    ELSE 'large'
                END AS bucket,
                COUNT(*) AS n
            FROM decisions
            WHERE date >= CURRENT_DATE - INTERVAL '1 day' * %s
            GROUP BY bucket
            """,
            (days,),
        )
        size_rows = cur.fetchall()

        # Ticker concentration (positions has no sector column).
        cur.execute(
            """
            SELECT ticker, COUNT(*) AS n
            FROM decisions
            WHERE date >= CURRENT_DATE - INTERVAL '1 day' * %s
            GROUP BY ticker
            ORDER BY n DESC
            LIMIT 20
            """,
            (days,),
        )
        ticker_rows = cur.fetchall()

        cur.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE date >= CURRENT_DATE - INTERVAL '1 day' * %s",
            (days,),
        )
        total = cur.fetchone()["n"]

        cur.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE date >= CURRENT_DATE - INTERVAL '1 day' * %s AND action = 'hold'",
            (days,),
        )
        holds = cur.fetchone()["n"]

    round_trips = analyze_round_trips(days=days, gap_days=7, min_pairs=2)
    return {
        "window_days": days,
        "decision_count": total or 0,
        "hold_rate": (holds / total) if total else 0.0,
        "size_histogram": {r["bucket"]: r["n"] for r in size_rows},
        "ticker_concentration": {r["ticker"]: r["n"] for r in ticker_rows},
        "round_trip_count": sum(t.pair_count for t in round_trips),
    }


# --- Tool Definitions for Claude ---

TOOL_DEFINITIONS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 6,
    },
    {
        "name": "get_market_snapshot",
        "description": "Current market: sectors, indices, movers, unusual volume.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_portfolio_state",
        "description": "Positions, open orders, cash, buying power.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_active_theses",
        "description": "Active trade theses with direction, triggers, invalidation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Filter by ticker"},
            },
            "required": [],
        },
    },
    {
        "name": "create_thesis",
        "description": (
            "Create trade thesis. Rejects if ticker has active thesis or is held. "
            "IMPORTANT: Thesis text is NARRATIVE only. Do NOT include current share counts, entry prices, P&L, or any numeric state in the `thesis`, `entry_trigger`, `exit_trigger`, or `invalidation` fields — those are computed from the positions table at read time. Numeric state you embed here will drift and cause incorrect decisions. "
            "REQUIRED: cite at least one news/macro/thesis signal via `signal_refs`. Calls without a valid signal_ref are rejected. Use adopt_thesis for orphan-position housekeeping where you have no fresh evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "direction": {"type": "string", "enum": ["long", "short", "avoid"]},
                "thesis": {"type": "string", "description": "Core reasoning"},
                "entry_trigger": {"type": "string", "description": "Entry conditions"},
                "exit_trigger": {"type": "string", "description": "Exit conditions"},
                "invalidation": {"type": "string", "description": "What proves thesis wrong"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "signal_refs": {
                    "type": "array",
                    "minItems": 1,
                    "description": (
                        "REQUIRED. Signals supporting this thesis. At least one entry "
                        "must reference a valid ID from get_news_signals / "
                        "get_macro_signals / get_active_theses. Invalid IDs are "
                        "stripped, and if none remain valid the call is rejected."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["news_signal", "macro_signal", "thesis"],
                            },
                            "id": {"type": "integer"},
                        },
                        "required": ["type", "id"],
                    },
                },
            },
            "required": [
                "ticker", "direction", "thesis",
                "entry_trigger", "exit_trigger", "invalidation", "confidence",
                "signal_refs",
            ],
        },
    },
    {
        "name": "adopt_thesis",
        "description": (
            "Adopt an existing portfolio position by creating a thesis. Use for orphan "
            "positions (held but no thesis). REQUIRES ticker to be in portfolio. "
            "Cite supporting signals via `signal_refs` if available — adopted positions "
            "may legitimately have none, in which case omit the field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "direction": {"type": "string", "enum": ["long", "short", "avoid"]},
                "thesis": {"type": "string", "description": "Why you believe in this position"},
                "exit_trigger": {"type": "string", "description": "When to exit"},
                "invalidation": {"type": "string", "description": "What proves thesis wrong"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "signal_refs": {
                    "type": "array",
                    "description": "Optional supporting signals (same shape as create_thesis).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["news_signal", "macro_signal", "thesis"],
                            },
                            "id": {"type": "integer"},
                        },
                        "required": ["type", "id"],
                    },
                },
            },
            "required": ["ticker", "direction", "thesis", "exit_trigger", "invalidation", "confidence"],
        },
    },
    {
        "name": "update_thesis",
        "description": (
            "Update thesis fields and/or append new signal citations. "
            "IMPORTANT: Thesis text is NARRATIVE only. Do NOT include current share counts, entry prices, P&L, or any numeric state in the `thesis`, `entry_trigger`, `exit_trigger`, or `invalidation` fields — those are computed from the positions table at read time. Numeric state you embed here will drift and cause incorrect decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "thesis_id": {"type": "integer"},
                "thesis": {"type": "string"},
                "entry_trigger": {"type": "string"},
                "exit_trigger": {"type": "string"},
                "invalidation": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "add_signal_refs": {
                    "type": "array",
                    "description": (
                        "Signals to append to the thesis as new evidence accumulates. "
                        "Idempotent — previously cited signals are not duplicated."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["news_signal", "macro_signal", "thesis"],
                            },
                            "id": {"type": "integer"},
                        },
                        "required": ["type", "id"],
                    },
                },
            },
            "required": ["thesis_id"],
        },
    },
    {
        "name": "close_thesis",
        "description": "Close thesis as invalidated, expired, or executed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thesis_id": {"type": "integer"},
                "status": {"type": "string", "enum": ["invalidated", "expired", "executed"]},
                "reason": {"type": "string"},
            },
            "required": ["thesis_id", "status", "reason"],
        },
    },
    {
        "name": "get_curated_news",
        "description": (
            "PREFERRED: Curated news signals — Haiku-filtered to the most relevant "
            "for today's market regime. Use this by default. Each line is [#<id>] "
            "(use IDs for signal_refs citation, same as get_news_signals)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Filter by ticker"},
                "days": {"type": "integer", "description": "Lookback days (default: 7)"},
                "target_n": {"type": "integer", "description": "Approx. number of signals to return (default: 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_news_signals",
        "description": (
            "RAW FIREHOSE: Unfiltered recent ticker news signals (could be 900+ items "
            "across 7 days). Use get_curated_news by default; reach for this only when "
            "you need to look past the filter for a specific ticker or theme."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Filter by ticker"},
                "days": {"type": "integer", "description": "Lookback days (default: 7)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_macro_context",
        "description": "Macro signals: Fed, trade, geopolitical, sector trends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback days (default: 7)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_macro_signals",
        "description": (
            "Recent macro signals as a list with IDs. Use this to find macro signal "
            "IDs to cite via signal_refs on create_thesis / update_thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback days (default: 7)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_signal_attribution",
        "description": "Win rates by signal type from historical outcomes.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_decision_history",
        "description": "Recent decisions with 7d/30d P&L outcomes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Lookback days (default: 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_recent_playbooks",
        "description": (
            "Past playbooks (planned actions) for recent sessions. "
            "Use BEFORE writing today's playbook to check whether today's "
            "actions would reverse what you wrote in recent sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Count (default: 3)"},
            },
            "required": [],
        },
    },
    {
        "name": "write_playbook",
        "description": (
            "Write today's playbook for the executor. REQUIRED every session. "
            "You author intents — NOT share counts. The trader resolves intents "
            "to exact shares against live portfolio state at execution time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "market_outlook": {"type": "string"},
                "priority_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "action": {"type": "string", "enum": ["buy", "sell"]},
                            "thesis_id": {"type": "integer"},
                            "reasoning": {"type": "string"},
                            "intent_type": {
                                "type": "string",
                                "enum": [
                                    "exit_full", "exit_partial_pct", "exit_dollar", "trim_to_portfolio_pct",
                                    "invest_dollar", "invest_portfolio_pct", "invest_buying_power_pct", "add_to_target_pct",
                                ],
                            },
                            "intent_magnitude": {"type": "number"},
                            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                        "required": ["ticker", "action", "reasoning", "confidence", "intent_type"],
                    },
                },
                "watch_list": {"type": "array", "items": {"type": "string"}},
                "risk_notes": {"type": "string"},
            },
            "required": ["market_outlook", "priority_actions", "watch_list", "risk_notes"],
        },
    },
    {
        "name": "get_strategy_identity",
        "description": "Trading identity: risk posture, sector biases, signal preferences.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_rules",
        "description": "Active strategy rules (constraints and preferences).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_strategy_history",
        "description": "Recent strategy reflection memos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Count (default: 5)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_retired_rules",
        "description": "Read-only. List the most recently retired strategy rules.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 50}},
        },
    },
    {
        "name": "get_rule_bind_history",
        "description": "Read-only. Decisions that cited a given rule within the window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "integer"},
                "days": {"type": "integer", "default": 30},
            },
            "required": ["rule_id"],
        },
    },
    {
        "name": "get_theses",
        "description": "Read-only. Theses filtered by status (all|active|closed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["all", "active", "closed"], "default": "all"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "get_thesis_lineage",
        "description": "Read-only. Decisions tagged with a thesis, with outcomes.",
        "input_schema": {
            "type": "object",
            "properties": {"thesis_id": {"type": "integer"}},
            "required": ["thesis_id"],
        },
    },
    {
        "name": "get_recent_decisions",
        "description": "Read-only. Compact recent decisions for behavior review.",
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "default": 14}},
        },
    },
    {
        "name": "get_decision_detail",
        "description": "Read-only. Full decision detail plus all referenced signal IDs.",
        "input_schema": {
            "type": "object",
            "properties": {"decision_id": {"type": "integer"}},
            "required": ["decision_id"],
        },
    },
    {
        "name": "get_flip_flop_report",
        "description": "Read-only. Tickers with N+ opposing-action pairs in the window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 30},
                "min_reversals": {"type": "integer", "default": 3},
            },
        },
    },
    {
        "name": "get_executor_behavior_summary",
        "description": "Read-only. Sizing, ticker concentration, round trips, hold rate.",
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "default": 14}},
        },
    },
    {
        "name": "get_session_memos",
        "description": "Read-only. Most recent strategy memos.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}},
    },
    {
        "name": "get_reflection_actions",
        "description": "Read-only. Per-session reflection-stage actions taken.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}},
    },
    {
        "name": "get_session_summary",
        "description": "Read-only. Per-session decisions, P&L (outcome_7d sum), failures, cost within a window.",
        "input_schema": {"type": "object", "properties": {"days": {"type": "integer", "default": 14}}},
    },
]


TOOL_HANDLERS = {
    "get_market_snapshot": tool_get_market_snapshot,
    "get_portfolio_state": tool_get_portfolio_state,
    "get_active_theses": tool_get_active_theses,
    "create_thesis": tool_create_thesis,
    "adopt_thesis": tool_adopt_thesis,
    "update_thesis": tool_update_thesis,
    "close_thesis": tool_close_thesis,
    "get_curated_news": tool_get_curated_news,
    "get_news_signals": tool_get_news_signals,
    "get_macro_context": tool_get_macro_context,
    "get_macro_signals": tool_get_macro_signals,
    "get_signal_attribution": tool_get_signal_attribution,
    "get_decision_history": tool_get_decision_history,
    "get_recent_playbooks": tool_get_recent_playbooks,
    "write_playbook": tool_write_playbook,
    "get_strategy_identity": tool_get_strategy_identity,
    "get_strategy_rules": tool_get_strategy_rules,
    "get_strategy_history": tool_get_strategy_history,
    "get_retired_rules": tool_get_retired_rules,
    "get_rule_bind_history": tool_get_rule_bind_history,
    "get_theses": tool_get_theses,
    "get_thesis_lineage": tool_get_thesis_lineage,
    "get_recent_decisions": tool_get_recent_decisions,
    "get_decision_detail": tool_get_decision_detail,
    "get_flip_flop_report": tool_get_flip_flop_report,
    "get_executor_behavior_summary": tool_get_executor_behavior_summary,
    "get_session_memos": tool_get_session_memos,
    "get_reflection_actions": tool_get_reflection_actions,
    "get_session_summary": tool_get_session_summary_window,
}
