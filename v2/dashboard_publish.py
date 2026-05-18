"""Dashboard data gathering for public GitHub Pages dashboard.

Queries the DB and structures data for JSON export.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .claude_client import _call_with_retry, get_claude_client
from .dashboard_og import (
    render_attribution_og,
    render_home_og,
    render_mistakes_og,
    render_thesis_og,
    render_trade_og,
)
from .dashboard_pages import (
    render_activity_page,
    render_attribution_page,
    render_changelog_page,
    render_homepage,
    render_how_it_works_hub,
    render_learning_hub,
    render_mistakes_page,
    render_performance_page,
    render_thesis_page,
    render_trade_page,
)
from .database.connection import get_cursor
from .database.trading_db import (
    get_closed_losers,
    get_recent_strategy_memos,
    get_retired_rules,
    get_signal_attribution,
)
from .executor import get_net_deposits

logger = logging.getLogger("dashboard_publish")

# Path to static assets directory (relative to project root)
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public_dashboard")


def _redact_order_id(order_id):
    """Truncate full Alpaca order UUIDs before publishing to the public dashboard.

    The frontend already truncates for display; the unredacted UUIDs were still
    being shipped in the publicly-fetchable JSON (P1.8). Match the frontend's
    `shortOrderId` shape (8 chars + '...') so display is unchanged.
    """
    if order_id is None:
        return None
    s = str(order_id)
    return s[:8] + "..." if len(s) > 12 else s


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal, date, and datetime types."""

    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def get_deposit_history() -> list[dict]:
    """Get chronological list of cash deposits/withdrawals from Alpaca.

    Returns list of {date, amount} dicts sorted by date.
    """
    from .executor import get_trading_client

    client = get_trading_client()
    history = []
    page_token = None

    while True:
        params = {"activity_types": "CSD,CSW", "page_size": 100, "direction": "asc"}
        if page_token:
            params["page_token"] = page_token

        activities = client.get("/account/activities", params)
        if not activities:
            break

        for a in activities:
            history.append({
                "date": a["date"],
                "amount": Decimal(str(a["net_amount"])),
            })

        if len(activities) < 100:
            break
        page_token = activities[-1]["id"]

    return history


def _enrich_snapshots_with_deposits(snapshots: list[dict], deposit_history: list[dict]) -> None:
    """Add cumulative_deposits to each snapshot dict, in place.

    Deposits are credited on the first snapshot where portfolio_value reflects
    the new cash. We detect this by checking if the portfolio jump between
    consecutive snapshots is close to a known deposit amount.
    """
    if not snapshots:
        return

    if not deposit_history:
        # No deposit data — assume first snapshot value is the only deposit
        base = snapshots[0]["portfolio_value"]
        for s in snapshots:
            s["cumulative_deposits"] = base
        return

    # Sort deposits chronologically and build a list of (date_str, amount)
    sorted_deposits = sorted(deposit_history, key=lambda x: str(x["date"]))

    # For each snapshot, cumulative_deposits = sum of deposits whose effect
    # is visible. A deposit on date D is reflected in the snapshot on or after
    # the date where portfolio_value jumps by roughly the deposit amount.
    # Simpler approach: credit deposit on the first snapshot where
    # snap.date > deposit.date (next trading day after deposit settles).
    cum = Decimal("0")
    dep_idx = 0
    for s in snapshots:
        snap_date = str(s["date"])
        # Credit all deposits with date strictly before this snapshot
        while dep_idx < len(sorted_deposits) and str(sorted_deposits[dep_idx]["date"]) < snap_date:
            cum += sorted_deposits[dep_idx]["amount"]
            dep_idx += 1
        s["cumulative_deposits"] = cum

    # If the first snapshot still has 0 cumulative_deposits, the very first
    # deposit happened on or before the first snapshot — credit it.
    # Only snapshots dated <= first_snap_date were missed by the strict-<
    # first loop; later snapshots already saw those deposits and crediting
    # them again would double-count.
    if snapshots[0]["cumulative_deposits"] == 0 and sorted_deposits:
        first_dep_date = str(sorted_deposits[0]["date"])
        first_snap_date = str(snapshots[0]["date"])
        if first_dep_date <= first_snap_date:
            credit = Decimal("0")
            for d in sorted_deposits:
                if str(d["date"]) <= first_snap_date:
                    credit += d["amount"]
                else:
                    break
            for s in snapshots:
                if str(s["date"]) <= first_snap_date:
                    s["cumulative_deposits"] += credit


def _enrich_snapshots_with_spy_match(
    snapshots: list[dict],
    deposit_history: list[dict],
    benchmark: list[dict],
) -> None:
    """Add `spy_value_if_deposited` to each snapshot, in place.

    Models a shadow portfolio that buys SPY at each deposit's date and marks
    to market on every snapshot date. Mirrors the same crediting rule used by
    `_enrich_snapshots_with_deposits` so a deposit hitting cumulative_deposits
    on snapshot S also adds SPY shares to the shadow portfolio on S.

    Powers the equity-curve chart's deposit-matched SPY benchmark — "what the
    same cash would have been worth had it gone into SPY instead." Without
    this match, comparing actual equity to a single-lump SPY normalization is
    misleading once the account has multiple deposits.
    """
    if not snapshots:
        return

    spy_by_date = {str(b["date"]): float(b["close"]) for b in (benchmark or [])}

    if not spy_by_date or not deposit_history:
        for s in snapshots:
            s["spy_value_if_deposited"] = None
        return

    sorted_spy_dates = sorted(spy_by_date.keys())

    def spy_close_on_or_after(d_str: str) -> float | None:
        for sd in sorted_spy_dates:
            if sd >= d_str:
                return spy_by_date[sd]
        return None

    sorted_deposits = sorted(deposit_history, key=lambda x: str(x["date"]))
    # (date_str, amount, spy_price_at_purchase) — drop deposits we can't price
    # (e.g. pre-history of benchmark window).
    purchases: list[tuple[str, float, float]] = []
    for d in sorted_deposits:
        d_date = str(d["date"])
        spy_buy = spy_close_on_or_after(d_date)
        if spy_buy and spy_buy > 0:
            purchases.append((d_date, float(d["amount"]), spy_buy))

    if not purchases:
        for s in snapshots:
            s["spy_value_if_deposited"] = None
        return

    first_snap_date = str(snapshots[0]["date"])

    for s in snapshots:
        snap_date = str(s["date"])
        spy_now = spy_by_date.get(snap_date)
        if spy_now is None:
            # Snapshot date isn't a SPY trading day — fall back to most recent
            # prior close so the shadow portfolio still has a value.
            for sd in reversed(sorted_spy_dates):
                if sd <= snap_date:
                    spy_now = spy_by_date[sd]
                    break
        if spy_now is None:
            # Snapshot precedes every SPY close in the window (e.g. a weekend
            # snapshot at the start of the benchmark range). Use the first
            # available close so the chart still has a baseline point.
            spy_now = spy_close_on_or_after(snap_date)
        if spy_now is None:
            s["spy_value_if_deposited"] = None
            continue

        if snap_date <= first_snap_date:
            credited = [p for p in purchases if p[0] <= first_snap_date]
        else:
            credited = [p for p in purchases if p[0] < snap_date]

        shares = sum(amt / buy_price for (_, amt, buy_price) in credited)
        s["spy_value_if_deposited"] = round(shares * spy_now, 2)


def _enrich_snapshots_with_twr_value(snapshots: list[dict]) -> None:
    """Add twr_value to each snapshot, in place.

    twr_value = snapshots[0].portfolio_value compounded by each day's
    trading-only growth factor. External cash flows (deposits/withdrawals)
    are subtracted from the growth calculation so the resulting line
    represents "what one dollar of starting capital grew to" — usable on
    the same dollar axis as SPY for apples-to-apples comparison.

    Requires `cumulative_deposits` to already be populated on each snapshot.
    """
    if not snapshots:
        return

    snapshots[0]["twr_value"] = snapshots[0]["portfolio_value"]

    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        deposit = curr.get("cumulative_deposits", Decimal("0")) - prev.get("cumulative_deposits", Decimal("0"))
        start_capital = prev["portfolio_value"] + deposit
        if start_capital > 0:
            growth = curr["portfolio_value"] / start_capital
            curr["twr_value"] = prev["twr_value"] * growth
        else:
            curr["twr_value"] = prev["twr_value"]


def compute_performance_stats(*, snapshots: list[dict],
                              decisions: list[dict]) -> dict:
    """Derive Performance-page stats from raw snapshots and closed decisions.

    Each snapshot must carry a "value" key. Each decision may carry an
    "outcome_30d_pct" key (None for still-open decisions, which are
    excluded from the win-rate computation).
    """
    if not snapshots and not decisions:
        return {
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "avg_days_held": 0.0,
            "best_day_pct": 0.0,
            "worst_day_pct": 0.0,
        }

    values = [float(s["value"]) for s in snapshots]

    max_dd = 0.0
    if values:
        peak = values[0]
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100.0 if peak else 0.0
            if dd < max_dd:
                max_dd = dd

    best, worst = 0.0, 0.0
    for prev, curr in zip(values, values[1:], strict=False):
        if prev <= 0:
            continue
        pct = (curr - prev) / prev * 100.0
        if pct > best:
            best = pct
        if pct < worst:
            worst = pct

    closed = [d for d in decisions if d.get("outcome_30d_pct") is not None]
    if closed:
        wins = sum(1 for d in closed if float(d["outcome_30d_pct"]) > 0)
        win_rate = wins / len(closed) * 100.0
    else:
        win_rate = 0.0

    avg_days_held = 0.0

    return {
        "max_drawdown_pct": round(max_dd, 4),
        "win_rate_pct": round(win_rate, 4),
        "avg_days_held": round(avg_days_held, 4),
        "best_day_pct": round(best, 4),
        "worst_day_pct": round(worst, 4),
    }


def render_sparkline_svg(snapshots: list[dict]) -> str:
    """Render the last 90 days of equity as an inline SVG polyline.

    Returns "" when fewer than 7 snapshots are supplied — the homepage
    template hides the sparkline in that case.
    """
    if not snapshots or len(snapshots) < 7:
        return ""

    series = [float(s["value"]) for s in snapshots[-90:]]
    n = len(series)
    lo, hi = min(series), max(series)
    span = hi - lo if hi > lo else 1.0
    flat = (hi == lo)

    width, height, pad_y = 400.0, 60.0, 5.0
    plot_h = height - 2 * pad_y

    points = []
    for i, v in enumerate(series):
        x = (i / (n - 1)) * width if n > 1 else width / 2
        if flat:
            y = height / 2
        else:
            y = pad_y + (1.0 - (v - lo) / span) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    return (
        f'<svg class="sparkline" viewBox="0 0 400 60" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{" ".join(points)}" />'
        f'</svg>'
    )


def gather_dashboard_data(session_date: date, net_deposits: Decimal | None = None) -> dict:
    """Gather all dashboard data in a single DB connection.

    Args:
        session_date: The trading session date.
        net_deposits: Total net cash deposited (from Alpaca activities).
            Used for accurate total return calculation excluding cash infusions.

    Returns dict with keys: summary, snapshots, positions, decisions, theses, benchmark.
    Handles empty DB gracefully.
    """
    with get_cursor() as cur:
        # Snapshots: last 90 days, ordered ASC
        cur.execute(
            """
            SELECT date, portfolio_value, cash, buying_power
            FROM account_snapshots
            WHERE date > %s - INTERVAL '90 days'
            ORDER BY date ASC
            """,
            (session_date,),
        )
        snapshots = cur.fetchall()

        # Positions: all, ordered by ticker
        cur.execute(
            "SELECT ticker, shares, avg_cost, updated_at FROM positions ORDER BY ticker"
        )
        positions = cur.fetchall()

        # Decisions: last 30 days, ordered DESC
        cur.execute(
            """
            SELECT id, date, ticker, action, quantity, price, reasoning,
                   outcome_7d, outcome_30d, order_id
            FROM decisions
            WHERE date > %s - INTERVAL '30 days'
            ORDER BY date DESC, id DESC
            """,
            (session_date,),
        )
        decisions = cur.fetchall()

        # Theses: active only, ordered DESC
        cur.execute(
            """
            SELECT id, ticker, direction, confidence, thesis,
                   entry_trigger, exit_trigger, created_at
            FROM theses
            WHERE status = 'active'
            ORDER BY created_at DESC
            """
        )
        theses = cur.fetchall()

        # Latest snapshot for summary
        cur.execute(
            """
            SELECT portfolio_value, cash, long_market_value
            FROM account_snapshots
            ORDER BY date DESC LIMIT 1
            """
        )
        latest = cur.fetchone()

        # First snapshot ever (for total return)
        cur.execute(
            """
            SELECT portfolio_value, date
            FROM account_snapshots
            ORDER BY date ASC LIMIT 1
            """
        )
        first = cur.fetchone()

        # Previous snapshot (for daily P&L)
        cur.execute(
            """
            SELECT portfolio_value
            FROM account_snapshots
            ORDER BY date DESC LIMIT 1 OFFSET 1
            """
        )
        previous = cur.fetchone()

    # Fetch SPY benchmark for the same date range as snapshots
    benchmark = []
    if snapshots:
        start = snapshots[0]["date"]
        end = snapshots[-1]["date"]
        benchmark = fetch_spy_benchmark(start, end)

    # Enrich snapshots with cumulative deposit data for accurate return calc
    snapshot_dicts = [dict(r) for r in snapshots]
    try:
        deposit_history = get_deposit_history()
    except Exception:
        logger.warning("Could not fetch deposit history", exc_info=True)
        deposit_history = []
    _enrich_snapshots_with_deposits(snapshot_dicts, deposit_history)
    _enrich_snapshots_with_twr_value(snapshot_dicts)
    _enrich_snapshots_with_spy_match(snapshot_dicts, deposit_history, benchmark)

    # Daily deposit is the jump in cumulative_deposits between the last two
    # snapshots — matches the credit semantics in _enrich_snapshots_with_deposits.
    daily_deposit = Decimal("0")
    if len(snapshot_dicts) >= 2:
        daily_deposit = (
            snapshot_dicts[-1].get("cumulative_deposits", Decimal("0"))
            - snapshot_dicts[-2].get("cumulative_deposits", Decimal("0"))
        )

    # Build summary (needs daily_deposit to exclude cash-ins from daily P&L)
    summary = _build_summary(latest, first, previous, len(positions), session_date,
                             net_deposits, daily_deposit=daily_deposit)

    # Renderer-facing aliases. The dashboard pages spec uses `total_return_pct`
    # for the hero strip; the publisher historically emitted `total_pnl_pct`.
    # Keep both so the legacy summary contract still holds.
    summary["total_return_pct"] = summary["total_pnl_pct"]

    # vs_spy_pct: portfolio total return minus a deposit-matched SPY shadow
    # portfolio's return over the same window. Same denominator (net_deposits)
    # so the spread reads as alpha. None when we lack the data to compute it.
    vs_spy_pct = None
    if snapshot_dicts and net_deposits and net_deposits != 0:
        spy_value = snapshot_dicts[-1].get("spy_value_if_deposited")
        if spy_value is not None:
            spy_return_pct = (Decimal(str(spy_value)) - net_deposits) / net_deposits * 100
            vs_spy_pct = summary["total_pnl_pct"] - spy_return_pct
    summary["vs_spy_pct"] = vs_spy_pct

    # NEW: gather full ID lists for page emission (link permanence)
    with get_cursor() as cur2:
        pages = gather_all_pages_data(cur2)

    # Mistakes log (closed losers + recently retired rules)
    try:
        closed_losers = get_closed_losers(reference_date=session_date, limit=15)
    except Exception:
        logger.warning("Failed to gather closed losers", exc_info=True)
        closed_losers = []
    try:
        retired_rules = get_retired_rules(reference_date=session_date, limit=10)
    except Exception:
        logger.warning("Failed to gather retired rules", exc_info=True)
        retired_rules = []
    mistakes = {
        "closed_losers": [dict(r) for r in closed_losers],
        "retired_rules": [dict(r) for r in retired_rules],
    }

    # Signal attribution snapshot
    try:
        attribution_rows = get_signal_attribution()
    except Exception:
        logger.warning("Failed to gather signal attribution", exc_info=True)
        attribution_rows = []
    attribution = [dict(r) for r in attribution_rows]

    try:
        memo_rows = get_recent_strategy_memos(n=10)
    except Exception:
        logger.warning("Failed to gather strategy memos", exc_info=True)
        memo_rows = []
    memos = [
        {
            "id": m["id"],
            "session_date": m["session_date"],
            "memo_type": m.get("memo_type"),
            "content": m["content"],
        }
        for m in memo_rows
    ]

    decision_dicts = [
        {**dict(r), "order_id": _redact_order_id(dict(r).get("order_id"))}
        for r in decisions
    ]

    performance = compute_performance_stats(
        snapshots=[
            {"value": s.get("twr_value", s.get("portfolio_value"))}
            for s in snapshot_dicts
        ],
        decisions=[
            {"outcome_30d_pct": d.get("outcome_30d")} for d in decision_dicts
        ],
    )

    return {
        "summary": summary,
        "snapshots": snapshot_dicts,
        "positions": [dict(r) for r in positions],
        "decisions": decision_dicts,
        "theses": [dict(r) for r in theses],
        "benchmark": benchmark,
        "mistakes": mistakes,
        "attribution": attribution,
        "memos": memos,
        "performance": performance,
        "_pages": pages,  # NEW
    }


def _fetch_signal_refs(cur, link_table: str, link_col: str, link_id: int) -> list[dict]:
    """Hydrate signal citations for a decision or thesis.

    INNER JOINs to news_signals/macro_signals/theses so orphan FKs (parent
    deleted) drop out — same defense pattern used by attribution.py and
    patterns.py. Returns one flat list with `signal_type` set on each row;
    consumers render-side group by type.
    """
    refs: list[dict] = []

    # f-string interpolation is safe because link_table and link_col come from
    # hardcoded callers below (decision_signals/decision_id, thesis_signals/
    # thesis_id), not user input. Assert the invariant so a future refactor
    # trips immediately.
    assert link_table in {"decision_signals", "thesis_signals"}, link_table
    assert link_col in {"decision_id", "thesis_id"}, link_col

    cur.execute(
        f"""
        SELECT ls.signal_id, ns.ticker, ns.headline, ns.category, ns.sentiment,
               ns.published_at
        FROM {link_table} ls
        JOIN news_signals ns ON ns.id = ls.signal_id
        WHERE ls.{link_col} = %s AND ls.signal_type = 'news_signal'
        ORDER BY ns.published_at DESC NULLS LAST, ls.signal_id
        """,  # noqa: S608
        (link_id,),
    )
    for row in cur.fetchall():
        d = dict(row)
        d["signal_type"] = "news_signal"
        refs.append(d)

    cur.execute(
        f"""
        SELECT ls.signal_id, ms.headline, ms.category, ms.sentiment,
               ms.affected_sectors, ms.published_at
        FROM {link_table} ls
        JOIN macro_signals ms ON ms.id = ls.signal_id
        WHERE ls.{link_col} = %s AND ls.signal_type = 'macro_signal'
        ORDER BY ms.published_at DESC NULLS LAST, ls.signal_id
        """,  # noqa: S608
        (link_id,),
    )
    for row in cur.fetchall():
        d = dict(row)
        d["signal_type"] = "macro_signal"
        refs.append(d)

    cur.execute(
        f"""
        SELECT ls.signal_id, t.ticker, t.direction, t.thesis, t.confidence, t.status
        FROM {link_table} ls
        JOIN theses t ON t.id = ls.signal_id
        WHERE ls.{link_col} = %s AND ls.signal_type = 'thesis'
        ORDER BY ls.signal_id
        """,  # noqa: S608
        (link_id,),
    )
    for row in cur.fetchall():
        d = dict(row)
        d["signal_type"] = "thesis"
        refs.append(d)

    return refs


def gather_trade_detail(cur, decision_id: int) -> dict | None:
    """Return full detail for one decision page: decision + thesis + position + signal_refs.

    Caller passes a cursor so this can run in any open transaction.
    Returns None if the decision_id doesn't exist.

    thesis_id is resolved via decisions.playbook_action_id -> playbook_actions.thesis_id
    since the decisions table does not carry thesis_id directly.
    """
    cur.execute(
        """
        SELECT d.id, d.date, d.ticker, d.action, d.quantity, d.price, d.reasoning,
               d.outcome_7d, d.outcome_30d, d.order_id,
               pa.thesis_id
        FROM decisions d
        LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
        WHERE d.id = %s
        """,
        (decision_id,),
    )
    decision = cur.fetchone()
    if decision is None:
        return None
    decision = dict(decision)
    decision["order_id"] = _redact_order_id(decision.get("order_id"))

    thesis = None
    if decision.get("thesis_id"):
        cur.execute(
            """
            SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
                   invalidation, confidence, status
            FROM theses WHERE id = %s
            """,
            (decision["thesis_id"],),
        )
        row = cur.fetchone()
        thesis = dict(row) if row else None

    cur.execute(
        "SELECT ticker, shares, avg_cost FROM positions WHERE ticker = %s",
        (decision["ticker"],),
    )
    pos_row = cur.fetchone()
    position = dict(pos_row) if pos_row else None

    signal_refs = _fetch_signal_refs(cur, "decision_signals", "decision_id", decision_id)

    return {
        "decision": decision,
        "thesis": thesis,
        "position": position,
        "signal_refs": signal_refs,
    }


def gather_thesis_detail(cur, thesis_id: int) -> dict | None:
    """Return full detail for one thesis page: thesis + decisions + position."""
    cur.execute(
        """
        SELECT id, ticker, direction, thesis, entry_trigger, exit_trigger,
               invalidation, confidence, status
        FROM theses WHERE id = %s
        """,
        (thesis_id,),
    )
    thesis = cur.fetchone()
    if thesis is None:
        return None
    thesis = dict(thesis)

    cur.execute(
        """
        SELECT d.id, d.date, d.ticker, d.action, d.quantity, d.price,
               d.outcome_7d, d.outcome_30d
        FROM decisions d
        JOIN playbook_actions pa ON pa.id = d.playbook_action_id
        WHERE pa.thesis_id = %s
        ORDER BY d.date DESC, d.id DESC
        """,
        (thesis_id,),
    )
    decisions = [dict(r) for r in cur.fetchall()]

    cur.execute(
        "SELECT ticker, shares, avg_cost FROM positions WHERE ticker = %s",
        (thesis["ticker"],),
    )
    pos_row = cur.fetchone()
    position = dict(pos_row) if pos_row else None

    signal_refs = _fetch_signal_refs(cur, "thesis_signals", "thesis_id", thesis_id)

    return {
        "thesis": thesis,
        "decisions": decisions,
        "position": position,
        "signal_refs": signal_refs,
    }


def gather_all_pages_data(cur) -> dict:
    """Return ID lists for every decision and thesis we need to emit pages for.

    No date filter — Cloudflare Pages does full-bundle replacement on each
    deploy, so any URL not in this deploy will 404. Link permanence is a hard
    requirement of the audience-growth strategy.
    """
    cur.execute("SELECT id FROM decisions ORDER BY id")
    decision_ids = [r["id"] for r in cur.fetchall()]
    cur.execute("SELECT id FROM theses ORDER BY id")
    thesis_ids = [r["id"] for r in cur.fetchall()]
    return {"decision_ids": decision_ids, "thesis_ids": thesis_ids}


def emit_home_og_image(summary: dict, deploy_dir: str) -> None:
    """Render the homepage OG card to deploy_dir/og/home.png."""
    try:
        png = render_home_og(summary)
        og_dir = os.path.join(deploy_dir, "og")
        os.makedirs(og_dir, exist_ok=True)
        with open(os.path.join(og_dir, "home.png"), "wb") as f:
            f.write(png)
    except Exception:
        logger.warning("Failed to render homepage OG image", exc_info=True)


def emit_static_pages(data: dict, deploy_dir: str, base_url: str) -> None:
    """Write /mistakes/index.html, /attribution/index.html, and the
    matching OG PNGs into deploy_dir.

    No-op when base_url is empty (local-only build path).
    """
    if not base_url:
        return

    mistakes = data.get("mistakes") or {"closed_losers": [], "retired_rules": []}
    attribution = data.get("attribution") or []

    # /mistakes/index.html
    try:
        html = render_mistakes_page(
            closed_losers=mistakes.get("closed_losers", []),
            retired_rules=mistakes.get("retired_rules", []),
            base_url=base_url,
        )
        page_dir = os.path.join(deploy_dir, "mistakes")
        os.makedirs(page_dir, exist_ok=True)
        with open(os.path.join(page_dir, "index.html"), "w") as f:
            f.write(html)
    except Exception:
        logger.warning("Failed to render /mistakes/", exc_info=True)

    # /attribution/index.html
    try:
        html = render_attribution_page(
            attribution=attribution,
            base_url=base_url,
        )
        page_dir = os.path.join(deploy_dir, "attribution")
        os.makedirs(page_dir, exist_ok=True)
        with open(os.path.join(page_dir, "index.html"), "w") as f:
            f.write(html)
    except Exception:
        logger.warning("Failed to render /attribution/", exc_info=True)

    # OG images
    og_dir = os.path.join(deploy_dir, "og")
    os.makedirs(og_dir, exist_ok=True)
    losers = mistakes.get("closed_losers", [])
    top_loser = losers[0] if losers else None
    try:
        png = render_mistakes_og(top_loser=top_loser)
        with open(os.path.join(og_dir, "mistakes.png"), "wb") as f:
            f.write(png)
    except Exception:
        logger.warning("Failed to render mistakes OG", exc_info=True)
    try:
        png = render_attribution_og(attribution=attribution)
        with open(os.path.join(og_dir, "attribution.png"), "wb") as f:
            f.write(png)
    except Exception:
        logger.warning("Failed to render attribution OG", exc_info=True)


def emit_detail_pages(cur, decision_ids: list[int], thesis_ids: list[int],
                      deploy_dir: str, base_url: str) -> dict:
    """Render per-trade and per-thesis HTML pages into deploy_dir.

    Returns a stats dict: {trades_written, theses_written, failed}.
    Per-page failures are isolated: one bad render doesn't abort the run.
    """
    stats = {"trades_written": 0, "theses_written": 0, "failed": 0}

    for did in decision_ids:
        try:
            detail = gather_trade_detail(cur, did)
            if detail is None:
                continue
            html = render_trade_page(
                decision=detail["decision"],
                thesis=detail["thesis"],
                position=detail["position"],
                base_url=base_url,
                signal_refs=detail.get("signal_refs", []),
            )
            page_dir = os.path.join(deploy_dir, "trade", str(did))
            os.makedirs(page_dir, exist_ok=True)
            with open(os.path.join(page_dir, "index.html"), "w") as f:
                f.write(html)
            stats["trades_written"] += 1
        except Exception:
            logger.warning("Failed to render trade page %s", did, exc_info=True)
            stats["failed"] += 1

    for tid in thesis_ids:
        try:
            detail = gather_thesis_detail(cur, tid)
            if detail is None:
                continue
            html = render_thesis_page(
                thesis=detail["thesis"],
                decisions=detail["decisions"],
                position=detail["position"],
                base_url=base_url,
                signal_refs=detail.get("signal_refs", []),
            )
            page_dir = os.path.join(deploy_dir, "thesis", str(tid))
            os.makedirs(page_dir, exist_ok=True)
            with open(os.path.join(page_dir, "index.html"), "w") as f:
                f.write(html)
            stats["theses_written"] += 1
        except Exception:
            logger.warning("Failed to render thesis page %s", tid, exc_info=True)
            stats["failed"] += 1

    return stats


def emit_og_images(cur, decision_ids: list[int], thesis_ids: list[int],
                   deploy_dir: str) -> dict:
    """Render OG PNGs for each decision and thesis into deploy_dir/og/."""
    stats = {"trades_written": 0, "theses_written": 0, "failed": 0}

    trade_dir = os.path.join(deploy_dir, "og", "trade")
    thesis_dir = os.path.join(deploy_dir, "og", "thesis")
    os.makedirs(trade_dir, exist_ok=True)
    os.makedirs(thesis_dir, exist_ok=True)

    for did in decision_ids:
        try:
            detail = gather_trade_detail(cur, did)
            if detail is None:
                continue
            png = render_trade_og(detail["decision"])
            with open(os.path.join(trade_dir, f"{did}.png"), "wb") as f:
                f.write(png)
            stats["trades_written"] += 1
        except Exception:
            logger.warning("Failed to render trade OG %s", did, exc_info=True)
            stats["failed"] += 1

    for tid in thesis_ids:
        try:
            detail = gather_thesis_detail(cur, tid)
            if detail is None:
                continue
            png = render_thesis_og(detail["thesis"])
            with open(os.path.join(thesis_dir, f"{tid}.png"), "wb") as f:
                f.write(png)
            stats["theses_written"] += 1
        except Exception:
            logger.warning("Failed to render thesis OG %s", tid, exc_info=True)
            stats["failed"] += 1

    return stats


def _build_summary(latest, first, previous, positions_count, session_date,
                   net_deposits=None, daily_deposit=Decimal("0")):
    """Build summary dict from query results.

    daily_deposit: net cash deposited between `previous` and `latest` snapshots.
        Subtracted from the daily delta so the card reflects trading P&L only.
    """
    if not latest:
        return {
            "portfolio_value": 0,
            "cash": 0,
            "invested": 0,
            "positions_count": 0,
            "last_updated": session_date.isoformat(),
            "daily_pnl": 0,
            "daily_pnl_pct": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "inception_date": None,
        }

    portfolio_value = latest["portfolio_value"]
    cash = latest["cash"]
    long_market_value = latest.get("long_market_value") or (portfolio_value - cash)

    # Daily P&L — subtract same-day deposits so card reflects trading only.
    daily_pnl = Decimal("0")
    daily_pnl_pct = Decimal("0")
    if previous and previous["portfolio_value"]:
        prev_value = previous["portfolio_value"]
        daily_pnl = portfolio_value - prev_value - daily_deposit
        base = prev_value + daily_deposit
        if base != 0:
            daily_pnl_pct = (daily_pnl / base) * 100

    # Total P&L: investment return only (excludes cash infusions)
    total_pnl = Decimal("0")
    total_pnl_pct = Decimal("0")
    inception_date = None
    if first:
        inception_date = first["date"]
    if net_deposits is not None and net_deposits != 0:
        total_pnl = portfolio_value - net_deposits
        total_pnl_pct = (total_pnl / net_deposits) * 100
    elif first and first["portfolio_value"]:
        # Fallback if net_deposits not available
        first_value = first["portfolio_value"]
        total_pnl = portfolio_value - first_value
        if first_value != 0:
            total_pnl_pct = (total_pnl / first_value) * 100

    return {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "invested": long_market_value,
        "positions_count": positions_count,
        "last_updated": session_date.isoformat(),
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "inception_date": inception_date,
    }


def fetch_spy_benchmark(start_date: date, end_date: date) -> list[dict]:
    """Fetch SPY daily bars from Alpaca for benchmark comparison.

    Returns list of {date, close} dicts, or [] on error.
    """
    try:
        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        client = StockHistoricalDataClient(api_key, secret_key)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.max.time()),
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request)
        spy_bars = list(bars["SPY"])

        if not spy_bars:
            return []

        return [
            {"date": bar.timestamp.strftime("%Y-%m-%d"), "close": bar.close}
            for bar in spy_bars
        ]
    except Exception:
        logger.warning("Failed to fetch SPY benchmark data", exc_info=True)
        return []


def write_json_files(data: dict, repo_path: str) -> list[str]:
    """Write dashboard data as separate JSON files for GitHub Pages.

    Creates a data/ subdirectory under repo_path and writes each key
    from the data dict as a separate JSON file.

    Returns list of file paths written.
    """
    data_dir = os.path.join(repo_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    files_written = []
    for key in (
        "summary", "snapshots", "positions", "decisions",
        "theses", "benchmark", "mistakes", "attribution", "memos",
        "performance",
    ):
        if key not in data:
            continue
        file_path = os.path.join(data_dir, f"{key}.json")
        with open(file_path, "w") as f:
            json.dump(data[key], f, cls=_DecimalEncoder, indent=2)
        logger.info("Wrote %s", file_path)
        files_written.append(file_path)

    return files_written


# Static asset filenames to copy from public_dashboard/
_STATIC_ASSETS = ("styles.css", "app.js")
_CHANGELOG_POINTER_KEY = "changelog_last_published_sha"
_CHANGELOG_PROMPT_VERSION = "public_changelog_v1"
_CHANGELOG_MODEL = "claude-haiku-4-5"


def get_current_git_sha(repo_path: str | None = None) -> str | None:
    """Return the current repository HEAD SHA, or None if git is unavailable."""
    root = repo_path or os.path.dirname(os.path.dirname(__file__))
    try:
        result = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:
        logger.warning("Failed to resolve current git SHA", exc_info=True)
        return None
    sha = result.stdout.strip()
    return sha or None


def get_changelog_pointer(cur) -> str | None:
    """Return the last git SHA successfully published to the dashboard."""
    cur.execute(
        "SELECT value FROM dashboard_publish_state WHERE key = %s",
        (_CHANGELOG_POINTER_KEY,),
    )
    row = cur.fetchone()
    return row["value"] if row else None


def update_changelog_pointer(cur, sha: str) -> None:
    """Persist the git SHA that was successfully published to the dashboard."""
    cur.execute(
        """
        INSERT INTO dashboard_publish_state (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = EXCLUDED.updated_at
        """,
        (_CHANGELOG_POINTER_KEY, sha),
    )


def read_changelog_pointer() -> str | None:
    """Read the last successfully published changelog SHA from the DB."""
    with get_cursor() as cur:
        return get_changelog_pointer(cur)


def persist_changelog_pointer(sha: str) -> None:
    """Write the latest successfully published changelog SHA to the DB."""
    with get_cursor() as cur:
        update_changelog_pointer(cur, sha)


def fetch_changelog_commits(repo_path: str | None = None,
                            from_sha: str | None = None,
                            to_sha: str | None = None,
                            bootstrap_limit: int = 30) -> list[dict]:
    """Fetch new changelog commit rows from git.

    When `from_sha` is present, only commits in `from_sha..to_sha` are read.
    On first publish, there is no pointer yet, so this bootstraps with the most
    recent `bootstrap_limit` commits and the successful publish then advances
    the pointer.
    """
    root = repo_path or os.path.dirname(os.path.dirname(__file__))
    target = to_sha or "HEAD"
    log_args = [
        "git", "-C", root, "log", "--no-merges",
        "--pretty=format:%cI%x1f%H%x1f%h%x1f%s%x1f%b%x1e",
    ]
    if from_sha:
        if from_sha == target:
            return []
        log_args.append(f"{from_sha}..{target}")
    else:
        log_args.extend([f"-n{bootstrap_limit}", target])

    try:
        result = subprocess.run(
            log_args,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:
        logger.warning("Failed to generate changelog from git history", exc_info=True)
        return []

    commits: list[dict] = []
    for raw in result.stdout.split("\x1e"):
        # Strip only newlines — str.strip() would eat the trailing \x1f
        # that delimits an empty body and break the field count below.
        raw = raw.strip("\n")
        if not raw:
            continue
        parts = raw.split("\x1f", 4)
        if len(parts) != 5:
            continue
        committed_at, sha, short_sha, subject, body = parts
        commits.append({
            "sha": sha,
            "short_sha": short_sha,
            "committed_at": committed_at,
            "subject": subject,
            "body": body.strip(),
            "files": fetch_commit_files(sha, repo_path=root),
        })
    return commits


def fetch_commit_files(sha: str, repo_path: str | None = None) -> list[str]:
    """Return files touched by a commit, best-effort."""
    root = repo_path or os.path.dirname(os.path.dirname(__file__))
    try:
        result = subprocess.run(
            ["git", "-C", root, "show", "--name-only", "--pretty=format:", sha],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:
        logger.warning("Failed to fetch files for changelog commit %s", sha, exc_info=True)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _commit_date(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)[:10]


def group_changelog_commits(commits: list[dict]) -> list[dict]:
    """Group flat changelog commit rows into date buckets for rendering."""
    entries_by_date: dict[str, list[dict]] = {}
    for commit in commits:
        commit_date = _commit_date(commit.get("committed_at"))
        entries_by_date.setdefault(commit_date, []).append({
            "sha": commit.get("sha"),
            "short_sha": commit.get("short_sha"),
            "subject": commit.get("subject"),
        })

    entries = []
    for commit_date, items in entries_by_date.items():
        commit_count = len(items)
        noun = "commit" if commit_count == 1 else "commits"
        entries.append({
            "date": commit_date,
            "title": "Repository updates",
            "summary": f"{commit_count} {noun} published from git history.",
            "items": items,
        })
    return entries


def generate_changelog_entries(repo_path: str | None = None,
                               from_sha: str | None = None,
                               to_sha: str | None = None,
                               bootstrap_limit: int = 30) -> list[dict]:
    """Build changelog entries from git without reading stored DB rows."""
    commits = fetch_changelog_commits(
        repo_path=repo_path,
        from_sha=from_sha,
        to_sha=to_sha,
        bootstrap_limit=bootstrap_limit,
    )
    return group_changelog_commits(commits)


def _extract_text_blocks(message) -> str:
    parts = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _changelog_prompt(commits: list[dict]) -> str:
    payload = [
        {
            "sha": c.get("sha"),
            "short_sha": c.get("short_sha"),
            "committed_at": c.get("committed_at"),
            "subject": c.get("subject"),
            "body": c.get("body") or "",
            "files": c.get("files") or [],
        }
        for c in commits
    ]
    return (
        "You are writing a public changelog for Pinchy, an AI trading dashboard.\n\n"
        "Audience: people following a public AI-operated trading account.\n\n"
        "Create public product notes from these repository commits. Group related "
        "commits into meaningful entries. Ignore internal-only cleanup unless it "
        "changes reliability, transparency, safety, publishing, dashboard UX, or "
        "trading behavior. Do not invent user-visible behavior. Do not mention raw "
        "implementation details unless they explain a user-facing change or safety "
        "improvement.\n\n"
        "Return JSON only in this shape:\n"
        '{"entries":[{"title":"Short public title","summary":"One sentence summary",'
        '"bullets":["Specific readable note"],"commit_shas":["full_sha"]}]}\n\n'
        "Commits:\n"
        f"{json.dumps(payload, indent=2)}"
    )


def validate_changelog_entries(raw_entries: list[dict], commits: list[dict]) -> list[dict]:
    """Validate and normalize LLM changelog entries."""
    known_shas = {c["sha"] for c in commits}
    entries = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        bullets = entry.get("bullets") or []
        commit_shas = entry.get("commit_shas") or []
        if not title or not summary:
            continue
        if not isinstance(bullets, list) or not isinstance(commit_shas, list):
            continue
        normalized_shas = [str(s) for s in commit_shas if str(s) in known_shas]
        if not normalized_shas:
            continue
        entries.append({
            "title": title[:120],
            "summary": summary[:500],
            "bullets": [str(b).strip()[:300] for b in bullets if str(b).strip()][:5],
            "commit_shas": normalized_shas,
        })
    return entries


def summarize_changelog_commits(commits: list[dict],
                                model: str = _CHANGELOG_MODEL) -> list[dict]:
    """Ask Claude to turn raw commits into public changelog entries."""
    if not commits:
        return []
    try:
        message = _call_with_retry(
            get_claude_client(),
            model=model,
            max_tokens=2000,
            temperature=0,
            system="Return strict JSON only. Do not wrap the response in markdown.",
            messages=[{"role": "user", "content": _changelog_prompt(commits)}],
            stage_name="dashboard_publish",
            purpose="changelog_summary",
        )
        text = _extract_text_blocks(message)
        payload = json.loads(text)
        return validate_changelog_entries(payload.get("entries") or [], commits)
    except Exception:
        logger.warning("Failed to summarize changelog commits", exc_info=True)
        return []


def store_changelog_commits(cur, commits: list[dict]) -> None:
    """Persist changelog commits idempotently."""
    for commit in commits:
        cur.execute(
            """
            INSERT INTO dashboard_changelog_commits
                (sha, short_sha, committed_at, subject, body, files, published_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (sha) DO NOTHING
            """,
            (
                commit["sha"],
                commit["short_sha"],
                commit["committed_at"],
                commit["subject"],
                commit.get("body"),
                json.dumps(commit.get("files") or []),
            ),
        )


def store_changelog_entries(cur, entries: list[dict], *,
                            from_sha: str | None, to_sha: str,
                            model: str | None) -> None:
    """Persist LLM-written changelog entries."""
    for entry in entries:
        cur.execute(
            """
            INSERT INTO dashboard_changelog_entries
                (from_sha, to_sha, title, summary, bullets, commit_shas,
                 model, prompt_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                from_sha,
                to_sha,
                entry["title"],
                entry["summary"],
                json.dumps(entry.get("bullets") or []),
                json.dumps(entry.get("commit_shas") or []),
                model,
                _CHANGELOG_PROMPT_VERSION,
            ),
        )


def get_recent_changelog_entries(cur, limit: int = 60) -> list[dict]:
    """Read recent stored LLM entries, falling back to raw commits."""
    cur.execute(
        """
        SELECT id, created_at, title, summary, bullets, commit_shas
        FROM dashboard_changelog_entries
        ORDER BY created_at DESC, id DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    if rows:
        entries = []
        for row in rows:
            d = dict(row)
            entries.append({
                "date": _commit_date(d.get("created_at")),
                "title": d.get("title"),
                "summary": d.get("summary"),
                "bullets": d.get("bullets") or [],
                "commit_shas": d.get("commit_shas") or [],
            })
        return entries

    cur.execute(
        """
        SELECT sha, short_sha, committed_at, subject
        FROM dashboard_changelog_commits
        ORDER BY committed_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return group_changelog_commits([dict(r) for r in cur.fetchall()])


def _detect_how_it_works_state(deploy_dir: str) -> dict:
    """Return readiness flags for /about/, /internals/, /trace/ at publish time."""
    return {
        "about": os.path.exists(os.path.join(deploy_dir, "about", "index.html")),
        "internals": os.path.exists(os.path.join(deploy_dir, "internals", "index.html")),
        "trace": os.path.exists(os.path.join(deploy_dir, "trace", "index.html")),
    }


def _select_today_move(decisions: list[dict],
                       portfolio_value) -> dict | None:
    """Most-recent significant non-hold decision (notional ≥ $100)."""
    if not decisions:
        return None
    pv = float(portfolio_value or 0)
    for d in decisions:
        action = (d.get("action") or "").lower()
        if action == "hold":
            continue
        qty = float(d.get("quantity") or 0)
        price = float(d.get("price") or 0)
        notional = qty * price
        if notional < 100:
            continue
        return {
            "id": d["id"],
            "ticker": d.get("ticker"),
            "action": action,
            "notional": notional,
            "pct_of_portfolio": (notional / pv * 100.0) if pv > 0 else 0.0,
            "reasoning": d.get("reasoning"),
        }
    return None


def _select_attribution_top(attribution: list[dict], min_samples: int = 5) -> dict | None:
    if not attribution:
        return None
    eligible = [a for a in attribution
                if (a.get("sample_size") or 0) >= min_samples]
    return eligible[0] if eligible else None


def _select_worst_loser(mistakes: dict) -> dict | None:
    losers = (mistakes or {}).get("closed_losers") or []
    return losers[0] if losers else None


def _select_latest_memo(memos: list[dict]) -> dict | None:
    return memos[0] if memos else None


def emit_homepage(data: dict, deploy_dir: str, base_url: str) -> None:
    """Render and write index.html — replaces the old static index.html copy."""
    summary = data.get("summary") or {}
    snapshots = data.get("snapshots") or []
    sparkline_input = [
        {"value": s.get("twr_value", s.get("portfolio_value", s.get("value")))}
        for s in snapshots
    ]
    sparkline = render_sparkline_svg(sparkline_input)
    today_move = _select_today_move(
        data.get("decisions") or [],
        portfolio_value=summary.get("portfolio_value"),
    )
    attribution_top = _select_attribution_top(data.get("attribution") or [])
    worst_loser = _select_worst_loser(data.get("mistakes") or {})
    memo = _select_latest_memo(data.get("memos") or [])
    how_it_works = _detect_how_it_works_state(deploy_dir)
    theses = [t for t in (data.get("theses") or [])
              if (t.get("status") or "active") == "active"]

    html = render_homepage(
        summary=summary,
        theses=theses,
        sparkline_svg=sparkline,
        today_move=today_move,
        attribution_top=attribution_top,
        worst_loser=worst_loser,
        memo=memo,
        how_it_works_state=how_it_works,
        base_url=base_url,
    )
    with open(os.path.join(deploy_dir, "index.html"), "w") as f:
        f.write(html)


def emit_new_pages(data: dict, deploy_dir: str, base_url: str) -> None:
    """Render performance, activity, learning, how-it-works pages."""
    summary = data.get("summary") or {}
    performance = data.get("performance") or {}
    memos = data.get("memos") or []
    attribution = data.get("attribution") or []
    mistakes = data.get("mistakes") or {}

    pages = (
        ("performance", render_performance_page(
            summary=summary, performance=performance, base_url=base_url)),
        ("activity", render_activity_page(
            base_url=base_url, memos=memos)),
        ("changelog", render_changelog_page(
            entries=data.get("changelog") or [], base_url=base_url)),
        ("learning", render_learning_hub(
            attribution_top3=attribution[:3],
            losers_top3=(mistakes.get("closed_losers") or [])[:3],
            retired_rules_count=len(mistakes.get("retired_rules") or []),
            base_url=base_url)),
        ("how-it-works", render_how_it_works_hub(
            child_state=_detect_how_it_works_state(deploy_dir),
            base_url=base_url)),
    )
    for slug, html in pages:
        page_dir = os.path.join(deploy_dir, slug)
        os.makedirs(page_dir, exist_ok=True)
        with open(os.path.join(page_dir, "index.html"), "w") as f:
            f.write(html)


def assemble_deploy_dir(data: dict, deploy_dir: str, assets_dir: str,
                        base_url: str = "") -> str:
    """Assemble the full deploy directory: static assets, JSON, detail pages, OG images.

    `data` must include a `_pages` key with `decision_ids` and `thesis_ids`
    (added by the extended `gather_dashboard_data` flow). When `_pages` is
    missing, only the static + JSON path runs (legacy behavior).
    """
    os.makedirs(deploy_dir, exist_ok=True)

    # Static assets
    for filename in _STATIC_ASSETS:
        src = os.path.join(assets_dir, filename)
        dst = os.path.join(deploy_dir, filename)
        shutil.copy2(src, dst)

    write_json_files(data, deploy_dir)

    emit_home_og_image(data.get("summary", {}), deploy_dir)
    emit_static_pages(data, deploy_dir, base_url=base_url)

    # Per-trade / per-thesis pages + OG images
    pages = data.get("_pages")
    if pages and base_url:
        with get_cursor() as cur:
            page_stats = emit_detail_pages(
                cur,
                decision_ids=pages.get("decision_ids", []),
                thesis_ids=pages.get("thesis_ids", []),
                deploy_dir=deploy_dir,
                base_url=base_url,
            )
            og_stats = emit_og_images(
                cur,
                decision_ids=pages.get("decision_ids", []),
                thesis_ids=pages.get("thesis_ids", []),
                deploy_dir=deploy_dir,
            )
        logger.info("Detail pages: %s; OG images: %s", page_stats, og_stats)

    # Render homepage + the 4 new pages last so they can detect which
    # /about/, /internals/, /trace/ children have already been written.
    if base_url:
        emit_homepage(data, deploy_dir, base_url=base_url)
        emit_new_pages(data, deploy_dir, base_url=base_url)

    return deploy_dir


def deploy_to_cloudflare(deploy_dir: str) -> bool:
    """Deploy dashboard directory to Cloudflare Pages via wrangler.

    Requires CLOUDFLARE_PAGES_PROJECT env var.
    Wrangler authenticates via CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN env vars.

    Returns True if deployed successfully.
    Raises RuntimeError on failure.
    """
    project = os.environ.get("CLOUDFLARE_PAGES_PROJECT")
    if not project:
        raise RuntimeError("CLOUDFLARE_PAGES_PROJECT not set")

    try:
        result = subprocess.run(
            ["wrangler", "pages", "deploy", deploy_dir,
             "--project-name", project, "--branch", "main"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Wrangler deploy timed out after {e.timeout}s — bailing rather "
            "than blocking the session"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"Wrangler deploy failed: {result.stderr.strip()}")

    logger.info("Deployed to Cloudflare Pages: %s", result.stdout.strip())
    return True


@dataclass
class DashboardStageResult:
    """Result of the dashboard publishing stage."""
    published: bool = False
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


def run_dashboard_stage(session_date: date | None = None) -> DashboardStageResult:
    """Run the full dashboard publish pipeline: gather -> assemble -> deploy."""
    if session_date is None:
        session_date = date.today()

    result = DashboardStageResult()

    project = os.environ.get("CLOUDFLARE_PAGES_PROJECT")
    if not project:
        result.skipped = True
        logger.info("Dashboard stage skipped — CLOUDFLARE_PAGES_PROJECT not set")
        return result

    # Fetch net deposits from Alpaca for accurate return calculation
    net_deposits = None
    try:
        net_deposits = get_net_deposits()
    except Exception as e:
        logger.warning("Could not fetch net deposits from Alpaca: %s", e)

    # Gather data
    try:
        data = gather_dashboard_data(session_date, net_deposits=net_deposits)
    except Exception as e:
        result.errors.append(f"Data gathering failed: {e}")
        logger.error("Failed to gather dashboard data: %s", e)
        return result

    publish_sha = get_current_git_sha()
    data["changelog"] = []
    if publish_sha:
        try:
            with get_cursor() as cur:
                last_published_sha = get_changelog_pointer(cur)
                commits = fetch_changelog_commits(
                    from_sha=last_published_sha,
                    to_sha=publish_sha,
                )
                store_changelog_commits(cur, commits)
                entries = summarize_changelog_commits(commits)
                if entries:
                    store_changelog_entries(
                        cur,
                        entries,
                        from_sha=last_published_sha,
                        to_sha=publish_sha,
                        model=_CHANGELOG_MODEL,
                    )
                data["changelog"] = get_recent_changelog_entries(cur)
        except Exception as e:
            logger.warning("Could not prepare changelog entries: %s", e)

    # Assemble deploy directory
    base_url = os.environ.get("DASHBOARD_URL", "").rstrip("/")
    deploy_dir = tempfile.mkdtemp(prefix="dashboard_deploy_")
    try:
        assemble_deploy_dir(data, deploy_dir, _ASSETS_DIR, base_url=base_url)
    except Exception as e:
        result.errors.append(f"Deploy assembly failed: {e}")
        logger.error("Failed to assemble deploy directory: %s", e)
        return result

    # Deploy to Cloudflare
    try:
        deploy_to_cloudflare(deploy_dir)
    except Exception as e:
        result.errors.append(f"Cloudflare deploy failed: {e}")
        logger.error("Failed to deploy to Cloudflare: %s", e)
        return result
    finally:
        shutil.rmtree(deploy_dir, ignore_errors=True)

    result.published = True
    if publish_sha:
        try:
            persist_changelog_pointer(publish_sha)
        except Exception as e:
            result.errors.append(f"Changelog pointer update failed: {e}")
            logger.error("Failed to update changelog pointer: %s", e)
    logger.info("Dashboard publish complete (published=%s)", result.published)
    return result
