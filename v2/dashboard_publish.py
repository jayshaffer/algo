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

from .dashboard_og import (
    render_attribution_og,
    render_home_og,
    render_mistakes_og,
    render_thesis_og,
    render_trade_og,
)
from .dashboard_pages import (
    render_attribution_page,
    render_homepage_meta,
    render_mistakes_page,
    render_thesis_page,
    render_trade_page,
)
from .database.connection import get_cursor
from .database.trading_db import (
    get_closed_losers,
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

    return {
        "summary": summary,
        "snapshots": snapshot_dicts,
        "positions": [dict(r) for r in positions],
        "decisions": [
            {**dict(r), "order_id": _redact_order_id(dict(r).get("order_id"))}
            for r in decisions
        ],
        "theses": [dict(r) for r in theses],
        "benchmark": benchmark,
        "mistakes": mistakes,
        "attribution": attribution,
        "_pages": pages,  # NEW
    }


def gather_trade_detail(cur, decision_id: int) -> dict | None:
    """Return full detail for one decision page: decision + thesis + position.

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

    return {"decision": decision, "thesis": thesis, "position": position}


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

    return {"thesis": thesis, "decisions": decisions, "position": position}


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


def inject_homepage_og_meta(deploy_dir: str, summary: dict, base_url: str) -> None:
    """Replace the <!-- OG_META --> placeholder in deploy_dir/index.html."""
    index_path = os.path.join(deploy_dir, "index.html")
    with open(index_path) as f:
        html = f.read()
    if "<!-- OG_META -->" not in html:
        return
    block = render_homepage_meta(summary, base_url=base_url)
    html = html.replace("<!-- OG_META -->", block)
    with open(index_path, "w") as f:
        f.write(html)


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
        "theses", "benchmark", "mistakes", "attribution",
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
_STATIC_ASSETS = ("index.html", "styles.css", "app.js")


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

    # Inject homepage OG meta (no-op if placeholder absent)
    try:
        inject_homepage_og_meta(deploy_dir, data.get("summary", {}), base_url=base_url)
    except Exception:
        logger.warning("Failed to inject homepage OG meta", exc_info=True)

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
    logger.info("Dashboard publish complete (published=%s)", result.published)
    return result
