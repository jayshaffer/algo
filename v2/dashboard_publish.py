"""Dashboard data gathering for public GitHub Pages dashboard.

Queries the DB and structures data for JSON export.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from .database.connection import get_cursor
from .executor import get_net_deposits

logger = logging.getLogger("dashboard_publish")


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal, date, and datetime types."""

    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def gather_dashboard_data(session_date: date, net_deposits: Optional[Decimal] = None) -> dict:
    """Gather all dashboard data in a single DB connection.

    Args:
        session_date: The trading session date.
        net_deposits: Total net cash deposited (from Alpaca activities).
            Used for accurate total return calculation excluding cash infusions.

    Returns dict with keys: summary, snapshots, positions, decisions, theses.
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

    # Build summary
    summary = _build_summary(latest, first, previous, len(positions), session_date, net_deposits)

    return {
        "summary": summary,
        "snapshots": [dict(r) for r in snapshots],
        "positions": [dict(r) for r in positions],
        "decisions": [dict(r) for r in decisions],
        "theses": [dict(r) for r in theses],
    }


def _build_summary(latest, first, previous, positions_count, session_date, net_deposits=None):
    """Build summary dict from query results."""
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

    # Daily P&L
    daily_pnl = Decimal("0")
    daily_pnl_pct = Decimal("0")
    if previous and previous["portfolio_value"]:
        prev_value = previous["portfolio_value"]
        daily_pnl = portfolio_value - prev_value
        if prev_value != 0:
            daily_pnl_pct = (daily_pnl / prev_value) * 100

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


def write_json_files(data: dict, repo_path: str) -> list[str]:
    """Write dashboard data as separate JSON files for GitHub Pages.

    Creates a data/ subdirectory under repo_path and writes each key
    from the data dict as a separate JSON file.

    Returns list of file paths written.
    """
    data_dir = os.path.join(repo_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    files_written = []
    for key in ("summary", "snapshots", "positions", "decisions", "theses"):
        file_path = os.path.join(data_dir, f"{key}.json")
        with open(file_path, "w") as f:
            json.dump(data[key], f, cls=_DecimalEncoder, indent=2)
        logger.info("Wrote %s", file_path)
        files_written.append(file_path)

    return files_written


def push_to_github(repo_path: str) -> bool:
    """Stage, commit, and push dashboard data to GitHub.

    Returns True if pushed successfully, False if nothing to commit.
    Raises RuntimeError if git push fails.
    """
    add_result = subprocess.run(
        ["git", "add", "data/"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        raise RuntimeError(f"git add failed: {add_result.stderr.strip()}")

    commit_result = subprocess.run(
        ["git", "commit", "-m", f"Update dashboard data {date.today().isoformat()}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if commit_result.returncode != 0:
        logger.info("Nothing to commit: %s", commit_result.stdout.strip())
        return False

    push_result = subprocess.run(
        ["git", "push"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if push_result.returncode != 0:
        raise RuntimeError(push_result.stderr)

    return True


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

    result = subprocess.run(
        ["wrangler", "pages", "deploy", deploy_dir,
         "--project-name", project, "--branch", "main"],
        capture_output=True,
        text=True,
    )
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


def run_dashboard_stage(session_date: Optional[date] = None) -> DashboardStageResult:
    """Run the full dashboard publish pipeline: gather -> write -> push."""
    if session_date is None:
        session_date = date.today()

    result = DashboardStageResult()

    repo_path = os.environ.get("DASHBOARD_REPO_PATH")
    if not repo_path:
        result.skipped = True
        logger.info("Dashboard stage skipped — DASHBOARD_REPO_PATH not set")
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

    # Write JSON files
    try:
        write_json_files(data, repo_path)
    except Exception as e:
        result.errors.append(f"JSON writing failed: {e}")
        logger.error("Failed to write JSON files: %s", e)
        return result

    # Push to GitHub
    try:
        pushed = push_to_github(repo_path)
    except Exception as e:
        result.errors.append(f"Git push failed: {e}")
        logger.error("Failed to push to GitHub: %s", e)
        return result

    result.published = pushed
    logger.info("Dashboard publish complete (published=%s)", pushed)
    return result
