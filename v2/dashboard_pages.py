"""HTML page rendering for the public dashboard.

Pure-ish functions that turn data dicts into HTML strings. No DB access; the
caller (v2/dashboard_publish.py) gathers data and passes it in.
"""

from decimal import Decimal
from string import Template

_HOMEPAGE_META_TEMPLATE = Template(
    '<meta property="og:title" content="$title" />\n'
    '<meta property="og:description" content="$description" />\n'
    '<meta property="og:image" content="$image_url" />\n'
    '<meta property="og:url" content="$page_url" />\n'
    '<meta name="twitter:card" content="summary_large_image" />\n'
    '<meta name="twitter:title" content="$title" />\n'
    '<meta name="twitter:description" content="$description" />\n'
    '<meta name="twitter:image" content="$image_url" />\n'
)


def _fmt_money(value) -> str:
    if value is None:
        return "$0.00"
    return f"${Decimal(value):,.2f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "0.00%"
    return f"{Decimal(value):+.2f}%"


def render_homepage_meta(summary: dict, base_url: str) -> str:
    """Return the OG/Twitter card meta block for the homepage."""
    title = "Bikini Bottom Capital"
    daily_pnl = summary.get("daily_pnl") or 0
    daily_pct = summary.get("daily_pnl_pct") or 0
    portfolio = summary.get("portfolio_value") or 0
    description = (
        f"Portfolio: {_fmt_money(portfolio)} · "
        f"Today: {_fmt_money(daily_pnl)} ({_fmt_pct(daily_pct)})"
    )
    return _HOMEPAGE_META_TEMPLATE.substitute(
        title=title,
        description=description,
        image_url=f"{base_url.rstrip('/')}/og/home.png",
        page_url=base_url.rstrip("/") + "/",
    )
