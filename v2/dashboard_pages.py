"""HTML page rendering for the public dashboard.

Pure-ish functions that turn data dicts into HTML strings. No DB access; the
caller (v2/dashboard_publish.py) gathers data and passes it in.
"""

from decimal import Decimal
from html import escape as _esc
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


def _fmt_money(value: Decimal | int | float | None) -> str:
    if value is None:
        return "$0.00"
    return f"${Decimal(value):,.2f}"


def _fmt_pct(value: Decimal | int | float | None) -> str:
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


_TRADE_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>$title — Bikini Bottom Capital</title>
<meta property="og:title" content="$title" />
<meta property="og:description" content="$description" />
<meta property="og:image" content="$og_image" />
<meta property="og:url" content="$page_url" />
<meta property="og:type" content="article" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="$title" />
<meta name="twitter:description" content="$description" />
<meta name="twitter:image" content="$og_image" />
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>$action_caps $ticker</h2>
<p class="trade-summary">$qty shares at $$price on $trade_date</p>
<h3>Reasoning</h3>
<p>$reasoning</p>
$thesis_section
$outcome_section
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")

_THESIS_LINK_TEMPLATE = Template(
    '<h3>Thesis</h3>'
    '<p><a href="/thesis/$tid/">$thesis_text</a> '
    '<span class="thesis-meta">($direction, $confidence confidence)</span></p>'
)

_OUTCOME_TEMPLATE = Template(
    '<h3>Outcome</h3><p>7-day: $o7 · 30-day: $o30</p>'
)


def _fmt_outcome(value: Decimal | int | float | None) -> str:
    if value is None:
        return "pending"
    return f"{Decimal(value):+.2f}%"


def render_trade_page(decision: dict, thesis: dict | None,
                      position: dict | None, base_url: str) -> str:
    """Return the full HTML page for one trade."""
    base = base_url.rstrip("/")
    ticker = _esc(str(decision["ticker"]))
    action = str(decision["action"]).lower()
    title = f"{action.upper()} {ticker}"
    qty = decision.get("quantity") or 0
    price = decision.get("price") or 0
    description = f"{action.upper()} {qty} {ticker} @ ${price}"

    if thesis:
        thesis_section = _THESIS_LINK_TEMPLATE.substitute(
            tid=thesis["id"],
            thesis_text=_esc(str(thesis.get("thesis", ""))),
            direction=_esc(str(thesis.get("direction", ""))),
            confidence=_esc(str(thesis.get("confidence", ""))),
        )
    else:
        thesis_section = ""

    if decision.get("outcome_7d") is not None or decision.get("outcome_30d") is not None:
        outcome_section = _OUTCOME_TEMPLATE.substitute(
            o7=_fmt_outcome(decision.get("outcome_7d")),
            o30=_fmt_outcome(decision.get("outcome_30d")),
        )
    else:
        outcome_section = ""

    return _TRADE_PAGE_TEMPLATE.substitute(
        title=title,
        action_caps=action.upper(),
        ticker=ticker,
        qty=qty,
        price=price,
        trade_date=decision["date"].isoformat() if hasattr(decision["date"], "isoformat") else str(decision["date"]),
        reasoning=_esc(str(decision.get("reasoning") or "")),
        thesis_section=thesis_section,
        outcome_section=outcome_section,
        description=_esc(description),
        og_image=f"{base}/og/trade/{decision['id']}.png",
        page_url=f"{base}/trade/{decision['id']}/",
    )
