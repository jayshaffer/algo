"""HTML page rendering for the public dashboard.

Pure-ish functions that turn data dicts into HTML strings. No DB access; the
caller (v2/dashboard_publish.py) gathers data and passes it in.
"""

from decimal import Decimal
from html import escape as _esc
from string import Template

_META_BLOCK_TEMPLATE = Template(
    '<meta property="og:title" content="$title" />\n'
    '<meta property="og:description" content="$description" />\n'
    '<meta property="og:image" content="$og_image" />\n'
    '<meta property="og:url" content="$page_url" />\n'
    '<meta property="og:type" content="$og_type" />\n'
    '<meta name="twitter:card" content="summary_large_image" />\n'
    '<meta name="twitter:title" content="$title" />\n'
    '<meta name="twitter:description" content="$description" />\n'
    '<meta name="twitter:image" content="$og_image" />\n'
)


def _render_meta_block(*, title: str, description: str, og_image: str,
                       page_url: str, og_type: str = "website") -> str:
    """Return the <meta> tags shared by every emitted page.

    Both inputs must already be HTML-safe (caller escapes).
    """
    return _META_BLOCK_TEMPLATE.substitute(
        title=title,
        description=description,
        og_image=og_image,
        page_url=page_url,
        og_type=og_type,
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
    base = base_url.rstrip("/")
    return _render_meta_block(
        title=title,
        description=description,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/",
        og_type="website",
    )


_TRADE_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>$title — Bikini Bottom Capital</title>
$meta_block
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>$action_caps $ticker</h2>
<p class="trade-summary">$qty_display shares at $price_display on $trade_date</p>
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

    # Safely coerce IDs to int
    decision_id = int(decision["id"])

    # Raw (unescaped) values for composition
    raw_ticker = str(decision["ticker"])
    raw_qty = decision.get("quantity") or 0
    raw_price = decision.get("price") or 0
    action_upper = str(decision.get("action", "")).lower().upper()

    # Escaped values for direct HTML/attribute output
    ticker_esc = _esc(raw_ticker)
    action_caps = _esc(action_upper)

    # title uses escaped values
    title = f"{action_caps} {ticker_esc}"

    # og:description — build from raw, escape once
    description_raw = f"{action_upper} {raw_qty} {raw_ticker} @ {_fmt_money(raw_price)}"
    description = _esc(description_raw)

    # Display values for body
    qty_display = _esc(str(raw_qty))
    price_display = _fmt_money(raw_price)  # e.g. "$450.25"

    # trade_date — isoformat() output is always safe ASCII; escape fallback path
    trade_date = (
        decision["date"].isoformat()
        if hasattr(decision["date"], "isoformat")
        else _esc(str(decision["date"]))
    )

    if thesis:
        tid = int(thesis["id"])
        raw_thesis_text = str(thesis.get("thesis", ""))
        thesis_text = _esc(raw_thesis_text) if raw_thesis_text else f"Thesis #{tid}"
        thesis_section = _THESIS_LINK_TEMPLATE.substitute(
            tid=tid,
            thesis_text=thesis_text,
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

    meta_block = _render_meta_block(
        title=title,
        description=description,
        og_image=f"{base}/og/trade/{decision_id}.png",
        page_url=f"{base}/trade/{decision_id}/",
        og_type="article",
    )

    return _TRADE_PAGE_TEMPLATE.substitute(
        title=title,
        action_caps=action_caps,
        ticker=ticker_esc,
        qty_display=qty_display,
        price_display=price_display,
        trade_date=trade_date,
        reasoning=_esc(str(decision.get("reasoning") or "")),
        thesis_section=thesis_section,
        outcome_section=outcome_section,
        meta_block=meta_block,
    )
