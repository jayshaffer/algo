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


_THESIS_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
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
<h2>$ticker — $direction thesis</h2>
<p class="thesis-meta">Confidence: $confidence · Status: $status</p>
<h3>Thesis</h3>
<p>$thesis_text</p>
$triggers_section
$decisions_section
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")


def _render_triggers_section(thesis: dict) -> str:
    parts = []
    if thesis.get("entry_trigger"):
        parts.append(f"<p><strong>Entry:</strong> {_esc(str(thesis['entry_trigger']))}</p>")
    if thesis.get("exit_trigger"):
        parts.append(f"<p><strong>Exit:</strong> {_esc(str(thesis['exit_trigger']))}</p>")
    if thesis.get("invalidation"):
        parts.append(f"<p><strong>Invalidation:</strong> {_esc(str(thesis['invalidation']))}</p>")
    if not parts:
        return ""
    return "<h3>Triggers</h3>" + "".join(parts)


def _render_decisions_section(decisions: list[dict]) -> str:
    if not decisions:
        return ""
    rows = []
    for d in decisions:
        did = int(d["id"])
        qty = d.get("quantity") or 0
        price = _fmt_money(d.get("price") or 0)
        action_upper = _esc(str(d.get("action", "")).upper())
        trade_date = (
            d["date"].isoformat()
            if hasattr(d["date"], "isoformat")
            else _esc(str(d["date"]))
        )
        rows.append(
            f'<li><a href="/trade/{did}/">{trade_date} '
            f'{action_upper} {_esc(str(qty))} @ {price}</a></li>'
        )
    return "<h3>Related decisions</h3><ul>" + "".join(rows) + "</ul>"


def render_thesis_page(thesis: dict, decisions: list[dict],
                       position: dict | None, base_url: str) -> str:
    """Return the full HTML page for one thesis."""
    base = base_url.rstrip("/")

    # Coerce ID to int for URL construction
    thesis_id = int(thesis["id"])

    # Raw values for composition
    raw_ticker = str(thesis["ticker"])
    raw_direction = str(thesis.get("direction", ""))

    # Escaped values for direct HTML output
    ticker_esc = _esc(raw_ticker)
    direction_esc = _esc(raw_direction)
    confidence_esc = _esc(str(thesis.get("confidence", "")))
    status_esc = _esc(str(thesis.get("status", "")))
    thesis_text_esc = _esc(str(thesis.get("thesis", "")))

    # title built from escaped pieces
    title = f"{ticker_esc} — {direction_esc} thesis"

    # og:description — build from raw, escape once
    description_raw = str(thesis.get("thesis", ""))[:160].replace("\n", " ").rstrip()
    description = _esc(description_raw)

    meta_block = _render_meta_block(
        title=title,
        description=description,
        og_image=f"{base}/og/thesis/{thesis_id}.png",
        page_url=f"{base}/thesis/{thesis_id}/",
        og_type="article",
    )

    return _THESIS_PAGE_TEMPLATE.substitute(
        title=title,
        ticker=ticker_esc,
        direction=direction_esc,
        confidence=confidence_esc,
        status=status_esc,
        thesis_text=thesis_text_esc,
        triggers_section=_render_triggers_section(thesis),
        decisions_section=_render_decisions_section(decisions),
        meta_block=meta_block,
    )


_MISTAKES_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>What didn't work — Bikini Bottom Capital</title>
$meta_block
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>What didn't work</h2>
<p class="subtitle">Closed losers (last 30 days) and retired rules (last 90 days). No spin.</p>
$losers_section
$rules_section
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")


def _render_loser_row(d: dict) -> str:
    did = int(d["id"])
    ticker = _esc(str(d["ticker"]))
    action_caps = _esc(str(d.get("action", "")).upper())
    qty = _esc(str(d.get("quantity") or 0))
    price = _fmt_money(d.get("price") or 0)
    o30 = _fmt_outcome(d.get("outcome_30d"))
    trade_date = (
        d["date"].isoformat()
        if hasattr(d["date"], "isoformat")
        else _esc(str(d["date"]))
    )
    reasoning = _esc(str(d.get("reasoning") or ""))
    return (
        f'<li class="loser-row">'
        f'<a href="/trade/{did}/"><strong>{action_caps} {ticker}</strong></a>'
        f' — {trade_date} · {qty} @ {price} · '
        f'<span class="loser-outcome">{o30}</span>'
        f'<p class="loser-reason">{reasoning}</p>'
        f'</li>'
    )


def _render_rule_row(r: dict) -> str:
    text = _esc(str(r.get("rule_text") or ""))
    reason = _esc(str(r.get("retirement_reason") or ""))
    retired_at = r.get("retired_at")
    if hasattr(retired_at, "isoformat"):
        retired_at = retired_at.isoformat()
    retired_at_esc = _esc(str(retired_at or ""))
    return (
        f'<li class="rule-row">'
        f'<p>{text}</p>'
        f'<p class="rule-meta">retired {retired_at_esc} — {reason}</p>'
        f'</li>'
    )


def render_mistakes_page(closed_losers: list[dict], retired_rules: list[dict],
                         base_url: str) -> str:
    """Return the full HTML for /mistakes/index.html."""
    base = base_url.rstrip("/")

    if closed_losers:
        rows = "".join(_render_loser_row(d) for d in closed_losers)
        losers_section = (
            "<h3>Closed losers</h3>"
            f'<ul class="loser-list">{rows}</ul>'
        )
    else:
        losers_section = (
            '<h3>Closed losers</h3>'
            '<p class="empty-state">No closed losers in window. '
            'Either we got lucky or we didn\'t trade enough.</p>'
        )

    if retired_rules:
        rows = "".join(_render_rule_row(r) for r in retired_rules)
        rules_section = (
            "<h3>Retired rules</h3>"
            f'<ul class="rule-list">{rows}</ul>'
        )
    else:
        rules_section = ""

    meta_block = _render_meta_block(
        title="What didn't work — Bikini Bottom Capital",
        description="Closed losers and retired rules. The receipts most accounts hide.",
        og_image=f"{base}/og/mistakes.png",
        page_url=f"{base}/mistakes/",
        og_type="article",
    )

    return _MISTAKES_PAGE_TEMPLATE.substitute(
        meta_block=meta_block,
        losers_section=losers_section,
        rules_section=rules_section,
    )


_ATTRIBUTION_PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>What's actually working — Bikini Bottom Capital</title>
$meta_block
<link rel="stylesheet" href="/styles.css" />
</head>
<body>
<header><div class="container"><h1><a href="/">&#9875; Bikini Bottom Capital</a></h1></div></header>
<main class="container">
<section class="panel">
<h2>What's actually working</h2>
<p class="subtitle">Signal-attribution scores from the last 90 days of decisions.</p>
$body
</section>
</main>
<footer><div class="container"><p><a href="/">Back to dashboard</a></p></div></footer>
</body>
</html>
""")


def _render_attribution_table(attribution: list[dict]) -> str:
    rows: list[str] = []
    for r in attribution:
        category = _esc(str(r.get("category") or ""))
        sample_7d = _esc(str(r.get("sample_size") or 0))
        sample_30d = _esc(str(r.get("sample_size_30d") or 0))
        out_7d = _fmt_outcome(r.get("avg_outcome_7d"))
        out_30d = _fmt_outcome(r.get("avg_outcome_30d"))
        rows.append(
            "<tr>"
            f"<td>{category}</td>"
            f'<td class="num">{sample_7d}</td>'
            f'<td class="num">{sample_30d}</td>'
            f'<td class="num">{out_7d}</td>'
            f'<td class="num">{out_30d}</td>'
            "</tr>"
        )
    body = "".join(rows)
    return (
        '<table class="attribution-table">'
        "<thead><tr>"
        "<th>Signal type</th>"
        '<th class="num">N (7d)</th>'
        '<th class="num">N (30d)</th>'
        '<th class="num">Avg 7d</th>'
        '<th class="num">Avg 30d</th>'
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


def render_attribution_page(attribution: list[dict], base_url: str) -> str:
    """Return the full HTML for /attribution/index.html."""
    base = base_url.rstrip("/")

    if attribution:
        body = _render_attribution_table(attribution)
    else:
        body = (
            '<p class="empty-state">'
            "Not enough samples yet. Attribution scores require at least "
            "5 closed decisions per signal type."
            "</p>"
        )

    meta_block = _render_meta_block(
        title="What's actually working — Bikini Bottom Capital",
        description="Signal-attribution scores. Which inputs predicted, which were noise.",
        og_image=f"{base}/og/attribution.png",
        page_url=f"{base}/attribution/",
        og_type="article",
    )

    return _ATTRIBUTION_PAGE_TEMPLATE.substitute(
        meta_block=meta_block,
        body=body,
    )
