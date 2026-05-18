"""HTML page rendering for the public dashboard.

Pure-ish functions that turn data dicts into HTML strings. No DB access; the
caller (v2/dashboard_publish.py) gathers data and passes it in.
"""

from decimal import Decimal
from html import escape as _esc
from string import Template
from urllib.parse import quote

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


_NAV_ITEMS = (
    ("home", "/", "Home"),
    ("strategy", "/strategy/", "Strategy"),
    ("performance", "/performance/", "Performance"),
    ("activity", "/activity/", "Activity"),
    ("learning", "/learning/", "Learning"),
    ("changelog", "/changelog/", "Changelog"),
    ("how-it-works", "/how-it-works/", "How it works"),
)


def _render_nav(active_nav: str) -> str:
    parts = ['<nav class="site-nav"><div class="container">']
    parts.append('<a class="logo" href="/">⌬ Pinchy</a>')
    parts.append('<button class="hamburger" aria-label="Menu">☰</button>')
    parts.append('<div class="links">')
    for key, href, label in _NAV_ITEMS:
        cls = ' class="active"' if key == active_nav else ''
        parts.append(f'<a{cls} href="{href}">{label}</a>')
    parts.append('</div></div></nav>')
    return "".join(parts)


_FOOTER_HTML = (
    '<footer><div class="container">'
    '<p>Is mayonnaise a financial instrument?</p>'
    '<p class="attribution">Data from '
    '<a href="https://alpaca.markets" target="_blank" rel="noopener">Alpaca</a></p>'
    '</div></footer>'
)


_PAGE_SHELL_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>$title — Pinchy</title>
$meta_block
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍍</text></svg>" />
<link rel="stylesheet" href="/styles.css" />
$head_extra
</head>
<body data-page="$active_nav">
$nav
<main class="container">
$breadcrumbs
$content
</main>
$footer
<script src="/app.js"></script>
</body>
</html>
""")


def _render_page_shell(*, title: str, description: str, active_nav: str,
                       content: str, og_image: str, page_url: str,
                       og_type: str = "website",
                       head_extra: str = "",
                       data_page: str | None = None,
                       breadcrumbs: list[tuple[str, str | None]] | None = None,
                       back_href: str | None = None) -> str:
    """Wrap page content in the shared <html> + nav + footer scaffolding.

    `data_page` overrides what's emitted as `<body data-page="…">`. Defaults
    to `active_nav`. Use it on permalink pages where the nav highlight (e.g.
    "learning") differs from the app.js dispatch key (e.g. "mistakes").
    """
    meta_block = _render_meta_block(
        title=_esc(title),
        description=_esc(description),
        og_image=og_image,
        page_url=page_url,
        og_type=og_type,
    )
    breadcrumbs_html = _render_breadcrumbs(breadcrumbs, back_href)
    return _PAGE_SHELL_TEMPLATE.substitute(
        title=_esc(title),
        meta_block=meta_block,
        head_extra=head_extra,
        active_nav=_esc(data_page or active_nav),
        nav=_render_nav(active_nav),
        breadcrumbs=breadcrumbs_html,
        content=content,
        footer=_FOOTER_HTML,
    )


def _render_breadcrumbs(
    breadcrumbs: list[tuple[str, str | None]] | None,
    back_href: str | None,
) -> str:
    if not breadcrumbs and not back_href:
        return ""

    current_title = ""
    if breadcrumbs:
        current_title = breadcrumbs[-1][0]

    parts = ['<div class="breadcrumbs-bar drilldown-header">']
    if back_href:
        parts.append(
            f'<button class="back-button" type="button" aria-label="Go back" '
            f'data-back-fallback="{_esc(back_href)}">←</button>'
        )
    if current_title:
        parts.append(f'<h1 class="drilldown-title">{_esc(current_title)}</h1>')
    parts.append("</div>")
    return "".join(parts)


def _fmt_money(value: Decimal | int | float | None) -> str:
    if value is None:
        return "$0.00"
    return f"${Decimal(value):,.2f}"


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def _ticker_href(ticker: str | None) -> str:
    return f"/ticker/{quote(str(ticker or '').upper(), safe='')}/"


def _fmt_signed_pct(value) -> str:
    if value is None:
        return "—"
    v = float(value)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _fmt_pct(value: Decimal | int | float | None) -> str:
    if value is None:
        return "0.00%"
    return f"{Decimal(value):+.2f}%"


def render_homepage_meta(summary: dict, base_url: str) -> str:
    """Return the OG/Twitter card meta block for the homepage."""
    title = "Pinchy"
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


def _fmt_signal_date(value) -> str:
    """Render a published_at timestamp as YYYY-MM-DD; empty string if missing."""
    if value is None:
        return ""
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return _esc(str(value))


def _render_signal_refs_section(refs: list[dict] | None) -> str:
    """Render the 'Cited evidence' block for trade and thesis pages.

    Refs are expected in the shape returned by `_fetch_signal_refs`:
      - news_signal:  {signal_id, ticker, headline, category, sentiment, published_at}
      - macro_signal: {signal_id, headline, category, sentiment, affected_sectors,
                       published_at}
      - thesis:       {signal_id, ticker, direction, thesis, confidence, status}

    Returns "" when there are no refs so callers can splice unconditionally.
    """
    if not refs:
        return ""

    news = [r for r in refs if r.get("signal_type") == "news_signal"]
    macro = [r for r in refs if r.get("signal_type") == "macro_signal"]
    theses = [r for r in refs if r.get("signal_type") == "thesis"]

    parts: list[str] = ['<h3>Cited evidence</h3>']

    def _meta_join(*bits) -> str:
        return " · ".join([b for b in bits if b])

    if news:
        items = []
        for r in news:
            ticker = _esc(str(r.get("ticker") or ""))
            headline = _esc(str(r.get("headline") or ""))
            meta = _meta_join(
                _esc(str(r.get("category") or "")),
                _esc(str(r.get("sentiment") or "")),
                _fmt_signal_date(r.get("published_at")),
            )
            meta_html = f' <span class="signal-meta">({meta})</span>' if meta else ""
            ticker_html = f"<strong>{ticker}</strong> " if ticker else ""
            items.append(f"<li>{ticker_html}{headline}{meta_html}</li>")
        parts.append('<h4>News</h4><ul class="signal-refs bounded-list">' + "".join(items) + "</ul>")

    if macro:
        items = []
        for r in macro:
            headline = _esc(str(r.get("headline") or ""))
            sectors = r.get("affected_sectors")
            sectors_str = ""
            if sectors:
                if isinstance(sectors, (list, tuple)):
                    sectors_str = ", ".join(_esc(str(s)) for s in sectors)
                else:
                    sectors_str = _esc(str(sectors))
            meta = _meta_join(
                _esc(str(r.get("category") or "")),
                _esc(str(r.get("sentiment") or "")),
                f"sectors: {sectors_str}" if sectors_str else "",
                _fmt_signal_date(r.get("published_at")),
            )
            meta_html = f' <span class="signal-meta">({meta})</span>' if meta else ""
            items.append(f"<li>{headline}{meta_html}</li>")
        parts.append('<h4>Macro</h4><ul class="signal-refs bounded-list">' + "".join(items) + "</ul>")

    if theses:
        items = []
        for r in theses:
            tid = int(r["signal_id"])
            ticker = _esc(str(r.get("ticker") or ""))
            direction = _esc(str(r.get("direction") or ""))
            thesis_text = _truncate(str(r.get("thesis") or ""), 120)
            label = _esc(thesis_text) if thesis_text else f"Thesis #{tid}"
            heading_bits = " ".join([b for b in (ticker, direction) if b])
            prefix = f"{heading_bits} — " if heading_bits else ""
            items.append(
                f'<li><a href="/thesis/{tid}/">{prefix}{label}</a></li>'
            )
        parts.append('<h4>Theses</h4><ul class="signal-refs bounded-list">' + "".join(items) + "</ul>")

    return "".join(parts)


def render_trade_page(decision: dict, thesis: dict | None,
                      position: dict | None, base_url: str,
                      signal_refs: list[dict] | None = None) -> str:
    """Return the full HTML page for one trade."""
    base = base_url.rstrip("/")
    decision_id = int(decision["id"])

    raw_ticker = str(decision["ticker"])
    raw_qty = decision.get("quantity") or 0
    raw_price = decision.get("price") or 0
    action_upper = str(decision.get("action", "")).lower().upper()

    ticker_esc = _esc(raw_ticker)
    action_caps = _esc(action_upper)
    qty_display = _esc(str(raw_qty))
    price_display = _fmt_money(raw_price)

    trade_date = (
        decision["date"].isoformat()
        if hasattr(decision["date"], "isoformat")
        else _esc(str(decision["date"]))
    )

    thesis_section = ""
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

    outcome_section = ""
    if decision.get("outcome_7d") is not None or decision.get("outcome_30d") is not None:
        outcome_section = _OUTCOME_TEMPLATE.substitute(
            o7=_fmt_outcome(decision.get("outcome_7d")),
            o30=_fmt_outcome(decision.get("outcome_30d")),
        )

    signal_refs_section = _render_signal_refs_section(signal_refs)

    title_raw = f"{action_upper} {raw_ticker}"
    description_raw = f"{action_upper} {raw_qty} {raw_ticker} @ {_fmt_money(raw_price)}"

    content = (
        f'<section class="section">'
        f'<h2>{action_caps} {ticker_esc}</h2>'
        f'<p class="trade-summary">{qty_display} shares at {price_display} on {trade_date}</p>'
        f'<h3>Reasoning</h3>'
        f'<p class="long-text">{_esc(str(decision.get("reasoning") or ""))}</p>'
        f'{thesis_section}{outcome_section}{signal_refs_section}'
        f'</section>'
    )

    return _render_page_shell(
        title=title_raw,
        description=description_raw,
        active_nav="activity",
        content=content,
        og_image=f"{base}/og/trade/{decision_id}.png",
        page_url=f"{base}/trade/{decision_id}/",
        og_type="article",
        breadcrumbs=[
            ("Home", "/"),
            ("Activity", "/activity/"),
            (raw_ticker, _ticker_href(raw_ticker)),
            (f"Trade {decision_id}", None),
        ],
        back_href=_ticker_href(raw_ticker),
    )


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
    return '<h3>Related decisions</h3><ul class="related-decisions bounded-list">' + "".join(rows) + "</ul>"


def render_thesis_page(thesis: dict, decisions: list[dict],
                       position: dict | None, base_url: str,
                       signal_refs: list[dict] | None = None) -> str:
    """Return the full HTML page for one thesis."""
    base = base_url.rstrip("/")
    thesis_id = int(thesis["id"])

    raw_ticker = str(thesis["ticker"])
    raw_direction = str(thesis.get("direction", ""))

    ticker_esc = _esc(raw_ticker)
    direction_esc = _esc(raw_direction)
    confidence_esc = _esc(str(thesis.get("confidence", "")))
    status_esc = _esc(str(thesis.get("status", "")))
    thesis_text_esc = _esc(str(thesis.get("thesis", "")))

    title_raw = f"{raw_ticker} — {raw_direction} thesis"
    description_raw = str(thesis.get("thesis", ""))[:160].replace("\n", " ").rstrip()

    content = (
        f'<section class="section">'
        f'<h2>{ticker_esc} — {direction_esc} thesis</h2>'
        f'<p class="thesis-meta">Confidence: {confidence_esc} · Status: {status_esc}</p>'
        f'<h3>Thesis</h3><p class="long-text">{thesis_text_esc}</p>'
        f'{_render_triggers_section(thesis)}'
        f'{_render_signal_refs_section(signal_refs)}'
        f'{_render_decisions_section(decisions)}'
        f'</section>'
    )

    return _render_page_shell(
        title=title_raw,
        description=description_raw,
        active_nav="strategy",
        content=content,
        og_image=f"{base}/og/thesis/{thesis_id}.png",
        page_url=f"{base}/thesis/{thesis_id}/",
        og_type="article",
        breadcrumbs=[
            ("Home", "/"),
            ("Strategy", "/strategy/"),
            (raw_ticker, _ticker_href(raw_ticker)),
            (f"Thesis {thesis_id}", None),
        ],
        back_href="/strategy/",
    )


def render_ticker_page(*, ticker: str, decisions: list[dict],
                       theses: list[dict], position: dict | None,
                       base_url: str) -> str:
    """Return an aggregate page for everything known about one ticker."""
    base = base_url.rstrip("/")
    ticker_esc = _esc(ticker)

    if position:
        pos_body = (
            '<div class="stat-row">'
            + _stat("Shares", _esc(str(position.get("shares") or 0)))
            + _stat("Avg cost", _fmt_money(position.get("avg_cost")))
            + '</div>'
        )
    else:
        pos_body = '<p class="empty-state">No open position.</p>'

    if theses:
        thesis_rows = []
        for t in theses[:20]:
            tid = int(t["id"])
            direction = _esc(str(t.get("direction") or ""))
            confidence = _esc(str(t.get("confidence") or ""))
            body = _esc(_truncate(str(t.get("thesis") or ""), 220))
            thesis_rows.append(
                f'<a class="card thesis-summary" href="/thesis/{tid}/">'
                f'<div class="lbl">{direction} · {confidence}</div>'
                f'<p>{body}</p></a>'
            )
        theses_body = f'<div class="card-grid strategy-theses-grid">{"".join(thesis_rows)}</div>'
    else:
        theses_body = '<p class="empty-state">No theses for this ticker.</p>'

    if decisions:
        rows = []
        for d in decisions[:50]:
            did = int(d["id"])
            action = (str(d.get("action") or "")).lower()
            badge_cls = (
                f"badge badge-{action}" if action in ("buy", "sell", "hold") else "badge"
            )
            qty = _esc(str(d.get("quantity") or 0))
            price = _fmt_money(d.get("price") or 0)
            decision_date = _esc(str(d.get("date") or ""))
            reasoning = _esc(_truncate(str(d.get("reasoning") or ""), 180))
            rows.append(
                f'<a class="decision-row" href="/trade/{did}/">'
                f'<span class="decision-main"><span class="{badge_cls}">'
                f'{action.upper() or "—"}</span><span class="decision-date">'
                f'{decision_date}</span></span>'
                f'<span class="decision-metrics"><span>{qty} sh</span>'
                f'<span>{price}</span></span>'
                f'<span class="decision-reason">{reasoning}</span></a>'
            )
        decisions_body = f'<div class="decision-list ticker-decisions">{"".join(rows)}</div>'
    else:
        decisions_body = '<p class="empty-state">No decisions for this ticker.</p>'

    content = (
        '<section class="hero">'
        '<p class="tag">Ticker drill-down</p>'
        f'<h1>{ticker_esc}</h1>'
        '<p class="intro">Position, theses, and decisions grouped by ticker.</p>'
        '</section>'
        '<section class="section"><div class="head"><h2>Position</h2></div>'
        f'{pos_body}</section>'
        '<section class="section"><div class="head"><h2>Theses</h2></div>'
        f'{theses_body}</section>'
        '<section class="section"><div class="head"><h2>Decisions</h2>'
        '<a class="more" href="/activity/#decisions">All decisions →</a></div>'
        f'{decisions_body}</section>'
    )

    return _render_page_shell(
        title=f"{ticker} ticker",
        description=f"Ticker drill-down for {ticker}.",
        active_nav="activity",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}{_ticker_href(ticker)}",
        og_type="article",
        breadcrumbs=[
            ("Home", "/"),
            ("Activity", "/activity/"),
            (ticker, None),
        ],
        back_href="/activity/",
    )



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
    reasoning = _esc(_truncate(str(d.get("reasoning") or ""), 220))
    return (
        f'<li class="loser-row">'
        f'<a href="/trade/{did}/"><strong>{action_caps} {ticker}</strong></a>'
        f' — {trade_date} · {qty} @ {price} · '
        f'<span class="loser-outcome">{o30}</span>'
        f'<p class="loser-reason">{reasoning}</p>'
        f'</li>'
    )


def _render_rule_row(r: dict) -> str:
    text = _esc(_truncate(str(r.get("rule_text") or ""), 300))
    reason = _esc(_truncate(str(r.get("retirement_reason") or ""), 220))
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
            '<section class="section"><div class="head">'
            '<h2>Closed losers</h2></div>'
            f'<ul class="loser-list">{rows}</ul></section>'
        )
    else:
        losers_section = (
            '<section class="section"><div class="head">'
            '<h2>Closed losers</h2></div>'
            '<p class="empty-state">No closed losers in window. '
            'Either we got lucky or we didn\'t trade enough.</p></section>'
        )

    if retired_rules:
        rows = "".join(_render_rule_row(r) for r in retired_rules)
        rules_section = (
            '<section class="section"><div class="head">'
            '<h2>Retired rules</h2></div>'
            f'<ul class="rule-list">{rows}</ul></section>'
        )
    else:
        rules_section = ""

    return _render_page_shell(
        title="What didn't work",
        description="Closed losers and retired rules. The receipts most accounts hide.",
        active_nav="learning",
        data_page="mistakes",
        content=losers_section + rules_section,
        og_image=f"{base}/og/mistakes.png",
        page_url=f"{base}/mistakes/",
        og_type="article",
        breadcrumbs=[("Home", "/"), ("Learning", "/learning/"), ("Mistakes", None)],
        back_href="/learning/",
    )



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
            "5 closed decisions per signal type.</p>"
        )

    content = (
        '<section class="section">'
        '<div class="head"><h2>What\'s actually working</h2></div>'
        '<p class="subtitle">'
        'Signal-attribution scores from the last 90 days of decisions.</p>'
        + body + '</section>'
    )

    return _render_page_shell(
        title="What's actually working",
        description="Signal-attribution scores. Which inputs predicted, which were noise.",
        active_nav="learning",
        data_page="attribution",
        content=content,
        og_image=f"{base}/og/attribution.png",
        page_url=f"{base}/attribution/",
        og_type="article",
        breadcrumbs=[("Home", "/"), ("Learning", "/learning/"), ("Attribution", None)],
        back_href="/learning/",
    )


def _hero_chip(t: dict) -> str:
    ticker = _esc(t["ticker"])
    blurb = _esc(t.get("thesis") or "")
    tid = int(t["id"])
    return (
        f'<a class="chip" href="/thesis/{tid}/">'
        f'<span class="ticker">{ticker}</span> {blurb}'
        f'</a>'
    )


def _render_homepage_hero(summary: dict, theses: list[dict],
                          sparkline_svg: str) -> str:
    portfolio = _fmt_money(summary.get("portfolio_value"))
    daily = _fmt_signed_pct(summary.get("daily_pnl_pct"))
    total = _fmt_signed_pct(summary.get("total_return_pct"))
    vs_spy = _fmt_signed_pct(summary.get("vs_spy_pct"))
    daily_class = "gain" if (summary.get("daily_pnl_pct") or 0) >= 0 else "loss"

    day_n = summary.get("day_number") or 0
    last_updated = _esc(str(summary.get("last_updated") or ""))

    intro_html = (
        '<p class="intro">'
        'A live AI-managed brokerage account, with performance, memos, '
        'and decisions published after each trading session.'
        '</p>'
    )

    return (
        f'<section class="hero">'
        f'<p class="tag">Day {day_n} · Updated {last_updated}</p>'
        f'<h1>{portfolio}'
        f'<span class="strip {daily_class}">'
        f' {daily} today · {total} all time · {vs_spy} vs S&amp;P</span></h1>'
        f'{intro_html}'
        f'{sparkline_svg}'
        f'</section>'
    )


def _render_latest_decisions(decisions: list[dict] | None) -> str:
    if not decisions:
        return (
            '<section class="section"><div class="head">'
            '<h2>Latest decisions</h2></div>'
            '<p class="empty-state">'
            'No decisions published yet — '
            '<a href="/activity/">see the full log →</a>'
            '</p></section>'
        )

    rows = []
    for d in decisions[:25]:
        action = (d.get("action") or "").lower()
        badge_cls = (
            f"badge badge-{action}" if action in ("buy", "sell", "hold") else "badge"
        )
        ticker_raw = d.get("ticker") or ""
        ticker = _esc(ticker_raw)
        quantity = d.get("quantity")
        qty = "—" if quantity is None else _esc(str(quantity))
        price = _fmt_money(d.get("price"))
        decision_date = _esc(str(d.get("date") or ""))
        reasoning = _esc(_truncate(d.get("reasoning") or "", 120))
        rows.append(
            f'<a class="decision-row" href="{_ticker_href(ticker_raw)}">'
            f'<span class="decision-main">'
            f'<span class="{badge_cls}">{action.upper() or "—"}</span>'
            f'<span class="ticker">{ticker}</span>'
            f'<span class="decision-date">{decision_date}</span>'
            f'</span>'
            f'<span class="decision-metrics">'
            f'<span>{qty} sh</span>'
            f'<span>{price}</span>'
            f'</span>'
            f'<span class="decision-reason">{reasoning}</span>'
            f'</a>'
        )

    return (
        f'<section class="section"><div class="head">'
        f'<h2>Latest decisions</h2>'
        f'<a class="more" href="/activity/#decisions">All decisions →</a>'
        f'</div>'
        f'<div class="decision-list">{"".join(rows)}</div>'
        f'</section>'
    )


def _render_recent_learnings(attribution_top: dict | None,
                             worst_loser: dict | None) -> str:
    if not attribution_top and not worst_loser:
        return ""
    if attribution_top:
        cat = _esc(attribution_top.get("category") or "")
        n = attribution_top.get("sample_size") or 0
        avg = _fmt_signed_pct(attribution_top.get("avg_outcome_30d"))
        working = (
            f'<a class="card" href="/attribution/">'
            f'<div class="lbl">What\'s working</div>'
            f'<h3 class="gain">{cat}</h3>'
            f'<p>{n} trades · {avg} avg</p></a>'
        )
    else:
        working = (
            '<div class="card disabled"><div class="lbl">What\'s working</div>'
            '<p>Not enough samples yet.</p></div>'
        )
    if worst_loser:
        ticker = _esc(worst_loser.get("ticker") or "")
        pct = _fmt_signed_pct(worst_loser.get("outcome_30d_pct"))
        didnt = (
            f'<a class="card" href="/mistakes/">'
            f'<div class="lbl">What didn\'t</div>'
            f'<h3 class="loss"><span class="ticker">{ticker}</span> {pct}</h3>'
            f'<p>Worst recent closed loser.</p></a>'
        )
    else:
        didnt = (
            '<div class="card disabled"><div class="lbl">What didn\'t</div>'
            '<p>No closed losers in window.</p></div>'
        )
    return (
        '<section class="section"><div class="head">'
        '<h2>Recent learnings</h2>'
        '<a class="more" href="/learning/">Learning →</a>'
        '</div>'
        f'<div class="card-grid">{working}{didnt}</div>'
        '</section>'
    )


def _render_memo_block(memo: dict | None) -> str:
    if not memo:
        return ""
    body = _esc(_truncate(memo.get("content") or "", 280))
    session_date = _esc(str(memo.get("session_date") or ""))
    return (
        '<section class="section"><div class="head">'
        '<h2>From today\'s session memo</h2>'
        '<a class="more" href="/strategy/#memos">All memos →</a>'
        '</div>'
        f'<blockquote class="memo-block">'
        f'<div class="meta">{session_date}</div>'
        f'{body}</blockquote>'
        '</section>'
    )


def _render_range_control() -> str:
    """Toolbar above the three-chart grid. Client JS reads data-range
    and re-renders the charts with a filtered slice of the snapshots
    JSON. Default active button = 1W, kept in sync with
    DEFAULT_RANGE_DAYS in public_dashboard/app.js."""
    buttons = [
        ("7", "1W", True),
        ("30", "1M", False),
        ("365", "1Y", False),
        ("all", "All", False),
    ]
    parts = ['<div class="range-control" role="group" aria-label="Time range">']
    for data_range, label, active in buttons:
        cls = "range-btn is-active" if active else "range-btn"
        pressed = "true" if active else "false"
        parts.append(
            f'<button type="button" data-range="{data_range}" '
            f'class="{cls}" aria-pressed="{pressed}">{label}</button>'
        )
    parts.append('</div>')
    return "".join(parts)


def _render_homepage_charts() -> str:
    range_html = _render_range_control()
    return (
        '<section class="section front-charts">'
        '<div class="head"><h2>Performance</h2>'
        '<a class="more" href="/performance/">Full view →</a></div>'
        f'{range_html}'
        '<div class="chart-grid">'
        '<div class="chart-panel primary">'
        '<div class="chart-title">Equity curve</div>'
        '<div class="chart-wrap"><canvas id="equity-chart"></canvas></div>'
        '<p class="empty-state" id="chart-empty" style="display:none;">No snapshot data yet</p>'
        '</div>'
        '<div class="chart-panel">'
        '<div class="chart-title">Cumulative P&amp;L</div>'
        '<div class="chart-wrap"><canvas id="pnl-chart"></canvas></div>'
        '<p class="empty-state" id="pnl-empty" style="display:none;">No snapshot data yet</p>'
        '</div>'
        '<div class="chart-panel">'
        '<div class="chart-title">vs S&amp;P 500</div>'
        '<div class="chart-wrap"><canvas id="benchmark-chart"></canvas></div>'
        '<p class="empty-state" id="benchmark-empty" style="display:none;">No benchmark data yet</p>'
        '</div>'
        '</div></section>'
    )


def _render_strategy_memo_focus(memo: dict | None) -> str:
    memo_html = '<p class="empty-state">No memo published for the latest session yet.</p>'
    link_html = ""
    if memo:
        mid = int(memo["id"])
        memo_body = _esc(_truncate(memo.get("content") or "", 360))
        memo_date = _esc(str(memo.get("session_date") or "Latest memo"))
        link_html = f'<a class="more" href="/memo/{mid}/">Full memo →</a>'
        memo_html = (
            f'<blockquote class="memo-block memo-feature">'
            f'<div class="meta">{memo_date}</div>{memo_body}</blockquote>'
        )

    return (
        '<section class="section memo-focus">'
        '<div class="focus-main">'
        f'<div class="head"><h2>Latest memo</h2>{link_html}</div>'
        f'{memo_html}'
        '</div></section>'
    )


def _methodology_link(label: str, child_path: str, ready: bool) -> str:
    href = child_path if ready else "/how-it-works/"
    return f'<a href="{href}">{_esc(label)}</a>'


def _render_methodology_strip(state: dict) -> str:
    state = state or {}
    return (
        '<div class="methodology-strip">'
        'Built by an AI agent (Claude Haiku for execution, Sonnet for strategy). '
        + _methodology_link("How it works", "/about/", state.get("about", False))
        + ' · '
        + _methodology_link("Sample tool-call trace", "/trace/", state.get("trace", False))
        + ' · '
        + _methodology_link("Model & cost", "/internals/", state.get("internals", False))
        + '</div>'
    )


def render_homepage(*, summary: dict, theses: list[dict],
                    sparkline_svg: str, today_move: dict | None,
                    attribution_top: dict | None, worst_loser: dict | None,
                    memo: dict | None, how_it_works_state: dict,
                    base_url: str, performance: dict | None = None,
                    decisions: list[dict] | None = None) -> str:
    """Render the curated landing homepage."""
    base = base_url.rstrip("/")
    daily_pnl = summary.get("daily_pnl") or 0
    portfolio = summary.get("portfolio_value") or 0
    description = (
        f"Portfolio: {_fmt_money(portfolio)} · "
        f"Today: {_fmt_money(daily_pnl)} ({_fmt_signed_pct(summary.get('daily_pnl_pct'))})"
    )

    content = (
        _render_homepage_hero(summary, theses, sparkline_svg)
        + _render_homepage_charts()
        + _render_latest_decisions(decisions or ([today_move] if today_move else []))
        + _render_recent_learnings(attribution_top, worst_loser)
        + _render_methodology_strip(how_it_works_state)
    )

    return _render_page_shell(
        title="Pinchy",
        description=description,
        active_nav="home",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/",
        head_extra=_CHART_JS_CDN,
    )


_CHART_JS_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/'
    'chart.js@4.4.7/dist/chart.umd.min.js"></script>'
)


def _stat(lbl: str, val: str, cls: str = "") -> str:
    return (
        f'<div class="stat"><div class="lbl">{_esc(lbl)}</div>'
        f'<div class="val {cls}">{val}</div></div>'
    )


def render_performance_page(*, summary: dict, performance: dict,
                            base_url: str) -> str:
    base = base_url.rstrip("/")
    portfolio = _fmt_money(summary.get("portfolio_value"))
    daily = _fmt_signed_pct(summary.get("daily_pnl_pct"))
    total = _fmt_signed_pct(summary.get("total_return_pct"))
    vs_spy = _fmt_signed_pct(summary.get("vs_spy_pct"))

    stat_strip = (
        '<div class="stat-row">'
        + _stat("Portfolio", portfolio)
        + _stat("Today", daily)
        + _stat("All time", total)
        + _stat("vs S&P", vs_spy)
        + '</div>'
    )

    p = performance or {}
    stats_panel = (
        '<section class="section"><div class="head"><h2>Stats</h2></div>'
        '<div class="stat-row">'
        + _stat("Max drawdown", f"{p.get('max_drawdown_pct', 0):+.2f}%")
        + _stat("Win rate", f"{p.get('win_rate_pct', 0):.1f}%")
        + _stat("Avg days held", f"{p.get('avg_days_held', 0):.1f}")
        + _stat("Best day", f"{p.get('best_day_pct', 0):+.2f}%")
        + _stat("Worst day", f"{p.get('worst_day_pct', 0):+.2f}%")
        + '</div></section>'
    )

    range_html = _render_range_control()
    charts = (
        f'<section class="section range-section">{range_html}</section>'
        '<section class="section">'
        '<div class="head"><h2>Equity curve</h2></div>'
        '<p class="section-subtitle">Account value over time. '
        'SPY line shows what the same deposits would be worth in an index fund.</p>'
        '<div class="chart-wrap"><canvas id="equity-chart"></canvas></div>'
        '<p class="empty-state" id="chart-empty" style="display:none;">No snapshot data yet</p>'
        '</section>'
        '<section class="section">'
        '<div class="head"><h2>Cumulative P&amp;L</h2></div>'
        '<p class="section-subtitle">Trading gain/loss in dollars, '
        'with cash deposits subtracted out.</p>'
        '<div class="chart-wrap"><canvas id="pnl-chart"></canvas></div>'
        '<p class="empty-state" id="pnl-empty" style="display:none;">No snapshot data yet</p>'
        '</section>'
        '<section class="section"><div class="head"><h2>Performance vs S&amp;P 500</h2></div>'
        '<div class="chart-wrap"><canvas id="benchmark-chart"></canvas></div>'
        '<p class="empty-state" id="benchmark-empty" style="display:none;">No benchmark data yet</p>'
        '</section>'
    )

    content = stat_strip + charts + stats_panel

    return _render_page_shell(
        title="Performance",
        description=f"Equity curve and benchmark comparison. Portfolio: {portfolio}.",
        active_nav="performance",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/performance/",
        head_extra=_CHART_JS_CDN,
        breadcrumbs=[("Home", "/"), ("Performance", None)],
        back_href="/",
    )


def _render_memos_section(memos: list[dict]) -> str:
    if not memos:
        return (
            '<section class="section" id="memos">'
            '<div class="head"><h2>Recent memos</h2></div>'
            '<p class="empty-state">No memos yet.</p></section>'
        )
    items = []
    for m in memos[:10]:
        mid = int(m["id"])
        body = _esc(_truncate(m.get("content") or "", 180))
        d = _esc(str(m.get("session_date") or ""))
        items.append(
            f'<a class="memo-row" href="/memo/{mid}/">'
            f'<span class="memo-row-date">{d}</span>'
            f'<span class="memo-row-body">{body}</span>'
            f'</a>'
        )
    return (
        '<section class="section" id="memos">'
        '<div class="head"><h2>Recent memos</h2></div>'
        '<div class="memo-list">'
        + "".join(items)
        + '</div></section>'
    )


def _render_strategy_theses_section(theses: list[dict]) -> str:
    if not theses:
        body = '<p class="empty-state">No active theses.</p>'
    else:
        cards = []
        for t in theses:
            tid = int(t["id"])
            ticker = _esc(t.get("ticker") or "")
            thesis = _esc(_truncate(t.get("thesis") or "", 240))
            direction = _esc(t.get("direction") or "")
            confidence = _esc(str(t.get("confidence") or ""))
            meta = " · ".join(bit for bit in (direction, confidence) if bit)
            meta_html = f'<p class="thesis-meta">{meta}</p>' if meta else ""
            cards.append(
                f'<a class="card thesis-summary" href="/thesis/{tid}/">'
                f'<div class="lbl">Active thesis</div>'
                f'<h3><span class="ticker">{ticker}</span></h3>'
                f'{meta_html}<p>{thesis}</p></a>'
            )
        body = f'<div class="card-grid strategy-theses-grid">{"".join(cards)}</div>'
    return (
        '<section class="section" id="theses">'
        '<div class="head"><h2>Active theses</h2></div>'
        f'{body}</section>'
    )


def render_strategy_page(*, theses: list[dict], memos: list[dict],
                         base_url: str) -> str:
    base = base_url.rstrip("/")
    latest = memos[0] if memos else None

    content = (
        '<section class="hero">'
        '<p class="tag">Strategy state</p>'
        '<h1>Strategy</h1>'
        '<p class="intro">Active theses and session memos from the trading agent.</p>'
        '</section>'
        + _render_strategy_theses_section(theses)
        + _render_strategy_memo_focus(latest)
        + _render_memos_section(memos[1:] if latest else memos)
    )

    return _render_page_shell(
        title="Strategy",
        description="Active theses and recent strategy reflection memos.",
        active_nav="strategy",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/strategy/",
        breadcrumbs=[("Home", "/"), ("Strategy", None)],
        back_href="/",
    )


def render_memo_page(*, memo: dict, base_url: str) -> str:
    """Return the full HTML page for one strategy memo."""
    base = base_url.rstrip("/")
    memo_id = int(memo["id"])
    session_date = _esc(str(memo.get("session_date") or ""))
    memo_type = _esc(str(memo.get("memo_type") or "reflection"))
    body = _esc(str(memo.get("content") or ""))
    title = f"Memo {session_date}" if session_date else f"Memo #{memo_id}"

    content = (
        '<section class="hero">'
        '<p class="tag">Strategy memo</p>'
        f'<h1>{_esc(title)}</h1>'
        f'<p class="intro">{memo_type}</p>'
        '</section>'
        '<section class="section">'
        '<div class="head"><h2>Full memo</h2>'
        '<a class="more" href="/strategy/#memos">All memos →</a></div>'
        f'<article class="memo-detail long-text">{body}</article>'
        '</section>'
    )

    return _render_page_shell(
        title=title,
        description=f"Strategy memo from {session_date}.",
        active_nav="strategy",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/memo/{memo_id}/",
        og_type="article",
        breadcrumbs=[
            ("Home", "/"),
            ("Strategy", "/strategy/"),
            (title, None),
        ],
        back_href="/strategy/",
    )


def render_activity_page(*, base_url: str, memos: list[dict]) -> str:
    base = base_url.rstrip("/")

    holdings = (
        '<section class="section" id="holdings">'
        '<div class="head"><h2>Current holdings</h2></div>'
        '<div class="table-wrap"><table id="positions-table">'
        '<thead><tr><th>Ticker</th><th class="num">Shares</th>'
        '<th class="num">Avg Cost</th></tr></thead><tbody></tbody></table></div>'
        '<p class="empty-state" id="positions-empty" style="display:none;">'
        'No open positions</p></section>'
    )

    decisions = (
        '<section class="section" id="decisions">'
        '<div class="head"><h2>Decisions log</h2></div>'
        '<div class="table-wrap"><table id="decisions-table">'
        '<thead><tr><th>Date</th><th>Ticker</th><th>Action</th>'
        '<th class="num">Qty</th><th>Reasoning</th>'
        '<th class="num">Order ID</th></tr></thead><tbody></tbody></table></div>'
        '<p class="empty-state" id="decisions-empty" style="display:none;">'
        'No decisions yet</p></section>'
    )

    content = holdings + decisions

    return _render_page_shell(
        title="Activity",
        description="Holdings and decisions log.",
        active_nav="activity",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/activity/",
        breadcrumbs=[("Home", "/"), ("Activity", None)],
        back_href="/",
    )


def render_changelog_page(*, entries: list[dict], base_url: str) -> str:
    """Return the public changelog page."""
    base = base_url.rstrip("/")

    if entries:
        if any(entry.get("title") and entry.get("summary") for entry in entries):
            cards = []
            for entry in entries:
                entry_date = _esc(str(entry.get("date") or ""))
                title = _esc(str(entry.get("title") or "Update"))
                summary = _esc(str(entry.get("summary") or ""))
                bullets = "".join(
                    f"<li>{_esc(str(b))}</li>" for b in (entry.get("bullets") or [])
                )
                sha_bits = []
                for sha in entry.get("commit_shas") or []:
                    full_sha = _esc(str(sha))
                    sha_bits.append(f'<code title="{full_sha}">{full_sha[:7]}</code>')
                sha_html = (
                    f'<p class="changelog-shas">{" ".join(sha_bits)}</p>'
                    if sha_bits else ""
                )
                bullet_html = f"<ul>{bullets}</ul>" if bullets else ""
                # Raw git-derived entries carry their per-commit list under
                # "items" rather than "bullets"; render it too so subjects
                # appear when no LLM-curated bullets exist.
                item_rows = []
                for item in entry.get("items") or []:
                    if isinstance(item, dict):
                        short = _esc(str(item.get("short_sha") or item.get("sha") or ""))
                        full = _esc(str(item.get("sha") or short))
                        subj = _esc(str(item.get("subject") or ""))
                    else:
                        short = full = ""
                        subj = _esc(str(item))
                    sha_cell = f'<code title="{full}">{short}</code> ' if short else ""
                    item_rows.append(f"<li>{sha_cell}{subj}</li>")
                items_html = f"<ul class=\"changelog-items\">{''.join(item_rows)}</ul>" if item_rows else ""
                cards.append(
                    '<article class="changelog-entry">'
                    f'<p class="changelog-date">{entry_date}</p>'
                    f"<h2>{title}</h2>"
                    f"<p>{summary}</p>"
                    f"{bullet_html}{items_html}{sha_html}</article>"
                )
            body = "".join(cards)
        else:
            rows = []
            for entry in entries:
                entry_date = _esc(str(entry.get("date") or ""))
                for item in entry.get("items") or []:
                    if isinstance(item, dict):
                        short_sha = _esc(str(item.get("short_sha") or item.get("sha") or ""))
                        full_sha = _esc(str(item.get("sha") or short_sha))
                        subject = _esc(str(item.get("subject") or ""))
                    else:
                        short_sha = ""
                        full_sha = ""
                        subject = _esc(str(item))
                    sha_cell = (
                        f'<code title="{full_sha}">{short_sha}</code>' if short_sha else "—"
                    )
                    rows.append(
                        "<tr>"
                        f"<td>{entry_date}</td>"
                        f"<td>{sha_cell}</td>"
                        f"<td>{subject}</td>"
                        "</tr>"
                    )
            body = (
                '<div class="table-wrap"><table class="changelog-table">'
                "<thead><tr><th>Date</th><th>SHA</th><th>Change</th></tr></thead>"
                f'<tbody>{"".join(rows)}</tbody></table></div>'
            )
    else:
        body = '<p class="empty-state">No public updates posted yet.</p>'

    content = (
        '<section class="hero">'
        '<p class="tag">Public build notes</p>'
        '<h1>Changelog</h1>'
        '<p class="intro">Notable changes to the trading agent and public dashboard.</p>'
        '</section>'
        f'<section class="section changelog-list">{body}</section>'
    )

    return _render_page_shell(
        title="Changelog",
        description="Public build notes for the Pinchy trading agent and dashboard.",
        active_nav="changelog",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/changelog/",
        breadcrumbs=[("Home", "/"), ("Changelog", None)],
        back_href="/",
    )


def render_learning_hub(*, attribution_top3: list[dict],
                        losers_top3: list[dict], retired_rules_count: int,
                        base_url: str) -> str:
    base = base_url.rstrip("/")

    if attribution_top3:
        rows = "".join(
            f'<li><strong>{_esc(a.get("category") or "")}</strong> · '
            f'{a.get("sample_size") or 0} trades · '
            f'{_fmt_signed_pct(a.get("avg_outcome_30d"))} avg</li>'
            for a in attribution_top3[:3]
        )
        working_body = f'<ul>{rows}</ul>'
    else:
        working_body = '<p>Not enough samples yet.</p>'

    if losers_top3:
        rows = "".join(
            f'<li><span class="ticker">{_esc(loser.get("ticker") or "")}</span> '
            f'<span class="loss">{_fmt_signed_pct(loser.get("outcome_30d_pct"))}</span></li>'
            for loser in losers_top3[:3]
        )
        didnt_body = (
            f'<ul>{rows}</ul>'
            f'<p>{retired_rules_count} retired rule(s) recently.</p>'
        )
    else:
        didnt_body = (
            '<p>No closed losers in window.</p>'
            f'<p>{retired_rules_count} retired rule(s) recently.</p>'
        )

    content = (
        '<section class="hero"><h1>What this thing has learned</h1></section>'
        '<section class="section"><div class="card-grid">'
        f'<a class="card" href="/attribution/">'
        f'<div class="lbl">What\'s working</div>'
        f'<h3>Top signals</h3>{working_body}'
        f'<p class="more">See all →</p></a>'
        f'<a class="card" href="/mistakes/">'
        f'<div class="lbl">What didn\'t</div>'
        f'<h3>Recent losers</h3>{didnt_body}'
        f'<p class="more">See all →</p></a>'
        '</div></section>'
    )

    return _render_page_shell(
        title="Learning",
        description="What this AI agent has learned: signals that work, mistakes it's made.",
        active_nav="learning",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/learning/",
        breadcrumbs=[("Home", "/"), ("Learning", None)],
        back_href="/",
    )


_HOW_IT_WORKS_CHILDREN = (
    ("about", "/about/", "Methodology",
     "How decisions get made — the agentic loop, the prompts, the data."),
    ("internals", "/internals/", "Model & cost",
     "Which Claude model runs each stage, how often, and what it costs."),
    ("trace", "/trace/", "Tool-call trace",
     "A real strategist session — every tool call, redacted but unedited."),
)


def render_how_it_works_hub(*, child_state: dict, base_url: str) -> str:
    base = base_url.rstrip("/")
    cards = []
    for key, href, title, blurb in _HOW_IT_WORKS_CHILDREN:
        ready = bool(child_state.get(key))
        if ready:
            cards.append(
                f'<a class="card" href="{href}">'
                f'<h3>{_esc(title)}</h3>'
                f'<p>{_esc(blurb)}</p>'
                f'<p class="more">Read →</p></a>'
            )
        else:
            cards.append(
                f'<div class="card disabled">'
                f'<h3>{_esc(title)}</h3>'
                f'<p>{_esc(blurb)}</p>'
                f'<p class="more">Coming soon</p></div>'
            )

    content = (
        '<section class="hero"><h1>How this thing works</h1>'
        '<p class="intro">Pinchy is a real Alpaca brokerage account operated '
        'end-to-end by Claude, which also wrote every line of the code running it. '
        'After every market close, the agent reviews the day&rsquo;s news, updates '
        'its trade theses, places orders, and publishes every decision, mistake, '
        'and lesson here.</p></section>'
        f'<section class="section"><div class="card-grid">{"".join(cards)}</div></section>'
    )

    return _render_page_shell(
        title="How it works",
        description="Methodology, model & cost transparency, and a sample tool-call trace.",
        active_nav="how-it-works",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/how-it-works/",
        breadcrumbs=[("Home", "/"), ("How it works", None)],
        back_href="/",
    )
