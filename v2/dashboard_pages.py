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


_NAV_ITEMS = (
    ("home", "/", "Home"),
    ("performance", "/performance/", "Performance"),
    ("activity", "/activity/", "Activity"),
    ("learning", "/learning/", "Learning"),
    ("how-it-works", "/how-it-works/", "How it works"),
)


def _render_nav(active_nav: str) -> str:
    parts = ['<nav class="site-nav"><div class="container">']
    parts.append('<span class="logo">⌬ Bikini Bottom Capital</span>')
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
<title>$title — Bikini Bottom Capital</title>
$meta_block
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍍</text></svg>" />
<link rel="stylesheet" href="/styles.css" />
$head_extra
</head>
<body data-page="$active_nav">
$nav
<main class="container">
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
                       data_page: str | None = None) -> str:
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
    return _PAGE_SHELL_TEMPLATE.substitute(
        title=_esc(title),
        meta_block=meta_block,
        head_extra=head_extra,
        active_nav=_esc(data_page or active_nav),
        nav=_render_nav(active_nav),
        content=content,
        footer=_FOOTER_HTML,
    )


def _fmt_money(value: Decimal | int | float | None) -> str:
    if value is None:
        return "$0.00"
    return f"${Decimal(value):,.2f}"


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


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

    chips_html = ""
    if theses:
        chip_items = "".join(_hero_chip(t) for t in theses[:3])
        chips_html = (
            f'<div class="label">Currently betting on</div>'
            f'<div class="chips">{chip_items}</div>'
        )

    return (
        f'<section class="hero">'
        f'<p class="tag">Day {day_n} · Updated {last_updated}</p>'
        f'<h1>{portfolio}'
        f'<span class="strip {daily_class}">'
        f' {daily} today · {total} all time · {vs_spy} vs S&amp;P</span></h1>'
        f'{chips_html}'
        f'{sparkline_svg}'
        f'</section>'
    )


def _render_today_move(today_move: dict | None) -> str:
    if not today_move:
        return (
            '<section class="section"><div class="head">'
            '<h2>Today\'s move</h2></div>'
            '<p class="empty-state">'
            'No new positions in the last 5 sessions — '
            '<a href="/activity/">see the full log →</a>'
            '</p></section>'
        )
    did = int(today_move["id"])
    action = (today_move.get("action") or "").lower()
    badge_cls = f"badge badge-{action}" if action in ("buy", "sell", "hold") else "badge"
    ticker = _esc(today_move.get("ticker") or "")
    notional = _fmt_money(today_move.get("notional"))
    pct = float(today_move.get("pct_of_portfolio") or 0)
    reasoning = _esc(_truncate(today_move.get("reasoning") or "", 150))
    return (
        f'<section class="section"><div class="head">'
        f'<h2>Today\'s move</h2>'
        f'<a class="more" href="/activity/#decisions">All decisions →</a>'
        f'</div>'
        f'<a class="move-card" href="/trade/{did}/">'
        f'<div class="head">'
        f'<span class="{badge_cls}">{action.upper()}</span> '
        f'<span class="ticker">{ticker}</span> · {notional} · {pct:.1f}% of portfolio'
        f'</div>'
        f'<p class="reasoning">{reasoning}</p>'
        f'</a></section>'
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
        '<a class="more" href="/activity/#memos">All memos →</a>'
        '</div>'
        f'<blockquote class="memo-block">'
        f'<div class="meta">{session_date}</div>'
        f'{body}</blockquote>'
        '</section>'
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
                    base_url: str) -> str:
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
        + _render_today_move(today_move)
        + _render_recent_learnings(attribution_top, worst_loser)
        + _render_memo_block(memo)
        + _render_methodology_strip(how_it_works_state)
    )

    return _render_page_shell(
        title="Bikini Bottom Capital",
        description=description,
        active_nav="home",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/",
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

    charts = (
        '<section class="section"><div class="head"><h2>Equity curve</h2></div>'
        '<div class="chart-wrap"><canvas id="equity-chart"></canvas></div>'
        '<p class="empty-state" id="chart-empty" style="display:none;">No snapshot data yet</p>'
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
        body = _esc(m.get("content") or "")
        d = _esc(str(m.get("session_date") or ""))
        items.append(
            f'<blockquote class="memo-block">'
            f'<div class="meta">{d}</div>{body}</blockquote>'
        )
    return (
        '<section class="section" id="memos">'
        '<div class="head"><h2>Recent memos</h2></div>'
        + "".join(items)
        + '</section>'
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

    theses = (
        '<section class="section" id="theses">'
        '<div class="head"><h2>Active theses</h2></div>'
        '<div id="theses-list"></div>'
        '<p class="empty-state" id="theses-empty" style="display:none;">'
        'No active theses</p></section>'
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

    content = holdings + theses + decisions + _render_memos_section(memos)

    return _render_page_shell(
        title="Activity",
        description="Holdings, active theses, decisions log, and recent memos.",
        active_nav="activity",
        content=content,
        og_image=f"{base}/og/home.png",
        page_url=f"{base}/activity/",
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
            f'<li><span class="ticker">{_esc(l.get("ticker") or "")}</span> '
            f'<span class="loss">{_fmt_signed_pct(l.get("outcome_30d_pct"))}</span></li>'
            for l in losers_top3[:3]
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
    )
