"""OG image generation for per-trade and per-thesis link previews.

Pure Pillow — no headless browser, no external service. Output is 1200x630
PNG bytes ready to write to the deploy directory.
"""

from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import ImageFont as _ImageFontModule

OG_WIDTH = 1200
OG_HEIGHT = 630
_BG = (8, 24, 32)        # dark teal
_FG = (220, 240, 230)    # warm off-white
_ACCENT = (0, 212, 170)  # bikini-bottom green
_MUTED = (140, 160, 150)

_FONT_CACHE: dict[int, _ImageFontModule.FreeTypeFont] = {}


def _load_font(size: int):
    """Return a PIL FreeTypeFont at *size* pts, loaded from Pillow's bundled DejaVu Sans TTF.

    Results are cached by size so repeated calls are O(1) after the first.
    """
    if size not in _FONT_CACHE:
        from PIL import ImageFont
        _FONT_CACHE[size] = ImageFont.load_default(size=size)
    return _FONT_CACHE[size]


def _canvas():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (OG_WIDTH, OG_HEIGHT), _BG)
    draw = ImageDraw.Draw(img)
    # Accent bar across the top
    draw.rectangle([(0, 0), (OG_WIDTH, 8)], fill=_ACCENT)
    # Footer line
    draw.text((48, OG_HEIGHT - 48), "BIKINI BOTTOM CAPITAL", fill=_MUTED, font=_load_font(24))
    return img, draw


def _to_png_bytes(img) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_trade_og(decision: dict) -> bytes:
    """Return PNG bytes (1200x630) for the OG card of one trade."""
    img, draw = _canvas()
    ticker = str(decision.get("ticker", "?"))
    action = str(decision.get("action", "")).upper()
    qty = decision.get("quantity") or 0
    price = decision.get("price")
    price_str = f"${Decimal(price):,.2f}" if price is not None else ""

    draw.text((48, 90), action, fill=_ACCENT, font=_load_font(64))
    draw.text((48, 170), ticker, fill=_FG, font=_load_font(220))
    if qty:
        draw.text((48, 430), f"{qty} shares", fill=_FG, font=_load_font(42))
    if price_str:
        draw.text((48, 490), price_str, fill=_MUTED, font=_load_font(56))

    return _to_png_bytes(img)


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _wrap_pixels(draw, text: str, font, max_width_px: int, max_lines: int) -> list[str]:
    """Wrap *text* to fit within *max_width_px* using *font*'s metrics.

    Splits on whitespace; never breaks individual words mid-character. Returns
    at most *max_lines* lines, with a trailing ellipsis on the last line if the
    text was truncated.
    """
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip() if current else word
        if draw.textlength(candidate, font=font) <= max_width_px:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
            if len(lines) == max_lines:
                # Already filled. Apply ellipsis to last line and stop.
                last = lines[-1]
                while last and draw.textlength(last + "…", font=font) > max_width_px:
                    last = last[:-1].rstrip()
                lines[-1] = last + "…"
                return lines
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def render_home_og(summary: dict) -> bytes:
    """Return PNG bytes (1200x630) for the OG card of the homepage."""
    img, draw = _canvas()

    portfolio_value = summary.get("portfolio_value") or 0
    daily_pnl = summary.get("daily_pnl") or 0
    daily_pnl_pct = summary.get("daily_pnl_pct") or 0

    try:
        pv = Decimal(str(portfolio_value))
        pv_str = f"${pv:,.2f}"
    except Exception:
        pv_str = "$0.00"

    try:
        pnl = Decimal(str(daily_pnl))
        pct = Decimal(str(daily_pnl_pct))
        sign = "+" if pnl > 0 else ""
        pnl_str = f"Today: {sign}${abs(pnl):,.2f} ({sign}{float(pct):+.2f}%)"
        pnl_color = _ACCENT if pnl > 0 else _MUTED
    except Exception:
        pnl_str = "Today: $0.00 (+0.00%)"
        pnl_color = _MUTED

    draw.text((48, 90), "BIKINI BOTTOM CAPITAL", fill=_ACCENT, font=_load_font(64))
    draw.text((48, 170), pv_str, fill=_FG, font=_load_font(180))
    draw.text((48, 410), pnl_str, fill=pnl_color, font=_load_font(42))

    return _to_png_bytes(img)


def render_mistakes_og(top_loser: dict | None) -> bytes:
    """Return PNG bytes (1200x630) for the /mistakes/ OG card."""
    img, draw = _canvas()

    draw.text((48, 90), "WHAT DIDN'T WORK", fill=_ACCENT, font=_load_font(56))

    if top_loser:
        ticker = str(top_loser.get("ticker", "?"))
        outcome = top_loser.get("outcome_30d")
        if outcome is not None:
            try:
                outcome_str = f"{Decimal(str(outcome)):+.2f}% (30d)"
            except Exception:
                outcome_str = ""
        else:
            outcome_str = ""
        draw.text((48, 200), ticker, fill=_FG, font=_load_font(220))
        if outcome_str:
            draw.text((48, 460), outcome_str, fill=_MUTED, font=_load_font(48))
    else:
        draw.text(
            (48, 240),
            "No closed losers in window.",
            fill=_FG,
            font=_load_font(56),
        )

    return _to_png_bytes(img)


def render_attribution_og(attribution: list[dict]) -> bytes:
    """Return PNG bytes (1200x630) showing the top-5 signal-attribution bars.

    Layout: title at top, baseline at y=400. Bars are 80px wide, 50px gap,
    starting at x=180. Positive bars (avg_outcome_30d > 0) draw upward in
    accent color; negative bars draw downward in muted color. Categories
    are labelled below the baseline.
    """
    img, draw = _canvas()

    draw.text((48, 50), "WHAT'S ACTUALLY WORKING", fill=_ACCENT, font=_load_font(48))
    draw.text((48, 110), "signal attribution (avg 30d outcome)", fill=_MUTED, font=_load_font(28))

    if not attribution:
        draw.text((48, 280), "Not enough samples yet.", fill=_FG, font=_load_font(56))
        return _to_png_bytes(img)

    BASELINE = 400
    BAR_W = 80
    GAP = 50
    X0 = 180
    MAX_BAR_PX = 200

    # Top-5 by sample_size (largest sample first), or fall back to whatever's
    # there if fewer rows.
    top = sorted(
        attribution,
        key=lambda r: int(r.get("sample_size") or 0),
        reverse=True,
    )[:5]

    # Find scale across the displayed slice
    scores = [
        float(r.get("avg_outcome_30d") or 0)
        for r in top
    ]
    max_abs = max((abs(s) for s in scores), default=1.0) or 1.0

    # Baseline line
    draw.line([(48, BASELINE), (1200 - 48, BASELINE)], fill=_MUTED, width=2)

    label_font = _load_font(24)
    for i, row in enumerate(top):
        x = X0 + i * (BAR_W + GAP)
        score = float(row.get("avg_outcome_30d") or 0)
        height_px = int(round((abs(score) / max_abs) * MAX_BAR_PX))
        if score >= 0:
            top_y = BASELINE - height_px
            color = _ACCENT
            draw.rectangle([(x, top_y), (x + BAR_W, BASELINE)], fill=color)
        else:
            color = _MUTED
            draw.rectangle([(x, BASELINE), (x + BAR_W, BASELINE + height_px)], fill=color)

        # Category label
        category = str(row.get("category") or "")
        if len(category) > 10:
            category = category[:9] + "…"
        draw.text((x, BASELINE + MAX_BAR_PX + 20), category, fill=_FG, font=label_font)

    return _to_png_bytes(img)


def render_thesis_og(thesis: dict) -> bytes:
    """Return PNG bytes (1200x630) for the OG card of one thesis."""
    img, draw = _canvas()
    ticker = str(thesis.get("ticker", "?"))
    direction = str(thesis.get("direction", "")).upper()
    confidence = str(thesis.get("confidence", "")).lower()
    thesis_text = _truncate(str(thesis.get("thesis", "")), 240)

    label = f"{direction} THESIS" if direction else "THESIS"
    label_font = _load_font(48)
    ticker_font = _load_font(180)
    meta_font = _load_font(28)
    body_font = _load_font(36)

    draw.text((48, 90), label, fill=_ACCENT, font=label_font)
    draw.text((48, 160), ticker, fill=_FG, font=ticker_font)
    if confidence:
        draw.text((48, 360), f"confidence: {confidence}", fill=_MUTED, font=meta_font)

    if thesis_text:
        wrapped = _wrap_pixels(
            draw, thesis_text, body_font,
            max_width_px=OG_WIDTH - 96,  # 48px padding each side
            max_lines=4,
        )
        for i, line in enumerate(wrapped):
            draw.text((48, 410 + i * 44), line, fill=_FG, font=body_font)

    return _to_png_bytes(img)
