"""OG image generation for per-trade and per-thesis link previews.

Pure Pillow — no headless browser, no external service. Output is 1200x630
PNG bytes ready to write to the deploy directory.
"""

from decimal import Decimal
from io import BytesIO

OG_WIDTH = 1200
OG_HEIGHT = 630
_BG = (8, 24, 32)        # dark teal
_FG = (220, 240, 230)    # warm off-white
_ACCENT = (0, 212, 170)  # bikini-bottom green
_MUTED = (140, 160, 150)

_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont"] = {}


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
