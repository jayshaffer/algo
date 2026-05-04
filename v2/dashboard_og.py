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


def _canvas():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (OG_WIDTH, OG_HEIGHT), _BG)
    draw = ImageDraw.Draw(img)
    # Accent bar across the top
    draw.rectangle([(0, 0), (OG_WIDTH, 8)], fill=_ACCENT)
    # Footer line
    draw.text((48, OG_HEIGHT - 56), "BIKINI BOTTOM CAPITAL", fill=_MUTED)
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

    draw.text((48, 80), action, fill=_ACCENT)
    draw.text((48, 130), ticker, fill=_FG)
    if qty:
        draw.text((48, 360), f"{qty} shares", fill=_FG)
    if price_str:
        draw.text((48, 410), price_str, fill=_MUTED)

    return _to_png_bytes(img)
