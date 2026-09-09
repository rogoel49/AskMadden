"""Phase 5.4: generate the PWA / home-screen icons programmatically.

    python design/icons/build_icons.py

Writes, next to this script:
    icon-192.png          Web App Manifest icon (Android Chrome minimum)
    icon-512.png          Web App Manifest icon (Android Chrome minimum)
    apple-touch-icon.png  180x180, what iOS Safari actually uses for the
                          Home Screen (it ignores manifest icons)

Design: the app's --bg (#05070a) as a full-bleed square, with a green
(--green, #39e39a) "AM" monogram in Anton -- the same face as the
wordmark in design/askmadden-ui-mockup.html. Full-bleed on purpose: iOS
rounds the corners itself, and Android's adaptive-icon mask can crop up
to ~10% per edge, so the monogram is kept inside the central safe zone
and the same PNGs serve as both "any" and "maskable" icons.

The Anton TTF is fetched from Google Fonts (the mockup already loads it
from there) into a cache dir and is never committed. Offline, this
falls back to DejaVu Sans Bold so the build still succeeds -- the
committed PNGs were built with Anton.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BG = "#05070a"
GREEN = "#39e39a"
TEXT = "AM"
FONT_CSS_URL = "https://fonts.googleapis.com/css2?family=Anton&display=swap"
FALLBACK_TTF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
OUTPUTS = {"icon-192.png": 192, "icon-512.png": 512, "apple-touch-icon.png": 180}


def _fetch_anton(cache_dir: Path) -> Path | None:
    """Download Anton-Regular.ttf via Google Fonts' CSS endpoint; None if offline."""
    ttf = cache_dir / "Anton-Regular.ttf"
    if ttf.exists():
        return ttf
    try:
        # A non-browser UA makes Google Fonts serve a plain TTF url (no woff2 unicode-range split).
        req = urllib.request.Request(FONT_CSS_URL, headers={"User-Agent": "python-urllib"})
        css = urllib.request.urlopen(req, timeout=20).read().decode()
        m = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)", css)
        if not m:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        ttf.write_bytes(urllib.request.urlopen(m.group(1), timeout=20).read())
        return ttf
    except Exception as exc:  # noqa: BLE001 -- offline is an expected, non-fatal case
        print(f"note: could not fetch Anton ({exc}); falling back to DejaVu Sans Bold", file=sys.stderr)
        return None


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def render(size: int, font_path: Path) -> Image.Image:
    """Render one square icon at `size` px, rendered 4x and downsampled for clean edges."""
    scale = 4
    px = size * scale
    img = Image.new("RGB", (px, px), BG)
    draw = ImageDraw.Draw(img)
    # Monogram cap-height ~ 42% of the icon: well inside Android's maskable safe zone.
    font = _load_font(font_path, int(px * 0.42))
    left, top, right, bottom = draw.textbbox((0, 0), TEXT, font=font, anchor="ls")
    w, h = right - left, bottom - top
    x = (px - w) / 2 - left
    y = (px - h) / 2 - top
    draw.text((x, y), TEXT, font=font, fill=GREEN, anchor="ls")
    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    cache_dir = Path.home() / ".cache" / "askmadden-fonts"
    font_path = _fetch_anton(cache_dir) or FALLBACK_TTF
    print(f"font: {font_path}")
    for name, size in OUTPUTS.items():
        out = HERE / name
        render(size, font_path).save(out, optimize=True)
        print(f"wrote {out.relative_to(HERE.parents[1])} ({size}x{size}, {out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
