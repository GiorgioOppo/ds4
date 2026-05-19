#!/usr/bin/env python3
"""Regenerate the placeholder AppIcon PNGs for the macOS app bundle.

The Mac App Store validator (error 90236) rejects bundles that lack a
1024x1024 (512pt @2x) icon, so the app needs a full AppIcon set. The
PNGs this script writes are committed to the repo at
``Sources/DeepSeekUI/Resources/Assets.xcassets/AppIcon.appiconset/`` and
are picked up automatically by Xcode's asset catalog compiler.

The design is intentionally minimal — a navy-to-cyan radial gradient
with a stylized "DS" wordmark, sized for the macOS icon grid (Apple
HIG: square canvas with the artwork inset, no rounded corners; the OS
masks them in the Dock). Swap in branded artwork before shipping; the
filenames and sizes below match what AppIcon.appiconset/Contents.json
references.

Usage:

    python3 Tools/generate-appicon.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# (file name, pixel size). The size mapping mirrors
# AppIcon.appiconset/Contents.json — keep the two in sync.
ICON_SIZES: list[tuple[str, int]] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

# Brand colors: deep navy → cyan radial gradient. Picked to read as
# "tech / AI" at small sizes and stay distinct against both light and
# dark macOS Dock backgrounds.
BG_INNER = (74, 144, 226)     # #4A90E2 cyan-blue
BG_OUTER = (16, 26, 64)       # #101A40 deep navy
FG = (240, 246, 255)          # #F0F6FF near-white wordmark


def render_icon(size: int) -> Image.Image:
    """Render the AppIcon at ``size``x``size`` pixels."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Radial gradient. Iterate pixel-by-pixel for the small sizes; for
    # larger sizes draw concentric rings to keep render time bounded.
    # The result looks identical visually because the gradient is
    # smooth and the icon is square.
    cx = cy = size / 2.0
    max_r = math.hypot(cx, cy)
    bands = max(64, size // 4)
    draw = ImageDraw.Draw(img)
    for i in range(bands, 0, -1):
        t = i / bands
        r = int(max_r * t)
        col = (
            int(BG_INNER[0] * (1 - t) + BG_OUTER[0] * t),
            int(BG_INNER[1] * (1 - t) + BG_OUTER[1] * t),
            int(BG_INNER[2] * (1 - t) + BG_OUTER[2] * t),
            255,
        )
        draw.ellipse(
            (int(cx - r), int(cy - r), int(cx + r), int(cy + r)),
            fill=col,
        )

    # Inset rounded-square plate so the wordmark sits on a flat
    # background rather than mid-gradient. macOS masks to a rounded
    # square automatically; we just nudge readability.
    inset = max(1, size // 12)
    plate_radius = max(2, size // 5)
    plate_box = (inset, inset, size - inset, size - inset)
    plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(plate)
    pdraw.rounded_rectangle(
        plate_box,
        radius=plate_radius,
        fill=(255, 255, 255, 24),     # subtle frosted overlay
        outline=(255, 255, 255, 64),
        width=max(1, size // 256),
    )
    img.alpha_composite(plate)

    # Wordmark. Use the bundled DejaVu Sans Bold if available
    # (Pillow ships it on most distros); otherwise fall back to the
    # default bitmap font, which is still legible at small sizes.
    text = "DS"
    target_h = int(size * 0.48)
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(candidate):
            try:
                font = ImageFont.truetype(candidate, target_h)
                break
            except Exception:
                font = None
    if font is None:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (size - tw) / 2 - bbox[0]
    ty = (size - th) / 2 - bbox[1]
    # Soft shadow under the wordmark for contrast on the gradient.
    shadow_off = max(1, size // 128)
    draw.text((tx + shadow_off, ty + shadow_off), text,
              font=font, fill=(0, 0, 0, 140))
    draw.text((tx, ty), text, font=font, fill=FG + (255,))
    return img


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    out = here / "Sources/DeepSeekUI/Resources/Assets.xcassets/AppIcon.appiconset"
    out.mkdir(parents=True, exist_ok=True)
    for name, size in ICON_SIZES:
        img = render_icon(size)
        img.save(out / name, format="PNG", optimize=True)
        print(f"wrote {out / name} ({size}x{size})")


if __name__ == "__main__":
    main()
