from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT_DIR / "assets"
ICONSET_DIR = ASSET_DIR / "ResearchPodcastRadar.iconset"
SOURCE_PNG = ASSET_DIR / "ResearchPodcastRadar_icon_summer_source.png"
PNG_1024 = ASSET_DIR / "ResearchPodcastRadar_icon_1024.png"
ICNS_PATH = ASSET_DIR / "ResearchPodcastRadar.icns"


def mix(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def radial_disc(size: int, center: tuple[int, int], radius: int) -> Image.Image:
    disc = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pixels = disc.load()
    cx, cy = center
    inner = (72, 246, 117)
    outer = (29, 185, 84)
    edge = (13, 142, 66)
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            dx = x - cx
            dy = y - cy
            dist = math.hypot(dx, dy)
            if dist > radius:
                continue
            t = min(1.0, dist / radius)
            light = max(0.0, 1.0 - math.hypot(x - (cx - radius * 0.36), y - (cy - radius * 0.42)) / (radius * 1.2))
            base = outer if t < 0.84 else edge
            r = mix(base[0], inner[0], light * 0.58)
            g = mix(base[1], inner[1], light * 0.58)
            b = mix(base[2], inner[2], light * 0.58)
            alpha = 255 if t < 0.98 else round(255 * (1.0 - (t - 0.98) / 0.02))
            pixels[x, y] = (r, g, b, max(0, min(255, alpha)))
    return disc


def draw_icon(size: int = 1024) -> Image.Image:
    scale = 3
    canvas_size = size * scale
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    pad = 76 * scale
    radius = 216 * scale
    shadow_draw.rounded_rectangle(
        [pad, pad + 20 * scale, canvas_size - pad, canvas_size - pad + 20 * scale],
        radius=radius,
        fill=(0, 0, 0, 128),
    )
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(28 * scale)))

    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        [pad, pad, canvas_size - pad, canvas_size - pad],
        radius=radius,
        fill=255,
    )

    base = Image.new("RGBA", img.size, (0, 0, 0, 0))
    base_pixels = base.load()
    for y in range(canvas_size):
        for x in range(canvas_size):
            diagonal = (x + y) / (canvas_size * 2)
            vignette = math.hypot(x - canvas_size / 2, y - canvas_size / 2) / (canvas_size * 0.68)
            green_glow = max(0.0, 1.0 - math.hypot(x - canvas_size * 0.38, y - canvas_size * 0.34) / (canvas_size * 0.72))
            r = mix(13, 19, diagonal) + round(green_glow * 5) - round(vignette * 6)
            g = mix(15, 23, diagonal) + round(green_glow * 16) - round(vignette * 5)
            b = mix(16, 18, diagonal) + round(green_glow * 9) - round(vignette * 4)
            base_pixels[x, y] = (max(0, r), max(0, g), max(0, b), 255)
    base.putalpha(mask)
    img.alpha_composite(base)

    rim = Image.new("RGBA", img.size, (0, 0, 0, 0))
    rim_draw = ImageDraw.Draw(rim)
    rim_draw.rounded_rectangle(
        [pad + 9 * scale, pad + 9 * scale, canvas_size - pad - 9 * scale, canvas_size - pad - 9 * scale],
        radius=radius - 9 * scale,
        outline=(70, 255, 120, 62),
        width=3 * scale,
    )
    img.alpha_composite(rim)

    disc_size = canvas_size
    disc = radial_disc(disc_size, (canvas_size // 2, canvas_size // 2), 326 * scale)
    img.alpha_composite(disc)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx = canvas_size // 2
    cy = canvas_size // 2

    draw.ellipse(
        [cx - 326 * scale, cy - 326 * scale, cx + 326 * scale, cy + 326 * scale],
        outline=(190, 255, 200, 70),
        width=5 * scale,
    )

    wave_color = (7, 12, 10, 236)
    wave_specs = (
        ((-235, -78, 250, 176), 203, 338, 42),
        ((-184, -30, 198, 137), 204, 336, 34),
        ((-132, 17, 146, 97), 205, 334, 28),
    )
    for box_offsets, start, end, width in wave_specs:
        x1, y1, x2, y2 = [value * scale for value in box_offsets]
        draw.arc(
            [cx + x1, cy + y1, cx + x2, cy + y2],
            start=start,
            end=end,
            fill=wave_color,
            width=width * scale,
        )

    draw.ellipse(
        [cx - 55 * scale, cy + 98 * scale, cx + 55 * scale, cy + 208 * scale],
        fill=(8, 14, 11, 214),
    )
    draw.ellipse(
        [cx - 23 * scale, cy + 130 * scale, cx + 23 * scale, cy + 176 * scale],
        fill=(45, 235, 98, 210),
    )

    sweep = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sweep_draw = ImageDraw.Draw(sweep)
    sweep_draw.pieslice(
        [cx - 316 * scale, cy - 316 * scale, cx + 316 * scale, cy + 316 * scale],
        start=300,
        end=331,
        fill=(255, 255, 255, 34),
    )
    sweep.putalpha(sweep.split()[-1].filter(ImageFilter.GaussianBlur(4 * scale)))
    img.alpha_composite(sweep)
    img.alpha_composite(overlay)

    final = img.resize((size, size), Image.Resampling.LANCZOS)
    return final


def load_icon_source(size: int = 1024) -> Image.Image:
    """Load the Summer artwork and normalize it for a macOS icon bundle."""
    if not SOURCE_PNG.exists():
        return draw_icon(size)

    with Image.open(SOURCE_PNG) as source:
        source.load()
        width, height = source.size
        edge = min(width, height)
        left = (width - edge) // 2
        top = (height - edge) // 2
        icon = source.crop((left, top, left + edge, top + edge))
        icon = icon.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")

    # The generated source uses a pure-black canvas around the macOS squircle.
    # Turn only those border-black pixels transparent while retaining the
    # midnight-navy icon face and its dark-blue line art.
    red, green, blue, original_alpha = icon.split()
    brightness = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    border_alpha = brightness.point(
        lambda value: 0 if value <= 2 else 255 if value >= 14 else round((value - 2) * 255 / 12)
    )
    icon.putalpha(ImageChops.multiply(original_alpha, border_alpha))
    return icon


def write_iconset(source: Image.Image) -> None:
    if ICONSET_DIR.exists():
        shutil.rmtree(ICONSET_DIR)
    ICONSET_DIR.mkdir(parents=True, exist_ok=True)
    sizes = (
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    )
    for pixel_size, name in sizes:
        source.resize((pixel_size, pixel_size), Image.Resampling.LANCZOS).save(ICONSET_DIR / name)


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    icon = load_icon_source()
    icon.save(PNG_1024)
    write_iconset(icon)
    subprocess.run(["iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(ICNS_PATH)], check=True)
    print(PNG_1024)
    print(ICNS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
