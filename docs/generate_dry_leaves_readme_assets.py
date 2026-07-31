"""Regenerate the dry-leaf README visuals from a multi-object result folder.

Usage:
    python docs/generate_dry_leaves_readme_assets.py \
        sessions/dry-leaves-20260731-224313/out

The relight intentionally mirrors ``web/static/js/relight.js``: sRGB albedo is
multiplied by ambient + Lambertian light and the exported perceptual roughness
drives the same isotropic GGX term used by the interactive viewport.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
INK = (238, 240, 242)
MUTED = (151, 158, 164)
ACCENT = (244, 166, 58)
PANEL = (20, 23, 27)
EDGE = (77, 82, 87)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    windows = Path("C:/Windows/Fonts")
    name = "consolab.ttf" if mono and bold else "consola.ttf" if mono else \
        "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(str(windows / name), size)
    except OSError:
        return ImageFont.load_default()


def contain(image: Image.Image, box: tuple[int, int], margin: int = 0) -> Image.Image:
    target = (box[0] - margin * 2, box[1] - margin * 2)
    copy = image.copy()
    copy.thumbnail(target, Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", box, (0, 0, 0, 0))
    layer.alpha_composite(copy.convert("RGBA"),
                          ((box[0] - copy.width) // 2, (box[1] - copy.height) // 2))
    return layer


def checker(size: tuple[int, int], cell: int = 28) -> Image.Image:
    image = Image.new("RGB", size, (19, 18, 16))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(12, 12, 11))
    return image


def masked_map(path: Path, alpha_path: Path, size: int,
               roughness: bool = False) -> Image.Image:
    alpha = Image.open(alpha_path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    if roughness:
        raw = np.asarray(Image.open(path), dtype=np.float32)
        raw /= 65535.0 if raw.max() > 255 else 255.0
        grey = Image.fromarray(raw, "F").resize((size, size), Image.Resampling.BILINEAR)
        values = np.asarray(grey, dtype=np.float32)
        rgb = np.repeat(np.clip(values * 255, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        image = Image.fromarray(rgb, "RGB")
    else:
        image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    image.putalpha(alpha)
    return image


def card(canvas: Image.Image, bounds: tuple[int, int, int, int], title: str,
         kicker: str, content: Image.Image, accent: bool = False) -> None:
    x0, y0, x1, y1 = bounds
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(bounds, radius=8, fill=PANEL, outline=EDGE, width=2)
    if accent:
        draw.rounded_rectangle((x0, y0, x1, y0 + 8), radius=8, fill=ACCENT)
        draw.rectangle((x0, y0 + 4, x1, y0 + 8), fill=ACCENT)
    draw.text((x0 + 28, y0 + 24), title, font=font(30, bold=True), fill=INK)
    draw.text((x0 + 28, y0 + 64), kicker.upper(), font=font(15, mono=True), fill=MUTED)
    content_box = (x0 + 20, y0 + 100, x1 - 20, y1 - 20)
    available = (content_box[2] - content_box[0], content_box[3] - content_box[1])
    fitted = contain(content, available, margin=8)
    canvas.alpha_composite(fitted, (content_box[0], content_box[1]))


def atlas_overview(out: Path) -> None:
    width, height = 2000, 690
    image = Image.new("RGBA", (width, height), (15, 18, 22, 255))
    detection = Image.open(out / "qa" / "object_detection.png").convert("RGBA")
    albedo = masked_map(out / "albedo_srgb.png", out / "alpha.png", 520)
    normal = masked_map(out / "normal_gl_rgba.png", out / "alpha.png", 520)
    cards = [(35, 35, 650, 655), (692, 35, 1307, 655), (1349, 35, 1964, 655)]
    card(image, cards[0], "Detect + track", "8 objects across 4 rotations", detection, True)
    card(image, cards[1], "Albedo atlas", "lighting removed · shared UVs", albedo)
    card(image, cards[2], "OpenGL normals", "independently aligned · shared UVs", normal)
    image.convert("RGB").save(ASSET_DIR / "dry-leaves-atlas-workflow.webp",
                              "WEBP", quality=88, method=6)


def roughness_comparison(out: Path) -> None:
    width, height = 2050, 660
    image = Image.new("RGBA", (width, height), (15, 18, 22, 255))
    specs = [
        ("Consensus", "robust delivered map", "roughness_consensus.png"),
        ("GGX", "long-tailed PBR lobe", "roughness_ggx.png"),
        ("Beckmann", "Gaussian slope model", "roughness_beckmann.png"),
        ("Ward", "independent comparison", "roughness_ward.png"),
    ]
    gap, margin = 34, 34
    card_width = (width - margin * 2 - gap * 3) // 4
    for index, (title, kicker, filename) in enumerate(specs):
        x0 = margin + index * (card_width + gap)
        content = masked_map(out / filename, out / "alpha.png", 440, roughness=True)
        card(image, (x0, 34, x0 + card_width, 626), title, kicker, content,
             accent=index == 0)
    image.convert("RGB").save(ASSET_DIR / "dry-leaves-roughness-comparison.webp",
                              "WEBP", quality=88, method=6)


def _ggx_g1(no_x: np.ndarray, alpha2: np.ndarray) -> np.ndarray:
    return 2.0 * no_x / (no_x + np.sqrt(alpha2 + (1.0 - alpha2) * no_x * no_x))


def relight(out: Path) -> None:
    map_size = 620
    albedo = np.asarray(Image.open(out / "albedo_srgb.png").convert("RGB").resize(
        (map_size, map_size), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    normal = np.asarray(Image.open(out / "normal_gl_rgba.png").convert("RGB").resize(
        (map_size, map_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    normal = normal * 2.0 - 1.0
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-7)
    alpha_mask = np.asarray(Image.open(out / "alpha.png").convert("L").resize(
        (map_size, map_size), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    raw_roughness = np.asarray(Image.open(out / "roughness_consensus.png"), dtype=np.float32)
    raw_roughness /= 65535.0 if raw_roughness.max() > 255 else 255.0
    roughness = np.asarray(Image.fromarray(raw_roughness, "F").resize(
        (map_size, map_size), Image.Resampling.BILINEAR), dtype=np.float32)

    ambient, specular, elevation = 0.12, 1.0, 35.0
    elevation_rad = math.radians(elevation)
    ch, sz = math.cos(elevation_rad), math.sin(elevation_rad)
    frames: list[Image.Image] = []
    for frame_index in range(36):
        azimuth = math.radians(frame_index * 10.0)
        light = np.array([math.cos(azimuth) * ch, math.sin(azimuth) * ch, sz],
                         dtype=np.float32)
        diffuse = np.maximum(normal @ light, 0.0)
        lit = albedo * (ambient + diffuse[..., None])

        view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vector = light + view
        half_vector /= np.linalg.norm(half_vector)
        no_v = np.maximum(normal[..., 2], 0.0001)
        no_h = np.maximum(normal @ half_vector, 0.0001)
        vo_h = max(float(half_vector[2]), 0.0)
        r = np.clip(roughness, 0.02, 1.0)
        micro_alpha = np.maximum(r * r, 0.0004)
        alpha2 = micro_alpha * micro_alpha
        q = no_h * no_h * (alpha2 - 1.0) + 1.0
        distribution = alpha2 / (math.pi * q * q)
        geometry = _ggx_g1(diffuse, alpha2) * _ggx_g1(no_v, alpha2)
        fresnel = 0.04 + 0.96 * (1.0 - vo_h) ** 5
        spec = distribution * geometry * fresnel / np.maximum(4.0 * no_v * diffuse, 0.0001)
        lit += (spec * diffuse * specular)[..., None]
        lit = np.clip(lit, 0.0, 1.0)

        canvas = checker((640, 730), cell=28)
        background = np.asarray(canvas.crop((10, 100, 630, 720)), dtype=np.float32) / 255.0
        mixed = background * (1.0 - alpha_mask[..., None]) + lit * alpha_mask[..., None]
        canvas.paste(Image.fromarray(np.clip(mixed * 255, 0, 255).astype(np.uint8), "RGB"),
                     (10, 100))

        draw = ImageDraw.Draw(canvas)
        cx, cy, radius = 48, 45, 31
        hud = (159, 190, 55)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     outline=hud, width=2)
        endpoint = (cx + int(math.cos(azimuth) * 25),
                    cy - int(math.sin(azimuth) * 25))
        draw.line((cx, cy, *endpoint), fill=hud, width=5)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=hud)
        draw.ellipse((endpoint[0] - 6, endpoint[1] - 6,
                      endpoint[0] + 6, endpoint[1] + 6), fill=hud)
        draw.text((92, 20), "DRY-LEAF ATLAS | 360 DEG RELIGHT",
                  font=font(14, bold=True, mono=True), fill=hud)
        draw.text((92, 43), "8 OBJECTS | GGX ROUGHNESS",
                  font=font(12, mono=True), fill=(189, 184, 163))
        draw.text((92, 63), "ELEVATION 35° | AMBIENT 12%",
                  font=font(12, mono=True), fill=(189, 184, 163))
        frames.append(canvas.quantize(colors=192, method=Image.Quantize.MEDIANCUT,
                                      dither=Image.Dither.FLOYDSTEINBERG))

    destination = ASSET_DIR / "dry-leaves-relight.gif"
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=90,
                   loop=0, disposal=2, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, help="multi-object reconstruction output directory")
    args = parser.parse_args()
    out = args.out.resolve()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    atlas_overview(out)
    roughness_comparison(out)
    relight(out)


if __name__ == "__main__":
    main()
