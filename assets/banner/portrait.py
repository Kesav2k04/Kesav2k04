#!/usr/bin/env python3
"""
Portrait pipeline for the profile banner.

Photo -> head-and-shoulders crop -> background segmentation -> tone prep ->
1-bit Floyd-Steinberg dither (serpentine) -> horizontal run-length dots.

The dithered grids are cached to portrait.npz so re-running the banner
generator is instant. Delete the cache (or pass force=True) after changing
any tuning constant below.

Debug renders:  python assets/banner/portrait.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PHOTO = HERE / "source-photo.jpg"
CACHE = HERE / "portrait.npz"
DEBUG_DIR = HERE / "_debug"

# --- tuning ---------------------------------------------------------------
# Crop box in source-photo pixels: head + shoulders, NOT a tight face crop
# (over-zoomed reads aggressive). Source photo is 930 x 982.
CROP = (205, 168, 765, 814)          # left, top, right, bottom

GRID_W, GRID_H = 260, 300            # dot lattice; coarser = less 1080p moire
CONTRAST = 1.30                      # 1.3x only -- 2.4x reads harsh and skull-like
UNSHARP = dict(radius=3, percent=140, threshold=3)
AUTOCONTRAST_CUTOFF = 1

# The shirt is a high-contrast plaid: left alone it out-shouts the face, which
# is the one thing the portrait exists to show. Fading the lower frame toward
# the panel colour dissolves the torso instead of cropping it, and drops total
# ink by roughly a third.
FADE_START, FADE_END, FADE_GAMMA = 0.56, 1.00, 1.25

GRABCUT_ITERS = 8
WORK_W = 360                         # segmentation working width
MASK_CLOSE = 5                       # binary closing radius, in mask pixels
MASK_ERODE = 1                       # hard-clears error-diffusion bleed at the edge

# Seeds for GrabCut, as fractions of the working image.
#
# A plain rect prior fails here: the backdrop is brick plus a shop window, not
# the flat wall the method assumes, so it labels wall as subject. It also needs
# a real background region to build a colour model from -- everything outside
# the generous silhouette below starts as probable background, and only the
# tight core is asserted as definite subject.
SILHOUETTE_HEAD = (0.51, 0.28, 0.23, 0.31)     # cx, cy, rx, ry -- probable fg
SILHOUETTE_TORSO = [(0.30, 0.46), (0.72, 0.46), (1.00, 0.86),
                    (1.00, 1.00), (0.00, 1.00), (0.00, 0.84)]

CORE_HEAD = (0.51, 0.29, 0.13, 0.20)           # definite fg -- face + hair
CORE_BOXES = [(0.44, 0.47, 0.59, 0.70),        # neck + collar
              (0.32, 0.78, 0.70, 0.99)]        # chest

SEED_BG_BOXES = [(0.00, 0.00, 0.17, 0.34),     # top-left  wall
                 (0.83, 0.00, 1.00, 0.34),     # top-right window
                 (0.00, 0.00, 1.00, 0.02),     # border ring
                 (0.00, 0.00, 0.02, 0.60),
                 (0.98, 0.00, 1.00, 0.60)]


def _load_crop() -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(PHOTO)).convert("RGB")
    return im.crop(CROP)


def _segment(crop: Image.Image) -> np.ndarray:
    """Boolean subject mask at GRID resolution.

    GrabCut with a rect prior, then close / fill / keep-largest-component.
    A busy background (this photo has brick and a shop window) is exactly the
    case the flat-backdrop advice warns about, so the mask is cleaned hard
    rather than trusted raw.
    """
    work = crop.resize((WORK_W, int(WORK_W * crop.height / crop.width)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.array(work), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]

    def ellipse(spec):
        cx, cy, rx, ry = spec
        return (((xx - cx * w) / (rx * w)) ** 2
                + ((yy - cy * h) / (ry * h)) ** 2) <= 1.0

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    poly = np.array([[int(x * w), int(y * h)] for x, y in SILHOUETTE_TORSO], np.int32)
    silhouette = np.zeros((h, w), np.uint8)
    cv2.fillPoly(silhouette, [poly], 1)
    mask[(silhouette > 0) | ellipse(SILHOUETTE_HEAD)] = cv2.GC_PR_FGD

    mask[ellipse(CORE_HEAD)] = cv2.GC_FGD
    for x0, y0, x1, y1 in CORE_BOXES:
        mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = cv2.GC_FGD

    for x0, y0, x1, y1 in SEED_BG_BOXES:
        mask[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1),
             int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)] = cv2.GC_BGD

    cv2.grabCut(bgr, mask, None, np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64), GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)

    fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    fg = ndimage.binary_closing(fg, ndimage.generate_binary_structure(2, 2),
                                iterations=MASK_CLOSE)
    fg = ndimage.binary_fill_holes(fg)

    lab, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = lab == (int(np.argmax(sizes)) + 1)

    small = Image.fromarray((fg * 255).astype(np.uint8)).resize(
        (GRID_W, GRID_H), Image.LANCZOS)
    return np.array(small) > 127


def _tone(img: Image.Image, fade_to: float) -> np.ndarray:
    g = img.convert("L").resize((GRID_W, GRID_H), Image.LANCZOS)
    g = ImageOps.autocontrast(g, cutoff=AUTOCONTRAST_CUTOFF)
    g = ImageEnhance.Contrast(g).enhance(CONTRAST)
    g = g.filter(ImageFilter.UnsharpMask(**UNSHARP))
    a = np.asarray(g, dtype=np.float64)

    ys = np.arange(GRID_H) / (GRID_H - 1)
    ramp = np.clip((ys - FADE_START) / (FADE_END - FADE_START), 0, 1) ** FADE_GAMMA
    return a * (1 - ramp[:, None]) + fade_to * ramp[:, None]


def _dither(gray: np.ndarray) -> np.ndarray:
    """1-bit Floyd-Steinberg, serpentine scan order. 1 = light, 0 = dark."""
    a = gray.copy()
    h, w = a.shape
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        left_to_right = (y % 2 == 0)
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        d = 1 if left_to_right else -1
        for x in xs:
            old = a[y, x]
            new = 255.0 if old >= 128.0 else 0.0
            out[y, x] = 1 if new > 0 else 0
            err = old - new
            if 0 <= x + d < w:
                a[y, x + d] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    a[y + 1, x - d] += err * 3 / 16
                a[y + 1, x] += err * 5 / 16
                if 0 <= x + d < w:
                    a[y + 1, x + d] += err * 1 / 16
    return out


def build(force: bool = False) -> dict[str, np.ndarray]:
    """Return {'dark': ink, 'light': ink, 'mask': mask} boolean grids."""
    if CACHE.exists() and not force:
        z = np.load(CACHE)
        return {k: z[k] for k in ("dark", "light", "mask")}

    crop = _load_crop()
    mask = _segment(crop)

    # Dark mode: composite the subject on black first, so error diffusion never
    # carries subject tone out into the panel. Dots then draw the lit subject.
    rgb = np.asarray(crop.convert("RGB"))
    big_mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        crop.size, Image.LANCZOS)) > 127
    cut = Image.fromarray(np.where(big_mask[..., None], rgb, 0).astype(np.uint8))

    keep = ndimage.binary_erosion(mask, iterations=MASK_ERODE)

    ink_dark = _dither(_tone(cut, 0.0)).astype(bool) & keep

    # Light mode: dots draw the dark parts. The usual advice is to keep the
    # background in light mode, but that only works off a flat backdrop -- here
    # it renders brick and shop shelving in full detail and buries the face, so
    # the same subject mask is used for both themes.
    on_white = Image.fromarray(np.where(big_mask[..., None], rgb, 255).astype(np.uint8))
    ink_light = (~_dither(_tone(on_white, 255.0)).astype(bool)) & keep

    np.savez_compressed(CACHE, dark=ink_dark, light=ink_light, mask=mask)
    return {"dark": ink_dark, "light": ink_light, "mask": mask}


def runs(ink: np.ndarray) -> list[tuple[int, int, int]]:
    """Horizontal run-length encode: list of (x0, y, length_in_cells)."""
    out = []
    h, w = ink.shape
    for y in range(h):
        row = ink[y]
        if not row.any():
            continue
        d = np.diff(row.astype(np.int8))
        starts = list(np.flatnonzero(d == 1) + 1)
        ends = list(np.flatnonzero(d == -1) + 1)
        if row[0]:
            starts.insert(0, 0)
        if row[-1]:
            ends.append(w)
        for a, b in zip(starts, ends):
            out.append((int(a), y, int(b - a)))
    return out


def _debug():
    DEBUG_DIR.mkdir(exist_ok=True)
    data = build(force=True)
    crop = _load_crop()
    crop.save(DEBUG_DIR / "1-crop.png")
    Image.fromarray((data["mask"] * 255).astype(np.uint8)).save(DEBUG_DIR / "2-mask.png")

    for name in ("dark", "light"):
        ink = data[name]
        px = np.where(ink, 255, 0).astype(np.uint8) if name == "dark" \
            else np.where(ink, 0, 255).astype(np.uint8)
        img = Image.fromarray(px).resize((GRID_W * 2, GRID_H * 2), Image.NEAREST)
        img.save(DEBUG_DIR / f"3-dither-{name}.png")
        r = runs(ink)
        print(f"{name:5s}  ink {ink.sum():6d} cells  {len(r):6d} runs  "
              f"avg run {ink.sum()/max(1,len(r)):.2f}  coverage {ink.mean()*100:.1f}%")


if __name__ == "__main__":
    _debug()
