#!/usr/bin/env python3
"""
Generator for the animated terminal banner on github.com/Kesav2k04.

Outputs (repo-relative):
    assets/banner-dark.svg     1180x560  dark-mode header
    assets/banner-light.svg    1180x560  light-mode header
    assets/footer-dark.svg     1180x120  dark-mode footer
    assets/footer-light.svg    1180x120  light-mode footer

This script plus portrait.py and portrait.npz are the source of truth. Never
hand-edit the SVGs -- change the data below and re-run:

    python assets/banner/generate_banner.py

Design notes
------------
* Left panel is a 1-bit dithered portrait (see portrait.py) drawn as horizontal
  <path> runs with shape-rendering="crispEdges" -- never font glyphs, they mush
  below ~2px.
* Portrait quality needs ~17k dots; living per-dot motion needs ~1k. Those are
  incompatible in one layer, so there are two: a dense portrait that dissolves
  in drift bands, and a sparse traveller swarm that morphs between three
  glyphs. Travellers stay hidden during the portrait phase, otherwise their
  thicker dots crowd the fine dither.
* Drift is a linear function of position, so quantising it into bands would
  mathematically recreate a square grid and the dissolve would look blocky.
  Per-dot noise (sigma 4) is added before grouping; see DRIFT_NOISE.
* The intro reveal needs its own grouping (60 interleaved random groups, not
  spatial regions, or it reveals patch-by-patch instead of shimmering in).
  That is why the portrait is emitted twice.
* Right panel rows lock their width with textLength +
  lengthAdjust="spacingAndGlyphs", so values stay right-aligned in any font the
  visitor happens to have. Leader dots are computed, never hand-placed.
* No links inside the SVG -- GitHub strips them. Clickable things live in the
  README as shields.io badges.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portrait as pt                                    # noqa: E402

SEED = 20260728
N_TRAVELLERS = 800
N_INTRO_GROUPS = 60
N_DRIFT_BANDS = 94
DRIFT_FRACTION = 0.42        # how far a band travels toward the first glyph
DRIFT_NOISE = 4.0            # sigma, in banner px -- breaks the grid trap

# loop timeline, seconds
T_PORTRAIT, T_GLYPH, T_TRANS = 3.0, 2.0, 1.3
INTRO_DUR, INTRO_STAGGER, INTRO_FADE = 0.6, 1.98, 0.15

W, H = 1180, 560
MONO = "JetBrains Mono,Fira Code,SFMono-Regular,ui-monospace,Consolas,Liberation Mono,monospace"
ADV = 0.60  # monospace advance width as a fraction of font-size

OUT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# palettes -- portrait hue must differ from the UI chrome, or the face blends
# into its own frame
# --------------------------------------------------------------------------

DARK = dict(
    name="dark",
    bg="#0D1117", panel="#0F1626", border="#1F2A3C",
    label="#70A5FD", dim="#5A6785", value="#C9D1D9", leader="#4A5878",
    accent="#38BDAE", dots="#BF91F3", ambient="#2A3550", live="#F7768E",
    pill_bg="#BF91F3", pill_fg="#0D1117",
    chrome=("#F7768E", "#E0AF68", "#9ECE6A"),
)

LIGHT = dict(
    name="light",
    bg="#FFFFFF", panel="#F6F8FA", border="#D8DEE6",
    label="#2563EB", dim="#6B7280", value="#1F2328", leader="#B6C0CE",
    accent="#0F766E", dots="#7C3AED", ambient="#CBD5E1", live="#DC2626",
    pill_bg="#7C3AED", pill_fg="#FFFFFF",
    chrome=("#FF5F57", "#FEBC2E", "#28C840"),
)


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------

TITLE = "kesav@research:~$  ./profile.sh --live"

# (label, value, extra_gap_before)
ROWS = [
    ("Subject",       "Kesav Kumar Jayakumar",                             False),
    ("Role",          "AI Researcher  ·  Cloud-Edge Systems",              False),
    ("Base",          "India  ·  UTC+05:30",                               False),
    ("Education",     "B.Tech Information Technology  ·  CGPA 8.41 / 10",  False),
    ("Focus",         "VLM evaluation  ·  distributed GPU execution",      False),
    ("Status",        "Open to research supervision  ·  2027 intake",      False),
    ("Papers",        "ICCET 2026 published  ·  NeurIPS WS in review",     True),
    ("Artifacts",     "PyPI  ·  HF dataset  ·  2 Zenodo DOIs",             False),
    ("Upstream",      "LangChain  ·  EleutherAI  ·  HF  ·  Future-AGI",    False),
    ("Core.Lang",     "Python  ·  Java  ·  C++  ·  Go  ·  Kotlin  ·  TS",  True),
    ("Core.ML",       "PyTorch  ·  Transformers  ·  TensorRT  ·  QLoRA",   False),
    ("Core.Infra",    "AWS  ·  GCP  ·  Docker  ·  Actions  ·  Tailscale",  False),
    ("Grid.Mail",     "kesavk659@gmail.com",                               True),
    ("Grid.ORCID",    "0009-0006-9909-7851",                               False),
    ("Grid.LinkedIn", "in/kesav-kumar-j-5414aa253",                        False),
    ("Grid.Live",     "visualpc.vercel.app",                               False),
]

PILL = "@Kesav2k04"
MOTTO = "Measure first  ·  Build right  ·  Ship with evidence"

# geometry
BAR_H = 40
PANEL_T, PANEL_B = 56, 536
LP_X, LP_W = 24, 446
RP_X, RP_R = 486, 1156
PAD = 20

FIELD = (LP_X + PAD, 104, LP_X + LP_W - PAD, 492)   # x0, y0, x1, y1

FS_ROW = 14
ROW_TOP, ROW_STEP, ROW_GAP = 106, 23, 8


def tw(text: str, fs: float) -> float:
    """Rendered width of monospace text -- locked in by textLength."""
    return len(text) * fs * ADV


# --------------------------------------------------------------------------
# traveller glyphs (all designed in a 130 x 100 space)
# --------------------------------------------------------------------------

def _seg(p0, p1, n, thick, rng):
    """Dots along a segment: evenly spaced lengthwise, scattered across width.

    Even spacing matters -- random t leaves clumps and gaps, which is what makes
    a thin stroke read as noise instead of a line.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    if L == 0 or n <= 0:
        return np.empty((0, 2))
    u = d / L
    nrm = np.array([-u[1], u[0]])
    t = ((np.arange(n) + 0.5) / n)[:, None]
    t = t + (rng.random((n, 1)) - 0.5) * (0.55 / n)
    off = (rng.normal(0, thick * 0.30, (n, 1)) if thick <= 2.2
           else (rng.random((n, 1)) - 0.5) * thick)
    return p0 + t * d + off * nrm


def _disc(c, r, n, rng):
    if n <= 0:
        return np.empty((0, 2))
    rr = r * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * math.pi
    return np.stack([c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)], axis=1)


def _many(segs, n, thick, rng):
    lens = np.array([np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs], float)
    share = np.maximum(1, np.round(lens / lens.sum() * n).astype(int))
    return np.vstack([_seg(a, b, k, thick, rng) for (a, b), k in zip(segs, share)])


def _rect_outline(x0, y0, x1, y1, n, thick, rng):
    return _many([((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                  ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))], n, thick, rng)


def _fit(pts, n, rng):
    """Force a cloud to exactly n points (drop extras / duplicate with jitter)."""
    if len(pts) > n:
        return pts[rng.permutation(len(pts))[:n]]
    while len(pts) < n:
        idx = rng.permutation(len(pts))[:n - len(pts)]
        pts = np.vstack([pts, pts[idx] + (rng.random((len(idx), 2)) - 0.5) * 0.8])
    return pts


def shape_code(n, rng):
    """The </> glyph -- what I write."""
    segs = [((46, 22), (14, 50)), ((14, 50), (46, 78)),      # <
            ((58, 84), (78, 16)),                            # /
            ((90, 22), (122, 50)), ((122, 50), (90, 78))]    # >
    return _fit(_many(segs, n, 6.8, rng), n, rng)


def shape_graph(n, rng):
    """A three-layer network -- what I evaluate.

    Fully-connected layers turn into visual mush at this dot budget, so only
    near-neighbour edges are drawn; it still reads as a network.
    """
    layers = [[(x, y) for y in np.linspace(14, 86, k)]
              for x, k in zip((18.0, 65.0, 112.0), (4, 5, 3))]
    wiring = [(0, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]),
              (1, [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (4, 2)])]
    edges = [(layers[i][a], layers[i + 1][b]) for i, prs in wiring for a, b in prs]

    n_edge = int(n * 0.46)
    pts = [_many(edges, n_edge, 1.4, rng)]
    per = max(1, (n - n_edge) // sum(len(l) for l in layers))
    for layer in layers:
        for c in layer:
            pts.append(_disc(c, 5.0, per, rng))
    return _fit(np.vstack(pts), n, rng)


def shape_die(n, rng):
    """An accelerator die -- what I run it on."""
    pts = [_rect_outline(24, 14, 106, 86, int(n * 0.36), 3.8, rng),
           _rect_outline(42, 32, 88, 68, int(n * 0.20), 2.8, rng)]
    core = [(gx, gy) for gx in np.linspace(51, 79, 5) for gy in np.linspace(40, 60, 4)]
    pts.append(np.vstack([_disc(c, 1.4, 3, rng) for c in core]))

    pins = []
    for x in np.linspace(34, 96, 5):
        pins += [((x, 14), (x, 3)), ((x, 86), (x, 97))]
    for y in np.linspace(28, 72, 4):
        pins += [((24, y), (12, y)), ((106, y), (118, y))]
    pts.append(_many(pins, int(n * 0.26), 2.6, rng))
    return _fit(np.vstack(pts), n, rng)


def _match(a, b):
    """Optimal transport so each dot takes the shortest path to its target."""
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return b[linear_sum_assignment(cost)[1]]


def build_glyphs():
    rng = np.random.default_rng(SEED)
    A = shape_code(N_TRAVELLERS, rng)
    B = _match(A, shape_graph(N_TRAVELLERS, rng))
    C = _match(B, shape_die(N_TRAVELLERS, rng))

    fx0, fy0, fx1, fy1 = FIELD
    s = min((fx1 - fx0) / 130.0, (fy1 - fy0) / 100.0)
    ox = fx0 + ((fx1 - fx0) - 130 * s) / 2
    oy = fy0 + ((fy1 - fy0) - 100 * s) / 2
    place = lambda p: np.stack([ox + p[:, 0] * s, oy + p[:, 1] * s], axis=1)
    return place(A), place(B), place(C)


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------

LOOP = T_PORTRAIT + 4 * T_TRANS + 3 * T_GLYPH      # 14.2s
INTRO_END = INTRO_STAGGER + INTRO_DUR + INTRO_FADE

_stops = [0.0, T_PORTRAIT]
for i in range(3):
    _stops += [_stops[-1] + T_TRANS, _stops[-1] + T_TRANS + T_GLYPH]
_stops += [LOOP]
KT = [round(t / LOOP, 4) for t in _stops]
# KT = portrait-out, g1-in, g1-out, g2-in, g2-out, g3-in, g3-out, portrait-back
KT_PORTRAIT = f"{KT[0]};{KT[1]};{KT[2]};{KT[7]};{KT[8]}"
KT_TRAVEL = ";".join(str(k) for k in KT[:9])


# --------------------------------------------------------------------------
# SVG assembly
# --------------------------------------------------------------------------

def text(x, y, s, fs, fill, *, anchor="start", weight=400, lock=True, spacing=None):
    attrs = [f'x="{x:g}"', f'y="{y:g}"', f'font-family="{MONO}"',
             f'font-size="{fs:g}"', f'font-weight="{weight}"', f'fill="{fill}"']
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if lock:
        attrs.append(f'textLength="{tw(s, fs):.1f}" lengthAdjust="spacingAndGlyphs"')
    if spacing is not None:
        attrs.append(f'letter-spacing="{spacing}"')
    return f'<text {" ".join(attrs)}>{escape(s)}</text>'


def portrait_geometry():
    fx0, fy0, fx1, fy1 = FIELD
    cell = min((fx1 - fx0) / pt.GRID_W, (fy1 - fy0) / pt.GRID_H)
    ox = fx0 + ((fx1 - fx0) - pt.GRID_W * cell) / 2
    oy = fy0 + ((fy1 - fy0) - pt.GRID_H * cell) / 2
    return cell, ox, oy


def run_path(run, cell, ox, oy):
    x0, y, length = run
    x = ox + (x0 + 0.5) * cell
    yy = oy + (y + 0.5) * cell
    return f"M{x:.2f} {yy:.2f}h{(length - 1) * cell:.2f}"


def left_panel(p, runs, glyphs):
    """Portrait (intro layer + drift-band loop layer) plus traveller swarm."""
    A, B, C = glyphs
    cell, ox, oy = portrait_geometry()
    out = []

    out.append(
        f'<rect x="{LP_X}" y="{PANEL_T}" width="{LP_W}" height="{PANEL_B-PANEL_T}" '
        f'rx="10" fill="{p["panel"]}" stroke="{p["border"]}"/>'
    )
    out.append(text(LP_X + PAD, PANEL_T + 24, "VISUAL.MAP", 12, p["accent"],
                    weight=700, lock=False, spacing="2.2"))

    # ambient texture behind the portrait
    arng = random.Random(SEED + 7)
    fx0, fy0, fx1, fy1 = FIELD
    out.append(f'<g fill="{p["ambient"]}" opacity="0.5">'
               f'<animate attributeName="opacity" values="0.25;0.55;0.25" dur="7s" '
               f'repeatCount="indefinite"/>')
    for _ in range(130):
        out.append(f'<circle cx="{arng.uniform(fx0-4, fx1+4):.1f}" '
                   f'cy="{arng.uniform(fy0-4, fy1+4):.1f}" '
                   f'r="{arng.uniform(0.7, 1.4):.2f}"/>')
    out.append('</g>')

    stroke = (f'stroke="{p["dots"]}" stroke-width="{cell:.3f}" fill="none" '
              f'stroke-linecap="square" shape-rendering="crispEdges"')

    # ---- intro layer: 60 interleaved random groups -------------------------
    order = list(range(len(runs)))
    random.Random(SEED + 13).shuffle(order)
    out.append(
        f'<g {stroke}><animate attributeName="opacity" from="1" to="0" '
        f'begin="{INTRO_STAGGER + INTRO_DUR:.2f}s" dur="{INTRO_FADE}s" fill="freeze"/>'
    )
    for gi in range(N_INTRO_GROUPS):
        members = order[gi::N_INTRO_GROUPS]
        if not members:
            continue
        d = "".join(run_path(runs[i], cell, ox, oy) for i in members)
        begin = round(gi * INTRO_STAGGER / N_INTRO_GROUPS, 3)
        out.append(
            f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" '
            f'begin="{begin}s" dur="{INTRO_DUR}s" fill="freeze"/>'
            f'<path d="{d}"/></g>'
        )
    out.append('</g>')

    # ---- loop layer: 94 drift bands ---------------------------------------
    centres = np.array([[ox + (r[0] + (r[2] - 1) / 2 + 0.5) * cell,
                         oy + (r[1] + 0.5) * cell] for r in runs])
    target = A.mean(axis=0)
    drift = (target - centres) * DRIFT_FRACTION

    nrng = np.random.default_rng(SEED + 29)
    score = np.hypot(*(centres - target).T) + nrng.normal(0, DRIFT_NOISE, len(runs))
    band = np.argsort(np.argsort(score)) * N_DRIFT_BANDS // len(runs)

    out.append(
        f'<g {stroke} opacity="0"><animate attributeName="opacity" '
        f'values="1;1;0;0;1" keyTimes="{KT_PORTRAIT}" begin="{INTRO_END:.2f}s" '
        f'dur="{LOOP}s" repeatCount="indefinite"/>'
    )
    for b in range(N_DRIFT_BANDS):
        idx = np.flatnonzero(band == b)
        if not len(idx):
            continue
        d = "".join(run_path(runs[i], cell, ox, oy) for i in idx)
        dx, dy = drift[idx].mean(axis=0)
        out.append(
            f'<g><animateTransform attributeName="transform" type="translate" '
            f'values="0 0;0 0;{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};0 0" '
            f'keyTimes="{KT_PORTRAIT}" begin="{INTRO_END:.2f}s" dur="{LOOP}s" '
            f'calcMode="spline" keySplines=".4 0 .2 1;.4 0 .2 1;0 0 1 1;.4 0 .2 1" '
            f'repeatCount="indefinite"/><path d="{d}"/></g>'
        )
    out.append('</g>')

    # ---- traveller swarm ---------------------------------------------------
    out.append(
        f'<g fill="{p["dots"]}" opacity="0"><animate attributeName="opacity" '
        f'values="0;0;1;1;1;1;1;1;0" keyTimes="{KT_TRAVEL}" '
        f'begin="{INTRO_END:.2f}s" dur="{LOOP}s" repeatCount="indefinite"/>'
    )
    for i in range(N_TRAVELLERS):
        a, b, c = A[i], B[i], C[i]
        vals = ";".join(f"{q[0]:.1f} {q[1]:.1f}"
                        for q in (a, a, a, a, b, b, c, c, a))
        out.append(
            f'<circle r="1.9"><animateTransform attributeName="transform" '
            f'type="translate" values="{vals}" keyTimes="{KT_TRAVEL}" '
            f'begin="{INTRO_END:.2f}s" dur="{LOOP}s" calcMode="spline" '
            f'keySplines="0 0 1 1;0 0 1 1;0 0 1 1;.5 0 .2 1;0 0 1 1;.5 0 .2 1;'
            f'0 0 1 1;.5 0 .2 1" repeatCount="indefinite"/></circle>'
        )
    out.append('</g>')

    caption = f"1-bit dither  ·  {pt.GRID_W} × {pt.GRID_H} grid  ·  {sum(r[2] for r in runs):,} dots"
    out.append(text(LP_X + PAD, PANEL_B - 15, caption, 11.5, p["dim"], lock=False))
    return out


def right_panel(p):
    out = [
        f'<rect x="{RP_X}" y="{PANEL_T}" width="{RP_R-RP_X}" '
        f'height="{PANEL_B-PANEL_T}" rx="10" fill="{p["panel"]}" '
        f'stroke="{p["border"]}"/>'
    ]
    ix0, ix1 = RP_X + PAD, RP_R - PAD
    out.append(text(ix0, PANEL_T + 24, "SYSTEM.INFO", 13, p["label"],
                    weight=700, lock=False, spacing="2.2"))

    bw, bx = 62, ix1 - 62
    out.append(f'<rect x="{bx}" y="{PANEL_T+10}" width="{bw}" height="20" rx="10" '
               f'fill="none" stroke="{p["live"]}" opacity="0.55"/>')
    out.append(f'<circle cx="{bx+13}" cy="{PANEL_T+20}" r="3.6" fill="{p["live"]}">'
               f'<animate attributeName="opacity" values="1;0.2;1" dur="1.6s" '
               f'repeatCount="indefinite"/></circle>')
    out.append(text(bx + 23, PANEL_T + 24, "LIVE", 12, p["live"], lock=False,
                    spacing="1.4"))
    out.append(f'<line x1="{ix0}" y1="{PANEL_T+38}" x2="{ix1}" y2="{PANEL_T+38}" '
               f'stroke="{p["border"]}" stroke-dasharray="2 4"/>')

    y = ROW_TOP
    for label, value, gap in ROWS:
        if gap:
            y += ROW_GAP
        out.append(text(ix0, y, label, FS_ROW, p["label"]))
        out.append(text(ix1, y, value, FS_ROW, p["value"], anchor="end"))
        lx0 = ix0 + tw(label, FS_ROW) + 8
        lx1 = ix1 - tw(value, FS_ROW) - 8
        if lx1 - lx0 > 6:
            out.append(f'<line x1="{lx0:.1f}" y1="{y-4}" x2="{lx1:.1f}" y2="{y-4}" '
                       f'stroke="{p["leader"]}" stroke-width="1" '
                       f'stroke-dasharray="1.5 4.5" stroke-linecap="round"/>')
        y += ROW_STEP

    py = PANEL_B - 42
    pw = tw(PILL, 14) + 26
    out.append(f'<rect x="{ix0}" y="{py}" width="{pw:.1f}" height="26" rx="13" '
               f'fill="{p["pill_bg"]}"/>')
    out.append(text(ix0 + 13, py + 18, PILL, 14, p["pill_fg"], weight=700))
    out.append(text(ix1, py + 18, MOTTO, 12.5, p["dim"], anchor="end"))
    return out


def banner(p, runs, glyphs):
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Kesav Kumar Jayakumar -- dot-matrix portrait beside a '
        f'terminal readout of his research profile">',
        f'<rect width="{W}" height="{H}" rx="14" fill="{p["bg"]}"/>',
        f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="13.5" '
        f'fill="none" stroke="{p["border"]}"/>',
    ]
    for i, c in enumerate(p["chrome"]):
        out.append(f'<circle cx="{26+i*22}" cy="20" r="5.5" fill="{c}" opacity="0.9"/>')
    out.append(text(100, 25, TITLE, 13, p["dim"]))
    cx = 100 + tw(TITLE, 13) + 6
    out.append(f'<rect x="{cx:.1f}" y="15" width="7" height="13" fill="{p["accent"]}">'
               f'<animate attributeName="opacity" values="1;1;0;0" dur="1.1s" '
               f'repeatCount="indefinite"/></rect>')
    out.append(f'<line x1="0" y1="{BAR_H}" x2="{W}" y2="{BAR_H}" '
               f'stroke="{p["border"]}"/>')

    out += left_panel(p, runs, glyphs)
    out += right_panel(p)
    out.append('</svg>')
    return "\n".join(out)


def footer(p):
    fh = 120
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{fh}" '
           f'viewBox="0 0 {W} {fh}" role="img" aria-label="profile footer">',
           f'<rect width="{W}" height="{fh}" rx="14" fill="{p["panel"]}" '
           f'stroke="{p["border"]}"/>']

    rng = random.Random(SEED + 31)
    n = 78
    for i in range(n):
        x = 40 + i * ((W - 80) / (n - 1))
        base = 46 + 14 * math.sin(i / 5.2)
        r = 2.4 + 1.3 * abs(math.sin(i / 3.1))
        delay = round(i * 0.045, 3)
        col = p["dots"] if rng.random() > 0.35 else p["accent"]
        out.append(
            f'<circle cx="{x:.1f}" cy="{base:.1f}" r="{r:.2f}" fill="{col}" '
            f'opacity="0.35">'
            f'<animate attributeName="opacity" values="0.2;0.95;0.2" dur="3.4s" '
            f'begin="{delay}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{base:.1f};{base-7:.1f};{base:.1f}" '
            f'dur="3.4s" begin="{delay}s" repeatCount="indefinite"/></circle>'
        )

    out.append(f'<line x1="40" y1="78" x2="{W-40}" y2="78" stroke="{p["border"]}" '
               f'stroke-dasharray="2 4"/>')
    left = "kesav@research:~$  exit  —  thanks for reading"
    out.append(text(40, 102, left, 13, p["dim"]))
    cx = 40 + tw(left, 13) + 6
    out.append(f'<rect x="{cx:.1f}" y="92" width="7" height="13" fill="{p["accent"]}">'
               f'<animate attributeName="opacity" values="1;1;0;0" dur="1.1s" '
               f'repeatCount="indefinite"/></rect>')
    out.append(text(W - 40, 102, MOTTO, 13, p["label"], anchor="end"))
    out.append('</svg>')
    return "\n".join(out)


def main():
    ink = pt.build()
    glyphs = build_glyphs()
    print(f"loop {LOOP:.1f}s  intro {INTRO_END:.2f}s  keyTimes {KT_PORTRAIT}")
    for p in (DARK, LIGHT):
        runs = pt.runs(ink[p["name"]])
        for kind, svg in (("banner", banner(p, runs, glyphs)),
                          ("footer", footer(p))):
            path = OUT / f'{kind}-{p["name"]}.svg'
            path.write_text(svg, encoding="utf-8")
            print(f'{path.name:20s} {len(svg.encode("utf-8"))/1024:8.1f} KB'
                  f'{"  " + str(len(runs)) + " runs" if kind == "banner" else ""}')


if __name__ == "__main__":
    main()
