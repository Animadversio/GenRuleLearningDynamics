"""
flower_rule_lib.py

Naturalistic (image-domain) version of the discrete rule-learning benchmarks.

Core idea
---------
A flower image is a *polar rendering of an (R, C) symbol grid*:

    ring r  (whorl, radial band)      ->  row r    of the latent grid
    slot c  (angular petal position)  ->  column c of the latent grid
    symbol s[r, c] in [0, n_symbols)  ->  a visual attribute of that petal

Therefore every rule already studied on flat/grid latents transfers verbatim:

    latent grid  --render-->  RGB image  --DiT trains here--  sample
    sample       --decode-->  latent grid  --rule check-->  valid?

Rules supported (`rule=` argument), ordered by score-function complexity, same
spectrum as the synthetic benchmark:

    exact_k   : exactly K of the D = R*C petals carry symbol 1     (degree 1)
    row_perm  : every ring is a permutation of the n_symbols colors
                (all-distinct per ring; needs n_slots == n_symbols)  (intermediate)
    row_k     : every ring has exactly K petals with symbol 1      (degree 1/ring)
    latin_sq  : R == C == n_symbols; each symbol once per ring and
                once per radial spoke                              (intermediate)
    parity    : contiguous groups of G petals have an even number
                of symbol-1 petals                                 (degree G)

Visual channels
---------------
Independent attributes on the same petals, each carrying its own rule:

    "color" : petal fill hue           (n_symbols distinguishable palette entries)
    "spot"  : dark blotch near the petal tip present / absent   (binary)
    "shape" : rounded vs. pointed petal outline                 (binary)

Multiple channels multiply the latent support size, which is what makes a
2048-sample dataset possible on a 12-petal flower at low G.

Nuisance variation (the "naturalistic" part; none of it is rule-relevant)
------------------------------------------------------------------------
global rotation, global scale, center offset, per-petal angular / length /
width jitter, per-petal color jitter, radial shading, background tint,
sensor noise, anti-aliasing via supersampled rendering.

Rotation handling
-----------------
`rot_mode` controls the identifiability of the petal *index origin*, which
matters because parity groups are contiguous blocks of petals:

    "fixed"  : petal (0,0) always points up
    "jitter" : global rotation within +-`rot_jit` * (half petal spacing), so
               nearest-slot rounding still recovers the true index  (default)
    "free"   : arbitrary global rotation; the origin is recovered from the
               stem, which rotates with the flower (requires draw_stem=True)

Typical usage
-------------
    from genrule.datasets.images.flower import (FlowerConfig, PRESETS, sample_latents,
                                      render_batch, decode_batch, check_latents)

    cfg    = PRESETS["daisy12"]
    latent = sample_latents("parity", 2048, cfg, group_size=4, seed=0)
    imgs   = render_batch(latent, cfg, seed=0)          # (N, S, S, 3) uint8
    rec, ok = decode_batch(imgs, cfg)                   # (N, R, C) int, (N,) bool
    valid  = check_latents(rec, "parity", cfg, group_size=4)
"""

import os
import sys
from dataclasses import dataclass, field, replace, asdict
from math import comb, factorial, pi

import cv2
import numpy as np
from .colors import BG_COLOR, CENTER_COLOR, CENTER_DOT_COLOR, STEM_COLOR, PETAL_PALETTE, _to_lab, palette_separation
from ..latents import sample_latents, check_latents, rule_subunit_ok, support_size, corrupt_latents



# ---------------------------------------------------------------------------
# Palette  (all colors are RGB uint8 triples)
# ---------------------------------------------------------------------------


#: Petal fill colors, chosen to be far apart in CIELAB and far from
#: background / disk / stem.  `n_symbols` uses the first n entries.

CHANNELS = ("color", "spot", "shape")






# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FlowerConfig:
    """Geometry, appearance and nuisance settings for the flower renderer.

    All lengths are fractions of `img_size` unless stated otherwise.
    """
    # ── latent shape ──────────────────────────────────────────────────────
    n_rings:   int = 1      # R — radial whorls
    n_slots:   int = 12     # C — angular petal positions per whorl
    n_symbols: int = 2      # alphabet size of the latent grid
    channels: tuple = ("color",)

    # ── canvas ────────────────────────────────────────────────────────────
    img_size:    int = 64
    supersample: int = 4    # render at img_size*SS, then INTER_AREA downsample

    # ── geometry ──────────────────────────────────────────────────────────
    center_r:    float = 0.135   # disk-floret semi-axis
    ring_inner:  float = 0.20    # inner edge of the innermost petal band
    ring_outer:  float = 0.49    # outer edge of the outermost petal band
    ring_gap:    float = 0.015   # radial gap between adjacent petal bands
    petal_wid_k: float = 0.30    # petal half-width as a fraction of the
                                 # angular arc between neighbouring petals
                                 # (< 0.5 guarantees a background gap; the
                                 # slack must cover the angular jitter below)
    petal_wid_cap: float = 0.75  # ... but never wider than this * semi-major,
                                 # so petals stay radially elongated when many
                                 # rings squeeze the radial band
    alternate_rings: bool = True  # offset odd rings by half a petal spacing

    # "shape" channel: symbol 1 renders a lanceolate (long, narrow, tapered)
    # petal instead of a rounded one.  The length/width ratio between the two
    # classes sets how separable the blob aspect ratio is — 1.25/0.55 = 2.3x,
    # which the decoder resolves with ~3 sigma margin at img_size=64.
    shape_len_k: float = 1.25
    shape_wid_k: float = 0.55

    # ── nuisance ──────────────────────────────────────────────────────────
    rot_mode:   str   = "jitter"  # "fixed" | "jitter" | "free"
    rot_jit:    float = 0.30      # in units of half the petal spacing
    angle_jit:  float = 0.05      # per-petal, in units of petal spacing
    len_jit:    float = 0.08      # per-petal relative
    wid_jit:    float = 0.08      # per-petal relative
    color_jit:  float = 0.045     # per-petal, relative to 255
    shade:      float = 0.20      # radial shading depth on petals
    bg_jit:     float = 0.02      # background tint
    scale_jit:  float = 0.04      # global scale
    center_jit: float = 0.012     # global center offset
    noise:      float = 0.012     # additive sensor noise (sigma, relative)
    draw_stem:  bool  = True
    outline:    float = 0.72      # petal outline = fill * outline.  Must stay
                                  # clearly *lighter* than spot_dark, otherwise
                                  # the decoder cannot tell a petal edge from a
                                  # spot (they are both just dark pixels).
    spot_dark:  float = 0.30      # blotch = fill * spot_dark
    spot_len_k: float = 0.42      # blotch semi-axes, relative to the petal's
    spot_wid_k: float = 0.55      # ... so the dark fraction is shape-invariant

    # ── derived ───────────────────────────────────────────────────────────
    @property
    def n_petals(self):
        """D = R * C — number of petals = number of latent cells."""
        return self.n_rings * self.n_slots

    @property
    def base_angle(self):
        """Angle of petal (0, 0) in the un-rotated frame (image coords, y down)."""
        return -pi / 2

    def ring_bands(self):
        """Radial bands, one per ring.

        Returns
        -------
        list of (rho, a) in fractions of img_size
            rho — band center radius, a — petal semi-major axis
        """
        edges = np.linspace(self.ring_inner, self.ring_outer, self.n_rings + 1)
        bands = []
        for r in range(self.n_rings):
            lo = edges[r] + self.ring_gap / 2
            hi = edges[r + 1] - self.ring_gap / 2
            bands.append(((lo + hi) / 2, (hi - lo) / 2))
        return bands

    def ring_offset(self, r):
        """Angular offset of ring r, in radians."""
        if self.alternate_rings and (r % 2 == 1):
            return pi / self.n_slots
        return 0.0


PRESETS = {
    # single whorl, 12 petals — the plain "center oval + D petals" flower
    "daisy12": FlowerConfig(n_rings=1, n_slots=12, img_size=64),
    # two whorls, 24 petals — support-rich enough for N >= 2048 at low G
    "aster24": FlowerConfig(n_rings=2, n_slots=12, img_size=96),
    # square grid for the Latin-square rule (6 rings x 6 slots, 6 colors)
    "latin6":  FlowerConfig(n_rings=6, n_slots=6, img_size=160, n_symbols=6,
                            ring_inner=0.13, ring_outer=0.50, center_r=0.085),
}


# ---------------------------------------------------------------------------
# Latent sampling / checking / support size
# ---------------------------------------------------------------------------













# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _jit(rng, scale):
    return 1.0 + rng.uniform(-scale, scale)


def _petal_polygon(cx, cy, a, b, ang, pointed):
    """Petal outline as a polygon.

    `pointed=False` gives an ellipse-like rounded petal; `pointed=True` tapers
    the outer end to a tip (visually distinct at low resolution, and separable
    by a solidity / tip-sharpness statistic).
    """
    t = np.linspace(0, 2 * pi, 48, endpoint=False)
    u = np.cos(t) * a
    v = np.sin(t) * b
    if pointed:
        # squeeze the width toward the outer end -> lanceolate petal
        v = v * np.clip(1.0 - 0.85 * np.clip(u / a, 0, 1) ** 1.5, 0.05, 1.0)
    ca, sa = np.cos(ang), np.sin(ang)
    xs = cx + u * ca - v * sa
    ys = cy + u * sa + v * ca
    return np.stack([xs, ys], axis=1).astype(np.int32)


def render_flower(latent, cfg, rng=None, return_meta=False):
    """Render one flower image from a latent grid.

    Parameters
    ----------
    latent : (R, C) int array, or dict {channel: (R, C) int array}
        A bare array is interpreted as the "color" channel.
    cfg    : FlowerConfig
    rng    : np.random.Generator or None
    return_meta : bool — also return ground-truth nuisance parameters
                  (for debugging only; the decoder never sees them)

    Returns
    -------
    img  : (img_size, img_size, 3) uint8, RGB
    meta : dict (only if return_meta)
    """
    rng = np.random.default_rng() if rng is None else rng
    if not isinstance(latent, dict):
        latent = {"color": np.asarray(latent)}
    for ch in latent:
        if ch not in CHANNELS:
            raise ValueError(f"unknown channel {ch!r}; expected one of {CHANNELS}")

    R, C = cfg.n_rings, cfg.n_slots
    SS = cfg.supersample
    S = cfg.img_size * SS

    # ── canvas ────────────────────────────────────────────────────────────
    bg = np.clip(np.array(BG_COLOR, float) * _jit(rng, cfg.bg_jit)
                 + rng.normal(0, 255 * cfg.bg_jit, 3), 0, 255)
    img = np.tile(bg.astype(np.uint8), (S, S, 1))

    # ── global frame ──────────────────────────────────────────────────────
    if cfg.rot_mode == "fixed":
        rot = cfg.base_angle
    elif cfg.rot_mode == "jitter":
        rot = cfg.base_angle + rng.uniform(-1, 1) * cfg.rot_jit * pi / C
    elif cfg.rot_mode == "free":
        if not cfg.draw_stem:
            raise ValueError("rot_mode='free' needs draw_stem=True as an anchor")
        rot = rng.uniform(-pi, pi)
    else:
        raise ValueError(f"unknown rot_mode: {cfg.rot_mode}")

    scale = _jit(rng, cfg.scale_jit)
    cx = S / 2 + rng.uniform(-1, 1) * cfg.center_jit * S
    cy = S / 2 + rng.uniform(-1, 1) * cfg.center_jit * S

    # ── stem + leaf (opposite the petal-0 direction, so it rotates along) ──
    if cfg.draw_stem:
        sa = rot + pi
        tip = (int(cx + np.cos(sa) * S), int(cy + np.sin(sa) * S))
        cv2.line(img, (int(cx), int(cy)), tip, STEM_COLOR,
                 max(1, int(0.022 * S)), cv2.LINE_AA)
        lang = sa + rng.choice([-1, 1]) * rng.uniform(0.5, 0.9)
        lr = 0.42 * S
        lc = (int(cx + np.cos(sa) * lr * 0.75), int(cy + np.sin(sa) * lr * 0.75))
        cv2.ellipse(img, lc, (int(0.10 * S), int(0.045 * S)),
                    np.degrees(lang), 0, 360, STEM_COLOR, -1, cv2.LINE_AA)

    # ── petals: outer rings first so inner whorls occlude their bases ─────
    bands = cfg.ring_bands()
    petal_meta = []
    for r in range(R - 1, -1, -1):
        rho_f, a_f = bands[r]
        rho, a = rho_f * S * scale, a_f * S * scale
        arc = 2 * pi * rho / C
        b = min(cfg.petal_wid_k * arc, cfg.petal_wid_cap * a)
        for c in range(C):
            ang = rot + cfg.ring_offset(r) + 2 * pi * c / C \
                  + rng.uniform(-1, 1) * cfg.angle_jit * (2 * pi / C)
            aa, bb = a * _jit(rng, cfg.len_jit), b * _jit(rng, cfg.wid_jit)
            px, py = cx + np.cos(ang) * rho, cy + np.sin(ang) * rho

            sym = int(latent["color"][r, c]) if "color" in latent else 0
            fill = np.clip(np.array(PETAL_PALETTE[sym], float)
                           + rng.normal(0, 255 * cfg.color_jit, 3), 0, 255)

            pointed = bool(latent["shape"][r, c]) if "shape" in latent else False
            if pointed:
                aa, bb = aa * cfg.shape_len_k, bb * cfg.shape_wid_k
            poly = _petal_polygon(px, py, aa, bb, ang, pointed)
            cv2.fillPoly(img, [poly], tuple(fill.astype(int).tolist()), cv2.LINE_AA)
            cv2.polylines(img, [poly], True,
                          tuple((fill * cfg.outline).astype(int).tolist()),
                          max(1, int(0.006 * S)), cv2.LINE_AA)

            mask = np.zeros((S, S), np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            # radial shading: darken the petal base, brighten the tip
            if cfg.shade > 0:
                ys, xs = np.nonzero(mask)
                if len(xs):
                    proj = ((xs - px) * np.cos(ang) + (ys - py) * np.sin(ang)) / max(aa, 1e-6)
                    f = (1.0 - cfg.shade / 2) + cfg.shade * (proj + 1) / 2
                    img[ys, xs] = np.clip(img[ys, xs] * f[:, None], 0, 255).astype(np.uint8)

            # spot: a basal blotch, sized *relative to the petal* and clipped to
            # it, so the dark-pixel fraction the decoder measures is invariant
            # to the petal's shape class and to the ring radius.
            if "spot" in latent and int(latent["spot"][r, c]) == 1:
                sp = np.zeros((S, S), np.uint8)
                cv2.ellipse(sp,
                            (int(px + np.cos(ang) * 0.18 * aa),
                             int(py + np.sin(ang) * 0.18 * aa)),
                            (max(1, int(cfg.spot_len_k * aa)),
                             max(1, int(cfg.spot_wid_k * bb))),
                            np.degrees(ang), 0, 360, 255, -1, cv2.LINE_AA)
                ys, xs = np.nonzero(sp & mask)
                if len(xs):
                    img[ys, xs] = (img[ys, xs] * cfg.spot_dark).astype(np.uint8)

            petal_meta.append(dict(ring=r, slot=c, angle=ang, rho=rho,
                                   xy=(px, py), sym=sym))

    # ── disk floret on top ────────────────────────────────────────────────
    cr = cfg.center_r * S * scale
    cv2.ellipse(img, (int(cx), int(cy)),
                (int(cr * _jit(rng, 0.08)), int(cr * _jit(rng, 0.08))),
                rng.uniform(0, 180), 0, 360, CENTER_COLOR, -1, cv2.LINE_AA)
    for _ in range(int(9 * cfg.n_rings ** 0.5)):
        t, rr = rng.uniform(0, 2 * pi), cr * np.sqrt(rng.uniform(0, 0.62))
        cv2.circle(img, (int(cx + np.cos(t) * rr), int(cy + np.sin(t) * rr)),
                   max(1, int(0.016 * S)), CENTER_DOT_COLOR, -1, cv2.LINE_AA)

    # ── downsample + sensor noise ─────────────────────────────────────────
    img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    if cfg.noise > 0:
        img = np.clip(img.astype(np.float32)
                      + rng.normal(0, 255 * cfg.noise, img.shape), 0, 255).astype(np.uint8)

    if return_meta:
        return img, dict(rot=rot, scale=scale, cx=cx / SS, cy=cy / SS,
                         petals=petal_meta)
    return img


def render_batch(latents, cfg, seed=0, progress=False):
    """Render a batch of latents.

    Parameters
    ----------
    latents : (N, R, C) int array, or dict {channel: (N, R, C)}
    cfg     : FlowerConfig
    seed    : int — per-sample rng is seeded as (seed, i) for reproducibility

    Returns
    -------
    (N, img_size, img_size, 3) uint8
    """
    if not isinstance(latents, dict):
        latents = {"color": np.asarray(latents)}
    N = next(iter(latents.values())).shape[0]
    it = range(N)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="render")
    out = np.empty((N, cfg.img_size, cfg.img_size, 3), np.uint8)
    for i in it:
        rng = np.random.default_rng([seed, i])
        out[i] = render_flower({k: v[i] for k, v in latents.items()}, cfg, rng)
    return out


# ---------------------------------------------------------------------------
# CV read-out (image -> latent grid)
# ---------------------------------------------------------------------------

#: material classes in the map returned by `_material_map`
MAT_BG, MAT_DISK, MAT_STEM, MAT_PETAL = 0, 1, 2, 3


def _material_map(lab, cfg, bg_lab):
    """Label every pixel with the nearest known material reference color.

    Independent per-material distance thresholds can double-claim a pixel:
    teal petals sit only 54 LAB units from the stem green, so a threshold wide
    enough to catch the shaded part of the stem also eats part of a teal petal
    and splits it in two.  A single nearest-reference assignment cannot
    double-claim, and needs no tolerances.

    The reference set encodes the rendering convention: each petal color is
    listed at every lightness the renderer can produce (flat fill, shaded base,
    outline, spot blotch), all mapping to MAT_PETAL.

    Parameters
    ----------
    lab    : (S, S, 3) float32 CIELAB image
    cfg    : FlowerConfig
    bg_lab : (3,) float32 — background color measured from the image border

    Returns
    -------
    (S, S) uint8 map of MAT_* labels
    """
    refs = [np.asarray(bg_lab, np.float32)]
    kinds = [MAT_BG]

    def add(rgb, kind):
        refs.append(_to_lab(np.clip(rgb, 0, 255)))
        kinds.append(kind)

    add(CENTER_COLOR, MAT_DISK)
    add(CENTER_DOT_COLOR, MAT_DISK)
    add(STEM_COLOR, MAT_STEM)
    add(np.array(STEM_COLOR, float) * (1 - cfg.shade / 2), MAT_STEM)
    for s in range(cfg.n_symbols):
        base = np.array(PETAL_PALETTE[s], float)
        for k in (1.0, 1.0 - cfg.shade / 2, cfg.outline, cfg.spot_dark):
            add(base * k, MAT_PETAL)

    ref = np.stack(refs)[None, None]                      # (1, 1, M, 3)
    d = np.linalg.norm(lab[:, :, None, :] - ref, axis=-1)  # (S, S, M)
    return np.asarray(kinds, np.uint8)[np.argmin(d, axis=-1)]


def _circ_mean_mod(angles, period):
    """Circular mean of `angles` modulo `period`, returned in [0, period)."""
    z = np.mean(np.exp(1j * 2 * pi * np.asarray(angles) / period))
    return (np.angle(z) / (2 * pi) * period) % period


def denoise_ksize(cfg):
    """Median-filter kernel appropriate to this resolution (odd; 1 = off).

    A *generated* image is not a clean render: a diffusion model leaves
    high-frequency residual, and a rank-deficient one leaves a lot of it (see
    docs/dit_rank_deficiency.md).  Nearest-material classification is per-pixel,
    so isolated speckle becomes spurious foreground and shatters the component
    map.  The kernel is tied to the petal minor axis so it only ever removes
    structure smaller than a petal: ~1/20 of the image width, clipped to 5 px so
    thin petals at low resolution are never smoothed away.
    """
    return int(min(5, max(1, 2 * round(cfg.img_size / 64.0) - 1)))


def expected_petal_area(cfg):
    """Area in pixels of an average petal, from the renderer's own geometry.

    Mirrors the ellipse the renderer draws: semi-major ``a*S``, semi-minor
    ``min(petal_wid_k * 2*pi*rho*S / C, petal_wid_cap * a*S)``, area ``pi*a*b``.
    """
    C = cfg.n_slots
    S = float(cfg.img_size)
    return float(pi * np.mean([
        (a * S) * min(cfg.petal_wid_k * 2 * pi * rho * S / C, cfg.petal_wid_cap * a * S)
        for rho, a in cfg.ring_bands()]))


def decode_flower(img, cfg, ctr_margin=1.30, min_area_frac=0.28,
                  spot_frac_thresh=0.05, dark_rel=0.62, return_diag=False,
                  denoise=None):
    """Recover the latent grid from a rendered flower image.

    Pipeline
    --------
    1. nearest-material classification of every pixel (background measured
       from the image border, plus the known disk / stem / petal colors at
       every lightness the renderer produces)  ->  foreground mask
    2. disk-material blob  ->  flower center (cx, cy)
    3. stem material  ->  rotation anchor (rot_mode="free" only)
    4. connected components of (foreground - disk - stem)  ->  petal blobs
    5. blob centroid  ->  (radius -> ring index, angle -> slot index)
    6. eroded-blob median LAB  ->  nearest palette entry  ->  color symbol
       dark-pixel fraction     ->  spot channel
       tip-width ratio         ->  shape channel

    Parameters
    ----------
    img  : (S, S, 3) uint8 RGB
    cfg  : FlowerConfig
    ctr_margin    : exclude this multiple of the detected disk radius
    min_area_frac : blobs smaller than this fraction of the expected petal
                    area are discarded as debris
    return_diag   : also return intermediate masks / per-blob records
    denoise       : median-filter kernel applied before classification.
                    None -> `denoise_ksize(cfg)`; 1 or 0 -> disabled.  Needed
                    only for generated images; clean renders decode identically
                    either way (verified at 48-192 px).

    Returns
    -------
    dict with
      symbols : {channel: (R, C) int}  — -1 marks an unfilled cell
      ok      : bool  — exactly R*C blobs, every cell filled exactly once
      n_blobs : int
      center  : (cx, cy)
      rot     : estimated global rotation
      diag    : present only if return_diag
    """
    S = cfg.img_size
    R, C = cfg.n_rings, cfg.n_slots
    k = denoise_ksize(cfg) if denoise is None else int(denoise)
    if k >= 3:
        img = cv2.medianBlur(np.ascontiguousarray(img), k)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # ── 1. background + material classification ───────────────────────────
    border = np.concatenate([lab[:2].reshape(-1, 3), lab[-2:].reshape(-1, 3),
                             lab[:, :2].reshape(-1, 3), lab[:, -2:].reshape(-1, 3)])
    bg_lab = np.median(border, axis=0)
    mat = _material_map(lab, cfg, bg_lab)
    fg = mat != MAT_BG

    # ── 2. center ─────────────────────────────────────────────────────────
    # The disk's anti-aliased rim is a blend of disk gold and petal color, so a
    # pure color mask leaves a thin ring that would bridge every petal into one
    # blob.  Excluding a slightly enlarged *disk* instead of the color mask
    # removes the rim regardless of what it blended with.
    ctr_m = mat == MAT_DISK
    cx, cy = S / 2, S / 2
    r_ctr = cfg.center_r * S
    if ctr_m.sum() >= 4:
        n, lbl, stats, cents = cv2.connectedComponentsWithStats(
            ctr_m.astype(np.uint8), connectivity=8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cx, cy = cents[k]
            r_ctr = np.sqrt(stats[k, cv2.CC_STAT_AREA] / pi)
    yy, xx = np.mgrid[0:S, 0:S]
    rad_px = np.hypot(xx - cx, yy - cy)
    ctr_m = rad_px < r_ctr * ctr_margin

    # ── 3. stem ───────────────────────────────────────────────────────────
    stem_m = cv2.dilate((mat == MAT_STEM).astype(np.uint8),
                        np.ones((3, 3), np.uint8)).astype(bool)
    rot_stem = None
    if cfg.draw_stem and stem_m.sum() >= 4:
        ys, xs = np.nonzero(stem_m)
        d = np.hypot(xs - cx, ys - cy)
        j = np.argsort(d)[-max(3, len(d) // 20):]        # farthest stem pixels
        rot_stem = _circ_mean_mod(
            np.arctan2(ys[j] - cy, xs[j] - cx), 2 * pi) - pi

    # ── 4. petal blobs ────────────────────────────────────────────────────
    petal_m = (fg & ~ctr_m & ~stem_m).astype(np.uint8)

    bands = cfg.ring_bands()
    exp_area = expected_petal_area(cfg)

    def _components(mask):
        n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = [k for k in range(1, n)
                if stats[k, cv2.CC_STAT_AREA] >= min_area_frac * exp_area]
        return lbl, stats, cents, keep

    lbl, stats, cents, keep = _components(petal_m)
    # Two petals whose background gap closed (jitter, or a malformed generated
    # sample) fuse into one blob.  A single erosion usually severs a 1-2 px
    # bridge; adopt it only if it recovers more petals without shattering them.
    if len(keep) < R * C:
        eroded = cv2.erode(petal_m, np.ones((3, 3), np.uint8), iterations=1)
        l2, s2, c2, k2 = _components(eroded)
        if len(k2) > len(keep):
            lbl, stats, cents, keep = l2, s2, c2, k2
    n_fused = max(0, R * C - len(keep))

    recs = []
    for k in keep:
        bx, by = cents[k]
        dx, dy = bx - cx, by - cy
        rho = np.hypot(dx, dy) / S
        ring = int(np.argmin([abs(rho - rf) for rf, _ in bands]))
        recs.append(dict(label=k, xy=(bx, by), rho=rho, ring=ring,
                         phi=np.arctan2(dy, dx),
                         area=int(stats[k, cv2.CC_STAT_AREA])))

    # ── 5. angular index ──────────────────────────────────────────────────
    delta = 2 * pi / C
    if cfg.rot_mode == "free":
        rot = rot_stem if rot_stem is not None else cfg.base_angle
    else:
        if recs:      # refine within the jitter window using the blobs themselves
            resid = [(rc["phi"] - cfg.ring_offset(rc["ring"])) % delta for rc in recs]
            off = _circ_mean_mod(resid, delta)
            base = cfg.base_angle % delta
            rot = cfg.base_angle + ((off - base + delta / 2) % delta) - delta / 2
        else:
            rot = cfg.base_angle

    for rc in recs:
        phi0 = rot + cfg.ring_offset(rc["ring"])
        rc["slot"] = int(np.round((rc["phi"] - phi0) / delta)) % C

    # ── 6. per-blob attributes ────────────────────────────────────────────
    pal_lab = np.stack([_to_lab(c) for c in PETAL_PALETTE[:cfg.n_symbols]])
    er = np.ones((3, 3), np.uint8)
    for rc in recs:
        m = (lbl == rc["label"]).astype(np.uint8)
        core = cv2.erode(m, er, iterations=1)
        if core.sum() < 3:
            core = m
        px = lab[core.astype(bool)]

        # color: the brightest half of the blob — immune to the outline, the
        # radial shading, and a blotch covering up to ~50% of the petal
        order = np.argsort(px[:, 0])[len(px) // 2:]
        med = np.median(px[order], axis=0)
        rc["sym_color"] = int(np.argmin(np.linalg.norm(pal_lab - med, axis=1)))
        rc["color_margin"] = float(np.sort(np.linalg.norm(pal_lab - med, axis=1))[1]
                                   - np.min(np.linalg.norm(pal_lab - med, axis=1)))

        # spot: a dark blotch inside the petal.  `dark_rel` is a *relative*
        # lightness drop: sRGB scaling by k moves L* by roughly k**0.73, so
        # spot_dark=0.30 -> 0.41*L while outline=0.72 -> 0.78*L, and a cut at
        # 0.62*L separates them with margin at every palette color.
        dark = px[:, 0] < dark_rel * med[0]
        rc["dark_frac"] = float(dark.mean())
        rc["sym_spot"] = int(rc["dark_frac"] > spot_frac_thresh)

        # shape: lanceolate petals are longer and narrower, so the blob's
        # second-moment aspect ratio separates the two classes.  The decision
        # boundary is the geometric mean of the two nominal aspects, which
        # cancels the (multiplicative) shrinkage from center-disk clipping.
        mu = cv2.moments(m)
        if mu["m00"] > 0:
            cov = np.array([[mu["mu20"], mu["mu11"]], [mu["mu11"], mu["mu02"]]]) / mu["m00"]
            l2v, l1v = np.linalg.eigvalsh(cov)      # ascending: minor, major
            rc["aspect"] = float(np.sqrt(max(l1v, 1e-9) / max(l2v, 1e-9)))
        else:
            rc["aspect"] = 1.0
        rho_f, a_f = bands[rc["ring"]]
        a_px = a_f * S
        b_px = min(cfg.petal_wid_k * 2 * pi * rho_f * S / C, cfg.petal_wid_cap * a_px)
        thresh_aspect = (a_px / max(b_px, 1e-6)) * np.sqrt(cfg.shape_len_k / cfg.shape_wid_k)
        rc["sym_shape"] = int(rc["aspect"] > thresh_aspect)

    # ── assemble ──────────────────────────────────────────────────────────
    symbols = {ch: np.full((R, C), -1, int) for ch in cfg.channels}
    hits = np.zeros((R, C), int)
    for rc in recs:
        hits[rc["ring"], rc["slot"]] += 1
        for ch in cfg.channels:
            symbols[ch][rc["ring"], rc["slot"]] = rc[f"sym_{ch}"]

    ok = bool(len(recs) == R * C and (hits == 1).all())
    out = dict(symbols=symbols, ok=ok, n_blobs=len(recs), n_fused=n_fused,
               center=(cx, cy), rot=rot)
    if return_diag:
        out["diag"] = dict(fg=fg, mat=mat, ctr=ctr_m, stem=stem_m, petal=petal_m,
                           labels=lbl, recs=recs, hits=hits, rot_stem=rot_stem,
                           exp_area=exp_area)
    return out


def decode_batch(imgs, cfg, channel="color", progress=False, **kw):
    """Decode a batch of images.

    Parameters
    ----------
    imgs    : (N, S, S, 3) uint8
    channel : which channel's grid to return as the primary array
    kw      : forwarded to `decode_flower`

    Returns
    -------
    grids : (N, R, C) int   — -1 where a cell could not be read
    ok    : (N,) bool       — clean segmentation (R*C blobs, no collisions)
    all_  : list of dicts   — full per-image decode results
    """
    it = range(len(imgs))
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="decode")
    res = [decode_flower(imgs[i], cfg, **kw) for i in it]
    grids = np.stack([r["symbols"][channel] for r in res])
    ok = np.array([r["ok"] for r in res])
    return grids, ok, res


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
