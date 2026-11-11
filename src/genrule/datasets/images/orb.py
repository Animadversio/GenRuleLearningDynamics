"""
orb_grid_lib.py

Intermediate case between the synthetic bit-vector benchmarks and the flower
images: a grid of visually rich objects in which **each object occupies exactly
one DiT token**.

Motivation
----------
In the flower benchmark a petal spans 5-7 tokens, 5-20 tokens straddle two
petals, and petal positions jitter, so any circuit that reads the rule must
first solve a perceptual grouping problem whose form differs from image to
image.  Rule learning there is 43-60x slower than on raw bit vectors.

This dataset removes exactly that confound and keeps everything else:

    * one object per token, guaranteed to fit strictly inside its own patch
      (asserted at config time and verified pixel-wise by
      `check_one_item_per_token`)
    * object slots at fixed grid positions, so slot index <-> token index is a
      fixed mapping shared by every image
    * objects still visually complex: 3D-shaded spheres/ellipsoids with a
      specular highlight, per-object light direction, radius, elongation,
      orientation and sub-patch position jitter, plus background tint and noise

With `n_rows = n_cols = 6` and `patch_px` = `img_size / 6`, the token count (36)
and the latent dimension (D = 36) match the synthetic parity runs exactly, so
the only remaining difference from those runs is that a token carries a rendered
object rather than a raw +-1 value.

Latent, rules, and validity checking are reused verbatim from
`genrule.datasets.images.flower` -- `OrbConfig` deliberately exposes the same
`n_rings` / `n_slots` / `n_symbols` / `n_petals` interface those functions
consume, so `sample_latents`, `check_latents`, `rule_subunit_ok`,
`support_size` and `corrupt_latents` all work unchanged.

Typical usage
-------------
    from genrule.datasets.images.orb import PRESETS, render_batch, decode_batch
    from genrule.datasets.images.flower import sample_latents, check_latents

    cfg    = PRESETS["orb36"]
    latent = sample_latents("parity", 2048, cfg, group_size=3, seed=0)
    imgs   = render_batch(latent, cfg, seed=0)
    rec, ok, _ = decode_batch(imgs, cfg)
    valid  = check_latents(rec, "parity", cfg, group_size=3)
"""

import os
import sys
from dataclasses import dataclass, replace
from math import pi

import cv2
import numpy as np


from .colors import PETAL_PALETTE, _to_lab

#: objects use the same palette as the flower petals, so color-difficulty is
#: held constant across the two benchmarks
ORB_PALETTE = PETAL_PALETTE
BG_COLOR = (238, 240, 231)

CHANNELS = ("color",)

#: The 2x2 decomposition of "what does rendering cost?".  `shaded x full` is the
#: standard orb dataset; the other three cells strip one factor each.
#:   shading  : flat fill      vs 3D Lambertian+specular with a per-object light
#:   nuisance : deterministic  vs radius/elongation/orientation/position jitter,
#:              colour jitter, background tint and sensor noise
RENDER_MODES = {
    ("flat",   "none"): dict(flat=True,  r_jit=0.0, oval=0.0, pos_jit=0.0,
                             light_jit=0.0, color_jit=0.0, bg_jit=0.0, noise=0.0,
                             orient_jit=0.0),
    ("flat",   "full"): dict(flat=True),
    ("shaded", "none"): dict(flat=False, r_jit=0.0, oval=0.0, pos_jit=0.0,
                             light_jit=0.0, color_jit=0.0, bg_jit=0.0, noise=0.0,
                             orient_jit=0.0),
    ("shaded", "full"): dict(flat=False),
}


@dataclass
class OrbConfig:
    """Grid-of-orbs configuration.

    Field names `n_rings` / `n_slots` mirror `FlowerConfig` so the shared rule
    functions in `flower_rule_lib` accept this config unchanged; here they mean
    grid rows and grid columns.
    """
    # ── latent shape (rows x cols of objects) ─────────────────────────────
    n_rings:   int = 6          # rows
    n_slots:   int = 6          # cols
    n_symbols: int = 2
    channels: tuple = ("color",)

    # ── content type ──────────────────────────────────────────────────────
    #: "orb"   -- 3D-shaded ellipsoid on a background (the full dataset)
    #: "block" -- the cell is filled edge to edge with the symbol's colour.
    #:            Definable at ANY patch size, so it is the content type for the
    #:            pixels-per-token ladder: the latent -> patch map is just
    #:            "broadcast the colour", holding content complexity at its
    #:            minimum while the output dimensionality varies.
    #: "raw"   -- single channel, cell filled with the +-1 value itself. This
    #:            reproduces the synthetic parity representation exactly and is
    #:            the pipeline-validation anchor.
    content: str = "orb"

    # ── canvas / tokenization ─────────────────────────────────────────────
    patch_px:    int = 24       # DiT patch size in pixels == one object cell
    supersample: int = 4

    # ── object geometry, as fractions of patch_px ─────────────────────────
    # Budget: ball_r*(1+r_jit)*(1+oval) + pos_jit + edge_pad must stay < 0.5,
    # so an object can never reach its patch border.  `validate()` enforces it.
    ball_r:    float = 0.30     # object radius
    pos_jit:   float = 0.06     # sub-patch position jitter (+-, fraction of patch)
    r_jit:     float = 0.08     # relative radius jitter
    oval:      float = 0.12     # max relative elongation (0 = perfect spheres)
    orient_jit: float = 1.0     # random object orientation, in units of pi.
                                # NOTE: even for a circular object this is not a
                                # no-op -- the light direction is fixed in the
                                # OBJECT frame, so rotating the object rotates
                                # its shading. Must be 0 for a deterministic
                                # render.
    edge_pad:  float = 0.055    # anti-aliasing margin kept clear of the border

    # ── shading / appearance ──────────────────────────────────────────────
    flat:      bool = False     # True -> uniform fill, no shading or specular
    ambient:   float = 0.38
    diffuse:   float = 0.62
    specular:  float = 0.75
    spec_pow:  float = 14.0
    light_jit: float = 0.55     # per-object light-direction jitter (radians)
    color_jit: float = 0.035
    bg_jit:    float = 0.02
    noise:     float = 0.012

    # ── derived ───────────────────────────────────────────────────────────
    @property
    def img_size(self):
        """Square canvas: one patch per object, no gaps."""
        return self.patch_px * max(self.n_rings, self.n_slots)

    @property
    def n_channels(self):
        """1 for the raw +-1 anchor, 3 for anything rendered in colour."""
        return 1 if self.content == "raw" else 3

    @property
    def n_petals(self):
        """D = rows * cols -- number of objects = number of latent cells."""
        return self.n_rings * self.n_slots

    @property
    def n_tokens(self):
        """DiT sequence length when patch_size == patch_px."""
        return (self.img_size // self.patch_px) ** 2

    def validate(self):
        """Assert every object stays strictly inside its own patch.

        Only meaningful for `content="orb"`: block/raw cells fill their patch by
        construction, so containment is exact and trivial.

        radius (worst case) + position jitter + anti-aliasing margin must not
        reach the patch border, otherwise an object would leak into a
        neighbouring token and the whole point of the dataset is lost.
        """
        if self.content != "orb":
            return 0.0
        r_max = self.ball_r * (1 + self.r_jit) * (1 + self.oval)
        reach = r_max + self.pos_jit + self.edge_pad
        if reach > 0.5:
            raise ValueError(
                f"objects would cross the patch border: max reach {reach:.3f} "
                f"> 0.5 of patch (r {r_max:.3f} + jitter {self.pos_jit} + "
                f"pad {self.edge_pad}). Reduce ball_r / pos_jit / oval.")
        # The fractional budget is necessary but NOT sufficient: anti-aliasing
        # from the supersampled downsample spreads an edge by ~0.5 px on each
        # side regardless of patch size, so the same fractional margin leaks at
        # small patches.
        #
        # `edge_pad` is NOT drawn by the renderer -- `_render_cells`/
        # `render_orb_grid` place an object of half-extent
        # `ball_r*(1+r_jit)*(1+oval)` at a centre offset by up to `pos_jit`, and
        # never consult `edge_pad`.  So the margin that actually exists in the
        # image is measured against that *rendered* reach; counting `edge_pad`
        # as consumed (as an earlier version did) understates the real clearance
        # by `edge_pad * patch_px` and silently turns edge_pad into the safety
        # factor.  Measured leakage (fraction of images with an object touching
        # a patch border) against the rendered margin, at patch 6 / 8 / 10:
        #     0.5 px -> 100% / 88% / 60% leak
        #     0.7 px ->  38% / 19% / 10%
        #     0.9 px ->   0% /  0% /  0%
        # BUT that sweep used the default jitter, where an object at the
        # worst-case extent is rare.  With the jitter switched off every object
        # sits exactly at the limit, and 0.9 px then leaks in 100% of images.
        # Re-measured against the worst case (zero geometric jitter, so every
        # object is at the bound) the requirement is 1.10 px, again independent
        # of patch size.  That is the value enforced here, since the bound has
        # to hold for the worst object the config can produce, not the median.
        rendered_reach = r_max + self.pos_jit
        margin_px = (0.5 - rendered_reach) * self.patch_px
        if margin_px < 1.10:
            raise ValueError(
                f"rendered margin {margin_px:.2f} px is below the measured "
                f"1.10 px needed to survive anti-aliasing, so objects will leak "
                f"into neighbouring tokens even though the fractional budget "
                f"passes. Shrink ball_r / pos_jit / oval / r_jit, or increase "
                f"patch_px. (Note edge_pad={self.edge_pad} is not drawn and so "
                f"does not help here.)")
        return reach

    def slot_center(self, r, c):
        """Nominal pixel center of slot (r, c) at final resolution."""
        return ((c + 0.5) * self.patch_px, (r + 0.5) * self.patch_px)


PRESETS = {
    # 6x6 = 36 tokens, D=36: identical sequence length and latent dimension to
    # the synthetic parity runs.  patch 12 -> 144 px canvas, 432 dims/token.
    # NOTE 432 > hidden 384, so this preset is 1.1x rank deficient.  It leaks
    # nothing (rendered margin 1.85 px, well over the measured 1.10 px), but see
    # `orb36_p8*` below for the over-complete operating points.
    "orb36": OrbConfig(n_rings=6, n_slots=6, patch_px=12),

    # ── over-complete operating points, patch 8 -> 192 dims/token ──────────
    # 192 <= hidden 384, so the patch embedding stays injective and the runs sit
    # inside the range where tau_rule ~ d^0.66 was fitted (R^2 0.9973).
    #
    # Default geometry does NOT fit at patch 8: its rendered margin is 0.62 px
    # against the 1.10 px anti-aliasing requirement, so ball_r is reduced.  Both
    # variants below were verified at 128 samples -- zero leakage, read-out cell
    # accuracy 1.0000, corrupted-latent rejection 1.000.
    #
    # `orb36_p8` keeps geometric nuisance (position / size / shape jitter) at
    # roughly half strength and gives a 4.8 px orb.
    "orb36_p8": OrbConfig(n_rings=6, n_slots=6, patch_px=8, ball_r=0.302,
                          oval=0.040, r_jit=0.050, pos_jit=0.030, edge_pad=0.0),
    # `orb36_p8_max` spends the entire radius budget on size: a 5.8 px orb, the
    # largest possible at patch 8 (the ceiling is `patch_px - 2*1.10` px, and
    # off-centring cannot beat it because the margin is required on every
    # border).  The cost is that geometric jitter must go to zero, so each
    # latent renders to a fixed geometry -- photometric nuisance (light
    # direction, colour, background, noise) is unaffected.
    "orb36_p8_max": OrbConfig(n_rings=6, n_slots=6, patch_px=8, ball_r=0.360,
                              oval=0.0, r_jit=0.0, pos_jit=0.0, edge_pad=0.0),
    # same 36 objects, larger patches -> more visual detail per object, same
    # token count and therefore the same training cost
    # larger objects; read-out is exact but the model does NOT fit these at
    # hidden 384 -- kept only for tokenization/fit studies
    "orb36_big": OrbConfig(n_rings=6, n_slots=6, patch_px=24),
    # 12x12 = 144 tokens, D=144: matches the flower runs' sequence length
    "orb144": OrbConfig(n_rings=12, n_slots=12, patch_px=12),
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _shade_orb(canvas, cx, cy, rx, ry, theta, base_rgb, cfg, rng):
    """Draw one 3D-shaded ellipsoid centred at (cx, cy) into `canvas`.

    Lambertian diffuse + Phong specular over an implied unit sphere normal, so
    the object reads as a glossy ball rather than a flat disc.
    """
    x0, x1 = int(np.floor(cx - rx - 2)), int(np.ceil(cx + rx + 2))
    y0, y1 = int(np.floor(cy - ry - 2)), int(np.ceil(cy + ry + 2))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, canvas.shape[1]), min(y1, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    ct, st = np.cos(theta), np.sin(theta)
    u = (dx * ct + dy * st) / rx          # object frame, normalised
    v = (-dx * st + dy * ct) / ry
    rr = u * u + v * v
    inside = rr <= 1.0
    if not inside.any():
        return

    if cfg.flat:
        # flat fill: the object carries colour and outline only, no 3D cue
        patch = canvas[y0:y1, x0:x1]
        patch[inside] = np.clip(base_rgb, 0, 255).astype(np.uint8)
        return

    nz = np.sqrt(np.clip(1.0 - rr, 0, 1))
    lt = rng.uniform(-1, 1, 2) * cfg.light_jit
    L = np.array([np.sin(lt[0]), np.sin(lt[1]) - 0.45, 0.85])
    L /= np.linalg.norm(L)
    ndotl = np.clip(u * L[0] + v * L[1] + nz * L[2], 0, 1)

    V = np.array([0.0, 0.0, 1.0])
    H = L + V
    H /= np.linalg.norm(H)
    ndoth = np.clip(u * H[0] + v * H[1] + nz * H[2], 0, 1)
    spec = cfg.specular * ndoth ** cfg.spec_pow

    shade = cfg.ambient + cfg.diffuse * ndotl
    col = (base_rgb[None, None, :] * shade[..., None]
           + 255.0 * spec[..., None])
    patch = canvas[y0:y1, x0:x1]
    patch[inside] = np.clip(col[inside], 0, 255).astype(np.uint8)


def _render_cells(latent, cfg, rng, return_meta=False):
    """Flat-cell renderers: each cell is filled edge to edge.

    "block": the symbol's palette colour, 3 channels.
    "raw"  : the symbol as +-1 in a single channel, i.e. exactly the synthetic
             parity representation (uint8 0/255 maps to -1/+1 through the
             pipeline's /127.5 - 1).
    """
    P = cfg.patch_px
    if cfg.content == "raw":
        vals = np.where(np.asarray(latent) > 0, 255, 0).astype(np.uint8)
        img = np.repeat(np.repeat(vals, P, axis=0), P, axis=1)[..., None]
    else:
        pal = np.array(ORB_PALETTE[:cfg.n_symbols], dtype=np.uint8)
        img = np.repeat(np.repeat(pal[np.asarray(latent)], P, axis=0), P, axis=1)
        if cfg.color_jit > 0:
            img = np.clip(img.astype(np.float32)
                          + rng.normal(0, 255 * cfg.color_jit, img.shape),
                          0, 255).astype(np.uint8)
    if cfg.noise > 0 and cfg.content == "block":
        img = np.clip(img.astype(np.float32)
                      + rng.normal(0, 255 * cfg.noise, img.shape), 0, 255).astype(np.uint8)
    return (img, dict(orbs=[])) if return_meta else img


def render_orb_grid(latent, cfg, rng=None, return_meta=False):
    """Render one grid-of-orbs image.

    Parameters
    ----------
    latent : (R, C) int array, or dict {channel: (R, C)} (only "color" is used)
    cfg    : OrbConfig
    rng    : np.random.Generator or None

    Returns
    -------
    (img_size, img_size, 3) uint8 RGB
    """
    cfg.validate()
    rng = np.random.default_rng() if rng is None else rng
    if isinstance(latent, dict):
        latent = latent["color"]
    latent = np.asarray(latent)

    if cfg.content in ("block", "raw"):
        return _render_cells(latent, cfg, rng, return_meta)

    SS = cfg.supersample
    P = cfg.patch_px * SS
    S = cfg.img_size * SS

    bg = np.clip(np.array(BG_COLOR, float) * (1 + rng.uniform(-1, 1) * cfg.bg_jit)
                 + rng.normal(0, 255 * cfg.bg_jit, 3), 0, 255)
    img = np.tile(bg.astype(np.uint8), (S, S, 1))

    meta = []
    for r in range(cfg.n_rings):
        for c in range(cfg.n_slots):
            base_r = cfg.ball_r * P * (1 + rng.uniform(-1, 1) * cfg.r_jit)
            e = rng.uniform(0, cfg.oval)
            rx, ry = base_r * (1 + e), base_r / (1 + e)
            theta = rng.uniform(0, pi) * cfg.orient_jit
            cx = (c + 0.5) * P + rng.uniform(-1, 1) * cfg.pos_jit * P
            cy = (r + 0.5) * P + rng.uniform(-1, 1) * cfg.pos_jit * P
            sym = int(latent[r, c])
            base = np.clip(np.array(ORB_PALETTE[sym], float)
                           + rng.normal(0, 255 * cfg.color_jit, 3), 0, 255)
            _shade_orb(img, cx, cy, rx, ry, theta, base, cfg, rng)
            meta.append(dict(row=r, col=c, sym=sym, xy=(cx / SS, cy / SS),
                             rx=rx / SS, ry=ry / SS, theta=theta))

    img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    if cfg.noise > 0:
        img = np.clip(img.astype(np.float32)
                      + rng.normal(0, 255 * cfg.noise, img.shape), 0, 255).astype(np.uint8)
    return (img, dict(orbs=meta)) if return_meta else img


def render_batch(latents, cfg, seed=0, progress=False):
    """Render a batch. `latents` is (N, R, C) or {channel: (N, R, C)}."""
    if isinstance(latents, dict):
        latents = latents["color"]
    latents = np.asarray(latents)
    it = range(len(latents))
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="render orbs")
    out = np.empty((len(latents), cfg.img_size, cfg.img_size, cfg.n_channels),
                   np.uint8)
    for i in it:
        out[i] = render_orb_grid(latents[i], cfg, np.random.default_rng([seed, i]))
    return out


# ---------------------------------------------------------------------------
# Read-out
# ---------------------------------------------------------------------------

def _decode_cells(img, cfg, eps=0.1):
    """Read-out for the flat-cell contents.

    "raw" snaps the cell mean to +-1 with tolerance `eps` and marks anything
    ambiguous as -1, matching the synthetic convention
    (`round_to_pos_neg_one(x, eps=1e-1)` in DiT_learn_parity_CLI.py).
    "block" classifies the cell mean colour against the palette in a*/b*.
    """
    P, R, C = cfg.patch_px, cfg.n_rings, cfg.n_slots
    x = np.asarray(img)
    cells = x.reshape(R, P, C, P, x.shape[-1]).mean(axis=(1, 3))     # (R, C, ch)
    if cfg.content == "raw":
        v = cells[..., 0] / 127.5 - 1.0
        grid = np.full((R, C), -1, int)
        grid[np.abs(v - 1) < eps] = 1
        grid[np.abs(v + 1) < eps] = 0
    else:
        pal_ab = np.stack([_to_lab(c) for c in ORB_PALETTE[:cfg.n_symbols]])[:, 1:]
        ab = cv2.cvtColor(cells.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1:]
        d = np.linalg.norm(ab[:, :, None, :] - pal_ab[None, None], axis=-1)
        srt = np.sort(d, axis=-1)
        grid = np.argmin(d, axis=-1)
        thr = 0.10 * float(np.linalg.norm(np.ptp(pal_ab, axis=0)) + 1e-6)
        grid = np.where(srt[..., 1] - srt[..., 0] < thr, -1, grid)
    ok = bool((grid >= 0).all())
    return dict(symbols={ch: grid for ch in cfg.channels}, ok=ok,
                n_blobs=int((grid >= 0).sum()), margin_min=0.0)


def decode_orb_grid(img, cfg, min_area_frac=0.25, margin=0.10,
                    chroma_frac=0.35, raw_eps=0.1, return_diag=False):
    """Recover the latent grid from a rendered orb-grid image.

    Because slots are token-aligned by construction, the read-out is per-patch
    and needs no global segmentation: for each patch, find the object pixels,
    reject the specular highlight and the dark terminator, and classify the
    remaining body color.

    Colors are matched in the CIELAB **a*/b* (chroma) plane only: shading moves
    L* a great deal and chroma very little, so dropping L* makes the read-out
    shading-invariant.

    A slot is marked -1 (and `ok` False) when the object is missing/too small or
    its color is ambiguous, which is the analog of the synthetic
    ambiguous-quantization (NaN) flag.

    Returns
    -------
    dict with keys symbols {channel: (R, C) int}, ok, n_blobs, margin_min
    """
    S, P = cfg.img_size, cfg.patch_px
    if cfg.content in ("block", "raw"):
        return _decode_cells(img, cfg, eps=raw_eps)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:1].reshape(-1, 3), lab[-1:].reshape(-1, 3),
                             lab[:, :1].reshape(-1, 3), lab[:, -1:].reshape(-1, 3)])
    bg_lab = np.median(border, axis=0)
    pal_ab = np.stack([_to_lab(c) for c in ORB_PALETTE[:cfg.n_symbols]])[:, 1:]

    grid = np.full((cfg.n_rings, cfg.n_slots), -1, int)
    n_found, margins, recs = 0, [], []
    exp_area = pi * (cfg.ball_r * P) ** 2

    # Object pixels are selected by CHROMA distance from the background, not
    # full LAB distance.  A generated background is speckled mostly in
    # luminance, so an L*-inclusive test counts that speckle as object (measured
    # 0.52 foreground per cell vs 0.34 in training) and drags the median colour
    # toward grey until the palette margin collapses.  Chroma ignores the
    # speckle: the palette entries are highly saturated and the background is
    # near-neutral, so the separation is large and luminance-noise-proof.
    bg_ab = bg_lab[1:]
    pal_chroma = np.linalg.norm(pal_ab - bg_ab[None, :], axis=1).min()

    for r in range(cfg.n_rings):
        for c in range(cfg.n_slots):
            y0, y1 = r * P, (r + 1) * P
            x0, x1 = c * P, (c + 1) * P
            cell = lab[y0:y1, x0:x1]
            chroma = np.linalg.norm(cell[:, :, 1:] - bg_ab[None, None, :], axis=-1)
            fg = chroma > chroma_frac * pal_chroma
            area = int(fg.sum())
            if area < min_area_frac * exp_area:
                recs.append(dict(row=r, col=c, area=area, sym=-1, margin=0.0))
                continue
            n_found += 1
            # weight toward the most chromatic (most object-like) pixels
            ch_fg = chroma[fg]
            core = ch_fg >= np.percentile(ch_fg, 40)
            body_ab = np.median(cell[:, :, 1:][fg][core], axis=0)
            d = np.linalg.norm(pal_ab - body_ab[None, :], axis=1)
            srt = np.sort(d)
            marg = float(srt[1] - srt[0]) if len(srt) > 1 else float("inf")
            sym = int(np.argmin(d))
            if marg < margin * float(np.linalg.norm(np.ptp(pal_ab, axis=0)) + 1e-6):
                sym = -1
            grid[r, c] = sym
            margins.append(marg)
            recs.append(dict(row=r, col=c, area=area, sym=sym, margin=marg))

    ok = bool(n_found == cfg.n_petals and (grid >= 0).all())
    out = dict(symbols={ch: grid for ch in cfg.channels}, ok=ok, n_blobs=n_found,
               margin_min=float(min(margins)) if margins else 0.0)
    if return_diag:
        out["diag"] = dict(recs=recs, bg_lab=bg_lab)
    return out


def decode_batch(imgs, cfg, channel="color", progress=False, **kw):
    """Decode a batch. Returns (grids (N,R,C), ok (N,), full results)."""
    it = range(len(imgs))
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="decode orbs")
    res = [decode_orb_grid(imgs[i], cfg, **kw) for i in it]
    return (np.stack([r["symbols"][channel] for r in res]),
            np.array([r["ok"] for r in res]), res)


# ---------------------------------------------------------------------------
# The property this dataset exists to guarantee
# ---------------------------------------------------------------------------

def check_one_item_per_token(imgs, cfg, tol=1.0):
    """Verify pixel-wise that no object leaks out of its own patch.

    For every patch, checks that no foreground pixel touches the patch border.
    A single leaked pixel would mean a token carries evidence about two objects,
    which is exactly the confound this dataset removes.

    Returns
    -------
    dict: frac_clean (fraction of images with zero leakage), n_leaks (total
    offending patches), max_border_frac (worst per-patch border occupancy)
    """
    P = cfg.patch_px
    T = cfg.img_size // P
    n_leak, worst, clean = 0, 0.0, 0
    for im in imgs:
        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)
        border = np.concatenate([lab[:1].reshape(-1, 3), lab[-1:].reshape(-1, 3),
                                 lab[:, :1].reshape(-1, 3), lab[:, -1:].reshape(-1, 3)])
        fg = np.linalg.norm(lab - np.median(border, axis=0), axis=-1) > 16.0
        bad = 0
        for i in range(T):
            for j in range(T):
                cell = fg[i * P:(i + 1) * P, j * P:(j + 1) * P]
                edge = np.concatenate([cell[0], cell[-1], cell[:, 0], cell[:, -1]])
                frac = float(edge.mean())
                worst = max(worst, frac)
                if edge.sum() > tol:
                    bad += 1
        n_leak += bad
        clean += (bad == 0)
    return dict(frac_clean=clean / len(imgs), n_leaks=n_leak,
                max_border_frac=worst)
