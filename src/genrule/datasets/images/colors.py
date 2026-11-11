"""Shared RGB palette and CIELAB helpers, unchanged from the research implementation."""
import numpy as np
import cv2

BG_COLOR         = (238, 240, 231)   # pale sage paper


CENTER_COLOR     = (176, 118,  34)   # dark gold disk floret


CENTER_DOT_COLOR = (116,  72,  16)   # texture speckles on the disk


STEM_COLOR       = ( 56, 108,  56)   # dark green stem / leaf


PETAL_PALETTE = [
    (214,  51, 108),   # 0 crimson
    (250, 176,   5),   # 1 amber
    ( 76, 110, 245),   # 2 indigo
    ( 12, 166, 120),   # 3 teal
    (156,  54, 181),   # 4 purple
    (255, 107,  53),   # 5 vermilion
]


def _to_lab(rgb):
    """RGB uint8 triple or (...,3) array -> float32 CIELAB."""
    arr = np.asarray(rgb, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3) if lab.shape[0] > 1 else lab.reshape(3)


def palette_separation(n_symbols):
    """Minimum pairwise CIELAB distance among the first `n_symbols` petal colors.

    A sanity metric: the decoder classifies petal color by nearest-palette in
    LAB, so this bounds how much color jitter / shading it can tolerate.
    """
    labs = np.stack([_to_lab(c) for c in PETAL_PALETTE[:n_symbols]])
    d = np.linalg.norm(labs[:, None] - labs[None, :], axis=-1)
    d[np.diag_indices(n_symbols)] = np.inf
    return float(d.min())
