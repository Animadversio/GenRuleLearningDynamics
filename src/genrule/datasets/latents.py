"""Legacy symbol-grid sampling and rule checks shared by image datasets.

The first migration preserves the original encodings and return values.
"""
import numpy as np
from math import comb, factorial
from genrule.rules.parity import sample_group_parity_vec, sample_ensuring_uniqueness
from genrule.rules.exact_k import sample_exact_k_vec
from genrule.rules.row_k import _sample_row_exact_k
from genrule.rules.latin_square import sample_latin_square_vec, check_latin_square_batch, valid_set_size as latin_valid_set_size

def _as_grid(flat_pm1, cfg):
    """(N, D) in {-1,+1}  ->  (N, R, C) in {0, 1}."""
    return ((np.asarray(flat_pm1) + 1) // 2).reshape(-1, cfg.n_rings, cfg.n_slots)


def sample_latents(rule, N, cfg, seed=None, unique=True, **params):
    """Sample N latent grids satisfying `rule`.

    Parameters
    ----------
    rule   : {"parity", "exact_k", "row_k", "latin_sq"}
    N      : int   — number of grids
    cfg    : FlowerConfig
    seed   : int or None
    unique : bool  — reject duplicates (recommended; memorization needs it)
    params : rule-specific
        parity   : group_size (int, must divide R*C), parity (0=even, default 0)
        exact_k  : K (int)
        row_k    : K (int) — petals per ring carrying symbol 1
        latin_sq : none

    Returns
    -------
    np.ndarray (N, R, C) int  — values in [0, n_symbols)
    """
    if seed is not None:
        np.random.seed(seed)
    D, R, C = cfg.n_petals, cfg.n_rings, cfg.n_slots

    if rule == "parity":
        G = params["group_size"]
        p = params.get("parity", 0)
        if D % G:
            raise ValueError(f"group_size {G} must divide n_petals {D}")
        fn = lambda N: sample_group_parity_vec(N=N, sample_len=D, group_size=G, parity=p)
    elif rule == "exact_k":
        K = params["K"]
        fn = lambda N: sample_exact_k_vec(N=N, D=D, K=K)
    elif rule == "row_k":
        K = params["K"]
        fn = lambda N: _sample_row_exact_k(C, K, N=N * R).reshape(N, R * C)
    elif rule == "row_perm":
        if C != cfg.n_symbols:
            raise ValueError(f"row_perm needs n_slots == n_symbols, got {C}, {cfg.n_symbols}")
        fn = lambda N: np.stack([
            np.concatenate([np.random.permutation(C) for _ in range(R)])
            for _ in range(N)])
        flat = sample_ensuring_uniqueness(N, fn) if unique else fn(N)
        return np.asarray(flat).reshape(N, R, C)
    elif rule == "latin_sq":
        if not (R == C == cfg.n_symbols):
            raise ValueError(f"latin_sq needs n_rings == n_slots == n_symbols, "
                             f"got {R}, {C}, {cfg.n_symbols}")
        flat = (sample_ensuring_uniqueness(N, lambda N: sample_latin_square_vec(N, R))
                if unique else sample_latin_square_vec(N, R))
        return np.asarray(flat).reshape(N, R, C)
    else:
        raise ValueError(f"unknown rule: {rule}")

    sup = support_size(rule, cfg, **params)
    if unique and N > sup:
        raise ValueError(f"cannot draw {N} unique latents: support is only {sup}")
    flat = sample_ensuring_uniqueness(N, fn) if unique else fn(N)
    return _as_grid(flat, cfg)


def check_latents(latent, rule, cfg, **params):
    """Rule validity of latent grids.

    Cells equal to -1 (decode failure) make the whole sample invalid.

    Parameters
    ----------
    latent : (N, R, C) int
    rule, cfg, params : see `sample_latents`

    Returns
    -------
    np.ndarray (N,) bool
    """
    x = np.asarray(latent)
    N = x.shape[0]
    bad = (x < 0).reshape(N, -1).any(axis=1)
    flat = x.reshape(N, -1)

    if rule == "parity":
        G = params["group_size"]
        p = params.get("parity", 0)
        pm1 = np.where(flat > 0, 1, -1).reshape(N, -1, G)
        prod = pm1.prod(axis=2)                       # (N, D/G) in {-1,+1}
        ok = (prod == (-1) ** p).all(axis=1)
    elif rule == "exact_k":
        ok = (flat == 1).sum(axis=1) == params["K"]
    elif rule == "row_k":
        ok = ((x == 1).sum(axis=2) == params["K"]).all(axis=1)
    elif rule == "row_perm":
        srt = np.sort(x, axis=2)
        ok = (srt == np.arange(cfg.n_slots)[None, None, :]).all(axis=(1, 2))
    elif rule == "latin_sq":
        ok = check_latin_square_batch(flat, cfg.n_rings)
    else:
        raise ValueError(f"unknown rule: {rule}")

    return np.asarray(ok) & ~bad


def rule_subunit_ok(latent, rule, cfg, **params):
    """Per-subunit satisfaction of the rule — the sub-rule diagnostic.

    A sample can fail the rule while satisfying most of its local constraints;
    tracking the subunit rate separates "learned the constraint but not
    globally" from "learned nothing", exactly as PerGroup_Accuracy does for
    synthetic parity.

    Subunits are parity groups (parity), whole samples (exact_k), or rings
    (row_k / row_perm / latin_sq).

    Parameters
    ----------
    latent : (N, R, C) int  — -1 cells count as failures

    Returns
    -------
    np.ndarray (N, n_units) bool
    """
    x = np.asarray(latent)
    N = x.shape[0]
    flat = x.reshape(N, -1)
    bad = (flat < 0).any(axis=1, keepdims=True)

    if rule == "parity":
        G = params["group_size"]
        p = params.get("parity", 0)
        pm1 = np.where(flat > 0, 1, -1).reshape(N, -1, G)
        ok = pm1.prod(axis=2) == (-1) ** p
    elif rule == "exact_k":
        ok = ((flat == 1).sum(axis=1) == params["K"])[:, None]
    elif rule == "row_k":
        ok = (x == 1).sum(axis=2) == params["K"]
    elif rule == "row_perm":
        ok = (np.sort(x, axis=2) == np.arange(cfg.n_slots)[None, None, :]).all(axis=2)
    elif rule == "latin_sq":
        n = cfg.n_rings
        ref = np.arange(n)[None, None, :]
        ok = np.concatenate([(np.sort(x, axis=2) == ref).all(axis=2),
                             (np.sort(x, axis=1).transpose(0, 2, 1) == ref).all(axis=2)],
                            axis=1)
    else:
        raise ValueError(f"unknown rule: {rule}")

    return np.asarray(ok) & ~bad


def support_size(rule, cfg, **params):
    """Number of distinct latent grids satisfying `rule` (one channel)."""
    D, R, C = cfg.n_petals, cfg.n_rings, cfg.n_slots
    if rule == "parity":
        G = params["group_size"]
        return (2 ** (G - 1)) ** (D // G)
    if rule == "exact_k":
        return comb(D, params["K"])
    if rule == "row_k":
        return comb(C, params["K"]) ** R
    if rule == "row_perm":
        return factorial(C) ** R
    if rule == "latin_sq":
        return latin_valid_set_size(R)
    raise ValueError(f"unknown rule: {rule}")


def corrupt_latents(latent, rule, cfg, n_edit=1, seed=0, **params):
    """Produce rule-*violating* latents by minimally editing valid ones.

    Used to verify the CV read-out can *detect* invalidity rather than merely
    accepting whatever it sees.

    Parameters
    ----------
    latent : (N, R, C) int — assumed valid
    n_edit : int — cells edited per sample
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(latent).copy()
    N, R, C = out.shape
    for i in range(N):
        for _ in range(n_edit):
            r, c = rng.integers(R), rng.integers(C)
            cur = out[i, r, c]
            choices = [v for v in range(cfg.n_symbols) if v != cur]
            out[i, r, c] = choices[rng.integers(len(choices))]
    still_valid = check_latents(out, rule, cfg, **params)
    return out, still_valid
