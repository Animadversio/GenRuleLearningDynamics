"""Explicitly named legacy onset algorithms and paper image-audit protocol."""
from dataclasses import dataclass
import numpy as np

def first_sustained_crossing(steps, vals, threshold, n_consec=5, above=True):
    """Return the first step at which `vals` crosses `threshold` for
    `n_consec` consecutive evaluation points.

    Parameters
    ----------
    steps    : array-like of int   — eval step numbers
    vals     : array-like of float — metric values at each step
    threshold: float
    n_consec : int                 — required consecutive crossings (default 5)
    above    : bool                — True = look for vals > threshold (default)

    Returns
    -------
    float : the step number at the *start* of the first sustained run,
            or np.nan if the threshold is never sustained.
    """
    steps = np.asarray(steps)
    vals  = np.asarray(vals)
    mask  = vals > threshold if above else vals < threshold
    count = 0
    for i, m in enumerate(mask):
        if m:
            count += 1
            if count >= n_consec:
                return float(steps[i - n_consec + 1])
        else:
            count = 0
    return np.nan


def mem_threshold(n_train, support, n_draws=1024, k_hits=3, cap=0.9):
    """Legacy threshold arithmetic: min(cap, max(2*N/support, k_hits/n_draws)).

    For the paper extensions, use the default fixed floor 3/1024, even for
    2048-draw evaluations. This is not a three-match constraint for conditional
    ratios. `n_train` assumes distinct valid training examples.
    """
    return min(cap, max(2.0 * n_train / support, k_hits / float(n_draws)))

def median_smooth(vals, window=3):
    """Running median, edge-padded.  Removes isolated single-point excursions
    while leaving sustained departures untouched (a window of 3 survives any
    run of 2+ consecutive points)."""
    vals = np.asarray(vals, float)
    if window < 3 or len(vals) < window:
        return vals
    h = window // 2
    padded = np.concatenate([np.repeat(vals[0], h), vals, np.repeat(vals[-1], h)])
    return np.array([np.median(padded[i:i + window]) for i in range(len(vals))])


def legacy_robust_onset(steps, vals, thresh=0.5, min_step=300, median_window=3):
    """First step after which `vals` NEVER drops back below `thresh`.

    A run that only touches the threshold and falls back has not learned the
    rule -- the 2x rank-deficient flower rung oscillates across 0.5 for 210k
    steps and must not register an onset.  But a *single* eval dipping below is
    sampling noise, not a regression, and taking it at face value made the
    synthetic exact-K onsets swing by up to 1244x across thresholds.

    Measured dip structure, counting only evals after the curve first reaches
    0.9: synthetic exact-K K = 4 / 6 / 12 / 18 have 1 / 0 / 2 / 2 sub-threshold
    excursions out of ~1250 evals, and **every one of them is a single point**.
    A running median of width 3 removes exactly those and nothing else: any run
    of 2 or more consecutive sub-threshold points survives it.

    `median_window=1` disables the smoothing.

    Returns nan if the metric is below `thresh` at the final eval.
    """
    steps = np.asarray(steps)
    vals = median_smooth(vals, median_window)
    last_bad = -1
    for i in range(len(vals)):
        if steps[i] >= min_step and vals[i] < thresh:
            last_bad = i
    j = last_bad + 1
    return float(steps[j]) if j < len(steps) else float("nan")

@dataclass(frozen=True)
class OnsetResult:
    """Observed time, censoring budget and explicit observation status."""
    onset: float | None
    status: str
    budget: float | None


def withhold_small_valid_counts(values, n_valid, minimum=30):
    """Withhold conditional estimates before smoothing, as in the image audit."""
    values, counts = np.asarray(values, float), np.asarray(n_valid, float)
    if values.shape != counts.shape:
        raise ValueError('values and valid counts must have matching shapes')
    return np.where(np.isfinite(counts) & (counts >= minimum), values, np.nan)


def persistence_onset(steps, values, threshold, *, min_step=0, median_window=3):
    """Reproduce the Block/Orb image-audit persistence criterion.

    Uses legacy centered, repeated-edge median smoothing. Nonfinite smoothed
    values fail persistence. A missing final value means incomplete evidence,
    not an observed failure. Short curves retain legacy smoothing behavior.
    """
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    if steps.ndim != 1 or values.shape != steps.shape:
        raise ValueError('steps and values must be matching one-dimensional arrays')
    if not np.isfinite(steps).all() or np.any(np.diff(steps) <= 0):
        raise ValueError('steps must be finite and strictly increasing')
    if not np.isfinite(threshold) or not np.isfinite(min_step):
        raise ValueError('threshold and min_step must be finite')
    if not isinstance(median_window, (int, np.integer)) or median_window < 1 or median_window % 2 != 1:
        raise ValueError('median_window must be a positive odd integer')
    budget = float(steps[-1]) if len(steps) else None
    smoothed = median_smooth(values, median_window)
    keep = steps >= min_step
    steps, smoothed = steps[keep], smoothed[keep]
    if not len(steps) or not np.isfinite(smoothed[-1]):
        return OnsetResult(None, 'missing_data', budget)
    bad = np.flatnonzero(~np.isfinite(smoothed) | (smoothed < threshold))
    first = int(bad[-1]) + 1 if len(bad) else 0
    if first == len(steps):
        return OnsetResult(None, 'right_censored', budget)
    return OnsetResult(float(steps[first]), 'observed', budget)
