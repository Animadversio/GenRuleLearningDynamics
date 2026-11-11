"""Metrics with explicit sample counts and denominators."""
from dataclasses import dataclass
import numpy as np


def exact_match_mask(train, generated):
    """Exact row membership for finite numeric latent sequences of any dimension.

    NaN/Inf generated rows never match. Inputs are already decoded sequences;
    floating image similarity is a separate metric. Training duplicates do not
    change membership. Unlike the legacy packed-binary implementation, this
    function also supports dimensions above 64.
    """
    train, generated = np.asarray(train), np.asarray(generated)
    if train.ndim != 2 or generated.ndim != 2 or train.shape[1] != generated.shape[1]:
        raise ValueError('train and generated must be matrices with the same width')
    if not np.isfinite(train).all():
        raise ValueError('training sequences must be finite')
    known = {tuple(row) for row in train}
    return np.array([bool(np.isfinite(row).all()) and tuple(row) in known
                     for row in generated], dtype=bool)


@dataclass(frozen=True)
class OutputMetrics:
    n_total: int
    n_valid: int
    n_train_match: int
    validity: float
    training_match_all: float
    training_match_valid: float
    novel_valid_yield: float


def summarize_outputs(valid, train_match):
    """Summarize sample masks, assuming training examples satisfy the rule.

    `valid` must already include the applicable parsing/readability gate.
    Empty denominators produce NaN rather than pretending an estimate is zero.
    """
    valid, train_match = np.asarray(valid, dtype=bool), np.asarray(train_match, dtype=bool)
    if valid.ndim != 1 or valid.shape != train_match.shape:
        raise ValueError('valid and train_match must be matching one-dimensional masks')
    if np.any(train_match & ~valid):
        raise ValueError('a match to valid training examples cannot be rule-invalid')
    total, nvalid, nmatch = len(valid), int(valid.sum()), int(train_match.sum())
    ratio = lambda numerator, denominator: numerator / denominator if denominator else float('nan')
    return OutputMetrics(total, nvalid, nmatch, ratio(nvalid,total),
                         ratio(nmatch,total), ratio(nmatch,nvalid), ratio(nvalid-nmatch,total))
