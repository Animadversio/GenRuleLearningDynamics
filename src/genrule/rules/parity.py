import numpy as np

def sample_parity(K, parity=0):
    """
    Sample x in {-1,1}^K *uniformly* subject to
    prod_i x[i] = (-1)**parity.

    Parameters
    ----------
    K : int
        Length of vector.
    parity : {0,1}
        0 ⇒ even number of -1’s ⇒ prod = +1
        1 ⇒ odd number of -1’s  ⇒ prod = -1

    Returns
    -------
    x : np.ndarray of shape (K,), dtype=int
        A random vector satisfying the parity constraint.
    """
    # sample first K-1 entries freely
    x = np.random.choice([-1,1], size=K)
    # compute their product
    p = x[:-1].prod()
    # pick the last entry so that overall product = (-1)**parity
    xK = ((-1)**parity) // p
    x[-1] = xK
    # concatenate and return
    return x


def sample_parity_vec(N, K, parity=0):
    x = np.random.choice([-1,1], size=(N, K))
    pk_1 = x[:, :-1].prod(axis=1)
    xK = ((-1)**parity) // pk_1
    x[:, -1] = xK
    return x


def sample_group_parity_vec(N, sample_len, group_size=1, parity=0):
    num_groups = sample_len // group_size
    x = sample_parity_vec(N=N * num_groups, K=group_size, parity=parity)
    x = x.reshape(N, num_groups * group_size)
    return x


def parity_func(x, axis=0):
    parity = x.prod(axis=axis) # this is -1, 1 -1 is odd, mapping to 1, 1 is even mapping to 0
    return (- parity + 1) // 2



def sample_ensuring_uniqueness(N, sample_func):
    collected_samples = []
    seen = set()

    while len(collected_samples) < N:
        # Sample more than needed to reduce iterations
        batch_size = max(N - len(collected_samples), N // 2)
        x_batch = sample_func(N=batch_size)
        # Convert batch to tuples for set operations
        batch_tuples = set(tuple(row) for row in x_batch)
        # Find new unique samples
        new_samples = batch_tuples - seen
        # Update seen set
        seen.update(new_samples)
        # Add new samples to collected samples
        for sample_tuple in new_samples:
            collected_samples.append(np.array(sample_tuple))
            if len(collected_samples) == N:
                break

    return np.array(collected_samples)

def round_to_pos_neg_one(x, eps = 1e-2):
    import torch as th
    result = th.full_like(x, float('nan'))
    result[th.abs(x - 1) < eps] = 1.0
    result[th.abs(x - (-1)) < eps] = -1
    return result


# ── row-prefix parity ─────────────────────────────────────────────────────────
# Standard group parity ties the number of constraints to the group size: with
# D=36, G=2 imposes 18 constraints and G=6 imposes 6, so changing G changes the
# rule degree, the constraint count AND the support size all at once
# (support = 2^(D - D/G), which runs from 2^18 at G=2 to 2^30 at G=6).
#
# Row-prefix parity decouples them.  Lay the D bits out as `n_rows` rows of
# `row_len`.  In each row the FIRST `group_size` bits must have the specified
# parity; the remaining `row_len - group_size` bits are free.  Then
#
#     number of constraints = n_rows          (independent of group_size)
#     support               = 2^(D - n_rows)  (independent of group_size)
#     chance validity       = 2^-n_rows       (independent of group_size)
#
# so a sweep over `group_size` varies ONLY the degree of each constraint.

def sample_row_prefix_parity_vec(N, n_rows, row_len, group_size, parity=0,
                                 rng=None):
    """N x (n_rows*row_len) in {-1,+1}; first `group_size` bits of each row
    satisfy the parity constraint, the rest are uniform."""
    if not 1 <= group_size <= row_len:
        raise ValueError(f"group_size {group_size} must be in [1, {row_len}]")
    rng = np.random.default_rng() if rng is None else rng
    x = rng.choice([-1, 1], size=(N, n_rows, row_len))
    # fix the last bit of each constrained prefix to enforce the parity
    if group_size == 1:
        x[:, :, 0] = (-1) ** parity
    else:
        pref = x[:, :, :group_size - 1].prod(axis=2)
        x[:, :, group_size - 1] = ((-1) ** parity) * pref
    return x.reshape(N, n_rows * row_len)


def row_prefix_parity_check(x, n_rows, row_len, group_size, parity=0):
    """Per-row satisfaction of the prefix constraint.

    `x` is (..., n_rows*row_len) in {-1,+1} (NaN allowed; a NaN anywhere in a
    prefix makes that row unsatisfied).  Returns a boolean array (..., n_rows).
    """
    x = np.asarray(x)
    rows = x.reshape(*x.shape[:-1], n_rows, row_len)
    pref = rows[..., :group_size]
    prod = pref.prod(axis=-1)
    return prod == ((-1) ** parity)


def row_prefix_support(n_rows, row_len, group_size):
    """Number of distinct valid vectors.  Independent of `group_size`."""
    return 2 ** (n_rows * row_len - n_rows)
