"""Original binary membership implementation; assumes finite {-1,+1} inputs.

The packed integer implementation is for the original small-D tasks; callers
must not use it as a general arbitrary-dimensional exact-match routine.
"""
import torch as th

def compute_membership_counts(train_int: th.Tensor,
                              gen_int: th.Tensor,
                              group_size: int):
    """
    Args:
        train_int:    Tensor of shape [N_train, D], values in {−1, +1}
        gen_int:      Tensor of shape [N_gen, D], values in {−1, +1}
        group_size:   int, width of each bit‐group (must divide D)

    Returns a dict with:
        sample_mem_num    – how many full samples appear in the train set
        sample_mem_ratio  – sample_mem_num / N_gen
        bitgroup_mem_num  – how many bit‐groups appear in the train set
        bitgroup_mem_ratio– bitgroup_mem_num / (N_gen * D / group_size)
    """
    # ---- full‐sample codes ----
    N_gen, D = gen_int.shape

    # map −1→0, +1→1
    train_bits = (train_int > 0).long()  # [N_train, D]
    gen_bits   = (gen_int   > 0).long()  # [N_gen,   D]

    # build powers‐of‐two vector [1,2,4,…,2^(D−1)]
    weights = 1 << th.arange(D, device=train_int.device, dtype=th.long)

    # pack each row into a single integer code
    train_codes = (train_bits * weights).sum(dim=1).unique()
    gen_codes   = (gen_bits   * weights).sum(dim=1)

    # vectorized membership test
    mask = th.isin(gen_codes, train_codes)
    sample_mem_num   = int(mask.sum().item())
    sample_mem_ratio = sample_mem_num / N_gen

    # ---- bit‐group codes ----
    # reshape into rows of width `group_size`
    BG_train = train_bits.view(-1, group_size)  # [N_train * (D/group_size), group_size]
    BG_gen   = gen_bits.view(-1,   group_size)  # [N_gen   * (D/group_size), group_size]

    # powers‐of‐two for each group
    gw = 1 << th.arange(group_size, device=train_int.device, dtype=th.long)

    train_group_codes = (BG_train * gw).sum(dim=1).unique()
    gen_group_codes   = (BG_gen   * gw).sum(dim=1)

    mask2 = th.isin(gen_group_codes, train_group_codes)
    bitgroup_mem_num   = int(mask2.sum().item())
    bitgroup_mem_ratio = bitgroup_mem_num / gen_group_codes.numel()

    return {
        "sample_mem_num":    sample_mem_num,
        "sample_mem_ratio":  sample_mem_ratio,
        "bitgroup_mem_num":  bitgroup_mem_num,
        "bitgroup_mem_ratio": bitgroup_mem_ratio,
    }
