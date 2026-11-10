from __future__ import annotations
import itertools
from dataclasses import dataclass
from math import comb
import numpy as np

FLIP, KEEP = "T", "F"


@dataclass(frozen=True)
class SwitchTask:
    """One instance family: D people, a rule, and a way of phrasing it."""

    n_people: int = 12
    #: "parity"  -- each group of `group_size` consecutive people must contain an
    #:              even number of flips (group_size = n_people is the global
    #:              odd-parity form of the original prompt, see `parity_target`)
    #: "exact_k" -- exactly `K` people flip, in total
    rule: str = "parity"
    group_size: int = 12
    K: int = 3
    #: For global parity the light must end ON, i.e. an ODD number of flips.
    #: For grouped parity each group must return the light to its previous
    #: state, i.e. an EVEN number of flips per group.
    parity_target: int = 1          # 1 = odd total (light ends on), 0 = even
    #: 0 bare instruction, 1 one-sentence story, 2 full paragraph, 3 + distractors
    dressing: int = 2
    #: "compact" -> "TFTF..."   "spaced" -> "T F T F ..."
    #:
    #: This is not cosmetic.  Qwen3's BPE merges runs of T/F, so a compact
    #: answer is 16-19 tokens for 36 characters and the degenerate answers are
    #: the cheapest sequences in the space: all-F is FIVE tokens (FFFFFFFF x4 +
    #: FFFF) and TFTF... is the single token "FT" repeated seventeen times.  The
    #: model cannot express "the 7th character" as a decision at all, and a
    #: repetition-biased decoder gets the periodic templates for free.
    #:
    #: Spacing makes it exactly one token per position (verified: 36 tokens for
    #: 36 letters), which is also what the DiT and GPT parity benchmarks in this
    #: repo do -- so only "spaced" is comparable to them.
    answer_format: str = "compact"

    def __post_init__(self):
        if self.answer_format not in ("compact", "spaced"):
            raise ValueError(f"unknown answer_format {self.answer_format!r}")
        if self.rule not in ("parity", "exact_k"):
            raise ValueError(f"unknown rule {self.rule!r}")
        if self.rule == "parity" and self.n_people % self.group_size:
            raise ValueError(f"group_size {self.group_size} must divide "
                             f"n_people {self.n_people}")
        if self.rule == "exact_k" and not 0 <= self.K <= self.n_people:
            raise ValueError(f"K={self.K} out of range for {self.n_people} people")

    @property
    def is_global_parity(self) -> bool:
        return self.rule == "parity" and self.group_size == self.n_people


def check(answer: str, task: SwitchTask) -> bool:
    """Does this T/F string satisfy the rule?  Assumes the format is already valid."""
    if task.rule == "exact_k":
        return answer.count(FLIP) == task.K
    G = task.group_size
    want = task.parity_target if task.is_global_parity else 0
    return all(answer[i:i + G].count(FLIP) % 2 == want
               for i in range(0, task.n_people, G))


def support_size(task: SwitchTask) -> int:
    """How many of the 2^D strings satisfy the rule."""
    if task.rule == "exact_k":
        return comb(task.n_people, task.K)
    G, n_groups = task.group_size, task.n_people // task.group_size
    return (2 ** (G - 1)) ** n_groups          # one parity bit fixed per group


def chance_rate(task: SwitchTask) -> float:
    """Validity of a uniformly random T/F string -- the baseline to beat."""
    return support_size(task) / 2 ** task.n_people


def enumerate_valid(task: SwitchTask):
    """All valid answers.  Only tractable for small D; used for coverage checks."""
    if task.n_people > 20:
        raise ValueError("refusing to enumerate 2^%d strings" % task.n_people)
    return [s for s in ("".join(p) for p in
                        itertools.product(FLIP + KEEP, repeat=task.n_people))
            if check(s, task)]


def sample_valid(task: SwitchTask, n: int, rng=None, unique: bool = True):
    """Draw valid answers, for building supervised fine-tuning targets."""
    rng = np.random.default_rng(0) if rng is None else rng
    out, seen = [], set()
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 200 * n:
            raise RuntimeError(f"only found {len(out)}/{n}; support is "
                               f"{support_size(task):,}")
        bits = rng.integers(0, 2, task.n_people)
        if task.rule == "exact_k":
            bits[:] = 0
            bits[rng.choice(task.n_people, task.K, replace=False)] = 1
        else:
            G = task.group_size
            want = task.parity_target if task.is_global_parity else 0
            for i in range(0, task.n_people, G):        # fix the last bit of each group
                if bits[i:i + G].sum() % 2 != want:
                    bits[i + G - 1] ^= 1
        s = "".join(FLIP if b else KEEP for b in bits)
        if unique and s in seen:
            continue
        seen.add(s); out.append(s)
    return out
