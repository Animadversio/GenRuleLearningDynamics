"""Legacy completion parsing and language scoring."""
import re
import numpy as np
from .task import SwitchTask, FLIP, KEEP, check, support_size, chance_rate

_STRIP = re.compile(r"[\s`'\"*.,;:!?()\[\]{}<>|_-]+")


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Return (answer segment, finished).

    A reasoning model emits `<think> ... </think>` before its answer.  Only the
    part *after* the closing tag is an answer; the reasoning itself is full of
    candidate strings the model is still evaluating.  If the block was opened and
    never closed the generation was truncated, so there is no answer at all --
    scraping T/F runs out of unfinished reasoning silently manufactures scores,
    which is exactly what happened when a 1536-token budget was too small for the
    longer prompts (100% of them hit the cap, 0% closed the tag, and the metric
    still reported 3-31% "correct").
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], True
    if "<think>" in text:
        return "", False                # opened, never closed => truncated
    return text, True


def extract_answer(text: str, task: SwitchTask, lenient: bool = True) -> str | None:
    """Pull the candidate T/F string out of a model completion.

    `lenient=False` demands the completion be *exactly* the string, which is what
    the prompt asks for.  `lenient=True` additionally strips whitespace and
    punctuation and, failing that, takes the last run of D T/F characters
    anywhere in the text -- so that "reasoning then answering" is scored on the
    rule rather than on obedience.  Reporting both separates the two failures.

    Reasoning is removed first in both cases; see `strip_reasoning`.
    """
    text, finished = strip_reasoning(text)
    if not finished:
        return None
    raw = text.strip()
    if task.answer_format == "spaced":
        # strict == exactly the D letters, single-space separated
        parts = raw.split(" ")
        if len(parts) == task.n_people and all(p in (FLIP, KEEP) for p in parts):
            return "".join(parts)
        if not lenient:
            return None
    elif len(raw) == task.n_people and set(raw) <= {FLIP, KEEP}:
        return raw
    if not lenient:
        return None
    squeezed = _STRIP.sub("", raw)
    if len(squeezed) == task.n_people and set(squeezed) <= {FLIP, KEEP}:
        return squeezed
    # Maximal runs only, and the run must be exactly D long.  Taking "the last
    # window of D" out of a longer run manufactures answers from output that
    # never terminated: at dressing >= 1 the model emits an unbounded
    # "T F T F ..." stream and hits the token cap at ~80 letters having committed
    # to nothing, yet a sliding window always finds some valid 36-substring.
    runs = re.findall(f"[{FLIP}{KEEP}]+", squeezed)
    exact = [r for r in runs if len(r) == task.n_people]
    if len(exact) == 1 and not any(len(r) > task.n_people for r in runs):
        return exact[0]
    return None


def verify(text: str, task: SwitchTask) -> dict:
    """Score one completion.  Format and rule are reported separately."""
    _, finished = strip_reasoning(text)
    strict = extract_answer(text, task, lenient=False)
    loose = extract_answer(text, task, lenient=True)
    return dict(
        truncated=not finished,
        format_ok=strict is not None,           # obeyed "exactly D characters"
        parsed_ok=loose is not None,            # an answer was recoverable
        answer=loose,
        rule_ok=bool(loose is not None and check(loose, task)),
        # the headline number: obeyed the format AND the rule
        strict_ok=bool(strict is not None and check(strict, task)),
    )


def minimal_period(s: str) -> int:
    """Smallest p with s[i] == s[i % p] for all i; == len(s) when aperiodic.

    The model's failure mode is emitting a short repeating template rather than
    deriving an answer, and templates satisfy group-parity rules by accident.
    Period is the statistic that separates the two: uniform draws from the valid
    support are aperiodic, so a median period below D means the cell is measuring
    the template, not the rule.
    """
    n = len(s)
    # Only p <= n/2 counts: at p = n-4 just four positions are tested, so random
    # strings pass by chance (a random 36-bit string scored "period 32").  A
    # period is only evidence of a template if it repeats at least twice.
    for p in range(1, n // 2 + 1):
        # NOT requiring p to divide n: a template truncated mid-repeat is still a
        # template.  "FFTTFTFT" x4.5 (36 chars) is obviously periodic, but with a
        # divisibility test it scored period 36 and passed as aperiodic.
        if all(s[i] == s[i % p] for i in range(n)):
            return p
    return n


def score_batch(texts, task: SwitchTask) -> dict:
    """Aggregate over completions, including the diversity of what was produced.

    `coverage` is the fraction of the *valid support* actually visited, and
    `top1_share` the frequency of the single most common valid answer -- the two
    numbers that reveal mode collapse, which is the failure mode reinforcement
    learning with a verifiable reward is known to induce.
    """
    res = [verify(t, task) for t in texts]
    n = max(1, len(res))
    valid = [r["answer"] for r in res if r["rule_ok"]]
    uniq = set(valid)
    top1 = max((valid.count(a) for a in uniq), default=0)
    top1_answer = max(uniq, key=valid.count) if uniq else None
    sup = support_size(task)
    # A PERIODIC string satisfies "even flips per group" whenever its period
    # divides the group and repeats an even number of flips -- e.g. TFTFTF...
    # has two flips in every block of four, so it solves G=4 for free, and all-F
    # solves every G.  Periodic strings are exactly this model's default output,
    # so a cell can show a 464x lift containing no rule-following whatsoever.
    # Measure the minimal period of every valid answer: a genuine rule-follower
    # sampling the support produces aperiodic strings (period == D).
    periods = [minimal_period(a) for a in valid]
    triv = sum(p < len(a) for p, a in zip(periods, valid))
    return dict(
        n=len(res),
        truncated=sum(r["truncated"] for r in res) / n,
        format_ok=sum(r["format_ok"] for r in res) / n,
        parsed_ok=sum(r["parsed_ok"] for r in res) / n,
        rule_ok=sum(r["rule_ok"] for r in res) / n,
        strict_ok=sum(r["strict_ok"] for r in res) / n,
        chance=chance_rate(task),
        lift=(sum(r["rule_ok"] for r in res) / n) / chance_rate(task),
        n_unique_valid=len(uniq),
        coverage=len(uniq) / sup,
        top1_share=top1 / max(1, len(valid)),
        top1_answer=top1_answer,
        # fraction of VALID outputs that are periodic, i.e. templated not derived
        trivial_share=triv / max(1, len(valid)),
        median_period=float(np.median(periods)) if periods else float("nan"),
        support=sup,
    )
