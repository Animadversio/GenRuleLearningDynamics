"""Verbatim experimental prompts and answer formatting."""
from .task import SwitchTask

_NAMES = ("Ana", "Ben", "Cleo", "Dev", "Eve", "Femi", "Gus", "Hana",
          "Ivan", "Jo", "Kit", "Lena", "Milo", "Nia", "Omar", "Pia",
          "Quinn", "Rosa", "Sam", "Tara", "Uma", "Vik", "Wren", "Xu",
          "Yuki", "Zane", "Ada", "Bo", "Cass", "Dara", "Eli", "Fay",
          "Gil", "Hugo", "Ines", "Jae")


def _rule_clause(task: SwitchTask) -> str:
    """The rule, stated the way each dressing level wants it."""
    if task.rule == "exact_k":
        return f"exactly {task.K} of the {task.n_people} people flip the switch"
    if task.is_global_parity:
        end = "on" if task.parity_target else "off"
        return f"the light is {end} after the last person leaves"
    G = task.group_size
    return (f"the light is off again after every group of {G} consecutive "
            f"people has passed through")


def render_prompt(task: SwitchTask) -> str:
    """Build the user-facing prompt at this task's dressing level."""
    D, d = task.n_people, task.dressing
    tail = (f"Answer with exactly {D} characters and nothing else."
            if task.answer_format == "compact" else
            f"Write the {D} letters separated by single spaces, and nothing else.")

    if d == 0:                                   # bare: no story at all
        if task.rule == "exact_k":
            body = (f"Output a string of {D} characters, each T or F, "
                    f"containing exactly {task.K} occurrences of T.")
        elif task.is_global_parity:
            par = "odd" if task.parity_target else "even"
            body = (f"Output a string of {D} characters, each T or F, "
                    f"containing an {par} number of T.")
        else:
            body = (f"Output a string of {D} characters, each T or F, such that "
                    f"every block of {task.group_size} consecutive characters "
                    f"contains an even number of T.")
        return f"{body} {tail}"

    if d == 1:                                   # one sentence
        return (f"A light starts off and {D} people each may flip its switch "
                f"once, in order; write T for flip and F for no flip so that "
                f"{_rule_clause(task)}. {tail}")

    setup = (
        "A room contains a single light controlled by one switch. Flipping the "
        "switch toggles the light: on becomes off, off becomes on. "
        f"{_people_word(D).capitalize()} people enter the room one at a time, and "
        "each person may flip the switch at most once before leaving. The light "
        "is off before the first person enters"
    )
    if d == 2:                                   # the original paragraph
        cond = (f", and {_rule_clause(task)}")
        return (f"{setup}{cond}. Write the {_people_word(D)} actions in order, "
                f"using T if that person flips the switch and F if they do not. "
                f"{tail}")

    # d >= 3: same rule, buried in irrelevant detail
    names = ", ".join(_NAMES[:D])
    return (f"{setup}. The people, in order, are {names}. The room also contains "
            f"a bookshelf, a wooden chair, and a window that stays shut "
            f"throughout. Nobody speaks, and the door closes on its own after "
            f"each visit. Ignore everything except the switch: what matters is "
            f"only that {_rule_clause(task)}. Write the {_people_word(D)} "
            f"actions in order, using T if that person flips the switch and F "
            f"if they do not. {tail}")


_WORDS = {8: "eight", 12: "twelve", 16: "sixteen", 24: "twenty-four",
          36: "thirty-six", 6: "six", 4: "four", 10: "ten", 20: "twenty"}


def _people_word(D: int) -> str:
    return _WORDS.get(D, str(D))


def format_answer(answer: str, task: SwitchTask) -> str:
    """Render a bare T/F string the way this task asks for it.

    The spaced form carries a LEADING space on purpose.  Qwen's BPE merges a
    letter with the space before it, so " T T F" is the two tokens {' T',' F'}
    repeated, one per position -- but "T T F" makes position 0 the distinct token
    'T', giving three token types and an asymmetric first position.  Measured
    over 200 random strings, the leading-space form is exactly D tokens from a
    2-token set every time.

    Digits do NOT work: '0'/'1' do not merge with the preceding space, so
    "0 1 0 1..." costs 71 tokens for 36 bits.  Words do (' on'/' off',
    ' yes'/' no' are each single tokens), if a more natural pair is ever wanted.
    """
    return answer if task.answer_format == "compact" else \
        "".join(" " + c for c in answer)
