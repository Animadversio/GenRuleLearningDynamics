"""Public Light-switch interface; implementation split into task/prompts/parsing."""
from .task import SwitchTask, FLIP, KEEP, check, support_size, chance_rate, enumerate_valid, sample_valid
from .prompts import render_prompt, format_answer
from .parsing import strip_reasoning, extract_answer, verify, minimal_period, score_batch
