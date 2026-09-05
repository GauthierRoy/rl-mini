"""Rewards for countdown. All signatures compatible with TRL GRPOTrainer."""

import re

from envs.countdown import check_equation, extract_answer


def correctness_reward(completions, numbers, target, **kwargs) -> list[float]:
    out = []
    for comp, nums, tgt in zip(completions, numbers, target, strict=False):
        txt = comp[0]["content"] if isinstance(comp, list) else comp
        eq = extract_answer(txt)
        ok, _ = check_equation(list(nums), int(tgt), eq)
        out.append(2.0 if ok else 0.0)
    return out


def format_reward(completions, **kwargs) -> list[float]:
    pat = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    out = []
    for c in completions:
        txt = c[0]["content"] if isinstance(c, list) else c
        out.append(0.5 if pat.search(txt) else 0.0)
    return out
