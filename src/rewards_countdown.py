"""Rewards for countdown. All signatures compatible with TRL GRPOTrainer."""

import re

from envs.countdown import grade_equation


def candidate_equations(text: str) -> list[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return [m.group(1).strip()]
    tail = text.rsplit("</think>", 1)[1] if "</think>" in text else text
    eqs = []
    for chunk in re.split(r"[\n;]+", tail):
        if not re.search(r"\d", chunk):
            continue
        eq = re.sub(r"^[^0-9(\-]*", "", chunk.strip().split("=")[0]).strip()
        if eq:
            eqs.append(eq)
    return eqs


def extract_equation(text: str) -> str:
    eqs = candidate_equations(text)
    return eqs[-1] if eqs else ""


def correctness_reward(completions, numbers, target, **kwargs) -> list[float]:
    """Correctness-only with arithmetic shaping (best candidate wins):
    2.0 exact solve, 0.3 exact numbers but wrong value, 0.1 valid equation
    with allowed numbers, else 0. No format requirements."""
    out = []
    for comp, nums, tgt in zip(completions, numbers, target, strict=False):
        txt = comp[0]["content"] if isinstance(comp, list) else comp
        best = 0.0
        for eq in candidate_equations(txt):
            score, _ = grade_equation(list(nums), int(tgt), eq)
            best = max(best, score)
            if best == 2.0:
                break
        out.append(best)
    return out


def overlong_penalty(completions, **kwargs) -> list[float]:
    """DAPO-style soft band: free below ~175 tok, linear to -0.5 at the cap.
    Word counts approximate tokens (~0.75x); truncated rollouts are masked
    from the loss separately via mask_truncated_completions."""
    out = []
    for c in completions:
        txt = c[0]["content"] if isinstance(c, list) else c
        n = len(txt.split())
        if n <= 130:
            out.append(0.0)
        else:
            out.append(-0.5 * min(1.0, (n - 130) / 60))
    return out


def format_reward(completions, **kwargs) -> list[float]:
    full = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    think = re.compile(r"<think>.*?</think>", re.DOTALL)
    answer = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    out = []
    for c in completions:
        txt = c[0]["content"] if isinstance(c, list) else c
        if full.search(txt):
            out.append(0.5)
        elif think.search(txt) and re.search(r"\d", txt.rsplit("</think>", 1)[1]):
            out.append(0.5)
        elif "<think>" in txt and "<answer>" in txt:
            out.append(0.3)
        elif think.search(txt) or "</think>" in txt:
            out.append(0.3)
        elif answer.search(txt):
            # Non-thinking rollouts start after a pre-closed <think> block,
            # so a lone <answer> block is full format compliance.
            out.append(0.5)
        elif "<think>" in txt or "<answer>" in txt:
            out.append(0.1)
        else:
            out.append(0.0)
    return out
