"""Rewards for countdown. All signatures compatible with TRL GRPOTrainer."""

import re

from envs.countdown import check_equation


def extract_equation(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[1]
        lines = [ln.strip() for ln in tail.splitlines() if re.search(r"\d", ln)]
        if lines:
            eq = lines[-1].split("=")[0]
            return re.sub(r"^[^0-9(\-]*", "", eq).strip()
    return ""


def correctness_reward(completions, numbers, target, **kwargs) -> list[float]:
    out = []
    for comp, nums, tgt in zip(completions, numbers, target, strict=False):
        txt = comp[0]["content"] if isinstance(comp, list) else comp
        eq = extract_equation(txt)
        ok, _ = check_equation(list(nums), int(tgt), eq)
        out.append(2.0 if ok else 0.0)
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
