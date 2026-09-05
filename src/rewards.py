"""Verifiable rewards for GSM8K/MATH: correctness + format + length control."""

import re

THINK_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def extract_answer(text: str) -> str:
    # GSM8K answers end with "#### <number>"; MATH uses \boxed{}
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    return text.strip().split()[-1][:50] if text.strip() else ""


def correctness_reward(completions, solution, **kwargs) -> list[float]:
    """2.0 if extracted answer matches gold, else 0.0. Uses math-verify if available."""
    try:
        from math_verify import parse, verify

        use_mv = True
    except ImportError:
        use_mv = False
    out = []
    for comp, gold in zip(completions, solution, strict=False):
        txt = comp[0]["content"] if isinstance(comp, list) else comp
        pred = extract_answer(txt)
        gold_txt = str(gold[0] if isinstance(gold, list) else gold)
        gold_ans = extract_answer(gold_txt)
        if use_mv:
            try:
                ok = verify(parse(f"${gold_ans}$"), parse(f"${pred}$"))
            except Exception:
                ok = pred == gold_ans
        else:
            ok = pred == gold_ans
        out.append(2.0 if ok else 0.0)
    return out


def format_reward(completions, **kwargs) -> list[float]:
    return [
        0.5 if THINK_RE.search(c[0]["content"] if isinstance(c, list) else c) else 0.0
        for c in completions
    ]


def length_penalty(completions, max_len: int = 512, weight: float = 0.3, **kwargs) -> list[float]:
    """Linear penalty past 60% of budget — phase 2 only. Keeps reasoning concise."""
    out = []
    for c in completions:
        txt = c[0]["content"] if isinstance(c, list) else c
        num_words = len(txt.split())
        over = max(0, num_words - int(0.6 * max_len)) / max_len
        out.append(-weight * over)
    return out
