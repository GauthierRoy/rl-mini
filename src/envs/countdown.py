"""Countdown synthetic generator: infinite solvable tasks by reverse-generation.
Core synthetic-data lesson: generate solution first, then problem. Guarantees solvability.
"""

import ast
import logging
import random
import re

logger = logging.getLogger(__name__)


def _safe_eval(expr: str) -> float | None:
    # only digits, +-*/(), spaces
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
        logger.debug("rejected by charset: %r", expr[:100])
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        for n in ast.walk(tree):
            if not isinstance(
                n,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Constant,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.USub,
                    ast.UAdd,
                    ast.Load,
                ),
            ):
                logger.warning("rejected by AST (%s): %r", type(n).__name__, expr[:100])
                return None
        v = eval(compile(tree, "<e>", "eval"), {"__builtins__": {}})
        if isinstance(v, (int, float)) and abs(v) < 1e6:
            return float(v)
        logger.debug("rejected by range/type (%r): %r", v, expr[:100])
        return None
    except Exception as e:
        logger.debug("eval failed (%s): %r", e, expr[:100])
        return None


def generate_task(
    num_numbers: int = 4, low: int = 1, high: int = 10, target_fixed: int = 24
) -> dict:
    """Reverse-generate: sample numbers, sample random binary tree, compute target.
    If target != fixed, keep numbers but recompute reachable target (guaranteed solvable).
    For simplicity: sample numbers, brute-force search a reachable target near 24."""
    for _ in range(100):
        nums = [random.randint(low, high) for _ in range(num_numbers)]
        targets = reachable_targets(nums)
        if not targets:
            continue
        # prefer 24 if reachable, else closest
        tgt = 24 if 24 in targets else min(targets, key=lambda t: (abs(t - 24), t))
        return {"numbers": nums, "target": tgt}
    nums = [3, 5, 7, 10]
    return {"numbers": nums, "target": 24}


def reachable_targets(nums: list[int]) -> set[int]:
    """BFS over expression states. Returns integer targets reachable exactly."""
    # simpler: enumerate all binary trees via brute force for n<=4
    results = set()
    for expr_val in _enumerate(nums):
        if expr_val.denominator == 1 and 1 <= expr_val.numerator <= 100:
            results.add(int(expr_val))
    return results


def _enumerate(nums):
    from fractions import Fraction

    if len(nums) == 1:
        yield Fraction(nums[0])
        return
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
            a, b = Fraction(nums[i]), Fraction(nums[j])
            for _op, fn in [
                ("+", lambda x, y: x + y),
                ("*", lambda x, y: x * y),
                ("-", lambda x, y: x - y),
            ]:
                yield from _enumerate(rest + [float(fn(a, b))])
                # note: keep ints as floats ok for small search; exactness via Fraction below
            if b != 0 and a % b == 0:
                yield from _enumerate(rest + [float(a / b)])


def grade_equation(numbers: list[int], target: int, equation: str) -> tuple[float, str]:
    """2.0 exact solve; 0.3 exact numbers but wrong value; 0.1 parses using
    only allowed numbers; else 0.0. Echo-safe: the worked example uses
    out-of-range numbers so copying it scores 0 on every tier."""
    nums_used = list(map(int, re.findall(r"\d+", equation)))
    # multiset check
    from collections import Counter

    if not nums_used:
        return 0.0, "no numbers found"
    v = _safe_eval(equation)
    if v is None:
        return 0.0, "unparseable"
    if Counter(nums_used) == Counter(numbers):
        if abs(v - target) < 1e-6:
            return 2.0, "correct"
        return 0.3, f"evals to {v}, want {target}"
    if not (Counter(nums_used) - Counter(numbers)):
        return 0.1, "valid equation, wrong numbers"
    return 0.0, f"uses numbers outside {numbers}"


def check_equation(numbers: list[int], target: int, equation: str) -> tuple[bool, str]:
    """Verify: uses each given number exactly once, evals to target."""
    score, reason = grade_equation(numbers, target, equation)
    return score == 2.0, reason


PROMPT_TMPL = (
    "Using each of {numbers} exactly once with + - * / and parentheses, reach {target}.\n"
    "Write your final equation on the last line.\n"
    "Example: with [20, 8, 2]: (20-8)*2"
)

THINK_PROMPT_TMPL = (
    "Using each of {numbers} exactly once with + - * / and parentheses, reach {target}.\n"
    "Reason briefly (a few steps at most), then write the final equation alone on the last line.\n"
    "Example: (20-8)*2"
)


def make_prompt(numbers, target, thinking: bool = False) -> str:
    tmpl = THINK_PROMPT_TMPL if thinking else PROMPT_TMPL
    return tmpl.format(numbers=numbers, target=target)


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""
