"""Eval: pass@1 + maj@8 + avg length on GSM8K / MATH-500 / SVAMP."""

import argparse
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams


def norm_answer(t: str) -> str:
    m = re.search(r"####\s*(-?[\d,\.]+)", t)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"\\boxed\{([^}]+)\}", t)
    return m.group(1).strip() if m else ""


def run(model, dataset, n=200, k=8):
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
        qs, golds = ds["question"], [norm_answer(a) for a in ds["answer"]]
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(n))
        qs, golds = ds["problem"], [norm_answer(s) for s in ds["solution"]]
    else:  # svamp
        ds = load_dataset("ChilleD/SVAMP", split="test").select(
            range(min(n, len(load_dataset("ChilleD/SVAMP", split="test"))))
        )
        qs, golds = ds["question"], [str(a).replace(",", "") for a in ds["answer"]]
    llm = LLM(model=model, gpu_memory_utilization=0.5)
    sp1 = SamplingParams(temperature=0.0, max_tokens=512)
    spk = SamplingParams(temperature=0.9, top_p=1.0, max_tokens=512, n=k)
    p1 = llm.generate(
        [f"Solve step by step inside <think></think>, then <answer></answer>.\n{q}" for q in qs],
        sp1,
    )
    pk = llm.generate(
        [f"Solve step by step inside <think></think>, then <answer></answer>.\n{q}" for q in qs],
        spk,
    )
    from collections import Counter

    pass1, maj, lens = 0, 0, []
    for g, o1, ok in zip(golds, p1, pk, strict=False):
        a1 = norm_answer(o1.outputs[0].text)
        pass1 += a1 == g
        lens.append(len(o1.outputs[0].text.split()))
        votes = Counter(norm_answer(o.text) for o in ok.outputs)
        maj += votes.most_common(1)[0][0] == g
    print(
        f"{dataset}: pass@1={pass1 / len(qs):.3f} maj@{k}={maj / len(qs):.3f} avg_len={sum(lens) / len(lens):.0f}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", default="gsm8k")
    p.add_argument("--num-samples", type=int, default=200)
    a = p.parse_args()
    for d in a.datasets.split(","):
        run(a.model, d.strip(), a.num_samples)
