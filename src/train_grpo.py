"""GRPO entrypoint: TRL GRPOTrainer + GSM8K. Config-driven via yaml."""

import argparse

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer

from grpo_compat import build_grpo_config
from rewards import correctness_reward, format_reward, length_penalty

SYSTEM = (
    "Solve step by step inside <think></think>, then put the final answer inside <answer></answer>."
)


def load_gsm8k(tokenizer):
    ds = load_dataset("openai/gsm8k", "main")

    def fmt(x):
        return {
            "prompt": [{"role": "user", "content": f"{SYSTEM}\n{x['question']}"}],
            "solution": [x["answer"]],
        }

    return ds["train"].map(fmt, remove_columns=ds["train"].column_names)


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_gsm8k(tok)
    rw = cfg.get("reward_weights", {})
    funcs, weights = [], []
    if rw.get("correctness", 0):
        funcs.append(correctness_reward)
        weights.append(rw["correctness"])
    if rw.get("format", 0):
        funcs.append(format_reward)
        weights.append(rw["format"])
    if rw.get("length_penalty", 0):
        funcs.append(
            lambda c, **k: length_penalty(
                c, max_len=cfg.get("max_completion_length", 512), weight=1.0
            )
        )
        weights.append(rw["length_penalty"])
    peft = None
    if cfg.get("use_peft"):
        peft = LoraConfig(
            r=cfg.get("lora_r", 32),
            lora_alpha=cfg.get("lora_alpha", 64),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        )
    args = build_grpo_config(
        output_dir=cfg["output_dir"],
        max_steps=cfg.get("max_steps", 1200),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        num_generations=cfg.get("num_generations", 8),
        max_prompt_length=cfg.get("max_prompt_length", 256),
        max_completion_length=cfg.get("max_completion_length", 512),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        beta=float(cfg.get("beta", 0.001)),
        temperature=float(cfg.get("temperature", 0.9)),
        use_vllm=bool(cfg.get("use_vllm", True)),
        log_completions=True,
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 100),
        report_to=cfg.get("log_with", "wandb") if cfg.get("log_with") != "none" else "none",
        reward_weights=weights or None,
    )
    GRPOTrainer(
        model=cfg["model"],
        args=args,
        train_dataset=ds,
        reward_funcs=funcs,
        peft_config=peft,
        processing_class=tok,
    ).train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
