"""Train GRPO on synthetic countdown. Usage: python src/train_countdown.py --config configs/countdown_05b.yaml"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer

from envs.countdown import generate_task, make_prompt
from grpo_compat import build_grpo_config
from rewards_countdown import correctness_reward, format_reward


def build_dataset(n: int, num_numbers: int, seed: int = 0, thinking: bool = False) -> Dataset:
    import random

    random.seed(seed)
    rows = {"prompt": [], "numbers": [], "target": []}
    for _ in range(n):
        t = generate_task(num_numbers=num_numbers)
        rows["prompt"].append(
            [{"role": "user", "content": make_prompt(t["numbers"], t["target"], thinking)}]
        )
        rows["numbers"].append(t["numbers"])
        rows["target"].append(t["target"])
    return Dataset.from_dict(rows)


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    thinking = bool(cfg.get("thinking", False))
    if "qwen3" in cfg["model"].lower():
        # Qwen3/3.5 template manages a native <think> block. Non-thinking
        # appends an empty closed block (short direct answers, fits budget);
        # thinking leaves it open for native reasoning traces.
        _chat = tok.apply_chat_template

        def _think_mode(*args, **kwargs):
            kwargs.setdefault("enable_thinking", thinking)
            return _chat(*args, **kwargs)

        tok.apply_chat_template = _think_mode
    ds = build_dataset(cfg.get("num_tasks", 2000), cfg.get("num_numbers", 4), thinking=thinking)
    funcs = [correctness_reward, format_reward]
    weights = [2.0, 0.5]
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
        max_steps=cfg.get("max_steps", 500),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        num_generations=cfg.get("num_generations", 8),
        max_prompt_length=128,
        max_completion_length=cfg.get("max_completion_length", 256),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        beta=float(cfg.get("beta", 0.001)),
        temperature=float(cfg.get("temperature", 0.9)),
        top_p=float(cfg.get("top_p", 1.0)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        # Memory-efficient fused CE kernels (fix OOM on 248k vocab). Unknown
        # keys are warned-and-dropped by grpo_compat on older TRL versions.
        use_liger_loss=bool(cfg.get("use_liger_loss", False)),
        use_liger_kernel=bool(cfg.get("use_liger_kernel", False)),
        use_vllm=False,
        log_completions=True,
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 100),
        report_to="none",
        reward_weights=weights,
    )
    trainer = GRPOTrainer(
        model=cfg["model"],
        args=args,
        train_dataset=ds,
        reward_funcs=funcs,
        peft_config=peft,
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume", False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
