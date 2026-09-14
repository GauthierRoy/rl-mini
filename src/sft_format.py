"""SFT cold-start for countdown: bare equations so GRPO starts with signal.
Usage: python src/sft_format.py --config configs/sft_format_08b.yaml"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from envs.countdown import find_equation, generate_task, grade_equation, make_prompt


def build_sft_dataset(cfg, tok):
    import random

    from datasets import Dataset

    random.seed(cfg.get("seed", 0))
    texts = []
    need, guard = cfg.get("num_tasks", 200), 0
    while len(texts) < need and guard < need * 20:
        guard += 1
        t = generate_task(num_numbers=cfg.get("num_numbers", 4))
        eq = find_equation(t["numbers"], t["target"])
        if eq is None:
            continue
        score, _ = grade_equation(t["numbers"], t["target"], eq)
        assert score == 2.0, (t, eq)
        msgs = [
            {"role": "user", "content": make_prompt(t["numbers"], t["target"])},
            {"role": "assistant", "content": eq},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False)
        if not text.endswith(tok.eos_token):
            text += tok.eos_token
        texts.append(text)
    assert len(texts) == need, f"only built {len(texts)}/{need}"
    return Dataset.from_dict({"text": texts})


def main(cfg_path):
    import inspect

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = build_sft_dataset(cfg, tok)
    peft = LoraConfig(
        r=cfg.get("lora_r", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
    )
    kwargs = dict(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 50),
        report_to="none",
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        dataset_text_field="text",
    )
    seq_len = cfg.get("max_seq_length", 256)
    accepted = set(inspect.signature(SFTConfig).parameters)
    if "max_seq_length" in accepted:
        kwargs["max_seq_length"] = seq_len
    elif "max_length" in accepted:
        kwargs["max_length"] = seq_len
    args = SFTConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(cfg["model"], trust_remote_code=True),
        args=args,
        train_dataset=ds,
        peft_config=peft,
        processing_class=tok,
    )
    trainer.train()
    merged_dir = cfg.get("merged_dir", cfg["output_dir"] + "-merged")
    m = trainer.model.merge_and_unload()
    m.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print("saved merged model to", merged_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
