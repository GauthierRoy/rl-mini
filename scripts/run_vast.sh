#!/bin/bash
# Budget template: Colab free or cheap 4090 spot. Keep total <$10:
# single short run only, no ablation matrix.
# Disk: 60GB. Env: WANDB_API_KEY=<key>
set -e
pip install -r requirements.txt
# Phase 1: correctness+format only, short run (~300-500 steps max)
python src/train_grpo.py --config configs/grpo_qwen15b.yaml
# Phase 2: enable length penalty — set reward_weights.length_penalty: 0.3, resume
# python src/train_grpo.py --config configs/grpo_qwen15b_len.yaml
python src/eval.py --model outputs/qwen15b-grpo --datasets gsm8k,math500,svamp --num-samples 500
echo "Budget: free / <\$10 total"
