#!/bin/bash
# Vast.ai / RunPod template: pytorch:2.4-cuda12.4, 1x RTX 4090 ($0.40-0.55/hr) or A100 40GB ($0.85-1.1/hr)
# Disk: 60GB. Env: WANDB_API_KEY=<key>
set -e
pip install -r requirements.txt
# Phase 1: correctness+format only (~10-14h on 4090 for 1200 steps, ~$6-8)
python src/train_grpo.py --config configs/grpo_qwen15b.yaml
# Phase 2: enable length penalty — set reward_weights.length_penalty: 0.3, resume
# python src/train_grpo.py --config configs/grpo_qwen15b_len.yaml
python src/eval.py --model outputs/qwen15b-grpo --datasets gsm8k,math500,svamp --num-samples 500
echo "Total est. cost: \$15-40 (4090) / \$30-80 (A100) incl. ablations"
