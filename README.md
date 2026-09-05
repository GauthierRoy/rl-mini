# TinyR1-Zero — GRPO Reasoning on a 1.5B Model (<$80)

Reproduce DeepSeek R1-Zero with GRPO on `Qwen2.5-1.5B-Instruct`, then beat the naive baseline with **length-controlled reasoning** (accuracy up, tokens down).

Why this is resume-strong for you:
- Fills your one gap: you have SFT 3B + vLLM 250K/day + quant, but no public RL-training story.
- Maps 1:1 to 2025-26 job posts: GRPO / RLVR / R1 / verifiable rewards / efficient reasoning.
- Production angle: ties training (GRPO) to your inference strength (vLLM latency/cost).

## What you'll learn (mapped to code)
- Policy gradients without critic: group baseline `A_i = (r_i - mean)/std` → `src/rewards.py`
- Clipping + KL control (`beta`, `epsilon`) → `configs/grpo_qwen15b.yaml`
- Reward engineering + hacking analysis → `src/rewards.py`, `notebooks/3_error_analysis.ipynb`
- Rigorous eval: pass@1, maj@8, length vs accuracy → `src/eval.py`

## Stack
TRL `GRPOTrainer` + vLLM sampling + LoRA (full-ft as stretch) + W&B + `math-verify`

- Train: GSM8K train
- Test ID: GSM8K test | OOD: MATH-500, SVAMP
- Base: `Qwen/Qwen2.5-1.5B-Instruct` (Colab smoke: `Qwen2.5-0.5B-Instruct`)

## 3-week plan
**Wk1 — Baselines + infra:** SFT eval (no train), GRPO smoke on Colab (50 steps, G=4), W&B + eval harness working.
**Wk2 — Core GRPO:** Full run on Vast (1x 4090/A100, ~800-1500 steps, G=8), tune correctness + format rewards, log length explosion.
**Wk3 — Differentiator + writeup:** Add length penalty / overlong filtering, ablate `G`, `beta`, `lr`. Ship curves + failure gallery + demo.
**Wk4 buffer:** Blog post (you already write) + resume bullet + Gradio demo served with vLLM.

Target result: `+10-20% pass@1 on GSM8K vs base, flat or up on MATH-500, -30-40% avg tokens with length penalty`.

## Quickstart
```bash
pip install -r requirements.txt
# smoke (Colab T4, ~15 min):
python src/train_grpo.py --config configs/smoke_colab.yaml
python src/eval.py --model outputs/smoke --datasets gsm8k --num-samples 100
# full (Vast 4090/A100):
bash scripts/run_vast.sh
```

See `configs/` for hyperparams and `scripts/run_vast.sh` for exact Vast template + cost.
