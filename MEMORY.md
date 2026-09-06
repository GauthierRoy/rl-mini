# MEMORY — RL for LLM project

## Who / goal
- Gauthier Roy, MLE (Lifen, prod SFT 3B + vLLM 250K/day + quant, RecSys'25 oral, GT research).
- Goal: resume piece + learn RL for LLMs. Focus: Reasoning + RLVR/GRPO. Time: 2-4 weeks. Compute: free (Colab) or <$10 total.

## Key decisions
- Pure GSM8K reproduction = tutorial, too weak for this level. Need original/custom.
- Skepticism validated: RLVR rarely creates new math ability; it improves pass@1 sampling efficiency (base wins at pass@256). Frame project as reliability + efficiency, not "new intelligence".
- RLVR needs verifiable reward = any Python checker (not just math): exact match, execution, JSON schema, ranking metric, fairness gap.
- Plan: lab rat first, flagship second. Synthetic data is core skill for main project.

## Shortlist (from brainstorm #1-25)
- #24 Countdown from scratch (TinyZero pure RL, no SFT, 0.5B vs 1.5B vs 3B) → CHOSEN as step 1 to learn GRPO mechanics.
- #21 Hacking zoo (farm 6 hacks: length explosion, empty think, box spam, lang switch, repetition, KL collapse → fix with Dr-GRPO) → overlay on #24.
- #15 Bias-penalized cold-start recs (sequel to RecSys paper) → leading flagship candidate. Reward = NDCG + format - lambda*bias_gap. All code-computable.
- #11 Quant-robust R1 (FP16 vs AWQ/FP8 delta via LLM Compressor + vLLM) → free add-on to any run.
- XML/JSON schema extraction also considered as alt flagship (parse_ok + field F1 + brevity).

## Step 1 — Countdown (in repo now)
- Files: `src/envs/countdown.py` (reverse-gen generator + checker), `src/rewards_countdown.py`, `src/train_countdown.py`, `configs/countdown_05b.yaml` (Colab), `configs/countdown_15b.yaml` (Vast).
- Lesson: reverse-generation (solution→problem) guarantees solvability = synthetic trick #1.
- Run: `python src/train_countdown.py --config configs/countdown_05b.yaml`. Expect flat ~100 steps → jump. 0.5B may fail (expected), 1.5B learns.
- Cost: free on Colab (0.5B smoke); 1.5B as single short budget run (<$10 total, cheap 4090 spot or Colab).
- Smoke finding: format reward 0/40 hits in 50 steps (cold start). Fix: one-shot format demo in prompts + partial format credit (0.1/0.3/0.5).
- Rollback rule: if next run's format reward is still always 0.0 or always 0.5 (binary, nothing in between), revert partial credit to the simple binary version.
- Countdown 0.5B full run (300 steps, G=8, Colab T4, Sep 2026): FAILED as predicted. correctness mean 0.0 every step, format mean 0.007-0.025 (rare <think> flickers, never consolidates), frac_zero_std 0.2-0.8, 47-67% completions truncated at 256 tok. Cause: cold start — Instruct answers in markdown/LaTeX, never emits <answer>, so reward is always 0. Fixes queued: assistant-prefill "<think>", dense shaping (partial credit for valid equation), 3-number curriculum, mask truncated. Runbook: `notebooks/countdown_colab.ipynb`.

## Main project — synthetic-data first (OPEN: recs vs extraction)
- User wants synthetic generation as core learning goal.
- 4 synthetic skills to cover:
  1. Reverse-gen (done in countdown)
  2. Difficulty curriculum (3→4 numbers; easy→hard users)
  3. Decontam + filtering (dedupe, held-out seeds, solvable filter)
  4. Bias-controlled gen (for #15: paired users identical except gender-coding, same true taste → clean bias_gap; impossible with real data alone)
- Next: pick flagship track (#15 recs with synthetic users/items vs extraction with synthetic notes+JSON), then build generator same pattern as countdown.

## Open questions
- [ ] Lock flagship: #15 recs or extraction?
- [ ] Base for flagship: Qwen2.5-1.5B-Instruct vs Qwen2.5-Math-1.5B?
- [ ] Data for flagship: MovieLens-100k cold split vs synthetic fashion vs AI4Privacy?

## Costs (est)
- Budget: free / <$10 total. Countdown 0.5B on Colab free; 1.5B as one short run only, no multi-ablation matrix.
