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
- #27 Inference-perf add-ons (NEW): speculative decoding — at least model-free n-gram matching (vLLM `--speculative-config ngram`, no draft model needed), optionally draft-model based (trained 0.8B as draft for a bigger verifier; Qwen3.5 also ships MTP heads → `--speculative-config mtp`) + flash kernels (flash-linear-attention / causal-conv1d for the gated-delta arch — the missing-kernel warnings in training). Plays to existing vLLM/quant strength; serving-side story next to the training story.
- XML/JSON schema extraction also considered as alt flagship (parse_ok + field F1 + brevity).
- #26 PII redaction/extraction as flagship (NEW): verifiable reward = entity F1 + schema validity + brevity; synthetic PII notes generatable same pattern as countdown (reverse-gen solvable entities, bias-controlled demographics). Direct leverage: working on PII models at work right now — domain data intuition + real-world framing for free. Design notes (brainstorm): CSV as output schema (compact, fast to parse/verify vs JSON); multi-task via input task tokens (one model, `<EXTRACT>`/`<REDACT>`/... prefixes); small encoder backbone for inference speed (e.g. ModernBERT-small — confirm exact model name); synthetic-PII-generation-in-the-loop à la OpenMed training, mixed with RLVR.

## Step 1 — Countdown (in repo now)
- Files: `src/envs/countdown.py` (reverse-gen generator + checker), `src/rewards_countdown.py`, `src/train_countdown.py`, `configs/countdown_05b.yaml` (Colab), `configs/countdown_15b.yaml` (Vast).
- Lesson: reverse-generation (solution→problem) guarantees solvability = synthetic trick #1.
- Run: `python src/train_countdown.py --config configs/countdown_05b.yaml`. Expect flat ~100 steps → jump. 0.5B may fail (expected), 1.5B learns.
- Cost: free on Colab (0.5B smoke); 1.5B as single short budget run (<$10 total, cheap 4090 spot or Colab).
- Smoke finding: format reward 0/40 hits in 50 steps (cold start). Fix: one-shot format demo in prompts + partial format credit (0.1/0.3/0.5).
- Rollback rule: if next run's format reward is still always 0.0 or always 0.5 (binary, nothing in between), revert partial credit to the simple binary version.
- Countdown 0.5B full run (300 steps, G=8, Colab T4, Sep 2026): FAILED as predicted. correctness mean 0.0 every step, format mean 0.007-0.025 (rare <think> flickers, never consolidates), frac_zero_std 0.2-0.8, 47-67% completions truncated at 256 tok. Cause: cold start — Instruct answers in markdown/LaTeX, never emits <answer>, so reward is always 0. Fixes queued: assistant-prefill "<think>", dense shaping (partial credit for valid equation), 3-number curriculum, mask truncated. Runbook: `notebooks/countdown_colab.ipynb`.
- Small-model slot switched to Qwen3-0.6B (IFEval 27.9→54.5, MATH 34.4→55.2 vs 2.5-0.5B; same VRAM). Train non-thinking: Qwen3 template defaults to thinking mode (verified), both trainers force `enable_thinking=False` so rollouts fit the token budget and `<answer>` verifier.
- Qwen3.5-0.8B-text-only run (Sep 2026, live): format flickers appearing early (`<answer>` + near-miss equations) = healthy bootstrap, unlike dead 0.5B run. Gates: keep if format mean climbs by step ~25-30; kill if format mean <0.05 at step ~50 or correctness still 0.0 at ~100.
- Qwen3.5-0.8B full run (300 steps, G=8, T4, Sep 2026): PARTIAL — format mean 0.005→0.06 (12x, `<answer>`-only, ~60% of rollouts by end) but correctness never consolidated (2 logged hits, both verbatim copies of the prompt's worked example). Length 244→109 tok (no explosion), entropy 0.7-0.9 stable (no collapse), zero_std →0 from step ~120, clip never fires (tiny LoRA steps). Verdict: alive, not dead — needs structural fixes + more steps.
- Root cause A (format ceiling): Qwen3-style template appends EMPTY closed `<think></think>` after the assistant header in non-thinking mode, so generation starts after `</think>` — model can never emit `<think>` content, levels 0.3/0.5 unreachable by construction. Fix: prompt + reward go answer-only (drop `<think>` requirement), keep `enable_thinking=False`.
- Root cause B (correctness doesn't consolidate): both correct hits echoed the worked example `(10-2)*3` on tasks containing {10,2,3}; subset-allowed checker rewards echoing. Fix: require each number EXACTLY once (TinyZero rule) + change worked example to numbers outside task range (or mark do-not-copy).
- Think run v1 (Sep 2026, killed ~step 30): ALL rewards 0 — native traces overflow the 256 tok cap, truncated before any answer. Fix for v2: concise prompt ("a few steps at most"), DAPO-style overlong band penalty (free <~175 tok, linear to -0.5 at cap, thinking runs only), `mask_truncated_completions: true` (truncated = no signal, not wrong), relaunch with `resume: true`.
- TODO next launch (from DAPO/Dr.GRPO/TinyZero deep dive, Sep 2026): `beta: 0.0`, `epsilon_high: 0.28`, `loss_type: dr_grpo` + `scale_rewards: none`, `mask_truncated_completions: true`, rescale rewards to correctness 1.0 / format 0.1 (raw scale matters with no std norm). Needs 4 new passthrough knobs in `train_countdown.py` (same safe pattern as `lora_target_modules`) + `configs/countdown_q35_08b_v2.yaml`. Do NOT apply mid-run.
- Cold-start ladder if 0.8B stalls: (1) assistant prefill `Let me solve this step by step.\n<think>`, (2) 3-number curriculum, (3) SFT pre-format: ~59 filtered correct-format traces, 1 epoch, so GRPO skips format-learning entirely (Unsloth Qwen3-4B recipe rationale), then accuracy-only RL, (4) base model (`Qwen/Qwen3-0.6B-Base`).
- Monitor gates: `frac_reward_zero_std > 0.8` = wasted batch; length up + reward flat = loss bias; entropy →0 = collapse (re-add small beta), entropy explosion = gibberish.

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
