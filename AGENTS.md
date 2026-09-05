# AGENTS.md

## Project
TinyR1-Zero: GRPO reasoning on Qwen2.5-1.5B. TRL `GRPOTrainer` + vLLM + LoRA.
See `README.md` and `MEMORY.md` for background.

## Commands
- Install runtime: `pip install -r requirements.txt`
- Lint all: `bash scripts/lint.sh`

## Style
Be minimalist. Comment only when necessary (non-obvious logic). No obvious/redundant comments.

## Definition of done
At the end of every task, before saying finish, you MUST run:
`bash scripts/lint.sh`
Fix all errors and re-run until it passes. Do not report completion with failing lint.
