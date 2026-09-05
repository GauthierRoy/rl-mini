"""Version-tolerant GRPOConfig construction.

TRL drops/renames args over time (e.g. `max_prompt_length` was deleted
in 2026). Pass everything, keep what the installed version accepts.
"""

import inspect

from trl import GRPOConfig

_ACCEPTED = set(inspect.signature(GRPOConfig).parameters)


def build_grpo_config(**kwargs) -> GRPOConfig:
    dropped = [k for k in kwargs if k not in _ACCEPTED]
    for k in dropped:
        print(f"warn: installed GRPOConfig has no {k!r}, dropping it")
    return GRPOConfig(**{k: v for k, v in kwargs.items() if k in _ACCEPTED})
