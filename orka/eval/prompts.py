"""Prompt file I/O for HF eval / AWQ calibration."""

from __future__ import annotations

from pathlib import Path


def _read_prompt_file(path: Path, max_prompts: int | None = None) -> list[str]:
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be positive")
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError("prompt file must contain at least one non-empty prompt")
    return prompts

