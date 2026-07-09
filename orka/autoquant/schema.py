"""Data contract for autoquant decisions. TensorConfig is one tensor's decision;
to/from_allocation_map (de)serialize the per-tensor map consumed by `orka pack`."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TensorConfig:
    method: str            # "rvq" | "int8" | "fp16"
    bits: int
    stages: int            # rvq stages (0 for int8/fp16)
    normalization: str     # "block-max" | "none" | ...
    keep_fp16: bool
    source: str            # "policy" | "llm" | "cache"
    confidence: float
    rationale: str


def to_allocation_map(cfgs: dict[str, TensorConfig]) -> dict[str, dict]:
    return {name: asdict(c) for name, c in cfgs.items()}


def from_allocation_map(m: dict[str, dict]) -> dict[str, TensorConfig]:
    return {name: TensorConfig(**d) for name, d in m.items()}
