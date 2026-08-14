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


def to_packer_allocation(
    cfgs: dict[str, TensorConfig],
    group_size: int = 8,
    source: str = "",
) -> dict:
    """Convert AutoQuant TensorConfig maps into the packer's allocation schema.

    RVQ tensors become ``stages = [1 << bits] * stages``. int8/fp16 tensors are
    recorded as dense exceptions so pack can passthrough them via only_tensors.
    """
    tensors: dict[str, dict] = {}
    dense: dict[str, dict] = {}
    for name, c in cfgs.items():
        meta = {"rationale": c.rationale, "source": c.source, "confidence": c.confidence}
        if c.method == "rvq":
            stages = [1 << int(c.bits)] * int(c.stages)
            spec = (
                "vq-" + str(c.bits)
                if c.stages == 1
                else "rvq-" + "-".join(str(c.bits) for _ in range(c.stages))
            )
            tensors[name] = {
                "spec": spec,
                "stages": stages,
                "bits_per_weight": (c.bits * c.stages) / group_size,
                "normalization": c.normalization,
                **meta,
            }
        else:
            dense[name] = {"method": c.method, "keep_fp16": c.keep_fp16, **meta}
    return {
        "format": "orka-allocation",
        "source": source,
        "group_size": group_size,
        "tensors": tensors,
        "dense": dense,
    }
