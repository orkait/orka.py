"""Quantization spec: vq-/rvq-/rvq-mixed parsing + RVQ-mixed family bits +
PayloadEstimate / estimate_payload (artifact size from params + group + codebook).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from orka._util import _index_bits_for_size


@dataclass(frozen=True)
class PayloadEstimate:
    params: int
    group_size: int
    codebook_size: int
    index_bits: int
    vector_count: int
    index_bytes: int
    scale_block_vectors: int
    scale_bytes: int
    bits_per_weight: float
    total_payload_bytes: int


def estimate_payload(
    params: int,
    group_size: int,
    codebook_size: int,
    scale_block_vectors: int = 64,
    scale_bits: int = 0,
) -> PayloadEstimate:
    if params <= 0:
        raise ValueError("params must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than 1")
    if scale_block_vectors <= 0:
        raise ValueError("scale_block_vectors must be positive")
    if scale_bits < 0:
        raise ValueError("scale_bits must be non-negative")

    index_bits = math.ceil(math.log2(codebook_size))
    vector_count = math.ceil(params / group_size)
    index_bytes = math.ceil(vector_count * index_bits / 8)
    scale_count = math.ceil(vector_count / scale_block_vectors)
    scale_bytes = math.ceil(scale_count * scale_bits / 8)
    return PayloadEstimate(
        params=params,
        group_size=group_size,
        codebook_size=codebook_size,
        index_bits=index_bits,
        vector_count=vector_count,
        index_bytes=index_bytes,
        scale_block_vectors=scale_block_vectors,
        scale_bytes=scale_bytes,
        bits_per_weight=(index_bytes + scale_bytes) * 8 / params,
        total_payload_bytes=index_bytes + scale_bytes,
    )

QUANT_SPEC_MAX_PER_STAGE_BITS = 64
QUANT_SPEC_MAX_TOTAL_BITS = 64

RVQ_MIXED_FAMILY_BITS = {
    "embedding": [16, 16, 16],
    "attention": [16, 8],
    "mlp": [16, 8],
    "other": [16],
}


def rvq_mixed_family_stages() -> dict[str, list[int]]:
    return {fam: [1 << b for b in bits] for fam, bits in RVQ_MIXED_FAMILY_BITS.items()}


def is_rvq_mixed_spec(spec: str | None) -> bool:
    return spec == "rvq-mixed"


def parse_quant_spec(spec: str) -> list[int]:
    if not isinstance(spec, str):
        raise ValueError(f"quant spec must be a string: {spec!r}")
    if spec.startswith("rvq-"):
        body = spec[4:]
        prefix = "rvq-"
    elif spec.startswith("vq-"):
        body = spec[3:]
        prefix = "vq-"
    else:
        raise ValueError(f"quant spec must start with 'vq-' or 'rvq-': {spec!r}")
    parts = body.split("-")
    if not parts or not all(p for p in parts):
        raise ValueError(f"empty stage in quant spec: {spec!r}")
    bits = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"non-integer bits in quant spec: {p!r}")
        b = int(p)
        if b < 1 or b > QUANT_SPEC_MAX_PER_STAGE_BITS:
            raise ValueError(
                f"per-stage bits must be 1..{QUANT_SPEC_MAX_PER_STAGE_BITS}: got {b}"
            )
        bits.append(b)
    total = sum(bits)
    if total > QUANT_SPEC_MAX_TOTAL_BITS:
        raise ValueError(
            f"total bits per vector must be ≤ {QUANT_SPEC_MAX_TOTAL_BITS}: got {total}"
        )
    if prefix == "vq-" and len(bits) > 1:
        raise ValueError(
            f"'vq-' is single-stage; use 'rvq-' for {len(bits)} stages: {spec!r}"
        )
    if prefix == "rvq-" and len(bits) < 2:
        raise ValueError(f"'rvq-' requires ≥2 stages; got {len(bits)}: {spec!r}")
    return [1 << b for b in bits]

def quant_spec_from_sizes(sizes: Sequence[int]) -> str:
    parts = [str(_index_bits_for_size(int(k))) for k in sizes]
    prefix = "vq-" if len(parts) == 1 else "rvq-"
    return prefix + "-".join(parts)


def _resolve_quant_stages(
    quant_mode: str | None,
    codebook_sizes: Sequence[int] | None,
    codebook_size: int,
) -> list[int]:
    if codebook_sizes:
        return [int(x) for x in codebook_sizes]
    if quant_mode:
        return parse_quant_spec(quant_mode)
    return [int(codebook_size)]
