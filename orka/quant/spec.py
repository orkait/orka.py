"""Quantization spec: vq-/rvq-/rvq-mixed parsing + RVQ-mixed family bits +
PayloadEstimate / estimate_payload (artifact size from params + group + codebook).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from orka.core._util import _index_bits_for_size


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
#: Ceiling on the summed PER-STAGE bit widths. For a VQ stage that width is an index cost
#: per group_size-weight vector; for a scalar ('s') stage it is a cost per WEIGHT. The two
#: units are not comparable, so scalar stages carry their own ceiling below.
QUANT_SPEC_MAX_TOTAL_BITS = 64
#: Ceiling on the summed scalar-stage bits, which ARE bits per weight. Without it the
#: per-vector ceiling above let a planar spec through at any rate: 'rvq-' + 8 x 's8'
#: counted as 64 (accepted) while actually costing 8 x 8 = 64 bits per weight, i.e. twice
#: fp32. Set above the shipped planar candidates (rvq-s8-s8-s8 = 24) so those still parse.
QUANT_SPEC_MAX_SCALAR_BITS_PER_WEIGHT = 32

# Default mixed precision for MoE and Dense architectures
RVQ_MIXED_FAMILY_BITS = {
    "embedding": [12, "s4", "s4"],  # High fidelity linguistic brain
    "shared_expert": [16, 8],       # The MoE 'Teacher' (Always active)
    "expert": [12, "s4"],           # Routed experts (Sparse logic)
    "router": [16, 16],             # Sensitive gating layers
    "attention": [16, 8],           # Logic layers
    "mlp": [16, 8],                 # Logic layers
    "other": [16],
}


def rvq_mixed_family_stages() -> dict[str, list[int | str]]:
    """Returns family -> list of (codebook_size or 's<bits>')"""
    res = {}
    for fam, bits in RVQ_MIXED_FAMILY_BITS.items():
        stages = []
        for b in bits:
            if isinstance(b, str) and b.startswith("s"):
                stages.append(b)
            else:
                stages.append(1 << int(b))
        res[fam] = stages
    return res


def is_rvq_mixed_spec(spec: str | None) -> bool:
    return spec == "rvq-mixed"


def parse_quant_spec(spec: str) -> list[int | str]:
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
    
    stages = []
    total_bits = 0
    scalar_bits_per_weight = 0
    for p in parts:
        if p.startswith("s"):
            b_str = p[1:]
            if not b_str.isdigit():
                raise ValueError(f"invalid scalar bits: {p!r}")
            b = int(b_str)
            stages.append(p) # Store as 's4'
            scalar_bits_per_weight += b
        else:
            if not p.isdigit():
                raise ValueError(f"non-integer bits: {p!r}")
            b = int(p)
            stages.append(1 << b)

        if b < 1 or b > QUANT_SPEC_MAX_PER_STAGE_BITS:
             raise ValueError(f"bits must be 1..{QUANT_SPEC_MAX_PER_STAGE_BITS}")
        total_bits += b

    if total_bits > QUANT_SPEC_MAX_TOTAL_BITS:
        raise ValueError(f"total bits ≤ {QUANT_SPEC_MAX_TOTAL_BITS}")
    # Scalar stages cost their bits PER WEIGHT, so they need a ceiling in that unit -
    # the per-vector total above cannot see a planar spec that stores more than fp32.
    if scalar_bits_per_weight > QUANT_SPEC_MAX_SCALAR_BITS_PER_WEIGHT:
        raise ValueError(
            f"scalar stages cost {scalar_bits_per_weight} bits per weight in {spec!r}; "
            f"the limit is {QUANT_SPEC_MAX_SCALAR_BITS_PER_WEIGHT} "
            f"(a planar spec above that stores more than the fp32 source)"
        )

    if prefix == "vq-" and len(stages) > 1:
        raise ValueError("vq- is single-stage")
    return stages


def quant_spec_from_sizes(sizes: Sequence[int | str]) -> str:
    parts = []
    for k in sizes:
        if isinstance(k, str) and k.startswith("s"):
            parts.append(k)
        else:
            parts.append(str(_index_bits_for_size(int(k))))
    prefix = "vq-" if len(parts) == 1 else "rvq-"
    return prefix + "-".join(parts)


def _resolve_quant_stages(
    quant_mode: str | None,
    codebook_sizes: Sequence[int] | None,
    codebook_size: int,
) -> list[int | str]:
    if codebook_sizes:
        return [int(x) for x in codebook_sizes]
    if quant_mode:
        return parse_quant_spec(quant_mode)
    return [int(codebook_size)]

