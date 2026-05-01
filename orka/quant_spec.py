"""Quantization spec parsing (vq-/rvq-/rvq-mixed) and tensor family classification."""

from __future__ import annotations

from typing import Sequence

from orka.core import _index_bits_for_size


def classify_tensor_family(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in ("embed", "embedding", "wte", "wpe")):
        return "embedding"
    if any(
        marker in lowered
        for marker in (".mlp.", "mlp", "gate_proj", "up_proj", "down_proj", "c_fc")
    ):
        return "mlp"
    if any(
        marker in lowered
        for marker in (
            "attn",
            "attention",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "c_attn",
        )
    ):
        return "attention"
    return "other"

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
