"""Build the Architecture section of the journey from config + tensor shapes, using orka's
ArchProfile (structural head/recurrent detection) and classify_tensor_family."""
from __future__ import annotations

import re

from orka.quant import ArchProfile, classify_tensor_family

from .schema import Architecture, FamilyBreakdown, LayerBlock, ModuleEntry

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _treatment(name: str, shape, profile: ArchProfile) -> str:
    if profile.is_output_head(name, shape):
        return "keep_fp16"
    if profile.is_recurrent(name):
        return "skip_error_comp"
    return "quantize"


def build_architecture(config: dict, shapes: dict) -> Architecture:
    vocab = config.get("vocab_size")
    profile = ArchProfile.from_shapes(shapes, vocab)
    tied = bool(config.get("tie_word_embeddings", False))

    fam_params: dict[str, int] = {}
    for name, shape in shapes.items():
        fam = classify_tensor_family(name)
        fam_params[fam] = fam_params.get(fam, 0) + _numel(shape)
    total = sum(fam_params.values()) or 1
    breakdown = [
        FamilyBreakdown(family=f, params=p, pct=round(100 * p / total, 1))
        for f, p in sorted(fam_params.items(), key=lambda kv: -kv[1])
    ]

    has_moe = any("expert" in n.lower() for n in shapes)
    has_ssm = len(profile.recurrent_names) > 0
    arch_class = "moe" if has_moe else ("hybrid" if has_ssm else "dense")
    flags = {"tied_head": tied, "has_moe": has_moe, "has_ssm": has_ssm}

    blocks: dict[int, list[ModuleEntry]] = {}
    for name, shape in shapes.items():
        if len(shape) < 2:
            continue  # 1-D params (norms, A_log, biases) not shown as quant modules
        m = _LAYER_RE.search(name)
        idx = int(m.group(1)) if m else -1
        blocks.setdefault(idx, []).append(ModuleEntry(
            name=name, shape=list(shape),
            family=classify_tensor_family(name),
            treatment=_treatment(name, shape, profile),
        ))
    layers = [LayerBlock(index=i, modules=blocks[i]) for i in sorted(blocks)]

    return Architecture(arch_class=arch_class, flags=flags,
                        param_breakdown=breakdown, layers=layers)
