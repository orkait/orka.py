"""Static, weight-free estimate of compression ratio + perplexity from the measured RD
anchors (orka frontier work). Transparent heuristic, every output labeled source=estimated
with the anchor in notes. Upgradeable to a fitted predictor later (out of scope)."""
from __future__ import annotations

from .schema import Architecture, ModelMeta, Result

# Smoothed ppl-ratio vs bpw for the full config (rvq-12-12 + em-aq + hessian), untied
# baseline. Measured anchors: 3.0 -> 1.345 (artifact), 4.0 -> 1.26 (sweep); the rest
# smoothed monotonic. Labeled estimated; this is the only "guessed" constant.
_PPL_ANCHORS = [(2.5, 2.2), (2.75, 1.6), (3.0, 1.35), (3.5, 1.22), (4.0, 1.15)]
# rvq-12-12 codebook overhead per quantized 2-D tensor: 2 stages * K(4096) * group(8) * 2 B.
_CODEBOOK_OVERHEAD_BYTES = 2 * 4096 * 8 * 2


def _interp(bpw: float) -> float:
    pts = _PPL_ANCHORS
    if bpw <= pts[0][0]:
        return pts[0][1]
    if bpw >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= bpw <= x1:
            return y0 + (y1 - y0) * (bpw - x0) / (x1 - x0)
    return pts[-1][1]


def _vocab_width_params(arch: Architecture) -> int:
    return sum(f.params for f in arch.param_breakdown if f.family == "embedding")


def estimate(meta: ModelMeta, arch: Architecture, bpw: float = 3.0,
             keep_head: bool = True, lattice: bool = False) -> Result:
    total = max(meta.params_total, 1)
    head = _vocab_width_params(arch) if (keep_head and meta.tie_word_embeddings) else 0
    body = max(total - head, 0)

    n_quant_tensors = sum(
        1 for blk in arch.layers for m in blk.modules if m.treatment != "keep_fp16"
    )
    quant_bytes = body * bpw / 8.0 + n_quant_tensors * _CODEBOOK_OVERHEAD_BYTES
    passthrough_bytes = head * 2  # fp16
    orka_bytes = max(quant_bytes + passthrough_bytes, 1.0)
    ratio = meta.fp16_bytes / orka_bytes

    ppl = _interp(bpw)
    notes = [f"estimated from RD anchor bpw={bpw:.2f}->{ppl:.2f}"]
    if arch.flags.get("has_moe"):
        ppl *= 0.97
        notes.append("MoE compresses well (-3%)")
    if meta.tie_word_embeddings and not keep_head:
        ppl *= 1.4
        notes.append("tied head quantized -> ppl penalty")
    if lattice and arch.flags.get("has_ssm"):
        ppl *= 1.3
        notes.append("E8 lattice Pareto-loses on hybrid (+30%)")
    if keep_head and meta.tie_word_embeddings:
        notes.append("tied head+embed kept fp16 (auto)")

    return Result(
        source="estimated", bpw=bpw, ratio=round(ratio, 2),
        fp16_mb=round(meta.fp16_bytes / 1e6, 1), orka_mb=round(orka_bytes / 1e6, 1),
        ppl_ratio=round(ppl, 3), notes=notes,
    )
