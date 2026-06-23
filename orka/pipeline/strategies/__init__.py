"""Compression strategies (the "tricks") that pack_checkpoint composes onto the base
RVQ codec, and a registry that tracks exactly what each one does, the flag that enables
it, where its code lives, and where it is wired into the pipeline.

The registry is the single source of truth for "what tricks exist and how they are
wired". Keep STRATEGY_REGISTRY in sync when adding/moving a strategy (STRATEGIES.md is
the human-readable rendering). Implementations that already had a natural home
(normalization/rotation/outliers in orka.transforms, RVQ in orka.codebook) stay there
and are catalogued by reference; the pack-time refinement/re-assignment strategies live
in this package.
"""

from __future__ import annotations

from orka.pipeline.strategies.base import PostAssignmentStrategy
from orka.pipeline.strategies.error_compensation import (
    ErrorCompensationStrategy,
    maybe_compensate_candidate,
)
from orka.pipeline.strategies.refinement import (
    EMAQStrategy,
    MSEScaleStrategy,
    _refine_scales_ls,
    _run_em_aq_refinement,
)

# Ordered post-assignment strategy pipeline. The streamed worker applies these in order;
# each decides whether it runs (see PostAssignmentStrategy). Order is load-bearing:
# error_compensation first (sets c["_compensated"]), then em_aq (skipped if compensated),
# then mse_scale. Add a new trick by appending an instance here - no pipeline edit.
POST_ASSIGNMENT_STRATEGIES: list[PostAssignmentStrategy] = [
    ErrorCompensationStrategy(),
    EMAQStrategy(),
    MSEScaleStrategy(),
]

__all__ = [
    "PostAssignmentStrategy",
    "POST_ASSIGNMENT_STRATEGIES",
    "ErrorCompensationStrategy",
    "EMAQStrategy",
    "MSEScaleStrategy",
    "maybe_compensate_candidate",
    "_refine_scales_ls",
    "_run_em_aq_refinement",
    "STRATEGY_REGISTRY",
]


# Each entry documents one composable compression strategy. `module` is where the code
# lives; `enabled_by` is the pack_checkpoint arg that turns it on; `wired_at` is the
# pipeline stage that invokes it; `effect` is the bit/quality trade-off.
STRATEGY_REGISTRY = [
    {
        "name": "rvq",
        "summary": "Residual vector quantization - the base codec (k-means codebooks + indices, multi-stage).",
        "module": "orka.codebook",
        "enabled_by": "always (codebook_size / codebook_sizes)",
        "wired_at": "per-tensor stage loop (learn_codebook_auto + quantize_vectors_auto)",
        "effect": "sets the base bpw (bits = ceil(log2 k) per stage / group_size).",
    },
    {
        "name": "normalization",
        "summary": "Per-block scale normalization (block-max / channel-block-max / slrq-block / awq variants).",
        "module": "orka.transforms.normalize",
        "enabled_by": "normalization=",
        "wired_at": "pre-stage, per tensor (_apply_normalization)",
        "effect": "tighter codeword fit; adds a scale sidecar (bpw +~0.03-0.5 depending on block size).",
    },
    {
        "name": "rotation",
        "summary": "Incoherence rotation (random orthogonal / Hadamard) to spread outliers before VQ.",
        "module": "orka.transforms.rotate",
        "enabled_by": "rotation=",
        "wired_at": "pre-vectorize, per tensor (_rotate_tensor_to_2d)",
        "effect": "lowers reconstruction error on heavy-tailed weights; stores a seed, no extra bpw.",
    },
    {
        "name": "outliers",
        "summary": "Exact-store the largest-magnitude weights in a sparse sidecar.",
        "module": "orka.transforms (_extract_outliers)",
        "enabled_by": "outlier_frac>0",
        "wired_at": "pre-stage, per tensor",
        "effect": "removes the worst quant errors; adds a sparse (index,value) sidecar.",
    },
    {
        "name": "salient",
        "summary": "SLRQ - keep one exact salient weight per block (Hessian-guided escape).",
        "module": "orka.pipeline.pack (salient extraction)",
        "enabled_by": "slrq_salient + normalization=slrq-block",
        "wired_at": "pre-stage, per tensor",
        "effect": "protects the highest-importance weight per block; small sidecar.",
    },
    {
        "name": "hessian_weighting",
        "summary": "AWQ-style importance weights (E[x^2] per column) feed weighted k-means.",
        "module": "orka.pipeline.pack (vector_weights / sample_weights)",
        "enabled_by": "awq_activations provided",
        "wired_at": "per-tensor, before stage learning",
        "effect": "centroids pulled toward high-energy columns; no extra bpw.",
    },
    {
        "name": "error_compensation",
        "summary": "GPTQ-style block-OBS index re-assignment minimising output error (frozen codebooks).",
        "module": "orka.pipeline.strategies.error_compensation",
        "enabled_by": "error_compensation=True (torch, rotation=none, activations)",
        "wired_at": "per-tensor, post-assignment (maybe_compensate_candidate); skips EM-AQ when applied",
        "effect": "lower output error at no extra bpw; rewrites the index stream.",
    },
    {
        "name": "em_aq",
        "summary": "EM-AQ joint optimization - iteratively re-learn each stage's codebook vs the residual.",
        "module": "orka.pipeline.strategies.refinement",
        "enabled_by": "em_aq_passes>0",
        "wired_at": "per-tensor, after the greedy stage loop (_run_em_aq_refinement)",
        "effect": "tightens multi-stage RVQ fit at no extra bpw; pack-time cost only.",
    },
    {
        "name": "mse_scale",
        "summary": "MSE-optimal block scales: replace block-max with least-squares s*=<w,r>/<r,r>.",
        "module": "orka.pipeline.strategies.refinement",
        "enabled_by": "mse_scale=True (rotation=none, block-max-family normalization)",
        "wired_at": "per-tensor, after refinement (_refine_scales_ls)",
        "effect": "strictly lower per-block error at no extra bpw / inference cost.",
    },
]
