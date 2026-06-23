"""Pack pipeline configuration: tunable thresholds, allowed-value sets, and pure
argument validation - separated from the pack_checkpoint orchestration so the
"what's legal / what knobs exist" surface is in one readable place.

Only pure validation (raise on bad input) lives here. Steps that *mutate* config
(device resolution, the error-compensation downgrade, partition normalisation, random
rotation-seed generation) stay in pack_checkpoint where that state is owned.
"""

from __future__ import annotations

from orka._features import AWQ_DISABLED_MESSAGE, awq_feature_enabled


# --- Tunable thresholds ---------------------------------------------------------
# Named so they are not magic numbers buried in the pipeline body.

# Mixed precision: keep a layer dense (skip VQ) when its measured loss-delta from the
# sensitivity map exceeds this.
SENSITIVITY_SKIP_LOSS_DELTA = 1.5

# Embedding tensors cap their VQ group width here even when a larger group_size is
# requested - wide groups at a fixed codebook size collapse embedding fidelity.
EMBEDDING_MAX_GROUP_SIZE = 8

# Floor for Hessian-proxy importance weights, so no dimension/vector gets zero weight
# (which would drop it from the weighted k-means distance metric).
IMPORTANCE_WEIGHT_FLOOR = 1e-6

# MSE-optimal scale refinement (_refine_scales_ls): a block whose residual energy <r,r>
# is below DENOM_FLOOR is degenerate (keep its old scale); a refined scale whose
# magnitude is below MIN_MAGNITUDE is treated as numerically zero (keep the old scale).
LS_SCALE_DENOM_FLOOR = 1e-8
LS_SCALE_MIN_MAGNITUDE = 1e-12

# Per-tensor streaming prefetch: how many tensors the producer reads ahead, and the
# poll timeout the consumer waits on the queue with.
PREFETCH_QUEUE_DEPTH = 4
PREFETCH_POLL_TIMEOUT_S = 0.1


# --- Allowed argument values ----------------------------------------------------
CODEBOOK_MODES = {"per-tensor", "global", "family"}
BACKENDS = {"auto", "numpy", "torch"}
NORMALIZATIONS = {
    "none",
    "block-max",
    "channel-block-max",
    "awq",
    "awq-block-max",
    "slrq-block",
}
ROTATIONS = {"none", "orthogonal", "hadamard"}


def validate_pack_args(
    *,
    codebook_mode: str,
    backend: str,
    normalization: str,
    rotation: str,
    awq_activations: dict | None,
    tensor_partition_count: int | None,
    tensor_partition_index: int | None,
) -> None:
    """Validate config-combination legality. Raises ValueError/RuntimeError on bad
    input; returns None. Pure - mutates nothing."""
    if codebook_mode not in CODEBOOK_MODES:
        raise ValueError("codebook_mode must be 'per-tensor', 'global', or 'family'")
    if backend not in BACKENDS:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            "normalization must be 'none', 'block-max', 'channel-block-max', 'awq', 'awq-block-max', or 'slrq-block'"
        )
    # The feature gate guards only the legacy AWQ *normalization* modes. Calibration
    # activations alone are allowed: they feed Hessian-proxy importance weighting,
    # which changes codebook learning, not the format.
    if normalization in {"awq", "awq-block-max"} and not awq_feature_enabled():
        raise RuntimeError(AWQ_DISABLED_MESSAGE)
    if normalization == "awq" and awq_activations is None:
        raise ValueError(
            "normalization 'awq' requires calibration activations "
            "(--awq-calibration with --awq-model-dir, or --awq-activations-file)"
        )
    if rotation not in ROTATIONS:
        raise ValueError("rotation must be 'none', 'orthogonal', or 'hadamard'")
    if tensor_partition_count is not None:
        if tensor_partition_count < 1:
            raise ValueError("tensor_partition_count must be >= 1")
        if tensor_partition_index is None:
            raise ValueError(
                "tensor_partition_index is required when tensor_partition_count is set"
            )
        if tensor_partition_index < 0 or tensor_partition_index >= tensor_partition_count:
            raise ValueError(
                "tensor_partition_index must be in [0, tensor_partition_count)"
            )
    if (
        tensor_partition_count is not None
        and tensor_partition_count > 1
        and codebook_mode != "per-tensor"
    ):
        raise ValueError(
            "tensor partitions require per-tensor codebooks. Use --codebook-mode per-tensor."
        )
