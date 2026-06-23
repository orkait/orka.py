"""Error-compensation strategy: GPTQ-style block-OBS re-assignment with frozen codebooks.

After the greedy per-stage assignment, re-pick each vector's codeword to minimise the
*output* error E[(Wx - What x)^2] using the calibration activation covariance (block
Optimal Brain Surgeon). Codebooks stay fixed; only the index stream changes. When it
runs, EM-AQ is skipped (EM-AQ would re-learn codebooks against uncompensated residuals
and undo the compensation).

Gated to: torch backend, rotation=none, calibration activations present for the tensor.
Wired in pack_checkpoint per-tensor, before EM-AQ / scale refinement.
"""

from __future__ import annotations

from orka.pipeline.strategies.base import PostAssignmentStrategy


class ErrorCompensationStrategy(PostAssignmentStrategy):
    name = "error_compensation"

    def applies(self, ctx, c: dict) -> bool:
        return ctx.error_compensation

    def apply(self, ctx, c: dict) -> None:
        # Record whether compensation actually ran so em_aq can skip (it would re-learn
        # codebooks against uncompensated residuals and undo this).
        c["_compensated"] = maybe_compensate_candidate(
            c,
            backend=ctx.backend,
            awq_activations=ctx.awq_activations,
            resolved_device=ctx.resolved_device,
            progress_file=ctx.progress_file,
            out_dir=ctx.out_dir,
        )


def maybe_compensate_candidate(
    c: dict,
    *,
    backend: str,
    awq_activations: dict | None,
    resolved_device: str,
    progress_file,
    out_dir,
) -> bool:
    """Apply error-compensated re-assignment to candidate ``c`` in place. Returns True
    when applied (caller then skips EM-AQ for this tensor), False when preconditions are
    not met."""
    from orka._format import _write_indices
    from orka._tensor import _is_torch_tensor
    from orka._util import _report_progress

    if backend != "torch":
        return False
    if awq_activations is None or c["name"] not in awq_activations:
        return False
    if c.get("rotation", "none") != "none":
        return False
    shape = c["shape"]
    if len(shape) < 2:
        return False
    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    group = int(c["group_size"])
    if (
        cols % group != 0
        or int(c["packed_values"]) != rows * cols
        or int(c["padded_values"]) != int(c["packed_values"])
    ):
        return False
    if any(
        sd.get("group_size", group) != group
        for sd in c["stages_data"].values()
    ):
        return False  # scalar stages use a different vector layout

    import numpy as np
    import torch

    from orka.compensation import compensated_assign

    X = torch.as_tensor(awq_activations[c["name"]], dtype=torch.float32)
    if X.dim() != 2 or int(X.shape[1]) != cols:
        return False

    W = c["vectors_orig"]
    W = (
        W if _is_torch_tensor(W) else torch.as_tensor(np.asarray(W))
    ).to(device=resolved_device, dtype=torch.float32).reshape(rows, cols)
    stage_keys = sorted(c["stages_data"].keys())
    cbs = []
    for s_key in stage_keys:
        cb = c["stages_data"][s_key]["cb"]
        cbs.append(
            (
                cb
                if _is_torch_tensor(cb)
                else torch.as_tensor(np.asarray(cb), dtype=torch.float32)
            ).to(resolved_device)
        )
    _report_progress(
        progress_file, f"  Error-compensated re-assignment: {c['name']}"
    )
    idxs, decoded = compensated_assign(W, cbs, group, X)
    for s_key, idx in zip(stage_keys, idxs):
        stage_meta = c["stages_meta"][s_key]
        _write_indices(
            out_dir / stage_meta["indices"],
            idx.cpu(),
            stage_meta["index_bits"],
            stage_meta.get("encoding", "raw"),
        )
        c["stages_data"][s_key]["indices"] = idx
    c["decoded_sum"] = decoded.reshape(-1, group)
    c["vectors_residual"] = None
    c["refined_metrics"] = None
    return True
