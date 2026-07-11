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
            profile=ctx.arch_profile,
        )


def maybe_compensate_candidate(
    c: dict,
    *,
    backend: str,
    awq_activations: dict | None,
    resolved_device: str,
    progress_file,
    out_dir,
    profile=None,
) -> bool:
    """Apply error-compensated re-assignment to candidate ``c`` in place. Returns True
    when applied (caller then skips EM-AQ for this tensor), False when preconditions are
    not met.

    ``profile`` is the checkpoint's ``ArchProfile`` (the single source of truth for which
    tensors block-OBS must skip - output head / recurrent block). pack_checkpoint builds it
    from the checkpoint shapes; a direct call without one falls back to name-only detection.
    Block-OBS minimises the LINEAR output error E[(Wx - What x)^2], valid only when the
    layer feeds a locally-linear path - invalid for the softmax head and SSM/recurrent
    scans (verified on FalconH1-0.5B: 4bpw 1.10 -> 1.50 with error-comp on those)."""
    from orka._runtime import _BG_WRITER
    from orka.core._format import _write_stage_indices
    from orka.core._tensor import _is_torch_tensor
    from orka.core._util import _report_progress
    from orka.quant import ArchProfile

    if profile is None:
        profile = ArchProfile.from_shapes({})  # name-fallback-only profile

    # Global preconditions (backend / no activations at all) stay quiet here - they are
    # one-time misconfigurations surfaced once upfront by pack_checkpoint, not per tensor.
    # Tensor-SPECIFIC skips are logged so a no-op is never silent (finding: a silent skip
    # on an SSM layer cost an afternoon of "why did error-comp make ppl worse" debugging).
    if backend != "torch":
        return False
    if awq_activations is None:
        return False
    name = c["name"]
    if (reason := profile.error_comp_skip_reason(name)) is not None:
        _report_progress(progress_file, f"  error-comp skipped {name}: {reason}")
        return False
    if name not in awq_activations:
        _report_progress(progress_file, f"  error-comp skipped {name}: no calibration activations for this tensor")
        return False
    if c.get("rotation", "none") != "none":
        _report_progress(progress_file, f"  error-comp skipped {name}: rotation={c.get('rotation')} (block-OBS needs unrotated weights)")
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
        _report_progress(progress_file, f"  error-comp skipped {name}: incompatible layout (padded/packed values)")
        return False
    if any(
        sd.get("group_size", group) != group
        for sd in c["stages_data"].values()
    ):
        _report_progress(progress_file, f"  error-comp skipped {name}: scalar/mixed-group stage layout")
        return False  # scalar stages use a different vector layout

    import numpy as np
    import torch

    from orka.quant.compensation import compensated_assign

    X = torch.as_tensor(awq_activations[c["name"]], dtype=torch.float32)
    if X.dim() != 2 or int(X.shape[1]) != cols:
        _report_progress(progress_file, f"  error-comp skipped {name}: activation shape {tuple(X.shape)} != (*, {cols})")
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
        # Queued on the single writer thread, so it lands after the greedy write and
        # reuses the encoding that write recorded (_write_stage_indices resolves it
        # at execution time).
        _BG_WRITER.submit(
            _write_stage_indices,
            out_dir / stage_meta["indices"],
            idx.cpu(),
            stage_meta["index_bits"],
            stage_meta,
        )
        c["stages_data"][s_key]["indices"] = idx
    c["decoded_sum"] = decoded.reshape(-1, group)
    c["vectors_residual"] = None
    c["refined_metrics"] = None
    return True
