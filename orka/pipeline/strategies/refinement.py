"""Post-assignment refinement strategies (no extra bits, pack-time cost only):

  _run_em_aq_refinement  EM-AQ joint optimization. Per tensor, decode the multi-stage
                         additive sum, then EM over the additive quantizer: re-learn
                         each stage's codebook against the residual target, re-quantize,
                         update the running sum in place. Refined codebooks/indices are
                         streamed to disk via the background writer.
  _refine_scales_ls      MSE-optimal block scales. Replace block-max scale with the
                         least-squares s* = <w, r> / <r, r> for the fixed assignment.

Both are catalogued in orka.pipeline.strategies.STRATEGY_REGISTRY.
"""

from __future__ import annotations

from pathlib import Path

from orka._format import _cast_codebook_storage, _write_codebook, _write_indices
from orka._runtime import _BG_WRITER, _check_ram_cap
from orka._tensor import _decode_to_vectors_format, _is_torch_tensor, _vectors_subtract
from orka._util import _report_progress, _safe_tensor_name
from orka.codebook import learn_codebook_auto, quantize_vectors_auto
from orka.metrics import _stage_quality_metrics
from orka.pipeline.pack_config import LS_SCALE_DENOM_FLOOR, LS_SCALE_MIN_MAGNITUDE
from orka.pipeline.pack_helpers import _sample_vectors_and_weights
from orka.pipeline.strategies.base import PostAssignmentStrategy


def _offload_to_cpu(t):
    """Move torch tensor to CPU; passthrough numpy/list."""
    if _is_torch_tensor(t):
        return t.detach().cpu()
    return t


def _run_em_aq_refinement(
    *,
    candidates: list,
    n_stages: int,
    skipped_tensors: set,
    sample_vectors: int | None,
    backend: str,
    resolved_device: str,
    tensor_dir: Path,
    progress_file: Path | None,
    em_aq_passes: int = 3,
) -> None:
    if em_aq_passes <= 0:
        _report_progress(progress_file, "--- EM-AQ disabled (em_aq_passes=0) ---")
    else:
        _report_progress(progress_file, "--- Starting Joint Optimization (EM-AQ) ---")
    joint_iterations = max(0, int(em_aq_passes))

    def _is_skipped(c: dict) -> bool:
        base = c["name"].replace(".weight", "")
        return base in skipped_tensors or c["name"] in skipped_tensors

    for i, c in enumerate(candidates):
        _check_ram_cap()
        if _is_skipped(c):
            continue

        c_n_stages = len(c["stages_data"])

        # 1. Decode current full_sum
        full_sum = None
        for stage_i in range(c_n_stages):
            sd = c["stages_data"][stage_i]
            s_group_size = sd.get("group_size", c["group_size"])
            t_template = c["vectors_orig"]
            if s_group_size == 1 and c["group_size"] > 1:
                t_template = c["vectors_orig"].reshape(-1, 1)

            dec = _decode_to_vectors_format(
                t_template, sd["cb"], sd["indices"], backend, resolved_device
            )
            if s_group_size == 1 and c["group_size"] > 1:
                dec = dec.reshape(c["vectors_orig"].shape)
            full_sum = dec if full_sum is None else full_sum + dec
            del dec

        current_full_sum = full_sum

        if joint_iterations > 0 and c_n_stages > 1:
            _report_progress(progress_file, f"  Joint Refining {c['name']} ({i+1}/{len(candidates)})...")
            # 2. EM-AQ Loop for this tensor
            for joint_iter in range(joint_iterations):
                for stage_i in range(c_n_stages):
                    _check_ram_cap()
                    sd = c["stages_data"][stage_i]
                    s_group_size = sd.get("group_size", c["group_size"])
                    k = c["stages_meta"][stage_i]["codebook_size"]

                    if sample_vectors is not None and k >= sample_vectors:
                        continue

                    t_template = c["vectors_orig"]
                    if s_group_size == 1 and c["group_size"] > 1:
                        t_template = c["vectors_orig"].reshape(-1, 1)

                    old_dec_raw = _decode_to_vectors_format(
                        t_template, sd["cb"], sd["indices"], backend, resolved_device
                    )
                    old_dec = old_dec_raw
                    if s_group_size == 1 and c["group_size"] > 1:
                        old_dec = old_dec_raw.reshape(c["vectors_orig"].shape)

                    target = _vectors_subtract(
                        c["vectors_orig"], (current_full_sum - old_dec)
                    )

                    target_train = target
                    if s_group_size == 1 and c["group_size"] > 1:
                        target_train = target.reshape(-1, 1)

                    sw = c.get("sample_weights") if s_group_size > 1 else None
                    training, sw_train = _sample_vectors_and_weights(
                        target_train, sw, sample_vectors
                    )
                    vw = c.get("vector_weights") if s_group_size > 1 else None

                    cb, _, _ = learn_codebook_auto(
                        training, min(k, len(training)), 2, backend, resolved_device,
                        vector_weights=vw, initial_codebook=sd["cb"],
                        sample_weights=sw_train,
                    )
                    stage_meta = c["stages_meta"][stage_i]
                    cb, _cb_dtype = _cast_codebook_storage(
                        cb, dtype=stage_meta.get("codebook_dtype", "float32")
                    )
                    stage_meta["codebook_dtype"] = _cb_dtype
                    indices, _ = quantize_vectors_auto(target_train, cb, backend, resolved_device, vector_weights=vw)

                    new_dec_raw = _decode_to_vectors_format(
                        target_train, cb, indices, backend, resolved_device
                    )
                    new_dec = new_dec_raw
                    if s_group_size == 1 and c["group_size"] > 1:
                        new_dec = new_dec_raw.reshape(c["vectors_orig"].shape)

                    if hasattr(current_full_sum, "sub_"):
                        # In-place PyTorch math avoids creating 3 massive intermediate tensors
                        current_full_sum.sub_(old_dec).add_(new_dec)
                    else:
                        current_full_sum = (current_full_sum - old_dec) + new_dec

                    c["stages_data"][stage_i] = {"cb": cb, "indices": indices, "group_size": s_group_size}

                    safe = _safe_tensor_name(c["name"])
                    cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                    _BG_WRITER.submit(_write_codebook, cb_path, cb, _cb_dtype)
                    idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
                    _BG_WRITER.submit(
                        _write_indices,
                        idx_path,
                        indices.cpu() if hasattr(indices, "cpu") else indices,
                        stage_meta["index_bits"],
                        stage_meta.get("encoding", "raw"),
                    )

                    del old_dec_raw, old_dec, target, target_train, training, cb, indices, new_dec_raw, new_dec
                    import gc
                    gc.collect()

        c["decoded_sum"] = _offload_to_cpu(current_full_sum)
        del current_full_sum
        c["refined_metrics"] = _stage_quality_metrics(c, backend)
        c["source_flat"] = None
        c["decoded_sum"] = None

        for stage_i in range(c_n_stages):
            if "indices" in c["stages_data"][stage_i]:
                c["stages_data"][stage_i]["indices"] = None

        if "vectors_orig" in c:
            c["vectors_orig"] = None
        if "vectors" in c:
            c["vectors"] = None

        import gc
        gc.collect()


def _refine_scales_ls(c: dict, *, mse_scale: bool, block_scale_size: int, out_dir) -> None:
    """MSE-optimal block scales (post-assignment refinement).

    orka stores scale = block max, so the codebook fit per block is dominated by
    the single largest weight. For the FIXED codeword assignment, the least-squares
    scale s* = <w, r> / <r, r> (r = normalized VQ reconstruction) minimizes the
    per-block reconstruction error - the same idea as llm-compressor's MSE observer,
    adapted to vector quant. Strictly reduces error at zero extra bits / inference cost.

    Excludes salient/outlier override positions from the fit, and leaves outlier-
    containing blocks at block-max (outlier sidecar values are stored scale-relative).
    Gated to rotation=none + block-max-family normalization, where the scale acts
    directly on the stored weights.
    """
    if not mse_scale:
        return
    if (c.get("rotation") or "none") != "none":
        c.pop("_mse_v", None)
        return
    if c.get("normalization") not in (
        "slrq-block", "block-max", "channel-block-max", "awq-block-max"
    ):
        c.pop("_mse_v", None)
        return
    scales = c.get("row_scales")
    v = c.pop("_mse_v", None)  # clean normalized weights stashed pre-stages
    meta = c.get("stages_meta")
    if scales is None or v is None or not meta:
        return
    import numpy as np
    import torch

    from orka._format import _fp16_storage_roundtrip, _read_codebook, _read_indices

    def _num(x, npd, tdt):
        # Flat-CPU conversion; forces dtype so object-arrays of py ints/floats
        # convert. Returns None on any non-numeric input (skip gracefully).
        if x is None:
            return None
        try:
            if _is_torch_tensor(x):
                return x.detach().cpu().reshape(-1).to(tdt)
            return torch.as_tensor(np.asarray(x, dtype=npd)).reshape(-1).to(tdt)
        except Exception:
            return None

    v = _num(v, np.float32, torch.float32)
    sc = _num(scales, np.float32, torch.float32)
    if v is None or sc is None:
        return
    total = int(v.numel())
    bs = int(c.get("block_scale_size") or block_scale_size)
    if bs <= 0 or total % bs != 0 or int(sc.numel()) != total // bs:
        return
    nb = total // bs
    # Final normalized VQ reconstruction r = sum_s codebook_s[indices_s]. The
    # in-memory indices are freed by EM-AQ, so read the final stored codebook +
    # indices back from disk (exactly what decode reconstructs).
    r = torch.zeros(total, dtype=torch.float32)
    try:
        for sm in meta:
            g = int(sm.get("group_size", 8))
            if g <= 0 or total % g != 0:
                return
            cb_np = _read_codebook(
                out_dir / sm["codebook"], g, sm.get("codebook_dtype", "float16")
            )
            idx_np = _read_indices(
                out_dir / sm["indices"], int(sm["index_bits"]), total // g,
                packed=bool(sm.get("packed", False)),
                encoding=sm.get("encoding", "raw"),
            )
            cb = torch.as_tensor(np.asarray(cb_np, dtype=np.float32))
            idx = torch.as_tensor(np.asarray(idx_np, dtype=np.int64)).reshape(-1)
            rr = cb[idx].reshape(-1)
            if int(rr.numel()) != total:
                return
            r += rr
    except Exception:
        return
    mask = torch.ones(total, dtype=torch.float32)
    sal = _num(c.get("salient_indices"), np.int64, torch.long)
    if sal is not None and int(sal.numel()) == nb:
        fs = torch.arange(nb) * bs + sal
        mask[fs[(fs >= 0) & (fs < total)]] = 0.0
    skip_block = torch.zeros(nb, dtype=torch.bool)
    op = _num(c.get("outlier_positions"), np.int64, torch.long)
    if op is not None and int(op.numel()) > 0:
        op = op[(op >= 0) & (op < total)]
        mask[op] = 0.0
        skip_block[(op // bs).clamp_(max=nb - 1)] = True
    vb = (v * mask).reshape(nb, bs)
    rb = (r * mask).reshape(nb, bs)
    num = (vb * rb).sum(1)
    den = (rb * rb).sum(1)
    # v is the NORMALIZED weight (orig = scale_old * v), so the LS-optimal scale
    # is s* = scale_old * <v, r> / <r, r>.
    factor = torch.where(den > LS_SCALE_DENOM_FLOOR, num / den, torch.ones(nb))
    s_new = sc * factor
    s_new = torch.where(skip_block, sc, s_new)  # outlier blocks keep block-max
    s_new = torch.where(torch.isfinite(s_new) & (s_new.abs() > LS_SCALE_MIN_MAGNITUDE), s_new, sc)
    c["row_scales"] = _fp16_storage_roundtrip(s_new.numpy().astype(np.float32))


class EMAQStrategy(PostAssignmentStrategy):
    name = "em_aq"

    def applies(self, ctx, c: dict) -> bool:
        # Skipped when error compensation already ran (it would undo the compensation).
        return ctx.n_stages > 1 and ctx.em_aq_passes > 0 and not c.get("_compensated")

    def apply(self, ctx, c: dict) -> None:
        _run_em_aq_refinement(
            candidates=[c],
            n_stages=ctx.n_stages,
            skipped_tensors=ctx.skipped_tensors,
            sample_vectors=ctx.sample_vectors,
            backend=ctx.backend,
            resolved_device=ctx.resolved_device,
            tensor_dir=ctx.tensor_dir,
            progress_file=ctx.progress_file,
            em_aq_passes=ctx.em_aq_passes,
        )


class MSEScaleStrategy(PostAssignmentStrategy):
    name = "mse_scale"

    def applies(self, ctx, c: dict) -> bool:
        return ctx.mse_scale

    def apply(self, ctx, c: dict) -> None:
        try:
            _refine_scales_ls(
                c, mse_scale=ctx.mse_scale, block_scale_size=ctx.block_scale_size, out_dir=ctx.out_dir
            )
        except Exception as exc:  # refinement is optional; never fail the pack
            c.pop("_mse_v", None)
            _report_progress(ctx.progress_file, f"  mse_scale refinement skipped: {exc}")
