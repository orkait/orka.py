"""Per-tensor pack pipeline.

PackCtx bundles the resolved pack configuration + run accumulators (built once by
pack_checkpoint after arg resolution), and process_streamed_per_tensor_candidate does the
per-tensor work: the RVQ stage loop, then the post-assignment strategies
(error_compensation -> em_aq -> mse_scale), then the manifest entry. Pulled out of pack.py
so pack_checkpoint is the orchestrator and this module is the worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orka._format import _cast_codebook_storage, _write_codebook, _write_indices
from orka._runtime import _BG_WRITER, _check_ram_cap
from orka._tensor import _decode_to_vectors_format, _is_torch_tensor, _vectors_subtract
from orka._util import (
    _derive_seed,
    _index_bits_for_size,
    _report_progress,
    _safe_tensor_name,
)
from orka.codebook import (
    _codebook_cache_key,
    _codebook_cache_load,
    _codebook_cache_save,
    learn_codebook_auto,
    quantize_vectors_auto,
)
from orka.pipeline.pack_helpers import _sample_vectors_and_weights, _weights_digest
from orka.pipeline.pack_manifest import _finalize_tensor_manifest_entry
from orka.pipeline.strategies import (
    _refine_scales_ls,
    _run_em_aq_refinement,
    maybe_compensate_candidate,
)


@dataclass
class PackCtx:
    """Resolved pack configuration + run accumulators, threaded through the per-tensor
    pipeline. Built once by pack_checkpoint after argument resolution."""

    backend: str
    n_stages: int
    group_size: int
    resolved_device: str
    out_dir: Path
    tensor_dir: Path
    sample_vectors: int | None
    progress_file: object
    iterations: int
    em_aq_passes: int
    block_scale_size: int
    rotation: str
    rotation_seed: int | None
    mse_scale: bool
    src_sig: str
    skipped_tensors: set
    codebook_dtype: str
    codebook_cache_dir: object
    awq_activations: dict | None
    outlier_frac: float
    normalization: str
    max_tensors: int | None
    error_compensation: bool
    stages_spec: list
    family_stages_resolved: dict | None
    tensor_stages_resolved: dict | None
    manifest: dict
    total_index_bytes: int = 0


def _offload(t):
    if _is_torch_tensor(t):
        return t.detach().cpu()
    return t


def _onload(t, device):
    if _is_torch_tensor(t):
        return t.to(device=device)
    return t


def stage_spec_for_candidate(ctx: PackCtx, c: dict, stage_i: int):
    tensor_stages_resolved = ctx.tensor_stages_resolved
    family_stages_resolved = ctx.family_stages_resolved
    stages_spec = ctx.stages_spec
    if tensor_stages_resolved is not None:
        stages_for_c = tensor_stages_resolved.get(
            c["name"]
        ) or tensor_stages_resolved.get(c["name"].replace(".weight", ""))
        if stages_for_c is not None:
            if stage_i >= len(stages_for_c):
                return None
            return stages_for_c[stage_i]
    if family_stages_resolved is not None:
        stages_for_c = family_stages_resolved[c["family"]]
        if stage_i >= len(stages_for_c):
            return None
        return stages_for_c[stage_i]
    if stage_i >= len(stages_spec):
        return None
    return stages_spec[stage_i]


def process_streamed_per_tensor_candidate(ctx: PackCtx, c: dict, stream_index: int) -> None:
    # Unpack resolved config so the body below reads exactly as it did inline.
    backend = ctx.backend
    n_stages = ctx.n_stages
    group_size = ctx.group_size
    resolved_device = ctx.resolved_device
    out_dir = ctx.out_dir
    tensor_dir = ctx.tensor_dir
    sample_vectors = ctx.sample_vectors
    progress_file = ctx.progress_file
    iterations = ctx.iterations
    em_aq_passes = ctx.em_aq_passes
    block_scale_size = ctx.block_scale_size
    rotation = ctx.rotation
    rotation_seed = ctx.rotation_seed
    mse_scale = ctx.mse_scale
    src_sig = ctx.src_sig
    skipped_tensors = ctx.skipped_tensors
    codebook_dtype = ctx.codebook_dtype
    codebook_cache_dir = ctx.codebook_cache_dir
    awq_activations = ctx.awq_activations
    outlier_frac = ctx.outlier_frac
    normalization = ctx.normalization
    max_tensors = ctx.max_tensors
    error_compensation = ctx.error_compensation
    manifest = ctx.manifest
    total_index_bytes = ctx.total_index_bytes

    for stage_i in range(n_stages):
        _check_ram_cap()
        k_spec = stage_spec_for_candidate(ctx, c, stage_i)
        if k_spec is None:
            continue

        _report_progress(
            progress_file,
            f"Quantizing {c['name']} ({stream_index}) | Stage {stage_i + 1}/{n_stages} (Spec: {k_spec})",
        )
        safe = _safe_tensor_name(c["name"])
        is_scalar_stage = isinstance(k_spec, str) and k_spec.startswith("s")
        if is_scalar_stage:
            k = 1 << int(k_spec[1:])
            v_res = c["vectors_residual"].reshape(-1, 1)
            c_group_size = 1
        else:
            k = int(k_spec)
            v_res = c["vectors_residual"]
            c_group_size = c["group_size"]

        if backend == "torch":
            c["vectors_orig"] = _onload(c["vectors_orig"], resolved_device)
            v_res = _onload(v_res, resolved_device)
            if c["decoded_sum"] is not None:
                c["decoded_sum"] = _onload(c["decoded_sum"], resolved_device)

        cache_key = (
            _codebook_cache_key(
                [
                    "per-tensor",
                    src_sig,
                    c["name"],
                    c_group_size,
                    k,
                    sample_vectors,
                    iterations,
                    backend,
                    normalization,
                    rotation,
                    rotation_seed,
                    outlier_frac,
                    max_tensors,
                    stage_i,
                    _weights_digest(c.get("sample_weights")),
                ]
            )
            if stage_i == 0
            else None
        )
        vw = c.get("vector_weights") if not is_scalar_stage else None
        sw = c.get("sample_weights") if not is_scalar_stage else None
        cached = _codebook_cache_load(codebook_cache_dir, cache_key) if cache_key else None
        if cached is not None:
            cb = cached
            training_count = sample_vectors or len(v_res)
        else:
            training, sw_train = _sample_vectors_and_weights(v_res, sw, sample_vectors)
            cb_seed = _derive_seed(
                ["per-tensor", src_sig, c["name"], c_group_size, k, stage_i]
            )
            cb, _, _ = learn_codebook_auto(
                training,
                min(k, len(training)),
                iterations,
                backend,
                resolved_device,
                vector_weights=vw,
                seed=cb_seed,
                sample_weights=sw_train,
            )
            training_count = len(training)
            if cache_key:
                _codebook_cache_save(codebook_cache_dir, cache_key, cb)

        # Round centroids to storage precision BEFORE assignment so the
        # indices, metrics, and on-disk codebook all agree exactly.
        cb, cb_dtype = _cast_codebook_storage(cb, dtype=codebook_dtype)
        cb_path = tensor_dir / (
            f"{safe}.codebook.f32" if n_stages == 1 else f"{safe}.s{stage_i}.codebook.f32"
        )
        _write_codebook(cb_path, cb, dtype=cb_dtype)

        indices, _ = quantize_vectors_auto(v_res, cb, backend, resolved_device, vector_weights=vw)
        c["stages_data"][stage_i] = {
            "cb": cb,
            "indices": indices,
            "group_size": c_group_size,
        }
        index_bits = _index_bits_for_size(len(cb))
        idx_path = tensor_dir / (
            f"{safe}.indices" if n_stages == 1 else f"{safe}.s{stage_i}.indices"
        )
        _, idx_encoding = _write_indices(idx_path, indices, index_bits)
        stage_bytes = idx_path.stat().st_size
        total_index_bytes += stage_bytes

        decoded_raw = _decode_to_vectors_format(v_res, cb, indices, backend, resolved_device)
        decoded = decoded_raw.reshape(c["vectors_residual"].shape) if is_scalar_stage else decoded_raw
        if c["decoded_sum"] is None:
            c["decoded_sum"] = decoded
        else:
            c["decoded_sum"] = c["decoded_sum"] + decoded
        c["vectors_residual"] = _vectors_subtract(c["vectors_orig"], c["decoded_sum"])

        if backend == "torch":
            c["vectors_residual"] = _offload(c["vectors_residual"])
            c["decoded_sum"] = _offload(c["decoded_sum"])
            c["vectors_orig"] = _offload(c["vectors_orig"])
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass

        if (
            n_stages > 1
            and em_aq_passes > 0
            and stage_i == n_stages - 1
        ):
            c.pop("vectors_residual", None)
            c.pop("decoded_sum", None)
            import gc
            gc.collect()

        c["stages_meta"].append(
            {
                "stage": stage_i,
                "codebook": str(cb_path.relative_to(out_dir)),
                "codebook_size": len(cb),
                "codebook_dtype": cb_dtype,
                "index_bits": index_bits,
                "packed": index_bits % 8 != 0,
                "encoding": idx_encoding,
                "indices": str(idx_path.relative_to(out_dir)),
                "index_bytes": stage_bytes,
                "training_vector_count": training_count,
                "codebook_family": c["family"],
                "group_size": c_group_size,
            }
        )

    compensated = False
    if error_compensation:
        compensated = maybe_compensate_candidate(
            c,
            backend=backend,
            awq_activations=awq_activations,
            resolved_device=resolved_device,
            progress_file=progress_file,
            out_dir=out_dir,
        )

    if n_stages > 1 and em_aq_passes > 0 and not compensated:
        _run_em_aq_refinement(
            candidates=[c],
            n_stages=n_stages,
            skipped_tensors=skipped_tensors,
            sample_vectors=sample_vectors,
            backend=backend,
            resolved_device=resolved_device,
            tensor_dir=tensor_dir,
            progress_file=progress_file,
            em_aq_passes=em_aq_passes,
        )

    try:
        _refine_scales_ls(c, mse_scale=mse_scale, block_scale_size=block_scale_size, out_dir=out_dir)
    except Exception as _e:  # refinement is optional; never fail the pack
        c.pop("_mse_v", None)
        _report_progress(progress_file, f"  mse_scale refinement skipped: {_e}")

    _BG_WRITER.wait()
    manifest["tensors"].append(
        _finalize_tensor_manifest_entry(
            c,
            n_stages=n_stages,
            group_size=group_size,
            block_scale_size=block_scale_size,
            rotation=rotation,
            backend=backend,
            out_dir=out_dir,
            tensor_dir=tensor_dir,
        )
    )
    ctx.total_index_bytes = total_index_bytes
