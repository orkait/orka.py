"""``pack_checkpoint``: source -> normalize -> rotate -> outliers -> RVQ stages
-> joint EM-AQ -> manifest write.

Currently a single ~700 LOC orchestrator. Splitting into named phases is a
follow-up; would not change semantics.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Sequence

from orka._format import (
    ORKA_VERSION,
    _write_codebook,
    _write_f32_vector,
    _write_indices,
    _write_outliers,
    _write_passthrough_tensors,
    _write_pillars,
    _write_salient,
)
from orka._runtime import (
    _BG_WRITER,
    _maybe_fallback_cuda_to_cpu,
    _resolve_torch_device,
    _check_ram_cap,
)
from orka._tensor import (
    _concat_vector_parts,
    _decode_to_vectors_format,
    _is_numpy_array,
    _is_torch_tensor,
    _numpy_float32_array,
    _sample_vector_rows,
    _tensor_shape,
    _torch_f32,
    _vectors_subtract,
)
from orka._util import (
    _derive_seed,
    _index_bits_for_size,
    _report_progress,
    _safe_tensor_name,
    _source_signature,
)
from orka._checkpoint import _load_tensors
from orka._features import AWQ_DISABLED_MESSAGE, awq_feature_enabled
from orka.codebook import (
    _codebook_cache_key,
    _codebook_cache_load,
    _codebook_cache_save,
    learn_codebook_auto,
    quantize_vectors_auto,
)
from orka.metrics import _stage_quality_metrics
from orka.quant import classify_tensor_family
from orka.transforms import (
    _apply_normalization,
    _extract_outliers,
    _rotate_tensor_to_2d,
)


def _weights_digest(sample_weights) -> str:
    """Cache-key component for importance weights. Content-addressed so a
    cached codebook is never reused when calibration activations changed."""
    if sample_weights is None:
        return "unweighted"
    import hashlib

    arr = (
        sample_weights.detach().cpu().numpy()
        if hasattr(sample_weights, "detach")
        else sample_weights
    )
    import numpy as np

    payload = np.asarray(arr, dtype="<f4").tobytes()
    return "sw-" + hashlib.blake2b(payload, digest_size=8).hexdigest()


def _sample_vectors_and_weights(vectors, weights, sample_vectors: int | None):
    """Sample training rows and their per-sample weights at identical positions.

    Mirrors ``_sample_vector_rows`` (deterministic linspace positions) so the
    weight of each sampled row stays aligned with the row itself.
    """
    if (
        sample_vectors is None
        or sample_vectors <= 0
        or sample_vectors >= len(vectors)
    ):
        return vectors, weights
    if _is_torch_tensor(vectors):
        import torch

        positions = (
            torch.linspace(
                0,
                len(vectors) - 1,
                steps=sample_vectors,
                device=vectors.device,
                dtype=torch.float64,
            )
            .round()
            .to(dtype=torch.long)
            .clamp_(max=len(vectors) - 1)
        )
        sampled = vectors.index_select(0, positions)
        if weights is None:
            return sampled, None
        if _is_torch_tensor(weights):
            return sampled, weights.index_select(0, positions.to(weights.device))
        return sampled, weights[positions.detach().cpu().numpy()]
    import numpy as np

    positions = np.linspace(0, len(vectors) - 1, sample_vectors, dtype=np.int64)
    sampled = vectors[positions]
    return sampled, (weights[positions] if weights is not None else None)


def _numpy_vectors_from_tensor(tensor: object, group_size: int, limit: int | None):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy backend requires numpy") from exc
    flat = _numpy_float32_array(tensor).reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = np.pad(flat, (0, group_size - remainder), mode="constant")
    return original_len, int(flat.shape[0]), flat.reshape(-1, group_size)


def _torch_vectors_from_tensor(
    tensor: object, group_size: int, limit: int | None, device: str
):
    import torch

    _, arr = _torch_f32(tensor, device)
    flat = arr.reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = torch.nn.functional.pad(flat, (0, group_size - remainder))
    return original_len, int(flat.shape[0]), flat.reshape(-1, group_size)


def _persist_tensor_sidecars(c: dict, tensor_dir: Path, out_dir: Path) -> tuple:
    """Write per-tensor scale / awq_col / outlier / salient sidecars.

    Returns (scale_path, scale_bytes, scale_count, awq_col_meta, outlier_meta, salient_meta, pillar_meta).
    """
    safe = _safe_tensor_name(c["name"])
    scale_path = None
    scale_bytes = 0
    scale_count = 0
    norm = c["normalization"]
    if norm == "awq":
        scale_path = tensor_dir / f"{safe}.col_l2_scale.f32"
        _write_f32_vector(scale_path, c["row_scales"])
        scale_bytes = scale_path.stat().st_size
        scale_count = len(c["row_scales"])
    elif norm in ("block-max", "channel-block-max", "slrq-block", "awq-block-max"):
        scale_path = tensor_dir / f"{safe}.block_max_scale.f32"
        _write_f32_vector(scale_path, c["row_scales"])
        scale_bytes = scale_path.stat().st_size
        scale_count = len(c["row_scales"])

    awq_col_meta = None
    if norm == "awq-block-max" and c.get("awq_col_scales") is not None:
        awq_col_path = tensor_dir / f"{safe}.awq_col_scale.f32"
        _write_f32_vector(awq_col_path, c["awq_col_scales"])
        awq_col_meta = {
            "path": str(awq_col_path.relative_to(out_dir)),
            "count": len(c["awq_col_scales"]),
            "bytes": awq_col_path.stat().st_size,
        }

    outlier_meta = None
    if c.get("outlier_positions") is not None and len(c["outlier_positions"]) > 0:
        out_idx_path = tensor_dir / f"{safe}.outliers.idx"
        out_val_path = tensor_dir / f"{safe}.outliers.val"
        positions_dtype, values_dtype = _write_outliers(
            out_idx_path, out_val_path, c["outlier_positions"], c["outlier_values"]
        )
        outlier_meta = {
            "count": int(len(c["outlier_positions"])),
            "positions": str(out_idx_path.relative_to(out_dir)),
            "values": str(out_val_path.relative_to(out_dir)),
            "positions_dtype": positions_dtype,
            "values_dtype": values_dtype,
            "positions_bytes": out_idx_path.stat().st_size,
            "values_bytes": out_val_path.stat().st_size,
        }

    pillar_meta = None
    if c.get("pillar_positions") is not None and len(c["pillar_positions"]) > 0:
        p_idx_path = tensor_dir / f"{safe}.pillars.idx"
        p_val_path = tensor_dir / f"{safe}.pillars.f2"
        _write_pillars(p_idx_path, p_val_path, c["pillar_positions"], c["pillar_values"])
        pillar_meta = {
            "count": int(len(c["pillar_positions"])),
            "positions": str(p_idx_path.relative_to(out_dir)),
            "values": str(p_val_path.relative_to(out_dir)),
            "positions_bytes": p_idx_path.stat().st_size,
            "values_bytes": p_val_path.stat().st_size,
        }

    salient_meta = None
    if c.get("salient_indices") is not None:
        s_idx_path = tensor_dir / f"{safe}.salient.idx"
        s_val_path = tensor_dir / f"{safe}.salient.val"
        indices_dtype, weights_dtype = _write_salient(
            s_idx_path, s_val_path, c["salient_indices"], c["salient_weights"]
        )
        salient_meta = {
            "count": int(len(c["salient_weights"])),
            "indices": str(s_idx_path.relative_to(out_dir)),
            "weights": str(s_val_path.relative_to(out_dir)),
            "indices_dtype": indices_dtype,
            "weights_dtype": weights_dtype,
            "indices_bytes": s_idx_path.stat().st_size,
            "weights_bytes": s_val_path.stat().st_size,
        }

    return scale_path, scale_bytes, scale_count, awq_col_meta, outlier_meta, salient_meta, pillar_meta


def _build_tensor_manifest_entry(
    c: dict,
    *,
    n_stages: int,
    group_size: int,
    block_scale_size: int,
    rotation: str,
    backend: str,
    out_dir: Path,
    tensor_dir: Path,
    scale_path,
    scale_bytes: int,
    scale_count: int,
    awq_col_meta,
    outlier_meta,
    salient_meta,
    pillar_meta,
) -> dict:
    """Build the per-tensor manifest dict entry."""
    safe = _safe_tensor_name(c["name"])
    metrics = c.get("refined_metrics") or _stage_quality_metrics(c, backend)
    first = c["stages_meta"][0]
    last_idx_path = tensor_dir / (
        f"{safe}.indices" if n_stages == 1 else f"{safe}.s0.indices"
    )
    index_bytes_total = sum(s["index_bytes"] for s in c["stages_meta"])
    vector_count = c.get("vector_count") or (c["packed_values"] // c["group_size"])
    return {
        "name": c["name"],
        "shape": c["shape"],
        "packed_values": c["packed_values"],
        "padded_values": c["padded_values"],
        "vector_count": vector_count,
        "training_vector_count": first["training_vector_count"],
        "group_size": c["group_size"],
        "codebook_size": first["codebook_size"],
        "index_bits": first["index_bits"],
        "index_bytes": index_bytes_total,
        "n_stages": n_stages,
        "stages": c["stages_meta"],
        "total_bits_per_vector": sum(s["index_bits"] for s in c["stages_meta"]),
        "mse": metrics["mse"],
        "sse": metrics["sse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "max_abs_error": metrics["max_abs_error"],
        "source_l2_sq": metrics["source_l2_sq"],
        "reconstructed_l2_sq": metrics["reconstructed_l2_sq"],
        "dot": metrics["dot"],
        "relative_rmse": metrics["relative_rmse"],
        "cosine_similarity": metrics["cosine_similarity"],
        "sqnr": metrics["sqnr"],
        "indices": str(last_idx_path.relative_to(out_dir)),
        "codebook": c["stages_meta"][0]["codebook"],
        "codebook_family": c["family"],
        "normalization": c["normalization"],
        "scales": str(scale_path.relative_to(out_dir)) if scale_path else None,
        "scale_count": scale_count,
        "scale_bytes": scale_bytes,
        "block_scale_size": (
            block_scale_size
            if c["normalization"]
            in ("block-max", "channel-block-max", "awq-block-max", "slrq-block")
            else None
        ),
        "awq_col_scales": awq_col_meta,
        "outliers": outlier_meta,
        "pillars": pillar_meta,
        "salient": salient_meta,
        "rotation_seed": c.get("rotation_seed"),
        "rotation": c.get("rotation", "none"),
    }


def _release_candidate_payload(c: dict) -> None:
    for key in (
        "source_flat",
        "decoded_sum",
        "vectors_orig",
        "vectors",
        "vectors_residual",
        "row_scales",
        "awq_col_scales",
        "salient_weights",
        "salient_indices",
        "outlier_positions",
        "outlier_values",
        "pillar_positions",
        "pillar_values",
        "vector_weights",
        "sample_weights",
    ):
        if key in c:
            c[key] = None
    stages_data = c.get("stages_data")
    if isinstance(stages_data, dict):
        for stage_data in stages_data.values():
            if isinstance(stage_data, dict) and "indices" in stage_data:
                stage_data["indices"] = None


def _finalize_tensor_manifest_entry(
    c: dict,
    *,
    n_stages: int,
    group_size: int,
    block_scale_size: int,
    rotation: str,
    backend: str,
    out_dir: Path,
    tensor_dir: Path,
) -> dict:
    scale_path, scale_bytes, scale_count, awq_col_meta, outlier_meta, salient_meta, pillar_meta = (
        _persist_tensor_sidecars(c, tensor_dir, out_dir)
    )
    entry = _build_tensor_manifest_entry(
        c,
        n_stages=n_stages,
        group_size=group_size,
        block_scale_size=block_scale_size,
        rotation=rotation,
        backend=backend,
        out_dir=out_dir,
        tensor_dir=tensor_dir,
        scale_path=scale_path,
        scale_bytes=scale_bytes,
        scale_count=scale_count,
        awq_col_meta=awq_col_meta,
        outlier_meta=outlier_meta,
        salient_meta=salient_meta,
        pillar_meta=pillar_meta,
    )
    _release_candidate_payload(c)
    return entry


def _persist_manifest(
    *,
    candidates: list,
    manifest: dict,
    out_dir: Path,
    tensor_dir: Path,
    skipped_tensors: set,
    n_stages: int,
    group_size: int,
    block_scale_size: int,
    rotation: str,
    backend: str,
    total_index_bytes: int,
    progress_file: Path | None,
) -> None:
    """Write per-tensor sidecars + assemble manifest.json."""
    _report_progress(progress_file, "--- Writing packed tensors & generating manifest ---")
    for i, c in enumerate(candidates):
        base_name = c["name"].replace(".weight", "")
        if base_name in skipped_tensors or c["name"] in skipped_tensors:
            continue
        _report_progress(progress_file, f"  Writing {c['name']} ({i + 1}/{len(candidates)})...")
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

    manifest["total_index_bytes"] = total_index_bytes
    manifest["tensor_count"] = len(manifest["tensors"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

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
                    _BG_WRITER.submit(_write_codebook, cb_path, cb)
                    idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
                    _BG_WRITER.submit(
                        _write_indices,
                        idx_path,
                        indices.cpu() if hasattr(indices, "cpu") else indices,
                        c["stages_meta"][stage_i]["index_bits"],
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


def pack_checkpoint(
    source: Path,
    out_dir: Path,
    group_size: int,
    codebook_size: int = 256,
    iterations: int = 12,
    max_values_per_tensor: int | None = None,
    codebook_mode: str = "per-tensor",
    sample_vectors: int | None = None,
    backend: str = "auto",
    normalization: str = "none",
    device: str = "cpu",
    codebook_sizes: Sequence[int] | None = None,
    family_stages_map: dict[str, Sequence[int]] | None = None,
    outlier_frac: float = 0.0,
    rotation: str = "none",
    rotation_seed: int | None = None,
    awq_activations: dict | None = None,
    awq_alpha: float = 0.5,
    max_tensors: int | None = None,
    only_tensors: list[str] | None = None,
    only_tensors_passthrough: bool = True,
    progress_file: Path | None = None,
    sensitivity_map: dict | None = None,
    codebook_cache_dir: Path | None = None,
    block_scale_size: int = 32,
    em_aq_passes: int = 3,
    slrq_salient: bool = True,
    tensor_partition_count: int | None = None,
    tensor_partition_index: int | None = None,
) -> dict:
    if codebook_mode not in {"per-tensor", "global", "family"}:
        raise ValueError(
            "codebook_mode must be 'per-tensor', 'global', or 'family'"
        )
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if normalization not in {
        "none",
        "block-max",
        "channel-block-max",
        "awq",
        "awq-block-max",
        "slrq-block",
    }:
        raise ValueError(
            "normalization must be 'none', 'block-max', 'channel-block-max', 'awq', 'awq-block-max', or 'slrq-block'"
        )
    # The feature gate guards only the legacy AWQ *normalization* modes.
    # Calibration activations alone are allowed: they feed Hessian-proxy
    # importance weighting, which changes codebook learning, not the format.
    if normalization in {"awq", "awq-block-max"} and not awq_feature_enabled():
        raise RuntimeError(AWQ_DISABLED_MESSAGE)
    if rotation not in {"none", "orthogonal", "hadamard"}:
        raise ValueError("rotation must be 'none', 'orthogonal', or 'hadamard'")
    if backend == "torch":
        device = _maybe_fallback_cuda_to_cpu(device, backend)
        resolved_device = str(_resolve_torch_device(device))
    else:
        resolved_device = "cpu"

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

    if tensor_partition_count == 1:
        tensor_partition_count = 1
        tensor_partition_index = 0

    if rotation == "orthogonal" and rotation_seed is None:
        rotation_seed = int.from_bytes(os.urandom(8), "little")

    # Mixed-Precision Sensitivity Logic
    skipped_tensors = set()
    if sensitivity_map is not None:
        for entry in sensitivity_map.get("layers", []):
            if (
                entry["loss_delta"] > 1.5
                or "embed" in entry["layer"]
                or "lm_head" in entry["layer"]
            ):
                skipped_tensors.add(entry["layer"])

    src_sig = _source_signature(source)

    if family_stages_map is not None:
        if codebook_mode != "per-tensor":
            raise ValueError(
                "family_stages_map (mixed mode) requires codebook_mode='per-tensor'"
            )
        family_stages_resolved = {
            fam: [
                int(k) if not (isinstance(k, str) and k.startswith("s")) else k
                for k in stages
            ]
            for fam, stages in family_stages_map.items()
        }
        stages_spec = []
        n_stages = max(len(s) for s in family_stages_resolved.values())
    else:
        family_stages_resolved = None
        stages_spec = list(codebook_sizes) if codebook_sizes else [codebook_size]
        n_stages = len(stages_spec)
        if n_stages < 1:
            raise ValueError("at least one codebook stage is required")

    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "orka",
        "version": ORKA_VERSION,
        "source": str(source),
        "group_size": group_size,
        "requested_codebook_size": stages_spec[0]
        if (family_stages_resolved is None and n_stages == 1)
        else None,
        "codebook_sizes": list(stages_spec) if family_stages_resolved is None else None,
        "family_stages_map": family_stages_resolved,
        "n_stages": n_stages,
        "codebook_mode": codebook_mode,
        # Per-tensor mode adapts group size by family; per-tensor entries carry
        # the resolved value. Top-level group_size is the requested baseline.
        "dynamic_group_sizing": codebook_mode == "per-tensor",
        "sample_vectors": sample_vectors,
        "backend": backend,
        "device": resolved_device,
        "normalization": normalization,
        "outlier_frac": outlier_frac,
        "rotation": rotation,
        "rotation_seed": rotation_seed,
        "awq_enabled": awq_activations is not None,
        "hessian_weighted": awq_activations is not None,
        "em_aq_passes": em_aq_passes,
        "slrq_salient": slrq_salient,
        "tensor_partition_count": (
            None if tensor_partition_count is None else tensor_partition_count
        ),
        "tensor_partition_index": (
            None if tensor_partition_index is None else tensor_partition_index
        ),
        "tensors": [],
    }

    def _offload(t):
        if _is_torch_tensor(t):
            return t.detach().cpu()
        return t

    def _onload(t, device):
        if _is_torch_tensor(t):
            return t.to(device=device)
        return t

    candidates = []
    awq_fallbacks: list[str] = []
    _passthrough: dict[str, object] = {}

    # Prefetch Queue for Concurrency
    prefetch_queue = queue.Queue(maxsize=4)
    prefetch_done = threading.Event()
    _prefetch_exc: list[BaseException] = []
    _prefetch_state = {"candidate_count": 0}

    def _prefetch_worker():
        try:
            tensors_emitted = 0
            for name, tensor in _load_tensors(source):
                _check_ram_cap()

                shape = _tensor_shape(tensor)
                name_lower = name.lower()
                is_candidate = len(shape) >= 2

                # Exclude biases, norms, and architectural sidecars.
                if any(
                    x in name_lower
                    for x in (".bias", ".norm", ".layernorm", "rotary_emb", "attention.bias")
                ):
                    is_candidate = False

                # Non-candidates (norms, biases) always pass through so any
                # partial artifact stays completable.
                if not is_candidate:
                    _passthrough[name] = tensor
                    continue

                if only_tensors is not None:
                    base_name = name.replace(".weight", "")
                    if name not in only_tensors and base_name not in only_tensors:
                        # Unlisted candidates: passthrough by default (legacy
                        # behaviour); skipped entirely for partitioned runs
                        # (sequential packing) so partial artifacts stay small.
                        if only_tensors_passthrough:
                            _passthrough[name] = tensor
                        continue

                if max_tensors is not None and tensors_emitted >= max_tensors:
                    break
                
                # Skipped tensors stay FP16 in the artifact (passthrough), not quantized.
                if name.replace(".weight", "") in skipped_tensors or name in skipped_tensors:
                    _passthrough[name] = tensor
                    continue

                if tensor_partition_count is not None:
                    slot = _prefetch_state["candidate_count"] % tensor_partition_count
                    _prefetch_state["candidate_count"] += 1
                    if slot != tensor_partition_index:
                        continue
                else:
                    _prefetch_state["candidate_count"] += 1

                row_scales = None
                source_flat = None
                awq_col_scales = None
                salient_weights = None
                salient_indices = None
                if normalization in {
                    "block-max",
                    "channel-block-max",
                    "awq",
                    "awq-block-max",
                    "slrq-block",
                }:
                    (
                        tensor, row_scales, source_flat, awq_col_scales,
                        salient_weights, salient_indices
                    ) = _apply_normalization(
                        tensor, name, normalization, awq_activations, awq_alpha,
                        block_scale_size, backend, resolved_device, awq_fallbacks,
                        slrq_salient=slrq_salient,
                    )

                # --- Post-Normalization Pre-processing ---
                # Capture source_flat if not already set by normalization.
                # This is mandatory for quality metrics verification.
                if source_flat is None:
                    if backend == "torch":
                        _, _arr = _torch_f32(tensor, resolved_device)
                        source_flat = _arr.reshape(-1).detach().cpu()
                    else:
                        source_flat = _numpy_float32_array(tensor).reshape(-1)

                tensor_seed = None
                tensor_rotation = "none"
                if rotation in {"orthogonal", "hadamard"}:
                    tensor, tensor_seed = _rotate_tensor_to_2d(
                        tensor, name, rotation, rotation_seed, backend, resolved_device
                    )
                    tensor_rotation = rotation

                # --- DYNAMIC FAMILY-AWARE GROUP SIZING (per-tensor mode only) ---
                # Different layer types have different sensitivity to group size.
                # Attention: phase-sensitive, needs tight groups.
                # MLP: high channel redundancy, tolerates larger groups.
                # Embedding: linguistic pillars, needs high fidelity.
                # Shared codebooks (global/family) require one vector width across
                # all tensors they cover, so the override only applies per-tensor.
                family = classify_tensor_family(name)
                resolved_group_size = group_size
                if codebook_mode == "per-tensor":
                    if family == "embedding":
                        resolved_group_size = min(group_size, 8)
                    elif family == "attention":
                        # Attention projections are phase-sensitive; use tighter groups
                        resolved_group_size = max(4, group_size // 2) if group_size > 4 else group_size
                    elif family in ("mlp", "expert", "shared_expert"):
                        # MLP/expert layers have high channel redundancy; relax group size
                        resolved_group_size = min(group_size * 2, 32)
                    # router/other: keep the user-specified group_size

                if backend == "torch":
                    packed_values, padded_values, vectors = _torch_vectors_from_tensor(
                        tensor, resolved_group_size, max_values_per_tensor, resolved_device
                    )
                else:
                    packed_values, padded_values, vectors = _numpy_vectors_from_tensor(
                        tensor, resolved_group_size, max_values_per_tensor
                    )
                
                # --- HESSIAN-PROXY IMPORTANCE WEIGHTS ---
                # h_j = E[x_j^2] per input column from calibration activations.
                # Two weight sets feed weighted k-means:
                #   vector_weights  - within-group dimension pattern (global average),
                #                     scales the distance metric per dimension.
                #   sample_weights  - per-vector scalar = mean importance of the columns
                #                     that vector covers, tiled across rows (row-major
                #                     flatten). Pulls centroids toward high-energy
                #                     column groups in the Lloyd update.
                vw = None
                sw = None
                if (awq_activations is not None and name in awq_activations and shape[-1] % resolved_group_size == 0):
                    import torch
                    H_diag = torch.as_tensor(awq_activations[name], dtype=torch.float32).pow(2).mean(dim=0)
                    cols = int(shape[-1])
                    groups_per_row = cols // resolved_group_size
                    h_groups = H_diag.reshape(groups_per_row, resolved_group_size)
                    vw = h_groups.mean(dim=0).clamp(min=1e-6).tolist()
                    rows_count = padded_values // cols
                    if rows_count * cols == padded_values:
                        sw_row = h_groups.mean(dim=1).clamp(min=1e-6)
                        sw_row = sw_row / sw_row.mean()
                        sw_full = sw_row.repeat(rows_count)
                        if backend == "torch":
                            sw = sw_full
                        else:
                            sw = sw_full.numpy()

                prefetch_queue.put({
                    "name": name, "shape": shape, "source_flat": source_flat,
                    "packed_values": packed_values, "padded_values": padded_values,
                    "vectors": vectors, "row_scales": row_scales, "awq_col_scales": awq_col_scales,
                    "salient_weights": salient_weights, "salient_indices": salient_indices,
                    "normalization": normalization,
                    "block_scale_size": (
                        block_scale_size
                        if normalization
                        in ("block-max", "channel-block-max", "awq-block-max", "slrq-block")
                        else None
                    ),
                    "family": family, "rotation_seed": tensor_seed,
                    "rotation": tensor_rotation,
                    "group_size": resolved_group_size,
                    "vector_weights": vw, "sample_weights": sw, "stages_data": {},
                })
                tensors_emitted += 1

        except BaseException as exc:
            _prefetch_exc.append(exc)
        finally:
            prefetch_done.set()

    total_index_bytes = 0
    streamed_tensor_count = 0

    def _stage_spec_for_candidate(c: dict, stage_i: int):
        if family_stages_resolved is not None:
            stages_for_c = family_stages_resolved[c["family"]]
            if stage_i >= len(stages_for_c):
                return None
            return stages_for_c[stage_i]
        if stage_i >= len(stages_spec):
            return None
        return stages_spec[stage_i]

    def _process_streamed_per_tensor_candidate(c: dict, stream_index: int) -> None:
        nonlocal total_index_bytes
        for stage_i in range(n_stages):
            _check_ram_cap()
            k_spec = _stage_spec_for_candidate(c, stage_i)
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

            cb_path = tensor_dir / (
                f"{safe}.codebook.f32" if n_stages == 1 else f"{safe}.s{stage_i}.codebook.f32"
            )
            _write_codebook(cb_path, cb)

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
            _write_indices(idx_path, indices, index_bits)
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
                    "index_bits": index_bits,
                    "packed": index_bits % 8 != 0,
                    "indices": str(idx_path.relative_to(out_dir)),
                    "index_bytes": stage_bytes,
                    "training_vector_count": training_count,
                    "codebook_family": c["family"],
                    "group_size": c_group_size,
                }
            )

        if n_stages > 1 and em_aq_passes > 0:
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

    prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    prefetch_thread.start()

    while not prefetch_done.is_set() or not prefetch_queue.empty():
        if _prefetch_exc:
            break
        try:
            c = prefetch_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        _report_progress(progress_file, f"Prepared {c['name']} {c['shape']} (Ready for Quantization)")

        # --- Frequency-Aware Pillar Protection (SmolLM/Qwen research branch) ---
        is_embedding = c["name"].lower() in (
            "model.embed_tokens.weight", "gpt_neox.embed_in.weight", 
            "embed_out.weight", "lm_head.weight"
        )
        pillar_positions = None
        pillar_values = None

        if is_embedding and sensitivity_map and "top_tokens" in sensitivity_map:
            top_token_ids = sensitivity_map["top_tokens"]
            _report_progress(progress_file, f"    Applying Frequency-Aware Pillar Protection ({len(top_token_ids)} tokens)")

            vocab_size, hidden_dim = c["shape"][0], c["shape"][1]
            p_pos = []
            for tid in top_token_ids:
                if tid < vocab_size:
                    start = tid * hidden_dim
                    p_pos.extend(range(start, start + hidden_dim))

            if p_pos:
                if _is_torch_tensor(c["vectors"]):
                    import torch
                    import numpy as np
                    flat = c["vectors"].reshape(-1)
                    pillar_positions = np.array(p_pos, dtype=np.int64)
                    pillar_values = flat[pillar_positions].detach().cpu().numpy().astype(np.float32)
                    mask = torch.ones_like(flat)
                    mask[pillar_positions] = 0
                    c["vectors"] = (flat * mask).reshape(c["vectors"].shape)
                else:
                    import numpy as np
                    flat = c["vectors"].reshape(-1)
                    pillar_positions = np.array(p_pos, dtype=np.int64)
                    pillar_values = flat[pillar_positions].astype(np.float32)
                    flat[pillar_positions] = 0
                    c["vectors"] = flat.reshape(c["vectors"].shape)

        # --- Standard Outlier Extraction ---
        # If freq-aware didn't run, or if we want to extract additional magnitude outliers
        positions, values, new_vectors = _extract_outliers(c["vectors"], outlier_frac, c["packed_values"])

        if pillar_positions is not None:
            # Combine freq-aware pillars with magnitude outliers
            import numpy as np
            if positions is not None:
                c["outlier_positions"] = np.concatenate([pillar_positions, positions])
                c["outlier_values"] = np.concatenate([pillar_values, values])
            else:
                c["outlier_positions"] = pillar_positions
                c["outlier_values"] = pillar_values
        else:
            c["outlier_positions"] = positions
            c["outlier_values"] = values

        c["vectors"] = _offload(new_vectors)
        c["vectors_orig"] = c["vectors"]
        c["vectors_residual"] = c["vectors"]
        c["decoded_sum"] = None
        c["stages_meta"] = []
        if codebook_mode == "per-tensor":
            streamed_tensor_count += 1
            _process_streamed_per_tensor_candidate(c, streamed_tensor_count)
        else:
            candidates.append(c)
        prefetch_queue.task_done()

    prefetch_thread.join()

    if _prefetch_exc:
        exc = _prefetch_exc[0]
        # Preserve the original exception type when it carries semantic meaning
        # (e.g. SystemRAMExceededError, CappedOutOfMemoryError). Plain wrapping
        # to RuntimeError loses that signal for callers/CLI.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise exc
        from orka._runtime import SystemRAMExceededError, CappedOutOfMemoryError
        if isinstance(exc, (SystemRAMExceededError, CappedOutOfMemoryError)):
            raise type(exc)(f"prefetch worker: {exc}") from exc
        raise RuntimeError(f"prefetch worker failed: {exc}") from exc
    if not candidates and streamed_tensor_count == 0 and tensor_partition_count is None:
        raise RuntimeError(
            "prefetch worker produced 0 candidates - no quantizable tensors found "
            "(check model path, tensor shapes, and device errors above)"
        )
    if not candidates and streamed_tensor_count == 0 and tensor_partition_count is not None:
        if _prefetch_state["candidate_count"] == 0:
            raise RuntimeError(
                "prefetch worker produced 0 candidates - no quantizable tensors found "
                "(check model path, tensor shapes, and device errors above)"
            )

    if _passthrough:
        passthrough_path = out_dir / "passthrough.safetensors"
        _write_passthrough_tensors(passthrough_path, _passthrough)
        manifest["passthrough_count"] = len(_passthrough)

    if codebook_mode == "per-tensor":
        _BG_WRITER.wait()
        manifest["total_index_bytes"] = total_index_bytes
        manifest["tensor_count"] = len(manifest["tensors"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    total_index_bytes = 0

    for stage_i in range(n_stages):
        _report_progress(
            progress_file, f"--- Starting Stage {stage_i + 1}/{n_stages} ---"
        )
        stage_codebooks = {}
        if (
            family_stages_resolved is None
            and codebook_mode in {"global", "family"}
            and candidates
        ):
            k = stages_spec[stage_i]
            vector_groups = {}
            for c in candidates:
                key = "global" if codebook_mode == "global" else c["family"]
                vector_groups.setdefault(key, []).append(c["vectors_residual"])
            for key, parts in vector_groups.items():
                cache_key = (
                    _codebook_cache_key(
                        [
                            "shared",
                            src_sig,
                            codebook_mode,
                            key,
                            group_size,
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
                            "awq-weighted" if awq_activations else "unweighted",
                        ]
                    )
                    if stage_i == 0
                    else None
                )
                cached = (
                    _codebook_cache_load(codebook_cache_dir, cache_key)
                    if cache_key
                    else None
                )
                if cached is not None:
                    cb = cached
                    training_count = (
                        int(sample_vectors)
                        if sample_vectors
                        else sum(len(p) for p in parts)
                    )
                else:
                    if sample_vectors is None or sample_vectors <= 0:
                        sampled_parts = parts
                    else:
                        total_count = sum(len(p) for p in parts)
                        if total_count <= sample_vectors:
                            sampled_parts = parts
                        else:
                            sampled_parts = []
                            remaining_budget = int(sample_vectors)
                            for idx, p in enumerate(parts):
                                share = max(
                                    1, int(round(sample_vectors * len(p) / total_count))
                                )
                                share = min(share, len(p), remaining_budget)
                                if idx == len(parts) - 1:
                                    share = min(remaining_budget, len(p))
                                sampled_parts.append(_sample_vector_rows(p, share))
                                remaining_budget -= share
                                if remaining_budget <= 0:
                                    break
                    training = _concat_vector_parts(sampled_parts)
                    vw = None

                    cb_seed = _derive_seed(
                        ["shared", src_sig, codebook_mode, key, group_size, k, stage_i]
                    )
                    cb, _, _ = learn_codebook_auto(
                        training,
                        min(k, len(training)),
                        iterations,
                        backend,
                        resolved_device,
                        vector_weights=vw,
                        seed=cb_seed,
                    )
                    if cache_key:
                        _codebook_cache_save(codebook_cache_dir, cache_key, cb)
                if n_stages == 1:
                    cb_path = out_dir / "codebooks" / f"{key}.codebook.f32"
                else:
                    cb_path = out_dir / "codebooks" / f"{key}.s{stage_i}.codebook.f32"
                _write_codebook(cb_path, cb)
                stage_codebooks[key] = (cb, cb_path)

        for i, c in enumerate(candidates):
            _check_ram_cap()
            base_name = c["name"].replace(".weight", "")
            if base_name in skipped_tensors or c["name"] in skipped_tensors:
                continue
            
            # Current stage definition from spec
            k_spec = stages_spec[stage_i] if stage_i < len(stages_spec) else None
            if family_stages_resolved is not None:
                stages_for_c = family_stages_resolved[c["family"]]
                if stage_i >= len(stages_for_c):
                    continue
                k_spec = stages_for_c[stage_i]

            if k_spec is None:
                continue

            _report_progress(
                progress_file,
                f"Quantizing {c['name']} ({i + 1}/{len(candidates)}) | Stage {stage_i + 1}/{n_stages} (Spec: {k_spec})",
            )
            safe = _safe_tensor_name(c["name"])
            
            # --- SCALAR STAGE DETECTION ---
            is_scalar_stage = isinstance(k_spec, str) and k_spec.startswith("s")
            if is_scalar_stage:
                k = 1 << int(k_spec[1:])
                # Reshape residual to scalar [N*G, 1]
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
            
            # Learn or Load Codebook
            if codebook_mode in {"global", "family"}:
                # Shared codebooks are learned unweighted; assign unweighted too.
                vw = None
                key = "global" if codebook_mode == "global" else c["family"]
                cb, cb_path = stage_codebooks[key]
                training_count = sample_vectors or len(v_res)
            else:
                vw = c.get("vector_weights") if not is_scalar_stage else None
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
                cached = (
                    _codebook_cache_load(codebook_cache_dir, cache_key)
                    if cache_key
                    else None
                )
                if cached is not None:
                    cb = cached
                    training_count = sample_vectors or len(v_res)
                else:
                    sw = c.get("sample_weights") if not is_scalar_stage else None
                    training, sw_train = _sample_vectors_and_weights(
                        v_res, sw, sample_vectors
                    )
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
                
                cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                _write_codebook(cb_path, cb)

            indices, _ = quantize_vectors_auto(
                v_res, cb, backend, resolved_device, vector_weights=vw
            )
            
            # Cache for joint refinement
            c["stages_data"][stage_i] = {
                "cb": cb,
                "indices": indices,
                "group_size": c_group_size
            }
            index_bits = _index_bits_for_size(len(cb))
            idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
            _write_indices(idx_path, indices, index_bits)
            stage_bytes = idx_path.stat().st_size
            total_index_bytes += stage_bytes

            # Decode and update sum/residual
            # Re-group decoded scalar back to original vector group size if needed
            decoded_raw = _decode_to_vectors_format(
                v_res, cb, indices, backend, resolved_device
            )
            if is_scalar_stage:
                decoded = decoded_raw.reshape(c["vectors_residual"].shape)
            else:
                decoded = decoded_raw

            if c["decoded_sum"] is None:
                c["decoded_sum"] = decoded
            else:
                c["decoded_sum"] = c["decoded_sum"] + decoded
            
            c["vectors_residual"] = _vectors_subtract(
                c["vectors_orig"], c["decoded_sum"]
            )

            if backend == "torch":
                c["vectors_residual"] = _offload(c["vectors_residual"])
                c["decoded_sum"] = _offload(c["decoded_sum"])
                c["vectors_orig"] = _offload(c["vectors_orig"])
                try:
                    import torch as _t
                    _t.cuda.empty_cache()
                except Exception: pass
                
            # If we are doing EM-AQ, we don't need residual/decoded_sum in RAM
            # because EM-AQ recalculates them from vectors_orig.
            if n_stages > 1 and codebook_mode == "per-tensor" and em_aq_passes > 0 and stage_i == n_stages - 1:
                if "vectors_residual" in c:
                    del c["vectors_residual"]
                if "decoded_sum" in c:
                    del c["decoded_sum"]
                import gc
                gc.collect()

            c["stages_meta"].append(
                {
                    "stage": stage_i,
                    "codebook": str(cb_path.relative_to(out_dir)),
                    "codebook_size": len(cb),
                    "index_bits": index_bits,
                    "packed": index_bits % 8 != 0,
                    "indices": str(idx_path.relative_to(out_dir)),
                    "index_bytes": stage_bytes,
                    "training_vector_count": training_count,
                    "codebook_family": c["family"],
                    "group_size": c_group_size,
                }
            )

    if n_stages > 1 and codebook_mode == "per-tensor" and em_aq_passes > 0:
        _run_em_aq_refinement(
            candidates=candidates,
            n_stages=n_stages,
            skipped_tensors=skipped_tensors,
            sample_vectors=sample_vectors,
            backend=backend,
            resolved_device=resolved_device,
            tensor_dir=tensor_dir,
            progress_file=progress_file,
            em_aq_passes=em_aq_passes,
        )

    _BG_WRITER.wait()

    _persist_manifest(
        candidates=candidates,
        manifest=manifest,
        out_dir=out_dir,
        tensor_dir=tensor_dir,
        skipped_tensors=skipped_tensors,
        n_stages=n_stages,
        group_size=group_size,
        block_scale_size=block_scale_size,
        rotation=rotation,
        backend=backend,
        total_index_bytes=total_index_bytes,
        progress_file=progress_file,
    )
    return manifest
