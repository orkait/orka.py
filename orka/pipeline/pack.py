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
    _write_salient,
)
from orka._runtime import (
    _BG_WRITER,
    _maybe_fallback_cuda_to_cpu,
    _resolve_torch_device,
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
    progress_file: Path | None = None,
    sensitivity_map: dict | None = None,
    codebook_cache_dir: Path | None = None,
    block_scale_size: int = 32,
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
        "awq",
        "awq-block-max",
        "slrq-block",
    }:
        raise ValueError(
            "normalization must be 'none', 'block-max', 'awq', 'awq-block-max', or 'slrq-block'"
        )
    if rotation not in {"none", "orthogonal", "hadamard"}:
        raise ValueError("rotation must be 'none', 'orthogonal', or 'hadamard'")
    if backend == "torch":
        device = _maybe_fallback_cuda_to_cpu(device, backend)
        resolved_device = str(_resolve_torch_device(device))
    else:
        resolved_device = "cpu"

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
            fam: [int(k) for k in stages] for fam, stages in family_stages_map.items()
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
        "sample_vectors": sample_vectors,
        "backend": backend,
        "device": resolved_device,
        "normalization": normalization,
        "outlier_frac": outlier_frac,
        "rotation": rotation,
        "rotation_seed": rotation_seed,
        "awq_enabled": awq_activations is not None,
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

    def _prefetch_worker():
        try:
            for i, (name, tensor) in enumerate(_load_tensors(source)):
                if max_tensors is not None and prefetch_queue.qsize() + len(candidates) >= max_tensors:
                    break
                shape = _tensor_shape(tensor)
                if len(shape) < 2:
                    _passthrough[name] = tensor
                    continue
                
                # Skipped tensors stay FP16 in the artifact (passthrough), not quantized.
                if name.replace(".weight", "") in skipped_tensors or name in skipped_tensors:
                    _passthrough[name] = tensor
                    continue

                row_scales = None
                source_flat = None
                awq_col_scales = None
                salient_weights = None
                salient_indices = None
                if normalization in {"block-max", "awq", "awq-block-max", "slrq-block"}:
                    (
                        tensor, row_scales, source_flat, awq_col_scales,
                        salient_weights, salient_indices
                    ) = _apply_normalization(
                        tensor, name, normalization, awq_activations, awq_alpha,
                        block_scale_size, backend, resolved_device, awq_fallbacks,
                    )

                # Capture pre-rotation flat when rotation is on but normalization didn't set it.
                # _stage_quality_metrics needs source_flat to compare un-rotated decode output.
                if source_flat is None and rotation != "none":
                    if backend == "torch":
                        _, _arr = _torch_f32(tensor, resolved_device)
                        source_flat = _arr.reshape(-1).detach().cpu()
                    else:
                        source_flat = _numpy_float32_array(tensor).reshape(-1)

                tensor_seed = None
                if rotation == "orthogonal":
                    tensor, tensor_seed = _rotate_tensor_to_2d(
                        tensor, name, rotation, rotation_seed, backend, resolved_device
                    )

                if backend == "torch":
                    packed_values, padded_values, vectors = _torch_vectors_from_tensor(
                        tensor, group_size, max_values_per_tensor, resolved_device
                    )
                else:
                    packed_values, padded_values, vectors = _numpy_vectors_from_tensor(
                        tensor, group_size, max_values_per_tensor
                    )
                
                vw = None
                if (awq_activations is not None and name in awq_activations and shape[-1] % group_size == 0):
                    import torch
                    H_diag = torch.as_tensor(awq_activations[name], dtype=torch.float32).pow(2).mean(dim=0)
                    vw = H_diag.reshape(-1, group_size).mean(dim=0).clamp(min=1e-6).tolist()

                prefetch_queue.put({
                    "name": name, "shape": shape, "source_flat": source_flat,
                    "packed_values": packed_values, "padded_values": padded_values,
                    "vectors": vectors, "row_scales": row_scales, "awq_col_scales": awq_col_scales,
                    "salient_weights": salient_weights, "salient_indices": salient_indices,
                    "normalization": normalization, "block_scale_size": block_scale_size if normalization in ("block-max", "awq-block-max", "slrq-block") else None,
                    "family": classify_tensor_family(name), "rotation_seed": tensor_seed,
                    "vector_weights": vw, "stages_data": {},
                })
        except BaseException as exc:
            _prefetch_exc.append(exc)
        finally:
            prefetch_done.set()

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
        
        positions, values, new_vectors = _extract_outliers(c["vectors"], outlier_frac, c["packed_values"])
        c["outlier_positions"] = positions
        c["outlier_values"] = values
        c["vectors"] = _offload(new_vectors)
        c["vectors_orig"] = c["vectors"]
        c["vectors_residual"] = c["vectors"]
        c["decoded_sum"] = None
        c["stages_meta"] = []
        candidates.append(c)
        prefetch_queue.task_done()

    prefetch_thread.join()

    if _prefetch_exc:
        raise RuntimeError(f"prefetch worker failed: {_prefetch_exc[0]}") from _prefetch_exc[0]
    if not candidates:
        raise RuntimeError(
            "prefetch worker produced 0 candidates - no quantizable tensors found "
            "(check model path, tensor shapes, and device errors above)"
        )

    if _passthrough:
        passthrough_path = out_dir / "passthrough.safetensors"
        _write_passthrough_tensors(passthrough_path, _passthrough)
        manifest["passthrough_count"] = len(_passthrough)

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
            base_name = c["name"].replace(".weight", "")
            if base_name in skipped_tensors or c["name"] in skipped_tensors:
                continue
            _report_progress(
                progress_file,
                f"Quantizing {c['name']} ({i + 1}/{len(candidates)}) | Stage {stage_i + 1}/{n_stages}",
            )
            safe = _safe_tensor_name(c["name"])
            if backend == "torch":
                c["vectors_orig"] = _onload(c["vectors_orig"], resolved_device)
                c["vectors_residual"] = _onload(c["vectors_residual"], resolved_device)
                if c["decoded_sum"] is not None:
                    c["decoded_sum"] = _onload(c["decoded_sum"], resolved_device)
            if family_stages_resolved is not None:
                stages_for_c = family_stages_resolved[c["family"]]
                if stage_i >= len(stages_for_c):
                    continue
                k = stages_for_c[stage_i]
                training = _sample_vector_rows(c["vectors_residual"], sample_vectors)
                cb_seed = _derive_seed(
                    ["family-mixed", src_sig, c["name"], group_size, k, stage_i]
                )
                cb, _, _ = learn_codebook_auto(
                    training,
                    min(k, len(training)),
                    iterations,
                    backend,
                    resolved_device,
                    seed=cb_seed,
                )
                training_count = len(training)
                cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                _write_codebook(cb_path, cb)
            elif codebook_mode in {"global", "family"}:
                k = stages_spec[stage_i]
                key = "global" if codebook_mode == "global" else c["family"]
                cb, cb_path = stage_codebooks[key]
                training_count = sample_vectors or len(c["vectors_residual"])
            else:
                k = stages_spec[stage_i]
                cache_key = (
                    _codebook_cache_key(
                        [
                            "per-tensor",
                            src_sig,
                            c["name"],
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
                    training_count = sample_vectors or len(c["vectors_residual"])
                else:
                    training = _sample_vector_rows(
                        c["vectors_residual"], sample_vectors
                    )
                    vw = c.get("vector_weights")

                    cb_seed = _derive_seed(
                        ["per-tensor", src_sig, c["name"], group_size, k, stage_i]
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
                    training_count = len(training)
                    if cache_key:
                        _codebook_cache_save(codebook_cache_dir, cache_key, cb)
                if n_stages == 1:
                    cb_path = tensor_dir / f"{safe}.codebook.f32"
                else:
                    cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                _write_codebook(cb_path, cb)

            indices, _ = quantize_vectors_auto(
                c["vectors_residual"], cb, backend, resolved_device
            )
            
            # Cache for joint refinement
            c["stages_data"][stage_i] = {
                "cb": cb,
                "indices": indices
            }
            index_bits = _index_bits_for_size(len(cb))
            if n_stages == 1:
                idx_path = tensor_dir / f"{safe}.indices"
            else:
                idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
            _write_indices(idx_path, indices, index_bits)
            stage_bytes = idx_path.stat().st_size
            total_index_bytes += stage_bytes

            decoded = _decode_to_vectors_format(
                c["vectors_orig"], cb, indices, backend, resolved_device
            )
            if c["decoded_sum"] is None:
                c["decoded_sum"] = decoded
            else:
                if _is_torch_tensor(c["decoded_sum"]):
                    c["decoded_sum"] = c["decoded_sum"] + decoded
                elif _is_numpy_array(c["decoded_sum"]):
                    c["decoded_sum"] = c["decoded_sum"] + decoded
                else:
                    c["decoded_sum"] = [
                        [a + b for a, b in zip(ra, rb)]
                        for ra, rb in zip(c["decoded_sum"], decoded)
                    ]
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
                except Exception:
                    pass

            c["stages_meta"].append(
                {
                    "stage": stage_i,
                    "codebook": str(cb_path.relative_to(out_dir)),
                    "codebook_size": len(cb),
                    "index_bits": index_bits,
                    "indices": str(idx_path.relative_to(out_dir)),
                    "index_bytes": stage_bytes,
                    "training_vector_count": training_count,
                    "codebook_family": c["family"],
                }
            )

    # Joint-Optimized Additive Quantization (AQLM EM-style)
    # We run the greedy stage-by-stage first, then do a joint refinement pass.
    if n_stages > 1 and codebook_mode == "per-tensor":
        _report_progress(progress_file, "--- Starting Joint Optimization (EM-AQ) ---")
        joint_iterations = 3
        
        # Pre-calculate full sum once
        for i, c in enumerate(candidates):
            base_name = c["name"].replace(".weight", "")
            if base_name in skipped_tensors or c["name"] in skipped_tensors:
                continue
            
            full_sum = None
            for stage_i in range(n_stages):
                sd = c["stages_data"][stage_i]
                dec = _decode_to_vectors_format(
                    c["vectors_orig"], sd["cb"], sd["indices"], backend, resolved_device
                )
                full_sum = dec if full_sum is None else full_sum + dec
            c["current_full_sum"] = full_sum

        for joint_iter in range(joint_iterations):
            _report_progress(
                progress_file,
                f"Joint Refinement Pass {joint_iter + 1}/{joint_iterations}",
            )
            for stage_i in range(n_stages):
                _report_progress(progress_file, f"    Refining stage {stage_i + 1}/{n_stages}...")
                for i, c in enumerate(candidates):
                    base_name = c["name"].replace(".weight", "")
                    if base_name in skipped_tensors or c["name"] in skipped_tensors:
                        continue

                    k = c["stages_meta"][stage_i]["codebook_size"]

                    # When k >= sample_vectors the codebook already memorizes the sample;
                    # joint refinement gives near-zero quality gain but dominates runtime.
                    if sample_vectors is not None and k >= sample_vectors:
                        continue

                    # O(1) Residual Update: target = orig - (full_sum - current_stage_dec)
                    sd = c["stages_data"][stage_i]
                    old_dec = _decode_to_vectors_format(
                        c["vectors_orig"], sd["cb"], sd["indices"], backend, resolved_device
                    )
                    
                    target = _vectors_subtract(c["vectors_orig"], (c["current_full_sum"] - old_dec))
                    training = _sample_vector_rows(target, sample_vectors)
                    vw = c.get("vector_weights")

                    cb, _, _ = learn_codebook_auto(
                        training, min(k, len(training)), 2, backend, resolved_device,
                        vector_weights=vw, initial_codebook=sd["cb"],
                    )

                    indices, _ = quantize_vectors_auto(target, cb, backend, resolved_device)
                    
                    # Update full sum: subtract old dec, add new dec
                    new_dec = _decode_to_vectors_format(c["vectors_orig"], cb, indices, backend, resolved_device)
                    c["current_full_sum"] = (c["current_full_sum"] - old_dec) + new_dec
                    
                    # Update cache
                    c["stages_data"][stage_i] = {"cb": cb, "indices": indices}

                    safe = _safe_tensor_name(c["name"])
                    cb_path = tensor_dir / f"{safe}.s{stage_i}.codebook.f32"
                    _BG_WRITER.submit(_write_codebook, cb_path, cb)

                    idx_path = tensor_dir / f"{safe}.s{stage_i}.indices"
                    _BG_WRITER.submit(_write_indices, idx_path, indices.cpu() if hasattr(indices, "cpu") else indices, c["stages_meta"][stage_i]["index_bits"])
                    c["decoded_sum"] = None

        # Sync decoded_sum after all passes
        for c in candidates:
            if "current_full_sum" in c:
                c["decoded_sum"] = _offload(c["current_full_sum"])
                del c["current_full_sum"]
            
            # Update metrics in manifest with joint-refined values
            refined_metrics = _stage_quality_metrics(c, backend)
            c["refined_metrics"] = refined_metrics

    _BG_WRITER.wait()

    _report_progress(progress_file, "--- Writing packed tensors & generating manifest ---")
    for i, c in enumerate(candidates):
        base_name = c["name"].replace(".weight", "")
        if base_name in skipped_tensors or c["name"] in skipped_tensors:
            continue
        _report_progress(progress_file, f"  Writing {c['name']} ({i+1}/{len(candidates)})...")
        safe = _safe_tensor_name(c["name"])
        scale_path = None
        scale_bytes = 0
        scale_count = 0
        if c["normalization"] == "awq":
            scale_path = tensor_dir / f"{safe}.col_l2_scale.f32"
            _write_f32_vector(scale_path, c["row_scales"])
            scale_bytes = scale_path.stat().st_size
            scale_count = len(c["row_scales"])
        elif c["normalization"] in ("block-max", "slrq-block"):
            scale_path = tensor_dir / f"{safe}.block_max_scale.f32"
            _write_f32_vector(scale_path, c["row_scales"])
            scale_bytes = scale_path.stat().st_size
            scale_count = len(c["row_scales"])
        elif c["normalization"] == "awq-block-max":
            scale_path = tensor_dir / f"{safe}.block_max_scale.f32"
            _write_f32_vector(scale_path, c["row_scales"])
            scale_bytes = scale_path.stat().st_size
            scale_count = len(c["row_scales"])

        awq_col_meta = None
        if (
            c["normalization"] == "awq-block-max"
            and c.get("awq_col_scales") is not None
        ):
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
            _write_outliers(
                out_idx_path, out_val_path, c["outlier_positions"], c["outlier_values"]
            )
            outlier_meta = {
                "count": int(len(c["outlier_positions"])),
                "positions": str(out_idx_path.relative_to(out_dir)),
                "values": str(out_val_path.relative_to(out_dir)),
                "positions_bytes": out_idx_path.stat().st_size,
                "values_bytes": out_val_path.stat().st_size,
            }

        salient_meta = None
        if c.get("salient_indices") is not None:
            s_idx_path = tensor_dir / f"{safe}.salient.idx"
            s_val_path = tensor_dir / f"{safe}.salient.val"
            
            sw = c["salient_weights"].numpy() if hasattr(c["salient_weights"], "numpy") else c["salient_weights"]
            si = c["salient_indices"].numpy() if hasattr(c["salient_indices"], "numpy") else c["salient_indices"]
            
            sw.astype("<f4").tofile(str(s_val_path))
            si.astype("<u4").tofile(str(s_idx_path))
            
            salient_meta = {
                "count": int(len(sw)),
                "indices": str(s_idx_path.relative_to(out_dir)),
                "weights": str(s_val_path.relative_to(out_dir)),
                "indices_bytes": s_idx_path.stat().st_size,
                "weights_bytes": s_val_path.stat().st_size,
            }

        metrics = c.get("refined_metrics") or _stage_quality_metrics(c, backend)
        first = c["stages_meta"][0]
        last_idx_path = tensor_dir / (
            f"{safe}.indices" if n_stages == 1 else f"{safe}.s0.indices"
        )
        index_bytes_total = sum(s["index_bytes"] for s in c["stages_meta"])
        manifest["tensors"].append(
            {
                "name": c["name"],
                "shape": c["shape"],
                "packed_values": c["packed_values"],
                "padded_values": c["padded_values"],
                "vector_count": len(c["vectors_orig"]),
                "training_vector_count": first["training_vector_count"],
                "group_size": group_size,
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
                "indices": str(last_idx_path.relative_to(out_dir)),
                "codebook": c["stages_meta"][0]["codebook"],
                "codebook_family": c["family"],
                "normalization": c["normalization"],
                "scales": str(scale_path.relative_to(out_dir)) if scale_path else None,
                "scale_count": scale_count,
                "scale_bytes": scale_bytes,
                "block_scale_size": block_scale_size
                if c["normalization"] in ("block-max", "awq-block-max", "slrq-block")
                else None,
                "awq_col_scales": awq_col_meta,
                "outliers": outlier_meta,
                "salient": salient_meta,
                "rotation_seed": c.get("rotation_seed"),
                "rotation": rotation if c.get("rotation_seed") is not None else "none",
            }
        )

    manifest["total_index_bytes"] = total_index_bytes
    manifest["tensor_count"] = len(manifest["tensors"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest

