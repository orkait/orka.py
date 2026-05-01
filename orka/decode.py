"""Artifact decode pipeline: verify, reconstruct (JSON/safetensors), report.

Mirrors the pack pipeline in reverse: stage-sum -> outlier-inject ->
un-rotate -> un-normalize -> salient-inject.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from orka.core import (
    ORKA_VERSION,
    _dir_size,
    _product,
    _reshape_flat,
)
from orka.io_format import (
    _flatten_float_values,
    _index_bit_spec,
    _load_tensors,
    _read_codebook,
    _read_f32_vector,
    _read_indices,
    _tensor_shape,
)
from orka.metrics import _quality_from_totals, quality_metrics_from_flat
from orka.transforms import (
    _apply_block_max_scales,
    _apply_col_l2_scales,
    _apply_row_l2_scales,
    _fwht_numpy,
    _generate_orthogonal_numpy,
    _read_outliers,
    _unrotate_flat,
)


def report_artifact(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    tensors = manifest.get("tensors", [])
    total_index_bytes = sum(int(t.get("index_bytes", 0)) for t in tensors)
    total_codebook_bytes = 0
    total_scale_bytes = sum(int(t.get("scale_bytes", 0)) for t in tensors)
    total_outlier_bytes = sum(
        int((t.get("outliers") or {}).get("positions_bytes", 0))
        + int((t.get("outliers") or {}).get("values_bytes", 0))
        for t in tensors
    )
    total_outlier_count = sum(
        int((t.get("outliers") or {}).get("count", 0)) for t in tensors
    )
    original_fp16_bytes = 0
    weighted_error = 0.0
    weighted_values = 0
    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0
    counted_codebooks = set()

    for tensor in tensors:
        cb_paths = [out_dir / s["codebook"] for s in tensor.get("stages", [])] or [
            out_dir / tensor["codebook"]
        ]
        for codebook_path in cb_paths:
            if codebook_path.exists() and codebook_path not in counted_codebooks:
                total_codebook_bytes += codebook_path.stat().st_size
                counted_codebooks.add(codebook_path)
        shape = [int(x) for x in tensor.get("shape", [])]
        if shape:
            original_fp16_bytes += _product(shape) * 2
        value_count = int(tensor.get("packed_values", 0))
        weighted_error += float(tensor.get("mse", 0.0)) * value_count
        weighted_values += value_count
        sse += float(tensor.get("sse", float(tensor.get("mse", 0.0)) * value_count))
        abs_error_sum += float(tensor.get("mae", 0.0)) * value_count
        max_abs_error = max(max_abs_error, float(tensor.get("max_abs_error", 0.0)))
        source_l2_sq += float(tensor.get("source_l2_sq", 0.0))
        reconstructed_l2_sq += float(tensor.get("reconstructed_l2_sq", 0.0))
        dot += float(tensor.get("dot", 0.0))

    artifact_bytes = _dir_size(out_dir)
    worst_tensors = sorted(
        (
            {
                "name": tensor.get("name"),
                "shape": tensor.get("shape"),
                "mse": tensor.get("mse", 0.0),
                "relative_rmse": tensor.get("relative_rmse"),
                "cosine_similarity": tensor.get("cosine_similarity"),
                "index_bytes": tensor.get("index_bytes", 0),
            }
            for tensor in tensors
        ),
        key=lambda item: float(item["mse"]),
        reverse=True,
    )[:10]

    compression_ratio = (
        original_fp16_bytes / artifact_bytes if artifact_bytes > 0 else 0.0
    )
    aggregate_metrics = _quality_from_totals(
        value_count=weighted_values,
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )
    return {
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "source": manifest.get("source"),
        "tensor_count": len(tensors),
        "group_size": manifest.get("group_size"),
        "requested_codebook_size": manifest.get("requested_codebook_size"),
        "codebook_mode": manifest.get("codebook_mode", "per-tensor"),
        "normalization": manifest.get("normalization", "none"),
        "total_index_bytes": total_index_bytes,
        "total_codebook_bytes": total_codebook_bytes,
        "total_scale_bytes": total_scale_bytes,
        "total_outlier_bytes": total_outlier_bytes,
        "total_outlier_count": total_outlier_count,
        "artifact_bytes": artifact_bytes,
        "original_fp16_bytes": original_fp16_bytes,
        "compression_ratio_fp16_to_artifact": compression_ratio,
        "weighted_mse": weighted_error / weighted_values if weighted_values else 0.0,
        "rmse": aggregate_metrics["rmse"],
        "mae": aggregate_metrics["mae"],
        "max_abs_error": aggregate_metrics["max_abs_error"],
        "relative_rmse": aggregate_metrics["relative_rmse"],
        "cosine_similarity": aggregate_metrics["cosine_similarity"],
        "source_norm": math.sqrt(source_l2_sq),
        "reconstructed_norm": math.sqrt(reconstructed_l2_sq),
        "worst_tensors": worst_tensors,
    }

def _decode_tensor(out_dir: Path, tensor_meta: dict) -> list[float]:
    group_size = int(tensor_meta["group_size"])
    padded_values = int(tensor_meta["padded_values"])
    index_count = math.ceil(padded_values / group_size)
    stages = tensor_meta.get("stages")
    if not stages:
        stages = [
            {
                "codebook": tensor_meta["codebook"],
                "codebook_size": int(tensor_meta["codebook_size"]),
                "index_bits": int(tensor_meta["index_bits"]),
                "indices": tensor_meta["indices"],
            }
        ]

    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        decoded_np = np.zeros(index_count * group_size, dtype=np.float32)
        for stage in stages:
            cb = np.fromfile(str(out_dir / stage["codebook"]), dtype="<f4").reshape(-1, group_size)
            idxs = np.asarray(_read_indices(out_dir / stage["indices"], int(stage["index_bits"]), index_count), dtype=np.int64)
            decoded_np += cb[idxs].reshape(-1)
        decoded = decoded_np[: int(tensor_meta["packed_values"])].tolist()
    else:
        decoded = [0.0] * (index_count * group_size)
        for stage in stages:
            cb = _read_codebook(
                out_dir / stage["codebook"], int(stage["codebook_size"]), group_size
            )
            idxs = _read_indices(
                out_dir / stage["indices"], int(stage["index_bits"]), index_count
            )
            offset = 0
            for index in idxs:
                row = cb[index]
                for j in range(group_size):
                    decoded[offset + j] += row[j]
                offset += group_size
        decoded = decoded[: int(tensor_meta["packed_values"])]
    outl = tensor_meta.get("outliers")
    if outl:
        positions, values = _read_outliers(
            out_dir / outl["positions"], out_dir / outl["values"]
        )
        for pos, val in zip(positions, values):
            decoded[int(pos)] = float(val)
            
    rotation = tensor_meta.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tensor_meta.get("rotation_seed") or 0)
        decoded = _unrotate_flat(decoded, tensor_meta["shape"], rotation, seed)
    norm = tensor_meta.get("normalization", "none")
    if norm == "row-l2":
        scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        decoded = _apply_row_l2_scales(decoded, tensor_meta["shape"], scales)
    elif norm in ("col-l2", "awq"):
        scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        decoded = _apply_col_l2_scales(decoded, tensor_meta["shape"], scales)
    elif norm in ("block-max", "slrq-block"):
        scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        block_size = int(tensor_meta.get("block_scale_size") or 32)
        decoded = _apply_block_max_scales(decoded, scales, block_size)

    elif norm == "awq-block-max":
        block_scales = _read_f32_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"])
        )
        block_size = int(tensor_meta.get("block_scale_size") or 32)
        decoded = _apply_block_max_scales(decoded, block_scales, block_size)
        awq_meta = tensor_meta.get("awq_col_scales")
        if awq_meta:
            awq_scales = _read_f32_vector(
                out_dir / awq_meta["path"], int(awq_meta["count"])
            )
            decoded = _apply_col_l2_scales(decoded, tensor_meta["shape"], awq_scales)

    salient = tensor_meta.get("salient")
    if salient:
        s_idx = np.fromfile(str(out_dir / salient["indices"]), dtype="<u4")
        s_val = np.fromfile(str(out_dir / salient["weights"]), dtype="<f4")
        
        # SLRQ: re-inject salient weights AFTER scaling to avoid double-scaling.
        for b_idx, (local_idx, weight) in enumerate(zip(s_idx, s_val)):
            flat_idx = b_idx * int(tensor_meta.get("block_scale_size", 16)) + int(local_idx)
            if flat_idx < len(decoded):
                decoded[flat_idx] = float(weight)

    return decoded


def verify_artifact(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    source_tensors = {name: tensor for name, tensor in _load_tensors(source)}
    verified = 0
    weighted_error = 0.0
    weighted_values = 0
    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0
    max_mse_delta = 0.0
    worst_tensors = []

    tensors_list = manifest.get("tensors", [])
    for i, tensor_meta in enumerate(tensors_list):
        name = tensor_meta["name"]
        print(f"  Validating tensor {name} ({i+1}/{len(tensors_list)})...", flush=True)
        if name not in source_tensors:
            raise KeyError(f"source tensor missing during verification: {name}")
        original = _flatten_float_values(
            source_tensors[name], int(tensor_meta["packed_values"])
        )
        decoded = _decode_tensor(out_dir, tensor_meta)
        if len(original) != len(decoded):
            raise ValueError(f"decoded value count mismatch for {name}")

        metrics = quality_metrics_from_flat(original, decoded)
        mse = metrics["mse"]
        mse_delta = abs(mse - float(tensor_meta.get("mse", 0.0)))
        max_mse_delta = max(max_mse_delta, mse_delta)
        weighted_error += mse * len(original)
        weighted_values += len(original)
        sse += metrics["sse"]
        abs_error_sum += metrics["mae"] * metrics["value_count"]
        max_abs_error = max(max_abs_error, metrics["max_abs_error"])
        source_l2_sq += metrics["source_l2_sq"]
        reconstructed_l2_sq += metrics["reconstructed_l2_sq"]
        dot += metrics["dot"]
        verified += 1
        worst_tensors.append(
            {
                "name": name,
                "mse": mse,
                "relative_rmse": metrics["relative_rmse"],
                "cosine_similarity": metrics["cosine_similarity"],
                "manifest_mse": tensor_meta.get("mse", 0.0),
                "mse_delta": mse_delta,
            }
        )

    worst_tensors.sort(key=lambda item: item["mse"], reverse=True)
    aggregate_metrics = _quality_from_totals(
        value_count=weighted_values,
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )
    return {
        "artifact": str(out_dir),
        "source": str(source),
        "verified_tensors": verified,
        "weighted_mse": weighted_error / weighted_values if weighted_values else 0.0,
        "rmse": aggregate_metrics["rmse"],
        "mae": aggregate_metrics["mae"],
        "max_abs_error": aggregate_metrics["max_abs_error"],
        "relative_rmse": aggregate_metrics["relative_rmse"],
        "cosine_similarity": aggregate_metrics["cosine_similarity"],
        "max_mse_delta": max_mse_delta,
        "worst_tensors": worst_tensors[:10],
    }


def _decoded_tensor_map(out_dir: Path, manifest: dict) -> dict:
    tensors = {}
    for tensor_meta in manifest.get("tensors", []):
        decoded = _decode_tensor(out_dir, tensor_meta)
        shape = [int(x) for x in tensor_meta.get("shape", [])]
        tensors[tensor_meta["name"]] = {
            "shape": shape,
            "flat": decoded,
            "values": _reshape_flat(decoded, shape),
        }
    return tensors


def _complete_decoded_tensor_map(out_dir: Path, manifest: dict) -> dict:
    tensors = {}
    packed_names = {t["name"] for t in manifest.get("tensors", [])}

    # Load passthrough tensors from artifact (self-contained, no source needed).
    passthrough_path = out_dir / "passthrough.safetensors"
    if passthrough_path.exists():
        for name, tensor in _load_tensors(passthrough_path):
            shape = _tensor_shape(tensor)
            flat = _flatten_float_values(tensor)
            tensors[name] = {"shape": shape, "flat": flat, "values": _reshape_flat(flat, shape)}

    # Fall back to source for anything still missing (backward compat, sensitivity-map skips).
    source = Path(manifest["source"])
    if source.exists():
        for name, tensor in _load_tensors(source):
            if name in packed_names or name in tensors:
                continue
            shape = _tensor_shape(tensor)
            flat = _flatten_float_values(tensor)
            tensors[name] = {"shape": shape, "flat": flat, "values": _reshape_flat(flat, shape)}

    tensors.update(_decoded_tensor_map(out_dir, manifest))
    return tensors

def _write_json_reconstruction(
    out_dir: Path, output_path: Path, manifest: dict, tensors: dict
) -> None:
    output = {
        "format": "orka-reconstruction",
        "version": ORKA_VERSION,
        "source_artifact": str(out_dir),
        "source_checkpoint": manifest.get("source"),
        "tensor_count": len(tensors),
        "tensors": {
            name: {
                "shape": tensor["shape"],
                "values": tensor["values"],
            }
            for name, tensor in tensors.items()
        },
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")


def _write_safetensors_reconstruction(output_path: Path, tensors: dict) -> None:
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors reconstruction requires numpy and safetensors"
        ) from exc

    arrays = {}
    for name, tensor in tensors.items():
        arrays[name] = np.asarray(tensor["flat"], dtype=np.float32).reshape(
            tensor["shape"]
        )
    save_file(arrays, str(output_path))

def _decode_tensor_torch(out_dir: Path, tm: dict, device: str):
    """Decode a single quantized tensor on GPU, return torch tensor in original shape."""
    import torch
    import numpy as np

    group_size = int(tm["group_size"])
    padded_values = int(tm["padded_values"])
    packed_values = int(tm["packed_values"])
    index_count = math.ceil(padded_values / group_size)
    shape = [int(x) for x in tm["shape"]]

    stages = tm.get("stages") or [{
        "codebook": tm["codebook"],
        "codebook_size": int(tm["codebook_size"]),
        "index_bits": int(tm["index_bits"]),
        "indices": tm["indices"],
    }]

    decoded = torch.zeros(index_count * group_size, dtype=torch.float32, device=device)
    for stage in stages:
        cb_np = np.fromfile(str(out_dir / stage["codebook"]), dtype="<f4").reshape(-1, group_size)
        idxs_np = np.frombuffer(
            (out_dir / stage["indices"]).read_bytes(),
            dtype=_index_bit_spec(int(stage["index_bits"]))[1],
        ).astype(np.int64)
        cb = torch.from_numpy(cb_np).to(device)
        idxs = torch.from_numpy(idxs_np).to(device)
        decoded.add_(cb[idxs].reshape(-1))
    decoded = decoded[:packed_values]

    outl = tm.get("outliers")
    if outl:
        positions, values = _read_outliers(out_dir / outl["positions"], out_dir / outl["values"])
        if positions:
            pos_t = torch.tensor(list(positions), dtype=torch.long, device=device)
            val_t = torch.tensor(list(values), dtype=torch.float32, device=device)
            decoded[pos_t] = val_t

    rotation = tm.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tm.get("rotation_seed") or 0)
        rows = shape[0]
        cols = 1
        for s in shape[1:]:
            cols *= int(s)
        arr = decoded[:rows * cols].reshape(rows, cols)
        if rotation == "hadamard":
            unrotated = torch.from_numpy(_fwht_numpy(arr.cpu().numpy())).to(device)
        else:
            q = torch.from_numpy(_generate_orthogonal_numpy(cols, seed)).to(device)
            unrotated = arr @ q.T
        decoded = unrotated.reshape(-1)

    norm = tm.get("normalization", "none")
    if norm in ("block-max", "awq-block-max", "slrq-block"):
        scales = np.fromfile(
            str(out_dir / tm["scales"]), dtype="<f4", count=int(tm["scale_count"])
        )
        block_size = int(tm.get("block_scale_size") or 32)
        scales_t = torch.from_numpy(scales).to(device)
        n = decoded.numel()
        pad = (-n) % block_size
        if pad:
            decoded = torch.cat([decoded, torch.zeros(pad, dtype=torch.float32, device=device)])
        decoded = (decoded.reshape(-1, block_size) * scales_t[:decoded.numel() // block_size, None]).reshape(-1)
        if pad:
            decoded = decoded[:n]
        if norm == "awq-block-max":
            awq_meta = tm.get("awq_col_scales")
            if awq_meta:
                awq_scales = np.fromfile(
                    str(out_dir / awq_meta["path"]), dtype="<f4", count=int(awq_meta["count"])
                )
                awq_t = torch.from_numpy(awq_scales).to(device)
                cols = shape[-1]
                rows = decoded.numel() // cols
                decoded = (decoded[:rows * cols].reshape(rows, cols) * awq_t[None, :]).reshape(-1)
    elif norm == "row-l2":
        scales = np.fromfile(
            str(out_dir / tm["scales"]), dtype="<f4", count=int(tm["scale_count"])
        )
        scales_t = torch.from_numpy(scales).to(device)
        cols = decoded.numel() // scales_t.numel()
        decoded = (decoded.reshape(-1, cols) * scales_t[:, None]).reshape(-1)
    elif norm in ("col-l2", "awq"):
        scales = np.fromfile(
            str(out_dir / tm["scales"]), dtype="<f4", count=int(tm["scale_count"])
        )
        scales_t = torch.from_numpy(scales).to(device)
        cols = scales_t.numel()
        rows = decoded.numel() // cols
        decoded = (decoded[:rows * cols].reshape(rows, cols) * scales_t[None, :]).reshape(-1)

    salient = tm.get("salient")
    if salient:
        s_idx_np = np.fromfile(str(out_dir / salient["indices"]), dtype="<u4")
        s_val_np = np.fromfile(str(out_dir / salient["weights"]), dtype="<f4")
        
        s_idx = torch.from_numpy(s_idx_np.astype(np.int64)).to(device)
        s_val = torch.from_numpy(s_val_np).to(device)
        
        # SLRQ: re-inject salient weights AFTER scaling to avoid double-scaling.
        block_size = int(tm.get("block_scale_size", 16))
        b_count = len(s_idx)
        b_indices = torch.arange(b_count, device=device)
        flat_indices = b_indices * block_size + s_idx
        
        # Guard against padding
        mask = flat_indices < decoded.numel()
        decoded[flat_indices[mask]] = s_val[mask]

    return decoded.reshape(shape)


def _write_complete_safetensors_reconstruction(
    out_dir: Path, output_path: Path, manifest: dict, device: str | None = None
) -> dict:
    """Reconstruct full model. Uses GPU streaming path when device='cuda' to avoid Python list bloat."""
    if device is not None and "cuda" in str(device).lower():
        try:
            import torch
            if torch.cuda.is_available():
                from safetensors.torch import save_file as save_torch
                from safetensors import safe_open
                arrays: dict = {}
                packed_names = {t["name"] for t in manifest.get("tensors", [])}
                # Passthrough first
                pp = out_dir / "passthrough.safetensors"
                if pp.exists():
                    with safe_open(str(pp), framework="pt") as f:
                        for name in f.keys():
                            arrays[name] = f.get_tensor(name).contiguous()
                # Source fallback for anything missing
                source = Path(manifest["source"])
                if source.exists():
                    with safe_open(str(source), framework="pt") as f:
                        for name in f.keys():
                            if name in packed_names or name in arrays:
                                continue
                            arrays[name] = f.get_tensor(name).contiguous()
                # GPU decode quantized tensors, move to CPU immediately to free GPU memory
                for tm in manifest.get("tensors", []):
                    dec_gpu = _decode_tensor_torch(out_dir, tm, device)
                    arrays[tm["name"]] = dec_gpu.cpu().contiguous()
                    del dec_gpu
                    torch.cuda.empty_cache()
                save_torch(arrays, str(output_path))
                return {"out": str(output_path), "tensor_count": len(arrays), "format": "safetensors"}
        except Exception as exc:
            print(f"GPU reconstruction failed ({exc}); falling back to numpy path", flush=True)
    # CPU/numpy fallback (the slow path)
    tensors = _complete_decoded_tensor_map(out_dir, manifest)
    _write_safetensors_reconstruction(output_path, tensors)
    return {
        "out": str(output_path),
        "tensor_count": len(tensors),
        "format": "safetensors",
    }

def reconstruct_artifact(
    out_dir: Path, output_path: Path, output_format: str = "json"
) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    if output_format not in {"json", "safetensors"}:
        raise ValueError("output_format must be 'json' or 'safetensors'")

    manifest = json.loads(manifest_path.read_text())
    tensors = _decoded_tensor_map(out_dir, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        _write_json_reconstruction(out_dir, output_path, manifest, tensors)
    else:
        _write_safetensors_reconstruction(output_path, tensors)

    return {
        "out": str(output_path),
        "tensor_count": len(tensors),
        "format": output_format,
    }
