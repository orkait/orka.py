"""report_artifact: aggregate per-tensor metrics + size breakdown from manifest."""

from __future__ import annotations

import json
import math
from pathlib import Path

from orka._util import _dir_size, _product
from orka.metrics import _quality_from_totals


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
    total_salient_bytes = sum(
        int((t.get("salient") or {}).get("indices_bytes", 0))
        + int((t.get("salient") or {}).get("weights_bytes", 0))
        for t in tensors
    )
    total_outlier_count = sum(
        int((t.get("outliers") or {}).get("count", 0)) for t in tensors
    )
    total_passthrough_bytes = 0
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
    passthrough_path = out_dir / "passthrough.safetensors"
    if passthrough_path.exists():
        total_passthrough_bytes = passthrough_path.stat().st_size
        try:
            from orka._checkpoint import _load_tensors

            for _name, tensor in _load_tensors(passthrough_path):
                shape = [int(x) for x in getattr(tensor, "shape", [])]
                if shape:
                    original_fp16_bytes += _product(shape) * 2
        except Exception:
            pass
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
        "total_salient_bytes": total_salient_bytes,
        "total_outlier_count": total_outlier_count,
        "total_passthrough_bytes": total_passthrough_bytes,
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
