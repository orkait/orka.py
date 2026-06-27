"""verify_artifact: decode every tensor and recompute MSE vs source."""

from __future__ import annotations

import json
from pathlib import Path

from orka.core._checkpoint import _load_tensors
from orka.core._tensor import _numpy_float32_array
from orka.eval.metrics import _quality_from_totals, quality_metrics_from_flat
from orka.pipeline.decode import _decode_tensor


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
    verified_passthrough = 0
    worst_tensors = []

    tensors_list = manifest.get("tensors", [])
    for i, tensor_meta in enumerate(tensors_list):
        name = tensor_meta["name"]
        print(f"  Validating tensor {name} ({i+1}/{len(tensors_list)})...", flush=True)
        if name not in source_tensors:
            raise KeyError(f"source tensor missing during verification: {name}")
        original = _numpy_float32_array(source_tensors[name]).reshape(-1)[
            : int(tensor_meta["packed_values"])
        ]
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

    passthrough_path = out_dir / "passthrough.safetensors"
    if passthrough_path.exists():
        source = Path(manifest["source"])
        source_tensors = source_tensors or {name: tensor for name, tensor in _load_tensors(source)}
        passthrough_tensors = {name: tensor for name, tensor in _load_tensors(passthrough_path)}
        for name, tensor in passthrough_tensors.items():
            if name not in source_tensors:
                raise KeyError(f"source passthrough tensor missing during verification: {name}")
            import numpy as np

            original = _numpy_float32_array(source_tensors[name]).reshape(-1)
            reconstructed = _numpy_float32_array(tensor).reshape(-1)
            if original.shape[0] != reconstructed.shape[0]:
                raise ValueError(f"passthrough value count mismatch for {name}")
            if not np.array_equal(original, reconstructed):
                raise ValueError(f"passthrough tensor mismatch for {name}")
            verified_passthrough += 1

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
        "sqnr": aggregate_metrics["sqnr"],
        "max_mse_delta": max_mse_delta,
        "verified_passthrough_tensors": verified_passthrough,
        "worst_tensors": worst_tensors[:10],
    }
