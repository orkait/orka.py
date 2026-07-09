"""Reconstruction quality metrics: SSE, MSE, RMSE, MAE, cosine similarity, relative RMSE.

Includes pack-side stage metrics (_stage_quality_metrics) and post-denormalization
metric variants (_denorm_metrics_from_flat).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from orka.core._tensor import _is_numpy_array, _is_torch_tensor
from orka.transforms.normalize import (
    _apply_block_max_scales,
    _apply_block_max_scales_numpy,
    _apply_col_l2_scales,
    _apply_col_l2_scales_numpy,
)
from orka.transforms.rotate import _unrotate_flat


def _quality_from_totals(
    value_count: int,
    sse: float,
    abs_error_sum: float,
    max_abs_error: float,
    source_l2_sq: float,
    reconstructed_l2_sq: float,
    dot: float,
) -> dict:
    mse = sse / value_count if value_count else 0.0
    rmse = math.sqrt(mse)
    if source_l2_sq > 0:
        relative_rmse = math.sqrt(sse / source_l2_sq)
    else:
        relative_rmse = 0.0 if sse == 0 else float("inf")

    denom = math.sqrt(source_l2_sq) * math.sqrt(reconstructed_l2_sq)
    if denom > 0:
        cosine = dot / denom
    else:
        cosine = 1.0 if source_l2_sq == 0 and reconstructed_l2_sq == 0 else 0.0

    if sse > 0 and source_l2_sq > 0:
        sqnr = 10.0 * math.log10(source_l2_sq / sse)
    elif sse == 0 and source_l2_sq > 0:
        sqnr = float("inf")
    else:
        sqnr = 0.0

    return {
        "value_count": value_count,
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "mae": abs_error_sum / value_count if value_count else 0.0,
        "max_abs_error": max_abs_error,
        "source_l2_sq": source_l2_sq,
        "reconstructed_l2_sq": reconstructed_l2_sq,
        "dot": dot,
        "relative_rmse": relative_rmse,
        "cosine_similarity": cosine,
        "sqnr": sqnr,
    }


def quality_metrics_from_flat(
    source: Sequence[float], reconstructed: Sequence[float]
) -> dict:
    if len(source) != len(reconstructed):
        raise ValueError("source and reconstructed values must have the same length")
    return _quality_metrics_for_numpy_flat(source, reconstructed)


def _quality_metrics_for_numpy_flat(
    source, reconstructed, chunk_size: int = 1_000_000
) -> dict:
    import numpy as np

    src = np.asarray(source, dtype=np.float32).reshape(-1)
    rec = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    if src.shape[0] != rec.shape[0]:
        print(f"DEBUG: Size Mismatch! Source: {src.shape[0]}, Recon: {rec.shape[0]}", flush=True)
        raise ValueError("source and reconstructed arrays must have the same size")

    sse = 0.0
    abs_error_sum = 0.0
    max_abs_error = 0.0
    source_l2_sq = 0.0
    reconstructed_l2_sq = 0.0
    dot = 0.0

    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        s = src[start:end]
        r = rec[start:end]
        diff = s - r
        abs_diff = np.abs(diff)
        sse += float(np.sum(diff * diff))
        abs_error_sum += float(np.sum(abs_diff))
        max_abs_error = max(
            max_abs_error, float(np.max(abs_diff)) if abs_diff.size else 0.0
        )
        source_l2_sq += float(np.sum(s * s))
        reconstructed_l2_sq += float(np.sum(r * r))
        dot += float(np.sum(s * r))

    return _quality_from_totals(
        value_count=int(src.shape[0]),
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )

def _stage_quality_metrics(candidate: dict, backend: str) -> dict:
    decoded_sum = candidate["decoded_sum"]
    orig = candidate["vectors_orig"]
    if _is_torch_tensor(decoded_sum):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch metrics requires torch") from exc
        if _is_torch_tensor(orig) and orig.device != decoded_sum.device:
            decoded_sum = decoded_sum.to(orig.device)
        diff = orig - decoded_sum
        sse = float((diff * diff).sum().detach().cpu().item())
        abs_diff = diff.abs()
        abs_error_sum = float(abs_diff.sum().detach().cpu().item())
        max_abs_error = (
            float(abs_diff.max().detach().cpu().item()) if abs_diff.numel() else 0.0
        )
        source_l2_sq = float((orig * orig).sum().detach().cpu().item())
        reconstructed_l2_sq = float(
            (decoded_sum * decoded_sum).sum().detach().cpu().item()
        )
        dot = float((orig * decoded_sum).sum().detach().cpu().item())
        value_count = int(orig.numel())
    elif _is_numpy_array(decoded_sum):
        import numpy as np

        diff = orig - decoded_sum
        abs_diff = np.abs(diff)
        sse = float(np.sum(diff * diff))
        abs_error_sum = float(np.sum(abs_diff))
        max_abs_error = float(np.max(abs_diff)) if abs_diff.size else 0.0
        source_l2_sq = float(np.sum(orig * orig))
        reconstructed_l2_sq = float(np.sum(decoded_sum * decoded_sum))
        dot = float(np.sum(orig * decoded_sum))
        value_count = int(orig.size)
    else:
        flat_src = []
        flat_rec = []
        for ro, rd in zip(orig, decoded_sum):
            flat_src.extend(float(v) for v in ro)
            flat_rec.extend(float(v) for v in rd)
        return _denorm_metrics_from_flat(candidate, flat_src, flat_rec)

    metrics = _quality_from_totals(
        value_count=value_count,
        sse=sse,
        abs_error_sum=abs_error_sum,
        max_abs_error=max_abs_error,
        source_l2_sq=source_l2_sq,
        reconstructed_l2_sq=reconstructed_l2_sq,
        dot=dot,
    )
    norm = candidate.get("normalization", "none")
    rotation = candidate.get("rotation", "none")
    rot_seed = candidate.get("rotation_seed")
    has_rotation = rotation in {"orthogonal", "hadamard"}
    outlier_positions = candidate.get("outlier_positions")
    outlier_values = candidate.get("outlier_values")
    has_outliers = outlier_positions is not None and len(outlier_positions) > 0
    if norm == "none" and not has_rotation and not has_outliers:
        return metrics

    import numpy as np
    if _is_torch_tensor(decoded_sum):
        flat_decoded = (
            decoded_sum.reshape(-1).detach().cpu().numpy()[: candidate["packed_values"]]
        ).copy()
    elif _is_numpy_array(decoded_sum):
        flat_decoded = decoded_sum.reshape(-1)[: candidate["packed_values"]].copy()
    else:
        flat_decoded = np.asarray(
            list(
                decoded_sum.reshape(-1)
                if hasattr(decoded_sum, "reshape")
                else [v for row in decoded_sum for v in row]
            )[: candidate["packed_values"]],
            dtype=np.float32,
        )
    # Patch outliers (stored in normalized+rotated space, written by _decode_tensor before un-rotate).
    if has_outliers:
        positions = np.asarray(list(outlier_positions), dtype=np.int64)
        values = np.asarray(list(outlier_values), dtype=np.float32)
        flat_decoded[positions] = values

    # Patch Concept Pillars (also stored in normalized space)
    pillar_positions = candidate.get("pillar_positions")
    pillar_values = candidate.get("pillar_values")
    if pillar_positions is not None:
        p_pos = np.asarray(list(pillar_positions), dtype=np.int64)
        p_val = np.asarray(list(pillar_values), dtype=np.float32)
        flat_decoded[p_pos] = p_val

    if has_rotation:
        flat_decoded = np.asarray(
            _unrotate_flat(
                flat_decoded,
                candidate["shape"],
                rotation,
                int(rot_seed or 0),
            ),
            dtype=np.float32,
        )
    if norm == "none":
        return _quality_metrics_for_numpy_flat(candidate["source_flat"], flat_decoded)
    return _denorm_metrics_from_flat(candidate, candidate["source_flat"], flat_decoded)


def _denorm_metrics_from_flat(candidate: dict, source_flat, decoded_flat) -> dict:
    norm = candidate.get("normalization", "none")
    if norm == "none":
        return (
            _quality_metrics_for_numpy_flat(source_flat, decoded_flat)
            if _is_numpy_array(decoded_flat)
            else quality_metrics_from_flat(source_flat, decoded_flat)
        )
    if norm == "awq":
        if _is_numpy_array(decoded_flat):
            denorm = _apply_col_l2_scales_numpy(
                decoded_flat, candidate["shape"], candidate["row_scales"]
            )
            return _quality_metrics_for_numpy_flat(candidate["source_flat"], denorm)
        denorm = _apply_col_l2_scales(
            decoded_flat, candidate["shape"], candidate["row_scales"]
        )
        return quality_metrics_from_flat(candidate["source_flat"], denorm)
    # block-scales-only inverse: awq-block-max is handled in its own branch below (block +
    # col scales). This is the narrower grouping, intentionally not stores_block_scales().
    if norm in ("block-max", "channel-block-max", "slrq-block"):
        block_size = candidate.get("block_scale_size") or 32
        if _is_numpy_array(decoded_flat):
            import numpy as np

            denorm = _apply_block_max_scales_numpy(
                decoded_flat, candidate["row_scales"], block_size
            )
            if norm == "slrq-block" and candidate.get("salient_indices") is not None:
                salient_idx = np.asarray(candidate["salient_indices"], dtype=np.int64)
                salient_val = np.asarray(candidate["salient_weights"], dtype=np.float32)
                for b_idx, (local_idx, weight) in enumerate(zip(salient_idx, salient_val)):
                    flat_idx = b_idx * block_size + int(local_idx)
                    if flat_idx < denorm.shape[0]:
                        denorm[flat_idx] = weight
            return _quality_metrics_for_numpy_flat(candidate["source_flat"], denorm)
        denorm = _apply_block_max_scales(
            decoded_flat, candidate["row_scales"], block_size
        )
        if norm == "slrq-block" and candidate.get("salient_indices") is not None:
            for b_idx, (local_idx, weight) in enumerate(
                zip(candidate["salient_indices"], candidate["salient_weights"])
            ):
                flat_idx = b_idx * block_size + int(local_idx)
                if flat_idx < len(denorm):
                    denorm[flat_idx] = float(weight)
        return quality_metrics_from_flat(candidate["source_flat"], denorm)
    if norm == "awq-block-max":
        block_size = candidate.get("block_scale_size") or 32
        awq_scales = candidate.get("awq_col_scales")
        block_scales = candidate["row_scales"]
        if _is_numpy_array(decoded_flat):
            stage = _apply_block_max_scales_numpy(
                decoded_flat, block_scales, block_size
            )
            if awq_scales is not None:
                stage = _apply_col_l2_scales_numpy(
                    stage, candidate["shape"], awq_scales
                )
            return _quality_metrics_for_numpy_flat(candidate["source_flat"], stage)
        stage = _apply_block_max_scales(decoded_flat, block_scales, block_size)
        if awq_scales is not None:
            stage = _apply_col_l2_scales(stage, candidate["shape"], awq_scales)
        return quality_metrics_from_flat(candidate["source_flat"], stage)
    raise ValueError(f"unknown normalization: {norm}")
