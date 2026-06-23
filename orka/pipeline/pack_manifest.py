"""Per-tensor sidecar persistence + manifest assembly for the pack pipeline.

Writes the scale / awq-col / outlier / pillar / salient sidecars, builds each tensor's
manifest entry (shape, stages, quality metrics, sidecar refs), and assembles the final
manifest.json. Split out of pack.py so the on-disk-layout concern is in one place and so
pack_pipeline can finalize entries without importing pack (avoids a circular import).
"""

from __future__ import annotations

import json
from pathlib import Path

from orka._format import (
    _write_float_vector,
    _write_outliers,
    _write_pillars,
    _write_salient,
)
from orka._util import _index_bits_for_size, _report_progress, _safe_tensor_name
from orka.metrics import _stage_quality_metrics
from orka.transforms.normalize import stores_block_scales


def _persist_tensor_sidecars(c: dict, tensor_dir: Path, out_dir: Path) -> tuple:
    """Write per-tensor scale / awq_col / outlier / salient sidecars.

    Returns (scale_path, scale_bytes, scale_count, awq_col_meta, outlier_meta, salient_meta, pillar_meta).
    """
    safe = _safe_tensor_name(c["name"])
    scale_path = None
    scale_bytes = 0
    scale_count = 0
    norm = c["normalization"]
    scale_dtype = None
    if norm == "awq":
        scale_path = tensor_dir / f"{safe}.col_l2_scale.f32"
        scale_dtype = _write_float_vector(scale_path, c["row_scales"], dtype="float16")
        scale_bytes = scale_path.stat().st_size
        scale_count = len(c["row_scales"])
    elif stores_block_scales(norm):
        scale_path = tensor_dir / f"{safe}.block_max_scale.f32"
        scale_dtype = _write_float_vector(scale_path, c["row_scales"], dtype="float16")
        scale_bytes = scale_path.stat().st_size
        scale_count = len(c["row_scales"])

    awq_col_meta = None
    if norm == "awq-block-max" and c.get("awq_col_scales") is not None:
        awq_col_path = tensor_dir / f"{safe}.awq_col_scale.f32"
        awq_col_dtype = _write_float_vector(
            awq_col_path, c["awq_col_scales"], dtype="float16"
        )
        awq_col_meta = {
            "path": str(awq_col_path.relative_to(out_dir)),
            "count": len(c["awq_col_scales"]),
            "bytes": awq_col_path.stat().st_size,
            "dtype": awq_col_dtype,
        }
    c["scale_dtype"] = scale_dtype

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
        salient_index_bits = _index_bits_for_size(int(c.get("block_scale_size") or 32))
        weights_dtype = _write_salient(
            s_idx_path, s_val_path, c["salient_indices"], c["salient_weights"], salient_index_bits
        )
        salient_meta = {
            "count": int(len(c["salient_weights"])),
            "indices": str(s_idx_path.relative_to(out_dir)),
            "weights": str(s_val_path.relative_to(out_dir)),
            "indices_bits": salient_index_bits,
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
        "scale_dtype": c.get("scale_dtype"),
        "block_scale_size": (
            block_scale_size if stores_block_scales(c["normalization"]) else None
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
        "col_importance",
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
    # EM-AQ rewrites index streams asynchronously; compressed sizes can differ
    # from the greedy pass, so refresh byte counts from disk.
    for stage_meta in c.get("stages_meta", []):
        stage_path = out_dir / stage_meta["indices"]
        if stage_path.exists():
            stage_meta["index_bytes"] = stage_path.stat().st_size
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

    manifest["total_index_bytes"] = sum(
        int(t.get("index_bytes", 0)) for t in manifest["tensors"]
    )
    manifest["tensor_count"] = len(manifest["tensors"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
