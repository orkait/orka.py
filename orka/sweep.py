"""Pack/report matrix sweeps over (group_size, codebook, mode, normalization)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

from orka.core import (
    ORKA_VERSION,
    _human_bytes,
    _index_bits_for_size,
    _require_non_empty,
    _resolve_torch_device,
    _safe_tensor_name,
)
from orka.decode import report_artifact, verify_artifact
from orka.pack import pack_checkpoint
from orka.quant_spec import (
    is_rvq_mixed_spec,
    parse_quant_spec,
    quant_spec_from_sizes,
    rvq_mixed_family_stages,
)


def _sweep_artifact_root(out_path: Path) -> Path:
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}.artifacts")
    return out_path.parent / f"{out_path.name}.artifacts"


def _sweep_artifact_name(
    group_size: int,
    stages: Sequence[int],
    codebook_mode: str,
    normalization: str,
    label: str | None = None,
) -> str:
    if label:
        stage_part = label
    elif len(stages) == 1:
        stage_part = f"k{stages[0]}"
    else:
        stage_part = "rvq" + "+".join(f"k{k}" for k in stages)
    return (
        f"g{group_size}-{stage_part}-"
        f"{_safe_tensor_name(codebook_mode)}-{_safe_tensor_name(normalization)}.orka"
    )


def _reset_sweep_run_dir(path: Path, artifact_root: Path) -> None:
    if not path.exists():
        return
    root = artifact_root.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"refusing to remove sweep artifact outside root: {path}")
    if not path.name.endswith(".orka"):
        raise ValueError(f"refusing to remove non-Orka sweep artifact: {path}")
    shutil.rmtree(path)


def _cosine_per_mb(report: dict) -> float:
    artifact_mb = float(report["artifact_bytes"]) / 1_000_000.0
    if artifact_mb <= 0:
        return 0.0
    return float(report["cosine_similarity"]) / artifact_mb


def _best_run(runs: Sequence[dict], key: str, reverse: bool) -> dict | None:
    if not runs:
        return None
    return dict(sorted(runs, key=lambda run: float(run[key]), reverse=reverse)[0])


def _sweep_run_summary(
    artifact_dir: Path,
    group_size: int,
    codebook_size: int,
    codebook_mode: str,
    normalization: str,
    report: dict,
) -> dict:
    return {
        "artifact": str(artifact_dir),
        "group_size": group_size,
        "codebook_size": codebook_size,
        "codebook_mode": codebook_mode,
        "normalization": normalization,
        "tensor_count": report["tensor_count"],
        "artifact_bytes": report["artifact_bytes"],
        "artifact_size": _human_bytes(report["artifact_bytes"]),
        "original_fp16_bytes": report["original_fp16_bytes"],
        "compression_ratio_fp16_to_artifact": report[
            "compression_ratio_fp16_to_artifact"
        ],
        "total_index_bytes": report["total_index_bytes"],
        "total_codebook_bytes": report["total_codebook_bytes"],
        "total_scale_bytes": report["total_scale_bytes"],
        "weighted_mse": report["weighted_mse"],
        "rmse": report["rmse"],
        "mae": report["mae"],
        "max_abs_error": report["max_abs_error"],
        "relative_rmse": report["relative_rmse"],
        "cosine_similarity": report["cosine_similarity"],
        "cosine_per_mb": _cosine_per_mb(report),
    }


def sweep_checkpoint(
    source: Path,
    out_path: Path,
    group_sizes: Sequence[int],
    codebook_sizes: Sequence[int],
    codebook_modes: Sequence[str],
    normalizations: Sequence[str],
    iterations: int,
    max_values_per_tensor: int | None = None,
    sample_vectors: int | None = None,
    backend: str = "auto",
    device: str = "cpu",
    verify_runs: bool = False,
    quant_modes: Sequence[str] = (),
    outlier_frac: float = 0.0,
    rotation: str = "none",
    rotation_seed: int | None = None,
    awq_activations: dict | None = None,
    awq_alpha: float = 0.5,
    awq_alphas: Sequence[float] | None = None,
    max_tensors: int | None = None,
    progress_file: Path | None = None,
    sensitivity_map: dict | None = None,
) -> dict:
    _require_non_empty("group_sizes", group_sizes)
    _require_non_empty("codebook_modes", codebook_modes)
    _require_non_empty("normalizations", normalizations)
    if not quant_modes and not codebook_sizes:
        raise ValueError("at least one of quant_modes or codebook_sizes is required")
    alpha_values = [float(a) for a in awq_alphas] if awq_alphas else [float(awq_alpha)]

    stage_specs: list[
        tuple[list[int] | None, str, int, dict[str, list[int]] | None]
    ] = []
    for k in codebook_sizes or []:
        label = quant_spec_from_sizes([int(k)])
        stage_specs.append(([int(k)], label, int(k), None))
    for mode in quant_modes:
        if is_rvq_mixed_spec(mode):
            family_map = rvq_mixed_family_stages()
            stage_specs.append((None, mode, family_map["other"][0], family_map))
        else:
            stages = parse_quant_spec(mode)
            stage_specs.append((stages, mode, stages[0], None))

    artifact_root = _sweep_artifact_root(out_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    runs = []

    for group_size in group_sizes:
        for stages, label, primary_k, family_map in stage_specs:
            for codebook_mode in codebook_modes:
                if family_map is not None and codebook_mode != "per-tensor":
                    continue
                for normalization in normalizations:
                    norm_uses_awq = (
                        normalization in {"awq", "awq-block-max"}
                        and awq_activations is not None
                    )
                    alphas_for_norm = (
                        alpha_values if norm_uses_awq else [float(awq_alpha)]
                    )
                    for cur_alpha in alphas_for_norm:
                        alpha_label = (
                            f"a{cur_alpha:.2f}".replace(".", "_")
                            if norm_uses_awq and len(alphas_for_norm) > 1
                            else None
                        )
                        base_name = _sweep_artifact_name(
                            int(group_size),
                            stages or [primary_k],
                            codebook_mode,
                            normalization,
                            label=label,
                        )
                        artifact_name = (
                            base_name
                            if alpha_label is None
                            else f"{base_name[: -len('.orka')]}-{alpha_label}.orka"
                        )
                        artifact_dir = artifact_root / artifact_name
                        _reset_sweep_run_dir(artifact_dir, artifact_root)
                        print(f"Sweep Run: Packing {artifact_name}...", flush=True)
                        pack_checkpoint(
                            source=source,
                            out_dir=artifact_dir,
                            group_size=int(group_size),
                            codebook_size=primary_k,
                            codebook_sizes=stages,
                            iterations=iterations,
                            max_values_per_tensor=max_values_per_tensor,
                            codebook_mode=codebook_mode,
                            sample_vectors=sample_vectors,
                            backend=backend,
                            device=device,
                            normalization=normalization,
                            family_stages_map=family_map,
                            outlier_frac=outlier_frac,
                            rotation=rotation,
                            rotation_seed=rotation_seed,
                            awq_activations=awq_activations,
                            awq_alpha=cur_alpha,
                            max_tensors=max_tensors,
                        )
                        report = report_artifact(artifact_dir)
                        run = _sweep_run_summary(
                            artifact_dir=artifact_dir,
                            group_size=int(group_size),
                            codebook_size=primary_k,
                            codebook_mode=codebook_mode,
                            normalization=normalization,
                            report=report,
                        )
                        run["quant_mode"] = label or "custom"
                        run["awq_alpha"] = cur_alpha if norm_uses_awq else None
                        if family_map is not None:
                            run["stages"] = None
                            run["family_stages_map"] = {
                                fam: [_index_bits_for_size(k) for k in s]
                                for fam, s in family_map.items()
                            }
                            run["bits_per_vector"] = None
                            run["bits_per_weight"] = None
                        else:
                            run["stages"] = list(stages)
                            run["bits_per_vector"] = sum(
                                _index_bits_for_size(k) for k in stages
                            )
                            run["bits_per_weight"] = run["bits_per_vector"] / int(
                                group_size
                            )
                        if verify_runs:
                            run["verify"] = verify_artifact(artifact_dir)
                        runs.append(run)

    summary = {
        "format": "orka-sweep",
        "version": ORKA_VERSION,
        "source": str(source),
        "out": str(out_path),
        "artifact_root": str(artifact_root),
        "backend": backend,
        "device": str(_resolve_torch_device(device)) if backend == "torch" else "cpu",
        "sample_vectors": sample_vectors,
        "iterations": iterations,
        "matrix": {
            "group_sizes": [int(value) for value in group_sizes],
            "codebook_sizes": [int(value) for value in codebook_sizes],
            "codebook_modes": list(codebook_modes),
            "normalizations": list(normalizations),
        },
        "run_count": len(runs),
        "best_by_cosine_similarity": _best_run(runs, "cosine_similarity", reverse=True),
        "best_by_relative_rmse": _best_run(runs, "relative_rmse", reverse=False),
        "best_by_cosine_per_mb": _best_run(runs, "cosine_per_mb", reverse=True),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    return summary
