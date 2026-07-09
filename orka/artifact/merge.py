"""Merge partitioned Orka artifacts into one complete artifact."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from orka.core._checkpoint import _load_tensors
from orka.core._format import _write_passthrough_tensors


def _files_equal(a: Path, b: Path, chunk_size: int = 1 << 20) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            a_chunk = fa.read(chunk_size)
            b_chunk = fb.read(chunk_size)
            if a_chunk != b_chunk:
                return False
            if not a_chunk:
                return True


def _copy_artifact_files(source: Path, target: Path) -> None:
    for source_path in source.rglob("*"):
        if source_path.is_dir():
            continue
        relative = source_path.relative_to(source)
        if relative.as_posix() in {"manifest.json", "passthrough.safetensors"}:
            continue
        target_path = target / relative
        if target_path.exists():
            if not _files_equal(source_path, target_path):
                raise RuntimeError(
                    f"merge conflict for {relative}. Files differ across partitions."
                )
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_path.read_bytes())


def _check_manifest_compatibility(base: dict, candidate: dict, candidate_dir: Path) -> None:
    # awq_enabled / hessian_weighted are deliberately NOT required: they
    # describe how codebooks were learned (sequential packing legitimately
    # mixes weighted and unweighted blocks), not the on-disk format. The
    # format-relevant normalization mode is checked below.
    required_fields = [
        "format",
        "version",
        "source",
        "group_size",
        "backend",
        "normalization",
        "rotation",
        "outlier_frac",
        "codebook_mode",
        "n_stages",
    ]

    for field in required_fields:
        if base.get(field) != candidate.get(field):
            raise ValueError(
                f"{field} mismatch while merging {candidate_dir}: {base.get(field)!r} != {candidate.get(field)!r}"
            )

    if base.get("codebook_mode") == "per-tensor" and candidate.get("codebook_sizes") != base.get(
        "codebook_sizes"
    ):
        raise ValueError(
            f"codebook_sizes mismatch while merging {candidate_dir}"
        )

    if base.get("family_stages_map") != candidate.get("family_stages_map"):
        raise ValueError(
            f"family_stages_map mismatch while merging {candidate_dir}"
        )


def merge_orka_artifacts(
    input_artifacts: Iterable[Path],
    out_dir: Path,
) -> dict:
    input_artifacts = [Path(path).resolve() for path in input_artifacts]
    if len(input_artifacts) < 2:
        raise ValueError("need at least two input artifacts to merge")

    if out_dir.resolve() in input_artifacts:
        raise ValueError("output directory cannot be one of the input artifacts")

    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"output directory already exists with content: {out_dir}")

    manifests: list[dict] = []
    for path in input_artifacts:
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        manifests.append(manifest)

    base = manifests[0]
    for manifest_path, manifest in zip(input_artifacts[1:], manifests[1:]):
        _check_manifest_compatibility(base, manifest, manifest_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    for artifact in input_artifacts:
        _copy_artifact_files(artifact, out_dir)

    merged_tensors: dict[str, dict] = {}
    for manifest in manifests:
        for tensor_meta in manifest.get("tensors", []):
            tensor_name = tensor_meta["name"]
            if tensor_name in merged_tensors:
                raise ValueError(f"duplicate tensor in merge: {tensor_name}")
            merged_tensors[tensor_name] = tensor_meta

    merged_tensors_list = [merged_tensors[name] for name in sorted(merged_tensors)]

    passthrough_tensors: dict[str, object] = {}
    for artifact in input_artifacts:
        passthrough_path = artifact / "passthrough.safetensors"
        if not passthrough_path.exists():
            continue
        for name, tensor in _load_tensors(passthrough_path):
            if name in merged_tensors:
                continue
            if name not in passthrough_tensors:
                passthrough_tensors[name] = tensor

    if passthrough_tensors:
        _write_passthrough_tensors(
            out_dir / "passthrough.safetensors",
            passthrough_tensors,
        )
        base["passthrough_count"] = len(passthrough_tensors)
    else:
        base["passthrough_count"] = 0

    base["tensors"] = merged_tensors_list
    base["tensor_count"] = len(merged_tensors_list)
    base["total_index_bytes"] = sum(
        int(tensor_meta.get("index_bytes", 0))
        for tensor_meta in merged_tensors_list
    )
    base["tensor_partition_count"] = None
    base["tensor_partition_index"] = None

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(base, indent=2) + "\n")
    return base
