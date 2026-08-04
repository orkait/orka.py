"""Check that a packed artifact's sidecars match its manifest.

Reuses the decoder's own readers, so a tensor that verifies here is one the decoder can read.
Reads indices and codebooks but skips the dequantization math, so it is I/O bound rather than
compute bound.
"""
from __future__ import annotations

import json
from pathlib import Path

from orka.core._format import _read_float_vector
from orka.pipeline.decode import _read_stage_arrays, _resolve_stages


def verify_artifact(artifact_dir: Path, *, stop_after: int | None = None) -> dict:
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "tensors": 0, "checked": 0,
                "problems": [{"name": "-", "part": "manifest.json", "error": "missing"}]}

    manifest = json.loads(manifest_path.read_text())
    tensors = manifest.get("tensors", [])
    problems: list[dict] = []
    checked = 0

    for tm in tensors:
        name = tm.get("name", "?")
        group_size = int(tm.get("group_size", 8))
        padded = int(tm.get("padded_values") or tm.get("packed_values") or 0)
        for stage in _resolve_stages(tm, group_size):
            try:
                _read_stage_arrays(artifact_dir, stage, group_size, padded)
            except Exception as exc:
                problems.append({"name": name, "part": f"stage{stage.get('stage', '?')}",
                                 "error": f"{type(exc).__name__}: {exc}"})
        if tm.get("scales"):
            try:
                _read_float_vector(artifact_dir / tm["scales"], int(tm["scale_count"]),
                                   tm.get("scale_dtype") or "float32")
            except Exception as exc:
                problems.append({"name": name, "part": "scales",
                                 "error": f"{type(exc).__name__}: {exc}"})
        checked += 1
        if stop_after is not None and len(problems) >= stop_after:
            break

    passthrough = artifact_dir / "passthrough.safetensors"
    if passthrough.exists() and passthrough.stat().st_size == 0:
        problems.append({"name": "-", "part": "passthrough.safetensors", "error": "empty"})

    return {"ok": not problems, "tensors": len(tensors), "checked": checked,
            "problems": problems}


def format_problems(result: dict, limit: int = 12) -> str:
    if result["ok"]:
        return f"artifact OK: {result['checked']}/{result['tensors']} tensors verified"
    lines = [f"artifact BAD: {len(result['problems'])} problems "
             f"over {result['checked']}/{result['tensors']} tensors"]
    for p in result["problems"][:limit]:
        lines.append(f"  {p['part']:10s} {p['name'][:56]}: {p['error'][:90]}")
    if len(result["problems"]) > limit:
        lines.append(f"  ... +{len(result['problems']) - limit} more")
    return "\n".join(lines)
