"""Low-rank correction sidecars: W ~ decode(W) + A @ B^T.

After packing (and optionally distillation), the residual W - decode(W) of
each tensor is factored with a truncated SVD. The top-r factors are stored as
fp16 sidecars and added back as the LAST decode step. Rank r costs
(rows + cols) * r * 2 bytes and buys back disproportionate quality at low
bits-per-weight - the residual's energy concentrates in few directions.
"""

from __future__ import annotations

import json
from pathlib import Path

from orka._checkpoint import _load_tensors
from orka._tensor import _numpy_float32_array
from orka._util import _safe_tensor_name
from orka.metrics import quality_metrics_from_flat
from orka.pipeline.decode import _decode_tensor


def _correct_tensor(
    out_dir: Path, tm: dict, source_tensor, *, rank: int, device: str
) -> dict:
    import numpy as np
    import torch

    shape = [int(x) for x in tm["shape"]]
    rows = shape[0]
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    packed = int(tm["packed_values"])
    if packed != rows * cols:
        return {"name": tm["name"], "skipped": "partial tensor"}
    effective_rank = min(rank, rows, cols)

    target = _numpy_float32_array(source_tensor).reshape(-1)[:packed]
    tm.pop("lowrank", None)  # measure the residual without a stale correction
    decoded = np.asarray(_decode_tensor(out_dir, tm), dtype=np.float32)
    before = quality_metrics_from_flat(target, decoded)

    residual = torch.from_numpy((target - decoded).reshape(rows, cols)).to(device)
    u, s, v = torch.svd_lowrank(residual, q=min(effective_rank + 4, rows, cols))
    a = (u[:, :effective_rank] * s[:effective_rank][None, :]).cpu().numpy()
    b = v[:, :effective_rank].cpu().numpy()

    safe = _safe_tensor_name(tm["name"])
    tensor_dir = out_dir / "tensors"
    a_path = tensor_dir / f"{safe}.lowrank.a"
    b_path = tensor_dir / f"{safe}.lowrank.b"
    # fp16 storage; round factors in memory first so metrics match disk.
    a16 = a.astype(np.float16)
    b16 = b.astype(np.float16)
    a16.tofile(str(a_path))
    b16.tofile(str(b_path))

    corrected = (
        decoded.reshape(rows, cols)
        + a16.astype(np.float32) @ b16.astype(np.float32).T
    ).reshape(-1)
    after = quality_metrics_from_flat(target, corrected)

    if after["mse"] >= before["mse"]:
        # Correction did not help (already near-exact); drop the sidecars.
        a_path.unlink(missing_ok=True)
        b_path.unlink(missing_ok=True)
        return {
            "name": tm["name"],
            "mse_before": before["mse"],
            "mse_after": before["mse"],
            "improved": False,
        }

    tm["lowrank"] = {
        "rank": effective_rank,
        "a": str(a_path.relative_to(out_dir)),
        "b": str(b_path.relative_to(out_dir)),
        "dtype": "float16",
        "a_bytes": a_path.stat().st_size,
        "b_bytes": b_path.stat().st_size,
    }
    for key in (
        "mse", "sse", "rmse", "mae", "max_abs_error", "source_l2_sq",
        "reconstructed_l2_sq", "dot", "relative_rmse", "cosine_similarity", "sqnr",
    ):
        tm[key] = after[key]
    return {
        "name": tm["name"],
        "mse_before": before["mse"],
        "mse_after": after["mse"],
        "improved": True,
        "sidecar_bytes": tm["lowrank"]["a_bytes"] + tm["lowrank"]["b_bytes"],
    }


def correct_artifact(
    artifact_dir: Path,
    *,
    rank: int = 8,
    device: str = "cpu",
    max_tensors: int | None = None,
    progress: bool = True,
) -> dict:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError("correct requires torch") from exc
    if rank < 1:
        raise ValueError("rank must be >= 1")

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Orka manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    packed_meta = {t["name"]: t for t in manifest.get("tensors", [])}

    results = []
    done = 0
    for name, tensor in _load_tensors(source):
        if name not in packed_meta:
            continue
        if max_tensors is not None and done >= max_tensors:
            break
        if progress:
            print(f"Correcting {name} ({done + 1})...", flush=True)
        results.append(
            _correct_tensor(
                artifact_dir, packed_meta[name], tensor, rank=rank, device=device
            )
        )
        done += 1

    manifest["lowrank_correction"] = {
        "rank": rank,
        "tensor_count": sum(1 for r in results if r.get("improved")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "artifact": str(artifact_dir),
        "tensor_count": len(results),
        "improved_count": sum(1 for r in results if r.get("improved")),
        "results": results,
    }
