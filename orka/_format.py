"""Orka artifact format I/O. Single source of truth for on-disk layout.

Manifest version + sidecar I/O for: indices, codebooks, scales, outliers,
salient (SLRQ), and FP16 passthrough tensors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from orka._tensor import _is_torch_tensor


ORKA_VERSION = 1


_INDEX_BIT_SPECS = (
    (8, "<u1", "<B"),
    (16, "<u2", "<H"),
    (32, "<u4", "<I"),
    (64, "<u8", "<Q"),
)


def _index_bit_spec(index_bits: int) -> tuple[int, str, str]:
    for ceiling, np_dtype, struct_fmt in _INDEX_BIT_SPECS:
        if index_bits <= ceiling:
            return ceiling, np_dtype, struct_fmt
    raise ValueError(f"index_bits above 64 are not supported: got {index_bits}")


def _write_indices(path: Path, indices: Sequence[int], index_bits: int) -> None:
    import numpy as np
    ceiling, np_dtype, struct_fmt = _index_bit_spec(index_bits)
    if _is_torch_tensor(indices):
        indices = indices.detach().cpu().numpy()
    path.write_bytes(np.asarray(indices, dtype=np_dtype).tobytes())


def _write_codebook(path: Path, codebook: Sequence[Sequence[float]]) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_torch_tensor(codebook):
        arr = codebook.detach().cpu().to(dtype=__import__("torch").float32).numpy()
    else:
        arr = np.asarray(codebook, dtype=np.float32)
    np.ascontiguousarray(arr, dtype="<f4").tofile(str(path))


def _write_f32_vector(path: Path, values) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_torch_tensor(values):
        values = values.detach().cpu().numpy()
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes())


def _read_f32_vector(path: Path, expected_count: int):
    import numpy as np
    arr = np.fromfile(str(path), dtype="<f4")
    if arr.shape[0] != expected_count:
        raise ValueError(
            f"f32 vector size mismatch for {path}: expected {expected_count}, got {arr.shape[0]}"
        )
    return arr


def _read_indices(path: Path, index_bits: int, expected_count: int):
    import numpy as np
    _, np_dtype, _ = _index_bit_spec(index_bits)
    arr = np.fromfile(str(path), dtype=np_dtype)
    if arr.shape[0] != expected_count:
        raise ValueError(
            f"index count mismatch for {path}: expected {expected_count}, got {arr.shape[0]}"
        )
    return arr


def _write_passthrough_tensors(path: Path, tensors: dict) -> None:
    """Save non-quantized tensors (1D norms, sensitivity-skipped 2D) preserving original dtype."""
    has_torch_tensor = any(_is_torch_tensor(t) for t in tensors.values())
    if has_torch_tensor:
        try:
            import torch  # noqa: F401
            from safetensors.torch import save_file as save_torch
            torch_dict = {}
            for name, tensor in tensors.items():
                if _is_torch_tensor(tensor):
                    torch_dict[name] = tensor.detach().cpu().contiguous()
                else:
                    import numpy as np
                    torch_dict[name] = torch.from_numpy(np.asarray(tensor, dtype=np.float32))
            save_torch(torch_dict, str(path))
            return
        except Exception as exc:
            raise RuntimeError(f"failed to write passthrough tensors (torch): {exc}") from exc
    # Pure numpy path
    import numpy as np
    arrays = {name: np.asarray(t) for name, t in tensors.items()}
    try:
        from safetensors.numpy import save_file
        save_file(arrays, str(path))
    except Exception as exc:
        raise RuntimeError(f"failed to write passthrough tensors (numpy): {exc}") from exc

def _write_outliers(idx_path: Path, val_path: Path, positions, values) -> None:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier writing requires numpy") from exc
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    val_arr = np.asarray(values, dtype=np.float32)
    np.asarray(positions, dtype="<u4").tofile(str(idx_path))
    val_arr.astype("<f4").tofile(str(val_path))


def _read_outliers(idx_path: Path, val_path: Path):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier reading requires numpy") from exc
    positions = np.fromfile(str(idx_path), dtype="<u4")
    values = np.fromfile(str(val_path), dtype="<f4").astype(np.float32)
    if len(positions) != len(values):
        raise ValueError(f"outlier count mismatch: {len(positions)} != {len(values)}")
    return positions, values


def _write_pillars(idx_path: Path, val_path: Path, positions, values) -> None:
    """Save critical Concept Pillars in FP16 (<f2) for space efficiency."""
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("pillar writing requires numpy") from exc
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    val_arr = np.asarray(values, dtype=np.float32)
    # Check for FP16 overflow just in case
    if np.abs(val_arr).max() > 65504.0:
        val_arr = np.clip(val_arr, -65504.0, 65504.0)
    np.asarray(positions, dtype="<u4").tofile(str(idx_path))
    val_arr.astype("<f2").tofile(str(val_path))


def _read_pillars(idx_path: Path, val_path: Path):
    """Read Concept Pillars from FP16 (<f2) sidecar."""
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("pillar reading requires numpy") from exc
    positions = np.fromfile(str(idx_path), dtype="<u4")
    values = np.fromfile(str(val_path), dtype="<f2").astype(np.float32)
    return positions, values




def _write_salient(idx_path: Path, val_path: Path, salient_indices, salient_weights) -> None:
    """Write SLRQ salient sidecars: per-block index (uint32) + value (float32)."""
    sw = salient_weights.numpy() if hasattr(salient_weights, "numpy") else salient_weights
    si = salient_indices.numpy() if hasattr(salient_indices, "numpy") else salient_indices
    sw.astype("<f4").tofile(str(val_path))
    si.astype("<u4").tofile(str(idx_path))


def _read_salient(idx_path: Path, val_path: Path):
    """Read SLRQ salient sidecars. Returns (indices ndarray uint32, values ndarray float32)."""
    import numpy as np
    s_idx = np.fromfile(str(idx_path), dtype="<u4")
    s_val = np.fromfile(str(val_path), dtype="<f4")
    return s_idx, s_val
