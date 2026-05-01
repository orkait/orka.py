"""Orka artifact format I/O. Single source of truth for on-disk layout.

Manifest version + sidecar I/O for: indices, codebooks, scales, outliers,
salient (SLRQ), and FP16 passthrough tensors.
"""

from __future__ import annotations

import struct
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
    ceiling, np_dtype, struct_fmt = _index_bit_spec(index_bits)
    if _is_torch_tensor(indices):
        indices = indices.detach().cpu().tolist()
    if hasattr(indices, "astype") and hasattr(indices, "tobytes"):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy index writing requires numpy") from exc
        path.write_bytes(np.asarray(indices, dtype=np_dtype).tobytes())
        return
    if ceiling == 8:
        path.write_bytes(bytes(int(i) & 0xFF for i in indices))
        return
    with path.open("wb") as f:
        for index in indices:
            f.write(struct.pack(struct_fmt, int(index)))


def _write_codebook(path: Path, codebook: Sequence[Sequence[float]]) -> None:
    if _is_torch_tensor(codebook):
        codebook = codebook.detach().cpu().tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for row in codebook:
            for value in row:
                f.write(struct.pack("<f", float(value)))


def _write_f32_vector(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_torch_tensor(values):
        values = values.detach().cpu().tolist()
    if hasattr(values, "astype") and hasattr(values, "tobytes"):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy scale writing requires numpy") from exc
        path.write_bytes(np.asarray(values, dtype="<f4").tobytes())
        return

    with path.open("wb") as f:
        for value in values:
            f.write(struct.pack("<f", float(value)))


def _read_f32_vector(path: Path, expected_count: int) -> list[float]:
    data = path.read_bytes()
    expected = expected_count * 4
    if len(data) != expected:
        raise ValueError(
            f"f32 vector size mismatch for {path}: expected {expected}, got {len(data)}"
        )
    return [value[0] for value in struct.iter_unpack("<f", data)]


def _read_codebook(
    path: Path, codebook_size: int, group_size: int
) -> list[list[float]]:
    data = path.read_bytes()
    expected = codebook_size * group_size * 4
    if len(data) != expected:
        raise ValueError(
            f"codebook size mismatch for {path}: expected {expected}, got {len(data)}"
        )
    floats = [value[0] for value in struct.iter_unpack("<f", data)]
    return [floats[i : i + group_size] for i in range(0, len(floats), group_size)]


def _read_indices(path: Path, index_bits: int, expected_count: int) -> list[int]:
    ceiling, _, struct_fmt = _index_bit_spec(index_bits)
    data = path.read_bytes()
    bytes_per = ceiling // 8
    if ceiling == 8:
        indices = list(data)
    else:
        if len(data) % bytes_per:
            raise ValueError(
                f"index file size not multiple of {bytes_per} bytes: {path}"
            )
        indices = [value[0] for value in struct.iter_unpack(struct_fmt, data)]
    if len(indices) != expected_count:
        raise ValueError(
            f"index count mismatch for {path}: expected {expected_count}, got {len(indices)}"
        )
    return indices


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

_FP16_MAX = 65504.0


def _write_outliers(idx_path: Path, val_path: Path, positions, values) -> None:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier writing requires numpy") from exc
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    val_arr = np.asarray(values, dtype=np.float32)
    over = np.abs(val_arr) > _FP16_MAX
    if bool(over.any()):
        n_over = int(over.sum())
        max_abs = float(np.max(np.abs(val_arr)))
        print(
            f"WARN: {n_over} outlier value(s) exceed fp16 range ({max_abs:.1f} > {_FP16_MAX}); "
            f"will be clipped to ±inf at {val_path.name}",
            file=os.sys.stderr,
        )
    np.asarray(positions, dtype="<u4").tofile(str(idx_path))
    val_arr.astype("<f2").tofile(str(val_path))


def _read_outliers(idx_path: Path, val_path: Path) -> tuple[list[int], list[float]]:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier reading requires numpy") from exc
    positions = np.fromfile(str(idx_path), dtype="<u4")
    values = np.fromfile(str(val_path), dtype="<f2").astype(np.float32)
    if len(positions) != len(values):
        raise ValueError(f"outlier count mismatch: {len(positions)} != {len(values)}")
    return positions.tolist(), values.tolist()




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
