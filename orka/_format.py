"""Orka artifact format I/O. Single source of truth for on-disk layout.

Manifest version + sidecar I/O for: indices, codebooks, scales, outliers,
salient (SLRQ), and FP16 passthrough tensors.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

from orka._tensor import _is_torch_tensor


# v2: fp16 codebooks/scales (with f32 overflow fallback) + optional zlib
# index streams. v1 artifacts read fine: missing manifest fields default to
# float32 / raw.
ORKA_VERSION = 2


_FLOAT_VALUE_DTYPES = {
    "float16": "<f2",
    "float32": "<f4",
}

_UNSIGNED_VALUE_DTYPES = {
    "uint8": "<u1",
    "uint16": "<u2",
    "uint32": "<u4",
    "uint64": "<u8",
}


def _smallest_unsigned_dtype(max_value: int) -> str:
    if max_value <= 255:
        return "uint8"
    if max_value <= 65535:
        return "uint16"
    if max_value <= 4294967295:
        return "uint32"
    return "uint64"


def _float_value_dtype(value_dtype: str) -> str:
    try:
        return _FLOAT_VALUE_DTYPES[value_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported float value dtype: {value_dtype}") from exc


def _unsigned_value_dtype(value_dtype: str) -> str:
    try:
        return _UNSIGNED_VALUE_DTYPES[value_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported unsigned value dtype: {value_dtype}") from exc


def _compact_float_dtype(values, requested: str) -> str:
    if requested != "float16":
        return requested
    import numpy as np

    if values.size == 0:
        return "float16"
    max_abs = float(np.max(np.abs(values)))
    if not np.isfinite(max_abs) or max_abs > 65504.0:
        return "float32"
    return "float16"


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


def _pack_indices(indices, bits: int):
    """Pack integer indices into a contiguous big-endian bitstream at exact bit width.

    Each index occupies exactly ``bits`` bits (MSB first); no padding to byte/int
    boundaries except the final byte. Lossless inverse of ``_unpack_indices``.
    """
    import numpy as np

    arr = np.asarray(indices, dtype=np.uint64).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    bitmat = ((arr[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bitmat.reshape(-1))


def _unpack_indices(packed, bits: int, count: int):
    """Inverse of ``_pack_indices``: recover ``count`` indices of ``bits`` width."""
    import numpy as np

    if count == 0:
        return np.zeros(0, dtype=np.int64)
    allbits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: count * bits]
    bitmat = allbits.reshape(count, bits).astype(np.uint64)
    weights = np.uint64(1) << np.arange(bits - 1, -1, -1, dtype=np.uint64)
    return (bitmat * weights).sum(axis=1).astype(np.int64)


def _write_indices(
    path: Path,
    indices: Sequence[int],
    index_bits: int,
    encoding: str | None = None,
) -> tuple[bool, str]:
    """Write indices to disk. Bit-packs when ``index_bits`` is not byte-aligned;
    then entropy-codes the stream with zlib when that actually shrinks it.

    ``encoding=None`` auto-picks ("zlib" if smaller else "raw"); passing
    "zlib"/"raw" forces the choice (EM-AQ rewrites must keep the encoding the
    stage metadata already recorded). Returns (bit_packed, encoding).
    """
    import zlib

    import numpy as np

    if _is_torch_tensor(indices):
        indices = indices.detach().cpu().numpy()
    if index_bits % 8 != 0:
        raw = _pack_indices(indices, index_bits).tobytes()
        bit_packed = True
    else:
        _, np_dtype, _ = _index_bit_spec(index_bits)
        raw = np.asarray(indices, dtype=np_dtype).tobytes()
        bit_packed = False

    if encoding is None:
        compressed = zlib.compress(raw, 6)
        encoding = "zlib" if len(compressed) < len(raw) else "raw"
        payload = compressed if encoding == "zlib" else raw
    elif encoding == "zlib":
        payload = zlib.compress(raw, 6)
    elif encoding == "raw":
        payload = raw
    else:
        raise ValueError(f"unknown index encoding: {encoding}")
    path.write_bytes(payload)
    return bit_packed, encoding


def _fp16_storage_roundtrip(values):
    """Round values to the fp16 grid IN MEMORY when they fit (else unchanged).

    Sidecars (scales, outliers, salient, pillars) are stored fp16 with an f32
    overflow fallback. Rounding at capture time makes the in-memory values the
    pipeline computes with byte-identical to what decode reads back, so
    manifest metrics match verify exactly. Mirrors ``_compact_float_dtype``.
    """
    if values is None:
        return None
    if _is_torch_tensor(values):
        import torch

        if values.numel() == 0:
            return values
        max_abs = float(values.detach().abs().max().item())
        if not math.isfinite(max_abs) or max_abs > 65504.0:
            return values
        return values.detach().to(torch.float16).to(values.dtype)
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    if _compact_float_dtype(arr, "float16") == "float32":
        return arr
    return arr.astype(np.float16).astype(np.float32)


def _cast_codebook_storage(codebook, dtype: str = "float16"):
    """Round a learned codebook to its storage dtype IN MEMORY and return
    (cast_codebook_f32, actual_dtype).

    Assignment, metrics, and the on-disk file must all see the exact same
    centroid values, so the fp16 rounding happens before quantization, not at
    write time. Falls back to float32 when values overflow fp16.
    """
    import numpy as np

    if dtype == "float32":
        return codebook, "float32"
    if _is_torch_tensor(codebook):
        import torch

        max_abs = float(codebook.detach().abs().max().item()) if codebook.numel() else 0.0
        if not math.isfinite(max_abs) or max_abs > 65504.0:
            return codebook, "float32"
        return codebook.detach().to(torch.float16).to(torch.float32), "float16"
    arr = np.asarray(codebook, dtype=np.float32)
    actual = _compact_float_dtype(arr, dtype)
    if actual == "float32":
        return arr, "float32"
    return arr.astype(np.float16).astype(np.float32), "float16"


def _write_codebook(path: Path, codebook: Sequence[Sequence[float]], dtype: str = "float32") -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_torch_tensor(codebook):
        arr = codebook.detach().cpu().to(dtype=__import__("torch").float32).numpy()
    else:
        arr = np.asarray(codebook, dtype=np.float32)
    np.ascontiguousarray(arr.astype(_float_value_dtype(dtype))).tofile(str(path))


def _read_codebook(path: Path, group_size: int, dtype: str = "float32"):
    import numpy as np

    arr = np.fromfile(str(path), dtype=_float_value_dtype(dtype))
    return arr.astype(np.float32).reshape(-1, group_size)


def _write_float_vector(path: Path, values, dtype: str = "float16") -> str:
    """Write a float sidecar vector; fp16 by default with f32 overflow
    fallback. Returns the actual dtype written."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_torch_tensor(values):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.float32)
    actual = _compact_float_dtype(arr, dtype)
    path.write_bytes(arr.astype(_float_value_dtype(actual)).tobytes())
    return actual


def _write_f32_vector(path: Path, values) -> None:
    _write_float_vector(path, values, dtype="float32")


def _read_float_vector(path: Path, expected_count: int, dtype: str = "float32"):
    import numpy as np
    arr = np.fromfile(str(path), dtype=_float_value_dtype(dtype)).astype(np.float32)
    if arr.shape[0] != expected_count:
        raise ValueError(
            f"float vector size mismatch for {path}: expected {expected_count}, got {arr.shape[0]}"
        )
    return arr


def _read_f32_vector(path: Path, expected_count: int):
    return _read_float_vector(path, expected_count, dtype="float32")


def _read_indices(
    path: Path,
    index_bits: int,
    expected_count: int,
    packed: bool = False,
    encoding: str = "raw",
):
    import numpy as np

    data = path.read_bytes()
    if encoding == "zlib":
        import zlib

        data = zlib.decompress(data)
    elif encoding != "raw":
        raise ValueError(f"unknown index encoding: {encoding}")
    if packed:
        raw = np.frombuffer(data, dtype=np.uint8)
        return _unpack_indices(raw, index_bits, expected_count)
    _, np_dtype, _ = _index_bit_spec(index_bits)
    arr = np.frombuffer(data, dtype=np_dtype)
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

def _write_outliers(
    idx_path: Path,
    val_path: Path,
    positions,
    values,
    value_dtype: str = "float16",
) -> tuple[str, str]:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier writing requires numpy") from exc
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    pos_arr = np.asarray(positions, dtype=np.uint64)
    val_arr = np.asarray(values, dtype=np.float32)
    position_dtype = _smallest_unsigned_dtype(int(pos_arr.max()) if pos_arr.size else 0)
    value_dtype = _compact_float_dtype(val_arr, value_dtype)
    pos_arr.astype(_unsigned_value_dtype(position_dtype)).tofile(str(idx_path))
    val_arr.astype(_float_value_dtype(value_dtype)).tofile(str(val_path))
    return position_dtype, value_dtype


def _read_outliers(
    idx_path: Path,
    val_path: Path,
    position_dtype: str = "uint32",
    value_dtype: str = "float32",
):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("outlier reading requires numpy") from exc
    positions = np.fromfile(str(idx_path), dtype=_unsigned_value_dtype(position_dtype))
    values = np.fromfile(str(val_path), dtype=_float_value_dtype(value_dtype)).astype(np.float32)
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




def _write_salient(
    idx_path: Path,
    val_path: Path,
    salient_indices,
    salient_weights,
    weight_dtype: str = "float16",
) -> tuple[str, str]:
    """Write SLRQ salient sidecars."""
    import numpy as np

    sw = salient_weights.numpy() if hasattr(salient_weights, "numpy") else salient_weights
    si = salient_indices.numpy() if hasattr(salient_indices, "numpy") else salient_indices
    sw = np.asarray(sw, dtype=np.float32)
    si = np.asarray(si, dtype=np.uint64)
    index_dtype = _smallest_unsigned_dtype(int(si.max()) if si.size else 0)
    weight_dtype = _compact_float_dtype(sw, weight_dtype)
    sw.astype(_float_value_dtype(weight_dtype)).tofile(str(val_path))
    si.astype(_unsigned_value_dtype(index_dtype)).tofile(str(idx_path))
    return index_dtype, weight_dtype


def _read_salient(
    idx_path: Path,
    val_path: Path,
    index_dtype: str = "uint32",
    weight_dtype: str = "float32",
):
    """Read SLRQ salient sidecars."""
    import numpy as np
    s_idx = np.fromfile(str(idx_path), dtype=_unsigned_value_dtype(index_dtype))
    s_val = np.fromfile(str(val_path), dtype=_float_value_dtype(weight_dtype)).astype(np.float32)
    if len(s_idx) != len(s_val):
        raise ValueError(f"salient count mismatch: {len(s_idx)} != {len(s_val)}")
    return s_idx, s_val
