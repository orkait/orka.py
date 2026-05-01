"""Tensor checkpoint loading and on-disk I/O for indices, codebooks, scales."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable, Sequence

from orka.core import _is_torch_tensor


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

def _load_tensors(path: Path) -> Iterable[tuple[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open() as f:
            loaded = json.load(f)
        tensors = loaded.get("tensors", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(tensors, dict):
            raise ValueError("JSON input must be an object or contain a tensors object")
        for name, tensor in tensors.items():
            yield name, tensor
        return

    if suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError(
                "safetensors input requires the safetensors package"
            ) from exc
        try:
            import torch  # noqa: F401
        except Exception:
            framework = "np"
        else:
            framework = "pt"
        with safe_open(str(path), framework=framework) as handle:
            for name in handle.keys():
                try:
                    yield name, handle.get_tensor(name)
                except TypeError as exc:
                    if framework == "np":
                        raise RuntimeError(
                            f"safetensors tensor {name} uses a dtype that requires torch loading"
                        ) from exc
                    raise
        return

    if suffix in {".pt", ".pth", ".bin"}:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch checkpoint input requires torch") from exc
        loaded = torch.load(path, map_location="cpu")
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state, dict):
            raise ValueError(
                "checkpoint must load to a tensor dictionary or contain state_dict"
            )
        for name, tensor in state.items():
            yield name, tensor
        return

    raise ValueError(f"unsupported input format: {path.suffix}")

def _infer_shape(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        first = _infer_shape(value[0])
        for item in value[1:]:
            if _infer_shape(item) != first:
                raise ValueError(
                    "nested JSON tensor values must have a rectangular shape"
                )
        return [len(value)] + first
    return []


def _flatten_nested(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_nested(item))
        return out
    return [float(value)]


def _flatten_float_values(tensor: object, limit: int | None = None) -> list[float]:
    if hasattr(tensor, "detach"):
        flat = tensor.detach().float().cpu().reshape(-1)
        if limit is not None:
            flat = flat[:limit]
        return [float(x) for x in flat.tolist()]

    if hasattr(tensor, "reshape") and hasattr(tensor, "tolist"):
        flat = tensor.reshape(-1)
        if hasattr(flat, "astype"):
            flat = flat.astype("float32")
        values = [float(x) for x in flat.tolist()]
        return values[:limit] if limit is not None else values

    if isinstance(tensor, (list, tuple)):
        values = _flatten_nested(tensor)
        return values[:limit] if limit is not None else values

    raise TypeError("expected a tensor-like object")


def _numpy_float32_array(tensor: object):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy backend requires numpy") from exc

    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(tensor, dtype=np.float32)

def _tensor_shape(tensor: object) -> list[int]:
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        return [int(x) for x in shape]
    return _infer_shape(tensor) if isinstance(tensor, (list, tuple)) else []


def _tensor_numel(tensor: object) -> int:
    if hasattr(tensor, "numel"):
        return int(tensor.numel())
    size = getattr(tensor, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(tensor, (list, tuple)):
        shape = _infer_shape(tensor)
        total = 1
        for dim in shape:
            total *= dim
        return total
    return 0

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
