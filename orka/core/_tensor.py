"""Backend primitives. ALL numpy/torch detection + dispatch lives here.

Other modules call these instead of branching on backend themselves.
"""

from __future__ import annotations

import math
from typing import Sequence

from orka._runtime import _resolve_torch_device


def _is_numpy_array(value: object) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "reshape")
        and hasattr(value, "astype")
        and hasattr(value, "tolist")
    )


def _is_torch_tensor(value: object) -> bool:
    return hasattr(value, "detach") and hasattr(value, "to")


def _torch_f32(tensor, device):
    import torch

    resolved = _resolve_torch_device(device)
    if _is_torch_tensor(tensor):
        return resolved, tensor.detach().to(device=resolved, dtype=torch.float32)
    return resolved, torch.as_tensor(tensor, dtype=torch.float32, device=resolved)

def _numpy_float32_array(tensor: object):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy backend requires numpy") from exc

    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(tensor, dtype=np.float32)

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

def _torch_float32_matrix(values, device: str):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    resolved = _resolve_torch_device(device)
    if _is_torch_tensor(values):
        rows = values.detach().to(device=resolved, dtype=torch.float32)
    else:
        rows = torch.as_tensor(values, dtype=torch.float32, device=resolved)
    if rows.ndim != 2:
        raise ValueError("torch VQ expects a 2D vector matrix")
    return rows.contiguous()


def _sample_vector_rows(vectors, sample_vectors: int | None):
    if sample_vectors is None or sample_vectors <= 0 or sample_vectors >= len(vectors):
        return vectors
    if _is_torch_tensor(vectors):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch vector sampling requires torch") from exc
        positions = (
            torch.linspace(
                0,
                len(vectors) - 1,
                steps=sample_vectors,
                device=vectors.device,
                dtype=torch.float64,
            )
            .round()
            .to(dtype=torch.long)
            .clamp_(max=len(vectors) - 1)
        )
        return vectors.index_select(0, positions)
    if hasattr(vectors, "shape") and hasattr(vectors, "__getitem__"):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy vector sampling requires numpy") from exc
        positions = np.linspace(0, len(vectors) - 1, sample_vectors, dtype=np.int64)
        return vectors[positions]

    raise TypeError(f"unsupported tensor type for vector sampling: {type(vectors)}")


def _concat_vector_parts(parts: Sequence[object]):
    if not parts:
        raise ValueError("cannot concatenate empty vector group")
    if _is_torch_tensor(parts[0]):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch vector concatenation requires torch") from exc
        return torch.cat(list(parts), dim=0)
    if _is_numpy_array(parts[0]):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy vector concatenation requires numpy") from exc
        return np.concatenate(parts, axis=0)

    raise TypeError(f"unsupported tensor type for concatenation: {type(parts[0])}")

def _decode_to_vectors_format(
    vectors_template, codebook, indices, backend: str, device: str
):
    if _is_torch_tensor(vectors_template):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch decode requires torch") from exc
        cb = _torch_float32_matrix(codebook, str(vectors_template.device))
        if _is_torch_tensor(indices):
            idx = indices.detach().to(device=cb.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, dtype=torch.long, device=cb.device)
        return cb[idx]
    if _is_numpy_array(vectors_template):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy decode requires numpy") from exc
        cb = np.asarray(codebook, dtype=np.float32)
        idx = np.asarray(indices, dtype=np.int64)
        return cb[idx]
    raise TypeError(f"unsupported vectors_template type: {type(vectors_template)}")


def _vectors_subtract(a, b):
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch subtract requires torch") from exc
        ta = a if _is_torch_tensor(a) else torch.as_tensor(a, dtype=torch.float32)
        tb = (
            b
            if _is_torch_tensor(b)
            else torch.as_tensor(b, dtype=torch.float32, device=ta.device)
        )
        if tb.device != ta.device:
            tb = tb.to(ta.device)
        return ta - tb
    if _is_numpy_array(a) or _is_numpy_array(b):
        import numpy as np

        return np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    raise TypeError(f"unsupported types for vector subtract: {type(a)}, {type(b)}")

