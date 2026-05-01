"""Rotation transforms: orthogonal QR (per-tensor seeded) + Hadamard FWHT."""

from __future__ import annotations

import math
from typing import Sequence

from orka._tensor import _numpy_float32_array, _torch_f32


def _fwht_torch(x):
    import torch

    x = x.contiguous().clone()
    n = int(x.shape[-1])
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-2 last dim, got {n}")
    leading_shape = list(x.shape[:-1])
    h = 1
    while h < n:
        view = x.view(*leading_shape, n // (2 * h), 2, h)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        x = view.reshape(*leading_shape, n)
        h *= 2
    return x * (1.0 / math.sqrt(n))


def _fwht_numpy(x):
    import numpy as np

    x = np.array(x, dtype=np.float32, copy=True)
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-2 last dim, got {n}")
    leading_shape = list(x.shape[:-1])
    h = 1
    while h < n:
        view = x.reshape(*leading_shape, n // (2 * h), 2, h)
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        x = view.reshape(*leading_shape, n)
        h *= 2
    return x / math.sqrt(n)

def _tensor_rotation_seed(global_seed: int, name: str) -> int:
    import hashlib

    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(h, "little") ^ int(global_seed)) & ((1 << 63) - 1)


def _generate_orthogonal_torch(n: int, seed: int, device, dtype):
    import torch

    q_np = _generate_orthogonal_numpy(n, seed)
    return torch.from_numpy(q_np).to(device=device, dtype=dtype)


def _generate_orthogonal_numpy(n: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
    a = rng.standard_normal((n, n)).astype(np.float32)
    q, _ = np.linalg.qr(a)
    return q.astype(np.float32)

def _rotate_tensor_to_2d(
    tensor, name: str, rotation: str, rotation_seed: int, backend: str, device: str
):
    if rotation == "none":
        return tensor, None
    if rotation not in {"orthogonal", "hadamard"}:
        raise ValueError(f"unknown rotation mode: {rotation}")

    if rotation == "hadamard":
        if backend == "torch":
            _, t = _torch_f32(tensor, device)
            shape = list(t.shape)
            rows, cols = shape[0], 1
            for s in shape[1:]:
                cols *= int(s)
            if cols & (cols - 1) != 0:
                raise ValueError(
                    f"hadamard rotation requires power-of-2 last-dim product, tensor {name} has {cols}"
                )
            return _fwht_torch(t.reshape(rows, cols)).reshape(shape), 0

        arr = _numpy_float32_array(tensor)
        shape = [int(x) for x in arr.shape]
        rows, cols = shape[0], 1
        for s in shape[1:]:
            cols *= int(s)
        if cols & (cols - 1) != 0:
            raise ValueError(
                f"hadamard rotation requires power-of-2 last-dim product, tensor {name} has {cols}"
            )
        return _fwht_numpy(arr.reshape((rows, cols))).reshape(shape), 0

    seed = _tensor_rotation_seed(rotation_seed, name)
    if backend == "torch":
        import torch

        resolved, t = _torch_f32(tensor, device)
        shape = list(t.shape)
        rows, cols = shape[0], 1
        for s in shape[1:]:
            cols *= int(s)
        q = _generate_orthogonal_torch(cols, seed, resolved, torch.float32)
        return (t.reshape(rows, cols) @ q).reshape(shape), seed
    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows, cols = shape[0], 1
    for s in shape[1:]:
        cols *= int(s)
    q = _generate_orthogonal_numpy(cols, seed)
    return (arr.reshape(rows, cols) @ q).reshape(shape), seed


def _unrotate_flat(flat, shape, rotation: str, seed: int):
    import numpy as np

    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    arr = np.asarray(flat, dtype=np.float32)[: rows * cols].reshape(rows, cols)
    if rotation == "hadamard":
        unrotated = _fwht_numpy(arr)
    else:
        q = _generate_orthogonal_numpy(cols, seed)
        unrotated = arr @ q.T
    return unrotated.reshape(-1).tolist()
