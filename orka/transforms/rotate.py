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


def _largest_pow2_divisor(n: int) -> int:
    if n <= 0:
        return 0
    return n & (-n)


def _block_fwht_torch(x, block_size: int):
    n = int(x.shape[-1])
    if n % block_size != 0:
        raise ValueError(f"block FWHT: dim {n} not divisible by block_size {block_size}")
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block FWHT: block_size {block_size} not power-of-2")
    leading = list(x.shape[:-1])
    n_blocks = n // block_size
    reshaped = x.reshape(*leading, n_blocks, block_size)
    transformed = _fwht_torch(reshaped)
    return transformed.reshape(*leading, n)


def _block_fwht_numpy(x, block_size: int):
    import numpy as np

    arr = np.array(x, dtype=np.float32, copy=True)
    n = arr.shape[-1]
    if n % block_size != 0:
        raise ValueError(f"block FWHT: dim {n} not divisible by block_size {block_size}")
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block FWHT: block_size {block_size} not power-of-2")
    leading = list(arr.shape[:-1])
    n_blocks = n // block_size
    reshaped = arr.reshape(*leading, n_blocks, block_size)
    transformed = _fwht_numpy(reshaped)
    return transformed.reshape(*leading, n)


def _hadamard_block_size(cols: int, min_block: int = 4) -> int:
    """Pick block size for Hadamard. Returns cols if pow2, else largest pow2 divisor.
    Raises if no usable divisor >= min_block."""
    if cols & (cols - 1) == 0:
        return cols
    div = _largest_pow2_divisor(cols)
    if div < min_block:
        raise ValueError(
            f"hadamard: cols {cols} has no power-of-2 divisor >= {min_block}"
        )
    return div

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
            block_size = _hadamard_block_size(cols)
            return _block_fwht_torch(t.reshape(rows, cols), block_size).reshape(shape), 0

        arr = _numpy_float32_array(tensor)
        shape = [int(x) for x in arr.shape]
        rows, cols = shape[0], 1
        for s in shape[1:]:
            cols *= int(s)
        block_size = _hadamard_block_size(cols)
        return _block_fwht_numpy(arr.reshape((rows, cols)), block_size).reshape(shape), 0

    seed = _tensor_rotation_seed(rotation_seed, name)
    if backend == "torch":
        import torch

        resolved, t = _torch_f32(tensor, device)
        shape = list(t.shape)
        rows, cols = shape[0], 1
        for s in shape[1:]:
            cols *= int(s)
        
        # Sanity cap: N*N matrix allocation for orthogonal rotation.
        # 16384 * 16384 * 4 bytes = 1.0 GB. 
        if cols > 16384:
            raise ValueError(
                f"tensor {name} too wide for orthogonal rotation (cols={cols} > 16384). "
                "Large tensors like attention masks should be skipped via sensitivity map or excluded from candidates."
            )
            
        q = _generate_orthogonal_torch(cols, seed, resolved, torch.float32)
        return (t.reshape(rows, cols) @ q).reshape(shape), seed
    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows, cols = shape[0], 1
    for s in shape[1:]:
        cols *= int(s)
        
    if cols > 16384:
        raise ValueError(
            f"tensor {name} too wide for orthogonal rotation (cols={cols} > 16384). "
            "Large tensors like attention masks should be skipped via sensitivity map or excluded from candidates."
        )
        
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
        block_size = _hadamard_block_size(cols)
        unrotated = _block_fwht_numpy(arr, block_size)
    else:
        q = _generate_orthogonal_numpy(cols, seed)
        unrotated = arr @ q.T
    return unrotated.reshape(-1)
