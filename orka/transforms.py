"""Pre-VQ pipeline transforms: normalization, rotation, outlier extraction.

Pipeline order during pack: normalize -> rotate -> outlier-extract -> vectorize.
Decode reverses: outlier-inject -> un-rotate -> un-normalize.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

from orka.core import _is_numpy_array, _is_torch_tensor, _product, _torch_f32
from orka.io_format import _numpy_float32_array


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

def _normalize_tensor_awq_block_max_torch(tensor, X, alpha, block_size, device):
    import torch

    resolved, arr = _torch_f32(tensor, device)
    X_t = X.to(device=resolved, dtype=torch.float32)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_t.shape[1] != rows.shape[1]:
        norm_tensor, block_scales, src_flat = _normalize_tensor_block_max_torch(
            tensor, block_size, device
        )
        return norm_tensor, block_scales, src_flat, None

    act_mag = X_t.abs().mean(dim=0).clamp(min=1e-6)
    s = act_mag**alpha
    awq_scales = 1.0 / s
    scaled_rows = rows / awq_scales[None, :]

    flat = scaled_rows.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]

    return (
        normalized.reshape(arr.shape),
        scales.detach().cpu(),
        arr.reshape(-1).detach().cpu(),
        awq_scales.detach().cpu(),
    )


def _normalize_tensor_block_max_torch(tensor, block_size: int, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
    return (
        normalized.reshape(arr.shape),
        scales.detach().cpu(),
        arr.reshape(-1).detach().cpu(),
    )


def _normalize_tensor_block_max_numpy(tensor, block_size: int):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    blocks = flat.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales).astype(np.float32)
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
    return normalized.reshape(arr.shape), scales, arr.reshape(-1)


def _normalize_tensor_slrq_block_torch(tensor, block_size: int, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    
    # Salient protection
    abs_blocks = blocks.abs()
    salient_indices = abs_blocks.argmax(dim=1)
    row_indices = torch.arange(blocks.shape[0], device=device)
    salient_weights = blocks[row_indices, salient_indices].clone()
    
    blocks[row_indices, salient_indices] = 0.0
    max_rem = blocks.abs().amax(dim=1)
    safe = torch.where(max_rem == 0, torch.ones_like(max_rem), max_rem)
    safe = torch.exp2(torch.ceil(torch.log2(safe)))
    
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
        
    return (
        normalized.reshape(arr.shape),
        safe.detach().cpu(),
        salient_weights.detach().cpu(),
        salient_indices.detach().cpu(),
        arr.reshape(-1).detach().cpu(),
    )


def _normalize_tensor_slrq_block_numpy(tensor, block_size: int):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    blocks = flat.reshape(-1, block_size)
    
    # Salient protection
    abs_blocks = np.abs(blocks)
    salient_indices = np.argmax(abs_blocks, axis=1)
    row_indices = np.arange(blocks.shape[0])
    salient_weights = blocks[row_indices, salient_indices].copy()
    
    blocks[row_indices, salient_indices] = 0.0
    max_rem = np.abs(blocks).max(axis=1)
    safe = np.where(max_rem == 0, 1.0, max_rem).astype(np.float32)
    safe = np.exp2(np.ceil(np.log2(safe))).astype(np.float32)
    
    normalized = (blocks / safe[:, np.newaxis]).reshape(-1)
    if pad:
        normalized = normalized[:n]
        
    return (
        normalized.reshape(arr.shape),
        safe,
        salient_weights,
        salient_indices,
        arr.reshape(-1),
    )


def _apply_block_max_scales(flat, scales, block_size: int):
    out = []
    n = len(flat)
    block_idx = 0
    for i in range(0, n, block_size):
        end = min(i + block_size, n)
        scale = float(scales[block_idx]) if block_idx < len(scales) else 1.0
        out.extend(float(flat[j]) * scale for j in range(i, end))
        block_idx += 1
    return out


def _apply_block_max_scales_numpy(flat, scales, block_size: int):
    import numpy as np

    arr = np.asarray(flat, dtype=np.float32)
    n = arr.shape[0]
    pad = (-n) % block_size
    if pad:
        arr = np.pad(arr, (0, pad), mode="constant")
    blocks = arr.reshape(-1, block_size)
    scale_arr = np.asarray(scales, dtype=np.float32)
    if scale_arr.shape[0] != blocks.shape[0]:
        raise ValueError(
            f"block scale count {scale_arr.shape[0]} != block count {blocks.shape[0]}"
        )
    out = (blocks * scale_arr[:, None]).reshape(-1)
    if pad:
        out = out[:n]
    return out


def _normalize_tensor_row_l2_numpy(tensor):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    scales = np.linalg.norm(rows, axis=1).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales).astype(np.float32)
    normalized = (rows / safe[:, None]).reshape(arr.shape)
    return normalized, scales, arr.reshape(-1)


def _normalize_tensor_col_l2_numpy(tensor):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    scales = np.linalg.norm(rows, axis=0).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales).astype(np.float32)
    normalized = (rows / safe[None, :]).reshape(arr.shape)
    return normalized, scales, arr.reshape(-1)


def _normalize_tensor_row_l2_torch(tensor, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    rows = arr.reshape(arr.shape[0], -1)
    scales = torch.linalg.vector_norm(rows, ord=2, dim=1).to(dtype=torch.float32)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (rows / safe[:, None]).reshape(arr.shape)
    return normalized, scales.detach().cpu(), arr.reshape(-1).detach().cpu()


def _normalize_tensor_awq_torch(tensor, X, alpha, device):
    import torch

    resolved, arr = _torch_f32(tensor, device)
    X_t = X.to(device=resolved, dtype=torch.float32)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_t.shape[1] != rows.shape[1]:
        return _normalize_tensor_col_l2_torch(tensor, device)
    act_mag = X_t.abs().mean(dim=0).clamp(min=1e-6)
    s = act_mag**alpha
    scales = 1.0 / s
    normalized = (rows / scales[None, :]).reshape(arr.shape)
    return normalized, scales.detach().cpu(), arr.reshape(-1).detach().cpu()


def _normalize_tensor_awq_numpy(tensor, X, alpha):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    X_arr = _numpy_float32_array(X)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_arr.shape[1] != rows.shape[1]:
        return _normalize_tensor_col_l2_numpy(tensor)
    act_mag = np.mean(np.abs(X_arr), axis=0)
    act_mag = np.maximum(act_mag, 1e-6)
    s = act_mag**alpha
    scales = 1.0 / s
    normalized = (rows / scales[None, :]).reshape(arr.shape)
    return normalized, scales.astype(np.float32), arr.reshape(-1)


def _normalize_tensor_col_l2_torch(tensor, device):
    import torch

    _, arr = _torch_f32(tensor, device)
    rows = arr.reshape(arr.shape[0], -1)
    scales = torch.linalg.vector_norm(rows, ord=2, dim=0).to(dtype=torch.float32)
    safe = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (rows / safe[None, :]).reshape(arr.shape)
    return normalized, scales.detach().cpu(), arr.reshape(-1).detach().cpu()


def _apply_normalization(
    tensor, name, normalization, awq_activations, awq_alpha,
    block_scale_size, backend, device, awq_fallbacks,
):
    is_torch = backend == "torch"
    has_awq = awq_activations is not None and name in awq_activations
    awq_col_scales = None

    def _row_l2():
        return (_normalize_tensor_row_l2_torch(tensor, device) if is_torch
                else _normalize_tensor_row_l2_numpy(tensor))

    def _col_l2():
        return (_normalize_tensor_col_l2_torch(tensor, device) if is_torch
                else _normalize_tensor_col_l2_numpy(tensor))

    def _block_max():
        return (_normalize_tensor_block_max_torch(tensor, block_scale_size, device) if is_torch
                else _normalize_tensor_block_max_numpy(tensor, block_scale_size))

    def _slrq_block():
        # returns tensor, scales, salient_weights, salient_indices, source_flat
        return (_normalize_tensor_slrq_block_torch(tensor, block_scale_size, device) if is_torch
                else _normalize_tensor_slrq_block_numpy(tensor, block_scale_size))

    salient_weights = None
    salient_indices = None

    if normalization == "row-l2":
        tensor, row_scales, source_flat = _row_l2()
    elif normalization == "col-l2":
        tensor, row_scales, source_flat = _col_l2()
    elif normalization == "slrq-block":
        tensor, row_scales, salient_weights, salient_indices, source_flat = _slrq_block()
    elif normalization == "awq":
        if not has_awq:
            awq_fallbacks.append(name)
            tensor, row_scales, source_flat = _col_l2()
        elif is_torch:
            tensor, row_scales, source_flat = _normalize_tensor_awq_torch(
                tensor, awq_activations[name], awq_alpha, device)
        else:
            tensor, row_scales, source_flat = _normalize_tensor_awq_numpy(
                tensor, awq_activations[name], awq_alpha)
    elif normalization == "awq-block-max":
        if not is_torch:
            raise RuntimeError("awq-block-max requires --backend torch")
        if not has_awq:
            awq_fallbacks.append(name)
            tensor, row_scales, source_flat = _block_max()
        else:
            tensor, row_scales, source_flat, awq_col_scales = (
                _normalize_tensor_awq_block_max_torch(
                    tensor, awq_activations[name], awq_alpha, block_scale_size, device))
    else:
        tensor, row_scales, source_flat = _block_max()
    return tensor, row_scales, source_flat, awq_col_scales, salient_weights, salient_indices


def _apply_col_l2_scales(flat, shape, scales):
    try:
        import numpy as np
        return _apply_col_l2_scales_numpy(flat, shape, scales).tolist()
    except ImportError:
        pass

    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    if len(scales) != cols:
        raise ValueError("col scale count does not match tensor cols")
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(float(flat[r * cols + c]) * float(scales[c]))
    return out


def _apply_col_l2_scales_numpy(flat, shape, scales):
    import numpy as np

    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    arr = np.asarray(flat, dtype=np.float32)[: rows * cols].reshape(rows, cols)
    col_scales = np.asarray(scales, dtype=np.float32)
    if col_scales.shape[0] != cols:
        raise ValueError("col scale count does not match tensor cols")
    return (arr * col_scales[None, :]).reshape(-1)

def _extract_outliers(vectors, outlier_frac: float, packed_values: int):
    if outlier_frac is None or outlier_frac <= 0:
        return None, None, vectors
    n = int(packed_values)
    k = int(outlier_frac * n)
    if k <= 0:
        return None, None, vectors
    if _is_torch_tensor(vectors):
        import torch

        flat = vectors.reshape(-1).clone()
        relevant = flat[:n]
        _, topk_idx = torch.topk(relevant.abs(), k)
        positions = topk_idx.detach().cpu().to(torch.int64).numpy()
        values = flat[topk_idx].detach().cpu().to(torch.float32).numpy()
        flat[topk_idx] = 0
        return positions, values, flat.reshape(vectors.shape)
    if _is_numpy_array(vectors):
        import numpy as np

        flat = vectors.reshape(-1).copy()
        relevant = flat[:n]
        order = np.argpartition(np.abs(relevant), -k)[-k:]
        positions = order.astype(np.int64)
        values = flat[positions].astype(np.float32)
        flat[positions] = 0
        return positions, values, flat.reshape(vectors.shape)
    flat = []
    for row in vectors:
        flat.extend(float(v) for v in row)
    pairs = [(abs(flat[i]), i) for i in range(n)]
    pairs.sort(reverse=True)
    positions = [p for _, p in pairs[:k]]
    values = [flat[p] for p in positions]
    for p in positions:
        flat[p] = 0.0
    g = len(vectors[0])
    new_vectors = [flat[i * g : (i + 1) * g] for i in range(len(vectors))]
    return positions, values, new_vectors


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

def _apply_row_l2_scales(
    flat: Sequence[float], shape: Sequence[int], scales: Sequence[float]
) -> list[float]:
    try:
        import numpy as np
        return _apply_row_l2_scales_numpy(flat, shape, scales).tolist()
    except ImportError:
        pass

    row_count = int(shape[0])
    row_width = _product([int(x) for x in shape[1:]])
    if len(scales) != row_count:
        raise ValueError("row scale count does not match tensor rows")
    out = []
    for row_i in range(row_count):
        scale = float(scales[row_i])
        start = row_i * row_width
        out.extend(float(v) * scale for v in flat[start : start + row_width])
    return out


def _apply_row_l2_scales_numpy(flat, shape: Sequence[int], scales):
    import numpy as np

    row_count = int(shape[0])
    row_width = _product([int(x) for x in shape[1:]])
    values = np.asarray(flat, dtype=np.float32)[: row_count * row_width].reshape(
        row_count, row_width
    )
    row_scales = np.asarray(scales, dtype=np.float32)
    if row_scales.shape[0] != row_count:
        raise ValueError("row scale count does not match tensor rows")
    return (values * row_scales[:, None]).reshape(-1)
