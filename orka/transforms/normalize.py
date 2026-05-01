"""Normalization variants: block-max, awq, awq-block-max, slrq-block.

Each mode has numpy + torch implementations side by side. Dispatcher
``_apply_normalization`` picks the right one based on backend/availability.
"""

from __future__ import annotations

from typing import Sequence

from orka._tensor import _numpy_float32_array, _torch_f32
from orka._util import _product


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

def _normalize_tensor_awq_torch(tensor, X, alpha, device):
    import torch

    resolved, arr = _torch_f32(tensor, device)
    X_t = X.to(device=resolved, dtype=torch.float32)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_t.shape[1] != rows.shape[1]:
        raise RuntimeError(f"awq calibration shape {tuple(X_t.shape)} mismatches tensor cols {rows.shape[1]}")
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
        raise RuntimeError(f"awq calibration shape {tuple(X_arr.shape)} mismatches tensor cols {rows.shape[1]}")
    act_mag = np.mean(np.abs(X_arr), axis=0)
    act_mag = np.maximum(act_mag, 1e-6)
    s = act_mag**alpha
    scales = 1.0 / s
    normalized = (rows / scales[None, :]).reshape(arr.shape)
    return normalized, scales.astype(np.float32), arr.reshape(-1)


def _apply_normalization(
    tensor, name, normalization, awq_activations, awq_alpha,
    block_scale_size, backend, device, awq_fallbacks,
):
    is_torch = backend == "torch"
    has_awq = awq_activations is not None and name in awq_activations
    awq_col_scales = None

    def _block_max():
        return (_normalize_tensor_block_max_torch(tensor, block_scale_size, device) if is_torch
                else _normalize_tensor_block_max_numpy(tensor, block_scale_size))

    def _slrq_block():
        # returns tensor, scales, salient_weights, salient_indices, source_flat
        return (_normalize_tensor_slrq_block_torch(tensor, block_scale_size, device) if is_torch
                else _normalize_tensor_slrq_block_numpy(tensor, block_scale_size))

    salient_weights = None
    salient_indices = None

    if normalization == "slrq-block":
        tensor, row_scales, salient_weights, salient_indices, source_flat = _slrq_block()
    elif normalization == "awq":
        if not has_awq:
            raise RuntimeError(f"awq normalization requires --awq-calibration activations for tensor {name}")
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
            raise RuntimeError(f"awq-block-max requires --awq-calibration activations for tensor {name}")
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

