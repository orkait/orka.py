"""Normalization variants: block-max, awq, awq-block-max, slrq-block.

Each mode has numpy + torch implementations side by side. Dispatcher
``_apply_normalization`` picks the right one based on backend/availability.
"""

from __future__ import annotations

from typing import Sequence

from orka._format import _fp16_storage_roundtrip
from orka._tensor import _numpy_float32_array, _torch_f32
from orka._util import _product


def _normalize_tensor_awq_block_max_torch(tensor, X, alpha, block_size, device):
    import torch

    resolved, arr = _torch_f32(tensor, device)
    X_t = X.to(device=resolved, dtype=torch.float32)
    shape = [int(x) for x in arr.shape]
    rows = arr.reshape(shape[0], -1)
    if X_t.shape[1] != rows.shape[1]:
        raise RuntimeError(
            f"awq-block-max calibration shape {tuple(X_t.shape)} mismatches tensor cols {rows.shape[1]}"
        )

    act_mag = X_t.abs().mean(dim=0).clamp(min=1e-6)
    s = act_mag**alpha
    awq_scales = _fp16_storage_roundtrip(1.0 / s)
    scaled_rows = rows / awq_scales[None, :]

    flat = scaled_rows.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    scales = blocks.abs().amax(dim=1)
    safe = _fp16_storage_roundtrip(
        torch.where(scales == 0, torch.ones_like(scales), scales)
    )
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
    safe = _fp16_storage_roundtrip(
        torch.where(scales == 0, torch.ones_like(scales), scales)
    )
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
    return (
        normalized.reshape(arr.shape),
        safe.detach().cpu(),
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
    safe = _fp16_storage_roundtrip(np.where(scales == 0, 1.0, scales).astype(np.float32))
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]
    return normalized.reshape(arr.shape), safe, arr.reshape(-1)


def _normalize_tensor_channel_block_max_torch(tensor, block_size: int, device):
    """Channel-aware block-max: blocks are aligned to hidden-dim columns.

    For a [rows, cols] weight matrix, we reshape to [rows, cols//block_size, block_size]
    so each scale factor governs a contiguous slice of channels within a single row.
    This prevents an outlier in one channel from inflating the scale for unrelated channels.
    Falls back to standard flat block-max if cols is not divisible by block_size.
    """
    import torch

    _, arr = _torch_f32(tensor, device)
    shape = list(arr.shape)
    source_flat = arr.reshape(-1).detach().cpu()

    if len(shape) < 2 or shape[-1] % block_size != 0:
        # Fallback to standard flat block-max for 1D or non-divisible tensors
        return _normalize_tensor_block_max_torch(tensor, block_size, device)

    rows = _product(shape[:-1])
    cols = shape[-1]
    mat = arr.reshape(rows, cols)

    # Reshape so blocks are channel-aligned: [rows, cols // block_size, block_size]
    blocks_per_row = cols // block_size
    blocked = mat.reshape(rows, blocks_per_row, block_size)
    scales = blocked.abs().amax(dim=2)  # [rows, blocks_per_row]
    safe = _fp16_storage_roundtrip(
        torch.where(scales == 0, torch.ones_like(scales), scales)
    )
    normalized = (blocked / safe.unsqueeze(2)).reshape(rows, cols)

    # Flatten scales in row-major order for compatibility with decode
    scales_flat = safe.reshape(-1)
    normalized_flat = normalized.reshape(-1)

    return (
        normalized_flat.reshape(arr.shape),
        scales_flat.detach().cpu(),
        source_flat,
    )


def _normalize_tensor_channel_block_max_numpy(tensor, block_size: int):
    """Channel-aware block-max (NumPy): blocks aligned to hidden-dim columns."""
    import numpy as np

    arr = _numpy_float32_array(tensor)
    shape = list(arr.shape)
    source_flat = arr.reshape(-1).copy()

    if len(shape) < 2 or shape[-1] % block_size != 0:
        return _normalize_tensor_block_max_numpy(tensor, block_size)

    rows = 1
    for s in shape[:-1]:
        rows *= int(s)
    cols = shape[-1]
    mat = arr.reshape(rows, cols)

    blocks_per_row = cols // block_size
    blocked = mat.reshape(rows, blocks_per_row, block_size)
    scales = np.abs(blocked).max(axis=2).astype(np.float32)  # [rows, blocks_per_row]
    safe = _fp16_storage_roundtrip(np.where(scales == 0, 1.0, scales).astype(np.float32))
    normalized = (blocked / safe[:, :, None]).reshape(rows, cols)

    scales_flat = safe.reshape(-1)
    normalized_flat = normalized.reshape(-1)

    return normalized_flat.reshape(arr.shape), scales_flat, source_flat


def _normalize_tensor_slrq_block_torch(tensor, block_size: int, device, salient_enabled: bool = True):
    import torch

    _, arr = _torch_f32(tensor, device)
    source_flat = arr.reshape(-1).detach().cpu().clone()
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)

    salient_weights = None
    salient_indices = None
    if salient_enabled:
        abs_blocks = blocks.abs()
        salient_indices = abs_blocks.argmax(dim=1)
        row_indices = torch.arange(blocks.shape[0], device=device)
        salient_weights = _fp16_storage_roundtrip(
            blocks[row_indices, salient_indices].clone()
        )
        blocks[row_indices, salient_indices] = 0.0
        max_for_anchor = blocks.abs().amax(dim=1)
    else:
        max_for_anchor = blocks.abs().amax(dim=1)

    safe = torch.where(max_for_anchor == 0, torch.ones_like(max_for_anchor), max_for_anchor)
    pow2 = torch.exp2(torch.ceil(torch.log2(safe)))
    # Floor at fp16 smallest subnormal (2**-24). A block whose anchor-max is tiny
    # (e.g. unused embedding rows ~1e-8) yields a power-of-2 scale below fp16 range;
    # the fp16 storage roundtrip would flush it to 0.0, making blocks / safe divide
    # by zero -> NaN that then poisons the whole tensor.
    pow2 = torch.clamp(pow2, min=2.0 ** -24)
    safe = _fp16_storage_roundtrip(pow2)

    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]

    return (
        normalized.reshape(arr.shape),
        safe.detach().cpu(),
        salient_weights.detach().cpu() if salient_weights is not None else None,
        salient_indices.detach().cpu() if salient_indices is not None else None,
        source_flat,
    )


def _normalize_tensor_slrq_block_numpy(tensor, block_size: int, salient_enabled: bool = True):
    import numpy as np

    arr = _numpy_float32_array(tensor)
    source_flat = arr.reshape(-1).copy()
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    pad = (-n) % block_size
    if pad:
        flat = np.pad(flat, (0, pad), mode="constant")
    blocks = flat.reshape(-1, block_size)

    salient_weights = None
    salient_indices = None
    if salient_enabled:
        abs_blocks = np.abs(blocks)
        salient_indices = np.argmax(abs_blocks, axis=1)
        row_indices = np.arange(blocks.shape[0])
        salient_weights = _fp16_storage_roundtrip(
            blocks[row_indices, salient_indices].copy()
        )
        blocks[row_indices, salient_indices] = 0.0
        max_for_anchor = np.abs(blocks).max(axis=1)
    else:
        max_for_anchor = np.abs(blocks).max(axis=1)

    safe = np.where(max_for_anchor == 0, 1.0, max_for_anchor).astype(np.float32)
    pow2 = np.exp2(np.ceil(np.log2(safe))).astype(np.float32)
    # Floor at fp16 smallest subnormal (2**-24). A block whose anchor-max is tiny
    # (e.g. unused embedding rows ~1e-8) yields a power-of-2 scale below fp16 range;
    # the fp16 storage roundtrip would flush it to 0.0, making blocks / safe divide
    # by zero -> NaN that then poisons the whole tensor.
    pow2 = np.maximum(pow2, np.float32(2.0 ** -24))
    safe = _fp16_storage_roundtrip(pow2)

    normalized = (blocks / safe[:, np.newaxis]).reshape(-1)
    if pad:
        normalized = normalized[:n]

    return (
        normalized.reshape(arr.shape),
        safe,
        salient_weights,
        salient_indices,
        source_flat,
    )


def _apply_block_max_scales(flat, scales, block_size: int):
    return _apply_block_max_scales_numpy(flat, scales, block_size).tolist()


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
    scales = _fp16_storage_roundtrip(1.0 / s)
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
    scales = _fp16_storage_roundtrip(1.0 / s)
    normalized = (rows / scales[None, :]).reshape(arr.shape)
    return normalized, scales.astype(np.float32), arr.reshape(-1)


def _apply_normalization(
    tensor, name, normalization, awq_activations, awq_alpha,
    block_scale_size, backend, device, awq_fallbacks,
    slrq_salient: bool = True,
):
    is_torch = backend == "torch"
    has_awq = awq_activations is not None and name in awq_activations
    
    # Defaults
    row_scales = None
    source_flat = None
    awq_col_scales = None
    salient_weights = None
    salient_indices = None

    def _get_none():
        if is_torch:
            _, arr = _torch_f32(tensor, device)
            return arr, None, arr.reshape(-1).detach().cpu()
        arr = _numpy_float32_array(tensor)
        return arr, None, arr.reshape(-1)

    if normalization == "slrq-block":
        if is_torch:
            tensor, row_scales, salient_weights, salient_indices, source_flat = _normalize_tensor_slrq_block_torch(
                tensor, block_scale_size, device, salient_enabled=slrq_salient)
        else:
            tensor, row_scales, salient_weights, salient_indices, source_flat = _normalize_tensor_slrq_block_numpy(
                tensor, block_scale_size, salient_enabled=slrq_salient)
    
    elif normalization == "awq":
        if not has_awq:
            tensor, row_scales, source_flat = _get_none()
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
            tensor, row_scales, source_flat = _normalize_tensor_block_max_torch(tensor, block_scale_size, device)
        else:
            tensor, row_scales, source_flat, awq_col_scales = _normalize_tensor_awq_block_max_torch(
                tensor, awq_activations[name], awq_alpha, block_scale_size, device)
                
    elif normalization == "channel-block-max":
        if is_torch:
            tensor, row_scales, source_flat = _normalize_tensor_channel_block_max_torch(tensor, block_scale_size, device)
        else:
            tensor, row_scales, source_flat = _normalize_tensor_channel_block_max_numpy(tensor, block_scale_size)

    elif normalization == "block-max":
        if is_torch:
            tensor, row_scales, source_flat = _normalize_tensor_block_max_torch(tensor, block_scale_size, device)
        else:
            tensor, row_scales, source_flat = _normalize_tensor_block_max_numpy(tensor, block_scale_size)
            
    else:
        # No normalization
        tensor, row_scales, source_flat = _get_none()

    return tensor, row_scales, source_flat, awq_col_scales, salient_weights, salient_indices


def _apply_col_l2_scales(flat, shape, scales):
    return _apply_col_l2_scales_numpy(flat, shape, scales).tolist()


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
