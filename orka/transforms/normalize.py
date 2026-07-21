"""Normalization variants: block-max, awq, awq-block-max, slrq-block.

Each mode has numpy + torch implementations side by side. Dispatcher
``_apply_normalization`` picks the right one based on backend/availability.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from orka.core._format import _fp16_storage_roundtrip
from orka.core._tensor import _numpy_float32_array, _torch_f32
from orka.core._util import _product

#: fp16's smallest positive subnormal. A block scale below this survives the
#: ``scales == 0`` guard but flushes to 0.0 in the fp16 sidecar, so the normalize
#: divide emits inf/nan. Real trigger: dead vocab rows in an embedding.
_MIN_STORABLE_SCALE = 2.0 ** -24


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
        torch.where(scales == 0, torch.ones_like(scales), scales).clamp(min=_MIN_STORABLE_SCALE)
    )
    normalized = (blocks / safe[:, None]).reshape(-1)
    if pad:
        normalized = normalized[:n]

    return (
        normalized.reshape(arr.shape),
        safe.detach().cpu(),
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
        torch.where(scales == 0, torch.ones_like(scales), scales).clamp(min=_MIN_STORABLE_SCALE)
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
    safe = _fp16_storage_roundtrip(
        np.maximum(np.where(scales == 0, 1.0, scales), _MIN_STORABLE_SCALE).astype(np.float32)
    )
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
        torch.where(scales == 0, torch.ones_like(scales), scales).clamp(min=_MIN_STORABLE_SCALE)
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
    safe = _fp16_storage_roundtrip(
        np.maximum(np.where(scales == 0, 1.0, scales), _MIN_STORABLE_SCALE).astype(np.float32)
    )
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
        # anchor-max over the salient-removed block: mirror the zeroing into the
        # already-computed abs_blocks instead of a second full-tensor .abs() pass.
        abs_blocks[row_indices, salient_indices] = 0.0
        max_for_anchor = abs_blocks.amax(dim=1)
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
        abs_blocks[row_indices, salient_indices] = 0.0     # reuse instead of a 2nd full abs
        max_for_anchor = abs_blocks.max(axis=1)
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


def apply_block_scales(decoded, scales, block_size: int, *, backend="numpy", device="cpu"):
    """Inverse of block-max-family normalization: multiply each block by its stored scale.

    Backend-parametric so the numpy decode/reconstruct path and the torch inference path
    share one implementation instead of each reimplementing the pad/reshape/scale math.
    ``decoded`` is a flat numpy array (numpy) or a flat torch tensor on ``device`` (torch);
    ``scales`` is the numpy scale vector read from the sidecar.
    """
    if backend == "torch":
        import torch

        scales_t = torch.from_numpy(scales).to(device)
        n = decoded.numel()
        pad = (-n) % block_size
        if pad:
            decoded = torch.cat([decoded, torch.zeros(pad, dtype=torch.float32, device=device)])
        out = (decoded.reshape(-1, block_size) * scales_t[: decoded.numel() // block_size, None]).reshape(-1)
        return out[:n] if pad else out
    return _apply_block_max_scales_numpy(decoded, scales, block_size)


def apply_col_scales(decoded, shape, scales, *, backend="numpy", device="cpu"):
    """Inverse of awq col-l2 normalization: multiply each column by its stored scale.

    Backend-parametric, mirroring ``apply_block_scales``. Column layout follows the numpy
    helper: rows = shape[0], cols = product(shape[1:]). ``scales`` is the numpy col-scale
    vector read from the sidecar.
    """
    if backend == "torch":
        import torch

        rows = int(shape[0])
        cols = _product([int(s) for s in shape[1:]])
        scales_t = torch.from_numpy(scales).to(device)
        flat = decoded[: rows * cols].reshape(rows, cols)
        return (flat * scales_t[None, :]).reshape(-1)
    return _apply_col_l2_scales_numpy(decoded, shape, scales)


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


@dataclass
class NormalizationResult:
    """Outcome of one normalization mode. Named slots replace the old positional
    6-tuple; each handler fills only the slots its mode produces (the rest stay None)."""

    tensor: object
    row_scales: object = None
    source_flat: object = None
    awq_col_scales: object = None
    salient_weights: object = None
    salient_indices: object = None


# A handler maps (tensor + resolved context) -> NormalizationResult. The context is passed
# as keyword args; handlers accept **_ so the registry can call them uniformly.
NormalizationHandler = Callable[..., NormalizationResult]


def _normalize_none(tensor, *, is_torch, device, **_) -> NormalizationResult:
    if is_torch:
        _, arr = _torch_f32(tensor, device)
        return NormalizationResult(tensor=arr, source_flat=arr.reshape(-1).detach().cpu())
    arr = _numpy_float32_array(tensor)
    return NormalizationResult(tensor=arr, source_flat=arr.reshape(-1))


def _normalize_slrq_block(tensor, *, is_torch, device, block_scale_size, slrq_salient, **_) -> NormalizationResult:
    fn = _normalize_tensor_slrq_block_torch if is_torch else _normalize_tensor_slrq_block_numpy
    args = (tensor, block_scale_size, device) if is_torch else (tensor, block_scale_size)
    t, row_scales, sal_w, sal_i, source_flat = fn(*args, salient_enabled=slrq_salient)
    return NormalizationResult(
        tensor=t, row_scales=row_scales, source_flat=source_flat,
        salient_weights=sal_w, salient_indices=sal_i,
    )


def _normalize_awq(tensor, *, is_torch, device, name, awq_activations, awq_alpha, has_awq, **_) -> NormalizationResult:
    if not has_awq:
        return _normalize_none(tensor, is_torch=is_torch, device=device)
    if is_torch:
        t, row_scales, source_flat = _normalize_tensor_awq_torch(tensor, awq_activations[name], awq_alpha, device)
    else:
        t, row_scales, source_flat = _normalize_tensor_awq_numpy(tensor, awq_activations[name], awq_alpha)
    return NormalizationResult(tensor=t, row_scales=row_scales, source_flat=source_flat)


def _normalize_awq_block_max(tensor, *, is_torch, device, name, awq_activations, awq_alpha, has_awq, block_scale_size, **_) -> NormalizationResult:
    if not is_torch:
        raise RuntimeError("awq-block-max requires --backend torch")
    if not has_awq:
        t, row_scales, source_flat = _normalize_tensor_block_max_torch(tensor, block_scale_size, device)
        return NormalizationResult(tensor=t, row_scales=row_scales, source_flat=source_flat)
    t, row_scales, source_flat, awq_col_scales = _normalize_tensor_awq_block_max_torch(
        tensor, awq_activations[name], awq_alpha, block_scale_size, device)
    return NormalizationResult(tensor=t, row_scales=row_scales, source_flat=source_flat, awq_col_scales=awq_col_scales)


def _normalize_channel_block_max(tensor, *, is_torch, device, block_scale_size, **_) -> NormalizationResult:
    if is_torch:
        t, row_scales, source_flat = _normalize_tensor_channel_block_max_torch(tensor, block_scale_size, device)
    else:
        t, row_scales, source_flat = _normalize_tensor_channel_block_max_numpy(tensor, block_scale_size)
    return NormalizationResult(tensor=t, row_scales=row_scales, source_flat=source_flat)


def _normalize_block_max(tensor, *, is_torch, device, block_scale_size, **_) -> NormalizationResult:
    if is_torch:
        t, row_scales, source_flat = _normalize_tensor_block_max_torch(tensor, block_scale_size, device)
    else:
        t, row_scales, source_flat = _normalize_tensor_block_max_numpy(tensor, block_scale_size)
    return NormalizationResult(tensor=t, row_scales=row_scales, source_flat=source_flat)


# Mode -> handler. Unknown / "none" falls through to _normalize_none. Register a new mode
# with register_normalization() - the dispatcher does not change (open/closed).
NORMALIZATION_REGISTRY: dict[str, NormalizationHandler] = {
    "slrq-block": _normalize_slrq_block,
    "awq": _normalize_awq,
    "awq-block-max": _normalize_awq_block_max,
    "channel-block-max": _normalize_channel_block_max,
    "block-max": _normalize_block_max,
}


def register_normalization(mode: str, handler: NormalizationHandler) -> None:
    """Register a normalization mode handler so it dispatches without editing the chain."""
    NORMALIZATION_REGISTRY[mode] = handler


def normalization_modes() -> list[str]:
    """Modes the dispatcher recognizes (plus the implicit 'none' fallback)."""
    return sorted(NORMALIZATION_REGISTRY)


# Single source of truth for "this mode persists a per-block scale sidecar"
# (the block_max_scale vector). awq-block-max is included: it stores block scales AND
# awq column scales. Shared by pack, manifest, decode, distill, refinement, and the
# inference loader so the membership cannot drift across call sites.
#
# NOTE: this is the "stores block scales at all" grouping (all four). The decode/metrics
# inverse paths use a narrower three-mode branch (awq-block-max handled separately because
# it also needs col scales) - that grouping is intentionally NOT this set.
BLOCK_SCALE_NORMALIZATIONS = frozenset(
    {"block-max", "channel-block-max", "slrq-block", "awq-block-max"}
)


def stores_block_scales(normalization) -> bool:
    """True if the normalization mode persists a per-block scale sidecar."""
    return normalization in BLOCK_SCALE_NORMALIZATIONS


def _apply_normalization(
    tensor, name, normalization, awq_activations, awq_alpha,
    block_scale_size, backend, device, awq_fallbacks,
    slrq_salient: bool = True,
):
    is_torch = backend == "torch"
    has_awq = awq_activations is not None and name in awq_activations
    handler = NORMALIZATION_REGISTRY.get(normalization, _normalize_none)
    res = handler(
        tensor,
        name=name,
        is_torch=is_torch,
        device=device,
        awq_activations=awq_activations,
        awq_alpha=awq_alpha,
        has_awq=has_awq,
        block_scale_size=block_scale_size,
        slrq_salient=slrq_salient,
    )
    return (
        res.tensor, res.row_scales, res.source_flat,
        res.awq_col_scales, res.salient_weights, res.salient_indices,
    )


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
