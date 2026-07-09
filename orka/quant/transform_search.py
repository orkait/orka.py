"""Pure helpers for the per-tensor transform search (allocate increment 2).

See docs/design/per-tensor-transform-search.md. These two functions are the cheap,
side-effect-free core of the search:

- ``scalar_quant_proxy``: an O(N) per-block scalar-quant MSE that ranks how
  quantizable a (transformed) distribution is, standing in for a full VQ k-means
  probe across the transform grid.
- ``transform_overhead_bits``: the storage overhead a transform adds to a tensor's
  rate, so the Lagrangian bit-allocation charges each candidate honestly.

Both are numpy-only and carry no pack/device state, so they unit-test without
touching the pack pipeline.
"""
from __future__ import annotations

import numpy as np


def scalar_quant_reconstruct(values, *, bits: int = 4, block_size: int = 128):
    """Per-block symmetric-uniform ``bits``-bit scalar quantize-then-dequantize.

    Returns the reconstruction as a float64 array the same length as ``values``.
    Each block's scale is ``max_abs / (2**(bits-1) - 1)``; values round to the nearest
    level and clip to the symmetric range.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.shape[0])
    if n == 0:
        return v
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    lim = max((1 << (bits - 1)) - 1, 1)  # symmetric levels; bits=1 -> 1 (sign)
    pad = (-n) % block_size
    if pad:
        v = np.concatenate([v, np.zeros(pad, dtype=np.float64)])
    blocks = v.reshape(-1, block_size)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scale = np.where(amax > 0.0, amax / lim, 1.0)
    recon = np.clip(np.round(blocks / scale), -lim, lim) * scale
    return recon.reshape(-1)[:n]


def scalar_quant_proxy(
    values,
    *,
    bits: int = 4,
    block_size: int = 128,
    original_scales=None,
) -> float:
    """Per-block symmetric-uniform ``bits``-bit scalar quantization MSE.

    A cheap stand-in for full VQ distortion when ranking transforms: a rotation that
    suppresses outliers or a normalization that equalizes block scale lowers this MSE
    the same way it lowers VQ MSE.

    ``values`` are in transform space (already normalized/rotated). For an orthogonal
    rotation this MSE equals the original-space MSE (isometry), so nothing extra is
    needed. For a *scaling* normalization, pass ``original_scales`` - the per-block
    scale that was divided out to produce ``values`` - and the per-block error is
    reweighted by ``scale**2`` so the result is in original weight units. Without it,
    a block-max-normalized tensor scores artificially low (its values are shrunk).
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.shape[0])
    if n == 0:
        return 0.0

    recon = scalar_quant_reconstruct(v, bits=bits, block_size=block_size)
    err = v - recon  # [n], original (transform) space

    if original_scales is None:
        return float(np.mean(err ** 2))

    # Reweight each block's error by its original scale**2 (normalized -> original).
    s = np.asarray(original_scales, dtype=np.float64).reshape(-1)
    n_blocks = (n + block_size - 1) // block_size
    if s.shape[0] != n_blocks:
        raise ValueError("original_scales length must match the block count")
    pad = (-n) % block_size
    if pad:
        err = np.concatenate([err, np.zeros(pad, dtype=np.float64)])
    err2 = (err.reshape(n_blocks, block_size) * s[:, None]) ** 2
    return float(np.mean(err2.reshape(-1)[:n]))


def transform_overhead_bits(
    normalization: str | None,
    rotation: str | None,
    *,
    numel: int,
    block_size: int = 128,
    scale_bits: int = 16,
    salient_frac: float = 0.0,
    salient_bits_each: int = 48,
) -> int:
    """Extra storage bits a transform config adds to a tensor's rate.

    Added to the index bits so the Lagrangian allocator compares candidates at their
    true cost. Exact for block scales (``n_blocks * scale_bits``); the salient sidecar
    is an estimate (``salient_count * salient_bits_each`` - fp16 value + ~32-bit
    position by default). Rotations cost nothing on disk (a seed only).
    """
    if numel < 0:
        raise ValueError("numel must be >= 0")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if not 0.0 <= salient_frac <= 1.0:
        raise ValueError("salient_frac must be in [0, 1]")

    n_blocks = (numel + block_size - 1) // block_size
    bits = 0
    if normalization in ("block-max", "channel-block-max", "awq-block-max"):
        bits += n_blocks * scale_bits
    elif normalization == "slrq-block":
        bits += n_blocks * scale_bits
        bits += int(round(salient_frac * numel)) * salient_bits_each
    # "none", "awq", None: no per-block sidecar.
    # rotation ("hadamard"/"orthogonal"/"none"/None): 0 disk overhead (seed only).
    return int(bits)


_SCALING_NORMS = ("block-max", "channel-block-max", "awq-block-max", "slrq-block")

# Default v1 search grid: (normalization, rotation). orthogonal QR is excluded
# (O(N^2)/tensor too costly to probe); slrq-block is proxied as block-max.
DEFAULT_TRANSFORM_GRID = (
    ("none", "none"),
    ("block-max", "none"),
    ("none", "hadamard"),
    ("block-max", "hadamard"),
)


def apply_transform(weight, normalization: str | None, rotation: str | None, *, norm_block: int = 128, device: str | None = None):
    """Apply (normalization, rotation) to a 2D weight for the allocate probe.

    Returns ``(transformed_2d, denorm_factor)``. ``denorm_factor`` = mean(block
    scale**2) converts a distortion measured on the transformed values back to
    original weight units (1.0 when no scaling normalization; rotation is isometric
    so it contributes 1). Mirrors the normalize-then-rotate order in the pack.

    The Hadamard FWHT is the dominant cost; when ``device`` names a CUDA device it
    runs the block FWHT on the GPU (~8.6x even paying the host roundtrip, measured),
    returning numpy so the caller's interface is unchanged.
    """
    W = np.asarray(weight, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    elif W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    rows, cols = W.shape
    flat = W.reshape(-1)
    n = int(flat.size)

    factor = 1.0
    if normalization in _SCALING_NORMS:
        pad = (-n) % norm_block
        fp = np.concatenate([flat, np.zeros(pad)]) if pad else flat
        blk = fp.reshape(-1, norm_block)
        amax = np.max(np.abs(blk), axis=1)
        scales = np.where(amax > 0.0, amax, 1.0)
        factor = float(np.mean(scales ** 2))
        Wn = (blk / scales[:, None]).reshape(-1)[:n].reshape(rows, cols)
    elif normalization in ("none", "awq", None):
        Wn = W
    else:
        raise ValueError(f"unsupported normalization for proxy: {normalization!r}")

    if rotation == "hadamard":
        from orka.transforms.rotate import _hadamard_block_size

        hb = _hadamard_block_size(cols)
        if device is not None and "cuda" in str(device):
            import torch

            from orka.transforms.rotate import _block_fwht_torch

            t = torch.as_tensor(Wn, dtype=torch.float32, device=device).reshape(rows, cols)
            Wr = _block_fwht_torch(t, hb).detach().cpu().numpy().astype(np.float64)
        else:
            from orka.transforms.rotate import _block_fwht_numpy

            Wr = np.asarray(_block_fwht_numpy(Wn, hb), dtype=np.float64)
    elif rotation in ("none", None):
        Wr = Wn
    else:
        raise ValueError(f"unsupported rotation for proxy: {rotation!r}")
    return Wr, factor


def rank_transforms(weight, grid=DEFAULT_TRANSFORM_GRID, *, bits: int = 4, norm_block: int = 128):
    """Rank transform configs by the scalar-quant proxy (lower MSE first).

    Returns ``[((normalization, rotation), proxy_mse), ...]`` sorted ascending.
    Configs that are infeasible for this tensor (e.g. Hadamard on a width with no
    usable power-of-two block) are skipped.
    """
    scored = []
    for norm, rot in grid:
        try:
            mse = transform_proxy_distortion(weight, norm, rot, bits=bits, norm_block=norm_block)
        except ValueError:
            continue
        scored.append(((norm, rot), mse))
    scored.sort(key=lambda x: x[1])
    return scored


def transform_proxy_distortion(
    weight,
    normalization: str | None,
    rotation: str | None,
    *,
    bits: int = 4,
    norm_block: int = 128,
) -> float:
    """Original-space scalar-quant MSE of a (normalization, rotation) config on a 2D
    weight - the ranking signal for the per-tensor transform search.

    Faithful order, matching the pack: block-max normalize (flattened ``norm_block``
    blocks) -> Hadamard rotate (along columns) -> b-bit scalar quant -> inverse-rotate
    the error (Hadamard is its own orthonormal inverse) -> denormalize -> MSE.

    The internal scalar quant uses ONE per-tensor scale (not per-block), so a finer
    block-max normalization and an outlier-spreading rotation actually lower the score
    - a per-block scalar quant would already bake in block-max and hide its benefit.

    v1 grid: ``normalization`` in {``none``, ``block-max``, ``slrq-block`` (treated as
    block-max for the proxy)}, ``rotation`` in {``none``, ``hadamard``}. Raises
    ValueError if Hadamard has no usable block for this width (caller drops the config).
    """
    W = np.asarray(weight, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    elif W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    rows, cols = W.shape
    flat = W.reshape(-1)
    n = int(flat.size)
    if n == 0:
        return 0.0

    # 1. block-max normalization over flattened norm_block blocks.
    scales = None
    if normalization in _SCALING_NORMS:
        pad = (-n) % norm_block
        fp = np.concatenate([flat, np.zeros(pad)]) if pad else flat
        blk = fp.reshape(-1, norm_block)
        amax = np.max(np.abs(blk), axis=1)
        scales = np.where(amax > 0.0, amax, 1.0)  # per-block divisor
        vn = (blk / scales[:, None]).reshape(-1)[:n]
    elif normalization not in ("none", "awq", None):
        raise ValueError(f"unsupported normalization for proxy: {normalization!r}")
    else:
        vn = flat
    Wn = vn.reshape(rows, cols)

    # 2. Hadamard rotation along columns (isometric).
    hb = None
    if rotation == "hadamard":
        from orka.transforms.rotate import _block_fwht_numpy, _hadamard_block_size

        hb = _hadamard_block_size(cols)  # raises if no usable pow2 block
        Wr = np.asarray(_block_fwht_numpy(Wn, hb), dtype=np.float64)
    elif rotation in ("none", None):
        Wr = Wn
    else:
        raise ValueError(f"unsupported rotation for proxy: {rotation!r}")

    # 3. b-bit scalar quant with ONE per-tensor scale (block_size = n).
    recon = scalar_quant_reconstruct(Wr.reshape(-1), bits=bits, block_size=n)
    err_r = (Wr.reshape(-1) - recon).reshape(rows, cols)

    # 4. inverse rotation -> normalized space (Hadamard self-inverse).
    if rotation == "hadamard":
        from orka.transforms.rotate import _block_fwht_numpy

        err_n = np.asarray(_block_fwht_numpy(err_r, hb), dtype=np.float64).reshape(-1)
    else:
        err_n = err_r.reshape(-1)

    # 5. denormalize the error -> original weight space.
    if scales is not None:
        pad = (-n) % norm_block
        en = np.concatenate([err_n, np.zeros(pad)]) if pad else err_n
        en = (en.reshape(-1, norm_block) * scales[:, None]).reshape(-1)[:n]
    else:
        en = err_n
    return float(np.mean(en ** 2))
