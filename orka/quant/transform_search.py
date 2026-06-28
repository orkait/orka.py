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
    q = np.clip(np.round(blocks / scale), -lim, lim) * scale
    err2 = (blocks - q) ** 2  # [n_blocks, block_size]

    if original_scales is not None:
        s = np.asarray(original_scales, dtype=np.float64).reshape(-1)
        if s.shape[0] != blocks.shape[0]:
            raise ValueError("original_scales length must match the block count")
        err2 = err2 * (s[:, None] ** 2)

    return float(np.mean(err2.reshape(-1)[:n]))  # mean over real (unpadded) elements


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
