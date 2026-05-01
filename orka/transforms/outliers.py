"""Outlier extraction: top-K-magnitude weights kept as fp16 sidecar.

On-disk sidecar I/O lives in ``orka._format`` (single source of truth for format).
"""

from __future__ import annotations

from orka._tensor import _is_numpy_array, _is_torch_tensor


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
    raise TypeError(f"unsupported tensor type for outlier extraction: {type(vectors)}")
