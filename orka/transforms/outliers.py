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
