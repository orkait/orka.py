"""Outlier extraction: top-K weights kept as fp16 sidecar.

Selection is by magnitude, or - when calibration column importance is
available - by Hessian-proxy salience ``h_col * w^2`` (the contribution of a
weight's quantization error to layer OUTPUT error, diagonal approximation).
A large weight on a dead input column wastes escape budget; a modest weight
on a hot column deserves it.

On-disk sidecar I/O lives in ``orka.core._format`` (single source of truth for format).
"""

from __future__ import annotations

from orka.core._tensor import _is_numpy_array, _is_torch_tensor


def _outlier_scores(flat_abs, col_importance, cols: int | None):
    """Salience scores for a flattened (row-major) weight vector."""
    if col_importance is None or not cols:
        return flat_abs * flat_abs if hasattr(flat_abs, "__mul__") else flat_abs
    n = int(flat_abs.shape[0])
    if n % cols != 0:
        return flat_abs * flat_abs
    if _is_torch_tensor(flat_abs):
        import torch

        h = torch.as_tensor(
            col_importance, dtype=torch.float32, device=flat_abs.device
        ).reshape(-1)
        if int(h.shape[0]) != cols:
            return flat_abs * flat_abs
        return (flat_abs * flat_abs) * h.repeat(n // cols)
    import numpy as np

    h = np.asarray(col_importance, dtype=np.float32).reshape(-1)
    if h.shape[0] != cols:
        return flat_abs * flat_abs
    return (flat_abs * flat_abs) * np.tile(h, n // cols)


def _extract_outliers(
    vectors,
    outlier_frac: float,
    packed_values: int,
    col_importance=None,
    cols: int | None = None,
):
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
        scores = _outlier_scores(relevant.abs(), col_importance, cols)
        _, topk_idx = torch.topk(scores, k)
        positions = topk_idx.detach().cpu().to(torch.int64).numpy()
        values = flat[topk_idx].detach().cpu().to(torch.float32).numpy()
        flat[topk_idx] = 0
        return positions, values, flat.reshape(vectors.shape)
    if _is_numpy_array(vectors):
        import numpy as np

        flat = vectors.reshape(-1).copy()
        relevant = flat[:n]
        scores = _outlier_scores(np.abs(relevant), col_importance, cols)
        order = np.argpartition(scores, -k)[-k:]
        positions = order.astype(np.int64)
        values = flat[positions].astype(np.float32)
        flat[positions] = 0
        return positions, values, flat.reshape(vectors.shape)
    raise TypeError(f"unsupported tensor type for outlier extraction: {type(vectors)}")
