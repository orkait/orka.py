"""Pure input/vector-prep helpers for the pack pipeline.

No pack-specific state - vector extraction from a tensor, deterministic training-row
sampling (with aligned per-sample weights), and the importance-weight cache digest.
Shared by pack.py (pack_checkpoint) and pack_refine.py.
"""

from __future__ import annotations

from orka._tensor import _is_torch_tensor, _numpy_float32_array, _torch_f32


def _weights_digest(sample_weights) -> str:
    """Cache-key component for importance weights. Content-addressed so a
    cached codebook is never reused when calibration activations changed."""
    if sample_weights is None:
        return "unweighted"
    import hashlib

    arr = (
        sample_weights.detach().cpu().numpy()
        if hasattr(sample_weights, "detach")
        else sample_weights
    )
    import numpy as np

    payload = np.asarray(arr, dtype="<f4").tobytes()
    return "sw-" + hashlib.blake2b(payload, digest_size=8).hexdigest()


def _sample_vectors_and_weights(vectors, weights, sample_vectors: int | None):
    """Sample training rows and their per-sample weights at identical positions.

    Mirrors ``_sample_vector_rows`` (deterministic linspace positions) so the
    weight of each sampled row stays aligned with the row itself.
    """
    if (
        sample_vectors is None
        or sample_vectors <= 0
        or sample_vectors >= len(vectors)
    ):
        return vectors, weights
    if _is_torch_tensor(vectors):
        import torch

        positions = (
            torch.linspace(
                0,
                len(vectors) - 1,
                steps=sample_vectors,
                device=vectors.device,
                dtype=torch.float64,
            )
            .round()
            .to(dtype=torch.long)
            .clamp_(max=len(vectors) - 1)
        )
        sampled = vectors.index_select(0, positions)
        if weights is None:
            return sampled, None
        if _is_torch_tensor(weights):
            return sampled, weights.index_select(0, positions.to(weights.device))
        return sampled, weights[positions.detach().cpu().numpy()]
    import numpy as np

    positions = np.linspace(0, len(vectors) - 1, sample_vectors, dtype=np.int64)
    sampled = vectors[positions]
    return sampled, (weights[positions] if weights is not None else None)


def _numpy_vectors_from_tensor(tensor: object, group_size: int, limit: int | None):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy backend requires numpy") from exc
    flat = _numpy_float32_array(tensor).reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = np.pad(flat, (0, group_size - remainder), mode="constant")
    return original_len, int(flat.shape[0]), flat.reshape(-1, group_size)


def _torch_vectors_from_tensor(
    tensor: object, group_size: int, limit: int | None, device: str
):
    import torch

    _, arr = _torch_f32(tensor, device)
    flat = arr.reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    original_len = int(flat.shape[0])
    remainder = original_len % group_size
    if remainder:
        flat = torch.nn.functional.pad(flat, (0, group_size - remainder))
    return original_len, int(flat.shape[0]), flat.reshape(-1, group_size)
