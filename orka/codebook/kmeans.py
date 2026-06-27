"""K-means dispatch: pick numpy or torch backend by input/device.

Backend implementations live in _kmeans_numpy / _kmeans_torch."""
from __future__ import annotations

import math
from typing import Sequence

from orka._runtime import _resolve_torch_device, _check_ram_cap
from orka.core._tensor import _is_numpy_array, _is_torch_tensor, _torch_float32_matrix


from orka.codebook._kmeans_numpy import (  # noqa: F401
    _kmeans_parallel_init_numpy, _numpy_assign, _numpy_centroid_sums, _learn_codebook_numpy,
)
from orka.codebook._kmeans_torch import (  # noqa: F401
    _kmeans_pp_init_torch, _torch_assign, _learn_codebook_torch,
)


def learn_codebook_auto(
    vectors,
    codebook_size: int,
    iterations: int,
    backend: str,
    device: str = "cpu",
    vector_weights=None,
    seed: int | None = None,
    initial_codebook=None,
    sample_weights=None,
):
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if backend == "torch":
        return _learn_codebook_torch(
            vectors,
            codebook_size,
            iterations,
            device,
            vector_weights=vector_weights,
            seed=seed,
            initial_codebook=initial_codebook,
            sample_weights=sample_weights,
        )
    if not _is_numpy_array(vectors):
        raise RuntimeError("NumPy backend requires NumPy array tensors")
    return _learn_codebook_numpy(
        vectors,
        codebook_size,
        iterations,
        seed=seed,
        initial_codebook=initial_codebook,
        vector_weights=vector_weights,
        sample_weights=sample_weights,
    )


def quantize_vectors_auto(
    vectors, codebook, backend: str, device: str = "cpu", vector_weights=None
):
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if backend == "torch":
        return _torch_assign(vectors, codebook, device, vector_weights=vector_weights)
    if not _is_numpy_array(vectors):
        raise RuntimeError("NumPy backend requires NumPy array tensors")
    return _numpy_assign(vectors, codebook, vector_weights=vector_weights)
