"""K-means: scalable ++ init (k-means||) + Lloyd iterations + nearest-centroid assign.

Backends: numpy (CPU) and torch (CPU/CUDA, GEMM-based distance).
"""

from __future__ import annotations

import math
from typing import Sequence

from orka._runtime import _resolve_torch_device, _check_ram_cap
from orka._tensor import _is_numpy_array, _is_torch_tensor, _torch_float32_matrix


def _kmeans_pp_init_torch(
    rows, k: int, seed: int | None = None, oversample_factor: float = 2.0
):
    """Scalable K-Means++ (k-means||) init, memory-bounded.

    Memory bound: ``per_round_candidates`` capped at ``min(oversample_factor * k, n // 5)``
    so the (batch x candidates) distance matrix never blows past ~256 MB.
    Total accumulated candidates capped at ``5 * k`` regardless of round count.
    """
    import torch

    n = int(rows.shape[0])
    if k >= n:
        return rows.clone()
    gen = torch.Generator(device=rows.device)
    if seed is not None:
        gen.manual_seed(int(seed) & ((1 << 63) - 1))

    first = int(torch.randint(n, (1,), generator=gen, device=rows.device).item())
    min_d2 = torch.sum((rows - rows[first]) ** 2, dim=1)
    candidate_chunks: list = [rows[first].unsqueeze(0)]
    total_candidates = 1
    candidate_cap = 5 * k  # absolute upper bound across all rounds

    rounds = 5
    per_round_cap = max(1, min(int(oversample_factor * k), n // 5))
    for _ in range(rounds):
        if total_candidates >= candidate_cap:
            break
        sum_d2 = min_d2.sum().item()
        if sum_d2 == 0:
            break
        probs = min_d2 / sum_d2
        rand_vals = torch.rand(n, generator=gen, device=rows.device)
        chosen = torch.where(rand_vals < probs * per_round_cap)[0]
        if chosen.numel() == 0:
            break
        # Cap candidates this round to avoid distance-matrix blow-up.
        if int(chosen.numel()) > per_round_cap:
            chosen = chosen[:per_round_cap]

        new_centers = rows[chosen].contiguous()
        candidate_chunks.append(new_centers)
        total_candidates += int(new_centers.shape[0])

        # Update min_d2 with GEMM-form distance, chunking by ~256 MB matrix budget.
        c_norm_sq = torch.sum(new_centers * new_centers, dim=1, keepdim=True).T
        c_count = int(new_centers.shape[0])
        # 256 MB / (4 bytes * c_count) rows per chunk
        batch_size = max(256, min(65536, (1 << 28) // (4 * max(c_count, 1))))
        for i in range(0, n, batch_size):
            batch_rows = rows[i : i + batch_size]
            r_norm_sq = torch.sum(batch_rows * batch_rows, dim=1, keepdim=True)
            dists = torch.addmm(
                (r_norm_sq + c_norm_sq),
                batch_rows,
                new_centers.T,
                alpha=-2.0,
                beta=1.0,
            )
            min_d2[i : i + batch_size] = torch.minimum(
                min_d2[i : i + batch_size], dists.min(dim=1)[0]
            )
            del dists, batch_rows, r_norm_sq
        del new_centers, c_norm_sq

    centroids = torch.cat(candidate_chunks, dim=0)
    if centroids.shape[0] > k:
        # Reduce oversampled candidates to exactly k via classic K-Means++ on the subset.
        subset = centroids
        sub_n = int(subset.shape[0])
        final_idx = torch.empty(k, dtype=torch.long, device=rows.device)
        final_idx[0] = 0
        sub_d2 = torch.sum((subset - subset[0]) ** 2, dim=1)
        for j in range(1, k):
            sum_d2 = sub_d2.sum().item()
            if sum_d2 == 0:
                final_idx[j] = j % sub_n
                continue
            probs = sub_d2 / sum_d2
            cumprobs = torch.cumsum(probs, dim=0)
            r = torch.rand(1, generator=gen, device=rows.device).item()
            chosen_idx = int(torch.searchsorted(cumprobs, r).item())
            chosen_idx = min(chosen_idx, sub_n - 1)
            final_idx[j] = chosen_idx
            d2 = torch.sum((subset - subset[chosen_idx]) ** 2, dim=1)
            sub_d2 = torch.minimum(sub_d2, d2)
        centroids = subset.index_select(0, final_idx)

    # Pad if undersampled (possible when sum_d2 hit zero early).
    if centroids.shape[0] < k:
        deficit = k - int(centroids.shape[0])
        extra_idx = torch.randint(n, (deficit,), generator=gen, device=rows.device)
        centroids = torch.cat([centroids, rows.index_select(0, extra_idx)], dim=0)

    return centroids[:k]


def _kmeans_pp_init_numpy(rows, k: int, seed: int | None = None):
    import numpy as np

    n, d = rows.shape
    if k >= n:
        return rows.copy()
    centroids = np.empty((k, d), dtype=np.float32)
    rng = (
        np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        if seed is not None
        else np.random.default_rng()
    )
    centroids[0] = rows[rng.integers(n)]
    min_d2 = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        diff = rows - centroids[i - 1]
        d2 = np.sum(diff * diff, axis=1, dtype=np.float64)
        np.minimum(min_d2, d2, out=min_d2)
        total = float(min_d2.sum())
        if not math.isfinite(total) or total <= 0:
            idx = int(rng.integers(n))
        else:
            probs = min_d2 / total
            idx = int(rng.choice(n, p=probs))
        centroids[i] = rows[idx]
    return centroids

def _numpy_assign(vectors, codebook, chunk_size: int = 65536):
    import numpy as np

    rows = np.asarray(vectors, dtype=np.float32)
    centroids = np.asarray(codebook, dtype=np.float32)
    indices = np.empty(rows.shape[0], dtype=np.int64)
    total = 0.0
    width = rows.shape[1]
    
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    c_norm_sq = np.sum(centroids * centroids, axis=1)

    for start in range(0, rows.shape[0], chunk_size):
        end = min(start + chunk_size, rows.shape[0])
        chunk = rows[start:end]
        r_norm_sq = np.sum(chunk * chunk, axis=1)
        
        # GEMM for the cross term
        # dists = r_norm_sq[:, None] + c_norm_sq[None, :] - 2 * (chunk @ centroids.T)
        dists = r_norm_sq[:, None] + c_norm_sq[None, :] - 2 * np.dot(chunk, centroids.T)
        
        chosen = np.argmin(dists, axis=1)
        indices[start:end] = chosen
        total += float(dists[np.arange(chosen.shape[0]), chosen].sum())

    return indices, total / (rows.shape[0] * width)


def _torch_assign(vectors, codebook, device: str, chunk_size: int = 65536):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    resolved = _resolve_torch_device(device)
    # Detect if we can use half precision (FP16 is generally safe for distance ranking)
    use_half = resolved.type == "cuda"
    dtype = torch.float16 if use_half else torch.float32

    rows = _torch_float32_matrix(vectors, device).to(dtype)
    centroids = _torch_float32_matrix(codebook, device).to(dtype)
    
    indices_parts = []
    total = 0.0
    width = int(rows.shape[1])
    k = int(centroids.shape[0])
    
    # Pre-calculate squared norms for centroids: ||b||^2
    c_norm_sq = torch.sum(centroids.to(torch.float32) * centroids.to(torch.float32), dim=1, keepdim=True).T.to(dtype)
    
    effective_chunk = max(256, min(chunk_size, (1 << 28) // max(k, 1)))

    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), effective_chunk):
            end = min(start + effective_chunk, int(rows.shape[0]))
            chunk = rows[start:end]
            
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
            r_norm_sq = torch.sum(chunk.to(torch.float32) * chunk.to(torch.float32), dim=1, keepdim=True).to(dtype)
            
            dists = torch.addmm(
                (r_norm_sq + c_norm_sq),
                chunk,
                centroids.T,
                alpha=-2.0,
                beta=1.0
            )
            
            chosen = torch.argmin(dists, dim=1)
            indices_parts.append(chosen.detach().cpu())
            
            # Accumulate error in float32 for precision
            total += float(
                dists[torch.arange(chosen.shape[0], device=rows.device), chosen]
                .to(torch.float32)
                .sum()
                .detach()
                .cpu()
                .item()
            )

    indices = (
        torch.cat(indices_parts).to(dtype=torch.int64)
        if indices_parts
        else torch.empty(0, dtype=torch.int64)
    )
    return indices, total / (int(rows.shape[0]) * width)

def _learn_codebook_numpy(
    vectors, codebook_size: int, iterations: int, seed: int | None = None,
    initial_codebook=None,
):
    import numpy as np

    rows = np.asarray(vectors, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("NumPy VQ expects a 2D vector matrix")
    if rows.shape[0] == 0:
        raise ValueError("at least one vector is required")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    n = rows.shape[0]
    k = min(codebook_size, n)
    effective_iters = min(iterations, 3) if k >= int(n * 0.9) else iterations
    if initial_codebook is not None:
        codebook = np.asarray(initial_codebook, dtype=np.float32)[:k].copy()
    elif k == 1:
        codebook = rows[[n // 2]].copy()
    else:
        codebook = _kmeans_pp_init_numpy(rows, k, seed=seed)

    for _ in range(effective_iters):
        _check_ram_cap()
        indices, _ = _numpy_assign(rows, codebook)
        sums = np.zeros_like(codebook)
        counts = np.bincount(indices, minlength=k).astype(np.float32)
        np.add.at(sums, indices, rows)
        nonzero = counts > 0
        codebook[nonzero] = sums[nonzero] / counts[nonzero, None]

    indices, mse = _numpy_assign(rows, codebook)
    return codebook, indices, float(mse)


def _learn_codebook_torch(
    vectors,
    codebook_size: int,
    iterations: int,
    device: str,
    vector_weights=None,
    seed: int | None = None,
    initial_codebook=None,
):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    rows = _torch_float32_matrix(vectors, device)
    if rows.shape[0] == 0:
        raise ValueError("at least one vector is required")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    n = int(rows.shape[0])
    k = min(int(codebook_size), n)
    # When k >= n/2 each centroid has ~1 sample on average; centroids barely move after init.
    effective_iters = min(iterations, 3) if k >= int(n * 0.9) else iterations
    with torch.no_grad():
        if initial_codebook is not None:
            codebook = _torch_float32_matrix(initial_codebook, str(rows.device))[:k].clone()
        elif k == 1:
            codebook = rows[[n // 2]].clone()
        else:
            codebook = _kmeans_pp_init_torch(rows, k, seed=seed)

        sums = torch.zeros_like(codebook)
        counts = torch.zeros(k, dtype=torch.float32, device=rows.device)

        for _ in range(effective_iters):
            _check_ram_cap()
            if vector_weights is not None:
                W = torch.as_tensor(
                    vector_weights, dtype=torch.float32, device=rows.device
                )
                weighted_rows = rows * torch.sqrt(W)
                weighted_cb = codebook * torch.sqrt(W)
                indices, _ = _torch_assign(weighted_rows, weighted_cb, str(rows.device))
            else:
                indices, _ = _torch_assign(rows, codebook, str(rows.device))
            
            chosen = indices.to(device=rows.device, dtype=torch.long)
            
            # Reuse buffers
            sums.zero_()
            counts.zero_()
            
            sums.index_add_(0, chosen, rows)
            counts.index_put_((chosen,), torch.ones(len(chosen), device=rows.device), accumulate=True)
            
            nonzero = counts > 0
            codebook[nonzero] = sums[nonzero] / counts[nonzero, None]

        indices, mse = _torch_assign(rows, codebook, str(rows.device))
    return codebook.detach().cpu(), indices, float(mse)


def learn_codebook_auto(
    vectors,
    codebook_size: int,
    iterations: int,
    backend: str,
    device: str = "cpu",
    vector_weights=None,
    seed: int | None = None,
    initial_codebook=None,
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
        )
    if not _is_numpy_array(vectors):
        raise RuntimeError("NumPy backend requires NumPy array tensors")
    return _learn_codebook_numpy(vectors, codebook_size, iterations, seed=seed,
                                 initial_codebook=initial_codebook)


def quantize_vectors_auto(vectors, codebook, backend: str, device: str = "cpu"):
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if backend == "torch":
        return _torch_assign(vectors, codebook, device)
    if not _is_numpy_array(vectors):
        raise RuntimeError("NumPy backend requires NumPy array tensors")
    return _numpy_assign(vectors, codebook)
