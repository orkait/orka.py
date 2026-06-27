"""K-means: scalable ++ init (k-means||) + Lloyd iterations + nearest-centroid assign.

Backends: numpy (CPU) and torch (CPU/CUDA, GEMM-based distance).
"""

from __future__ import annotations

import math
from typing import Sequence

from orka._runtime import _resolve_torch_device, _check_ram_cap
from orka.core._tensor import _is_numpy_array, _is_torch_tensor, _torch_float32_matrix


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

    if k > 2048:
        perm = torch.randperm(n, generator=gen, device=rows.device)
        return rows[perm[:k]].clone()

    # Use FP16 on CUDA for 2× faster distance computation (ranking is identical)
    use_half = rows.device.type == "cuda"
    dist_dtype = torch.float16 if use_half else torch.float32
    rows_dist = rows.to(dist_dtype)

    first = int(torch.randint(n, (1,), generator=gen, device=rows.device).item())
    min_d2 = torch.sum((rows_dist - rows_dist[first]) ** 2, dim=1).float()
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

        new_centers = rows_dist[chosen].contiguous()
        candidate_chunks.append(rows[chosen].contiguous())
        total_candidates += int(new_centers.shape[0])

        # Update min_d2 with GEMM-form distance, chunking by ~256 MB matrix budget.
        c_norm_sq = torch.sum(new_centers * new_centers, dim=1, keepdim=True).T
        c_count = int(new_centers.shape[0])
        # Budget per chunk: 256 MB / (elem_size * c_count)
        elem_size = 2 if use_half else 4
        batch_size = max(256, min(65536, (1 << 28) // (elem_size * max(c_count, 1))))
        for i in range(0, n, batch_size):
            batch_rows = rows_dist[i : i + batch_size]
            r_norm_sq = torch.sum(batch_rows * batch_rows, dim=1, keepdim=True)
            dists = torch.addmm(
                (r_norm_sq + c_norm_sq),
                batch_rows,
                new_centers.T,
                alpha=-2.0,
                beta=1.0,
            )
            min_d2[i : i + batch_size] = torch.minimum(
                min_d2[i : i + batch_size], dists.min(dim=1)[0].float()
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


def _kmeans_parallel_init_numpy(rows, k: int, seed: int | None = None, oversample_factor: float = 2.0):
    """Scalable K-Means++ (k-means||) init for numpy.

    Same algorithm as the torch version: 5 rounds of proportional oversampling,
    candidate cap at 5*k, GEMM-based distance update, classic K-Means++ reduction
    on the candidate subset. O(n * rounds) distance work instead of O(n * k).
    """
    import numpy as np

    n, d = rows.shape
    if k >= n:
        return rows.copy()

    rng = (
        np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        if seed is not None
        else np.random.default_rng()
    )

    if k > 2048:
        idx = rng.choice(n, size=k, replace=False)
        return rows[idx].copy()

    first = int(rng.integers(n))
    min_d2 = np.sum((rows - rows[first]) ** 2, axis=1, dtype=np.float64)
    candidate_chunks = [rows[[first]]]
    total_candidates = 1
    candidate_cap = 5 * k

    rounds = 5
    per_round_cap = max(1, min(int(oversample_factor * k), n // 5))

    for _ in range(rounds):
        if total_candidates >= candidate_cap:
            break
        sum_d2 = float(min_d2.sum())
        if not math.isfinite(sum_d2) or sum_d2 <= 0:
            break
        probs = min_d2 / sum_d2
        rand_vals = rng.random(n)
        chosen = np.where(rand_vals < probs * per_round_cap)[0]
        if len(chosen) == 0:
            break
        if len(chosen) > per_round_cap:
            chosen = chosen[:per_round_cap]

        new_centers = rows[chosen]
        candidate_chunks.append(new_centers)
        total_candidates += len(new_centers)

        # Budget: keep distance matrix ≤ 64 MB (float32).
        c_norm_sq = np.sum(new_centers * new_centers, axis=1, dtype=np.float32)
        nc = max(len(chosen), 1)
        chunk_size = max(64, min(65536, (1 << 26) // (4 * nc)))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = rows[start:end]
            r_norm_sq = np.sum(batch * batch, axis=1, dtype=np.float32)
            dists = (r_norm_sq[:, None] + c_norm_sq[None, :]
                     - 2.0 * (batch @ new_centers.T)).astype(np.float32)
            np.minimum(min_d2[start:end], dists.min(axis=1), out=min_d2[start:end])

    centroids = np.concatenate(candidate_chunks, axis=0)

    if len(centroids) > k:
        sub_n = len(centroids)
        final_indices = [0]
        sub_d2 = np.sum((centroids - centroids[0]) ** 2, axis=1, dtype=np.float64)
        for j in range(1, k):
            sum_d2 = float(sub_d2.sum())
            if not math.isfinite(sum_d2) or sum_d2 <= 0:
                final_indices.append(j % sub_n)
                continue
            probs = sub_d2 / sum_d2
            chosen_idx = int(rng.choice(sub_n, p=probs))
            final_indices.append(chosen_idx)
            d2 = np.sum((centroids - centroids[chosen_idx]) ** 2, axis=1, dtype=np.float64)
            np.minimum(sub_d2, d2, out=sub_d2)
        centroids = centroids[final_indices]

    if len(centroids) < k:
        deficit = k - len(centroids)
        extra_idx = rng.integers(n, size=deficit)
        centroids = np.concatenate([centroids, rows[extra_idx]], axis=0)

    return centroids[:k].astype(np.float32).copy()

def _numpy_assign(vectors, codebook, chunk_size: int = 65536, r_norm_sq=None, vector_weights=None):
    import numpy as np

    rows = np.asarray(vectors, dtype=np.float32)
    centroids = np.asarray(codebook, dtype=np.float32)
    if vector_weights is not None:
        W = np.asarray(vector_weights, dtype=np.float32)
        sqrt_W = np.sqrt(W)
        rows = rows * sqrt_W
        centroids = centroids * sqrt_W
        r_norm_sq = np.sum(rows * rows, axis=1, dtype=np.float32)

    row_norms = None
    if r_norm_sq is not None:
        row_norms = np.asarray(r_norm_sq, dtype=np.float32).reshape(-1)
        if row_norms.shape[0] != rows.shape[0]:
            raise ValueError("r_norm_sq length must match vectors")
    indices = np.empty(rows.shape[0], dtype=np.int64)
    total = 0.0
    width = rows.shape[1]
    k = len(centroids)

    # Adaptive chunk: keep distance matrix ≤ 64 MB (float32, 4 bytes/elem).
    effective_chunk = max(64, min(chunk_size, (1 << 26) // max(k, 1)))

    c_norm_sq = np.sum(centroids * centroids, axis=1, dtype=np.float32)

    for start in range(0, rows.shape[0], effective_chunk):
        end = min(start + effective_chunk, rows.shape[0])
        chunk = rows[start:end]
        chunk_norm_sq = (
            np.sum(chunk * chunk, axis=1, dtype=np.float32)
            if row_norms is None
            else row_norms[start:end]
        )
        dists = (
            chunk_norm_sq[:, None]
            + c_norm_sq[None, :]
            - 2.0 * np.dot(chunk, centroids.T)
        )
        chosen = np.argmin(dists, axis=1)
        indices[start:end] = chosen
        total += float(dists[np.arange(chosen.shape[0]), chosen].sum())

    return indices, total / (rows.shape[0] * width)


def _numpy_centroid_sums(rows, indices, k: int, sample_weights=None):
    import numpy as np

    vectors = np.asarray(rows, dtype=np.float32)
    assignments = np.asarray(indices, dtype=np.int64)
    if assignments.shape[0] != vectors.shape[0]:
        raise ValueError("indices length must match rows")
    if assignments.size and (assignments.min() < 0 or assignments.max() >= k):
        raise IndexError("centroid index out of bounds")
    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if sw.shape[0] != vectors.shape[0]:
            raise ValueError("sample_weights length must match rows")
        vectors = vectors * sw[:, None]

    sums = np.empty((k, vectors.shape[1]), dtype=np.float32)
    for dim in range(vectors.shape[1]):
        sums[:, dim] = np.bincount(
            assignments,
            weights=vectors[:, dim],
            minlength=k,
        )[:k]
    return sums


def _torch_assign(vectors, codebook, device: str, chunk_size: int = 65536, r_norm_sq=None, vector_weights=None):
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

    if vector_weights is not None:
        W = torch.as_tensor(vector_weights, dtype=torch.float32, device=resolved)
        sqrt_W = torch.sqrt(W).to(dtype)
        rows = rows * sqrt_W
        centroids = centroids * sqrt_W
        r_norm_sq = torch.sum(rows.to(torch.float32) * rows.to(torch.float32), dim=1, keepdim=True).to(dtype)

    indices_parts = []
    total = 0.0
    width = int(rows.shape[1])
    k = int(centroids.shape[0])

    # Pre-calculate squared norms for centroids: ||b||^2
    c_norm_sq = torch.sum(centroids.to(torch.float32) * centroids.to(torch.float32), dim=1, keepdim=True).T.to(dtype)

    if r_norm_sq is None:
        r_norm_sq = torch.sum(rows.to(torch.float32) * rows.to(torch.float32), dim=1, keepdim=True).to(dtype)
    else:
        r_norm_sq = _torch_float32_matrix(r_norm_sq, device).to(dtype)

    effective_chunk = max(256, min(chunk_size, (1 << 28) // max(k, 1)))

    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), effective_chunk):
            end = min(start + effective_chunk, int(rows.shape[0]))
            chunk = rows[start:end]
            r_norm_sq_chunk = r_norm_sq[start:end]

            dists = torch.addmm(
                (r_norm_sq_chunk + c_norm_sq),
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
    initial_codebook=None, vector_weights=None, sample_weights=None,
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
        codebook = _kmeans_parallel_init_numpy(rows, k, seed=seed)

    r_norm_sq = np.sum(rows * rows, axis=1, dtype=np.float32)

    from tqdm import tqdm
    pbar = tqdm(range(effective_iters), desc="      K-Means Iterations", leave=False)
    report_interval = max(1, effective_iters // 5)
    for iter_i in pbar:
        if (iter_i + 1) % report_interval == 0 or iter_i == 0 or iter_i == effective_iters - 1:
            print(f"      [Lloyd] Iteration {iter_i + 1}/{effective_iters}", flush=True)
        _check_ram_cap()
        indices, _ = _numpy_assign(
            rows, codebook, r_norm_sq=r_norm_sq if vector_weights is None else None,
            vector_weights=vector_weights
        )
        sums = _numpy_centroid_sums(rows, indices, k, sample_weights=sample_weights)
        if sample_weights is None:
            counts = np.bincount(indices, minlength=k).astype(np.float32)
        else:
            counts = np.bincount(
                indices,
                weights=np.asarray(sample_weights, dtype=np.float32).reshape(-1),
                minlength=k,
            ).astype(np.float32)
        nonzero = counts > 0
        codebook[nonzero] = sums[nonzero] / counts[nonzero, None]

    indices, mse = _numpy_assign(
        rows, codebook, r_norm_sq=r_norm_sq if vector_weights is None else None,
        vector_weights=vector_weights
    )
    return codebook, indices, float(mse)


def _learn_codebook_torch(
    vectors,
    codebook_size: int,
    iterations: int,
    device: str,
    vector_weights=None,
    seed: int | None = None,
    initial_codebook=None,
    sample_weights=None,
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
    effective_iters = min(iterations, 3) if k >= int(n * 0.9) else iterations

    # Pre-resolve target dtype and pre-calculate row norms once
    resolved = _resolve_torch_device(device)
    use_half = resolved.type == "cuda"
    dtype = torch.float16 if use_half else torch.float32

    rows_dtype = rows.to(dtype)
    r_norm_sq = torch.sum(rows.to(torch.float32) * rows.to(torch.float32), dim=1, keepdim=True).to(dtype)
    sw_t = None
    if sample_weights is not None:
        sw_t = torch.as_tensor(sample_weights, dtype=torch.float32, device=rows.device).reshape(-1)
        if int(sw_t.shape[0]) != n:
            raise ValueError("sample_weights length must match rows")

    with torch.no_grad():
        if initial_codebook is not None:
            codebook = _torch_float32_matrix(initial_codebook, str(rows.device))[:k].clone()
        elif k == 1:
            codebook = rows[[n // 2]].clone()
        else:
            codebook = _kmeans_pp_init_torch(rows, k, seed=seed)

        sums = torch.zeros_like(codebook)
        counts = torch.zeros(k, dtype=torch.float32, device=rows.device)

        from tqdm import tqdm
        pbar = tqdm(range(effective_iters), desc="      K-Means Iterations", leave=False)
        report_interval = max(1, effective_iters // 5)
        for iter_i in pbar:
            if (iter_i + 1) % report_interval == 0 or iter_i == 0 or iter_i == effective_iters - 1:
                print(f"      [Lloyd] Iteration {iter_i + 1}/{effective_iters}", flush=True)
            _check_ram_cap()

            # Save old codebook to monitor convergence shift
            old_codebook = codebook.clone()

            indices, _ = _torch_assign(
                rows_dtype,
                codebook,
                str(rows.device),
                r_norm_sq=r_norm_sq if vector_weights is None else None,
                vector_weights=vector_weights,
            )

            chosen = indices.to(device=rows.device, dtype=torch.long)

            # Reuse buffers
            sums.zero_()
            counts.zero_()

            if sw_t is None:
                sums.index_add_(0, chosen, rows)
                counts.index_put_((chosen,), torch.ones(len(chosen), device=rows.device), accumulate=True)
            else:
                sums.index_add_(0, chosen, rows * sw_t[:, None])
                counts.index_put_((chosen,), sw_t, accumulate=True)

            nonzero = counts > 0
            codebook[nonzero] = sums[nonzero] / counts[nonzero, None]

            # Early stopping check: if centroids shift by less than 1e-5 relative to scale
            if iter_i > 0:
                max_diff = torch.max(torch.abs(codebook - old_codebook)).item()
                if max_diff < 1e-5:
                    print(f"      [Lloyd] Converged early at iteration {iter_i + 1} (max diff: {max_diff:.2e})", flush=True)
                    break

        indices, mse = _torch_assign(
            rows_dtype,
            codebook,
            str(rows.device),
            r_norm_sq=r_norm_sq if vector_weights is None else None,
            vector_weights=vector_weights,
        )
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
