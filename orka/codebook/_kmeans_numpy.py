"""K-means numpy backend: scalable ++ init, assign, Lloyd iterations."""
from __future__ import annotations

import math
from typing import Sequence

from orka._runtime import _resolve_torch_device, _check_ram_cap
from orka.core._tensor import _is_numpy_array, _is_torch_tensor, _torch_float32_matrix


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


