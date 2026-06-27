"""K-means torch backend: ++ init, GEMM-based assign, Lloyd iterations."""
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
    total_t = torch.zeros((), dtype=torch.float32, device=resolved)
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
            indices_parts.append(chosen)                       # keep on GPU - no per-chunk .cpu()

            # Accumulate selected-distance error on-device in float32; one host
            # transfer at the end instead of a .item() sync every chunk.
            total_t += dists.gather(1, chosen.unsqueeze(1)).to(torch.float32).sum()

    indices = (
        torch.cat(indices_parts).to(dtype=torch.int64).cpu()   # single device->host transfer
        if indices_parts
        else torch.empty(0, dtype=torch.int64)
    )
    return indices, float(total_t.item()) / (int(rows.shape[0]) * width)


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


