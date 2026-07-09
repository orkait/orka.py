"""K-means torch backend: ++ init, GEMM-based assign, Lloyd iterations."""
from __future__ import annotations

import math
from collections.abc import Sequence

from orka._runtime import _check_ram_cap, _resolve_torch_device
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



def _torch_assign(vectors, codebook, device: str, chunk_size: int = 65536, r_norm_sq=None, vector_weights=None, keep_device=False):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc
    from orka.inference._assign_kernel import assign as _fused_assign

    resolved = _resolve_torch_device(device)
    rows = _torch_float32_matrix(vectors, device).float()
    centroids = _torch_float32_matrix(codebook, device).float()

    # Hessian / importance weighting: weighted L2 == plain L2 on sqrt(w)-scaled vectors.
    if vector_weights is not None:
        sqrt_W = torch.sqrt(torch.as_tensor(vector_weights, dtype=torch.float32, device=resolved))
        rows = rows * sqrt_W
        centroids = centroids * sqrt_W

    width = int(rows.shape[1])
    with torch.no_grad():
        # Fused dist+argmin Triton kernel on CUDA (~4x the chunked-addmm path, never
        # materializes the [N,k] matrix so it can't OOM at large k); addmm fallback on
        # CPU. fp32 throughout, so the nearest centroid is exact rather than fp16-ranked
        # (the ||v||^2 term is constant across centroids and drops out of the argmin).
        idx = _fused_assign(rows, centroids)                       # int64 [N]
        sel = centroids.index_select(0, idx)                       # [N, d]
        mse = ((rows - sel) ** 2).sum() / (rows.shape[0] * width)
    idx = idx.to(torch.int64)
    # Lloyd iterations consume idx on-device; only the final/external call needs the
    # host copy. Skipping the per-iter .cpu() drops a GPU<->host roundtrip each loop.
    return (idx if keep_device else idx.cpu()), float(mse.item())


def _det_segment_sum(keys, values, k):
    """Deterministic sum of `values` ([N] or [N, d]) grouped by integer `keys` in
    [0, k). Sorts by key then segment-reduces, so the float accumulation order is
    fixed - unlike atomic index_add/index_put on CUDA, which is the only source of
    byte non-determinism in the pack. torch.segment_reduce does the per-cluster sum
    in one fused kernel (56x faster here than a manual cumsum-boundary-diff, whose
    cumsum along dim 0 was a strided, memory-bound scan); bincount gives integer-
    exact, order-independent segment lengths."""
    import torch

    tail = tuple(values.shape[1:])
    if keys.numel() == 0:
        return torch.zeros((k,) + tail, device=values.device, dtype=values.dtype)
    order = torch.argsort(keys, stable=True)
    lengths = torch.bincount(keys, minlength=k)
    return torch.segment_reduce(
        values.index_select(0, order), "sum", lengths=lengths, axis=0, unsafe=True
    )


def _faiss_kmeans_enabled() -> bool:
    """Opt-in via ORKA_KMEANS_FAISS={1,true,yes}. Off by default so the pack stays
    byte-reproducible regardless of whether faiss happens to be installed; enabling
    it swaps the unweighted CUDA k-means for faiss's GPU Lloyd (~2x faster here, same
    reconstruction MSE, byte-deterministic per seed)."""
    from orka import config

    if not config.kmeans_faiss_enabled():
        return False
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def _learn_codebook_faiss(rows, k: int, iterations: int, device: str, seed: int | None):
    """faiss GPU Lloyd for the unweighted path. Returns orka's
    (codebook_cpu, indices, mse); nearest-centroid indices + mse are recomputed with
    orka's fused assign so the encoding matches the torch path exactly."""
    import faiss
    import numpy as np
    import torch

    resolved = _resolve_torch_device(device)
    rows_f = _torch_float32_matrix(rows, device).float()
    n, d = int(rows_f.shape[0]), int(rows_f.shape[1])
    k = min(int(k), n)
    xb = np.ascontiguousarray(rows_f.detach().cpu().numpy(), dtype="float32")
    km = faiss.Kmeans(
        d, k, niter=int(iterations), gpu=True,
        seed=int(seed or 0) & 0x7FFFFFFF,
        max_points_per_centroid=n,  # use all points (no faiss subsampling)
    )
    km.train(xb)
    codebook = torch.from_numpy(np.ascontiguousarray(km.centroids)).to(resolved).float()
    indices, mse = _torch_assign(rows_f, codebook, str(resolved))
    return codebook.detach().cpu(), indices, float(mse)


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

    resolved = _resolve_torch_device(device)

    # Opt-in faiss GPU Lloyd: unweighted CUDA path only (faiss has no per-dimension
    # or per-sample weighting, and no warm-start). ~2x the torch path at equal MSE;
    # falls through to the deterministic torch path on any failure.
    if (
        resolved.type == "cuda"
        and k > 1
        and vector_weights is None
        and sample_weights is None
        and initial_codebook is None
        and _faiss_kmeans_enabled()
    ):
        try:
            return _learn_codebook_faiss(rows, k, effective_iters, device, seed)
        except Exception:
            pass  # fall back to the torch path below

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

        from tqdm import tqdm
        pbar = tqdm(range(effective_iters), desc="      K-Means Iterations", leave=False)
        report_interval = max(1, effective_iters // 5)
        for iter_i in pbar:
            if (iter_i + 1) % report_interval == 0 or iter_i == 0 or iter_i == effective_iters - 1:
                print(f"      [Lloyd] Iteration {iter_i + 1}/{effective_iters}", flush=True)
            _check_ram_cap()

            old_codebook = codebook.clone()

            indices, _ = _torch_assign(
                rows_dtype,
                codebook,
                str(rows.device),
                r_norm_sq=r_norm_sq if vector_weights is None else None,
                vector_weights=vector_weights,
                keep_device=True,
            )

            chosen = indices.to(dtype=torch.long)

            # Deterministic centroid sums/counts: atomic index_add/index_put use a
            # non-deterministic float-accumulation order on CUDA (the sole source of
            # byte-non-determinism in the pack). A sorted segment-reduce sums in
            # a fixed order -> reproducible codebooks. Unweighted counts are an exact
            # integer bincount (no segment-sum needed).
            k_cb = codebook.shape[0]
            sums = _det_segment_sum(chosen, rows if sw_t is None else rows * sw_t[:, None], k_cb)
            counts = (torch.bincount(chosen, minlength=k_cb).to(rows.dtype)
                      if sw_t is None else _det_segment_sum(chosen, sw_t, k_cb))

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


