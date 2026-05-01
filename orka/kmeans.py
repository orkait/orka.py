"""K-means codebook learning, nearest-centroid assignment, codebook caching, and
vector helpers used by the RVQ pipeline.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from orka.core import _is_numpy_array, _is_torch_tensor


def _codebook_cache_key(parts: Sequence[object]) -> str:
    import hashlib

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()

def _codebook_cache_load(cache_dir: Path | None, key: str):
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.npy"
    if not path.exists():
        return None
    try:
        import numpy as np

        return np.load(str(path), allow_pickle=False)
    except Exception:
        return None


def _codebook_cache_save(cache_dir: Path | None, key: str, codebook) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npy"
    import numpy as np

    if _is_torch_tensor(codebook):
        cb_np = codebook.detach().cpu().to(dtype=__import__("torch").float32).numpy()
    elif _is_numpy_array(codebook):
        cb_np = np.asarray(codebook, dtype=np.float32)
    else:
        cb_np = np.asarray([list(row) for row in codebook], dtype=np.float32)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, cb_np, allow_pickle=False)
    tmp.replace(path)

def _kmeans_pp_init_torch(
    rows, k: int, seed: int | None = None, oversample_factor: float = 2.0
):
    import torch

    n = int(rows.shape[0])
    d = int(rows.shape[1])
    if k >= n:
        return rows.clone()
    gen = torch.Generator(device=rows.device)
    if seed is not None:
        gen.manual_seed(int(seed) & ((1 << 63) - 1))

    # K-Means|| (Scalable K-Means++)
    first = int(torch.randint(n, (1,), generator=gen, device=rows.device).item())
    centroids = [rows[first]]
    min_d2 = torch.sum((rows - rows[first]) ** 2, dim=1)

    # We sample ~ l points per step. l = oversample_factor * k
    # We do this log(n) times, but usually a small constant like 5 is enough
    for _ in range(5):
        if len(centroids) >= k:
            break
        sum_d2 = min_d2.sum().item()
        if sum_d2 == 0:
            break
        probs = min_d2 / sum_d2
        # Sample l points
        l = int(oversample_factor * k)
        rand_vals = torch.rand(n, generator=gen, device=rows.device)
        chosen = torch.where(rand_vals < probs * l)[0]

        if chosen.numel() == 0:
            break

        new_centers = rows[chosen]
        for c in new_centers:
            centroids.append(c)
            
        # Update min_d2 efficiently using GEMM
        # Pre-calculate squared norms for new centers
        c_norm_sq = torch.sum(new_centers * new_centers, dim=1, keepdim=True).T
        
        batch_size = max(1024, (1 << 28) // max(int(new_centers.shape[0]), 1))
        for i in range(0, n, batch_size):
            batch_rows = rows[i : i + batch_size]
            r_norm_sq = torch.sum(batch_rows * batch_rows, dim=1, keepdim=True)
            
            dists = torch.addmm(
                (r_norm_sq + c_norm_sq),
                batch_rows,
                new_centers.T,
                alpha=-2.0,
                beta=1.0
            )
            
            min_d2[i : i + batch_size] = torch.minimum(
                min_d2[i : i + batch_size], dists.min(dim=1)[0]
            )
        del dists, batch_rows, new_centers


    centroids = torch.stack(centroids)
    if centroids.shape[0] > k:
        # If we oversampled, run K-Means++ on the sampled set to reduce to k
        subset = centroids
        final_centers = [subset[0]]
        sub_d2 = torch.sum((subset - subset[0]) ** 2, dim=1)
        for _ in range(1, k):
            sum_d2 = sub_d2.sum().item()
            if sum_d2 == 0:
                break
            probs = sub_d2 / sum_d2
            cumprobs = torch.cumsum(probs, dim=0)
            r = torch.rand(1, generator=gen, device=rows.device).item()
            chosen_idx = int(torch.searchsorted(cumprobs, r).item())
            chosen_idx = min(chosen_idx, subset.shape[0] - 1)
            final_centers.append(subset[chosen_idx])
            d2 = torch.sum((subset - subset[chosen_idx]) ** 2, dim=1)
            sub_d2 = torch.minimum(sub_d2, d2)
        centroids = torch.stack(final_centers)

    # Pad if we have less than k
    while centroids.shape[0] < k:
        centroids = torch.cat(
            [centroids, rows[torch.randint(n, (1,), generator=gen, device=rows.device)]]
        )

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

def _sample_vector_rows(vectors, sample_vectors: int | None):
    if sample_vectors is None or sample_vectors <= 0 or sample_vectors >= len(vectors):
        return vectors
    if _is_torch_tensor(vectors):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch vector sampling requires torch") from exc
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
        return vectors.index_select(0, positions)
    if hasattr(vectors, "shape") and hasattr(vectors, "__getitem__"):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy vector sampling requires numpy") from exc
        positions = np.linspace(0, len(vectors) - 1, sample_vectors, dtype=np.int64)
        return vectors[positions]

    if sample_vectors == 1:
        return [vectors[len(vectors) // 2]]
    last = len(vectors) - 1
    return [
        vectors[round(i * last / (sample_vectors - 1))] for i in range(sample_vectors)
    ]


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


def _torch_float32_matrix(values, device: str):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch backend requires torch") from exc

    resolved = _resolve_torch_device(device)
    if _is_torch_tensor(values):
        rows = values.detach().to(device=resolved, dtype=torch.float32)
    else:
        rows = torch.as_tensor(values, dtype=torch.float32, device=resolved)
    if rows.ndim != 2:
        raise ValueError("torch VQ expects a 2D vector matrix")
    return rows.contiguous()


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

def _concat_vector_parts(parts: Sequence[object]):
    if not parts:
        raise ValueError("cannot concatenate empty vector group")
    if _is_torch_tensor(parts[0]):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch vector concatenation requires torch") from exc
        return torch.cat(list(parts), dim=0)
    if _is_numpy_array(parts[0]):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy vector concatenation requires numpy") from exc
        return np.concatenate(parts, axis=0)

    out = []
    for part in parts:
        out.extend(part)
    return out

def _decode_to_vectors_format(
    vectors_template, codebook, indices, backend: str, device: str
):
    if _is_torch_tensor(vectors_template):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch decode requires torch") from exc
        cb = _torch_float32_matrix(codebook, str(vectors_template.device))
        if _is_torch_tensor(indices):
            idx = indices.detach().to(device=cb.device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, dtype=torch.long, device=cb.device)
        return cb[idx]
    if _is_numpy_array(vectors_template):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy decode requires numpy") from exc
        cb = np.asarray(codebook, dtype=np.float32)
        idx = np.asarray(indices, dtype=np.int64)
        return cb[idx]
    return [list(codebook[int(i)]) for i in indices]


def _vectors_subtract(a, b):
    if _is_torch_tensor(a) or _is_torch_tensor(b):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch subtract requires torch") from exc
        ta = a if _is_torch_tensor(a) else torch.as_tensor(a, dtype=torch.float32)
        tb = (
            b
            if _is_torch_tensor(b)
            else torch.as_tensor(b, dtype=torch.float32, device=ta.device)
        )
        if tb.device != ta.device:
            tb = tb.to(ta.device)
        return ta - tb
    if _is_numpy_array(a) or _is_numpy_array(b):
        import numpy as np

        return np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return [[float(x) - float(y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _decode_vectors_to_flat(vectors, codebook, indices, backend: str):
    if backend == "torch":
        centroids = _torch_float32_matrix(codebook, "auto")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch decode requires torch") from exc
        assigned = (
            indices.detach().to(device=centroids.device, dtype=torch.long)
            if _is_torch_tensor(indices)
            else torch.as_tensor(indices, dtype=torch.long, device=centroids.device)
        )
        return centroids[assigned].reshape(-1).detach().cpu()
    if backend in {"auto", "numpy"} and _is_numpy_array(vectors):
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("NumPy decode requires numpy") from exc
        centroids = np.asarray(codebook, dtype=np.float32)
        assigned = np.asarray(indices, dtype=np.int64)
        return centroids[assigned].reshape(-1)
    if backend == "numpy":
        raise RuntimeError("NumPy backend requires NumPy array tensors")

    decoded = []
    for index in indices:
        decoded.extend(float(v) for v in codebook[int(index)])
    return decoded
