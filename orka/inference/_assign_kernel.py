"""Fused nearest-centroid (argmin) Triton kernel.

The assign step (argmin_k ||v - c_k||^2 over a codebook) is the dominant op in
both packing (k-means Lloyd) and QAT. The torch path materializes a [chunk, k]
distance matrix in HBM and re-reads it to argmin - memory-bound. This kernel
computes each distance tile in registers and keeps a running per-row min+argmin,
so the [chunk, k] matrix never touches HBM. Distance uses ||c||^2 - 2 v.c (the
||v||^2 term is constant across centroids, so it drops out of the argmin), giving
the same result as torch.cdist / addmm.

Falls back to the torch addmm path when Triton is unavailable or the group dim is
unusual. group dim d (=group_size, typ. 8) is small, so the v.c dot is unrolled
manually (tl.dot needs a contraction dim >= 16).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton optional
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _argmin_kernel(
        v_ptr, c_ptr, csq_ptr, out_ptr,
        N, K,
        D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)          # [BN]
        rmask = rows < N
        d_idx = tl.arange(0, D)                                # [D] (padded, multiple of 16)
        v = tl.load(v_ptr + rows[:, None] * D + d_idx[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)   # [BN, D]

        best_d = tl.full([BLOCK_N], float("inf"), tl.float32)
        best_i = tl.zeros([BLOCK_N], tl.int32)

        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)                    # [BK]
            kmask = kk < K
            c = tl.load(c_ptr + kk[:, None] * D + d_idx[None, :],
                        mask=kmask[:, None], other=0.0).to(tl.float32)   # [BK, D]
            csq = tl.load(csq_ptr + kk, mask=kmask, other=0.0).to(tl.float32)  # [BK]
            # v.c via tl.dot (D padded to a multiple of 16); fp32, no tf32 (exactness)
            vc = tl.dot(v, tl.trans(c), allow_tf32=False)     # [BN, BK]
            dist = csq[None, :] - 2.0 * vc                     # argmin-equiv (||v||^2 dropped)
            dist = tl.where(kmask[None, :], dist, float("inf"))
            tile_min = tl.min(dist, axis=1)                    # [BN]
            tile_arg = tl.argmin(dist, axis=1).to(tl.int32) + k0
            upd = tile_min < best_d
            best_i = tl.where(upd, tile_arg, best_i)
            best_d = tl.where(upd, tile_min, best_d)

        tl.store(out_ptr + rows, best_i.to(tl.int64), mask=rmask)


def triton_assign(vectors: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """argmin over codebook, fused. Returns int64 [N]. CUDA + Triton only.
    Pads the group dim to a multiple of 16 (tl.dot constraint); zero-pad changes
    neither v.c nor ||c||^2, so the argmin is identical."""
    N, d = vectors.shape
    K = cb.shape[0]
    csq = (cb.float() * cb.float()).sum(1).contiguous()
    D = ((d + 15) // 16) * 16
    if D != d:
        vectors = torch.nn.functional.pad(vectors.float(), (0, D - d))
        cb = torch.nn.functional.pad(cb.float(), (0, D - d))
    vectors = vectors.contiguous().float()
    cb = cb.contiguous().float()
    out = torch.empty(N, dtype=torch.int64, device=vectors.device)
    BLOCK_N, BLOCK_K = 64, 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _argmin_kernel[grid](vectors, cb, csq, out, N, K,
                         D=D, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out


def assign(vectors: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid index. Triton-fused on CUDA; torch addmm fallback otherwise."""
    if _HAVE_TRITON and vectors.is_cuda and vectors.shape[1] <= 16:
        return triton_assign(vectors, cb)
    csq = (cb * cb).sum(1)
    return torch.addmm(csq.unsqueeze(0), vectors, cb.t().contiguous(), beta=1.0, alpha=-2.0).argmin(1)
