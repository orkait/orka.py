"""Triton VQ-GEMM kernel: fused VQ decode + matmul.

Operation: y = x @ W.T  where W is stored as RVQ codebooks + indices.

Kernel strategy: tile over (M output features, N tokens). For each K-tile of
BLOCK_K columns (BLOCK_K a multiple of B=32 so tl.dot's inner dim >= 16):
  1. Vectorized 2D gather of W_vq tile [BLOCK_M, BLOCK_K]:
       group_of_col = col // G ; within = col % G
       flat_group   = row * GPR + group_of_col
       idx_vals     = indices[flat_group]            (2D load)
       w_tile      += codebook[idx_vals*G + within]  (2D gather, per stage)
  2. Apply block scales [BLOCK_M, BLOCK_K]
  3. acc += x_tile [BLOCK_N, BLOCK_K] @ w_tile.T [BLOCK_K, BLOCK_M]

Salient + outlier corrections applied as sparse matmul after the kernel.
All loads are 1D/2D (Triton 3.x rejects 3D pointer loads).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Decode-optimized kernel (N=1 autoregressive hot path)
# ---------------------------------------------------------------------------
# Decode is HBM-bandwidth-bound. The minimum read is the indices; the codebook
# (192KB for 3 stages) is L2-resident and x (2KB) lives in registers. So we do
# the codebook[idx].x_g dot INLINE rather than materializing a per-group LUT (a
# 6MB LUT would exceed L2 and get re-read from HBM, the regression we hit).
#
# Per output row m: y[m] = sum_g scale[m,b(g)] * sum_s codebook_s[idx_s[m,g]].x_g
# HBM traffic = indices only (~M*(K/G)*S*2 B) vs dense M*K*2 B -> ~2.7x less.

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64},  num_warps=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 512}, num_warps=8),
    ],
    key=["M", "GPR", "N_STAGES"],
)
@triton.jit
def _vq_decode_kernel(
    x_ptr,                                  # [K] fp16 input (single token)
    idx0_ptr, idx1_ptr, idx2_ptr,           # [M * GPR] int16 per stage
    cb0_ptr, cb1_ptr, cb2_ptr,              # [cb_size * G] fp16 per stage
    scale_ptr,                              # [M * BPR] fp16
    y_ptr,                                  # [M] fp32 output
    M, GPR, BPR,
    G: tl.constexpr,                        # group_size = 8
    GROUPS_PER_BLOCK: tl.constexpr,         # B // G
    N_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_ids < M
    e = tl.arange(0, G)                      # [G]
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for g in tl.range(0, GPR):
        # x_g [G] - same for every row, hot in L1/registers
        x_g = tl.load(x_ptr + g * G + e).to(tl.float32)          # [G]

        b = g // GROUPS_PER_BLOCK
        scale = tl.load(scale_ptr + m_ids * BPR + b, mask=m_mask, other=0.0).to(tl.float32)

        # Stage 0: codebook[idx].x_g  (codebook gather from L2)
        idx0 = tl.load(idx0_ptr + m_ids * GPR + g, mask=m_mask, other=0).to(tl.int32)
        cbe0 = tl.load(cb0_ptr + idx0[:, None] * G + e[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        dot = tl.sum(cbe0 * x_g[None, :], axis=1)                # [BLOCK_M]

        if N_STAGES >= 2:
            idx1 = tl.load(idx1_ptr + m_ids * GPR + g, mask=m_mask, other=0).to(tl.int32)
            cbe1 = tl.load(cb1_ptr + idx1[:, None] * G + e[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
            dot += tl.sum(cbe1 * x_g[None, :], axis=1)

        if N_STAGES >= 3:
            idx2 = tl.load(idx2_ptr + m_ids * GPR + g, mask=m_mask, other=0).to(tl.int32)
            cbe2 = tl.load(cb2_ptr + idx2[:, None] * G + e[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
            dot += tl.sum(cbe2 * x_g[None, :], axis=1)

        acc += scale * dot

    tl.store(y_ptr + m_ids, acc, mask=m_mask)


def _vq_decode_n1(layer, x_2d: torch.Tensor) -> torch.Tensor:
    """Inline-dot decode for N=1. x_2d: [1, K] -> y: [1, M] fp16."""
    G = layer.group_size
    B = layer.block_size
    K = layer.in_features
    M = layer.out_features
    GPR = K // G
    BPR = K // B

    x_flat = x_2d.view(K).contiguous()
    y = torch.empty(M, dtype=torch.float32, device=x_2d.device)

    def _buf(s, attr):
        idx = s if s < layer.n_stages else 0
        return getattr(layer, f"{attr}_{idx}")

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _vq_decode_kernel[grid](
        x_flat,
        _buf(0, "indices"), _buf(1, "indices"), _buf(2, "indices"),
        _buf(0, "codebook"), _buf(1, "codebook"), _buf(2, "codebook"),
        layer.scales,
        y,
        M, GPR, BPR,
        G, B // G, layer.n_stages,
    )
    return y.view(1, M).to(torch.float16)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K", "N_STAGES"],
)
@triton.jit
def _vq_gemm_kernel(
    x_ptr, x_stride_n, x_stride_k,
    y_ptr, y_stride_n, y_stride_m,
    # indices [out_features * GPR] int16 per stage, row-major: row * GPR + group
    idx0_ptr, idx1_ptr, idx2_ptr,
    # codebooks [CB_SIZE * G] fp16 per stage
    cb0_ptr, cb1_ptr, cb2_ptr,
    # scales [out_features * BPR] fp16, row-major: row * BPR + block
    scale_ptr,
    M, N, K,
    GPR,     # K // G  (groups per row)
    BPR,     # K // B  (scale blocks per row)
    G: tl.constexpr,          # group_size = 8
    B: tl.constexpr,          # block_size = 32
    N_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_start = tl.program_id(0) * BLOCK_M
    n_start = tl.program_id(1) * BLOCK_N

    m_ids = m_start + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    n_ids = n_start + tl.arange(0, BLOCK_N)    # [BLOCK_N]
    m_mask = m_ids < M
    n_mask = n_ids < N

    acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        col_ids = k_start + tl.arange(0, BLOCK_K)      # [BLOCK_K]
        k_mask = col_ids < K
        group_of_col = col_ids // G                     # [BLOCK_K]
        within = col_ids % G                            # [BLOCK_K]
        block_of_col = col_ids // B                      # [BLOCK_K]

        # flat group index per (row, col): [BLOCK_M, BLOCK_K]
        flat_group = m_ids[:, None] * GPR + group_of_col[None, :]

        # Stage 0 (always present)
        idx0 = tl.load(idx0_ptr + flat_group, mask=m_mask[:, None] & k_mask[None, :], other=0).to(tl.int32)
        cb_off0 = idx0 * G + within[None, :]
        w_tile = tl.load(cb0_ptr + cb_off0, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        if N_STAGES >= 2:
            idx1 = tl.load(idx1_ptr + flat_group, mask=m_mask[:, None] & k_mask[None, :], other=0).to(tl.int32)
            cb_off1 = idx1 * G + within[None, :]
            w_tile = w_tile + tl.load(cb1_ptr + cb_off1, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        if N_STAGES >= 3:
            idx2 = tl.load(idx2_ptr + flat_group, mask=m_mask[:, None] & k_mask[None, :], other=0).to(tl.int32)
            cb_off2 = idx2 * G + within[None, :]
            w_tile = w_tile + tl.load(cb2_ptr + cb_off2, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Block scales [BLOCK_M, BLOCK_K]
        scale_off = m_ids[:, None] * BPR + block_of_col[None, :]
        scales = tl.load(scale_ptr + scale_off, mask=m_mask[:, None] & k_mask[None, :], other=1.0).to(tl.float32)
        w_tile = w_tile * scales

        # x tile [BLOCK_N, BLOCK_K]
        x_tile = tl.load(
            x_ptr + n_ids[:, None] * x_stride_n + col_ids[None, :] * x_stride_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # acc [BLOCK_N, BLOCK_M] += x_tile [BLOCK_N, BLOCK_K] @ w_tile.T [BLOCK_K, BLOCK_M]
        acc = tl.dot(x_tile, tl.trans(w_tile), acc)

    tl.store(
        y_ptr + n_ids[:, None] * y_stride_n + m_ids[None, :] * y_stride_m,
        acc.to(tl.float16),
        mask=n_mask[:, None] & m_mask[None, :],
    )


def vq_linear_forward(layer, x: torch.Tensor) -> torch.Tensor:
    """Run VQLinear forward: Triton VQ-GEMM + sparse correction.

    x: [..., in_features]
    returns: [..., out_features]
    """
    G = layer.group_size
    B = layer.block_size
    K = layer.in_features
    M = layer.out_features

    assert G == 8, f"kernel requires group_size=8, got {G}"
    assert B == 32, f"kernel requires block_size=32, got {B}"
    assert K % G == 0, f"in_features={K} must be divisible by G={G}"
    assert K % B == 0, f"in_features={K} must be divisible by B={B}"

    orig_shape = x.shape
    x_2d = x.reshape(-1, K).contiguous().to(torch.float16)
    N = x_2d.shape[0]

    # Decode hot path: N=1 uses the LUT kernel (codebook.x precomputed once,
    # then per-row table lookup) - much cheaper than the tiled gather+GEMM.
    if N == 1:
        y = _vq_decode_n1(layer, x_2d)
        if layer.corr_indices.numel() > 0:
            sp = layer._correction_sparse()
            y = y + torch.sparse.mm(sp, x_2d.float().T.contiguous()).T.to(torch.float16)
        if layer.bias is not None:
            y = y + layer.bias
        return y.reshape(*orig_shape[:-1], M).to(x.dtype)

    y = torch.empty((N, M), dtype=torch.float16, device=x_2d.device)

    GPR = K // G
    BPR = K // B

    def _buf(s, attr):
        # Pass tensors directly so Triton infers pointer type. Unused stages
        # reuse stage-0's tensor (never read - guarded by N_STAGES constexpr).
        idx = s if s < layer.n_stages else 0
        return getattr(layer, f"{attr}_{idx}")

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731

    _vq_gemm_kernel[grid](
        x_2d, x_2d.stride(0), x_2d.stride(1),
        y, y.stride(0), y.stride(1),
        _buf(0, "indices"), _buf(1, "indices"), _buf(2, "indices"),
        _buf(0, "codebook"), _buf(1, "codebook"), _buf(2, "codebook"),
        layer.scales,
        M, N, K, GPR, BPR,
        G, B, layer.n_stages,
    )

    # Sparse correction: W_correction [M, K] -> y += x @ W_correction.T
    # The coalesced sparse tensor is built once and cached on the layer; only
    # the sparse.mm runs per forward.
    if layer.corr_indices.numel() > 0:
        sp = layer._correction_sparse()
        correction = torch.sparse.mm(sp, x_2d.float().T.contiguous()).T.to(torch.float16)
        y = y + correction

    if layer.bias is not None:
        y = y + layer.bias

    return y.reshape(*orig_shape[:-1], M).to(x.dtype)
