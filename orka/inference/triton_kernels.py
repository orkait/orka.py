"""Triton VQ kernels - the FALLBACK backend.

Used when the CUDA extension (orka.inference.cuda_decode) is unavailable or the layer
is unsupported. The dispatcher that chooses between this and the CUDA backend lives in
orka.inference.dispatch; see orka/inference/__init__.py for the full dispatch map.

  _vq_decode_kernel  N=1 decode matvec
  _vq_gemm_kernel    N>1 tiled gather-GEMM
Both read indices/scales group-major or row-major via a GROUP_MAJOR constexpr.

GEMM strategy: tile over (M out-features, N tokens); per K-tile gather the W_vq tile
(codebook[indices] summed over stages), apply block scales, then acc += x_tile @
w_tile.T. All loads are 1D/2D (Triton 3.x rejects 3D pointer loads).
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
    GROUP_MAJOR: tl.constexpr,              # indices/scales laid out [GPR, M] / [BPR, M]
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
        # group-major [GPR,M]/[BPR,M] coalesces across rows; row-major is the legacy layout
        if GROUP_MAJOR:
            i_off = g * M + m_ids
            s_off = b * M + m_ids
        else:
            i_off = m_ids * GPR + g
            s_off = m_ids * BPR + b
        scale = tl.load(scale_ptr + s_off, mask=m_mask, other=0.0).to(tl.float32)

        # Stage 0: codebook[idx].x_g  (codebook gather from L2)
        idx0 = tl.load(idx0_ptr + i_off, mask=m_mask, other=0).to(tl.int32)
        cbe0 = tl.load(cb0_ptr + idx0[:, None] * G + e[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        dot = tl.sum(cbe0 * x_g[None, :], axis=1)                # [BLOCK_M]

        if N_STAGES >= 2:
            idx1 = tl.load(idx1_ptr + i_off, mask=m_mask, other=0).to(tl.int32)
            cbe1 = tl.load(cb1_ptr + idx1[:, None] * G + e[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
            dot += tl.sum(cbe1 * x_g[None, :], axis=1)

        if N_STAGES >= 3:
            idx2 = tl.load(idx2_ptr + i_off, mask=m_mask, other=0).to(tl.int32)
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
        GROUP_MAJOR=bool(getattr(layer, "_group_major", False)),
    )
    return y.view(1, M).to(torch.float16)


# ---------------------------------------------------------------------------
# Decode kernel reading bit-PLANED indices (N=1 hot path)
# ---------------------------------------------------------------------------
# Same matvec as _vq_decode_kernel, but each index is reconstructed from two
# byte-aligned planes instead of an int16/uint8 buffer:
#   idx = lo | (hi << 8),  hi extracted from a (HI_BITS)-packed byte (MSB-first).
# HBM traffic = lo (1 B) + hi (HI_BITS/8 B) per index = the packed width, < int16.
# Uniform HI_BITS across stages; 2-stage; HI_BITS in {2,4} so a value never straddles.

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64},  num_warps=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 256}, num_warps=8),
    ],
    key=["M", "GPR"],
)
@triton.jit
def _vq_decode_planes_kernel(
    x_ptr,
    lo0_ptr, hi0_ptr, lo1_ptr, hi1_ptr,
    cb0_ptr, cb1_ptr,
    scale_ptr, y_ptr,
    M, GPR, BPR,
    G: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    N_STAGES: tl.constexpr,
    HI_BITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_MAJOR: tl.constexpr,
):
    pid = tl.program_id(0)
    m_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_ids < M
    e = tl.arange(0, G)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    per = 8 // HI_BITS
    himask = (1 << HI_BITS) - 1

    for g in tl.range(0, GPR):
        x_g = tl.load(x_ptr + g * G + e).to(tl.float32)
        b = g // GROUPS_PER_BLOCK
        if GROUP_MAJOR:
            i_off = g * M + m_ids
            s_off = b * M + m_ids
        else:
            i_off = m_ids * GPR + g
            s_off = m_ids * BPR + b
        scale = tl.load(scale_ptr + s_off, mask=m_mask, other=0.0).to(tl.float32)

        lo0 = tl.load(lo0_ptr + i_off, mask=m_mask, other=0).to(tl.int32)
        byte0 = tl.load(hi0_ptr + (i_off * HI_BITS) // 8, mask=m_mask, other=0).to(tl.int32)
        shift0 = 8 - ((i_off % per) + 1) * HI_BITS
        idx0 = lo0 | (((byte0 >> shift0) & himask) << 8)
        cbe0 = tl.load(cb0_ptr + idx0[:, None] * G + e[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
        dot = tl.sum(cbe0 * x_g[None, :], axis=1)

        if N_STAGES >= 2:
            lo1 = tl.load(lo1_ptr + i_off, mask=m_mask, other=0).to(tl.int32)
            byte1 = tl.load(hi1_ptr + (i_off * HI_BITS) // 8, mask=m_mask, other=0).to(tl.int32)
            shift1 = 8 - ((i_off % per) + 1) * HI_BITS
            idx1 = lo1 | (((byte1 >> shift1) & himask) << 8)
            cbe1 = tl.load(cb1_ptr + idx1[:, None] * G + e[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
            dot += tl.sum(cbe1 * x_g[None, :], axis=1)

        acc += scale * dot

    tl.store(y_ptr + m_ids, acc, mask=m_mask)


def _vq_decode_planes_n1(layer, x_2d: torch.Tensor):
    """N=1 decode reading bit-planed indices. Returns y [1,M] fp16, or None if the
    layer's plane layout is unsupported (non-uniform hi-bits / >2 stages / not planed)."""
    widths = getattr(layer, "_plane_width", None)
    if not widths or layer.n_stages != 2:
        return None
    if widths[0] == 0 or any(w != widths[0] for w in widths):
        return None
    hi_bits = widths[0] - 8
    if hi_bits not in (2, 4):
        return None
    G, B = layer.group_size, layer.block_size
    K, M = layer.in_features, layer.out_features
    GPR, BPR = K // G, K // B
    x_flat = x_2d.view(K).contiguous()
    y = torch.empty(M, dtype=torch.float32, device=x_2d.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _vq_decode_planes_kernel[grid](
        x_flat,
        layer.indices_lo_0, layer.indices_hi_0, layer.indices_lo_1, layer.indices_hi_1,
        layer.codebook_0, layer.codebook_1,
        layer.scales, y,
        M, GPR, BPR,
        G, B // G, layer.n_stages, hi_bits,
        GROUP_MAJOR=bool(getattr(layer, "_group_major", False)),
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
        # Canonical Triton matmul config space (larger N/K tiles, deeper
        # pipelining) - num_stages>2 overlaps the codebook gather with the dot.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
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
    GROUP_MAJOR: tl.constexpr,    # indices [GPR,M] / scales [BPR,M] vs legacy row-major
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
        if GROUP_MAJOR:
            flat_group = group_of_col[None, :] * M + m_ids[:, None]
        else:
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
        if GROUP_MAJOR:
            scale_off = block_of_col[None, :] * M + m_ids[:, None]
        else:
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


# The VQLinear forward dispatcher lives in orka.inference.dispatch (it orchestrates
# this Triton backend and the CUDA backend in cuda_decode).
