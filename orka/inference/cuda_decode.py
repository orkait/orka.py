"""Optional CUDA fast-path for the N=1 RVQ-12-12 autoregressive decode hot path.

Fuses two hand-written kernels for single-token decode (the memory-bound matvec
regime that dominates generation):

  1. VQ GEMV - float4 (128-bit) codeword gather + FMA, group-major index layout.
     A roofline probe showed the codebook gather is the sole bottleneck (the index
     stream, scales and ALU all run near HBM peak); loading each 8-half codeword as
     ONE float4 instead of four half2 loads (4x fewer load instructions) lifts the
     gather from ~63 to ~110 GB/s, beating cuBLAS fp16 GEMV per-layer on >=1B shapes.
  2. Salient/outlier correction - warp-per-row CSR spmv (one warp reduces one output
     row's nnz), replacing the per-call torch.sparse.mm.

Net effect vs the Triton N=1 path: ~6x faster (the shipped Triton kernel reads the
indices row-major = strided/uncoalesced; this path is group-major = coalesced),
reaching dense-fp16 parity at 3-4x compression with exact rvq-12-12 quality.

Strictly opt-in with transparent fallback: compiles on first use via torch's
cpp_extension; ANY failure (no nvcc/ninja, unsupported layer shape) returns None and
the caller uses the Triton path. Applies only when:
    N == 1, n_stages == 2, group_size == 8, block_size == 32, in_features % 32 == 0, cuda.
"""

from __future__ import annotations

import torch

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>

// VQ GEMV: y[m] = sum_g scale[m,g] * (cb0[i0[m,g]] + cb1[i1[m,g]]) . x_g
// indices group-major [GROUPS, M] (coalesced across m); codeword = one float4 load.
__global__ void gemv_f4(
    const unsigned short* __restrict__ i0, const unsigned short* __restrict__ i1,
    const float4* __restrict__ cb0, const float4* __restrict__ cb1,   // [cb_size] (8 halves each)
    const __half* __restrict__ scale, const __half2* __restrict__ x,
    float* __restrict__ y, int M, int GROUPS, int GPB, int GPSPLIT, int TH){
  int pk = blockIdx.y; int g_lo = pk * GPSPLIT, g_hi = min(g_lo + GPSPLIT, GROUPS);
  int m = blockIdx.x * TH + threadIdx.x; if (m >= M) return;
  float acc = 0.0f;
  for (int g = g_lo; g < g_hi; g++) {
    int base = g * M + m; int a = i0[base], b = i1[base];
    int blk = g / GPB; float s = __half2float(scale[blk * M + m]);
    float4 w0 = __ldg(&cb0[a]), w1 = __ldg(&cb1[b]);
    const __half2* xg = &x[g * 4];
    const __half2* p0 = (const __half2*)&w0; const __half2* p1 = (const __half2*)&w1;
    float dot = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
      __half2 w = __hadd2(p0[e], p1[e]); __half2 xv = __ldg(&xg[e]);
      float2 wf = __half22float2(w), xf = __half22float2(xv);
      dot += wf.x * xf.x + wf.y * xf.y;
    }
    acc += s * dot;
  }
  atomicAdd(&y[m], acc);
}

// Correction: y[m] += sum_{j in row m} val[j] * x[col[j]]   (warp per row)
__global__ void spmv_warp(
    const int* __restrict__ rp, const int* __restrict__ col, const __half* __restrict__ val,
    const __half* __restrict__ x, float* __restrict__ y, int M){
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; int lane = threadIdx.x & 31;
  if (warp >= M) return;
  int s = rp[warp], e = rp[warp + 1]; float acc = 0.0f;
  for (int j = s + lane; j < e; j += 32) acc += __half2float(val[j]) * __half2float(__ldg(&x[col[j]]));
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[warp] += acc;
}

torch::Tensor decode(torch::Tensor i0, torch::Tensor i1, torch::Tensor cb0, torch::Tensor cb1,
                     torch::Tensor scale, torch::Tensor x, int M, int GROUPS, int GPB, int KSPLIT, int TH){
  auto y = torch::zeros({M}, torch::dtype(torch::kFloat32).device(x.device()));
  int bx = (M + TH - 1) / TH, gps = (GROUPS + KSPLIT - 1) / KSPLIT; dim3 grid(bx, KSPLIT);
  gemv_f4<<<grid, TH>>>(
    (const unsigned short*)i0.data_ptr<int16_t>(), (const unsigned short*)i1.data_ptr<int16_t>(),
    (const float4*)cb0.data_ptr<at::Half>(), (const float4*)cb1.data_ptr<at::Half>(),
    (const __half*)scale.data_ptr<at::Half>(), (const __half2*)x.data_ptr<at::Half>(),
    y.data_ptr<float>(), M, GROUPS, GPB, gps, TH);
  return y;
}

void correct(torch::Tensor y, torch::Tensor rp, torch::Tensor col, torch::Tensor val, torch::Tensor x, int M){
  int th = 128, bx = (M * 32 + th - 1) / th;
  spmv_warp<<<bx, th>>>(rp.data_ptr<int>(), col.data_ptr<int>(), (const __half*)val.data_ptr<at::Half>(),
                        (const __half*)x.data_ptr<at::Half>(), y.data_ptr<float>(), M);
}

// Decode the full weight to dense [M,K] fp16 (row-major) for the N>1 prefill path:
// one thread per group (g,m) writes W[m, g*8..g*8+7] = scale * (cb0[i0]+cb1[i1]).
// The gather is paid once, then cuBLAS reuses W over all N tokens (vs the Triton
// gather-GEMM re-gathering per tile per token).
__global__ void ddense(
    const unsigned short* __restrict__ i0, const unsigned short* __restrict__ i1,
    const float4* __restrict__ cb0, const float4* __restrict__ cb1,
    const __half* __restrict__ scale, __half* __restrict__ W, int M, int GROUPS, int GPB){
  int gm = blockIdx.x * blockDim.x + threadIdx.x; if (gm >= M * GROUPS) return;
  int g = gm / M, m = gm % M; int blk = g / GPB;     // group-major: i0[g*M+m]
  float4 w0 = __ldg(&cb0[i0[gm]]), w1 = __ldg(&cb1[i1[gm]]);
  __half2 sh = __half2half2(scale[blk * M + m]);
  const __half2* p0 = (const __half2*)&w0; const __half2* p1 = (const __half2*)&w1;
  __half2* out = (__half2*)(W + m * (GROUPS * 8) + g * 8);
  #pragma unroll
  for (int e = 0; e < 4; e++) out[e] = __hmul2(sh, __hadd2(p0[e], p1[e]));
}
torch::Tensor decode_dense(torch::Tensor i0, torch::Tensor i1, torch::Tensor cb0, torch::Tensor cb1,
                           torch::Tensor scale, int M, int GROUPS, int GPB){
  auto W = torch::empty({M, GROUPS * 8}, torch::dtype(torch::kFloat16).device(i0.device()));
  int n = M * GROUPS, th = 256, bx = (n + th - 1) / th;
  ddense<<<bx, th>>>(
    (const unsigned short*)i0.data_ptr<int16_t>(), (const unsigned short*)i1.data_ptr<int16_t>(),
    (const float4*)cb0.data_ptr<at::Half>(), (const float4*)cb1.data_ptr<at::Half>(),
    (const __half*)scale.data_ptr<at::Half>(), (__half*)W.data_ptr<at::Half>(), M, GROUPS, GPB);
  return W;
}
'''

_MODULE = None          # compiled extension, or False if compilation failed
_CFG_CACHE: dict = {}    # (M, GPR) -> (KSPLIT, TH)


def _get_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE or None
    try:
        from torch.utils.cpp_extension import load_inline
        _MODULE = load_inline(
            name="orka_cuda_decode",
            cpp_sources=(
                "torch::Tensor decode(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
                "torch::Tensor,torch::Tensor,int,int,int,int,int);\n"
                "void correct(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int);\n"
                "torch::Tensor decode_dense(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
                "torch::Tensor,int,int,int);"
            ),
            cuda_sources=_CUDA_SRC,
            functions=["decode", "correct", "decode_dense"],
            verbose=False,
        )
    except Exception:
        _MODULE = False
        return None
    return _MODULE


def supported(layer, n_tokens: int) -> bool:
    return (
        n_tokens == 1
        and getattr(layer, "n_stages", 0) == 2
        and getattr(layer, "group_size", 0) == 8
        and getattr(layer, "block_size", 0) == 32
        and layer.in_features % 32 == 0
        and getattr(layer, "_group_major", False)  # kernel reads group-major directly
        and layer.scales.is_cuda
        and _get_module() is not None
    )


def _pick_cfg(mod, M, GPR, run) -> tuple[int, int]:
    """One-time micro-autotune per (M, GPR) shape; cached. Picks (KSPLIT, TH)."""
    import time
    key = (M, GPR)
    if key in _CFG_CACHE:
        return _CFG_CACHE[key]
    best_t, best = 1e9, (64, 256)
    for th in (128, 256):
        for ks in (8, 16, 32, 64, 128):
            if ks > GPR:
                continue
            for _ in range(5):
                run(ks, th)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                run(ks, th)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if dt < best_t:
                best_t, best = dt, (ks, th)
    _CFG_CACHE[key] = best
    return best


def _build_buffers(layer, M, GPR, BPR):
    """Build + cache the group-major index/scale buffers and the correction CSR.

    The artifact stores indices row-major [M, GPR]; the kernel needs group-major
    [GPR, M] for coalesced loads. Done once per layer, cached on the layer.
    """
    cache = getattr(layer, "_cuda_decode_cache", None)
    if cache is not None:
        return cache
    # Indices/scales are stored group-major ([GPR,M]/[BPR,M]) - exactly the layout the
    # kernel reads (i[g*M+m]) - so reference them directly, no transposed copy.
    i0 = layer.indices_0
    i1 = layer.indices_1
    sc = layer.scales
    c0 = layer.codebook_0.reshape(-1)   # [cb,8] contiguous -> view, float4-addressable
    c1 = layer.codebook_1.reshape(-1)
    # Reuse the layer's registered CSR correction buffers directly - no rebuild,
    # no second copy (this is the same data the N>1 cuSPARSE path uses).
    csr = None
    if layer.corr_col.numel() > 0:
        csr = (layer.corr_rowptr, layer.corr_col, layer.corr_val)
    cache = (i0, i1, c0, c1, sc, csr)
    layer._cuda_decode_cache = cache
    return cache


def forward_n1(layer, x: torch.Tensor):
    """Full N=1 forward via the CUDA fast path. Returns the output tensor, or None
    (unsupported / compile failed) so the caller falls back to the Triton path."""
    mod = _get_module()
    if mod is None:
        return None
    G, B = layer.group_size, layer.block_size
    K, M = layer.in_features, layer.out_features
    GPR, BPR, GPB = K // G, K // B, B // G

    x2 = x.reshape(-1, K)
    if x2.shape[0] != 1:
        return None
    i0, i1, c0, c1, sc, csr = _build_buffers(layer, M, GPR, BPR)
    xf = x2.reshape(K).to(torch.float16).contiguous()

    ks, th = _pick_cfg(mod, M, GPR, lambda k, t: mod.decode(i0, i1, c0, c1, sc, xf, M, GPR, GPB, k, t))
    yf = mod.decode(i0, i1, c0, c1, sc, xf, M, GPR, GPB, ks, th)
    if csr is not None:
        mod.correct(yf, csr[0], csr[1], csr[2], xf, M)

    y = yf.view(1, M).to(torch.float16)
    if layer.bias is not None:
        y = y + layer.bias
    return y.reshape(*x.shape[:-1], M).to(x.dtype)


# N>1 prefill: above this token count, fused decode-to-dense + cuBLAS beats the Triton
# gather-GEMM (~2x). Below it the one-time decode cost is not amortized; use Triton.
PREFILL_MIN_TOKENS = 256


def supported_prefill(layer, n_tokens: int) -> bool:
    return (
        n_tokens >= PREFILL_MIN_TOKENS
        and getattr(layer, "n_stages", 0) == 2
        and getattr(layer, "group_size", 0) == 8
        and getattr(layer, "block_size", 0) == 32
        and getattr(layer, "_group_major", False)
        and layer.scales.is_cuda
        and _get_module() is not None
    )


def forward_prefill(layer, x: torch.Tensor):
    """N>1 prefill via fused decode-to-dense + cuBLAS. Decodes the weight once (the
    gather is paid once, not per token), then F.linear reuses it over all N tokens.
    The dense weight is transient (one layer at a time, freed) so VRAM stays compressed
    for the decode path. Returns the output, or None to fall back to Triton."""
    import torch.nn.functional as F
    mod = _get_module()
    if mod is None:
        return None
    K, M = layer.in_features, layer.out_features
    GPR, GPB = K // layer.group_size, layer.block_size // layer.group_size
    c = getattr(layer, "_ddense_buf", None)
    if c is None:
        c = (layer.indices_0.contiguous(), layer.indices_1.contiguous(),
             layer.codebook_0.reshape(-1).contiguous(), layer.codebook_1.reshape(-1).contiguous(),
             layer.scales.contiguous())
        layer._ddense_buf = c
    i0, i1, c0, c1, sc = c
    W = mod.decode_dense(i0, i1, c0, c1, sc, M, GPR, GPB)   # [M,K] fp16, transient
    if layer.corr_col.numel() > 0:
        cnt = (layer.corr_rowptr[1:] - layer.corr_rowptr[:-1]).long()
        rows = torch.repeat_interleave(torch.arange(M, device=W.device), cnt)
        W[rows, layer.corr_col.long()] += layer.corr_val.to(W.dtype)
    x2 = x.reshape(-1, K)
    y = F.linear(x2, W.to(x.dtype))
    if layer.bias is not None:
        y = y + layer.bias
    return y.reshape(*x.shape[:-1], M).to(x.dtype)
