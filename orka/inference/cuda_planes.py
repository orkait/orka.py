"""CUDA warp-per-row GEMV for N=1 decode of bit-planed RVQ indices.

The decode matvec is the hot path. A naive one-thread-per-row kernel is ~7-10x a dense
cuBLAS GEMV (the GPR-group reduction runs serially per thread, latency-bound). Splitting
that reduction across a **warp** (32 lanes, shfl-reduce) - the group-8 ``cuda_decode.gemv_f4``
trick - makes it match/beat dense while reading ~9x less data: an isolated [3072,1024]
tensor measured 23 us vs 26 us dense (0.92x), max rel err 3e-4.

General over: any even group_size (G==4 uses a float2 codeword fast path; larger G loops
__half2), and n_stages in {1,2,3} (active stages summed per codeword). Each index is
rebuilt from a uint8 low plane + a (HI_BITS)-packed high plane
(idx = lo | ((hi_byte>>shift)&mask)<<8). Uniform planed stages, hi-bits in {2,4},
row-major, correction-free; opt-in with transparent fallback to the Triton path.
"""

from __future__ import annotations

import torch

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_fp16.h>

// One warp per output row m; 32 lanes split the GPR-group loop and shfl-reduce.
// Sums N_STAGES codewords per group. Unused stage pointers alias stage 0 (never read).
__global__ void gemv_planes(
    const unsigned char* __restrict__ lo0, const unsigned char* __restrict__ hi0,
    const unsigned char* __restrict__ lo1, const unsigned char* __restrict__ hi1,
    const unsigned char* __restrict__ lo2, const unsigned char* __restrict__ hi2,
    const __half* __restrict__ cb0, const __half* __restrict__ cb1, const __half* __restrict__ cb2,
    const __half* __restrict__ scale, const __half* __restrict__ x,
    float* __restrict__ y, int M, int GPR, int BPR, int GPB, int G, int HI_BITS, int N_STAGES){
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int m = gid >> 5, lane = gid & 31;
  if (m >= M) return;
  int per = 8 / HI_BITS, himask = (1 << HI_BITS) - 1, GH = G >> 1;
  const __half2* x2 = (const __half2*)x;
  float acc = 0.0f;
  for (int g = lane; g < GPR; g += 32) {
    int p = m * GPR + g;
    int blk = g / GPB;
    float s = __half2float(scale[m * BPR + blk]);
    int sh = 8 - ((p % per) + 1) * HI_BITS;
    int bo = (p * HI_BITS) >> 3;
    int i0 = lo0[p] | ((((int)__ldg(&hi0[bo]) >> sh) & himask) << 8);
    int i1 = (N_STAGES >= 2) ? (lo1[p] | ((((int)__ldg(&hi1[bo]) >> sh) & himask) << 8)) : 0;
    int i2 = (N_STAGES >= 3) ? (lo2[p] | ((((int)__ldg(&hi2[bo]) >> sh) & himask) << 8)) : 0;
    const __half2* c0 = (const __half2*)(cb0 + i0 * G);
    const __half2* c1 = (const __half2*)(cb1 + i1 * G);
    const __half2* c2 = (const __half2*)(cb2 + i2 * G);
    const __half2* xv = x2 + g * GH;
    float dot = 0.0f;
    if (G == 4 && N_STAGES == 2) {              // float2 fast path (one 8B load/stage)
      float2 r0 = __ldg((const float2*)c0), r1 = __ldg((const float2*)c1);
      const __half2* a = (const __half2*)&r0; const __half2* b = (const __half2*)&r1;
      __half2 w0 = __hadd2(a[0], b[0]), w1 = __hadd2(a[1], b[1]);
      float2 wf0 = __half22float2(w0), wf1 = __half22float2(w1);
      float2 xf0 = __half22float2(__ldg(&xv[0])), xf1 = __half22float2(__ldg(&xv[1]));
      dot = wf0.x * xf0.x + wf0.y * xf0.y + wf1.x * xf1.x + wf1.y * xf1.y;
    } else {
      for (int e = 0; e < GH; e++) {
        __half2 w = __ldg(&c0[e]);
        if (N_STAGES >= 2) w = __hadd2(w, __ldg(&c1[e]));
        if (N_STAGES >= 3) w = __hadd2(w, __ldg(&c2[e]));
        float2 wf = __half22float2(w), xf = __half22float2(__ldg(&xv[e]));
        dot += wf.x * xf.x + wf.y * xf.y;
      }
    }
    acc += s * dot;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[m] = acc;
}

torch::Tensor decode_planes(
    torch::Tensor lo0, torch::Tensor hi0, torch::Tensor lo1, torch::Tensor hi1,
    torch::Tensor lo2, torch::Tensor hi2,
    torch::Tensor cb0, torch::Tensor cb1, torch::Tensor cb2,
    torch::Tensor scale, torch::Tensor x,
    int M, int GPR, int BPR, int GPB, int G, int HI_BITS, int N_STAGES){
  auto y = torch::zeros({M}, torch::dtype(torch::kFloat32).device(x.device()));
  int th = 256, bx = (M * 32 + th - 1) / th;
  gemv_planes<<<bx, th>>>(
    lo0.data_ptr<unsigned char>(), hi0.data_ptr<unsigned char>(),
    lo1.data_ptr<unsigned char>(), hi1.data_ptr<unsigned char>(),
    lo2.data_ptr<unsigned char>(), hi2.data_ptr<unsigned char>(),
    (const __half*)cb0.data_ptr<at::Half>(), (const __half*)cb1.data_ptr<at::Half>(),
    (const __half*)cb2.data_ptr<at::Half>(),
    (const __half*)scale.data_ptr<at::Half>(), (const __half*)x.data_ptr<at::Half>(),
    y.data_ptr<float>(), M, GPR, BPR, GPB, G, HI_BITS, N_STAGES);
  return y;
}
'''

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE or None
    try:
        from torch.utils.cpp_extension import load_inline
        _MODULE = load_inline(
            name="orka_cuda_planes",
            cpp_sources=(
                "torch::Tensor decode_planes(torch::Tensor,torch::Tensor,torch::Tensor,"
                "torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,"
                "torch::Tensor,torch::Tensor,torch::Tensor,int,int,int,int,int,int,int);"
            ),
            cuda_sources=_CUDA_SRC,
            functions=["decode_planes"],
            verbose=False,
        )
    except Exception as exc:
        _MODULE = False
        # Surface the failure once: without this kernel, N=1 decode silently falls to
        # the slower Triton path (measured ~0.66x -> 0.47x). The usual cause is a
        # missing `ninja` (torch's JIT extension builder needs it); installing it
        # recovers the fast path. Warn so deployments don't run slow unknowingly.
        import warnings
        hint = " (install `ninja` - torch JIT needs it)" if "Ninja" in str(exc) else ""
        warnings.warn(
            f"orka: CUDA plane-decode kernel unavailable, using the slower fallback"
            f"{hint}. Reason: {type(exc).__name__}: {str(exc)[:120]}",
            RuntimeWarning, stacklevel=2,
        )
        return None
    return _MODULE


def supported(layer) -> bool:
    widths = getattr(layer, "_plane_width", None)
    if not widths or layer.n_stages not in (1, 2, 3):
        return False
    if widths[0] == 0 or any(w != widths[0] for w in widths):
        return False
    if (widths[0] - 8) not in (2, 4):
        return False
    return (
        layer.group_size % 2 == 0                    # half2 codeword (float2 fast path at G=4)
        and not getattr(layer, "_group_major", False)  # kernel reads row-major indices
        and layer.corr_col.numel() == 0
        and layer.scales.is_cuda
        and _get_module() is not None
    )


def forward_n1(layer, x_2d: torch.Tensor):
    """N=1 decode via the warp-per-row CUDA plane GEMV. x_2d [1,K] -> y [1,M] fp16, or None."""
    mod = _get_module()
    if mod is None:
        return None
    G, B = layer.group_size, layer.block_size
    K, M = layer.in_features, layer.out_features
    GPR, BPR = K // G, K // B
    hi_bits = layer._plane_width[0] - 8
    xf = x_2d.reshape(K).to(torch.float16).contiguous()

    # Per-stage plane/codebook buffers; unused stages (n_stages<3) alias stage 0.
    def lo(s): return getattr(layer, f"indices_lo_{s}")
    def hi(s): return getattr(layer, f"indices_hi_{s}")
    def cb(s): return getattr(layer, f"codebook_{s}").reshape(-1)
    s1 = 1 if layer.n_stages >= 2 else 0
    s2 = 2 if layer.n_stages >= 3 else 0

    y = mod.decode_planes(
        lo(0), hi(0), lo(s1), hi(s1), lo(s2), hi(s2),
        cb(0), cb(s1), cb(s2),
        layer.scales, xf, M, GPR, BPR, B // G, G, hi_bits, layer.n_stages,
    )
    return y.view(1, M).to(torch.float16)
