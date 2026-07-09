"""orka inference: serve a packed .orka artifact with weights kept in VQ form.

Architecture (how the pieces fit):

  build / serve
    export.py   export_inference()  -> load an HF model, swap nn.Linear for VQLinear
    fast.py     load_fast()         -> alternative: dense (export_vllm) + torch.compile,
                                       for mamba/hybrid models where the SSM (not the
                                       linears) dominates and reconstruct-to-dense wins

  storage (one layer) - vq_linear.py
    VQLinear           registered buffers: indices_{s} (group-major [GPR,M]),
                       codebook_{s}, scales (group-major [BPR,M]), CSR correction
                       (corr_rowptr / corr_col / corr_val).
    build_vq_linear()  populate them from the artifact.

  forward dispatch (VQLinear.forward -> dispatch.vq_linear_forward) - dispatch.py
    N == 1 (decode, memory-bound matvec):
      cuda_decode.forward_n1        float4 gather GEMV + warp-spmv correction
        -> fallback triton_kernels._vq_decode_kernel
           -> fallback VQLinear._forward_python (dense)
    N  > 1 (prefill):
      cuda_decode.forward_prefill   fused decode-to-dense + cuBLAS (N >= 256)
        -> fallback triton_kernels._vq_gemm_kernel (gather-GEMM)

  backends
    cuda_decode.py     CUDA kernels (gemv_f4 N=1, spmv_warp correction, ddense prefill),
                       compiled lazily; any failure falls back to Triton transparently.
    triton_kernels.py  Triton kernels (_vq_decode_kernel, _vq_gemm_kernel) - the fallback.

Indices/scales are stored group-major so both backends read them coalesced (the
`_group_major` flag gates this; legacy row-major still works via the GROUP_MAJOR
constexpr in the Triton kernels).
"""

from orka.inference.export import export_inference
from orka.inference.fast import load_fast
from orka.inference.vq_linear import VQLinear, build_vq_linear

__all__ = ["VQLinear", "build_vq_linear", "export_inference", "load_fast"]
