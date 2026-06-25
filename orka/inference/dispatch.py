"""VQLinear forward dispatcher: route each call to the fastest available backend.

This is the single place that decides, per call, between the CUDA backend
(orka.inference.cuda_decode) and the Triton fallback (orka.inference.triton_kernels):

    N == 1 : cuda_decode.forward_n1 (float4)      -> Triton _vq_decode_kernel -> python
    N  > 1 : cuda_decode.forward_prefill (N>=256) -> Triton _vq_gemm_kernel

Any CUDA-side failure / unsupported layer returns None and falls back transparently.
See orka/inference/__init__.py for the full architecture map.
"""

from __future__ import annotations

import torch
import triton

from orka.inference.triton_kernels import _vq_decode_n1, _vq_gemm_kernel


def vq_linear_forward(layer, x: torch.Tensor) -> torch.Tensor:
    """Run a VQLinear forward, picking the backend by token count (see module docstring).

    x: [..., in_features] -> [..., out_features].
    """
    try:
        from orka.inference import cuda_decode
    except Exception:
        cuda_decode = None
    G = layer.group_size
    B = layer.block_size
    K = layer.in_features
    M = layer.out_features

    # The Triton kernels (_vq_decode_n1 / _vq_gemm_kernel) are general over group_size
    # and block_size (G/B are constexpr args). The CUDA float4 fast path is the only
    # group_size==8 specialization, and it self-gates via cuda_decode.supported()
    # (returns False for other G -> transparent Triton fallback). So the dispatcher only
    # needs the divisibility invariants the layout relies on, not a fixed G/B.
    assert K % G == 0, f"in_features={K} must be divisible by G={G}"
    assert K % B == 0, f"in_features={K} must be divisible by B={B}"

    orig_shape = x.shape
    x_2d = x.reshape(-1, K).contiguous().to(torch.float16)
    N = x_2d.shape[0]

    # Decode hot path (N=1, memory-bound matvec). Prefer the fused CUDA fast path
    # (float4 gather GEMV + warp-spmv correction, group-major coalesced indices)
    # when available - ~6x over the Triton path, dense-fp16 parity at >=1B. Any
    # failure (no nvcc, unsupported layer) returns None and we fall back below.
    if N == 1:
        if cuda_decode is not None:
            try:
                if cuda_decode.supported(layer, N):
                    out = cuda_decode.forward_n1(layer, x)
                    if out is not None:
                        return out
            except Exception:
                pass
        y = _vq_decode_n1(layer, x_2d)
        if layer.corr_col.numel() > 0:
            sp = layer._correction_sparse()
            y = y + torch.sparse.mm(sp, x_2d.float().T.contiguous()).T.to(torch.float16)
        if layer.bias is not None:
            y = y + layer.bias
        return y.reshape(*orig_shape[:-1], M).to(x.dtype)

    # N>1 prefill: for long-enough prompts, fused decode-to-dense + cuBLAS beats the
    # Triton gather-GEMM (~2x). Falls back to Triton below the token threshold / on any
    # failure / unsupported layer.
    if cuda_decode is not None:
        try:
            if cuda_decode.supported_prefill(layer, N):
                out = cuda_decode.forward_prefill(layer, x)
                if out is not None:
                    return out
        except Exception:
            pass

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
        GROUP_MAJOR=bool(getattr(layer, "_group_major", False)),
    )

    # Sparse correction: W_correction [M, K] -> y += x @ W_correction.T
    # The coalesced sparse tensor is built once and cached on the layer; only
    # the sparse.mm runs per forward.
    if layer.corr_col.numel() > 0:
        sp = layer._correction_sparse()
        correction = torch.sparse.mm(sp, x_2d.float().T.contiguous()).T.to(torch.float16)
        y = y + correction

    if layer.bias is not None:
        y = y + layer.bias

    return y.reshape(*orig_shape[:-1], M).to(x.dtype)
