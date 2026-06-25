"""torch custom op wrapping the bit-plane VQ kernels.

VQLinear.forward calls Triton kernels and branches on Python/tensor properties, which
makes torch.compile graph-break at every layer - so compile/CUDA-graphs can't capture
the per-token decode loop (where the launch + dispatch overhead lives). Registering the
plane matvec/GEMM as a single ``torch.library.custom_op`` makes Dynamo treat it as one
opaque node instead of a break: the surrounding model compiles into one graph and CUDA
graphs can capture it. ``register_fake`` gives compile the output shape without running
the kernel.

Only the uniform 2-stage planed case (hi-bits in {2,4}, no correction) routes here; the
caller falls back to the eager path otherwise.
"""

from __future__ import annotations

import torch
import triton

from orka.inference.triton_kernels import _vq_decode_planes_kernel, _vq_gemm_planes_kernel


@torch.library.custom_op("orka::vq_plane_linear", mutates_args=())
def vq_plane_linear(
    x: torch.Tensor,
    lo0: torch.Tensor, hi0: torch.Tensor, lo1: torch.Tensor, hi1: torch.Tensor,
    cb0: torch.Tensor, cb1: torch.Tensor, scales: torch.Tensor,
    M: int, GPR: int, BPR: int, G: int, B: int, hi_bits: int, group_major: bool,
) -> torch.Tensor:
    """x [N, K] fp16 -> y [N, M] fp16, reading bit-planed 2-stage indices."""
    N, K = x.shape
    if N == 1:
        y = torch.empty(M, dtype=torch.float32, device=x.device)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
        _vq_decode_planes_kernel[grid](
            x.reshape(K).contiguous(),
            lo0, hi0, lo1, hi1, cb0, cb1, scales, y,
            M, GPR, BPR, G, B // G, 2, hi_bits,
            GROUP_MAJOR=group_major,
        )
        return y.view(1, M).to(torch.float16)
    y = torch.empty((N, M), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    _vq_gemm_planes_kernel[grid](
        x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1),
        lo0, hi0, lo1, hi1, cb0, cb1, scales,
        M, N, K, GPR, BPR, G, B, 2, hi_bits,
        GROUP_MAJOR=group_major,
    )
    return y


@vq_plane_linear.register_fake
def _vq_plane_linear_fake(
    x, lo0, hi0, lo1, hi1, cb0, cb1, scales,
    M, GPR, BPR, G, B, hi_bits, group_major,
):
    return x.new_empty((x.shape[0], M), dtype=torch.float16)


def plane_op_supported(layer) -> bool:
    """True if the layer's plane layout is the uniform 2-stage case the op handles."""
    widths = getattr(layer, "_plane_width", None)
    if not widths or layer.n_stages != 2:
        return False
    if widths[0] == 0 or any(w != widths[0] for w in widths):
        return False
    return (widths[0] - 8) in (2, 4) and layer.corr_col.numel() == 0
