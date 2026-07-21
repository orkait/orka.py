"""VQLinear: the inference-time VQ layer (decode/forward). Construction lives in _vq_build."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLinear(nn.Module):
    """Linear layer backed by VQ codebooks.

    All weight state lives in registered buffers (frozen; not optimized).
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        n_stages: int,
        group_size: int,
        block_size: int,
        cb_sizes: list[int] | int,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.n_stages = n_stages
        self.group_size = group_size
        self.block_size = block_size
        self._n1_ok = None  # cached cuda_planes.supported() result (N=1 hot path)
        # Per-stage codebook sizes (RVQ stages may use different cb sizes:
        # 256/896/4096 from the per-tensor allocation). int16 indices hold all.
        if isinstance(cb_sizes, int):
            cb_sizes = [cb_sizes] * n_stages
        self.cb_sizes = list(cb_sizes)
        self._triton_ok: bool | None = None

        total = out_features * in_features
        n_groups = math.ceil(total / group_size)
        n_scale_blocks = math.ceil(total / block_size)

        # Per-stage index width = bits to address the codebook. Storage tiers:
        #   width <= 8        -> uint8 buffer indices_{s}        (#83)
        #   width in {10,12}  -> byte-aligned bit-planes lo/hi   (this change; 5-6 bpw)
        #   else (9,11,13-16) -> int16 buffer indices_{s}
        # Planes engage only when the high bits (width-8) divide a byte (2 or 4) so no
        # index straddles a byte - keeps the read coalesced and the kernel branch-free.
        self._plane_width = []
        for s in range(n_stages):
            width = max(1, (self.cb_sizes[s] - 1).bit_length())
            planed = 8 < width <= 12 and (width - 8) in (2, 4)
            self._plane_width.append(width if planed else 0)
            if planed:
                hi_bits = width - 8
                hi_bytes = math.ceil(n_groups * hi_bits / 8)
                self.register_buffer(f"indices_lo_{s}", torch.zeros(n_groups, dtype=torch.uint8))
                self.register_buffer(f"indices_hi_{s}", torch.zeros(hi_bytes, dtype=torch.uint8))
            else:
                # int16 is SIGNED (max 32767): codebooks > 32768 (e.g. vq-15/16, or a
                # codebook capped to a large vector_count) overflow to negative indices
                # -> cb[neg] wraps to the wrong entry. Promote to int32 above the int16
                # range; uint8 for <=256, int16 for the common <=32768 case.
                if self.cb_sizes[s] <= 256:
                    idx_dtype = torch.uint8
                elif self.cb_sizes[s] <= 32768:
                    idx_dtype = torch.int16
                else:
                    idx_dtype = torch.int32
                self.register_buffer(f"indices_{s}", torch.zeros(n_groups, dtype=idx_dtype))
            self.register_buffer(f"codebook_{s}", torch.zeros(self.cb_sizes[s], group_size, dtype=torch.float16))

        self.register_buffer("scales", torch.ones(n_scale_blocks, dtype=torch.float16))

        # Sparse correction (salient+outlier deltas) in CSR: rowptr [out+1] int32,
        # col [nnz] int32, val [nnz] fp16. CSR is what BOTH consumers want directly -
        # the cuSPARSE N>1 path and the warp-spmv N=1 kernel - so it is stored once
        # (no COO + CSR duplication) and ~2.5x smaller than the old COO form.
        self.register_buffer("corr_rowptr", torch.zeros(out_features + 1, dtype=torch.int32))
        self.register_buffer("corr_col", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("corr_val", torch.zeros(0, dtype=torch.float16))

        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16))
        else:
            self.register_buffer("bias", None)

    @property
    def weight(self):
        """Device/dtype sentinel. orka serves the projection through forward(); the
        weight matrix is never materialized. Some model code (e.g. the mamba/SSM
        fast path in falcon_h1/jamba) only probes `in_proj.weight.device`/`.dtype`
        to pick a kernel path, not the matrix itself - return a 1-elem tensor on the
        right device so those checks succeed while the projection stays compressed."""
        w = getattr(self, "_weight_sentinel", None)
        if w is None or w.device != self.scales.device or w.dtype != self.scales.dtype:
            w = torch.empty(1, device=self.scales.device, dtype=self.scales.dtype)
            object.__setattr__(self, "_weight_sentinel", w)
        return w

    # ------------------------------------------------------------------
    # Correction sparse tensor (rebuilt on first forward or after .to())
    # ------------------------------------------------------------------

    def _correction_sparse(self) -> torch.Tensor | None:
        cached = getattr(self, "_corr_sp_cache", None)
        if cached is not None:
            # The cache is a plain attribute, not a registered buffer, so
            # nn.Module._apply (.to() / .cuda()) does not migrate it with the rest
            # of the layer. Follow the layer instead of failing on a device mismatch.
            if cached.device != self.scales.device:
                cached = cached.to(self.scales.device)
                self._corr_sp_cache = cached
            return cached
        if self.corr_col.numel() == 0:
            return None
        # Wrap the stored CSR buffers as a torch CSR sparse tensor for cuSPARSE
        # CSR x dense (N>1 path). Built once and cached. The raw int32/fp16 buffers
        # are then FREED: the int64/fp32 sparse copy would otherwise sit alongside
        # them (~2.5x the correction footprint - at 9B scale that alone OOMs a 12GB
        # card). Inference needs one copy; state_dict round-trips must happen
        # before the first forward.
        sp = torch.sparse_csr_tensor(
            self.corr_rowptr.long(),
            self.corr_col.long(),
            self.corr_val.float(),
            size=(self.out_features, self.in_features),
            device=self.corr_col.device,
        )
        self._corr_sp_cache = sp
        # resize_(0) would keep the storage alive; reassigning the buffers drops it.
        self.corr_col = torch.empty(0, dtype=torch.int32, device=sp.device)
        self.corr_val = torch.empty(0, dtype=torch.float16, device=sp.device)
        self.corr_rowptr = torch.empty(0, dtype=torch.int32, device=sp.device)
        return sp

    def _correction_csr(self):
        """Live CSR correction as ``(rowptr, col, val)``, or None when there is none.

        ``_correction_sparse`` frees the raw buffers once it has cached the sparse
        tensor, so no reader may assume ``corr_col`` still holds the data - after any
        correction-carrying forward it lives in ``_corr_sp_cache`` instead. This is the
        one accessor that knows which storage is currently live; read the correction
        through it rather than off the buffers.
        """
        if self.corr_col.numel() > 0:
            return self.corr_rowptr, self.corr_col, self.corr_val
        sp = getattr(self, "_corr_sp_cache", None)
        if sp is None:
            return None
        return sp.crow_indices(), sp.col_indices(), sp.values()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # After a forward the raw buffers are empty, so the default buffer save emits
        # a zero-nnz correction and the checkpoint silently loses it. Re-emit from the
        # cache in the registered buffer dtypes, so serialization no longer depends on
        # whether the layer has been run.
        if self.corr_col.numel() > 0:
            return
        csr = self._correction_csr()
        if csr is None:
            return
        rowptr, col, val = csr
        destination[prefix + "corr_rowptr"] = rowptr.detach().to(torch.int32)
        destination[prefix + "corr_col"] = col.detach().to(torch.int32)
        destination[prefix + "corr_val"] = val.detach().to(torch.float16)

    # ------------------------------------------------------------------
    # Weight reconstruction (for testing / fallback)
    # ------------------------------------------------------------------

    def _stage_indices_int(self, s: int) -> torch.Tensor:
        """Return stage ``s`` indices as int64 [n_groups], regardless of storage tier
        (int16/uint8 buffer, or lo/hi bit-planes reconstructed as ``lo | hi<<8``)."""
        width = self._plane_width[s] if s < len(self._plane_width) else 0
        if width == 0:
            return getattr(self, f"indices_{s}").to(torch.int64)
        lo = getattr(self, f"indices_lo_{s}")
        hi = getattr(self, f"indices_hi_{s}")
        n = lo.numel()
        hi_bits = width - 8
        per = 8 // hi_bits
        mask = (1 << hi_bits) - 1
        # MSB-first packing: slot j (0=top) occupies bits [8-(j+1)*hi_bits, ...)
        shifts = torch.tensor(
            [8 - (j + 1) * hi_bits for j in range(per)], device=hi.device, dtype=torch.int32
        )
        hi_vals = ((hi.to(torch.int32)[:, None] >> shifts[None, :]) & mask).reshape(-1)[:n]
        return lo.to(torch.int64) | (hi_vals.to(torch.int64) << 8)

    def reconstruct_weight(self) -> torch.Tensor:
        """Decode full W [out, in] fp32. Expensive - for testing only."""
        dev = self.scales.device
        G, B = self.group_size, self.block_size
        M = self.out_features
        total = M * self.in_features
        padded = math.ceil(total / G) * G
        # Indices/scales may be stored group-major ([GPR,M]/[BPR,M]); transpose back
        # to row-major element/block order for the dense decode.
        gm = bool(getattr(self, "_group_major", False))

        decoded = torch.zeros(padded, dtype=torch.float32, device=dev)
        for s in range(self.n_stages):
            idxs = self._stage_indices_int(s)
            if gm and idxs.numel() == total // G:
                idxs = idxs.view(total // G // M, M).t().reshape(-1)
            cb = getattr(self, f"codebook_{s}").float()
            decoded.add_(cb[idxs].reshape(-1))
        decoded = decoded[:total]

        n_blocks = math.ceil(total / B)
        pad_b = n_blocks * B - total
        sc = self.scales
        if gm and sc.numel() == n_blocks:
            sc = sc.view(n_blocks // M, M).t().reshape(-1)
        if pad_b:
            decoded = F.pad(decoded, (0, pad_b))
        decoded = (decoded.reshape(n_blocks, B) * sc[:n_blocks, None].float()).reshape(-1)[:total]

        # Apply the correction from whichever storage is live: the raw CSR buffers
        # before the first forward, the cached sparse tensor after (the forward path
        # frees the buffers into it). Reading the buffers directly here dropped the
        # correction entirely once a forward had run - a silent numeric error.
        csr = self._correction_csr()
        if csr is not None:
            rowptr, col, val = (t.to(dev) for t in csr)
            counts = (rowptr[1:] - rowptr[:-1]).long()
            rows = torch.repeat_interleave(
                torch.arange(self.out_features, device=dev), counts
            )
            flat = rows * self.in_features + col.long()
            mask = flat < total
            decoded[flat[mask]] += val.float()[mask]

        return decoded.reshape(self.out_features, self.in_features)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _triton_available(self) -> bool:
        if self._triton_ok is None:
            try:
                from orka.inference.dispatch import vq_linear_forward  # noqa: F401
                self._triton_ok = True
            except Exception:
                self._triton_ok = False
        return self._triton_ok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bit-planed layers use the fused plane decode kernel on the N=1 hot path
        # (reads lo/hi planes directly, < int16 traffic); other shapes fall back to the
        # dense reconstruct path. Non-planed layers keep the standard kernel dispatch.
        if any(getattr(self, "_plane_width", ()) or ()):
            return self._forward_planed(x)
        if self._triton_available():
            from orka.inference.dispatch import vq_linear_forward
            return vq_linear_forward(self, x)
        return self._forward_python(x)

    def _forward_planed(self, x: torch.Tensor) -> torch.Tensor:
        K, M = self.in_features, self.out_features
        # Fast eager N=1 decode: warp-per-row CUDA plane GEMV (matches/beats dense cuBLAS,
        # ~9x less HBM traffic). Opt-in; transparent fallback to the custom op below.
        xr = x.reshape(-1, K)
        if xr.shape[0] == 1 and xr.is_cuda:
            # supported() (~9 checks + a module lookup) is invariant per layer, but
            # this runs once per token per layer (~19k/decode). Cache it so the N=1
            # hot path is a single attribute read, not the full eligibility probe -
            # the decode is dispatch-bound, so per-call CPU work directly costs util.
            ok = self._n1_ok
            if ok is None:
                from orka.inference import cuda_planes
                ok = self._n1_ok = bool(cuda_planes.supported(self))
            if ok:
                from orka.inference import cuda_planes
                try:
                    y = cuda_planes.forward_n1(self, xr)
                    if y is not None:
                        if self.bias is not None:
                            y = y + self.bias
                        return y.reshape(*x.shape[:-1], M).to(x.dtype)
                except Exception:
                    pass
        # Compile-traceable path: a single custom op (no graph break) for the uniform
        # 2-stage, correction-free case. torch.compile/CUDA-graphs can capture this.
        from orka.inference.plane_ops import plane_op_supported, vq_plane_linear
        if x.is_cuda and plane_op_supported(self):
            xf = x.reshape(-1, K).to(torch.float16)
            y = vq_plane_linear(
                xf,
                self.indices_lo_0, self.indices_hi_0, self.indices_lo_1, self.indices_hi_1,
                self.codebook_0, self.codebook_1, self.scales,
                M, K // self.group_size, K // self.block_size,
                self.group_size, self.block_size, self._plane_width[0] - 8,
                bool(getattr(self, "_group_major", False)),
            )
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*x.shape[:-1], M).to(x.dtype)

        # Eager fallback: layers with sparse correction / non-uniform plane widths.
        x_2d = x.reshape(-1, K)
        if x_2d.is_cuda:
            xf = x_2d.to(torch.float16)
            y = None
            try:
                from orka.inference.triton_kernels import _vq_decode_planes_n1, _vq_gemm_planes
                y = _vq_decode_planes_n1(self, xf) if xf.shape[0] == 1 else _vq_gemm_planes(self, xf)
            except Exception:
                y = None
            if y is not None:
                if self.corr_col.numel() > 0:
                    sp = self._correction_sparse()
                    y = y + torch.sparse.mm(sp, x_2d.float().T.contiguous()).T.to(torch.float16)
                if self.bias is not None:
                    y = y + self.bias
                return y.reshape(*x.shape[:-1], M).to(x.dtype)
        return self._forward_python(x)

    def _forward_python(self, x: torch.Tensor) -> torch.Tensor:
        w = self.reconstruct_weight()
        out = F.linear(x.float(), w)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        nnz = int(self.corr_col.numel())
        return (
            f"out={self.out_features}, in={self.in_features}, "
            f"stages={self.n_stages}, G={self.group_size}, B={self.block_size}, "
            f"corr_nnz={nnz}"
        )


# ------------------------------------------------------------------
# Factory: load VQLinear from a .orka artifact
# ------------------------------------------------------------------
