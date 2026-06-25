"""VQLinear: inference-time VQ layer.

Stores decoded codebooks + decompressed indices in GPU memory.
forward() dispatches to the Triton VQ-GEMM kernel when available,
falling back to pure-PyTorch reconstruct+matmul.

Weight reconstruction:
  W = W_vq + W_correction
  W_vq:        RVQ codebook decode * slrq scale    (handled by Triton kernel)
  W_correction: sparse delta matrix for salient +  (precomputed at load time;
                outlier positions                   applied as sparse matmul)

This split lets the Triton kernel focus on the dense tiled path while salient
(~1 entry per 32 weights) and outliers (~0.5% of weights) are corrected once
per forward via torch.sparse.mm, which is efficient on CUDA.

Memory per layer (group=8, block=32, cb=4096, stages=3):
  indices   [n_stages, n_groups] int16   ~2.25 B/param
  codebooks [n_stages, 4096, 8] fp16    negligible
  scales    [n_scale_blocks]    fp16    ~0.06 B/param
  correction sparse COO               ~correction_nnz * 6 B
"""

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
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.n_stages = n_stages
        self.group_size = group_size
        self.block_size = block_size
        # Per-stage codebook sizes (RVQ stages may use different cb sizes:
        # 256/896/4096 from the per-tensor allocation). int16 indices hold all.
        if isinstance(cb_sizes, int):
            cb_sizes = [cb_sizes] * n_stages
        self.cb_sizes = list(cb_sizes)
        self._triton_ok: Optional[bool] = None

        total = out_features * in_features
        n_groups = math.ceil(total / group_size)
        n_scale_blocks = math.ceil(total / block_size)

        for s in range(n_stages):
            # Index width follows the codebook size: uint8 holds [0,255] for <=256-entry
            # codebooks (8-bit stages, e.g. rvq-8-8) - half the VRAM of int16; int16 holds
            # up to 4096. The Triton kernels infer the pointer dtype, so they read either
            # transparently; the CUDA float4 path is int16-only and self-gates (see
            # cuda_decode.supported). uint8 not int8: 256 entries need [0,255] unsigned.
            idx_dtype = torch.uint8 if self.cb_sizes[s] <= 256 else torch.int16
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

    def _correction_sparse(self) -> Optional[torch.Tensor]:
        if self.corr_col.numel() == 0:
            return None
        # Wrap the stored CSR buffers as a torch CSR sparse tensor for cuSPARSE
        # CSR x dense (N>1 path). Built once and cached; rebuilt only on device change.
        cached = getattr(self, "_corr_sp_cache", None)
        if cached is not None and cached.device == self.corr_col.device:
            return cached
        sp = torch.sparse_csr_tensor(
            self.corr_rowptr.long(),
            self.corr_col.long(),
            self.corr_val.float(),
            size=(self.out_features, self.in_features),
            device=self.corr_col.device,
        )
        self._corr_sp_cache = sp
        return sp

    # ------------------------------------------------------------------
    # Weight reconstruction (for testing / fallback)
    # ------------------------------------------------------------------

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
            idxs = getattr(self, f"indices_{s}").long()
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

        # Apply correction directly from the stored CSR buffers - independent of the
        # forward path's cached sparse tensor.
        if self.corr_col.numel() > 0:
            counts = (self.corr_rowptr[1:] - self.corr_rowptr[:-1]).long()
            rows = torch.repeat_interleave(
                torch.arange(self.out_features, device=counts.device), counts
            )
            flat = rows * self.in_features + self.corr_col.long()
            mask = flat < total
            decoded[flat[mask]] += self.corr_val.float()[mask]

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
        if self._triton_available():
            from orka.inference.dispatch import vq_linear_forward
            return vq_linear_forward(self, x)
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

def _register_layer_buffers(layer, artifact_dir, stages, group_size, block_size, total, tensor_meta):
    """Load codebooks/indices (per stage) and block scales into the layer's
    registered buffers. Returns the loaded scales array (or None)."""
    import numpy as np
    from orka._format import _read_codebook, _read_indices, _read_float_vector
    from orka.transforms.normalize import stores_block_scales

    # --- Load codebooks + indices ---
    for s, stage in enumerate(stages):
        s_group = int(stage.get("group_size", group_size))
        s_n_groups = math.ceil(total / s_group)
        idxs = _read_indices(
            artifact_dir / stage["indices"],
            int(stage["index_bits"]),
            s_n_groups,
            packed=bool(stage.get("packed", False)),
            encoding=stage.get("encoding", "raw"),
        )
        cb = _read_codebook(
            artifact_dir / stage["codebook"],
            s_group,
            stage.get("codebook_dtype", "float16"),
        )
        idx_buf = getattr(layer, f"indices_{s}")
        idx_buf.copy_(torch.from_numpy(np.ascontiguousarray(idxs).copy()).to(idx_buf.dtype))
        getattr(layer, f"codebook_{s}").copy_(torch.from_numpy(cb).to(torch.float16))

    # --- Load scales ---
    norm = tensor_meta.get("normalization", "none")
    scales_np = None
    if stores_block_scales(norm):
        scale_dtype = tensor_meta.get("scale_dtype") or "float32"
        n_scale = math.ceil(total / block_size)
        scales_full = _read_float_vector(artifact_dir / tensor_meta["scales"], int(tensor_meta["scale_count"]), scale_dtype)
        scales_np = scales_full[:n_scale]
        layer.scales.copy_(torch.from_numpy(scales_np).to(torch.float16))

    return scales_np


def _build_csr_correction(layer, artifact_dir, tensor_meta, n_stages, group_size, block_size, scales_np, in_features, out_features, total):
    """Precompute sparse correction (salient + outliers) and store it into the
    layer's CSR buffers.

    The Triton kernel produces W_vq = vq_decode * block_scale at every position.
    The artifact overrides some positions; we store delta = final - W_vq so
    forward() applies them with one sparse matmul.

    orka decode ordering (must be matched exactly):
      1. vq decode
      2. outliers OVERWRITE (pre-scale)  -> final_outlier = outlier_val * block_scale
      3. block scaling                   -> scales everything incl. outliers
      4. salient OVERWRITE (post-scale)  -> final_salient = salient_val  (verbatim)
    Salient is applied last, so at a shared position salient wins. We dedup
    into a position->final-value map with salient priority (NOT summed).
    """
    import numpy as np
    from orka._format import _read_salient, _read_outliers

    def _vq_decoded_at(flat_pos_np):
        """W_vq = vq_decode * block_scale at given flat positions."""
        flat_t = torch.from_numpy(flat_pos_np.astype(np.int64))
        group_ids = flat_t // group_size
        within_g = flat_t % group_size
        dec = torch.zeros(len(flat_t), dtype=torch.float32)
        for s in range(n_stages):
            idxs_s = getattr(layer, f"indices_{s}").long()[group_ids]
            cb_s = getattr(layer, f"codebook_{s}").float()
            dec += cb_s[idxs_s, within_g]
        if scales_np is not None:
            scale_ids = (flat_t // block_size).numpy()
            dec *= torch.from_numpy(scales_np[scale_ids])
        return dec.numpy()

    def _scale_at(flat_pos_np):
        if scales_np is None:
            return np.ones(len(flat_pos_np), dtype=np.float32)
        return scales_np[(flat_pos_np // block_size).astype(np.int64)]

    # position -> final value (salient written after outliers => wins)
    final_vals: dict[int, float] = {}

    outl = tensor_meta.get("outliers")
    if outl and outl.get("count", 0) > 0:
        positions, values = _read_outliers(
            artifact_dir / outl["positions"],
            artifact_dir / outl["values"],
            int(outl["count"]),
            outl.get("positions_dtype", "uint32"),
            outl.get("values_dtype", "float32"),
        )
        flat_pos = positions.astype(np.int64)
        mask = flat_pos < total
        fp = flat_pos[mask]
        # outlier final value = stored value * block scale (it was overwritten pre-scale)
        outl_final = values[mask] * _scale_at(fp)
        for p, v in zip(fp.tolist(), outl_final.tolist()):
            final_vals[p] = v

    salient = tensor_meta.get("salient")
    if salient and salient.get("count", 0) > 0:
        s_idx_raw, s_val_raw = _read_salient(
            artifact_dir / salient["indices"],
            artifact_dir / salient["weights"],
            int(salient["count"]),
            int(salient["indices_bits"]),
            salient.get("weights_dtype", "float32"),
        )
        n_scale = math.ceil(total / block_size)
        s_idx = s_idx_raw[:n_scale].astype(np.int64)
        s_val = s_val_raw[:n_scale]
        b_ids = np.arange(len(s_idx), dtype=np.int64)
        flat_pos = b_ids * block_size + s_idx
        mask = flat_pos < total
        fp = flat_pos[mask]
        # salient final value = stored value verbatim (overwritten post-scale)
        for p, v in zip(fp.tolist(), s_val[mask].tolist()):
            final_vals[p] = v  # salient overwrites any outlier at same pos

    if final_vals:
        pos_arr = np.fromiter(final_vals.keys(), dtype=np.int64, count=len(final_vals))
        fin_arr = np.fromiter(final_vals.values(), dtype=np.float32, count=len(final_vals))
        vq_at = _vq_decoded_at(pos_arr)
        delta = fin_arr - vq_at
        rows = (pos_arr // in_features).astype(np.int64)
        cols = (pos_arr % in_features).astype(np.int64)
        # CSR: sort by row, build rowptr, store col (sorted) + val (fp16).
        order = np.argsort(rows, kind="stable")
        cols_s = cols[order].astype(np.int32)
        delta_s = delta[order].astype(np.float16)
        rowptr = np.zeros(out_features + 1, dtype=np.int64)
        rowptr[1:] = np.cumsum(np.bincount(rows, minlength=out_features))
        layer.corr_rowptr.copy_(torch.from_numpy(rowptr.astype(np.int32)))
        layer.corr_col.resize_(len(cols_s)).copy_(torch.from_numpy(cols_s))
        layer.corr_val.resize_(len(delta_s)).copy_(torch.from_numpy(delta_s))


def _to_group_major(layer, n_stages, group_size, block_size, in_features, out_features):
    """Transpose index/scale buffers to group-major ([GPR,M] / [BPR,M]) for
    coalesced kernel reads.

    Done LAST: the correction delta above reads row-major indices. In-place copy_
    keeps the buffers registered; the transient transpose copy is freed at load
    time (not during inference), so the inference footprint holds only one layout.
    """
    if in_features % group_size == 0 and in_features % block_size == 0:
        gpr = in_features // group_size
        bpr = in_features // block_size
        ok = True
        for s in range(n_stages):
            buf = getattr(layer, f"indices_{s}")
            if buf.numel() != out_features * gpr:
                ok = False
                break
        if ok and layer.scales.numel() == out_features * bpr:
            for s in range(n_stages):
                buf = getattr(layer, f"indices_{s}")
                buf.copy_(buf.view(out_features, gpr).t().contiguous().reshape(-1))
            layer.scales.copy_(layer.scales.view(out_features, bpr).t().contiguous().reshape(-1))
            layer._group_major = True


def build_vq_linear(
    artifact_dir: Path,
    tensor_meta: dict,
    bias: Optional[torch.Tensor],
    device: str | torch.device = "cpu",
) -> VQLinear:
    """Construct and populate a VQLinear from one tensor's .orka metadata."""
    import numpy as np
    from orka._format import (
        _read_codebook, _read_indices, _read_salient, _read_outliers,
        _read_float_vector, _float_value_dtype,
    )

    shape = [int(x) for x in tensor_meta["shape"]]
    out_features = shape[0]
    in_features = 1
    for s in shape[1:]:
        in_features *= s

    stages = tensor_meta.get("stages") or [{
        "codebook": tensor_meta["codebook"],
        "codebook_size": int(tensor_meta["codebook_size"]),
        "index_bits": int(tensor_meta["index_bits"]),
        "indices": tensor_meta["indices"],
    }]

    group_size = int(tensor_meta.get("group_size", 8))
    block_size = int(tensor_meta.get("block_scale_size") or 32)
    cb_sizes = [int(st["codebook_size"]) for st in stages]
    n_stages = len(stages)
    total = out_features * in_features

    layer = VQLinear(
        out_features=out_features,
        in_features=in_features,
        n_stages=n_stages,
        group_size=group_size,
        block_size=block_size,
        cb_sizes=cb_sizes,
        bias=bias,
    )

    scales_np = _register_layer_buffers(
        layer, artifact_dir, stages, group_size, block_size, total, tensor_meta
    )

    _build_csr_correction(
        layer, artifact_dir, tensor_meta, n_stages, group_size, block_size,
        scales_np, in_features, out_features, total,
    )

    _to_group_major(layer, n_stages, group_size, block_size, in_features, out_features)

    return layer.to(device)
