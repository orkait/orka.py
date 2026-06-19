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
            # int16 holds [0, 4095] (max cb_size=4096) without overflow
            self.register_buffer(f"indices_{s}", torch.zeros(n_groups, dtype=torch.int16))
            self.register_buffer(f"codebook_{s}", torch.zeros(self.cb_sizes[s], group_size, dtype=torch.float16))

        self.register_buffer("scales", torch.ones(n_scale_blocks, dtype=torch.float16))

        # Sparse correction: W_correction in COO format [out, in].
        # Encodes both salient and outlier deltas precomputed at load time.
        # Stored as indices [2, nnz] + values [nnz] fp32.
        self.register_buffer("corr_indices", torch.zeros(2, 0, dtype=torch.int32))
        self.register_buffer("corr_values", torch.zeros(0, dtype=torch.float32))

        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16))
        else:
            self.register_buffer("bias", None)

    # ------------------------------------------------------------------
    # Correction sparse tensor (rebuilt on first forward or after .to())
    # ------------------------------------------------------------------

    def _correction_sparse(self) -> Optional[torch.Tensor]:
        if self.corr_indices.numel() == 0:
            return None
        # Cache the coalesced sparse tensor; rebuild only if device changed.
        cached = getattr(self, "_corr_sp_cache", None)
        if cached is not None and cached.device == self.corr_indices.device:
            return cached
        sp = torch.sparse_coo_tensor(
            self.corr_indices.long(),
            self.corr_values,
            size=(self.out_features, self.in_features),
            device=self.corr_indices.device,
        ).coalesce()
        self._corr_sp_cache = sp
        return sp

    # ------------------------------------------------------------------
    # Weight reconstruction (for testing / fallback)
    # ------------------------------------------------------------------

    def reconstruct_weight(self) -> torch.Tensor:
        """Decode full W [out, in] fp32. Expensive - for testing only."""
        dev = self.scales.device
        G, B = self.group_size, self.block_size
        total = self.out_features * self.in_features
        padded = math.ceil(total / G) * G

        decoded = torch.zeros(padded, dtype=torch.float32, device=dev)
        for s in range(self.n_stages):
            idxs = getattr(self, f"indices_{s}").long()
            cb = getattr(self, f"codebook_{s}").float()
            decoded.add_(cb[idxs].reshape(-1))
        decoded = decoded[:total]

        n_blocks = math.ceil(total / B)
        pad_b = n_blocks * B - total
        if pad_b:
            decoded = F.pad(decoded, (0, pad_b))
        decoded = (decoded.reshape(n_blocks, B) * self.scales[:n_blocks, None].float()).reshape(-1)[:total]

        # Apply sparse correction
        sp = self._correction_sparse()
        if sp is not None:
            rows = sp.indices()[0]
            cols = sp.indices()[1]
            vals = sp.values()
            flat = rows * self.in_features + cols
            mask = flat < total
            decoded[flat[mask]] += vals[mask]

        return decoded.reshape(self.out_features, self.in_features)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _triton_available(self) -> bool:
        if self._triton_ok is None:
            try:
                from orka.inference.kernel import vq_linear_forward  # noqa: F401
                self._triton_ok = True
            except Exception:
                self._triton_ok = False
        return self._triton_ok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._triton_available():
            from orka.inference.kernel import vq_linear_forward
            return vq_linear_forward(self, x)
        return self._forward_python(x)

    def _forward_python(self, x: torch.Tensor) -> torch.Tensor:
        w = self.reconstruct_weight()
        out = F.linear(x.float(), w)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        nnz = self.corr_indices.shape[1] if self.corr_indices.numel() else 0
        return (
            f"out={self.out_features}, in={self.in_features}, "
            f"stages={self.n_stages}, G={self.group_size}, B={self.block_size}, "
            f"corr_nnz={nnz}"
        )


# ------------------------------------------------------------------
# Factory: load VQLinear from a .orka artifact
# ------------------------------------------------------------------

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
        _float_value_dtype,
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
        getattr(layer, f"indices_{s}").copy_(torch.from_numpy(idxs.astype(np.int16)))
        getattr(layer, f"codebook_{s}").copy_(torch.from_numpy(cb).to(torch.float16))

    # --- Load scales ---
    norm = tensor_meta.get("normalization", "none")
    scales_np = None
    if norm in ("slrq-block", "block-max", "channel-block-max", "awq-block-max"):
        scale_dtype = tensor_meta.get("scale_dtype") or "float32"
        scales_raw = np.fromfile(
            str(artifact_dir / tensor_meta["scales"]),
            dtype=_float_value_dtype(scale_dtype),
        ).astype(np.float32)
        n_scale = math.ceil(total / block_size)
        scales_np = scales_raw[:n_scale]
        layer.scales.copy_(torch.from_numpy(scales_np).to(torch.float16))

    # --- Precompute sparse correction (salient + outliers) ---
    # The Triton kernel produces W_vq = vq_decode * block_scale at every position.
    # The artifact overrides some positions; we store delta = final - W_vq so
    # forward() applies them with one sparse matmul.
    #
    # orka decode ordering (must be matched exactly):
    #   1. vq decode
    #   2. outliers OVERWRITE (pre-scale)  -> final_outlier = outlier_val * block_scale
    #   3. block scaling                   -> scales everything incl. outliers
    #   4. salient OVERWRITE (post-scale)  -> final_salient = salient_val  (verbatim)
    # Salient is applied last, so at a shared position salient wins. We dedup
    # into a position->final-value map with salient priority (NOT summed).
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
            salient.get("indices_dtype", "uint32"),
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
        rows = (pos_arr // in_features).astype(np.int32)
        cols = (pos_arr % in_features).astype(np.int32)
        layer.corr_indices.resize_(2, len(rows))
        layer.corr_indices[0].copy_(torch.from_numpy(rows))
        layer.corr_indices[1].copy_(torch.from_numpy(cols))
        layer.corr_values.resize_(len(delta))
        layer.corr_values.copy_(torch.from_numpy(delta))

    return layer.to(device)
