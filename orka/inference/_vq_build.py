"""Build a VQLinear from a .orka tensor (buffer loading, CSR correction, group-major)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from orka.inference._vq_core import VQLinear


def _register_layer_buffers(layer, artifact_dir, stages, group_size, block_size, total, tensor_meta):
    """Load codebooks/indices (per stage) and block scales into the layer's
    registered buffers. Returns the loaded scales array (or None)."""
    import numpy as np
    from orka.core._format import _read_codebook, _read_indices, _read_float_vector
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
        width = layer._plane_width[s] if s < len(layer._plane_width) else 0
        if width:
            from orka.core._format import _pack_index_planes
            lo, hi = _pack_index_planes(idxs, width)
            getattr(layer, f"indices_lo_{s}").copy_(torch.from_numpy(lo))
            getattr(layer, f"indices_hi_{s}").copy_(torch.from_numpy(hi))
        else:
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
    from orka.core._format import _read_salient, _read_outliers

    def _vq_decoded_at(flat_pos_np):
        """W_vq = vq_decode * block_scale at given flat positions."""
        flat_t = torch.from_numpy(flat_pos_np.astype(np.int64))
        group_ids = flat_t // group_size
        within_g = flat_t % group_size
        dec = torch.zeros(len(flat_t), dtype=torch.float32)
        for s in range(n_stages):
            idxs_s = layer._stage_indices_int(s)[group_ids]
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
        # C-level dict build (same insertion order + last-write-wins as a python loop,
        # but no per-element bytecode - matters when corrections number 100k+).
        final_vals.update(zip(fp.tolist(), outl_final.tolist()))

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
        # salient final value = stored value verbatim (overwritten post-scale);
        # update() overwrites any outlier at the same position (salient wins).
        final_vals.update(zip(fp.tolist(), s_val[mask].tolist()))

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
    # Bit-planed tensors stay row-major (the packed high plane cannot be transposed in
    # place); they decode through the dense reconstruct path, which handles row-major.
    if any(getattr(layer, "_plane_width", ()) or ()):
        return
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
    from orka.core._format import (
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
