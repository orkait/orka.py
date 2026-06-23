"""Per-tensor decode: numpy default + torch GPU streaming variant."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from orka._format import (
    _float_value_dtype,
    _read_codebook,
    _read_float_vector,
    _read_indices,
    _read_outliers,
    _read_pillars,
    _read_salient,
)
from orka.transforms.normalize import (
    _apply_block_max_scales_numpy,
    _apply_col_l2_scales_numpy,
    stores_block_scales,
)
from orka.transforms.rotate import (
    _block_fwht_torch,
    _generate_orthogonal_numpy,
    _hadamard_block_size,
    _unrotate_flat,
)


def _read_lowrank(out_dir: Path, lr_meta: dict):
    import numpy as np

    dtype = _float_value_dtype(lr_meta.get("dtype", "float16"))
    rank = int(lr_meta["rank"])
    a = np.fromfile(str(out_dir / lr_meta["a"]), dtype=dtype).astype(np.float32)
    b = np.fromfile(str(out_dir / lr_meta["b"]), dtype=dtype).astype(np.float32)
    return a.reshape(-1, rank), b.reshape(-1, rank)


def _apply_lowrank_numpy(decoded, shape, lr_meta, out_dir: Path):
    import numpy as np

    rows = int(shape[0])
    cols = 1
    for s in shape[1:]:
        cols *= int(s)
    a, b = _read_lowrank(out_dir, lr_meta)
    mat = np.asarray(decoded, dtype=np.float32)[: rows * cols].reshape(rows, cols)
    return (mat + a @ b.T).reshape(-1)


def _resolve_stages(tm: dict, group_size: int):
    """Backend-agnostic stage list: explicit stages, else single-stage fallback.

    Identical control flow for the numpy and torch decode paths; only the
    numeric accumulation differs per backend.
    """
    stages = tm.get("stages")
    if not stages:
        stages = [
            {
                "codebook": tm["codebook"],
                "codebook_size": int(tm["codebook_size"]),
                "index_bits": int(tm["index_bits"]),
                "indices": tm["indices"],
            }
        ]
    return stages


def _read_stage_arrays(out_dir: Path, stage: dict, group_size: int, padded_values: int):
    """Backend-agnostic per-stage read: returns (codebook, indices) numpy arrays.

    The stage metadata math (s_group_size, s_index_count) and the codebook /
    indices I/O are byte-identical regardless of backend; each backend gathers
    ``codebook[indices]`` in its own numeric primitives.
    """
    s_group_size = int(stage.get("group_size", group_size))
    s_index_count = math.ceil(padded_values / s_group_size)
    cb = _read_codebook(
        out_dir / stage["codebook"],
        s_group_size,
        stage.get("codebook_dtype", "float32"),
    )
    idxs = _read_indices(
        out_dir / stage["indices"], int(stage["index_bits"]), s_index_count,
        packed=bool(stage.get("packed", False)),
        encoding=stage.get("encoding", "raw"),
    )
    return cb, idxs


def _decode_tensor(out_dir: Path, tensor_meta: dict):
    import numpy as np

    group_size = int(tensor_meta["group_size"])
    padded_values = int(tensor_meta["padded_values"])
    packed_values = int(tensor_meta["packed_values"])
    index_count = math.ceil(padded_values / group_size)
    stages = _resolve_stages(tensor_meta, group_size)

    decoded_np = np.zeros(index_count * group_size, dtype=np.float32)
    for stage in stages:
        cb, idxs = _read_stage_arrays(out_dir, stage, group_size, padded_values)
        decoded_np += cb[idxs.astype(np.int64, copy=False)].reshape(-1)

    decoded = decoded_np[:packed_values].copy()

    outl = tensor_meta.get("outliers")
    if outl:
        positions, values = _read_outliers(
            out_dir / outl["positions"],
            out_dir / outl["values"],
            int(outl["count"]),
            outl.get("positions_dtype", "uint32"),
            outl.get("values_dtype", "float32"),
        )
        if positions.size:
            decoded[positions.astype(np.int64, copy=False)] = values

    pillars = tensor_meta.get("pillars")
    if pillars:
        positions, values = _read_pillars(
            out_dir / pillars["positions"], out_dir / pillars["values"]
        )
        if positions.size:
            decoded[positions.astype(np.int64, copy=False)] = values

    rotation = tensor_meta.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tensor_meta.get("rotation_seed") or 0)
        decoded = _unrotate_flat(decoded, tensor_meta["shape"], rotation, seed)

    norm = tensor_meta.get("normalization", "none")
    scale_dtype = tensor_meta.get("scale_dtype") or "float32"
    if norm == "awq":
        scales = _read_float_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"]), scale_dtype
        )
        decoded = _apply_col_l2_scales_numpy(decoded, tensor_meta["shape"], scales)
    # block-scales-only inverse: awq-block-max is NOT here - it needs block + col scales
    # and is handled in its own branch below (this is the narrower grouping, not
    # stores_block_scales()).
    elif norm in ("block-max", "channel-block-max", "slrq-block"):
        scales = _read_float_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"]), scale_dtype
        )
        block_size = int(tensor_meta.get("block_scale_size") or 32)
        decoded = _apply_block_max_scales_numpy(decoded, scales, block_size)
    elif norm == "awq-block-max":
        block_scales = _read_float_vector(
            out_dir / tensor_meta["scales"], int(tensor_meta["scale_count"]), scale_dtype
        )
        block_size = int(tensor_meta.get("block_scale_size") or 32)
        decoded = _apply_block_max_scales_numpy(decoded, block_scales, block_size)
        awq_meta = tensor_meta.get("awq_col_scales")
        if awq_meta:
            awq_scales = _read_float_vector(
                out_dir / awq_meta["path"], int(awq_meta["count"]),
                awq_meta.get("dtype") or "float32",
            )
            decoded = _apply_col_l2_scales_numpy(decoded, tensor_meta["shape"], awq_scales)

    salient = tensor_meta.get("salient")
    if salient:
        s_idx, s_val = _read_salient(
            out_dir / salient["indices"],
            out_dir / salient["weights"],
            int(salient["count"]),
            int(salient["indices_bits"]),
            salient.get("weights_dtype", "float32"),
        )
        if s_idx.size:
            block_size = int(tensor_meta.get("block_scale_size") or 32)
            b_count = s_idx.shape[0]
            flat_indices = np.arange(b_count, dtype=np.int64) * block_size + s_idx.astype(np.int64, copy=False)
            mask = flat_indices < decoded.shape[0]
            decoded[flat_indices[mask]] = s_val[mask]

    lr_meta = tensor_meta.get("lowrank")
    if lr_meta:
        decoded = _apply_lowrank_numpy(decoded, tensor_meta["shape"], lr_meta, out_dir)

    return decoded


def _decode_tensor_torch(out_dir: Path, tm: dict, device: str):
    """Decode a single quantized tensor on GPU, return torch tensor in original shape."""
    import torch
    import numpy as np

    group_size = int(tm["group_size"])
    padded_values = int(tm["padded_values"])
    packed_values = int(tm["packed_values"])
    index_count = math.ceil(padded_values / group_size)
    shape = [int(x) for x in tm["shape"]]

    stages = _resolve_stages(tm, group_size)

    decoded = torch.zeros(index_count * group_size, dtype=torch.float32, device=device)
    for stage in stages:
        cb_np, idxs_raw = _read_stage_arrays(out_dir, stage, group_size, padded_values)
        idxs_np = np.asarray(idxs_raw, dtype=np.int64)
        cb = torch.from_numpy(cb_np).to(device)
        idxs = torch.from_numpy(idxs_np).to(device)
        decoded.add_(cb[idxs].reshape(-1))
    decoded = decoded[:packed_values]

    outl = tm.get("outliers")
    if outl:
        positions, values = _read_outliers(
            out_dir / outl["positions"],
            out_dir / outl["values"],
            int(outl["count"]),
            outl.get("positions_dtype", "uint32"),
            outl.get("values_dtype", "float32"),
        )
        if positions.size:
            pos_t = torch.from_numpy(positions.astype(np.int64, copy=False)).to(device)
            val_t = torch.from_numpy(values).to(device)
            decoded[pos_t] = val_t

    pillars = tm.get("pillars")
    if pillars:
        positions, values = _read_pillars(out_dir / pillars["positions"], out_dir / pillars["values"])
        if positions.size:
            pos_t = torch.from_numpy(positions.astype(np.int64, copy=False)).to(device)
            val_t = torch.from_numpy(values).to(device)
            decoded[pos_t] = val_t

    rotation = tm.get("rotation", "none")
    if rotation in {"orthogonal", "hadamard"}:
        seed = int(tm.get("rotation_seed") or 0)
        rows = shape[0]
        cols = 1
        for s in shape[1:]:
            cols *= int(s)
        arr = decoded[:rows * cols].reshape(rows, cols)
        if rotation == "hadamard":
            block_size = _hadamard_block_size(cols)
            unrotated = _block_fwht_torch(arr, block_size)
        else:
            q = torch.from_numpy(_generate_orthogonal_numpy(cols, seed)).to(device)
            unrotated = arr @ q.T
        decoded = unrotated.reshape(-1)

    norm = tm.get("normalization", "none")
    scale_np_dtype = _float_value_dtype(tm.get("scale_dtype") or "float32")
    # torch path: block scales for all four block-family modes; awq-block-max applies an
    # extra col-scale step inside this branch (so it stays the all-four grouping here).
    if stores_block_scales(norm):
        scales = _read_float_vector(out_dir / tm["scales"], int(tm["scale_count"]), tm.get("scale_dtype") or "float32")
        block_size = int(tm.get("block_scale_size") or 32)
        scales_t = torch.from_numpy(scales).to(device)
        n = decoded.numel()
        pad = (-n) % block_size
        if pad:
            decoded = torch.cat([decoded, torch.zeros(pad, dtype=torch.float32, device=device)])
        decoded = (decoded.reshape(-1, block_size) * scales_t[:decoded.numel() // block_size, None]).reshape(-1)
        if pad:
            decoded = decoded[:n]
        if norm == "awq-block-max":
            awq_meta = tm.get("awq_col_scales")
            if awq_meta:
                awq_scales = _read_float_vector(out_dir / awq_meta["path"], int(awq_meta["count"]), awq_meta.get("dtype") or "float32")
                awq_t = torch.from_numpy(awq_scales).to(device)
                cols = shape[-1]
                rows = decoded.numel() // cols
                decoded = (decoded[:rows * cols].reshape(rows, cols) * awq_t[None, :]).reshape(-1)
    elif norm == "awq":
        scales = _read_float_vector(out_dir / tm["scales"], int(tm["scale_count"]), tm.get("scale_dtype") or "float32")
        scales_t = torch.from_numpy(scales).to(device)
        cols = scales_t.numel()
        rows = decoded.numel() // cols
        decoded = (decoded[:rows * cols].reshape(rows, cols) * scales_t[None, :]).reshape(-1)

    salient = tm.get("salient")
    if salient:
        s_idx_np, s_val_np = _read_salient(
            out_dir / salient["indices"],
            out_dir / salient["weights"],
            int(salient["count"]),
            int(salient["indices_bits"]),
            salient.get("weights_dtype", "float32"),
        )
        
        s_idx = torch.from_numpy(s_idx_np.astype(np.int64)).to(device)
        s_val = torch.from_numpy(s_val_np).to(device)
        
        # SLRQ: re-inject salient weights AFTER scaling to avoid double-scaling.
        block_size = int(tm.get("block_scale_size") or 32)
        b_count = len(s_idx)
        b_indices = torch.arange(b_count, device=device)
        flat_indices = b_indices * block_size + s_idx
        
        # Guard against padding
        mask = flat_indices < decoded.numel()
        decoded[flat_indices[mask]] = s_val[mask]

    lr_meta = tm.get("lowrank")
    if lr_meta:
        a_np, b_np = _read_lowrank(out_dir, lr_meta)
        a_t = torch.from_numpy(a_np).to(device)
        b_t = torch.from_numpy(b_np).to(device)
        rows = shape[0]
        cols = decoded.numel() // rows
        decoded = (
            decoded[: rows * cols].reshape(rows, cols) + a_t @ b_t.T
        ).reshape(-1)

    return decoded.reshape(shape)
