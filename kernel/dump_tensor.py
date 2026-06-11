"""Dump one real Orka tensor's decode inputs as a flat f32 blob + JSON header,
plus a random activation matrix and the reference matmul result. The C ggml
harness reconstructs the weight inside a ggml custom op and must reproduce the
matmul byte-for-byte (to f32 precision).

The blob is laid out as a single contiguous f32 array so it can live in one
ggml tensor - mirroring how a GGUF ORKA_VQ weight would carry codebooks +
indices + scales + sidecars in its backend buffer.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orka._format import _read_indices, _read_outliers, _read_salient  # noqa: E402
from orka.pipeline.decode import (  # noqa: E402
    _decode_tensor,
    _read_codebook,
    _read_float_vector,
    _read_lowrank,
)


def dump(artifact: Path, tensor_name: str | None, out_prefix: Path, n_cols_out: int = 4):
    manifest = json.loads((artifact / "manifest.json").read_text())
    tensors = manifest["tensors"]
    tm = next((t for t in tensors if t["name"] == tensor_name), None) if tensor_name else tensors[1]
    if tm is None:
        raise SystemExit(f"tensor not found: {tensor_name}")

    shape = [int(x) for x in tm["shape"]]
    rows = shape[0]
    cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    group_size = int(tm["group_size"])
    padded = int(tm["padded_values"])
    packed = int(tm["packed_values"])

    blob_parts = []
    meta = {
        "name": tm["name"], "rows": rows, "cols": cols,
        "group_size": group_size, "padded_values": padded, "packed_values": packed,
        "stages": [],
    }
    offset = 0

    def add(arr):
        nonlocal offset
        arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        blob_parts.append(arr)
        off = offset
        offset += arr.shape[0]
        return off, int(arr.shape[0])

    def add_i32(arr):
        # Integer payloads that can exceed 2^24 (e.g. absolute outlier
        # positions) lose precision as float32. Store the int32 bit pattern in
        # the f32 blob and reinterpret on the C side.
        nonlocal offset
        arr = np.ascontiguousarray(arr, dtype=np.int32).view(np.float32).reshape(-1)
        blob_parts.append(arr)
        off = offset
        offset += arr.shape[0]
        return off, int(arr.shape[0])

    stages_meta = tm.get("stages")
    for st in stages_meta:
        g = int(st.get("group_size", group_size))
        s_count = math.ceil(padded / g)
        cb = _read_codebook(artifact / st["codebook"], g, st.get("codebook_dtype", "float32"))
        idx = np.asarray(
            _read_indices(
                artifact / st["indices"], int(st["index_bits"]), s_count,
                packed=bool(st.get("packed", False)), encoding=st.get("encoding", "raw"),
            ),
            dtype=np.float32,
        )
        cb_off, cb_len = add(cb.reshape(-1))
        idx_off, idx_len = add(idx)
        meta["stages"].append({
            "group_size": g, "codebook_size": int(cb.shape[0]),
            "cb_off": cb_off, "idx_off": idx_off, "idx_count": s_count,
        })

    norm = tm.get("normalization", "none")
    meta["block_scale_size"] = 0
    if norm in ("block-max", "channel-block-max", "slrq-block", "awq-block-max") and tm.get("scales"):
        scales = _read_float_vector(artifact / tm["scales"], int(tm["scale_count"]),
                                    tm.get("scale_dtype") or "float32")
        off, n = add(scales)
        meta["block_scale_size"] = int(tm.get("block_scale_size") or 32)
        meta["scale_off"] = off
        meta["scale_count"] = n

    meta["outlier_count"] = 0
    outl = tm.get("outliers")
    if outl:
        pos, val = _read_outliers(artifact / outl["positions"], artifact / outl["values"],
                                  outl.get("positions_dtype", "uint32"), outl.get("values_dtype", "float32"))
        if pos.size:
            off_p, _ = add_i32(pos.astype(np.int64))  # absolute positions, may exceed 2^24
            off_v, _ = add(val)
            meta.update(outlier_count=int(pos.shape[0]), outlier_pos_off=off_p, outlier_val_off=off_v)

    meta["salient_count"] = 0
    sal = tm.get("salient")
    if sal:
        s_idx, s_val = _read_salient(artifact / sal["indices"], artifact / sal["weights"],
                                     sal.get("indices_dtype", "uint32"), sal.get("weights_dtype", "float32"))
        if s_idx.size:
            off_i, _ = add(s_idx.astype(np.float32))
            off_v, _ = add(s_val)
            meta.update(salient_count=int(s_idx.shape[0]), salient_idx_off=off_i, salient_val_off=off_v)

    meta["lowrank_rank"] = 0
    lr = tm.get("lowrank")
    if lr:
        a, b = _read_lowrank(artifact, lr)
        off_a, _ = add(a.reshape(-1))
        off_b, _ = add(b.reshape(-1))
        meta.update(lowrank_rank=int(lr["rank"]), lowrank_a_off=off_a, lowrank_b_off=off_b)

    blob = np.concatenate(blob_parts).astype("<f4")

    # Reference: decode W exactly, GEMM with a fixed-seed random activation.
    W = np.asarray(_decode_tensor(artifact, tm), dtype=np.float32).reshape(rows, cols)
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((cols, n_cols_out)).astype(np.float32)
    y_ref = (W @ x).astype("<f4")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    blob.tofile(str(out_prefix) + ".blob")
    # ggml stores [cols, x_cols] column-major: element (c, j) at j*cols + c.
    # That equals the C-order flatten of x.T, so dump the transpose.
    np.ascontiguousarray(x.T, dtype="<f4").tofile(str(out_prefix) + ".x")
    y_ref.tofile(str(out_prefix) + ".yref")
    meta["blob_len"] = int(blob.shape[0])
    meta["x_cols"] = n_cols_out
    Path(str(out_prefix) + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"dumped {tm['name']}: rows={rows} cols={cols} blob_floats={blob.shape[0]} x={x.shape} -> {out_prefix}.*")


if __name__ == "__main__":
    artifact = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/orka_ggml/t")
    dump(artifact, name, out)
