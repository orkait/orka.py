"""Byte-exact verification of the ORKA_VQ C kernel against the Python decoder.

Loads real packed tensors from an .orka artifact, decodes each one twice -
once through orka.pipeline.decode._decode_tensor (the reference), once through
the C kernel via ctypes - and asserts the two agree to f32 precision. The
kernel receives already-unpacked indices and f32 codebooks/scales, exactly
the inputs a GGML loader would hand it after stream decode.
"""

from __future__ import annotations

import ctypes
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orka._format import _read_indices, _read_outliers, _read_salient  # noqa: E402
from orka.pipeline.decode import (  # noqa: E402
    _apply_block_max_scales_numpy,  # noqa: F401 (kept for parity reference)
    _decode_tensor,
    _read_codebook,
    _read_float_vector,
    _read_lowrank,
)


class _Stage(ctypes.Structure):
    _fields_ = [
        ("group_size", ctypes.c_int32),
        ("codebook_size", ctypes.c_int32),
        ("codebook", ctypes.POINTER(ctypes.c_float)),
        ("indices", ctypes.POINTER(ctypes.c_int32)),
    ]


class _Tensor(ctypes.Structure):
    _fields_ = [
        ("packed_values", ctypes.c_int64),
        ("padded_values", ctypes.c_int64),
        ("rows", ctypes.c_int32),
        ("cols", ctypes.c_int32),
        ("n_stages", ctypes.c_int32),
        ("stages", ctypes.POINTER(_Stage)),
        ("block_scale_size", ctypes.c_int32),
        ("scale_count", ctypes.c_int32),
        ("scales", ctypes.POINTER(ctypes.c_float)),
        ("outlier_count", ctypes.c_int32),
        ("outlier_pos", ctypes.POINTER(ctypes.c_int64)),
        ("outlier_val", ctypes.POINTER(ctypes.c_float)),
        ("salient_count", ctypes.c_int32),
        ("salient_idx", ctypes.POINTER(ctypes.c_int32)),
        ("salient_val", ctypes.POINTER(ctypes.c_float)),
        ("lowrank_rank", ctypes.c_int32),
        ("lowrank_a", ctypes.POINTER(ctypes.c_float)),
        ("lowrank_b", ctypes.POINTER(ctypes.c_float)),
    ]


def _fptr(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i32ptr(arr):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _i64ptr(arr):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def kernel_decode(lib, out_dir: Path, tm: dict) -> np.ndarray:
    keep = []  # hold numpy arrays alive for the duration of the C call
    group_size = int(tm["group_size"])
    padded = int(tm["padded_values"])
    packed = int(tm["packed_values"])

    stages_meta = tm.get("stages") or [
        {
            "codebook": tm["codebook"],
            "index_bits": int(tm["index_bits"]),
            "indices": tm["indices"],
            "group_size": group_size,
        }
    ]
    c_stages = (_Stage * len(stages_meta))()
    for i, st in enumerate(stages_meta):
        g = int(st.get("group_size", group_size))
        s_count = math.ceil(padded / g)
        cb = _read_codebook(out_dir / st["codebook"], g, st.get("codebook_dtype", "float32"))
        idx = np.asarray(
            _read_indices(
                out_dir / st["indices"], int(st["index_bits"]), s_count,
                packed=bool(st.get("packed", False)),
                encoding=st.get("encoding", "raw"),
            ),
            dtype=np.int32,
        )
        cb_arr, cb_ptr = _fptr(cb.reshape(-1))
        idx_arr, idx_ptr = _i32ptr(idx)
        keep += [cb_arr, idx_arr]
        c_stages[i].group_size = g
        c_stages[i].codebook_size = int(cb.shape[0])
        c_stages[i].codebook = cb_ptr
        c_stages[i].indices = idx_ptr

    t = _Tensor()
    t.packed_values = packed
    t.padded_values = padded
    shape = [int(x) for x in tm["shape"]]
    t.rows = shape[0]
    t.cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    t.n_stages = len(stages_meta)
    t.stages = c_stages

    norm = tm.get("normalization", "none")
    if norm in ("block-max", "channel-block-max", "slrq-block", "awq-block-max") and tm.get("scales"):
        scales = _read_float_vector(
            out_dir / tm["scales"], int(tm["scale_count"]), tm.get("scale_dtype") or "float32"
        )
        sc_arr, sc_ptr = _fptr(scales)
        keep.append(sc_arr)
        t.block_scale_size = int(tm.get("block_scale_size") or 32)
        t.scale_count = int(scales.shape[0])
        t.scales = sc_ptr

    outl = tm.get("outliers")
    if outl:
        pos, val = _read_outliers(
            out_dir / outl["positions"], out_dir / outl["values"],
            outl.get("positions_dtype", "uint32"), outl.get("values_dtype", "float32"),
        )
        if pos.size:
            p_arr, p_ptr = _i64ptr(pos)
            v_arr, v_ptr = _fptr(val)
            keep += [p_arr, v_arr]
            t.outlier_count = int(pos.shape[0])
            t.outlier_pos = p_ptr
            t.outlier_val = v_ptr

    salient = tm.get("salient")
    if salient:
        s_idx, s_val = _read_salient(
            out_dir / salient["indices"], out_dir / salient["weights"],
            salient.get("indices_dtype", "uint32"), salient.get("weights_dtype", "float32"),
        )
        if s_idx.size:
            si_arr, si_ptr = _i32ptr(s_idx)
            sv_arr, sv_ptr = _fptr(s_val)
            keep += [si_arr, sv_arr]
            t.salient_count = int(s_idx.shape[0])
            t.salient_idx = si_ptr
            t.salient_val = sv_ptr

    lr = tm.get("lowrank")
    if lr:
        a, b = _read_lowrank(out_dir, lr)
        a_arr, a_ptr = _fptr(a.reshape(-1))
        b_arr, b_ptr = _fptr(b.reshape(-1))
        keep += [a_arr, b_arr]
        t.lowrank_rank = int(lr["rank"])
        t.lowrank_a = a_ptr
        t.lowrank_b = b_ptr

    out = np.zeros(packed, dtype=np.float32)
    rc = lib.orka_vq_dequantize(ctypes.byref(t), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if rc != 0:
        raise RuntimeError(f"kernel returned {rc} for {tm['name']}")
    del keep
    return out


def main() -> int:
    artifact = Path(sys.argv[1] if len(sys.argv) > 1 else ".local_runs/dist/SmolLM2-135M-4bpw.orka")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    lib = ctypes.CDLL(str(Path(__file__).resolve().parent / "liborka_vq.so"))
    lib.orka_vq_dequantize.restype = ctypes.c_int
    lib.orka_vq_dequantize.argtypes = [ctypes.POINTER(_Tensor), ctypes.POINTER(ctypes.c_float)]

    manifest = json.loads((artifact / "manifest.json").read_text())
    tensors = manifest["tensors"][:limit]
    worst = 0.0
    for tm in tensors:
        ref = np.asarray(_decode_tensor(artifact, tm), dtype=np.float32)
        got = kernel_decode(lib, artifact, tm)
        max_abs = float(np.max(np.abs(ref - got))) if ref.size else 0.0
        denom = float(np.max(np.abs(ref))) or 1.0
        rel = max_abs / denom
        worst = max(worst, rel)
        flag = "OK" if rel < 1e-6 else "MISMATCH"
        print(f"{flag} {tm['name']}: n_stages={tm['n_stages']} max_abs={max_abs:.3e} rel={rel:.3e}")
    print(f"\nworst relative diff across {len(tensors)} tensors: {worst:.3e}")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
